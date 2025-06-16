#include "EinsumTree.h"
#include "EinsumNode.h"
#include "Optimizer.h"
#include <iostream>
#include <cstring>

using mini_jit::dim_t;
using mini_jit::exec_t;

mini_jit::einsum::EinsumNode *mini_jit::einsum::EinsumTree::parse_einsum_expression(std::string const &einsum_expression,
                                                                                    std::vector<int64_t> &dimension_sizes)
{
    // Check allowed characters
    for (char c : einsum_expression)
    {
        if (!(c == '[' || c == ']' || c == '-' || c == '>' ||
              (c >= '0' && c <= '9') || c == ','))
        {
            throw std::invalid_argument(std::string("Invalid character in einsum expression: ") + c);
        }
    }

    mini_jit::einsum::EinsumNode *root_node = parse_einsum_expression_recursive(einsum_expression);
    initialize_einsum_nodes(root_node, dimension_sizes);
    return root_node;
}

mini_jit::einsum::EinsumNode *mini_jit::einsum::EinsumTree::parse_einsum_expression_recursive(std::string const &einsum_expression)
{
    if (einsum_expression.empty())
    {
        return nullptr;
    }

    std::string leftInputExpression = "";
    std::string rightInputExpression = "";
    std::string output = einsum_expression;

    // find most right arrow
    size_t arrowPos = einsum_expression.rfind("->");
    if (arrowPos != std::string::npos)
    {
        std::string inputs = einsum_expression.substr(0, arrowPos);
        output = einsum_expression.substr(arrowPos + 3, einsum_expression.size() - arrowPos - 4);

        // if the first char is not a bracket, there is only one input
        int64_t splitInputPos = -1;
        if (inputs[0] == '[')
        {
            // split inputs by comma
            int64_t brackets = 0;
            int64_t currentPos = 0;
            for (char c : inputs)
            {
                switch (c)
                {
                case '[':
                    brackets++;
                    break;
                case ']':
                    brackets--;
                    break;
                case ',':
                    if (brackets == 0)
                    {
                        splitInputPos = currentPos;
                    }
                    break;
                default:
                    break;
                }

                // return early if we found the comma already
                if (splitInputPos != -1)
                {
                    break;
                }

                currentPos++;
            }
        }

        if (splitInputPos == -1)
        {
            leftInputExpression = inputs.substr(1, inputs.size() - 2);
            rightInputExpression = "";
        }
        else
        {
            // remove outer brackets
            leftInputExpression = inputs.substr(1, splitInputPos - 2);
            rightInputExpression = inputs.substr(splitInputPos + 2, inputs.size() - splitInputPos - 3);
        }
    }

    return new EinsumNode(get_dimensions_from_expression(output),
                          output,
                          parse_einsum_expression_recursive(leftInputExpression),
                          parse_einsum_expression_recursive(rightInputExpression));
}

std::vector<int64_t> mini_jit::einsum::EinsumTree::get_dimensions_from_expression(std::string const &einsum_expression)
{
    std::vector<int64_t> dims;

    // example input: 1,2,3,4
    for (auto value : einsum_expression | std::views::split(','))
    {
        dims.push_back(std::stoll(std::string(value.begin(), value.end())));
    }

    return dims;
}

void mini_jit::einsum::EinsumTree::initialize_einsum_nodes(EinsumNode *einsum_node,
                                                           std::vector<int64_t> &dimension_sizes)
{
    if (einsum_node == nullptr)
    {
        return;
    }

    initialize_einsum_nodes(einsum_node->leftChild, dimension_sizes);
    initialize_einsum_nodes(einsum_node->rightChild, dimension_sizes);

    //////////////////////////////////////////////////////////////////
    // GATHER AND SORT ALL USED IDS
    //////////////////////////////////////////////////////////////////

    // this is already given
    std::vector<int64_t> *output_dimension_ids = &einsum_node->output_dimension_ids;
    // these should be initialized
    std::vector<int64_t> *operation_dim_ids = &einsum_node->dimension_ids;
    std::vector<int64_t> *dim_sizes = &einsum_node->dim_sizes;
    // local vector of sizes of the output dimensions
    std::vector<int64_t> out_dim_sizes = einsum_node->dim_sizes;

    // add ids from output
    for (auto dim_id : einsum_node->output_dimension_ids)
    {
        operation_dim_ids->push_back(dim_id);
        dim_sizes->push_back(dimension_sizes[dim_id]);
        out_dim_sizes.push_back(dimension_sizes[dim_id]);
    }
    // add ids from children (for 1 child, the ids are the same as the output ids)
    if (einsum_node->get_number_of_children() == 2)
    {
        for (auto dim_id : einsum_node->leftChild->output_dimension_ids)
        {
            // check if the dimension id is already in the list
            if (std::find(operation_dim_ids->begin(), operation_dim_ids->end(), dim_id) == operation_dim_ids->end())
            {
                operation_dim_ids->push_back(dim_id);
                dim_sizes->push_back(dimension_sizes[dim_id]);
            }
        }
        for (auto dim_id : einsum_node->rightChild->output_dimension_ids)
        {
            // check if the dimension id is already in the list
            if (std::find(operation_dim_ids->begin(), operation_dim_ids->end(), dim_id) == operation_dim_ids->end())
            {
                operation_dim_ids->push_back(dim_id);
                dim_sizes->push_back(dimension_sizes[dim_id]);
            }
        }
    }

    // compute the size of the output tensor
    einsum_node->tensor_size = 1;
    for (auto &dim_id : *output_dimension_ids)
    {
        einsum_node->tensor_size *= dimension_sizes[dim_id];
    }

    //////////////////////////////////////////////////////////////////
    // INIT VECTORS
    //////////////////////////////////////////////////////////////////

    std::vector<dim_t> *dim_types = &einsum_node->dim_types;
    dim_types->resize(operation_dim_ids->size(), dim_t::k);
    std::vector<exec_t> *exec_types = &einsum_node->exec_types;
    exec_types->resize(operation_dim_ids->size(), exec_t::seq);
    std::vector<int64_t> *strides_in0 = &einsum_node->strides_in0;
    strides_in0->resize(operation_dim_ids->size(), 0);
    std::vector<int64_t> *strides_in1 = &einsum_node->strides_in1;
    strides_in1->resize(operation_dim_ids->size(), 0);
    std::vector<int64_t> *strides_out = &einsum_node->strides_out;
    strides_out->resize(operation_dim_ids->size(), 0);

    for (size_t i = 0; i < operation_dim_ids->size(); i++)
    {
        //////////////////////////////////////////////////////////////////
        // SET TYPE
        //////////////////////////////////////////////////////////////////
        // output + left = M
        // output + right = N

        int64_t dim_id = (*operation_dim_ids)[i];

        if (einsum_node->get_number_of_children() == 2)
        {
            // Dimension M
            if (contains(*output_dimension_ids, dim_id) &&
                contains(einsum_node->leftChild->output_dimension_ids, dim_id))
            {
                (*dim_types)[i] = dim_t::m;
            }
            // Dimension N
            else if (contains(*output_dimension_ids, dim_id) &&
                     contains(einsum_node->rightChild->output_dimension_ids, dim_id))
            {
                (*dim_types)[i] = dim_t::n;
            }
        }
        else
        {
            (*dim_types)[i] = dim_t::c;
        }

        // stride_in0
        if (einsum_node->leftChild != nullptr &&
            contains(einsum_node->leftChild->output_dimension_ids, dim_id))
        {
            int64_t stride = 1;
            auto it = std::find(einsum_node->leftChild->output_dimension_ids.begin(),
                                einsum_node->leftChild->output_dimension_ids.end(),
                                dim_id);
            size_t index = std::distance(einsum_node->leftChild->output_dimension_ids.begin(), it);
            for (size_t j = index + 1; j < einsum_node->leftChild->output_dimension_ids.size(); ++j)
            {
                stride *= dimension_sizes[einsum_node->leftChild->output_dimension_ids[j]];
            }
            (*strides_in0)[i] = stride;
        }

        // stride_in1
        if (einsum_node->rightChild != nullptr &&
            contains(einsum_node->rightChild->output_dimension_ids, dim_id))
        {
            int64_t stride = 1;
            auto it = std::find(einsum_node->rightChild->output_dimension_ids.begin(),
                                einsum_node->rightChild->output_dimension_ids.end(),
                                dim_id);
            size_t index = std::distance(einsum_node->rightChild->output_dimension_ids.begin(), it);
            for (size_t j = index + 1; j < einsum_node->rightChild->output_dimension_ids.size(); ++j)
            {
                stride *= dimension_sizes[einsum_node->rightChild->output_dimension_ids[j]];
            }
            (*strides_in1)[i] = stride;
        }

        // stride_out
        if (contains(*output_dimension_ids, dim_id))
        {
            int64_t stride = 1;
            auto it = std::find(output_dimension_ids->begin(), output_dimension_ids->end(), dim_id);
            size_t index = std::distance(output_dimension_ids->begin(), it);
            for (size_t j = index + 1; j < output_dimension_ids->size(); ++j)
            {
                stride *= dimension_sizes[einsum_node->output_dimension_ids[j]];
            }
            (*strides_out)[i] = stride;
        }
    }
}

void mini_jit::einsum::EinsumTree::optimize_einsum_nodes(EinsumNode *einsum_node,
                                                         int64_t thread_target,
                                                         int64_t max_kernel_size)
{
    if (einsum_node == nullptr)
    {
        return;
    }

    // no optimizations for input nodes
    if (einsum_node->get_number_of_children() == 0)
    {
        return;
    }

    // optimize children
    if (einsum_node->leftChild)
    {
        optimize_einsum_nodes(einsum_node->leftChild, thread_target, max_kernel_size);
    }
    if (einsum_node->rightChild)
    {
        optimize_einsum_nodes(einsum_node->rightChild, thread_target, max_kernel_size);
    }

    // optimize current node
    mini_jit::ir::Optimizer::optimize(einsum_node->dim_types,
                                      einsum_node->exec_types,
                                      einsum_node->dim_sizes,
                                      einsum_node->strides_in0,
                                      einsum_node->strides_in1,
                                      einsum_node->strides_out,
                                      thread_target,
                                      max_kernel_size);
}

void mini_jit::einsum::EinsumTree::lower_einsum_nodes_to_tensor_operations(EinsumNode *einsum_node,
                                                                           std::vector<int64_t> &dimension_sizes,
                                                                           mini_jit::dtype_t dtype)
{
    if (einsum_node == nullptr)
    {
        return;
    }

    // operations for all nodes
    einsum_node->dtype = dtype;
    einsum_node->computational_operations = 0.0;

    // no further operations for input nodes
    if (einsum_node->get_number_of_children() == 0)
    {
        return;
    }

    // lower children
    lower_einsum_nodes_to_tensor_operations(einsum_node->leftChild, dimension_sizes, dtype);
    lower_einsum_nodes_to_tensor_operations(einsum_node->rightChild, dimension_sizes, dtype);

    // lower current node
    int prim_count = std::count(einsum_node->exec_types.begin(), einsum_node->exec_types.end(), exec_t::prim);
    mini_jit::ptype_t main_ptype = mini_jit::ptype_t::none;
    if (prim_count == 2)
    {
        main_ptype = mini_jit::ptype_t::identity;
        einsum_node->computational_operations = 0.0; // no operations for identity
    }
    else if (prim_count == 3)
    {
        main_ptype = mini_jit::ptype_t::gemm;
        einsum_node->computational_operations = 2.0f;
        for (int64_t size : einsum_node->dim_sizes)
        {
            einsum_node->computational_operations *= size;
        }
    }
    else if (prim_count == 4)
    {
        main_ptype = mini_jit::ptype_t::brgemm;
        einsum_node->computational_operations = 2.0f;
        for (int64_t size : einsum_node->dim_sizes)
        {
            einsum_node->computational_operations *= size;
        }
    }

    // add child ops
    einsum_node->computational_operations += einsum_node->leftChild ? einsum_node->leftChild->computational_operations : 0.0;
    einsum_node->computational_operations += einsum_node->rightChild ? einsum_node->rightChild->computational_operations : 0.0;

    einsum_node->operation.setup(einsum_node->dtype,
                                 ptype_t::none,
                                 main_ptype,
                                 ptype_t::none,
                                 einsum_node->dim_types,
                                 einsum_node->exec_types,
                                 einsum_node->dim_sizes,
                                 einsum_node->strides_in0,
                                 einsum_node->strides_in1,
                                 einsum_node->strides_out);
}

void mini_jit::einsum::EinsumTree::execute(EinsumNode *root_node,
                                           std::vector<int64_t> &dimension_sizes,
                                           std::map<std::string, void const *> &tensor_inputs)
{
    if (root_node == nullptr)
    {
        return;
    }

    const int64_t tensor_size = root_node->tensor_size;

    if (root_node->dtype == mini_jit::dtype_t::fp32)
    {
        if (root_node->tensor_out == nullptr)
        {
            root_node->tensor_out = new float[tensor_size]{0.0f};
        }
        else
        {
            std::fill(static_cast<float *>(root_node->tensor_out),
                      static_cast<float *>(root_node->tensor_out) + tensor_size,
                      0.0f);
        }
    }
    else if (root_node->dtype == mini_jit::dtype_t::fp64)
    {
        if (root_node->tensor_out == nullptr)
        {
            root_node->tensor_out = new double[tensor_size]{0.0};
        }
        else
        {
            std::fill(static_cast<double *>(root_node->tensor_out),
                      static_cast<double *>(root_node->tensor_out) + tensor_size,
                      0.0);
        }
    }

    // we are a leaf node! set tensor_out to the input tensor
    if (root_node->get_number_of_children() == 0)
    {
        // check if input tensor is given
        auto it = tensor_inputs.find(root_node->tensor_expression);
        if (it != tensor_inputs.end())
        {
            if (root_node->dtype == mini_jit::dtype_t::fp32)
            {
                std::copy(
                    static_cast<const float *>(it->second),
                    static_cast<const float *>(it->second) + tensor_size,
                    static_cast<float *>(root_node->tensor_out));
            }
            else if (root_node->dtype == mini_jit::dtype_t::fp64)
            {
                std::copy(
                    static_cast<const double *>(it->second),
                    static_cast<const double *>(it->second) + tensor_size,
                    static_cast<double *>(root_node->tensor_out));
            }
        }
        else
        {
            throw std::invalid_argument("Error: No input tensor found for leaf node with expression: " + root_node->tensor_expression);
        }
    }
    // we are not a leaf node -> compute children and execute operation
    else
    {
        // compute children
        if (root_node->leftChild != nullptr)
        {
            execute(root_node->leftChild, dimension_sizes, tensor_inputs);
        }

        if (root_node->rightChild != nullptr)
        {
            execute(root_node->rightChild, dimension_sizes, tensor_inputs);
        }

        // execute operation
        auto ptrRight = root_node->rightChild ? root_node->rightChild->tensor_out : nullptr;
        root_node->operation.execute(root_node->leftChild->tensor_out,
                                     ptrRight,
                                     root_node->tensor_out);
    }
}
