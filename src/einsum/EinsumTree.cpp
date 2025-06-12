#include "EinsumTree.h"
#include "EinsumNode.h"
#include "Optimizer.h"
#include <iostream>
#include <cstring>

using mini_jit::dim_t;
using mini_jit::exec_t;

mini_jit::einsum::EinsumNode *mini_jit::einsum::EinsumTree::parse_einsum_expression(std::string const &einsum_expression,
                                                                                    std::vector<int64_t> &dimension_sizes,
                                                                                    mini_jit::dtype_t dtype,
                                                                                    int64_t thread_target,
                                                                                    int64_t max_kernel_size)
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

    initialize_einsum_nodes(root_node, 
                            dimension_sizes, 
                            dtype, 
                            thread_target, 
                            max_kernel_size);

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
                                                           std::vector<int64_t> &dimension_sizes,
                                                           mini_jit::dtype_t dtype,
                                                           int64_t thread_target,
                                                           int64_t max_kernel_size)
{
    if (einsum_node == nullptr)
    {
        return;
    }

    einsum_node->dtype = dtype;
    einsum_node->computational_operations = 0.0;

    // no children? -> input node and no setup needed
    if(einsum_node->get_number_of_children() == 0)
    {
        return;
    }

    //////////////////////////////////////////////////////////////////
    // GATHER AND SORT ALL USED IDS
    //////////////////////////////////////////////////////////////////
    std::vector<int64_t> dim_ids;
    std::vector<int64_t> dim_sizes;

    for (auto &dim_id : einsum_node->dimension_ids)
    {
        dim_ids.push_back(dim_id);
        dim_sizes.push_back(dimension_sizes[dim_id]);
    }
    if (einsum_node->get_number_of_children() == 2)
    {
        for (auto &dim_id : einsum_node->leftChild->dimension_ids)
        {
            // check if the dimension id is already in the list
            if (std::find(dim_ids.begin(), dim_ids.end(), dim_id) == dim_ids.end())
            {
                dim_ids.push_back(dim_id);
                dim_sizes.push_back(dimension_sizes[dim_id]);
            }
        }
        for (auto &dim_id : einsum_node->rightChild->dimension_ids)
        {
            // check if the dimension id is already in the list
            if (std::find(dim_ids.begin(), dim_ids.end(), dim_id) == dim_ids.end())
            {
                dim_ids.push_back(dim_id);
                dim_sizes.push_back(dimension_sizes[dim_id]);
            }
        }
    }

    //////////////////////////////////////////////////////////////////
    // INIT VECTORS
    //////////////////////////////////////////////////////////////////
    std::vector<dim_t> dim_types(dim_ids.size(), dim_t::k);
    std::vector<exec_t> exec_types(dim_ids.size(), exec_t::seq);
    std::vector<int64_t> strides_in0(dim_ids.size(), 0);
    std::vector<int64_t> strides_in1(dim_ids.size(), 0);
    std::vector<int64_t> strides_out(dim_ids.size(), 0);

    std::vector<int64_t> &out_dim_ids = einsum_node->dimension_ids;

    for (size_t i = 0; i < dim_ids.size(); i++)
    {
        //////////////////////////////////////////////////////////////////
        // SET TYPE
        //////////////////////////////////////////////////////////////////
        // output + left = M
        // output + right = N
        int64_t dim_id = dim_ids[i];

        if (einsum_node->get_number_of_children() == 2)
        {
            // Dimension M
            if (contains(out_dim_ids, dim_id) &&
                contains(einsum_node->leftChild->dimension_ids, dim_id))
            {
                dim_types[i] = dim_t::m;
            }
            // Dimension N
            else if (contains(out_dim_ids, dim_id) &&
                     contains(einsum_node->rightChild->dimension_ids, dim_id))
            {
                dim_types[i] = dim_t::n;
            }
        }
        else
        {
            dim_types[i] = dim_t::c;
        }

        // stride_in0
        if (einsum_node->leftChild != nullptr &&
            contains(einsum_node->leftChild->dimension_ids, dim_id))
        {
            // dimension_sizes[dim_id]

            // find out at which index is dim id
            // dim_id=2 dimension_ids=1,2,3,4 -> 1
            // gehe von 1+1 bis ende und multipliziere

            int64_t stride = 1;
            auto it = std::find(einsum_node->leftChild->dimension_ids.begin(), 
                                einsum_node->leftChild->dimension_ids.end(), 
                                dim_id);
            size_t index = std::distance(einsum_node->leftChild->dimension_ids.begin(), it);
            for (size_t j = index + 1; j < einsum_node->leftChild->dimension_ids.size(); ++j)
            {
                stride *= dimension_sizes[einsum_node->leftChild->dimension_ids[j]];
            }
            strides_in0[i] = stride;
        }

        // stride_in1
        if (einsum_node->rightChild != nullptr &&
            contains(einsum_node->rightChild->dimension_ids, dim_id))
        {
            int64_t stride = 1;
            auto it = std::find(einsum_node->rightChild->dimension_ids.begin(), 
                                einsum_node->rightChild->dimension_ids.end(), 
                                dim_id);
            size_t index = std::distance(einsum_node->rightChild->dimension_ids.begin(), it);
            for (size_t j = index + 1; j < einsum_node->rightChild->dimension_ids.size(); ++j)
            {
                stride *= dimension_sizes[einsum_node->rightChild->dimension_ids[j]];
            }
            strides_in1[i] = stride;
        }

        // stride_out
        if (contains(out_dim_ids, dim_id))
        {
            int64_t stride = 1;
            auto it = std::find(out_dim_ids.begin(), out_dim_ids.end(), dim_id);
            size_t index = std::distance(out_dim_ids.begin(), it);
            for (size_t j = index + 1; j < out_dim_ids.size(); ++j)
            {
                stride *= dimension_sizes[einsum_node->dimension_ids[j]];
            }
            strides_out[i] = stride;
        }
    }

    mini_jit::ir::Optimizer::optimize(dim_types, 
                                      exec_types, 
                                      dim_sizes, 
                                      strides_in0, 
                                      strides_in1, 
                                      strides_out, 
                                      thread_target, 
                                      max_kernel_size);

    int prim_count = std::count(exec_types.begin(), exec_types.end(), exec_t::prim);
    mini_jit::ptype_t main_ptype = mini_jit::ptype_t::none;
    if(prim_count == 2)
    {
        main_ptype = mini_jit::ptype_t::identity;
        einsum_node->computational_operations = 0.0; // no operations for identity
    }
    else if(prim_count == 3)
    {
        main_ptype = mini_jit::ptype_t::gemm;
        einsum_node->computational_operations = 2.0f;
        for(int64_t size : dim_sizes)
        {
            einsum_node->computational_operations *= size;
        }
    }
    else if(prim_count == 4)
    {
        main_ptype = mini_jit::ptype_t::brgemm;
        einsum_node->computational_operations = 2.0f;
        for(int64_t size : dim_sizes)
        {
            einsum_node->computational_operations *= size;
        }
    }

    // add child ops
    einsum_node->computational_operations += einsum_node->leftChild ? einsum_node->leftChild->computational_operations : 0.0;
    einsum_node->computational_operations += einsum_node->rightChild ? einsum_node->rightChild->computational_operations : 0.0;

    einsum_node->operation.setup(dtype,
                                 ptype_t::none,
                                 main_ptype,
                                 ptype_t::none,
                                 dim_types,
                                 exec_types,
                                 dim_sizes,
                                 strides_in0,
                                 strides_in1,
                                 strides_out);

    // set values for children
    if (einsum_node->leftChild)
    {
        initialize_einsum_nodes(einsum_node->leftChild, dimension_sizes, dtype, thread_target, max_kernel_size);
    }
    if (einsum_node->rightChild)
    {
        initialize_einsum_nodes(einsum_node->rightChild, dimension_sizes, dtype, thread_target, max_kernel_size);
    }
}

void mini_jit::einsum::EinsumTree::execute(EinsumNode *root_node,
                                           std::vector<int64_t> &dimension_sizes,
                                           std::map<std::string, void const *> &tensor_inputs,
                                           mini_jit::dtype_t dtype)
{
    if (root_node == nullptr)
    {
        return;
    }

    // we are a leaf node! set tensor_out to the input tensor
    if(root_node->get_number_of_children() == 0)
    {
        // leaf node, check if input tensor is given
        if (tensor_inputs.find(root_node->tensor_expression) != tensor_inputs.end())
        {
            // delete old output tensor if it exists
            if (root_node->tensor_out != nullptr)
            {
                if (root_node->dtype == mini_jit::dtype_t::fp32)
                {
                    delete[] static_cast<float *>(root_node->tensor_out);
                }
                else if (root_node->dtype == mini_jit::dtype_t::fp64)
                {
                    delete[] static_cast<double *>(root_node->tensor_out);
                }
                root_node->tensor_out = nullptr;
            }
            // copy input tensor to output tensor
            int64_t output_tensor_size = 1;
            for (auto dim_id : root_node->dimension_ids)
            {
                output_tensor_size *= dimension_sizes[dim_id];
            }
            if (dtype == mini_jit::dtype_t::fp32)
            {
                root_node->tensor_out = new float[output_tensor_size]{0.0f};
            }
            else if (dtype == mini_jit::dtype_t::fp64)
            {
                root_node->tensor_out = new double[output_tensor_size]{0.0};
            }
            int64_t dtype_size = dtype == mini_jit::dtype_t::fp32 ? sizeof(float) : sizeof(double);
            std::memcpy(root_node->tensor_out,
                        tensor_inputs.at(root_node->tensor_expression),
                        output_tensor_size * dtype_size);
        }
        else
        {
            throw std::invalid_argument("Error: No input tensor found for leaf node with expression: " 
                + root_node->tensor_expression);
        }
        return;
    }

    // compute left child
    if (root_node->leftChild != nullptr)
    {
        execute(root_node->leftChild, dimension_sizes, tensor_inputs, dtype);
    }

    // compute right child
    if (root_node->rightChild != nullptr)
    {
        execute(root_node->rightChild, dimension_sizes, tensor_inputs, dtype);
    }

    // delete old output tensor if it exists
    if (root_node->tensor_out != nullptr)
    {
        if (root_node->dtype == mini_jit::dtype_t::fp32)
        {
            delete[] static_cast<float *>(root_node->tensor_out);
        }
        else if (root_node->dtype == mini_jit::dtype_t::fp64)
        {
            delete[] static_cast<double *>(root_node->tensor_out);
        }
        root_node->tensor_out = nullptr;
    }
    // create new output tensor
    int64_t output_tensor_size = 1;
    for (auto dim_id : root_node->dimension_ids)
    {
        output_tensor_size *= dimension_sizes[dim_id];
    }
    if(dtype == mini_jit::dtype_t::fp32)
    {
        root_node->tensor_out = new float[output_tensor_size]{0.0f};
    }
    else if (dtype == mini_jit::dtype_t::fp64)
    {
        root_node->tensor_out = new double[output_tensor_size]{0.0};
    }
    
    // execute operation
    auto ptrLeft = root_node->leftChild ? root_node->leftChild->tensor_out : nullptr;
    auto ptrRight = root_node->rightChild ? root_node->rightChild->tensor_out : nullptr;
    root_node->operation.execute(ptrLeft, 
                                 ptrRight, 
                                 root_node->tensor_out);
}
