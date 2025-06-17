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

    mini_jit::einsum::EinsumNode *l_root_node = parse_einsum_expression_recursive(einsum_expression);

    // SWAP NODES
    swap_nodes(l_root_node);

    initialize_einsum_nodes(l_root_node, dimension_sizes);
    return l_root_node;
}

mini_jit::einsum::EinsumNode *mini_jit::einsum::EinsumTree::parse_einsum_expression_recursive(std::string const &einsum_expression)
{
    if (einsum_expression.empty())
    {
        return nullptr;
    }

    std::string l_left_input_expression = "";
    std::string l_right_input_expression = "";
    std::string l_output = einsum_expression;

    // find most right arrow
    size_t l_arrow_pos = einsum_expression.rfind("->");
    if (l_arrow_pos != std::string::npos)
    {
        std::string l_inputs = einsum_expression.substr(0, l_arrow_pos);
        l_output = einsum_expression.substr(l_arrow_pos + 3, einsum_expression.size() - l_arrow_pos - 4);

        // if the first char is not a bracket, there is only one input
        int64_t l_split_input_pos = -1;
        if (l_inputs[0] == '[')
        {
            // split inputs by comma
            int64_t l_brackets = 0;
            int64_t l_current_pos = 0;
            for (char c : l_inputs)
            {
                switch (c)
                {
                case '[':
                    l_brackets++;
                    break;
                case ']':
                    l_brackets--;
                    break;
                case ',':
                    if (l_brackets == 0)
                    {
                        l_split_input_pos = l_current_pos;
                    }
                    break;
                default:
                    break;
                }

                // return early if we found the comma already
                if (l_split_input_pos != -1)
                {
                    break;
                }

                l_current_pos++;
            }
        }

        if (l_split_input_pos == -1)
        {
            l_left_input_expression = l_inputs.substr(1, l_inputs.size() - 2);
            l_right_input_expression = "";
        }
        else
        {
            // remove outer brackets
            l_left_input_expression = l_inputs.substr(1, l_split_input_pos - 2);
            l_right_input_expression = l_inputs.substr(l_split_input_pos + 2, l_inputs.size() - l_split_input_pos - 3);
        }
    }

    return new EinsumNode(get_dimensions_from_expression(l_output),
                          l_output,
                          parse_einsum_expression_recursive(l_left_input_expression),
                          parse_einsum_expression_recursive(l_right_input_expression));
}

std::vector<int64_t> mini_jit::einsum::EinsumTree::get_dimensions_from_expression(std::string const &einsum_expression)
{
    std::vector<int64_t> l_dims;

    // example input: 1,2,3,4
    for (auto value : einsum_expression | std::views::split(','))
    {
        l_dims.push_back(std::stoll(std::string(value.begin(), value.end())));
    }

    return l_dims;
}

void mini_jit::einsum::EinsumTree::initialize_einsum_nodes(EinsumNode *einsum_node,
                                                           std::vector<int64_t> &dimension_sizes)
{
    if (einsum_node == nullptr)
    {
        return;
    }

    initialize_einsum_nodes(einsum_node->m_left_child, dimension_sizes);
    initialize_einsum_nodes(einsum_node->m_right_child, dimension_sizes);

    //////////////////////////////////////////////////////////////////
    // GATHER AND SORT ALL USED IDS
    //////////////////////////////////////////////////////////////////

    // this is already given
    std::vector<int64_t> *l_output_dimension_ids = &einsum_node->m_output_dimension_ids;
    // these will be initialized
    std::vector<int64_t> *l_operation_dim_ids = &einsum_node->m_dimension_ids;
    std::vector<int64_t> *l_dim_sizes = &einsum_node->m_dim_sizes;
    // local vector of sizes of the output dimensions
    std::vector<int64_t> l_out_dim_sizes = einsum_node->m_dim_sizes;

    // add ids from output
    for (auto dim_id : einsum_node->m_output_dimension_ids)
    {
        l_operation_dim_ids->push_back(dim_id);
        l_dim_sizes->push_back(dimension_sizes[dim_id]);
        l_out_dim_sizes.push_back(dimension_sizes[dim_id]);
    }
    // add ids from children (for 1 child, the ids are the same as the output ids)
    if (einsum_node->get_number_of_children() == 2)
    {
        for (auto dim_id : einsum_node->m_left_child->m_output_dimension_ids)
        {
            // check if the dimension id is already in the list
            if (!contains(*l_operation_dim_ids, dim_id))
            {
                l_operation_dim_ids->push_back(dim_id);
                l_dim_sizes->push_back(dimension_sizes[dim_id]);
            }
        }
        for (auto dim_id : einsum_node->m_right_child->m_output_dimension_ids)
        {
            // check if the dimension id is already in the list
            if (!contains(*l_operation_dim_ids, dim_id))
            {
                l_operation_dim_ids->push_back(dim_id);
                l_dim_sizes->push_back(dimension_sizes[dim_id]);
            }
        }
    }

    // compute the size of the output tensor
    einsum_node->m_tensor_size = 1;
    for (auto &dim_id : *l_output_dimension_ids)
    {
        einsum_node->m_tensor_size *= dimension_sizes[dim_id];
    }

    //////////////////////////////////////////////////////////////////
    // INIT VECTORS
    //////////////////////////////////////////////////////////////////

    std::vector<dim_t> *l_dim_types = &einsum_node->m_dim_types;
    l_dim_types->resize(l_operation_dim_ids->size(), dim_t::k);
    std::vector<exec_t> *l_exec_types = &einsum_node->m_exec_types;
    l_exec_types->resize(l_operation_dim_ids->size(), exec_t::seq);
    std::vector<int64_t> *l_strides_in0 = &einsum_node->m_strides_in0;
    l_strides_in0->resize(l_operation_dim_ids->size(), 0);
    std::vector<int64_t> *l_strides_in1 = &einsum_node->m_strides_in1;
    l_strides_in1->resize(l_operation_dim_ids->size(), 0);
    std::vector<int64_t> *l_strides_out = &einsum_node->m_strides_out;
    l_strides_out->resize(l_operation_dim_ids->size(), 0);

    for (size_t i = 0; i < l_operation_dim_ids->size(); i++)
    {
        //////////////////////////////////////////////////////////////////
        // SET TYPE
        //////////////////////////////////////////////////////////////////
        // output + left = M
        // output + right = N

        int64_t l_dim_id = (*l_operation_dim_ids)[i];

        if (einsum_node->get_number_of_children() == 2)
        {
            // Dimension M
            if (contains(*l_output_dimension_ids, l_dim_id) &&
                contains(einsum_node->m_left_child->m_output_dimension_ids, l_dim_id))
            {
                (*l_dim_types)[i] = dim_t::m;
            }
            // Dimension N
            else if (contains(*l_output_dimension_ids, l_dim_id) &&
                     contains(einsum_node->m_right_child->m_output_dimension_ids, l_dim_id))
            {
                (*l_dim_types)[i] = dim_t::n;
            }
        }
        else
        {
            (*l_dim_types)[i] = dim_t::c;
        }

        // stride_in0
        if (einsum_node->m_left_child != nullptr &&
            contains(einsum_node->m_left_child->m_output_dimension_ids, l_dim_id))
        {
            int64_t stride = 1;
            auto it = std::find(einsum_node->m_left_child->m_output_dimension_ids.begin(),
                                einsum_node->m_left_child->m_output_dimension_ids.end(),
                                l_dim_id);
            size_t index = std::distance(einsum_node->m_left_child->m_output_dimension_ids.begin(), it);
            for (size_t j = index + 1; j < einsum_node->m_left_child->m_output_dimension_ids.size(); ++j)
            {
                stride *= dimension_sizes[einsum_node->m_left_child->m_output_dimension_ids[j]];
            }
            (*l_strides_in0)[i] = stride;
        }

        // stride_in1
        if (einsum_node->m_right_child != nullptr &&
            contains(einsum_node->m_right_child->m_output_dimension_ids, l_dim_id))
        {
            int64_t stride = 1;
            auto it = std::find(einsum_node->m_right_child->m_output_dimension_ids.begin(),
                                einsum_node->m_right_child->m_output_dimension_ids.end(),
                                l_dim_id);
            size_t index = std::distance(einsum_node->m_right_child->m_output_dimension_ids.begin(), it);
            for (size_t j = index + 1; j < einsum_node->m_right_child->m_output_dimension_ids.size(); ++j)
            {
                stride *= dimension_sizes[einsum_node->m_right_child->m_output_dimension_ids[j]];
            }
            (*l_strides_in1)[i] = stride;
        }

        // stride_out
        if (contains(*l_output_dimension_ids, l_dim_id))
        {
            int64_t stride = 1;
            auto it = std::find(l_output_dimension_ids->begin(), l_output_dimension_ids->end(), l_dim_id);
            size_t index = std::distance(l_output_dimension_ids->begin(), it);
            for (size_t j = index + 1; j < l_output_dimension_ids->size(); ++j)
            {
                stride *= dimension_sizes[einsum_node->m_output_dimension_ids[j]];
            }
            (*l_strides_out)[i] = stride;
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
    optimize_einsum_nodes(einsum_node->m_left_child, thread_target, max_kernel_size);
    optimize_einsum_nodes(einsum_node->m_right_child, thread_target, max_kernel_size);

    // optimize current node
    mini_jit::ir::Optimizer::optimize(einsum_node->m_dim_types,
                                      einsum_node->m_exec_types,
                                      einsum_node->m_dim_sizes,
                                      einsum_node->m_strides_in0,
                                      einsum_node->m_strides_in1,
                                      einsum_node->m_strides_out,
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
    einsum_node->m_dtype = dtype;
    einsum_node->m_computational_operations = 0.0;

    // no further operations for input nodes
    if (einsum_node->get_number_of_children() == 0)
    {
        return;
    }

    // lower children
    lower_einsum_nodes_to_tensor_operations(einsum_node->m_left_child, dimension_sizes, dtype);
    lower_einsum_nodes_to_tensor_operations(einsum_node->m_right_child, dimension_sizes, dtype);

    // lower current node
    int l_prim_count = std::count(einsum_node->m_exec_types.begin(), einsum_node->m_exec_types.end(), exec_t::prim);
    mini_jit::ptype_t l_main_ptype = mini_jit::ptype_t::none;
    if (l_prim_count == 2)
    {
        l_main_ptype = mini_jit::ptype_t::identity;
        einsum_node->m_computational_operations = 0.0; // no operations for identity
    }
    else if (l_prim_count == 3)
    {
        l_main_ptype = mini_jit::ptype_t::gemm;
        einsum_node->m_computational_operations = 2.0f;
        for (int64_t size : einsum_node->m_dim_sizes)
        {
            einsum_node->m_computational_operations *= size;
        }
    }
    else if (l_prim_count == 4)
    {
        l_main_ptype = mini_jit::ptype_t::brgemm;
        einsum_node->m_computational_operations = 2.0f;
        for (int64_t size : einsum_node->m_dim_sizes)
        {
            einsum_node->m_computational_operations *= size;
        }
    }

    // add child ops
    einsum_node->m_computational_operations += einsum_node->m_left_child ? einsum_node->m_left_child->m_computational_operations : 0.0;
    einsum_node->m_computational_operations += einsum_node->m_right_child ? einsum_node->m_right_child->m_computational_operations : 0.0;

    einsum_node->m_operation.setup(einsum_node->m_dtype,
                                   ptype_t::none,
                                   l_main_ptype,
                                   ptype_t::none,
                                   einsum_node->m_dim_types,
                                   einsum_node->m_exec_types,
                                   einsum_node->m_dim_sizes,
                                   einsum_node->m_strides_in0,
                                   einsum_node->m_strides_in1,
                                   einsum_node->m_strides_out);
}

void mini_jit::einsum::EinsumTree::execute(EinsumNode *root_node,
                                           std::vector<int64_t> &dimension_sizes,
                                           std::map<std::string, void const *> &tensor_inputs)
{
    if (root_node == nullptr)
    {
        return;
    }

    const int64_t l_tensor_size = root_node->m_tensor_size;

    if (root_node->m_dtype == mini_jit::dtype_t::fp32)
    {
        if (root_node->m_tensor_out == nullptr)
        {
            root_node->m_tensor_out = new float[l_tensor_size]{0.0f};
        }
        else
        {
            std::fill(static_cast<float *>(root_node->m_tensor_out),
                      static_cast<float *>(root_node->m_tensor_out) + l_tensor_size,
                      0.0f);
        }
    }
    else if (root_node->m_dtype == mini_jit::dtype_t::fp64)
    {
        if (root_node->m_tensor_out == nullptr)
        {
            root_node->m_tensor_out = new double[l_tensor_size]{0.0};
        }
        else
        {
            std::fill(static_cast<double *>(root_node->m_tensor_out),
                      static_cast<double *>(root_node->m_tensor_out) + l_tensor_size,
                      0.0);
        }
    }

    // we are a leaf node! set tensor_out to the input tensor
    if (root_node->get_number_of_children() == 0)
    {
        // check if input tensor is given
        auto it = tensor_inputs.find(root_node->m_tensor_expression);
        if (it != tensor_inputs.end())
        {
            if (root_node->m_dtype == mini_jit::dtype_t::fp32)
            {
                std::copy(
                    static_cast<const float *>(it->second),
                    static_cast<const float *>(it->second) + l_tensor_size,
                    static_cast<float *>(root_node->m_tensor_out));
            }
            else if (root_node->m_dtype == mini_jit::dtype_t::fp64)
            {
                std::copy(
                    static_cast<const double *>(it->second),
                    static_cast<const double *>(it->second) + l_tensor_size,
                    static_cast<double *>(root_node->m_tensor_out));
            }
        }
        else
        {
            throw std::invalid_argument("Error: No input tensor found for leaf node with expression: " + root_node->m_tensor_expression);
        }
    }
    // we are not a leaf node -> compute children and execute operation
    else
    {
        // compute children
        execute(root_node->m_left_child, dimension_sizes, tensor_inputs);
        execute(root_node->m_right_child, dimension_sizes, tensor_inputs);

        // execute operation
        auto l_ptr_right_child = root_node->m_right_child ? root_node->m_right_child->m_tensor_out : nullptr;
        root_node->m_operation.execute(root_node->m_left_child->m_tensor_out,
                                       l_ptr_right_child,
                                       root_node->m_tensor_out);
    }
}

void mini_jit::einsum::EinsumTree::optimize_einsum_tree(EinsumNode *root_node)
{
    // swap

    // reorder

    // permutation nodes
}

void mini_jit::einsum::EinsumTree::swap_nodes(EinsumNode *einsum_node)
{
    if (einsum_node == nullptr || einsum_node->get_number_of_children() == 0)
    {
        return;
    }

    if (einsum_node->get_number_of_children() == 1)
    {
        swap_nodes(einsum_node->m_left_child);
        return;
    }

    // recursively swap children
    swap_nodes(einsum_node->m_left_child);
    swap_nodes(einsum_node->m_right_child);

    int64_t l_unit_stride_root_node = einsum_node->m_output_dimension_ids.size() - 1;
    int64_t l_unit_stride_left_child = einsum_node->m_left_child->m_output_dimension_ids.size() - 1;
    int64_t l_unit_stride_right_child = einsum_node->m_right_child->m_output_dimension_ids.size() - 1;

    // swap nodes if
    // (A) the right child and the root node have the same unit stride, AND
    // (B) if the right childs output dimension ids contain the left childs unit stride somewhere
    if (einsum_node->m_output_dimension_ids[l_unit_stride_root_node] == einsum_node->m_right_child->m_output_dimension_ids[l_unit_stride_right_child] &&
        contains(einsum_node->m_right_child->m_output_dimension_ids, einsum_node->m_left_child->m_output_dimension_ids[l_unit_stride_left_child]))
    {
        std::cout << "Swapping nodes: " << std::endl;
        EinsumNode *l_temp_node = einsum_node->m_left_child;

        einsum_node->m_left_child = einsum_node->m_right_child;
        einsum_node->m_right_child = l_temp_node;
    }
}
