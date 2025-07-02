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

    // plain swapping is not needed anymore,
    // since the reordering will do swapping as well
    // swap_nodes(l_root_node);

    // reordering is necessary to make the tree executable
    // it also needs to be done before initializing the nodes
    reorder_node_dimensions(l_root_node);

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

void mini_jit::einsum::EinsumTree::initialize_einsum_nodes(EinsumNode *root_node,
                                                           std::vector<int64_t> &dimension_sizes)
{
    if (root_node == nullptr)
    {
        return;
    }

    initialize_einsum_nodes(root_node->m_left_child, dimension_sizes);
    initialize_einsum_nodes(root_node->m_right_child, dimension_sizes);

    //////////////////////////////////////////////////////////////////
    // GATHER AND SORT ALL USED IDS
    //////////////////////////////////////////////////////////////////

    // this is already given
    std::vector<int64_t> *l_output_dimension_ids = &root_node->m_output_dimension_ids;
    // these will be initialized
    std::vector<int64_t> *l_operation_dim_ids = &root_node->m_dimension_ids;
    std::vector<int64_t> *l_dim_sizes = &root_node->m_dim_sizes;
    // local vector of sizes of the output dimensions
    std::vector<int64_t> l_out_dim_sizes = root_node->m_dim_sizes;

    // add ids from output
    for (auto dim_id : root_node->m_output_dimension_ids)
    {
        l_operation_dim_ids->push_back(dim_id);
        l_dim_sizes->push_back(dimension_sizes[dim_id]);
        l_out_dim_sizes.push_back(dimension_sizes[dim_id]);
    }
    // add ids from children (for 1 child, the ids are the same as the output ids)
    if (root_node->get_number_of_children() == 2)
    {
        for (auto dim_id : root_node->m_left_child->m_output_dimension_ids)
        {
            // check if the dimension id is already in the list
            if (!contains(*l_operation_dim_ids, dim_id))
            {
                l_operation_dim_ids->push_back(dim_id);
                l_dim_sizes->push_back(dimension_sizes[dim_id]);
            }
        }
        for (auto dim_id : root_node->m_right_child->m_output_dimension_ids)
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
    root_node->m_tensor_size = 1;
    for (auto &dim_id : *l_output_dimension_ids)
    {
        root_node->m_tensor_size *= dimension_sizes[dim_id];
    }

    //////////////////////////////////////////////////////////////////
    // INIT VECTORS
    //////////////////////////////////////////////////////////////////

    std::vector<dim_t> *l_dim_types = &root_node->m_dim_types;
    l_dim_types->resize(l_operation_dim_ids->size(), dim_t::k);
    std::vector<exec_t> *l_exec_types = &root_node->m_exec_types;
    l_exec_types->resize(l_operation_dim_ids->size(), exec_t::seq);
    std::vector<int64_t> *l_strides_in0 = &root_node->m_strides_in0;
    l_strides_in0->resize(l_operation_dim_ids->size(), 0);
    std::vector<int64_t> *l_strides_in1 = &root_node->m_strides_in1;
    l_strides_in1->resize(l_operation_dim_ids->size(), 0);
    std::vector<int64_t> *l_strides_out = &root_node->m_strides_out;
    l_strides_out->resize(l_operation_dim_ids->size(), 0);

    for (size_t i = 0; i < l_operation_dim_ids->size(); i++)
    {
        //////////////////////////////////////////////////////////////////
        // SET TYPE
        //////////////////////////////////////////////////////////////////
        // output + left = M
        // output + right = N

        int64_t l_dim_id = (*l_operation_dim_ids)[i];

        if (root_node->get_number_of_children() == 2)
        {
            // Dimension M
            if (contains(*l_output_dimension_ids, l_dim_id) &&
                contains(root_node->m_left_child->m_output_dimension_ids, l_dim_id))
            {
                (*l_dim_types)[i] = dim_t::m;
            }
            // Dimension N
            else if (contains(*l_output_dimension_ids, l_dim_id) &&
                     contains(root_node->m_right_child->m_output_dimension_ids, l_dim_id))
            {
                (*l_dim_types)[i] = dim_t::n;
            }
        }
        else
        {
            (*l_dim_types)[i] = dim_t::c;
        }

        // stride_in0
        if (root_node->m_left_child != nullptr &&
            contains(root_node->m_left_child->m_output_dimension_ids, l_dim_id))
        {
            int64_t stride = 1;
            auto it = std::find(root_node->m_left_child->m_output_dimension_ids.begin(),
                                root_node->m_left_child->m_output_dimension_ids.end(),
                                l_dim_id);
            size_t index = std::distance(root_node->m_left_child->m_output_dimension_ids.begin(), it);
            for (size_t j = index + 1; j < root_node->m_left_child->m_output_dimension_ids.size(); ++j)
            {
                stride *= dimension_sizes[root_node->m_left_child->m_output_dimension_ids[j]];
            }
            (*l_strides_in0)[i] = stride;
        }

        // stride_in1
        if (root_node->m_right_child != nullptr &&
            contains(root_node->m_right_child->m_output_dimension_ids, l_dim_id))
        {
            int64_t stride = 1;
            auto it = std::find(root_node->m_right_child->m_output_dimension_ids.begin(),
                                root_node->m_right_child->m_output_dimension_ids.end(),
                                l_dim_id);
            size_t index = std::distance(root_node->m_right_child->m_output_dimension_ids.begin(), it);
            for (size_t j = index + 1; j < root_node->m_right_child->m_output_dimension_ids.size(); ++j)
            {
                stride *= dimension_sizes[root_node->m_right_child->m_output_dimension_ids[j]];
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
                stride *= dimension_sizes[root_node->m_output_dimension_ids[j]];
            }
            (*l_strides_out)[i] = stride;
        }
    }
}

void mini_jit::einsum::EinsumTree::optimize_einsum_nodes(EinsumNode *root_node,
                                                         int64_t thread_target,
                                                         int64_t max_kernel_size,
                                                         int64_t min_kernel_size)
{
    if (root_node == nullptr)
    {
        return;
    }

    // no optimizations for input nodes
    if (root_node->get_number_of_children() == 0)
    {
        return;
    }

    // optimize children
    optimize_einsum_nodes(root_node->m_left_child, thread_target, max_kernel_size, min_kernel_size);
    optimize_einsum_nodes(root_node->m_right_child, thread_target, max_kernel_size, min_kernel_size);

    // optimize current node
    mini_jit::ir::Optimizer::optimize(root_node->m_dim_types,
                                      root_node->m_exec_types,
                                      root_node->m_dim_sizes,
                                      root_node->m_strides_in0,
                                      root_node->m_strides_in1,
                                      root_node->m_strides_out,
                                      thread_target,
                                      max_kernel_size,
                                      min_kernel_size);
}

void mini_jit::einsum::EinsumTree::lower_einsum_nodes_to_tensor_operations(EinsumNode *root_node,
                                                                           std::vector<int64_t> &dimension_sizes,
                                                                           mini_jit::dtype_t dtype)
{
    if (root_node == nullptr)
    {
        return;
    }

    // operations for all nodes
    root_node->m_dtype = dtype;
    root_node->m_computational_operations = 0.0;

    // no further operations for input nodes
    if (root_node->get_number_of_children() == 0)
    {
        return;
    }

    // lower children
    lower_einsum_nodes_to_tensor_operations(root_node->m_left_child, dimension_sizes, dtype);
    lower_einsum_nodes_to_tensor_operations(root_node->m_right_child, dimension_sizes, dtype);

    // lower current node
    int l_prim_count = std::count(root_node->m_exec_types.begin(), root_node->m_exec_types.end(), exec_t::prim);
    mini_jit::ptype_t l_main_ptype = mini_jit::ptype_t::none;
    if (l_prim_count == 2)
    {
        l_main_ptype = mini_jit::ptype_t::identity;
        root_node->m_computational_operations = 0.0; // no operations for identity
    }
    else if (l_prim_count == 3)
    {
        l_main_ptype = mini_jit::ptype_t::gemm;
        root_node->m_computational_operations = 2.0f;
        for (int64_t size : root_node->m_dim_sizes)
        {
            root_node->m_computational_operations *= size;
        }
    }
    else if (l_prim_count == 4)
    {
        l_main_ptype = mini_jit::ptype_t::brgemm;
        root_node->m_computational_operations = 2.0f;
        for (int64_t size : root_node->m_dim_sizes)
        {
            root_node->m_computational_operations *= size;
        }
    }

    // add child ops
    root_node->m_computational_operations += root_node->m_left_child ? root_node->m_left_child->m_computational_operations : 0.0;
    root_node->m_computational_operations += root_node->m_right_child ? root_node->m_right_child->m_computational_operations : 0.0;

    root_node->m_operation.setup(root_node->m_dtype,
                                 ptype_t::none,
                                 l_main_ptype,
                                 ptype_t::none,
                                 root_node->m_dim_types,
                                 root_node->m_exec_types,
                                 root_node->m_dim_sizes,
                                 root_node->m_strides_in0,
                                 root_node->m_strides_in1,
                                 root_node->m_strides_out);
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
            throw std::invalid_argument("Error: No input tensor found for leaf node with expression: " +
                                        root_node->m_tensor_expression);
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

void mini_jit::einsum::EinsumTree::reorder_node_dimensions(EinsumNode *root_node)
{
    if (root_node == nullptr || root_node->get_number_of_children() == 0)
    {
        // since we view root_node as the parent node where the children are reordered,
        // we do not need to do anything for leaf nodes
        return;
    }

    // one child -> identity operation and not a contraction
    // -> no need to reorder dimensions here
    if (root_node->get_number_of_children() == 1)
    {
        reorder_node_dimensions(root_node->m_left_child);
        return;
    }

    int64_t l_unit_stride_root_node = root_node->m_output_dimension_ids.size() - 1;
    int64_t l_unit_stride_left_child = root_node->m_left_child->m_output_dimension_ids.size() - 1;

    // Assumption: The parent's order is correct (parent specifies the order)
    int64_t l_parent_dim_id = root_node->m_output_dimension_ids[l_unit_stride_root_node];

    // Find unit stride for M
    if (contains(root_node->m_right_child->m_output_dimension_ids,
                 root_node->m_output_dimension_ids[l_unit_stride_root_node]))
    {
        // swap children
        EinsumNode *l_temp_node = root_node->m_left_child;

        root_node->m_left_child = root_node->m_right_child;
        root_node->m_right_child = l_temp_node;

        l_unit_stride_left_child = root_node->m_left_child->m_output_dimension_ids.size() - 1;
    }

    auto l_dim_child_m_it = std::find_if(root_node->m_left_child->m_output_dimension_ids.begin(),
                                         root_node->m_left_child->m_output_dimension_ids.end(),
                                         [l_parent_dim_id](const int64_t dim_id)
                                         {
                                             return (dim_id == l_parent_dim_id);
                                         });

    // no M found
    if (l_dim_child_m_it == root_node->m_left_child->m_output_dimension_ids.end())
    {
        throw std::invalid_argument("EinsumTree: No M dimension found for child " +
                                    root_node->m_left_child->m_tensor_expression +
                                    " and parent " +
                                    root_node->m_tensor_expression);
    }
    // M is in output dims but not the right-most element
    else if (l_dim_child_m_it < root_node->m_left_child->m_output_dimension_ids.end() - 1)
    {
        EinsumNode *l_left_child_permute = new EinsumNode(root_node->m_left_child->m_output_dimension_ids,
                                                          root_node->m_left_child->m_tensor_expression,
                                                          root_node->m_left_child,
                                                          nullptr);

        // move M dimension to the right-most position
        // Calculate the position in the new vector and rotate
        size_t l_m_position = std::distance(root_node->m_left_child->m_output_dimension_ids.begin(), l_dim_child_m_it);
        auto l_new_m_it = l_left_child_permute->m_output_dimension_ids.begin() + l_m_position;
        std::rotate(l_new_m_it, l_new_m_it + 1, l_left_child_permute->m_output_dimension_ids.end());

        // update expression of the new permute node
        std::string l_new_expression = "";
        for (size_t i = 0; i < l_left_child_permute->m_output_dimension_ids.size(); i++)
        {
            if (i > 0)
            {
                l_new_expression += ",";
            }
            l_new_expression += std::to_string(l_left_child_permute->m_output_dimension_ids[i]);
        }
        l_left_child_permute->m_tensor_expression = l_new_expression;

        // insert into the tree
        root_node->m_left_child = l_left_child_permute;
    }

    // Find K in left child
    int64_t l_k_dim_index = l_unit_stride_left_child;
    for (int i = l_unit_stride_left_child; i >= 0; i--)
    {
        if (contains(root_node->m_right_child->m_output_dimension_ids,
                     root_node->m_left_child->m_output_dimension_ids[i]))
        {
            l_k_dim_index = i;
            break;
        }
    }

    int64_t l_k_dim_id = root_node->m_left_child->m_output_dimension_ids[l_k_dim_index];
    auto l_dim_child_k_it = std::find_if(root_node->m_right_child->m_output_dimension_ids.begin(),
                                         root_node->m_right_child->m_output_dimension_ids.end(),
                                         [l_k_dim_id](const int64_t dim_id)
                                         {
                                             return (dim_id == l_k_dim_id);
                                         });

    // no K found
    if (l_dim_child_k_it == root_node->m_right_child->m_output_dimension_ids.end())
    {
        throw std::invalid_argument("EinsumTree: No K dimension found for child " +
                                    root_node->m_right_child->m_tensor_expression +
                                    " and parent " +
                                    root_node->m_tensor_expression);
    }
    // K is in output dims but not the right-most element
    else if (l_dim_child_k_it < root_node->m_right_child->m_output_dimension_ids.end() - 1)
    {
        EinsumNode *l_right_child_permute = new EinsumNode(root_node->m_right_child->m_output_dimension_ids,
                                                           root_node->m_right_child->m_tensor_expression,
                                                           root_node->m_right_child,
                                                           nullptr);

        // move K dimension to the right-most position
        // Calculate the position in the new vector and rotate
        size_t l_k_position = std::distance(root_node->m_right_child->m_output_dimension_ids.begin(), l_dim_child_k_it);
        auto l_new_k_it = l_right_child_permute->m_output_dimension_ids.begin() + l_k_position;
        std::rotate(l_new_k_it, l_new_k_it + 1, l_right_child_permute->m_output_dimension_ids.end());

        // update expression of the new permute node
        std::string l_new_expression = "";
        for (size_t i = 0; i < l_right_child_permute->m_output_dimension_ids.size(); i++)
        {
            if (i > 0)
            {
                l_new_expression += ",";
            }
            l_new_expression += std::to_string(l_right_child_permute->m_output_dimension_ids[i]);
        }
        l_right_child_permute->m_tensor_expression = l_new_expression;

        // insert into the tree
        root_node->m_right_child = l_right_child_permute;
    }

    // recursively call children
    reorder_node_dimensions(root_node->m_left_child);
    reorder_node_dimensions(root_node->m_right_child);
}

void mini_jit::einsum::EinsumTree::swap_nodes(EinsumNode *root_node)
{
    if (root_node == nullptr || root_node->get_number_of_children() == 0)
    {
        return;
    }

    if (root_node->get_number_of_children() == 1)
    {
        swap_nodes(root_node->m_left_child);
        return;
    }

    // recursively swap children
    swap_nodes(root_node->m_left_child);
    swap_nodes(root_node->m_right_child);

    int64_t l_unit_stride_root_node = root_node->m_output_dimension_ids.size() - 1;
    int64_t l_unit_stride_left_child = root_node->m_left_child->m_output_dimension_ids.size() - 1;
    int64_t l_unit_stride_right_child = root_node->m_right_child->m_output_dimension_ids.size() - 1;

    // swap nodes if
    // (A) the right child and the root node have the same unit stride, AND
    // (B) if the right childs output dimension ids contain the left childs unit stride somewhere
    if (root_node->m_output_dimension_ids[l_unit_stride_root_node] == root_node->m_right_child->m_output_dimension_ids[l_unit_stride_right_child] &&
        contains(root_node->m_right_child->m_output_dimension_ids, root_node->m_left_child->m_output_dimension_ids[l_unit_stride_left_child]))
    {
        EinsumNode *l_temp_node = root_node->m_left_child;
        root_node->m_left_child = root_node->m_right_child;
        root_node->m_right_child = l_temp_node;
    }
}

std::string mini_jit::einsum::EinsumTree::to_string(EinsumNode *root_node)
{
    if (root_node == nullptr)
    {
        return "";
    }

    if (root_node->get_number_of_children() == 0)
    {
        return root_node->m_tensor_expression;
    }
    else if (root_node->get_number_of_children() == 1)
    {
        return "[" + to_string(root_node->m_left_child) + "]->[" + root_node->m_tensor_expression + "]";
    }
    else
    {
        return "[" + to_string(root_node->m_left_child) + "],[" + to_string(root_node->m_right_child) + "]->[" + root_node->m_tensor_expression + "]";
    }
}
