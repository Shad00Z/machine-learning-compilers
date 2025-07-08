#ifndef MINI_JIT_EINSUM_EINSUM_NODE_H
#define MINI_JIT_EINSUM_EINSUM_NODE_H

#include <mlc/TensorOperation.h>
#include <vector>

namespace mini_jit
{
    namespace einsum
    {
        struct EinsumNode
        {
            /// The IDs of the dimensions in the output tensor
            std::vector<int64_t> m_output_dimension_ids;

            /// The IDs of the dimensions in the operation
            std::vector<int64_t> m_dimension_ids;

            /// The data type of the tensor
            mini_jit::dtype_t m_dtype = mini_jit::dtype_t::fp32;

            /// Primititve type for the first touch kernel
            mini_jit::ptype_t m_prim_first_touch = mini_jit::ptype_t::none;

            /// Primitive type for the main kernel
            mini_jit::ptype_t m_prim_main = mini_jit::ptype_t::none;

            /// Primitive type for the last touch kernel
            mini_jit::ptype_t m_prim_last_touch = mini_jit::ptype_t::none;

            /// Dimension types of the loops (m, n, k, c)
            std::vector<mini_jit::dim_t> m_dim_types;

            /// Execution types of the loops (seq, shared, prim)
            std::vector<mini_jit::exec_t> m_exec_types;

            /// Sizes of the dimensions (loops)
            std::vector<int64_t> m_dim_sizes;

            /// Strides of the first input tensor
            std::vector<int64_t> m_strides_in0;

            /// Strides of the second input tensor
            std::vector<int64_t> m_strides_in1;

            /// Strides of the output tensor
            std::vector<int64_t> m_strides_out;

            /// Size of the output tensor
            int64_t m_tensor_size = 1;

            /// The output tensor for this node
            void* m_tensor_out = nullptr;

            /// The tensor operation associated with this node
            mini_jit::TensorOperation m_operation;

            /// String representation of the einsum expression
            std::string m_tensor_expression = "";

            /// The left child node in the einsum tree
            EinsumNode* m_left_child = nullptr;

            /// The right child node in the einsum tree
            EinsumNode* m_right_child = nullptr;

            /// The number of operations performed by this node
            double m_computational_operations = 0.0;

            /**
             *
             */
            EinsumNode(std::vector<int64_t> const& output_dimension_ids,
                       std::string                 tensor_expression,
                       EinsumNode*                 left_child,
                       EinsumNode*                 right_child)
                : m_output_dimension_ids(output_dimension_ids), m_tensor_expression(tensor_expression), m_left_child(left_child), m_right_child(right_child)
            {
            }

            ~EinsumNode()
            {
                if (m_left_child != nullptr)
                {
                    delete m_left_child;
                }
                if (m_right_child != nullptr)
                {
                    delete m_right_child;
                }
                if (m_tensor_out != nullptr)
                {
                    if (m_dtype == mini_jit::dtype_t::fp32)
                    {
                        delete[] static_cast<float*>(m_tensor_out);
                    }
                    else if (m_dtype == mini_jit::dtype_t::fp64)
                    {
                        delete[] static_cast<double*>(m_tensor_out);
                    }
                }
            }

            int64_t get_number_of_children() const
            {
                return (m_left_child != nullptr) + (m_right_child != nullptr);
            }
        };
    } // namespace einsum
} // namespace mini_jit
#endif // MINI_JIT_EINSUM_EINSUM_NODE_H