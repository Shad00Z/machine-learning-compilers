#ifndef MINI_JIT_EINSUM_EINSUM_NODE_H
#define MINI_JIT_EINSUM_EINSUM_NODE_H

#include <vector>
#include "TensorOperation.h"

namespace mini_jit
{
    namespace einsum
    {
        struct EinsumNode
        {
            /// The IDs of the dimensions in the output tensor
            std::vector<int64_t> output_dimension_ids;

            /// The IDs of the dimensions in the operation
            std::vector<int64_t> dimension_ids;

            /// The data type of the tensor
            mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

            /// Primititve type for the first touch kernel
            mini_jit::ptype_t prim_first_touch = mini_jit::ptype_t::none;

            /// Primitive type for the main kernel
            mini_jit::ptype_t prim_main = mini_jit::ptype_t::none;

            /// Primitive type for the last touch kernel
            mini_jit::ptype_t prim_last_touch = mini_jit::ptype_t::none;

            /// Dimension types of the loops (m, n, k, c)
            std::vector<mini_jit::dim_t> dim_types;

            /// Execution types of the loops (seq, shared, prim)
            std::vector<mini_jit::exec_t> exec_types;

            /// Sizes of the dimensions (loops)
            std::vector<int64_t> dim_sizes;

            /// Strides of the first input tensor
            std::vector<int64_t> strides_in0;

            /// Strides of the second input tensor
            std::vector<int64_t> strides_in1;

            /// Strides of the output tensor
            std::vector<int64_t> strides_out;

            /// Size of the output tensor
            int64_t tensor_size = 1;

            /// The tensor operation associated with this node
            mini_jit::TensorOperation operation;

            /// The output tensor for this node
            void *tensor_out = nullptr;

            /// String representation of the einsum expression
            std::string tensor_expression = "";

            /// The left child node in the einsum tree
            EinsumNode *leftChild = nullptr;

            /// The right child node in the einsum tree
            EinsumNode *rightChild = nullptr;

            /// The number of operations performed by this node
            double computational_operations = 0.0;

            /**
             *
             */
            EinsumNode(std::vector<int64_t> const &output_dimension_ids,
                       std::string tensor_expression,
                       EinsumNode *left,
                       EinsumNode *right)
                : output_dimension_ids(output_dimension_ids), tensor_expression(tensor_expression), leftChild(left), rightChild(right)
            {
            }

            ~EinsumNode()
            {
                if (leftChild != nullptr)
                {
                    delete leftChild;
                }
                if (rightChild != nullptr)
                {
                    delete rightChild;
                }
                if (tensor_out != nullptr)
                {
                    if (dtype == mini_jit::dtype_t::fp32)
                    {
                        delete[] static_cast<float *>(tensor_out);
                    }
                    else if (dtype == mini_jit::dtype_t::fp64)
                    {
                        delete[] static_cast<double *>(tensor_out);
                    }
                }
            }

            int64_t get_number_of_children() const
            {
                int64_t result = 0;

                if (leftChild != nullptr)
                {
                    result++;
                }
                if (rightChild != nullptr)
                {
                    result++;
                }

                return result;
            }
        };
    }
}
#endif // MINI_JIT_EINSUM_EINSUM_NODE_H