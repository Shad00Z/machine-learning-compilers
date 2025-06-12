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
            /// The IDs of the dimensions in the einsum expression
            std::vector<int64_t> dimension_ids;

            /// String representation of the einsum expression
            std::string tensor_expression = "";

            /// The data type of the tensor
            mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

            /// The left child node in the einsum tree
            EinsumNode *leftChild = nullptr;

            /// The right child node in the einsum tree
            EinsumNode *rightChild = nullptr;

            /// The tensor operation associated with this node
            mini_jit::TensorOperation operation;

            /// The output tensor for this node
            void *tensor_out = nullptr;

            /// The number of operations performed by this node
            double computational_operations = 0.0;

            /**
             *
             */
            EinsumNode(std::vector<int64_t> const &dimension_ids,
                      std::string tensor_expression,
                       EinsumNode *left,
                       EinsumNode *right)
                : dimension_ids(dimension_ids), tensor_expression(tensor_expression), leftChild(left), rightChild(right)
            {
            }

            ~EinsumNode()
            {
                if(leftChild != nullptr)
                {
                    delete leftChild;
                }
                if(rightChild != nullptr)
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

            int64_t get_number_of_children()
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