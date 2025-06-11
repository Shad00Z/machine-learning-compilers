#ifndef MINI_JIT_EINSUM_EINSUM_NODE_H
#define MINI_JIT_EINSUM_EINSUM_NODE_H

#include <vector>

#include "Dimension.h"
#include "TensorOperation.h"

namespace mini_jit
{
    namespace einsum
    {
        struct EinsumNode
        {
            // The node is the output for a contraction / permutation
            std::vector<mini_jit::ir::Dimension> dimensions;

            // The node is an input for a contraction / permutation
            std::vector<mini_jit::ir::Dimension> dimensions_in;

            std::vector<int64_t> dimension_ids;

            EinsumNode *leftChild, *rightChild;

            mini_jit::TensorOperation operation;

            /**
             *
             */
            EinsumNode(std::vector<int64_t> const &dimension_ids,
                       EinsumNode *left,
                       EinsumNode *right)
                : dimension_ids(dimension_ids), leftChild(left), rightChild(right)
            {
            }

            ~EinsumNode()
            {
                delete leftChild;
                delete rightChild;
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