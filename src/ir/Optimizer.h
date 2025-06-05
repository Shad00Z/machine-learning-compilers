#ifndef MINI_JIT_IR_OPTIMIZER_H
#define MINI_JIT_IR_OPTIMIZER_H

#include "Dimension.h"
#include "types.h"
#include <vector>

namespace mini_jit
{
    namespace ir
    {
        class Optimizer;
    }
}

class mini_jit::ir::Optimizer
{
public:
    // Static class
    Optimizer() = delete;

    /**
     * @brief Optimize the dimensions of a tensor operation.
     *
     * @param dimensions A vector of dimensions to be optimized.
     * @param thread_target The target number of threads for optimization.
     */
    static void optimize(std::vector<Dimension> &dimensions,
                         int64_t thread_target);

private:
    /**
     * @brief Identify primitive dimensions in the tensor operation and adjust their order.
     *
     * @param dimensions A vector of dimensions to be processed.
     */
    static void identifyPrimitives(std::vector<Dimension> &dimensions);

    /**
     * @brief Split large dimensions into smaller ones.
     *
     * @param dimensions A vector of dimensions to be processed.
     */
    static void splitDimensions(std::vector<Dimension> &dimensions);

    /**
     * @brief Turn sequential dimensions into shared dimensions.
     * 
     * @param dimensions A vector of dimensions to be processed.
     * @param thread_target The target number of threads for optimization.
     */
    static void createSharedLoops(std::vector<Dimension> &dimensions,
                                  int64_t thread_target);

    // Helper functions

    /**
     * @brief Find the best split for a given dimension size and type.
     *
     * @param i_size The size of the dimension to be split.
     * @param i_max_size The maximum size allowed for the dimension.
     * @param i_type The type of the dimension (e.g., M, N, K).
     * @param o_size_0 Output size for the first part of the split (SEQ).
     * @param o_size_1 Output size for the second part of the split (PRIM).
     */
    static void findBestSplit(int64_t i_size,
                              int64_t i_max_size,
                              dim_t i_type,
                              int64_t &o_size_0,
                              int64_t &o_size_1);

    static void findLargestMultipleOfDivisor(int64_t i_divisor,
                                             int64_t i_size,
                                             int64_t i_max_size,
                                             int64_t &o_size_0,
                                             int64_t &o_size_1);
};

#endif