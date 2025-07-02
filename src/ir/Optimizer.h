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

/**
 * @brief The Optimizer class provides methods to optimize tensor operations
 * by adjusting dimensions, splitting large dimensions, and creating shared loops.
 * 
 * It is recommened to only call the `optimize` method, which will
 * internally call the other methods in the correct order.
 */
class mini_jit::ir::Optimizer
{
public:
    //! Deleted constructor to prevent instantiation of the static Optimizer class.
    Optimizer() = delete;

    /**
     * @brief Optimize the dimensions of a tensor operation.
     *
     * @param dimensions A vector of dimensions to be optimized.
     * @param thread_target The target number of threads for optimization.
     * @param max_kernel_size The maximum size of a kernel dimension
     * @param min_kernel_size The minimum size of a kernel dimension
     */
    static void optimize(std::vector<Dimension> &dimensions,
                         int64_t thread_target,
                         int64_t max_kernel_size,
                         int64_t min_kernel_size);

    /**
     * @brief Optimize the dimensions of a tensor operation.
     *
     * @param dim_types A vector of dimension types (M, N, K).
     * @param exec_types A vector of execution types (Prim, Seq, Shared).
     * @param dim_sizes A vector of dimension sizes.
     * @param strides_in0 A vector of strides for the first input tensor.
     * @param strides_in1 A vector of strides for the second input tensor.
     * @param strides_out A vector of strides for the output tensor.
     * @param thread_target The target number of threads for optimization.
     * @param max_kernel_size The maximum size of a kernel dimension
     * @param min_kernel_size The minimum size of a kernel dimension
     */
    static void optimize(std::vector<dim_t> &dim_types,
                         std::vector<exec_t> &exec_types,
                         std::vector<int64_t> &dim_sizes,
                         std::vector<int64_t> &strides_in0,
                         std::vector<int64_t> &strides_in1,
                         std::vector<int64_t> &strides_out,
                         int64_t thread_target,
                         int64_t max_kernel_size,
                         int64_t min_kernel_size);

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
     * @param max_kernel_size The maximum size allowed for a kernel dimension.
     * @param min_kernel_size The minimum size allowed for a kernel dimension.
     */
    static void splitDimensions(std::vector<Dimension> &dimensions,
                                int64_t max_kernel_size,
                                int64_t min_kernel_size);

    /**
     * @brief Fuse small dimensions into larger dimensions.
     * 
     * @param dimensions A vector of dimensions to be processed.
     * @param min_kernel_size The minimum size for a kernel dimension to be considered for fusion.
     */
    static void fuseDimensions(std::vector<Dimension> &dimensions,
                               int64_t min_kernel_size);

    /**
     * @brief Turn sequential dimensions into shared dimensions.
     *
     * @param dimensions A vector of dimensions to be processed.
     * @param thread_target The target number of threads for optimization.
     */
    static void createSharedLoops(std::vector<Dimension> &dimensions,
                                  int64_t thread_target);

private:
    // Helper functions

    /**
     * @brief Find the best split for a given dimension size and type.
     *
     * @param i_size The size of the dimension to be split.
     * @param i_max_kernel_size The maximum size allowed for the dimension.
     * @param i_min_kernel_size The minimum size allowed for the dimension.
     * @param i_type The type of the dimension (e.g., M, N, K).
     * @param o_size_0 Output size for the first part of the split (SEQ).
     * @param o_size_1 Output size for the second part of the split (PRIM).
     */
    static void findBestSplit(int64_t i_size,
                              int64_t i_max_kernel_size,
                              int64_t i_min_kernel_size,
                              dim_t i_type,
                              int64_t &o_size_0,
                              int64_t &o_size_1);

    /**
     * @brief Finds the largest multiple of a given divisor that divides the dimension 
     * size without rest and is less than or equal to the maximum kernel size.
     *
     * For example, if the divisor is 16, the size is 1600, 
     * and the maximum kernel size is 1024, the function will set
     * o_size_0 to 2 and o_size_1 to 800. 800 is the largest multiple
     * of 16 that is less than or equal to 1024. o_size_0 is then set
     * to i_size / o_size_1, which is 1600 / 800 = 2.
     * 
     * @param i_divisor The divisor to find the largest multiple of.
     * @param i_size The size of the dimension to be processed.
     * @param i_max_size The maximum size allowed for the divisor.
     * @param i_min_size The minimum size allowed for the divisor and multiplicand.
     * @param o_size_0 The input size divided by the largest multiple of the divisor.
     * @param o_size_1 The largest multiple of the divisor that is less than or equal to the maximum kernel size.
     */
    static void findLargestMultipleOfDivisor(int64_t i_divisor,
                                             int64_t i_size,
                                             int64_t i_max_size,
                                             int64_t i_min_size,
                                             int64_t &o_size_0,
                                             int64_t &o_size_1);
};

#endif