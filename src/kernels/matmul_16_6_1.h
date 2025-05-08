#ifndef MINI_JIT_MATMUL_16_6_1_H
#define MINI_JIT_MATMUL_16_6_1_H

#include "../Kernel.h"
#include <cstdint>

namespace mini_jit
{
    namespace kernels
    {
        /**
         * @brief Kernel for batch-reduce matrix multiplication.
         * @param a Pointer to first of a batch of A matrices.
         * @param b Pointer to first of a batch of B matrices.
         * @param c Pointer to C matrix.
         * @param ld_a Leading dimension of A.
         * @param ld_b Leading dimension of B.
         * @param ld_c Leading dimension of C.
         * @param br_stride_a Stride (in elements, not bytes) between A matrices.
         * @param br_stride_b Stride (in elements, not bytes) between B matrices.
         */
        void matmul_16_6_1( mini_jit::Kernel &kernel );      
    }
};

#endif