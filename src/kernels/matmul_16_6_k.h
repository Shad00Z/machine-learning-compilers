#ifndef MINI_JIT_MATMUL_16_6_K_H
#define MINI_JIT_MATMUL_16_6_K_H

#include "../Kernel.h"
#include <cstdint>

namespace mini_jit
{
    namespace kernels
    {
        /**
         * @brief Kernel for batch-reduce matrix multiplication.
         * @param kernel Kernel object to be filled with instructions.
         */
        void matmul_16_6_k( mini_jit::Kernel &kernel );      
    }
};

#endif