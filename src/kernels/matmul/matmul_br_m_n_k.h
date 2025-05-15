#ifndef MINI_JIT_MATMUL_BR_M_N_K_H
#define MINI_JIT_MATMUL_BR_M_N_K_H

#include "Kernel.h"

namespace mini_jit
{
    namespace kernels
    {
        /**
         * @brief Kernel for batch-reduce matrix multiplication.
         * @param kernel Kernel object to be filled with instructions.
         * @param m number of rows in A and C.
         * @param n number of columns in B and C.
         * @param k number of columns in A and rows in B.
         * @param br_size batch-reduce size.
         */
        void matmul_br_m_n_k( mini_jit::Kernel &kernel, 
                              int m, 
                              int n, 
                              int k,
                              int br_size );
    }
};

#endif