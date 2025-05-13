#ifndef MINI_JIT_MATMUL_M_N_K_H
#define MINI_JIT_MATMUL_M_N_K_H

#include "Kernel.h"

namespace mini_jit
{
    namespace kernels
    {
        namespace internal
        {
            /**
             * @brief Generates on M loop for matrix multiplication.
             * @param kernel Kernel object to be filled with instructions.
             * @param mLoopIterations number of M loop iterations.
             * @param k number of columns in A and rows in B.
             */
            void generateMnLoop( mini_jit::Kernel &kernel, 
                                int mLoopIterations, 
                                int k );

            void generateMn1Loop( mini_jit::Kernel &kernel );

            /**
             * @brief Generates on N loop for matrix multiplication.
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of M loop iterations.
             * @param n number of M loop iterations.
             * @param k number of columns in A and rows in B.
             */
            void generateNLoop(mini_jit::Kernel &kernel, int m, int n, int k);
        }

        /**
         * @brief Kernel for batch-reduce matrix multiplication.
         * @param kernel Kernel object to be filled with instructions.
         * @param m number of rows in A and C.
         * @param k number of columns in A and rows in B.
         */
        void matmul_m_n_k( mini_jit::Kernel &kernel, 
                           int m, 
                           int n, 
                           int k );      
    }
};

#endif