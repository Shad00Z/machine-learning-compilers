#ifndef MINI_JIT_MATMUL_M_N_K_H
#define MINI_JIT_MATMUL_M_N_K_H

#include "../Kernel.h"

namespace mini_jit
{
    namespace kernels
    {
        namespace internal
        {
            /**
             * @brief Generates an M loop for matrix multiplication where M = 1.
             * @param kernel Kernel object to be filled with instructions.
             * @param mLoopIterations number of M loop iterations.
             * @param mLoopRemainder remaining iterations for M loop.
             * @param k number of columns in A and rows in B.
             */
            void generateNLoopRest1( mini_jit::Kernel &kernel,
                                     int mLoopIterations,
                                     int mLoopRemainder,
                                     int k );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 2.
             * @param kernel Kernel object to be filled with instructions.
             * @param mLoopIterations number of M loop iterations.
             * @param mLoopRemainder remaining iterations for M loop.
             * @param k number of columns in A and rows in B.
             */
            void generateNLoopRest2( mini_jit::Kernel &kernel,
                                     int mLoopIterations,
                                     int mLoopRemainder,
                                     int k );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 3.
             * @param kernel Kernel object to be filled with instructions.
             * @param mLoopIterations number of M loop iterations.
             * @param mLoopRemainder remaining iterations for M loop.
             * @param k number of columns in A and rows in B.
             */
            void generateNLoopRest3( mini_jit::Kernel &kernel,
                                     int mLoopIterations,
                                     int mLoopRemainder,
                                     int k );
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