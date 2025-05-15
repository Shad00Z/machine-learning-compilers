#ifndef MINI_JIT_MATMUL_M_N_K_H
#define MINI_JIT_MATMUL_M_N_K_H

#include "Kernel.h"

namespace mini_jit
{
    namespace kernels
    {
        namespace matmul
        {
            namespace internal
            {
                /**
                 * @brief Generates an N loop for matrix multiplication where N = 1.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param mLoopIterations number of M loop iterations.
                 * @param mLoopRemainder remaining iterations for M loop.
                 * @param k number of columns in A and rows in B.
                 */
                void generateN1Loop(mini_jit::Kernel &kernel,
                                    int mLoopIterations,
                                    int mLoopRemainder,
                                    int k);

                /**
                 * @brief Generates an N loop for matrix multiplication where N = 2.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param mLoopIterations number of M loop iterations.
                 * @param mLoopRemainder remaining iterations for M loop.
                 * @param k number of columns in A and rows in B.
                 */
                void generateN2Loop(mini_jit::Kernel &kernel,
                                    int mLoopIterations,
                                    int mLoopRemainder,
                                    int k);

                /**
                 * @brief Generates an N loop for matrix multiplication where N = 3.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param mLoopIterations number of M loop iterations.
                 * @param mLoopRemainder remaining iterations for M loop.
                 * @param k number of columns in A and rows in B.
                 */
                void generateN3Loop(mini_jit::Kernel &kernel,
                                    int mLoopIterations,
                                    int mLoopRemainder,
                                    int k);
            }
            /**
             * @brief Kernel for batch-reduce matrix multiplication.
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in A and C.
             * @param n number of columns in B and C.
             * @param k number of columns in A and rows in B.
             */
            void matmul_m_n_k(mini_jit::Kernel &kernel,
                              int m,
                              int n,
                              int k);
        }
    }
};

#endif