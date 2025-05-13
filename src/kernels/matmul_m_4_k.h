#ifndef MINI_JIT_MATMUL_M_6_K_H
#define MINI_JIT_MATMUL_M_6_K_H

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
            void generateMLoop( mini_jit::Kernel &kernel, 
                                int mLoopIterations, 
                                int k );

            void generateMLoopRest1( mini_jit::Kernel &kernel );

            void generateMLoopRest2( mini_jit::Kernel &kernel );

            void generateMLoopRest3( mini_jit::Kernel &kernel );

            void generateMLoopRest4( mini_jit::Kernel &kernel );

            void generateMLoopRest5( mini_jit::Kernel &kernel );

            void generateMLoopRest6( mini_jit::Kernel &kernel );

            void generateMLoopRest7( mini_jit::Kernel &kernel );
        }

        /**
         * @brief Kernel for batch-reduce matrix multiplication.
         * @param kernel Kernel object to be filled with instructions.
         * @param m number of rows in A and C.
         * @param k number of columns in A and rows in B.
         */
        void matmul_m_4_k( mini_jit::Kernel &kernel, 
                           int m, 
                           int k );      
    }
};

#endif