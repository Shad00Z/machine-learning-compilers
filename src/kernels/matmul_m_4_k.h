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
             * @brief Generates an M loop for matrix multiplication where M % 8 = 0.
             * @param kernel Kernel object to be filled with instructions.
             * @param mLoopIterations number of M loop iterations.
             * @param k number of columns in A and rows in B.
             */
            void generateMLoop( mini_jit::Kernel &kernel, 
                                int mLoopIterations, 
                                int k );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 1.
             * @param kernel Kernel object to be filled with instructions.
             */
            void generateMLoopRest1( mini_jit::Kernel &kernel );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 2.
             * @param kernel Kernel object to be filled with instructions.
             */
            void generateMLoopRest2( mini_jit::Kernel &kernel );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 3.
             * @param kernel Kernel object to be filled with instructions.
             */
            void generateMLoopRest3( mini_jit::Kernel &kernel );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 4.
             * @param kernel Kernel object to be filled with instructions.
             */
            void generateMLoopRest4( mini_jit::Kernel &kernel );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 5.
             * @param kernel Kernel object to be filled with instructions.
             */
            void generateMLoopRest5( mini_jit::Kernel &kernel );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 6.
             * @param kernel Kernel object to be filled with instructions.
             */
            void generateMLoopRest6( mini_jit::Kernel &kernel );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 7.
             * @param kernel Kernel object to be filled with instructions.
             */
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