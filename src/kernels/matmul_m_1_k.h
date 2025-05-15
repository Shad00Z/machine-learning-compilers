#ifndef MINI_JIT_MATMUL_M_1_K_H
#define MINI_JIT_MATMUL_M_1_K_H

#include "../Kernel.h"

namespace mini_jit
{
    namespace kernels
    {
        namespace internal
        {
            /**
             * @brief Generates an M loop for matrix multiplication where M % 8 = 0 and N = 1.
             * @param kernel Kernel object to be filled with instructions.
             * @param mLoopIterations number of M loop iterations.
             * @param k number of columns in A and rows in B.
             */
            void generateM8N1Loop( mini_jit::Kernel &kernel, 
                                   int mLoopIterations, 
                                   int k );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 1 and N = 1.
             * @param kernel Kernel object to be filled with instructions.
             */
            void generateM1N1Loop( mini_jit::Kernel &kernel );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 2 and N = 1.
             * @param kernel Kernel object to be filled with instructions.
             */
            void generateM2N1Loop( mini_jit::Kernel &kernel );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 3 and N = 1.
             * @param kernel Kernel object to be filled with instructions.
             */
            void generateM3N1Loop( mini_jit::Kernel &kernel );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 4 and N = 1.
             * @param kernel Kernel object to be filled with instructions.
             */
            void generateM4N1Loop( mini_jit::Kernel &kernel );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 5 and N = 1.
             * @param kernel Kernel object to be filled with instructions.
             */
            void generateM5N1Loop( mini_jit::Kernel &kernel );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 6 and N = 1.
             * @param kernel Kernel object to be filled with instructions.
             */
            void generateM6N1Loop( mini_jit::Kernel &kernel );

            /**
             * @brief Generates an M loop for matrix multiplication where M = 7 and N = 1.
             * @param kernel Kernel object to be filled with instructions.
             */
            void generateM7N1Loop( mini_jit::Kernel &kernel );
        }

        /**
         * @brief Kernel for batch-reduce matrix multiplication.
         * @param kernel Kernel object to be filled with instructions.
         * @param m number of rows in A and C.
         * @param k number of columns in A and rows in B.
         */
        void matmul_m_1_k( mini_jit::Kernel &kernel, 
                           int m, 
                           int k );      
    }
};

#endif