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
             * @param m number of rows in A and C.
             * @param k number of columns in A and rows in B.
             */
            void generateMLoop( mini_jit::Kernel &kernel, 
                                int m, 
                                int k );

            void generateM1Loop( mini_jit::Kernel &kernel, 
                                 int k );
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