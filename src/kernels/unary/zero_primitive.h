#ifndef MINI_JIT_UNARY_ZERO_PRIMITVE_H
#define MINI_JIT_UNARY_ZERO_PRIMITVE_H

#include "Kernel.h"

namespace mini_jit
{
    namespace kernels
    {
        namespace unary
        {
            /**
             * @brief Kernel for zeroing out a matrix.
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void zero( mini_jit::Kernel &kernel, 
                       int m, 
                       int n );
        }
    }
};

#endif //MINI_JIT_UNARY_ZERO_PRIMITIVE_H