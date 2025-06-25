#ifndef MINI_JIT_UNARY_ZERO_PRIMITIVE_H
#define MINI_JIT_UNARY_ZERO_PRIMITIVE_H

#include "Kernel.h"

namespace mini_jit
{
    namespace kernels
    {
        namespace unary
        {
            /**
             * @brief Kernel for zeroing out a matrix using neon and EOR.
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             * @param trans_b 0 if B is stored in column-major order, 1 if B is stored in row-major order.
             */
            void zero( mini_jit::Kernel &kernel, 
                       uint32_t m, 
                       uint32_t n,
                       uint32_t trans_b );
        }
    }
};

#endif // MINI_JIT_UNARY_ZERO_PRIMITIVE_H