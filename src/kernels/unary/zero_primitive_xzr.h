#ifndef MINI_JIT_UNARY_ZERO_XZR_PRIMITVE_H
#define MINI_JIT_UNARY_ZERO_XZR_PRIMITVE_H

#include "Kernel.h"

namespace mini_jit
{
    namespace kernels
    {
        namespace unary
        {
            /**
             * @brief Kernel for zeroing out a matrix using the XZR register.
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             * @param trans_b 0 if B is stored in column-major order, 1 if B is stored in row-major order.
             */
            void zero_xzr( mini_jit::Kernel &kernel, 
                           u_int32_t m, 
                           u_int32_t n,
                           u_int32_t trans_b );
        }
    }
};

#endif //MINI_JIT_UNARY_ZERO_XZR_PRIMITVE_H