#ifndef MINI_JIT_UNARY_MUL_PRIMITIVE_H
#define MINI_JIT_UNARY_MUL_PRIMITIVE_H

#include "Kernel.h"

namespace mini_jit
{
    namespace kernels
    {
        namespace binary
        {
            /**
             * @brief Kernel that multiplies two matrices element-wise.
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void mul(mini_jit::Kernel &kernel, 
                     u_int32_t m, 
                     u_int32_t n );
        }
    }
};

#endif //MINI_JIT_UNARY_MUL_PRIMITIVE_H