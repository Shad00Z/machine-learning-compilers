#ifndef MINI_JIT_UNARY_ADD_PRIMITVE_H
#define MINI_JIT_UNARY_ADD_PRIMITVE_H

#include "Kernel.h"

namespace mini_jit
{
    namespace kernels
    {
        namespace binary
        {
            /**
             * @brief Kernel that adds two matrices element-wise.
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void add(mini_jit::Kernel &kernel, 
                     u_int32_t m, 
                     u_int32_t n );
        }
    }
};

#endif //MINI_JIT_UNARY_ADD_PRIMITVE_H