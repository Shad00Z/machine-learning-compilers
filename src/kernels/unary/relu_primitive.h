#ifndef MINI_JIT_UNARY_RELU_PRIMITVE_H
#define MINI_JIT_UNARY_RELU_PRIMITVE_H

#include "Kernel.h"

namespace mini_jit
{
    namespace kernels
    {
        namespace unary
        {
            /**
             * @brief Kernel that applies ReLU activation function to a matrix.
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void relu( mini_jit::Kernel &kernel, 
                       u_int32_t m, 
                       u_int32_t n );
        }
    }
};

#endif //MINI_JIT_UNARY_RELU_PRIMITIVE_H