#ifndef MINI_JIT_UNARY_SIGMOID_PRIMITIVE_H
#define MINI_JIT_UNARY_SIGMOID_PRIMITIVE_H

#include "Kernel.h"

namespace mini_jit
{
    namespace kernels
    {
        namespace unary
        {
            /**
             * @brief Kernel that applies sigmoid activation function to the input and stores it into the output.
             * Uses polynomial approximation: σ(x) = 1 / (1 + e^(-x)) for fast SIMD computation.
             * 
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void sigmoid(mini_jit::Kernel &kernel, 
                         u_int32_t m, 
                         u_int32_t n);
        }
    }
};

#endif // MINI_JIT_UNARY_SIGMOID_PRIMITIVE_H 