#ifndef MINI_JIT_UNARY_SIGMOID_TAYLOR_PRIMITIVE_H
#define MINI_JIT_UNARY_SIGMOID_TAYLOR_PRIMITIVE_H

#include "Kernel.h"

namespace mini_jit
{
    namespace kernels
    {
        namespace unary
        {
            /**
             * @brief Kernel that applies sigmoid activation function to the input and stores it into the output.
             * Uses polynomial approximation: σ(x) ≈ 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5 for precise 
             * SIMD computation in [-2,2].
             * 
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void sigmoid_taylor(mini_jit::Kernel &kernel, 
                                u_int32_t m, 
                                u_int32_t n);
        }
    }
};

#endif // MINI_JIT_UNARY_SIGMOID_TAYLOR_PRIMITIVE_H 