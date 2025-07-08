#ifndef MINI_JIT_UNARY_FAST_SIGMOID_PRIMITIVE_H
#define MINI_JIT_UNARY_FAST_SIGMOID_PRIMITIVE_H

#include <cstdint>
#include <mlc/Kernel.h>

namespace mini_jit
{
    namespace kernels
    {
        namespace unary
        {
            /**
             * @brief Kernel that applies the fast sigmoid activation function to a matrix.
             *
             * Specifically, it computes the function:
             * f(x) = 0.5 * (x / (1 + abs(x)) + 1)
             * for each element in the matrix.
             *
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void fast_sigmoid(mini_jit::Kernel& kernel,
                              u_int32_t         m,
                              u_int32_t         n);
        } // namespace unary
    } // namespace kernels
}; // namespace mini_jit

#endif // MINI_JIT_UNARY_FAST_SIGMOID_PRIMITIVE_H