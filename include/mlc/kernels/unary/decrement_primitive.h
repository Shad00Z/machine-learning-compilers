#ifndef MINI_JIT_UNARY_DECREMENT_PRIMITIVE_H
#define MINI_JIT_UNARY_DECREMENT_PRIMITIVE_H

#include <cstdint>
#include <mlc/Kernel.h>

namespace mini_jit
{
    namespace kernels
    {
        namespace unary
        {
            /**
             * @brief Kernel that decrements the input by one and stores it into the output.
             *
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void decrement(mini_jit::Kernel& kernel,
                           u_int32_t         m,
                           u_int32_t         n);
        } // namespace unary
    } // namespace kernels
}; // namespace mini_jit

#endif // MINI_JIT_UNARY_DECREMENT_PRIMITIVE_H