#ifndef MINI_JIT_UNARY_MIN_PRIMITIVE_H
#define MINI_JIT_UNARY_MIN_PRIMITIVE_H

#include <cstdint>
#include <mlc/Kernel.h>

namespace mini_jit
{
    namespace kernels
    {
        namespace binary
        {
            /**
             * @brief Kernel that computes the element-wise minimum of two matrices.
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void min(mini_jit::Kernel& kernel,
                     u_int32_t         m,
                     u_int32_t         n);
        } // namespace binary
    } // namespace kernels
}; // namespace mini_jit

#endif // MINI_JIT_UNARY_MIN_PRIMITIVE_H