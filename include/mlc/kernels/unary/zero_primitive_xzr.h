#ifndef MINI_JIT_UNARY_ZERO_XZR_PRIMITIVE_H
#define MINI_JIT_UNARY_ZERO_XZR_PRIMITIVE_H

#include <cstdint>
#include <mlc/Kernel.h>

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
            void zero_xzr(mini_jit::Kernel& kernel,
                          u_int32_t         m,
                          u_int32_t         n,
                          u_int32_t         trans_b);
        } // namespace unary
    } // namespace kernels
}; // namespace mini_jit

#endif // MINI_JIT_UNARY_ZERO_XZR_PRIMITIVE_H