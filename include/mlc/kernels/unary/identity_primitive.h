#ifndef MINI_JIT_UNARY_IDENTITY_PRIMITIVE_H
#define MINI_JIT_UNARY_IDENTITY_PRIMITIVE_H

#include <cstdint>
#include <mlc/Kernel.h>

namespace mini_jit
{
    namespace kernels
    {
        namespace unary
        {
            /**
             * @brief Kernel for performing the identity operation on a matrix.
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void identity(mini_jit::Kernel& kernel,
                          uint32_t          m,
                          uint32_t          n);
        } // namespace unary
    } // namespace kernels
}; // namespace mini_jit

#endif // MINI_JIT_UNARY_IDENTITY_PRIMITIVE_H