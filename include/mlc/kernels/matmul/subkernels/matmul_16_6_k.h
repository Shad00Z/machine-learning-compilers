#ifndef MINI_JIT_MATMUL_16_6_K_H
#define MINI_JIT_MATMUL_16_6_K_H

#include <mlc/Kernel.h>

namespace mini_jit
{
    namespace kernels
    {
        namespace matmul
        {
            namespace subkernels
            {
                /**
                 * @brief Kernel for batch-reduce matrix multiplication.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param k number of columns in A and rows in B.
                 */
                void matmul_16_6_k(mini_jit::Kernel& kernel,
                                   int               k);
            } // namespace subkernels
        } // namespace matmul
    } // namespace kernels
}; // namespace mini_jit

#endif