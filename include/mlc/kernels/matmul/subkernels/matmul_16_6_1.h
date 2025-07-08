#ifndef MINI_JIT_MATMUL_16_6_1_H
#define MINI_JIT_MATMUL_16_6_1_H

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
                 */
                void matmul_16_6_1(mini_jit::Kernel& kernel);
            } // namespace subkernels
        } // namespace matmul
    } // namespace kernels
}; // namespace mini_jit

#endif