#ifndef MINI_JIT_UNARY_RELU_TRANS_PRIMITIVE_H
#define MINI_JIT_UNARY_RELU_TRANS_PRIMITIVE_H

#include <cstdint>
#include <mlc/Kernel.h>

namespace mini_jit
{
    namespace kernels
    {
        namespace unary
        {
            namespace internal
            {
                /**
                 * @brief Kernel for transposing and performing relu on a 4x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reluM4N4(mini_jit::Kernel& kernel,
                              int               m);

                /**
                 * @brief Kernel for transposing and performing relu on a 3x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reluM3N4(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 2x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reluM2N4(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reluM1N4(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 4x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void reluM4N3(mini_jit::Kernel& kernel,
                              int               m);

                /**
                 * @brief Kernel for transposing and performing relu on a 4x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void reluM4N2(mini_jit::Kernel& kernel,
                              int               m);

                /**
                 * @brief Kernel for transposing and performing relu on a 4x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void reluM4N1(mini_jit::Kernel& kernel,
                              int               m);

                /**
                 * @brief Kernel for transposing and performing relu on a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reluM1N4(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 1x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reluM1N3(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 1x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reluM1N2(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 1x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reluM1N1(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reluM1N4(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 2x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reluM2N3(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 2x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reluM2N2(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 2x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reluM2N1(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reluM1N4(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 4x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void reluM3N3(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 4x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void reluM3N2(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing and performing relu on a 4x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void reluM3N1(mini_jit::Kernel& kernel);
            } // namespace internal
            /**
             * @brief Kernel that applies ReLU activation function to a matrix while transposing the output.
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void relu_trans(mini_jit::Kernel& kernel,
                            int               m,
                            int               n);
        } // namespace unary
    } // namespace kernels
}; // namespace mini_jit

#endif // MINI_JIT_UNARY_RELU_TRANS_PRIMITIVE_H