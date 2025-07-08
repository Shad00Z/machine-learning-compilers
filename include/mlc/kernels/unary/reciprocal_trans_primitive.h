#ifndef MINI_JIT_UNARY_RECIPROCAL_TRANS_PRIMITIVE_H
#define MINI_JIT_UNARY_RECIPROCAL_TRANS_PRIMITIVE_H

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
                 * @brief Kernel for transposing a 4x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reciprocalM4N4(mini_jit::Kernel& kernel,
                                    int               m);

                /**
                 * @brief Kernel for transposing a 3x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reciprocalM3N4(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 2x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reciprocalM2N4(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reciprocalM1N4(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 4x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void reciprocalM4N3(mini_jit::Kernel& kernel,
                                    int               m);

                /**
                 * @brief Kernel for transposing a 4x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void reciprocalM4N2(mini_jit::Kernel& kernel,
                                    int               m);

                /**
                 * @brief Kernel for transposing a 4x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void reciprocalM4N1(mini_jit::Kernel& kernel,
                                    int               m);

                /**
                 * @brief Kernel for transposing a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reciprocalM1N4(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 1x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reciprocalM1N3(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 1x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reciprocalM1N2(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 1x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reciprocalM1N1(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reciprocalM1N4(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 2x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reciprocalM2N3(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 2x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reciprocalM2N2(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 2x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reciprocalM2N1(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void reciprocalM1N4(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 4x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void reciprocalM3N3(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 4x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void reciprocalM3N2(mini_jit::Kernel& kernel);

                /**
                 * @brief Kernel for transposing a 4x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void reciprocalM3N1(mini_jit::Kernel& kernel);
            } // namespace internal
            /**
             * @brief Kernel for performing the reciprocal operation on a matrix while transposing the output.
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void reciprocal_trans(mini_jit::Kernel& kernel,
                                  int               m,
                                  int               n);
        } // namespace unary
    } // namespace kernels
}; // namespace mini_jit

#endif // MINI_JIT_UNARY_RECIPROCAL_TRANS_PRIMITIVE_H