#ifndef MINI_JIT_UNARY_DECREMENT_TRANS_PRIMITIVE_H
#define MINI_JIT_UNARY_DECREMENT_TRANS_PRIMITIVE_H

#include "Kernel.h"

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
                void decrementM4N4( mini_jit::Kernel &kernel, 
                                    int m );

                /**
                 * @brief Kernel for transposing a 3x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void decrementM3N4( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 2x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void decrementM2N4( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void decrementM1N4( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 4x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void decrementM4N3( mini_jit::Kernel &kernel,
                                    int m );

                /**
                 * @brief Kernel for transposing a 4x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void decrementM4N2( mini_jit::Kernel &kernel,
                                    int m );

                /**
                 * @brief Kernel for transposing a 4x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void decrementM4N1( mini_jit::Kernel &kernel,
                                    int m );

                /**
                 * @brief Kernel for transposing a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void decrementM1N4( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 1x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void decrementM1N3( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 1x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void decrementM1N2( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 1x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void decrementM1N1( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void decrementM1N4( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 2x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void decrementM2N3( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 2x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void decrementM2N2( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 2x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void decrementM2N1( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void decrementM1N4( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 4x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void decrementM3N3( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 4x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void decrementM3N2( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 4x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void decrementM3N1( mini_jit::Kernel &kernel );
            }
            /**
             * @brief Kernel for performing the decrement operation on a matrix while transposing the output.
             * 
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void decrement_trans( mini_jit::Kernel &kernel, 
                                  int m, 
                                  int n );
        }
    }
};

#endif // MINI_JIT_UNARY_DECREMENT_TRANS_PRIMITIVE_H