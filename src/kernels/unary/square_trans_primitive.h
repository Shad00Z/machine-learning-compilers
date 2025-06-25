#ifndef MINI_JIT_UNARY_SQUARE_TRANS_PRIMITIVE_H
#define MINI_JIT_UNARY_SQUARE_TRANS_PRIMITIVE_H

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
                void squareM4N4( mini_jit::Kernel &kernel, 
                                   int m );

                /**
                 * @brief Kernel for transposing a 3x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void squareM3N4( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 2x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void squareM2N4( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void squareM1N4( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 4x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void squareM4N3( mini_jit::Kernel &kernel,
                                   int m );

                /**
                 * @brief Kernel for transposing a 4x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void squareM4N2( mini_jit::Kernel &kernel,
                                   int m );

                /**
                 * @brief Kernel for transposing a 4x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void squareM4N1( mini_jit::Kernel &kernel,
                                   int m );

                /**
                 * @brief Kernel for transposing a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void squareM1N4( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 1x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void squareM1N3( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 1x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void squareM1N2( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 1x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void squareM1N1( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void squareM1N4( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 2x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void squareM2N3( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 2x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void squareM2N2( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 2x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void squareM2N1( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 1x4 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 */
                void squareM1N4( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 4x3 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void squareM3N3( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 4x2 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void squareM3N2( mini_jit::Kernel &kernel );

                /**
                 * @brief Kernel for transposing a 4x1 matrix.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m Number of loop cycles to perform.
                 */
                void squareM3N1( mini_jit::Kernel &kernel );
            }
            /**
             * @brief Kernel for performing the square operation on a matrix while transposing the output.
             * @param kernel Kernel object to be filled with instructions.
             * @param m number of rows in the matrix.
             * @param n number of columns in the matrix.
             */
            void square_trans( mini_jit::Kernel &kernel, 
                                 int m, 
                                 int n );
        }
    }
};

#endif // MINI_JIT_UNARY_SQUARE_TRANS_PRIMITIVE_H