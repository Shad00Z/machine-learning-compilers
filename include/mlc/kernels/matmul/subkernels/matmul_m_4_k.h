#ifndef MINI_JIT_MATMUL_M_6_K_H
#define MINI_JIT_MATMUL_M_6_K_H

#include <mlc/Kernel.h>

namespace mini_jit
{
    namespace kernels
    {
        namespace matmul
        {
            namespace subkernels
            {
                namespace internal
                {
                    /**
                     * @brief Generates an M loop for matrix multiplication where M % 16 = 0.
                     * @param kernel Kernel object to be filled with instructions.
                     * @param mLoopIterations number of M loop iterations.
                     * @param k number of columns in A and rows in B.
                     */
                    void generateM16N4Loop(mini_jit::Kernel& kernel,
                                           int               mLoopIterations,
                                           int               k);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 1.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM1N4Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM2N4Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 3.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM3N4Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 4.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM4N4Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 5.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM5N4Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 6.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM6N4Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 7.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM7N4Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 8.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM8N4Loop(mini_jit::Kernel& kernel);
                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 9.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM9N4Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 10.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM10N4Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 11.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM11N4Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 12.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM12N4Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 13.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM13N4Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 14.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM14N4Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 15.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM15N4Loop(mini_jit::Kernel& kernel);
                } // namespace internal

                /**
                 * @brief Kernel for batch-reduce matrix multiplication.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m number of rows in A and C.
                 * @param k number of columns in A and rows in B.
                 */
                void matmul_m_4_k(mini_jit::Kernel& kernel,
                                  int               m,
                                  int               k);
            } // namespace subkernels
        } // namespace matmul
    } // namespace kernels
}; // namespace mini_jit

#endif