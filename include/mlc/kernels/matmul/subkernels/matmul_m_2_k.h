#ifndef MINI_JIT_MATMUL_M_2_K_H
#define MINI_JIT_MATMUL_M_2_K_H

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
                    void generateM16N2Loop(mini_jit::Kernel& kernel,
                                           int               mLoopIterations,
                                           int               k);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 1 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM1N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 2 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM2N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 3 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM3N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 4 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM4N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 5 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM5N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 6 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM6N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 7 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM7N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 8 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM8N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 9 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM9N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 10 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM10N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 11 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM11N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 12 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM12N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 13 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM13N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 14 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM14N2Loop(mini_jit::Kernel& kernel);

                    /**
                     * @brief Generates an M loop for matrix multiplication where M = 15 and N = 2.
                     * @param kernel Kernel object to be filled with instructions.
                     */
                    void generateM15N2Loop(mini_jit::Kernel& kernel);
                } // namespace internal

                /**
                 * @brief Kernel for batch-reduce matrix multiplication.
                 * @param kernel Kernel object to be filled with instructions.
                 * @param m number of rows in A and C.
                 * @param k number of columns in A and rows in B.
                 */
                void matmul_m_2_k(mini_jit::Kernel& kernel,
                                  int               m,
                                  int               k);
            } // namespace subkernels
        } // namespace matmul
    } // namespace kernels
}; // namespace mini_jit

#endif