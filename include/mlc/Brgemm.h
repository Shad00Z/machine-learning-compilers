#ifndef MINI_JIT_BRGEMM_H
#define MINI_JIT_BRGEMM_H

#include <cstdint>
#include <mlc/Kernel.h>
#include <mlc/types.h>

namespace mini_jit
{
    class Brgemm;
}

class mini_jit::Brgemm
{
private:
    /// kernel
    Kernel* m_kernel = nullptr;

    /**
     * @brief Creates a new kernel and deletes the existing one.
     */
    void reset_kernel();

public:
    /// @brief Destructor
    ~Brgemm() noexcept
    {
        if (m_kernel)
        {
            delete m_kernel;
            m_kernel = nullptr;
        }
    }

    /**
     * @brief Generate a kernel for batch-reduce matrix multiplication.
     * @param m number of rows in A and C.
     * @param n number of columns in B and C.
     * @param k number of columns in A and rows in B.
     * @param br_size batch-reduce size.
     * @param trans_a 0 if A is stored in column-major order, 1 if A is stored in row-major order.
     * @param trans_b 0 if B is stored in column-major order, 1 if B is stored in row-major order.
     * @param trans_c 0 if C is stored in column-major order, 1 if C is stored in row-major order.
     * @param dtype data type of the matrices.
     * @return error_t::success on success, another error_t value otherwise.
     **/
    error_t generate(uint32_t          m,
                     uint32_t          n,
                     uint32_t          k,
                     uint32_t          br_size,
                     uint32_t          trans_a,
                     uint32_t          trans_b,
                     uint32_t          trans_c,
                     mini_jit::dtype_t dtype);

    /*
     * Kernel type.
     * The kernel is a function that takes the following parameters:
     * - a: Pointer to first of a batch of A matrices.
     * - b: Pointer to first of a batch of B matrices.
     * - c: Pointer to C matrix.
     * - ld_a: Leading dimension of A.
     * - ld_b: Leading dimension of B.
     * - ld_c: Leading dimension of C.
     * - br_stride_a: Stride (in elements, not bytes) between A matrices.
     * - br_stride_b: Stride (in elements, not bytes) between B matrices.
     */
    using kernel_t = void (*)(void const* a,
                              void const* b,
                              void*       c,
                              int64_t     ld_a,
                              int64_t     ld_b,
                              int64_t     ld_c,
                              int64_t     br_stride_a,
                              int64_t     br_stride_b);

    /**
     * @brief Get the generated kernel: C += sum_i(A_i * B_i).
     * @return pointer to the generated kernel.
     **/
    kernel_t get_kernel() const;
};

#endif