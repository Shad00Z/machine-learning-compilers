#ifndef MINI_JIT_BINARY_H
#define MINI_JIT_BINARY_H

#include "types.h"
#include "Kernel.h"
#include <cstdint>

namespace mini_jit
{
    class Binary;
}

class mini_jit::Binary
{
private:
    /// kernel
    Kernel *m_kernel = nullptr;

    /**
     * @brief Creates a new kernel and deletes the existing one.
     */
    void reset_kernel();

public:
    /// @brief Destructor
    ~Binary() noexcept
    {
        if (m_kernel)
        {
            delete m_kernel;
            m_kernel = nullptr;
        }
    }

    /**
     * @brief Generate a kernel for a binary primitive.
     * @param m       Number of rows.
     * @param n       Number of columns.
     * @param trans_c 0 if C is stored in column-major order, 1 if C is stored in row-major order.
     * @param dtype   Data type of the matrices.
     * @param ptype   Primitive type.
     * @return error_t::success on success, another error_t value otherwise.
     **/
    error_t generate(uint32_t m,
                     uint32_t n,
                     uint32_t trans_c,
                     mini_jit::dtype_t dtype,
                     mini_jit::ptype_t ptype);

    /*
     * Kernel type.
     * The kernel is a function that takes the following parameters:
     * - a:    Pointer to input matrix A.
     * - b:    Pointer to input matrix B.
     * - c:    Pointer to output matrix C.
     * - ld_a: Leading dimension of A.
     * - ld_b: Leading dimension of B.
     * - ld_c: Leading dimension of C.
     */
    using kernel_t = void (*)(void const *a,
                              void *b,
                              int64_t ld_a,
                              int64_t ld_b);

    /**
     * @brief Get the generated kernel: C := op(A, B).
     * @return pointer to the generated kernel.
     **/
    kernel_t get_kernel() const;
};
#endif
