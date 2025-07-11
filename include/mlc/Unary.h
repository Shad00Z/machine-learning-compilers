#ifndef MINI_JIT_UNARY_H
#define MINI_JIT_UNARY_H

#include <cstdint>
#include <mlc/Kernel.h>
#include <mlc/types.h>

namespace mini_jit
{
    class Unary;
}

class mini_jit::Unary
{
private:
    /// kernel
    Kernel* m_kernel = nullptr;
    void*   m_extra  = nullptr; // pointer to extra/context data

    /**
     * @brief Creates a new kernel and deletes the existing one.
     */
    void reset_kernel();

public:
    /// @brief Destructor
    ~Unary() noexcept
    {
        if (m_kernel)
        {
            delete m_kernel;
            m_kernel = nullptr;
        }
    }

    /**
     * @brief Generate a kernel for a unary primitive.
     * @param m       Number of rows in A and B.
     * @param n       Number of columns in A and B.
     * @param trans_b 0 if B is stored in column-major order, 1 if B is stored in row-major order.
     * @param dtype   Data type of the matrices.
     * @param ptype   Primitive type.
     * @return error_t::success on success, another error_t value otherwise.
     **/
    error_t generate(uint32_t          m,
                     uint32_t          n,
                     uint32_t          trans_b,
                     mini_jit::dtype_t dtype,
                     mini_jit::ptype_t ptype);

    /*
     * Generalized kernel type.
     * The kernel is a function that takes the following parameters:
     * - a:    Pointer to column-major matrix A, nullptr if zero kernel.
     * - b:    Pointer to matrix B.
     * - ld_a: Leading dimension of A.
     * - ld_b: Leading dimension of B.
     * - extra: Optional pointer to extra/context data (e.g., lookup table).
     */
    using kernel_t = void (*)(void const* a,
                              void*       b,
                              int64_t     ld_a,
                              int64_t     ld_b,
                              void*       extra);

    /**
     * @brief Get the generated kernel: B := op(A).
     * @return pointer to the generated kernel.
     **/
    kernel_t get_kernel() const;

    /**
     * @brief Set extra/context pointer for kernels that need it (e.g., lookup table).
     */
    void set_extra(void* extra);

    /**
     * @brief Get the extra/context pointer.
     */
    void* get_extra() const;
};
#endif
