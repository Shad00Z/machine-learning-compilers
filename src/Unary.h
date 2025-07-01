#ifndef MINI_JIT_UNARY_H
#define MINI_JIT_UNARY_H

#include "types.h"
#include "Kernel.h"
#include <cstdint>

namespace mini_jit
{
    class Unary;
}

class mini_jit::Unary
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
    error_t generate(uint32_t m,
                     uint32_t n,
                     uint32_t trans_b,
                     mini_jit::dtype_t dtype,
                     mini_jit::ptype_t ptype);

    /*
     * Kernel type.
     * The kernel is a function that takes the following parameters:
     * - a:    Pointer to column-major matrix A, nullptr if zero kernel.
     * - b:    Pointer to matrix B.
     * - ld_a: Leading dimension of A.
     * - ld_b: Leading dimension of B.
     */
    using kernel_t = void (*)(void const *a,
                              void *b,
                              int64_t ld_a,
                              int64_t ld_b);

    /*
     * Kernel type.
     * The kernel is a function that takes the following parameters:
     * - a:    Pointer to column-major matrix A, nullptr if zero kernel.
     * - b:    Pointer to matrix B.
     * - tab:  Pointer to lookup table.
     * - ld_a: Leading dimension of A.
     * - ld_b: Leading dimension of B.
     */
    using kernel_t_sig = void (*)(void const *a,
                                  void *b,
                                  void *tab,
                                  int64_t ld_a,
                                  int64_t ld_b);

    /**
     * @brief Get the generated kernel: B := op(A).
     * @return pointer to the generated kernel.
     **/
    kernel_t get_kernel() const;
};
#endif
