#include "Kernel.h"
#include "Unary.h"
#include "kernels/unary/all_unary_primitives.h"
#include <iostream>

mini_jit::error_t mini_jit::Unary::generate(uint32_t m,
                                            uint32_t n,
                                            uint32_t trans_b,
                                            dtype_t dtype,
                                            ptype_t ptype)
{
    if (m <= 0)
    {
        std::cout << ("M must be greater than 0") << std::endl;
        return error_t::wrong_dimension;
    }
    else if (m > 2048)
    {
        std::cout << ("M must not be greater than 2048") << std::endl;
        return error_t::wrong_dimension;
    }
    else if (n <= 0)
    {
        std::cout << ("N must be greater than 0") << std::endl;
        return error_t::wrong_dimension;
    }
    else if (n > 2048)
    {
        std::cout << ("N must not be greater than 2048") << std::endl;
        return error_t::wrong_dimension;
    }
    else if (trans_b != 0 && trans_b != 1)
    {
        std::cout << ("Invalid trans_b parameter value") << std::endl;
        return error_t::wrong_matrix_ordering_format;
    }

    reset_kernel();

    switch (ptype)
    {
    case ptype_t::zero:
        mini_jit::kernels::unary::zero(*m_kernel, m, n, trans_b);
        break;
    case ptype_t::identity:
        if (0 == trans_b)
        {
            mini_jit::kernels::unary::identity(*m_kernel, m, n);
        }
        else if (1 == trans_b)
        {
            mini_jit::kernels::unary::identity_trans(*m_kernel, m, n);
        }
        break;
    case ptype_t::relu:
        if (0 == trans_b)
        {
            mini_jit::kernels::unary::relu(*m_kernel, m, n);
        }
        else if (1 == trans_b)
        {
            mini_jit::kernels::unary::relu_trans(*m_kernel, m, n);
        }
        break;
    case ptype_t::square:
        if (0 == trans_b)
        {
            mini_jit::kernels::unary::square(*m_kernel, m, n);
        }
        else if (1 == trans_b)
        {
            mini_jit::kernels::unary::square_trans(*m_kernel, m, n);
        }
        break;
    case ptype_t::reciprocal:
        if (0 == trans_b)
        {
            mini_jit::kernels::unary::reciprocal(*m_kernel, m, n);
        }
        else if (1 == trans_b)
        {
            mini_jit::kernels::unary::reciprocal_trans(*m_kernel, m, n);
        }
        break;
    default:
        std::cout << ("Invalid primitive type") << std::endl;
        return error_t::wrong_ptype;
    }

    return error_t::success;
}

mini_jit::Unary::kernel_t mini_jit::Unary::get_kernel() const
{
    return reinterpret_cast<kernel_t>(const_cast<void *>(m_kernel->get_kernel()));
}

void mini_jit::Unary::reset_kernel()
{
    if (m_kernel)
    {
        delete m_kernel;
        m_kernel = nullptr;
    }
    m_kernel = new mini_jit::Kernel();
}