#include <iostream>
#include <mlc/Binary.h>
#include <mlc/Kernel.h>
#include <mlc/kernels/binary/all_binary_primitives.h>

mini_jit::error_t mini_jit::Binary::generate(uint32_t m,
                                             uint32_t n,
                                             uint32_t trans_c,
                                             dtype_t  dtype,
                                             ptype_t  ptype)
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
    else if (trans_c != 0 && trans_c != 1)
    {
        std::cout << ("Invalid trans_c parameter value") << std::endl;
        return error_t::wrong_matrix_ordering_format;
    }

    reset_kernel();

    switch (ptype)
    {
    case ptype_t::add:
        if (0 == trans_c)
        {
            mini_jit::kernels::binary::add(*m_kernel, m, n);
        }
        else if (1 == trans_c)
        {
            std::cout << "Transposition for add primitive is not supported" << std::endl;
            return error_t::operation_not_supported;
        }
        break;
    case ptype_t::sub:
        if (0 == trans_c)
        {
            mini_jit::kernels::binary::sub(*m_kernel, m, n);
        }
        else if (1 == trans_c)
        {
            std::cout << "Transposition for sub primitive is not supported" << std::endl;
            return error_t::operation_not_supported;
        }
        break;
    case ptype_t::mul:
        if (0 == trans_c)
        {
            mini_jit::kernels::binary::mul(*m_kernel, m, n);
        }
        else if (1 == trans_c)
        {
            std::cout << "Transposition for mul primitive is not supported" << std::endl;
            return error_t::operation_not_supported;
        }
        break;
    case ptype_t::div:
        if (0 == trans_c)
        {
            mini_jit::kernels::binary::div(*m_kernel, m, n);
        }
        else if (1 == trans_c)
        {
            std::cout << "Transposition for div primitive is not supported" << std::endl;
            return error_t::operation_not_supported;
        }
        break;
    case ptype_t::min:
        if (0 == trans_c)
        {
            mini_jit::kernels::binary::min(*m_kernel, m, n);
        }
        else if (1 == trans_c)
        {
            std::cout << "Transposition for min primitive is not supported" << std::endl;
            return error_t::operation_not_supported;
        }
        break;
    case ptype_t::max:
        if (0 == trans_c)
        {
            mini_jit::kernels::binary::max(*m_kernel, m, n);
        }
        else if (1 == trans_c)
        {
            std::cout << "Transposition for max primitive is not supported" << std::endl;
            return error_t::operation_not_supported;
        }
        break;
    default:
        std::cout << ("Invalid primitive type") << std::endl;
        return error_t::wrong_ptype;
    }

    return error_t::success;
}

mini_jit::Binary::kernel_t mini_jit::Binary::get_kernel() const
{
    return reinterpret_cast<kernel_t>(const_cast<void*>(m_kernel->get_kernel()));
}

void mini_jit::Binary::reset_kernel()
{
    if (m_kernel)
    {
        delete m_kernel;
        m_kernel = nullptr;
    }
    m_kernel = new mini_jit::Kernel();
}