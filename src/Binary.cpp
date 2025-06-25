#include "Kernel.h"
#include "Binary.h"
#include "kernels/binary/all_binary_primitives.h"
#include <iostream>

mini_jit::error_t mini_jit::Binary::generate(uint32_t m,
                                             uint32_t n,
                                             uint32_t trans_c,
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
    else if (trans_c != 0 && trans_c != 1)
    {
        std::cout << ("Invalid trans_c parameter value") << std::endl;
        return error_t::wrong_matrix_ordering_format;
    }

    reset_kernel();

    switch (ptype)
    {
    
    default:
        std::cout << ("Invalid primitive type") << std::endl;
        return error_t::wrong_ptype;
    }

    return error_t::success;
}

mini_jit::Binary::kernel_t mini_jit::Binary::get_kernel() const
{
    return reinterpret_cast<kernel_t>(const_cast<void *>(m_kernel->get_kernel()));
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