#include "Brgemm.h"
#include "Kernel.h"
#include "kernels/matmul/matmul_m_n_k.h"
#include "kernels/matmul/matmul_br_m_n_k.h"
#include <iostream>

mini_jit::error_t mini_jit::Brgemm::generate(uint32_t m,
                                             uint32_t n,
                                             uint32_t k,
                                             uint32_t br_size,
                                             uint32_t trans_a,
                                             uint32_t trans_b,
                                             uint32_t trans_c,
                                             dtype_t dtype)
{
    /**
     * Currently supported:
     * BR_SIZE: Not defined
     * trans_a, trans_b, trans_c: Column-major
     * dtype: fp32
     */

    if (m <= 0)
    {
        std::cout << ("M must be greater than 0") << std::endl;
        return error_t::wrong_dimension;
    }
    else if (m >= 1025)
    {
        std::cout << ("M must not greater than 1024") << std::endl;
        return error_t::wrong_dimension;
    }
    else if (n <= 0)
    {
        std::cout << ("N must be greater than 0") << std::endl;
        return error_t::wrong_dimension;
    }
    else if (n >= 1025)
    {
        std::cout << ("N must not be greater than 1024") << std::endl;
        return error_t::wrong_dimension;
    }
    else if (k <= 0)
    {
        std::cout << ("K must be greater than 0") << std::endl;
        return error_t::wrong_dimension;
    }
    else if (k >= 2049)
    {
        std::cout << ("K must not be greater than 2048") << std::endl;
        return error_t::wrong_dimension;
    }
    else if (br_size <= 0)
    {
        std::cout << ("BR_SIZE must greater than 0") << std::endl;
        return error_t::wrong_dimension;
    }
    else if (br_size >= 1025)
    {
        std::cout << ("BR_SIZE must not be greater than 1024") << std::endl;
        return error_t::wrong_dimension;
    }
    else if (trans_a != 0 || trans_b != 0 || trans_c != 0)
    {
        std::cout << ("Matrix ordering must be column-major") << std::endl;
        return error_t::wrong_matrix_ordering_format;
    }
    else if (dtype != dtype_t::fp32)
    {
        std::cout << ("Matrix data type must be fp32") << std::endl;
        return error_t::wrong_dtype;
    }
    else
    {
        if (br_size == 1)
        {
            mini_jit::kernels::matmul::matmul_m_n_k(*m_kernel, m, n, k);
        }
        else
        {
            mini_jit::kernels::matmul::matmul_br_m_n_k(*m_kernel, m, n, k, br_size);
        }

        // Valid matrix kernel
        return error_t::success;
    }
}

mini_jit::Brgemm::kernel_t mini_jit::Brgemm::get_kernel() const
{
    return reinterpret_cast<kernel_t>(const_cast<void *>(m_kernel->get_kernel()));
}