#include "Brgemm.h"
#include "Kernel.h"
#include "kernels/matmul_16_6_1.h"
#include "kernels/matmul_16_6_k.h"
#include <iostream>

/**
 * @brief Generate a kernel for batch-reduce matrix multiplication if
 *        all conditions are met.
 */
mini_jit::Brgemm::error_t mini_jit::Brgemm::generate( uint32_t m,
                                                      uint32_t n,
                                                      uint32_t k,
                                                      uint32_t br_size, 
                                                      uint32_t trans_a,
                                                      uint32_t trans_b,
                                                      uint32_t trans_c,
                                                      dtype_t  dtype )
{
    /**
     * Currently supported:
     * M = 16
     * N =  6
     * K =  1
     * BR_SIZE: Not defined
     * trans_a, trans_b, trans_c: Column-major
     * dtype: fp32
     */

    if( m != 16 )
    {
        std::cout << ( "M must be 16" ) << std::endl;
        return mini_jit::Brgemm::error_t::wrong_m_dimension;
    }
    else if ( n != 6 )
    {
        std::cout << ( "N must be 6" ) << std::endl;
        return mini_jit::Brgemm::error_t::wrong_n_dimension;
    }
    else if ( k == 0 )
    {
        std::cout << ( "K must not be 0" ) << std::endl;
        return mini_jit::Brgemm::error_t::wrong_k_dimension;
    }
    else if ( br_size != 4 ) // for now, we don't check br_size
    {
        std::cout << ( "BR_SIZE must be 4" ) << std::endl;
        return mini_jit::Brgemm::error_t::wrong_batch_reduce_size;
    }
    else if ( trans_a != 0 || trans_b != 0 || trans_c != 0 )
    {
        std::cout << ( "Matrix ordering must be column-major" ) << std::endl;
        return mini_jit::Brgemm::error_t::wrong_matrix_ordering_format;
    }
    else if ( dtype != dtype_t::fp32 )
    {
        std::cout << ( "Matrix data type must be fp32" ) << std::endl;
        return mini_jit::Brgemm::error_t::wrong_matrix_datatype;
    }
    else
    {
        if ( k == 1 )
        {
            mini_jit::kernels::matmul_16_6_1( m_kernel );
        }
        else
        {
            mini_jit::kernels::matmul_16_6_k( m_kernel );
        }

        // Valid matrix kernel
        return mini_jit::Brgemm::error_t::success;
    }
}

mini_jit::Brgemm::kernel_t mini_jit::Brgemm::get_kernel() const 
{
    return reinterpret_cast<kernel_t>(const_cast<void*>(m_kernel.get_kernel()));
}