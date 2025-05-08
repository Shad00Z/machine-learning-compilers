#include "Brgemm.h"
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
        throw std::invalid_argument( "M must be 16" );
        return mini_jit::Brgemm::error_t::wrong_m_dimension;
    }
    else if ( n != 6 )
    {
        throw std::invalid_argument( "N must be 6" );
        return mini_jit::Brgemm::error_t::wrong_n_dimension;
    }
    else if ( k != 1 )
    {
        throw std::invalid_argument( "K must be 1" );
        return mini_jit::Brgemm::error_t::wrong_k_dimension;
    }
    else if ( br_size != 4 ) // for now, we don't check br_size
    {
        throw std::invalid_argument( "BR_SIZE must be 4" );
        return mini_jit::Brgemm::error_t::wrong_batch_reduce_size;
    }
    else if ( trans_a != 0 || trans_b != 0 || trans_c != 0 )
    {
        throw std::invalid_argument( "Matrix ordering must be column-major" );
        return mini_jit::Brgemm::error_t::wrong_matrix_ordering_format;
    }
    else if ( dtype != dtype_t::fp32 )
    {
        throw std::invalid_argument( "Matrix data type must be fp32" );
        return mini_jit::Brgemm::error_t::wrong_matrix_datatype;
    }
    else
    {
        // Valid matrix kernel
        return mini_jit::Brgemm::error_t::success;
    }
}

// Return the generated kernel
mini_jit::Brgemm::kernel_t mini_jit::Brgemm::get_kernel() const
{
    return nullptr;
}
