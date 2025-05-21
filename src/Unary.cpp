#include "Kernel.h"
#include "Unary.h"
#include "kernels/unary/zero_primitive.h"
#include "kernels/unary/relu_primitive.h"
#include <iostream>

mini_jit::Unary::error_t mini_jit::Unary::generate( uint32_t m,
                                                    uint32_t n,
                                                    uint32_t trans_b,
                                                    dtype_t  dtype,
                                                    ptype_t  ptype  )
{
    if( m <= 0 )
    {
        std::cout << ( "M must be greater than 0" ) << std::endl;
        return mini_jit::Unary::error_t::wrong_m_dimension;
    }
    else if ( m > 2048 )
    {
        std::cout << ( "M must not be greater than 2048" ) << std::endl;
        return mini_jit::Unary::error_t::wrong_m_dimension;
    }
    else if ( n <= 0 )
    {
        std::cout << ( "N must be greater than 0" ) << std::endl;
        return mini_jit::Unary::error_t::wrong_n_dimension;
    }
    else if ( n > 2048 )
    {
        std::cout << ( "N must not be greater than 2048" ) << std::endl;
        return mini_jit::Unary::error_t::wrong_n_dimension;
    }

    switch (ptype)
    {
    case ptype_t::zero:
        mini_jit::kernels::unary::zero(m_kernel, m, n, trans_b);
        break;
    case ptype_t::identity:
        // call identity primitive kernel
        break;
    case ptype_t::relu:
        if(trans_b != 0)
        {
            std::cout << ( "ReLU kernel does not support transposition" ) << std::endl;
            return mini_jit::Unary::error_t::operation_not_supported;
        }
        else
        {
            mini_jit::kernels::unary::relu(m_kernel, m, n);
        }
        break;
    default:
        std::cout << ( "Invalid primitive type" ) << std::endl;
        return mini_jit::Unary::error_t::wrong_ptype;
    }

    return mini_jit::Unary::error_t::success;
}

mini_jit::Unary::kernel_t mini_jit::Unary::get_kernel() const 
{
    return reinterpret_cast<kernel_t>(const_cast<void*>(m_kernel.get_kernel()));
}