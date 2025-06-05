#ifndef MINI_JIT_IR_DIMENSION_H
#define MINI_JIT_IR_DIMENSION_H

#include "types.h"
#include <cstdint>
#include <stdexcept>

namespace mini_jit
{
    namespace ir
    {
        struct Dimension
        {
            //! Type of the dimension (M, N, K)
            dim_t type = dim_t::undefined;
            //! Execution type (Prim, Seq, Shared, ...)
            exec_t exec_type = exec_t::undefined;
            //! Dimension size
            int64_t size = 0;
            //! Stride in the first input tensor
            int64_t stride_in0 = 0;
            //! Stride in the second input tensor
            int64_t stride_in1 = 0;
            //! Stride in the output tensor
            int64_t stride_out = 0;

            Dimension(dim_t type,
                      exec_t exec_type,
                      int64_t size,
                      int64_t stride_in0,
                      int64_t stride_in1,
                      int64_t stride_out)
                : type(type),
                  exec_type(exec_type),
                  size(size),
                  stride_in0(stride_in0),
                  stride_in1(stride_in1),
                  stride_out(stride_out)
            {
                if (size <= 0)
                {
                    throw std::invalid_argument("Dimension size needs to be greater than 0");
                }
            }
        };
    }
}

#endif