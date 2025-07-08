#ifndef MINI_JIT_IR_DIMENSION_H
#define MINI_JIT_IR_DIMENSION_H

#include <cstdint>
#include <mlc/types.h>
#include <stdexcept>

namespace mini_jit
{
    namespace ir
    {
        /**
         * @brief The Dimension struct represents a dimension in a tensor operation.
         * It contains information about the type of dimension (M, N, K), execution type (Prim, Seq, Shared),
         * size, and strides for the input and output tensors.
         */
        struct Dimension
        {
            //! Type of the dimension (M, N, K)
            dim_t type = dim_t::m;
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

            /**
             * @brief Construct a new Dimension object.
             *
             * @param type Type of the dimension (M, N, K).
             * @param exec_type Execution type (Prim, Seq, Shared, ...).
             * @param size Size of the dimension.
             * @param stride_in0 Stride in the first input tensor.
             * @param stride_in1 Stride in the second input tensor.
             * @param stride_out Stride in the output tensor.
             */
            Dimension(dim_t   type,
                      exec_t  exec_type,
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
    } // namespace ir
} // namespace mini_jit

#endif