#ifndef MINI_JIT_TENSOR_OPERATION_H
#define MINI_JIT_TENSOR_OPERATION_H

#include <cstdint>
#include <span>
#include <vector>

#include "types.h"
#include "Unary.h"
#include "Brgemm.h"

namespace mini_jit
{
    class TensorOperation;
}

class mini_jit::TensorOperation
{
private:
    /// used dtype
    mini_jit::dtype_t m_dtype;

    /// first touch primitive type
    mini_jit::ptype_t m_prim_first_touch;
    /// first touch primitive
    mini_jit::Unary::kernel_t m_prim_first_touch_kernel;

    /// first touch primitive type
    mini_jit::ptype_t m_prim_main;
    /// main primitive
    mini_jit::Brgemm::kernel_t m_prim_main_kernel;

    /// first touch primitive type
    mini_jit::ptype_t m_prim_last_touch;
    /// last touch primitive;
    mini_jit::Unary::kernel_t m_prim_last_touch_kernel;

    /// dimension types of the loops (m, n, k)
    std::vector<dim_t> m_dim_types;
    /// execution types of the loops (seq, shared, prim)
    std::vector<exec_t> m_exec_types;
    /// sizes of the dimensions (loops)
    std::vector<int64_t> m_loop_sizes;
    /// strides of the first input tensor
    std::vector<int64_t> m_strides_in0;
    /// strides of the second input tensor
    std::vector<int64_t> m_strides_in1;
    /// strides of the output tensor
    std::vector<int64_t> m_strides_out;
    /// location of first primitive loop
    int64_t m_id_first_primitive_loop;

public:
    /**
     * Setup for a binary tensor contraction or a unary tensor operation.
     *
     * @param dtype             Datatype of all tensor elements.
     * @param prim_first_touch  Type of the first touch primitive.
     * @param prim_main         Type of the main primitive.
     * @param prim_last_touch   Type of the last touch primitive.
     * @param dim_types         Dimension type of the loops (c, m, n, or k).
     * @param exec_types        Execution type of the loops (seq, shared, or prim).
     * @param dim_sizes         Sizes of the dimensions.
     * @param strides_in0       Strides of the first input tensor.
     * @param strides_in1       Strides of the second input tensor (ignored if unary).
     * @param strides_out       Strides of the output tensor.
     * @return error_t::success on success, another error_t value otherwise.
     **/
    error_t
    setup(dtype_t dtype,
          ptype_t prim_first_touch,
          ptype_t prim_main,
          ptype_t prim_last_touch,
          std::span<const dim_t> dim_types,
          std::span<const exec_t> exec_types,
          std::span<const int64_t> dim_sizes,
          std::span<const int64_t> strides_in0,
          std::span<const int64_t> strides_in1,
          std::span<const int64_t> strides_out);

    /**
     * Execute the tensor operation.
     *
     * @param tensor_in0 First input tensor.
     * @param tensor_in1 Second input tensor (use nullptr if unary).
     * @param tensor_out Output tensor.
     **/
    void execute(void const *tensor_in0,
                 void const *tensor_in1,
                 void *tensor_out);

    /**
     * General-purpose loop implementation featuring first and last touch operations.
     * No threading is applied.
     *
     * @param id_loop      Dimension id of the loop which is executed.
     * @param ptr_in0      Pointer to the first input tensor's data.
     * @param ptr_in1      Pointer to the second input tensor's data (use nullptr if unary).
     * @param ptr_out      Pointer to the output tensor's data.
     * @param first_access True if first time accessing data of output tensor.
     * @param last_access  True if last time accessing data of output tensor.
     **/
    void execute_iter(int64_t id_loop,
                      char const *ptr_in0,
                      char const *ptr_in1,
                      char *ptr_out,
                      bool first_access,
                      bool last_access);
};

#endif