#include "TensorOperation.h"

mini_jit::error_t mini_jit::TensorOperation::setup(dtype_t dtype,
                                                   ptype_t prim_first_touch,
                                                   ptype_t prim_main,
                                                   ptype_t prim_last_touch,
                                                   std::span<const dim_t> dim_types,
                                                   std::span<const exec_t> exec_types,
                                                   std::span<const int64_t> dim_sizes,
                                                   std::span<const int64_t> strides_in0,
                                                   std::span<const int64_t> strides_in1,
                                                   std::span<const int64_t> strides_out)
{
    // Check the number of dimensions
    if (dim_types.size() != dim_sizes.size() || dim_types.size() != strides_in0.size() || dim_types.size() != strides_in1.size() || dim_types.size() != strides_out.size())
    {
        return error_t::wrong_dimension;
    }

    // Check data type
    if (dtype != dtype_t::fp32)
    {
        return error_t::wrong_dtype;
    }

    mini_jit::Unary l_unary;
    mini_jit::Brgemm l_brgemm;

    if (ptype_t::none != prim_first_touch)
    {
        l_unary.generate(dim_sizes[0], dim_sizes[1], 0, dtype, prim_first_touch);
        m_prim_first_touch = l_unary.get_kernel();
    }

    if (ptype_t::none != prim_main)
    {
        l_brgemm.generate(dim_sizes[0], dim_sizes[1], dim_sizes[2], dim_sizes[3], 0, 0, 0, dtype);
        m_prim_main = l_brgemm.get_kernel();
    }

    if (ptype_t::none != prim_last_touch)
    {
        l_unary.generate(dim_sizes[0], dim_sizes[1], 0, dtype, prim_last_touch);
        m_prim_last_touch = l_unary.get_kernel();
    }

    return error_t::success;
}

void mini_jit::TensorOperation::execute(void const *tensor_in0,
                                        void const *tensor_in1,
                                        void *tensor_out)
{
}

void mini_jit::TensorOperation::execute_iter(int64_t id_loop,
                                             char const *ptr_in0,
                                             char const *ptr_in1,
                                             char *ptr_out,
                                             bool first_access,
                                             bool last_access)
{
}