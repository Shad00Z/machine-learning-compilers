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

    // Assigning to member
    m_dim_types.assign(dim_types.begin(), dim_types.end());
    m_exec_types.assign(exec_types.begin(), exec_types.end());
    m_loop_sizes.assign(dim_sizes.begin(), dim_sizes.end());
    m_strides_in0.assign(strides_in0.begin(), strides_in0.end());
    m_strides_in1.assign(strides_in1.begin(), strides_in1.end());
    m_strides_out.assign(strides_out.begin(), strides_out.end());

    // First "prim" position
    auto it = std::find(exec_types.begin(), exec_types.end(), exec_t::prim);

    if (it != exec_types.end())
    {
        m_id_first_primitive_loop = std::distance(exec_types.begin(), it);
    }
    else
    {
        m_id_first_primitive_loop = 0;
    }

    // Check data type
    if (dtype != dtype_t::fp32)
    {
        return error_t::wrong_dtype;
    }

    // Assigning to member
    m_dtype = dtype;

    mini_jit::Unary l_unary;
    mini_jit::Brgemm l_brgemm;

    if (ptype_t::none != prim_first_touch)
    {
        l_unary.generate(dim_sizes[0], dim_sizes[1], 0, dtype, prim_first_touch);
        m_prim_first_touch_kernel = l_unary.get_kernel();
    }
    m_prim_first_touch = prim_first_touch;

    if (ptype_t::none != prim_main)
    {
        l_brgemm.generate(dim_sizes[0], dim_sizes[1], dim_sizes[2], dim_sizes[3], 0, 0, 0, dtype);
        m_prim_main_kernel = l_brgemm.get_kernel();
    }
    m_prim_main = prim_main;

    if (ptype_t::none != prim_last_touch)
    {
        l_unary.generate(dim_sizes[0], dim_sizes[1], 0, dtype, prim_last_touch);
        m_prim_last_touch_kernel = l_unary.get_kernel();
    }
    m_prim_last_touch = prim_last_touch;

    return error_t::success;
}

void mini_jit::TensorOperation::execute(void const *tensor_in0,
                                        void const *tensor_in1,
                                        void *tensor_out)
{
    auto ptr_in0 = static_cast<char const *>(tensor_in0);
    auto ptr_in1 = static_cast<char const *>(tensor_in1);
    auto ptr_out = static_cast<char *>(tensor_out);

    execute_iter(0, ptr_in0, ptr_in1, ptr_out, true, true);
}

void mini_jit::TensorOperation::execute_iter(int64_t id_loop,
                                             char const *ptr_in0,
                                             char const *ptr_in1,
                                             char *ptr_out,
                                             bool first_access,
                                             bool last_access)
{
    int64_t l_size = m_loop_sizes[id_loop];
    int64_t l_stride_in0 = m_strides_in0[id_loop];
    int64_t l_stride_in1 = m_strides_in1[id_loop];
    int64_t l_stride_out = m_strides_out[id_loop];

    for (int64_t l_iter = 0; l_iter < l_size; l_iter++)
    {
        bool first_access = 0 ? id_loop == 0 : 1;
        bool last_access = 1 ? id_loop == 0 && l_iter == l_size - 1 : 0;

        // Calculate new pointer positions
        char const *sub_ptr_in0 = ptr_in0 + l_iter * m_loop_sizes[2] * 4;                   // 4 for fp32
        char const *sub_ptr_in1 = ptr_in1 + m_loop_sizes[1] * l_iter * m_loop_sizes[2] * 4; // 4 for fp32
        char *sub_ptr_out = ptr_out + l_iter * m_loop_sizes[1] * m_loop_sizes[0] * 4;       // 4 for fp32

        // Recursive Call
        if (id_loop + 1 < m_id_first_primitive_loop)
        {
            execute_iter(id_loop + 1, sub_ptr_in0, sub_ptr_in1, sub_ptr_out, first_access, last_access);
        }
        else
        {
            if (m_prim_first_touch != ptype_t::none)
            {
                sub_ptr_in1 = nullptr;
            }

            // Main
            m_prim_main_kernel(sub_ptr_in0,                      // A
                               sub_ptr_in1,                      // B
                               sub_ptr_out,                      // C
                               m_loop_sizes[3],                  // m
                               m_loop_sizes[4],                  // n
                               m_loop_sizes[5],                  // k
                               m_loop_sizes[3],                  // ldA
                               m_loop_sizes[2] * m_loop_sizes[5] // ldB
                                                                 // missing: m_loop_sizes[0] * m_loop_sizes[3] // ldC
            );

            if (m_prim_last_touch != ptype_t::none)
            {
            }
        }
    }
}