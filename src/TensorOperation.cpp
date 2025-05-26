#include "TensorOperation.h"
#include "kernels/matmul/matmul_m_n_k.h"
#include <iostream>
#include <algorithm>

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
    /////////////////////////////////////////////////////////////////////
    // Check the number of dimensions
    /////////////////////////////////////////////////////////////////////
    if (dim_types.size() != dim_sizes.size() || dim_types.size() != strides_in0.size() || dim_types.size() != strides_in1.size() || dim_types.size() != strides_out.size())
    {
        return error_t::wrong_dimension;
    }

    /////////////////////////////////////////////////////////////////////
    // Assign member variables
    /////////////////////////////////////////////////////////////////////
    m_dim_types.assign(dim_types.begin(), dim_types.end());
    m_exec_types.assign(exec_types.begin(), exec_types.end());
    m_loop_sizes.assign(dim_sizes.begin(), dim_sizes.end());
    m_strides_in0.assign(strides_in0.begin(), strides_in0.end());
    m_strides_in1.assign(strides_in1.begin(), strides_in1.end());
    m_strides_out.assign(strides_out.begin(), strides_out.end());
    m_dtype = dtype;
    m_idx_m = 0;
    m_idx_n = 0;
    m_idx_k = 0;

    /////////////////////////////////////////////////////////////////////
    // Check allowed data type
    /////////////////////////////////////////////////////////////////////
    if (dtype != dtype_t::fp32)
    {
        return error_t::wrong_dtype;
    }

    /////////////////////////////////////////////////////////////////////
    // Check allowed primitive types
    /////////////////////////////////////////////////////////////////////
    std::vector<ptype_t> allowed_first_touch_types = {ptype_t::none, ptype_t::zero, ptype_t::relu};
    std::vector<ptype_t> allowed_main_types = {ptype_t::none, ptype_t::identity, ptype_t::brgemm, ptype_t::gemm};
    std::vector<ptype_t> allowed_last_touch_types = {ptype_t::none, ptype_t::relu};

    if (std::find(allowed_first_touch_types.begin(), allowed_first_touch_types.end(), prim_first_touch) == allowed_first_touch_types.end())
    {
        return error_t::wrong_ptype;
    }
    if (std::find(allowed_main_types.begin(), allowed_main_types.end(), prim_main) == allowed_main_types.end())
    {
        return error_t::wrong_ptype;
    }
    if (std::find(allowed_last_touch_types.begin(), allowed_last_touch_types.end(), prim_last_touch) == allowed_last_touch_types.end())
    {
        return error_t::wrong_ptype;
    }

    /////////////////////////////////////////////////////////////////////
    // Find first "prim" position
    /////////////////////////////////////////////////////////////////////
    auto it = std::find(exec_types.begin(), exec_types.end(), exec_t::prim);

    if (it != exec_types.end())
    {
        m_id_first_primitive_loop = std::distance(exec_types.begin(), it);
    }
    else
    {
        m_id_first_primitive_loop = 0;
    }

    /////////////////////////////////////////////////////////////////////
    // Generate kernels
    /////////////////////////////////////////////////////////////////////
    if (prim_first_touch != ptype_t::none)
    {
        m_prim_first_touch_unary.generate(dim_sizes[3], 
                                          dim_sizes[4], 
                                          0, 
                                          dtype, 
                                          prim_first_touch);
    }
    if (prim_main == ptype_t::gemm)
    {
        m_prim_main_gemm.generate(dim_sizes[3], 
                                  dim_sizes[4], 
                                  dim_sizes[5], 
                                  1, 
                                  0, 
                                  0, 
                                  0, 
                                  dtype);
    }
    else if (prim_main == ptype_t::brgemm)
    {
        m_prim_main_gemm.generate(dim_sizes[3], 
                                  dim_sizes[4], 
                                  dim_sizes[5], 
                                  dim_sizes[2], 
                                  0, 
                                  0, 
                                  0, 
                                  dtype);
    }
    else if (prim_main == ptype_t::identity)
    {
        // TODO: check if transpose or not
        m_prim_main_unary.generate(dim_sizes[3], 
                                   dim_sizes[4], 
                                   0, 
                                   dtype, 
                                   prim_main);
    }
    if (prim_last_touch != ptype_t::none)
    {
        m_prim_last_touch_unary.generate(dim_sizes[3], 
                                         dim_sizes[4], 
                                         0, 
                                         dtype, 
                                         prim_last_touch);
    }

    m_prim_first_touch = prim_first_touch;
    m_prim_main = prim_main;
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

    for (int64_t l_iter = 0; l_iter < l_size; l_iter++)
    {
        switch (id_loop)
        {
        case 0:
            m_idx_m = l_iter;
            break;
        case 1:
            m_idx_n = l_iter;
            break;
        case 2:
            m_idx_k = l_iter;
            break;
        default:
            break;
        }

        bool is_first = false;
        bool is_last = false;
        if (m_idx_k == 0)
        {
            is_first = true;
        }
        else if (m_idx_k == l_size - 1)
        {
            is_last = true;
        }

        /*
         * r = m_loop_sizes[0]
         * p = m_loop_sizes[1]
         * t = m_loop_sizes[2]
         * s = m_loop_sizes[3]
         * q = m_loop_sizes[4]
         * u = m_loop_sizes[5]
         */
        int64_t offset_A = m_strides_in0[2] * m_idx_k + m_strides_in0[0] * m_idx_m;
        int64_t offset_B = m_strides_in1[1] * m_idx_n + m_strides_in1[2] * m_idx_k;
        int64_t offset_C = m_strides_out[1] * m_idx_n + m_strides_out[0] * m_idx_m;

        char const *sub_ptr_in0 = ptr_in0 + offset_A * dtype_size();
        char const *sub_ptr_in1 = ptr_in1 + offset_B * dtype_size();
        char *sub_ptr_out = ptr_out + offset_C * dtype_size();

        // Recursive Call
        if (id_loop + 1 < m_id_first_primitive_loop)
        {
            execute_iter(id_loop + 1, 
                         ptr_in0, 
                         ptr_in1, 
                         ptr_out, 
                         is_first,
                         is_last);
        }
        else
        {
            /////////////////////////////////////////////////////////////////////
            // First Touch
            /////////////////////////////////////////////////////////////////////
            if (is_first && m_prim_first_touch != ptype_t::none)
            {
                mini_jit::Unary::kernel_t l_prim_first_touch_kernel = m_prim_first_touch_unary.get_kernel();
                if (m_prim_first_touch == ptype_t::zero)
                {
                    l_prim_first_touch_kernel(nullptr,
                                              sub_ptr_out,
                                              0,                                  // ldA = 0
                                              m_loop_sizes[0] * m_loop_sizes[3]); // ldB = r * s
                }
                else if (m_prim_first_touch == ptype_t::relu)
                {
                    l_prim_first_touch_kernel(sub_ptr_out,
                                              sub_ptr_out,
                                              m_loop_sizes[0] * m_loop_sizes[3],  // ldA = r * s
                                              m_loop_sizes[0] * m_loop_sizes[3]); // ldB = r * s
                }
            }
            /////////////////////////////////////////////////////////////////////
            // Main
            /////////////////////////////////////////////////////////////////////
            if (m_prim_main == ptype_t::gemm)
            {
                mini_jit::Brgemm::kernel_t l_prim_main_kernel = m_prim_main_gemm.get_kernel();
                l_prim_main_kernel(sub_ptr_in0,                                         // A
                                   sub_ptr_in1,                                         // B
                                   sub_ptr_out,                                         // C
                                   m_loop_sizes[3],                                     // ldA = s
                                   m_loop_sizes[2] * m_loop_sizes[5],                   // ldB = t * u
                                   m_loop_sizes[0] * m_loop_sizes[3],                   // ldC = r * s
                                   1,
                                   1);
            }
            else if (m_prim_main == ptype_t::brgemm)
            {
                mini_jit::Brgemm::kernel_t l_prim_main_kernel = m_prim_main_gemm.get_kernel();
                l_prim_main_kernel(sub_ptr_in0,                                         // A
                                   sub_ptr_in1,                                         // B
                                   sub_ptr_out,                                         // C
                                   m_loop_sizes[3],                                     // ldA = s
                                   m_loop_sizes[2] * m_loop_sizes[5],                   // ldB = t * u
                                   m_loop_sizes[0] * m_loop_sizes[3],                   // ldC = r * s
                                   m_loop_sizes[0] * m_loop_sizes[3] * m_loop_sizes[5], // br_size_A = r * s * u
                                   m_loop_sizes[5]);                                    // br_size_B = u
            }
            else if (m_prim_main == ptype_t::identity)
            {
                mini_jit::Unary::kernel_t l_prim_main_kernel = m_prim_main_unary.get_kernel();
                l_prim_main_kernel(sub_ptr_in0,                                          // A
                                   sub_ptr_out,                                          // C
                                   m_loop_sizes[5] * m_loop_sizes[3],                    // u * s
                                   m_loop_sizes[2] * m_loop_sizes[5] * m_loop_sizes[0]); // t * u * r
            }
            /////////////////////////////////////////////////////////////////////
            // Last Touch
            /////////////////////////////////////////////////////////////////////
            // in theory, zero kernel is not possible here but kept for consistency
            if (is_last && m_prim_last_touch != ptype_t::none)
            {
                mini_jit::Unary::kernel_t l_prim_last_touch_kernel = m_prim_last_touch_unary.get_kernel();
                if (m_prim_last_touch == ptype_t::zero)
                {
                    l_prim_last_touch_kernel(nullptr,
                                             sub_ptr_out,
                                             0,                                  // ldA = 0
                                             m_loop_sizes[0] * m_loop_sizes[3]); // ldB = r * s
                }
                else if (m_prim_last_touch == ptype_t::relu)
                {
                    l_prim_last_touch_kernel(sub_ptr_out,
                                             sub_ptr_out,
                                             m_loop_sizes[0] * m_loop_sizes[3],  // ldA = r * s
                                             m_loop_sizes[0] * m_loop_sizes[3]); // ldB = r * s
                }
            }
        }
    }
}