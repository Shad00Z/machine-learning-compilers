#include "TensorOperation.h"
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
    // Check the number of prim exec types
    /////////////////////////////////////////////////////////////////////
    int prim_count = std::count(exec_types.begin(), exec_types.end(), exec_t::prim);
    if (prim_main == ptype_t::brgemm && prim_count != 4)
    {
        return error_t::wrong_exec_type;
    }
    else if (prim_main == ptype_t::gemm && prim_count != 3)
    {
        return error_t::wrong_exec_type;
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

    m_dim_s = -1;
    m_dim_q = -1;
    m_dim_u = -1;
    m_dim_r = -1;
    m_dim_p = -1;
    m_dim_t = -1;
    m_exists_seq_k = false;

    /////////////////////////////////////////////////////////////////////
    // Read PRIM dimensions using dim types
    /////////////////////////////////////////////////////////////////////
    for (size_t i = m_dim_types.size() - 1; i > 0; i--)
    {
        if (m_exec_types[i] == exec_t::prim)
        {
            if (m_dim_types[i] == dim_t::m && m_dim_s == -1)
            {
                m_dim_s = m_loop_sizes[i];
            }
            else if (m_dim_types[i] == dim_t::n && m_dim_q == -1)
            {
                m_dim_q = m_loop_sizes[i];
            }
            else if (m_dim_types[i] == dim_t::k && m_dim_u == -1)
            {
                m_dim_u = m_loop_sizes[i];
            }
            else if (m_dim_types[i] == dim_t::k)
            {
                m_dim_t = m_loop_sizes[i];
            }
        }
    }

    /////////////////////////////////////////////////////////////////////
    // Read SEQ dimensions using dim types
    /////////////////////////////////////////////////////////////////////
    for (size_t i = 0; i < m_dim_types.size(); ++i)
    {
        if (m_exec_types[i] == exec_t::seq)
        {
            if (m_dim_types[i] == dim_t::m)
            {
                m_dim_r = m_loop_sizes[i];
            }
            else if (m_dim_types[i] == dim_t::n)
            {
                m_dim_p = m_loop_sizes[i];
            }
            else if (m_dim_types[i] == dim_t::k)
            {
                m_exists_seq_k = true;
                m_dim_t = m_loop_sizes[i];
            }
        }
    }

    /////////////////////////////////////////////////////////////////////
    // Generate kernels
    /////////////////////////////////////////////////////////////////////
    if (prim_first_touch != ptype_t::none)
    {
        m_unary_first_touch.generate(m_dim_s,
                                     m_dim_q,
                                     0,
                                     dtype,
                                     prim_first_touch);
        m_kernel_first_touch = m_unary_first_touch.get_kernel();
    }
    if (prim_main == ptype_t::gemm)
    {
        m_brgemm_main.generate(m_dim_s,
                               m_dim_q,
                               m_dim_u,
                               1,
                               0,
                               0,
                               0,
                               dtype);
        m_kernel_gemm_main = m_brgemm_main.get_kernel();
    }
    else if (prim_main == ptype_t::brgemm)
    {
        m_brgemm_main.generate(m_dim_s,
                               m_dim_q,
                               m_dim_u,
                               m_dim_t,
                               0,
                               0,
                               0,
                               dtype);
        m_kernel_gemm_main = m_brgemm_main.get_kernel();
    }
    else if (prim_main == ptype_t::identity)
    {
        // TODO: check if transpose or not
        m_unary_main.generate(m_dim_s,
                              m_dim_q,
                              0,
                              dtype,
                              prim_main);
        m_kernel_unary_main = m_unary_main.get_kernel();
    }
    if (prim_last_touch != ptype_t::none)
    {
        m_unary_last_touch.generate(m_dim_s,
                                    m_dim_q,
                                    0,
                                    dtype,
                                    prim_last_touch);
        m_kernel_last_touch = m_unary_last_touch.get_kernel();
    }

    m_kernel_first_touch_type = prim_first_touch;
    m_kernel_main_type = prim_main;
    m_kernel_last_touch_type = prim_last_touch;

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

        if (m_kernel_main_type == ptype_t::brgemm &&
            !m_exists_seq_k)
        {
            // is_first = true;
            is_last = true;
        }

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
            if (is_first)
            {
                execute_kernel_first_touch(sub_ptr_out,
                                           m_dim_r * m_dim_s); // ld = r * s
            }

            execute_kernel_main(sub_ptr_in0,
                                sub_ptr_in1,
                                sub_ptr_out,
                                m_dim_s,                     // ldA = s
                                m_dim_t * m_dim_u,           // ldB = t * u
                                m_dim_r * m_dim_s,           // ldC = r * s
                                m_dim_r * m_dim_s * m_dim_u, // br_size_A = r * s * u
                                m_dim_u);                    // br_size_B = u

            if (is_last)
            {
                execute_kernel_last_touch(sub_ptr_out,
                                          m_dim_r * m_dim_s); // ld = r * s
            }
        }
    }
}

void mini_jit::TensorOperation::execute_kernel_first_touch(char *ptr_out,
                                                           int64_t ldOut)
{
    if (m_kernel_first_touch_type == ptype_t::zero)
    {
        m_kernel_first_touch(nullptr,
                             ptr_out,
                             0,
                             ldOut);
    }
    else if (m_kernel_first_touch_type == ptype_t::relu)
    {
        m_kernel_first_touch(ptr_out,
                             ptr_out,
                             ldOut,
                             ldOut);
    }
}

void mini_jit::TensorOperation::execute_kernel_main(char const *ptr_in0,
                                                    char const *ptr_in1,
                                                    char *ptr_out,
                                                    int64_t ldA,
                                                    int64_t ldB,
                                                    int64_t ldC,
                                                    int64_t br_size_A,
                                                    int64_t br_size_B)
{
    if (m_kernel_main_type == ptype_t::gemm)
    {
        m_kernel_gemm_main(ptr_in0,
                           ptr_in1,
                           ptr_out,
                           ldA,
                           ldB,
                           ldC,
                           1,
                           1);
    }
    else if (m_kernel_main_type == ptype_t::brgemm)
    {
        m_kernel_gemm_main(ptr_in0,
                           ptr_in1,
                           ptr_out,
                           ldA,
                           ldB,
                           ldC,
                           br_size_A,
                           br_size_B);
    }
    else if (m_kernel_main_type == ptype_t::identity)
    {
        m_kernel_unary_main(ptr_in0,
                            ptr_out,
                            ldA,
                            ldC);
    }
}

void mini_jit::TensorOperation::execute_kernel_last_touch(char *ptr_out,
                                                          int64_t ldOut)
{
    if (m_kernel_last_touch_type == ptype_t::zero)
    {
        m_kernel_last_touch(nullptr,
                            ptr_out,
                            0,
                            ldOut);
    }
    else if (m_kernel_last_touch_type == ptype_t::relu)
    {
        m_kernel_last_touch(ptr_out,
                            ptr_out,
                            ldOut,
                            ldOut);
    }
}