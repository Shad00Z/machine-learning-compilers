#include "TensorOperation.h"
#include <algorithm>
#include <ostream>
#include <iostream>

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
    else if (prim_main == ptype_t::identity && prim_count != 2)
    {
        return error_t::wrong_exec_type;
    }

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
    // Assign member variables
    /////////////////////////////////////////////////////////////////////
    m_dim_types.assign(dim_types.begin(), dim_types.end());
    m_exec_types.assign(exec_types.begin(), exec_types.end());
    m_dim_sizes.assign(dim_sizes.begin(), dim_sizes.end());
    m_strides_in0.assign(strides_in0.begin(), strides_in0.end());
    m_strides_in1.assign(strides_in1.begin(), strides_in1.end());
    m_strides_out.assign(strides_out.begin(), strides_out.end());
    m_dtype = dtype;

    m_dim_id_prim_M = -1;
    m_dim_id_prim_N = -1;
    m_dim_id_prim_K = -1;
    m_dim_id_prim_BR = -1;
    m_dim_id_seq_M = -1;
    m_dim_id_seq_N = -1;
    m_dim_id_seq_K = -1;
    m_dim_id_sha_M = -1;
    m_dim_id_sha_N = -1;
    m_num_parallel_loops = 0;

    /////////////////////////////////////////////////////////////////////
    // Find first PRIM and SEQ dimensions in exec types
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

    it = std::find(exec_types.begin(), exec_types.end(), exec_t::seq);
    if (it != exec_types.end())
    {
        m_id_first_seq_loop = std::distance(exec_types.begin(), it);
    }
    else
    {
        m_id_first_seq_loop = -1;
    }

    /////////////////////////////////////////////////////////////////////
    // Find SHARED dimensions in exec types
    /////////////////////////////////////////////////////////////////////
    for (size_t i = 0; i < m_exec_types.size(); ++i)
    {
        if (m_exec_types[i] == exec_t::shared)
        {
            m_shared_loop_ids.push_back(i);
            m_shared_loop_sizes.push_back(m_dim_sizes[i]);
        }
    }

    /////////////////////////////////////////////////////////////////////
    // Read PRIM dimensions using dim types (No Copy)
    /////////////////////////////////////////////////////////////////////
    // convert to int so negative values are allowed
    int l_dim_types_size = static_cast<int>(m_dim_types.size());
    for (int i = l_dim_types_size - 1; i >= 0; i--)
    {
        if (m_exec_types[i] == exec_t::prim)
        {
            if (m_dim_id_prim_M == -1 && m_dim_types[i] == dim_t::m)
            {
                m_dim_id_prim_M = i;
            }
            else if (m_dim_id_prim_N == -1 && m_dim_types[i] == dim_t::n)
            {
                m_dim_id_prim_N = i;
            }
            else if (m_dim_id_prim_K == -1 && m_dim_types[i] == dim_t::k)
            {
                m_dim_id_prim_K = i;
            }
            else if (m_dim_id_prim_K != -1 && m_dim_id_prim_BR == -1 && m_dim_types[i] == dim_t::k)
            {
                m_dim_id_prim_BR = i;
            }
        }
    }

    /////////////////////////////////////////////////////////////////////
    // Read SEQ and SHARED dimensions using dim types
    /////////////////////////////////////////////////////////////////////
    for (size_t i = 0; i < m_dim_types.size(); ++i)
    {
        if (m_exec_types[i] == exec_t::seq)
        {
            if (m_dim_types[i] == dim_t::m)
            {
                m_dim_id_seq_M = i;
            }
            else if (m_dim_types[i] == dim_t::n)
            {
                m_dim_id_seq_N = i;
            }
            else if (m_dim_types[i] == dim_t::k)
            {
                m_dim_id_seq_K = i;
            }
        }
        else if (m_exec_types[i] == exec_t::shared)
        {
            if (m_dim_types[i] == dim_t::m)
            {
                m_dim_id_sha_M = i;
                m_num_parallel_loops++;
            }
            else if (m_dim_types[i] == dim_t::n)
            {
                m_dim_id_sha_N = i;
                m_num_parallel_loops++;
            }
        }
    }

    /////////////////////////////////////////////////////////////////////
    // Find M and N dimensions for dim_t::c (PRIM)
    /////////////////////////////////////////////////////////////////////
    if (prim_main == ptype_t::identity)
    {
        // For unary operations with dim_t::c, treat the first primitive dimension as M and the second as N
        for (size_t i = 0; i < m_dim_types.size(); ++i)
        {
            if (m_exec_types[i] == exec_t::prim && m_dim_types[i] == dim_t::c)
            {
                if (m_strides_in0[i] == 1)
                {
                    m_dim_id_prim_M = i;
                }
                else
                {
                    m_dim_id_prim_N = i;
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////////////
    // Check for Transposition
    /////////////////////////////////////////////////////////////////////
    if (m_dim_id_prim_M != -1)
    {
        int64_t l_stride_in0 = m_strides_in0[m_dim_id_prim_M];
        int64_t l_stride_out = m_strides_out[m_dim_id_prim_M];
        // set transpose flag to true if the strides are different
        m_transpose_output = l_stride_in0 != l_stride_out;
    }
    else
    {
        // idk if we can check for transposition without M
        m_transpose_output = false;
    }

    /////////////////////////////////////////////////////////////////////
    // Adjust strides based on primitive type and transposition
    /////////////////////////////////////////////////////////////////////
    if (prim_main == ptype_t::identity)
    {
        if (!m_transpose_output)
        {
            m_adjusted_stride_in0 = m_strides_in0[m_dim_id_prim_N];
            m_adjusted_stride_in1 = 0;
            m_adjusted_stride_out = m_strides_out[m_dim_id_prim_N];
        }
        else
        {
            m_adjusted_stride_in0 = m_strides_in0[m_dim_id_prim_N];
            m_adjusted_stride_in1 = 0;
            m_adjusted_stride_out = m_strides_out[m_dim_id_prim_M];
        }
    }
    else
    {
        // GEMM & BRGEMM
        m_adjusted_stride_in0 = m_strides_in0[m_dim_id_prim_K];
        m_adjusted_stride_in1 = m_strides_in1[m_dim_id_prim_N];
        m_adjusted_stride_out = m_strides_out[m_dim_id_prim_N];
    }

    /////////////////////////////////////////////////////////////////////
    // Generate kernels
    /////////////////////////////////////////////////////////////////////
    if (prim_first_touch != ptype_t::none)
    {
        m_unary_first_touch.generate(m_dim_sizes[m_dim_id_prim_M],
                                     m_dim_sizes[m_dim_id_prim_N],
                                     m_transpose_output,
                                     dtype,
                                     prim_first_touch);
        m_kernel_first_touch = m_unary_first_touch.get_kernel();
    }
    if (prim_main == ptype_t::gemm)
    {
        // does not support transposition
        m_brgemm_main.generate(m_dim_sizes[m_dim_id_prim_M],
                               m_dim_sizes[m_dim_id_prim_N],
                               m_dim_sizes[m_dim_id_prim_K],
                               1,
                               0,
                               0,
                               0,
                               dtype);
        m_kernel_gemm_main = m_brgemm_main.get_kernel();
    }
    else if (prim_main == ptype_t::brgemm)
    {
        // does not support transposition
        m_brgemm_main.generate(m_dim_sizes[m_dim_id_prim_M],
                               m_dim_sizes[m_dim_id_prim_N],
                               m_dim_sizes[m_dim_id_prim_K],
                               m_dim_sizes[m_dim_id_prim_BR],
                               0,
                               0,
                               0,
                               dtype);
        m_kernel_gemm_main = m_brgemm_main.get_kernel();
    }
    else if (prim_main == ptype_t::identity)
    {
        m_unary_main.generate(m_dim_sizes[m_dim_id_prim_M],
                              m_dim_sizes[m_dim_id_prim_N],
                              m_transpose_output,
                              dtype,
                              prim_main);
        m_kernel_unary_main = m_unary_main.get_kernel();
    }
    if (prim_last_touch != ptype_t::none)
    {
        m_unary_last_touch.generate(m_dim_sizes[m_dim_id_prim_M],
                                    m_dim_sizes[m_dim_id_prim_N],
                                    m_transpose_output,
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

    if (m_num_parallel_loops == 0)
    {
        // No shared loops, execute sequentially
        execute_iter(0,
                     ptr_in0,
                     ptr_in1,
                     ptr_out,
                     true,
                     true);
    }
    else
    {
        // Shared loops, execute in parallel
        execute_iter_parallel(ptr_in0,
                              ptr_in1,
                              ptr_out,
                              true,
                              true);
    }
}

void mini_jit::TensorOperation::execute_iter(int64_t id_loop,
                                             char const *ptr_in0,
                                             char const *ptr_in1,
                                             char *ptr_out,
                                             bool first_access,
                                             bool last_access)
{
    // there is only one iteration if the dimension is the first primitive
    const int64_t l_size = id_loop != m_id_first_primitive_loop ? m_dim_sizes[id_loop] : 1;
    const int64_t dtype_sz = dtype_size();
    const int64_t l_stride_in0 = m_strides_in0[id_loop] * dtype_sz;
    const int64_t l_stride_in1 = m_strides_in1[id_loop] * dtype_sz;
    const int64_t l_stride_out = m_strides_out[id_loop] * dtype_sz;

    for (int64_t l_iter = 0; l_iter < l_size; l_iter++)
    {
        bool is_first = first_access;
        bool is_last = last_access;
        // if the size is 1, it is always the first and last access
        if (l_size > 1 && m_dim_types[id_loop] == dim_t::k)
        {
            is_first = first_access && (l_iter == 0);
            is_last = last_access && (l_iter == m_dim_sizes[id_loop] - 1);
        }

        char const *sub_ptr_in0 = ptr_in0 + l_iter * l_stride_in0;
        char const *sub_ptr_in1 = ptr_in1 + l_iter * l_stride_in1;
        char *sub_ptr_out = ptr_out + l_iter * l_stride_out;

        // Recursive Call
        if (id_loop + 1 < m_id_first_primitive_loop)
        {
            execute_iter(id_loop + 1,
                         sub_ptr_in0,
                         sub_ptr_in1,
                         sub_ptr_out,
                         is_first,
                         is_last);
        }
        else
        {
            if (is_first)
            {
                execute_kernel_first_touch(sub_ptr_out,
                                           m_adjusted_stride_out);
            }

            execute_kernel_main(sub_ptr_in0,
                                sub_ptr_in1,
                                sub_ptr_out,
                                m_adjusted_stride_in0,
                                m_adjusted_stride_in1,
                                m_adjusted_stride_out,
                                m_dim_id_prim_BR != -1 ? m_strides_in0[m_dim_id_prim_BR] : 1,
                                m_dim_id_prim_BR != -1 ? m_strides_in1[m_dim_id_prim_BR] : 1);

            if (is_last)
            {
                execute_kernel_last_touch(sub_ptr_out,
                                          m_adjusted_stride_out);
            }
        }
    }
}

void mini_jit::TensorOperation::execute_iter_parallel(char const *ptr_in0,
                                                      char const *ptr_in1,
                                                      char *ptr_out,
                                                      bool first_access,
                                                      bool last_access)
{
    // Compute total number of iterations over shared loops
    int64_t l_size_parallel_loops = 1;
    for (auto current_loop_size : m_shared_loop_sizes)
    {
        l_size_parallel_loops *= current_loop_size;
    }

#pragma omp parallel for
    for (int64_t l_it_all = 0; l_it_all < l_size_parallel_loops; ++l_it_all)
    {
        // Unflatten l_it_all into loop indices
        int64_t remainder = l_it_all;
        std::vector<int64_t> loop_indices(m_shared_loop_ids.size());

        for (int64_t i = m_shared_loop_ids.size() - 1; i >= 0; --i)
        {
            loop_indices[i] = remainder % m_shared_loop_sizes[i];
            remainder /= m_shared_loop_sizes[i];
        }

        // Compute pointer offsets using strides and loop indices
        char const *sub_ptr_in0 = ptr_in0;
        char const *sub_ptr_in1 = ptr_in1;
        char *sub_ptr_out = ptr_out;

        const int64_t dtype_sz = dtype_size();
        for (size_t i = 0; i < m_shared_loop_ids.size(); ++i)
        {
            const int64_t dim_id = m_shared_loop_ids[i];
            const int64_t idx = loop_indices[i];

            sub_ptr_in0 += idx * m_strides_in0[dim_id] * dtype_sz;
            sub_ptr_in1 += idx * m_strides_in1[dim_id] * dtype_sz;
            sub_ptr_out += idx * m_strides_out[dim_id] * dtype_sz;
        }

        // Call remaining loops
        execute_iter((m_id_first_seq_loop != -1) ? m_id_first_seq_loop : m_id_first_primitive_loop,
                     sub_ptr_in0,
                     sub_ptr_in1,
                     sub_ptr_out,
                     first_access,
                     last_access);
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