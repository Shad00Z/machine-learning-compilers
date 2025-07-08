#include <catch2/catch.hpp>
#include <iostream>
#include <mlc/ir/Dimension.h>
#include <mlc/ir/IRConverter.h>
#include <mlc/ir/Optimizer.h>
#include <mlc/types.h>

using mini_jit::dim_t;
using mini_jit::exec_t;
using mini_jit::ir::Dimension;

TEST_CASE("Test Optimizer for BRGEMM", "[ir][optimizer][brgemm]")
{
    std::vector<mini_jit::ir::Dimension> dimensions;

    std::vector<dim_t>   dim_types   = {dim_t::m, dim_t::m, dim_t::n, dim_t::n, dim_t::k, dim_t::k};
    std::vector<exec_t>  exec_types  = {exec_t::seq, exec_t::seq, exec_t::seq, exec_t::seq, exec_t::seq, exec_t::seq};
    std::vector<int64_t> dim_sizes   = {64, 25, 64, 25, 64, 25};
    std::vector<int64_t> strides_in0 = {25, 1, 0, 0, 40000, 1600};
    std::vector<int64_t> strides_in1 = {0, 0, 40000, 1600, 25, 1};
    std::vector<int64_t> strides_out = {25, 1, 40000, 1600, 0, 0};

    const int64_t thread_target   = 64;
    const int64_t max_kernel_size = 1024;
    const int64_t min_kernel_size = 1;

    mini_jit::ir::IRConverter::convertConfigToDimensions(dim_types,
                                                         exec_types,
                                                         dim_sizes,
                                                         strides_in0,
                                                         strides_in1,
                                                         strides_out,
                                                         dimensions);

    mini_jit::ir::Optimizer::optimize(dimensions,
                                      thread_target,
                                      max_kernel_size,
                                      min_kernel_size);

    int prim_count        = 0;
    int shared_loop_count = 0;

    int64_t count_prim_m = 0;
    int64_t count_prim_n = 0;
    int64_t count_prim_k = 0;

    for (const auto& dim : dimensions)
    {
        if (dim.exec_type == exec_t::prim)
        {
            // Count primitive dimensions
            prim_count++;
            REQUIRE(dim.size <= max_kernel_size);
            if (dim.type == dim_t::m)
            {
                count_prim_m++;
            }
            else if (dim.type == dim_t::n)
            {
                count_prim_n++;
            }
            else if (dim.type == dim_t::k)
            {
                count_prim_k++;
            }
        }
        else if (dim.exec_type == exec_t::shared)
        {
            // Count shared loop dimensions
            shared_loop_count += dim.size;
        }
    }
    // 4 prims for BRGEMM
    REQUIRE(prim_count == 4);
    REQUIRE(count_prim_m == 1);
    REQUIRE(count_prim_n == 1);
    REQUIRE(count_prim_k == 2);
    REQUIRE(shared_loop_count <= thread_target);
}

TEST_CASE("Test Optimizer for GEMM", "[ir][optimizer][gemm]")
{
    std::vector<mini_jit::ir::Dimension> dimensions;

    std::vector<dim_t>   dim_types   = {dim_t::m, dim_t::n, dim_t::k};
    std::vector<exec_t>  exec_types  = {exec_t::seq, exec_t::seq, exec_t::seq};
    std::vector<int64_t> dim_sizes   = {1600, 1600, 512};
    std::vector<int64_t> strides_in0 = {1, 0, 1600};
    std::vector<int64_t> strides_in1 = {0, 512, 1};
    std::vector<int64_t> strides_out = {1, 1600, 0};

    const int64_t thread_target   = 16;
    const int64_t max_kernel_size = 512;
    const int64_t min_kernel_size = 1;

    mini_jit::ir::IRConverter::convertConfigToDimensions(dim_types,
                                                         exec_types,
                                                         dim_sizes,
                                                         strides_in0,
                                                         strides_in1,
                                                         strides_out,
                                                         dimensions);

    mini_jit::ir::Optimizer::optimize(dimensions,
                                      thread_target,
                                      max_kernel_size,
                                      min_kernel_size);

    int prim_count        = 0;
    int shared_loop_count = 0;

    int64_t count_prim_m = 0;
    int64_t count_prim_n = 0;
    int64_t count_prim_k = 0;

    for (const auto& dim : dimensions)
    {
        if (dim.exec_type == exec_t::prim)
        {
            // Count primitive dimensions
            prim_count++;
            REQUIRE(dim.size <= max_kernel_size);
            if (dim.type == dim_t::m)
            {
                count_prim_m++;
            }
            else if (dim.type == dim_t::n)
            {
                count_prim_n++;
            }
            else if (dim.type == dim_t::k)
            {
                count_prim_k++;
            }
        }
        else if (dim.exec_type == exec_t::shared)
        {
            // Count shared loop dimensions
            shared_loop_count += dim.size;
        }
    }
    // 3 prims for GEMM
    REQUIRE(prim_count == 3);
    REQUIRE(count_prim_m == 1);
    REQUIRE(count_prim_n == 1);
    REQUIRE(count_prim_k == 1);
    REQUIRE(shared_loop_count <= thread_target);
}

TEST_CASE("Test Optimizer for Identity", "[ir][optimizer][identity]")
{
    std::vector<mini_jit::ir::Dimension> dimensions;

    std::vector<dim_t>   dim_types   = {dim_t::c, dim_t::c, dim_t::c};
    std::vector<exec_t>  exec_types  = {exec_t::seq, exec_t::seq, exec_t::seq};
    std::vector<int64_t> dim_sizes   = {1600, 1600, 1600};
    std::vector<int64_t> strides_in0 = {1, 8, 1600};
    std::vector<int64_t> strides_in1 = {0, 0, 0};
    std::vector<int64_t> strides_out = {1, 1600, 8};

    const int64_t thread_target   = 16;
    const int64_t max_kernel_size = 512;
    const int64_t min_kernel_size = 1;

    // convert config object to IR
    mini_jit::ir::IRConverter::convertConfigToDimensions(dim_types,
                                                         exec_types,
                                                         dim_sizes,
                                                         strides_in0,
                                                         strides_in1,
                                                         strides_out,
                                                         dimensions);

    // Optimize with max 16 threads (16 shared loop iterations) and max kernel size of 512
    mini_jit::ir::Optimizer::optimize(dimensions,
                                      thread_target,
                                      max_kernel_size,
                                      min_kernel_size);

    int prim_count        = 0;
    int shared_loop_count = 0;
    for (const auto& dim : dimensions)
    {
        if (dim.exec_type == exec_t::prim)
        {
            // Count primitive dimensions
            prim_count++;
            REQUIRE(dim.size <= max_kernel_size);
        }
        else if (dim.exec_type == exec_t::shared)
        {
            // Count shared loop dimensions
            shared_loop_count += dim.size;
        }
        // All dimensions should be of type 'c'
        REQUIRE(dim.type == dim_t::c);
    }
    REQUIRE(prim_count == 2);
    REQUIRE(shared_loop_count <= thread_target);
}

TEST_CASE("Test Optimizer for Dimension Fusion and Splitting", "[ir][optimizer][fusion]")
{
    std::vector<mini_jit::ir::Dimension> dimensions;

    std::vector<dim_t>   dim_types   = {dim_t::m, dim_t::m, dim_t::n};
    std::vector<exec_t>  exec_types  = {exec_t::seq, exec_t::seq, exec_t::seq};
    std::vector<int64_t> dim_sizes   = {32, 4, 32};
    std::vector<int64_t> strides_in0 = {4, 1, 32};
    std::vector<int64_t> strides_in1 = {4, 1, 32};
    std::vector<int64_t> strides_out = {4, 1, 32};

    const int64_t thread_target   = 1024;
    const int64_t max_kernel_size = 32;
    const int64_t min_kernel_size = 8;

    mini_jit::ir::IRConverter::convertConfigToDimensions(dim_types,
                                                         exec_types,
                                                         dim_sizes,
                                                         strides_in0,
                                                         strides_in1,
                                                         strides_out,
                                                         dimensions);

    mini_jit::ir::Optimizer::optimize(dimensions,
                                      thread_target,
                                      max_kernel_size,
                                      min_kernel_size);

    for (const auto& dim : dimensions)
    {
        REQUIRE(dim.size <= max_kernel_size);
        REQUIRE(dim.size >= min_kernel_size);
    }
}