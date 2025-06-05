#include <catch2/catch.hpp>
#include <iostream>
#include "types.h"
#include "ir/Optimizer.h"
#include "ir/Dimension.h"
#include "ir/IRConverter.h"
#include <limits.h>

using mini_jit::dim_t;
using mini_jit::exec_t;
using mini_jit::ir::Dimension;

TEST_CASE("Test Optimizer #1")
{
    std::vector<mini_jit::ir::Dimension> dimensions;

    std::vector<dim_t> dim_types = {dim_t::m, dim_t::m, dim_t::n, dim_t::n, dim_t::k, dim_t::k};
    std::vector<exec_t> exec_types = {exec_t::seq, exec_t::seq, exec_t::seq, exec_t::seq, exec_t::seq, exec_t::seq};
    std::vector<int64_t> dim_sizes = {64, 25, 64, 25, 64, 25};
    std::vector<int64_t> strides_in0 = {25, 1, 0, 0, 40000, 1600};
    std::vector<int64_t> strides_in1 = {0, 0, 40000, 1600, 25, 1};
    std::vector<int64_t> strides_out = {25, 1, 40000, 1600, 0, 0};

    // convert config object to IR
    mini_jit::ir::IRConverter::convertConfigToDimensions(
        dim_types, exec_types, dim_sizes, strides_in0, strides_in1, strides_out, dimensions);

    std::cout << "Before optimization:" << std::endl;
    for (const auto &dim : dimensions)
    {
        std::cout << "type: " << mini_jit::to_string(dim.type)
                  << ", exec_type: " << mini_jit::to_string(dim.exec_type)
                  << ", size: " << dim.size
                  << ", stride_in0: " << dim.stride_in0
                  << ", stride_in1: " << dim.stride_in1
                  << ", stride_out: " << dim.stride_out
                  << std::endl;
    }

    mini_jit::ir::Optimizer::optimize(dimensions, INT_MAX);

    std::cout << "After optimization:" << std::endl;
    for (const auto &dim : dimensions)
    {
        std::cout << "type: " << mini_jit::to_string(dim.type)
                  << ", exec_type: " << mini_jit::to_string(dim.exec_type)
                  << ", size: " << dim.size
                  << ", stride_in0: " << dim.stride_in0
                  << ", stride_in1: " << dim.stride_in1
                  << ", stride_out: " << dim.stride_out
                  << std::endl;
    }
    std::cout << "####################################################" << std::endl;
}

TEST_CASE("Test Optimizer #2")
{
    std::vector<mini_jit::ir::Dimension> dimensions;

    std::vector<dim_t> dim_types = {dim_t::m, dim_t::n, dim_t::k};
    std::vector<exec_t> exec_types = {exec_t::seq, exec_t::seq, exec_t::seq};
    std::vector<int64_t> dim_sizes = {1600, 1600, 1600};
    std::vector<int64_t> strides_in0 = {1, 0, 1600};
    std::vector<int64_t> strides_in1 = {0, 1600, 1};
    std::vector<int64_t> strides_out = {1, 1600, 0};

    // convert config object to IR
    mini_jit::ir::IRConverter::convertConfigToDimensions(
        dim_types, exec_types, dim_sizes, strides_in0, strides_in1, strides_out, dimensions);

    std::cout << "Before optimization:" << std::endl;
    for (const auto &dim : dimensions)
    {
        std::cout << "type: " << mini_jit::to_string(dim.type)
                  << ", exec_type: " << mini_jit::to_string(dim.exec_type)
                  << ", size: " << dim.size
                  << ", stride_in0: " << dim.stride_in0
                  << ", stride_in1: " << dim.stride_in1
                  << ", stride_out: " << dim.stride_out
                  << std::endl;
    }

    mini_jit::ir::Optimizer::optimize(dimensions, 5);

    std::cout << "After optimization:" << std::endl;
    for (const auto &dim : dimensions)
    {
        std::cout << "type: " << mini_jit::to_string(dim.type)
                  << ", exec_type: " << mini_jit::to_string(dim.exec_type)
                  << ", size: " << dim.size
                  << ", stride_in0: " << dim.stride_in0
                  << ", stride_in1: " << dim.stride_in1
                  << ", stride_out: " << dim.stride_out
                  << std::endl;
    }
    std::cout << "####################################################" << std::endl;
}