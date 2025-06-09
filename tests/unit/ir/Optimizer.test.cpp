#include <catch2/catch.hpp>
#include "types.h"
#include "Optimizer.h"
#include "Dimension.h"
#include "IRConverter.h"
#include <climits>
#include <iostream>

using mini_jit::dim_t;
using mini_jit::exec_t;
using mini_jit::ir::Dimension;

TEST_CASE("Test Optimizer - Identity")
{
    std::vector<mini_jit::ir::Dimension> dimensions;

    std::vector<dim_t> dim_types = {dim_t::c, dim_t::c, dim_t::c};
    std::vector<exec_t> exec_types = {exec_t::seq, exec_t::seq, exec_t::seq};
    std::vector<int64_t> dim_sizes = {1600, 1600, 1600};
    std::vector<int64_t> strides_in0 = {1, 8, 1600};
    std::vector<int64_t> strides_in1 = {0, 0, 0};
    std::vector<int64_t> strides_out = {1, 1600, 8};

    // convert config object to IR
    mini_jit::ir::IRConverter::convertConfigToDimensions(dim_types,
                                                         exec_types,
                                                         dim_sizes,
                                                         strides_in0,
                                                         strides_in1,
                                                         strides_out,
                                                         dimensions);

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

    // Optimize with max 16 threads (16 shared loop iterations) and max kernel size of 512
    mini_jit::ir::Optimizer::optimize(dimensions,
                                      INT_MAX,
                                      512);

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