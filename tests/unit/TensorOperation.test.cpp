#include <catch2/catch.hpp>
#include <random>
#include <iostream>
#include <vector>
#include <span>

#include "Brgemm.h"
#include "TensorOperation.h"
#include "constants.h"
#include "types.h"

void runTensorOperationTest(mini_jit::ptype_t first_touch_type,
                            mini_jit::ptype_t main_type,
                            mini_jit::ptype_t last_touch_type,
                            std::span<const mini_jit::exec_t> exec_types)
{
    const int R = 3;
    const int P = GENERATE(3, 7);
    const int T = GENERATE(3, 7);
    const int S = GENERATE(3, 4, 5, 7);
    const int Q = GENERATE(3, 4, 5, 7);
    const int U = GENERATE(3, 4, 5, 7);

    const int SIZE_A = (R * S) * (T * U);
    const int SIZE_B = (T * U) * (P * Q);
    const int SIZE_C = (R * S) * (P * Q);

    float *A = new float[SIZE_A];
    float *A_raw = new float[SIZE_A];
    float *B = new float[SIZE_B];
    float *C = new float[SIZE_C];
    float *C_expected = new float[SIZE_C];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    int id_Raw = 0;
    for (int t = 0; t < T; ++t)
    {
        for (int r = 0; r < R; ++r)
        {
            for (int u = 0; u < U; ++u)
            {
                for (int s = 0; s < S; ++s)
                {
                    int row = r * S + s;
                    int col = t * U + u;
                    int idx = col * (R * S) + row;

                    float val = dist(gen);
                    A[idx] = val;
                    A_raw[id_Raw++] = val;
                }
            }
        }
    }

    for (int i = 0; i < SIZE_B; ++i)
    {
        B[i] = dist(gen);
    }

    if (first_touch_type == mini_jit::ptype_t::zero)
    {
        for (int i = 0; i < SIZE_C; ++i)
        {
            // dont init with zero, to test if the kernel sets it to zero
            C[i] = dist(gen);
            C_expected[i] = 0.0f;
        }
    }
    else
    {
        for (int i = 0; i < SIZE_C; ++i)
        {
            C[i] = 0.0f;
            C_expected[i] = 0.0f;
        }
    }

    if (main_type == mini_jit::ptype_t::brgemm || main_type == mini_jit::ptype_t::gemm)
    {
        // Reference GEMM calculation
        for (int col = 0; col < (P * Q); ++col)
        {
            for (int row = 0; row < (R * S); ++row)
            {
                float sum = 0.0f;
                for (int k = 0; k < (T * U); ++k)
                {
                    sum += A[row + k * (R * S)] * B[k + col * (T * U)];
                }
                C_expected[row + col * (R * S)] = sum;
            }
        }
    }

    if (last_touch_type == mini_jit::ptype_t::relu)
    {
        auto fRelu = [](float x)
        { return x > 0.0f ? x : 0.0f; };
        for (int i = 0; i < SIZE_C; ++i)
        {
            C_expected[i] = fRelu(C_expected[i]);
        }
    }

    std::vector<mini_jit::dim_t> dim_types = {
        mini_jit::dim_t::m,
        mini_jit::dim_t::n,
        mini_jit::dim_t::k,
        mini_jit::dim_t::m,
        mini_jit::dim_t::n,
        mini_jit::dim_t::k};

    std::vector<int64_t> dim_sizes = {R, P, T, S, Q, U};

    std::vector<int64_t> strides_in0 = {U * S,
                                        0,
                                        R * U * S,
                                        1,
                                        0,
                                        S};
    std::vector<int64_t> strides_in1 = {0,
                                        Q * T * U,
                                        U,
                                        0,
                                        T * U,
                                        1};
    std::vector<int64_t> strides_out = {S,
                                        Q * R * S,
                                        0,
                                        1,
                                        R * S,
                                        0};

    mini_jit::TensorOperation l_top;
    l_top.setup(mini_jit::dtype_t::fp32,
                first_touch_type,
                main_type,
                last_touch_type,
                dim_types,
                exec_types,
                dim_sizes,
                strides_in0,
                strides_in1,
                strides_out);

    l_top.execute(A_raw, B, C);

    for (int i = 0; i < SIZE_C; ++i)
    {
        REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] A_raw;
    delete[] B;
    delete[] C;
    delete[] C_expected;
}

TEST_CASE("Reference test for ZERO + GEMM tensor operation kernel with variable R, P, T, S, Q, U", "[tensor_operation][parameterized][zero][gemm]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::zero;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::gemm;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::none;

    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::seq,
        mini_jit::exec_t::seq,
        mini_jit::exec_t::seq,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim};
    runTensorOperationTest(first_touch_type,
                           main_type,
                           last_touch_type,
                           exec_types);
}

TEST_CASE("Reference test for ZERO + BRGEMM tensor operation kernel with variable R, P, T, S, Q, U", "[tensor_operation][parameterized][zero][brgemm]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::zero;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::brgemm;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::none;

    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::seq,
        mini_jit::exec_t::seq,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim};
    runTensorOperationTest(first_touch_type,
                           main_type,
                           last_touch_type,
                           exec_types);
}

TEST_CASE("Reference test for ZERO + GEMM + RELU tensor operation kernel with variable R, P, T, S, Q, U", "[tensor_operation][parameterized][zero][gemm][relu]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::zero;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::gemm;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::relu;

    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::seq,
        mini_jit::exec_t::seq,
        mini_jit::exec_t::seq,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim};
    runTensorOperationTest(first_touch_type,
                           main_type,
                           last_touch_type,
                           exec_types);
}

TEST_CASE("Reference test for ZERO + BRGEMM + RELU tensor operation kernel with variable R, P, T, S, Q, U", "[tensor_operation][parameterized][zero][brgemm][relu]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::zero;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::brgemm;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::relu;

    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::seq,
        mini_jit::exec_t::seq,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim};
    runTensorOperationTest(first_touch_type,
                           main_type,
                           last_touch_type,
                           exec_types);
}

TEST_CASE("Reference test for ZERO + BRGEMM + RELU tensor operation kernel with variable shared(R), P, T, S, Q, U", "[tensor_operation][parameterized][zero][brgemm][relu]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::zero;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::brgemm;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::relu;

    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::shared,
        mini_jit::exec_t::seq,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim};
    runTensorOperationTest(first_touch_type,
                           main_type,
                           last_touch_type,
                           exec_types);
}

TEST_CASE("Reference test for BRGEMM tensor operation kernel with variable shared(R), shared(P), T, S, Q, U", "[tensor_operation][parameterized][brgemm]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::none;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::brgemm;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::none;

    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::shared,
        mini_jit::exec_t::shared,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim};
    runTensorOperationTest(first_touch_type,
                           main_type,
                           last_touch_type,
                           exec_types);
}

TEST_CASE("Reference test for ZERO + BRGEMM + RELU tensor operation kernel with variable shared(R), shared(P), T, S, Q, U", "[tensor_operation][parameterized][zero][brgemm][relu]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::zero;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::brgemm;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::relu;

    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::shared,
        mini_jit::exec_t::shared,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim};
    runTensorOperationTest(first_touch_type,
                           main_type,
                           last_touch_type,
                           exec_types);
}