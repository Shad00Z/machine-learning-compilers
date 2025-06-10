#include <catch2/catch.hpp>
#include <random>
#include <iostream>
#include <vector>
#include <span>
#include <iomanip>

#include "Brgemm.h"
#include "TensorOperation.h"
#include "Optimizer.h"
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

TEST_CASE("Reference test for IDENTITY layout transformation trus → turs", "[tensor_operation][layout_transform][identity]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::none;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::identity;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::none;

    const int T = GENERATE(3, 4, 7);
    const int R = GENERATE(3, 4, 7);
    const int U = GENERATE(3, 4, 7);
    const int S = GENERATE(3, 4, 7);

    const int SIZE = T * R * U * S;

    float* A = new float[SIZE];
    float* C = new float[SIZE];
    float* C_expected = new float[SIZE];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < SIZE; ++i)
    {
        A[i] = dist(gen);
    }

    // Compute C_expected
    for (int t = 0; t < T; ++t)
    {
        for (int r = 0; r < R; ++r)
        {
            for (int u = 0; u < U; ++u)
            {
                for (int s = 0; s < S; ++s)
                {
                    // Calculate index in output format (t,r,u,s) using strides_out
                    int l_idx_c_exp = t * (U * R * S) + r * S + u * (R * S) + s;
                    // Calculate index in input format (t,u,r,s) using strides_in0
                    int l_idx_a = t * (R * U * S) + u * S + r * (U * S) + s;
                    C_expected[l_idx_c_exp] = A[l_idx_a];
                }
            }
        }
    }

    std::vector<mini_jit::dim_t> dim_types = {
        mini_jit::dim_t::c, // t
        mini_jit::dim_t::c, // r
        mini_jit::dim_t::c, // u
        mini_jit::dim_t::c  // s
    };

    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::seq,  // t
        mini_jit::exec_t::seq,  // r
        mini_jit::exec_t::prim, // u
        mini_jit::exec_t::prim  // s
    };

    std::vector<int64_t> dim_sizes = {
        T, R, U, S
    };

    std::vector<int64_t> strides_in0 = {
        R * U * S,  // t
        U * S,      // r
        S,          // u
        1           // s
    };

    std::vector<int64_t> strides_in1 = {0, 0, 0, 0};

    std::vector<int64_t> strides_out = {
        U * R * S,  // t
        S,          // r
        R * S,      // u
        1           // s
    };

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

    l_top.execute(A, nullptr, C);

    for (int i = 0; i < SIZE; ++i)
    {
        REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] C;
    delete[] C_expected;
}

TEST_CASE("Reference test for SHARED IDENTITY layout transformation trus → turs", "[tensor_operation][layout_transform][identity][shared]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::none;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::identity;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::none;

    const int T = GENERATE(3, 4, 7);
    const int R = GENERATE(3, 4, 7);
    const int U = GENERATE(3, 4, 7);
    const int S = GENERATE(3, 4, 7);

    const int SIZE = T * R * U * S;

    float* A = new float[SIZE];
    float* C = new float[SIZE];
    float* C_expected = new float[SIZE];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < SIZE; ++i)
    {
        A[i] = dist(gen);
    }

    // Compute C_expected
    for (int t = 0; t < T; ++t)
    {
        for (int r = 0; r < R; ++r)
        {
            for (int u = 0; u < U; ++u)
            {
                for (int s = 0; s < S; ++s)
                {
                    // Calculate index in output format (t,r,u,s) using strides_out
                    int l_idx_c_exp = t * (U * R * S) + r * S + u * (R * S) + s;
                    // Calculate index in input format (t,u,r,s) using strides_in0
                    int l_idx_a = t * (R * U * S) + u * S + r * (U * S) + s;
                    C_expected[l_idx_c_exp] = A[l_idx_a];
                }
            }
        }
    }

    std::vector<mini_jit::dim_t> dim_types = {
        mini_jit::dim_t::c, // t
        mini_jit::dim_t::c, // r
        mini_jit::dim_t::c, // u
        mini_jit::dim_t::c  // s
    };

    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::shared,  // t
        mini_jit::exec_t::shared,  // r
        mini_jit::exec_t::prim, // u
        mini_jit::exec_t::prim  // s
    };

    std::vector<int64_t> dim_sizes = {
        T, R, U, S
    };

    std::vector<int64_t> strides_in0 = {
        R * U * S,  // t
        U * S,      // r
        S,          // u
        1           // s
    };

    std::vector<int64_t> strides_in1 = {0, 0, 0, 0};

    std::vector<int64_t> strides_out = {
        U * R * S,  // t
        S,          // r
        R * S,      // u
        1           // s
    };

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

    l_top.execute(A, nullptr, C);

    for (int i = 0; i < SIZE; ++i) {
        REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] C;
    delete[] C_expected;
}

TEST_CASE("Reference test for ZERO + IDENTITY layout transformation trus → turs", "[tensor_operation][layout_transform][zero][identity]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::zero;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::identity;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::none;

    const int T = GENERATE(3, 4, 7);
    const int R = GENERATE(3, 4, 7);
    const int U = GENERATE(3, 4, 7);
    const int S = GENERATE(3, 4, 7);

    const int SIZE = T * R * U * S;

    float* A = new float[SIZE];
    float* C = new float[SIZE];
    float* C_expected = new float[SIZE];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < SIZE; ++i)
    {
        A[i] = 0;
    }

    // Compute C_expected
    for (int t = 0; t < T; ++t)
    {
        for (int r = 0; r < R; ++r)
        {
            for (int u = 0; u < U; ++u)
            {
                for (int s = 0; s < S; ++s)
                {
                    // Calculate index in output format (t,r,u,s) using strides_out
                    int l_idx_c_exp = t * (U * R * S) + r * S + u * (R * S) + s;
                    // Calculate index in input format (t,u,r,s) using strides_in0
                    int l_idx_a = t * (R * U * S) + u * S + r * (U * S) + s;
                    C_expected[l_idx_c_exp] = A[l_idx_a];
                }
            }
        }
    }

    std::vector<mini_jit::dim_t> dim_types = {
        mini_jit::dim_t::c, // t
        mini_jit::dim_t::c, // r
        mini_jit::dim_t::c, // u
        mini_jit::dim_t::c  // s
    };

    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::seq,  // t
        mini_jit::exec_t::seq,  // r
        mini_jit::exec_t::prim, // u
        mini_jit::exec_t::prim  // s
    };

    std::vector<int64_t> dim_sizes = {
        T, R, U, S
    };

    std::vector<int64_t> strides_in0 = {
        R * U * S,  // t
        U * S,      // r
        S,          // u
        1           // s
    };

    std::vector<int64_t> strides_in1 = {0, 0, 0, 0};

    std::vector<int64_t> strides_out = {
        U * R * S,  // t
        S,          // r
        R * S,      // u
        1           // s
    };

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

    l_top.execute(A, nullptr, C);

    for (int i = 0; i < SIZE; ++i)
    {
        REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] C;
    delete[] C_expected;
}

TEST_CASE("Reference test for IDENTITY + RELU layout transformation trus → turs", "[tensor_operation][layout_transform][relu][identity]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::none;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::identity;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::relu;

    const int T = GENERATE(3, 4, 7);
    const int R = GENERATE(3, 4, 7);
    const int U = GENERATE(3, 4, 7);
    const int S = GENERATE(3, 4, 7);

    const int SIZE = T * R * U * S;

    float* A = new float[SIZE];
    float* C = new float[SIZE];
    float* C_expected = new float[SIZE];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < SIZE; ++i)
    {
        A[i] = dist(gen);
    }

    // Compute C_expected
    for (int t = 0; t < T; ++t)
    {
        for (int r = 0; r < R; ++r)
        {
            for (int u = 0; u < U; ++u)
            {
                for (int s = 0; s < S; ++s)
                {
                    // Calculate index in output format (t,r,u,s) using strides_out
                    int l_idx_c_exp = t * (U * R * S) + r * S + u * (R * S) + s;
                    // Calculate index in input format (t,u,r,s) using strides_in0
                    int l_idx_a = t * (R * U * S) + u * S + r * (U * S) + s;
                    if ( A[l_idx_a] < 0 ) 
                    {
                        C_expected[l_idx_c_exp] = 0; // Apply ReLU
                    } else 
                    {
                        C_expected[l_idx_c_exp] = A[l_idx_a];
                    }
                }
            }
        }
    }

    std::vector<mini_jit::dim_t> dim_types = {
        mini_jit::dim_t::c, // t
        mini_jit::dim_t::c, // r
        mini_jit::dim_t::c, // u
        mini_jit::dim_t::c  // s
    };

    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::seq,  // t
        mini_jit::exec_t::seq,  // r
        mini_jit::exec_t::prim, // u
        mini_jit::exec_t::prim  // s
    };

    std::vector<int64_t> dim_sizes = {
        T, R, U, S
    };

    std::vector<int64_t> strides_in0 = {
        R * U * S,  // t
        U * S,      // r
        S,          // u
        1           // s
    };

    std::vector<int64_t> strides_in1 = {0, 0, 0, 0};

    std::vector<int64_t> strides_out = {
        U * R * S,  // t
        S,          // r
        R * S,      // u
        1           // s
    };

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

    l_top.execute(A, nullptr, C);

    for (int i = 0; i < SIZE; ++i)
    {
        REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] C;
    delete[] C_expected;
}

TEST_CASE("Reference test for ZERO + IDENTITY_TRANS tensor operation kernel with variable R, S", "[tensor_operation][parameterized][zero][identity_trans]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::zero;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::identity;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::none;

    const int R = 4;
    const int S = 4;

    const int SIZE_A = R * S;
    const int SIZE_C = R * S;

    float *A = new float[SIZE_A];
    float *C = new float[SIZE_C];
    float *C_expected = new float[SIZE_C];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // Initialize input tensor A
    for (int i = 0; i < SIZE_A; ++i)
    {
        A[i] = dist(gen);
    }

    // Initialize input tensor C
    for (int i = 0; i < SIZE_C; ++i)
    {
        C[i] = dist(gen);
    }

    // Initialize expected output (transposed)
    for (int r = 0; r < R; ++r)
    {
        for (int s = 0; s < S; ++s)
        {
            C_expected[s * R + r] = A[r * S + s];
        }
    }

    // Define dimension types (all 'c' for unary operation)
    std::vector<mini_jit::dim_t> dim_types = {
        mini_jit::dim_t::c,
        mini_jit::dim_t::c
    };

    // Define execution types
    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim
    };

    // Define dimension sizes
    std::vector<int64_t> dim_sizes = {
        R,
        S
    };

    // Define strides
    std::vector<int64_t> strides_in0 = {
        S,
        1
    };
    std::vector<int64_t> strides_in1 = {
        0,
        0
    };
    std::vector<int64_t> strides_out = {
        1,
        R
    };

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

    // Execute with nullptr for second input (unary operation)
    l_top.execute(A, nullptr, C);

    // Verify results
    for (int i = 0; i < SIZE_C; ++i) {
        REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] C;
    delete[] C_expected;
}

TEST_CASE("Reference test for ZERO + IDENTITY_TRANS + RELU tensor operation kernel with variable R, S", "[tensor_operation][parameterized][zero][identity][relu]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::zero;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::identity;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::relu;

    const int R = 4;
    const int S = 4;

    const int SIZE_A = R * S;
    const int SIZE_C = R * S;

    float *A = new float[SIZE_A];
    float *C = new float[SIZE_C];
    float *C_expected = new float[SIZE_C];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // Initialize input tensor A
    for (int i = 0; i < SIZE_A; ++i)
    {
        A[i] = dist(gen);
    }

    // Initialize expected output (transposed  + ReLU)
    for (int r = 0; r < R; ++r)
    {
        for (int s = 0; s < S; ++s)
        {
            C_expected[s * R + r] =  std::max(0.0f,A[r * S + s]);
        }
    }

    // Define dimension types (all 'c' for unary operation)
    std::vector<mini_jit::dim_t> dim_types = {
        mini_jit::dim_t::c,
        mini_jit::dim_t::c
    };

    // Define execution types
    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim
    };

    // Define dimension sizes
    std::vector<int64_t> dim_sizes = {
        R,
        S
    };

    // Define strides
    std::vector<int64_t> strides_in0 = {
        S,
        1
    };
    std::vector<int64_t> strides_in1 = {
        0,
        0
    };
    std::vector<int64_t> strides_out = {
        1,
        R
    };

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

    // Execute with nullptr for second input (unary operation)
    l_top.execute(A, nullptr, C);

    // Verify results
    for (int i = 0; i < SIZE_C; ++i)
    {
        REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] C;
    delete[] C_expected;
}

TEST_CASE("Reference test for ZERO + IDENTITY + RELU optimized tensor operation kernel with variable R, S", "[tensor_operation][parameterized][zero][identity][relu][optimized]")
{
    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::zero;
    const mini_jit::ptype_t main_type = mini_jit::ptype_t::identity;
    const mini_jit::ptype_t last_touch_type = mini_jit::ptype_t::relu;

    const int R = 8;
    const int S = 8;

    const int SIZE_A = R * S;
    const int SIZE_C = R * S;

    float *A = new float[SIZE_A];
    float *C = new float[SIZE_C];
    float *C_expected = new float[SIZE_C];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // Initialize input tensor A
    for (int i = 0; i < SIZE_A; ++i)
    {
        A[i] = dist(gen);
    }

    // Initialize expected output (identity + ReLU operation)
    for (int i = 0; i < SIZE_C; ++i)
    {
        C_expected[i] = std::max(0.0f, A[i]);
    }

    // Define dimension types (all 'c' for unary operation)
    std::vector<mini_jit::dim_t> dim_types = {
        mini_jit::dim_t::c,
        mini_jit::dim_t::c
    };

    // Define execution types
    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::seq,
        mini_jit::exec_t::seq
    };

    // Define dimension sizes
    std::vector<int64_t> dim_sizes = {
        R,
        S
    };

    // Define strides
    std::vector<int64_t> strides_in0 = {
        S,
        1
    };
    std::vector<int64_t> strides_in1 = {
        0,
        0
    };
    std::vector<int64_t> strides_out = {
        S,
        1
    };

    // max kernel size of 4
    mini_jit::ir::Optimizer::optimize(dim_types,
                                      exec_types,
                                      dim_sizes,
                                      strides_in0,
                                      strides_in1,
                                      strides_out,
                                      256,
                                      4);

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

    // Execute with nullptr for second input (unary operation)
    l_top.execute(A, nullptr, C);

    // Verify results
    for (int i = 0; i < SIZE_C; ++i)
    {
        REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] C;
    delete[] C_expected;
}