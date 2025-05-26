#include <catch2/catch.hpp>
#include <random>
#include <iostream>
#include <vector>

#include "Brgemm.h"
#include "TensorOperation.h"
#include "constants.h"
#include "types.h"

TEST_CASE("Reference test for tensor operation kernel with variable S, Q, U", "[matmul][parameterized]")
{
    const int R = GENERATE(2, 3, 5);
    const int P = GENERATE(2, 3, 5);
    const int T = 1;
    const int S = GENERATE(take(1, random(1, 16)));
    const int Q = GENERATE(take(1, random(1, 16)));
    const int U = GENERATE(take(1, random(1, 16)));

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
    std::uniform_real_distribution<float> dist(-0.5f, 100.0f);

    int aVal = 0;
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

                    A[idx] = aVal;
                    A_raw[id_Raw++] = aVal;
                    aVal++;
                }
            }
        }
    }

    for (int i = 0; i < SIZE_B; ++i)
    {
        B[i] = i;
    }

    for (int i = 0; i < SIZE_C; ++i)
    {
        C[i] = 0.0f;
        C_expected[i] = 0.0f;
    }

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

    // // Print matrix A_raw (column-major order)
    // std::cout << "Matrix A_raw (column-major order):" << std::endl;

    // for (int row = 0; row < (R * S); ++row)
    // {
    //     for (int col = 0; col < (T * U); ++col)
    //     {
    //         std::cout << A_raw[row + col * (R * S)] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // // Print matrix A (column-major order)
    // std::cout << "Matrix A (how kernel interprets A_raw):" << std::endl;

    // for (int row = 0; row < (R * S); ++row)
    // {
    //     for (int col = 0; col < (T * U); ++col)
    //     {
    //         std::cout << A[row + col * (R * S)] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // // Print matrix B (column-major order)
    // std::cout << "Matrix B (column-major order):" << std::endl;
    // for (int row = 0; row < (T * U); ++row)
    // {
    //     for (int col = 0; col < (P * Q); ++col)
    //     {

    //         std::cout << B[row + col * (T * U)] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // // Print matrix C_expected (column-major order)
    // std::cout << "Matrix C reference (column-major order):" << std::endl;
    // for (int row = 0; row < (R * S); ++row)
    // {
    //     for (int col = 0; col < (P * Q); ++col)
    //     {
    //         std::cout << C_expected[row + col * (R * S)] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    std::vector<mini_jit::dim_t> dim_types = {
        mini_jit::dim_t::m,
        mini_jit::dim_t::n,
        mini_jit::dim_t::k,
        mini_jit::dim_t::m,
        mini_jit::dim_t::n,
        mini_jit::dim_t::k};

    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::seq,
        mini_jit::exec_t::seq,
        mini_jit::exec_t::seq,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim};

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
                mini_jit::ptype_t::none,
                mini_jit::ptype_t::gemm,
                mini_jit::ptype_t::none,
                dim_types,
                exec_types,
                dim_sizes,
                strides_in0,
                strides_in1,
                strides_out);

    l_top.execute(A_raw, B, C);

    // // Print result matrix C (column-major order)
    // std::cout << "Matrix C (column-major order):" << std::endl;
    // for (int row = 0; row < (R * S); ++row)
    // {
    //     for (int col = 0; col < (P * Q); ++col)
    //     {
    //         std::cout << C[row + col * (R * S)] << " ";
    //     }
    //     std::cout << std::endl;
    // }

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
