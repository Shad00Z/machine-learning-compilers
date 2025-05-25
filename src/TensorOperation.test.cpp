#include <catch2/catch.hpp>
#include <random>
#include <iostream>
#include <vector>

#include "Brgemm.h"
#include "TensorOperation.h"
#include "constants.h"
#include "types.h"

// TEST_CASE("Reference test for tensor operation kernel with variable M, N, K", "[matmul][parameterized]")
// {
//     const int M = GENERATE(32);
//     const int N = GENERATE(32);
//     const int K = GENERATE(8);

//     float *A = new float[M * K];
//     float *B = new float[K * N];
//     float *C = new float[M * N];
//     float *C_expected = new float[M * N];

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> dist(-0.5f, 100.0f);

//     for (int i = 0; i < M * K; ++i)
//     {
//         A[i] = dist(gen);
//     }

//     for (int i = 0; i < K * N; ++i)
//     {
//         B[i] = dist(gen);
//     }

//     for (int i = 0; i < M * N; ++i)
//     {
//         C[i] = C_expected[i] = dist(gen);
//     }

//     // Reference GEMM calculation
//     for (int col = 0; col < N; ++col)
//     {
//         for (int row = 0; row < M; ++row)
//         {
//             float sum = 0.0f;
//             for (int k = 0; k < K; ++k)
//             {
//                 sum += A[row + k * M] * B[k + col * K];
//             }
//             C_expected[row + col * M] += sum;
//         }
//     }

//     std::vector<mini_jit::dim_t> dim_types = {
//         mini_jit::dim_t::m,
//         mini_jit::dim_t::n,
//         mini_jit::dim_t::k,
//         mini_jit::dim_t::m,
//         mini_jit::dim_t::n,
//         mini_jit::dim_t::k};

//     std::vector<mini_jit::exec_t> exec_types = {
//         mini_jit::exec_t::seq,
//         mini_jit::exec_t::seq,
//         mini_jit::exec_t::seq,
//         mini_jit::exec_t::prim,
//         mini_jit::exec_t::prim,
//         mini_jit::exec_t::prim};

//     std::vector<int64_t> dim_sizes = {32, 32, 8, 32, 32, 32};

//     // std::vector<int64_t> strides_in0 = {1024, 0, 32768, 1, 0, 32}; // A[M, K]
//     std::vector<int64_t> strides_in0 = {8192, 0, 1024, 1, 0, 32}; // A[M, K]
//     // std::vector<int64_t> strides_in1 = {0, 8192, 32, 0, 1024, 1}; // B[K, N]
//     std::vector<int64_t> strides_in1 = {0, 8192, 1024, 0, 32, 1}; // B[K, N]
//     // std::vector<int64_t> strides_out = {32, 32768, 0, 1, 1024, 0}; // C[M, N]
//     std::vector<int64_t> strides_out = {32768, 1024, 0, 1, 32, 0}; // C[M, N]

//     mini_jit::TensorOperation l_top;
//     l_top.setup(mini_jit::dtype_t::fp32,
//                 mini_jit::ptype_t::none,
//                 mini_jit::ptype_t::gemm,
//                 mini_jit::ptype_t::none,
//                 dim_types,
//                 exec_types,
//                 dim_sizes,
//                 strides_in0,
//                 strides_in1,
//                 strides_out);

//     l_top.execute(A, B, C);

//     for (int i = 0; i < M * N; ++i)
//     {
//         REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
//     }

//     delete[] A;
//     delete[] B;
//     delete[] C;
//     delete[] C_expected;
}
