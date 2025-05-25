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
//     const int M = 4;
//     const int N = 4;
//     const int K = 2;

//     float *A = new float[2 * M * K];
//     float *B = new float[K * N];
//     float *C = new float[2 * M * N];
//     float *C_expected = new float[2 * M * N];

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> dist(-0.5f, 100.0f);

//     for (int i = 0; i < 2 * M * K; ++i)
//     {
//         A[i] = i;
//     }

//     for (int i = 0; i < K * N; ++i)
//     {
//         B[i] = i;
//     }

//     for (int i = 0; i < 2 * M * N; ++i)
//     {
//         C[i] = 0.0f;
//         C_expected[i] = 0.0f;
//     }

//     // Reference GEMM calculation
//     for (int col = 0; col < N; ++col)
//     {
//         for (int row = 0; row < M * 2; ++row)
//         {
//             float sum = 0.0f;
//             for (int k = 0; k < K; ++k)
//             {
//                 sum += A[row + k * 2 * M] * B[k + col * K];
//             }
//             C_expected[row + col * M * 2] += sum;
//         }
//     }

//     // Print matrix A (column-major order)
//     std::cout << "Matrix A (column-major order):" << std::endl;

//     for (int row = 0; row < 2 * M; ++row)
//     {
//         for (int col = 0; col < K; ++col)
//         {
//             std::cout << A[row + col * (2 * M)] << " ";
//         }
//         std::cout << std::endl;
//     }
//     // Print matrix B (column-major order)
//     std::cout << "Matrix B (column-major order):" << std::endl;
//     for (int row = 0; row < K; ++row)
//     {
//         for (int col = 0; col < N; ++col)
//         {

//             std::cout << B[row + col * K] << " ";
//         }
//         std::cout << std::endl;
//     }
//     // Print matrix C_expected (column-major order)
//     std::cout << "Matrix C_expected (column-major order):" << std::endl;
//     for (int row = 0; row < 2 * M; ++row)
//     {
//         for (int col = 0; col < N; ++col)
//         {
//             std::cout << C_expected[row + col * (2 * M)] << " ";
//         }
//         std::cout << std::endl;
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

//     //                                r, p, t, s, q, u
//     std::vector<int64_t> dim_sizes = {2, 1, 1, M, N, K};

//     std::vector<int64_t> strides_in0 = {dim_sizes[5] * dim_sizes[3], // u * s,
//                                         0,
//                                         dim_sizes[0] * dim_sizes[5] * dim_sizes[3], // r * u * s
//                                         1,
//                                         0,
//                                         dim_sizes[3]}; // A[M, K] // s
//     std::vector<int64_t> strides_in1 = {0,
//                                         dim_sizes[4] * dim_sizes[2] * dim_sizes[5], // q * t * u
//                                         dim_sizes[5],                               // u
//                                         0,
//                                         dim_sizes[2] * dim_sizes[5], // t * u
//                                         1};                          // B[K, N]
//     std::vector<int64_t> strides_out = {dim_sizes[3],
//                                         dim_sizes[4] * dim_sizes[0] * dim_sizes[3], // q * r * s
//                                         0,
//                                         1,
//                                         dim_sizes[0] * dim_sizes[3], // r * s
//                                         0};                          // C[M, N]

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

//     // Print result matrix C (column-major order)
//     std::cout << "Result Matrix C (column-major order):" << std::endl;
//     for (int row = 0; row < 2 * M; ++row)
//     {
//         for (int col = 0; col < N; ++col)
//         {
//             std::cout << C[row + col * (2 * M)] << " ";
//         }
//         std::cout << std::endl;
//     }

//     for (int i = 0; i < 2 * M * N; ++i)
//     {
//         REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
//     }

//     delete[] A;
//     delete[] B;
//     delete[] C;
//     delete[] C_expected;
// }
