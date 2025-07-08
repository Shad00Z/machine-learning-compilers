#include <catch2/catch.hpp>
#include <iostream>
#include <mlc/Brgemm.h>
#include <mlc/constants.h>
#include <mlc/kernels/matmul/subkernels/matmul_m_3_k.h>
#include <random>
#include <vector>

TEST_CASE("Reference test for matmul_m_3_k kernel with variable M, N, K", "[matmul_3_k][parameterized]")
{
    const int M = GENERATE(8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
    const int N = GENERATE(3);
    const int K = GENERATE(1, 2, 3, 4, 5, 6, 7, 11, 13, 17);

    float* A          = new float[M * K];
    float* B          = new float[K * N];
    float* C          = new float[M * N];
    float* C_expected = new float[M * N];

    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dist(-0.5f, 100.0f);

    for (int i = 0; i < M * K; ++i)
    {
        A[i] = dist(gen);
    }

    for (int i = 0; i < K * N; ++i)
    {
        B[i] = dist(gen);
    }

    for (int i = 0; i < M * N; ++i)
    {
        C[i] = C_expected[i] = dist(gen);
    }

    // Reference GEMM calculation
    for (int col = 0; col < N; ++col)
    {
        for (int row = 0; row < M; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[row + k * M] * B[k + col * K];
            }
            C_expected[row + col * M] += sum;
        }
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::matmul::subkernels::matmul_m_3_k(l_kernel, M, K);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void*>(l_kernel.get_kernel()));
    l_kernel_t(A, B, C, M, K, M, 0, 0);

    for (int i = 0; i < M * N; ++i)
    {
        REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_expected;
}