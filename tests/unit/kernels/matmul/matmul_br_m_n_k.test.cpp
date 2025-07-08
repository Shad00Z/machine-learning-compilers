#include <catch2/catch.hpp>
#include <iostream>
#include <mlc/Brgemm.h>
#include <mlc/constants.h>
#include <mlc/kernels/matmul/matmul_br_m_n_k.h>
#include <random>

TEST_CASE("Reference test for batch reduce matmul kernel with variable M, N, K", "[br_matmul][parameterized]")
{
    const int M       = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    const int N       = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    const int K       = GENERATE(1, 16, 32, 64, 128);
    const int br_size = 16;

    float* A          = new float[M * K * br_size];
    float* B          = new float[K * N * br_size];
    float* C          = new float[M * N];
    float* C_expected = new float[M * N];

    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dist(-0.5f, 100.0f);

    for (int i = 0; i < M * K * br_size; ++i)
    {
        A[i] = dist(gen);
    }

    for (int i = 0; i < K * N * br_size; ++i)
    {
        B[i] = dist(gen);
    }

    for (int i = 0; i < M * N; ++i)
    {
        C[i] = C_expected[i] = dist(gen);
    }

    // Reference batched GEMM calculation
    for (int col = 0; col < N; ++col)
    {
        for (int row = 0; row < M; ++row)
        {
            float sum = 0.0f;
            for (int br = 0; br < br_size; ++br)
            {
                for (int k = 0; k < K; ++k)
                {
                    sum += A[br * M * K + row + k * M] * B[br * K * N + k + col * K];
                }
            }
            C_expected[row + col * M] += sum;
        }
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::matmul::matmul_br_m_n_k(l_kernel, M, N, K, br_size);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void*>(l_kernel.get_kernel()));
    l_kernel_t(A, B, C, M, K, M, M * K, K * N);

    for (int i = 0; i < M * N; ++i)
    {
        REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_expected;
}