#include <catch2/catch.hpp>
#include <random>
#include <vector>
#include <iostream>

#include "matmul_m_n_k.h"
#include "Brgemm.h"

TEST_CASE("Tests the matmul_m_n_k microkernel function with random matrices and M=17, N=12 and K=64", "[matmul_M17_N12_K64]")
{
    const int M = 17;
    const int N = 12;
    const int K = 64;

    float A[M * K];
    float B[K * N];
    float C[M * N];
    float C_expected[M * N];

    // Initialize matrices A and B with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < M * K; i++)
    {
        A[i] = dist(gen);
    }

    for (int i = 0; i < K * N; i++)
    {
        B[i] = dist(gen);
    }

    for (int i = 0; i < M * N; i++)
    {
        float value = dist(gen);
        C[i] = value;
        C_expected[i] = value;
    }

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
    mini_jit::kernels::matmul_m_n_k(l_kernel, M, N, K);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t( A, B, C, M, K, M, 0, 0 );

    for ( int i = 0; i < M * N; i++ )
    {
        REQUIRE( C[i] == Approx( C_expected[i] ).epsilon( 0.01 ) );
    }
}
