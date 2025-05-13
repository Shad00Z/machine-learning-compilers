#include <catch2/catch.hpp>
#include <random>
#include <vector>
#include <iostream>

#include "matmul_m_4_k.h"
#include "Brgemm.h"

TEST_CASE("Tests the matmul_m_4_k microkernel function with random matrices and M=16", "[matmul_M16_4_k]")
{
    const int M = 16;
    const int N = 4;
    const int K = 1;

    float A[M * K] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
    float B[K * N] = {0.0f, 1.0f, 2.0f, 3.0f};
    float C[M * N] = {0.0f};
    float C_expected[M * N] = {
        0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
        0.0f, 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        0.0f, 2.0f,  4.0f,  6.0f,  8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
        0.0f, 3.0f,  6.0f,  9.0f, 12.0f, 15.0f, 18.0f, 21.0f, 24.0f, 27.0f, 30.0f, 33.0f, 36.0f, 39.0f, 42.0f, 45.0f
    };

    mini_jit::Brgemm l_brgemm;
    
    mini_jit::Brgemm::error_t l_ret = l_brgemm.generate(M, N, K, 4, 0, 0, 0, mini_jit::Brgemm::dtype_t::fp32);
    REQUIRE( l_ret == mini_jit::Brgemm::error_t::success );

    mini_jit::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();
    l_kernel( A, B, C, M, K, M, 0, 0 );

    for ( int i = 0; i < M * N; i++ )
    {
        REQUIRE( C[i] == Approx( C_expected[i] ).epsilon( 0.01 ) );
    }
}

TEST_CASE("Tests the matmul_m_4_k microkernel function with random matrices and M=17 and K=64", "[matmul_M17_4_k]")
{
    const int M = 17;
    const int N = 4;
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

    mini_jit::Brgemm l_brgemm;
    
    mini_jit::Brgemm::error_t l_ret = l_brgemm.generate(M, N, K, 4, 0, 0, 0, mini_jit::Brgemm::dtype_t::fp32);
    REQUIRE( l_ret == mini_jit::Brgemm::error_t::success );

    mini_jit::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();
    l_kernel( A, B, C, M, K, M, 0, 0 );

    for ( int i = 0; i < M * N; i++ )
    {
        REQUIRE( C[i] == Approx( C_expected[i] ).epsilon( 0.01 ) );
    }
}

TEST_CASE("Tests the matmul_m_4_k microkernel function with random matrices and M=18 and K=64", "[matmul_M18_4_k]")
{
    const int M = 18;
    const int N = 4;
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

    mini_jit::Brgemm l_brgemm;
    
    mini_jit::Brgemm::error_t l_ret = l_brgemm.generate(M, N, K, 4, 0, 0, 0, mini_jit::Brgemm::dtype_t::fp32);
    REQUIRE( l_ret == mini_jit::Brgemm::error_t::success );

    mini_jit::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();
    l_kernel( A, B, C, M, K, M, 0, 0 );

    for ( int i = 0; i < M * N; i++ )
    {
        REQUIRE( C[i] == Approx( C_expected[i] ).epsilon( 0.01 ) );
    }
}

TEST_CASE("Tests the matmul_m_4_k microkernel function with random matrices and M=19 and K=64", "[matmul_M19_4_k]")
{
    const int M = 19;
    const int N = 4;
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

    mini_jit::Brgemm l_brgemm;
    
    mini_jit::Brgemm::error_t l_ret = l_brgemm.generate(M, N, K, 4, 0, 0, 0, mini_jit::Brgemm::dtype_t::fp32);
    REQUIRE( l_ret == mini_jit::Brgemm::error_t::success );

    mini_jit::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();
    l_kernel( A, B, C, M, K, M, 0, 0 );

    for ( int i = 0; i < M * N; i++ )
    {
        REQUIRE( C[i] == Approx( C_expected[i] ).epsilon( 0.01 ) );
    }
}