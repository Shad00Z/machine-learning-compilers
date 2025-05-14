#include <catch2/catch.hpp>
#include <random>
#include <vector>
#include <iostream>

#include "matmul_m_2_k.h"
#include "Brgemm.h"

TEST_CASE("Tests the matmul_m_2_k microkernel function with random matrices M=16, N=2, and K=1", "[matmul_M16_2_k]")
{
    const int M = 16;
    const int N = 2;
    const int K = 1;

    float A[M * K] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
    float B[K * N] = {0.0f, 1.0f};
    float C[M * N] = {0.0f};
    float C_expected[M * N] = {
        0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
        0.0f, 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
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

TEST_CASE("Tests the matmul_m_2_k microkernel function with random matrices and M=17, N=2, and K=64", "[matmul_M17_2_k]")
{
    const int M = 17;
    const int N = 2;
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

TEST_CASE("Tests the matmul_m_2_k microkernel function with random matrices and M=18, N=2 and K=64", "[matmul_M18_2_k]")
{
    const int M = 18;
    const int N = 2;
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

TEST_CASE("Tests the matmul_m_2_k microkernel function with random matrices and M=19, N=6 and K=64", "[matmul_M19_6_k]")
{
    const int M = 19;
    const int N = 6;
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

TEST_CASE("Tests the matmul_m_2_k microkernel function with random matrices and M=20, N=10, and K=64", "[matmul_M20_10_k]")
{
    const int M = 20;
    const int N = 10;
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

TEST_CASE("Tests the matmul_m_2_k microkernel function with random matrices and M=21, N=82 and K=64", "[matmul_M21_82_k]")
{
    const int M = 21;
    const int N = 82;
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

TEST_CASE("Tests the matmul_m_2_k microkernel function with random matrices and M=22, N=14, and K=64", "[matmul_M22_14_k]")
{
    const int M = 22;
    const int N = 14;
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

TEST_CASE("Tests the matmul_m_2_k microkernel function with random matrices and M=23, N=22 and K=64", "[matmul_M23_22_k]")
{
    const int M = 23;
    const int N = 22;
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

TEST_CASE("Tests the matmul_m_2_k microkernel function with random matrices and M=25, N=22 and K=64", "[matmul_M25_22_k]")
{
    const int M = 25;
    const int N = 22;
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
