#include <catch2/catch.hpp>
#include <random>
#include <iostream>

#include "matmul_m_n_k.h"
#include "Brgemm.h"
#include "constants.h"

TEST_CASE("Reference test for matmul kernel with variable M, N, K", "[matmul][parameterized]") {
    const int M = GENERATE(take(4, random(1, 64)));
    const int N = GENERATE(take(4, random(1, 64)));
    const int K = GENERATE(1, 16, 32, 64, 128);

    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];
    float* C_expected = new float[M * N];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.5f, 100.0f);

    for (int i = 0; i < M * K; ++i) {
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
    mini_jit::kernels::matmul::matmul_m_n_k(l_kernel, M, N, K);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t( A, B, C, M, K, M, 0, 0 );

    for (int i = 0; i < M * N; ++i) 
    {
        REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_expected;
}

TEST_CASE("Reference test for matmul kernel with variable M, N, K and lda>M, ldb>K or ldc>M", "[matmul][parameterized][larger strides]") {
    const int M = GENERATE(take(4, random(1, 64)));
    const int N = GENERATE(take(4, random(1, 64)));
    const int K = GENERATE(1, 16, 32, 64, 128);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> strideDist(1, 10);

    // Set strides larger than dimensions
    const int lda = M + strideDist(gen);
    const int ldb = K + strideDist(gen);
    const int ldc = M + strideDist(gen);

    // Allocate space for matrices larger than M, N, K
    float* A = new float[lda * K];
    float* B = new float[ldb * N];
    float* C = new float[ldc * N];
    float* C_expected = new float[ldc * N];

    std::uniform_real_distribution<float> dist(-0.5f, 100.0f);

    // Initialize A
    for (int k = 0; k < K; ++k) 
    {
        for (int m = 0; m < lda; ++m) 
        {
            A[m + k * lda] = (m < M) ? dist(gen) : 0.0f;
        }
    }

    // Initialize B
    for (int n = 0; n < N; ++n) 
    {
        for (int k = 0; k < ldb; ++k) 
        {
            B[k + n * ldb] = (k < K) ? dist(gen) : 0.0f;
        }
    }

    // Initialize C and C_expected
    for (int n = 0; n < N; ++n) 
    {
        for (int m = 0; m < ldc; ++m) 
        {
            float value = (m < M) ? dist(gen) : 0.0f;
            C[m + n * ldc] = value;
            C_expected[m + n * ldc] = value;
        }
    }

    // Reference GEMM calculation
    for (int col = 0; col < N; ++col) 
    {
        for (int row = 0; row < M; ++row) 
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) 
            {
                sum += A[row + k * lda] * B[k + col * ldb];
            }
            C_expected[row + col * ldc] += sum;
        }
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::matmul::matmul_m_n_k(l_kernel, M, N, K);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t( A, B, C, lda, ldb, ldc, 0, 0 );

    for (int n = 0; n < N; ++n) 
    {
        for (int m = 0; m < M; ++m) 
        {
            REQUIRE(C[m + n * ldc] == Approx(C_expected[m + n * ldc]).margin(FLOAT_ERROR_MARGIN));
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_expected;
}

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
    mini_jit::kernels::matmul::matmul_m_n_k(l_kernel, M, N, K);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t( A, B, C, M, K, M, 0, 0 );

    for ( int i = 0; i < M * N; i++ )
    {
        REQUIRE( C[i] == Approx( C_expected[i] ).margin(FLOAT_ERROR_MARGIN) );
    }
}

TEST_CASE("Tests the matmul_m_n_k microkernel function with random matrices and M=39, N=64 and K=64", "[matmul_M17_N64_K64]")
{
    const int M = 39;
    const int N = 64;
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
    mini_jit::kernels::matmul::matmul_m_n_k(l_kernel, M, N, K);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t( A, B, C, M, K, M, 0, 0 );

    for ( int i = 0; i < M * N; i++ )
    {
        REQUIRE( C[i] == Approx( C_expected[i] ).margin(FLOAT_ERROR_MARGIN) );
    }
}

TEST_CASE("Tests the matmul_m_n_k microkernel function with random matrices and M=19, N=64 and K=64", "[matmul_M31_N64_K22]")
{
    const int M = 19;
    const int N = 64;
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
    mini_jit::kernels::matmul::matmul_m_n_k(l_kernel, M, N, K);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t( A, B, C, M, K, M, 0, 0 );

    for ( int i = 0; i < M * N; i++ )
    {
        REQUIRE( C[i] == Approx( C_expected[i] ).margin(FLOAT_ERROR_MARGIN) );
    }
}

TEST_CASE("Tests the matmul_m_n_k microkernel function with random matrices and M=16, N=7 and K=4", "[matmul_M17_N7_K64]")
{
    const int M = 16;
    const int N = 7;
    const int K = 4;

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
    mini_jit::kernels::matmul::matmul_m_n_k(l_kernel, M, N, K);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t( A, B, C, M, K, M, 0, 0 );

    for ( int i = 0; i < M * N; i++ )
    {
        REQUIRE( C[i] == Approx( C_expected[i] ).margin(FLOAT_ERROR_MARGIN) );
    }
}

TEST_CASE("Tests the matmul_m_n_k microkernel function with random matrices and M=16, N=63 and K=4", "[matmul_M17_N7_K64]")
{
    const int M = 16;
    const int N = 63;
    const int K = 4;

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
    mini_jit::kernels::matmul::matmul_m_n_k(l_kernel, M, N, K);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t( A, B, C, M, K, M, 0, 0 );

    for ( int i = 0; i < M * N; i++ )
    {
        REQUIRE( C[i] == Approx( C_expected[i] ).margin(FLOAT_ERROR_MARGIN) );
    }
}

TEST_CASE("Tests the matmul_m_n_k microkernel function with random matrices and M=27, N=63 and K=4", "[matmul_M27_N63_K4]")
{
    const int M = 27;
    const int N = 63;
    const int K = 4;

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
    mini_jit::kernels::matmul::matmul_m_n_k(l_kernel, M, N, K);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t( A, B, C, M, K, M, 0, 0 );

    for ( int i = 0; i < M * N; i++ )
    {
        REQUIRE( C[i] == Approx( C_expected[i] ).margin(FLOAT_ERROR_MARGIN) );
    }
}

TEST_CASE("Tests the matmul_m_n_k microkernel function with random matrices and M=23, N=63 and K=22", "[matmul_M23_N63_K22]")
{
    const int M = 23;
    const int N = 63;
    const int K = 22;

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
    mini_jit::kernels::matmul::matmul_m_n_k(l_kernel, M, N, K);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t( A, B, C, M, K, M, 0, 0 );

    for ( int i = 0; i < M * N; i++ )
    {
        REQUIRE( C[i] == Approx( C_expected[i] ).margin(FLOAT_ERROR_MARGIN) );
    }
}