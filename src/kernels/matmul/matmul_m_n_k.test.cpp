#include <catch2/catch.hpp>
#include <random>
#include <iostream>

#include "matmul_m_n_k.h"
#include "Brgemm.h"
#include "constants.h"

TEST_CASE("Reference test for matmul kernel with variable M, N, K", "[matmul][parameterized]") {
    const int M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    const int N = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    const int K = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

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