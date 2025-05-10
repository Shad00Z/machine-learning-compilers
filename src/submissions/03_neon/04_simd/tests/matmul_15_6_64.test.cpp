#include <catch2/catch.hpp>
#include <random>
#include <iostream>

extern "C"
{
    void v1_matmul_15_6_64( float const *a,
                            float const *b,
                            float *c,
                            int64_t lda,
                            int64_t ldb,
                            int64_t ldc );

                            
    void v2_matmul_15_6_64( float const *a,
                            float const *b,
                            float *c,
                            int64_t lda,
                            int64_t ldb,
                            int64_t ldc );

    void v3_matmul_15_6_64( float const *a,
                            float const *b,
                            float *c,
                            int64_t lda,
                            int64_t ldb,
                            int64_t ldc );
}

TEST_CASE("Tests the v1_matmul_15_6_64 microkernel function with random matrices", "[v1_matmul_15_6_64]")
{
    const int M = 15;
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

    v1_matmul_15_6_64(A, B, C, M, K, M);

    for (int i = 0; i < M * N; i++)
    {
        REQUIRE(C[i] == Approx(C_expected[i]).epsilon(0.01));
    }
}

TEST_CASE("Tests the v2_matmul_15_6_64 microkernel function with random matrices", "[v2_matmul_15_6_64]")
{
    const int M = 15;
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

    v2_matmul_15_6_64(A, B, C, M, K, M);

    for (int i = 0; i < M * N; i++)
    {
        REQUIRE(C[i] == Approx(C_expected[i]).epsilon(0.01));
    }
}

TEST_CASE("Tests the v3_matmul_15_6_64 microkernel function with random matrices", "[v3_matmul_15_6_64]")
{
    const int M = 15;
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

    v3_matmul_15_6_64(A, B, C, M, K, M);

    for (int i = 0; i < M * N; i++)
    {
        REQUIRE(C[i] == Approx(C_expected[i]).epsilon(0.01));
    }
}
