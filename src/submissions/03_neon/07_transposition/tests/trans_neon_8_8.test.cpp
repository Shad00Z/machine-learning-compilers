#include <catch2/catch.hpp>
#include <random>
#include <iostream>


extern "C"
{
    void trans_neon_8_8( float const *a,
                         float const *b,
                         int64_t lda,
                         int64_t ldb );
}

TEST_CASE("Tests the trans_neon_8_8 microkernel function with random matrices", "[trans_neon_8_8]")
{
    const int M = 8;
    const int N = 8;

    float A[M * N];
    float B[M * N];
    float B_expected[M * N];

    // Initialize matrices A and B with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < M * N; ++i)
    {
        A[i] = i;  // Initialize A

        B[i] = 0;
        B_expected[i] = 0;
    }

    // Transpose matrix A
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            B_expected[i * M + j] = A[j * N + i];
        }
    }

    for (int i = 0; i < M * N; i++)
    {
        REQUIRE(B[i] == Approx(B_expected[i]).epsilon(0.01));
    }
}
