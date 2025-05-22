#include <catch2/catch.hpp>
#include <random>
#include <iostream>

#include "identity_trans_primitive.h"
#include "Unary.h"
#include "constants.h"

void test_identity_trans_primitive(uint32_t M,
                                   uint32_t N)
{
    float *A = new float[M * N];
    float *B = new float[N * M];
    float *A_expected = new float[M * N];
    float *B_expected = new float[N * M];

    // Initialize matrices A and B with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 30.0f);

    for (u_int32_t i = 0; i < M * N; i++)
    {
        A[i] = i;
        A_expected[i] = i;
        B[i] = dist(gen);
    }

    for (u_int32_t i = 0; i < M; ++i)
    {
        for (u_int32_t j = 0; j < N; ++j)
        {
            B_expected[i * N + j] = A[j * M + i];
        }
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::unary::identity_trans(l_kernel, M, N);
    mini_jit::Unary::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Unary::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t(A, B, M, N);

    for (u_int32_t i = 0; i < M * N; i++)
    {
        REQUIRE(A[i] == Approx(A_expected[i]).margin(FLOAT_ERROR_MARGIN));
        REQUIRE(B[i] == Approx(B_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] B;
    delete[] A_expected;
    delete[] B_expected;
}

TEST_CASE("Tests the transposition identity primitive with M=N=50", "[identity_trans_primitive][M=N=50]")
{
    uint32_t M = 50;
    uint32_t N = 50;
    test_identity_trans_primitive(M, N);
}

TEST_CASE("Tests the transposition identity primitive with M=N=64", "[identity_trans_primitive][M=N=64]")
{
    uint32_t M = 64;
    uint32_t N = 64;
    test_identity_trans_primitive(M, N);
}

// TEST_CASE("Tests the transposition identity primitive with M=N=512", "[identity_trans_primitive][M=N=512]")
// {
//     uint32_t M = 512;
//     uint32_t N = 512;
//     test_identity_trans_primitive(M, N);
// }

// TEST_CASE("Tests the transposition identity primitive with M=N=2048", "[identity_trans_primitive][M=N=2048]")
// {
//     uint32_t M = 2048;
//     uint32_t N = 2048;
//     test_identity_trans_primitive(M, N);
// }

TEST_CASE("Tests the transposition identity primitive with random matrices", "[identity_trans_primitive][parameterized]")
{
    uint32_t M = GENERATE(take(2, random(1, 32)));
    uint32_t N = GENERATE(take(2, random(1, 32)));
    test_identity_trans_primitive(M, N);
}
