#include <catch2/catch.hpp>
#include <random>
#include <iostream>

#include "identity_primitive.h"
#include "Unary.h"
#include "constants.h"

TEST_CASE("Tests the standard identity primitive with random matrices", "[str_identity_primitive][parameterized]")
{
    u_int32_t M = GENERATE(50, 64, 512, 2048);
    u_int32_t N = GENERATE(50, 64, 512, 2048);

    float* A = new float[M * N];
    float* B = new float[M * N];
    float* A_expected = new float[M * N];
    float* B_expected = new float[M * N];

    // Initialize matrices A and B with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 30.0f);

    for (u_int32_t i = 0; i < M * N; i++)
    {
        A[i] = i;
        A_expected[i] = i;
        B[i] = dist(gen);
        B_expected[i] = i;
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::unary::identity(l_kernel, M, N, 0);
    mini_jit::Unary::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Unary::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t(A, B, M, M);

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
