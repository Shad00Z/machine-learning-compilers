#include <catch2/catch.hpp>
#include <random>
#include <iostream>

#include "decrement_primitive.h"
#include "Unary.h"
#include "constants.h"

void test_decrement_primitive(uint32_t M,
                              uint32_t N)
{
    float* A = new float[M * N];
    float* B = new float[M * N];
    float* A_expected = new float[M * N];
    float* B_expected = new float[M * N];

    // Initialize matrices A and B with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (u_int32_t i = 0; i < M * N; i++)
    {
        float l_aValue = dist(gen);
        A[i] = l_aValue;
        A_expected[i] = l_aValue;
        B[i] = dist(gen);
        B_expected[i] = l_aValue - 1.0f;
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::unary::decrement(l_kernel, M, N);
    mini_jit::Unary::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Unary::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t(A, B, M, M, nullptr);

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

TEST_CASE("Tests the decrement primitive with different M and N", "[decrement_primitive][parameterized]")
{
    uint32_t M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    uint32_t N = GENERATE(1, 2, 3, 4);
    test_decrement_primitive(M, N);
}

TEST_CASE("Tests the decrement primitive with larger M and N", "[decrement_primitive][large]")
{
    uint32_t M = 64;
    uint32_t N = 65;
    test_decrement_primitive(M, N);
}
