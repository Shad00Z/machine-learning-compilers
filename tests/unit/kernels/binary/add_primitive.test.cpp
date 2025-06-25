#include <catch2/catch.hpp>
#include <random>
#include <iostream>

#include "add_primitive.h"
#include "Binary.h"
#include "constants.h"

void test_add_primitive(uint32_t M,
                        uint32_t N)
{
    float* A = new float[M * N];
    float* B = new float[M * N];
    float* C = new float[M * N];
    float* A_expected = new float[M * N];
    float* B_expected = new float[M * N];
    float* C_expected = new float[M * N];

    // Initialize matrices A and B with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (u_int32_t i = 0; i < M * N; i++)
    {
        float l_aValue = dist(gen);
        float l_bValue = dist(gen);

        A[i] = l_aValue;
        A_expected[i] = l_aValue;

        B[i] = l_bValue;
        B_expected[i] = l_bValue;

        C[i] = 0.0f;
        C_expected[i] = l_aValue + l_bValue;
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::binary::add(l_kernel, M, N);
    mini_jit::Binary::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Binary::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t(A, B, C, M, M, M);

    for (u_int32_t i = 0; i < M * N; i++)
    {
        REQUIRE(A[i] == Approx(A_expected[i]).margin(FLOAT_ERROR_MARGIN));
        REQUIRE(B[i] == Approx(B_expected[i]).margin(FLOAT_ERROR_MARGIN));
        REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] B;
    delete[] A_expected;
    delete[] B_expected;
}

TEST_CASE("Tests the add primitive with different M and N", "[add_primitive][parameterized]")
{
    uint32_t M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    uint32_t N = GENERATE(1, 2, 3, 4);
    test_add_primitive(M, N);
}

TEST_CASE("Tests the add primitive with larger M and N", "[add_primitive][large]")
{
    uint32_t M = 64;
    uint32_t N = 65;
    test_add_primitive(M, N);
}
