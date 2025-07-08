#include <catch2/catch.hpp>
#include <iostream>
#include <mlc/Unary.h>
#include <mlc/constants.h>
#include <mlc/kernels/unary/decrement_trans_primitive.h>
#include <random>

void test_decrement_trans_primitive(uint32_t M,
                                    uint32_t N)
{
    float* A          = new float[M * N];
    float* B          = new float[N * M];
    float* A_expected = new float[M * N];
    float* B_expected = new float[N * M];

    // Initialize matrices A and B with random values
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (u_int32_t i = 0; i < M * N; i++)
    {
        float l_aValue = dist(gen);
        A[i]           = l_aValue;
        A_expected[i]  = l_aValue;
        B[i]           = i;
    }

    for (u_int32_t i = 0; i < M; ++i)
    {
        for (u_int32_t j = 0; j < N; ++j)
        {
            B_expected[i * N + j] = A[j * M + i] - 1.0f;
        }
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::unary::decrement_trans(l_kernel, M, N);
    mini_jit::Unary::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Unary::kernel_t>(const_cast<void*>(l_kernel.get_kernel()));
    l_kernel_t(A, B, M, N, nullptr);

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

TEST_CASE("Tests the decrement trans primitive with different M and N", "[decrement_trans_primitive][parameterized]")
{
    uint32_t M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8);
    uint32_t N = GENERATE(1, 2, 3, 4, 5, 6, 7, 8);
    test_decrement_trans_primitive(M, N);
}

TEST_CASE("Tests the decrement trans primitive with larger M and N", "[decrement_trans_primitive][large]")
{
    uint32_t M = 64;
    uint32_t N = 65;
    test_decrement_trans_primitive(M, N);
}
