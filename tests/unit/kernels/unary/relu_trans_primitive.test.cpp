#include <catch2/catch.hpp>
#include <random>
#include <iostream>

#include "relu_trans_primitive.h"
#include "Unary.h"
#include "constants.h"

void test_relu_trans_primitive(uint32_t M,
                               uint32_t N)
{
    float *A = new float[M * N];
    float *B = new float[N * M];
    float *A_expected = new float[M * N];
    float *B_expected = new float[N * M];

    // Initialize matrices A and B with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (u_int32_t i = 0; i < M * N; i++)
    {
        float l_aValue = dist(gen);
        A[i] = l_aValue;
        A_expected[i] = l_aValue;
        B[i] = i;
    }

    auto fRelu = [](float x) { return x > 0.0f ? x : 0.0f; };

    for (u_int32_t i = 0; i < M; ++i)
    {
        for (u_int32_t j = 0; j < N; ++j)
        {
            B_expected[i * N + j] = fRelu(A[j * M + i]);
        }
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::unary::relu_trans(l_kernel, M, N);
    mini_jit::Unary::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Unary::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
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

TEST_CASE("Tests the ReLU trans primitive with different M and N", "[relu_trans_primitive][parameterized]")
{
    uint32_t M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8);
    uint32_t N = GENERATE(1, 2, 3, 4, 5, 6, 7, 8);
    test_relu_trans_primitive(M, N);
}

TEST_CASE("Tests the ReLU trans primitive with larger M and N", "[relu_trans_primitive][large]")
{
    uint32_t M = 64;
    uint32_t N = 65;
    test_relu_trans_primitive(M, N);
}

// TEST_CASE("Tests the relu transposition primitive with M=N=512", "[relu_trans_primitive][M=N=512]")
// {
//     uint32_t M = 512;
//     uint32_t N = 512;
//     test_relu_trans_primitive(M, N);
// }

// TEST_CASE("Tests the relu transposition primitive with M=N=2048", "[relu_trans_primitive][M=N=2048]")
// {
//     uint32_t M = 2048;
//     uint32_t N = 2048;
//     test_relu_trans_primitive(M, N);
// }
