#include <catch2/catch.hpp>
#include <random>
#include <iostream>

#include "relu_primitive.h"
#include "Unary.h"
#include "constants.h"

TEST_CASE("Tests the ReLu primitive with random matrices", "[relu_primitive][parameterized]")
{
    int M = GENERATE(take(16, random(1, 64)));
    int N = GENERATE(1, 2, 3, 7, 16, 64);

    float* A = new float[M * N];
    float* B = new float[M * N];
    float* A_expected = new float[M * N];
    float* B_expected = new float[M * N];

    // Initialize matrices A and B with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    auto fRelu = [](float x) { return x > 0.0f ? x : 0.0f; };

    for (int i = 0; i < M * N; i++)
    {
        A[i] = dist(gen);
        A_expected[i] = A[i];
        B[i] = dist(gen);
        B_expected[i] = fRelu(A[i]);
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::unary::relu(l_kernel, M, N);
    mini_jit::Unary::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Unary::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t(A, B, M, M);

    for (int i = 0; i < M * N; i++)
    {
        REQUIRE(A[i] == Approx(A_expected[i]).margin(FLOAT_ERROR_MARGIN));
        REQUIRE(B[i] == Approx(B_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete[] A;
    delete[] B;
    delete[] A_expected;
    delete[] B_expected;
}