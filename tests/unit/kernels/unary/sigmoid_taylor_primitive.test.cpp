#include <catch2/catch.hpp>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "sigmoid_taylor_primitive.h"
#include "Unary.h"
#include "constants.h"

void test_sigmoid_taylor_primitive(uint32_t M,
                                   uint32_t N)
{
    float* A = new float[M * N];
    float* B = new float[M * N];
    float* A_expected = new float[M * N];
    float* B_expected = new float[M * N];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    // sigmoid(x) ≈ 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5 (5th order Taylor)
    auto fSigmoidApprox = [](float x) {
        float x2 = x * x;
        float x3 = x2 * x;
        float x5 = x3 * x2;
        return 0.5f + 0.25f * x - 0.020833333f * x3 + 0.002083333f * x5;
    };

    for (u_int32_t i = 0; i < M * N; i++)
    {
        float l_aValue = dist(gen);
        A[i] = l_aValue;
        A_expected[i] = l_aValue;

        B[i] = dist(gen);
        B_expected[i] = fSigmoidApprox(A[i]);
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::unary::sigmoid_taylor(l_kernel, M, N);
    mini_jit::Unary::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Unary::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t(A, B, M, M, const_cast<void*>(static_cast<const void*>(sig_taylor_values)));

    for (u_int32_t i = 0; i < M * N; i++)
    {
        REQUIRE(A[i] == Approx(A_expected[i]).margin(FLOAT_ERROR_MARGIN));
        // For polynomial approximation, relaxed margin
        REQUIRE(B[i] == Approx(B_expected[i]).margin(0.00001f));
    }

    delete[] A;
    delete[] B;
    delete[] A_expected;
    delete[] B_expected;
}

TEST_CASE("Tests the sigmoid primitive with different M and N", "[sigmoid_taylor_primitive][parameterized]")
{
    uint32_t M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    uint32_t N = GENERATE(1, 2, 3);
    test_sigmoid_taylor_primitive(M, N);
}

TEST_CASE("Tests the sigmoid primitive with larger M and N", "[sigmoid_taylor_primitive][large]")
{
    uint32_t M = 64;
    uint32_t N = 65;
    test_sigmoid_taylor_primitive(M, N);
}
