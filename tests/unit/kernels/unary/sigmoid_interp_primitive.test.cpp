#include <catch2/catch.hpp>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "sigmoid_interp_primitive.h"
#include "Unary.h"
#include "constants.h"

void test_sigmoid_interp_primitive(uint32_t M,
                            uint32_t N)
{
    float* A = new float[M * N];
    float* B = new float[M * N];
    float* A_expected = new float[M * N];
    float* B_true = new float[M * N];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // True sigmoid function: σ(x) = 1 / (1 + e^(-x))
    auto fSigmoidTrue = [](float x) {
        return 1.0f / (1.0f + std::exp(-x));
    };

    for (u_int32_t i = 0; i < M * N; i++)
    {
        float l_aValue = dist(gen);
        A[i] = l_aValue;
        A_expected[i] = l_aValue;

        B[i] = dist(gen);
        B_true[i] = fSigmoidTrue(A[i]);
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::unary::sigmoid_interpolation(l_kernel, M, N);
    
    mini_jit::Unary::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Unary::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t(A, B, M, M, const_cast<void*>(static_cast<const void*>(sig_table)));

    for (u_int32_t i = 0; i < M * N; i++)
    {
        REQUIRE(A[i] == Approx(A_expected[i]).margin(FLOAT_ERROR_MARGIN));
        REQUIRE(B[i] == Approx(B_true[i]).margin(0.01f));
    }

    delete[] A;
    delete[] B;
    delete[] A_expected;
    delete[] B_true;
}

TEST_CASE("Tests the sigmoid interpolation primitive with different M and N", "[sigmoid_primitive][parameterized]")
{
    uint32_t M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8);
    uint32_t N = GENERATE(1, 2, 3);
    test_sigmoid_interp_primitive(M, N);
}
