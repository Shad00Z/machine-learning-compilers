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
    float* B_expected = new float[M * N];
    float* B_true = new float[M * N];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // True sigmoid function: σ(x) = 1 / (1 + e^(-x))
    auto fSigmoidTrue = [](float x) {
        return 1.0f / (1.0f + std::exp(-x));
    };

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
        // std::cout << A[i] << std::endl;
        A_expected[i] = l_aValue;

        B[i] = dist(gen);
        B_expected[i] = fSigmoidApprox(A[i]);
        B_true[i] = fSigmoidTrue(A[i]);
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::unary::sigmoid_interpolation(l_kernel, M, N);
    
    // Use kernel_t_sig signature with lookup table parameter
    mini_jit::Unary::kernel_t_sig l_kernel_t = reinterpret_cast<mini_jit::Unary::kernel_t_sig>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t(A, B, const_cast<void*>(static_cast<const void*>(sig_table)), M, M);

    // std::cout << "\nMatrix B" << std::endl;
    // for (u_int32_t i = 0; i < M * N; i++)
    // {
    //     if ( i % M == 0)
    //     {
    //         std::cout << ""<< std::endl;    
    //     }
    //     std::cout << "B[" << i << "] = " << B[i] << std::endl;
    // }

    // std::cout << "\nMatrix B_expected" << std::endl;
    // for (u_int32_t i = 0; i < M * N; i++)
    // {
    //     if ( i % M == 0)
    //     {
    //         std::cout << ""<< std::endl;    
    //     }
    //     std::cout << "B[" << i << "] = " << B_true[i] << std::endl;
    // }

    // for (u_int32_t i = 0; i < M * N; i++)
    // {
    //     REQUIRE(A[i] == Approx(A_expected[i]).margin(FLOAT_ERROR_MARGIN));
    //     // For polynomial approximation, relaxed margin
    //     REQUIRE(B[i] == Approx(B_expected[i]).margin(0.01f));
    // }

    // // Print comparison results
    std::cout << "\n=== Sigmoid Comparison Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Input\t\tTrue Sigmoid\tKernel Result\tDifference\tApprox (Taylor)" << std::endl;
    std::cout << "-----\t\t------------\t-------------\t----------\t--------------" << std::endl;
    
    for (u_int32_t i = 0; i < M * N; i++)
    {
        float difference = std::abs(B_true[i] - B[i]);
        std::cout << A[i] << "\t\t" 
                  << B_true[i] << "\t\t" 
                  << B[i] << "\t\t" 
                  << difference << "\t\t"
                  << B_expected[i] << std::endl;
        
        // Check accuracy - use a reasonable tolerance for interpolation
        REQUIRE(A[i] == Approx(A_expected[i]).margin(FLOAT_ERROR_MARGIN));
        REQUIRE(B[i] == Approx(B_true[i]).margin(0.01f)); // Allow 10% error for interpolation
    }

    // std::cout << "===================================\n" << std::endl;

    delete[] A;
    delete[] B;
    delete[] A_expected;
    delete[] B_expected;
    delete[] B_true;
}

TEST_CASE("Tests the sigmoid interpolation primitive with different M and N", "[sigmoid_primitive][parameterized]")
{
    uint32_t M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8);
    uint32_t N = GENERATE(1, 2, 3);
    test_sigmoid_interp_primitive(M, N);
}
