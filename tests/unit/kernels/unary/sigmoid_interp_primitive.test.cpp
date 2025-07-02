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

    float sig_table[33] = {
        0.000335f, 0.000553f, 0.000911f, 0.001503f, 0.002473f, 0.004070f, 0.006693f,
        0.011109f, 0.017986f, 0.029312f, 0.047426f, 0.075858f, 0.119203f, 0.182426f,
        0.268941f, 0.377541f, 0.500000f, 0.622459f, 0.731059f, 0.817574f, 0.880797f,
        0.924142f, 0.952574f, 0.970688f, 0.982014f, 0.988891f, 0.993307f, 0.995930f,
        0.997527f, 0.998497f, 0.999089f, 0.999447f, 0.999665f
    };

    // float sig_table[65] = {
    //     0.000335f, 0.000431f, 0.000552f, 0.000707f, 0.000900f,
    //     0.001142f, 0.001442f, 0.001812f, 0.002265f, 0.002817f,
    //     0.003483f, 0.004283f, 0.005235f, 0.006362f, 0.007687f,
    //     0.009237f, 0.011038f, 0.013120f, 0.015511f, 0.018243f,
    //     0.021345f, 0.024847f, 0.028776f, 0.033157f, 0.038009f,
    //     0.043353f, 0.049203f, 0.055572f, 0.062469f, 0.069898f,
    //     0.077858f, 0.086345f, 0.095345f, 0.104844f, 0.114818f,
    //     0.125241f, 0.136085f, 0.147318f, 0.158909f, 0.170824f,
    //     0.183033f, 0.195505f, 0.208211f, 0.221123f, 0.234217f,
    //     0.247469f, 0.260857f, 0.274361f, 0.287966f, 0.301656f,
    //     0.315421f, 0.329250f, 0.343139f, 0.357085f, 0.371086f,
    //     0.385147f, 0.399273f, 0.413473f, 0.427758f, 0.442144f,
    //     0.456646f, 0.471286f, 0.486085f, 0.501068f, 0.516262f
    // };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-18.0f, 18.0f);

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
    l_kernel_t(A, B, sig_table, M, M);

    // std::cout << "\nMatrix B" << std::endl;
    // for (u_int32_t i = 0; i < M * N; i++)
    // {
    //     std::cout << "B[" << i << "] = " << B[i] << std::endl;
    // }

    // std::cout << "\nMatrix B_expected" << std::endl;
    // for (u_int32_t i = 0; i < M * N; i++)
    // {
    //     std::cout << "B[" << i << "] = " << B_true[i] << std::endl;
    // }

    // for (u_int32_t i = 0; i < M * N; i++)
    // {
    //     REQUIRE(A[i] == Approx(A_expected[i]).margin(FLOAT_ERROR_MARGIN));
    //     // For polynomial approximation, relaxed margin
    //     REQUIRE(B[i] == Approx(B_expected[i]).margin(0.01f));
    // }

    // // Print comparison results
    // std::cout << "\n=== Sigmoid Comparison Results ===" << std::endl;
    // std::cout << std::fixed << std::setprecision(6);
    // std::cout << "Input\t\tTrue Sigmoid\tKernel Result\tDifference\tApprox (Taylor)" << std::endl;
    // std::cout << "-----\t\t------------\t-------------\t----------\t--------------" << std::endl;
    
    for (u_int32_t i = 0; i < M * N; i++)
    {
        float difference = std::abs(B_true[i] - B[i]);
        // std::cout << A[i] << "\t\t" 
        //           << B_true[i] << "\t\t" 
        //           << B[i] << "\t\t" 
        //           << difference << "\t\t"
        //           << B_expected[i] << std::endl;
        
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
    uint32_t M = GENERATE(24);
    uint32_t N = GENERATE(1);
    test_sigmoid_interp_primitive(M, N);
}

TEST_CASE("Tests the sigmoid interpolation primitive with different M%4=1 and N", "[sigmoid_primitive][parameterized]")
{
    uint32_t M = GENERATE(7, 15);
    uint32_t N = GENERATE(1);
    test_sigmoid_interp_primitive(M, N);
}

// TEST_CASE("Tests the sigmoid interpolation primitive with M=50 and N=50", "[sigmoid_primitive][parameterized]")
// {
//     uint32_t M = GENERATE(1024);
//     uint32_t N = GENERATE(1024);
//     test_sigmoid_interp_primitive(M, N);
// }

// TEST_CASE("4 Tests the sigmoid interpolation primitive with different M and N", "[sigmoid_primitive][parameterized]")
// {
//     uint32_t M = GENERATE(4);
//     uint32_t N = GENERATE(1);
//     test_sigmoid_interp_primitive(M, N);
// }

// TEST_CASE("5 Tests the sigmoid interpolation primitive with different M and N", "[sigmoid_primitive][parameterized]")
// {
//     uint32_t M = GENERATE(4);
//     uint32_t N = GENERATE(1);
//     test_sigmoid_interp_primitive(M, N);
// }

// TEST_CASE("6 Tests the sigmoid interpolation primitive with different M and N", "[sigmoid_primitive][parameterized]")
// {
//     uint32_t M = GENERATE(4);
//     uint32_t N = GENERATE(1);
//     test_sigmoid_interp_primitive(M, N);
// }

// TEST_CASE("7 Tests the sigmoid interpolation primitive with different M and N", "[sigmoid_primitive][parameterized]")
// {
//     uint32_t M = GENERATE(4);
//     uint32_t N = GENERATE(1);
//     test_sigmoid_interp_primitive(M, N);
// }

// TEST_CASE("Tests the sigmoid primitive with larger M and N", "[sigmoid_primitive][large]")
// {
//     uint32_t M = 64;
//     uint32_t N = 65;
//     test_sigmoid_primitive(M, N);
// }
