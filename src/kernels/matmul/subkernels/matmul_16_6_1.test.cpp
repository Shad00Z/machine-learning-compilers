#include <catch2/catch.hpp>
#include <iostream>

#include "matmul_16_6_1.h"
#include "Brgemm.h"
#include "constants.h"

TEST_CASE( "Tests the matmul_16_6_1 microkernel", "[matmul_16_6_1]" )
{
    const int M = 16;
    const int N = 6;
    const int K = 1;

    float A[M * K] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
    float B[K * N] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float C[M * N] = {0.0f};
    float C_expected[M * N] = {
        0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
        0.0f, 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        0.0f, 2.0f,  4.0f,  6.0f,  8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
        0.0f, 3.0f,  6.0f,  9.0f, 12.0f, 15.0f, 18.0f, 21.0f, 24.0f, 27.0f, 30.0f, 33.0f, 36.0f, 39.0f, 42.0f, 45.0f,
        0.0f, 4.0f,  8.0f, 12.0f, 16.0f, 20.0f, 24.0f, 28.0f, 32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
        0.0f, 5.0f, 10.0f, 15.0f, 20.0f, 25.0f, 30.0f, 35.0f, 40.0f, 45.0f, 50.0f, 55.0f, 60.0f, 65.0f, 70.0f, 75.0f
    };

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::matmul_16_6_1(l_kernel);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void*>(l_kernel.get_kernel()));
    l_kernel_t( A, B, C, M, K, M, 0, 0 );

    // Check the result
    for ( int i = 0; i < M * N; i++ )
    {
        REQUIRE( C[i] == Approx( C_expected[i] ).margin(FLOAT_ERROR_MARGIN) );
    }
}