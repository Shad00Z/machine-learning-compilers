#include <catch2/catch.hpp>
#include <random>
#include <iostream>

#include "identity_primitive.h"
#include "Unary.h"
#include "constants.h"

// TEST_CASE("Tests the identity primitive with random matrices", "[identity_primitive]")
// {
//     const int M = 5;
//     const int N = 5;
//     // int M = GENERATE(take(4, random(1, 64)));
//     // int N = GENERATE(take(4, random(1, 64)));

//     float A[M * N];
//     float B[N * M];
//     float B_expected[N * M];

//     // Initialize matrices A and B with random values
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> dist(1.0f, 30.0f);

        
//     for (int i = 0; i < M * N; i++)
//     {
//         A[i] = i;
//         B[i] = dist(gen);
//     }

//     printf("Matrix A: \n");
//     for (int i = 0; i < M; i++) // Row loop
//     {           
//         for (int j = 0; j < N; j++) // Column loop
//         {       
//             printf("%.1f ", A[i + j*M]); // Accessing element at (i,j)
//         }
//         printf("\n");
//     }

//     // Transpose matrix A
//     for (int i = 0; i < M; ++i)
//     {
//         for (int j = 0; j < N; ++j)
//         {
//             B_expected[i * N + j] = A[j * M + i];
//         }
//     }

//     printf("\nMatrix B_expected: \n");
//     for (int i = 0; i < N; i++) // Row loop
//     {           
//         for (int j = 0; j < M; j++) // Column loop
//         {       
//             printf("%f ", B_expected[j * N + i]); // Accessing element at (i,j)
//         }
//         printf("\n");
//     }

//     mini_jit::Kernel l_kernel;
//     mini_jit::kernels::unary::identity(l_kernel, M, N);
//     mini_jit::Unary::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Unary::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
//     l_kernel_t(A, B, M, N);

//     printf("\nMatrix B: \n");
//     for (int i = 0; i < N; i++) // Row loop
//     {           
//         for (int j = 0; j < M; j++) // Column loop
//         {       
//             printf("%f ", B[j * N + i]); // Accessing element at (i,j)
//         }
//         printf("\n");
//     }

    

//     for (int i = 0; i < M * N; i++)
//     {
//         REQUIRE(B[i] == Approx(B_expected[i]).margin(FLOAT_ERROR_MARGIN));
//     }
// }

TEST_CASE("Tests the identity primitive with random matrices", "[identity_primitive][parameterized]")
{
    int M = GENERATE(take(8, random(1, 64)));
    int N = GENERATE(take(8, random(1, 64)));

    float* A = new float[M * N];
    float* B = new float[N * M];
    float* A_expected = new float[M * N];
    float* B_expected = new float[N * M];

    // Initialize matrices A and B with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 30.0f);

    for (int i = 0; i < M * N; i++)
    {
        A[i] = i;
        A_expected[i] = A[i];
        B[i] = dist(gen);
    }

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            B_expected[i * N + j] = A[j * M + i];
        }
    }

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::unary::identity(l_kernel, M, N);
    mini_jit::Unary::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Unary::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t(A, B, M, N);

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
