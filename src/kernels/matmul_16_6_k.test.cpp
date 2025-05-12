#include <catch2/catch.hpp>
#include <random>
#include <vector>
#include <iostream>

#include "matmul_16_6_k.h"
#include "Brgemm.h"

TEST_CASE("Tests the matmul_16_6_k microkernel function with random matrices", "[matmul_16_6_k]")
{
    const int M = 16;
    const int N = 6;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> k_dist(1, 100); // Random K values between 1 and 100
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int test = 0; test < 10; ++test) // Run the test 10 times with different K values
    {
        int K = k_dist(gen);
        std::vector<float> A(M * K);
        std::vector<float> B(K * N);
        std::vector<float> C(M * N);
        std::vector<float> C_expected(M * N);

        // Initialize matrices A and B with random values
        for (int i = 0; i < M * K; i++)
        {
            A[i] = dist(gen);
        }

        for (int i = 0; i < K * N; i++)
        {
            B[i] = dist(gen);
        }

        for (int i = 0; i < M * N; i++)
        {
            float value = dist(gen);
            C[i] = value;
            C_expected[i] = value;
        }

        for (int col = 0; col < N; ++col)
        {
            for (int row = 0; row < M; ++row)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    sum += A[row + k * M] * B[k + col * K];
                }
                C_expected[row + col * M] += sum;
            }
        }

        mini_jit::Brgemm l_brgemm;
        mini_jit::Brgemm::error_t l_ret = l_brgemm.generate(M, N, K, 4, 0, 0, 0, mini_jit::Brgemm::dtype_t::fp32);
        REQUIRE(l_ret == mini_jit::Brgemm::error_t::success);

        mini_jit::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();
        l_kernel(A.data(), B.data(), C.data(), M, K, M, 0, 0);

        for (int i = 0; i < M * N; i++)
        {
            REQUIRE(C[i] == Approx(C_expected[i]).epsilon(0.01));
        }
    }
}