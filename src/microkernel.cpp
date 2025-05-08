#include "InstGen.h"
#include "Brgemm.h"
#include "Kernel.h"
#include <iostream>
#include <cstring>

using gpr_t = mini_jit::InstGen::gpr_t;
using simd_fp_t = mini_jit::InstGen::simd_fp_t;
using arr_spec_t = mini_jit::InstGen::arr_spec_t;

using dtype_t = mini_jit::Brgemm::dtype_t;

int main()
{
  // Setup Matrix Multiplication
  const int M = 16;
  const int N = 6;
  const int K = 1;

  float A[M * K] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
  float B[K * N] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float C[M * N] = {0.0f};
  float C_expected[M * N] = {0.0f};

  mini_jit::Brgemm l_brgemm;

  // Print matrix A
  std::cout << "Matrix A:" << std::endl;
  for ( int i = 0; i < M; ++i )
  {
      for ( int j = 0; j < K; ++j )
      {
          std::cout << A[i * K + j] << " ";
      }
      std::cout << "\n";
  }

  // Print matrix B
  std::cout << "Matrix B:" << std::endl;
  for ( int i = 0; i < K; ++i )
  {
      for ( int j = 0; j < N; ++j )
      {
          std::cout << B[i * N + j] << " ";
      }
      std::cout << "\n";
  }

  l_brgemm.generate( M, N, K, 4, 0, 0, 0, dtype_t::fp32 );

  // Execute the kernel
  mini_jit::Brgemm::kernel_t l_func = l_brgemm.get_kernel();
  l_func( A, B, C, M, K, M, 0, 0 );

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

    std::cout << "Expected Matrix C (Result):\n";
    for (int i = 0; i < M; ++i) 
    {
        for (int j = 0; j < N; ++j) 
        {
            std::cout << C_expected[i + j * M] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Expected Matrix C (Result):\n";
    for (int i = 0; i < M; ++i) 
    {
        for (int j = 0; j < N; ++j) 
        {
            std::cout << C[i + j * M] << " ";
        }
        std::cout << std::endl;
    }

  return EXIT_SUCCESS;
}