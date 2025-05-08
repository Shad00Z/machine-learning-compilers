#include "InstGen.h"
#include "Brgemm.h"
#include "Kernel.h"
#include <iostream>
#include <cstring>

#include <arm_neon.h>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <cstring>

using gpr_t = mini_jit::InstGen::gpr_t;
using simd_fp_t = mini_jit::InstGen::simd_fp_t;
using arr_spec_t = mini_jit::InstGen::arr_spec_t;

using dtype_t = mini_jit::Brgemm::dtype_t;

/*
 * Benchmarks the throughput for the different MatMul (loop) versions.
 *
 * @param n: number of loop iterations.
 * @param instruction: a string selecting the instruction to benchmark.
 * @param a pointer to column-major matrix A.
 * @param b pointer to column-major matrix B.
 * @param c pointer to column-major matrix C.
 */
 void benchmark_thr( int64_t n, 
                     std::string instruction,
                     float const *a,
                     float const *b,
                     float *c ) 
{
    std::cout << "-----------------------------------------------\n";
    double elapsedTime = 1;

    mini_jit::Brgemm l_brgemm;

    std::string l1_matmul( "matmul_16_6_1" );
    int res_1 = l1_matmul.compare( instruction );

    std::string l2_matmul( "matmul_16_6_k" );
    int res_2 = l2_matmul.compare( instruction );

    double opsPerMatmul = 1;

    // Time measuring
    if ( res_1 == 0 )
    {
        l_brgemm.generate( 16, 6, 1, 4, 0, 0, 0, dtype_t::fp32 );
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            // Execute the kernel
            mini_jit::Brgemm::kernel_t l_func = l_brgemm.get_kernel();
            l_func( a, b, c, 16, 1, 16, 0, 0 );
        }
        std::memset( c, 0, sizeof( c ) );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < n; i++ )
        {
            // Execute the kernel
            mini_jit::Brgemm::kernel_t l_func = l_brgemm.get_kernel();
            l_func( a, b, c, 16, 1, 16, 0, 0 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (M x N x K) x (FMLA OPs)
        opsPerMatmul = (16 * 6 * 1) * 2;
    }
    else if ( res_2 == 0 )
    {
        l_brgemm.generate( 16, 6, 64, 4, 0, 0, 0, dtype_t::fp32 );
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            // Execute the kernel
            mini_jit::Brgemm::kernel_t l_func = l_brgemm.get_kernel();
            l_func( a, b, c, 16, 64, 16, 0, 0 );
        }
        std::memset( c, 0, sizeof( c ) );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < n; i++ )
        {
            // Execute the kernel
            mini_jit::Brgemm::kernel_t l_func = l_brgemm.get_kernel();
            l_func( a, b, c, 16, 64, 16, 0, 0 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (M x N x K) x (FMLA OPs)
        opsPerMatmul = (16 * 6 * 64) * 2;
    }  

    double loopIterations = n;
    double totalFLOPs = opsPerMatmul * loopIterations;

    double flopsPerSec = totalFLOPs / elapsedTime;
    double gflops = totalFLOPs / ( elapsedTime * 1e9 );

    std::cout << "Measuring throughput for " << "Instruction\n";
    std::cout << "Total time (s):   " << elapsedTime << "\n";
    std::cout << "Instructions per Second:   " << flopsPerSec << "\n";
    std::cout << "Estimated GFLOPS:   " << gflops << " GFLOPS/sec\n";
    std::cout << "-----------------------------------------------\n";
}

int main() 
{
    const int M_l1 = 16;
    const int N_l1 = 6;
    const int K_l1 = 1;

    float A_l1[M_l1 * K_l1];
    float B_l1[K_l1 * N_l1];
    float C_l1[M_l1 * N_l1];

    // Initialize matrices
    for (int i = 0; i < M_l1 * K_l1; ++i)
    {
        A_l1[i] = static_cast<float>(i);
    }
    for (int j = 0; j < K_l1 * N_l1; ++j)
    {
        B_l1[j] = static_cast<float>(j);
    }
    std::memset(C_l1, 0, sizeof(C_l1));

    int64_t l_iter = 10000 * 15000;
    std::string l1_matmul( "matmul_16_6_1" );

    std::cout << "\nBenchmarking Matmul_16_6_1 throughput ...\n";
    benchmark_thr( l_iter, l1_matmul, A_l1, B_l1, C_l1 );
    std::memset(C_l1, 0, sizeof(C_l1));

    const int M_l2 = 16;
    const int N_l2 = 6;
    const int K_l2 = 64;

    float A_l2[M_l2 * K_l2];
    float B_l2[K_l2 * N_l2];
    float C_l2[M_l2 * N_l2];

    // Initialize matrices
    for (int i = 0; i < M_l2 * K_l2; ++i)
    {
        A_l2[i] = static_cast<float>(i);
    }
    for (int j = 0; j < K_l2 * N_l2; ++j)
    {
        B_l2[j] = static_cast<float>(j);
    }
    std::memset(C_l2, 0, sizeof(C_l2));

    int64_t l2_iter = 10000 * 2000;
    std::string l2_matmul( "matmul_16_6_k" );

    std::cout << "\nBenchmarking Matmul_16_6_64 throughput ...\n";
    benchmark_thr( l2_iter, l2_matmul, A_l2, B_l2, C_l2 );
    std::memset(C_l2, 0, sizeof(C_l2));

    return 0;
}
