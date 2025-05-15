#include "Brgemm.h"
#include "Kernel.h"
#include <iostream>
#include <cstring>

#include <arm_neon.h>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <fstream>

#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"

using gpr_t = mini_jit::registers::gpr_t;
using simd_fp_t = mini_jit::registers::simd_fp_t;
using arr_spec_t = mini_jit::registers::arr_spec_t;

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
        std::memset(c, 0, 16 * 6 * sizeof(float));

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
        std::memset(c, 0, 16 * 6 * sizeof(float));

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


void benchmark_gemm_kernel( uint32_t M, 
                            uint32_t N, 
                            uint32_t K,
                            uint32_t br_size)
{
    uint32_t trans_a   = 0;
    uint32_t trans_b   = 0;
    uint32_t trans_c   = 0;
    uint32_t lda = M;
    uint32_t ldb = K;
    uint32_t ldc = M;
    uint32_t br_stride_a = 0;
    uint32_t br_stride_b = 0;
    dtype_t dtype      = dtype_t::fp32;

    // Allocate and initialize matrices
    float A[M * K];
    float B[K * N];
    float C[M * N];

    for (int i = 0; i < M * K; ++i)
    {
        A[i] = static_cast<float>(i % 13);
    }    

    for (int i = 0; i < K * N; ++i)
    {
        B[i] = static_cast<float>(i % 7);
    }

    mini_jit::Brgemm brgemm;
    brgemm.generate(M, N, K, br_size, trans_a, trans_b, trans_c, dtype_t::fp32);
    mini_jit::Brgemm::kernel_t kernel = brgemm.get_kernel();

    // Run between 1-2 seconds
    int64_t num_reps = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    double elapsed = 0.0;

    do {
        kernel(A, B, C, M, K, M, 0, 0);
        ++num_reps;
        auto now = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time).count() / 1e6;
    } while (elapsed < 1.5);

    // Calculate GFLOPs
    double gflops = (2.0 * M * N * K * num_reps) / (elapsed * 1e9);

    // Write CSV row
    std::ofstream csv("benchmark/gemm_perf.csv", std::ios::app);
    csv << M << "," << N << "," << K << ","
        << br_size << ","
        << trans_a << "," << trans_b << "," << trans_c << ","
        << lda << "," << ldb << "," << ldc << ","
        << 0 << "," << 0 << ","
        << num_reps << ","
        << elapsed << ","
        << gflops << "\n";
}

void benchmark_brgemm_kernel( uint32_t M, 
                              uint32_t N, 
                              uint32_t K,
                              uint32_t br_size )
{
    uint32_t trans_a   = 0;
    uint32_t trans_b   = 0;
    uint32_t trans_c   = 0;
    uint32_t lda = M;
    uint32_t ldb = K;
    uint32_t ldc = M;
    uint32_t br_stride_a = M * K;
    uint32_t br_stride_b = K * N;
    dtype_t dtype      = dtype_t::fp32;

    // Allocate and initialize matrices
    float A[M * K * br_size];
    float B[K * N * br_size];
    float C[M * N];

    for (int i = 0; i < M * K * br_size; ++i)
    {
        A[i] = static_cast<float>(i % 13);
    }    

    for (int i = 0; i < K * N * br_size; ++i)
    {
        B[i] = static_cast<float>(i % 7);
    }

    mini_jit::Brgemm brgemm;
    brgemm.generate(M, N, K, br_size, trans_a, trans_b, trans_c, dtype_t::fp32);
    mini_jit::Brgemm::kernel_t kernel = brgemm.get_kernel();

    // Run between 1-2 seconds
    int64_t num_reps = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    double elapsed = 0.0;

    do {
        kernel( A, B, C, M, K, M, M*K, K*N );
        ++num_reps;
        auto now = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time).count() / 1e6;
    } while (elapsed < 1.0);

    // Calculate GFLOPs
    double gflops = (2.0 * M * N * K * br_size * num_reps) / (elapsed * 1e9);

    // Write CSV row
    std::ofstream csv("benchmark/brgemm_perf.csv", std::ios::app);
    csv << M << "," << N << "," << K << ","
        << br_size << ","
        << trans_a << "," << trans_b << "," << trans_c << ","
        << lda << "," << ldb << "," << ldc << ","
        << br_stride_a << "," << br_stride_b << ","
        << num_reps << ","
        << elapsed << ","
        << gflops << "\n";
}



void microkernel_benchmark()
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
    std::memset(C_l1, 0, M_l1 * N_l1 * sizeof(float));

    int64_t l_iter = 10000 * 15000;
    std::string l1_matmul( "matmul_16_6_1" );

    std::cout << "\nBenchmarking Matmul_16_6_1 throughput ...\n";
    benchmark_thr( l_iter, l1_matmul, A_l1, B_l1, C_l1 );
    std::memset(C_l1, 0, M_l1 * N_l1 * sizeof(float));

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
    std::memset(C_l2, 0, M_l2 * N_l2 * sizeof(float));

    int64_t l2_iter = 10000 * 2000;
    std::string l2_matmul( "matmul_16_6_k" );

    std::cout << "\nBenchmarking Matmul_16_6_64 throughput ...\n";
    benchmark_thr( l2_iter, l2_matmul, A_l2, B_l2, C_l2 );
    std::memset(C_l2, 0, M_l2 * N_l2 * sizeof(float));
}

void gemm_benchmark()
{
    std::ofstream csv("benchmark/gemm_perf.csv");
    csv << "m,n,k,br_size,trans_a,trans_b,trans_c,ld_a,ld_b,ld_c,br_stride_a,br_stride_b,num_reps,time,gflops\n";


    for (int M = 1; M <= 64; ++M) 
    {
        for (int N = 1; N <= 64; ++N) 
        {
            for (int K : {1, 16, 32, 64, 128}) 
            {
                benchmark_gemm_kernel(M, N, K, 1);
            }
        }
    }
}

void brgemm_benchmark()
{
    std::ofstream csv("benchmark/brgemm_perf.csv");
    csv << "m,n,k,br_size,trans_a,trans_b,trans_c,ld_a,ld_b,ld_c,br_stride_a,br_stride_b,num_reps,time,gflops\n";


    for (int M = 1; M <= 64; ++M) 
    {
        for (int N = 1; N <= 64; ++N) 
        {
            for (int K : {1, 16, 32, 64, 128}) 
            {
                benchmark_brgemm_kernel(M, N, K, 16);
            }
        }
    }
}

int main() 
{
    microkernel_benchmark();
    
    gemm_benchmark();

    brgemm_benchmark();

    return 0;
}
