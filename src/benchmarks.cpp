#include "benchmarks/all_benchmarks.h"

#include "Brgemm.h"
#include "Kernel.h"
#include <iostream>
#include <cstring>

#include <arm_neon.h>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <fstream>

void gemm_benchmark()
{
    std::cout << "Running GEMM benchmark..." << std::endl;
    std::string filename = "benchmarks/gemm_perf.csv";
    std::ofstream csv(filename);
    csv << "m,n,k,br_size,trans_a,trans_b,trans_c,ld_a,ld_b,ld_c,br_stride_a,br_stride_b,num_reps,time,gflops\n";
    csv.close();

    for (int M = 1; M <= 64; ++M)
    {
        for (int N = 1; N <= 64; ++N)
        {
            std::ofstream csv(filename, std::ios::app);
            for (int K : {1, 16, 32, 64, 128})
            {
                mini_jit::benchmarks::Matmul_m_n_k_bench bench(1.0, M, N, K);
                bench.run();
                mini_jit::Benchmark::benchmark_result result = bench.getResult();
                csv << M << "," << N << "," << K << ","
                    << 1 << ","
                    << 0 << "," << 0 << "," << 0 << ","
                    << M << "," << K << "," << M << ","
                    << 0 << "," << 0 << ","
                    << result.numReps << ","
                    << result.elapsedSeconds << ","
                    << result.gops << "\n";
            }
            csv.close();
        }
    }
    std::cout << "GEMM benchmark completed." << std::endl;
}

void brgemm_benchmark()
{
    std::cout << "Running BRGEMM benchmark..." << std::endl;
    std::string filename = "benchmarks/brgemm_perf.csv";
    std::ofstream csv(filename);
    csv << "m,n,k,br_size,trans_a,trans_b,trans_c,ld_a,ld_b,ld_c,br_stride_a,br_stride_b,num_reps,time,gflops\n";
    csv.close();

    for (int M = 1; M <= 64; ++M)
    {
        for (int N = 1; N <= 64; ++N)
        {
            std::ofstream csv(filename, std::ios::app);
            for (int K : {1, 16, 32, 64, 128})
            {
                mini_jit::benchmarks::Matmul_br_m_n_k_bench bench(1.0, M, N, K, 16);
                bench.run();
                mini_jit::Benchmark::benchmark_result result = bench.getResult();
                csv << M << "," << N << "," << K << ","
                    << 16 << ","
                    << 0 << "," << 0 << "," << 0 << ","
                    << M << "," << K << "," << M << ","
                    << M*K << "," << K*N << ","
                    << result.numReps << ","
                    << result.elapsedSeconds << ","
                    << result.gops << "\n";
            }
            csv.close();
        }
    }
    std::cout << "BRGEMM benchmark completed." << std::endl;
}

int main()
{
    //gemm_benchmark();

    brgemm_benchmark();

    return 0;
}
