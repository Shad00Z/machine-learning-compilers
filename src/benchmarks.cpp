#include "benchmarks/all_benchmarks.h"

#include "Brgemm.h"
#include <iostream>
#include <cstring>
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
                    << result.gflops << "\n";
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
                    << M * K << "," << K * N << ","
                    << result.numReps << ","
                    << result.elapsedSeconds << ","
                    << result.gflops << "\n";
            }
            csv.close();
        }
    }
    std::cout << "BRGEMM benchmark completed." << std::endl;
}

void matmul_benchmark(mini_jit::Benchmark &bench, std::ofstream &matmul_bm, std::string name)
{
    std::cout << "Running " << name << " benchmark" << std::endl;
    matmul_bm << "Running " << name << " benchmark" << std::endl;
    bench.run();
    mini_jit::Benchmark::benchmark_result result = bench.getResult();
    matmul_bm << "Total time (s):                  " << result.elapsedSeconds << std::endl;
    matmul_bm << "Total reps:                      " << result.numReps << std::endl;
    matmul_bm << "Total floating point operations: " << result.totalOperations << std::endl;
    matmul_bm << "Estimated GFLOPS/sec:            " << result.gflops << std::endl;
    matmul_bm << "--------------------------------------------------" << std::endl;
}

void unary_benchmark(mini_jit::Benchmark &bench, std::ofstream &unary_bm, std::string name)
{
    std::cout << "Running " << name << " benchmark" << std::endl;
    unary_bm << "Running " << name << " benchmark" << std::endl;
    bench.run();
    mini_jit::Benchmark::benchmark_result result = bench.getResult();
    unary_bm << "Total time (s):                       " << result.elapsedSeconds << std::endl;
    unary_bm << "Total reps:                           " << result.numReps << std::endl;
    unary_bm << "Total number of elements:             " << result.totalNumberElements << std::endl;
    unary_bm << "Total amount of processed data (GiB): " << result.totalDataProcessed << std::endl;
    unary_bm << "Bandwidth (GiB/s)                     " << result.gibps << std::endl;
    unary_bm << "--------------------------------------------------" << std::endl;
}

int main(int argc, char *argv[])
{
    // check console arguments
    bool has_gemm = false, has_brgemm = false, has_matmul = false, has_unary = false;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "gemm") == 0)
            has_gemm = true;
        else if (strcmp(argv[i], "brgemm") == 0)
            has_brgemm = true;
        else if (strcmp(argv[i], "matmul") == 0)
            has_matmul = true;
        else if (strcmp(argv[i], "unary") == 0)
            has_unary = true;
    }

    if (has_gemm)
    {
        gemm_benchmark();
    }

    if (has_brgemm)
    {
        brgemm_benchmark();
    }

    if (has_matmul)
    {
        mini_jit::benchmarks::Matmul_m_n_k_bench bench_mnk(3.0, 2048, 2048, 2048);
        mini_jit::benchmarks::Matmul_br_m_n_k_bench bench_brmnk(3.0, 1024, 1024, 1024, 16);
        std::ofstream matmul_bm("benchmarks/matmul_benchmarks.txt");
        matmul_benchmark(bench_mnk, matmul_bm, "Matmul_m_n_k 2048x2048x2048");
        matmul_benchmark(bench_brmnk, matmul_bm, "Matmul_br_m_n_k 1024x1024x1024 br=16");
        matmul_bm.close();
    }
    
    if (has_unary)
    {    
        const double RUN_TIME = 3.0;
        std::ofstream unary_bm("benchmarks/unary_benchmarks.txt");

        // identity_primitive benchmarks
        mini_jit::benchmarks::Identity_primitive_bench bench_identity_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::Identity_primitive_bench bench_identity_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::Identity_primitive_bench bench_identity_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::Identity_primitive_bench bench_identity_2048_2048(RUN_TIME, 2048, 2048);
        unary_benchmark(bench_identity_50_50, unary_bm, "identity_primitive 50x50");
        unary_benchmark(bench_identity_64_64, unary_bm, "identity_primitive 64x64");
        unary_benchmark(bench_identity_512_512, unary_bm, "identity_primitive 512x512");
        unary_benchmark(bench_identity_2048_2048, unary_bm, "identity_primitive 2048x2048");

        // identity_trans_primitive benchmarks
        mini_jit::benchmarks::Identity_trans_primitive_bench bench_identity_trans_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::Identity_trans_primitive_bench bench_identity_trans_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::Identity_trans_primitive_bench bench_identity_trans_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::Identity_trans_primitive_bench bench_identity_trans_2048_2048(RUN_TIME, 2048, 2048);
        unary_benchmark(bench_identity_trans_50_50, unary_bm, "identity_trans_primitive 50x50");
        unary_benchmark(bench_identity_trans_64_64, unary_bm, "identity_trans_primitive 64x64");
        unary_benchmark(bench_identity_trans_512_512, unary_bm, "identity_trans_primitive 512x512");
        unary_benchmark(bench_identity_trans_2048_2048, unary_bm, "identity_trans_primitive 2048x2048");

        // relu_primitive benchmarks
        mini_jit::benchmarks::Relu_primitive_bench bench_relu_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::Relu_primitive_bench bench_relu_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::Relu_primitive_bench bench_relu_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::Relu_primitive_bench bench_relu_2048_2048(RUN_TIME, 2048, 2048);
        unary_benchmark(bench_relu_50_50, unary_bm, "relu_primitive 50x50");
        unary_benchmark(bench_relu_64_64, unary_bm, "relu_primitive 64x64");
        unary_benchmark(bench_relu_512_512, unary_bm, "relu_primitive 512x512");
        unary_benchmark(bench_relu_2048_2048, unary_bm, "relu_primitive 2048x2048");

        // relu_trans_primitive benchmarks
        mini_jit::benchmarks::Relu_trans_primitive_bench bench_relu_trans_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::Relu_trans_primitive_bench bench_relu_trans_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::Relu_trans_primitive_bench bench_relu_trans_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::Relu_trans_primitive_bench bench_relu_trans_2048_2048(RUN_TIME, 2048, 2048);
        unary_benchmark(bench_relu_trans_50_50, unary_bm, "relu_trans_primitive 50x50");
        unary_benchmark(bench_relu_trans_64_64, unary_bm, "relu_trans_primitive 64x64");
        unary_benchmark(bench_relu_trans_512_512, unary_bm, "relu_trans_primitive 512x512");
        unary_benchmark(bench_relu_trans_2048_2048, unary_bm, "relu_trans_primitive 2048x2048");

        // zero_primitive benchmarks
        mini_jit::benchmarks::Zero_primitive_bench bench_zero_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::Zero_primitive_bench bench_zero_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::Zero_primitive_bench bench_zero_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::Zero_primitive_bench bench_zero_2048_2048(RUN_TIME, 2048, 2048);
        unary_benchmark(bench_zero_50_50, unary_bm, "zero_primitive 50x50");
        unary_benchmark(bench_zero_64_64, unary_bm, "zero_primitive 64x64");
        unary_benchmark(bench_zero_512_512, unary_bm, "zero_primitive 512x512");
        unary_benchmark(bench_zero_2048_2048, unary_bm, "zero_primitive 2048x2048");

        unary_bm.close();
    }

    return 0;
}
