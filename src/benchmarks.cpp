#include "benchmarks/all_benchmarks.h"

#include "Brgemm.h"
#include "ir/Optimizer.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <climits>

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
                mini_jit::benchmarks::MatmulMNKBench bench(1.0, M, N, K);
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
                mini_jit::benchmarks::MatmulBrMNKBench bench(1.0, M, N, K, 16);
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

void print_bandwidth(mini_jit::Benchmark &bench,
                     std::ofstream &bm_file,
                     std::string name)
{
    std::cout << "Running " << name << " benchmark" << std::endl;
    bm_file << "Running " << name << " benchmark" << std::endl;
    bench.run();
    mini_jit::Benchmark::benchmark_result result = bench.getResult();
    bm_file << "Total time (s):                       " << result.elapsedSeconds << std::endl;
    bm_file << "Total reps:                           " << result.numReps << std::endl;
    bm_file << "Total number of elements:             " << result.totalNumberElements << std::endl;
    bm_file << "Total amount of processed data (GiB): " << result.totalDataProcessed << std::endl;
    bm_file << "Bandwidth (GiB/s)                     " << result.gibps << std::endl;
    bm_file << "--------------------------------------------------" << std::endl;
}

void print_throughput(mini_jit::Benchmark &bench,
                      std::ofstream &bm_file,
                      std::string name)
{
    std::cout << "Running " << name << " benchmark" << std::endl;
    bm_file << "Running " << name << " benchmark" << std::endl;
    bench.run();
    mini_jit::Benchmark::benchmark_result result = bench.getResult();
    bm_file << "Total time (s):                       " << result.elapsedSeconds << std::endl;
    bm_file << "Total reps:                           " << result.numReps << std::endl;
    bm_file << "Total floating point operations:      " << result.totalOperations << std::endl;
    bm_file << "Estimated GFLOPS/sec:                 " << result.gflops << std::endl;
    bm_file << "--------------------------------------------------" << std::endl;
}

void optimized_tensor_benchmark(std::ofstream &top_opt_bm,
                                int64_t thread_target,
                                int64_t max_kernel_size,
                                int64_t min_kernel_size)
{
    const double RUN_TIME = 3.0;

    std::vector<mini_jit::dim_t> l_dims = {mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::k};
    std::vector<mini_jit::exec_t> l_execs = {mini_jit::exec_t::seq, mini_jit::exec_t::seq, mini_jit::exec_t::seq};
    std::vector<int64_t> l_sizes = {1600, 1600, 1600};
    std::vector<int64_t> l_strides_in0 = {1, 0, 1600};
    std::vector<int64_t> l_strides_in1 = {0, 1600, 1};
    std::vector<int64_t> l_strides_out = {1, 1600, 0};

    mini_jit::ir::Optimizer::optimize(l_dims,
                                      l_execs,
                                      l_sizes,
                                      l_strides_in0,
                                      l_strides_in1,
                                      l_strides_out,
                                      thread_target,
                                      max_kernel_size,
                                      min_kernel_size);

    int l_prim_count = 0;
    for (const auto &exec : l_execs)
    {
        if (exec == mini_jit::exec_t::prim)
        {
            l_prim_count++;
        }
    }

    mini_jit::benchmarks::TensorOperationBench tensor_bench(RUN_TIME,
                                                            mini_jit::dtype_t::fp32,
                                                            mini_jit::ptype_t::none,
                                                            l_prim_count == 4 ? mini_jit::ptype_t::brgemm : mini_jit::ptype_t::gemm,
                                                            mini_jit::ptype_t::none,
                                                            l_dims,
                                                            l_execs,
                                                            l_sizes,
                                                            l_strides_in0,
                                                            l_strides_in1,
                                                            l_strides_out);

    top_opt_bm << "Running SharedTensorOperationBench benchmark (thread_target: " << thread_target << ", max_kernel_size: " << max_kernel_size << ")" << std::endl;
    std::cout << "Running SharedTensorOperationBench benchmark (thread_target: " << thread_target << ", max_kernel_size: " << max_kernel_size << ")" << std::endl;
    tensor_bench.run();
    mini_jit::Benchmark::benchmark_result result = tensor_bench.getResult();
    top_opt_bm << "Total time (s):                  " << result.elapsedSeconds << std::endl;
    top_opt_bm << "Total reps:                      " << result.numReps << std::endl;
    top_opt_bm << "Total floating point operations: " << result.totalOperations << std::endl;
    top_opt_bm << "Estimated GFLOPS/sec:            " << result.gflops << std::endl;
    top_opt_bm << "--------------------------------------------------" << std::endl;
}

void optimized_tensor_benchmark_einsum(std::ofstream &top_opt_bm,
                                       int64_t thread_target,
                                       int64_t max_kernel_size,
                                       int64_t min_kernel_size)
{
    const double RUN_TIME = 3.0;

    std::string expression = "[2,0],[1,2]->[1,0]";
    std::vector<int64_t> dimension_sizes = {1600, 1600, 1600};
    mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

    // [2,0] -> A
    // [1,2] -> B
    std::map<std::string, void const *> tensor_inputs;

    const int64_t SIZE_A = dimension_sizes[2] * dimension_sizes[0];
    const int64_t SIZE_B = dimension_sizes[1] * dimension_sizes[2];

    float *tensor_A = new float[SIZE_A];
    float *tensor_B = new float[SIZE_B];

    tensor_inputs["2,0"] = tensor_A;
    tensor_inputs["1,2"] = tensor_B;

    // init matrices
    for (int64_t i = 0; i < SIZE_A; ++i)
    {
        tensor_A[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_B; ++i)
    {
        tensor_B[i] = i % 100;
    }

    mini_jit::benchmarks::EinsumTreeBench tensor_bench(RUN_TIME,
                                                       expression,
                                                       dimension_sizes,
                                                       dtype,
                                                       thread_target,
                                                       max_kernel_size,
                                                       min_kernel_size,
                                                       tensor_inputs);

    delete[] tensor_A;
    delete[] tensor_B;

    top_opt_bm << "Running EinsumTensorOperationBench benchmark (thread_target: " << thread_target << ", max_kernel_size: " << max_kernel_size << ")" << std::endl;
    std::cout << "Running EinsumTensorOperationBench benchmark (thread_target: " << thread_target << ", max_kernel_size: " << max_kernel_size << ")" << std::endl;
    tensor_bench.run();
    mini_jit::Benchmark::benchmark_result result = tensor_bench.getResult();
    top_opt_bm << "Total time (s):                  " << result.elapsedSeconds << std::endl;
    top_opt_bm << "Total reps:                      " << result.numReps << std::endl;
    top_opt_bm << "Total floating point operations: " << result.totalOperations << std::endl;
    top_opt_bm << "Estimated GFLOPS/sec:            " << result.gflops << std::endl;
    top_opt_bm << "--------------------------------------------------" << std::endl;
}

void einsum_benchmark_1(std::ofstream &einsum_bm,
                        double RUN_TIME,
                        int64_t thread_target,
                        int64_t max_kernel_size,
                        int64_t min_kernel_size)
{
    std::string expression = "[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]";
    std::vector<int64_t> dimension_sizes = {100, 72, 128, 128, 3, 71, 305, 32, 3};
    mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

    // [8,4] -> A
    // [7,3,8] -> B
    // [2,6,7] -> C
    // [1,5,6] -> D
    // [0,5] -> E
    std::map<std::string, void const *> tensor_inputs;
    const int64_t SIZE_A = dimension_sizes[8] * dimension_sizes[4];
    const int64_t SIZE_B = dimension_sizes[7] * dimension_sizes[3] * dimension_sizes[8];
    const int64_t SIZE_C = dimension_sizes[2] * dimension_sizes[6] * dimension_sizes[7];
    const int64_t SIZE_D = dimension_sizes[1] * dimension_sizes[5] * dimension_sizes[6];
    const int64_t SIZE_E = dimension_sizes[0] * dimension_sizes[5];
    float *tensor_A = new float[SIZE_A];
    float *tensor_B = new float[SIZE_B];
    float *tensor_C = new float[SIZE_C];
    float *tensor_D = new float[SIZE_D];
    float *tensor_E = new float[SIZE_E];

    tensor_inputs["8,4"] = tensor_A;
    tensor_inputs["7,3,8"] = tensor_B;
    tensor_inputs["2,6,7"] = tensor_C;
    tensor_inputs["1,5,6"] = tensor_D;
    tensor_inputs["0,5"] = tensor_E;

    // init matrices
    for (int64_t i = 0; i < SIZE_A; ++i)
    {
        tensor_A[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_B; ++i)
    {
        tensor_B[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_C; ++i)
    {
        tensor_C[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_D; ++i)
    {
        tensor_D[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_E; ++i)
    {
        tensor_E[i] = i % 100;
    }

    mini_jit::benchmarks::EinsumTreeBench einsum_bench(RUN_TIME,
                                                       expression,
                                                       dimension_sizes,
                                                       dtype,
                                                       thread_target,
                                                       max_kernel_size,
                                                       min_kernel_size,
                                                       tensor_inputs);

    einsum_bm << "Running EinsumTree benchmark #1" << std::endl;
    std::cout << "Running EinsumTree benchmark #1" << std::endl;
    einsum_bench.run();
    mini_jit::Benchmark::benchmark_result result = einsum_bench.getResult();
    einsum_bm << "Total time (s):                  " << result.elapsedSeconds << std::endl;
    einsum_bm << "Total reps:                      " << result.numReps << std::endl;
    einsum_bm << "Total floating point operations: " << result.totalOperations << std::endl;
    einsum_bm << "Estimated GFLOPS/sec:            " << result.gflops << std::endl;
    einsum_bm << "--------------------------------------------------" << std::endl;

    delete[] tensor_A;
    delete[] tensor_B;
    delete[] tensor_C;
    delete[] tensor_D;
    delete[] tensor_E;
}

void einsum_benchmark_2(std::ofstream &einsum_bm,
                        double RUN_TIME,
                        int64_t thread_target,
                        int64_t max_kernel_size,
                        int64_t min_kernel_size)
{
    std::string expression = "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]";
    std::vector<int64_t> dimension_sizes = {60, 60, 20, 20, 8, 8, 8, 8, 8, 8};
    mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

    //     // [3,6,8,9] -> *A
    //     // [2,5,7,9] -> *B
    //     // [0,4,5,6] -> *C
    //     // [1,4,7,8] -> *D
    std::map<std::string, void const *> tensor_inputs;

    const int64_t SIZE_A = dimension_sizes[3] * dimension_sizes[6] * dimension_sizes[8] * dimension_sizes[9];
    const int64_t SIZE_B = dimension_sizes[2] * dimension_sizes[5] * dimension_sizes[7] * dimension_sizes[9];
    const int64_t SIZE_C = dimension_sizes[0] * dimension_sizes[4] * dimension_sizes[5] * dimension_sizes[6];
    const int64_t SIZE_D = dimension_sizes[1] * dimension_sizes[4] * dimension_sizes[7] * dimension_sizes[8];

    float *tensor_A = new float[SIZE_A];
    float *tensor_B = new float[SIZE_B];
    float *tensor_C = new float[SIZE_C];
    float *tensor_D = new float[SIZE_D];

    tensor_inputs["3,6,8,9"] = tensor_A;
    tensor_inputs["2,5,7,9"] = tensor_B;
    tensor_inputs["0,4,5,6"] = tensor_C;
    tensor_inputs["1,4,7,8"] = tensor_D;

    // init matrices
    for (int64_t i = 0; i < SIZE_A; ++i)
    {
        tensor_A[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_B; ++i)
    {
        tensor_B[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_C; ++i)
    {
        tensor_C[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_D; ++i)
    {
        tensor_D[i] = i % 100;
    }

    mini_jit::benchmarks::EinsumTreeBench einsum_bench(RUN_TIME,
                                                       expression,
                                                       dimension_sizes,
                                                       dtype,
                                                       thread_target,
                                                       max_kernel_size,
                                                       min_kernel_size,
                                                       tensor_inputs);

    einsum_bm << "Running EinsumTree benchmark #2" << std::endl;
    std::cout << "Running EinsumTree benchmark #2" << std::endl;
    einsum_bench.run();
    mini_jit::Benchmark::benchmark_result result = einsum_bench.getResult();
    einsum_bm << "Total time (s):                  " << result.elapsedSeconds << std::endl;
    einsum_bm << "Total reps:                      " << result.numReps << std::endl;
    einsum_bm << "Total floating point operations: " << result.totalOperations << std::endl;
    einsum_bm << "Estimated GFLOPS/sec:            " << result.gflops << std::endl;
    einsum_bm << "--------------------------------------------------" << std::endl;

    delete[] tensor_A;
    delete[] tensor_B;
    delete[] tensor_C;
    delete[] tensor_D;
}

void einsum_benchmark_optimization_example_1(std::ofstream &einsum_bm,
                                             double RUN_TIME,
                                             int64_t thread_target,
                                             int64_t max_kernel_size,
                                             int64_t min_kernel_size)
{
    std::string expression = "[[7,3,8],[8,4]->[7,3,4]],[[0,5],[[5,1,6],[6,2,7]->[5,1,2,7]]->[0,1,2,7]]->[0,1,2,3,4]";
    std::vector<int64_t> dimension_sizes = {100, 72, 128, 128, 3, 71, 305, 32, 3};
    mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

    //     // [7,3,8] -> *A
    //     // [8,4] -> *B
    //     // [0,5] -> *C
    //     // [5,1,6] -> *D
    //     // [6,2,7] -> *E
    std::map<std::string, void const *> tensor_inputs;

    const int64_t SIZE_A = dimension_sizes[7] * dimension_sizes[3] * dimension_sizes[8];
    const int64_t SIZE_B = dimension_sizes[8] * dimension_sizes[4];
    const int64_t SIZE_C = dimension_sizes[0] * dimension_sizes[5];
    const int64_t SIZE_D = dimension_sizes[5] * dimension_sizes[1] * dimension_sizes[6];
    const int64_t SIZE_E = dimension_sizes[6] * dimension_sizes[2] * dimension_sizes[7];

    float *tensor_A = new float[SIZE_A];
    float *tensor_B = new float[SIZE_B];
    float *tensor_C = new float[SIZE_C];
    float *tensor_D = new float[SIZE_D];
    float *tensor_E = new float[SIZE_E];

    tensor_inputs["7,3,8"] = tensor_A;
    tensor_inputs["8,4"] = tensor_B;
    tensor_inputs["0,5"] = tensor_C;
    tensor_inputs["5,1,6"] = tensor_D;
    tensor_inputs["6,2,7"] = tensor_E;

    // init matrices
    for (int64_t i = 0; i < SIZE_A; ++i)
    {
        tensor_A[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_B; ++i)
    {
        tensor_B[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_C; ++i)
    {
        tensor_C[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_D; ++i)
    {
        tensor_D[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_E; ++i)
    {
        tensor_E[i] = i % 100;
    }

    mini_jit::benchmarks::EinsumTreeBench einsum_bench(RUN_TIME,
                                                       expression,
                                                       dimension_sizes,
                                                       dtype,
                                                       thread_target,
                                                       max_kernel_size,
                                                       min_kernel_size,
                                                       tensor_inputs);

    einsum_bm << "Running EinsumTree benchmark - Optimization Example #1" << std::endl;
    std::cout << "Running EinsumTree benchmark - Optimization Example #1" << std::endl;
    einsum_bench.run();
    mini_jit::Benchmark::benchmark_result result = einsum_bench.getResult();
    einsum_bm << "Total time (s):                  " << result.elapsedSeconds << std::endl;
    einsum_bm << "Total reps:                      " << result.numReps << std::endl;
    einsum_bm << "Total floating point operations: " << result.totalOperations << std::endl;
    einsum_bm << "Estimated GFLOPS/sec:            " << result.gflops << std::endl;
    einsum_bm << "--------------------------------------------------" << std::endl;

    delete[] tensor_A;
    delete[] tensor_B;
    delete[] tensor_C;
    delete[] tensor_D;
    delete[] tensor_E;
}

void einsum_benchmark_optimization_example_2(std::ofstream &einsum_bm,
                                             double RUN_TIME,
                                             int64_t thread_target,
                                             int64_t max_kernel_size,
                                             int64_t min_kernel_size)
{
    std::string expression = "[1,4,7,8],[[0,4,5,6],[[2,5,7,9],[3,6,8,9]->[2,5,7,3,6,8]]->[0,4,2,7,3,8]]->[0,1,2,3]";
    std::vector<int64_t> dimension_sizes = {60, 60, 20, 20, 8, 8, 8, 8, 8, 8};
    mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

    //     // [1,4,7,8] -> *A
    //     // [0,4,5,6] -> *B
    //     // [2,5,7,9] -> *C
    //     // [3,6,8,9] -> *D
    std::map<std::string, void const *> tensor_inputs;

    const int64_t SIZE_A = dimension_sizes[1] * dimension_sizes[4] * dimension_sizes[7] * dimension_sizes[8];
    const int64_t SIZE_B = dimension_sizes[0] * dimension_sizes[4] * dimension_sizes[5] * dimension_sizes[6];
    const int64_t SIZE_C = dimension_sizes[2] * dimension_sizes[5] * dimension_sizes[7] * dimension_sizes[9];
    const int64_t SIZE_D = dimension_sizes[3] * dimension_sizes[6] * dimension_sizes[8] * dimension_sizes[9];

    float *tensor_A = new float[SIZE_A];
    float *tensor_B = new float[SIZE_B];
    float *tensor_C = new float[SIZE_C];
    float *tensor_D = new float[SIZE_D];

    tensor_inputs["1,4,7,8"] = tensor_A;
    tensor_inputs["0,4,5,6"] = tensor_B;
    tensor_inputs["2,5,7,9"] = tensor_C;
    tensor_inputs["3,6,8,9"] = tensor_D;

    // init matrices
    for (int64_t i = 0; i < SIZE_A; ++i)
    {
        tensor_A[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_B; ++i)
    {
        tensor_B[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_C; ++i)
    {
        tensor_C[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_D; ++i)
    {
        tensor_D[i] = i % 100;
    }

    mini_jit::benchmarks::EinsumTreeBench einsum_bench(RUN_TIME,
                                                       expression,
                                                       dimension_sizes,
                                                       dtype,
                                                       thread_target,
                                                       max_kernel_size,
                                                       min_kernel_size,
                                                       tensor_inputs);

    einsum_bm << "Running EinsumTree benchmark - Optimization Example #2" << std::endl;
    std::cout << "Running EinsumTree benchmark - Optimization Example #2" << std::endl;
    einsum_bench.run();
    mini_jit::Benchmark::benchmark_result result = einsum_bench.getResult();
    einsum_bm << "Total time (s):                  " << result.elapsedSeconds << std::endl;
    einsum_bm << "Total reps:                      " << result.numReps << std::endl;
    einsum_bm << "Total floating point operations: " << result.totalOperations << std::endl;
    einsum_bm << "Estimated GFLOPS/sec:            " << result.gflops << std::endl;
    einsum_bm << "--------------------------------------------------" << std::endl;

    delete[] tensor_A;
    delete[] tensor_B;
    delete[] tensor_C;
    delete[] tensor_D;
}

void einsum_benchmark_optimization_example_3(std::ofstream &einsum_bm,
                                             double RUN_TIME, int64_t thread_target,
                                             int64_t max_kernel_size,
                                             int64_t min_kernel_size)
{
    std::string expression = "[[2,7,3],[3,8,4]->[2,7,8,4]],[[4,9,0],[[0,5,1],[1,6,2]->[0,5,6,2]]->[4,9,5,6,2]]->[5,6,7,8,9]";
    std::vector<int64_t> dimension_sizes = {40, 40, 40, 40, 40, 25, 25, 25, 25, 25};
    mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

    //     // [2,7,3] -> *A
    //     // [3,8,4] -> *B
    //     // [4,9,0] -> *C
    //     // [0,5,1] -> *D
    //     // [1,6,2] -> *E
    std::map<std::string, void const *> tensor_inputs;

    const int64_t SIZE_A = dimension_sizes[2] * dimension_sizes[7] * dimension_sizes[3];
    const int64_t SIZE_B = dimension_sizes[3] * dimension_sizes[8] * dimension_sizes[4];
    const int64_t SIZE_C = dimension_sizes[4] * dimension_sizes[9] * dimension_sizes[0];
    const int64_t SIZE_D = dimension_sizes[0] * dimension_sizes[5] * dimension_sizes[1];
    const int64_t SIZE_E = dimension_sizes[1] * dimension_sizes[6] * dimension_sizes[2];

    float *tensor_A = new float[SIZE_A];
    float *tensor_B = new float[SIZE_B];
    float *tensor_C = new float[SIZE_C];
    float *tensor_D = new float[SIZE_D];
    float *tensor_E = new float[SIZE_E];

    tensor_inputs["2,7,3"] = tensor_A;
    tensor_inputs["3,8,4"] = tensor_B;
    tensor_inputs["4,9,0"] = tensor_C;
    tensor_inputs["0,5,1"] = tensor_D;
    tensor_inputs["1,6,2"] = tensor_E;

    // init matrices
    for (int64_t i = 0; i < SIZE_A; ++i)
    {
        tensor_A[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_B; ++i)
    {
        tensor_B[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_C; ++i)
    {
        tensor_C[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_D; ++i)
    {
        tensor_D[i] = i % 100;
    }
    for (int64_t i = 0; i < SIZE_E; ++i)
    {
        tensor_E[i] = i % 100;
    }

    mini_jit::benchmarks::EinsumTreeBench einsum_bench(RUN_TIME,
                                                       expression,
                                                       dimension_sizes,
                                                       dtype,
                                                       thread_target,
                                                       max_kernel_size,
                                                       min_kernel_size,
                                                       tensor_inputs);

    einsum_bm << "Running EinsumTree benchmark - Optimization Example #3" << std::endl;
    std::cout << "Running EinsumTree benchmark - Optimization Example #3" << std::endl;
    einsum_bench.run();
    mini_jit::Benchmark::benchmark_result result = einsum_bench.getResult();
    einsum_bm << "Total time (s):                  " << result.elapsedSeconds << std::endl;
    einsum_bm << "Total reps:                      " << result.numReps << std::endl;
    einsum_bm << "Total floating point operations: " << result.totalOperations << std::endl;
    einsum_bm << "Estimated GFLOPS/sec:            " << result.gflops << std::endl;
    einsum_bm << "--------------------------------------------------" << std::endl;

    delete[] tensor_A;
    delete[] tensor_B;
    delete[] tensor_C;
    delete[] tensor_D;
    delete[] tensor_E;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN BENCHMARK ENTRY POINT
////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    // check console arguments
    bool has_gemm = false;
    bool has_brgemm = false;
    bool has_matmul = false;
    bool has_unary = false;
    bool has_tensor_operations = false;
    bool has_shared_tensor_operations = false;
    bool has_opt_tensor_operations = false;
    bool has_einsum_benchmark = false;
    bool has_opt_einsum_benchmark = false;
    bool has_reciprocal = false;
    bool has_sigmoid = false;
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
        else if (strcmp(argv[i], "top") == 0)
            has_tensor_operations = true;
        else if (strcmp(argv[i], "top-shared") == 0)
            has_shared_tensor_operations = true;
        else if (strcmp(argv[i], "top-opt") == 0)
            has_opt_tensor_operations = true;
        else if (strcmp(argv[i], "einsum") == 0)
            has_einsum_benchmark = true;
        else if (strcmp(argv[i], "opt-einsum") == 0)
            has_opt_einsum_benchmark = true;
        else if (strcmp(argv[i], "reciprocal") == 0)
            has_reciprocal = true;
        else if (strcmp(argv[i], "sigmoid") == 0)
            has_sigmoid = true;
        else
        {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [gemm|brgemm|matmul|unary|top|top-shared|top-opt|einsum|opt-einsum|reciprocal|sigmoid]" << std::endl;
            return 1;
        }
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
        mini_jit::benchmarks::MatmulMNKBench bench_mnk(3.0, 2048, 2048, 2048);
        mini_jit::benchmarks::MatmulBrMNKBench bench_brmnk(3.0, 1024, 1024, 1024, 16);
        std::ofstream matmul_bm("benchmarks/matmul_benchmarks.txt");
        print_throughput(bench_mnk, matmul_bm, "MatmulMNKBench 2048x2048x2048");
        print_throughput(bench_brmnk, matmul_bm, "MatmulBrMNKBench 1024x1024x1024 br=16");
        matmul_bm.close();
    }

    if (has_unary)
    {
        const double RUN_TIME = 3.0;
        std::ofstream unary_bm("benchmarks/print_bandwidths.txt");

        // IdentityPrimitiveBench benchmarks
        mini_jit::benchmarks::IdentityPrimitiveBench bench_identity_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::IdentityPrimitiveBench bench_identity_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::IdentityPrimitiveBench bench_identity_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::IdentityPrimitiveBench bench_identity_2048_2048(RUN_TIME, 2048, 2048);
        print_bandwidth(bench_identity_50_50, unary_bm, "IdentityPrimitiveBench 50x50");
        print_bandwidth(bench_identity_64_64, unary_bm, "IdentityPrimitiveBench 64x64");
        print_bandwidth(bench_identity_512_512, unary_bm, "IdentityPrimitiveBench 512x512");
        print_bandwidth(bench_identity_2048_2048, unary_bm, "IdentityPrimitiveBench 2048x2048");

        // IdentityTransPrimitiveBench benchmarks
        mini_jit::benchmarks::IdentityTransPrimitiveBench bench_identity_trans_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::IdentityTransPrimitiveBench bench_identity_trans_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::IdentityTransPrimitiveBench bench_identity_trans_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::IdentityTransPrimitiveBench bench_identity_trans_2048_2048(RUN_TIME, 2048, 2048);
        print_bandwidth(bench_identity_trans_50_50, unary_bm, "IdentityTransPrimitiveBench 50x50");
        print_bandwidth(bench_identity_trans_64_64, unary_bm, "IdentityTransPrimitiveBench 64x64");
        print_bandwidth(bench_identity_trans_512_512, unary_bm, "IdentityTransPrimitiveBench 512x512");
        print_bandwidth(bench_identity_trans_2048_2048, unary_bm, "IdentityTransPrimitiveBench 2048x2048");

        // ReluPrimitiveBench benchmarks
        mini_jit::benchmarks::ReLUPrimitiveBench bench_relu_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::ReLUPrimitiveBench bench_relu_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::ReLUPrimitiveBench bench_relu_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::ReLUPrimitiveBench bench_relu_2048_2048(RUN_TIME, 2048, 2048);
        print_bandwidth(bench_relu_50_50, unary_bm, "ReluPrimitiveBench 50x50");
        print_bandwidth(bench_relu_64_64, unary_bm, "ReluPrimitiveBench 64x64");
        print_bandwidth(bench_relu_512_512, unary_bm, "ReluPrimitiveBench 512x512");
        print_bandwidth(bench_relu_2048_2048, unary_bm, "ReluPrimitiveBench 2048x2048");

        // ReluTransPrimitiveBench benchmarks
        mini_jit::benchmarks::ReLUTransPrimitiveBench bench_relu_trans_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::ReLUTransPrimitiveBench bench_relu_trans_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::ReLUTransPrimitiveBench bench_relu_trans_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::ReLUTransPrimitiveBench bench_relu_trans_2048_2048(RUN_TIME, 2048, 2048);
        print_bandwidth(bench_relu_trans_50_50, unary_bm, "ReluTransPrimitiveBench 50x50");
        print_bandwidth(bench_relu_trans_64_64, unary_bm, "ReluTransPrimitiveBench 64x64");
        print_bandwidth(bench_relu_trans_512_512, unary_bm, "ReluTransPrimitiveBench 512x512");
        print_bandwidth(bench_relu_trans_2048_2048, unary_bm, "ReluTransPrimitiveBench 2048x2048");

        // SquarePrimitiveBench benchmarks
        mini_jit::benchmarks::SquarePrimitiveBench bench_square_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::SquarePrimitiveBench bench_square_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::SquarePrimitiveBench bench_square_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::SquarePrimitiveBench bench_square_2048_2048(RUN_TIME, 2048, 2048);
        print_throughput(bench_square_50_50, unary_bm, "SquarePrimitiveBench 50x50");
        print_throughput(bench_square_64_64, unary_bm, "SquarePrimitiveBench 64x64");
        print_throughput(bench_square_512_512, unary_bm, "SquarePrimitiveBench 512x512");
        print_throughput(bench_square_2048_2048, unary_bm, "SquarePrimitiveBench 2048x2048");

        // SquareTransPrimitiveBench benchmarks
        mini_jit::benchmarks::SquareTransPrimitiveBench bench_square_trans_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::SquareTransPrimitiveBench bench_square_trans_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::SquareTransPrimitiveBench bench_square_trans_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::SquareTransPrimitiveBench bench_square_trans_2048_2048(RUN_TIME, 2048, 2048);
        print_throughput(bench_square_trans_50_50, unary_bm, "SquareTransPrimitiveBench 50x50");
        print_throughput(bench_square_trans_64_64, unary_bm, "SquareTransPrimitiveBench 64x64");
        print_throughput(bench_square_trans_512_512, unary_bm, "SquareTransPrimitiveBench 512x512");
        print_throughput(bench_square_trans_2048_2048, unary_bm, "SquareTransPrimitiveBench 2048x2048");

        // ZeroEorPrimitiveBench benchmarks
        mini_jit::benchmarks::ZeroEorPrimitiveBench bench_zero_eor_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::ZeroEorPrimitiveBench bench_zero_eor_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::ZeroEorPrimitiveBench bench_zero_eor_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::ZeroEorPrimitiveBench bench_zero_eor_2048_2048(RUN_TIME, 2048, 2048);
        print_bandwidth(bench_zero_eor_50_50, unary_bm, "ZeroEorPrimitiveBench 50x50");
        print_bandwidth(bench_zero_eor_64_64, unary_bm, "ZeroEorPrimitiveBench 64x64");
        print_bandwidth(bench_zero_eor_512_512, unary_bm, "ZeroEorPrimitiveBench 512x512");
        print_bandwidth(bench_zero_eor_2048_2048, unary_bm, "ZeroEorPrimitiveBench 2048x2048");

        // ZeroXzrPrimitiveBench benchmarks
        mini_jit::benchmarks::ZeroXZRPrimitiveBench bench_zero_xzr_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::ZeroXZRPrimitiveBench bench_zero_xzr_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::ZeroXZRPrimitiveBench bench_zero_xzr_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::ZeroXZRPrimitiveBench bench_zero_xzr_2048_2048(RUN_TIME, 2048, 2048);
        print_bandwidth(bench_zero_xzr_50_50, unary_bm, "ZeroXzrPrimitiveBench 50x50");
        print_bandwidth(bench_zero_xzr_64_64, unary_bm, "ZeroXzrPrimitiveBench 64x64");
        print_bandwidth(bench_zero_xzr_512_512, unary_bm, "ZeroXzrPrimitiveBench 512x512");
        print_bandwidth(bench_zero_xzr_2048_2048, unary_bm, "ZeroXzrPrimitiveBench 2048x2048");

        unary_bm.close();
    }

    if (has_tensor_operations)
    {
        const double RUN_TIME = 3.0;
        std::ofstream top_bm("benchmarks/tensor_operation_benchmarks.txt");

        std::vector<mini_jit::dim_t> l_dims_1 = {mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::k, mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::k};
        std::vector<mini_jit::exec_t> l_execs_1 = {mini_jit::exec_t::seq, mini_jit::exec_t::seq, mini_jit::exec_t::seq, mini_jit::exec_t::prim, mini_jit::exec_t::prim, mini_jit::exec_t::prim};
        std::vector<int64_t> l_sizes_1 = {32, 32, 8, 32, 32, 32};
        std::vector<int64_t> l_strides_in0_1 = {8192, 0, 1024, 1, 0, 32};
        std::vector<int64_t> l_strides_in1_1 = {0, 8192, 1024, 0, 32, 1};
        std::vector<int64_t> l_strides_out_1 = {32768, 1024, 0, 1, 32, 0};
        mini_jit::benchmarks::TensorOperationBench tensor_bench_1(RUN_TIME,
                                                                  mini_jit::dtype_t::fp32,
                                                                  mini_jit::ptype_t::none,
                                                                  mini_jit::ptype_t::gemm,
                                                                  mini_jit::ptype_t::none,
                                                                  l_dims_1,
                                                                  l_execs_1,
                                                                  l_sizes_1,
                                                                  l_strides_in0_1,
                                                                  l_strides_in1_1,
                                                                  l_strides_out_1);

        std::vector<mini_jit::dim_t> l_dims_2 = {mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::k, mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::k};
        std::vector<mini_jit::exec_t> l_execs_2 = {mini_jit::exec_t::seq, mini_jit::exec_t::seq, mini_jit::exec_t::prim, mini_jit::exec_t::prim, mini_jit::exec_t::prim, mini_jit::exec_t::prim};
        std::vector<int64_t> l_sizes_2 = {32, 32, 8, 32, 32, 32};
        std::vector<int64_t> l_strides_in0_2 = {8192, 0, 1024, 1, 0, 32};
        std::vector<int64_t> l_strides_in1_2 = {0, 8192, 1024, 0, 32, 1};
        std::vector<int64_t> l_strides_out_2 = {32768, 1024, 0, 1, 32, 0};
        mini_jit::benchmarks::TensorOperationBench tensor_bench_2(RUN_TIME,
                                                                  mini_jit::dtype_t::fp32,
                                                                  mini_jit::ptype_t::none,
                                                                  mini_jit::ptype_t::brgemm,
                                                                  mini_jit::ptype_t::none,
                                                                  l_dims_2,
                                                                  l_execs_2,
                                                                  l_sizes_2,
                                                                  l_strides_in0_2,
                                                                  l_strides_in1_2,
                                                                  l_strides_out_2);

        std::vector<mini_jit::dim_t> l_dims_3 = {mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::k, mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::k};
        std::vector<mini_jit::exec_t> l_execs_3 = {mini_jit::exec_t::seq, mini_jit::exec_t::seq, mini_jit::exec_t::prim, mini_jit::exec_t::prim, mini_jit::exec_t::prim, mini_jit::exec_t::prim};
        std::vector<int64_t> l_sizes_3 = {32, 32, 8, 32, 32, 32};
        std::vector<int64_t> l_strides_in0_3 = {8192, 0, 1024, 1, 0, 32};
        std::vector<int64_t> l_strides_in1_3 = {0, 8192, 1024, 0, 32, 1};
        std::vector<int64_t> l_strides_out_3 = {32768, 1024, 0, 1, 32, 0};
        mini_jit::benchmarks::TensorOperationBench tensor_bench_3(RUN_TIME,
                                                                  mini_jit::dtype_t::fp32,
                                                                  mini_jit::ptype_t::zero,
                                                                  mini_jit::ptype_t::brgemm,
                                                                  mini_jit::ptype_t::relu,
                                                                  l_dims_3,
                                                                  l_execs_3,
                                                                  l_sizes_3,
                                                                  l_strides_in0_3,
                                                                  l_strides_in1_3,
                                                                  l_strides_out_3);

        print_throughput(tensor_bench_1, top_bm, "TensorOperationBench GEMM");
        print_throughput(tensor_bench_2, top_bm, "TensorOperationBench BRGEMM");
        print_throughput(tensor_bench_3, top_bm, "TensorOperationBench ZERO + BRGEMM + ReLU");
    }

    if (has_shared_tensor_operations)
    {
        const double RUN_TIME = 3.0;
        std::ofstream top_bm("benchmarks/shared_tensor_operation_benchmarks.txt");

        std::vector<mini_jit::dim_t> l_dims_1 = {mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::k, mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::k};
        std::vector<mini_jit::exec_t> l_execs_1 = {mini_jit::exec_t::shared, mini_jit::exec_t::seq, mini_jit::exec_t::seq, mini_jit::exec_t::prim, mini_jit::exec_t::prim, mini_jit::exec_t::prim};
        std::vector<int64_t> l_sizes_1 = {32, 32, 8, 32, 32, 32};
        std::vector<int64_t> l_strides_in0_1 = {8192, 0, 1024, 1, 0, 32};
        std::vector<int64_t> l_strides_in1_1 = {0, 8192, 1024, 0, 32, 1};
        std::vector<int64_t> l_strides_out_1 = {32768, 1024, 0, 1, 32, 0};
        mini_jit::benchmarks::TensorOperationBench tensor_bench_1(RUN_TIME,
                                                                  mini_jit::dtype_t::fp32,
                                                                  mini_jit::ptype_t::none,
                                                                  mini_jit::ptype_t::gemm,
                                                                  mini_jit::ptype_t::none,
                                                                  l_dims_1,
                                                                  l_execs_1,
                                                                  l_sizes_1,
                                                                  l_strides_in0_1,
                                                                  l_strides_in1_1,
                                                                  l_strides_out_1);

        std::vector<mini_jit::dim_t> l_dims_2 = {mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::k, mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::k};
        std::vector<mini_jit::exec_t> l_execs_2 = {mini_jit::exec_t::shared, mini_jit::exec_t::seq, mini_jit::exec_t::prim, mini_jit::exec_t::prim, mini_jit::exec_t::prim, mini_jit::exec_t::prim};
        std::vector<int64_t> l_sizes_2 = {32, 32, 8, 32, 32, 32};
        std::vector<int64_t> l_strides_in0_2 = {8192, 0, 1024, 1, 0, 32};
        std::vector<int64_t> l_strides_in1_2 = {0, 8192, 1024, 0, 32, 1};
        std::vector<int64_t> l_strides_out_2 = {32768, 1024, 0, 1, 32, 0};
        mini_jit::benchmarks::TensorOperationBench tensor_bench_2(RUN_TIME,
                                                                  mini_jit::dtype_t::fp32,
                                                                  mini_jit::ptype_t::none,
                                                                  mini_jit::ptype_t::brgemm,
                                                                  mini_jit::ptype_t::none,
                                                                  l_dims_2,
                                                                  l_execs_2,
                                                                  l_sizes_2,
                                                                  l_strides_in0_2,
                                                                  l_strides_in1_2,
                                                                  l_strides_out_2);

        std::vector<mini_jit::dim_t> l_dims_3 = {mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::k, mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::k};
        std::vector<mini_jit::exec_t> l_execs_3 = {mini_jit::exec_t::shared, mini_jit::exec_t::seq, mini_jit::exec_t::prim, mini_jit::exec_t::prim, mini_jit::exec_t::prim, mini_jit::exec_t::prim};
        std::vector<int64_t> l_sizes_3 = {32, 32, 8, 32, 32, 32};
        std::vector<int64_t> l_strides_in0_3 = {8192, 0, 1024, 1, 0, 32};
        std::vector<int64_t> l_strides_in1_3 = {0, 8192, 1024, 0, 32, 1};
        std::vector<int64_t> l_strides_out_3 = {32768, 1024, 0, 1, 32, 0};
        mini_jit::benchmarks::TensorOperationBench tensor_bench_3(RUN_TIME,
                                                                  mini_jit::dtype_t::fp32,
                                                                  mini_jit::ptype_t::zero,
                                                                  mini_jit::ptype_t::brgemm,
                                                                  mini_jit::ptype_t::relu,
                                                                  l_dims_3,
                                                                  l_execs_3,
                                                                  l_sizes_3,
                                                                  l_strides_in0_3,
                                                                  l_strides_in1_3,
                                                                  l_strides_out_3);

        print_throughput(tensor_bench_1, top_bm, "SharedTensorOperationBench GEMM");
        print_throughput(tensor_bench_2, top_bm, "SharedTensorOperationBench BRGEMM");
        print_throughput(tensor_bench_3, top_bm, "SharedTensorOperationBench ZERO + BRGEMM + ReLU");
    }

    if (has_opt_tensor_operations)
    {
        const double RUN_TIME = 3.0;
        std::ofstream top_bm("benchmarks/optimized_tensor_and_einsum_operation_benchmarks.txt");

        std::vector<mini_jit::dim_t> l_dims_1 = {mini_jit::dim_t::m, mini_jit::dim_t::m, mini_jit::dim_t::n, mini_jit::dim_t::n, mini_jit::dim_t::k, mini_jit::dim_t::k};
        std::vector<mini_jit::exec_t> l_execs_1 = {mini_jit::exec_t::seq, mini_jit::exec_t::seq, mini_jit::exec_t::seq, mini_jit::exec_t::seq, mini_jit::exec_t::seq, mini_jit::exec_t::seq};
        std::vector<int64_t> l_sizes_1 = {64, 25, 64, 25, 64, 25};
        std::vector<int64_t> l_strides_in0_1 = {25, 1, 0, 0, 40000, 1600};
        std::vector<int64_t> l_strides_in1_1 = {0, 0, 40000, 1600, 25, 1};
        std::vector<int64_t> l_strides_out_1 = {25, 1, 40000, 1600, 0, 0};

        // min_kernel_size was implemented after the benchmarks were run, so we set it to 1
        mini_jit::ir::Optimizer::optimize(l_dims_1,
                                          l_execs_1,
                                          l_sizes_1,
                                          l_strides_in0_1,
                                          l_strides_in1_1,
                                          l_strides_out_1,
                                          INT_MAX,
                                          512,
                                          1);

        int l_prim_count_1 = 0;
        for (const auto &exec : l_execs_1)
        {
            if (exec == mini_jit::exec_t::prim)
            {
                l_prim_count_1++;
            }
        }

        mini_jit::benchmarks::TensorOperationBench tensor_bench_1(RUN_TIME,
                                                                  mini_jit::dtype_t::fp32,
                                                                  mini_jit::ptype_t::none,
                                                                  l_prim_count_1 == 4 ? mini_jit::ptype_t::brgemm : mini_jit::ptype_t::gemm,
                                                                  mini_jit::ptype_t::none,
                                                                  l_dims_1,
                                                                  l_execs_1,
                                                                  l_sizes_1,
                                                                  l_strides_in0_1,
                                                                  l_strides_in1_1,
                                                                  l_strides_out_1);

        print_throughput(tensor_bench_1, top_bm, "Optimized TensorOperationBench GEMM/BRGEMM");

        top_bm << "#####################################################" << std::endl;
        top_bm << "Testing different kernel sizes" << std::endl;
        top_bm << "#####################################################" << std::endl;
        for (int64_t max_kernel_size : {1024, 512, 256, 125, 64, 32, 16})
        {
            optimized_tensor_benchmark(top_bm, 64, max_kernel_size, 1);
            optimized_tensor_benchmark(top_bm, 256, max_kernel_size, 1);
            optimized_tensor_benchmark_einsum(top_bm, 64, max_kernel_size, 1);
            optimized_tensor_benchmark_einsum(top_bm, 256, max_kernel_size, 1);
        }
    }

    if (has_einsum_benchmark)
    {
        std::ofstream einsum_bm("benchmarks/einsum_benchmark.txt");
        einsum_benchmark_1(einsum_bm, 3.0, 256, 64, 1);
        einsum_benchmark_2(einsum_bm, 3.0, 256, 64, 1);
    }

    if (has_opt_einsum_benchmark)
    {
        std::ofstream einsum_bm("benchmarks/opt_einsum_benchmark.txt");
        einsum_benchmark_optimization_example_1(einsum_bm, 3.0, 256, 64, 1);
        einsum_benchmark_optimization_example_2(einsum_bm, 3.0, 256, 64, 1);
        einsum_benchmark_optimization_example_3(einsum_bm, 3.0, 256, 64, 1);
    }

    if (has_reciprocal)
    {
        const double RUN_TIME = 3.0;
        std::ofstream reciprocal_bm("benchmarks/reciprocal_benchmark.txt");

        mini_jit::benchmarks::ReciprocalPrimitiveBench bench_reciprocal_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::ReciprocalPrimitiveBench bench_reciprocal_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::ReciprocalPrimitiveBench bench_reciprocal_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::ReciprocalPrimitiveBench bench_reciprocal_2048_2048(RUN_TIME, 2048, 2048);
        print_bandwidth(bench_reciprocal_50_50, reciprocal_bm, "ReciprocalPrimitiveBench 50x50");
        print_bandwidth(bench_reciprocal_64_64, reciprocal_bm, "ReciprocalPrimitiveBench 64x64");
        print_bandwidth(bench_reciprocal_512_512, reciprocal_bm, "ReciprocalPrimitiveBench 512x512");
        print_bandwidth(bench_reciprocal_2048_2048, reciprocal_bm, "ReciprocalPrimitiveBench 2048x2048");

        reciprocal_bm.close();
    }

    if (has_sigmoid)
    {
        const double RUN_TIME = 3.0;
        std::ofstream sigmoid_bm("benchmarks/sigmoid_benchmark.txt");

        mini_jit::benchmarks::FastSigmoidPrimitiveBench bench_sigmoid_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::FastSigmoidPrimitiveBench bench_sigmoid_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::FastSigmoidPrimitiveBench bench_sigmoid_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::FastSigmoidPrimitiveBench bench_sigmoid_2048_2048(RUN_TIME, 2048, 2048);
        print_bandwidth(bench_sigmoid_50_50, sigmoid_bm, "FastSigmoidPrimitiveBench 50x50");
        print_bandwidth(bench_sigmoid_64_64, sigmoid_bm, "FastSigmoidPrimitiveBench 64x64");
        print_bandwidth(bench_sigmoid_512_512, sigmoid_bm, "FastSigmoidPrimitiveBench 512x512");
        print_bandwidth(bench_sigmoid_2048_2048, sigmoid_bm, "FastSigmoidPrimitiveBench 2048x2048");

        mini_jit::benchmarks::SigmoidInterpolationPrimitiveBench bench_sigmoid_interpolation_50_50(RUN_TIME, 50, 50);
        mini_jit::benchmarks::SigmoidInterpolationPrimitiveBench bench_sigmoid_interpolation_64_64(RUN_TIME, 64, 64);
        mini_jit::benchmarks::SigmoidInterpolationPrimitiveBench bench_sigmoid_interpolation_512_512(RUN_TIME, 512, 512);
        mini_jit::benchmarks::SigmoidInterpolationPrimitiveBench bench_sigmoid_interpolation_2048_2048(RUN_TIME, 2048, 2048);
        print_bandwidth(bench_sigmoid_interpolation_50_50, sigmoid_bm, "SigmoidInterpolationPrimitiveBench 50x50");
        print_bandwidth(bench_sigmoid_interpolation_64_64, sigmoid_bm, "SigmoidInterpolationPrimitiveBench 64x64");
        print_bandwidth(bench_sigmoid_interpolation_512_512, sigmoid_bm, "SigmoidInterpolationPrimitiveBench 512x512");
        print_bandwidth(bench_sigmoid_interpolation_2048_2048, sigmoid_bm, "SigmoidInterpolationPrimitiveBench 2048x2048");

        sigmoid_bm.close();
    }

    return 0;
}
