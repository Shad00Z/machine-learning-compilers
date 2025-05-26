#include <random>
#include <chrono>
#include "TensorOperation.bench.h"
#include "benchmarks/Benchmark.h"

mini_jit::benchmarks::TensorOperationBench::TensorOperationBench(double run_time,
                                                                 dtype_t dtype,
                                                                 ptype_t prim_first_touch,
                                                                 ptype_t prim_main,
                                                                 ptype_t prim_last_touch,
                                                                 std::span<const dim_t> dim_types,
                                                                 std::span<const exec_t> exec_types,
                                                                 std::span<const int64_t> dim_sizes,
                                                                 std::span<const int64_t> strides_in0,
                                                                 std::span<const int64_t> strides_in1,
                                                                 std::span<const int64_t> strides_out) : Benchmark()
{
    m_run_time = run_time;
    m_tensor_op.setup(dtype,
                      prim_first_touch,
                      prim_main,
                      prim_last_touch,
                      dim_types,
                      exec_types,
                      dim_sizes,
                      strides_in0,
                      strides_in1,
                      strides_out);

    m_dim_sizes.assign(dim_sizes.begin(), dim_sizes.end());
}

void mini_jit::benchmarks::TensorOperationBench::run()
{
    const int R = m_dim_sizes[0];
    const int P = m_dim_sizes[1];
    const int T = m_dim_sizes[2];
    const int S = m_dim_sizes[3];
    const int Q = m_dim_sizes[4];
    const int U = m_dim_sizes[5];

    const int SIZE_A = (R * S) * (T * U);
    const int SIZE_B = (T * U) * (P * Q);
    const int SIZE_C = (R * S) * (P * Q);

    float *A = new float[SIZE_A];
    float *B = new float[SIZE_B];
    float *C = new float[SIZE_C];

    // init with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < SIZE_A; ++i)
    {
        A[i] = dist(gen);
    }

    for (int i = 0; i < SIZE_B; ++i)
    {
        B[i] = dist(gen);
    }

    for (int i = 0; i < SIZE_C; ++i)
    {
        C[i] = dist(gen);
    }

    // RUN
    long l_num_reps = 0;
    auto l_start_time = std::chrono::high_resolution_clock::now();
    double l_elapsed = 0.0;
    double l_run_time_ms = m_run_time * 1e6;
    do
    {
        m_tensor_op.execute(A, B, C);
        ++l_num_reps;
        auto l_now = std::chrono::high_resolution_clock::now();
        l_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(l_now - l_start_time).count();
    } while (l_elapsed < l_run_time_ms);
    l_elapsed /= 1e6; // Convert to seconds
    // END RUN

    // Calculate metrics
    long l_totalOperations = 2.0 * l_num_reps * (R * S) * (T * U) * (P * Q);
    double l_gflops = ((double)l_totalOperations) / (l_elapsed * 1e9);

    // Store the results
    m_benchmarkResult.numReps = l_num_reps;
    m_benchmarkResult.elapsedSeconds = l_elapsed;
    m_benchmarkResult.totalNumberElements = (R * S) * (T * U) * (P * Q) * l_num_reps;
    m_benchmarkResult.totalOperations = l_totalOperations;
    m_benchmarkResult.gflops = l_gflops;

    delete[] A;
    delete[] B;
    delete[] C;
}
