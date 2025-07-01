#include <random>
#include <chrono>
#include "square_trans_primitive.bench.h"
#include "benchmarks/Benchmark.h"
#include "kernels/unary/square_trans_primitive.h"
#include "Kernel.h"
#include "Unary.h"

mini_jit::benchmarks::SquareTransPrimitiveBench::SquareTransPrimitiveBench(double runTime,
                                                                           uint32_t m,
                                                                           uint32_t n) : Benchmark()
{
    m_M = m;
    m_N = n;
    m_runTime = runTime;
}

void mini_jit::benchmarks::SquareTransPrimitiveBench::run()
{
    m_A = new float[m_M * m_N];
    m_B = new float[m_M * m_N];

    // Initialize matrices A and B with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (uint32_t i = 0; i < m_M * m_N; i++)
    {
        m_A[i] = dist(gen);
        m_B[i] = dist(gen);
    }

    // Generate and get the kernel function
    mini_jit::Kernel l_kernel;
    mini_jit::kernels::unary::square_trans(l_kernel, m_M, m_N);
    mini_jit::Unary::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Unary::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));

    // RUN
    long l_num_reps = 0;
    auto l_start_time = std::chrono::high_resolution_clock::now();
    double l_elapsed = 0.0;
    double l_runTimeMs = m_runTime * 1e6;
    do
    {
        l_kernel_t(m_A, m_B, m_M, m_N);
        ++l_num_reps;
        auto l_now = std::chrono::high_resolution_clock::now();
        l_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(l_now - l_start_time).count();
    } while (l_elapsed < l_runTimeMs);
    l_elapsed /= 1e6; // Convert to seconds
    // END RUN

    // Calculate metrics
    long l_totalOperations = l_num_reps * (m_M * m_N);
    double l_gflops = ((double)l_totalOperations) / (l_elapsed * 1e9);

    // Store the results
    m_benchmarkResult.numReps = l_num_reps;
    m_benchmarkResult.elapsedSeconds = l_elapsed;
    m_benchmarkResult.totalOperations = l_totalOperations;
    m_benchmarkResult.gflops = l_gflops;

    delete[] m_A;
    delete[] m_B;
}
