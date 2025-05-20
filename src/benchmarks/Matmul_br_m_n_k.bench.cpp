#include <random>
#include "Benchmark.h"
#include "Matmul_br_m_n_k.bench.h"
#include "Kernel.h"
#include "Brgemm.h"
#include "kernels/matmul/matmul_br_m_n_k.h"

mini_jit::benchmarks::Matmul_br_m_n_k_bench::Matmul_br_m_n_k_bench(double runTime,
                                                                  int m,
                                                                  int n,
                                                                  int k,
                                                                  int brSize) : Benchmark()
{
    m_M = m;
    m_N = n;
    m_K = k;
    m_brSize = brSize;
    m_runTime = runTime;
}

void mini_jit::benchmarks::Matmul_br_m_n_k_bench::run()
{
    m_A = new float[m_M * m_K * m_brSize];
    m_B = new float[m_K * m_N * m_brSize];
    m_C = new float[m_M * m_N];

    // Initialize matrices A and B with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < m_M * m_K * m_brSize; i++)
    {
        m_A[i] = dist(gen);
    }
    for (int i = 0; i < m_K * m_N * m_brSize; i++)
    {
        m_B[i] = dist(gen);
    }
    // Initialize matrix C with zeros
    for (int i = 0; i < m_M * m_N; ++i)
    {
        m_C[i] = 0.0f;
    }

    // Generate and get the kernel function
    mini_jit::Kernel l_kernel;
    mini_jit::kernels::matmul::matmul_br_m_n_k(l_kernel, m_M, m_N, m_K, m_brSize);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));

    // RUN
    int l_num_reps = 0;
    auto l_start_time = std::chrono::high_resolution_clock::now();
    double l_elapsed = 0.0;
    do
    {
        l_kernel_t(m_A, m_B, m_C, m_M, m_K, m_M, m_M * m_K, m_K * m_N);
        ++l_num_reps;
        auto l_now = std::chrono::high_resolution_clock::now();
        l_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(l_now - l_start_time).count() / 1e6;
    } while (l_elapsed < m_runTime);
    // END RUN

    // Calculate metrics
    int l_totalOperations = 2.0 * m_M * m_N * m_K * l_num_reps * m_brSize;
    double l_gflops = ((double)l_totalOperations) / (l_elapsed * 1e9);

    // Store the results
    m_benchmarkResult = {
        l_num_reps,
        l_elapsed,
        l_totalOperations,
        l_gflops};

    delete[] m_A;
    delete[] m_B;
    delete[] m_C;
}
