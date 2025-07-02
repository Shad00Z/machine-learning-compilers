#include <random>
#include <chrono>
#include "sigmoid_interpolation_primitive.bench.h"
#include "benchmarks/Benchmark.h"
#include "kernels/unary/sigmoid_interp_primitive.h"
#include "Kernel.h"
#include "Unary.h"

mini_jit::benchmarks::SigmoidInterpolationPrimitiveBench::SigmoidInterpolationPrimitiveBench(double runTime,
                                                                                             uint32_t m,
                                                                                             uint32_t n) : Benchmark()
{
    m_M = m;
    m_N = n;
    m_runTime = runTime;
}

void mini_jit::benchmarks::SigmoidInterpolationPrimitiveBench::run()
{
    m_A = new float[m_M * m_N];
    m_B = new float[m_M * m_N];

    float sig_table[33] = {
        0.000335f, 0.000553f, 0.000911f, 0.001503f, 0.002473f, 0.004070f, 0.006693f,
        0.011109f, 0.017986f, 0.029312f, 0.047426f, 0.075858f, 0.119203f, 0.182426f,
        0.268941f, 0.377541f, 0.500000f, 0.622459f, 0.731059f, 0.817574f, 0.880797f,
        0.924142f, 0.952574f, 0.970688f, 0.982014f, 0.988891f, 0.993307f, 0.995930f,
        0.997527f, 0.998497f, 0.999089f, 0.999447f, 0.999665f
    };

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
    mini_jit::kernels::unary::sigmoid_interpolation(l_kernel, m_M, m_N);
    mini_jit::Unary::kernel_t_sig l_kernel_t = reinterpret_cast<mini_jit::Unary::kernel_t_sig>(const_cast<void *>(l_kernel.get_kernel()));

    // RUN
    long l_num_reps = 0;
    auto l_start_time = std::chrono::high_resolution_clock::now();
    double l_elapsed = 0.0;
    double l_runTimeMs = m_runTime * 1e6;
    do
    {
        l_kernel_t(m_A, m_B, sig_table, m_M, m_M);
        ++l_num_reps;
        auto l_now = std::chrono::high_resolution_clock::now();
        l_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(l_now - l_start_time).count();
    } while (l_elapsed < l_runTimeMs);
    l_elapsed /= 1e6; // Convert to seconds
    // END RUN

    // Calculate metrics
    long l_totalNumberElements = m_M * m_N * l_num_reps * 2;
    double l_totalDataProcessed = (sizeof(float) * l_totalNumberElements) / (1024.0 * 1024.0 * 1024.0);
    double l_gibps = l_totalDataProcessed / l_elapsed;

    // Store the results
    m_benchmarkResult.numReps = l_num_reps;
    m_benchmarkResult.elapsedSeconds = l_elapsed;
    m_benchmarkResult.totalNumberElements = m_M * m_N * l_num_reps;
    m_benchmarkResult.totalDataProcessed = l_totalDataProcessed;
    m_benchmarkResult.gibps = l_gibps;

    delete[] m_A;
    delete[] m_B;
}
