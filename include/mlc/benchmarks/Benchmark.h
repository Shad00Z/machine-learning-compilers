#ifndef MINI_JIT_BENCHMARK_H
#define MINI_JIT_BENCHMARK_H

namespace mini_jit
{
    class Benchmark;
}

class mini_jit::Benchmark
{
public:
    /*
     * This structure holds the result of a benchmark run.
     * @param numReps Number of repetitions of the benchmark.
     * @param elapsedSeconds Elapsed time in seconds.
     * @param totalNumberElements Total number of elements processed.
     * @param totalOperations Total number of operations performed.
     * @param gflops Giga FP operations per second.
     * @param totalDataProcessed Total data processed in GiB.
     * @param gibps Bandwidth in GiB/s.
     */
    struct benchmark_result
    {
        // all
        long   numReps             = 0;
        double elapsedSeconds      = 0.0f;
        long   totalNumberElements = 0;

        long   totalOperations = 0;
        double gflops          = 0.0f;

        double totalDataProcessed = 0.0f;
        double gibps              = 0.0f;
    };

    virtual ~Benchmark() {}
    //! Runs the benchmark.
    virtual void run() = 0;
    //! Returns the result of the benchmark.
    benchmark_result getResult()
    {
        return m_benchmarkResult;
    }

protected:
    benchmark_result m_benchmarkResult;
};

#endif // MINI_JIT_BENCHMARK_H