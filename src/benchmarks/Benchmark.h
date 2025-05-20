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
    * @param totalOperations Total number of operations performed.
    * @param gops Giga operations per second.
    */
    struct benchmark_result
    {
        long numReps = 0;
        double elapsedSeconds = 0.0f;
        long totalOperations = 0;
        double gops = 0.0f;
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