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
    * @param flopsPerSec Floating point operations per second.
    * @param gflops Giga floating point operations per second.    
    */
    struct benchmark_result
    {
        int numReps = 0;
        double elapsedSeconds = 0.0f;
        int totalOperations = 0;
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