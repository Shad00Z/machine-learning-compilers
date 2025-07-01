#ifndef FAST_SIGMOID_PRIMITIVE_BENCH_H
#define FAST_SIGMOID_PRIMITIVE_BENCH_H
#include "benchmarks/Benchmark.h"
#include <cstdint>

namespace mini_jit
{
    namespace benchmarks
    {
        class FastSigmoidPrimitiveBench : public Benchmark
        {
        public:
            /**
             * @brief Constructor for the benchmark for the fast sigmoid primitive.
             * @param runTime The time to run the benchmark in seconds.
             * @param m number of rows in A and B.
             * @param n number of columns in A and B.
             */
            FastSigmoidPrimitiveBench(double runTime,
                                      uint32_t m,
                                      uint32_t n);
            //! Destructor
            ~FastSigmoidPrimitiveBench() override = default;
            //! Runs the benchmark.
            void run() override;

        private:
            uint32_t m_M;
            uint32_t m_N;
            double m_runTime;
            float *m_A;
            float *m_B;
        };

    }
}

#endif // FAST_SIGMOID_PRIMITIVE_BENCH_H