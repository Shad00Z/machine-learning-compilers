#ifndef ZERO_XZR_PRIMITIVE_BENCH_H
#define ZERO_XZR_PRIMITIVE_BENCH_H
#include "benchmarks/Benchmark.h"
#include <cstdint>

namespace mini_jit
{
    namespace benchmarks
    {
        class ZeroXZRPrimitiveBench : public Benchmark
        {
        public:
            /**
             * @brief Constructor for the benchmark for the XZR Zero primitive.
             * @param runTime The time to run the benchmark in seconds.
             * @param m number of rows in A and B.
             * @param n number of columns in A and B.
             */
            ZeroXZRPrimitiveBench(double runTime,
                                  uint32_t m,
                                  uint32_t n);
            //! Destructor
            ~ZeroXZRPrimitiveBench() override = default;
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

#endif // ZERO_XZR_PRIMITIVE_BENCH_H