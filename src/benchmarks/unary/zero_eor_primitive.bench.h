#ifndef ZERO_EOR_PRIMITIVE_BENCH_H
#define ZERO_EOR_PRIMITIVE_BENCH_H
#include "benchmarks/Benchmark.h"
#include <cstdint>

namespace mini_jit
{
    namespace benchmarks
    {
        class Zero_eor_primitive_bench : public Benchmark
        {
        public:
            /**
             * @brief Constructor for the benchmark for the EOR Zero primitive.
             * @param runTime The time to run the benchmark in seconds.
             * @param m number of rows in A and B.
             * @param n number of columns in A and B.
             */
            Zero_eor_primitive_bench(double runTime,
                                     uint32_t m,
                                     uint32_t n);
            //! Destructor
            ~Zero_eor_primitive_bench() override = default;
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

#endif // ZERO_EOR_PRIMITIVE_BENCH_H