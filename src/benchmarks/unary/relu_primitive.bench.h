#ifndef RELU_PRIMITIVE_BENCH_H
#define RELU_PRIMITIVE_BENCH_H
#include "benchmarks/Benchmark.h"
#include <cstdint>

namespace mini_jit
{
    namespace benchmarks
    {
        class Relu_primitive_bench : public Benchmark
        {
        public:
            /**
             * @brief Constructor for the benchmark for the ReLu primitive.
             * @param runTime The time to run the benchmark in seconds.
             * @param m number of rows in A and B.
             * @param n number of columns in A and B.
             */
            Relu_primitive_bench(double runTime,
                                 uint32_t m,
                                 uint32_t n);
            //! Destructor
            ~Relu_primitive_bench() override = default;
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

#endif // RELU_PRIMITIVE_BENCH_H