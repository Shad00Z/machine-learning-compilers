#ifndef SQUARE_PRIMITIVE_BENCH_H
#define SQUARE_PRIMITIVE_BENCH_H
#include "benchmarks/Benchmark.h"
#include <cstdint>

namespace mini_jit
{
    namespace benchmarks
    {
        class SquarePrimitiveBench : public Benchmark
        {
        public:
            /**
             * @brief Constructor for the benchmark for square primitive.
             * @param runTime The time to run the benchmark in seconds.
             * @param m number of rows in A and B.
             * @param n number of columns in A and B.
             */
            SquarePrimitiveBench(double runTime,
                                     uint32_t m,
                                     uint32_t n);
            //! Destructor
            ~SquarePrimitiveBench() override = default;
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

#endif // SQUARE_PRIMITIVE_BENCH_H