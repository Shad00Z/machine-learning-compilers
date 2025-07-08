#ifndef ZERO_EOR_PRIMITIVE_BENCH_H
#define ZERO_EOR_PRIMITIVE_BENCH_H
#include <cstdint>
#include <mlc/benchmarks/Benchmark.h>

namespace mini_jit
{
    namespace benchmarks
    {
        class ZeroEorPrimitiveBench : public Benchmark
        {
        public:
            /**
             * @brief Constructor for the benchmark for the EOR Zero primitive.
             * @param runTime The time to run the benchmark in seconds.
             * @param m number of rows in A and B.
             * @param n number of columns in A and B.
             */
            ZeroEorPrimitiveBench(double   runTime,
                                  uint32_t m,
                                  uint32_t n);
            //! Destructor
            ~ZeroEorPrimitiveBench() override = default;
            //! Runs the benchmark.
            void run() override;

        private:
            uint32_t m_M;
            uint32_t m_N;
            double   m_runTime;
            float*   m_A;
            float*   m_B;
        };

    } // namespace benchmarks
} // namespace mini_jit

#endif // ZERO_EOR_PRIMITIVE_BENCH_H