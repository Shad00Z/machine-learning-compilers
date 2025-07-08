#ifndef RELU_PRIMITIVE_BENCH_H
#define RELU_PRIMITIVE_BENCH_H
#include <cstdint>
#include <mlc/benchmarks/Benchmark.h>

namespace mini_jit
{
    namespace benchmarks
    {
        class ReLUPrimitiveBench : public Benchmark
        {
        public:
            /**
             * @brief Constructor for the benchmark for the ReLU primitive.
             * @param runTime The time to run the benchmark in seconds.
             * @param m number of rows in A and B.
             * @param n number of columns in A and B.
             */
            ReLUPrimitiveBench(double   runTime,
                               uint32_t m,
                               uint32_t n);
            //! Destructor
            ~ReLUPrimitiveBench() override = default;
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

#endif // RELU_PRIMITIVE_BENCH_H