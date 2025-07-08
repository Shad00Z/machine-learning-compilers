#ifndef RELU_TRANS_PRIMITIVE_BENCH_H
#define RELU_TRANS_PRIMITIVE_BENCH_H
#include <cstdint>
#include <mlc/benchmarks/Benchmark.h>

namespace mini_jit
{
    namespace benchmarks
    {
        class ReLUTransPrimitiveBench : public Benchmark
        {
        public:
            /**
             * @brief Constructor for the benchmark for the transposition ReLU primitive.
             * @param runTime The time to run the benchmark in seconds.
             * @param m number of rows in A and B.
             * @param n number of columns in A and B.
             */
            ReLUTransPrimitiveBench(double   runTime,
                                    uint32_t m,
                                    uint32_t n);
            //! Destructor
            ~ReLUTransPrimitiveBench() override = default;
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

#endif // RELU_TRANS_PRIMITIVE_BENCH_H