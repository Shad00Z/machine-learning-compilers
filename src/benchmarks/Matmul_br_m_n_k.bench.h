#ifndef MATMUL_BR_M_N_K_BENCH_H
#define MATMUL_BR_M_N_K_BENCH_H
#include "Benchmark.h"

namespace mini_jit
{
    namespace benchmarks
    {
        /**
         * @brief Benchmark for matrix multiplication using BRGEMM.
         */
        class Matmul_br_m_n_k_bench : public Benchmark
        {
        public:
            /**
             * @brief Constructor for the benchmark for matrix multiplication using BRGEMM.
             * @param runTime The time to run the benchmark in seconds.
             * @param m number of rows in A and C.
             * @param n number of columns in B and C.
             * @param k number of columns in A and rows in B.
             * @param brSize The size of the batch-reduce.
             */
            Matmul_br_m_n_k_bench(double runTime,
                                  int m,
                                  int n,
                                  int k,
                                  int brSize);
            //! Destructor
            ~Matmul_br_m_n_k_bench() override = default;
            //! Runs the benchmark.
            void run() override;

        private:
            int m_M;
            int m_N;
            int m_K;
            int m_brSize;
            double m_runTime;
            float *m_A;
            float *m_B;
            float *m_C;
        };

    }
}

#endif // MATMUL_BR_M_N_K_BENCH_H