#ifndef MATMUL_M_N_K_BENCH_H
#define MATMUL_M_N_K_BENCH_H
#include "Benchmark.h"

namespace mini_jit
{
    namespace benchmarks
    {
        class Matmul_m_n_k_bench : public Benchmark
        {
        public:
            Matmul_m_n_k_bench(double runTime, 
                               int m, 
                               int n, 
                               int k);
            ~Matmul_m_n_k_bench() override = default;
            void run() override;

        private:
            int m_M;
            int m_N;
            int m_K;
            double m_runTime;
            float *m_A;
            float *m_B;
            float *m_C;
        };

    }
}

#endif // MATMUL_M_N_K_BENCH_H