#include <arm_neon.h>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <cstring>

extern "C" {
    void v1_matmul_64_64_64( float const *a,
                             float const *b,
                             float *c,
                             int64_t lda,
                             int64_t ldb,
                             int64_t ldc );

    void v2_matmul_64_64_64( float const *a,
                             float const *b,
                             float *c,
                             int64_t lda,
                             int64_t ldb,
                             int64_t ldc );
}

/*
 * Benchmarks the throughput for the different MatMul versions.
 *
 * @param n: number of loop iterations.
 * @param instruction: a string selecting the instruction to benchmark.
 * @param a pointer to column-major matrix A.
 * @param b pointer to column-major matrix B.
 * @param c pointer to column-major matrix C.
 */
void benchmark_thr( int64_t loop_iters, 
                    std::string instruction,
                    float const *a,
                    float const *b,
                    float *c ) 
{
    std::cout << "-----------------------------------------------\n";

    std::string v1_matmul( "v1_matmul" );
    int res_1 = v1_matmul.compare( instruction );

    std::string v2_matmul( "v2_matmul" );
    int res_2 = v2_matmul.compare( instruction );

    double elapsedTime = 1;

    // Time measuring
    if ( res_1 == 0 )
    {
        // Warmup
        for ( int j = 0; j < 100; j++ )
        {
            v1_matmul_64_64_64( a, 
                                b, 
                                c, 
                                64, 
                                64, 
                                64 );
        }
        std::memset( c, 0, 64 * 64 * sizeof( float ) );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int j = 0; j < loop_iters; j++ )
        {
            v1_matmul_64_64_64( a, 
                                b, 
                                c,
                                64, 
                                64, 
                                64 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;
    }
    else if ( res_2 == 0 )
    {
        // Warmup
        for ( int j = 0; j < 100; j++ )
        {
            v2_matmul_64_64_64( a, 
                                b, 
                                c, 
                                64, 
                                64, 
                                64 );
        }
        std::memset( c, 0, 64 * 64 * sizeof( float ) );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int j = 0; j < loop_iters; j++ )
        {
            v2_matmul_64_64_64( a, 
                                b, 
                                c, 
                                64, 
                                64, 
                                64 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;
    }

    // (M x N x K) x (FMLA OPs)
    double totalOps = ( 64 * 64 * 64 ) * 2;
    double opsPerIteration = totalOps * loop_iters;

    double opsPerSec = opsPerIteration / elapsedTime;
    double gflops = opsPerIteration / ( elapsedTime * 1e9 );

    std::cout << "Measuring throughput for " << "Instruction\n";
    std::cout << "Total time (s):   " << elapsedTime << "\n";
    std::cout << "Instructions per Second:   " << opsPerSec << "\n";
    std::cout << "Estimated GFLOPS:   " << gflops << " GFLOPS/sec\n";
    std::cout << "-----------------------------------------------\n";
}

int main() 
{
    const int M = 64;
    const int N = 64;
    const int K = 64;

    float A[M * K];
    float B[K * N];
    float C[M * N];

    // Initialize matrices
    for ( int i = 0; i < M * K; ++i )
    {
        A[i] = static_cast<float>( i );
    }
    for ( int j = 0; j < K * N; ++j )
    {
        B[j] = static_cast<float>( j );
    }
    for ( int k = 0; k < M * N; ++k )
    {
        C[k] = static_cast<float>( k );
    }

    int64_t l_iter = 300000;
    std::string v1_matmul( "v1_matmul" );
    std::string v2_matmul( "v2_matmul" );
    
    std::cout << "\nBenchmarking V1 Matmul throughput ...\n";
    benchmark_thr( l_iter, v1_matmul, A, B, C );
    std::memset( C, 0, M * N * sizeof( float ) );

    std::cout << "\nBenchmarking V2 Matmul throughput ...\n";
    benchmark_thr( l_iter, v2_matmul, A, B, C );

    return 0;
}
