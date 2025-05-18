#include <arm_neon.h>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <cstring>

extern "C" {
    void trans_neon_8_8( float const *a,
                         float const *b,
                         int64_t lda,
                         int64_t ldb );

    void trans_neon_4_4( float const *a,
                         float const *b,
                         int64_t lda,
                         int64_t ldb );
}

/*
 * Benchmarks the throughput for the different MatMul (loop) versions.
 *
 * @param n: number of loop iterations.
 * @param instruction: a string selecting the instruction to benchmark.
 * @param a pointer to column-major matrix A.
 * @param b pointer to column-major matrix B.
 * @param c pointer to column-major matrix C.
 */
 void benchmark_thr( int64_t iter, 
                     std::string instruction,
                     float const *a,
                     float const *b,
                     int m,
                     int n ) 
{
    std::cout << "-----------------------------------------------\n";
    double elapsedTime = 1;

    std::string trans_8_8( "neon_8_8" );
    int res_1 = trans_8_8.compare( instruction );

    std::string trans_4_4( "neon_4_4" );
    int res_2 = trans_4_4.compare( instruction );

    double opsPerMatmul = 1;
    double total_bytes = 0;

    // Time measuring
    if ( res_1 == 0 )
    {
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            trans_neon_8_8( a, b, m, n );
        }

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < iter; i++ )
        {
            trans_neon_8_8( a, b, m, n );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (64 elements * 4 bytes) * 2 (Read / Write) * iterations
        total_bytes = (64 * 4) * 2 * iter;
    }
    else if ( res_2 == 0 )
    {
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            trans_neon_4_4( a, b, m, n );
        }

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < iter; i++ )
        {
            trans_neon_4_4( a, b, m, n );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (16 elements * 4 bytes) * 2 (Read / Write) * iterations
        total_bytes = (16 * 4) * 2 * iter;
    }

    double movPerSec = total_bytes / ( elapsedTime * 1e9 );
    double gib_per_sec = total_bytes / ( elapsedTime * 1e9 ) / ( 1024 * 1024 * 1024 );

    std::cout << "Measuring throughput for transposition in GiB/s\n";
    std::cout << "Total time (s):   " << elapsedTime << "\n";
    std::cout << "Data movements per Second:   " << movPerSec << "\n";
    std::cout << "Estimated GiB/s:   " << gib_per_sec << " GiB/s\n";
    std::cout << "-----------------------------------------------\n";
}

int main() 
{
    const int M = 8;
    const int N = 8;

    float A[M * N];
    float B[M * N];

    // Initialize matrices
    for (int i = 0; i < M * N; ++i)
    {
        A[i] = static_cast<float>( i );
    }
    std::memset( B, 0, M * N * sizeof(float) ); 

    int64_t l_iter = 200000000;
    std::string neon_8_8( "neon_8_8" );

    std::cout << "\nBenchmarking trans_neon_8_8 performance ...\n";
    benchmark_thr( l_iter, neon_8_8, A, B, M, N );

    return 0;
}
