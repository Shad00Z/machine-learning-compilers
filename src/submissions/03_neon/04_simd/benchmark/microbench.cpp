#include <arm_neon.h>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <cstring>

extern "C" {
    void v1_matmul_14_6_64( float const *a,
                            float const *b,
                            float *c,
                            int64_t lda,
                            int64_t ldb,
                            int64_t ldc );

    void v2_matmul_14_6_64( float const *a,
                            float const *b,
                            float *c,
                            int64_t lda,
                            int64_t ldb,
                            int64_t ldc );

    void v3_matmul_14_6_64( float const *a,
                            float const *b,
                            float *c,
                            int64_t lda,
                            int64_t ldb,
                            int64_t ldc );

    void v4_matmul_14_6_64( float const *a,
                            float const *b,
                            float *c,
                            int64_t lda,
                            int64_t ldb,
                            int64_t ldc );

    void v1_matmul_15_6_64( float const *a,
                            float const *b,
                            float *c,
                            int64_t lda,
                            int64_t ldb,
                            int64_t ldc );

    void v2_matmul_15_6_64( float const *a,
                            float const *b,
                            float *c,
                            int64_t lda,
                            int64_t ldb,
                            int64_t ldc );

    void v3_matmul_15_6_64( float const *a,
                            float const *b,
                            float *c,
                            int64_t lda,
                            int64_t ldb,
                            int64_t ldc );

    void matmul_M_6_64( float const *a,
                        float const *b,
                        float *c,
                        int64_t lda,
                        int64_t ldb,
                        int64_t ldc );
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
 void benchmark_thr( int64_t n, 
                     std::string instruction,
                     float const *a,
                     float const *b,
                     float *c ) 
{
    std::cout << "-----------------------------------------------\n";
    double elapsedTime = 1;

    std::string v1_matmul_14( "v1_matmul_14" );
    int res_1 = v1_matmul_14.compare( instruction );

    std::string v2_matmul_14( "v2_matmul_14" );
    int res_2 = v2_matmul_14.compare( instruction );

    std::string v3_matmul_14( "v3_matmul_14" );
    int res_3 = v3_matmul_14.compare( instruction );

    std::string v4_matmul_14( "v4_matmul_14" );
    int res_4 = v4_matmul_14.compare( instruction );

    std::string v1_matmul_15( "v1_matmul_15" );
    int res_5 = v1_matmul_15.compare( instruction );

    std::string v2_matmul_15( "v2_matmul_15" );
    int res_6 = v2_matmul_15.compare( instruction );

    std::string v3_matmul_15( "v3_matmul_15" );
    int res_7 = v3_matmul_15.compare( instruction );

    std::string g_matmul_14( "g_matmul_14" );
    int res_8 = g_matmul_14.compare( instruction );

    std::string g_matmul_15( "g_matmul_15" );
    int res_9 = g_matmul_15.compare( instruction );

    double opsPerMatmul = 1;

    // Time measuring
    if ( res_1 == 0 )
    {
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            v1_matmul_14_6_64( a, 
                               b, 
                               c, 
                               14, 
                               64, 
                               14 );
        }
        std::memset( c, 0, 14 * 6 * sizeof( float ) );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < n; i++ )
        {
            v1_matmul_14_6_64( a, 
                               b, 
                               c, 
                               14, 
                               64, 
                               14 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (M x N x K) x (FMLA OPs)
        opsPerMatmul = ( 14 * 6 * 64 ) * 2;
    }
    else if ( res_2 == 0 )
    {
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            v2_matmul_14_6_64( a, 
                               b, 
                               c, 
                               14, 
                               64, 
                               14 );
        }
        std::memset( c, 0, 14 * 6 * sizeof( float ) );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < n; i++ )
        {
            v2_matmul_14_6_64( a, 
                               b, 
                               c, 
                               14, 
                               64, 
                               14 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (M x N x K) x (FMLA OPs)
        opsPerMatmul = ( 14 * 6 * 64 ) * 2;
    }
    else if ( res_3 == 0 )
    {
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            v3_matmul_14_6_64( a, 
                               b, 
                               c, 
                               14, 
                               64, 
                               14 );
        }
        std::memset( c, 0, 14 * 6 * sizeof( float ) );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < n; i++ )
        {
            v3_matmul_14_6_64( a, 
                               b, 
                               c, 
                               14, 
                               64, 
                               14 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (M x N x K) x (FMLA OPs)
        opsPerMatmul = ( 14 * 6 * 64 ) * 2;
    }
    else if ( res_4 == 0 )
    {
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            v4_matmul_14_6_64( a, 
                               b, 
                               c, 
                               14, 
                               64, 
                               14 );
        }
        std::memset( c, 0, 14 * 6 * sizeof( float ) );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < n; i++ )
        {
            v4_matmul_14_6_64( a, 
                             b, 
                             c, 
                             14, 
                             64, 
                             14 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (M x N x K) x (FMLA OPs)
        opsPerMatmul = ( 14 * 6 * 64 ) * 2;
    }
    else if ( res_5 == 0 )
    {
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            v1_matmul_15_6_64( a, 
                               b, 
                               c, 
                               15, 
                               64, 
                               15 );
        }
        std::memset( c, 0, 15 * 6 * sizeof( float ) );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < n; i++ )
        {
            v1_matmul_15_6_64( a, 
                               b, 
                               c, 
                               15, 
                               64, 
                               15 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (M x N x K) x (FMLA OPs)
        opsPerMatmul = ( 15 * 6 * 64 ) * 2;
    }
    else if ( res_6 == 0 )
    {
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            v2_matmul_15_6_64( a, 
                               b, 
                               c, 
                               15, 
                               64, 
                               15 );
        }
        std::memset( c, 0, 15 * 6 * sizeof( float ) );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < n; i++ )
        {
            v2_matmul_15_6_64( a, 
                               b, 
                               c, 
                               15, 
                               64, 
                               15 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (M x N x K) x (FMLA OPs)
        opsPerMatmul = ( 15 * 6 * 64 ) * 2;
    }
    else if ( res_7 == 0 )
    {
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            v3_matmul_15_6_64( a, 
                               b, 
                               c, 
                               15, 
                               64, 
                               15 );
        }
        std::memset( c, 0, 15 * 6 * sizeof( float ) );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < n; i++ )
        {
            v3_matmul_15_6_64( a, 
                               b, 
                               c, 
                               15, 
                               64, 
                               15 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (M x N x K) x (FMLA OPs)
        opsPerMatmul = ( 15 * 6 * 64 ) * 2;
    }
    else if ( res_8 == 0 )
    {
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            matmul_M_6_64( a, 
                           b, 
                           c, 
                           14, 
                           64, 
                           14 );
        }
        std::memset( c, 0, 14 * 6 * sizeof( float ) );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < n; i++ )
        {
            matmul_M_6_64( a, 
                           b, 
                           c, 
                           14, 
                           64, 
                           14 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (M x N x K) x (FMLA OPs)
        opsPerMatmul = ( 14 * 6 * 64 ) * 2;
    }
    else if ( res_9 == 0 )
    {
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            matmul_M_6_64( a, 
                           b, 
                           c, 
                           15, 
                           64, 
                           15 );
        }
        std::memset( c, 0, 15 * 6 * sizeof( float ) );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < n; i++ )
        {
            matmul_M_6_64( a, 
                           b, 
                           c, 
                           15, 
                           64, 
                           15 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (M x N x K) x (FMLA OPs)
        opsPerMatmul = ( 15 * 6 * 64 ) * 2;
    }

    double loopIterations = n;
    double totalFLOPs = opsPerMatmul * loopIterations;

    double flopsPerSec = totalFLOPs / elapsedTime;
    double gflops = totalFLOPs / ( elapsedTime * 1e9 );

    std::cout << "Measuring throughput for " << "Instruction\n";
    std::cout << "Total time (s):   " << elapsedTime << "\n";
    std::cout << "Instructions per Second:   " << flopsPerSec << "\n";
    std::cout << "Estimated GFLOPS:   " << gflops << " GFLOPS/sec\n";
    std::cout << "-----------------------------------------------\n";
}

int main() 
{
    const int M = 14;
    const int N = 6;
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
    std::memset( C, 0, sizeof( C ) );

    int64_t l_iter = 10000 * 2000;
    std::string v1_matmul_14( "v1_matmul_14" );
    std::string v2_matmul_14( "v2_matmul_14" );
    std::string v3_matmul_14( "v3_matmul_14" );
    std::string v4_matmul_14( "v4_matmul_14" );
    std::string g_matmul_14( "g_matmul_14" );

    std::cout << "\nBenchmarking V1_Matmul_14_6_64 throughput ...\n";
    benchmark_thr( l_iter, v1_matmul_14, A, B, C );
    std::memset( C, 0, 14 * 6 * sizeof( float ) );

    std::cout << "\nBenchmarking V2_Matmul_14_6_64 throughput ...\n";
    benchmark_thr( l_iter, v2_matmul_14, A, B, C );
    std::memset( C, 0, 14 * 6 * sizeof( float ) );

    std::cout << "\nBenchmarking V3_Matmul_14_6_64 throughput ...\n";
    benchmark_thr( l_iter, v3_matmul_14, A, B, C );
    std::memset( C, 0, 14 * 6 * sizeof( float ) );

    std::cout << "\nBenchmarking V4_Matmul_14_6_64 throughput ...\n";
    benchmark_thr( l_iter, v4_matmul_14, A, B, C );
    std::memset( C, 0, 14 * 6 * sizeof( float ) );

    std::cout << "\nBenchmarking Matmul_M_6_64 M=14 throughput ...\n";
    benchmark_thr( l_iter, g_matmul_14, A, B, C );
    std::memset( C, 0, 14 * 6 * sizeof( float ) );


    const int m = 15;
    const int n = 6;
    const int k = 64;

    float a[m * k];
    float b[k * n];
    float c[m * n];

    // Initialize matrices
    for ( int i = 0; i < m * k; ++i )
    {
        a[i] = static_cast<float>( i );
    }
    for ( int j = 0; j < k * n; ++j )
    {
        b[j] = static_cast<float>( j );
    }
    std::memset( c, 0, sizeof( c ) );

    std::string v1_matmul_15( "v1_matmul_15" );
    std::string v2_matmul_15( "v2_matmul_15" );
    std::string v3_matmul_15( "v3_matmul_15" );
    std::string g_matmul_15( "g_matmul_15" );

    std::cout << "\nBenchmarking V1_Matmul_15_6_64 throughput ...\n";
    benchmark_thr( l_iter, v1_matmul_15, a, b, c );
    std::memset( c, 0, 15 * 6 * sizeof( float ) );

    std::cout << "\nBenchmarking V2_Matmul_15_6_64 throughput ...\n";
    benchmark_thr( l_iter, v2_matmul_15, a, b, c );
    std::memset( c, 0, 15 * 6 * sizeof( float ) );

    std::cout << "\nBenchmarking V3_Matmul_15_6_64 throughput ...\n";
    benchmark_thr( l_iter, v3_matmul_15, a, b, c );
    std::memset( c, 0, 15 * 6 * sizeof( float ) );

    std::cout << "\nBenchmarking Matmul_M_6_64 M=15 throughput ...\n";
    benchmark_thr( l_iter, g_matmul_15, a, b, c );
    std::memset( c, 0, 15 * 6 * sizeof( float ) );

    return 0;
}
