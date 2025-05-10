#include <arm_neon.h>
#include <iostream>
#include <cstdint>
#include <chrono>

extern "C" {
    // throughput
    void fmla_4s_instr( int64_t l_n,
                        float32x4_t * l_a );

    void fmla_2s_instr( int64_t l_n,
                        float32x2_t * l_a );

    void fmadd_instr( int64_t l_n,
                      float * l_a );

    // latency
    void fmla_4s_source_lat_instr( int64_t l_n,
                                   float32x4_t * l_a );

    void fmla_4s_dest_lat_instr( int64_t l_n,
                                 float32x4_t * l_a );
}

float32x4_t g_4s_registers[32];
float32x2_t g_2s_registers[32];
float       g_registers[32];

void initialize_registers() 
{
    for ( int i = 0; i < 32; ++i ) 
    {
        g_4s_registers[i] = vdupq_n_f32( static_cast<float>( i + 1 ) );
        g_2s_registers[i] = vdup_n_f32( static_cast<float>( i + 1 ) );
    }
}

/*
 * Benchmarks the throughput of either the ADD or MUL instruction.
 *
 * @param n: number of loop iterations.
 * @param instruction: a string ("ADD" or "MUL") selecting the instruction to benchmark.
 */
void benchmark_thr( int64_t n, 
                    std::string instruction ) 
{
    std::cout << "-----------------------------------------------\n";
    double elapsedTime = 1;

    std::string fmla4( "FMLA_4s" );
    int res_1 = fmla4.compare( instruction );

    std::string fmla2( "FMLA_2s" );
    int res_2 = fmla2.compare( instruction );

    double totalOps = 0.0;

    // Time measuring
    if ( res_1 == 0 )
    {
        // Warmup
        fmla_4s_instr( 100, g_4s_registers );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        fmla_4s_instr( n, g_4s_registers );
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // per FMLA: 4 Muls, 4 Adds
        // 32 fmla
        // rept 100
        // n: loop iterations
        totalOps = (2 * 4) * 32 * 100 * n;
    }
    else if ( res_2 == 0 )
    {
        // Warmup
        fmla_2s_instr( 100, g_2s_registers );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        fmla_2s_instr( n, g_2s_registers );
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // per FMLA: 2 Muls, 2 Adds
        // 32 fmla
        // rept 100
        // n: loop iterations
        totalOps = (2 * 2) * 32 * 100 * n;
    }
    else
    {
        // Warmup
        fmadd_instr( 100, g_registers );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        fmadd_instr( n, g_registers );
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // per FMADD: 1 Mul, 1 Add
        // 32 fmadd
        // rept 100
        // n: loop iterations
        totalOps = (2 * 1) * 32 * 100 * n;
    }
    
    double opsPerSec = totalOps / elapsedTime;
    double gflops = totalOps / (elapsedTime * 1e9);

    std::cout << "Measuring throughput for " << "Instruction\n";
    std::cout << "Total time (s):   " << elapsedTime << "\n";
    std::cout << "Instructions per Second:   " << opsPerSec << "\n";
    std::cout << "Estimated GOPS:   " << gflops << " GFLOPs/sec\n";
    std::cout << "-----------------------------------------------\n";
}

/*
 * Benchmarks the latency of either the ADD or MUL instruction.
 *
 * @param n: number of loop iterations.
 * @param instruction: a string ("ADD" or "MUL") selecting the instruction to benchmark.
 */
void benchmark_lat( int64_t n, 
                    std::string instruction ) 
{
    std::cout << "-----------------------------------------------\n";
    double elapsedTime = 1;

    std::string fmuls( "FMUL_Source" );
    int res = fmuls.compare( instruction );

    // time measuring
    if ( res == 0 )
    {
        // Warmup
        fmla_4s_source_lat_instr( 100, g_4s_registers );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        fmla_4s_source_lat_instr( n, g_4s_registers );
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(l_end_time - l_start_time).count() / 1e6;
    }
    else 
    {
        // Warmup
        fmla_4s_dest_lat_instr( 100, g_4s_registers );

        auto l_start_time = std::chrono::high_resolution_clock::now();
        fmla_4s_dest_lat_instr( n, g_4s_registers );
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(l_end_time - l_start_time).count() / 1e6;
    }

    // per FMLA: 4 Muls, 4 Adds
    // 32 fmla
    // rept 100
    // n: loop iterations
    double totalOps = (2 * 4) * 32 * 100 * n;
    double opsPerSec = totalOps / elapsedTime;
    double gflops = totalOps / (elapsedTime * 1e9);

    std::cout << "Measuring latency for " << instruction << "Instruction\n";
    std::cout << "Total time (s):   " << elapsedTime << "\n";
    std::cout << "Instructions per Second:   " << opsPerSec << "\n";
    std::cout << "Estimated GOPS:   " << gflops << " GFLOPs/sec\n";
    std::cout << "-----------------------------------------------\n";
}

int main() 
{
    initialize_registers();

    int64_t l_iter = 10000000;
    std::string fmla4( "FMLA_4s" );
    std::string fmla2( "FMLA_2s" );
    std::string fmadd( "FMADD" );
    
    std::cout << "\nBenchmarking FMLA 4s throughput ...\n";
    benchmark_thr( l_iter, fmla4 );

    std::cout << "\nBenchmarking FMLA 2s throughput ...\n";
    benchmark_thr( l_iter, fmla2 );

    std::cout << "\nBenchmarking FMADD throughput ...\n";
    benchmark_thr( l_iter, fmadd );


    l_iter = 1500000;
    std::string fmlaS( "FMLA_Source" );
    std::string fmlaD( "FMLA_Destination" );

    std::cout << "\nBenchmarking FMLA 4s source register latency ...\n";
    benchmark_lat( l_iter, fmlaS );

    std::cout << "\nBenchmarking FMLA 4s destination register latency ...\n";
    benchmark_lat( l_iter, fmlaD );

    return 0;
}
