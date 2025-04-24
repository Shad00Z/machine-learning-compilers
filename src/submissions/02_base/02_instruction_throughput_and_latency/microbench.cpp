#include <iostream>
#include <cstdint>
#include <chrono>

extern "C" {
    void add_instr( int64_t l_n );

    void mul_instr( int64_t l_n );

    void add_lat_instr( int64_t l_n );

    void mul_lat_instr( int64_t l_n );
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

    std::string add( "ADD" );
    int res = add.compare( instruction );

    // Time measuring
    if ( res == 0 )
    {
        auto l_start_time = std::chrono::high_resolution_clock::now();
        add_instr( n );
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(l_end_time - l_start_time).count() / 1e6;
    }
    else 
    {
        auto l_start_time = std::chrono::high_resolution_clock::now();
        mul_instr( n );
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(l_end_time - l_start_time).count() / 1e6;
    }

    double totalOps = n * 25;
    double opsPerSec = totalOps / elapsedTime;
    double gops = opsPerSec / 1e9;

    std::cout << "Measuring throughput for " << "Instruction\n";
    std::cout << "Total time (s):   " << elapsedTime << "\n";
    std::cout << "Instructions per Second:   " << opsPerSec << "\n";
    std::cout << "Estimated GOPS:   " << gops << " GigaOps/sec\n";
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

    std::string add( "ADD" );
    int res = add.compare( instruction );

    // time measuring
    if ( res == 0 )
    {
        auto l_start_time = std::chrono::high_resolution_clock::now();
        add_lat_instr( n );
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(l_end_time - l_start_time).count() / 1e6;
    }
    else 
    {
        auto l_start_time = std::chrono::high_resolution_clock::now();
        mul_lat_instr( n );
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(l_end_time - l_start_time).count() / 1e6;
    }

    double totalOps = n * 25;
    double opsPerSec = totalOps / elapsedTime;
    double gops = opsPerSec / 1e9;

    std::cout << "Measuring latency for " << "Instruction\n";
    std::cout << "Total time (s):   " << elapsedTime << "\n";
    std::cout << "Instructions per Second:   " << opsPerSec << "\n";
    std::cout << "Estimated GOPS:   " << gops << " GigaOps/sec\n";
    std::cout << "-----------------------------------------------\n";
}

int main() 
{
    int64_t l_iter = 1500000000;
    std::string add( "ADD" );
    std::string mul( "MUL" );

    
    std::cout << "\nBenchmarking ADD throughput ...\n";
    benchmark_thr( l_iter, add );

    std::cout << "\nBenchmarking MUL throughput ...\n";
    benchmark_thr( l_iter, mul );

    l_iter = 150000000;

    std::cout << "\nBenchmarking ADD latency ...\n";
    benchmark_lat( l_iter, add );

    std::cout << "\nBenchmarking MUL latency ...\n";
    benchmark_lat( l_iter, mul );

    return 0;
}
