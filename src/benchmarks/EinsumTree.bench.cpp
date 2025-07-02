#include <random>
#include <chrono>
#include <span>
#include "EinsumTree.bench.h"
#include "benchmarks/Benchmark.h"
#include <iostream>

mini_jit::benchmarks::EinsumTreeBench::EinsumTreeBench(double run_time,
                                                       std::string const &einsum_expression,
                                                       std::vector<int64_t> &dimension_sizes,
                                                       mini_jit::dtype_t dtype,
                                                       int64_t thread_target,
                                                       int64_t max_kernel_size,
                                                       int64_t min_kernel_size,
                                                       std::map<std::string, void const *> &tensor_inputs) : Benchmark()
{
    m_run_time = run_time;
    m_dimension_sizes = dimension_sizes;
    m_tensor_inputs = tensor_inputs;
    m_root_node = mini_jit::einsum::EinsumTree::parse_einsum_expression(einsum_expression,
                                                                        dimension_sizes);

    mini_jit::einsum::EinsumTree::optimize_einsum_nodes(m_root_node, 
                                                        thread_target, 
                                                        max_kernel_size,
                                                        min_kernel_size);
    
    mini_jit::einsum::EinsumTree::lower_einsum_nodes_to_tensor_operations(m_root_node, 
                                                                          dimension_sizes, 
                                                                          dtype);
}

void mini_jit::benchmarks::EinsumTreeBench::run()
{
    // RUN
    long l_num_reps = 0;
    auto l_start_time = std::chrono::high_resolution_clock::now();
    double l_elapsed = 0.0;
    double l_run_time_ms = m_run_time * 1e6;
    do
    {
        mini_jit::einsum::EinsumTree::execute(m_root_node,
                                              m_dimension_sizes,
                                              m_tensor_inputs);
        ++l_num_reps;
        auto l_now = std::chrono::high_resolution_clock::now();
        l_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(l_now - l_start_time).count();
    } while (l_elapsed < l_run_time_ms);
    l_elapsed /= 1e6; // Convert to seconds
    // END RUN

    // Calculate metrics
    double l_totalOperations = m_root_node->m_computational_operations * l_num_reps;
    double l_gflops = ((double)l_totalOperations) / (l_elapsed * 1e9);

    // Store the results
    m_benchmarkResult.numReps = l_num_reps;
    m_benchmarkResult.elapsedSeconds = l_elapsed;
    m_benchmarkResult.totalOperations = l_totalOperations;
    m_benchmarkResult.gflops = l_gflops;
}
