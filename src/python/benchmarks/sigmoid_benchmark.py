"""
Sigmoid Performance Benchmark

This script benchmarks the performance of applying sigmoid functions to matrices
of different sizes using PyTorch, similar to the C++ benchmarks to compare the difference in performance.
"""

import torch
import time
import numpy as np
from typing import Callable, List, Tuple

def real_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Real sigmoid function: σ(x) = 1 / (1 + e^(-x))"""
    return torch.sigmoid(x)

def fast_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Fast sigmoid approximation: f(x) = 0.5 * (x / (1 + abs(x)) + 1)"""
    return 0.5 * (x / (1 + torch.abs(x)) + 1)

def taylor_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Taylor series approximation (5th order): f(x) = 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5"""
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    return 0.5 + 0.25 * x - 0.020833333 * x3 + 0.002083333 * x5

def benchmark_sigmoid(func: Callable, matrix_size: Tuple[int, int], run_time: float = 3.0, device: str = 'cpu') -> dict:
    """
    Benchmark a sigmoid function on a matrix of given size.
    
    Args:
        func: Sigmoid function to benchmark
        matrix_size: Tuple of (rows, cols) for matrix size
        run_time: Time to run benchmark in seconds
        device: Device to run on ('cpu') - fair comparison
    
    Returns:
        Dictionary with benchmark results
    """
    m, n = matrix_size
    total_elements = m * n
    
    # Create random input matrix
    input_matrix = torch.randn(m, n, device=device)
    
    # Warm up
    for _ in range(10):
        _ = func(input_matrix)
    
    # Benchmark
    start_time = time.time()
    num_reps = 0
    elapsed = 0.0
    
    while elapsed < run_time:
        result = func(input_matrix)
        num_reps += 1
        
        elapsed = time.time() - start_time
    
    # Calculate metrics
    total_processed_elements = total_elements * num_reps * 2  # Input + output
    total_data_processed_gb = (total_processed_elements * 4) / (1024**3)  # 4 bytes per float32
    bandwidth_gbps = total_data_processed_gb / elapsed
    
    return {
        'matrix_size': f"{m}x{n}",
        'total_time': elapsed,
        'num_reps': num_reps,
        'total_elements': total_elements * num_reps,
        'total_data_gb': total_data_processed_gb,
        'bandwidth_gbps': bandwidth_gbps,
        'elements_per_second': total_elements * num_reps / elapsed
    }

def format_benchmark_results(results: dict, func_name: str) -> str:
    """Format benchmark results"""
    lines = []
    lines.append(f"Running {func_name} {results['matrix_size']} benchmark")
    lines.append(f"Total time (s):                       {results['total_time']:.6f}")
    lines.append(f"Total reps:                           {results['num_reps']}")
    lines.append(f"Total number of elements:             {results['total_elements']}")
    lines.append(f"Total amount of processed data (GiB): {results['total_data_gb']:.6f}")
    lines.append(f"Bandwidth (GiB/s)                     {results['bandwidth_gbps']:.6f}")
    lines.append("-" * 50)
    return "\n".join(lines)

def main():
    matrix_sizes = [(50, 50), (64, 64), (512, 512), (2048, 2048)]
    run_time = 3.0
    
    # Check device (currently only cpu)
    device = 'cpu'
    
    # Prepare output
    output_lines = []
    output_lines.append(f"Running benchmarks on: {device.upper()}")
    output_lines.append(f"PyTorch version: {torch.__version__}")
    output_lines.append("=" * 50)
    
    sigmoid_functions = [
        ("RealSigmoid", real_sigmoid),
        ("FastSigmoid", fast_sigmoid),
        ("TaylorSigmoid", taylor_sigmoid)
    ]
    
    # Run benchmarks
    for func_name, sigmoid_func in sigmoid_functions:
        output_lines.append(f"\n{func_name} Benchmarks:")
        output_lines.append("-" * 50)
        
        for matrix_size in matrix_sizes:
            try:
                results = benchmark_sigmoid(sigmoid_func, 
                                            matrix_size, 
                                            run_time, 
                                            device)
                output_lines.append(format_benchmark_results(results, func_name))
            except Exception as e:
                error_msg = f"Error benchmarking {func_name} {matrix_size}: {e}"
                output_lines.append(error_msg)
                print(error_msg)
    
    # Save results to file
    output_file = "benchmarks/pytorch/pytorch_sigmoid_benchmark.txt"
    try:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results to file: {e}")
        print('\n'.join(output_lines))

if __name__ == "__main__":
    main() 