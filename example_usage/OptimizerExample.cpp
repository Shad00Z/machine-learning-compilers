#include <iostream>
#include <mlc/ir/IRConverter.h>
#include <mlc/ir/Optimizer.h>
#include <mlc/types.h>

/**
 * Prints the given dimensions, execution types, sizes, and strides.
 */
void print_config(const std::vector<mini_jit::dim_t>&  dim_types,
                  const std::vector<mini_jit::exec_t>& exec_types,
                  const std::vector<int64_t>&          dim_sizes,
                  const std::vector<int64_t>&          strides_in0,
                  const std::vector<int64_t>&          strides_in1,
                  const std::vector<int64_t>&          strides_out)
{
    std::cout << "Dimension types: ";
    for (const auto& dim_type : dim_types)
    {
        std::cout << to_string(dim_type) << " ";
    }
    std::cout << "\nExecution types: ";
    for (const auto& exec_type : exec_types)
    {
        std::cout << to_string(exec_type) << " ";
    }
    std::cout << "\nDimension sizes: ";
    for (const auto& dim_size : dim_sizes)
    {
        std::cout << dim_size << " ";
    }
    std::cout << "\nStrides in0: ";
    for (const auto& stride : strides_in0)
    {
        std::cout << stride << " ";
    }
    std::cout << "\nStrides in1: ";
    for (const auto& stride : strides_in1)
    {
        std::cout << stride << " ";
    }
    std::cout << "\nStrides out: ";
    for (const auto& stride : strides_out)
    {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
}

int main()
{
    // This example demonstrates how to use the Optimizer
    // As seen in the TensorOperation example, we can create a TensorOperation object
    // and set it up with the required parameters. In that example, we had to manually
    // defines the execution types and choose the main type of operation, such as GEMM or BRGEMM.
    // In this Optimizer example, we will use the Optimizer class to automatically
    // generate the execution types and main type based on the tensor dimensions and operations.
    std::cout << "Running the Optimizer example" << std::endl;

    // First, we will define some tensor dimensions and their properties.
    std::vector<mini_jit::dim_t> dim_types = {mini_jit::dim_t::m,
                                              mini_jit::dim_t::m,
                                              mini_jit::dim_t::n,
                                              mini_jit::dim_t::n,
                                              mini_jit::dim_t::k,
                                              mini_jit::dim_t::k};
    // Note that the execution types are all set to sequential (seq) initially.
    // The Optimizer will later determine the best execution types based on the dimensions.
    std::vector<mini_jit::exec_t> exec_types  = {mini_jit::exec_t::seq,
                                                 mini_jit::exec_t::seq,
                                                 mini_jit::exec_t::seq,
                                                 mini_jit::exec_t::seq,
                                                 mini_jit::exec_t::seq,
                                                 mini_jit::exec_t::seq};
    std::vector<int64_t>          dim_sizes   = {64,
                                                 25,
                                                 64,
                                                 25,
                                                 64,
                                                 25};
    std::vector<int64_t>          strides_in0 = {25,
                                                 1,
                                                 0,
                                                 0,
                                                 40000,
                                                 1600};
    std::vector<int64_t>          strides_in1 = {0,
                                                 0,
                                                 40000,
                                                 1600,
                                                 25,
                                                 1};
    std::vector<int64_t>          strides_out = {25,
                                                 1,
                                                 40000,
                                                 1600,
                                                 0,
                                                 0};

    // Next, we will define the target number of threads, maximum kernel size, and minimum kernel size.
    const int64_t thread_target   = 64;
    const int64_t max_kernel_size = 1024;
    const int64_t min_kernel_size = 32;

    // In our defined dimensions, we can see that some dimensions are smaller than the minimum kernel size
    // that we would like to have. The Optimizer will try to fuse these dimensions to create larger kernels.
    // However fusing the two M dimensions (64, 25) would result in a kernel size of 1600,
    // which is larger than the maximum kernel size we defined (1024).
    // The Optimizer will therefore try to split the M dimension again into more manageable sizes.
    // Furthermore, we set a thread target of 64, which means that the Optimizer will try to create
    // shared loops that can be executed in parallel, up to 64 threads.

    // Print the initial configuration
    std::cout << "Initial configuration:" << std::endl;
    print_config(dim_types,
                 exec_types,
                 dim_sizes,
                 strides_in0,
                 strides_in1,
                 strides_out);

    // We can call the Optimizer the following way:
    mini_jit::ir::Optimizer::optimize(dim_types,
                                      exec_types,
                                      dim_sizes,
                                      strides_in0,
                                      strides_in1,
                                      strides_out,
                                      thread_target,
                                      max_kernel_size,
                                      min_kernel_size);

    // After the optimization, we can print the optimized configuration
    std::cout << "Optimized configuration:" << std::endl;
    print_config(dim_types,
                 exec_types,
                 dim_sizes,
                 strides_in0,
                 strides_in1,
                 strides_out);

    std::cout << "---------------------------------" << std::endl;
    return 0;
}