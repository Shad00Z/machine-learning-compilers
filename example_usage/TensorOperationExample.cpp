#include <cstdint>
#include <iostream>
#include <mlc/TensorOperation.h>
#include <mlc/constants.h>
#include <mlc/types.h>
#include <random>
#include <vector>

int main()
{
    // This example demonstrates how to use the TensorOperation class to perform tensor operations
    std::cout << "Running the TensorOperation example" << std::endl;

    // First, we need to define some input tensors and their dimensions.
    // This example uses three tensors: A, B, and C.
    const int R = 3;
    const int P = 4;
    const int T = 5;
    const int S = 6;
    const int Q = 7;
    const int U = 8;

    // Tensor A has dimensions R, S, T, U
    // Tensor B has dimensions T, U, P, Q
    // Tensor C has dimensions R, S, P, Q
    const int SIZE_A = (R * S) * (T * U);
    const int SIZE_B = (T * U) * (P * Q);
    const int SIZE_C = (R * S) * (P * Q);

    float* A = new float[SIZE_A];
    float* B = new float[SIZE_B];
    float* C = new float[SIZE_C];

    // Next, we fill the input tensors with some data.
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < SIZE_A; ++i)
    {
        A[i] = dist(gen);
    }
    for (int i = 0; i < SIZE_B; ++i)
    {
        B[i] = dist(gen);
    }
    for (int i = 0; i < SIZE_C; ++i)
    {
        C[i] = dist(gen);
    }

    // To perform a tensor operation, we first need to format the data into a suitable structure
    // for our TensorOperation backened. We use vectors to hold the data and dimensions.

    // Define the dimensions of the tensors
    std::vector<int64_t> dim_sizes = {
        R,
        P,
        T,
        S,
        Q,
        U};

    // Define the types of the dimensions
    std::vector<mini_jit::dim_t> dim_types = {
        mini_jit::dim_t::m,
        mini_jit::dim_t::n,
        mini_jit::dim_t::k,
        mini_jit::dim_t::m,
        mini_jit::dim_t::n,
        mini_jit::dim_t::k};

    // Define the strides for the input tensors
    std::vector<int64_t> strides_in0 = {U * S,
                                        0,
                                        R * U * S,
                                        1,
                                        0,
                                        S};
    std::vector<int64_t> strides_in1 = {0,
                                        Q * T * U,
                                        U,
                                        0,
                                        T * U,
                                        1};
    std::vector<int64_t> strides_out = {S,
                                        Q * R * S,
                                        0,
                                        1,
                                        R * S,
                                        0};

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // GEMM Operation Example
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Lastly, we need to define the execution types for each dimension,
    // as well as the types of operations we want to perform.
    // A simple GEMM operation for example needs three primitive dimensions,
    // while the other dimensions can be sequential or shared (parallel execution).
    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::seq,
        mini_jit::exec_t::seq,
        mini_jit::exec_t::shared,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim};

    // We can also define a first and last touch type,
    // which will be executed before and after the main operation.
    mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::zero;
    mini_jit::ptype_t main_type        = mini_jit::ptype_t::gemm;
    mini_jit::ptype_t last_touch_type  = mini_jit::ptype_t::none;

    // Now we can create a TensorOperation object and set it up with the defined parameters.
    mini_jit::TensorOperation l_top;
    l_top.setup(mini_jit::dtype_t::fp32,
                first_touch_type,
                main_type,
                last_touch_type,
                dim_types,
                exec_types,
                dim_sizes,
                strides_in0,
                strides_in1,
                strides_out);

    // Lastly, we can execute the tensor operation.
    // The execute function takes the input tensors and an output tensor.
    // In this case, we will use C as the output tensor.
    // Since we are using a GEMM operation, this will compute C += A * B
    l_top.execute(A, B, C);
    // The result will be stored in C, which is the output tensor.

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // BRGEMM Operation Example
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // To perform a batch-reduce GEMM operation, we would need four primitive dimensions:
    // M, N, K as before and a fourth K dimension for the batch size.
    std::vector<mini_jit::exec_t> exec_types_br = {
        mini_jit::exec_t::seq,
        mini_jit::exec_t::seq,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim,
        mini_jit::exec_t::prim};

    // We also need to adjust the main type to BRGEMM for batch-reduce operations.
    mini_jit::ptype_t main_type_brgemm = mini_jit::ptype_t::brgemm;

    // We can also select a different last touch, for example, a fast sigmoid operation
    // to apply a non-linear activation function after the batch-reduce GEMM.
    mini_jit::ptype_t last_touch_type_brgemm = mini_jit::ptype_t::fast_sigmoid;

    // We can create another TensorOperation object for the batch-reduce GEMM operation.
    mini_jit::TensorOperation l_top_brgemm;
    l_top_brgemm.setup(mini_jit::dtype_t::fp32,
                       first_touch_type,
                       main_type_brgemm,
                       last_touch_type_brgemm,
                       dim_types,
                       exec_types_br,
                       dim_sizes,
                       strides_in0,
                       strides_in1,
                       strides_out);

    // Execute the batch-reduce GEMM operation.
    l_top_brgemm.execute(A, B, C);
    // The result will be stored in C, which is the output tensor for the batch-reduce operation.

    // Lastly, we can clean up the allocated memory for the input tensors.
    delete[] A;
    delete[] B;
    delete[] C;

    std::cout << "---------------------------------" << std::endl;
    return 0;
}