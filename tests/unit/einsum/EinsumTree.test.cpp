#include <catch2/catch.hpp>
#include "EinsumTree.h"
#include "EinsumNode.h"
#include "constants.h"
#include <vector>
#include <map>
#include <iostream>

TEST_CASE("EinsumTree Simple GEMM Test")
{
    std::string input = "[2,0],[1,2]->[1,0]";
    std::vector<int64_t> dimension_sizes{GENERATE(3, 7, 19),
                                         GENERATE(3, 7, 19),
                                         GENERATE(3, 7, 19)};
    mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

    mini_jit::einsum::EinsumNode *node = mini_jit::einsum::EinsumTree::parse_einsum_expression(input,
                                                                                               dimension_sizes);

    // verify that the tree was created correctly
    REQUIRE(mini_jit::einsum::EinsumTree::to_string(node) == input);

    mini_jit::einsum::EinsumTree::optimize_einsum_nodes(node,
                                                        256,
                                                        512);
    mini_jit::einsum::EinsumTree::lower_einsum_nodes_to_tensor_operations(node,
                                                                          dimension_sizes,
                                                                          dtype);
    // [0,1] -> *A
    // [1,2] -> *B
    std::map<std::string, void const *> tensor_inputs;

    const int64_t M = dimension_sizes[0];
    const int64_t N = dimension_sizes[1];
    const int64_t K = dimension_sizes[2];

    const int64_t SIZE_A = M * K;
    const int64_t SIZE_B = K * N;
    const int64_t SIZE_OUT = M * N;

    float *tensor_A = new float[SIZE_A];
    float *tensor_B = new float[SIZE_B];
    float *tensor_out_expected = new float[SIZE_OUT];

    tensor_inputs["2,0"] = tensor_A;
    tensor_inputs["1,2"] = tensor_B;

    // init matrices
    for (int64_t i = 0; i < SIZE_A; ++i)
    {
        tensor_A[i] = i * 3.1f;
    }
    for (int64_t i = 0; i < SIZE_B; ++i)
    {
        tensor_B[i] = i * 0.5f;
    }
    for (int64_t i = 0; i < SIZE_OUT; ++i)
    {
        tensor_out_expected[i] = 0.0f;
    }

    for (int col = 0; col < N; ++col)
    {
        for (int row = 0; row < M; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += tensor_A[row + k * M] * tensor_B[k + col * K];
            }
            tensor_out_expected[row + col * M] += sum;
        }
    }

    mini_jit::einsum::EinsumTree::execute(node,
                                          dimension_sizes,
                                          tensor_inputs);

    const float *tensor_out = static_cast<const float *>(node->m_tensor_out);

    // compare output tensor with expected output
    for (int64_t i = 0; i < SIZE_OUT; ++i)
    {
        REQUIRE(tensor_out[i] == Approx(tensor_out_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete node;
    delete[] tensor_A;
    delete[] tensor_B;
    delete[] tensor_out_expected;
}

TEST_CASE("EinsumTree Simple BRGEMM Test")
{
    std::string input = "[3,2,0],[3,1,2]->[1,0]";
    std::vector<int64_t> dimension_sizes{GENERATE(3, 7, 19),
                                         GENERATE(3, 7, 19),
                                         GENERATE(3, 7, 19),
                                         GENERATE(3, 7, 19)};
    mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

    mini_jit::einsum::EinsumNode *node = mini_jit::einsum::EinsumTree::parse_einsum_expression(input,
                                                                                               dimension_sizes);

    // verify that the tree was created correctly
    REQUIRE(mini_jit::einsum::EinsumTree::to_string(node) == input);

    mini_jit::einsum::EinsumTree::optimize_einsum_nodes(node,
                                                        256,
                                                        512);

    mini_jit::einsum::EinsumTree::lower_einsum_nodes_to_tensor_operations(node,
                                                                          dimension_sizes,
                                                                          dtype);

    // [3,2,0] -> *A
    // [3,1,2] -> *B
    std::map<std::string, void const *> tensor_inputs;

    const int64_t M = dimension_sizes[0];
    const int64_t N = dimension_sizes[1];
    const int64_t K = dimension_sizes[2];
    const int64_t br_size = dimension_sizes[3];

    const int64_t SIZE_A = M * K * br_size;
    const int64_t SIZE_B = K * N * br_size;
    const int64_t SIZE_OUT = M * N;

    float *tensor_A = new float[SIZE_A];
    float *tensor_B = new float[SIZE_B];
    float *tensor_out_expected = new float[SIZE_OUT];

    tensor_inputs["3,2,0"] = tensor_A;
    tensor_inputs["3,1,2"] = tensor_B;

    // init matrices
    for (int64_t i = 0; i < SIZE_A; ++i)
    {
        tensor_A[i] = i;
    }
    for (int64_t i = 0; i < SIZE_B; ++i)
    {
        tensor_B[i] = i;
    }
    for (int64_t i = 0; i < SIZE_OUT; ++i)
    {
        tensor_out_expected[i] = 0.0f;
    }

    // calculate expected output
    for (int col = 0; col < N; ++col)
    {
        for (int row = 0; row < M; ++row)
        {
            float sum = 0.0f;
            for (int br = 0; br < br_size; ++br)
            {
                for (int k = 0; k < K; ++k)
                {
                    sum += tensor_A[br * M * K + row + k * M] * tensor_B[br * K * N + k + col * K];
                }
            }
            tensor_out_expected[row + col * M] += sum;
        }
    }

    mini_jit::einsum::EinsumTree::execute(node,
                                          dimension_sizes,
                                          tensor_inputs);

    mini_jit::einsum::EinsumTree::optimize_einsum_nodes(node,
                                                        256,
                                                        512);

    mini_jit::einsum::EinsumTree::lower_einsum_nodes_to_tensor_operations(node,
                                                                          dimension_sizes,
                                                                          dtype);

    const float *tensor_out = static_cast<const float *>(node->m_tensor_out);

    // compare output tensor with expected output
    for (int64_t i = 0; i < SIZE_OUT; ++i)
    {
        REQUIRE(tensor_out[i] == Approx(tensor_out_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete node;
    delete[] tensor_A;
    delete[] tensor_B;
    delete[] tensor_out_expected;
}

TEST_CASE("EinsumTree Simple Permutation Test")
{
    std::string input = "[3,2,1,0]->[3,1,2,0]";
    std::vector<int64_t> dimension_sizes{GENERATE(3, 7, 19),
                                         GENERATE(3, 7, 19),
                                         GENERATE(3, 7, 19),
                                         GENERATE(3, 7, 19)};
    mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

    mini_jit::einsum::EinsumNode *node = mini_jit::einsum::EinsumTree::parse_einsum_expression(input,
                                                                                               dimension_sizes);

    // verify that the tree was created correctly
    REQUIRE(mini_jit::einsum::EinsumTree::to_string(node) == input);

    mini_jit::einsum::EinsumTree::optimize_einsum_nodes(node,
                                                        256,
                                                        512);

    mini_jit::einsum::EinsumTree::lower_einsum_nodes_to_tensor_operations(node,
                                                                          dimension_sizes,
                                                                          dtype);

    std::map<std::string, void const *> tensor_inputs;

    const int64_t S = dimension_sizes[0];
    const int64_t U = dimension_sizes[1];
    const int64_t R = dimension_sizes[2];
    const int64_t T = dimension_sizes[3];

    const int64_t SIZE_A = T * R * U * S;
    const int64_t SIZE_OUT = T * U * R * S;

    float *tensor_A = new float[SIZE_A];
    float *tensor_out_expected = new float[SIZE_OUT];

    tensor_inputs["3,2,1,0"] = tensor_A;

    // init matrices
    for (int64_t i = 0; i < SIZE_A; ++i)
    {
        tensor_A[i] = i;
    }

    // Calculate expected output
    for (int t = 0; t < T; ++t)
    {
        for (int r = 0; r < R; ++r)
        {
            for (int u = 0; u < U; ++u)
            {
                for (int s = 0; s < S; ++s)
                {
                    // Calculate index in output format (t,r,u,s) using strides_out
                    int l_idx_c_exp = t * (U * R * S) + r * S + u * (R * S) + s;
                    // Calculate index in input format (t,u,r,s) using strides_in0
                    int l_idx_a = t * (R * U * S) + u * S + r * (U * S) + s;
                    tensor_out_expected[l_idx_c_exp] = tensor_A[l_idx_a];
                }
            }
        }
    }

    mini_jit::einsum::EinsumTree::execute(node,
                                          dimension_sizes,
                                          tensor_inputs);

    const float *tensor_out = static_cast<const float *>(node->m_tensor_out);

    // compare output tensor with expected output
    for (int64_t i = 0; i < SIZE_OUT; ++i)
    {
        REQUIRE(tensor_out[i] == Approx(tensor_out_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete node;
    delete[] tensor_A;
    delete[] tensor_out_expected;
}

TEST_CASE("EinsumTree Complex Permutation + GEMM Test")
{
    std::string input = "[[0,2]->[2,0]],[1,2]->[1,0]";
    std::vector<int64_t> dimension_sizes{GENERATE(3, 7, 19),
                                         GENERATE(3, 7, 19),
                                         GENERATE(3, 7, 19)};
    mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

    mini_jit::einsum::EinsumNode *node = mini_jit::einsum::EinsumTree::parse_einsum_expression(input,
                                                                                               dimension_sizes);

    // verify that the tree was created correctly
    REQUIRE(mini_jit::einsum::EinsumTree::to_string(node) == input);

    mini_jit::einsum::EinsumTree::optimize_einsum_nodes(node,
                                                        256,
                                                        512);

    mini_jit::einsum::EinsumTree::lower_einsum_nodes_to_tensor_operations(node,
                                                                          dimension_sizes,
                                                                          dtype);

    std::map<std::string, void const *> tensor_inputs;

    const int64_t M = dimension_sizes[0];
    const int64_t N = dimension_sizes[1];
    const int64_t K = dimension_sizes[2];

    const int64_t SIZE_A = M * K;
    const int64_t SIZE_B = K * N;
    const int64_t SIZE_OUT = M * N;

    float *tensor_A = new float[SIZE_A];
    float *tensor_A_intermediate = new float[SIZE_A];
    float *tensor_B = new float[SIZE_B];
    float *tensor_out_expected = new float[SIZE_OUT];

    tensor_inputs["0,2"] = tensor_A;
    tensor_inputs["1,2"] = tensor_B;

    // init matrices
    for (int64_t i = 0; i < SIZE_A; ++i)
    {
        tensor_A[i] = i;
    }
    for (int64_t i = 0; i < SIZE_B; ++i)
    {
        tensor_B[i] = i;
    }
    for (int64_t i = 0; i < SIZE_OUT; ++i)
    {
        tensor_out_expected[i] = 0.0f;
    }

    // Calculate expected output
    // transpose tensor A
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            tensor_A_intermediate[i + j * M] = tensor_A[j + i * K];
        }
    }

    // perform GEMM with transposed A
    for (int col = 0; col < N; ++col)
    {
        for (int row = 0; row < M; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += tensor_A_intermediate[row + k * M] * tensor_B[k + col * K];
            }
            tensor_out_expected[row + col * M] += sum;
        }
    }

    mini_jit::einsum::EinsumTree::execute(node,
                                          dimension_sizes,
                                          tensor_inputs);

    const float *tensor_out = static_cast<const float *>(node->m_tensor_out);

    // compare output tensor with expected output
    for (int64_t i = 0; i < SIZE_OUT; ++i)
    {
        REQUIRE(tensor_out[i] == Approx(tensor_out_expected[i]).margin(FLOAT_ERROR_MARGIN));
    }

    delete node;
    delete[] tensor_A;
    delete[] tensor_A_intermediate;
    delete[] tensor_B;
    delete[] tensor_out_expected;
}

// TEST_CASE("EinsumTree Simple Swap Test")
// {
//     std::string input = "[2,0,3],[3,1]->[2,0,1]";
//     std::vector<int64_t> dimension_sizes{GENERATE(3, 7, 19),
//                                          GENERATE(3, 7, 19),
//                                          GENERATE(3, 7, 19),
//                                          GENERATE(3, 7, 19)};
//     mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

//     mini_jit::einsum::EinsumNode *node = mini_jit::einsum::EinsumTree::parse_einsum_expression(input,
//                                                                                                dimension_sizes);

//     mini_jit::einsum::EinsumTree::optimize_einsum_nodes(node,
//                                                         256,
//                                                         512);

//     mini_jit::einsum::EinsumTree::lower_einsum_nodes_to_tensor_operations(node,
//                                                                           dimension_sizes,
//                                                                           dtype);

//     // [3,1] -> *A
//     // [2,0,3] -> *B
//     std::map<std::string, void const *> tensor_inputs;

//     const int64_t M = dimension_sizes[1];
//     const int64_t N = dimension_sizes[0];
//     const int64_t K = dimension_sizes[3];
//     const int64_t br_size = dimension_sizes[2];

//     const int64_t SIZE_A = M * K;
//     const int64_t SIZE_B = K * N * br_size;
//     const int64_t SIZE_OUT = M * N * br_size;

//     float *tensor_A = new float[SIZE_A];
//     float *tensor_B = new float[SIZE_B];
//     float *tensor_out_expected = new float[SIZE_OUT];

//     tensor_inputs["3,1"] = tensor_A;
//     tensor_inputs["2,0,3"] = tensor_B;

//     // init matrices
//     for (int64_t i = 0; i < SIZE_A; ++i)
//     {
//         tensor_A[i] = i;
//     }
//     for (int64_t i = 0; i < SIZE_B; ++i)
//     {
//         tensor_B[i] = i;
//     }
//     for (int64_t i = 0; i < SIZE_OUT; ++i)
//     {
//         tensor_out_expected[i] = 0.0f;
//     }

//     // calculate expected output
//     for (int col = 0; col < N; ++col)
//     {
//         for (int row = 0; row < M; ++row)
//         {
//             float sum = 0.0f;
//             for (int br = 0; br < br_size; ++br)
//             {
//                 for (int k = 0; k < K; ++k)
//                 {
//                     sum += tensor_A[br * M * K + row + k * M] * tensor_B[br * K * N + k + col * K];
//                 }
//             }
//             tensor_out_expected[row + col * M] += sum;
//         }
//     }

//     mini_jit::einsum::EinsumTree::execute(node,
//                                           dimension_sizes,
//                                           tensor_inputs);

//     mini_jit::einsum::EinsumTree::optimize_einsum_nodes(node,
//                                                         256,
//                                                         512);

//     mini_jit::einsum::EinsumTree::lower_einsum_nodes_to_tensor_operations(node,
//                                                                           dimension_sizes,
//                                                                           dtype);

//     const float *tensor_out = static_cast<const float *>(node->tensor_out);

//     // compare output tensor with expected output
//     for (int64_t i = 0; i < SIZE_OUT; ++i)
//     {
//         REQUIRE(tensor_out[i] == Approx(tensor_out_expected[i]).margin(FLOAT_ERROR_MARGIN));
//     }

//     delete node;
//     delete[] tensor_A;
//     delete[] tensor_B;
//     delete[] tensor_out_expected;
// }

// TEST_CASE("huhn")
// {
//     std::string input = "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]";
//     std::vector<int64_t> dimension_sizes{60, 60, 20, 20, 8, 8, 8, 8, 8, 8};
//     mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

//     mini_jit::einsum::EinsumNode *node = mini_jit::einsum::EinsumTree::parse_einsum_expression(input, dimension_sizes, dtype);

//     // [3,6,8,9] -> *A
//     // [2,5,7,9] -> *B
//     // [0,4,5,6] -> *C
//     // [1,4,7,8] -> *D
//     std::map<std::string, void const *> tensor_inputs;

//     const int64_t SIZE_A = dimension_sizes[3] * dimension_sizes[6] * dimension_sizes[8] * dimension_sizes[9];
//     const int64_t SIZE_B = dimension_sizes[2] * dimension_sizes[5] * dimension_sizes[7] * dimension_sizes[9];
//     const int64_t SIZE_C = dimension_sizes[0] * dimension_sizes[4] * dimension_sizes[5] * dimension_sizes[6];
//     const int64_t SIZE_D = dimension_sizes[1] * dimension_sizes[4] * dimension_sizes[7] * dimension_sizes[8];

//     const int64_t SIZE_OUT = dimension_sizes[0] * dimension_sizes[1] * dimension_sizes[2] * dimension_sizes[3];

//     float *tensor_A = new float[SIZE_A];
//     float *tensor_B = new float[SIZE_B];
//     float *tensor_C = new float[SIZE_C];
//     float *tensor_D = new float[SIZE_D];

//     tensor_inputs["3,6,8,9"] = tensor_A;
//     tensor_inputs["2,5,7,9"] = tensor_B;
//     tensor_inputs["0,4,5,6"] = tensor_C;
//     tensor_inputs["1,4,7,8"] = tensor_D;

//     // init matrices
//     for (int64_t i = 0; i < SIZE_A; ++i)
//     {
//         tensor_A[i] = i;
//     }
//     for (int64_t i = 0; i < SIZE_B; ++i)
//     {
//         tensor_B[i] = i;
//     }
//     for (int64_t i = 0; i < SIZE_C; ++i)
//     {
//         tensor_C[i] = i;
//     }
//     for (int64_t i = 0; i < SIZE_D; ++i)
//     {
//         tensor_D[i] = i;
//     }

//     mini_jit::einsum::EinsumTree::execute(node, dimension_sizes, tensor_inputs);

//     const float *tensor_out = static_cast<const float *>(node->tensor_out);

//     // print output tensor
//     // for (int64_t i = 0; i < SIZE_OUT; ++i)
//     // {
//     //     std::cout << tensor_out[i] << " ";
//     // }
//     // std::cout << std::endl;

//     delete node;
//     delete[] tensor_A;
//     delete[] tensor_B;
//     delete[] tensor_C;
//     delete[] tensor_D;
// }