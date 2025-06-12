#include <catch2/catch.hpp>
#include "EinsumTree.h"
#include "EinsumNode.h"
#include <vector>
#include <map>
#include <iostream>

TEST_CASE("huhn")
{
    std::string input = "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]";
    std::vector<int64_t> dimension_sizes{60, 60, 20, 20, 8, 8, 8, 8, 8, 8};
    mini_jit::dtype_t dtype = mini_jit::dtype_t::fp32;

    mini_jit::einsum::EinsumNode *node = mini_jit::einsum::EinsumTree::parse_einsum_expression(input, dimension_sizes, dtype);

    // [3,6,8,9] -> *A
    // [2,5,7,9] -> *B
    // [0,4,5,6] -> *C
    // [1,4,7,8] -> *D
    std::map<std::string, void const *> tensor_inputs; 

    const int64_t SIZE_A = dimension_sizes[3] * dimension_sizes[6] * dimension_sizes[8] * dimension_sizes[9];
    const int64_t SIZE_B = dimension_sizes[2] * dimension_sizes[5] * dimension_sizes[7] * dimension_sizes[9];
    const int64_t SIZE_C = dimension_sizes[0] * dimension_sizes[4] * dimension_sizes[5] * dimension_sizes[6];
    const int64_t SIZE_D = dimension_sizes[1] * dimension_sizes[4] * dimension_sizes[7] * dimension_sizes[8];

    const int64_t SIZE_OUT = dimension_sizes[0] * dimension_sizes[1] * dimension_sizes[2] * dimension_sizes[3];

    float *tensor_A = new float[SIZE_A];
    float *tensor_B = new float[SIZE_B];
    float *tensor_C = new float[SIZE_C];
    float *tensor_D = new float[SIZE_D];

    tensor_inputs["3,6,8,9"] = tensor_A;
    tensor_inputs["2,5,7,9"] = tensor_B;
    tensor_inputs["0,4,5,6"] = tensor_C;
    tensor_inputs["1,4,7,8"] = tensor_D;

    // init matrices
    for (int64_t i = 0; i < SIZE_A; ++i)
    {
        tensor_A[i] = i;
    }
    for (int64_t i = 0; i < SIZE_B; ++i)
    {
        tensor_B[i] = i;
    }
    for (int64_t i = 0; i < SIZE_C; ++i)
    {
        tensor_C[i] = i;
    }
    for (int64_t i = 0; i < SIZE_D; ++i)
    {
        tensor_D[i] = i;
    }

    mini_jit::einsum::EinsumTree::execute(node, dimension_sizes, tensor_inputs, dtype);

    const float *tensor_out = static_cast<const float *>(node->tensor_out);

    // print output tensor
    // for (int64_t i = 0; i < SIZE_OUT; ++i)
    // {
    //     std::cout << tensor_out[i] << " ";
    // }
    // std::cout << std::endl;

    delete node;
    delete[] tensor_A;
    delete[] tensor_B;
    delete[] tensor_C;
    delete[] tensor_D;
}