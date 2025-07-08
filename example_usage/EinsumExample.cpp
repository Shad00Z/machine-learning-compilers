#include <iostream>
#include <map>
#include <mlc/constants.h>
#include <mlc/einsum/EinsumNode.h>
#include <mlc/einsum/EinsumTree.h>
#include <mlc/types.h>
#include <vector>

int main()
{
    // This example demonstrates how to use the EinsumTree to perform tensor operations
    std::cout << "Running the Einsum example" << std::endl;

    // First, we need to specify the einsum expression and the dimensions involved
    std::string          input = "[2,0],[1,2]->[1,0]";
    std::vector<int64_t> dimension_sizes{4,
                                         4,
                                         4};
    mini_jit::dtype_t    dtype = mini_jit::dtype_t::fp32;

    mini_jit::einsum::EinsumNode* node = mini_jit::einsum::EinsumTree::parse_einsum_expression(input,
                                                                                               dimension_sizes);

    // The to_string function provides a human-readable representation of the EinsumTree
    // beginning by the root node, which should be the same as the input string.
    std::cout << "Parsed Einsum Tree:" << std::endl;
    std::cout << mini_jit::einsum::EinsumTree::to_string(node) << std::endl;

    // Next, we can optimize the einsum nodes. This step is optional but can improve performance.
    mini_jit::einsum::EinsumTree::optimize_einsum_nodes(node,
                                                        256,
                                                        512,
                                                        16);

    // In order to make the einsum tree executable on hardware, we need to lower the einsum nodes
    // to tensor operations.
    mini_jit::einsum::EinsumTree::lower_einsum_nodes_to_tensor_operations(node,
                                                                          dimension_sizes,
                                                                          dtype);

    // The einsum tree is now ready for execution. We need to provide the input tensors.
    // The input tensors are specified as a map, where the keys are the dimension IDs
    // in the einsum expression, and the values are pointers to the actual data.
    // In this example, we will create two input tensors:
    // [0,1] -> *A
    // [1,2] -> *B
    std::map<std::string, void const*> tensor_inputs;

    const int64_t M = dimension_sizes[0];
    const int64_t N = dimension_sizes[1];
    const int64_t K = dimension_sizes[2];

    const int64_t SIZE_A   = M * K;
    const int64_t SIZE_B   = K * N;
    const int64_t SIZE_OUT = M * N;

    float* tensor_A = new float[SIZE_A];
    float* tensor_B = new float[SIZE_B];

    tensor_inputs["2,0"] = tensor_A;
    tensor_inputs["1,2"] = tensor_B;

    // We initialize the input tensors with some data.
    // In a real application, you would fill these tensors with meaningful data.
    for (int64_t i = 0; i < SIZE_A; ++i)
    {
        tensor_A[i] = i * 3.1f;
    }
    for (int64_t i = 0; i < SIZE_B; ++i)
    {
        tensor_B[i] = i * 0.5f;
    }

    // Now we can execute the einsum operation.
    // You may call the execute function as often as you like, even with different inputs.
    // The intermediate results of the einsum expression will be overwritten in each call.
    // Furthermore, the output tensor will be allocated automatically by the einsum tree.
    mini_jit::einsum::EinsumTree::execute(node,
                                          dimension_sizes,
                                          tensor_inputs);

    // The output tensor is now stored in the node. You can access it via the m_tensor_out member.
    float* output_tensor = static_cast<float*>(node->m_tensor_out);

    // The size of the output tensor is stored in m_tensor_size.
    std::cout << "Output tensor size: " << node->m_tensor_size << std::endl;

    // Lastly, the einsum tree needs to be deleted by the user.
    // It suffices to delete the root node, as the destructor will recursively delete all child nodes
    // and all allocated intermediate tensors, as well as the output tensor.
    delete node;
    // Regarding the input tensors, the user is responsible for deleting them.
    delete[] tensor_A;
    delete[] tensor_B;

    std::cout << "---------------------------------" << std::endl;
    return 0;
}