#ifndef MINI_JIT_EINSUM_EINSUM_TREE_H
#define MINI_JIT_EINSUM_EINSUM_TREE_H

#include "EinsumNode.h"
#include <string>
#include <vector>
#include <algorithm>
#include <ranges>

namespace mini_jit
{
    namespace einsum
    {
        class EinsumTree;
    }
}

/**
 * @brief The EinsumTree class provides methods to transform a string einsum expression
 * into a tree structure based on nodes and tensor operations.
 *
 */
class mini_jit::einsum::EinsumTree
{
public:
    /**
     * @brief Parses the einsum expression and creates an einsum tree.
     * In case the dimensions are not in an optimal order, permutation nodes will be inserted.
     *
     * @param einsum_expression The string representation of the einsum operation.
     * @param dimension_sizes A vector to store the sizes of the dimensions used in the expression.
     * @return The root EinsumNode representing the output of the parsed expression.
     */
    static EinsumNode *parse_einsum_expression(std::string const &einsum_expression,
                                               std::vector<int64_t> &dimension_sizes);

    /**
     * @brief Executes the einsum tree using tensor operations.
     *
     * @param root_node The root EinsumNode of the einsum tree.
     * @param dimension_sizes A vector containing the sizes of the dimensions used in the expression.
     * @param tensor_inputs A map containing the input tensors for the einsum operation.
     */
    static void execute(EinsumNode *root_node,
                        std::vector<int64_t> &dimension_sizes,
                        std::map<std::string, void const *> &tensor_inputs);

private:
    /**
     * @brief Helper function to parse the einsum expression and create an einsum tree recursively.
     *
     * @param einsum_expression The string representation of the einsum operation.
     * @return A EinsumNode representing the parsed expression.
     */
    static EinsumNode *parse_einsum_expression_recursive(std::string const &einsum_expression);

    /**
     * @brief Helper function to extract dimension IDs from the einsum expression.
     *
     * @param einsum_expression The string representation of the einsum operation.
     * @return A vector of dimension IDs extracted from the expression.
     */
    static std::vector<int64_t> get_dimensions_from_expression(std::string const &einsum_expression);

    /**
     * @brief Helper function to initialize the dimensions, sizes and strides of the given
     * einsum tree.
     *
     * @param root_node The root node of the einsum tree.
     * @param dimension_sizes The array with the dimension sizes sorted by id.
     * @return A EinsumNode representing the parsed expression.
     */
    static void initialize_einsum_nodes(EinsumNode *root_node,
                                        std::vector<int64_t> &dimension_sizes);

    /**
     * @brief Reorders the dimensions inside the nodes of the given einsum tree 
     * to ensure correct dimension positions for performant execution.
     * If this function is applied, no swapping of nodes is necessary.
     *
     * @param root_node The root node of the einsum tree.
     */
    static void reorder_node_dimensions(EinsumNode *root_node);

    /**
     * @brief Optimize the einsum tree by swapping nodes.
     *
     * @param root_node The root node of the einsum tree.
     */
    static void swap_nodes(EinsumNode *root_node);

    /**
     * @brief Helper function to check if a value is contained in a vector.
     *
     * @tparam T The type of the elements in the vector.
     * @param vec The vector to search in.
     * @param value The value to search for.
     * @return True if the value is found in the vector, false otherwise.
     */
    template <typename T>
    static bool contains(const std::vector<T> &vec, const T &value)
    {
        return std::find(vec.begin(), vec.end(), value) != vec.end();
    }

public:
    //! Deleted constructor to prevent instantiation of the static EinsumTree class.
    EinsumTree() = delete;

    /**
     * @brief Optimize the given einsum tree by applying transformations to the
     * dimensions, sizes and strides.
     *
     * @param root_node The root node of the einsum tree.
     * @param thread_target The target number of threads for parallel execution.
     * @param max_kernel_size The maximum size of the kernel to be used.
     * @param min_kernel_size The minimum size of the kernel to be used.
     */
    static void optimize_einsum_nodes(EinsumNode *root_node,
                                      int64_t thread_target,
                                      int64_t max_kernel_size,
                                      int64_t min_kernel_size);

    /**
     * @brief Lower the given einsum tree to executable tensor operations.
     *
     * @param root_node The root node of the einsum tree.
     * @param dimension_sizes The array with the dimension sizes sorted by id.
     * @param dtype The data type of the tensor.
     */
    static void lower_einsum_nodes_to_tensor_operations(EinsumNode *root_node,
                                                        std::vector<int64_t> &dimension_sizes,
                                                        mini_jit::dtype_t dtype);

    /**
     * @brief Convert the einsum tree to a string representation.
     * @param root_node The root node of the einsum tree.
     * @return A string representation of the einsum tree.
     */
    static std::string to_string(EinsumNode *root_node);
    
};

#endif // MINI_JIT_EINSUM_EINSUM_TREE_H