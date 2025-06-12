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
     * @brief Helper function to parse the einsum expression and create nodes.
     *
     * @param einsum_expression The string representation of the einsum operation.
     * @return The root EinsumNode representing the start of the parsed expression.
     */
    static EinsumNode *parse_einsum_expression(std::string const &einsum_expression,
                                               std::vector<int64_t> &dimension_sizes,
                                               mini_jit::dtype_t dtype,
                                               int64_t thread_target,
                                               int64_t max_kernel_size);

    static void execute(EinsumNode *root_node,
                        std::vector<int64_t> &dimension_sizes,
                        std::map<std::string, void const *> &tensor_inputs,
                        mini_jit::dtype_t dtype);

private:
    /**
     * @brief Helper function to parse the einsum expression and create nodes recursively.
     *
     * @param einsum_expression The string representation of the einsum operation.
     * @param dimension_sizes The array with the dimension sizes sorted by id.
     * @param parentNode The current parent node for potential children nodes.
     * @return A EinsumNode representing the parsed expression.
     */
    static EinsumNode *parse_einsum_expression_recursive(std::string const &einsum_expression);

    static std::vector<int64_t> get_dimensions_from_expression(std::string const &einsum_expression);

    /**
     * @brief Helper function to parse the einsum expression and create nodes recursively.
     *
     * @param einsum_node
     * @param dimension_sizes The array with the dimension sizes sorted by id.
     * @return A EinsumNode representing the parsed expression.
     */
    static void initialize_einsum_nodes(EinsumNode *einsum_node,
                                        std::vector<int64_t> &dimension_sizes,
                                        mini_jit::dtype_t dtype,
                                        int64_t thread_target,
                                        int64_t max_kernel_size);

    template <typename T>
    static bool contains(const std::vector<T> &vec, const T &value)
    {
        return std::find(vec.begin(), vec.end(), value) != vec.end();
    }

public:
    //! Deleted constructor to prevent instantiation of the static EinsumTree class.
    EinsumTree() = delete;
};

#endif // MINI_JIT_EINSUM_EINSUM_TREE_H