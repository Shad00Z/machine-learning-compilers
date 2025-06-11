#include <catch2/catch.hpp>
#include "EinsumTree.h"
#include <vector>

TEST_CASE("huhn")
{
    std::string input = "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]";
    std::vector<int64_t> dimension_sizes{60, 60, 20, 20, 8, 8, 8, 8, 8, 8};
    mini_jit::einsum::EinsumTree::parse_einsum_expression(input, dimension_sizes);
}