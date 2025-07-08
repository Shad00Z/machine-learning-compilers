#ifndef MINI_JIT_IR_CONVERTER_H
#define MINI_JIT_IR_CONVERTER_H

#include <cstdint>
#include <mlc/ir/Dimension.h>
#include <mlc/types.h>
#include <span>
#include <stdexcept>
#include <vector>

namespace mini_jit
{
    namespace ir
    {
        class IRConverter;
    }
} // namespace mini_jit

/**
 * @brief The IRConverter class provides methods to convert between configuration
 * parameters and IR Dimension representation.
 */
class mini_jit::ir::IRConverter
{
public:
    //! Deleted constructor to prevent instantiation of the static IRConverter class.
    IRConverter() = delete;

    /**
     * @brief Convert configuration parameters to a vector of Dimension objects.
     *
     * @param i_dim_types A span of dimension types (M, N, K).
     * @param i_exec_types A span of execution types (Prim, Seq, Shared).
     * @param i_dim_sizes A span of dimension sizes.
     * @param i_strides_in0 A span of strides for the first input tensor.
     * @param i_strides_in1 A span of strides for the second input tensor.
     * @param i_strides_out A span of strides for the output tensor.
     * @param o_dimensions A vector to store the converted Dimension objects.
     */
    static void convertConfigToDimensions(std::span<const dim_t>   i_dim_types,
                                          std::span<const exec_t>  i_exec_types,
                                          std::span<const int64_t> i_dim_sizes,
                                          std::span<const int64_t> i_strides_in0,
                                          std::span<const int64_t> i_strides_in1,
                                          std::span<const int64_t> i_strides_out,
                                          std::vector<Dimension>&  o_dimensions);

    /**
     * @brief Convert a vector of Dimension objects to configuration parameters.
     *
     * @param i_dimensions A vector of Dimension objects to be converted.
     * @param o_dim_types A vector to store the dimension types.
     * @param o_exec_types A vector to store the execution types.
     * @param o_dim_sizes A vector to store the dimension sizes.
     * @param o_strides_in0 A vector to store the strides for the first input tensor.
     * @param o_strides_in1 A vector to store the strides for the second input tensor.
     * @param o_strides_out A vector to store the strides for the output tensor.
     */
    static void convertDimensionsToConfig(const std::vector<Dimension>& i_dimensions,
                                          std::vector<dim_t>&           o_dim_types,
                                          std::vector<exec_t>&          o_exec_types,
                                          std::vector<int64_t>&         o_dim_sizes,
                                          std::vector<int64_t>&         o_strides_in0,
                                          std::vector<int64_t>&         o_strides_in1,
                                          std::vector<int64_t>&         o_strides_out);
};

#endif