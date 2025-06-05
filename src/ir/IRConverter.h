#ifndef MINI_JIT_IR_CONVERTER_H
#define MINI_JIT_IR_CONVERTER_H

#include "Dimension.h"
#include "types.h"
#include <vector>
#include <span>

namespace mini_jit
{
    namespace ir
    {
        class IRConverter;
    }
}

class mini_jit::ir::IRConverter
{
public:
    // Static class
    IRConverter() = delete;

    static void convertConfigToDimensions(std::span<const dim_t> i_dim_types,
                                          std::span<const exec_t> i_exec_types,
                                          std::span<const int64_t> i_dim_sizes,
                                          std::span<const int64_t> i_strides_in0,
                                          std::span<const int64_t> i_strides_in1,
                                          std::span<const int64_t> i_strides_out,
                                          std::vector<Dimension> &o_dimensions);

    static void convertDimensionsToConfig(const std::vector<Dimension> &i_dimensions,
                                          std::vector<dim_t> &o_dim_types,
                                          std::vector<exec_t> &o_exec_types,
                                          std::vector<int64_t> &o_dim_sizes,
                                          std::vector<int64_t> &o_strides_in0,
                                          std::vector<int64_t> &o_strides_in1,
                                          std::vector<int64_t> &o_strides_out);
};

#endif