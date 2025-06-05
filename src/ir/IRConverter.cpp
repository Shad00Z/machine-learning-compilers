#include "IRConverter.h"

void mini_jit::ir::IRConverter::convertConfigToDimensions(
    std::span<const dim_t> i_dim_types,
    std::span<const exec_t> i_exec_types,
    std::span<const int64_t> i_dim_sizes,
    std::span<const int64_t> i_strides_in0,
    std::span<const int64_t> i_strides_in1,
    std::span<const int64_t> i_strides_out,
    std::vector<Dimension> &o_dimensions)
{
    if (i_dim_types.size() != i_exec_types.size() ||
        i_dim_types.size() != i_dim_sizes.size() ||
        i_dim_types.size() != i_strides_in0.size() ||
        i_dim_types.size() != i_strides_in1.size() ||
        i_dim_types.size() != i_strides_out.size())
    {
        throw std::invalid_argument("All input spans must have the same size.");
    }

    o_dimensions.clear();
    o_dimensions.reserve(i_dim_types.size());
    for (size_t i = 0; i < i_dim_types.size(); ++i)
    {
        o_dimensions.emplace_back(i_dim_types[i], i_exec_types[i], i_dim_sizes[i],
                                  i_strides_in0[i], i_strides_in1[i], i_strides_out[i]);
    }
}

void mini_jit::ir::IRConverter::convertDimensionsToConfig(
    const std::vector<Dimension> &i_dimensions,
    std::vector<dim_t> &o_dim_types,
    std::vector<exec_t> &o_exec_types,
    std::vector<int64_t> &o_dim_sizes,
    std::vector<int64_t> &o_strides_in0,
    std::vector<int64_t> &o_strides_in1,
    std::vector<int64_t> &o_strides_out)
{
    const size_t n = i_dimensions.size();
    o_dim_types.clear();
    o_exec_types.clear();
    o_dim_sizes.clear();
    o_strides_in0.clear();
    o_strides_in1.clear();
    o_strides_out.clear();

    o_dim_types.reserve(n);
    o_exec_types.reserve(n);
    o_dim_sizes.reserve(n);
    o_strides_in0.reserve(n);
    o_strides_in1.reserve(n);
    o_strides_out.reserve(n);

    for (const auto &dim : i_dimensions)
    {
        o_dim_types.push_back(dim.type);
        o_exec_types.push_back(dim.exec_type);
        o_dim_sizes.push_back(dim.size);
        o_strides_in0.push_back(dim.stride_in0);
        o_strides_in1.push_back(dim.stride_in1);
        o_strides_out.push_back(dim.stride_out);
    }
}