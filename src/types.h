#ifndef MINI_JIT_TYPES_H
#define MINI_JIT_TYPES_H

#include <cstdint>
#include <string>

namespace mini_jit
{
    /// execution type
    enum class exec_t : uint32_t
    {
        seq = 0,
        prim = 1,
        shared = 2,
        undefined = 99
    };

    inline const std::string to_string(exec_t e) {
        switch (e) {
            case exec_t::seq: return "seq";
            case exec_t::prim: return "prim";
            case exec_t::shared: return "shared";
            case exec_t::undefined: return "undefined";
            default: return "unknown";
        }
    };

    /// primitive type
    enum class ptype_t : uint32_t
    {
        zero = 0,
        identity = 1,
        relu = 2,
        gemm = 3,
        brgemm = 4,
        square = 5,
        reciprocal = 6,
        add = 7,
        sub = 8,
        mul = 9,
        div = 10,
        min = 11,
        max = 12,
        none = 99
    };

    inline const std::string to_string(ptype_t p) {
        switch (p) {
            case ptype_t::zero: return "zero";
            case ptype_t::identity: return "identity";
            case ptype_t::relu: return "relu";
            case ptype_t::gemm: return "gemm";
            case ptype_t::brgemm: return "brgemm";
            case ptype_t::square: return "square";
            case ptype_t::reciprocal: return "reciprocal";
            case ptype_t::add: return "add";
            case ptype_t::sub: return "sub";
            case ptype_t::mul: return "mul";
            case ptype_t::div: return "div";
            case ptype_t::min: return "min";
            case ptype_t::max: return "max";
            case ptype_t::none: return "none";
            default: return "unknown";
        }
    }

    /// dimension type
    enum class dim_t : uint32_t
    {
        c = 0,
        m = 1,
        n = 2,
        k = 3,
        undefined = 99
    };

    inline const std::string to_string(dim_t d) {
        switch (d) {
            case dim_t::c: return "c";
            case dim_t::m: return "m";
            case dim_t::n: return "n";
            case dim_t::k: return "k";
            case dim_t::undefined: return "undefined";
            default: return "unknown";
        }
    }

    /// data type
    enum class dtype_t : uint32_t
    {
        fp32 = 0,
        fp64 = 1
    };

    inline const std::string to_string(dtype_t d) {
        switch (d) {
            case dtype_t::fp32: return "fp32";
            case dtype_t::fp64: return "fp64";
            default: return "unknown";
        }
    }

    /// error codes
    enum class error_t : int32_t
    {
        success = 0,
        wrong_dimension = 1,
        wrong_ptype = 3,
        operation_not_supported = 4,
        wrong_matrix_ordering_format = 5,
        wrong_dtype = 6,
        wrong_exec_type = 7,
    };

    inline const std::string to_string(error_t e) {
        switch (e) {
            case error_t::success: return "success";
            case error_t::wrong_dimension: return "wrong_dimension";
            case error_t::wrong_ptype: return "wrong_ptype";
            case error_t::operation_not_supported: return "operation_not_supported";
            case error_t::wrong_matrix_ordering_format: return "wrong_matrix_ordering_format";
            case error_t::wrong_dtype: return "wrong_dtype";
            case error_t::wrong_exec_type: return "wrong_exec_type";
            default: return "unknown";
        }
    }
}
#endif // MINI_JIT_TYPES_H