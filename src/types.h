#ifndef MINI_JIT_TYPES_H
#define MINI_JIT_TYPES_H

#include <cstdint>

namespace mini_jit
{
    /// execution type
    enum class exec_t : uint32_t
    {
        seq = 0,
        prim = 1,
        shared = 2,
    };

    /// primitive type
    enum class ptype_t : uint32_t
    {
        zero = 0,
        identity = 1,
        relu = 2,
        gemm = 3,
        brgemm = 4,
        none = 99
    };

    /// dimension type
    enum class dim_t : uint32_t
    {
        c = 0,
        m = 1,
        n = 2,
        k = 3,
        undefined = 99
    };

    /// data type
    enum class dtype_t : uint32_t
    {
        fp32 = 0,
        fp64 = 1
    };

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
}
#endif // MINI_JIT_TYPES_H