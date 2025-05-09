#ifndef MINI_JIT_REGISTERS_SIMD_H
#define MINI_JIT_REGISTERS_SIMD_H

#include <cstdint>

namespace mini_jit
{
    namespace registers
    {
        //! simd&fp registers
        typedef enum : uint32_t
        {
            v0 = 0,
            v1 = 1,
            v2 = 2,
            v3 = 3,
            v4 = 4,
            v5 = 5,
            v6 = 6,
            v7 = 7,
            v8 = 8,
            v9 = 9,
            v10 = 10,
            v11 = 11,
            v12 = 12,
            v13 = 13,
            v14 = 14,
            v15 = 15,
            v16 = 16,
            v17 = 17,
            v18 = 18,
            v19 = 19,
            v20 = 20,
            v21 = 21,
            v22 = 22,
            v23 = 23,
            v24 = 24,
            v25 = 25,
            v26 = 26,
            v27 = 27,
            v28 = 28,
            v29 = 29,
            v30 = 30,
            v31 = 31
        } simd_fp_t;

        //! neon arrangement specifiers
        typedef enum : uint32_t
        {
            s2 = 0x0,
            s4 = 0x40000000,
            d2 = 0x40400000
        } arr_spec_t;

        //! neon size specifiers
        typedef enum : uint32_t
        {
            s = 0x0,
            d = 0x1,
            q = 0x2
        } neon_size_spec_t;
    }
}

#endif // MINI_JIT_REGISTERS_SIMD_H