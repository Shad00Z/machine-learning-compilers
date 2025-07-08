#ifndef MINI_JIT_REGISTERS_GENERAL_PURPOSE_H
#define MINI_JIT_REGISTERS_GENERAL_PURPOSE_H

#include <cstdint>

namespace mini_jit
{
    namespace registers
    {
        //! general-purpose registers
        typedef enum : uint32_t
        {
            w0  = 0,
            w1  = 1,
            w2  = 2,
            w3  = 3,
            w4  = 4,
            w5  = 5,
            w6  = 6,
            w7  = 7,
            w8  = 8,
            w9  = 9,
            w10 = 10,
            w11 = 11,
            w12 = 12,
            w13 = 13,
            w14 = 14,
            w15 = 15,
            w16 = 16,
            w17 = 17,
            w18 = 18,
            w19 = 19,
            w20 = 20,
            w21 = 21,
            w22 = 22,
            w23 = 23,
            w24 = 24,
            w25 = 25,
            w26 = 26,
            w27 = 27,
            w28 = 28,
            w29 = 29,
            w30 = 30,

            x0  = 32 + 0,
            x1  = 32 + 1,
            x2  = 32 + 2,
            x3  = 32 + 3,
            x4  = 32 + 4,
            x5  = 32 + 5,
            x6  = 32 + 6,
            x7  = 32 + 7,
            x8  = 32 + 8,
            x9  = 32 + 9,
            x10 = 32 + 10,
            x11 = 32 + 11,
            x12 = 32 + 12,
            x13 = 32 + 13,
            x14 = 32 + 14,
            x15 = 32 + 15,
            x16 = 32 + 16,
            x17 = 32 + 17,
            x18 = 32 + 18,
            x19 = 32 + 19,
            x20 = 32 + 20,
            x21 = 32 + 21,
            x22 = 32 + 22,
            x23 = 32 + 23,
            x24 = 32 + 24,
            x25 = 32 + 25,
            x26 = 32 + 26,
            x27 = 32 + 27,
            x28 = 32 + 28,
            x29 = 32 + 29,
            x30 = 32 + 30,

            wzr = 31,
            xzr = 32 + 31,
            sp  = 64 + 32 + 31
        } gpr_t;
    } // namespace registers
} // namespace mini_jit

#endif // MINI_JIT_REGISTERS_GENERAL_PURPOSE_H