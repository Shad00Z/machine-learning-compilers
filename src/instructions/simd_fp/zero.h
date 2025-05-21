#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_ZERO_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_ZERO_H

#include <cstdint>
#include <stdexcept>
#include "registers/simd_fp_registers.h"
#include "eor.h"
using simd_fp_t = mini_jit::registers::simd_fp_t;
using arr_spec_t = mini_jit::registers::arr_spec_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace simd_fp
        {
            /**
             * @brief Generates an EOR instruction that zeros out the given register.
             * 8B zeroes out the lower half (64 bit) of the register, while 16B zeroes out the whole register (128 bit).
             *
             * @param reg register to zero out.
             * @param arr_spec arrangement specifier (8B or 16B).
             */
            constexpr uint32_t zero(simd_fp_t reg,
                                    arr_spec_t arr_spec)
            {
                return eor(reg, reg, reg, arr_spec);
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_ZERO_H
