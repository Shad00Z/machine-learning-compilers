#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_INS_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_INS_H

#include <cstdint>
#include <stdexcept>
#include "registers/simd_fp_registers.h"
using simd_fp_t = mini_jit::registers::simd_fp_t;
using arr_spec_t = mini_jit::registers::arr_spec_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace simd_fp
        {
            /**
             * @brief Generates an INS (element) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src2 source register.
             * @param imm5 5-bit immediate (destination index).
             * @param imm4 4-bit immediate (source index).
             *
             * @return instruction.
             **/
            constexpr uint32_t ins(simd_fp_t reg_dest,
                                   simd_fp_t reg_src,
                                   uint32_t imm5,
                                   uint32_t imm4,
                                   neon_size_spec_t size_spec)
            {
                if (size_spec != neon_size_spec_t::s && 
                    size_spec != neon_size_spec_t::d)
                {
                    throw std::invalid_argument("Invalid size specifier");
                }

                uint32_t l_ins = 0x6E000400;

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                // set source register id
                l_ins |= (reg_src & 0x1f) << 5;

                // set indices
                if (size_spec == neon_size_spec_t::s)
                {
                    l_ins |= (1) << 18;

                    // destination index
                    if (imm5 != 0 && imm5 != 1 && imm5 != 2 && imm5 != 3)
                    {
                        throw std::invalid_argument("Invalid index for destination register (s)");
                    }

                    l_ins |= (imm5 & 0x3) << 19;

                    // src index
                    if (imm4 != 0 && imm4 != 1 && imm4 != 2 && imm4 != 3)
                    {
                        throw std::invalid_argument("Invalid index for source register (s)");
                    }

                    l_ins |= (imm4 & 0x3) << 13;
                }
                else
                {
                    l_ins |= (1) << 19;

                    // destination index
                    if (imm5 != 0 && imm5 != 1)
                    {
                        throw std::invalid_argument("Invalid index for destination register (d)");
                    }

                    l_ins |= (imm5 & 0x1) << 20;

                    // src index
                    if (imm4 != 0 && imm4 != 1)
                    {
                        throw std::invalid_argument("Invalid index for source register (s)");
                    }

                    l_ins |= (imm4 & 0x1) << 14;
                }

                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_INS_H
