#ifndef MINI_JIT_INSTRUCTIONS_BASE_SUB_H
#define MINI_JIT_INSTRUCTIONS_BASE_SUB_H

#include <cstdint>
#include "registers/gp_registers.h"
using gpr_t = mini_jit::registers::gpr_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace base
        {
            /**
             * @brief Generates an SUB (immediate) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register.
             * @param imm12 12-bit immediate value.
             * @param shift shift value.
             *
             * @return instruction.
             */
            constexpr uint32_t sub(gpr_t reg_dest,
                                   gpr_t reg_src,
                                   uint32_t imm12,
                                   uint32_t shift)
            {
                uint32_t l_ins = 0x51000000;

                // set size
                l_ins |= (reg_dest & 0x20) << 26; // set bit 31
                // set destination register id
                l_ins |= (reg_dest & 0x1f);
                // set first source register id
                l_ins |= (reg_src & 0x1f) << 5;
                // set immediate value
                l_ins |= (imm12 & 0xfff) << 10;
                // set shift value
                l_ins |= (shift & 0x1) << 22;

                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_BASE_SUB_H