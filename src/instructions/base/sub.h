#ifndef MINI_JIT_INSTRUCTIONS_BASE_SUB_H
#define MINI_JIT_INSTRUCTIONS_BASE_SUB_H

#include <cstdint>
#include "../../registers/gp_registers.h"
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
                uint32_t l_sf = reg_dest & 0x20;
                l_ins |= l_sf << 26; // set bit 31

                // set destination register id
                uint32_t l_reg_id = reg_dest & 0x1f;
                l_ins |= l_reg_id;

                // set first source register id
                l_reg_id = reg_src & 0x1f;
                l_ins |= l_reg_id << 5;

                // set immediate value
                uint32_t l_imm = imm12 & 0xfff;
                l_ins |= l_imm << 10;

                // set shift value
                uint32_t l_cond = shift & 0x1;
                l_ins |= l_cond << 22;

                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_BASE_SUB_H