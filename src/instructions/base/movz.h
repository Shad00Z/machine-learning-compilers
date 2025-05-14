#ifndef MINI_JIT_INSTRUCTIONS_BASE_MOVZ_H
#define MINI_JIT_INSTRUCTIONS_BASE_MOVZ_H

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
             * @brief Generates an MOVZ instruction.
             *
             * @param reg_dest destination register.
             * @param imm16 16-bit unsigned immediate value.
             * @param shift amount by which to left shift the immediate value.
             */
            constexpr uint32_t movz(gpr_t reg_dest,
                                    uint16_t imm16,
                                    uint32_t shift)
            {
                uint32_t l_ins = 0x52800000;

                // set sf
                uint32_t l_sf = reg_dest & 0x20;
                l_ins |= l_sf << 26;

                // set destination register id
                uint32_t l_reg_id = reg_dest & 0x1f;
                l_ins |= l_reg_id;

                // set immediate value
                uint32_t l_imm = imm16 & 0xFFFF;
                l_ins |= l_imm << 5;

                // set shift value
                uint32_t l_shift = shift & 0x3;
                l_ins |= l_shift << 21;

                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_BASE_MOVZ_H