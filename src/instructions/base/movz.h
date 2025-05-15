#ifndef MINI_JIT_INSTRUCTIONS_BASE_MOVZ_H
#define MINI_JIT_INSTRUCTIONS_BASE_MOVZ_H

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
                l_ins |= (reg_dest & 0x20) << 26;
                // set destination register id
                l_ins |= (reg_dest & 0x1f);
                // set immediate value
                l_ins |= (imm16 & 0xFFFF) << 5;
                // set shift value
                l_ins |= (shift & 0x3) << 21;

                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_BASE_MOVZ_H