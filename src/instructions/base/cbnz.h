#ifndef MINI_JIT_INSTRUCTIONS_BASE_CBNZ_H
#define MINI_JIT_INSTRUCTIONS_BASE_CBNZ_H

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
             * @brief Generates a CBNZ instruction.
             *
             * @param reg general-purpose register.
             * @param imm19 immediate value (not the offset bytes!).
             *
             * @return instruction.
             **/
            constexpr uint32_t cbnz(gpr_t reg,
                                    int32_t imm19)
            {
                uint32_t l_ins = 0;
                uint32_t l_sf = reg & 0x20;

                l_ins |= l_sf << 26; // set bit 31
                l_ins |= 0b0110101 << 24;
                l_ins |= ((imm19 >> 2) & 0x510FFFFF) << 5;
                l_ins |= reg & 0x1F;

                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_BASE_CBNZ_H