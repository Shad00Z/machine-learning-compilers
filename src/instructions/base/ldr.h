#ifndef MINI_JIT_INSTRUCTIONS_BASE_LDR_H
#define MINI_JIT_INSTRUCTIONS_BASE_LDR_H

#include <cstdint>
#include <stdexcept>
#include "registers/gp_registers.h"
using gpr_t = mini_jit::registers::gpr_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace base
        {
            /**
             * @brief Generates a base LDR (12-bit immediate) instruction using unsigned offset encoding.
             *
             * @param reg_dest destination register.
             * @param reg_src source register (base address).
             * @param imm12 12-bit immediate value.
             */
            constexpr uint32_t ldr(gpr_t reg_dest,
                                   gpr_t reg_src,
                                   uint32_t imm)
            {
                uint32_t l_ins = 0xB9400000;

                // set size
                uint32_t l_sf = reg_dest & 0x20;
                l_ins |= l_sf << 25; // set bit 30

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                // set first source register id
                l_ins |= (reg_src & 0x1f) << 5;

                // check if immediate can be encoded
                uint32_t scale = (l_sf) ? 8 : 4;
                if (imm % scale != 0)
                {
                    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)");
                }

                // scale the immediate for encoding (right-shift)
                uint32_t scaleShift = (l_sf) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
                uint32_t l_imm = (imm >> scaleShift) & 0xFFF;

                // set 12 bit immediate value
                l_ins |= l_imm << 10;
                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_BASE_LDR_H