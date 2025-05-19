#ifndef MINI_JIT_INSTRUCTIONS_BASE_STR_H
#define MINI_JIT_INSTRUCTIONS_BASE_STR_H

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
             * @brief Generates an STR (12-bit immediate) instruction using unsigned offset encoding.
             *
             * @param reg_data register holding the data to be transferred.
             * @param reg_address register holding the memory address.
             * @param imm12 12-bit immediate value.
             */
            constexpr uint32_t str(gpr_t reg_data,
                                   gpr_t reg_address,
                                   uint32_t imm12)
            {
                uint32_t l_ins = 0xB9000000;

                // set size
                uint32_t l_sf = reg_data & 0x20;
                l_ins |= l_sf << 25; // set bit 30
                // set destination register id
                l_ins |= (reg_data & 0x1f);
                // set first source register id
                l_ins |= (reg_address & 0x1f) << 5;
                // check if immediate can be encoded
                uint32_t l_scale = (l_sf) ? 8 : 4;
                if (imm12 % l_scale != 0)
                {
                    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)");
                }

                // scale the immediate for encoding (right-shift)
                uint32_t l_scaleShift = (l_sf) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
                uint32_t l_imm = (imm12 >> l_scaleShift) & 0xFFF;

                // set 12 bit immediate value
                l_ins |= l_imm << 10;
                return l_ins;
            }

            /**
             * @brief Generates an STR (9-bit immediate) instruction using post-index encoding.
             *
             * @param reg_data register holding the data to be transferred.
             * @param reg_address register holding the memory address.
             * @param imm9 signed 9-bit immediate value.
             */
            constexpr uint32_t strPost(gpr_t reg_data,
                                       gpr_t reg_address,
                                       uint32_t imm9)
            {
                uint32_t l_ins = 0xB8000400;

                // set size
                uint32_t l_sf = reg_data & 0x20;
                l_ins |= l_sf << 25; // set bit 30
                // set destination register id
                l_ins |= (reg_data & 0x1f);
                // set first source register id
                l_ins |= (reg_address & 0x1f) << 5;
            
                // set 9 bit immediate value
                l_ins |= (imm9 & 0x1FF) << 12;
                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_BASE_STR_H