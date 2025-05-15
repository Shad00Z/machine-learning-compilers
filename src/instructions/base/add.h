#ifndef MINI_JIT_INSTRUCTIONS_BASE_ADD_H
#define MINI_JIT_INSTRUCTIONS_BASE_ADD_H

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
             * @brief Generates an ADD (immediate) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 source register.
             * @param imm12 12-bit immediate value.
             * @param shift shift value.
             *
             * @return instruction.
             */
            constexpr uint32_t add(gpr_t reg_dest,
                                gpr_t reg_src,
                                uint32_t imm12,
                                uint32_t shift)
            {
                uint32_t l_ins = 0x11000000;

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
                uint32_t l_shift = shift & 0x1;
                l_ins |= l_shift << 22;

                return l_ins;
            }

            /**
             * @brief Generates an ADD (shifted register) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register.
             * @param reg_src2 second source register.
             * @param imm6 6-bit immediate value.
             * @param shift shift value.
             *
             * @return instruction.
             */
            constexpr uint32_t add(gpr_t reg_dest,
                                gpr_t reg_src1,
                                gpr_t reg_src2,
                                uint32_t imm6,
                                uint32_t shift)
            {
                uint32_t l_ins = 0xB000000;

                uint32_t l_sf1 = reg_src1 & 0x20;
                uint32_t l_sf2 = reg_src2 & 0x20;
                uint32_t l_sf_dest = reg_dest & 0x20;
                if (l_sf1 != l_sf2)
                {
                    throw std::invalid_argument("ADD: both source registers must be of the same size");
                }
                else if (l_sf1 != l_sf_dest)
                {
                    throw std::invalid_argument("ADD: destination register must be of the same size as source registers");
                }

                // set size
                uint32_t l_sf = reg_dest & 0x20;
                l_ins |= l_sf << 26; // set bit 31

                // set immediate value
                uint32_t l_imm = imm6 & 0x3f;
                l_ins |= l_imm << 10;

                // set destination register id
                uint32_t l_reg_id = reg_dest & 0x1f;
                l_ins |= l_reg_id;

                // set first source register id
                l_reg_id = reg_src1 & 0x1f;
                l_ins |= l_reg_id << 5;

                // set second source register id
                l_reg_id = reg_src2 & 0x1f;
                l_ins |= l_reg_id << 16;

                // set shift value
                uint32_t l_shift = shift & 0x3;
                l_ins |= l_shift << 22;

                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_BASE_ADD_H