#ifndef MINI_JIT_INSTRUCTIONS_BASE_MUL_H
#define MINI_JIT_INSTRUCTIONS_BASE_MUL_H

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
             * @brief Generates an MUL (register) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register.
             * @param reg_src2 second source register.
             *
             * @return instruction.
             */
            constexpr uint32_t mul(gpr_t reg_dest,
                                   gpr_t reg_src1,
                                   gpr_t reg_src2)
            {
                uint32_t l_ins = 0x1B007C00;

                uint32_t l_sf1 = reg_src1 & 0x20;
                uint32_t l_sf2 = reg_src2 & 0x20;
                uint32_t l_sf_dest = reg_dest & 0x20;
                if (l_sf1 != l_sf2)
                {
                    throw std::invalid_argument("MUL: both source registers must be of the same size");
                }
                else if (l_sf1 != l_sf_dest)
                {
                    throw std::invalid_argument("MUL: destination register must be of the same size as source registers");
                }

                // set size
                uint32_t l_sf = reg_dest & 0x20;
                l_ins |= l_sf << 26; // set bit 31

                // set destination register id
                uint32_t l_reg_id = reg_dest & 0x1f;
                l_ins |= l_reg_id;

                // set first source register id
                l_reg_id = reg_src1 & 0x1f;
                l_ins |= l_reg_id << 5;

                // set second source register id
                l_reg_id = reg_src2 & 0x1f;
                l_ins |= l_reg_id << 16;

                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_BASE_MUL_H