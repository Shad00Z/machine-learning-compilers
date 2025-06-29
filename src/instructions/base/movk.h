#ifndef MINI_JIT_INSTRUCTIONS_BASE_MOVK_H
#define MINI_JIT_INSTRUCTIONS_BASE_MOVK_H

#include <cstdint>
#include <stdexcept>
#include "registers/gp_registers.h"
#include "orr.h"
#include "movz.h"
#include "add.h"
using gpr_t = mini_jit::registers::gpr_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace base
        {
            /**
             * @brief Generates an MOVK instruction.
             *
             * @param reg_dest destination register.
             * @param imm16 16-bit unsigned immediate value.
             * @param shift amount by which to left shift the immediate value.
             */
            constexpr uint32_t movk(gpr_t reg_dest,
                                    uint16_t imm16,
                                    uint32_t shift)
            {
                uint32_t l_ins = 0x72800000;

                // set sf
                l_ins |= (reg_dest & 0x20) << 26;

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                // set immediate value
                l_ins |= (imm16 & 0xFFFF) << 5;

                // w
                if ( reg_dest < 32 )
                {
                    if (shift != 0 && shift != 16)
                    {
                        throw std::invalid_argument("MOVK: invalid shift for w");
                    }  
                }
                // x
                else
                {
                    if (shift != 0 && shift != 16 && shift != 32 && shift != 48)
                    {
                        throw std::invalid_argument("MOVK: invalid shift for x");
                    }   
                }

                if (shift == 16)
                {
                    l_ins |= (0x1) << 21;
                }
                else if (shift == 32)
                {
                    l_ins |= (0x1) << 22;
                }
                else if (shift == 48)
                {
                    l_ins |= (0x3) << 21;
                }

                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_BASE_MOVK_H