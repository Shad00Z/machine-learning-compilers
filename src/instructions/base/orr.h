#ifndef MINI_JIT_INSTRUCTIONS_BASE_ORR_H
#define MINI_JIT_INSTRUCTIONS_BASE_ORR_H

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
             * @brief Generates an ORR (shifted register) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register.
             * @param reg_src2 second source register.
             * @param shift shift value.
             * @param amount amount to shift.
             *
             * @return instruction.
             **/
            constexpr uint32_t orr(gpr_t reg_dest,
                                   gpr_t reg_src1,
                                   gpr_t reg_src2,
                                   uint32_t shift,
                                   uint32_t amount)
            {
                uint32_t l_ins = 0x2a000000;

                // set sf
                uint32_t l_sf = reg_dest & 0x20;
                l_ins |= l_sf << 26;

                // set destination register id
                uint32_t l_reg_id = reg_dest & 0x1f;
                l_ins |= l_reg_id;

                // set first source register id
                l_reg_id = reg_src1 & 0x1f;
                l_ins |= l_reg_id << 5;

                // set amount to shift
                uint32_t l_amount = amount & 0x3f;
                l_ins |= l_amount << 10;

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
#endif // MINI_JIT_INSTRUCTIONS_BASE_ORR_H