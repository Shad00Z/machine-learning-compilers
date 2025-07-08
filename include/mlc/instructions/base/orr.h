#ifndef MINI_JIT_INSTRUCTIONS_BASE_ORR_H
#define MINI_JIT_INSTRUCTIONS_BASE_ORR_H

#include <cstdint>
#include <mlc/registers/gp_registers.h>
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
            constexpr uint32_t orr(gpr_t    reg_dest,
                                   gpr_t    reg_src1,
                                   gpr_t    reg_src2,
                                   uint32_t shift,
                                   uint32_t amount)
            {
                uint32_t l_ins = 0x2a000000;

                // set sf
                l_ins |= (reg_dest & 0x20) << 26;
                // set destination register id
                l_ins |= (reg_dest & 0x1f);
                // set first source register id
                l_ins |= (reg_src1 & 0x1f) << 5;
                // set amount to shift
                l_ins |= (amount & 0x3f) << 10;
                // set second source register id
                l_ins |= (reg_src2 & 0x1f) << 16;
                // set shift value
                l_ins |= (shift & 0x3) << 22;

                return l_ins;
            }
        } // namespace base
    } // namespace instructions
} // namespace mini_jit
#endif // MINI_JIT_INSTRUCTIONS_BASE_ORR_H