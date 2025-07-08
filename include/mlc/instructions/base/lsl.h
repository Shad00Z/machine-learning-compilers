#ifndef MINI_JIT_INSTRUCTIONS_BASE_LSL_H
#define MINI_JIT_INSTRUCTIONS_BASE_LSL_H

#include <cstdint>
#include <mlc/registers/gp_registers.h>
#include <stdexcept>
using gpr_t = mini_jit::registers::gpr_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace base
        {
            /**
             * @brief Generates a base LSL (immediate) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src source register.
             * @param imm12 immediate value.
             */
            constexpr uint32_t lsl(gpr_t    reg_dest,
                                   gpr_t    reg_src,
                                   uint32_t imm)
            {
                uint32_t l_ins = 0x53000000;

                bool is_64bit = (reg_dest & 0x20) != 0;

                // Validate immediate
                if (imm >= (is_64bit ? 64 : 32))
                {
                    throw std::invalid_argument("Shift amount out of range");
                }

                // Calculate immr and imms
                uint32_t immr = (is_64bit ? (64 - imm) % 64 : (32 - imm) % 32);
                uint32_t imms = (is_64bit ? 63 - imm : 31 - imm);

                // Set sf (bit 31) and N (bit 22) for 64-bit op
                if (is_64bit)
                {
                    l_ins |= (1 << 31); // Set sf to 1
                    l_ins |= (1 << 22); // Set N to 1
                }

                // Set immr and imms
                l_ins |= (immr & 0x3F) << 16;
                l_ins |= (imms & 0x3F) << 10;

                // destination register
                l_ins |= (reg_dest & 0x1F);

                // source register
                l_ins |= (reg_src & 0x1F) << 5;

                return l_ins;
            }
        } // namespace base
    } // namespace instructions
} // namespace mini_jit

#endif // MINI_JIT_INSTRUCTIONS_BASE_LSL_H