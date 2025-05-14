#ifndef MINI_JIT_INSTRUCTIONS_BASE_MOV_H
#define MINI_JIT_INSTRUCTIONS_BASE_MOV_H

#include <cstdint>
#include <stdexcept>
#include "../../registers/gp_registers.h"
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
             * @brief Generates an MOV (register) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src source register.
             *
             * @return instruction.
             **/
            constexpr uint32_t mov(gpr_t reg_dest,
                                   gpr_t reg_src)
            {
                return orr(reg_dest,
                           gpr_t::wzr,
                           reg_src,
                           0x0,
                           0x0);
            }

            /**
             * @brief Generates an MOV 16-bit immediate instruction.
             *
             * @param reg_dest destination register.
             * @param imm16 16-bit unsigned immediate value.
             *
             * @return instruction.
             */
            constexpr uint32_t mov(gpr_t reg_dest,
                                   uint64_t imm16)
            {
                bool is64bit = (reg_dest & 0x20) != 0;

                // movz allows placing a 16-bit immediate at bit positions 0, 16, 32, or 48.
                for (int shift = 0; shift < (is64bit ? 64 : 32); shift += 16)
                {
                    // Check if the immediate fits entirely within one 16-bit field at the given shift.
                    // ~(0xFFFFULL << shift) creates a mask that zeros out the 16-bit field we're targeting,
                    // and leaves 1s elsewhere.
                    // If ANDing with this mask results in zero, it means the rest of the bits are zero.
                    if ((imm16 & ~(0xFFFFULL << shift)) == 0)
                    {
                        // Extract the 16-bit portion of the immediate that we want to encode
                        uint16_t immMasked = (imm16 >> shift) & 0xFFFF;
                        return movz(reg_dest, immMasked, shift);
                    }
                }

                // immediate value could not be encoded using a single MOVZ
                // need to implement MOVZ+MOVK support for larger immediates
                throw std::invalid_argument("Immediate too large for a single MOVZ");
                return 0;
            }

            /**
             * @brief Generates an MOV (from/to SP) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 source register.
             *
             * @return instruction.
             */
            constexpr uint32_t movSP(gpr_t reg_dest,
                                     gpr_t reg_src)
            {
                return add(reg_dest,
                           reg_src,
                           0,
                           0);
            }

        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_BASE_CBNZ_H