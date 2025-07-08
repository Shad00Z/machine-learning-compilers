#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_UMOV_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_UMOV_H

#include <cstdint>
#include <mlc/registers/gp_registers.h>
#include <mlc/registers/simd_fp_registers.h>
#include <stdexcept>
using gpr_t            = mini_jit::registers::gpr_t;
using simd_fp_t        = mini_jit::registers::simd_fp_t;
using arr_spec_t       = mini_jit::registers::arr_spec_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace simd_fp
        {
            /**
             * @brief Generates an UMOV instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src source register.
             * @param imm5 index in range 0-3.
             * @param size_spec size specifier.
             *
             * @return instruction.
             **/
            constexpr uint32_t umov(gpr_t            reg_dest,
                                    simd_fp_t        reg_src,
                                    uint32_t         imm5,
                                    neon_size_spec_t size_spec)
            {
                if (size_spec != neon_size_spec_t::s &&
                    size_spec != neon_size_spec_t::d)
                {
                    throw std::invalid_argument("Invalid size specifier");
                }

                uint32_t l_ins = 0xE003C00;

                // set first source register id
                l_ins |= (reg_dest & 0x1f);

                // set first source register id
                l_ins |= (reg_src & 0x1f) << 5;

                // w
                if (reg_dest < 32)
                {
                    // set size specifier
                    l_ins |= (1) << 18;

                    if (imm5 != 0 && imm5 != 1 && imm5 != 2 && imm5 != 3)
                    {
                        throw std::invalid_argument("Invalid index for w");
                    }

                    // set index
                    l_ins |= (imm5 & 0x1f) << 19;
                }
                // x
                else
                {
                    // set bit 30
                    l_ins |= (1) << 30;

                    if (imm5 != 0 && imm5 != 1)
                    {
                        throw std::invalid_argument("Invalid index for x");
                    }

                    // set index
                    l_ins |= (imm5 & 0x1f) << 20;

                    // set size specifier
                    l_ins |= (1) << 19;
                }

                return l_ins;
            }
        } // namespace simd_fp
    } // namespace instructions
} // namespace mini_jit

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_UMOV_H
