#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_STR_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_STR_H

#include <cstdint>
#include <stdexcept>
#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
using gpr_t = mini_jit::registers::gpr_t;
using simd_fp_t = mini_jit::registers::simd_fp_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace simd_fp
        {
            /**
             * @brief Generates an STR (12-bit immediate) instruction using unsigned offset encoding.
             *
             * @param reg_data register holding the data to be transferred.
             * @param reg_address register holding the memory address.
             * @param imm12 12-bit immediate value.
             * @param size_spec size specifier (s, d, q).
             */
            constexpr uint32_t str(simd_fp_t reg_data,
                                   gpr_t reg_address,
                                   uint32_t imm12,
                                   neon_size_spec_t size_spec)
            {
                uint32_t l_ins = 0x3D000000;

                // set size
                uint32_t l_size = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                                                                              : 0;
                l_ins |= (l_size & 0x3) << 30; // set bit 31, 30

                // set destination register id
                l_ins |= (reg_data & 0x1f);

                // set first source register id
                l_ins |= (reg_address & 0x1f) << 5;

                // check if immediate can be encoded
                uint32_t l_scale = (size_spec == neon_size_spec_t::s) ? 4 : (size_spec == neon_size_spec_t::d) ? 8
                                                                                                               : 16;
                if (imm12 % l_scale != 0)
                {
                    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit) or 16 (128-bit)");
                }

                // scale the immediate for encoding (right-shift)
                uint32_t l_scaleShift = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                                                                                    : 4;
                uint32_t l_imm = (imm12 >> l_scaleShift) & 0xfff;
                l_ins |= l_imm << 10;

                // set op code
                uint32_t l_opc = (size_spec == neon_size_spec_t::q) ? 2 : 0;
                l_ins |= l_opc << 22;

                return l_ins;
            }

            /**
             * @brief Generates an STR (9-bit immediate) instruction using post-index encoding.
             *
             * @param reg_data register holding the data to be transferred.
             * @param reg_address register holding the memory address.
             * @param imm9 9-bit immediate value.
             * @param size_spec size specifier (s, d, q).
             */
            constexpr uint32_t strPost(simd_fp_t reg_data,
                                       gpr_t reg_address,
                                       uint32_t imm9,
                                       neon_size_spec_t size_spec)
            {
                uint32_t l_ins = 0x3C000400;

                // set size
                uint32_t l_size = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                                                                              : 0;

                uint32_t l_sf = l_size & 0x3;
                l_ins |= l_sf << 30; // set bit 31, 30

                // set destination register id
                uint32_t l_reg_id = reg_data & 0x1f;
                l_ins |= l_reg_id;

                // set first source register id
                l_reg_id = reg_address & 0x1f;
                l_ins |= l_reg_id << 5;

                // check if immediate can be encoded
                uint32_t l_scale = (size_spec == neon_size_spec_t::s) ? 4 : (size_spec == neon_size_spec_t::d) ? 8
                                                                                                               : 16;
                if (imm9 % l_scale != 0)
                {
                    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit) or 16 (128-bit)");
                }

                uint32_t l_imm = imm9 & 0x1ff;
                l_ins |= l_imm << 12;

                // set op code
                uint32_t l_opc = (size_spec == neon_size_spec_t::q) ? 1 : 0;
                l_ins |= l_opc << 23;

                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_STR_H