#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_LDR_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_LDR_H

#include <cstdint>
#include <mlc/registers/gp_registers.h>
#include <mlc/registers/simd_fp_registers.h>
#include <stdexcept>
using gpr_t            = mini_jit::registers::gpr_t;
using simd_fp_t        = mini_jit::registers::simd_fp_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace simd_fp
        {
            /**
             * @brief Generates an LDR (12-bit immediate) instruction using unsigned offset encoding.
             *
             * @param reg_dest destination register.
             * @param reg_src source register (base address).
             * @param imm12 12-bit immediate value.
             * @param size_spec size specifier (s, d, q).
             */
            constexpr uint32_t ldr(simd_fp_t        reg_dest,
                                   gpr_t            reg_src,
                                   uint32_t         imm12,
                                   neon_size_spec_t size_spec)
            {
                uint32_t l_ins = 0x3D400000;

                // set size
                uint32_t l_size = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                                                                              : 0;

                uint32_t l_sf = l_size & 0x3;
                l_ins |= l_sf << 30; // set bit 31, 30

                // set destination register id
                uint32_t l_reg_id = reg_dest & 0x1f;
                l_ins |= l_reg_id;

                // set first source register id
                l_reg_id = reg_src & 0x1f;
                l_ins |= l_reg_id << 5;

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
                uint32_t l_imm        = (imm12 >> l_scaleShift) & 0xfff;
                l_ins |= l_imm << 10;

                // set op code
                uint32_t l_opc = (size_spec == neon_size_spec_t::q) ? 3 : 1;
                l_ins |= l_opc << 22;

                return l_ins;
            }

            /**
             * @brief Generates an LDR (9-bit immediate) instruction using post-index encoding.
             *
             * @param reg_dest destination register.
             * @param reg_src source register (base address).
             * @param imm9 9-bit immediate value.
             * @param size_spec size specifier (s, d, q).
             */
            constexpr uint32_t ldrPost(simd_fp_t        reg_dest,
                                       gpr_t            reg_src,
                                       uint32_t         imm9,
                                       neon_size_spec_t size_spec)
            {
                uint32_t l_ins = 0x3C400400;

                // set size
                uint32_t l_size = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                                                                              : 0;

                uint32_t l_sf = l_size & 0x3;
                l_ins |= l_sf << 30; // set bit 31, 30

                // set destination register id
                uint32_t l_reg_id = reg_dest & 0x1f;
                l_ins |= l_reg_id;

                // set first source register id
                l_reg_id = reg_src & 0x1f;
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

            /**
             * @brief Generates an LDR (register, SIMD&FP) instruction using register offset.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register.
             * @param reg_src2 second source register.
             * @param option option (0:uxtw, 1:sxtw, 2:lsl, 3:sxtx).
             * @param size_spec size specifier (s, d, q).
             */
            constexpr uint32_t ldrReg(simd_fp_t        reg_dest,
                                      gpr_t            reg_src1,
                                      gpr_t            reg_src2,
                                      uint32_t         option,
                                      neon_size_spec_t size_spec)
            {
                uint32_t l_ins = 0x3C600800;

                // set size
                uint32_t l_size = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                                                                              : 0;

                uint32_t l_sf = l_size & 0x3;
                l_ins |= l_sf << 30; // set bit 31, 30

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                // set first source register id
                l_ins |= (reg_src1 & 0x1f) << 5;

                // set second source register id
                l_ins |= (reg_src2 & 0x1f) << 16;

                // Currently only uxtw supported
                if (option != 0)
                {
                    throw std::invalid_argument("Invalid option, only uxtw supported");
                }

                // set option
                if (option == 0)
                {
                    l_ins |= (1) << 14;
                }

                // set op code
                uint32_t l_opc = (size_spec == neon_size_spec_t::q) ? 1 : 0;
                l_ins |= l_opc << 23;

                return l_ins;
            }
        } // namespace simd_fp
    } // namespace instructions
} // namespace mini_jit

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_LDR_H