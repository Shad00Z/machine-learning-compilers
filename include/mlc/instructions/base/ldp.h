#ifndef MINI_JIT_INSTRUCTIONS_BASE_LDP_H
#define MINI_JIT_INSTRUCTIONS_BASE_LDP_H

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
            namespace internal
            {
                /**
                 * @brief Helper function to generate LDP instructions.
                 *
                 * @param reg_dest1 first destination register.
                 * @param reg_dest2 second destination register.
                 * @param reg_src source register (base address).
                 * @param imm7 7-bit immediate value.
                 * @param opc operation code.
                 * @param encoding encoding type (signed offset, post-index, pre-index).
                 */
                constexpr uint32_t ldpHelper(uint32_t reg_dest1,
                                             uint32_t reg_dest2,
                                             uint32_t reg_src,
                                             int32_t  imm7,
                                             uint32_t opc,
                                             uint32_t encoding)
                {
                    // LDP without VR
                    uint32_t l_ins = 0x28400000;
                    // set 2-bit opc
                    l_ins |= (opc & 0x3) << 30;
                    // set 4-bit VR encoding
                    l_ins |= (encoding & 0xF) << 23;
                    // set first destination register
                    l_ins |= (reg_dest1 & 0x1f);
                    // set source register
                    l_ins |= (reg_src & 0x1f) << 5;
                    // set second destination register
                    l_ins |= (reg_dest2 & 0x1f) << 10;
                    // set immediate value
                    l_ins |= (imm7 & 0x7f) << 15;

                    return l_ins;
                }
            } // namespace internal

            /**
             * @brief Generates a base LDP instruction using signed offset encoding.
             *
             * @param reg_dest1 first destination register.
             * @param reg_dest2 second destination register.
             * @param reg_src source register (base address).
             * @param imm7 7-bit immediate value.
             */
            constexpr uint32_t ldp(gpr_t   reg_dest1,
                                   gpr_t   reg_dest2,
                                   gpr_t   reg_src,
                                   int32_t imm7)
            {
                uint32_t l_sf1 = reg_dest1 & 0x20;
                uint32_t l_sf2 = reg_dest2 & 0x20;
                if (l_sf1 != l_sf2)
                {
                    throw std::invalid_argument("LDP: both destination registers must be of the same size");
                }

                // check if immediate can be encoded
                uint32_t l_scale = (l_sf1) ? 8 : 4;
                if (imm7 % l_scale != 0)
                {
                    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)");
                }

                // scale the immediate for encoding (right-shift)
                uint32_t l_scaleShift = (l_sf1) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
                uint32_t l_imm        = (imm7 >> l_scaleShift) & 0x7f;

                // set op code
                uint32_t l_opc = (l_sf1) ? 0x2 : 0x0;

                // encoding: 0010
                uint32_t l_encoding = 0x2;

                return internal::ldpHelper(reg_dest1,
                                           reg_dest2,
                                           reg_src,
                                           l_imm,
                                           l_opc,
                                           l_encoding);
            }

            /**
             * @brief Generates a base LDP instruction using post-index encoding.
             *
             * @param reg_dest1 first destination register.
             * @param reg_dest2 second destination register.
             * @param reg_src source register (base address).
             * @param imm7 7-bit immediate value.
             */
            constexpr uint32_t ldpPost(gpr_t   reg_dest1,
                                       gpr_t   reg_dest2,
                                       gpr_t   reg_src,
                                       int32_t imm7)
            {
                uint32_t l_sf1 = reg_dest1 & 0x20;
                uint32_t l_sf2 = reg_dest2 & 0x20;
                if (l_sf1 != l_sf2)
                {
                    throw std::invalid_argument("LDP: both destination registers must be of the same size");
                }

                // check if immediate can be encoded
                uint32_t l_scale = (l_sf1) ? 8 : 4;
                if (imm7 % l_scale != 0)
                {
                    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)");
                }

                // scale the immediate for encoding (right-shift)
                uint32_t l_scaleShift = (l_sf1) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
                uint32_t l_imm        = (imm7 >> l_scaleShift) & 0x7f;

                // set op code
                uint32_t l_opc = (l_sf1) ? 0x2 : 0x0;

                // encoding: 0001
                uint32_t l_encoding = 0x1;

                return internal::ldpHelper(reg_dest1,
                                           reg_dest2,
                                           reg_src,
                                           l_imm,
                                           l_opc,
                                           l_encoding);
            }

            /**
             * @brief Generates a base LDP instruction using pre-index encoding.
             *
             * @param reg_dest1 first destination register.
             * @param reg_dest2 second destination register.
             * @param reg_src source register (base address).
             * @param imm7 7-bit immediate value.
             */
            constexpr uint32_t ldpPre(gpr_t   reg_dest1,
                                      gpr_t   reg_dest2,
                                      gpr_t   reg_src,
                                      int32_t imm7)
            {
                uint32_t l_sf1 = reg_dest1 & 0x20;
                uint32_t l_sf2 = reg_dest2 & 0x20;
                if (l_sf1 != l_sf2)
                {
                    throw std::invalid_argument("LDP: both destination registers must be of the same size");
                }

                // check if immediate can be encoded
                uint32_t l_scale = (l_sf1) ? 8 : 4;
                if (imm7 % l_scale != 0)
                {
                    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)");
                }

                // scale the immediate for encoding (right-shift)
                uint32_t l_scaleShift = (l_sf1) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
                uint32_t l_imm        = (imm7 >> l_scaleShift) & 0x7f;

                // set op code
                uint32_t l_opc = (l_sf1) ? 0x2 : 0x0;

                // encoding: 0011
                uint32_t l_encoding = 0x3;

                return internal::ldpHelper(reg_dest1,
                                           reg_dest2,
                                           reg_src,
                                           l_imm,
                                           l_opc,
                                           l_encoding);
            }
        } // namespace base
    } // namespace instructions
} // namespace mini_jit

#endif // MINI_JIT_INSTRUCTIONS_BASE_LDP_H