#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_STP_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_STP_H

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
            namespace internal
            {
                /**
                 * @brief Helper function to generate STP instructions.
                 *
                 * @param reg_data1 first register holding the data to be transferred.
                 * @param reg_data2 second register holding the data to be transferred.
                 * @param reg_address register holding the memory address.
                 * @param imm7 7-bit immediate value.
                 * @param opc operation code.
                 * @param encoding encoding type (signed offset, post-index, pre-index).
                 */
                constexpr uint32_t stpHelper(uint32_t reg_data1,
                                             uint32_t reg_data2,
                                             uint32_t reg_address,
                                             int32_t  imm7,
                                             uint32_t opc,
                                             uint32_t encoding)
                {
                    // STP without VR - bits: 29 = 1, 27 = 1
                    uint32_t l_ins = 0x28000000;

                    // set 2-bit opc
                    l_ins |= (opc & 0x3) << 30;

                    // set 4-bit VR encoding
                    l_ins |= (encoding & 0xF) << 23;

                    // set first destination register
                    uint32_t l_reg_id = reg_data1 & 0x1f;
                    l_ins |= l_reg_id;
                    // set source register
                    l_reg_id = reg_address & 0x1f;
                    l_ins |= l_reg_id << 5;
                    // set second destination register
                    l_reg_id = reg_data2 & 0x1f;
                    l_ins |= l_reg_id << 10;
                    // set immediate value
                    uint32_t l_imm = imm7 & 0x7f;
                    l_ins |= l_imm << 15;

                    return l_ins;
                }
            } // namespace internal

            /**
             * @brief Generates an STP instruction using signed offset encoding.
             *
             * @param reg_data1 first register holding the data to be transferred.
             * @param reg_data2 second register holding the data to be transferred.
             * @param reg_address register holding the memory address.
             * @param imm7 7-bit immediate value.
             * @param size_spec size specifier (s, d, q).
             */
            constexpr uint32_t stp(simd_fp_t        reg_data1,
                                   simd_fp_t        reg_data2,
                                   gpr_t            reg_address,
                                   int32_t          imm7,
                                   neon_size_spec_t size_spec)
            {
                // check if immediate can be encoded
                uint32_t l_scale = (size_spec == neon_size_spec_t::s) ? 4 : (size_spec == neon_size_spec_t::d) ? 8
                                                                                                               : 16;
                if (imm7 % l_scale != 0)
                {
                    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit) or 16 (128-bit)");
                }

                // scale the immediate for encoding (right-shift)
                uint32_t l_scaleShift = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                                                                                    : 4;
                uint32_t l_imm        = (imm7 >> l_scaleShift) & 0x7f;

                // set op code
                uint32_t l_opc = size_spec & 0x3;

                // encoding: 1010
                uint32_t l_encoding = 0xA;

                return internal::stpHelper(reg_data1,
                                           reg_data2,
                                           reg_address,
                                           l_imm,
                                           l_opc,
                                           l_encoding);
            }

            /**
             * @brief Generates an STP instruction using post-index encoding.
             *
             * @param reg_data1 first register holding the data to be transferred.
             * @param reg_data2 second register holding the data to be transferred.
             * @param reg_address register holding the memory address.
             * @param imm7 7-bit immediate value.
             * @param size_spec size specifier (s, d, q).
             */
            constexpr uint32_t stpPost(simd_fp_t        reg_data1,
                                       simd_fp_t        reg_data2,
                                       gpr_t            reg_address,
                                       int32_t          imm7,
                                       neon_size_spec_t size_spec)
            {
                // check if immediate can be encoded
                uint32_t l_scale = (size_spec == neon_size_spec_t::s) ? 4 : (size_spec == neon_size_spec_t::d) ? 8
                                                                                                               : 16;
                if (imm7 % l_scale != 0)
                {
                    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit) or 16 (128-bit)");
                }

                // scale the immediate for encoding (right-shift)
                uint32_t l_scaleShift = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                                                                                    : 4;
                uint32_t l_imm        = (imm7 >> l_scaleShift) & 0x7f;

                // set op code
                uint32_t l_opc = size_spec & 0x3;

                // encoding: 1001
                uint32_t l_encoding = 0x9;

                return internal::stpHelper(reg_data1,
                                           reg_data2,
                                           reg_address,
                                           l_imm,
                                           l_opc,
                                           l_encoding);
            }

            /**
             * @brief Generates an STP instruction using pre-index encoding.
             *
             * @param reg_data1 first register holding the data to be transferred.
             * @param reg_data2 second register holding the data to be transferred.
             * @param reg_address register holding the memory address.
             * @param imm7 7-bit immediate value.
             * @param size_spec size specifier (s, d, q).
             */
            constexpr uint32_t stpPre(simd_fp_t        reg_data1,
                                      simd_fp_t        reg_data2,
                                      gpr_t            reg_address,
                                      int32_t          imm7,
                                      neon_size_spec_t size_spec)
            {
                // check if immediate can be encoded
                uint32_t l_scale = (size_spec == neon_size_spec_t::s) ? 4 : (size_spec == neon_size_spec_t::d) ? 8
                                                                                                               : 16;
                if (imm7 % l_scale != 0)
                {
                    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit) or 16 (128-bit)");
                }

                // scale the immediate for encoding (right-shift)
                uint32_t l_scaleShift = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                                                                                    : 4;
                uint32_t l_imm        = (imm7 >> l_scaleShift) & 0x7f;

                // set op code
                uint32_t l_opc = size_spec & 0x3;

                // encoding: 1011
                uint32_t l_encoding = 0xB;

                return internal::stpHelper(reg_data1,
                                           reg_data2,
                                           reg_address,
                                           l_imm,
                                           l_opc,
                                           l_encoding);
            }

        } // namespace simd_fp
    } // namespace instructions
} // namespace mini_jit

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_STP_H