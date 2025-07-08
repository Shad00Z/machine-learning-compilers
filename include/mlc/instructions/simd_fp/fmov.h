#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_FMOV_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_FMOV_H

#include <cstdint>
#include <mlc/registers/simd_fp_registers.h>
#include <stdexcept>
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
             * @brief Generates an FMOV (vector, immediate) instruction.
             *
             * @param reg_dest destination register.
             * @param imm8 8-bit immediate (sign bit, 3-bit exponent, 4-bit precision).
             * @param arr_spec arrangement specifier.
             *
             * @return instruction.
             **/
            constexpr uint32_t fmovVec(simd_fp_t  reg_dest,
                                       int32_t    imm8,
                                       arr_spec_t arr_spec)
            {
                if (arr_spec != arr_spec_t::s2 &&
                    arr_spec != arr_spec_t::s4 &&
                    arr_spec != arr_spec_t::d2)
                {
                    throw std::invalid_argument("Invalid arrangement specifier");
                }

                int32_t l_ins = 0xF00F400;

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                // move lower 5 immediate bits
                l_ins |= (imm8 & 0x1f) << 5;

                // move upper 3 immediate bits
                l_ins |= (imm8 & 0xe0) << 11;

                // set arrangement specifier
                if (arr_spec == arr_spec_t::s4)
                {
                    l_ins |= (0x1) << 30;
                }
                else if (arr_spec == arr_spec_t::d2)
                {
                    l_ins |= (0x1) << 29;
                    l_ins |= (0x1) << 30;
                }

                return l_ins;
            }

            /**
             * @brief Generates an FMOV (scalar, immediate) instruction.
             *
             * @param reg_dest destination register.
             * @param imm8 8-bit immediate (sign bit, 3-bit exponent, 4-bit precision).
             * @param size_spec size specifier.
             *
             * @return instruction.
             **/
            constexpr uint32_t fmovScalar(simd_fp_t        reg_dest,
                                          int32_t          imm8,
                                          neon_size_spec_t size_spec)
            {
                if (size_spec != neon_size_spec_t::s &&
                    size_spec != neon_size_spec_t::d)
                {
                    throw std::invalid_argument("Invalid size specifier");
                }

                uint32_t l_ins = 0x1E201000;

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                if (imm8 > 31 || imm8 < -31)
                {
                    throw std::invalid_argument("Invalid immediate (allowed range: -31, 31)");
                }

                // immediate
                l_ins |= (imm8 & 0xff) << 13;

                // set size specifier
                l_ins |= (size_spec & 0x3) << 22;

                return l_ins;
            }

            /**
             * @brief Generates an FMOV (vector, immediate) instruction.
             *
             * @param reg_dest destination register.
             * @param imm8 8-bit integer value to move.
             * @param arr_spec arrangement specifier.
             *
             * @return instruction.
             **/
            constexpr uint32_t fmovIntVec(simd_fp_t  reg_dest,
                                          int32_t    imm8,
                                          arr_spec_t arr_spec)
            {
                if (arr_spec != arr_spec_t::s2 &&
                    arr_spec != arr_spec_t::s4 &&
                    arr_spec != arr_spec_t::d2)
                {
                    throw std::invalid_argument("Invalid arrangement specifier");
                }

                int32_t l_ins = 0xF00F400;

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                if (imm8 > 31 || imm8 < -31)
                {
                    throw std::invalid_argument("Invalid immediate (allowed range: -31, 31)");
                }

                if (imm8 < 0)
                {
                    l_ins |= (0x1) << 18;
                    imm8 *= -1;
                }

                // immediate bits
                if (imm8 == 1)
                {
                    l_ins |= (0x3) << 16;
                    l_ins |= (0x1) << 9;
                }
                else if (imm8 == 2)
                {
                }
                else if (imm8 == 3)
                {
                    l_ins |= (0x1) << 8;
                }
                else if (imm8 < 8)
                {
                    l_ins |= (imm8 & 0x7) << 7;
                }
                else
                {
                    l_ins |= (0x1) << 16;

                    if (imm8 > 8 && imm8 < 16)
                    {
                        l_ins |= (imm8 & 0x7) << 6;
                    }
                    else if (imm8 > 16)
                    {
                        l_ins |= (imm8 & 0x1f) << 5;
                    }
                }

                // set arrangement specifier
                if (arr_spec == arr_spec_t::s4)
                {
                    l_ins |= (0x1) << 30;
                }
                else if (arr_spec == arr_spec_t::d2)
                {
                    l_ins |= (0x1) << 29;
                    l_ins |= (0x1) << 30;
                }

                return l_ins;
            }

            /**
             * @brief Generates an FMOV (scalar, immediate) instruction.
             *
             * @param reg_dest destination register.
             * @param imm8 8-bit integer value to move.
             * @param size_spec size specifier.
             *
             * @return instruction.
             **/
            constexpr uint32_t fmovIntScalar(simd_fp_t        reg_dest,
                                             int32_t          imm8,
                                             neon_size_spec_t size_spec)
            {
                if (size_spec != neon_size_spec_t::s &&
                    size_spec != neon_size_spec_t::d)
                {
                    throw std::invalid_argument("Invalid size specifier");
                }

                uint32_t l_ins = 0x1E201000;

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                if (imm8 > 31 || imm8 < -31)
                {
                    throw std::invalid_argument("Invalid immediate (allowed range: -31, 31)");
                }

                if (imm8 < 0)
                {
                    l_ins |= (0x1) << 20;
                    imm8 *= -1;
                }

                // immediate bits
                if (imm8 == 1)
                {
                    l_ins |= (0x7) << 17;
                }
                else if (imm8 == 2)
                {
                }
                else if (imm8 == 3)
                {
                    l_ins |= (0x1) << 16;
                }
                else if (imm8 < 8)
                {
                    l_ins |= (imm8 & 0x7) << 15;
                }
                else
                {
                    l_ins |= (0x1) << 18;

                    if (imm8 > 8 && imm8 < 16)
                    {
                        l_ins |= (imm8 & 0x7) << 14;
                    }
                    else if (imm8 > 16)
                    {
                        l_ins |= (imm8 & 0x1f) << 13;
                    }
                }

                // set size specifier
                l_ins |= (size_spec & 0x3) << 22;

                return l_ins;
            }
        } // namespace simd_fp
    } // namespace instructions
} // namespace mini_jit

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_FMOV_H