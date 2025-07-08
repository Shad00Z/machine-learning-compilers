#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_FRECPE_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_FRECPE_H

#include <cstdint>
#include <mlc/registers/simd_fp_registers.h>
#include <stdexcept>
using simd_fp_t   = mini_jit::registers::simd_fp_t;
using arr_spec_t  = mini_jit::registers::arr_spec_t;
using size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace simd_fp
        {
            /**
             * @brief Generates a vector FRECPE (Floating-point reciprocal estimate) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src source register.
             * @param arr_spec arrangement specifier (2s, 4s or 2d).
             */
            constexpr uint32_t frecpeVec(simd_fp_t  reg_dest,
                                         simd_fp_t  reg_src,
                                         arr_spec_t arr_spec)
            {
                u_int32_t l_ins = 0xEA1D800;

                // set destination register id - Rd
                l_ins |= (reg_dest & 0x1f);

                // set source register id - Rn
                l_ins |= (reg_src & 0x1f) << 5;

                // set arrangement specifier
                l_ins |= (arr_spec & 0x40400000);

                return l_ins;
            }

            /**
             * @brief Generates a scalar FRECPE (Floating-point reciprocal estimate) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src source register.
             * @param size_spec size specifier (s, d).
             */
            constexpr uint32_t frecpeScalar(simd_fp_t   reg_dest,
                                            simd_fp_t   reg_src,
                                            size_spec_t size_spec)
            {
                if (size_spec != neon_size_spec_t::s &&
                    size_spec != neon_size_spec_t::d)
                {
                    throw std::invalid_argument("Invalid size specifier");
                }

                u_int32_t l_ins = 0x5EA1D800;

                // set destination register id - Rd
                l_ins |= (reg_dest & 0x1f);

                // set source register id - Rn
                l_ins |= (reg_src & 0x1f) << 5;

                // set size specifier
                l_ins |= (size_spec & 0x1) << 22;

                return l_ins;
            }
        } // namespace simd_fp
    } // namespace instructions
} // namespace mini_jit

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_FRECPE_H