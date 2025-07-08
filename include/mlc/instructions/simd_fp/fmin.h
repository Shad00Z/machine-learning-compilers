#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_FMIN_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_FMIN_H

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
             * @brief Generates an FMIN (scalar) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register.
             * @param reg_src2 second source register.
             * @param size_spec size specifier (s or d).
             */
            constexpr uint32_t fminScalar(simd_fp_t        reg_dest,
                                          simd_fp_t        reg_src1,
                                          simd_fp_t        reg_src2,
                                          neon_size_spec_t size_spec)
            {
                if (size_spec != neon_size_spec_t::s && size_spec != neon_size_spec_t::d)
                {
                    throw std::invalid_argument("Invalid size specifier");
                }

                u_int32_t l_ins = 0x1E205800;

                // set destination register id - Rd
                l_ins |= (reg_dest & 0x1f);

                // set first source register id - Rn
                l_ins |= (reg_src1 & 0x1f) << 5;

                // set second source register id - Rm
                l_ins |= (reg_src2 & 0x1f) << 16;

                // set neon size specifier - size_spec
                l_ins |= size_spec << 22;

                return l_ins;
            }

            /**
             * @brief Generates an FMIN (vector) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register.
             * @param reg_src2 second source register.
             * @param arr_spec_t arrangement specifier.
             */
            constexpr uint32_t fminVec(simd_fp_t  reg_dest,
                                       simd_fp_t  reg_src1,
                                       simd_fp_t  reg_src2,
                                       arr_spec_t arr_spec)
            {
                if (arr_spec != arr_spec_t::s2 && arr_spec != arr_spec_t::s4 &&
                    arr_spec != arr_spec_t::d2)
                {
                    throw std::invalid_argument("Invalid arrangement specifier");
                }

                u_int32_t l_ins = 0xEA0F400;

                // set destination register id - Rd
                l_ins |= (reg_dest & 0x1f);

                // set first source register id - Rn
                l_ins |= (reg_src1 & 0x1f) << 5;

                // set second source register id - Rm
                l_ins |= (reg_src2 & 0x1f) << 16;

                // set arrangement specifier - arr_spec
                l_ins |= arr_spec;

                return l_ins;
            }
        } // namespace simd_fp
    } // namespace instructions
} // namespace mini_jit

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_FMIN_H