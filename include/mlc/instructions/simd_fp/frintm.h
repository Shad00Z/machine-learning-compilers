#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_FRINTM_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_FRINTM_H

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
             * @brief Generates an FRINTM (vector) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src source register.
             * @param arr_spec arrangement specifier.
             *
             * @return instruction.
             **/
            constexpr uint32_t frintmVec(simd_fp_t  reg_dest,
                                         simd_fp_t  reg_src,
                                         arr_spec_t arr_spec)
            {
                if (arr_spec != arr_spec_t::s2 &&
                    arr_spec != arr_spec_t::s4 &&
                    arr_spec != arr_spec_t::d2)
                {
                    throw std::invalid_argument("Invalid arrangement specifier");
                }

                uint32_t l_ins = 0xE219800;

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                // set first source register id
                l_ins |= (reg_src & 0x1f) << 5;

                // set arrangement specifier
                l_ins |= (arr_spec & 0x40400000);

                return l_ins;
            }

            /**
             * @brief Generates an FRINTM (scalar) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src source register.
             * @param size_spec size specifier.
             *
             * @return instruction.
             **/
            constexpr uint32_t frintmScalar(simd_fp_t        reg_dest,
                                            simd_fp_t        reg_src,
                                            neon_size_spec_t size_spec)
            {
                if (size_spec != neon_size_spec_t::s &&
                    size_spec != neon_size_spec_t::d)
                {
                    throw std::invalid_argument("Invalid size specifier");
                }

                uint32_t l_ins = 0x1E254000;

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                // set first source register id
                l_ins |= (reg_src & 0x1f) << 5;

                // set size specifier
                l_ins |= (size_spec & 0x3) << 22;

                return l_ins;
            }
        } // namespace simd_fp
    } // namespace instructions
} // namespace mini_jit

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_FRINTM_H