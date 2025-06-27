#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_FABS_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_FABS_H

#include <cstdint>
#include <stdexcept>
#include "registers/simd_fp_registers.h"
using simd_fp_t = mini_jit::registers::simd_fp_t;
using arr_spec_t = mini_jit::registers::arr_spec_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace simd_fp
        {
            /**
             * @brief Generates an FABS (vector) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src  source register.
             * @param arr_spec arrangement specifier.
             */
            constexpr uint32_t fabsVec(simd_fp_t reg_dest,
                                       simd_fp_t reg_src,
                                       arr_spec_t arr_spec)
            {
                if (arr_spec != arr_spec_t::s2 && 
                    arr_spec != arr_spec_t::s4 &&
                    arr_spec != arr_spec_t::d2)
                {
                    throw std::invalid_argument("Invalid arrangement specifier for fabsVec");
                }

                uint32_t l_ins = 0xEA0F800;

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                // set source register id
                l_ins |= (reg_src & 0x1f) << 5;

                // set arrangement specifier
                l_ins |= (arr_spec & 0x40400000);

                return l_ins;
            }

            /**
             * @brief Generates an FABS (scalar) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src  source register.
             * @param size_spec size specifier.
             */
            constexpr uint32_t fabsScalar(simd_fp_t reg_dest,
                                          simd_fp_t reg_src,
                                          neon_size_spec_t size_spec)
            {
                if (size_spec != neon_size_spec_t::s &&
                    size_spec != neon_size_spec_t::d)
                {
                    throw std::invalid_argument("Invalid size specifier for fabsScalar");
                }

                uint32_t l_ins = 0x1E20C000;

                // set ftype
                uint32_t ftype = (size_spec == neon_size_spec_t::s) ? 0 : 1;
                l_ins |= (ftype & 0x1) << 22;

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                // set source register id
                l_ins |= (reg_src & 0x1f) << 5;
                
                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_FABS_H