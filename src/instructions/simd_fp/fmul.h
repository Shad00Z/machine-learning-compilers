#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_FMUL_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_FMUL_H

#include <cstdint>
#include <stdexcept>
#include "registers/simd_fp_registers.h"
using simd_fp_t = mini_jit::registers::simd_fp_t;
using arr_spec_t = mini_jit::registers::arr_spec_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace simd_fp
        {
            /**
             * @brief Generates an FMUL (vector) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register.
             * @param reg_src2 second source register.
             * @param arr_spec arrangement specifier.
             *
             * @return instruction.
             **/
            constexpr uint32_t fmulVec(simd_fp_t reg_dest,
                                       simd_fp_t reg_src1,
                                       simd_fp_t reg_src2,
                                       arr_spec_t arr_spec)
            {
                if (arr_spec != arr_spec_t::s2 && 
                    arr_spec != arr_spec_t::s4 &&
                    arr_spec != arr_spec_t::d2)
                {
                    throw std::invalid_argument("Invalid arrangement specifier");
                }

                uint32_t l_ins = 0x2E20DC00;

                // set destination register id
                uint32_t l_reg_id = reg_dest & 0x1f;
                l_ins |= l_reg_id;

                // set first source register id
                l_reg_id = reg_src1 & 0x1f;
                l_ins |= l_reg_id << 5;

                // set second source register id
                l_reg_id = reg_src2 & 0x1f;
                l_ins |= l_reg_id << 16;

                // set arrangement specifier
                uint32_t l_arr_spec = arr_spec & 0x40400000;
                l_ins |= l_arr_spec;

                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_FMUL_H