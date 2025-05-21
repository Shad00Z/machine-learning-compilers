#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_EOR_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_EOR_H

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
             * @brief Generates an EOR (vector) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register.
             * @param reg_src2 second source register.
             * @param arr_spec arrangement specifier (8B or 16B).
             */
            constexpr uint32_t eor(simd_fp_t reg_dest,
                                   simd_fp_t reg_src1,
                                   simd_fp_t reg_src2,
                                   arr_spec_t arr_spec)
            {
                u_int32_t l_ins = 0x2E201C00;

                // set size specifier (bit 30)
                if(arr_spec == arr_spec_t::b8)
                {
                    // dont set anything
                    l_ins |= 0x0 << 30;
                }
                else if(arr_spec == arr_spec_t::b16)
                {
                    l_ins |= 0x1 << 30;
                }
                else
                {
                    throw std::invalid_argument("Invalid arrangement specifier");
                    
                }

                // set destination register id - Rd
                uint32_t l_reg_id = reg_dest & 0x1f;
                l_ins |= l_reg_id;

                // set first source register id - Rn
                l_reg_id = reg_src1 & 0x1f;
                l_ins |= (l_reg_id << 5);

                // set second source register id - Rm
                l_reg_id = reg_src2 & 0x1f;
                l_ins |= (l_reg_id << 16);

                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_EOR_H