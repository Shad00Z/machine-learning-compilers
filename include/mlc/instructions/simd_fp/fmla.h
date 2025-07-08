#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_FMLA_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_FMLA_H

#include <cstdint>
#include <mlc/registers/simd_fp_registers.h>
#include <stdexcept>
using simd_fp_t  = mini_jit::registers::simd_fp_t;
using arr_spec_t = mini_jit::registers::arr_spec_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace simd_fp
        {
            /**
             * @brief Generates an FMLA (vector) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register.
             * @param reg_src2 second source register.
             * @param arr_spec arrangement specifier.
             *
             * @return instruction.
             **/
            constexpr uint32_t fmlaVec(simd_fp_t  reg_dest,
                                       simd_fp_t  reg_src1,
                                       simd_fp_t  reg_src2,
                                       arr_spec_t arr_spec)
            {
                uint32_t l_ins = 0x0e20cc00;

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

            /**
             * @brief Generates an FMLA (by element) instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register.
             * @param reg_src2 second source register.
             * @param arr_spec arrangement specifier.
             *
             * @return instruction.
             **/
            constexpr uint32_t fmlaElem(simd_fp_t  reg_dest,
                                        simd_fp_t  reg_src1,
                                        simd_fp_t  reg_src2,
                                        arr_spec_t arr_spec)
            {
                // bit: 27, 26, 25, 24, 23, 12 = 1
                uint32_t l_ins = 0xF801000;

                // set destination register id
                uint32_t l_reg_id = reg_dest & 0x1f;
                l_ins |= l_reg_id;

                // set first source register id
                l_reg_id = reg_src1 & 0x1f;
                l_ins |= l_reg_id << 5;

                // set second source register id
                l_reg_id = reg_src2 & 0x1f;
                l_ins |= l_reg_id << 16; // why 16??

                // set arrangement specifier (bit 30, 22)
                uint32_t l_arr_spec = arr_spec & 0x40400000;
                l_ins |= l_arr_spec;

                return l_ins;
            }
        } // namespace simd_fp
    } // namespace instructions
} // namespace mini_jit

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_FMLA_H