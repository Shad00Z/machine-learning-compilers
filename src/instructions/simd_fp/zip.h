#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_ZIP_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_ZIP_H

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
            namespace internal
            {
                /**
                 * @brief Helper function to generate ZIP instructions.
                 *
                 * @param reg_dest destination register.
                 * @param reg_src1 first source register.
                 * @param reg_src2 second source register (base address).
                 * @param opc operation code.
                 * @param arr_spec arrangement specifier.
                 */
                constexpr uint32_t zipHelper(simd_fp_t reg_dest,
                                             simd_fp_t reg_src1,
                                             simd_fp_t reg_src2,
                                             uint32_t opc,
                                             arr_spec_t arr_spec)
                {
                    // TRN without opcode
                    uint32_t l_ins = 0xE003800;

                    // set size
                    uint32_t l_q = (arr_spec == arr_spec_t::s2) ? 0 : 1;
                    l_ins |= l_q << 30;

                    uint32_t l_size = (arr_spec == arr_spec_t::d2) ? 3 : 2;
                    l_ins |= l_size << 22;

                    // set opcode
                    l_ins |= opc << 14;

                    // set destination register id
                    uint32_t l_reg_id = reg_dest & 0x1f;
                    l_ins |= l_reg_id;

                    // set first source register id
                    l_reg_id = reg_src1 & 0x1f;
                    l_ins |= l_reg_id << 5;

                    // set second source register id
                    l_reg_id = reg_src2 & 0x1f;
                    l_ins |= l_reg_id << 16;

                    return l_ins;
                }
            }

            /**
             * @brief Generates an ZIP1 instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register.
             * @param reg_src2 second source register (base address).
             * @param arr_spec arrangement specifier.
             */
            constexpr uint32_t zip1(simd_fp_t reg_dest,
                                    simd_fp_t reg_src1,
                                    simd_fp_t reg_src2,
                                    arr_spec_t arr_spec)
            {
                uint32_t opc = 0;

                return internal::zipHelper(reg_dest,
                                           reg_src1,
                                           reg_src2,
                                           opc,
                                           arr_spec);
            }

            /**
             * @brief Generates an ZIP2 instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register.
             * @param reg_src2 second source register (base address).
             * @param arr_spec arrangement specifier.
             */
            constexpr uint32_t zip2(simd_fp_t reg_dest,
                                    simd_fp_t reg_src1,
                                    simd_fp_t reg_src2,
                                    arr_spec_t arr_spec)
            {
                uint32_t opc = 1;

                return internal::zipHelper(reg_dest,
                                           reg_src1,
                                           reg_src2,
                                           opc,
                                           arr_spec);
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_ZIP_H