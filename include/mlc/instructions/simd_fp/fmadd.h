#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_FMADD_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_FMADD_H

#include <cstdint>
#include <mlc/registers/simd_fp_registers.h>
#include <stdexcept>
using simd_fp_t        = mini_jit::registers::simd_fp_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace simd_fp
        {
            /**
             * @brief Generates an FMADD instruction.
             *
             * @param reg_dest destination register.
             * @param reg_src1 first source register (multiplicand).
             * @param reg_src2 second source register (multiplier).
             * @param reg_src3 third source register (addend).
             * @param size_spec size specifier (s or d).
             */
            constexpr uint32_t fmadd(simd_fp_t        reg_dest,
                                     simd_fp_t        reg_src1,
                                     simd_fp_t        reg_src2,
                                     simd_fp_t        reg_src3,
                                     neon_size_spec_t size_spec)
            {
                u_int32_t l_ins = 0x1F000000;

                // set destination register id - Rd
                uint32_t l_reg_id = reg_dest & 0x1f;
                l_ins |= l_reg_id;

                // set first source register id - Rn
                l_reg_id = reg_src1 & 0x1f;
                l_ins |= (l_reg_id << 5);

                // set second source register id - Rm
                l_reg_id = reg_src2 & 0x1f;
                l_ins |= (l_reg_id << 16);

                // set third source register id - Ra
                l_reg_id = reg_src3 & 0x1f;
                l_ins |= (l_reg_id << 10);

                // set size specifier
                l_ins |= (size_spec & 0x1) << 22;

                return l_ins;
            }
        } // namespace simd_fp
    } // namespace instructions
} // namespace mini_jit

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_FMADD_H