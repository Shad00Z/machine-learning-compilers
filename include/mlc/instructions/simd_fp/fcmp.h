#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_FCMP_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_FCMP_H

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
             * @brief Generates an FCMP (scalar) instruction.
             *
             * @param reg_src1 first source register.
             * @param reg_src2 second source register.
             * @param size_spec size specifier.
             * @param zero specifies the opcode.
             *
             * @return instruction.
             **/
            constexpr uint32_t fcmp(simd_fp_t        reg_src1,
                                    simd_fp_t        reg_src2,
                                    neon_size_spec_t size_spec,
                                    bool             zero)
            {
                if (size_spec != neon_size_spec_t::s &&
                    size_spec != neon_size_spec_t::d)
                {
                    throw std::invalid_argument("Invalid size specifier");
                }

                uint32_t l_ins = 0x1E202000;

                // set first source register id
                l_ins |= (reg_src1 & 0x1f) << 5;

                // set second source register id
                l_ins |= (reg_src2 & 0x1f) << 16;

                // set size specifier
                l_ins |= (size_spec & 0x3) << 22;

                // opcode
                if (zero)
                {
                    l_ins |= (1) << 3;
                }

                return l_ins;
            }
        } // namespace simd_fp
    } // namespace instructions
} // namespace mini_jit

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_FCMP_H
