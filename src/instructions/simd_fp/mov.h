#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_MOV_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_MOV_H

#include <cstdint>
#include <stdexcept>
#include "../../registers/gp_registers.h"
#include "../../registers/simd_fp_registers.h"
using gpr_t = mini_jit::registers::gpr_t;
using simd_fp_t = mini_jit::registers::simd_fp_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace simd_fp
        {
            /**
             * @brief Generates an MOV instruction from a general-purpose register to a vector element. This instruction can insert data into individual elements within a SIMD&FP register without clearing the remaining bits to zero.
             *
             * @param reg_dest destination SIMD&FP register.
             * @param reg_src source general-purpose register.
             * @param index index of the simd vector element to be replaced.
             * @param size_spec size specifier for the SIMD&FP register.
             */
            constexpr uint32_t mov(simd_fp_t reg_dest,
                                   gpr_t reg_src,
                                   uint32_t index,
                                   neon_size_spec_t size_spec)
            {
                u_int32_t l_ins = 0x4E001C00;

                // set destination register id
                uint32_t l_reg_id = reg_dest & 0x1f;
                l_ins |= l_reg_id;

                // set source register id
                l_reg_id = reg_src & 0x1f;
                l_ins |= l_reg_id << 5;

                // set imm5
                uint32_t imm5 = 0;

                // Encode element size as base pattern
                switch (size_spec) 
                {
                case neon_size_spec_t::s: imm5 = 0b00100; break;
                case neon_size_spec_t::d: imm5 = 0b01000; break;
                case neon_size_spec_t::q: throw std::invalid_argument("MOV instruction does not support Q size specifier"); break;
                }

                // Encode index bits
                switch (size_spec)
                {
                case neon_size_spec_t::s: imm5 |= (index & 0x3) << 3; break; // imm5<4:3>
                case neon_size_spec_t::d: imm5 |= (index & 0x1) << 4; break; // imm5<4>
                case neon_size_spec_t::q: throw std::invalid_argument("MOV instruction does not support Q size specifier"); break;
                }

                // if src is X, set x1000. otherwise leave as is I guess??
                if ((reg_src & 0x20) != 0)
                {
                    imm5 = (imm5 & 0b10000) | 0b01000;
                }

                // set imm5
                l_ins |= imm5 << 16;

                return l_ins;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_MOV_H