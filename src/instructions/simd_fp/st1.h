#ifndef MINI_JIT_INSTRUCTIONS_SIMD_FP_ST1_H
#define MINI_JIT_INSTRUCTIONS_SIMD_FP_ST1_H

#include <cstdint>
#include <stdexcept>
#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
using gpr_t = mini_jit::registers::gpr_t;
using simd_fp_t = mini_jit::registers::simd_fp_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace mini_jit
{
    namespace instructions
    {
        namespace simd_fp
        {
            namespace internal
            {
                //! Helper function to check if the size is valid
                constexpr void checkSizeST1(neon_size_spec_t size)
                {
                    if (size != neon_size_spec_t::s && size != neon_size_spec_t::d)
                    {
                        throw std::invalid_argument("Only s and d sizes are currently supported for ST1 instructions.");
                    }
                }

                //! Helper function to check if the index is valid
                constexpr void checkIndexST1(neon_size_spec_t size, uint32_t index)
                {
                    if (size == neon_size_spec_t::s && index > 3)
                    {
                        throw std::out_of_range("Index for s size must be between 0 and 3.");
                    }
                    else if (size == neon_size_spec_t::d && index > 1)
                    {
                        throw std::out_of_range("Index for d size must be between 0 and 1.");
                    }
                }

                //! Helper function to check if the post-index immediate is valid
                constexpr void checkPostIndexST1(neon_size_spec_t size, uint32_t post_index)
                {
                    if (size == neon_size_spec_t::s && post_index != 4)
                    {
                        throw std::invalid_argument("Post-index immediate for s size must be 4.");
                    }
                    else if (size == neon_size_spec_t::d && post_index != 8)
                    {
                        throw std::invalid_argument("Post-index immediate for d size must be 8.");
                    }
                }

            }

            /**!
             * @brief Generates an st1 instruction (single structure) with a lane index, e.g. st1 {V0.S}[0], [X0]
             * @param reg_dst Destination SIMD register.
             * @param reg_src Source general-purpose register containing the address.
             * @param index Index of the lane to load to.
             * @param size Size of the SIMD register (s or d).
             */
            constexpr uint32_t st1(simd_fp_t reg_dst,
                                   gpr_t reg_src,
                                   uint32_t index,
                                   neon_size_spec_t size)
            {
                internal::checkSizeST1(size);
                internal::checkIndexST1(size, index);

                uint32_t l_ins = 0xD000000;
                uint32_t l_opc = 0x4; // 100
                uint32_t Q = 0x0;
                uint32_t S = 0x0;
                if (size == neon_size_spec_t::s)
                {
                    S = index & 0x1;
                    Q = (index >> 1) & 0x1;
                }
                else if (size == neon_size_spec_t::d)
                {
                    S = 0x0;
                    Q = index & 0x1;
                }

                l_ins |= (size << 10);
                l_ins |= (l_opc << 13);
                l_ins |= (Q << 30);
                l_ins |= (S << 12);

                uint32_t l_reg_id = reg_dst & 0x1f;
                l_ins |= l_reg_id;

                l_reg_id = reg_src & 0x1f;
                l_ins |= l_reg_id << 5;

                return l_ins;
            }

            /**!
             * @brief Generates an st1 instruction (single structure) with a lane index and a register post-index, e.g. st1 {V0.S}[0], [X0], X1
             * @param reg_dst Destination SIMD register.
             * @param reg_src Source general-purpose register containing the address.
             * @param index Index of the lane to load to.
             * @param size Size of the SIMD register (s or d).
             * @param post_index Post-index register to add to the address in reg_src.
             */
            constexpr uint32_t st1(simd_fp_t reg_dst,
                                   gpr_t reg_src,
                                   uint32_t index,
                                   neon_size_spec_t size,
                                   gpr_t reg_post_index)
            {
                internal::checkSizeST1(size);
                internal::checkIndexST1(size, index);

                uint32_t l_ins = 0xD800000;
                uint32_t l_opc = 0x4; // 100
                uint32_t Q = 0x0;
                uint32_t S = 0x0;
                if (size == neon_size_spec_t::s)
                {
                    S = index & 0x1;
                    Q = (index >> 1) & 0x1;
                }
                else if (size == neon_size_spec_t::d)
                {
                    S = 0x0;
                    Q = index & 0x1;
                }

                l_ins |= (size << 10);
                l_ins |= (l_opc << 13);
                l_ins |= (Q << 30);
                l_ins |= (S << 12);

                uint32_t l_reg_id = reg_dst & 0x1f;
                l_ins |= l_reg_id;

                l_reg_id = reg_src & 0x1f;
                l_ins |= l_reg_id << 5;

                l_reg_id = reg_post_index & 0x1f;
                l_ins |= (l_reg_id << 16);

                return l_ins;
            }

            /**!
             * @brief Generates an st1 instruction (single structure) with a lane index and a post-index immediate, e.g. st1 {V0.S}[0], [X0], #4
             * @param reg_dst Destination SIMD register.
             * @param reg_src Source general-purpose register containing the address.
             * @param index Index of the lane to load to.
             * @param size Size of the SIMD register (s or d).
             * @param post_index Post-index immediate to add to the address in reg_src.
             */
            constexpr uint32_t st1(simd_fp_t reg_dst,
                                   gpr_t reg_src,
                                   uint32_t index,
                                   neon_size_spec_t size,
                                   uint32_t post_index)
            {
                internal::checkPostIndexST1(size, post_index);
                /*
                 * the post_index is not used in the instruction encoding!
                 * seems like #4 and #8 are used implicitly when the register
                 * is set to all 1s
                 */ 
                return st1(reg_dst,
                           reg_src,
                           index,
                           size,
                           gpr_t::wzr);
            }

        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_SIMD_FP_ST1_H
