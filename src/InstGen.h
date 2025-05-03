#ifndef MINI_JIT_INSTGEN_H
#define MINI_JIT_INSTGEN_H

#include <cstdint>
#include <string>

namespace mini_jit
{
  class InstGen;
}

class mini_jit::InstGen
{
public:
  //! general-purpose registers
  typedef enum : uint32_t
  {
    w0 = 0,
    w1 = 1,
    w2 = 2,
    w3 = 3,
    w4 = 4,
    w5 = 5,
    w6 = 6,
    w7 = 7,
    w8 = 8,
    w9 = 9,
    w10 = 10,
    w11 = 11,
    w12 = 12,
    w13 = 13,
    w14 = 14,
    w15 = 15,
    w16 = 16,
    w17 = 17,
    w18 = 18,
    w19 = 19,
    w20 = 20,
    w21 = 21,
    w22 = 22,
    w23 = 23,
    w24 = 24,
    w25 = 25,
    w26 = 26,
    w27 = 27,
    w28 = 28,
    w29 = 29,
    w30 = 30,

    x0 = 32 + 0,
    x1 = 32 + 1,
    x2 = 32 + 2,
    x3 = 32 + 3,
    x4 = 32 + 4,
    x5 = 32 + 5,
    x6 = 32 + 6,
    x7 = 32 + 7,
    x8 = 32 + 8,
    x9 = 32 + 9,
    x10 = 32 + 10,
    x11 = 32 + 11,
    x12 = 32 + 12,
    x13 = 32 + 13,
    x14 = 32 + 14,
    x15 = 32 + 15,
    x16 = 32 + 16,
    x17 = 32 + 17,
    x18 = 32 + 18,
    x19 = 32 + 19,
    x20 = 32 + 20,
    x21 = 32 + 21,
    x22 = 32 + 22,
    x23 = 32 + 23,
    x24 = 32 + 24,
    x25 = 32 + 25,
    x26 = 32 + 26,
    x27 = 32 + 27,
    x28 = 32 + 28,
    x29 = 32 + 29,
    x30 = 32 + 30,

    wzr = 31,
    xzr = 32 + 31,
    sp = 64 + 32 + 31
  } gpr_t;

  //! simd&fp registers
  typedef enum : uint32_t
  {
    v0 = 0,
    v1 = 1,
    v2 = 2,
    v3 = 3,
    v4 = 4,
    v5 = 5,
    v6 = 6,
    v7 = 7,
    v8 = 8,
    v9 = 9,
    v10 = 10,
    v11 = 11,
    v12 = 12,
    v13 = 13,
    v14 = 14,
    v15 = 15,
    v16 = 16,
    v17 = 17,
    v18 = 18,
    v19 = 19,
    v20 = 20,
    v21 = 21,
    v22 = 22,
    v23 = 23,
    v24 = 24,
    v25 = 25,
    v26 = 26,
    v27 = 27,
    v28 = 28,
    v29 = 29,
    v30 = 30,
    v31 = 31
  } simd_fp_t;

  //! arrangement specifiers
  typedef enum : uint32_t
  {
    s2 = 0x0,
    s4 = 0x40000000,
    d2 = 0x40400000
  } arr_spec_t;

  //! neon size specifiers
  typedef enum : uint32_t
  {
    s = 0x0,
    d = 0x1,
    q = 0x2
  } neon_size_spec_t;

  /**
   * @brief Generates a RET instruction.
   *
   * @return instruction.
   */
  static uint32_t ret();

  /**
   * @brief Generates a CBNZ instruction.
   *
   * @param reg general-purpose register.
   * @param imm19 immediate value (not the offset bytes!).
   *
   * @return instruction.
   **/
  static uint32_t base_br_cbnz(gpr_t reg,
                               int32_t imm19);

  /**
   * @brief Generates an ORR (shifted register) instruction.
   *
   * @param reg_dest destination register.
   * @param reg_src1 first source register.
   * @param reg_src2 second source register.
   * @param shift shift value.
   * @param amount amount to shift.
   *
   * @return instruction.
   **/
  static uint32_t base_orr_shifted_reg(gpr_t reg_dest,
                                       gpr_t reg_src1,
                                       gpr_t reg_src2,
                                       uint32_t shift,
                                       uint32_t amount);

  /**
   * @brief Generates an MOV (register) instruction.
   *
   * @param reg_dest destination register.
   * @param reg_src source register.
   *
   * @return instruction.
   **/
  static uint32_t mov_reg(gpr_t reg_dest,
                          gpr_t reg_src);

  /**
   * @brief Generates an MOVZ instruction.
   *
   * @param reg_dest destination register.
   * @param imm16 16-bit unsigned immediate value.
   * @param shift amount by which to left shift the immediate value.
   */
  static uint32_t movz(gpr_t reg_dest,
                       uint16_t imm16,
                       uint32_t shift);

  /**
   * @brief Generates an MOV 16-bit immediate instruction.
   *
   * @param reg_dest destination register.
   * @param imm16 16-bit unsigned immediate value.
   *
   * @return instruction.
   */
  static uint32_t mov_imm(gpr_t reg_dest,
                          uint64_t imm16);

  /**
   * @brief Generates a base LDR (12-bit immediate) instruction using unsigned offset encoding.
   *
   * @param reg_dest destination register.
   * @param reg_src source register (base address).
   * @param imm12 12-bit immediate value.
   */
  static uint32_t base_ldr_imm_uoff(gpr_t reg_dest,
                                    gpr_t reg_src,
                                    uint32_t imm12);

  /**
   * @brief Generates a base LDP instruction using signed offset encoding.
   *
   * @param reg_dest1 first destination register.
   * @param reg_dest2 second destination register.
   * @param reg_src source register (base address).
   * @param imm7 7-bit immediate value.
   */
  static uint32_t base_ldp_soff(gpr_t reg_dest1,
                                gpr_t reg_dest2,
                                gpr_t reg_src,
                                int32_t imm7);

  /**
   * @brief Generates a base LDP instruction using post-index encoding.
   *
   * @param reg_dest1 first destination register.
   * @param reg_dest2 second destination register.
   * @param reg_src source register (base address).
   * @param imm7 7-bit immediate value.
   */
  static uint32_t base_ldp_post(gpr_t reg_dest1,
                                gpr_t reg_dest2,
                                gpr_t reg_src,
                                int32_t imm7);

  /**
   * @brief Generates a base LDP instruction using pre-index encoding.
   *
   * @param reg_dest1 first destination register.
   * @param reg_dest2 second destination register.
   * @param reg_src source register (base address).
   * @param imm7 7-bit immediate value.
   */
  static uint32_t base_ldp_pre(gpr_t reg_dest1,
                               gpr_t reg_dest2,
                               gpr_t reg_src,
                               int32_t imm7);

  /**
   * @brief Generates a neon LDP instruction using signed offset encoding.
   *
   * @param reg_dest1 first destination register.
   * @param reg_dest2 second destination register.
   * @param reg_src source register (base address).
   * @param imm7 7-bit immediate value.
   * @param size_spec size specifier (s, d, q).
   */
  static uint32_t neon_ldp_soff(simd_fp_t reg_dest1,
                                simd_fp_t reg_dest2,
                                gpr_t reg_src,
                                int32_t imm7,
                                neon_size_spec_t size_spec);

  /**
   * @brief Generates a neon LDP instruction using post-index encoding.
   *
   * @param reg_dest1 first destination register.
   * @param reg_dest2 second destination register.
   * @param reg_src source register (base address).
   * @param imm7 7-bit immediate value.
   * @param size_spec size specifier (s, d, q).
   */
  static uint32_t neon_ldp_post(simd_fp_t reg_dest1,
                                simd_fp_t reg_dest2,
                                gpr_t reg_src,
                                int32_t imm7,
                                neon_size_spec_t size_spec);

  /**
   * @brief Generates a neon LDP instruction using pre-index encoding.
   *
   * @param reg_dest1 first destination register.
   * @param reg_dest2 second destination register.
   * @param reg_src source register (base address).
   * @param imm7 7-bit immediate value.
   * @param size_spec size specifier (s, d, q).
   */
  static uint32_t neon_ldp_pre(simd_fp_t reg_dest1,
                               simd_fp_t reg_dest2,
                               gpr_t reg_src,
                               int32_t imm7,
                               neon_size_spec_t size_spec);

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
  static uint32_t neon_dp_fmla_vector(simd_fp_t reg_dest,
                                      simd_fp_t reg_src1,
                                      simd_fp_t reg_src2,
                                      arr_spec_t arr_spec);

  /**
   * @brief Converts the given instruction to a hex string.
   *
   * @param inst instruction.
   *
   * @return hex string.
   **/
  static std::string to_string_hex(uint32_t inst);

  /**
   * @brief Converts the given instruction to a binary string.
   *
   * @param inst instruction.
   *
   * @return binary string.
   **/
  static std::string to_string_bin(uint32_t inst);

private:
  /**
   * @brief Helper function to generate LDP instructions.
   *
   * @param reg_dest1 first destination register.
   * @param reg_dest2 second destination register.
   * @param reg_src source register (base address).
   * @param imm7 7-bit immediate value.
   * @param opc operation code.
   * @param encoding encoding type (signed offset, post-index, pre-index).
   */
  static uint32_t ldp_help(uint32_t reg_dest1,
                           uint32_t reg_dest2,
                           uint32_t reg_src,
                           int32_t imm7,
                           uint32_t opc,
                           uint32_t encoding);
};

#endif