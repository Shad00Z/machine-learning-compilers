#include "InstGen.h"
#include <iostream>

using gpr_t = mini_jit::InstGen::gpr_t;
using simd_fp_t = mini_jit::InstGen::simd_fp_t;
using arr_spec_t = mini_jit::InstGen::arr_spec_t;

void base_load_instructions( uint32_t &l_ins,
  std::string &l_str,
  mini_jit::InstGen &l_gen )
{
  std::cout << "#############################" << std::endl;
  std::cout << "### base Load instructions ###" << std::endl;
  std::cout << "#############################" << std::endl;

  // LDR (unsigned offset)
  std::cout << "ldr w0, [x2, #8]" << std::endl;
  l_ins = l_gen.base_ldr_imm_uoff(gpr_t::w0,
                                  gpr_t::x2,
                                  8);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  // LDP (post-index)
  std::cout << "ldp w0, w1, [x2], #8" << std::endl;
  l_ins = l_gen.base_ldp_post(gpr_t::w0,
                              gpr_t::w1,
                              gpr_t::x2,
                              8);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  // LDP (pre-index)
  std::cout << "ldp w0, w1, [x2, #8]!" << std::endl;
  l_ins = l_gen.base_ldp_pre(gpr_t::w0,
                             gpr_t::w1,
                             gpr_t::x2,
                             8);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  // LDP (signed offset)
  std::cout << "ldp w0, w1, [x2, #8]" << std::endl;
  l_ins = l_gen.base_ldp_soff(gpr_t::w0,
                              gpr_t::w1,
                              gpr_t::x2,
                              8);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;
}

void neon_load_instructions( uint32_t &l_ins,
  std::string &l_str,
  mini_jit::InstGen &l_gen )
{
  std::cout << "#############################" << std::endl;
  std::cout << "### neon Load instructions ###" << std::endl;
  std::cout << "#############################" << std::endl;

  // neon LDR (unsigned offset)
  std::cout << "ldr q0, [x2, #16]" << std::endl;
  l_ins = l_gen.neon_ldr_imm_uoff(simd_fp_t::v0,
                                  gpr_t::x2,
                                  16,
                                  mini_jit::InstGen::neon_size_spec_t::q);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  // neon LDP (post-index)
  std::cout << "ldp q0, q1, [x2], #16" << std::endl;
  l_ins = l_gen.neon_ldp_post(simd_fp_t::v0,
                              simd_fp_t::v1,
                              gpr_t::x2,
                              16,
                              mini_jit::InstGen::neon_size_spec_t::q);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  // neon LDP (pre-index)
  std::cout << "ldp q0, q1, [x2, #16]!" << std::endl;
  l_ins = l_gen.neon_ldp_pre(simd_fp_t::v0,
                             simd_fp_t::v1,
                             gpr_t::x2,
                             16,
                             mini_jit::InstGen::neon_size_spec_t::q);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  // neon LDP (signed offset)
  std::cout << "ldp q0, q1, [x2, #16]" << std::endl;
  l_ins = l_gen.neon_ldp_soff(simd_fp_t::v0,
                              simd_fp_t::v1,
                              gpr_t::x2,
                              16,
                              mini_jit::InstGen::neon_size_spec_t::q);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  // neon LDP (post-index)
  std::cout << "ldp d0, d1, [x2], #8" << std::endl;
  l_ins = l_gen.neon_ldp_post(simd_fp_t::v0,
                              simd_fp_t::v1,
                              gpr_t::x2,
                              8,
                              mini_jit::InstGen::neon_size_spec_t::d);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  std::cout << "" << std::endl;
}

void base_store_instructions(uint32_t &l_ins,
                             std::string &l_str,
                             mini_jit::InstGen &l_gen)
{
  std::cout << "#############################" << std::endl;
  std::cout << "### base STP instructions ###" << std::endl;
  std::cout << "#############################" << std::endl;
  // STP (post-index)
  std::cout << "stp w0, w1, [x2], #8" << std::endl;
  l_ins = l_gen.base_stp_post(gpr_t::w0,
                              gpr_t::w1,
                              gpr_t::x2,
                              8);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  // STP (pre-index)
  std::cout << "stp w0, w1, [x2, #8]!" << std::endl;
  l_ins = l_gen.base_stp_pre(gpr_t::w0,
                             gpr_t::w1,
                             gpr_t::x2,
                             8);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  // STP (signed offset)
  std::cout << "stp w0, w1, [x2, #8]" << std::endl;
  l_ins = l_gen.base_stp_soff(gpr_t::w0,
                              gpr_t::w1,
                              gpr_t::x2,
                              8);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  std::cout << "" << std::endl;
}

void neon_store_instructions(uint32_t &l_ins,
                             std::string &l_str,
                             mini_jit::InstGen &l_gen)
{
  std::cout << "#############################" << std::endl;
  std::cout << "### neon STP instructions ###" << std::endl;
  std::cout << "#############################" << std::endl;
  // neon STP (post-index)
  std::cout << "stp q0, q1, [x2], #16" << std::endl;
  l_ins = l_gen.neon_stp_post(simd_fp_t::v0,
                              simd_fp_t::v1,
                              gpr_t::x2,
                              16,
                              mini_jit::InstGen::neon_size_spec_t::q);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  // neon STP (pre-index)
  std::cout << "stp q0, q1, [x2, #16]!" << std::endl;
  l_ins = l_gen.neon_stp_pre(simd_fp_t::v0,
                             simd_fp_t::v1,
                             gpr_t::x2,
                             16,
                             mini_jit::InstGen::neon_size_spec_t::q);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  // neon STP (signed offset)
  std::cout << "stp q0, q1, [x2, #16]" << std::endl;
  l_ins = l_gen.neon_stp_soff(simd_fp_t::v0,
                              simd_fp_t::v1,
                              gpr_t::x2,
                              16,
                              mini_jit::InstGen::neon_size_spec_t::q);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  std::cout << "" << std::endl;
}

void math_instructions(uint32_t &l_ins,
                       std::string &l_str,
                       mini_jit::InstGen &l_gen)
{
  std::cout << "#############################" << std::endl;
  std::cout << "### math instructions ###" << std::endl;
  std::cout << "#############################" << std::endl;

  // MUL
  std::cout << "mul w0, w1, w2" << std::endl;
  l_ins = l_gen.mul_reg(gpr_t::w0,
                        gpr_t::w1,
                        gpr_t::w2);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  std::cout << "" << std::endl;

  // ADD (shifted register)
  std::cout << "add w0, w1, w2" << std::endl;
  l_ins = l_gen.add_shifted_reg(gpr_t::w0,
                                gpr_t::w1,
                                gpr_t::w2,
                                0,
                                0);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  std::cout << "" << std::endl;
}

void fmla_instructions( uint32_t &l_ins,
  std::string &l_str,
  mini_jit::InstGen &l_gen )
{
  std::cout << "#############################" << std::endl;
  std::cout << "### fmla instructions ###" << std::endl;
  std::cout << "#############################" << std::endl;

  // FMLA (vector)
  std::cout << "fmla v0.2s, v1.2s, v2.2s" << std::endl;
  l_ins = l_gen.neon_dp_fmla_vector(simd_fp_t::v0,
                                    simd_fp_t::v1,
                                    simd_fp_t::v2,
                                    arr_spec_t::s2);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  std::cout << "" << std::endl;

  // FMLA (by element)
  std::cout << "fmla v0.4s, v1.4s, v2.[0]" << std::endl;
  l_ins = l_gen.neon_vec_dp_fmla_by_element(simd_fp_t::v0,
                                            simd_fp_t::v1,
                                            simd_fp_t::v2,
                                            arr_spec_t::s4);
  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  std::cout << "" << std::endl;
}

int main()
{
  mini_jit::InstGen l_gen;
  uint32_t l_ins = 0;
  std::string l_str;

  // // CBNZ
  // std::cout << "cbnz w0, #0" << std::endl;
  // l_ins = l_gen.base_br_cbnz(gpr_t::w0,
  //                            0x0);
  // l_str = l_gen.to_string_hex(l_ins);
  // std::cout << " " << l_str << std::endl;
  // l_str = l_gen.to_string_bin(l_ins);
  // std::cout << " " << l_str << std::endl;

  // // CBNZ
  // std::cout << "cbnz w5, #-100" << std::endl;
  // l_ins = l_gen.base_br_cbnz(gpr_t::w5,
  //                            -25);
  // l_str = l_gen.to_string_hex(l_ins);
  // std::cout << " " << l_str << std::endl;
  // l_str = l_gen.to_string_bin(l_ins);
  // std::cout << " " << l_str << std::endl;

  // // MOV (register)
  // std::cout << "mov w1, w0" << std::endl;
  // l_ins = l_gen.mov_reg(gpr_t::w1,
  //                       gpr_t::w0);
  // l_str = l_gen.to_string_hex(l_ins);
  // std::cout << " " << l_str << std::endl;
  // l_str = l_gen.to_string_bin(l_ins);
  // std::cout << " " << l_str << std::endl;

  // // MOV (immediate)
  // std::cout << "mov x0, #5" << std::endl;
  // l_ins = l_gen.mov_imm(gpr_t::x0,
  //                       5);
  // l_str = l_gen.to_string_hex(l_ins);
  // std::cout << " " << l_str << std::endl;
  // l_str = l_gen.to_string_bin(l_ins);
  // std::cout << " " << l_str << std::endl;

  base_load_instructions(l_ins, l_str, l_gen);
  neon_load_instructions(l_ins, l_str, l_gen);

  base_store_instructions(l_ins, l_str, l_gen);
  neon_store_instructions(l_ins, l_str, l_gen);

  math_instructions(l_ins, l_str, l_gen);
  fmla_instructions(l_ins, l_str, l_gen);

  // // FMLA (vector)
  // std::cout << "fmla v16.2s, v29.2s, v2.2s" << std::endl;
  // l_ins = l_gen.neon_dp_fmla_vector(simd_fp_t::v16,
  //                                   simd_fp_t::v29,
  //                                   simd_fp_t::v2,
  //                                   arr_spec_t::s2);
  // l_str = l_gen.to_string_hex(l_ins);
  // std::cout << " " << l_str << std::endl;
  // l_str = l_gen.to_string_bin(l_ins);
  // std::cout << " " << l_str << std::endl;

  // // FMLA (vector)
  // std::cout << "fmla v5.2d, v3.2d, v22.2d" << std::endl;
  // l_ins = l_gen.neon_dp_fmla_vector(simd_fp_t::v5,
  //                                   simd_fp_t::v3,
  //                                   simd_fp_t::v22,
  //                                   arr_spec_t::d2);
  // l_str = l_gen.to_string_hex(l_ins);
  // std::cout << " " << l_str << std::endl;
  // l_str = l_gen.to_string_bin(l_ins);
  // std::cout << " " << l_str << std::endl;

  // // FMLA (vector)
  // std::cout << "fmla v9.4s, v31.4s, v1.4s" << std::endl;
  // l_ins = l_gen.neon_dp_fmla_vector(simd_fp_t::v9,
  //                                   simd_fp_t::v31,
  //                                   simd_fp_t::v1,
  //                                   arr_spec_t::s4);
  // l_str = l_gen.to_string_hex(l_ins);
  // std::cout << " " << l_str << std::endl;
  // l_str = l_gen.to_string_bin(l_ins);
  // std::cout << " " << l_str << std::endl;

  return EXIT_SUCCESS;
}