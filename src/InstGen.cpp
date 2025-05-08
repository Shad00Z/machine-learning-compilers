#include "InstGen.h"
#include <sstream>
#include <iomanip>
#include <bitset>
#include <cassert>

std::string mini_jit::InstGen::to_string_hex(uint32_t inst)
{
  std::stringstream l_ss;
  l_ss << "0x" << std::hex
       << std::setfill('0')
       << std::setw(8)
       << inst;

  return l_ss.str();
}

std::string mini_jit::InstGen::to_string_bin(uint32_t inst)
{
  std::string l_res = "0b";
  l_res += std::bitset<32>(inst).to_string();

  return l_res;
}

uint32_t mini_jit::InstGen::ret()
{
  return 0xd65f03c0;
}

uint32_t mini_jit::InstGen::base_br_cbnz(gpr_t reg,
                                         int32_t imm19)
{
  uint32_t l_ins = 0x35000000;

  // set register id
  uint32_t l_reg_id = reg & 0x1f;
  l_ins |= l_reg_id;

  // set size of the register
  uint32_t l_reg_size = reg & 0x20;
  l_ins |= l_reg_size << (32 - 6);

  // set immediate
  uint32_t l_imm = imm19 & 0x7ffff;
  l_ins |= l_imm << 5;

  return l_ins;
}

uint32_t mini_jit::InstGen::base_orr_shifted_reg(gpr_t reg_dest,
                                                 gpr_t reg_src1,
                                                 gpr_t reg_src2,
                                                 uint32_t shift,
                                                 uint32_t amount)
{
  uint32_t l_ins = 0x2a000000;

  // set sf
  uint32_t l_sf = reg_dest & 0x20;
  l_ins |= l_sf << 26;

  // set destination register id
  uint32_t l_reg_id = reg_dest & 0x1f;
  l_ins |= l_reg_id;

  // set first source register id
  l_reg_id = reg_src1 & 0x1f;
  l_ins |= l_reg_id << 5;

  // set amount to shift
  uint32_t l_amount = amount & 0x3f;
  l_ins |= l_amount << 10;

  // set second source register id
  l_reg_id = reg_src2 & 0x1f;
  l_ins |= l_reg_id << 16;

  // set shift value
  uint32_t l_shift = shift & 0x3;
  l_ins |= l_shift << 22;

  return l_ins;
}

uint32_t mini_jit::InstGen::mov_reg(gpr_t reg_dest,
                                    gpr_t reg_src)
{

  return base_orr_shifted_reg(reg_dest,
                              wzr,
                              reg_src,
                              0x0,
                              0x0);
}

uint32_t mini_jit::InstGen::movz(gpr_t reg_dest,
                                 uint16_t imm16,
                                 uint32_t shift)
{
  uint32_t l_ins = 0x52800000;

  // set sf
  uint32_t l_sf = reg_dest & 0x20;
  l_ins |= l_sf << 26;

  // set destination register id
  uint32_t l_reg_id = reg_dest & 0x1f;
  l_ins |= l_reg_id;

  // set immediate value
  uint32_t l_imm = imm16 & 0xFFFF;
  l_ins |= l_imm << 5;

  // set shift value
  uint32_t l_shift = shift & 0x3;
  l_ins |= l_shift << 21;

  return l_ins;
}

uint32_t mini_jit::InstGen::mov_imm(gpr_t reg_dest, uint64_t imm)
{
  bool is64bit = (reg_dest & 0x20) != 0;

  // movz allows placing a 16-bit immediate at bit positions 0, 16, 32, or 48.
  for (int shift = 0; shift < (is64bit ? 64 : 32); shift += 16)
  {
    // Check if the immediate fits entirely within one 16-bit field at the given shift.
    // ~(0xFFFFULL << shift) creates a mask that zeros out the 16-bit field we're targeting,
    // and leaves 1s elsewhere.
    // If ANDing with this mask results in zero, it means the rest of the bits are zero.
    if ((imm & ~(0xFFFFULL << shift)) == 0)
    {
      // Extract the 16-bit portion of the immediate that we want to encode
      uint16_t imm16 = (imm >> shift) & 0xFFFF;
      return movz(reg_dest, imm16, shift);
    }
  }

  // immediate value could not be encoded using a single MOVZ
  // need to implement MOVZ+MOVK support for larger immediates
  throw std::invalid_argument("Immediate too large for a single MOVZ");
  return 0;
}

uint32_t mini_jit::InstGen::base_ldr_imm_uoff(gpr_t reg_dest,
                                              gpr_t reg_src,
                                              uint32_t imm)
{
  uint32_t l_ins = 0xB9400000;

  // set size
  uint32_t l_sf = reg_dest & 0x20;
  l_ins |= l_sf << 25; // set bit 30

  // set destination register id
  uint32_t l_reg_id = reg_dest & 0x1f;
  l_ins |= l_reg_id;

  // set first source register id
  l_reg_id = reg_src & 0x1f;
  l_ins |= l_reg_id << 5;

  // check if immediate can be encoded
  uint32_t scale = (l_sf) ? 8 : 4;
  if (imm % scale != 0)
  {
    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)");
  }

  // scale the immediate for encoding (right-shift)
  uint32_t scaleShift = (l_sf) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
  uint32_t l_imm = (imm >> scaleShift) & 0xFFF;

  // set 12 bit immediate value
  l_ins |= l_imm << 10;
  return l_ins;
}

uint32_t mini_jit::InstGen::base_ldp_soff(gpr_t reg_dest1,
                                          gpr_t reg_dest2,
                                          gpr_t reg_src,
                                          int32_t imm7)
{
  uint32_t l_sf1 = reg_dest1 & 0x20;
  uint32_t l_sf2 = reg_dest2 & 0x20;
  if (l_sf1 != l_sf2)
  {
    throw std::invalid_argument("LDP: both destination registers must be of the same size");
  }

  // check if immediate can be encoded
  uint32_t l_scale = (l_sf1) ? 8 : 4;
  if (imm7 % l_scale != 0)
  {
    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)");
  }

  // scale the immediate for encoding (right-shift)
  uint32_t l_scaleShift = (l_sf1) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
  uint32_t l_imm = (imm7 >> l_scaleShift) & 0x7f;

  // set op code
  uint32_t l_opc = (l_sf1) ? 0x2 : 0x0;

  // encoding: 0010
  uint32_t l_encoding = 0x2;

  return ldp_help(reg_dest1,
                  reg_dest2,
                  reg_src,
                  l_imm,
                  l_opc,
                  l_encoding);
}

uint32_t mini_jit::InstGen::base_ldp_post(gpr_t reg_dest1,
                                          gpr_t reg_dest2,
                                          gpr_t reg_src,
                                          int32_t imm7)
{
  uint32_t l_sf1 = reg_dest1 & 0x20;
  uint32_t l_sf2 = reg_dest2 & 0x20;
  if (l_sf1 != l_sf2)
  {
    throw std::invalid_argument("LDP: both destination registers must be of the same size");
  }

  // check if immediate can be encoded
  uint32_t l_scale = (l_sf1) ? 8 : 4;
  if (imm7 % l_scale != 0)
  {
    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)");
  }

  // scale the immediate for encoding (right-shift)
  uint32_t l_scaleShift = (l_sf1) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
  uint32_t l_imm = (imm7 >> l_scaleShift) & 0x7f;

  // set op code
  uint32_t l_opc = (l_sf1) ? 0x2 : 0x0;

  // encoding: 0001
  uint32_t l_encoding = 0x1;

  return ldp_help(reg_dest1,
                  reg_dest2,
                  reg_src,
                  l_imm,
                  l_opc,
                  l_encoding);
}

uint32_t mini_jit::InstGen::base_ldp_pre(gpr_t reg_dest1,
                                         gpr_t reg_dest2,
                                         gpr_t reg_src,
                                         int32_t imm7)
{
  uint32_t l_sf1 = reg_dest1 & 0x20;
  uint32_t l_sf2 = reg_dest2 & 0x20;
  if (l_sf1 != l_sf2)
  {
    throw std::invalid_argument("LDP: both destination registers must be of the same size");
  }

  // check if immediate can be encoded
  uint32_t l_scale = (l_sf1) ? 8 : 4;
  if (imm7 % l_scale != 0)
  {
    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)");
  }

  // scale the immediate for encoding (right-shift)
  uint32_t l_scaleShift = (l_sf1) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
  uint32_t l_imm = (imm7 >> l_scaleShift) & 0x7f;

  // set op code
  uint32_t l_opc = (l_sf1) ? 0x2 : 0x0;

  // encoding: 0011
  uint32_t l_encoding = 0x3;

  return ldp_help(reg_dest1,
                  reg_dest2,
                  reg_src,
                  l_imm,
                  l_opc,
                  l_encoding);
}

uint32_t mini_jit::InstGen::neon_ldr_imm_uoff(simd_fp_t reg_dest,
                                              gpr_t reg_src,
                                              uint32_t imm12,
                                              neon_size_spec_t size_spec)
{
  // check if immediate can be encoded
  uint32_t l_scale = (size_spec == neon_size_spec_t::s) ? 4 : (size_spec == neon_size_spec_t::d) ? 8
                                                      : 16;
  if (imm12 % l_scale != 0)
  {
    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit) or 16 (128-bit)");
  }

  // scale the immediate for encoding (right-shift)
  uint32_t l_scaleShift = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                            : 4;
  uint32_t l_imm = (imm12 >> l_scaleShift) & 0xfff;

  // set op code
  uint32_t l_opc_pre = (size_spec == neon_size_spec_t::q) ? 3 : 1;
  uint32_t l_opc = l_opc_pre & 0x3;

  // encoding: 101
  uint32_t l_encoding = 0x5;

  return ldr_help( reg_dest,
                   reg_src,
                   l_imm,
                   l_opc,
                   l_encoding );
}

uint32_t mini_jit::InstGen::neon_ldp_soff(simd_fp_t reg_dest1,
                                          simd_fp_t reg_dest2,
                                          gpr_t reg_src,
                                          int32_t imm7,
                                          neon_size_spec_t size_spec)
{
  // check if immediate can be encoded
  uint32_t l_scale = (size_spec == neon_size_spec_t::s) ? 4 : (size_spec == neon_size_spec_t::d) ? 8
                                                                                                 : 16;
  if (imm7 % l_scale != 0)
  {
    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit) or 16 (128-bit)");
  }

  // scale the immediate for encoding (right-shift)
  uint32_t l_scaleShift = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                                                                      : 4;
  uint32_t l_imm = (imm7 >> l_scaleShift) & 0x7f;

  // set op code
  uint32_t l_opc = size_spec & 0x3;

  // encoding: 1010
  uint32_t l_encoding = 0xA;

  return ldp_help(reg_dest1,
                  reg_dest2,
                  reg_src,
                  l_imm,
                  l_opc,
                  l_encoding);
}

uint32_t mini_jit::InstGen::neon_ldp_post(simd_fp_t reg_dest1,
                                          simd_fp_t reg_dest2,
                                          gpr_t reg_src,
                                          int32_t imm7,
                                          neon_size_spec_t size_spec)
{
  // check if immediate can be encoded
  uint32_t l_scale = (size_spec == neon_size_spec_t::s) ? 4 : (size_spec == neon_size_spec_t::d) ? 8
                                                                                                 : 16;
  if (imm7 % l_scale != 0)
  {
    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit) or 16 (128-bit)");
  }

  // scale the immediate for encoding (right-shift)
  uint32_t l_scaleShift = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                                                                      : 4;
  uint32_t l_imm = (imm7 >> l_scaleShift) & 0x7f;

  // set op code
  uint32_t l_opc = size_spec & 0x3;

  // encoding: 1001
  uint32_t l_encoding = 0x9;

  return ldp_help(reg_dest1,
                  reg_dest2,
                  reg_src,
                  l_imm,
                  l_opc,
                  l_encoding);
}

uint32_t mini_jit::InstGen::neon_ldp_pre(simd_fp_t reg_dest1,
                                         simd_fp_t reg_dest2,
                                         gpr_t reg_src,
                                         int32_t imm7,
                                         neon_size_spec_t size_spec)
{
  // check if immediate can be encoded
  uint32_t l_scale = (size_spec == neon_size_spec_t::s) ? 4 : (size_spec == neon_size_spec_t::d) ? 8
                                                                                                 : 16;
  if (imm7 % l_scale != 0)
  {
    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit) or 16 (128-bit)");
  }

  // scale the immediate for encoding (right-shift)
  uint32_t l_scaleShift = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                                                                      : 4;
  uint32_t l_imm = (imm7 >> l_scaleShift) & 0x7f;

  // set op code
  uint32_t l_opc = size_spec & 0x3;

  // encoding: 1011
  uint32_t l_encoding = 0xB;

  return ldp_help(reg_dest1,
                  reg_dest2,
                  reg_src,
                  l_imm,
                  l_opc,
                  l_encoding);
}

uint32_t mini_jit::InstGen::neon_dp_fmla_vector(simd_fp_t reg_dest,
                                                simd_fp_t reg_src1,
                                                simd_fp_t reg_src2,
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

uint32_t mini_jit::InstGen::neon_vec_dp_fmla_by_element( simd_fp_t reg_dest,
                                                         simd_fp_t reg_src1,
                                                         simd_fp_t reg_src2,
                                                         arr_spec_t arr_spec )
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

uint32_t mini_jit::InstGen::ldp_help(uint32_t reg_dest1,
  uint32_t reg_dest2,
  uint32_t reg_src,
  int32_t imm7,
  uint32_t opc,
  uint32_t encoding)
{
  // LDP without VR
  uint32_t l_ins = 0x28400000;

  // set 2-bit opc
  l_ins |= (opc & 0x3) << 30;

  // set 4-bit VR encoding
  l_ins |= (encoding & 0xF) << 23;

  // set first destination register
  uint32_t l_reg_id = reg_dest1 & 0x1f;
  l_ins |= l_reg_id;
  // set source register
  l_reg_id = reg_src & 0x1f;
  l_ins |= l_reg_id << 5;
  // set second destination register
  l_reg_id = reg_dest2 & 0x1f;
  l_ins |= l_reg_id << 10;
  // set immediate value
  uint32_t l_imm = imm7 & 0x7f;
  l_ins |= l_imm << 15;

  return l_ins;
}

/*
 * Store Base instructions
 */
uint32_t mini_jit::InstGen::base_str_imm_uoff(gpr_t reg_dest,
                                              gpr_t reg_src,
                                              uint32_t imm)
{
  uint32_t l_ins = 0xB9000000;

  // set size
  uint32_t l_sf = reg_dest & 0x20;
  l_ins |= l_sf << 25; // set bit 30

  // set destination register id
  uint32_t l_reg_id = reg_dest & 0x1f;
  l_ins |= l_reg_id;

  // set first source register id
  l_reg_id = reg_src & 0x1f;
  l_ins |= l_reg_id << 5;

  // check if immediate can be encoded
  uint32_t scale = (l_sf) ? 8 : 4;
  if (imm % scale != 0)
  {
    throw std::invalid_argument( "Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)" );
  }

  // scale the immediate for encoding (right-shift)
  uint32_t scaleShift = (l_sf) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
  uint32_t l_imm = (imm >> scaleShift) & 0xFFF;

  // set 12 bit immediate value
  l_ins |= l_imm << 10;
  return l_ins;
}

uint32_t mini_jit::InstGen::base_stp_pre(gpr_t reg_dest1,
                                         gpr_t reg_dest2,
                                         gpr_t reg_src,
                                         int32_t imm7)
{
  uint32_t l_sf1 = reg_dest1 & 0x20;
  uint32_t l_sf2 = reg_dest2 & 0x20;
  if (l_sf1 != l_sf2)
  {
    throw std::invalid_argument("STP: both destination registers must be of the same size");
  }

  // check if immediate can be encoded
  uint32_t l_scale = (l_sf1) ? 8 : 4;
  if (imm7 % l_scale != 0)
  {
    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)");
  }

  // scale the immediate for encoding (right-shift)
  uint32_t l_scaleShift = (l_sf1) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
  uint32_t l_imm = (imm7 >> l_scaleShift) & 0x7f;

  // set op code
  uint32_t l_opc = (l_sf1) ? 0x2 : 0x0;

  // encoding: 0011
  uint32_t l_encoding = 0x3;

  return stp_help(reg_dest1,
                  reg_dest2,
                  reg_src,
                  l_imm,
                  l_opc,
                  l_encoding);
}

uint32_t mini_jit::InstGen::base_stp_post(gpr_t   reg_dest1,
                                          gpr_t   reg_dest2,
                                          gpr_t   reg_src,
                                          int32_t imm7 )
{
  // Check size of destination registers
  uint32_t l_sf1 = reg_dest1 & 0x20;
  uint32_t l_sf2 = reg_dest2 & 0x20;
  if (l_sf1 != l_sf2)
  {
    throw std::invalid_argument( "STP: both destination registers must be of the same size" );
  }

  // check if immediate can be encoded
  uint32_t l_scale = (l_sf1) ? 8 : 4;
  if (imm7 % l_scale != 0)
  {
    throw std::invalid_argument( "Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)" );
  }

  // scale the immediate for encoding (right-shift)
  uint32_t l_scaleShift = (l_sf1) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
  uint32_t l_imm = (imm7 >> l_scaleShift) & 0x7f;

  // set op code
  uint32_t l_opc = (l_sf1) ? 0x2 : 0x0;

  // encoding: 0001
  uint32_t l_encoding = 0x1;

  return stp_help(reg_dest1,
                  reg_dest2,
                  reg_src,
                  l_imm,
                  l_opc,
                  l_encoding);
}

uint32_t mini_jit::InstGen::base_stp_soff(gpr_t reg_dest1,
                                          gpr_t reg_dest2,
                                          gpr_t reg_src,
                                          int32_t imm7)
{
  uint32_t l_sf1 = reg_dest1 & 0x20;
  uint32_t l_sf2 = reg_dest2 & 0x20;
  if (l_sf1 != l_sf2)
  {
    throw std::invalid_argument("STP: both destination registers must be of the same size");
  }

  // check if immediate can be encoded
  uint32_t l_scale = (l_sf1) ? 8 : 4;
  if (imm7 % l_scale != 0)
  {
    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)");
  }

  // scale the immediate for encoding (right-shift)
  uint32_t l_scaleShift = (l_sf1) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
  uint32_t l_imm = (imm7 >> l_scaleShift) & 0x7f;

  // set op code
  uint32_t l_opc = (l_sf1) ? 0x2 : 0x0;

  // encoding: 0010
  uint32_t l_encoding = 0x2;

  return stp_help( reg_dest1,
                   reg_dest2,
                   reg_src,
                   l_imm,
                   l_opc,
                   l_encoding );
}

/*
 * Store Neon instructions
 */
uint32_t mini_jit::InstGen::neon_stp_post(simd_fp_t reg_dest1,
                                          simd_fp_t reg_dest2,
                                          gpr_t reg_src,
                                          int32_t imm7,
                                          neon_size_spec_t size_spec)
{
  // check if immediate can be encoded
  uint32_t l_scale = (size_spec == neon_size_spec_t::s) ? 4 : (size_spec == neon_size_spec_t::d) ? 8
                                                          : 16;
  if (imm7 % l_scale != 0)
  {
    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit) or 16 (128-bit)");
  }

  // scale the immediate for encoding (right-shift)
  uint32_t l_scaleShift = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                              : 4;
  uint32_t l_imm = (imm7 >> l_scaleShift) & 0x7f;

  // set op code
  uint32_t l_opc = size_spec & 0x3;

  // encoding: 1001
  uint32_t l_encoding = 0x9;

  return stp_help(reg_dest1,
                  reg_dest2,
                  reg_src,
                  l_imm,
                  l_opc,
                  l_encoding);
}

uint32_t mini_jit::InstGen::neon_stp_pre(simd_fp_t reg_dest1,
                                         simd_fp_t reg_dest2,
                                         gpr_t reg_src,
                                         int32_t imm7,
                                         neon_size_spec_t size_spec)
{
  // check if immediate can be encoded
  uint32_t l_scale = (size_spec == neon_size_spec_t::s) ? 4 : (size_spec == neon_size_spec_t::d) ? 8
                                                          : 16;
  if (imm7 % l_scale != 0)
  {
    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit) or 16 (128-bit)");
  }

  // scale the immediate for encoding (right-shift)
  uint32_t l_scaleShift = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                              : 4;
  uint32_t l_imm = (imm7 >> l_scaleShift) & 0x7f;

  // set op code
  uint32_t l_opc = size_spec & 0x3;

  // encoding: 1011
  uint32_t l_encoding = 0xB;

  return stp_help(reg_dest1,
                  reg_dest2,
                  reg_src,
                  l_imm,
                  l_opc,
                  l_encoding );
}

uint32_t mini_jit::InstGen::neon_stp_soff(simd_fp_t reg_dest1,
                                          simd_fp_t reg_dest2,
                                          gpr_t reg_src,
                                          int32_t imm7,
                                          neon_size_spec_t size_spec)
{
  // check if immediate can be encoded
  uint32_t l_scale = (size_spec == neon_size_spec_t::s) ? 4 : (size_spec == neon_size_spec_t::d) ? 8
                                                          : 16;
  if (imm7 % l_scale != 0)
  {
    throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit) or 16 (128-bit)");
  }

  // scale the immediate for encoding (right-shift)
  uint32_t l_scaleShift = (size_spec == neon_size_spec_t::s) ? 2 : (size_spec == neon_size_spec_t::d) ? 3
                                                                                                      : 4;
  uint32_t l_imm = (imm7 >> l_scaleShift) & 0x7f;

  // set op code
  uint32_t l_opc = size_spec & 0x3;

  // encoding: 1010
  uint32_t l_encoding = 0xA;

  return stp_help(reg_dest1,
                  reg_dest2,
                  reg_src,
                  l_imm,
                  l_opc,
                  l_encoding);
}

uint32_t mini_jit::InstGen::stp_help(uint32_t reg_dest1,
                                     uint32_t reg_dest2,
                                     uint32_t reg_src,
                                     int32_t imm7,
                                     uint32_t opc,
                                     uint32_t encoding)
{
  // LDP without VR - bits: 29 = 1, 27 = 1
  uint32_t l_ins = 0x28000000;

  // set 2-bit opc
  l_ins |= (opc & 0x3) << 30;

  // set 4-bit VR encoding
  l_ins |= (encoding & 0xF) << 23;

  // set first destination register
  uint32_t l_reg_id = reg_dest1 & 0x1f;
  l_ins |= l_reg_id;
  // set source register
  l_reg_id = reg_src & 0x1f;
  l_ins |= l_reg_id << 5;
  // set second destination register
  l_reg_id = reg_dest2 & 0x1f;
  l_ins |= l_reg_id << 10;
  // set immediate value
  uint32_t l_imm = imm7 & 0x7f;
  l_ins |= l_imm << 15;

  return l_ins;
}

uint32_t mini_jit::InstGen::ldr_help(uint32_t reg_dest,
                                     uint32_t reg_src,
                                     int32_t imm12,
                                     uint32_t opc,
                                     uint32_t encoding)
{
  // LDP without VR - bits: 29 = 1, 28 = 1, 27 = 1
  uint32_t l_ins = 0x38000000;

  // set 2-bit opc
  l_ins |= (opc & 0x3) << 22;

  // set 3-bit VR encoding
  l_ins |= (encoding & 0x7) << 24;

  // set first destination register
  uint32_t l_reg_id = reg_dest & 0x1f;
  l_ins |= l_reg_id;

  // set source register
  l_reg_id = reg_src & 0x1f;
  l_ins |= l_reg_id << 5;

  // set immediate value
  uint32_t l_imm = imm12 & 0xFFF;
  l_ins |= l_imm << 10;

  return l_ins;
}

/*
 * Math instructions
 */
uint32_t mini_jit::InstGen::mul_reg( gpr_t reg_dest,
  gpr_t reg_src1,
  gpr_t reg_src2 )
{
uint32_t l_ins = 0x1B007C00;

uint32_t l_sf1 = reg_src1 & 0x20;
uint32_t l_sf2 = reg_src2 & 0x20;
uint32_t l_sf_dest = reg_dest & 0x20;
if (l_sf1 != l_sf2)
{
throw std::invalid_argument("MUL: both source registers must be of the same size");
}
else if ( l_sf1 != l_sf_dest )
{
throw std::invalid_argument("MUL: destination register must be of the same size as source registers");
}

// set size
uint32_t l_sf = reg_dest & 0x20;
l_ins |= l_sf << 26; // set bit 31

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

uint32_t mini_jit::InstGen::add_shifted_reg(gpr_t reg_dest,
                                            gpr_t reg_src1,
                                            gpr_t reg_src2,
                                            uint32_t imm6,
                                            uint32_t shift)
{
  uint32_t l_ins = 0xB000000;

  uint32_t l_sf1 = reg_src1 & 0x20;
  uint32_t l_sf2 = reg_src2 & 0x20;
  uint32_t l_sf_dest = reg_dest & 0x20;
  if (l_sf1 != l_sf2)
  {
    throw std::invalid_argument("ADD: both source registers must be of the same size");
  }
  else if ( l_sf1 != l_sf_dest )
  {
    throw std::invalid_argument("ADD: destination register must be of the same size as source registers");
  }

  // set size
  uint32_t l_sf = reg_dest & 0x20;
  l_ins |= l_sf << 26; // set bit 31

  // set immediate value
  uint32_t l_imm = imm6 & 0x3f;
  l_ins |= l_imm << 10;

  // set destination register id
  uint32_t l_reg_id = reg_dest & 0x1f;
  l_ins |= l_reg_id;

  // set first source register id
  l_reg_id = reg_src1 & 0x1f;
  l_ins |= l_reg_id << 5;

  // set second source register id
  l_reg_id = reg_src2 & 0x1f;
  l_ins |= l_reg_id << 16;

  // set shift value
  uint32_t l_shift = shift & 0x3;
  l_ins |= l_shift << 22;

  return l_ins;
}

uint32_t mini_jit::InstGen::mov_sp(gpr_t reg_dest,
                                    gpr_t reg_src)
{
    return add_immediate(reg_dest,
                         reg_src,
                         0,
                         0);
}

uint32_t mini_jit::InstGen::add_immediate(gpr_t reg_dest,
                                          gpr_t reg_src,
                                          uint32_t imm12,
                                          uint32_t shift)
{
  uint32_t l_ins = 0x11000000;

  // set size
  uint32_t l_sf = reg_dest & 0x20;
  l_ins |= l_sf << 26; // set bit 31

  // set destination register id
  uint32_t l_reg_id = reg_dest & 0x1f;
  l_ins |= l_reg_id;

  // set first source register id
  l_reg_id = reg_src & 0x1f;
  l_ins |= l_reg_id << 5;

  // set immediate value
  uint32_t l_imm = imm12 & 0xfff;
  l_ins |= l_imm << 10;

  // set shift value
  uint32_t l_shift = shift & 0x1;
  l_ins |= l_shift << 22;

  return l_ins;
}