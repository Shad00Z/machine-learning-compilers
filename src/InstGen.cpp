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
                              wzr, // 11111
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
  uint32_t l_imm = imm16 & 0xffff;
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
  uint32_t l_imm = (imm >> scaleShift) & 0xfff;

  // set 12 bit immediate value
  l_ins |= l_imm << 10;
  return l_ins;
}

uint32_t mini_jit::InstGen::ldp_help(gpr_t reg_dest1,
                                     gpr_t reg_dest2,
                                     gpr_t reg_src,
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