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
                              wzr,// 11111
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
  // Determine whether the destination register is 64-bit or 32-bit
  // Bit 5 (0x20) of the register ID typically indicates the register width
  bool is64bit = (reg_dest & 0x20) != 0;

  // Try to encode the immediate value using a single MOVZ instruction.
  // MOVZ allows placing a 16-bit immediate at bit positions 0, 16, 32, or 48.
  for (int shift = 0; shift < (is64bit ? 64 : 32); shift += 16) {
    // Check if the immediate fits entirely within one 16-bit field at the given shift.
    // ~(0xFFFFULL << shift) creates a mask that zeros out the 16-bit field we're targeting,
    // and leaves 1s elsewhere.
    // If ANDing with this mask results in zero, it means the rest of the bits are zero.
    if ((imm & ~(0xFFFFULL << shift)) == 0) {
      // Extract the 16-bit portion of the immediate that we want to encode
      uint16_t imm16 = (imm >> shift) & 0xFFFF;

      // Emit a MOVZ instruction with that 16-bit value and the appropriate left shift
      return movz(reg_dest, imm16, shift);
    }
  }

  // If we get here, the immediate value could not be encoded using a single MOVZ.
  // Fail fast for now, until full MOVZ+MOVK support is implemented
  assert(false && "Immediate not representable with single MOVZ");
  return 0;
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