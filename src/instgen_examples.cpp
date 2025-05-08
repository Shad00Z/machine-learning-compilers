#include "InstGen.h"
#include <iostream>

using gpr_t = mini_jit::InstGen::gpr_t;
using simd_fp_t = mini_jit::InstGen::simd_fp_t;
using arr_spec_t = mini_jit::InstGen::arr_spec_t;

int main()
{
  mini_jit::InstGen l_gen;
  uint32_t l_ins = 0;
  std::string l_str;

  std::cout << "ldr s28, [x6]" << std::endl;
  l_ins = mini_jit::InstGen::neon_ldr_imm_uoff(simd_fp_t::v28, gpr_t::x6, 0, mini_jit::InstGen::neon_size_spec_t::s);

  l_str = l_gen.to_string_hex(l_ins);
  std::cout << " " << l_str << std::endl;
  l_str = l_gen.to_string_bin(l_ins);
  std::cout << " " << l_str << std::endl;

  return EXIT_SUCCESS;
}