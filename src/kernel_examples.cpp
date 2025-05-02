#include "Kernel.h"
#include <iostream>

/**
 * @brief Kernel that returns 5.
 */
void example_0() {
  std::cout << "example_0" << std::endl;

  mini_jit::Kernel l_kernel;
  l_kernel.add_instr( 0xd28000a0 ); // mov x0, #5
  l_kernel.add_instr( 0xd65f03c0 ); // ret
  l_kernel.write( "example_0.bin" );
  l_kernel.set_kernel();

  int64_t (* l_func)() = nullptr;
  l_func = (int64_t (*)()) l_kernel.get_kernel();
  std::cout << "  result: " << l_func() << std::endl;
}

/**
 * @brief Kernel that uses the given immediate
 *        to set the return value.
 * @param imm16 16-bit immediate value.
 */
void example_1( int16_t imm16 ) {
  std::cout << "example_1" << std::endl;

  mini_jit::Kernel l_kernel;
  uint32_t l_ins = 0xd2800000;
  l_ins |= imm16 << 5;
  l_kernel.add_instr( l_ins ); // mov x0, #imm16
  l_kernel.add_instr( 0xd65f03c0 ); // ret
  l_kernel.write( "example_1.bin" );
  l_kernel.set_kernel();

  int64_t (* l_func)() = nullptr;
  l_func = (int64_t (*)()) l_kernel.get_kernel();
  std::cout << "  result: " << l_func() << std::endl;
}

/**
 * @brief Kernel that adds 5 to the passed value.
 */
void example_2() {
  std::cout << "example_2" << std::endl;

  mini_jit::Kernel l_kernel;
  l_kernel.add_instr( 0x91001400 ); // add x0, x0, #5
  l_kernel.add_instr( 0xd65f03c0 ); // ret
  l_kernel.write( "example_2.bin" );
  l_kernel.set_kernel();

  int64_t (* l_func)( int64_t ) = nullptr;
  l_func = (int64_t (*)( int64_t )) l_kernel.get_kernel();
  std::cout << "  result: " << l_func( 7 ) << std::endl;
}

/**
 * @brief Kernel that contains a loop.
 */
void example_3() {
  std::cout << "example_3" << std::endl;

  mini_jit::Kernel l_kernel;
  l_kernel.add_instr( 0xd2804000 ); // mov x0, #512
  l_kernel.add_instr( 0xd2800001 ); // mov x1, #0
  l_kernel.add_instr( 0xd1000400 ); // sub x0, x0, #1
  l_kernel.add_instr( 0x91000821 ); // add x1, x1, #2
  l_kernel.add_instr( 0xb5ffffc0 ); // cbnz x0, #-8
  l_kernel.add_instr( 0xaa0103e0 ); // mov x0, x1
  l_kernel.add_instr( 0xd65f03c0 ); // ret
  l_kernel.write( "example_3.bin" );
  l_kernel.set_kernel();

  int64_t (* l_func)() = nullptr;
  l_func = (int64_t (*)()) l_kernel.get_kernel();
  std::cout << "  result: " << l_func() << std::endl;
}

int main() {
  std::cout << "###########################" << std::endl;
  std::cout << "### welcome to mini_jit ###" << std::endl;
  std::cout << "###########################" << std::endl;

  example_0();
  example_1( 25 );
  example_2();
  example_3();

  return 0;
}