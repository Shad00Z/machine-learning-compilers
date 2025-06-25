#include "add_primitive.h"
#include "Kernel.h"

#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
#include "instructions/all_instructions.h"

using enum gpr_t;
using enum simd_fp_t;
using enum neon_size_spec_t;
using enum arr_spec_t;

using namespace mini_jit::instructions;

void mini_jit::kernels::binary::add(mini_jit::Kernel &kernel,
                                    u_int32_t m,
                                    u_int32_t n)
{
    // Inputs:
    // x0: pointer to A
    // x1: pointer to B
    // x2: pointer to C
    // x3: leading dimension of A
    // x4: leading dimension of B
    // x5: leading dimension of C

    kernel.add_instr(ret());
    kernel.write("add_primitive.bin");
    kernel.set_kernel();
}