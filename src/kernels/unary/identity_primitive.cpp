#include "identity_primitive.h"
#include "Kernel.h"

#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
#include "instructions/all_instructions.h"

using enum gpr_t;
using enum simd_fp_t;
using enum neon_size_spec_t;
using enum arr_spec_t;

using namespace mini_jit::instructions::base;
using namespace mini_jit::instructions::simd_fp;

void mini_jit::kernels::unary::identity(mini_jit::Kernel &kernel,
                                        u_int32_t m,
                                        u_int32_t n)
{
    // Inputs:
    // x0: pointer to A
    // x1: pointer to B
    // x2: leading dimension of A
    // x3: leading dimension of B

    // Prepare the kernel
    u_int32_t mLoopIterations = m / 8;
    u_int32_t mLoopRemainder = m % 8;

    // PCS
    kernel.add_instr(stpPre(x29, x30, sp, -16));
    kernel.add_instr(movSP(x29, sp));

    // Compute stride for A
    kernel.add_instr(lsl(x2, x2, 2));

    // Compute stride for B
    kernel.add_instr(lsl(x3, x3, 2));

    // Save pase matrix pointer
    kernel.add_instr(mov(x4, x1)); // B
    kernel.add_instr(mov(x10, x0)); // B

    // Set n loop counter
    kernel.add_instr(mov(x5, n));

    // Start n loop (1 column)
    kernel.add_label("n_loop");

    // Set m loop counter
    kernel.add_instr(mov(x6, mLoopIterations));

    // working pointer for B (rows)
    kernel.add_instr(mov(x7, x4)); 

    // working pointer for A (rows)
    kernel.add_instr(mov(x8, x10));

    if (mLoopIterations > 0)
    {
        kernel.add_label("m_8_loop");

        // load and store 8 rows of A and B
        kernel.add_instr(ldp(v0, v1, x8, 0, q));
        kernel.add_instr(stp(v0, v1, x7, 0, q));
        // jump by 8 rows
        kernel.add_instr(add(x8, x8, 8*4, 0));
        kernel.add_instr(add(x7, x7, 8*4, 0));
        // decrement m loop counter
        kernel.add_instr(sub(x6, x6, 1, 0));
        // check if loop counter is zero
        int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_8_loop");
        kernel.add_instr(cbnz(x6, -l_mLoopInstrCount * 4));
    }

    if (mLoopRemainder > 0)
    {
        switch (mLoopRemainder)
        {
        case 1:
            kernel.add_instr(ldr(v0, x8, 0, s));
            kernel.add_instr(str(v0, x7, 0, s));
            break;
        case 2:
            kernel.add_instr(ldr(v0, x8, 0, d));
            kernel.add_instr(str(v0, x7, 0, d));
            break;
        case 3:
            kernel.add_instr(ldrPost(v0, x8, 8, d));
            kernel.add_instr(ldr(v1, x8, 0, s));

            kernel.add_instr(strPost(v0, x7, 8, d));
            kernel.add_instr(str(v1, x7, 0, s));
            break;
        case 4:
            kernel.add_instr(ldr(v0, x8, 0, q));
            kernel.add_instr(str(v0, x7, 0, q));
            break;
        case 5:
            kernel.add_instr(ldrPost(v0, x8, 16, q));
            kernel.add_instr(ldr(v1, x8, 0, s));

            kernel.add_instr(strPost(v0, x7, 16, q));
            kernel.add_instr(str(v1, x7, 0, s));
            break;
        case 6:
            kernel.add_instr(ldrPost(v0, x8, 16, q));
            kernel.add_instr(ldr(v1, x8, 0, d));

            kernel.add_instr(strPost(v0, x7, 16, q));
            kernel.add_instr(str(v1, x7, 0, d));
            break;
        case 7:
            kernel.add_instr(ldrPost(v0, x8, 16, q));
            kernel.add_instr(ldrPost(v1, x8, 8, d));
            kernel.add_instr(ldr(v2, x8, 0, s));

            kernel.add_instr(strPost(v0, x7, 16, q));
            kernel.add_instr(strPost(v1, x7, 8, d));
            kernel.add_instr(str(v2, x7, 0, s));
            break;
        default:
            break;
        }
    }

    // jump to next column
    kernel.add_instr(add(x10, x10, x2, 0, 0));
    kernel.add_instr(add(x4, x4, x3, 0, 0));
    // decrement n loop counter
    kernel.add_instr(sub(x5, x5, 1, 0));
    // check if loop counter is zero
    int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
    kernel.add_instr(cbnz(x5, -l_nLoopInstrCount * 4));

    // Restore stack pointer
    kernel.add_instr(ldpPost(x29, x30, sp, 16));

    kernel.add_instr(ret());
    kernel.write("identity_primitive.bin");
    kernel.set_kernel();
}