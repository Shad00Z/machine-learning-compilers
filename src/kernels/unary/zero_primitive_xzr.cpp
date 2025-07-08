#include <mlc/Kernel.h>
#include <mlc/instructions/all_instructions.h>
#include <mlc/kernels/unary/zero_primitive_xzr.h>
#include <mlc/registers/gp_registers.h>
#include <mlc/registers/simd_fp_registers.h>

using enum gpr_t;
using namespace mini_jit::instructions::base;

void mini_jit::kernels::unary::zero_xzr(mini_jit::Kernel& kernel,
                                        u_int32_t         m,
                                        u_int32_t         n,
                                        u_int32_t         trans_b)
{
    // Inputs:
    // x0: pointer to A
    // x1: pointer to B
    // x2: leading dimension of A
    // x3: leading dimension of B

    if (1 == trans_b)
    {
        u_int32_t mTemp = m;
        m               = n;
        n               = mTemp;
    }

    // Prepare the kernel
    u_int32_t mLoopIterations = m / 8;
    u_int32_t mLoopRemainder  = m % 8;

    // PCS
    kernel.add_instr(stpPre(x29, x30, sp, -16));
    kernel.add_instr(movSP(x29, sp));

    // Compute stride for B
    kernel.add_instr(lsl(x3, x3, 2));

    // Save pase matrix pointer
    kernel.add_instr(mov(x4, x1)); // B

    // Set n loop counter
    kernel.add_instr(mov(x5, n));

    // Start n loop (1 column)
    kernel.add_label("n_loop");

    kernel.add_instr(mov(x6, mLoopIterations));
    // working pointer for B (rows)
    kernel.add_instr(mov(x7, x4));

    if (mLoopIterations > 0)
    {
        kernel.add_label("m_8_loop");
        // store 8 zeros
        kernel.add_instr(mov(x8, x7));
        kernel.add_instr(strPost(xzr, x8, 8));
        kernel.add_instr(strPost(xzr, x8, 8));
        kernel.add_instr(strPost(xzr, x8, 8));
        kernel.add_instr(str(xzr, x8, 0));

        // jump by 8 rows
        kernel.add_instr(add(x7, x7, 8 * 4, 0));
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
            kernel.add_instr(str(wzr, x7, 0));
            break;
        case 2:
            kernel.add_instr(str(xzr, x7, 0));
            break;
        case 3:
            kernel.add_instr(strPost(xzr, x7, 8));
            kernel.add_instr(str(wzr, x7, 0));
            break;
        case 4:
            kernel.add_instr(strPost(xzr, x7, 8));
            kernel.add_instr(str(xzr, x7, 0));
            break;
        case 5:
            kernel.add_instr(strPost(xzr, x7, 8));
            kernel.add_instr(strPost(xzr, x7, 8));
            kernel.add_instr(str(wzr, x7, 0));
            break;
        case 6:
            kernel.add_instr(strPost(xzr, x7, 8));
            kernel.add_instr(strPost(xzr, x7, 8));
            kernel.add_instr(str(xzr, x7, 0));
            break;
        case 7:
            kernel.add_instr(strPost(xzr, x7, 8));
            kernel.add_instr(strPost(xzr, x7, 8));
            kernel.add_instr(strPost(xzr, x7, 8));
            kernel.add_instr(str(wzr, x7, 0));
            break;
        default:
            break;
        }
    }

    // jump to next column
    kernel.add_instr(add(x4, x4, x3, 0, 0));
    // decrement n loop counter
    kernel.add_instr(sub(x5, x5, 1, 0));
    // check if loop counter is zero
    int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
    kernel.add_instr(cbnz(x5, -l_nLoopInstrCount * 4));

    // Restore stack pointer
    kernel.add_instr(ldpPost(x29, x30, sp, 16));

    kernel.add_instr(ret());
    kernel.write("zero_primitive_xzr.bin");
    kernel.set_kernel();
}