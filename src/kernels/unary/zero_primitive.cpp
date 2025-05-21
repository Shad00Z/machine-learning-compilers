#include "zero_primitive.h"
#include "Kernel.h"

#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
#include "instructions/all_instructions.h"

namespace inst = mini_jit::instructions;
namespace base = inst::base;
namespace simd_fp = inst::simd_fp;

void mini_jit::kernels::unary::zero(mini_jit::Kernel &kernel,
                                    u_int32_t m,
                                    u_int32_t n,
                                    u_int32_t trans_b)
{
    // Inputs:
    // x0: pointer to A
    // x1: pointer to B
    // x2: leading dimension of A
    // x3: leading dimension of B

    if(1 == trans_b)
    {
        u_int32_t mTemp = m;
        m = n;
        n = mTemp;
    }

    // Prepare the kernel
    u_int32_t mLoopIterations = m / 8;
    u_int32_t mLoopRemainder = m % 8;

    // PCS
    kernel.add_instr(base::stpPre(gpr_t::x29, gpr_t::x30, gpr_t::sp, -16));
    kernel.add_instr(base::movSP(gpr_t::x29, gpr_t::sp));

    // Compute stride for B
    kernel.add_instr(base::lsl(gpr_t::x3, gpr_t::x3, 2));

    // Save pase matrix pointer
    kernel.add_instr(base::mov(gpr_t::x4, gpr_t::x1)); // B

    // Set n loop counter
    kernel.add_instr(base::mov(gpr_t::x5, n));

    // create zero register
    kernel.add_instr(simd_fp::zero(simd_fp_t::v31, arr_spec_t::b16));

    // Start n loop (1 column)
    kernel.add_label("n_loop");

    // Set m loop counter
    kernel.add_instr(base::mov(gpr_t::x6, mLoopIterations));

    // working pointer for B (rows)
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));

    if (mLoopIterations > 0)
    {
        kernel.add_label("m_8_loop");
        // store 8 zeros
        kernel.add_instr(simd_fp::stp(simd_fp_t::v31, simd_fp_t::v31, gpr_t::x7, 0, neon_size_spec_t::q));
        // jump by 8 rows
        kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, 8*4, 0));
        // decrement m loop counter
        kernel.add_instr(base::sub(gpr_t::x6, gpr_t::x6, 1, 0));
        // check if loop counter is zero
        int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_8_loop");
        kernel.add_instr(base::cbnz(gpr_t::x6, -l_mLoopInstrCount * 4));
    }

    if (mLoopRemainder > 0)
    {
        switch (mLoopRemainder)
        {
        case 1:
            kernel.add_instr(base::str(gpr_t::wzr, gpr_t::x7, 0));
            break;
        case 2:
            kernel.add_instr(base::str(gpr_t::xzr, gpr_t::x7, 0));
            break;
        case 3:
            kernel.add_instr(base::strPost(gpr_t::xzr, gpr_t::x7, 8));  // 2
            kernel.add_instr(base::str(gpr_t::wzr, gpr_t::x7, 0));      // 1
            break;
        case 4:
            kernel.add_instr(simd_fp::str(simd_fp_t::v31, gpr_t::x7, 0, neon_size_spec_t::q)); // 4
            break;
        case 5:
            kernel.add_instr(simd_fp::strPost(simd_fp_t::v31, gpr_t::x7, 4*4, neon_size_spec_t::q)); // 4
            kernel.add_instr(base::str(gpr_t::wzr, gpr_t::x7, 0));                                   // 1
            break;
        case 6:
            kernel.add_instr(simd_fp::strPost(simd_fp_t::v31, gpr_t::x7, 4*4, neon_size_spec_t::q)); // 4
            kernel.add_instr(base::str(gpr_t::xzr, gpr_t::x7, 0));                                   // 2
            break;
        case 7:
            kernel.add_instr(simd_fp::strPost(simd_fp_t::v31, gpr_t::x7, 4*4, neon_size_spec_t::q)); // 4
            kernel.add_instr(base::strPost(gpr_t::xzr, gpr_t::x7, 8));                               // 2
            kernel.add_instr(base::str(gpr_t::wzr, gpr_t::x7, 0));                                   // 1
            break;
        default:
            break;
        }
    }

    // jump to next column
    kernel.add_instr(base::add(gpr_t::x4, gpr_t::x4, gpr_t::x3, 0, 0));
    // decrement n loop counter
    kernel.add_instr(base::sub(gpr_t::x5, gpr_t::x5, 1, 0));
    // check if loop counter is zero
    int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
    kernel.add_instr(base::cbnz(gpr_t::x5, -l_nLoopInstrCount * 4));

    // Restore stack pointer
    kernel.add_instr(base::ldpPost(gpr_t::x29, gpr_t::x30, gpr_t::sp, 16));

    kernel.add_instr(inst::ret());
    kernel.write("zero_primitive.bin");
    kernel.set_kernel();
}