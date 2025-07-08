#include <mlc/Kernel.h>
#include <mlc/instructions/all_instructions.h>
#include <mlc/kernels/unary/relu_primitive.h>
#include <mlc/registers/gp_registers.h>
#include <mlc/registers/simd_fp_registers.h>

using enum gpr_t;
using enum simd_fp_t;
using enum neon_size_spec_t;
using enum arr_spec_t;

using namespace mini_jit::instructions::base;
using namespace mini_jit::instructions::simd_fp;

void mini_jit::kernels::unary::relu(mini_jit::Kernel& kernel,
                                    u_int32_t         m,
                                    u_int32_t         n)
{
    // Inputs:
    // x0: pointer to A
    // x1: pointer to B
    // x2: leading dimension of A
    // x3: leading dimension of B

    // Prepare the kernel
    int mLoopIterations = m / 8;
    int mLoopRemainder  = m % 8;

    // PCS
    kernel.add_instr(stpPre(x29, x30, sp, -16));
    kernel.add_instr(movSP(x29, sp));

    // Compute strides (* 4, because of 4 bytes per fp32 element)
    kernel.add_instr(lsl(x2, x2, 2));
    kernel.add_instr(lsl(x3, x3, 2));

    // Save pase matrix pointers
    kernel.add_instr(mov(x4, x0)); // A
    kernel.add_instr(mov(x5, x1)); // B

    // Set n loop counter
    kernel.add_instr(mov(x6, n));

    // create zero register
    kernel.add_instr(zero(v31, b16));

    // Start n loop (1 column)
    kernel.add_label("n_loop");

    // Set m loop counter
    kernel.add_instr(mov(x7, mLoopIterations));

    // working pointers for rows
    kernel.add_instr(mov(x8, x4)); // A
    kernel.add_instr(mov(x9, x5)); // B

    if (mLoopIterations > 0)
    {
        kernel.add_label("m_8_loop");
        kernel.add_instr({
            // load 8 elements from A
            ldp(v0, v1, x8, 0, q),
            // compute f(x)=max(x,0)
            fmaxVec(v0, v0, v31, s4),
            fmaxVec(v1, v1, v31, s4),
            // store 8 elements to B
            stp(v0, v1, x9, 0, q),
            // jump by 8 rows
            add(x8, x8, 8 * 4, 0),
            add(x9, x9, 8 * 4, 0),
            // decrement m loop counter
            sub(x7, x7, 1, 0),
        });
        // check if loop counter is zero
        kernel.add_instr(cbnz(x7, -kernel.getInstrCountFromLabel("m_8_loop") * 4));
    }

    if (mLoopRemainder > 0)
    {
        switch (mLoopRemainder)
        {
        case 1:
            kernel.add_instr(ldr(v0, x8, 0, s));
            kernel.add_instr(fmaxScalar(v0, v0, v31, s));
            kernel.add_instr(str(v0, x9, 0, s));
            break;
        case 2:
            kernel.add_instr(ldr(v0, x8, 0, d));
            kernel.add_instr(fmaxVec(v0, v0, v31, s2));
            kernel.add_instr(str(v0, x9, 0, d));
            break;
        case 3:
            // 2
            kernel.add_instr(ldrPost(v0, x8, 2 * 4, d));
            kernel.add_instr(fmaxVec(v0, v0, v31, s2));
            kernel.add_instr(strPost(v0, x9, 2 * 4, d));
            // 1
            kernel.add_instr(ldr(v0, x8, 0, s));
            kernel.add_instr(fmaxScalar(v0, v0, v31, s));
            kernel.add_instr(str(v0, x9, 0, s));
            break;
        case 4:
            kernel.add_instr(ldr(v0, x8, 0, q));
            kernel.add_instr(fmaxVec(v0, v0, v31, s4));
            kernel.add_instr(str(v0, x9, 0, q));
            break;
        case 5:
            // 4
            kernel.add_instr(ldrPost(v0, x8, 4 * 4, q));
            kernel.add_instr(fmaxVec(v0, v0, v31, s4));
            kernel.add_instr(strPost(v0, x9, 4 * 4, q));
            // 1
            kernel.add_instr(ldr(v0, x8, 0, s));
            kernel.add_instr(fmaxScalar(v0, v0, v31, s));
            kernel.add_instr(str(v0, x9, 0, s));
            break;
        case 6:
            // 4
            kernel.add_instr(ldrPost(v0, x8, 4 * 4, q));
            kernel.add_instr(fmaxVec(v0, v0, v31, s4));
            kernel.add_instr(strPost(v0, x9, 4 * 4, q));
            // 2
            kernel.add_instr(ldr(v0, x8, 0, d));
            kernel.add_instr(fmaxVec(v0, v0, v31, s2));
            kernel.add_instr(str(v0, x9, 0, d));
            break;
        case 7:
            // 4
            kernel.add_instr(ldrPost(v0, x8, 4 * 4, q));
            kernel.add_instr(fmaxVec(v0, v0, v31, s4));
            kernel.add_instr(strPost(v0, x9, 4 * 4, q));
            // 2
            kernel.add_instr(ldrPost(v0, x8, 2 * 4, d));
            kernel.add_instr(fmaxVec(v0, v0, v31, s2));
            kernel.add_instr(strPost(v0, x9, 2 * 4, d));
            // 1
            kernel.add_instr(ldr(v0, x8, 0, s));
            kernel.add_instr(fmaxScalar(v0, v0, v31, s));
            kernel.add_instr(str(v0, x9, 0, s));
            break;
        default:
            break;
        }
    }

    // jump to next column
    kernel.add_instr(add(x4, x4, x2, 0, 0));
    kernel.add_instr(add(x5, x5, x3, 0, 0));
    // decrement n loop counter
    kernel.add_instr(sub(x6, x6, 1, 0));
    // check if loop counter is zero
    int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
    kernel.add_instr(cbnz(x6, -l_nLoopInstrCount * 4));

    // Restore stack pointer
    kernel.add_instr(ldpPost(x29, x30, sp, 16));

    kernel.add_instr(ret());
    kernel.write("relu_primitive.bin");
    kernel.set_kernel();
}