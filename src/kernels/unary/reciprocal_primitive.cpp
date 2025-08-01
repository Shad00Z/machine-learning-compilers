#include <mlc/Kernel.h>
#include <mlc/instructions/all_instructions.h>
#include <mlc/kernels/unary/reciprocal_primitive.h>
#include <mlc/registers/gp_registers.h>
#include <mlc/registers/simd_fp_registers.h>

using enum gpr_t;
using enum simd_fp_t;
using enum neon_size_spec_t;
using enum arr_spec_t;

using namespace mini_jit::instructions::base;
using namespace mini_jit::instructions::simd_fp;

void mini_jit::kernels::unary::reciprocal(mini_jit::Kernel& kernel,
                                          u_int32_t         m,
                                          u_int32_t         n)
{
    // Inputs:
    // x0: pointer to A
    // x1: pointer to B
    // x2: leading dimension of A
    // x3: leading dimension of B

    // Prepare the kernel
    int mLoopIterations = m / 16;
    int mLoopRemainder  = m % 16;

    kernel.add_instr({// PCS
                      stpPre(x29, x30, sp, -16),
                      movSP(x29, sp),

                      // Save callee-saved registers
                      stpPre(v8, v9, sp, -16, d),
                      stpPre(v10, v11, sp, -16, d),

                      // Compute strides (* 4, because of 4 bytes per fp32 element)
                      lsl(x2, x2, 2),
                      lsl(x3, x3, 2),

                      // Save base matrix pointers
                      mov(x4, x0), // A
                      mov(x5, x1), // B

                      // Set n loop counter
                      mov(x6, n)});

    // Start n loop (1 column)
    kernel.add_label("n_loop");

    // Set m loop counter
    kernel.add_instr({
        mov(x7, mLoopIterations),

        // working pointers for rows
        mov(x8, x4), // A
        mov(x9, x5)  // B
    });

    if (mLoopIterations > 0)
    {
        kernel.add_label("m_16_loop");
        kernel.add_instr({
            // load 16 elements from A
            ldp(v0, v1, x8, 0, q),
            ldp(v2, v3, x8, 32, q),

            frecpeVec(v4, v0, s4),
            frecpsVec(v10, v0, v4, s4),
            fmulVec(v4, v4, v10, s4),

            frecpeVec(v5, v1, s4),
            frecpsVec(v10, v1, v5, s4),
            fmulVec(v5, v5, v10, s4),

            frecpeVec(v6, v2, s4),
            frecpsVec(v10, v2, v6, s4),
            fmulVec(v6, v6, v10, s4),

            frecpeVec(v7, v3, s4),
            frecpsVec(v10, v3, v7, s4),
            fmulVec(v7, v7, v10, s4),

            // store 16 elements to B
            stp(v4, v5, x9, 0, q),
            stp(v6, v7, x9, 32, q),

            // jump by 16 rows
            add(x8, x8, 16 * 4, 0),
            add(x9, x9, 16 * 4, 0),

            // decrement m loop counter
            sub(x7, x7, 1, 0),
        });
        // check if loop counter is zero
        kernel.add_instr(cbnz(x7, -kernel.getInstrCountFromLabel("m_16_loop") * 4));
    }

    if (mLoopRemainder > 0)
    {
        switch (mLoopRemainder)
        {
        case 1:
            kernel.add_instr({// 1 element
                              ldr(v0, x8, 0, s),
                              frecpeScalar(v1, v0, s),
                              frecpsScalar(v10, v0, v1, s),
                              fmulScalar(v1, v1, v10, s),
                              str(v1, x9, 0, s)});
            break;
        case 2:
            kernel.add_instr({// 2 elements
                              ldr(v0, x8, 0, d),
                              frecpeVec(v1, v0, s2),
                              frecpsVec(v10, v0, v1, s2),
                              fmulVec(v1, v1, v10, s2),
                              str(v1, x9, 0, d)});
            break;
        case 3:
            kernel.add_instr({// 2 elements
                              ldr(v0, x8, 0, d),
                              frecpeVec(v1, v0, s2),
                              frecpsVec(v10, v0, v1, s2),
                              fmulVec(v1, v1, v10, s2),
                              str(v1, x9, 0, d),

                              // 1 element
                              ldr(v2, x8, 2 * 4, s),
                              frecpeScalar(v3, v2, s),
                              frecpsScalar(v10, v2, v3, s),
                              fmulScalar(v3, v3, v10, s),
                              str(v3, x9, 2 * 4, s)});
            break;
        case 4:
            kernel.add_instr({// 4 elements
                              ldr(v0, x8, 0, q),
                              frecpeVec(v1, v0, s4),
                              frecpsVec(v10, v0, v1, s4),
                              fmulVec(v1, v1, v10, s4),
                              str(v1, x9, 0, q)});
            break;
        case 5:
            kernel.add_instr({// 4 elements
                              ldr(v0, x8, 0, q),
                              frecpeVec(v1, v0, s4),
                              frecpsVec(v10, v0, v1, s4),
                              fmulVec(v1, v1, v10, s4),
                              str(v1, x9, 0, q),

                              // 1 element
                              ldr(v2, x8, 4 * 4, s),
                              frecpeScalar(v3, v2, s),
                              frecpsScalar(v10, v2, v3, s),
                              fmulScalar(v3, v3, v10, s),
                              str(v3, x9, 4 * 4, s)});
            break;
        case 6:
            kernel.add_instr({// 4 elements
                              ldr(v0, x8, 0, q),
                              frecpeVec(v1, v0, s4),
                              frecpsVec(v10, v0, v1, s4),
                              fmulVec(v1, v1, v10, s4),
                              str(v1, x9, 0, q),

                              // 2 elements
                              ldr(v2, x8, 4 * 4, d),
                              frecpeVec(v3, v2, s2),
                              frecpsVec(v10, v2, v3, s2),
                              fmulVec(v3, v3, v10, s2),
                              str(v3, x9, 4 * 4, d)});
            break;
        case 7:
            kernel.add_instr({// 4 elements
                              ldr(v0, x8, 0, q),
                              frecpeVec(v1, v0, s4),
                              frecpsVec(v10, v0, v1, s4),
                              fmulVec(v1, v1, v10, s4),
                              str(v1, x9, 0, q),

                              // 2 elements
                              ldr(v2, x8, 4 * 4, d),
                              frecpeVec(v3, v2, s2),
                              frecpsVec(v10, v2, v3, s2),
                              fmulVec(v3, v3, v10, s2),
                              str(v3, x9, 4 * 4, d),

                              // 1 element
                              ldr(v4, x8, 24, s),
                              frecpeScalar(v5, v4, s),
                              frecpsScalar(v10, v4, v5, s),
                              fmulScalar(v5, v5, v10, s),
                              str(v5, x9, 24, s)});
            break;
        case 8:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),
                              frecpeVec(v2, v0, s4),
                              frecpsVec(v10, v0, v2, s4),
                              fmulVec(v2, v2, v10, s4),

                              frecpeVec(v3, v1, s4),
                              frecpsVec(v10, v1, v3, s4),
                              fmulVec(v3, v3, v10, s4),
                              stp(v2, v3, x9, 0, q)});
            break;
        case 9:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),
                              frecpeVec(v2, v0, s4),
                              frecpsVec(v10, v0, v2, s4),
                              fmulVec(v2, v2, v10, s4),

                              frecpeVec(v3, v1, s4),
                              frecpsVec(v10, v1, v3, s4),
                              fmulVec(v3, v3, v10, s4),
                              stp(v2, v3, x9, 0, q),

                              // 1 element
                              ldr(v4, x8, 32, s),
                              frecpeScalar(v5, v4, s),
                              frecpsScalar(v10, v4, v5, s),
                              fmulScalar(v5, v5, v10, s),
                              str(v5, x9, 32, s)});
            break;
        case 10:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),
                              frecpeVec(v2, v0, s4),
                              frecpsVec(v10, v0, v2, s4),
                              fmulVec(v2, v2, v10, s4),

                              frecpeVec(v3, v1, s4),
                              frecpsVec(v10, v1, v3, s4),
                              fmulVec(v3, v3, v10, s4),
                              stp(v2, v3, x9, 0, q),

                              // 2 elements
                              ldr(v4, x8, 32, d),
                              frecpeVec(v5, v4, s2),
                              frecpsVec(v10, v4, v5, s2),
                              fmulVec(v5, v5, v10, s2),
                              str(v5, x9, 32, d)});
            break;
        case 11:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),
                              frecpeVec(v2, v0, s4),
                              frecpsVec(v10, v0, v2, s4),
                              fmulVec(v2, v2, v10, s4),

                              frecpeVec(v3, v1, s4),
                              frecpsVec(v10, v1, v3, s4),
                              fmulVec(v3, v3, v10, s4),
                              stp(v2, v3, x9, 0, q),

                              // 2 elements
                              ldr(v4, x8, 32, d),
                              frecpeVec(v5, v4, s2),
                              frecpsVec(v10, v4, v5, s2),
                              fmulVec(v5, v5, v10, s2),
                              str(v5, x9, 32, d),

                              // 1 element
                              ldr(v6, x8, 40, s),
                              frecpeScalar(v7, v6, s),
                              frecpsScalar(v10, v6, v7, s),
                              fmulScalar(v7, v7, v10, s),
                              str(v7, x9, 40, s)});
            break;
        case 12:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),
                              frecpeVec(v2, v0, s4),
                              frecpsVec(v10, v0, v2, s4),
                              fmulVec(v2, v2, v10, s4),

                              frecpeVec(v3, v1, s4),
                              frecpsVec(v10, v1, v3, s4),
                              fmulVec(v3, v3, v10, s4),
                              stp(v2, v3, x9, 0, q),

                              // 4 elements
                              ldr(v4, x8, 32, q),
                              frecpeVec(v5, v4, s4),
                              frecpsVec(v10, v4, v5, s4),
                              fmulVec(v5, v5, v10, s4),
                              str(v5, x9, 32, q)});
            break;
        case 13:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),
                              frecpeVec(v2, v0, s4),
                              frecpsVec(v10, v0, v2, s4),
                              fmulVec(v2, v2, v10, s4),

                              frecpeVec(v3, v1, s4),
                              frecpsVec(v10, v1, v3, s4),
                              fmulVec(v3, v3, v10, s4),
                              stp(v2, v3, x9, 0, q),

                              // 4 elements
                              ldr(v4, x8, 32, q),
                              frecpeVec(v5, v4, s4),
                              frecpsVec(v10, v4, v5, s4),
                              fmulVec(v5, v5, v10, s4),
                              str(v5, x9, 32, q),

                              // 1 element
                              ldr(v6, x8, 48, s),
                              frecpeScalar(v7, v6, s),
                              frecpsScalar(v10, v6, v7, s),
                              fmulScalar(v7, v7, v10, s),
                              str(v7, x9, 48, s)});
            break;
        case 14:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),
                              frecpeVec(v2, v0, s4),
                              frecpsVec(v10, v0, v2, s4),
                              fmulVec(v2, v2, v10, s4),

                              frecpeVec(v3, v1, s4),
                              frecpsVec(v10, v1, v3, s4),
                              fmulVec(v3, v3, v10, s4),
                              stp(v2, v3, x9, 0, q),

                              // 4 elements
                              ldr(v4, x8, 32, q),
                              frecpeVec(v5, v4, s4),
                              frecpsVec(v10, v4, v5, s4),
                              fmulVec(v5, v5, v10, s4),
                              str(v5, x9, 32, q),

                              // 2 elements
                              ldr(v6, x8, 48, d),
                              frecpeVec(v7, v6, s2),
                              frecpsVec(v10, v6, v7, s2),
                              fmulVec(v7, v7, v10, s2),
                              str(v7, x9, 48, d)});
            break;
        case 15:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),
                              frecpeVec(v2, v0, s4),
                              frecpsVec(v10, v0, v2, s4),
                              fmulVec(v2, v2, v10, s4),

                              frecpeVec(v3, v1, s4),
                              frecpsVec(v10, v1, v3, s4),
                              fmulVec(v3, v3, v10, s4),
                              stp(v2, v3, x9, 0, q),

                              // 4 elements
                              ldr(v4, x8, 32, q),
                              frecpeVec(v5, v4, s4),
                              frecpsVec(v10, v4, v5, s4),
                              fmulVec(v5, v5, v10, s4),
                              str(v5, x9, 32, q),

                              // 2 elements
                              ldr(v6, x8, 48, d),
                              frecpeVec(v7, v6, s2),
                              frecpsVec(v10, v6, v7, s2),
                              fmulVec(v7, v7, v10, s2),
                              str(v7, x9, 48, d),

                              // 1 element
                              ldr(v8, x8, 56, s),
                              frecpeScalar(v9, v8, s),
                              frecpsScalar(v10, v8, v9, s),
                              fmulScalar(v9, v9, v10, s),
                              str(v9, x9, 56, s)});
            break;
        default:
            break;
        }
    }

    kernel.add_instr({// jump to next column
                      add(x4, x4, x2, 0, 0),
                      add(x5, x5, x3, 0, 0),

                      // decrement n loop counter
                      sub(x6, x6, 1, 0)});
    // check if loop counter is zero
    int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
    kernel.add_instr(cbnz(x6, -l_nLoopInstrCount * 4));

    kernel.add_instr({// Restore callee-saved registers
                      ldpPost(v10, v11, sp, 16, d),
                      ldpPost(v8, v9, sp, 16, d),

                      // Restore stack pointer
                      ldpPost(x29, x30, sp, 16),

                      ret()});
    kernel.write("reciprocal_primitive.bin");
    kernel.set_kernel();
}