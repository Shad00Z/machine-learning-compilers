#include <mlc/Kernel.h>
#include <mlc/constants.h>
#include <mlc/instructions/all_instructions.h>
#include <mlc/kernels/unary/fast_sigmoid_primitive.h>
#include <mlc/registers/gp_registers.h>
#include <mlc/registers/simd_fp_registers.h>

using enum gpr_t;
using enum simd_fp_t;
using enum neon_size_spec_t;
using enum arr_spec_t;

using namespace mini_jit::instructions::base;
using namespace mini_jit::instructions::simd_fp;

void mini_jit::kernels::unary::fast_sigmoid(mini_jit::Kernel& kernel,
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

                      // Load constant values
                      fmovVec(v30, 0b01110000, s4),
                      fmovVec(v31, 0b01100000, s4),

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

            ////////////////////////////////////////////
            // Compute B = 0.5 * (A / (1 + abs(A)) + 1)
            ////////////////////////////////////////////
            // abs(A)
            fabsVec(v4, v0, s4),
            fabsVec(v5, v1, s4),
            fabsVec(v6, v2, s4),
            fabsVec(v7, v3, s4),

            // 1 + abs(A)
            faddVec(v4, v4, v30, s4),
            faddVec(v5, v5, v30, s4),
            faddVec(v6, v6, v30, s4),
            faddVec(v7, v7, v30, s4),

            // A / (1 + abs(A))
            fdivVec(v0, v0, v4, s4),
            fdivVec(v1, v1, v5, s4),
            fdivVec(v2, v2, v6, s4),
            fdivVec(v3, v3, v7, s4),

            // A / (1 + abs(A)) + 1
            faddVec(v0, v0, v30, s4),
            faddVec(v1, v1, v30, s4),
            faddVec(v2, v2, v30, s4),
            faddVec(v3, v3, v30, s4),

            // 0.5 * (A / (1 + abs(A)) + 1)
            fmulVec(v4, v0, v31, s4),
            fmulVec(v5, v1, v31, s4),
            fmulVec(v6, v2, v31, s4),
            fmulVec(v7, v3, v31, s4),
            ////////////////////////////////////////////

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

                              fabsScalar(v1, v0, s),
                              faddScalar(v1, v1, v30, s),
                              fdivScalar(v0, v0, v1, s),
                              faddScalar(v0, v0, v30, s),
                              fmulScalar(v1, v0, v31, s),

                              str(v1, x9, 0, s)});
            break;
        case 2:
            kernel.add_instr({// 2 elements
                              ldr(v0, x8, 0, d),

                              fabsVec(v1, v0, s2),
                              faddVec(v1, v1, v30, s2),
                              fdivVec(v0, v0, v1, s2),
                              faddVec(v0, v0, v30, s2),
                              fmulVec(v1, v0, v31, s2),

                              str(v1, x9, 0, d)});
            break;
        case 3:
            kernel.add_instr({// 2 elements
                              ldr(v0, x8, 0, d),

                              fabsVec(v1, v0, s2),
                              faddVec(v1, v1, v30, s2),
                              fdivVec(v0, v0, v1, s2),
                              faddVec(v0, v0, v30, s2),
                              fmulVec(v1, v0, v31, s2),

                              str(v1, x9, 0, d),

                              // 1 element
                              ldr(v2, x8, 2 * 4, s),

                              fabsScalar(v3, v2, s),
                              faddScalar(v3, v3, v30, s),
                              fdivScalar(v2, v2, v3, s),
                              faddScalar(v2, v2, v30, s),
                              fmulScalar(v3, v2, v31, s),

                              str(v3, x9, 2 * 4, s)});
            break;
        case 4:
            kernel.add_instr({// 4 elements
                              ldr(v0, x8, 0, q),

                              fabsVec(v1, v0, s4),
                              faddVec(v1, v1, v30, s4),
                              fdivVec(v0, v0, v1, s4),
                              faddVec(v0, v0, v30, s4),
                              fmulVec(v1, v0, v31, s4),

                              str(v1, x9, 0, q)});
            break;
        case 5:
            kernel.add_instr({// 4 elements
                              ldr(v0, x8, 0, q),

                              fabsVec(v1, v0, s4),
                              faddVec(v1, v1, v30, s4),
                              fdivVec(v0, v0, v1, s4),
                              faddVec(v0, v0, v30, s4),
                              fmulVec(v1, v0, v31, s4),

                              str(v1, x9, 0, q),

                              // 1 element
                              ldr(v2, x8, 4 * 4, s),

                              fabsScalar(v3, v2, s),
                              faddScalar(v3, v3, v30, s),
                              fdivScalar(v2, v2, v3, s),
                              faddScalar(v2, v2, v30, s),
                              fmulScalar(v3, v2, v31, s),

                              str(v3, x9, 4 * 4, s)});
            break;
        case 6:
            kernel.add_instr({// 4 elements
                              ldr(v0, x8, 0, q),

                              fabsVec(v1, v0, s4),
                              faddVec(v1, v1, v30, s4),
                              fdivVec(v0, v0, v1, s4),
                              faddVec(v0, v0, v30, s4),
                              fmulVec(v1, v0, v31, s4),

                              str(v1, x9, 0, q),

                              // 2 elements
                              ldr(v2, x8, 4 * 4, d),

                              fabsVec(v3, v2, s4),
                              faddVec(v3, v3, v30, s4),
                              fdivVec(v2, v2, v3, s4),
                              faddVec(v2, v2, v30, s4),
                              fmulVec(v3, v2, v31, s4),

                              str(v3, x9, 4 * 4, d)});
            break;
        case 7:
            kernel.add_instr({// 4 elements
                              ldr(v0, x8, 0, q),

                              fabsVec(v1, v0, s4),
                              faddVec(v1, v1, v30, s4),
                              fdivVec(v0, v0, v1, s4),
                              faddVec(v0, v0, v30, s4),
                              fmulVec(v1, v0, v31, s4),

                              str(v1, x9, 0, q),

                              // 2 elements
                              ldr(v2, x8, 4 * 4, d),

                              fabsVec(v3, v2, s4),
                              faddVec(v3, v3, v30, s4),
                              fdivVec(v2, v2, v3, s4),
                              faddVec(v2, v2, v30, s4),
                              fmulVec(v3, v2, v31, s4),

                              str(v3, x9, 4 * 4, d),

                              // 1 element
                              ldr(v4, x8, 6 * 4, s),

                              fabsScalar(v5, v4, s),
                              faddScalar(v5, v5, v30, s),
                              fdivScalar(v4, v4, v5, s),
                              faddScalar(v4, v4, v30, s),
                              fmulScalar(v5, v4, v31, s),

                              str(v5, x9, 6 * 4, s)});
            break;
        case 8:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),

                              fabsVec(v2, v0, s4),
                              fabsVec(v3, v1, s4),
                              faddVec(v2, v2, v30, s4),
                              faddVec(v3, v3, v30, s4),
                              fdivVec(v0, v0, v2, s4),
                              fdivVec(v1, v1, v3, s4),
                              faddVec(v0, v0, v30, s4),
                              faddVec(v1, v1, v30, s4),
                              fmulVec(v2, v0, v31, s4),
                              fmulVec(v3, v1, v31, s4),

                              stp(v2, v3, x9, 0, q)});
            break;
        case 9:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),

                              fabsVec(v2, v0, s4),
                              fabsVec(v3, v1, s4),
                              faddVec(v2, v2, v30, s4),
                              faddVec(v3, v3, v30, s4),
                              fdivVec(v0, v0, v2, s4),
                              fdivVec(v1, v1, v3, s4),
                              faddVec(v0, v0, v30, s4),
                              faddVec(v1, v1, v30, s4),
                              fmulVec(v2, v0, v31, s4),
                              fmulVec(v3, v1, v31, s4),

                              stp(v2, v3, x9, 0, q),

                              // 1 element
                              ldr(v4, x8, 8 * 4, s),

                              fabsScalar(v5, v4, s),
                              faddScalar(v5, v5, v30, s),
                              fdivScalar(v4, v4, v5, s),
                              faddScalar(v4, v4, v30, s),
                              fmulScalar(v5, v4, v31, s),

                              str(v5, x9, 8 * 4, s)});
            break;
        case 10:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),

                              fabsVec(v2, v0, s4),
                              fabsVec(v3, v1, s4),
                              faddVec(v2, v2, v30, s4),
                              faddVec(v3, v3, v30, s4),
                              fdivVec(v0, v0, v2, s4),
                              fdivVec(v1, v1, v3, s4),
                              faddVec(v0, v0, v30, s4),
                              faddVec(v1, v1, v30, s4),
                              fmulVec(v2, v0, v31, s4),
                              fmulVec(v3, v1, v31, s4),

                              stp(v2, v3, x9, 0, q),

                              // 2 elements
                              ldr(v4, x8, 8 * 4, d),

                              fabsVec(v5, v4, s4),
                              faddVec(v5, v5, v30, s4),
                              fdivVec(v4, v4, v5, s4),
                              faddVec(v4, v4, v30, s4),
                              fmulVec(v5, v4, v31, s4),

                              str(v5, x9, 8 * 4, d)});
            break;
        case 11:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),

                              fabsVec(v2, v0, s4),
                              fabsVec(v3, v1, s4),
                              faddVec(v2, v2, v30, s4),
                              faddVec(v3, v3, v30, s4),
                              fdivVec(v0, v0, v2, s4),
                              fdivVec(v1, v1, v3, s4),
                              faddVec(v0, v0, v30, s4),
                              faddVec(v1, v1, v30, s4),
                              fmulVec(v2, v0, v31, s4),
                              fmulVec(v3, v1, v31, s4),

                              stp(v2, v3, x9, 0, q),

                              // 2 elements
                              ldr(v4, x8, 8 * 4, d),

                              fabsVec(v5, v4, s4),
                              faddVec(v5, v5, v30, s4),
                              fdivVec(v4, v4, v5, s4),
                              faddVec(v4, v4, v30, s4),
                              fmulVec(v5, v4, v31, s4),

                              str(v5, x9, 8 * 4, d),

                              // 1 element
                              ldr(v6, x8, 10 * 4, s),

                              fabsScalar(v7, v6, s),
                              faddScalar(v7, v7, v30, s),
                              fdivScalar(v6, v6, v7, s),
                              faddScalar(v6, v6, v30, s),
                              fmulScalar(v7, v6, v31, s),

                              str(v7, x9, 10 * 4, s)});
            break;
        case 12:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),

                              fabsVec(v2, v0, s4),
                              fabsVec(v3, v1, s4),
                              faddVec(v2, v2, v30, s4),
                              faddVec(v3, v3, v30, s4),
                              fdivVec(v0, v0, v2, s4),
                              fdivVec(v1, v1, v3, s4),
                              faddVec(v0, v0, v30, s4),
                              faddVec(v1, v1, v30, s4),
                              fmulVec(v2, v0, v31, s4),
                              fmulVec(v3, v1, v31, s4),

                              stp(v2, v3, x9, 0, q),

                              // 4 elements
                              ldr(v4, x8, 8 * 4, q),

                              fabsVec(v5, v4, s4),
                              faddVec(v5, v5, v30, s4),
                              fdivVec(v4, v4, v5, s4),
                              faddVec(v4, v4, v30, s4),
                              fmulVec(v5, v4, v31, s4),

                              str(v5, x9, 8 * 4, q)});
            break;
        case 13:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),

                              fabsVec(v2, v0, s4),
                              fabsVec(v3, v1, s4),
                              faddVec(v2, v2, v30, s4),
                              faddVec(v3, v3, v30, s4),
                              fdivVec(v0, v0, v2, s4),
                              fdivVec(v1, v1, v3, s4),
                              faddVec(v0, v0, v30, s4),
                              faddVec(v1, v1, v30, s4),
                              fmulVec(v2, v0, v31, s4),
                              fmulVec(v3, v1, v31, s4),

                              stp(v2, v3, x9, 0, q),

                              // 4 elements
                              ldr(v4, x8, 8 * 4, q),

                              fabsVec(v5, v4, s4),
                              faddVec(v5, v5, v30, s4),
                              fdivVec(v4, v4, v5, s4),
                              faddVec(v4, v4, v30, s4),
                              fmulVec(v5, v4, v31, s4),

                              str(v5, x9, 8 * 4, q),

                              // 1 element
                              ldr(v6, x8, 12 * 4, s),

                              fabsScalar(v7, v6, s),
                              faddScalar(v7, v7, v30, s),
                              fdivScalar(v6, v6, v7, s),
                              faddScalar(v6, v6, v30, s),
                              fmulScalar(v7, v6, v31, s),

                              str(v7, x9, 12 * 4, s)});
            break;
        case 14:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),

                              fabsVec(v2, v0, s4),
                              fabsVec(v3, v1, s4),
                              faddVec(v2, v2, v30, s4),
                              faddVec(v3, v3, v30, s4),
                              fdivVec(v0, v0, v2, s4),
                              fdivVec(v1, v1, v3, s4),
                              faddVec(v0, v0, v30, s4),
                              faddVec(v1, v1, v30, s4),
                              fmulVec(v2, v0, v31, s4),
                              fmulVec(v3, v1, v31, s4),

                              stp(v2, v3, x9, 0, q),

                              // 4 elements
                              ldr(v4, x8, 8 * 4, q),

                              fabsVec(v5, v4, s4),
                              faddVec(v5, v5, v30, s4),
                              fdivVec(v4, v4, v5, s4),
                              faddVec(v4, v4, v30, s4),
                              fmulVec(v5, v4, v31, s4),

                              str(v5, x9, 8 * 4, q),

                              // 2 elements
                              ldr(v6, x8, 12 * 4, d),

                              fabsVec(v7, v6, s2),
                              faddVec(v7, v7, v30, s2),
                              fdivVec(v6, v6, v7, s2),
                              faddVec(v6, v6, v30, s2),
                              fmulVec(v7, v6, v31, s2),

                              str(v7, x9, 12 * 4, d)});
            break;
        case 15:
            kernel.add_instr({// 8 elements
                              ldp(v0, v1, x8, 0, q),

                              fabsVec(v2, v0, s4),
                              fabsVec(v3, v1, s4),
                              faddVec(v2, v2, v30, s4),
                              faddVec(v3, v3, v30, s4),
                              fdivVec(v0, v0, v2, s4),
                              fdivVec(v1, v1, v3, s4),
                              faddVec(v0, v0, v30, s4),
                              faddVec(v1, v1, v30, s4),
                              fmulVec(v2, v0, v31, s4),
                              fmulVec(v3, v1, v31, s4),

                              stp(v2, v3, x9, 0, q),

                              // 4 elements
                              ldr(v4, x8, 8 * 4, q),

                              fabsVec(v5, v4, s4),
                              faddVec(v5, v5, v30, s4),
                              fdivVec(v4, v4, v5, s4),
                              faddVec(v4, v4, v30, s4),
                              fmulVec(v5, v4, v31, s4),

                              str(v5, x9, 8 * 4, q),

                              // 2 elements
                              ldr(v6, x8, 12 * 4, d),

                              fabsVec(v7, v6, s2),
                              faddVec(v7, v7, v30, s2),
                              fdivVec(v6, v6, v7, s2),
                              faddVec(v6, v6, v30, s2),
                              fmulVec(v7, v6, v31, s2),

                              str(v7, x9, 12 * 4, d),

                              // 1 element
                              ldr(v8, x8, 14 * 4, s),

                              fabsScalar(v9, v8, s),
                              faddScalar(v9, v9, v30, s),
                              fdivScalar(v8, v8, v9, s),
                              faddScalar(v8, v8, v30, s),
                              fmulScalar(v9, v8, v31, s),

                              str(v9, x9, 14 * 4, s)});
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