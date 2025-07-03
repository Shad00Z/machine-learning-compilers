#include "div_primitive.h"
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

void mini_jit::kernels::binary::div(mini_jit::Kernel &kernel,
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

    // Prepare the kernel
    int mLoopIterations = m / 16;
    int mLoopRemainder = m % 16;

    kernel.add_instr({
        stpPre(x29, x30, sp, -16),
        movSP(x29, sp),

        // Callee-saved registers
        stpPre(x19, x20, sp, -16),
        stpPre(x21, x22, sp, -16),
        stpPre(x23, x24, sp, -16),
        stpPre(x25, x26, sp, -16),
        stpPre(x27, x28, sp, -16),

        stpPre(v8, v9, sp, -16, neon_size_spec_t::d),
        stpPre(v10, v11, sp, -16, neon_size_spec_t::d),
        stpPre(v12, v13, sp, -16, neon_size_spec_t::d),
        stpPre(v14, v15, sp, -16, neon_size_spec_t::d),

        // Strides
        lsl(x3, x3, 2), // leading dimension of A
        lsl(x4, x4, 2), // leading dimension of B
        lsl(x5, x5, 2), // leading dimension of C

        // Save base matrix pointers
        mov(x6, x0), // A
        mov(x7, x1), // B
        mov(x8, x2), // C

        // Set n loop counter
        mov(x9, n)
    });

    // Start n loop (1 column)
    kernel.add_label("n_loop");

    kernel.add_instr({
        // Set m loop counter
        mov(x10, mLoopIterations),

        // working pointers for rows
        mov(x11, x6),   // A
        mov(x12, x7),   // B
        mov(x13, x8)    // C
    });

    if (mLoopIterations > 0)
    {
        kernel.add_label("m_16_loop");
        kernel.add_instr({
            // load 16 elements from A
            ldp(v0, v1, x11, 0, q),
            ldp(v2, v3, x11, 32, q),

            // load 16 elements from B
            ldp(v4, v5, x12, 0, q),
            ldp(v6, v7, x12, 32, q),

            // compute C = A / B
            fdivVec(v8, v0, v4, s4),
            fdivVec(v9, v1, v5, s4),
            fdivVec(v10, v2, v6, s4),
            fdivVec(v11, v3, v7, s4),

            // store 16 elements to C
            stp(v8, v9, x13, 0, q),
            stp(v10, v11, x13, 32, q),

            // jump by 16 rows
            add(x11, x11, 16*4, 0),
            add(x12, x12, 16*4, 0),
            add(x13, x13, 16*4, 0),

            // decrement m loop counter
            sub(x10, x10, 1, 0),
        });
        // check if loop counter is zero
        kernel.add_instr(cbnz(x10, -kernel.getInstrCountFromLabel("m_16_loop") * 4));
    }

    if (mLoopRemainder > 0)
    {
        switch (mLoopRemainder)
        {
        case 1:
            kernel.add_instr({
                // 1 element
                ldr(v0, x11, 0, s),
                ldr(v1, x12, 0, s),
                fdivScalar(v2, v0, v1, s),
                str(v2, x13, 0, s)
            });
            break;
        case 2:
            kernel.add_instr({
                // 2 elements
                ldr(v0, x11, 0, d),
                ldr(v1, x12, 0, d),
                fdivVec(v2, v0, v1, s2),
                str(v2, x13, 0, d)
            });
            break;
        case 3:
            kernel.add_instr({
                // 2 elements
                ldr(v0, x11, 0, d),
                ldr(v1, x12, 0, d),
                fdivVec(v2, v0, v1, s2),
                str(v2, x13, 0, d),

                // 1 element
                ldr(v3, x11, 2*4, s),
                ldr(v4, x12, 2*4, s),
                fdivScalar(v5, v3, v4, s),
                str(v5, x13, 2*4, s)
            });
            break;
        case 4:
            kernel.add_instr({
                // 4 elements
                ldr(v0, x11, 0, q),
                ldr(v1, x12, 0, q),
                fdivVec(v2, v0, v1, s4),
                str(v2, x13, 0, q)
            });
            break;
        case 5:
            kernel.add_instr({
                // 4 elements
                ldr(v0, x11, 0, q),
                ldr(v1, x12, 0, q),
                fdivVec(v2, v0, v1, s4),
                str(v2, x13, 0, q),
                
                // 5 elements
                ldr(v3, x11, 4*4, s),
                ldr(v4, x12, 4*4, s),
                fdivScalar(v5, v3, v4, s),
                str(v5, x13, 4*4, s)
            });
            break;
        case 6:
            kernel.add_instr({
                // 4 elements
                ldr(v0, x11, 0, q),
                ldr(v1, x12, 0, q),
                fdivVec(v2, v0, v1, s4),
                str(v2, x13, 0, q),

                // 2 elements
                ldr(v3, x11, 4*4, d),
                ldr(v4, x12, 4*4, d),
                fdivVec(v5, v3, v4, s2),
                str(v5, x13, 4*4, d)
            });
            break;
        case 7:
            kernel.add_instr({
                // 4 elements
                ldr(v0, x11, 0, q),
                ldr(v1, x12, 0, q),
                fdivVec(v2, v0, v1, s4),
                str(v2, x13, 0, q),

                // 2 elements
                ldr(v3, x11, 4*4, d),
                ldr(v4, x12, 4*4, d),
                fdivVec(v5, v3, v4, s2),
                str(v5, x13, 4*4, d),

                // 1 element
                ldr(v6, x11, 6*4, s),
                ldr(v7, x12, 6*4, s),
                fdivScalar(v8, v6, v7, s),
                str(v8, x13, 6*4, s)
            });
            break;
        case 8:
            kernel.add_instr({
                // 8 elements
                ldp(v0, v1, x11, 0, q),
                ldp(v2, v3, x12, 0, q),
                fdivVec(v4, v0, v2, s4),
                fdivVec(v5, v1, v3, s4),
                stp(v4, v5, x13, 0, q)
            });
            break;
        case 9:
            kernel.add_instr({
                // 8 elements
                ldp(v0, v1, x11, 0, q),
                ldp(v2, v3, x12, 0, q),
                fdivVec(v4, v0, v2, s4),
                fdivVec(v5, v1, v3, s4),
                stp(v4, v5, x13, 0, q),

                // 1 element
                ldr(v6, x11, 8*4, s),
                ldr(v7, x12, 8*4, s),
                fdivScalar(v8, v6, v7, s),
                str(v8, x13, 8*4, s)
            });
            break;
        case 10:
            kernel.add_instr({
                // 8 elements
                ldp(v0, v1, x11, 0, q),
                ldp(v2, v3, x12, 0, q),
                fdivVec(v4, v0, v2, s4),
                fdivVec(v5, v1, v3, s4),
                stp(v4, v5, x13, 0, q),

                // 2 elements
                ldr(v6, x11, 8*4, d),
                ldr(v7, x12, 8*4, d),
                fdivVec(v8, v6, v7, s2),
                str(v8, x13, 8*4, d)
            });
            break;
        case 11:
            kernel.add_instr({
                // 8 elements
                ldp(v0, v1, x11, 0, q),
                ldp(v2, v3, x12, 0, q),
                fdivVec(v4, v0, v2, s4),
                fdivVec(v5, v1, v3, s4),
                stp(v4, v5, x13, 0, q),

                // 2 elements
                ldr(v6, x11, 8*4, d),
                ldr(v7, x12, 8*4, d),
                fdivVec(v8, v6, v7, s2),
                str(v8, x13, 8*4, d),

                // 1 element
                ldr(v9, x11, 10*4, s),
                ldr(v10, x12, 10*4, s),
                fdivScalar(v11, v9, v10, s),
                str(v11, x13, 10*4, s)
            });
            break;
        case 12:
            kernel.add_instr({
                // 8 elements
                ldp(v0, v1, x11, 0, q),
                ldp(v2, v3, x12, 0, q),
                fdivVec(v4, v0, v2, s4),
                fdivVec(v5, v1, v3, s4),
                stp(v4, v5, x13, 0, q),

                // 4 elements
                ldr(v6, x11, 8*4, q),
                ldr(v7, x12, 8*4, q),
                fdivVec(v8, v6, v7, s4),
                str(v8, x13, 8*4, q)
            });
            break;
        case 13:
            kernel.add_instr({
                // 8 elements
                ldp(v0, v1, x11, 0, q),
                ldp(v2, v3, x12, 0, q),
                fdivVec(v4, v0, v2, s4),
                fdivVec(v5, v1, v3, s4),
                stp(v4, v5, x13, 0, q),

                // 4 elements
                ldr(v6, x11, 8*4, q),
                ldr(v7, x12, 8*4, q),
                fdivVec(v8, v6, v7, s4),
                str(v8, x13, 8*4, q),

                // 1 element
                ldr(v9, x11, 12*4, s),
                ldr(v10, x12, 12*4, s),
                fdivScalar(v11, v9, v10, s),
                str(v11, x13, 12*4, s)
            });
            break;
        case 14:
            kernel.add_instr({
                // 8 elements
                ldp(v0, v1, x11, 0, q),
                ldp(v2, v3, x12, 0 , q),
                fdivVec(v4, v0, v2, s4),
                fdivVec(v5, v1, v3, s4),
                stp(v4, v5, x13, 0, q),

                // 4 elements
                ldr(v6, x11, 8*4, q),
                ldr(v7, x12, 8*4, q),
                fdivVec(v8, v6, v7, s4),
                str(v8, x13, 8*4, q),

                // 2 elements
                ldr(v9, x11, 12*4, d),
                ldr(v10, x12, 12*4, d),
                fdivVec(v11, v9, v10, s2),
                str(v11, x13, 12*4, d)
            });
            break;
        case 15:
            kernel.add_instr({
                // 8 elements
                ldp(v0, v1, x11, 0, q),
                ldp(v2, v3, x12, 0, q),
                fdivVec(v4, v0, v2, s4),
                fdivVec(v5, v1, v3, s4),
                stp(v4, v5, x13, 0, q),

                // 4 elements
                ldr(v6, x11, 8*4, q),
                ldr(v7, x12, 8*4, q),
                fdivVec(v8, v6, v7, s4),
                str(v8, x13, 8*4, q),

                // 2 elements
                ldr(v9, x11, 12*4, d),
                ldr(v10, x12, 12*4, d),
                fdivVec(v11, v9, v10, s2),
                str(v11, x13, 12*4, d),

                // 1 element
                ldr(v12, x11, 14*4, s),
                ldr(v13, x12, 14*4, s),
                fdivScalar(v14, v12, v13, s),
                str(v14, x13, 14*4, s)
            });
            break;
        default:
            break;
        }
    }

    kernel.add_instr({
        // jump to next column
        add(x6, x6, x3, 0, 0),
        add(x7, x7, x4, 0, 0),
        add(x8, x8, x5, 0, 0),

        // decrement n loop counter
        sub(x9, x9, 1, 0)
    });
    // check if n loop counter is zero
    int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
    kernel.add_instr(cbnz(x9, -l_nLoopInstrCount * 4));

    kernel.add_instr({
        // Restore callee-saved registers
        ldpPost(v14, v15, sp, 16, d),
        ldpPost(v12, v13, sp, 16, d),
        ldpPost(v10, v11, sp, 16, d),
        ldpPost(v8, v9, sp, 16, d),

        ldpPost(x27, x28, sp, 16),
        ldpPost(x25, x26, sp, 16),
        ldpPost(x23, x24, sp, 16),
        ldpPost(x21, x22, sp, 16),
        ldpPost(x19, x20, sp, 16),

        // Restore stack pointer
        ldpPost(x29, x30, sp, 16),

        ret()
    });
    kernel.write("add_primitive.bin");
    kernel.set_kernel();
}