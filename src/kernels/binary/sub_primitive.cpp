#include "sub_primitive.h"
#include "Kernel.h"

#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
#include "instructions/all_instructions.h"

using enum gpr_t;
using enum simd_fp_t;
using enum neon_size_spec_t;
using enum arr_spec_t;

using namespace mini_jit::instructions;

void mini_jit::kernels::binary::sub(mini_jit::Kernel &kernel,
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
        base::stpPre(x29, x30, sp, -16),
        base::movSP(x29, sp),

        // Callee-saved registers
        base::stpPre(x19, x20, sp, -16),
        base::stpPre(x21, x22, sp, -16),
        base::stpPre(x23, x24, sp, -16),
        base::stpPre(x25, x26, sp, -16),
        base::stpPre(x27, x28, sp, -16),

        simd_fp::stpPre(v8, v9, sp, -16, neon_size_spec_t::d),
        simd_fp::stpPre(v10, v11, sp, -16, neon_size_spec_t::d),
        simd_fp::stpPre(v12, v13, sp, -16, neon_size_spec_t::d),
        simd_fp::stpPre(v14, v15, sp, -16, neon_size_spec_t::d),

        // Strides
        base::lsl(x3, x3, 2), // leading dimension of A
        base::lsl(x4, x4, 2), // leading dimension of B
        base::lsl(x5, x5, 2), // leading dimension of C

        // Save base matrix pointers
        base::mov(x6, x0), // A
        base::mov(x7, x1), // B
        base::mov(x8, x2), // C

        // Set n loop counter
        base::mov(x9, n)
    });

    // Start n loop (1 column)
    kernel.add_label("n_loop");

    kernel.add_instr({
        // Set m loop counter
        base::mov(x10, mLoopIterations),

        // working pointers for rows
        base::mov(x11, x6),   // A
        base::mov(x12, x7),   // B
        base::mov(x13, x8)    // C
    });

    if (mLoopIterations > 0)
    {
        kernel.add_label("m_16_loop");
        kernel.add_instr({
            // load 16 elements from A
            simd_fp::ldp(v0, v1, x11, 0, q),
            simd_fp::ldp(v2, v3, x11, 32, q),

            // load 16 elements from B
            simd_fp::ldp(v4, v5, x12, 0, q),
            simd_fp::ldp(v6, v7, x12, 32, q),

            // compute C = A + B
            simd_fp::fsubVec(v8, v0, v4, s4),
            simd_fp::fsubVec(v9, v1, v5, s4),
            simd_fp::fsubVec(v10, v2, v6, s4),
            simd_fp::fsubVec(v11, v3, v7, s4),

            // store 16 elements to C
            simd_fp::stp(v8, v9, x13, 0, q),
            simd_fp::stp(v10, v11, x13, 32, q),

            // jump by 16 rows
            base::add(x11, x11, 16*4, 0),
            base::add(x12, x12, 16*4, 0),
            base::add(x13, x13, 16*4, 0),

            // decrement m loop counter
            base::sub(x10, x10, 1, 0),
        });
        // check if loop counter is zero
        kernel.add_instr(base::cbnz(x10, -kernel.getInstrCountFromLabel("m_16_loop") * 4));
    }

    if (mLoopRemainder > 0)
    {
        switch (mLoopRemainder)
        {
        case 1:
            kernel.add_instr({
                // 1 element
                simd_fp::ldr(v0, x11, 0, s),
                simd_fp::ldr(v1, x12, 0, s),
                simd_fp::fsubScalar(v2, v0, v1, s),
                simd_fp::str(v2, x13, 0, s)
            });
            break;
        case 2:
            kernel.add_instr({
                // 2 elements
                simd_fp::ldr(v0, x11, 0, d),
                simd_fp::ldr(v1, x12, 0, d),
                simd_fp::fsubVec(v2, v0, v1, s2),
                simd_fp::str(v2, x13, 0, d)
            });
            break;
        case 3:
            kernel.add_instr({
                // 2 elements
                simd_fp::ldr(v0, x11, 0, d),
                simd_fp::ldr(v1, x12, 0, d),
                simd_fp::fsubVec(v2, v0, v1, s2),
                simd_fp::str(v2, x13, 0, d),

                // 1 element
                simd_fp::ldr(v3, x11, 2*4, s),
                simd_fp::ldr(v4, x12, 2*4, s),
                simd_fp::fsubScalar(v5, v3, v4, s),
                simd_fp::str(v5, x13, 2*4, s)
            });
            break;
        case 4:
            kernel.add_instr({
                // 4 elements
                simd_fp::ldr(v0, x11, 0, q),
                simd_fp::ldr(v1, x12, 0, q),
                simd_fp::fsubVec(v2, v0, v1, s4),
                simd_fp::str(v2, x13, 0, q)
            });
            break;
        case 5:
            kernel.add_instr({
                // 4 elements
                simd_fp::ldr(v0, x11, 0, q),
                simd_fp::ldr(v1, x12, 0, q),
                simd_fp::fsubVec(v2, v0, v1, s4),
                simd_fp::str(v2, x13, 0, q),
                
                // 5 elements
                simd_fp::ldr(v3, x11, 4*4, s),
                simd_fp::ldr(v4, x12, 4*4, s),
                simd_fp::fsubScalar(v5, v3, v4, s),
                simd_fp::str(v5, x13, 4*4, s)
            });
            break;
        case 6:
            kernel.add_instr({
                // 4 elements
                simd_fp::ldr(v0, x11, 0, q),
                simd_fp::ldr(v1, x12, 0, q),
                simd_fp::fsubVec(v2, v0, v1, s4),
                simd_fp::str(v2, x13, 0, q),

                // 2 elements
                simd_fp::ldr(v3, x11, 4*4, d),
                simd_fp::ldr(v4, x12, 4*4, d),
                simd_fp::fsubVec(v5, v3, v4, s2),
                simd_fp::str(v5, x13, 4*4, d)
            });
            break;
        case 7:
            kernel.add_instr({
                // 4 elements
                simd_fp::ldr(v0, x11, 0, q),
                simd_fp::ldr(v1, x12, 0, q),
                simd_fp::fsubVec(v2, v0, v1, s4),
                simd_fp::str(v2, x13, 0, q),

                // 2 elements
                simd_fp::ldr(v3, x11, 4*4, d),
                simd_fp::ldr(v4, x12, 4*4, d),
                simd_fp::fsubVec(v5, v3, v4, s2),
                simd_fp::str(v5, x13, 4*4, d),

                // 1 element
                simd_fp::ldr(v6, x11, 6*4, s),
                simd_fp::ldr(v7, x12, 6*4, s),
                simd_fp::fsubScalar(v8, v6, v7, s),
                simd_fp::str(v8, x13, 6*4, s)
            });
            break;
        case 8:
            kernel.add_instr({
                // 8 elements
                simd_fp::ldp(v0, v1, x11, 0, q),
                simd_fp::ldp(v2, v3, x12, 0, q),
                simd_fp::fsubVec(v4, v0, v2, s4),
                simd_fp::fsubVec(v5, v1, v3, s4),
                simd_fp::stp(v4, v5, x13, 0, q)
            });
            break;
        case 9:
            kernel.add_instr({
                // 8 elements
                simd_fp::ldp(v0, v1, x11, 0, q),
                simd_fp::ldp(v2, v3, x12, 0, q),
                simd_fp::fsubVec(v4, v0, v2, s4),
                simd_fp::fsubVec(v5, v1, v3, s4),
                simd_fp::stp(v4, v5, x13, 0, q),

                // 1 element
                simd_fp::ldr(v6, x11, 8*4, s),
                simd_fp::ldr(v7, x12, 8*4, s),
                simd_fp::fsubScalar(v8, v6, v7, s),
                simd_fp::str(v8, x13, 8*4, s)
            });
            break;
        case 10:
            kernel.add_instr({
                // 8 elements
                simd_fp::ldp(v0, v1, x11, 0, q),
                simd_fp::ldp(v2, v3, x12, 0, q),
                simd_fp::fsubVec(v4, v0, v2, s4),
                simd_fp::fsubVec(v5, v1, v3, s4),
                simd_fp::stp(v4, v5, x13, 0, q),

                // 2 elements
                simd_fp::ldr(v6, x11, 8*4, d),
                simd_fp::ldr(v7, x12, 8*4, d),
                simd_fp::fsubVec(v8, v6, v7, s2),
                simd_fp::str(v8, x13, 8*4, d)
            });
            break;
        case 11:
            kernel.add_instr({
                // 8 elements
                simd_fp::ldp(v0, v1, x11, 0, q),
                simd_fp::ldp(v2, v3, x12, 0, q),
                simd_fp::fsubVec(v4, v0, v2, s4),
                simd_fp::fsubVec(v5, v1, v3, s4),
                simd_fp::stp(v4, v5, x13, 0, q),

                // 2 elements
                simd_fp::ldr(v6, x11, 8*4, d),
                simd_fp::ldr(v7, x12, 8*4, d),
                simd_fp::fsubVec(v8, v6, v7, s2),
                simd_fp::str(v8, x13, 8*4, d),

                // 1 element
                simd_fp::ldr(v9, x11, 10*4, s),
                simd_fp::ldr(v10, x12, 10*4, s),
                simd_fp::fsubScalar(v11, v9, v10, s),
                simd_fp::str(v11, x13, 10*4, s)
            });
            break;
        case 12:
            kernel.add_instr({
                // 8 elements
                simd_fp::ldp(v0, v1, x11, 0, q),
                simd_fp::ldp(v2, v3, x12, 0, q),
                simd_fp::fsubVec(v4, v0, v2, s4),
                simd_fp::fsubVec(v5, v1, v3, s4),
                simd_fp::stp(v4, v5, x13, 0, q),

                // 4 elements
                simd_fp::ldr(v6, x11, 8*4, q),
                simd_fp::ldr(v7, x12, 8*4, q),
                simd_fp::fsubVec(v8, v6, v7, s4),
                simd_fp::str(v8, x13, 8*4, q)
            });
            break;
        case 13:
            kernel.add_instr({
                // 8 elements
                simd_fp::ldp(v0, v1, x11, 0, q),
                simd_fp::ldp(v2, v3, x12, 0, q),
                simd_fp::fsubVec(v4, v0, v2, s4),
                simd_fp::fsubVec(v5, v1, v3, s4),
                simd_fp::stp(v4, v5, x13, 0, q),

                // 4 elements
                simd_fp::ldr(v6, x11, 8*4, q),
                simd_fp::ldr(v7, x12, 8*4, q),
                simd_fp::fsubVec(v8, v6, v7, s4),
                simd_fp::str(v8, x13, 8*4, q),

                // 1 element
                simd_fp::ldr(v9, x11, 12*4, s),
                simd_fp::ldr(v10, x12, 12*4, s),
                simd_fp::fsubScalar(v11, v9, v10, s),
                simd_fp::str(v11, x13, 12*4, s)
            });
            break;
        case 14:
            kernel.add_instr({
                // 8 elements
                simd_fp::ldp(v0, v1, x11, 0, q),
                simd_fp::ldp(v2, v3, x12, 0 , q),
                simd_fp::fsubVec(v4, v0, v2, s4),
                simd_fp::fsubVec(v5, v1, v3, s4),
                simd_fp::stp(v4, v5, x13, 0, q),

                // 4 elements
                simd_fp::ldr(v6, x11, 8*4, q),
                simd_fp::ldr(v7, x12, 8*4, q),
                simd_fp::fsubVec(v8, v6, v7, s4),
                simd_fp::str(v8, x13, 8*4, q),

                // 2 elements
                simd_fp::ldr(v9, x11, 12*4, d),
                simd_fp::ldr(v10, x12, 12*4, d),
                simd_fp::fsubVec(v11, v9, v10, s2),
                simd_fp::str(v11, x13, 12*4, d)
            });
            break;
        case 15:
            kernel.add_instr({
                // 8 elements
                simd_fp::ldp(v0, v1, x11, 0, q),
                simd_fp::ldp(v2, v3, x12, 0, q),
                simd_fp::fsubVec(v4, v0, v2, s4),
                simd_fp::fsubVec(v5, v1, v3, s4),
                simd_fp::stp(v4, v5, x13, 0, q),

                // 4 elements
                simd_fp::ldr(v6, x11, 8*4, q),
                simd_fp::ldr(v7, x12, 8*4, q),
                simd_fp::fsubVec(v8, v6, v7, s4),
                simd_fp::str(v8, x13, 8*4, q),

                // 2 elements
                simd_fp::ldr(v9, x11, 12*4, d),
                simd_fp::ldr(v10, x12, 12*4, d),
                simd_fp::fsubVec(v11, v9, v10, s2),
                simd_fp::str(v11, x13, 12*4, d),

                // 1 element
                simd_fp::ldr(v12, x11, 14*4, s),
                simd_fp::ldr(v13, x12, 14*4, s),
                simd_fp::fsubScalar(v14, v12, v13, s),
                simd_fp::str(v14, x13, 14*4, s)
            });
            break;
        default:
            break;
        }
    }

    kernel.add_instr({
        // jump to next column
        base::add(x6, x6, x3, 0, 0),
        base::add(x7, x7, x4, 0, 0),
        base::add(x8, x8, x5, 0, 0),

        // decrement n loop counter
        base::sub(x9, x9, 1, 0)
    });
    // check if n loop counter is zero
    int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
    kernel.add_instr(base::cbnz(x9, -l_nLoopInstrCount * 4));

    kernel.add_instr({
        // Restore callee-saved registers
        simd_fp::ldpPost(v14, v15, sp, 16, d),
        simd_fp::ldpPost(v12, v13, sp, 16, d),
        simd_fp::ldpPost(v10, v11, sp, 16, d),
        simd_fp::ldpPost(v8, v9, sp, 16, d),

        base::ldpPost(x27, x28, sp, 16),
        base::ldpPost(x25, x26, sp, 16),
        base::ldpPost(x23, x24, sp, 16),
        base::ldpPost(x21, x22, sp, 16),
        base::ldpPost(x19, x20, sp, 16),

        // Restore stack pointer
        base::ldpPost(x29, x30, sp, 16),

        ret()
    });

    kernel.write("sub_primitive.bin");
    kernel.set_kernel();
}