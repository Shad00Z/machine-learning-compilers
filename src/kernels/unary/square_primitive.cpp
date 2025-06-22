#include "square_primitive.h"
#include "Kernel.h"

#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
#include "instructions/all_instructions.h"

namespace inst = mini_jit::instructions;
namespace base = inst::base;
namespace simd_fp = inst::simd_fp;

void mini_jit::kernels::unary::square(mini_jit::Kernel &kernel,
                                      u_int32_t m,
                                      u_int32_t n)
{
    // Inputs:
    // x0: pointer to A
    // x1: pointer to B
    // x2: leading dimension of A
    // x3: leading dimension of B

    // Prepare the kernel
    int mLoopIterations = m / 8;
    int mLoopRemainder = m % 8;

    kernel.add_instr({
        // PCS
        base::stpPre(gpr_t::x29, gpr_t::x30, gpr_t::sp, -16),
        base::movSP(gpr_t::x29, gpr_t::sp),

        // Compute strides (* 4, because of 4 bytes per fp32 element)
        base::lsl(gpr_t::x2, gpr_t::x2, 2),
        base::lsl(gpr_t::x3, gpr_t::x3, 2),

        // Save base matrix pointers
        base::mov(gpr_t::x4, gpr_t::x0), // A
        base::mov(gpr_t::x5, gpr_t::x1), // B

        // Set n loop counter
        base::mov(gpr_t::x6, n)
    });

    // Start n loop (1 column)
    kernel.add_label("n_loop");

    // Set m loop counter
    kernel.add_instr({
        base::mov(gpr_t::x7, mLoopIterations),

        // working pointers for rows
        base::mov(gpr_t::x8, gpr_t::x4), // A
        base::mov(gpr_t::x9, gpr_t::x5)  // B
    });

    if (mLoopIterations > 0)
    {
        kernel.add_label("m_8_loop");
        kernel.add_instr({
            // load 8 elements from A into v0, v1
            simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x8, 0, neon_size_spec_t::q),

            // Zero output registers v2, v3
            simd_fp::eor(simd_fp_t::v2, simd_fp_t::v2, simd_fp_t::v2, arr_spec_t::b16),
            simd_fp::eor(simd_fp_t::v3, simd_fp_t::v3, simd_fp_t::v3, arr_spec_t::b16),

            // Square: v2 = 0 + (v0 × v0), v3 = 0 + (v1 × v1)
            simd_fp::fmlaVec(simd_fp_t::v2, simd_fp_t::v0, simd_fp_t::v0, arr_spec_t::s4),
            simd_fp::fmlaVec(simd_fp_t::v3, simd_fp_t::v1, simd_fp_t::v1, arr_spec_t::s4),

            // store 8 squared elements to B
            simd_fp::stp(simd_fp_t::v2, simd_fp_t::v3, gpr_t::x9, 0, neon_size_spec_t::q),

            // jump by 8 rows
            base::add(gpr_t::x8, gpr_t::x8, 8*4, 0),
            base::add(gpr_t::x9, gpr_t::x9, 8*4, 0),

            // decrement m loop counter
            base::sub(gpr_t::x7, gpr_t::x7, 1, 0),
        });
        // check if loop counter is zero
        kernel.add_instr(base::cbnz(gpr_t::x7, -kernel.getInstrCountFromLabel("m_8_loop") * 4));
    }

    if (mLoopRemainder > 0)
    {
        switch (mLoopRemainder)
        {
        case 1:
            kernel.add_instr({
                simd_fp::ldr(simd_fp_t::v0, gpr_t::x8, 0, neon_size_spec_t::s),
                simd_fp::eor(simd_fp_t::v1, simd_fp_t::v1, simd_fp_t::v1, arr_spec_t::b16),
                simd_fp::fmadd(simd_fp_t::v1, simd_fp_t::v0, simd_fp_t::v0, simd_fp_t::v1, neon_size_spec_t::s),
                simd_fp::str(simd_fp_t::v1, gpr_t::x9, 0, neon_size_spec_t::s)
            });
            break;
        case 2:
            kernel.add_instr({
                simd_fp::ldr(simd_fp_t::v0, gpr_t::x8, 0, neon_size_spec_t::d),
                simd_fp::eor(simd_fp_t::v1, simd_fp_t::v1, simd_fp_t::v1, arr_spec_t::b16),
                simd_fp::fmlaVec(simd_fp_t::v1, simd_fp_t::v0, simd_fp_t::v0, arr_spec_t::s2),
                simd_fp::str(simd_fp_t::v1, gpr_t::x9, 0, neon_size_spec_t::d)
            });
            break;
        case 3:
            // 2 elements
            kernel.add_instr({
                simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x8, 2*4, neon_size_spec_t::d),
                simd_fp::eor(simd_fp_t::v1, simd_fp_t::v1, simd_fp_t::v1, arr_spec_t::b16),
                simd_fp::fmlaVec(simd_fp_t::v1, simd_fp_t::v0, simd_fp_t::v0, arr_spec_t::s2),
                simd_fp::strPost(simd_fp_t::v1, gpr_t::x9, 2*4, neon_size_spec_t::d)
            });
            // 1 element
            kernel.add_instr({
                simd_fp::ldr(simd_fp_t::v0, gpr_t::x8, 0, neon_size_spec_t::s),
                simd_fp::eor(simd_fp_t::v1, simd_fp_t::v1, simd_fp_t::v1, arr_spec_t::b16),
                simd_fp::fmadd(simd_fp_t::v1, simd_fp_t::v0, simd_fp_t::v0, simd_fp_t::v1, neon_size_spec_t::s),
                simd_fp::str(simd_fp_t::v1, gpr_t::x9, 0, neon_size_spec_t::s)
            });
            break;
        case 4:
            kernel.add_instr({
                simd_fp::ldr(simd_fp_t::v0, gpr_t::x8, 0, neon_size_spec_t::q),
                simd_fp::eor(simd_fp_t::v1, simd_fp_t::v1, simd_fp_t::v1, arr_spec_t::b16),
                simd_fp::fmlaVec(simd_fp_t::v1, simd_fp_t::v0, simd_fp_t::v0, arr_spec_t::s4),
                simd_fp::str(simd_fp_t::v1, gpr_t::x9, 0, neon_size_spec_t::q)
            });
            break;
        case 5:
            // 4 elements
            kernel.add_instr({
                simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x8, 4*4, neon_size_spec_t::q),
                simd_fp::eor(simd_fp_t::v1, simd_fp_t::v1, simd_fp_t::v1, arr_spec_t::b16),
                simd_fp::fmlaVec(simd_fp_t::v1, simd_fp_t::v0, simd_fp_t::v0, arr_spec_t::s4),
                simd_fp::strPost(simd_fp_t::v1, gpr_t::x9, 4*4, neon_size_spec_t::q)
            });
            // 1 element
            kernel.add_instr({
                simd_fp::ldr(simd_fp_t::v0, gpr_t::x8, 0, neon_size_spec_t::s),
                simd_fp::eor(simd_fp_t::v2, simd_fp_t::v2, simd_fp_t::v2, arr_spec_t::b16),
                simd_fp::fmadd(simd_fp_t::v2, simd_fp_t::v0, simd_fp_t::v0, simd_fp_t::v2, neon_size_spec_t::s),
                simd_fp::str(simd_fp_t::v2, gpr_t::x9, 0, neon_size_spec_t::s)
            });
            break;
        case 6:
            // 4 elements
            kernel.add_instr({
                simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x8, 4*4, neon_size_spec_t::q),
                simd_fp::eor(simd_fp_t::v1, simd_fp_t::v1, simd_fp_t::v1, arr_spec_t::b16),
                simd_fp::fmlaVec(simd_fp_t::v1, simd_fp_t::v0, simd_fp_t::v0, arr_spec_t::s4),
                simd_fp::strPost(simd_fp_t::v1, gpr_t::x9, 4*4, neon_size_spec_t::q)
            });
            // 2 elements
            kernel.add_instr({
                simd_fp::ldr(simd_fp_t::v0, gpr_t::x8, 0, neon_size_spec_t::d),
                simd_fp::eor(simd_fp_t::v2, simd_fp_t::v2, simd_fp_t::v2, arr_spec_t::b16),
                simd_fp::fmlaVec(simd_fp_t::v2, simd_fp_t::v0, simd_fp_t::v0, arr_spec_t::s2),
                simd_fp::str(simd_fp_t::v2, gpr_t::x9, 0, neon_size_spec_t::d)
            });
            break;
        case 7:
            // 4 elements
            kernel.add_instr({
                simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x8, 4*4, neon_size_spec_t::q),
                simd_fp::eor(simd_fp_t::v1, simd_fp_t::v1, simd_fp_t::v1, arr_spec_t::b16),
                simd_fp::fmlaVec(simd_fp_t::v1, simd_fp_t::v0, simd_fp_t::v0, arr_spec_t::s4),
                simd_fp::strPost(simd_fp_t::v1, gpr_t::x9, 4*4, neon_size_spec_t::q)
            });
            // 2 elements
            kernel.add_instr({
                simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x8, 2*4, neon_size_spec_t::d),
                simd_fp::eor(simd_fp_t::v2, simd_fp_t::v2, simd_fp_t::v2, arr_spec_t::b16),
                simd_fp::fmlaVec(simd_fp_t::v2, simd_fp_t::v0, simd_fp_t::v0, arr_spec_t::s2),
                simd_fp::strPost(simd_fp_t::v2, gpr_t::x9, 2*4, neon_size_spec_t::d)
            });
            // 1 element
            kernel.add_instr({
                simd_fp::ldr(simd_fp_t::v0, gpr_t::x8, 0, neon_size_spec_t::s),
                simd_fp::eor(simd_fp_t::v3, simd_fp_t::v3, simd_fp_t::v3, arr_spec_t::b16),
                simd_fp::fmadd(simd_fp_t::v3, simd_fp_t::v0, simd_fp_t::v0, simd_fp_t::v3, neon_size_spec_t::s),
                simd_fp::str(simd_fp_t::v3, gpr_t::x9, 0, neon_size_spec_t::s)
            });
            break;
        default:
            break;
        }
    }

    kernel.add_instr({
        // jump to next column
        base::add(gpr_t::x4, gpr_t::x4, gpr_t::x2, 0, 0),
        base::add(gpr_t::x5, gpr_t::x5, gpr_t::x3, 0, 0),

        // decrement n loop counter
        base::sub(gpr_t::x6, gpr_t::x6, 1, 0)
    });
    // check if loop counter is zero
    int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
    kernel.add_instr(base::cbnz(gpr_t::x6, -l_nLoopInstrCount * 4));

    // Restore stack pointer
    kernel.add_instr(base::ldpPost(gpr_t::x29, gpr_t::x30, gpr_t::sp, 16));

    kernel.add_instr(inst::ret());
    kernel.write("square_primitive.bin");
    kernel.set_kernel();
}