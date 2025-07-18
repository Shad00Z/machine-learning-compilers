#include <mlc/instructions/all_instructions.h>
#include <mlc/kernels/matmul/matmul_m_n_k.h>
#include <mlc/kernels/matmul/subkernels/all_subkernels.h>
#include <mlc/registers/gp_registers.h>
#include <mlc/registers/simd_fp_registers.h>

using gpr_t            = mini_jit::registers::gpr_t;
using simd_fp_t        = mini_jit::registers::simd_fp_t;
using arr_spec_t       = mini_jit::registers::arr_spec_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace inst                = mini_jit::instructions;
namespace base                = inst::base;
namespace simd_fp             = inst::simd_fp;
namespace internal_subkernels = mini_jit::kernels::matmul::subkernels::internal;

void mini_jit::kernels::matmul::matmul_m_n_k(mini_jit::Kernel& kernel,
                                             int               m,
                                             int               n,
                                             int               k)
{
    // Prepare the kernel
    int nLoopIterations = n / 4;
    int nLoopRemainder  = n % 4;
    int mLoopIterations = m / 16;
    int mLoopRemainder  = m % 16;

    // PCS
    kernel.add_instr({base::stpPre(gpr_t::x29, gpr_t::x30, gpr_t::sp, -16),
                      base::movSP(gpr_t::x29, gpr_t::sp),

                      // // Save callee-saved registers
                      base::stpPre(gpr_t::x19, gpr_t::x20, gpr_t::sp, -16),
                      base::stpPre(gpr_t::x21, gpr_t::x22, gpr_t::sp, -16),
                      base::stpPre(gpr_t::x23, gpr_t::x24, gpr_t::sp, -16),
                      // base::stpPre(gpr_t::x25, gpr_t::x26, gpr_t::sp, -16),
                      // base::stpPre(gpr_t::x27, gpr_t::x28, gpr_t::sp, -16),

                      simd_fp::stpPre(simd_fp_t::v8, simd_fp_t::v9, gpr_t::sp, -16, neon_size_spec_t::d),
                      simd_fp::stpPre(simd_fp_t::v10, simd_fp_t::v11, gpr_t::sp, -16, neon_size_spec_t::d),
                      simd_fp::stpPre(simd_fp_t::v12, simd_fp_t::v13, gpr_t::sp, -16, neon_size_spec_t::d),
                      simd_fp::stpPre(simd_fp_t::v14, simd_fp_t::v15, gpr_t::sp, -16, neon_size_spec_t::d),

                      // Strides
                      // lsl #2 -> *4
                      base::lsl(gpr_t::x3, gpr_t::x3, 2),
                      base::lsl(gpr_t::x4, gpr_t::x4, 2),
                      base::lsl(gpr_t::x5, gpr_t::x5, 2),

                      base::lsl(gpr_t::x22, gpr_t::x4, 2), // ldb * 4 columns
                      base::lsl(gpr_t::x23, gpr_t::x5, 2), // ldc * 4 columns

                      // set base matrix pointers
                      base::mov(gpr_t::x20, gpr_t::x1),
                      base::mov(gpr_t::x21, gpr_t::x2),

                      // N loop counters
                      base::mov(gpr_t::x19, nLoopIterations)});

    if (nLoopIterations > 0)
    {
        // n_loop:
        kernel.add_label("n_loop");

        // Save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x0));   // A
        kernel.add_instr(base::mov(gpr_t::x9, gpr_t::x20));  // B
        kernel.add_instr(base::mov(gpr_t::x10, gpr_t::x21)); // C

        if (mLoopIterations > 0)
        {
            internal_subkernels::generateM16N4Loop(kernel, mLoopIterations, k);
        }

        if (mLoopRemainder > 0)
        {
            // set up k loop counter
            kernel.add_instr(base::mov(gpr_t::x14, k));
            // save base matrix pointers
            kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x8)); // A
            kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9)); // B
            kernel.add_instr(base::mov(gpr_t::x17, 0));         // row count B

            switch (mLoopRemainder)
            {
            case 1:
                internal_subkernels::generateM1N4Loop(kernel);
                break;
            case 2:
                internal_subkernels::generateM2N4Loop(kernel);
                break;
            case 3:
                internal_subkernels::generateM3N4Loop(kernel);
                break;
            case 4:
                internal_subkernels::generateM4N4Loop(kernel);
                break;
            case 5:
                internal_subkernels::generateM5N4Loop(kernel);
                break;
            case 6:
                internal_subkernels::generateM6N4Loop(kernel);
                break;
            case 7:
                internal_subkernels::generateM7N4Loop(kernel);
                break;
            case 8:
                internal_subkernels::generateM8N4Loop(kernel);
                break;
            case 9:
                internal_subkernels::generateM9N4Loop(kernel);
                break;
            case 10:
                internal_subkernels::generateM10N4Loop(kernel);
                break;
            case 11:
                internal_subkernels::generateM11N4Loop(kernel);
                break;
            case 12:
                internal_subkernels::generateM12N4Loop(kernel);
                break;
            case 13:
                internal_subkernels::generateM13N4Loop(kernel);
                break;
            case 14:
                internal_subkernels::generateM14N4Loop(kernel);
                break;
            case 15:
                internal_subkernels::generateM15N4Loop(kernel);
                break;
            default:
                break;
            }
        }

        // increase B and C pointers for next block
        // (jump 4 columns) 4*x4, 4*x5
        kernel.add_instr(base::add(gpr_t::x20, gpr_t::x20, gpr_t::x22, 0, 0));
        kernel.add_instr(base::add(gpr_t::x21, gpr_t::x21, gpr_t::x23, 0, 0));
        // decrement n loop counter
        kernel.add_instr(base::sub(gpr_t::x19, gpr_t::x19, 1, 0));

        // check if loop counter is zero
        int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
        kernel.add_instr(base::cbnz(gpr_t::x19, -l_nLoopInstrCount * 4));
        // END N LOOP
    }

    if (nLoopRemainder > 0)
    {
        // Save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x0));   // A
        kernel.add_instr(base::mov(gpr_t::x9, gpr_t::x20));  // B
        kernel.add_instr(base::mov(gpr_t::x10, gpr_t::x21)); // C

        switch (nLoopRemainder)
        {
        case 1:
            mini_jit::kernels::matmul::internal::generateN1Loop(kernel, mLoopIterations, mLoopRemainder, k);
            break;
        case 2:
            mini_jit::kernels::matmul::internal::generateN2Loop(kernel, mLoopIterations, mLoopRemainder, k);
            break;
        case 3:
            mini_jit::kernels::matmul::internal::generateN3Loop(kernel, mLoopIterations, mLoopRemainder, k);
            break;
        default:
            break;
        }
    }

    // Restore callee-saved registers
    kernel.add_instr(simd_fp::ldpPost(simd_fp_t::v14, simd_fp_t::v15, gpr_t::sp, 16, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldpPost(simd_fp_t::v12, simd_fp_t::v13, gpr_t::sp, 16, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldpPost(simd_fp_t::v10, simd_fp_t::v11, gpr_t::sp, 16, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldpPost(simd_fp_t::v8, simd_fp_t::v9, gpr_t::sp, 16, neon_size_spec_t::d));

    // kernel.add_instr(base::ldpPost(gpr_t::x27, gpr_t::x28, gpr_t::sp, 16));
    // kernel.add_instr(base::ldpPost(gpr_t::x25, gpr_t::x26, gpr_t::sp, 16));
    kernel.add_instr(base::ldpPost(gpr_t::x23, gpr_t::x24, gpr_t::sp, 16));
    kernel.add_instr(base::ldpPost(gpr_t::x21, gpr_t::x22, gpr_t::sp, 16));
    kernel.add_instr(base::ldpPost(gpr_t::x19, gpr_t::x20, gpr_t::sp, 16));

    // Restore stack pointer
    kernel.add_instr(base::ldpPost(gpr_t::x29, gpr_t::x30, gpr_t::sp, 16));

    kernel.add_instr(base::ret());

    kernel.write("matmul_m_n_k.bin");
    kernel.set_kernel();
}

void mini_jit::kernels::matmul::internal::generateN1Loop(mini_jit::Kernel& kernel,
                                                         int               mLoopIterations,
                                                         int               mLoopRemainder,
                                                         int               k)
{
    if (mLoopIterations > 0)
    {
        internal_subkernels::generateM16N1Loop(kernel, mLoopIterations, k);
    }

    if (mLoopRemainder > 0)
    {
        // set up k loop counter
        kernel.add_instr(base::mov(gpr_t::x14, k));
        // save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x8)); // A
        kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9)); // B
        kernel.add_instr(base::mov(gpr_t::x17, 0));         // row count B

        switch (mLoopRemainder)
        {
        case 1:
            internal_subkernels::generateM1N1Loop(kernel);
            break;
        case 2:
            internal_subkernels::generateM2N1Loop(kernel);
            break;
        case 3:
            internal_subkernels::generateM3N1Loop(kernel);
            break;
        case 4:
            internal_subkernels::generateM4N1Loop(kernel);
            break;
        case 5:
            internal_subkernels::generateM5N1Loop(kernel);
            break;
        case 6:
            internal_subkernels::generateM6N1Loop(kernel);
            break;
        case 7:
            internal_subkernels::generateM7N1Loop(kernel);
            break;
        case 8:
            internal_subkernels::generateM8N1Loop(kernel);
            break;
        case 9:
            internal_subkernels::generateM9N1Loop(kernel);
            break;
        case 10:
            internal_subkernels::generateM10N1Loop(kernel);
            break;
        case 11:
            internal_subkernels::generateM11N1Loop(kernel);
            break;
        case 12:
            internal_subkernels::generateM12N1Loop(kernel);
            break;
        case 13:
            internal_subkernels::generateM13N1Loop(kernel);
            break;
        case 14:
            internal_subkernels::generateM14N1Loop(kernel);
            break;
        case 15:
            internal_subkernels::generateM15N1Loop(kernel);
            break;
        default:
            break;
        }
    }
}

void mini_jit::kernels::matmul::internal::generateN2Loop(mini_jit::Kernel& kernel,
                                                         int               mLoopIterations,
                                                         int               mLoopRemainder,
                                                         int               k)
{
    if (mLoopIterations > 0)
    {
        internal_subkernels::generateM16N2Loop(kernel, mLoopIterations, k);
    }

    if (mLoopRemainder > 0)
    {
        // set up k loop counter
        kernel.add_instr(base::mov(gpr_t::x14, k));
        // save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x8)); // A
        kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9)); // B
        kernel.add_instr(base::mov(gpr_t::x17, 0));         // row count B

        switch (mLoopRemainder)
        {
        case 1:
            internal_subkernels::generateM1N2Loop(kernel);
            break;
        case 2:
            internal_subkernels::generateM2N2Loop(kernel);
            break;
        case 3:
            internal_subkernels::generateM3N2Loop(kernel);
            break;
        case 4:
            internal_subkernels::generateM4N2Loop(kernel);
            break;
        case 5:
            internal_subkernels::generateM5N2Loop(kernel);
            break;
        case 6:
            internal_subkernels::generateM6N2Loop(kernel);
            break;
        case 7:
            internal_subkernels::generateM7N2Loop(kernel);
            break;
        case 8:
            internal_subkernels::generateM8N2Loop(kernel);
            break;
        case 9:
            internal_subkernels::generateM9N2Loop(kernel);
            break;
        case 10:
            internal_subkernels::generateM10N2Loop(kernel);
            break;
        case 11:
            internal_subkernels::generateM11N2Loop(kernel);
            break;
        case 12:
            internal_subkernels::generateM12N2Loop(kernel);
            break;
        case 13:
            internal_subkernels::generateM13N2Loop(kernel);
            break;
        case 14:
            internal_subkernels::generateM14N2Loop(kernel);
            break;
        case 15:
            internal_subkernels::generateM15N2Loop(kernel);
            break;
        default:
            break;
        }
    }
}

void mini_jit::kernels::matmul::internal::generateN3Loop(mini_jit::Kernel& kernel,
                                                         int               mLoopIterations,
                                                         int               mLoopRemainder,
                                                         int               k)
{
    if (mLoopIterations > 0)
    {
        internal_subkernels::generateM16N3Loop(kernel, mLoopIterations, k);
    }

    if (mLoopRemainder > 0)
    {
        // set up k loop counter
        kernel.add_instr(base::mov(gpr_t::x14, k));
        // save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x8)); // A
        kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9)); // B
        kernel.add_instr(base::mov(gpr_t::x17, 0));         // row count B

        switch (mLoopRemainder)
        {
        case 1:
            internal_subkernels::generateM1N3Loop(kernel);
            break;
        case 2:
            internal_subkernels::generateM2N3Loop(kernel);
            break;
        case 3:
            internal_subkernels::generateM3N3Loop(kernel);
            break;
        case 4:
            internal_subkernels::generateM4N3Loop(kernel);
            break;
        case 5:
            internal_subkernels::generateM5N3Loop(kernel);
            break;
        case 6:
            internal_subkernels::generateM6N3Loop(kernel);
            break;
        case 7:
            internal_subkernels::generateM7N3Loop(kernel);
            break;
        case 8:
            internal_subkernels::generateM8N3Loop(kernel);
            break;
        case 9:
            internal_subkernels::generateM9N3Loop(kernel);
            break;
        case 10:
            internal_subkernels::generateM10N3Loop(kernel);
            break;
        case 11:
            internal_subkernels::generateM11N3Loop(kernel);
            break;
        case 12:
            internal_subkernels::generateM12N3Loop(kernel);
            break;
        case 13:
            internal_subkernels::generateM13N3Loop(kernel);
            break;
        case 14:
            internal_subkernels::generateM14N3Loop(kernel);
            break;
        case 15:
            internal_subkernels::generateM15N3Loop(kernel);
            break;
        default:
            break;
        }
    }
}
