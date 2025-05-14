#include "../registers/gp_registers.h"
#include "../registers/simd_fp_registers.h"
#include "../instructions/all_instructions.h"
#include "matmul_m_n_k.h"
#include "matmul_m_1_k.h"
#include "matmul_m_2_k.h"
#include "matmul_m_3_k.h"
#include "matmul_m_4_k.h"

#include <iostream>
#include <cstring>

using gpr_t = mini_jit::registers::gpr_t;
using simd_fp_t = mini_jit::registers::simd_fp_t;
using arr_spec_t = mini_jit::registers::arr_spec_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace inst = mini_jit::instructions;
namespace base = inst::base;
namespace simd_fp = inst::simd_fp;

void mini_jit::kernels::matmul_m_n_k(mini_jit::Kernel &kernel,
                                     int m,
                                     int n,
                                     int k)
{
    // Prepare the kernel
    int nLoopIterations = n / 4;
    int nLoopRemainder = n % 4;
    int mLoopIterations = m / 8;
    int mLoopRemainder = m % 8;

    // PCS
    kernel.add_instr(base::stpPre(gpr_t::x29, gpr_t::x30, gpr_t::sp, -16));
    kernel.add_instr(base::movSP(gpr_t::x29, gpr_t::sp));

    // // Save callee-saved registers
    kernel.add_instr(base::stpPre(gpr_t::x19, gpr_t::x20, gpr_t::sp, -16));
    kernel.add_instr(base::stpPre(gpr_t::x21, gpr_t::x22, gpr_t::sp, -16));
    kernel.add_instr(base::stpPre(gpr_t::x23, gpr_t::x24, gpr_t::sp, -16));
    kernel.add_instr(base::stpPre(gpr_t::x25, gpr_t::x26, gpr_t::sp, -16));
    kernel.add_instr(base::stpPre(gpr_t::x27, gpr_t::x28, gpr_t::sp, -16));

    kernel.add_instr(simd_fp::stpPre(simd_fp_t::v8, simd_fp_t::v9, gpr_t::sp, -16, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::stpPre(simd_fp_t::v10, simd_fp_t::v11, gpr_t::sp, -16, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::stpPre(simd_fp_t::v12, simd_fp_t::v13, gpr_t::sp, -16, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::stpPre(simd_fp_t::v14, simd_fp_t::v15, gpr_t::sp, -16, neon_size_spec_t::d));

    // Strides
    kernel.add_instr(base::mov(gpr_t::x6, 4));
    kernel.add_instr(base::mul(gpr_t::x3, gpr_t::x3, gpr_t::x6));
    kernel.add_instr(base::mul(gpr_t::x4, gpr_t::x4, gpr_t::x6));
    kernel.add_instr(base::mul(gpr_t::x5, gpr_t::x5, gpr_t::x6));

    kernel.add_instr(base::mul(gpr_t::x22, gpr_t::x4, gpr_t::x6)); // ldb * 4 columns
    kernel.add_instr(base::mul(gpr_t::x23, gpr_t::x5, gpr_t::x6)); // ldc * 4 columns

    // set base matrix pointers
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x1));
    kernel.add_instr(base::mov(gpr_t::x21, gpr_t::x2));

    // N loop counter
    kernel.add_instr(base::mov(gpr_t::x19, nLoopIterations));

    if (nLoopIterations > 0)
    {

        //n_loop:
        kernel.add_label("n_loop");

        // Save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x0)); // A
        kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x20)); // B
        kernel.add_instr(base::mov(gpr_t::x9, gpr_t::x21)); // C

        if (mLoopIterations > 0)
        {
            mini_jit::kernels::internal::generateM1N4Loop(kernel, mLoopIterations, k);
        }
    
        if (mLoopRemainder > 0)
        {
            // set up k loop counter
            kernel.add_instr(base::mov(gpr_t::x14, k));
            // save base matrix pointers
            kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x7)); // A
            kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8)); // B
            kernel.add_instr(base::mov(gpr_t::x17, 0));         // row count B
    
            if (mLoopRemainder == 1)
            {
                mini_jit::kernels::internal::generateM1N4LoopRest1(kernel);
            }
            else if (mLoopRemainder == 2)
            {
                mini_jit::kernels::internal::generateM1N4LoopRest2(kernel);
            }
            else if (mLoopRemainder == 3)
            {
                mini_jit::kernels::internal::generateM1N4LoopRest3(kernel, k);
            }
            else if (mLoopRemainder == 4)
            {
                mini_jit::kernels::internal::generateM1N4LoopRest4(kernel);
            }
            else if (mLoopRemainder == 5)
            {
                mini_jit::kernels::internal::generateM1N4LoopRest5(kernel);
            }
            else if (mLoopRemainder == 6)
            {
                mini_jit::kernels::internal::generateM1N4LoopRest6(kernel);
            }
            else if (mLoopRemainder == 7)
            {
                mini_jit::kernels::internal::generateM1N4LoopRest7(kernel, k);
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
        // // increase B and C pointers for next block
        // // (jump 6 columns) 6*x4, 6*x5
        // Save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x0)); // A
        kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x20)); // B
        kernel.add_instr(base::mov(gpr_t::x9, gpr_t::x21)); // C

        // prepare the kernel
        kernel.add_instr(base::mov(gpr_t::x11, mLoopIterations));

        if (nLoopRemainder == 1)
        {
            mini_jit::kernels::internal::generateNLoopRest1(kernel, mLoopIterations, mLoopRemainder, k);
        }
        else if (nLoopRemainder == 2)
        {
            mini_jit::kernels::internal::generateNLoopRest2(kernel, mLoopIterations, mLoopRemainder, k);
        }
        else if (nLoopRemainder == 3)
        {
            mini_jit::kernels::internal::generateNLoopRest3(kernel, mLoopIterations, mLoopRemainder, k);
        }
    }

    // Restore callee-saved registers
    kernel.add_instr(simd_fp::ldpPost(simd_fp_t::v14, simd_fp_t::v15, gpr_t::sp, 16, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldpPost(simd_fp_t::v12, simd_fp_t::v13, gpr_t::sp, 16, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldpPost(simd_fp_t::v10, simd_fp_t::v11, gpr_t::sp, 16, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldpPost(simd_fp_t::v8, simd_fp_t::v9, gpr_t::sp, 16, neon_size_spec_t::d));

    kernel.add_instr(base::ldpPost(gpr_t::x27, gpr_t::x28, gpr_t::sp, 16));
    kernel.add_instr(base::ldpPost(gpr_t::x25, gpr_t::x26, gpr_t::sp, 16));
    kernel.add_instr(base::ldpPost(gpr_t::x23, gpr_t::x24, gpr_t::sp, 16));
    kernel.add_instr(base::ldpPost(gpr_t::x21, gpr_t::x22, gpr_t::sp, 16));
    kernel.add_instr(base::ldpPost(gpr_t::x19, gpr_t::x20, gpr_t::sp, 16));

    // Restore stack pointer
    kernel.add_instr(base::ldpPost(gpr_t::x29, gpr_t::x30, gpr_t::sp, 16));

    kernel.add_instr(inst::ret());

    kernel.write("matmul_m_n_k.bin");
    kernel.set_kernel();
}

void mini_jit::kernels::internal::generateNLoopRest3(mini_jit::Kernel &kernel,
                                                     int mLoopIterations,
                                                     int mLoopRemainder,
                                                     int k)
{
    if (mLoopIterations > 0)
    {
        mini_jit::kernels::internal::generateMN3Loop(kernel, mLoopIterations, k);
    }

    if (mLoopRemainder > 0)
    {
        // set up k loop counter
        kernel.add_instr(base::mov(gpr_t::x14, k));
        // save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x7)); // A
        kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8)); // B
        kernel.add_instr(base::mov(gpr_t::x17, 0));         // row count B

        if (mLoopRemainder == 1)
        {
            mini_jit::kernels::internal::generateM1N3LoopRest1(kernel);
        }
        else if (mLoopRemainder == 2)
        {
            mini_jit::kernels::internal::generateM1N3LoopRest2(kernel);
        }
        else if (mLoopRemainder == 3)
        {
            mini_jit::kernels::internal::generateM1N3LoopRest3(kernel);
        }
        else if (mLoopRemainder == 4)
        {
            mini_jit::kernels::internal::generateM1N3LoopRest4(kernel);
        }
        else if (mLoopRemainder == 5)
        {
            mini_jit::kernels::internal::generateM1N3LoopRest5(kernel);
        }
        else if (mLoopRemainder == 6)
        {
            mini_jit::kernels::internal::generateM1N3LoopRest6(kernel);
        }
        else if (mLoopRemainder == 7)
        {
            mini_jit::kernels::internal::generateM1N3LoopRest7(kernel);
        }
    }
}

void mini_jit::kernels::internal::generateNLoopRest2(mini_jit::Kernel &kernel,
                                                     int mLoopIterations,
                                                     int mLoopRemainder,
                                                     int k)
{
    if (mLoopIterations > 0)
    {
        mini_jit::kernels::internal::generateMN2Loop(kernel, mLoopIterations, k);
    }

    if (mLoopRemainder > 0)
    {
        // set up k loop counter
        kernel.add_instr(base::mov(gpr_t::x14, k));
        // save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x7)); // A
        kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8)); // B
        kernel.add_instr(base::mov(gpr_t::x17, 0));         // row count B

        if (mLoopRemainder == 1)
        {
            mini_jit::kernels::internal::generateM1N2LoopRest1(kernel);
        }
        else if (mLoopRemainder == 2)
        {
            mini_jit::kernels::internal::generateM1N2LoopRest2(kernel);
        }
        else if (mLoopRemainder == 3)
        {
            mini_jit::kernels::internal::generateM1N2LoopRest3(kernel);
        }
        else if (mLoopRemainder == 4)
        {
            mini_jit::kernels::internal::generateM1N2LoopRest4(kernel);
        }
        else if (mLoopRemainder == 5)
        {
            mini_jit::kernels::internal::generateM1N2LoopRest5(kernel);
        }
        else if (mLoopRemainder == 6)
        {
            mini_jit::kernels::internal::generateM1N2LoopRest6(kernel);
        }
        else if (mLoopRemainder == 7)
        {
            mini_jit::kernels::internal::generateM1N2LoopRest7(kernel);
        }
    }
}

void mini_jit::kernels::internal::generateNLoopRest1(mini_jit::Kernel &kernel,
                                                     int mLoopIterations,
                                                     int mLoopRemainder,
                                                     int k)
{
    if (mLoopIterations > 0)
    {
        mini_jit::kernels::internal::generateMN1Loop(kernel, mLoopIterations, k);
    }

    if (mLoopRemainder > 0)
    {
        // set up k loop counter
        kernel.add_instr(base::mov(gpr_t::x14, k));
        // save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x7)); // A
        kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8)); // B
        kernel.add_instr(base::mov(gpr_t::x17, 0));         // row count B

        if (mLoopRemainder == 1)
        {
            mini_jit::kernels::internal::generateM1N1LoopRest1(kernel);
        }
        else if (mLoopRemainder == 2)
        {
            mini_jit::kernels::internal::generateM1N1LoopRest2(kernel);
        }
        else if (mLoopRemainder == 3)
        {
            mini_jit::kernels::internal::generateM1N1LoopRest3(kernel);
        }
        else if (mLoopRemainder == 4)
        {
            mini_jit::kernels::internal::generateM1N1LoopRest4(kernel);
        }
        else if (mLoopRemainder == 5)
        {
            mini_jit::kernels::internal::generateM1N1LoopRest5(kernel);
        }
        else if (mLoopRemainder == 6)
        {
            mini_jit::kernels::internal::generateM1N1LoopRest6(kernel);
        }
        else if (mLoopRemainder == 7)
        {
            mini_jit::kernels::internal::generateM1N1LoopRest7(kernel);
        }
    }
}
