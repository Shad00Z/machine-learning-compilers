#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
#include "instructions/all_instructions.h"
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

void mini_jit::kernels::matmul_m_4_k(mini_jit::Kernel &kernel,
                                     int m,
                                     int k)
{
    // Prepare the kernel
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

    // Save base matrix pointers
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x0)); // A
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x1)); // B
    kernel.add_instr(base::mov(gpr_t::x9, gpr_t::x2)); // C

    if (mLoopIterations > 0)
    {
        mini_jit::kernels::internal::generateMLoop(kernel, mLoopIterations, k);
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
            mini_jit::kernels::internal::generateMLoopRest1(kernel);
        }
        else if (mLoopRemainder == 2)
        {
            mini_jit::kernels::internal::generateMLoopRest2(kernel);
        }
        else if (mLoopRemainder == 3)
        {
            mini_jit::kernels::internal::generateMLoopRest3(kernel);
        }
        else if (mLoopRemainder == 4)
        {
            mini_jit::kernels::internal::generateMLoopRest4(kernel);
        }
        else if (mLoopRemainder == 5)
        {
            mini_jit::kernels::internal::generateMLoopRest5(kernel);
        }
        else if (mLoopRemainder == 6)
        {
            mini_jit::kernels::internal::generateMLoopRest6(kernel);
        }
        else if (mLoopRemainder == 7)
        {
            mini_jit::kernels::internal::generateMLoopRest7(kernel);
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

    kernel.write("matmul_m_4_k.bin");
    kernel.set_kernel();
}

void mini_jit::kernels::internal::generateMLoop(mini_jit::Kernel &kernel,
                                                int mLoopIterations,
                                                int k)
{
    // prepare the kernel
    kernel.add_instr(base::mov(gpr_t::x11, mLoopIterations));

    // START M_LOOP
    kernel.add_label("m_loop");
    // Load Matrix C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v2, simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::q));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v4, simd_fp_t::v5, gpr_t::x12, 0, neon_size_spec_t::q));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v6, simd_fp_t::v7, gpr_t::x12, 0, neon_size_spec_t::q));

    // Setup for Loop
    kernel.add_instr(base::mov(gpr_t::x14, k));         // K loop counter
    kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x7)); // Matrix A pointer
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8)); // Matrix B pointer
    kernel.add_instr(base::mov(gpr_t::x17, 0));         // Row index for Matrix B

    // START K_LOOP
    kernel.add_label("k_loop");
    //  Load column of A (8 values)
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x15, 0, neon_size_spec_t::q));

    // Load Column of Matrix B
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));

    // 1st Multiplication
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));

    // Load Column of Matrix B
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));

    // 2nd Multiplication
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v3, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));

    // Load Column of Matrix B
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));

    // 3rd Multiplication
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v4, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v5, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));

    // Load Column of Matrix B
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));

    // 4th Multiplication
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v7, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));

    // Decrement K
    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // END K_LOOP
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // Store Matrix C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v2, simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::q));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v4, simd_fp_t::v5, gpr_t::x12, 0, neon_size_spec_t::q));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v6, simd_fp_t::v7, gpr_t::x12, 0, neon_size_spec_t::q));

    // increase A and C pointers for next block
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, 8 * 4, 0));
    kernel.add_instr(base::add(gpr_t::x9, gpr_t::x9, 8 * 4, 0));

    // decrement M loop counter
    kernel.add_instr(base::sub(gpr_t::x11, gpr_t::x11, 1, 0));

    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_loop");
    kernel.add_instr(base::cbnz(gpr_t::x11, -l_mLoopInstrCount * 4));
    // END M_LOOP
}

void mini_jit::kernels::internal::generateMLoopRest1(mini_jit::Kernel &kernel)
{
    // Load Matrix C (1 value)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::s));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::s));

    // case_1_k_loop:
    kernel.add_label("case_1_k_loop");
    // load column of A (1 value)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v24, gpr_t::x15, 0, neon_size_spec_t::s));

    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    // B: COLUMN 3
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v3, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("case_1_k_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::s));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::s));
}

void mini_jit::kernels::internal::generateMLoopRest2(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (2 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::d));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::d));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::d));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::d));

    // case_2_k_loop:
    kernel.add_label("case_2_k_loop");
    // load column of A (2 values)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v24, gpr_t::x15, 0, neon_size_spec_t::d));

    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s2));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s2));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s2));
    // B: COLUMN 3
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v3, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s2));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("case_2_k_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::d));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::d));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::d));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::d));
}

void mini_jit::kernels::internal::generateMLoopRest3(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (3 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v0, gpr_t::x20, 0, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v0, gpr_t::x20, 1, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v0, gpr_t::x20, 2, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::mov(simd_fp_t::v0, gpr_t::wzr, 3, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v1, gpr_t::x20, 0, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v1, gpr_t::x20, 1, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v1, gpr_t::x20, 2, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::mov(simd_fp_t::v1, gpr_t::wzr, 3, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v2, gpr_t::x20, 0, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v2, gpr_t::x20, 1, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v2, gpr_t::x20, 2, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::mov(simd_fp_t::v2, gpr_t::wzr, 3, neon_size_spec_t::s));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v3, gpr_t::x20, 0, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v3, gpr_t::x20, 1, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v3, gpr_t::x20, 2, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::mov(simd_fp_t::v3, gpr_t::wzr, 3, neon_size_spec_t::s));

    // case_3_k_loop:
    kernel.add_label("case_3_k_loop");
    // load column of A (3 values)
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x15));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v24, gpr_t::x20, 0, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v24, gpr_t::x20, 1, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::ld1(simd_fp_t::v24, gpr_t::x20, 2, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::mov(simd_fp_t::v24, gpr_t::wzr, 3, neon_size_spec_t::s));

    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    // B: COLUMN 3
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v3, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("case_3_k_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (3 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::st1(simd_fp_t::v0, gpr_t::x20, 0, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::st1(simd_fp_t::v0, gpr_t::x20, 1, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::st1(simd_fp_t::v0, gpr_t::x20, 2, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::mov(simd_fp_t::v0, gpr_t::wzr, 3, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::st1(simd_fp_t::v1, gpr_t::x20, 0, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::st1(simd_fp_t::v1, gpr_t::x20, 1, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::st1(simd_fp_t::v1, gpr_t::x20, 2, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::mov(simd_fp_t::v1, gpr_t::wzr, 3, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::st1(simd_fp_t::v2, gpr_t::x20, 0, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::st1(simd_fp_t::v2, gpr_t::x20, 1, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::st1(simd_fp_t::v2, gpr_t::x20, 2, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::mov(simd_fp_t::v2, gpr_t::wzr, 3, neon_size_spec_t::s));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::st1(simd_fp_t::v3, gpr_t::x20, 0, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::st1(simd_fp_t::v3, gpr_t::x20, 1, neon_size_spec_t::s, 4));
    kernel.add_instr(simd_fp::st1(simd_fp_t::v3, gpr_t::x20, 2, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::mov(simd_fp_t::v3, gpr_t::wzr, 3, neon_size_spec_t::s));
}

void mini_jit::kernels::internal::generateMLoopRest4(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (4 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::q));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::q));

    // case_4_k_loop:
    kernel.add_label("case_4_k_loop");
    // load column of A (4 values)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v24, gpr_t::x15, 0, neon_size_spec_t::q));
    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    // B: COLUMN 3
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v3, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("case_4_k_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::q));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::q));
}

void mini_jit::kernels::internal::generateMLoopRest5(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (5 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x12, 16, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x12, 16, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v4, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v5, gpr_t::x12, 16, neon_size_spec_t::s));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v7, gpr_t::x12, 16, neon_size_spec_t::s));

    // case_5_k_loop:
    kernel.add_label("case_5_k_loop");
    // load column of A (5 values)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v24, gpr_t::x15, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v25, gpr_t::x15, 16, neon_size_spec_t::s));

    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, simd_fp_t::v1, neon_size_spec_t::s));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v3, simd_fp_t::v25, simd_fp_t::v29, simd_fp_t::v3, neon_size_spec_t::s));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v4, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v5, simd_fp_t::v25, simd_fp_t::v29, simd_fp_t::v5, neon_size_spec_t::s));
    // B: COLUMN 3
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v7, simd_fp_t::v25, simd_fp_t::v29, simd_fp_t::v7, neon_size_spec_t::s));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("case_5_k_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (5 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x12, 16, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x12, 16, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v4, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v5, gpr_t::x12, 16, neon_size_spec_t::s));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v7, gpr_t::x12, 16, neon_size_spec_t::s));
}

void mini_jit::kernels::internal::generateMLoopRest6(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (6 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x12, 16, neon_size_spec_t::d));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x12, 16, neon_size_spec_t::d));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v4, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v5, gpr_t::x12, 16, neon_size_spec_t::d));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v7, gpr_t::x12, 16, neon_size_spec_t::d));

    // case_6_k_loop:
    kernel.add_label("case_6_k_loop");
    // load column of A (6 values)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v24, gpr_t::x15, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v25, gpr_t::x15, 16, neon_size_spec_t::d));

    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s2));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v3, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s2));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v4, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v5, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s2));
    // B: COLUMN 3
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v7, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s2));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("case_6_k_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (6 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x12, 16, neon_size_spec_t::d));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x12, 16, neon_size_spec_t::d));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v4, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v5, gpr_t::x12, 16, neon_size_spec_t::d));
    // fourth column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v7, gpr_t::x12, 16, neon_size_spec_t::d));
}

void mini_jit::kernels::internal::generateMLoopRest7(mini_jit::Kernel &kernel)
{
// TODO: implement simd LDR with post index encoding

//     // LOAD MATRIX C (7 values)
//     mov x12, x9
//     // first column
//     mov x20, x12
//     ldr q0, [x20], #16
//     ld1 {v1.s}[0], [x20], #4
//     ld1 {v1.s}[1], [x20], #4
//     ld1 {v1.s}[2], [x20]
//     mov  v1.s[3], wzr
//     // second column
//     add x12, x12, x5
//     mov x20, x12
//     ldr q2, [x20], #16
//     ld1 {v3.s}[0], [x20], #4
//     ld1 {v3.s}[1], [x20], #4
//     ld1 {v3.s}[2], [x20]
//     mov  v3.s[3], wzr
//     // third column
//     add x12, x12, x5
//     mov x20, x12
//     ldr q4, [x20], #16
//     ld1 {v5.s}[0], [x20], #4
//     ld1 {v5.s}[1], [x20], #4
//     ld1 {v5.s}[2], [x20]
//     mov  v5.s[3], wzr
//     // fourth column
//     add x12, x12, x5
//     mov x20, x12
//     ldr q6, [x20], #16
//     ld1 {v7.s}[0], [x20], #4
//     ld1 {v7.s}[1], [x20], #4
//     ld1 {v7.s}[2], [x20]
//     mov  v7.s[3], wzr
//     // fifth column
//     add x12, x12, x5
//     mov x20, x12
//     ldr q8, [x20], #16
//     ld1 {v9.s}[0], [x20], #4
//     ld1 {v9.s}[1], [x20], #4
//     ld1 {v9.s}[2], [x20]
//     mov  v9.s[3], wzr
//     // sixth column
//     add x12, x12, x5
//     mov x20, x12
//     ldr q10, [x20], #16
//     ld1 {v11.s}[0], [x20], #4
//     ld1 {v11.s}[1], [x20], #4
//     ld1 {v11.s}[2], [x20]
//     mov  v11.s[3], wzr

// case_7_k_loop:
//     // load column of A (7 values)
//     mov x20, x15
//     ldr q24, [x20], #16
//     ld1 {v25.s}[0], [x20], #4
//     ld1 {v25.s}[1], [x20], #4
//     ld1 {v25.s}[2], [x20]
//     mov  v25.s[3], wzr

//     // B: COLUMN 0
//     ldr s29, [x16]
//     fmla v0.4s, v24.4s, v29.s[0]
//     fmla v1.4s, v25.4s, v29.s[0]
//     // B: COLUMN 1
//     add x16, x16, x4
//     ldr s29, [x16]
//     fmla v2.4s, v24.4s, v29.s[0]
//     fmla v3.4s, v25.4s, v29.s[0]
//     // B: COLUMN 2
//     add x16, x16, x4
//     ldr s29, [x16]
//     fmla v4.4s, v24.4s, v29.s[0]
//     fmla v5.4s, v25.4s, v29.s[0]
//     // B: COLUMN 3
//     add x16, x16, x4
//     ldr s29, [x16]
//     fmla v6.4s, v24.4s, v29.s[0]
//     fmla v7.4s, v25.4s, v29.s[0]
//     // B: COLUMN 4
//     add x16, x16, x4
//     ldr s29, [x16]
//     fmla v8.4s, v24.4s, v29.s[0]
//     fmla v9.4s, v25.4s, v29.s[0]
//     // B: COLUMN 5
//     add x16, x16, x4
//     ldr s29, [x16]
//     fmla v10.4s, v24.4s, v29.s[0]
//     fmla v11.4s, v25.4s, v29.s[0]

//     // move to next column of A
//     add x15, x15, x3
//     // move to next row of B
//     mov x16, x8
//     add x17, x17, #4
//     add x16, x16, x17

//     // decrement loop counter
//     sub x14, x14, #1
//     // check if loop counter is zero
//     cbnz x14, case_7_k_loop

//     // STORE MATRIX C (7 values)
//     mov x12, x9
//     // first column
//     mov x20, x12
//     str q0, [x20], #16
//     st1 {v1.s}[0], [x20], #4
//     st1 {v1.s}[1], [x20], #4
//     st1 {v1.s}[2], [x20]
//     mov  v1.s[3], wzr
//     // second column
//     add x12, x12, x5
//     mov x20, x12
//     str q2, [x20], #16
//     st1 {v3.s}[0], [x20], #4
//     st1 {v3.s}[1], [x20], #4
//     st1 {v3.s}[2], [x20]
//     mov  v3.s[3], wzr
//     // third column
//     add x12, x12, x5
//     mov x20, x12
//     str q4, [x20], #16
//     st1 {v5.s}[0], [x20], #4
//     st1 {v5.s}[1], [x20], #4
//     st1 {v5.s}[2], [x20]
//     mov  v5.s[3], wzr
//     // fourth column
//     add x12, x12, x5
//     mov x20, x12
//     str q6, [x20], #16
//     st1 {v7.s}[0], [x20], #4
//     st1 {v7.s}[1], [x20], #4
//     st1 {v7.s}[2], [x20]
//     mov  v7.s[3], wzr
//     // fifth column
//     add x12, x12, x5
//     mov x20, x12
//     str q8, [x20], #16
//     st1 {v9.s}[0], [x20], #4
//     st1 {v9.s}[1], [x20], #4
//     st1 {v9.s}[2], [x20]
//     mov  v9.s[3], wzr
//     // sixth column
//     add x12, x12, x5
//     mov x20, x12
//     str q10, [x20], #16
//     st1 {v11.s}[0], [x20], #4
//     st1 {v11.s}[1], [x20], #4
//     st1 {v11.s}[2], [x20]
//     mov  v11.s[3], wzr
}