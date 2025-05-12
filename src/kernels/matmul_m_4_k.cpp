

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
            mini_jit::kernels::internal::generateM1Loop(kernel);
        }
        else if (mLoopRemainder == 2)
        {
        }
        else if (mLoopRemainder == 3)
        {
        }
        else if (mLoopRemainder == 4)
        {
        }
        else if (mLoopRemainder == 5)
        {
        }
        else if (mLoopRemainder == 6)
        {
        }
        else if (mLoopRemainder == 7)
        {
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
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));

    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v2, simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::q));

    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v4, simd_fp_t::v5, gpr_t::x12, 0, neon_size_spec_t::q));

    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v6, simd_fp_t::v7, gpr_t::x12, 0, neon_size_spec_t::q));

    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v8, simd_fp_t::v9, gpr_t::x12, 0, neon_size_spec_t::q));

    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));

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
    kernel.add_instr(simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));

    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v2, simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::q));

    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v4, simd_fp_t::v5, gpr_t::x12, 0, neon_size_spec_t::q));

    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v6, simd_fp_t::v7, gpr_t::x12, 0, neon_size_spec_t::q));

    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v8, simd_fp_t::v9, gpr_t::x12, 0, neon_size_spec_t::q));

    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));

    // increase A and C pointers for next block
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, 8 * 4, 0));
    kernel.add_instr(base::add(gpr_t::x9, gpr_t::x9, 8 * 4, 0));

    // END M_LOOP
    // decrement M loop counter
    kernel.add_instr(base::sub(gpr_t::x11, gpr_t::x11, 1, 0));

    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_loop");
    kernel.add_instr(base::cbnz(gpr_t::x11, -l_mLoopInstrCount * 4));
}

void mini_jit::kernels::internal::generateM1Loop(mini_jit::Kernel &kernel)
{
    // Load Matrix C (1 value)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    //     // first column
    //     ldr s0, [x12]
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::s));
    //     // second column
    //     add x12, x12, x5
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    //     ldr s1, [x12]
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::s));
    //     // third column
    //     add x12, x12, x5
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    //     ldr s2, [x12]
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::s));
    //     // fourth column
    //     add x12, x12, x5
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    //     ldr s3, [x12]
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::s));

    // case_1_k_loop:
    kernel.add_label("case_1_k_loop");
    //     // load column of A (1 value)
    //     ldr s24, [x15]
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v24, gpr_t::x15, 0, neon_size_spec_t::s));

    //     // B: COLUMN 0
    //     ldr s29, [x16]
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    //     fmadd s0, s24, s29, s0
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    //     // B: COLUMN 1
    //     add x16, x16, x4
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    //     ldr s29, [x16]
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    //     fmadd s1, s24, s29, s1
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    //     // B: COLUMN 2
    //     add x16, x16, x4
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    //     ldr s29, [x16]
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    //     fmadd s2, s24, s29, s2
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    //     // B: COLUMN 3
    //     add x16, x16, x4
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    //     ldr s29, [x16]
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    //     fmadd s3, s24, s29, s3
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v3, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));

    //     // move to next column of A
    //     add x15, x15, x3
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    //     // move to next row of B
    //     mov x16, x8
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    //     add x17, x17, #4
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    //     add x16, x16, x17
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    //     // decrement loop counter
    //     sub x14, x14, #1
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    //     // check if loop counter is zero
    //     cbnz x14, case_1_k_loop
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("case_1_k_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    //     // STORE MATRIX C
    //     mov x12, x9
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    //     // first column
    //     str s0, [x12]
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::s));
    //     // second column
    //     add x12, x12, x5
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    //     str s1, [x12]
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::s));
    //     // third column
    //     add x12, x12, x5
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    //     str s2, [x12]
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::s));
    //     // fourth column
    //     add x12, x12, x5
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    //     str s3, [x12]
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::s));
}