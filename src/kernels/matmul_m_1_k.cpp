#include "../registers/gp_registers.h"
#include "../registers/simd_fp_registers.h"
#include "../instructions/all_instructions.h"
#include "matmul_m_1_k.h"

#include <iostream>
#include <cstring>

using gpr_t = mini_jit::registers::gpr_t;
using simd_fp_t = mini_jit::registers::simd_fp_t;
using arr_spec_t = mini_jit::registers::arr_spec_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace inst = mini_jit::instructions;
namespace base = inst::base;
namespace simd_fp = inst::simd_fp;

void mini_jit::kernels::matmul_m_1_k(mini_jit::Kernel &kernel,
                                     int m,
                                     int k)
{
    // Prepare the kernel
    int mLoopIterations = m / 8;
    int mLoopRemainder = m % 8;

    // PCS
    kernel.add_instr(base::stpPre(gpr_t::x29, gpr_t::x30, gpr_t::sp, -16));
    kernel.add_instr(base::movSP(gpr_t::x29, gpr_t::sp));

    // Save callee-saved registers
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
        mini_jit::kernels::internal::generateM8N1Loop(kernel, mLoopIterations, k);
    }

    if (mLoopRemainder > 0)
    {
        // set up k loop counter
        kernel.add_instr(base::mov(gpr_t::x14, k));
        // save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x7)); // A
        kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8)); // B
        kernel.add_instr(base::mov(gpr_t::x17, 0));         // row count B

        switch (mLoopRemainder)
        {
        case 1:
            mini_jit::kernels::internal::generateM1N1Loop(kernel);
            break;
        case 2:
            mini_jit::kernels::internal::generateM2N1Loop(kernel);
            break;
        case 3:
            mini_jit::kernels::internal::generateM3N1Loop(kernel);
            break;
        case 4:
            mini_jit::kernels::internal::generateM4N1Loop(kernel);
            break;
        case 5:
            mini_jit::kernels::internal::generateM5N1Loop(kernel);
            break;
        case 6:
            mini_jit::kernels::internal::generateM6N1Loop(kernel);
            break;
        case 7:
            mini_jit::kernels::internal::generateM7N1Loop(kernel);
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

    kernel.add_instr(base::ldpPost(gpr_t::x27, gpr_t::x28, gpr_t::sp, 16));
    kernel.add_instr(base::ldpPost(gpr_t::x25, gpr_t::x26, gpr_t::sp, 16));
    kernel.add_instr(base::ldpPost(gpr_t::x23, gpr_t::x24, gpr_t::sp, 16));
    kernel.add_instr(base::ldpPost(gpr_t::x21, gpr_t::x22, gpr_t::sp, 16));
    kernel.add_instr(base::ldpPost(gpr_t::x19, gpr_t::x20, gpr_t::sp, 16));

    // Restore stack pointer
    kernel.add_instr(base::ldpPost(gpr_t::x29, gpr_t::x30, gpr_t::sp, 16));

    kernel.add_instr(inst::ret());

    kernel.write("matmul_m_1_k.bin");
    kernel.set_kernel();
}

void mini_jit::kernels::internal::generateM8N1Loop(mini_jit::Kernel &kernel,
                                                   int mLoopIterations,
                                                   int k)
{
    // prepare the kernel
    kernel.add_instr(base::mov(gpr_t::x11, mLoopIterations));

    // START M_LOOP
    kernel.add_label("m8n1_loop");
    // Load Matrix C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));

    // Setup for Loop
    kernel.add_instr(base::mov(gpr_t::x14, k));         // K loop counter
    kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x7)); // Matrix A pointer
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8)); // Matrix B pointer
    kernel.add_instr(base::mov(gpr_t::x17, 0));         // Row index for Matrix B

    // START K_LOOP
    kernel.add_label("k_m8n1_loop");
    //  Load column of A (8 values)
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x15, 0, neon_size_spec_t::q));

    // Load Column of Matrix B
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));

    // 1st Multiplication
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));

    // Decrement K
    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // END K_LOOP
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m8n1_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // Store Matrix C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));

    // increase A and C pointers for next block
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, 8 * 4, 0));
    kernel.add_instr(base::add(gpr_t::x9, gpr_t::x9, 8 * 4, 0));

    // decrement M loop counter
    kernel.add_instr(base::sub(gpr_t::x11, gpr_t::x11, 1, 0));

    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m8n1_loop");
    kernel.add_instr(base::cbnz(gpr_t::x11, -l_mLoopInstrCount * 4));
    // END M_LOOP
}

void mini_jit::kernels::internal::generateM1N1Loop(mini_jit::Kernel &kernel)
{
    // Load Matrix C (1 value)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::s));

    // case_1_k_loop:
    kernel.add_label("k_m1n1_loop");
    // load column of A (1 value)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v24, gpr_t::x15, 0, neon_size_spec_t::s));

    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m1n1_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::s));
}

void mini_jit::kernels::internal::generateM2N1Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (2 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::d));

    // case_2_km1n1_loop:
    kernel.add_label("k_m2n1_loop");
    // load column of A (2 values)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v24, gpr_t::x15, 0, neon_size_spec_t::d));

    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s2));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m2n1_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::d));
}

void mini_jit::kernels::internal::generateM3N1Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (3 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x12));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x24, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x24, 0, neon_size_spec_t::s));

    // case_3_km1n1_loop:
    kernel.add_label("k_m3n1_loop");
    // load column of A (3 values)
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x15));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v24, gpr_t::x24, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v25, gpr_t::x24, 0, neon_size_spec_t::s));

    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, simd_fp_t::v1, neon_size_spec_t::s));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m3n1_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (3 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x12));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v0, gpr_t::x24, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x24, 0, neon_size_spec_t::s));
}

void mini_jit::kernels::internal::generateM4N1Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (4 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));

    // case_4_km1n1_loop:
    kernel.add_label("k_m4n1_loop");
    // load column of A (4 values)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v24, gpr_t::x15, 0, neon_size_spec_t::q));
    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m4n1_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));
}

void mini_jit::kernels::internal::generateM5N1Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (5 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x12, 16, neon_size_spec_t::s));

    // case_5_km1n1_loop:
    kernel.add_label("k_m5n1_loop");
    // load column of A (5 values)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v24, gpr_t::x15, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v25, gpr_t::x15, 16, neon_size_spec_t::s));

    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, simd_fp_t::v1, neon_size_spec_t::s));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m5n1_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (5 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x12, 16, neon_size_spec_t::s));
}

void mini_jit::kernels::internal::generateM6N1Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (6 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x12, 16, neon_size_spec_t::d));

    // case_6_km1n1_loop:
    kernel.add_label("k_m6n1_loop");
    // load column of A (6 values)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v24, gpr_t::x15, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v25, gpr_t::x15, 16, neon_size_spec_t::d));

    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s2));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m6n1_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (6 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x12, 16, neon_size_spec_t::d));
}

void mini_jit::kernels::internal::generateM7N1Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (7 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x20, 16, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v1, gpr_t::x20, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v2, gpr_t::x20, 0, neon_size_spec_t::s));

    // case_7_km1n1_loop:
    kernel.add_label("k_m7n1_loop");
    // load column of A (7 values)
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x15));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v24, gpr_t::x20, 16, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v25, gpr_t::x20, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v26, gpr_t::x20, 0, neon_size_spec_t::s));
    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v2, simd_fp_t::v26, simd_fp_t::v29, simd_fp_t::v2, neon_size_spec_t::s));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x8));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m7n1_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (7 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x9));
    // first column
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v0, gpr_t::x20, 16, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v1, gpr_t::x20, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v2, gpr_t::x20, 4, neon_size_spec_t::s));
}