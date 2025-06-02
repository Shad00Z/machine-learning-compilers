#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
#include "instructions/all_instructions.h"
#include "matmul_m_3_k.h"

#include <iostream>
#include <cstring>

using gpr_t = mini_jit::registers::gpr_t;
using simd_fp_t = mini_jit::registers::simd_fp_t;
using arr_spec_t = mini_jit::registers::arr_spec_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace inst = mini_jit::instructions;
namespace base = inst::base;
namespace simd_fp = inst::simd_fp;

void mini_jit::kernels::matmul::subkernels::matmul_m_3_k(mini_jit::Kernel &kernel,
                                                         int m,
                                                         int k)
{
    // Prepare the kernel
    int mLoopIterations = m / 16;
    int mLoopRemainder = m % 16;

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
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x0));  // A
    kernel.add_instr(base::mov(gpr_t::x9, gpr_t::x1));  // B
    kernel.add_instr(base::mov(gpr_t::x10, gpr_t::x2)); // C

    if (mLoopIterations > 0)
    {
        mini_jit::kernels::matmul::subkernels::internal::generateM16N3Loop(kernel, mLoopIterations, k);
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
            mini_jit::kernels::matmul::subkernels::internal::generateM1N3Loop(kernel);
            break;
        case 2:
            mini_jit::kernels::matmul::subkernels::internal::generateM2N3Loop(kernel);
            break;
        case 3:
            mini_jit::kernels::matmul::subkernels::internal::generateM3N3Loop(kernel);
            break;
        case 4:
            mini_jit::kernels::matmul::subkernels::internal::generateM4N3Loop(kernel);
            break;
        case 5:
            mini_jit::kernels::matmul::subkernels::internal::generateM5N3Loop(kernel);
            break;
        case 6:
            mini_jit::kernels::matmul::subkernels::internal::generateM6N3Loop(kernel);
            break;
        case 7:
            mini_jit::kernels::matmul::subkernels::internal::generateM7N3Loop(kernel);
            break;
        case 8:
            mini_jit::kernels::matmul::subkernels::internal::generateM8N3Loop(kernel);
            break;
        case 9:
            mini_jit::kernels::matmul::subkernels::internal::generateM9N3Loop(kernel);
            break;
        case 10:
            mini_jit::kernels::matmul::subkernels::internal::generateM10N3Loop(kernel);
            break;
        case 11:
            mini_jit::kernels::matmul::subkernels::internal::generateM11N3Loop(kernel);
            break;
        case 12:
            mini_jit::kernels::matmul::subkernels::internal::generateM12N3Loop(kernel);
            break;
        case 13:
            mini_jit::kernels::matmul::subkernels::internal::generateM13N3Loop(kernel);
            break;
        case 14:
            mini_jit::kernels::matmul::subkernels::internal::generateM14N3Loop(kernel);
            break;
        case 15:
            mini_jit::kernels::matmul::subkernels::internal::generateM15N3Loop(kernel);
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

    kernel.write("matmul_m_3_k.bin");
    kernel.set_kernel();
}

void mini_jit::kernels::matmul::subkernels::internal::generateM16N3Loop(mini_jit::Kernel &kernel,
                                                                        int mLoopIterations,
                                                                        int k)
{
    // prepare the kernel
    kernel.add_instr(base::mov(gpr_t::x11, mLoopIterations));

    // START M_LOOP
    kernel.add_label("m16n3_loop");
    // Load Matrix C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v2, simd_fp_t::v3, gpr_t::x12, 32, neon_size_spec_t::q));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v4, simd_fp_t::v5, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v6, simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::q));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v8, simd_fp_t::v9, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 32, neon_size_spec_t::q));

    // Setup for Loop
    kernel.add_instr(base::mov(gpr_t::x14, k));         // K loop counter
    kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x8)); // Matrix A pointer
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9)); // Matrix B pointer
    kernel.add_instr(base::mov(gpr_t::x17, 0));         // Row index for Matrix B

    // START K_LOOP
    kernel.add_label("k_m16n3_loop");
    //  Load column of A (8 values)
    kernel.add_instr(base::mov(gpr_t::x13, gpr_t::x15));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x13, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v26, simd_fp_t::v27, gpr_t::x13, 32, neon_size_spec_t::q));

    // Load Column of Matrix B
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));

    // 1st Multiplication
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v3, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s4));

    // Load Column of Matrix B
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));

    // 2nd Multiplication
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v4, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v5, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v7, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s4));

    // Load Column of Matrix B
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));

    // 3rd Multiplication
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v8, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v9, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v10, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v11, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s4));

    // Decrement K
    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // END K_LOOP
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m16n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // Store Matrix C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v2, simd_fp_t::v3, gpr_t::x12, 32, neon_size_spec_t::q));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v4, simd_fp_t::v5, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v6, simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::q));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v8, simd_fp_t::v9, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 32, neon_size_spec_t::q));

    // increase A and C pointers for next block
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, 16 * 4, 0));
    kernel.add_instr(base::add(gpr_t::x10, gpr_t::x10, 16 * 4, 0));

    // decrement M loop counter
    kernel.add_instr(base::sub(gpr_t::x11, gpr_t::x11, 1, 0));

    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m16n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x11, -l_mLoopInstrCount * 4));
    // END M_LOOP
}

void mini_jit::kernels::matmul::subkernels::internal::generateM1N3Loop(mini_jit::Kernel &kernel)
{
    // Load Matrix C (1 value)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::s));

    // case_1_k_loop:
    kernel.add_label("k_m1n3_loop");
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

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m1n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::s));
}

void mini_jit::kernels::matmul::subkernels::internal::generateM2N3Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (2 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::d));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::d));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::d));

    // case_2_km1n3_loop:
    kernel.add_label("k_m2n3_loop");
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

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m2n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::d));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::d));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::d));
}

void mini_jit::kernels::matmul::subkernels::internal::generateM3N3Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (3 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x12));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x24, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x24, 0, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x12));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v2, gpr_t::x24, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x24, 0, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x12));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v4, gpr_t::x24, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v5, gpr_t::x24, 0, neon_size_spec_t::s));

    // case_3_km1n3_loop:
    kernel.add_label("k_m3n3_loop");
    // load column of A (3 values)
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x15));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v24, gpr_t::x24, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v25, gpr_t::x24, 0, neon_size_spec_t::s));

    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, simd_fp_t::v1, neon_size_spec_t::s));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v3, simd_fp_t::v25, simd_fp_t::v29, simd_fp_t::v3, neon_size_spec_t::s));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v4, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v5, simd_fp_t::v25, simd_fp_t::v29, simd_fp_t::v5, neon_size_spec_t::s));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m3n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (3 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x12));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v0, gpr_t::x24, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x24, 0, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x12));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v2, gpr_t::x24, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x24, 0, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x12));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v4, gpr_t::x24, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v5, gpr_t::x24, 0, neon_size_spec_t::s));
}

void mini_jit::kernels::matmul::subkernels::internal::generateM4N3Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (4 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::q));

    // case_4_km1n3_loop:
    kernel.add_label("k_m4n3_loop");
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

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m4n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::q));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 0, neon_size_spec_t::q));
}

void mini_jit::kernels::matmul::subkernels::internal::generateM5N3Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (5 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
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

    // case_5_km1n3_loop:
    kernel.add_label("k_m5n3_loop");
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

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m5n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (5 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
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
}

void mini_jit::kernels::matmul::subkernels::internal::generateM6N3Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (6 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
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

    // case_6_km1n3_loop:
    kernel.add_label("k_m6n3_loop");
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

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    // check if loop counter is zero
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m6n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (6 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
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
}

void mini_jit::kernels::matmul::subkernels::internal::generateM7N3Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (7 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x20, 16, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v1, gpr_t::x20, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v2, gpr_t::x20, 0, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v3, gpr_t::x20, 16, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v4, gpr_t::x20, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v5, gpr_t::x20, 0, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v6, gpr_t::x20, 16, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v7, gpr_t::x20, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v8, gpr_t::x20, 0, neon_size_spec_t::s));

    // case_7_km1n3_loop:
    kernel.add_label("k_m7n3_loop");
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
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v3, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v4, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v5, simd_fp_t::v26, simd_fp_t::v29, simd_fp_t::v5, neon_size_spec_t::s));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v7, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v8, simd_fp_t::v26, simd_fp_t::v29, simd_fp_t::v8, neon_size_spec_t::s));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m7n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (7 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v0, gpr_t::x20, 16, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v1, gpr_t::x20, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v2, gpr_t::x20, 4, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v3, gpr_t::x20, 16, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v4, gpr_t::x20, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v5, gpr_t::x20, 4, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x12));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v6, gpr_t::x20, 16, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v7, gpr_t::x20, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v8, gpr_t::x20, 4, neon_size_spec_t::s));
}

void mini_jit::kernels::matmul::subkernels::internal::generateM8N3Loop(mini_jit::Kernel &kernel)
{
    // Load Matrix C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v2, simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::q));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v4, simd_fp_t::v5, gpr_t::x12, 0, neon_size_spec_t::q));

    // START K_LOOP
    kernel.add_label("k_m8n3_loop");
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

    // Decrement K
    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // END K_LOOP
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m8n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // Store Matrix C
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v2, simd_fp_t::v3, gpr_t::x12, 0, neon_size_spec_t::q));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v4, simd_fp_t::v5, gpr_t::x12, 0, neon_size_spec_t::q));
}

void mini_jit::kernels::matmul::subkernels::internal::generateM9N3Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (9 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::s));

    // case_9_k_loop:
    kernel.add_label("k_m9n3_loop");
    // load column of A (9 values)
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x15));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x24, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v26, gpr_t::x24, 32, neon_size_spec_t::s));
    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v2, simd_fp_t::v26, simd_fp_t::v29, simd_fp_t::v2, neon_size_spec_t::s));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v5, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v7, simd_fp_t::v26, simd_fp_t::v29, simd_fp_t::v7, neon_size_spec_t::s));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v10, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v11, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v12, simd_fp_t::v26, simd_fp_t::v29, simd_fp_t::v12, neon_size_spec_t::s));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m9n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (9 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::s));
}

void mini_jit::kernels::matmul::subkernels::internal::generateM10N3Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (10 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::d));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::d));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::d));

    // case_10_k_loop:
    kernel.add_label("k_m10n3_loop");
    // load column of A (10 values)
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x15));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x24, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v26, gpr_t::x24, 32, neon_size_spec_t::d));
    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s2));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v5, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v7, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s2));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v10, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v11, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v12, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s2));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m10n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (10 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::d));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::d));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::d));
}

void mini_jit::kernels::matmul::subkernels::internal::generateM11N3Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (11 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x12, 40, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v8, gpr_t::x12, 40, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v13, gpr_t::x12, 40, neon_size_spec_t::s));

    // case_11_k_loop:
    kernel.add_label("k_m11n3_loop");
    // load column of A (11 values)
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x15));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x24, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v26, gpr_t::x24, 32, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v27, gpr_t::x24, 40, neon_size_spec_t::s));
    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v3, simd_fp_t::v27, simd_fp_t::v29, simd_fp_t::v3, neon_size_spec_t::s));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v5, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v7, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v8, simd_fp_t::v27, simd_fp_t::v29, simd_fp_t::v8, neon_size_spec_t::s));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v10, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v11, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v12, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v13, simd_fp_t::v27, simd_fp_t::v29, simd_fp_t::v13, neon_size_spec_t::s));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m11n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (11 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x12, 40, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v8, gpr_t::x12, 40, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v13, gpr_t::x12, 40, neon_size_spec_t::s));
}

void mini_jit::kernels::matmul::subkernels::internal::generateM12N3Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (12 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::q));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::q));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::q));

    // case_12_k_loop:
    kernel.add_label("k_m12n3_loop");
    // load column of A (12 values)
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x15));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x24, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v26, gpr_t::x24, 32, neon_size_spec_t::q));
    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v5, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v7, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v10, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v11, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v12, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m12n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (12 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::q));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::q));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::q));
}

void mini_jit::kernels::matmul::subkernels::internal::generateM13N3Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (13 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x12, 48, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v8, gpr_t::x12, 48, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v13, gpr_t::x12, 48, neon_size_spec_t::s));

    // case_13_k_loop:
    kernel.add_label("k_m13n3_loop");
    // load column of A (13 values)
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x15));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x24, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v26, gpr_t::x24, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v27, gpr_t::x24, 48, neon_size_spec_t::s));
    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v3, simd_fp_t::v27, simd_fp_t::v29, simd_fp_t::v3, neon_size_spec_t::s));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v5, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v7, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v8, simd_fp_t::v27, simd_fp_t::v29, simd_fp_t::v8, neon_size_spec_t::s));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v10, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v11, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v12, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v13, simd_fp_t::v27, simd_fp_t::v29, simd_fp_t::v13, neon_size_spec_t::s));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m13n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (13 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x12, 48, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v8, gpr_t::x12, 48, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v13, gpr_t::x12, 48, neon_size_spec_t::s));
}

void mini_jit::kernels::matmul::subkernels::internal::generateM14N3Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (14 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x12, 48, neon_size_spec_t::d));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v8, gpr_t::x12, 48, neon_size_spec_t::d));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v13, gpr_t::x12, 48, neon_size_spec_t::d));

    // case_14_k_loop:
    kernel.add_label("k_m14n3_loop");
    // load column of A (14 values)
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x15));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x24, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v26, gpr_t::x24, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v27, gpr_t::x24, 48, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v28, gpr_t::x24, 56, neon_size_spec_t::s));
    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v3, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s2));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v5, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v7, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v8, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s2));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v10, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v11, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v12, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v13, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s2));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m14n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (14 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x12, 48, neon_size_spec_t::d));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v8, gpr_t::x12, 48, neon_size_spec_t::d));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v13, gpr_t::x12, 48, neon_size_spec_t::d));
}

void mini_jit::kernels::matmul::subkernels::internal::generateM15N3Loop(mini_jit::Kernel &kernel)
{
    // LOAD MATRIX C (15 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x12, 48, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v4, gpr_t::x12, 56, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v8, gpr_t::x12, 48, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v9, gpr_t::x12, 56, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v13, gpr_t::x12, 48, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v14, gpr_t::x12, 56, neon_size_spec_t::s));

    // case_15_k_loop:
    kernel.add_label("k_m15n3_loop");
    // load column of A (15 values)
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x15));
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x24, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v26, gpr_t::x24, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v27, gpr_t::x24, 48, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v28, gpr_t::x24, 56, neon_size_spec_t::s));
    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v3, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v4, simd_fp_t::v28, simd_fp_t::v29, simd_fp_t::v4, neon_size_spec_t::s));
    // B: COLUMN 1
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v5, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v7, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v8, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v9, simd_fp_t::v28, simd_fp_t::v29, simd_fp_t::v9, neon_size_spec_t::s));
    // B: COLUMN 2
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x4, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v10, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v11, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v12, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v13, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v14, simd_fp_t::v28, simd_fp_t::v29, simd_fp_t::v14, neon_size_spec_t::s));

    // move to next column of A
    kernel.add_instr(base::add(gpr_t::x15, gpr_t::x15, gpr_t::x3, 0, 0));
    // move to next row of B
    kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9));
    kernel.add_instr(base::add(gpr_t::x17, gpr_t::x17, 4, 0));
    kernel.add_instr(base::add(gpr_t::x16, gpr_t::x16, gpr_t::x17, 0, 0));

    // decrement loop counter
    kernel.add_instr(base::sub(gpr_t::x14, gpr_t::x14, 1, 0));
    int l_kLoopInstrCount = kernel.getInstrCountFromLabel("k_m15n3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x14, -l_kLoopInstrCount * 4));

    // STORE MATRIX C (15 values)
    kernel.add_instr(base::mov(gpr_t::x12, gpr_t::x10));
    // first column
    kernel.add_instr(simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x12, 48, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v4, gpr_t::x12, 56, neon_size_spec_t::s));
    // second column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v5, simd_fp_t::v6, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v7, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v8, gpr_t::x12, 48, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v9, gpr_t::x12, 56, neon_size_spec_t::s));
    // third column
    kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x5, 0, 0));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x12, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v12, gpr_t::x12, 32, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v13, gpr_t::x12, 48, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v14, gpr_t::x12, 56, neon_size_spec_t::s));
}
