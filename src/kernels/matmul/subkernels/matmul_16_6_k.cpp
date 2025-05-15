

#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
#include "instructions/all_instructions.h"
#include "matmul_16_6_k.h"

#include <iostream>
#include <cstring>

using gpr_t = mini_jit::registers::gpr_t;
using simd_fp_t = mini_jit::registers::simd_fp_t;
using arr_spec_t = mini_jit::registers::arr_spec_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace inst = mini_jit::instructions;
namespace base = inst::base;
namespace simd_fp = inst::simd_fp;

void mini_jit::kernels::matmul::subkernels::matmul_16_6_k( mini_jit::Kernel &kernel, 
                                                           int k )
{
    // PCS
    kernel.add_instr( base::stpPre(gpr_t::x29, gpr_t::x30, gpr_t::sp, -16) );
    kernel.add_instr( base::movSP(gpr_t::x29, gpr_t::sp) );

    // // Save callee-saved registers
    kernel.add_instr( base::stpPre(gpr_t::x19, gpr_t::x20, gpr_t::sp, -16) );
    kernel.add_instr( base::stpPre(gpr_t::x21, gpr_t::x22, gpr_t::sp, -16) );
    kernel.add_instr( base::stpPre(gpr_t::x23, gpr_t::x24, gpr_t::sp, -16) );
    kernel.add_instr( base::stpPre(gpr_t::x25, gpr_t::x26, gpr_t::sp, -16) );
    kernel.add_instr( base::stpPre(gpr_t::x27, gpr_t::x28, gpr_t::sp, -16) );

    kernel.add_instr( simd_fp::stpPre(simd_fp_t::v8, simd_fp_t::v9, gpr_t::sp, -16, neon_size_spec_t::d) );
    kernel.add_instr( simd_fp::stpPre(simd_fp_t::v10, simd_fp_t::v11, gpr_t::sp, -16, neon_size_spec_t::d) );
    kernel.add_instr( simd_fp::stpPre(simd_fp_t::v12, simd_fp_t::v13, gpr_t::sp, -16, neon_size_spec_t::d) );
    kernel.add_instr( simd_fp::stpPre(simd_fp_t::v14, simd_fp_t::v15, gpr_t::sp, -16, neon_size_spec_t::d) );

    // Strides
    kernel.add_instr( base::mov(gpr_t::x6, 4) );
    kernel.add_instr( base::mul(gpr_t::x3, gpr_t::x3, gpr_t::x6) );
    kernel.add_instr( base::mul(gpr_t::x4, gpr_t::x4, gpr_t::x6) );
    kernel.add_instr( base::mul(gpr_t::x5, gpr_t::x5, gpr_t::x6) );

    // Load Matrix C
    kernel.add_instr( base::mov(gpr_t::x8, gpr_t::x2) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x8, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v2, simd_fp_t::v3, gpr_t::x8, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::ldp(simd_fp_t::v4, simd_fp_t::v5, gpr_t::x8, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v6, simd_fp_t::v7, gpr_t::x8, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::ldp(simd_fp_t::v8, simd_fp_t::v9, gpr_t::x8, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x8, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::ldp(simd_fp_t::v12, simd_fp_t::v13, gpr_t::x8, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v14, simd_fp_t::v15, gpr_t::x8, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::ldp(simd_fp_t::v16, simd_fp_t::v17, gpr_t::x8, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v18, simd_fp_t::v19, gpr_t::x8, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::ldp(simd_fp_t::v20, simd_fp_t::v21, gpr_t::x8, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v22, simd_fp_t::v23, gpr_t::x8, 32, neon_size_spec_t::q) );

    // Setup for Loop
    kernel.add_instr( base::mov(gpr_t::x6, k) ); // K loop counter
    kernel.add_instr( base::mov(gpr_t::x7, gpr_t::x0) ); // Matrix A pointer
    kernel.add_instr( base::mov(gpr_t::x8, gpr_t::x1) ); // Matrix B pointer
    kernel.add_instr( base::mov(gpr_t::x9, 0) ); // Row index for Matrix B

    // START K_LOOP

    // Load Matrix A
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x7, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v26, simd_fp_t::v27, gpr_t::x7, 32, neon_size_spec_t::q) );

    // Load Column of Matrix B
    kernel.add_instr( simd_fp::ldr(simd_fp_t::v29, gpr_t::x8, 0, neon_size_spec_t::s) );

    // 1st Multiplication
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v2, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v3, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x4, 0, 0) );

    // Load Column of Matrix B
    kernel.add_instr( simd_fp::ldr(simd_fp_t::v29, gpr_t::x8, 0, neon_size_spec_t::s) );

    // 2nd Multiplication
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v4, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v5, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v7, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x4, 0, 0) );

    // Load Column of Matrix B
    kernel.add_instr( simd_fp::ldr(simd_fp_t::v29, gpr_t::x8, 0, neon_size_spec_t::s) );

    // 3rd Multiplication
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v8, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v9, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v10, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v11, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x4, 0, 0) );

    // Load Column of Matrix B
    kernel.add_instr( simd_fp::ldr(simd_fp_t::v29, gpr_t::x8, 0, neon_size_spec_t::s) );

    // 4th Multiplication
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v12, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v13, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v14, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v15, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x4, 0, 0) );

    // Load Column of Matrix B
    kernel.add_instr( simd_fp::ldr(simd_fp_t::v29, gpr_t::x8, 0, neon_size_spec_t::s) );

    // 5th Multiplication
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v16, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v17, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v18, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v19, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x4, 0, 0) );

    // Load Column of Matrix B
    kernel.add_instr( simd_fp::ldr(simd_fp_t::v29, gpr_t::x8, 0, neon_size_spec_t::s) );

    // 6th Multiplication
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v20, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v21, simd_fp_t::v25, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v22, simd_fp_t::v26, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v23, simd_fp_t::v27, simd_fp_t::v29, arr_spec_t::s4) );

    // Decrement K
    // move to next column of A
    kernel.add_instr( base::add(gpr_t::x7, gpr_t::x7, gpr_t::x3, 0, 0) ); 
    // move to next row of B
    kernel.add_instr( base::mov(gpr_t::x8, gpr_t::x1) );
    kernel.add_instr( base::add(gpr_t::x9, gpr_t::x9, 4, 0) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x9, 0, 0) );

    kernel.add_instr( base::sub(gpr_t::x6, gpr_t::x6, 1, 0) );
    kernel.add_instr( base::cbnz(gpr_t::x6, -168) );

    // Store Matrix C
    kernel.add_instr( base::mov(gpr_t::x8, gpr_t::x2) );
    kernel.add_instr( simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x8, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::stp(simd_fp_t::v2, simd_fp_t::v3, gpr_t::x8, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::stp(simd_fp_t::v4, simd_fp_t::v5, gpr_t::x8, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::stp(simd_fp_t::v6, simd_fp_t::v7, gpr_t::x8, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::stp(simd_fp_t::v8, simd_fp_t::v9, gpr_t::x8, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::stp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x8, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::stp(simd_fp_t::v12, simd_fp_t::v13, gpr_t::x8, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::stp(simd_fp_t::v14, simd_fp_t::v15, gpr_t::x8, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::stp(simd_fp_t::v16, simd_fp_t::v17, gpr_t::x8, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::stp(simd_fp_t::v18, simd_fp_t::v19, gpr_t::x8, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::stp(simd_fp_t::v20, simd_fp_t::v21, gpr_t::x8, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::stp(simd_fp_t::v22, simd_fp_t::v23, gpr_t::x8, 32, neon_size_spec_t::q) );

    // Restore callee-saved registers
    kernel.add_instr( simd_fp::ldpPost(simd_fp_t::v14, simd_fp_t::v15, gpr_t::sp, 16, neon_size_spec_t::d) );
    kernel.add_instr( simd_fp::ldpPost(simd_fp_t::v12, simd_fp_t::v13, gpr_t::sp, 16, neon_size_spec_t::d) );
    kernel.add_instr( simd_fp::ldpPost(simd_fp_t::v10, simd_fp_t::v11, gpr_t::sp, 16, neon_size_spec_t::d) );
    kernel.add_instr( simd_fp::ldpPost(simd_fp_t::v8, simd_fp_t::v9, gpr_t::sp, 16, neon_size_spec_t::d) );

    kernel.add_instr( base::ldpPost(gpr_t::x27, gpr_t::x28, gpr_t::sp, 16) );
    kernel.add_instr( base::ldpPost(gpr_t::x25, gpr_t::x26, gpr_t::sp, 16) );
    kernel.add_instr( base::ldpPost(gpr_t::x23, gpr_t::x24, gpr_t::sp, 16) );
    kernel.add_instr( base::ldpPost(gpr_t::x21, gpr_t::x22, gpr_t::sp, 16) );
    kernel.add_instr( base::ldpPost(gpr_t::x19, gpr_t::x20, gpr_t::sp, 16) );

    // Restore stack pointer
    kernel.add_instr( base::ldpPost(gpr_t::x29, gpr_t::x30, gpr_t::sp, 16) );
    
    kernel.add_instr( inst::ret() );

    kernel.write( "matmul_16_6_k.bin" );
    kernel.set_kernel();
}