#include "sigmoid_primitive.h"
#include "Kernel.h"

#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
#include "instructions/all_instructions.h"
#include <iostream>

namespace inst = mini_jit::instructions;
namespace base = inst::base;
namespace simd_fp = inst::simd_fp;

using enum gpr_t;
using enum simd_fp_t;
using enum neon_size_spec_t;
using enum arr_spec_t;

using base::stpPre;
using base::movSP;
using base::lsl;
using base::mov;
using base::movz;
using base::movk;
using base::add;
using base::sub;
using base::str;
using base::cbnz;
using base::ldpPost;
using simd_fp::mov;
using simd_fp::ldp;
using simd_fp::ldpPost;
using simd_fp::stp;
using simd_fp::stpPre;
using simd_fp::ldr;
using simd_fp::str;
using simd_fp::faddVec;
using simd_fp::faddScalar;
using simd_fp::fsubVec;
using simd_fp::fsubScalar;
using simd_fp::fmulVec;
using simd_fp::fmulScalar;
using simd_fp::fdivVec;
using simd_fp::fdivScalar;
using simd_fp::fmovVec;
using simd_fp::fmovScalar;
using simd_fp::fminVec;
using simd_fp::fminScalar;

void mini_jit::kernels::unary::sigmoid(mini_jit::Kernel &kernel,
                                       u_int32_t m,
                                       u_int32_t n)
{
    // Inputs:
    // x0: pointer to A (input)
    // x1: pointer to B (output)
    // x2: leading dimension of A
    // x3: leading dimension of B

    // Prepare the kernel
    int mLoopIterations = m / 16;
    int mLoopRemainder = m % 16;

    kernel.add_instr({
        // PCS
        stpPre(x29, x30, sp, -16),
        movSP(x29, sp),

        // Allocate additional stack space for constants
        sub(sp, sp, 64, 0),

        // Save callee-saved registers
        stpPre(v8, v9, sp, -16, d),
        stpPre(v10, v11, sp, -16, d),
        
        // Compute stride
        lsl(x2, x2, 2),
        lsl(x3, x3, 2),

        // Save base matrix pointers
        mov(x4, x0), // A (input)
        mov(x5, x1), // B (output)

        // Set n loop counter
        mov(x6, n),

        // Load fixed values
        // Load 0.5f (0x3F000000)
        movz(w11, 0x0000, 0),     // w11 = 0x00000000
        movk(w11, 0x3F00, 16),    // w11 = 0x3F000000 (0.5f)
        str(w11, sp, 0),
        str(w11, sp, 4),
        str(w11, sp, 8),
        str(w11, sp, 12),
        ldr(v31, sp, 0, q),       // v31 = {0.5, 0.5, 0.5, 0.5}

        // Load 0.25f (0x3E800000)
        movz(w12, 0x0000, 0),     // w12 = 0x00000000
        movk(w12, 0x3E80, 16),    // w12 = 0x3E800000 (0.25f)
        str(w12, sp, 16),
        str(w12, sp, 20),
        str(w12, sp, 24),
        str(w12, sp, 28),
        ldr(v30, sp, 16, q),      // v30 = {0.25, 0.25, 0.25, 0.25}

        movz(w13, 0xAAAB, 0),     // w13 = 0x0000AAAB
        movk(w13, 0xBCAA, 16),    // w13 = 0xBCAAAAAB (-1/48)
        str(w13, sp, 32),
        str(w13, sp, 36),
        str(w13, sp, 40),
        str(w13, sp, 44),
        ldr(v29, sp, 32, q),      // v29 = {-0.020833333, -0.020833333, -0.020833333, -0.020833333}

        // Load 1/480 ≈ 0.002083 (0x3B088889)
        movz(w14, 0x8889, 0),     // w14 = 0x00008889
        movk(w14, 0x3B08, 16),    // w14 = 0x3B088889 (1/480)
        str(w14, sp, 48),
        str(w14, sp, 52),
        str(w14, sp, 56),
        str(w14, sp, 60),
        ldr(v28, sp, 48, q),      // v28 = {0.002083333, 0.002083333, 0.002083333, 0.002083333}
    });

    // Start n loop (1 column)
    kernel.add_label("n_loop");

    // Set m loop counter
    kernel.add_instr({
        mov(x7, mLoopIterations),

        // working pointers for rows
        mov(x8, x4), // A
        mov(x9, x5)  // B
    });

    if (mLoopIterations > 0)
    {
        kernel.add_label("m_16_loop");
        kernel.add_instr({
            // Load 16 elements from A
            ldp(v0, v1, x8, 0, q),
            ldp(v2, v3, x8, 32, q),

            // Compute x^2 for all 4 groups
            fmulVec(v4, v0, v0, s4),
            fmulVec(v5, v1, v1, s4),
            fmulVec(v6, v2, v2, s4),
            fmulVec(v7, v3, v3, s4),

            // 0.5 + 0.25*x - 0.020833*x³ + 0.002083*x⁵
            // x^3 = x^2 * x
            fmulVec(v12, v4, v0, s4),
            fmulVec(v13, v5, v1, s4),
            fmulVec(v14, v6, v2, s4),
            fmulVec(v15, v7, v3, s4),

            // x^5 = x^3 * x^2  
            fmulVec(v16, v12, v4, s4),
            fmulVec(v17, v13, v5, s4),
            fmulVec(v18, v14, v6, s4),
            fmulVec(v19, v15, v7, s4),

            // 0.25 * x (reusing v4-v7)
            fmulVec(v4, v0, v30, s4),
            fmulVec(v5, v1, v30, s4),
            fmulVec(v6, v2, v30, s4),
            fmulVec(v7, v3, v30, s4),

            // -0.020833 * x^3
            fmulVec(v12, v12, v29, s4),
            fmulVec(v13, v13, v29, s4),
            fmulVec(v14, v14, v29, s4),
            fmulVec(v15, v15, v29, s4),

            // +0.002083 * x^5
            fmulVec(v16, v16, v28, s4),
            fmulVec(v17, v17, v28, s4),
            fmulVec(v18, v18, v28, s4),
            fmulVec(v19, v19, v28, s4),

            // 0.5 + 0.25*x
            faddVec(v4, v31, v4, s4),
            faddVec(v5, v31, v5, s4),
            faddVec(v6, v31, v6, s4),
            faddVec(v7, v31, v7, s4),

            // + (-0.020833*x^3)
            faddVec(v4, v4, v12, s4),
            faddVec(v5, v5, v13, s4),
            faddVec(v6, v6, v14, s4),
            faddVec(v7, v7, v15, s4),

            // + (+0.002083*x^5)
            faddVec(v0, v4, v16, s4),
            faddVec(v1, v5, v17, s4),
            faddVec(v2, v6, v18, s4),
            faddVec(v3, v7, v19, s4),

            // Store 16 elements to B
            stp(v0, v1, x9, 0, q),
            stp(v2, v3, x9, 32, q),

            // Jump by 16 rows
            add(x8, x8, 16*4, 0),
            add(x9, x9, 16*4, 0),

            // Decrement m loop counter
            sub(x7, x7, 1, 0),
        });

        // Check if loop counter is zero
        kernel.add_instr(cbnz(x7, -kernel.getInstrCountFromLabel("m_16_loop") * 4));
    }

    if (mLoopRemainder > 0)
    {
        switch (mLoopRemainder)
        {
        case 1:
            kernel.add_instr({
                // 1 element
                ldr(v0, x8, 0, s),
                
                // x^2
                fmulScalar(v1, v0, v0, s),
                // x^3
                fmulScalar(v2, v1, v0, s),
                // x^5 = x^3 * x^2
                fmulScalar(v3, v2, v1, s),
                
                // 0.25 * x
                fmulScalar(v4, v0, v30, s),
                // -0.020833 * x^3
                fmulScalar(v5, v2, v29, s),
                // +0.002083 * x^5
                fmulScalar(v6, v3, v28, s),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddScalar(v7, v31, v4, s),
                faddScalar(v7, v7, v5, s),
                faddScalar(v7, v7, v6, s),
                
                str(v7, x9, 0, s)
            });
            break;
        case 2:
            kernel.add_instr({
                // 2 elements
                ldr(v0, x8, 0, d),
                
                // x^2
                fmulVec(v1, v0, v0, s2),
                // x^3  
                fmulVec(v2, v1, v0, s2),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s2),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s2),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s2),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s2),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s2),
                faddVec(v7, v7, v5, s2),
                faddVec(v7, v7, v6, s2),
                
                str(v7, x9, 0, d)
            });
            break;
        case 3:
            kernel.add_instr({
                // 2 elements
                ldr(v0, x8, 0, d),
                
                // x^2
                fmulVec(v1, v0, v0, s2),
                // x^3  
                fmulVec(v2, v1, v0, s2),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s2),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s2),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s2),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s2),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s2),
                faddVec(v7, v7, v5, s2),
                faddVec(v7, v7, v6, s2),
                
                str(v7, x9, 0, d),
                
                // 1 element
                ldr(v0, x8, 2*4, s),
                
                // x^2
                fmulScalar(v1, v0, v0, s),
                // x^3
                fmulScalar(v2, v1, v0, s),
                // x^5 = x^3 * x^2
                fmulScalar(v3, v2, v1, s),
                
                // 0.25 * x
                fmulScalar(v4, v0, v30, s),
                // -0.020833 * x^3
                fmulScalar(v5, v2, v29, s),
                // +0.002083 * x^5
                fmulScalar(v6, v3, v28, s),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddScalar(v7, v31, v4, s),
                faddScalar(v7, v7, v5, s),
                faddScalar(v7, v7, v6, s),
                
                str(v7, x9, 2*4, s)
            });
            break;
        case 4:
            kernel.add_instr({
                // 4 elements - implement 5th order sigmoid polynomial
                ldr(v0, x8, 0, q),
                
                // x^2
                fmulVec(v1, v0, v0, s4),
                // x^3  
                fmulVec(v2, v1, v0, s4),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s4),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s4),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s4),
                faddVec(v7, v7, v5, s4),
                faddVec(v7, v7, v6, s4),
                
                str(v7, x9, 0, q)
            });
            break;
        case 5:
            kernel.add_instr({
                // 4 elements
                ldr(v0, x8, 0, q),
                
                // x^2
                fmulVec(v1, v0, v0, s4),
                // x^3  
                fmulVec(v2, v1, v0, s4),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s4),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s4),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s4),
                faddVec(v7, v7, v5, s4),
                faddVec(v7, v7, v6, s4),
                
                str(v7, x9, 0, q),
                
                // 1 element - implement 5th order sigmoid polynomial
                ldr(v0, x8, 4*4, s),
                
                // x^2
                fmulScalar(v1, v0, v0, s),
                // x^3
                fmulScalar(v2, v1, v0, s),
                // x^5 = x^3 * x^2
                fmulScalar(v3, v2, v1, s),
                
                // 0.25 * x
                fmulScalar(v4, v0, v30, s),
                // -0.020833 * x^3
                fmulScalar(v5, v2, v29, s),
                // +0.002083 * x^5
                fmulScalar(v6, v3, v28, s),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddScalar(v7, v31, v4, s),
                faddScalar(v7, v7, v5, s),
                faddScalar(v7, v7, v6, s),
                
                str(v7, x9, 4*4, s)
            });
            break;
        case 6:
            kernel.add_instr({
                // 4 elements
                ldr(v0, x8, 0, q),
                
                // x^2
                fmulVec(v1, v0, v0, s4),
                // x^3  
                fmulVec(v2, v1, v0, s4),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s4),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s4),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s4),
                faddVec(v7, v7, v5, s4),
                faddVec(v7, v7, v6, s4),
                
                str(v7, x9, 0, q),
                
                // 2 elements
                ldr(v0, x8, 4*4, d),
                
                // x^2
                fmulVec(v1, v0, v0, s2),
                // x^3  
                fmulVec(v2, v1, v0, s2),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s2),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s2),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s2),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s2),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s2),
                faddVec(v7, v7, v5, s2),
                faddVec(v7, v7, v6, s2),
                
                str(v7, x9, 4*4, d)
            });
            break;
        case 7:
            kernel.add_instr({
                // 4 elements
                ldr(v0, x8, 0, q),
                
                // x^2
                fmulVec(v1, v0, v0, s4),
                // x^3  
                fmulVec(v2, v1, v0, s4),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s4),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s4),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s4),
                faddVec(v7, v7, v5, s4),
                faddVec(v7, v7, v6, s4),
                
                str(v7, x9, 0, q),
                
                // 2 elements
                ldr(v0, x8, 4*4, d),
                
                // x^2
                fmulVec(v1, v0, v0, s2),
                // x^3  
                fmulVec(v2, v1, v0, s2),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s2),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s2),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s2),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s2),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s2),
                faddVec(v7, v7, v5, s2),
                faddVec(v7, v7, v6, s2),
                
                str(v7, x9, 4*4, d),
                
                // 1 element
                ldr(v0, x8, 6*4, s),
                
                // x^2
                fmulScalar(v1, v0, v0, s),
                // x^3
                fmulScalar(v2, v1, v0, s),
                // x^5 = x^3 * x^2
                fmulScalar(v3, v2, v1, s),
                
                // 0.25 * x
                fmulScalar(v4, v0, v30, s),
                // -0.020833 * x^3
                fmulScalar(v5, v2, v29, s),
                // +0.002083 * x^5
                fmulScalar(v6, v3, v28, s),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddScalar(v7, v31, v4, s),
                faddScalar(v7, v7, v5, s),
                faddScalar(v7, v7, v6, s),
                
                str(v7, x9, 6*4, s)
            });
            break;
        case 8:
            kernel.add_instr({
                // 8 elements
                ldr(v0, x8, 0, q),
                ldr(v1, x8, 16, q),
                
                // First 4 elements: x^2
                fmulVec(v2, v0, v0, s4),
                // First 4 elements: x^3  
                fmulVec(v3, v2, v0, s4),
                // First 4 elements: x^5 = x^3 * x^2
                fmulVec(v4, v3, v2, s4),
                
                // Second 4 elements: x^2
                fmulVec(v5, v1, v1, s4),
                // Second 4 elements: x^3  
                fmulVec(v6, v5, v1, s4),
                // Second 4 elements: x^5 = x^3 * x^2
                fmulVec(v7, v6, v5, s4),
                
                // First 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v8, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v9, v3, v29, s4),
                // +0.002083 * x^5
                fmulVec(v10, v4, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v11, v31, v8, s4),
                faddVec(v11, v11, v9, s4),
                faddVec(v11, v11, v10, s4),
                
                // Second 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v12, v1, v30, s4),
                // -0.020833 * x^3
                fmulVec(v13, v6, v29, s4),
                // +0.002083 * x^5
                fmulVec(v14, v7, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v15, v31, v12, s4),
                faddVec(v15, v15, v13, s4),
                faddVec(v15, v15, v14, s4),
                
                str(v11, x9, 0, q),
                str(v15, x9, 16, q)
            });
            break;
        case 9:
            kernel.add_instr({
                // 8 elements
                ldr(v0, x8, 0, q),
                ldr(v1, x8, 16, q),
                
                // First 4 elements: x^2
                fmulVec(v2, v0, v0, s4),
                // First 4 elements: x^3  
                fmulVec(v3, v2, v0, s4),
                // First 4 elements: x^5 = x^3 * x^2
                fmulVec(v4, v3, v2, s4),
                
                // Second 4 elements: x^2
                fmulVec(v5, v1, v1, s4),
                // Second 4 elements: x^3  
                fmulVec(v6, v5, v1, s4),
                // Second 4 elements: x^5 = x^3 * x^2
                fmulVec(v7, v6, v5, s4),
                
                // First 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v8, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v9, v3, v29, s4),
                // +0.002083 * x^5
                fmulVec(v10, v4, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v11, v31, v8, s4),
                faddVec(v11, v11, v9, s4),
                faddVec(v11, v11, v10, s4),
                
                // Second 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v12, v1, v30, s4),
                // -0.020833 * x^3
                fmulVec(v13, v6, v29, s4),
                // +0.002083 * x^5
                fmulVec(v14, v7, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v15, v31, v12, s4),
                faddVec(v15, v15, v13, s4),
                faddVec(v15, v15, v14, s4),
                
                str(v11, x9, 0, q),
                str(v15, x9, 16, q),
                
                // 1 element
                ldr(v0, x8, 8*4, s),
                
                // x^2
                fmulScalar(v1, v0, v0, s),
                // x^3
                fmulScalar(v2, v1, v0, s),
                // x^5 = x^3 * x^2
                fmulScalar(v3, v2, v1, s),
                
                // 0.25 * x
                fmulScalar(v4, v0, v30, s),
                // -0.020833 * x^3
                fmulScalar(v5, v2, v29, s),
                // +0.002083 * x^5
                fmulScalar(v6, v3, v28, s),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddScalar(v7, v31, v4, s),
                faddScalar(v7, v7, v5, s),
                faddScalar(v7, v7, v6, s),
                
                str(v7, x9, 8*4, s)
            });
            break;
        case 10:
            kernel.add_instr({
                // 8 elements
                ldr(v0, x8, 0, q),
                ldr(v1, x8, 16, q),
                
                // First 4 elements: x^2
                fmulVec(v2, v0, v0, s4),
                // First 4 elements: x^3  
                fmulVec(v3, v2, v0, s4),
                // First 4 elements: x^5 = x^3 * x^2
                fmulVec(v4, v3, v2, s4),
                
                // Second 4 elements: x^2
                fmulVec(v5, v1, v1, s4),
                // Second 4 elements: x^3  
                fmulVec(v6, v5, v1, s4),
                // Second 4 elements: x^5 = x^3 * x^2
                fmulVec(v7, v6, v5, s4),
                
                // First 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v8, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v9, v3, v29, s4),
                // +0.002083 * x^5
                fmulVec(v10, v4, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v11, v31, v8, s4),
                faddVec(v11, v11, v9, s4),
                faddVec(v11, v11, v10, s4),
                
                // Second 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v12, v1, v30, s4),
                // -0.020833 * x^3
                fmulVec(v13, v6, v29, s4),
                // +0.002083 * x^5
                fmulVec(v14, v7, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v15, v31, v12, s4),
                faddVec(v15, v15, v13, s4),
                faddVec(v15, v15, v14, s4),
                
                str(v11, x9, 0, q),
                str(v15, x9, 16, q),

                // 2 elements
                ldr(v0, x8, 8*4, d),
                
                // x^2
                fmulVec(v1, v0, v0, s2),
                // x^3  
                fmulVec(v2, v1, v0, s2),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s2),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s2),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s2),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s2),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s2),
                faddVec(v7, v7, v5, s2),
                faddVec(v7, v7, v6, s2),
                
                str(v7, x9, 8*4, d)
            });
            break;
        case 11:
            kernel.add_instr({
                // 8 elements
                ldr(v0, x8, 0, q),
                ldr(v1, x8, 16, q),
                
                // First 4 elements: x^2
                fmulVec(v2, v0, v0, s4),
                // First 4 elements: x^3  
                fmulVec(v3, v2, v0, s4),
                // First 4 elements: x^5 = x^3 * x^2
                fmulVec(v4, v3, v2, s4),
                
                // Second 4 elements: x^2
                fmulVec(v5, v1, v1, s4),
                // Second 4 elements: x^3  
                fmulVec(v6, v5, v1, s4),
                // Second 4 elements: x^5 = x^3 * x^2
                fmulVec(v7, v6, v5, s4),
                
                // First 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v8, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v9, v3, v29, s4),
                // +0.002083 * x^5
                fmulVec(v10, v4, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v11, v31, v8, s4),
                faddVec(v11, v11, v9, s4),
                faddVec(v11, v11, v10, s4),
                
                // Second 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v12, v1, v30, s4),
                // -0.020833 * x^3
                fmulVec(v13, v6, v29, s4),
                // +0.002083 * x^5
                fmulVec(v14, v7, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v15, v31, v12, s4),
                faddVec(v15, v15, v13, s4),
                faddVec(v15, v15, v14, s4),
                
                str(v11, x9, 0, q),
                str(v15, x9, 16, q),

                // 2 elements
                ldr(v0, x8, 8*4, d),
                
                // x^2
                fmulVec(v1, v0, v0, s2),
                // x^3  
                fmulVec(v2, v1, v0, s2),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s2),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s2),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s2),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s2),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s2),
                faddVec(v7, v7, v5, s2),
                faddVec(v7, v7, v6, s2),
                
                str(v7, x9, 8*4, d),
                
                // 1 element
                ldr(v0, x8, 10*4, s),
                
                // x^2
                fmulScalar(v1, v0, v0, s),
                // x^3
                fmulScalar(v2, v1, v0, s),
                // x^5 = x^3 * x^2
                fmulScalar(v3, v2, v1, s),
                
                // 0.25 * x
                fmulScalar(v4, v0, v30, s),
                // -0.020833 * x^3
                fmulScalar(v5, v2, v29, s),
                // +0.002083 * x^5
                fmulScalar(v6, v3, v28, s),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddScalar(v7, v31, v4, s),
                faddScalar(v7, v7, v5, s),
                faddScalar(v7, v7, v6, s),
                
                str(v7, x9, 10*4, s)
            });
            break;
        case 12:
            kernel.add_instr({
                // 8 elements
                ldr(v0, x8, 0, q),
                ldr(v1, x8, 16, q),
                
                // First 4 elements: x^2
                fmulVec(v2, v0, v0, s4),
                // First 4 elements: x^3  
                fmulVec(v3, v2, v0, s4),
                // First 4 elements: x^5 = x^3 * x^2
                fmulVec(v4, v3, v2, s4),
                
                // Second 4 elements: x^2
                fmulVec(v5, v1, v1, s4),
                // Second 4 elements: x^3  
                fmulVec(v6, v5, v1, s4),
                // Second 4 elements: x^5 = x^3 * x^2
                fmulVec(v7, v6, v5, s4),
                
                // First 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v8, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v9, v3, v29, s4),
                // +0.002083 * x^5
                fmulVec(v10, v4, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v11, v31, v8, s4),
                faddVec(v11, v11, v9, s4),
                faddVec(v11, v11, v10, s4),
                
                // Second 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v12, v1, v30, s4),
                // -0.020833 * x^3
                fmulVec(v13, v6, v29, s4),
                // +0.002083 * x^5
                fmulVec(v14, v7, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v15, v31, v12, s4),
                faddVec(v15, v15, v13, s4),
                faddVec(v15, v15, v14, s4),
                
                str(v11, x9, 0, q),
                str(v15, x9, 16, q),

                // 4 elements
                ldr(v0, x8, 8*4, q),
                
                // x^2
                fmulVec(v1, v0, v0, s4),
                // x^3  
                fmulVec(v2, v1, v0, s4),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s4),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s4),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s4),
                faddVec(v7, v7, v5, s4),
                faddVec(v7, v7, v6, s4),
                
                str(v7, x9, 8*4, q)
            });
            break;
        case 13:
            kernel.add_instr({
                // 8 elements
                ldr(v0, x8, 0, q),
                ldr(v1, x8, 16, q),
                
                // First 4 elements: x^2
                fmulVec(v2, v0, v0, s4),
                // First 4 elements: x^3  
                fmulVec(v3, v2, v0, s4),
                // First 4 elements: x^5 = x^3 * x^2
                fmulVec(v4, v3, v2, s4),
                
                // Second 4 elements: x^2
                fmulVec(v5, v1, v1, s4),
                // Second 4 elements: x^3  
                fmulVec(v6, v5, v1, s4),
                // Second 4 elements: x^5 = x^3 * x^2
                fmulVec(v7, v6, v5, s4),
                
                // First 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v8, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v9, v3, v29, s4),
                // +0.002083 * x^5
                fmulVec(v10, v4, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v11, v31, v8, s4),   // 0.5 + 0.25*x
                faddVec(v11, v11, v9, s4),   // + (-0.020833*x^3)
                faddVec(v11, v11, v10, s4),  // + (0.002083*x^5)
                
                // Second 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v12, v1, v30, s4),
                // -0.020833 * x^3
                fmulVec(v13, v6, v29, s4),
                // +0.002083 * x^5
                fmulVec(v14, v7, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v15, v31, v12, s4),
                faddVec(v15, v15, v13, s4),
                faddVec(v15, v15, v14, s4),
                
                str(v11, x9, 0, q),
                str(v15, x9, 16, q),

                // 4 elements
                ldr(v0, x8, 8*4, q),
                
                // x^2
                fmulVec(v1, v0, v0, s4),
                // x^3  
                fmulVec(v2, v1, v0, s4),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s4),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s4),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s4),
                faddVec(v7, v7, v5, s4),
                faddVec(v7, v7, v6, s4),
                
                str(v7, x9, 8*4, q),
                
                // 1 element
                ldr(v0, x8, 12*4, s),
                
                // x^2
                fmulScalar(v1, v0, v0, s),
                // x^3
                fmulScalar(v2, v1, v0, s),
                // x^5 = x^3 * x^2
                fmulScalar(v3, v2, v1, s),
                
                // 0.25 * x
                fmulScalar(v4, v0, v30, s),
                // -0.020833 * x^3
                fmulScalar(v5, v2, v29, s),
                // +0.002083 * x^5
                fmulScalar(v6, v3, v28, s),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddScalar(v7, v31, v4, s),
                faddScalar(v7, v7, v5, s),
                faddScalar(v7, v7, v6, s),
                
                str(v7, x9, 12*4, s)
            });
            break;
        case 14:
            kernel.add_instr({
                // 8 elements
                ldr(v0, x8, 0, q),
                ldr(v1, x8, 16, q),
                
                // First 4 elements: x^2
                fmulVec(v2, v0, v0, s4),
                // First 4 elements: x^3  
                fmulVec(v3, v2, v0, s4),
                // First 4 elements: x^5 = x^3 * x^2
                fmulVec(v4, v3, v2, s4),
                
                // Second 4 elements: x^2
                fmulVec(v5, v1, v1, s4),
                // Second 4 elements: x^3  
                fmulVec(v6, v5, v1, s4),
                // Second 4 elements: x^5 = x^3 * x^2
                fmulVec(v7, v6, v5, s4),
                
                // First 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v8, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v9, v3, v29, s4),
                // +0.002083 * x^5
                fmulVec(v10, v4, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v11, v31, v8, s4),
                faddVec(v11, v11, v9, s4),
                faddVec(v11, v11, v10, s4),
                
                // Second 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v12, v1, v30, s4),
                // -0.020833 * x^3
                fmulVec(v13, v6, v29, s4),
                // +0.002083 * x^5
                fmulVec(v14, v7, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v15, v31, v12, s4),
                faddVec(v15, v15, v13, s4),
                faddVec(v15, v15, v14, s4),
                
                str(v11, x9, 0, q),
                str(v15, x9, 16, q),

                // 4 elements
                ldr(v0, x8, 8*4, q),
                
                // x^2
                fmulVec(v1, v0, v0, s4),
                // x^3  
                fmulVec(v2, v1, v0, s4),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s4),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s4),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s4),
                faddVec(v7, v7, v5, s4),
                faddVec(v7, v7, v6, s4),
                
                str(v7, x9, 8*4, q),
                
                // 2 elements
                ldr(v0, x8, 12*4, d),
                
                // x^2
                fmulVec(v1, v0, v0, s2),
                // x^3  
                fmulVec(v2, v1, v0, s2),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s2),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s2),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s2),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s2),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s2),
                faddVec(v7, v7, v5, s2),
                faddVec(v7, v7, v6, s2),
                
                str(v7, x9, 12*4, d)
            });
            break;
        case 15:
            kernel.add_instr({
                // 8 elements
                ldr(v0, x8, 0, q),
                ldr(v1, x8, 16, q),
                
                // First 4 elements: x^2
                fmulVec(v2, v0, v0, s4),
                // First 4 elements: x^3  
                fmulVec(v3, v2, v0, s4),
                // First 4 elements: x^5 = x^3 * x^2
                fmulVec(v4, v3, v2, s4),
                
                // Second 4 elements: x^2
                fmulVec(v5, v1, v1, s4),
                // Second 4 elements: x^3  
                fmulVec(v6, v5, v1, s4),
                // Second 4 elements: x^5 = x^3 * x^2
                fmulVec(v7, v6, v5, s4),
                
                // First 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v8, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v9, v3, v29, s4),
                // +0.002083 * x^5
                fmulVec(v10, v4, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v11, v31, v8, s4),
                faddVec(v11, v11, v9, s4),
                faddVec(v11, v11, v10, s4),
                
                // Second 4 elements polynomial calculation
                // 0.25 * x
                fmulVec(v12, v1, v30, s4),
                // -0.020833 * x^3
                fmulVec(v13, v6, v29, s4),
                // +0.002083 * x^5
                fmulVec(v14, v7, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v15, v31, v12, s4),
                faddVec(v15, v15, v13, s4),
                faddVec(v15, v15, v14, s4),
                
                str(v11, x9, 0, q),
                str(v15, x9, 16, q),
                
                // 4 elements
                ldr(v0, x8, 8*4, q),
                
                // x^2
                fmulVec(v1, v0, v0, s4),
                // x^3  
                fmulVec(v2, v1, v0, s4),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s4),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s4),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s4),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s4),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s4),
                faddVec(v7, v7, v5, s4),
                faddVec(v7, v7, v6, s4),
                
                str(v7, x9, 8*4, q),
                
                // 2 elements
                ldr(v0, x8, 12*4, d),
                
                // x^2
                fmulVec(v1, v0, v0, s2),
                // x^3  
                fmulVec(v2, v1, v0, s2),
                // x^5 = x^3 * x^2
                fmulVec(v3, v2, v1, s2),
                
                // 0.25 * x
                fmulVec(v4, v0, v30, s2),
                // -0.020833 * x^3
                fmulVec(v5, v2, v29, s2),
                // +0.002083 * x^5
                fmulVec(v6, v3, v28, s2),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddVec(v7, v31, v4, s2),
                faddVec(v7, v7, v5, s2),
                faddVec(v7, v7, v6, s2),
                
                str(v7, x9, 12*4, d),
                
                // 1 element
                ldr(v0, x8, 14*4, s),
                
                // x^2
                fmulScalar(v1, v0, v0, s),
                // x^3
                fmulScalar(v2, v1, v0, s),
                // x^5 = x^3 * x^2
                fmulScalar(v3, v2, v1, s),
                
                // 0.25 * x
                fmulScalar(v4, v0, v30, s),
                // -0.020833 * x^3
                fmulScalar(v5, v2, v29, s),
                // +0.002083 * x^5
                fmulScalar(v6, v3, v28, s),
                
                // 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5
                faddScalar(v7, v31, v4, s),
                faddScalar(v7, v7, v5, s),
                faddScalar(v7, v7, v6, s),
                
                str(v7, x9, 14*4, s)
            });
            break;
        default:
            break;
        }
    }

    kernel.add_instr({
        // Jump to next column
        add(x4, x4, x2, 0, 0),
        add(x5, x5, x3, 0, 0),

        // Decrement n loop counter
        sub(x6, x6, 1, 0)
    });

    // Check if loop counter is zero
    int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
    kernel.add_instr(cbnz(x6, -l_nLoopInstrCount * 4));

    kernel.add_instr({
        // Restore callee-saved registers
        simd_fp::ldpPost(v10, v11, sp, 16, d),
        simd_fp::ldpPost(v8, v9, sp, 16, d),

        // Restore stack space allocated for constants
        add(sp, sp, 64, 0),

        // Restore stack pointer
        ldpPost(x29, x30, sp, 16),

        inst::ret()
    });

    kernel.write("sigmoid_primitive.bin");
    kernel.set_kernel();
} 