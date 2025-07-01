#include "sigmoid_interp_primitive.h"
#include "Kernel.h"
#include "Unary.h"

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
using simd_fp::ins;
using simd_fp::umov;
using simd_fp::frintnVec;
using simd_fp::frintnScalar;
using simd_fp::frintmVec;
using simd_fp::frintmScalar;
using simd_fp::fcvtmsVec;
using simd_fp::fcvtmsScalar;
using simd_fp::ldp;
using simd_fp::ldpPost;
using simd_fp::stp;
using simd_fp::stpPre;
using simd_fp::ldr;
using simd_fp::ldrReg;
using simd_fp::str;
using simd_fp::faddVec;
using simd_fp::faddScalar;
using simd_fp::fsubVec;
using simd_fp::fsubScalar;
using simd_fp::fmaxVec;
using simd_fp::fmaxScalar;
using simd_fp::fmlaVec;
using simd_fp::fmlaElem;
using simd_fp::fmadd;
using simd_fp::fmulVec;
using simd_fp::fmulScalar;
using simd_fp::fdivVec;
using simd_fp::fdivScalar;
using simd_fp::fmovIntVec;
using simd_fp::fmovIntScalar;
using simd_fp::fminVec;
using simd_fp::fminScalar;
using simd_fp::scvtfVec;
using simd_fp::scvtfScalar;
using simd_fp::scvtfBaseScalar;

void mini_jit::kernels::unary::sigmoid_interpolation(mini_jit::Kernel &kernel,
                                                     u_int32_t m,
                                                     u_int32_t n)
{
    // Inputs:
    // x0: pointer to A (input)
    // x1: pointer to B (output)
    // x2: pointer to Lookup Table 
    // x3: leading dimension of A
    // x4: leading dimension of B

    // Prepare the kernel
    int mLoopIterations = m / 4;
    int mLoopRemainder = m % 4;

    kernel.add_instr({
        // PCS - Proper stack frame setup
        stpPre(x29, x30, sp, -16),
        movSP(x29, sp),

        // Save callee-saved registers (save in pairs for 16-byte alignment)
        stpPre(x21, x22, sp, -16),
        stpPre(x23, x24, sp, -16),
        
        // Compute stride (convert to bytes)
        lsl(x3, x3, 2),  // x3 = ldA * 4 (stride in bytes)
        lsl(x4, x4, 2),  // x4 = ldB * 4 (stride in bytes)

        // Save base matrix pointers
        mov(x5, x0), // A (input)
        mov(x6, x1), // B (output)

        // Set n loop counter
        mov(x7, n),
    });

    // Start n loop (1 column)
    kernel.add_label("n_loop");

    // Set m loop counter
    kernel.add_instr({
        mov(x8, mLoopIterations),

        // working pointers for rows
        mov(x22, x5), // A (input pointer)
        mov(x23, x6)  // B (output pointer)
    });

    if (mLoopIterations > 0)
    {
        kernel.add_label("m_16_loop");
        kernel.add_instr({
            // Load 4 elements
            ldr(v0, x22, 0, q),

            // 1. Clamping: values in range [-8.0,8.0]
            fmovIntVec(v1, -8, s4),
            fmovIntVec(v2,  8, s4),
            fmaxVec(v0, v0, v1, s4),
            fminVec(v0, v0, v2, s4),

            // 2.1 Compute table indices
            faddVec(v4, v0, v2, s4),  // x + 8.0
            fmovIntVec(v5, 2, s4),
            fmulVec(v6, v4, v5, s4),  // 2 * (x + 8.0)

            // 2.2 Clamp indices to [0, 31] to prevent out-of-bounds access
            fmovIntVec(v18, 31, s4),  // max value = 31 (so i+1 <= 32)
            fminVec(v6, v6, v18, s4), // clamp to <= 31

            // // 3. Integer Parts - Potential SegFaults caused by fcvtms with floats
            // fcvtmsVec(v7, v6, s4),
            // scvtfVec(v8, v7, s4),
            // fsubVec(v9, v6, v8, s4),

            // 3. Integer Parts
            frintmVec(v7, v6, s4),      // gets the "integer" - XX.FP from v6 and floors the value
            fsubVec(v9, v6, v7, s4),    // gets the "float" - .XX 
            fcvtmsVec(v8, v7, s4),      // conversion from v7 float-integer to real integer for indexing

            // Table[i]
            // 4. Extract Lanes to GPRs
            umov(w10, v8, 0, s),
            umov(w11, v8, 1, s),
            umov(w12, v8, 2, s),
            umov(w13, v8, 3, s),

            // Multiply with datatype
            lsl(w10, w10, 2),
            lsl(w11, w11, 2),
            lsl(w12, w12, 2),
            lsl(w13, w13, 2),

            // 5. Load values from table at index i
            ldrReg(v10, x2, w10, 0, s),
            ldrReg(v11, x2, w11, 0, s),
            ldrReg(v12, x2, w12, 0, s),
            ldrReg(v13, x2, w13, 0, s),

            // 6. Calculate first vector
            ins(v14, v10, 0, 0, s),
            ins(v14, v11, 1, 0, s),
            ins(v14, v12, 2, 0, s),
            ins(v14, v13, 3, 0, s),

            // Table[i+1]
            // 4.1 Update Lanes in GPRs
            add(w10, w10, 4, 0),
            add(w11, w11, 4, 0),
            add(w12, w12, 4, 0),
            add(w13, w13, 4, 0),

            // 5.1 Load values from table at index i+1
            ldrReg(v18, x2, w10, 0, s),
            ldrReg(v19, x2, w11, 0, s),
            ldrReg(v20, x2, w12, 0, s),
            ldrReg(v21, x2, w13, 0, s),

            // 6.1 Calculate second vector
            ins(v15, v18, 0, 0, s),
            ins(v15, v19, 1, 0, s),
            ins(v15, v20, 2, 0, s),
            ins(v15, v21, 3, 0, s),

            // 7. Vectorized Interpolation
            fsubVec(v16, v15, v14, s4), // v16 = diff
            fmlaVec(v14, v9, v16, s4),  // v14 = t[i], v9 = frac, v16 = diff

            // Store 4 elements
            str(v14, x23, 0, q),

            // Advance pointers by 4 elements (16 bytes)
            add(x22, x22, 16, 0),    // advance input pointer
            add(x23, x23, 16, 0),    // advance output pointer

            // Decrement m loop counter
            sub(x8, x8, 1, 0),
        });

        // Check if loop counter is zero
        kernel.add_instr(cbnz(x8, -kernel.getInstrCountFromLabel("m_16_loop") * 4));
    }

    // Handle remainder elements if needed
    if (mLoopRemainder > 0)
    {
        // TODO: Handle remaining elements (1-3 elements)
        switch(mLoopRemainder)
        {
            case 1:
                kernel.add_instr({
                    // Load 1 element
                    ldr(v0, x22, 0, s),

                    // 1. Clamping: values in range [-8.0,8.0]
                    fmovIntScalar(v1, -8, s),
                    fmovIntScalar(v2,  8, s),
                    fmaxScalar(v0, v0, v1, s),
                    fminScalar(v0, v0, v2, s),

                    // 2.1 Compute table indices
                    faddScalar(v4, v0, v2, s),  // x + 8.0
                    fmovIntScalar(v5, 2, s),
                    fmulScalar(v6, v4, v5, s),  // 2 * (x + 8.0)

                    // 2.2 Clamp indices to [0, 31] to prevent out-of-bounds access
                    fmovIntScalar(v18, 31, s),  // max value = 31 (so i+1 <= 32)
                    fminScalar(v6, v6, v18, s), // clamp to <= 31

                    // 3. Integer Parts
                    frintmScalar(v7, v6, s),      // gets the "integer" - XX.FP from v6 and floors the value
                    fsubScalar(v9, v6, v7, s),    // gets the "float" - .XX 
                    fcvtmsScalar(v8, v7, s),      // conversion from v7 float-integer to real integer for indexing

                    // Table[i]
                    // 4. Extract Lanes to GPRs
                    umov(w10, v8, 0, s),

                    // Multiply with datatype
                    lsl(w10, w10, 2),

                    // 5. Load values from table at index i
                    ldrReg(v10, x2, w10, 0, s),

                    // 6. Calculate first vector
                    ins(v14, v10, 0, 0, s),

                    // Table[i+1]
                    // 4.1 Update Lanes in GPRs
                    add(w10, w10, 4, 0),

                    // 5.1 Load values from table at index i+1
                    ldrReg(v18, x2, w10, 0, s),

                    // 6.1 Calculate second vector
                    ins(v15, v18, 0, 0, s),

                    // 7. Vectorized Interpolation
                    fsubScalar(v16, v15, v14, s), // v16 = diff
                    fmadd(v14, v9, v16, v14, s),  // v14 = t[i], v9 = frac, v16 = diff

                    // Store 4 elements
                    str(v14, x23, 0, s),
                });
                break;
            case 2:
                kernel.add_instr({
                    // Load 2 elements
                    ldr(v0, x22, 0, d),

                    // 1. Clamping: values in range [-8.0,8.0]
                    fmovIntVec(v1, -8, s2),
                    fmovIntVec(v2,  8, s2),
                    fmaxVec(v0, v0, v1, s2),
                    fminVec(v0, v0, v2, s2),

                    // 2.1 Compute table indices
                    faddVec(v4, v0, v2, s2),  // x + 8.0
                    fmovIntVec(v5, 2, s2),
                    fmulVec(v6, v4, v5, s2),  // 2 * (x + 8.0)

                    // 2.2 Clamp indices to [0, 31] to prevent out-of-bounds access
                    fmovIntVec(v18, 31, s2),  // max value = 31 (so i+1 <= 32)
                    fminVec(v6, v6, v18, s2), // clamp to <= 31

                    // // 3. Integer Parts - Potential SegFaults caused by fcvtms with floats
                    // fcvtmsVec(v7, v6, s4),
                    // scvtfVec(v8, v7, s4),
                    // fsubVec(v9, v6, v8, s4),

                    // 3. Integer Parts
                    frintmVec(v7, v6, s2),      // gets the "integer" - XX.FP from v6 and floors the value
                    fsubVec(v9, v6, v7, s2),    // gets the "float" - .XX 
                    fcvtmsVec(v8, v7, s2),      // conversion from v7 float-integer to real integer for indexing

                    // Table[i]
                    // 4. Extract Lanes to GPRs
                    umov(w10, v8, 0, s),
                    umov(w11, v8, 1, s),

                    // Multiply with datatype
                    lsl(w10, w10, 2),
                    lsl(w11, w11, 2),

                    // 5. Load values from table at index i
                    ldrReg(v10, x2, w10, 0, s),
                    ldrReg(v11, x2, w11, 0, s),

                    // 6. Calculate first vector
                    ins(v14, v10, 0, 0, s),
                    ins(v14, v11, 1, 0, s),

                    // Table[i+1]
                    // 4.1 Update Lanes in GPRs
                    add(w10, w10, 4, 0),
                    add(w11, w11, 4, 0),

                    // 5.1 Load values from table at index i+1
                    ldrReg(v18, x2, w10, 0, s),
                    ldrReg(v19, x2, w11, 0, s),

                    // 6.1 Calculate second vector
                    ins(v15, v18, 0, 0, s),
                    ins(v15, v19, 1, 0, s),

                    // 7. Vectorized Interpolation
                    fsubVec(v16, v15, v14, s2), // v16 = diff
                    fmlaVec(v14, v9, v16, s2),  // v14 = t[i], v9 = frac, v16 = diff

                    // Store 4 elements
                    str(v14, x23, 0, d),
                });
                break;
            case 3:
                kernel.add_instr({
                    // Load 3 elements
                    ldr(v0, x22, 0, d),
                    ldr(v17, x22, 8, s),

                    // 1. Clamping: values in range [-8.0,8.0]
                    fmovIntVec(v1, -8, s2),
                    fmovIntScalar(v18, -8, s),

                    fmovIntVec(v2,  8, s2),
                    fmovIntScalar(v19,  8, s),
                    
                    fmaxVec(v0, v0, v1, s2),
                    fmaxScalar(v17, v17, v18, s),
                    
                    fminVec(v0, v0, v2, s2),
                    fminScalar(v17, v17, v19, s),

                    // 2.1 Compute table indices
                    faddVec(v4, v0, v2, s2),       // x + 8.0
                    faddScalar(v20, v17, v19, s),  // x + 8.0

                    fmovIntVec(v5, 2, s2),
                    fmovIntScalar(v21, 2, s),

                    fmulVec(v6, v4, v5, s2),       // 2 * (x + 8.0)
                    fmulScalar(v22, v20, v21, s),  // 2 * (x + 8.0)

                    // 2.2 Clamp indices to [0, 31] to prevent out-of-bounds access
                    fmovIntVec(v18, 31, s2),    // max value = 31 (so i+1 <= 32)
                    fmovIntScalar(v23, 31, s),  // max value = 31 (so i+1 <= 32)

                    fminVec(v6, v6, v18, s2),     // clamp to <= 31
                    fminScalar(v22, v22, v23, s), // clamp to <= 31

                    // 3. Integer Parts
                    frintmVec(v7, v6, s2),      // gets the "integer" - XX.FP from v6 and floors the value
                    frintmScalar(v23, v22, s),      // gets the "integer" - XX.FP from v22 and floors the value

                    fsubVec(v9, v6, v7, s2),    // gets the "float" - .XX 
                    fsubScalar(v24, v22, v23, s),    // gets the "float" - .XX 

                    fcvtmsVec(v8, v7, s2),      // conversion from v7 float-integer to real integer for indexing
                    fcvtmsScalar(v25, v23, s),      // conversion from v23 float-integer to real integer for indexing

                    // Table[i]
                    // 4. Extract Lanes to GPRs
                    umov(w10, v8, 0, s),
                    umov(w11, v8, 1, s),
                    umov(w12, v25, 0, s),

                    // Multiply with datatype
                    lsl(w10, w10, 2),
                    lsl(w11, w11, 2),
                    lsl(w12, w12, 2),

                    // 5. Load values from table at index i
                    ldrReg(v10, x2, w10, 0, s),
                    ldrReg(v11, x2, w11, 0, s),
                    ldrReg(v12, x2, w12, 0, s),

                    // 6. Calculate first vector
                    ins(v14, v10, 0, 0, s),
                    ins(v14, v11, 1, 0, s),
                    ins(v14, v12, 2, 0, s),

                    // Table[i+1]
                    // 4.1 Update Lanes in GPRs
                    add(w10, w10, 4, 0),
                    add(w11, w11, 4, 0),
                    add(w12, w12, 4, 0),

                    // 5.1 Load values from table at index i+1
                    ldrReg(v18, x2, w10, 0, s),
                    ldrReg(v19, x2, w11, 0, s),
                    ldrReg(v20, x2, w12, 0, s),

                    // 6.1 Calculate second vector
                    ins(v15, v18, 0, 0, s),
                    ins(v15, v19, 1, 0, s),
                    ins(v26, v20, 0, 0, s),

                    // 7. Vectorized Interpolation
                    fsubVec(v16, v15, v14, s2), // v16 = diff for vector elements
                    fmlaVec(v14, v9, v16, s2),  // v14 = table[i] + frac * diff for vector elements

                    // Scalar interpolation for third element
                    fsubScalar(v27, v26, v12, s), // v27 = table[i+1] - table[i] for third element
                    fmadd(v28, v24, v27, v12, s), // v28 = table[i] + frac * diff for third element

                    // Store 3 elements
                    str(v14, x23, 0, d),
                    str(v28, x23, 8, s),
                });
                break;
            default:
                break;
        }
    }

    kernel.add_instr({
        // Jump to next column
        add(x5, x5, x3, 0, 0),    // input pointer += stride
        add(x6, x6, x4, 0, 0),    // output pointer += stride

        // Decrement n loop counter
        sub(x7, x7, 1, 0)
    });

    // Check if loop counter is zero
    int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
    kernel.add_instr(cbnz(x7, -l_nLoopInstrCount * 4));

    kernel.add_instr({
        // Restore callee-saved registers (in reverse order)
        ldpPost(x23, x24, sp, 16),
        ldpPost(x21, x22, sp, 16),

        // Restore stack pointer
        ldpPost(x29, x30, sp, 16),

        inst::ret()
    });

    kernel.write("sigmoid_interp_primitive.bin");
    kernel.set_kernel();
}