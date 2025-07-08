#include <mlc/Kernel.h>
#include <mlc/Unary.h>
#include <mlc/instructions/all_instructions.h>
#include <mlc/kernels/unary/sigmoid_interp_primitive.h>
#include <mlc/registers/gp_registers.h>
#include <mlc/registers/simd_fp_registers.h>

using enum gpr_t;
using enum simd_fp_t;
using enum neon_size_spec_t;
using enum arr_spec_t;

using namespace mini_jit::instructions::base;
using namespace mini_jit::instructions::simd_fp;

void mini_jit::kernels::unary::sigmoid_interpolation(mini_jit::Kernel& kernel,
                                                     u_int32_t         m,
                                                     u_int32_t         n)
{
    // Inputs:
    // x0: pointer to A (input)
    // x1: pointer to B (output)
    // x2: leading dimension of A
    // x3: leading dimension of B
    // x4: pointer to Lookup Table

    // Prepare the kernel
    int mLoopIterations = m / 4;
    int mLoopRemainder  = m % 4;

    kernel.add_instr({
        // PCS - Proper stack frame setup
        stpPre(x29, x30, sp, -16),
        movSP(x29, sp),

        // Compute stride (convert to bytes)
        lsl(x2, x2, 2), // x2 = ldA * 4 (stride in bytes)
        lsl(x3, x3, 2), // x3 = ldB * 4 (stride in bytes)

        // Save base matrix pointers
        mov(x5, x0), // A (input)
        mov(x6, x1), // B (output)

        // Pre-calculated registers
        fmovIntVec(v31, -8, s4),
        fmovIntVec(v30, 8, s4),
        fmovIntVec(v29, 2, s4),
        fmovIntVec(v28, 31, s4), // max value = 31 (so i+1 <= 32)

        // Set n loop counter
        mov(x7, n),
    });

    // Start n loop (1 column)
    kernel.add_label("n_loop");

    // Set m loop counter
    kernel.add_instr({
        mov(x8, mLoopIterations),

        // working pointers for rows
        mov(x16, x5), // A (input pointer)
        mov(x17, x6)  // B (output pointer)
    });

    if (mLoopIterations > 0)
    {
        kernel.add_label("m_16_loop");
        kernel.add_instr({
            // Load 4 elements
            ldr(v0, x16, 0, q),

            // 1. Clamping: values in range [-8.0,8.0]
            fmaxVec(v0, v0, v31, s4),
            fminVec(v0, v0, v30, s4),

            // 2.1 Compute table indices
            faddVec(v1, v0, v30, s4), // x + 8.0
            fmulVec(v2, v1, v29, s4), // 2 * (x + 8.0)

            // 2.2 Clamp indices to [0, 31] to prevent out-of-bounds access
            fminVec(v2, v2, v28, s4), // clamp to <= 31

            // 3. Integer Parts
            frintmVec(v3, v2, s4),   // gets the "integer" - XX.FP from v2 and floors the value
            fsubVec(v5, v2, v3, s4), // gets the "float" - .XX
            fcvtmsVec(v4, v3, s4),   // conversion from v3 float-integer to real integer for indexing

            // Table[i]
            // 4. Extract Lanes to GPRs
            umov(w10, v4, 0, s),
            umov(w11, v4, 1, s),
            umov(w12, v4, 2, s),
            umov(w13, v4, 3, s),

            // Multiply with datatype
            lsl(w10, w10, 2),
            lsl(w11, w11, 2),
            lsl(w12, w12, 2),
            lsl(w13, w13, 2),

            // 5. Load values from table at index i
            ldrReg(v6, x4, w10, 0, s),
            ldrReg(v7, x4, w11, 0, s),
            ldrReg(v16, x4, w12, 0, s),
            ldrReg(v17, x4, w13, 0, s),

            // 6. Calculate first vector
            ins(v18, v6, 0, 0, s),
            ins(v18, v7, 1, 0, s),
            ins(v18, v16, 2, 0, s),
            ins(v18, v17, 3, 0, s),

            // Table[i+1]
            // 4.1 Update Lanes in GPRs
            add(w10, w10, 4, 0),
            add(w11, w11, 4, 0),
            add(w12, w12, 4, 0),
            add(w13, w13, 4, 0),

            // 5.1 Load values from table at index i+1
            ldrReg(v19, x4, w10, 0, s),
            ldrReg(v20, x4, w11, 0, s),
            ldrReg(v21, x4, w12, 0, s),
            ldrReg(v22, x4, w13, 0, s),

            // 6.1 Calculate second vector
            ins(v23, v19, 0, 0, s),
            ins(v23, v20, 1, 0, s),
            ins(v23, v21, 2, 0, s),
            ins(v23, v22, 3, 0, s),

            // 7. Vectorized Interpolation
            fsubVec(v24, v23, v18, s4), // v24 = diff
            fmlaVec(v18, v5, v24, s4),  // v18 = t[i], v5 = frac, v24 = diff

            // Store 4 elements
            str(v18, x17, 0, q),

            // Advance pointers by 4 elements (16 bytes)
            add(x16, x16, 16, 0), // advance input pointer
            add(x17, x17, 16, 0), // advance output pointer

            // Decrement m loop counter
            sub(x8, x8, 1, 0),
        });

        // Check if loop counter is zero
        kernel.add_instr(cbnz(x8, -kernel.getInstrCountFromLabel("m_16_loop") * 4));
    }

    // Handle remainder elements if needed
    if (mLoopRemainder > 0)
    {
        switch (mLoopRemainder)
        {
        case 1:
            kernel.add_instr({
                // Load 1 element
                ldr(v0, x16, 0, s),

                // 1. Clamping: values in range [-8.0,8.0]
                fmaxScalar(v0, v0, v31, s),
                fminScalar(v0, v0, v30, s),

                // 2.1 Compute table indices
                faddScalar(v1, v0, v30, s), // x + 8.0
                fmulScalar(v2, v1, v29, s), // 2 * (x + 8.0)

                // 2.2 Clamp indices to [0, 31] to prevent out-of-bounds access
                fminScalar(v2, v2, v28, s), // clamp to <= 31

                // 3. Integer Parts
                frintmScalar(v3, v2, s),   // gets the "integer" - XX.FP from v2 and floors the value
                fsubScalar(v5, v2, v3, s), // gets the "float" - .XX
                fcvtmsScalar(v4, v3, s),   // conversion from v3 float-integer to real integer for indexing

                // Table[i]
                // 4. Extract Lanes to GPRs
                umov(w10, v4, 0, s),

                // Multiply with datatype
                lsl(w10, w10, 2),

                // 5. & 6. Load values from table at index i
                ldrReg(v6, x4, w10, 0, s),

                // Table[i+1]
                // 4.1 Update Lanes in GPRs
                add(w10, w10, 4, 0),

                // 5.1 & 6.1 Load values from table at index i+1
                ldrReg(v7, x4, w10, 0, s),

                // 7. Vectorized Interpolation
                fsubScalar(v16, v7, v6, s), // v16 = diff
                fmadd(v6, v5, v16, v6, s),  // v6 = t[i], v9 = frac, v16 = diff

                // Store 4 elements
                str(v6, x17, 0, s),
            });
            break;
        case 2:
            kernel.add_instr({
                // Load 2 elements
                ldr(v0, x16, 0, d),

                // 1. Clamping: values in range [-8.0,8.0]
                fmaxVec(v0, v0, v31, s2),
                fminVec(v0, v0, v30, s2),

                // 2.1 Compute table indices
                faddVec(v1, v0, v30, s2), // x + 8.0
                fmulVec(v2, v1, v29, s2), // 2 * (x + 8.0)

                // 2.2 Clamp indices to [0, 31] to prevent out-of-bounds access
                fminVec(v2, v2, v28, s2), // clamp to <= 31

                // 3. Integer Parts
                frintmVec(v3, v2, s2),   // gets the "integer" - XX.FP from v2 and floors the value
                fsubVec(v5, v2, v3, s2), // gets the "float" - .XX
                fcvtmsVec(v4, v3, s2),   // conversion from v3 float-integer to real integer for indexing

                // Table[i]
                // 4. Extract Lanes to GPRs
                umov(w10, v4, 0, s),
                umov(w11, v4, 1, s),

                // Multiply with datatype
                lsl(w10, w10, 2),
                lsl(w11, w11, 2),

                // 5. Load values from table at index i
                ldrReg(v6, x4, w10, 0, s),
                ldrReg(v7, x4, w11, 0, s),

                // 6. Calculate first vector
                ins(v16, v6, 0, 0, s),
                ins(v16, v7, 1, 0, s),

                // Table[i+1]
                // 4.1 Update Lanes in GPRs
                add(w10, w10, 4, 0),
                add(w11, w11, 4, 0),

                // 5.1 Load values from table at index i+1
                ldrReg(v17, x4, w10, 0, s),
                ldrReg(v18, x4, w11, 0, s),

                // 6.1 Calculate second vector
                ins(v19, v17, 0, 0, s),
                ins(v19, v18, 1, 0, s),

                // 7. Vectorized Interpolation
                fsubVec(v20, v19, v16, s2), // v20 = diff
                fmlaVec(v16, v5, v20, s2),  // v16 = t[i], v5 = frac, v20 = diff

                // Store 4 elements
                str(v16, x17, 0, d),
            });
            break;
        case 3:
            kernel.add_instr({
                // Load 3 elements
                ldr(v0, x16, 0, d),
                ldr(v1, x16, 8, s),

                // 1. Clamping: values in range [-8.0,8.0]
                fmaxVec(v0, v0, v31, s2),
                fmaxScalar(v1, v1, v31, s),
                // Scalar
                fminVec(v0, v0, v30, s2),
                fminScalar(v1, v1, v30, s),

                // 2.1 Compute table indices
                faddVec(v2, v0, v30, s2),   // x + 8.0
                faddScalar(v3, v1, v30, s), // x + 8.0
                // Scalar
                fmulVec(v4, v2, v29, s2),   // 2 * (x + 8.0)
                fmulScalar(v5, v3, v29, s), // 2 * (x + 8.0)

                // 2.2 Clamp indices to [0, 31] to prevent out-of-bounds access
                fminVec(v4, v4, v28, s2),   // clamp to <= 31
                fminScalar(v5, v5, v28, s), // clamp to <= 31

                // 3. Integer Parts
                frintmVec(v6, v4, s2),   // gets the "integer" - XX.FP from v4 and floors the value
                frintmScalar(v7, v5, s), // gets the "integer" - XX.FP from v5 and floors the value

                fsubVec(v18, v4, v6, s2),   // gets the "float" - .XX
                fsubScalar(v19, v5, v7, s), // gets the "float" - .XX

                fcvtmsVec(v16, v6, s2),   // conversion from v6 float-integer to real integer for indexing
                fcvtmsScalar(v17, v7, s), // conversion from v7 float-integer to real integer for indexing

                // Table[i]
                // 4. Extract Lanes to GPRs
                umov(w10, v16, 0, s),
                umov(w11, v16, 1, s),
                umov(w12, v17, 0, s),

                // Multiply with datatype
                lsl(w10, w10, 2),
                lsl(w11, w11, 2),
                lsl(w12, w12, 2),

                // 5. Load values from table at index i
                ldrReg(v20, x4, w10, 0, s),
                ldrReg(v21, x4, w11, 0, s),
                ldrReg(v22, x4, w12, 0, s),

                // 6. Calculate first vector
                ins(v23, v20, 0, 0, s),
                ins(v23, v21, 1, 0, s),

                // Table[i+1]
                // 4.1 Update Lanes in GPRs
                add(w10, w10, 4, 0),
                add(w11, w11, 4, 0),
                add(w12, w12, 4, 0),

                // 5.1 Load values from table at index i+1
                ldrReg(v24, x4, w10, 0, s),
                ldrReg(v25, x4, w11, 0, s),
                ldrReg(v26, x4, w12, 0, s),

                // 6.1 Calculate second vector
                ins(v27, v24, 0, 0, s),
                ins(v27, v25, 1, 0, s),

                // 7. Vectorized Interpolation
                fsubVec(v16, v27, v23, s2), // v16 = diff for vector elements
                fmlaVec(v23, v18, v16, s2), // v23 = table[i] + frac * diff for vector elements

                // Scalar interpolation for third element
                fsubScalar(v17, v26, v22, s), // v22 = table[i+1] - table[i] for third element
                fmadd(v22, v19, v17, v22, s), // v22 = table[i] + frac * diff for third element

                // Store 3 elements
                str(v23, x17, 0, d),
                str(v22, x17, 8, s),
            });
            break;
        default:
            break;
        }
    }

    kernel.add_instr({                       // Jump to next column
                      add(x5, x5, x2, 0, 0), // input pointer += stride
                      add(x6, x6, x3, 0, 0), // output pointer += stride

                      // Decrement n loop counter
                      sub(x7, x7, 1, 0)});

    // Check if loop counter is zero
    int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
    kernel.add_instr(cbnz(x7, -l_nLoopInstrCount * 4));

    kernel.add_instr({// Restore stack pointer
                      ldpPost(x29, x30, sp, 16),

                      ret()});

    kernel.write("sigmoid_interp_primitive.bin");
    kernel.set_kernel();
}