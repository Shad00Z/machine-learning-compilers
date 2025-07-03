#include "identity_trans_primitive.h"
#include "Kernel.h"

#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
#include "instructions/all_instructions.h"

using enum gpr_t;
using enum simd_fp_t;
using enum neon_size_spec_t;
using enum arr_spec_t;

using namespace mini_jit::instructions::base;
using namespace mini_jit::instructions::simd_fp;
namespace internal_subkernels = mini_jit::kernels::unary::internal;

void mini_jit::kernels::unary::identity_trans(mini_jit::Kernel &kernel,
                                              int m,
                                              int n)
{
    // Prepare the kernel
    int mLoopIterations = m / 4;
    int mLoopRemainder = m % 4;

    int nLoopIterations = n / 4;
    int nLoopRemainder = n % 4;

    // PCS
    kernel.add_instr(stpPre(x29, x30, sp, -16));
    kernel.add_instr(movSP(x29, sp));

    // Save callee-saved registers
    kernel.add_instr(stpPre(x25, x26, sp, -16));
    kernel.add_instr(stpPre(x27, x28, sp, -16));

    kernel.add_instr(stpPre(v8, v9, sp, -16, d));
    kernel.add_instr(stpPre(v10, v11, sp, -16, d));
    kernel.add_instr(stpPre(v12, v13, sp, -16, d));
    kernel.add_instr(stpPre(v14, v15, sp, -16, d));

    // Save base matrix pointer
    kernel.add_instr(mov(x4, x0)); // A
    kernel.add_instr(mov(x5, x1)); // B

    // Compute stride for A and B
    kernel.add_instr(lsl(x2, x2, 2));
    kernel.add_instr(lsl(x3, x3, 2));

    // Set n loop counter
    kernel.add_instr(mov(x9, nLoopIterations));

    // Row and column Pointer for A and B
    kernel.add_instr(mov(x12, 0)); // Columns of A
    kernel.add_instr(mov(x13, 0)); // Rows of B

    // Some constant values:
    // Jumping 4 rows in A | B      - (x25)
    kernel.add_instr(mov(x25, 4*4)); 

    // Jumping 4 columns in A       - (x26)
    kernel.add_instr(lsl(x26, x2, 2));

    // Jumping 4 columns in B       - (x27)
    kernel.add_instr(lsl(x27, x3, 2));

    if ( nLoopIterations > 0)
    {
        // Start n loop (1 column)
        kernel.add_label("n_loop");
    
        if (mLoopIterations > 0)
        {
            internal_subkernels::identityM4N4( kernel, mLoopIterations );
        }
    
        if (mLoopRemainder > 0)
        {
            switch (mLoopRemainder)
            {
            case 1:
                internal_subkernels::identityM1N4( kernel );
                break;
            case 2:
                internal_subkernels::identityM2N4( kernel );
                break;
            case 3:
                internal_subkernels::identityM3N4( kernel );
                break;
            default:
                break;
            }
        }
        
        // Restore positions
        kernel.add_instr(mov(x4, x0));
        kernel.add_instr(mov(x5, x1));
    
        // Update Columns of A
        kernel.add_instr(add(x12, x12, x26, 0, 0));
    
        // Update Rows of B
        kernel.add_instr(add(x13, x13, x25, 0, 0));
    
        // Apply the updates:
        kernel.add_instr(add(x4, x4, x12, 0, 0));
        kernel.add_instr(add(x5, x5, x13, 0, 0));
    
        // decrement n loop counter
        kernel.add_instr(sub(x9, x9, 1, 0));
        // check if loop counter is zero
        int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
        kernel.add_instr(cbnz(x9, -l_nLoopInstrCount * 4));
    }

    // All iterations in the n dimension have been performed, only possibilities are now:
    // nRemainder == 3, 2, 1
    if ( nLoopRemainder > 0 )
    {
        if (mLoopIterations)
        {
            switch (nLoopRemainder)
            {
                case 1:
                    internal_subkernels::identityM4N1( kernel, mLoopIterations );
                    break;
                case 2:
                    internal_subkernels::identityM4N2( kernel, mLoopIterations );
                    break;
                case 3:
                    internal_subkernels::identityM4N3( kernel, mLoopIterations );
                    break;
                default:
                    break;
            }
        }
        
        if (mLoopRemainder > 0)
        {
            switch (mLoopRemainder)
            {
            case 1:
                if ( nLoopRemainder == 1 )
                {
                    internal_subkernels::identityM1N1( kernel );
                }
                else if ( nLoopRemainder == 2 )
                {
                    internal_subkernels::identityM1N2( kernel );
                }
                else if ( nLoopRemainder == 3 )
                {
                    internal_subkernels::identityM1N3( kernel );
                }
                break;
            case 2:
                if ( nLoopRemainder == 1 )
                {
                    internal_subkernels::identityM2N1( kernel );
                }
                else if ( nLoopRemainder == 2 )
                {
                    internal_subkernels::identityM2N2( kernel );
                }
                else if ( nLoopRemainder == 3 )
                {
                    internal_subkernels::identityM2N3( kernel );
                }
                break;
            case 3:
                if ( nLoopRemainder == 1 )
                {
                    internal_subkernels::identityM3N1( kernel );
                }
                else if ( nLoopRemainder == 2 )
                {
                    internal_subkernels::identityM3N2( kernel );
                }
                else if ( nLoopRemainder == 3 )
                {
                    internal_subkernels::identityM3N3( kernel );
                }
                break;
            default:
                break;
            }
        }
    }

    // Restore callee-saved registers
    kernel.add_instr(ldpPost(v14, v15, sp, 16, d));
    kernel.add_instr(ldpPost(v12, v13, sp, 16, d));
    kernel.add_instr(ldpPost(v10, v11, sp, 16, d));
    kernel.add_instr(ldpPost(v8, v9, sp, 16, d));

    kernel.add_instr(ldpPost(x27, x28, sp, 16));
    kernel.add_instr(ldpPost(x25, x26, sp, 16));

    // Restore stack pointer
    kernel.add_instr(ldpPost(x29, x30, sp, 16));

    kernel.add_instr(ret());
    kernel.write("identity_trans_primitive.bin");
    kernel.set_kernel();
}

void mini_jit::kernels::unary::internal::identityM4N4( mini_jit::Kernel &kernel,
                                                       int mLoopIterations )
{
    kernel.add_instr(mov(x6, mLoopIterations));

    kernel.add_label("m_4_n_4_loop");

    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 4x4 block of A (input matrix)
    kernel.add_instr(ldr(v0, x7, 0, q));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(ldr(v1, x7, 0, q));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(ldr(v2, x7, 0, q));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(ldr(v3, x7, 0, q));

    // Transpose 4x4 block
    // TRN
    kernel.add_instr(trn1(v4, v0, v2, s4));
    kernel.add_instr(trn1(v5, v1, v3, s4));
    kernel.add_instr(trn2(v6, v0, v2, s4));
    kernel.add_instr(trn2(v7, v1, v3, s4));

    // ZIP
    kernel.add_instr(zip1(v8, v4, v5, s4));
    kernel.add_instr(zip1(v9, v6, v7, s4));

    kernel.add_instr(zip2(v10, v4, v5, s4));
    kernel.add_instr(zip2(v11, v6, v7, s4));

    // Store 4x4 Block of B
    kernel.add_instr(str(v8, x8, 0, q));
    kernel.add_instr(add(x8, x8, x3, 0, 0));
    kernel.add_instr(str(v9, x8, 0, q));
    kernel.add_instr(add(x8, x8, x3, 0, 0));
    kernel.add_instr(str(v10, x8, 0, q));
    kernel.add_instr(add(x8, x8, x3, 0, 0));
    kernel.add_instr(str(v11, x8, 0, q));

    // Matrix A next 4 rows
    kernel.add_instr(add(x4, x4, x25, 0, 0));

    // Matrix B next 4 columns
    kernel.add_instr(add(x5, x5, x27, 0, 0));
    
    // decrement m loop counter
    kernel.add_instr(sub(x6, x6, 1, 0));
    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_4_n_4_loop");
    kernel.add_instr(cbnz(x6, -l_mLoopInstrCount * 4));
}

void mini_jit::kernels::unary::internal::identityM3N4( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 3x4 block of A (input matrix)
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v0, x17, 8, d));
    kernel.add_instr(ldr(v1, x17, 0, s));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v2, x17, 8, d));
    kernel.add_instr(ldr(v3, x17, 0, s));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v4, x17, 8, d));
    kernel.add_instr(ldr(v5, x17, 0, s));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v6, x17, 8, d));
    kernel.add_instr(ldr(v7, x17, 0, s));

    // Transpose 3x4 block
    // TRN (d)
    kernel.add_instr(trn1(v8, v0, v4, s4));
    kernel.add_instr(trn1(v9, v2, v6, s4));
    kernel.add_instr(trn2(v10, v0, v4, s4));
    kernel.add_instr(trn2(v11, v2, v6, s4));

    // TRN (s)
    kernel.add_instr(trn1(v12, v1, v5, s2));
    kernel.add_instr(trn1(v13, v3, v7, s2));

    // ZIP
    kernel.add_instr(zip1(v14, v8, v9, s4));
    kernel.add_instr(zip1(v15, v10, v11, s4));

    kernel.add_instr(zip1(v18, v12, v13, s2));
    kernel.add_instr(zip2(v19, v12, v13, s2));

    // Store 3x4 Block of B
    kernel.add_instr(str(v14, x8, 0, q));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(str(v15, x8, 0, q));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(strPost(v18, x8, 8, d));
    kernel.add_instr(str(v19, x8, 0, d));
}

void mini_jit::kernels::unary::internal::identityM2N4( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 2x4 block of A (input matrix)
    kernel.add_instr(ldr(v0, x7, 0, d));
    kernel.add_instr(add(x7, x7, x2, 0, 0));

    kernel.add_instr(ldr(v1, x7, 0, d));
    kernel.add_instr(add(x7, x7, x2, 0, 0));

    kernel.add_instr(ldr(v2, x7, 0, d));
    kernel.add_instr(add(x7, x7, x2, 0, 0));

    kernel.add_instr(ldr(v3, x7, 0, d));

    // Transpose 2x4 block
    // TRN
    kernel.add_instr(trn1(v4, v0, v2, s4));
    kernel.add_instr(trn1(v5, v1, v3, s4));

    kernel.add_instr(trn2(v6, v0, v2, s4));
    kernel.add_instr(trn2(v7, v1, v3, s4));

    // ZIP
    kernel.add_instr(zip1(v8, v4, v5, s4));
    kernel.add_instr(zip1(v9, v6, v7, s4));

    // Store 2x4 Block of B
    kernel.add_instr(str(v8, x8, 0, q));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(str(v9, x8, 0, q));
}

void mini_jit::kernels::unary::internal::identityM1N4( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 1x4 block of A (input matrix)
    kernel.add_instr(ldr(v0, x7, 0, s));
    kernel.add_instr(add(x7, x7, x2, 0, 0));

    kernel.add_instr(ldr(v1, x7, 0, s));
    kernel.add_instr(add(x7, x7, x2, 0, 0));

    kernel.add_instr(ldr(v2, x7, 0, s));
    kernel.add_instr(add(x7, x7, x2, 0, 0));

    kernel.add_instr(ldr(v3, x7, 0, s));

    // Transpose 1x4 block
    // TRN
    kernel.add_instr(trn1(v4, v0, v2, s2));
    kernel.add_instr(trn1(v5, v1, v3, s2));

    // ZIP
    kernel.add_instr(zip1(v6, v4, v5, s4));

    // Store 1x4 Block of B
    kernel.add_instr(str(v6, x8, 0, q));
}

void mini_jit::kernels::unary::internal::identityM4N3( mini_jit::Kernel &kernel,
                                                       int mLoopIterations )
{
    kernel.add_instr(mov(x6, mLoopIterations));
    kernel.add_label("m_4_n_3_loop");


    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 4x3 block of A (input matrix)
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v0, x17, 8, d));
    kernel.add_instr(ldr(v1, x17, 0, d));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v2, x17, 8, d));
    kernel.add_instr(ldr(v3, x17, 0, d));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v4, x17, 4, s));
    kernel.add_instr(ldrPost(v5, x17, 4, s));
    kernel.add_instr(ldrPost(v6, x17, 4, s));
    kernel.add_instr(ldr(v7, x17, 0, s));

    // Transpose 4x3 matrix
    // TRN
    kernel.add_instr(trn1(v8, v0, v2, s4));
    kernel.add_instr(trn2(v9, v0, v2, s4));
    kernel.add_instr(trn1(v10, v1, v3, s4));
    kernel.add_instr(trn2(v11, v1, v3, s4));


    // Store 4x3 Block of B
    kernel.add_instr(mov(x17, x8));

    kernel.add_instr(strPost(v8, x17, 8, d));
    kernel.add_instr(str(v4, x17, 0, s));
    kernel.add_instr(add(x8, x8, x3, 0, 0));
    kernel.add_instr(mov(x17, x8));

    kernel.add_instr(strPost(v9, x17, 8, d));
    kernel.add_instr(str(v5, x17, 0, s));
    kernel.add_instr(add(x8, x8, x3, 0, 0));
    kernel.add_instr(mov(x17, x8));

    kernel.add_instr(strPost(v10, x17, 8, d));
    kernel.add_instr(str(v6, x17, 0, s));
    kernel.add_instr(add(x8, x8, x3, 0, 0));
    kernel.add_instr(mov(x17, x8));

    kernel.add_instr(strPost(v11, x17, 8, d));
    kernel.add_instr(str(v7, x17, 0, s));


    // Matrix A next 4 rows
    kernel.add_instr(add(x4, x4, x25, 0, 0));

    // Matrix B next 1 columns
    kernel.add_instr(add(x5, x5, x27, 0, 0));
    
    // decrement m loop counter
    kernel.add_instr(sub(x6, x6, 1, 0));
    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_4_n_3_loop");
    kernel.add_instr(cbnz(x6, -l_mLoopInstrCount * 4));
}

void mini_jit::kernels::unary::internal::identityM4N2( mini_jit::Kernel &kernel,
                                                       int mLoopIterations )
{
    kernel.add_instr(mov(x6, mLoopIterations));
    kernel.add_label("m_4_n_2_loop");


    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 4x2 block of A (input matrix)
    kernel.add_instr(mov(x17, x7));
    kernel.add_instr(ldrPost(v0, x17, 8, d));
    kernel.add_instr(ldr(v1, x17, 0, d));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v2, x17, 8, d));
    kernel.add_instr(ldr(v3, x17, 0, d));

    // Transpose 4x2 matrix
    // TRN
    kernel.add_instr(trn1(v4, v0, v2, s4));
    kernel.add_instr(trn2(v5, v0, v2, s4));

    kernel.add_instr(trn1(v6, v1, v3, s4));
    kernel.add_instr(trn2(v7, v1, v3, s4));

    // Store 4x2 Block of B
    kernel.add_instr(str(v4, x8, 0, d));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(str(v5, x8, 0, d));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(str(v6, x8, 0, d));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(str(v7, x8, 0, d));


    // Matrix A next 4 rows
    kernel.add_instr(add(x4, x4, x25, 0, 0));

    // Matrix B next 1 columns
    kernel.add_instr(add(x5, x5, x27, 0, 0));
    
    // decrement m loop counter
    kernel.add_instr(sub(x6, x6, 1, 0));
    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_4_n_2_loop");
    kernel.add_instr(cbnz(x6, -l_mLoopInstrCount * 4));
}

void mini_jit::kernels::unary::internal::identityM4N1( mini_jit::Kernel &kernel,
                                                       int mLoopIterations )
{
    kernel.add_instr(mov(x6, mLoopIterations));
    kernel.add_label("m_4_n_1_loop");


    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 4x1 block of A (input matrix)
    kernel.add_instr(ldrPost(v0, x7, 4, s));
    kernel.add_instr(ldrPost(v1, x7, 4, s));
    kernel.add_instr(ldrPost(v2, x7, 4, s));
    kernel.add_instr(ldr(v3, x7, 0, s));

    // Store 4x1 Block of B
    kernel.add_instr(str(v0, x8, 0, s));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(str(v1, x8, 0, s));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(str(v2, x8, 0, s));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(str(v3, x8, 0, s));
    kernel.add_instr(add(x8, x8, x3, 0, 0));


    // Matrix A next 4 rows
    kernel.add_instr(add(x4, x4, x25, 0, 0));

    // Matrix B next 1 columns
    kernel.add_instr(add(x5, x5, x27, 0, 0));
    
    // decrement m loop counter
    kernel.add_instr(sub(x6, x6, 1, 0));
    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_4_n_1_loop");
    kernel.add_instr(cbnz(x6, -l_mLoopInstrCount * 4));
}

void mini_jit::kernels::unary::internal::identityM3N3( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 3x3 block of A (input matrix)
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v0, x17, 8, d));
    kernel.add_instr(ldr(v1, x17, 0, s));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v2, x17, 8, d));
    kernel.add_instr(ldr(v3, x17, 0, s));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v4, x17, 4, s));
    kernel.add_instr(ldrPost(v5, x17, 4, s));
    kernel.add_instr(ldr(v6, x17, 0, s));

    // Transpose 3x3 matrix
    // TRN
    kernel.add_instr(trn1(v7, v0, v2, s4));
    kernel.add_instr(trn2(v8, v0, v2, s4));

    // Store 3x3 Block of B
    kernel.add_instr(mov(x17, x8));

    kernel.add_instr(strPost(v7, x17, 8, d));
    kernel.add_instr(str(v4, x17, 0, s));
    kernel.add_instr(add(x8, x8, x3, 0, 0));
    kernel.add_instr(mov(x17, x8));

    kernel.add_instr(strPost(v8, x17, 8, d));
    kernel.add_instr(str(v5, x17, 0, s));
    kernel.add_instr(add(x8, x8, x3, 0, 0));
    kernel.add_instr(mov(x17, x8));

    kernel.add_instr(strPost(v1, x17, 4, s));
    kernel.add_instr(strPost(v3, x17, 4, s));
    kernel.add_instr(str(v6, x17, 0, s));
}

void mini_jit::kernels::unary::internal::identityM3N2( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 3x2 block of A (input matrix)
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v0, x17, 8, d));
    kernel.add_instr(ldr(v1, x17, 0, s));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v2, x17, 8, d));
    kernel.add_instr(ldr(v3, x17, 0, s));

    // Transpose 3x2 matrix
    // TRN
    kernel.add_instr(trn1(v4, v0, v2, s4));
    kernel.add_instr(trn2(v5, v0, v2, s4));

    // Store 3x2 Block of B
    kernel.add_instr(str(v4, x8, 0, d));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(str(v5, x8, 0, d));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(strPost(v1, x8, 4, s));
    kernel.add_instr(str(v3, x8, 0, s));
}

void mini_jit::kernels::unary::internal::identityM3N1( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 3x2 block of A (input matrix)
    kernel.add_instr(ldrPost(v0, x7, 4, s));
    kernel.add_instr(ldrPost(v1, x7, 4, s));
    kernel.add_instr(ldr(v2, x7, 0, s));

    // Store 3x2 Block of B
    kernel.add_instr(str(v0, x8, 0, s));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(str(v1, x8, 0, s));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(str(v2, x8, 0, s));
}

void mini_jit::kernels::unary::internal::identityM2N3( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 2x3 block of A (input matrix)
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldr(v0, x17, 0, d));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldr(v1, x17, 0, d));
    kernel.add_instr(add(x7, x7, x2, 0, 0));
    kernel.add_instr(mov(x17, x7));

    kernel.add_instr(ldrPost(v2, x17, 4, s));
    kernel.add_instr(ldr(v3, x17, 0, s));

    // Transpose 2x3 matrix
    // TRN
    kernel.add_instr(trn1(v4, v0, v1, s2));
    kernel.add_instr(trn2(v5, v0, v1, s2));

    // Store 2x3 Block of B
    kernel.add_instr(mov(x17, x8));

    kernel.add_instr(strPost(v4, x17, 8, d));
    kernel.add_instr(str(v2, x17, 0, s));
    kernel.add_instr(add(x8, x8, x3, 0, 0));
    kernel.add_instr(mov(x17, x8));

    kernel.add_instr(strPost(v5, x17, 8, d));
    kernel.add_instr(str(v3, x17, 0, s));
}

void mini_jit::kernels::unary::internal::identityM1N3( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 1x3 block of A (input matrix)
    kernel.add_instr(ldr(v0, x7, 0, s));
    kernel.add_instr(add(x7, x7, x2, 0, 0));

    kernel.add_instr(ldr(v1, x7, 0, s));
    kernel.add_instr(add(x7, x7, x2, 0, 0));

    kernel.add_instr(ldr(v2, x7, 0, s));

    // Store 1x3 Block of B
    kernel.add_instr(strPost(v0, x8, 4, s));
    kernel.add_instr(strPost(v1, x8, 4, s));
    kernel.add_instr(str(v2, x8, 0, s));
}

void mini_jit::kernels::unary::internal::identityM2N2( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 2x3 block of A (input matrix)
    kernel.add_instr(ldr(v0, x7, 0, d));
    kernel.add_instr(add(x7, x7, x2, 0, 0));

    kernel.add_instr(ldr(v1, x7, 0, d));

    // Transpose 2x3 matrix
    // TRN
    kernel.add_instr(trn1(v2, v0, v1, s2));
    kernel.add_instr(trn2(v3, v0, v1, s2));

    // Store 2x3 Block of B
    kernel.add_instr(str(v2, x8, 0, d));
    kernel.add_instr(add(x8, x8, x3, 0, 0));
    
    kernel.add_instr(str(v3, x8, 0, d));
}

void mini_jit::kernels::unary::internal::identityM2N1( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 2x1 block of A (input matrix)
    kernel.add_instr(ldrPost(v0, x7, 4, s));
    kernel.add_instr(ldr(v1, x7, 0, s));

    // Store 2x1 Block of B
    kernel.add_instr(str(v0, x8, 0, s));
    kernel.add_instr(add(x8, x8, x3, 0, 0));

    kernel.add_instr(strPost(v1, x8, 0, s));
}

void mini_jit::kernels::unary::internal::identityM1N2( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 1x2 block of A (input matrix)
    kernel.add_instr(ldr(v0, x7, 0, s));
    kernel.add_instr(add(x7, x7, x2, 0, 0));

    kernel.add_instr(ldr(v1, x7, 0, s));

    // Store 1x2 Block of B
    kernel.add_instr(strPost(v0, x8, 4, s));
    kernel.add_instr(str(v1, x8, 0, s));
}

void mini_jit::kernels::unary::internal::identityM1N1( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(mov(x7, x4));
    kernel.add_instr(mov(x8, x5));
    
    // Load 1x1 block of A (input matrix)
    kernel.add_instr(ldr(v0, x7, 0, s));

    // Store 1x1 Block of B
    kernel.add_instr(str(v0, x8, 0, s));
}
