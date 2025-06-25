#include "decrement_trans_primitive.h"
#include "Kernel.h"

#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
#include "instructions/all_instructions.h"

using simd_fp_t = mini_jit::registers::simd_fp_t;
using arr_spec_t = mini_jit::registers::arr_spec_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

using enum gpr_t;
using enum simd_fp_t;
using enum neon_size_spec_t;
using enum arr_spec_t;

namespace inst = mini_jit::instructions;
namespace base = inst::base;
namespace simd_fp = inst::simd_fp;
namespace internal_subkernels = mini_jit::kernels::unary::internal;

using base::stpPre;
using base::movSP;
using base::mov;
using base::lsl;
using base::add;
using base::sub;
using base::ldpPost;
using simd_fp::ldr;
using simd_fp::ldrPost;
using simd_fp::str;
using simd_fp::strPost;
using simd_fp::trn1;
using simd_fp::trn2;
using simd_fp::zip1;
using simd_fp::zip2;
using simd_fp::fmovVec;
using simd_fp::fmovScalar;
using simd_fp::fsubVec;
using simd_fp::fsubScalar;

void mini_jit::kernels::unary::decrement_trans(mini_jit::Kernel &kernel,
                                               int m,
                                               int n)
{
    // Prepare the kernel
    int mLoopIterations = m / 4;
    int mLoopRemainder = m % 4;

    int nLoopIterations = n / 4;
    int nLoopRemainder = n % 4;

    // PCS
    kernel.add_instr({
        stpPre(x29, x30, sp, -16),
        movSP(x29, sp),

        // Save callee-saved registers
        stpPre(x19, x20, sp, -16),
        stpPre(x21, x22, sp, -16),
        stpPre(x23, x24, sp, -16),
        stpPre(x25, x26, sp, -16),
        stpPre(x27, x28, sp, -16),

        simd_fp::stpPre(v8, v9, sp, -16, d),
        simd_fp::stpPre(v10, v11, sp, -16, d),
        simd_fp::stpPre(v12, v13, sp, -16, d),
        simd_fp::stpPre(v14, v15, sp, -16, d),

        // Save base matrix pointer
        mov(x4, x0), // A
        mov(x5, x1), // B

        // Compute stride for A and B
        lsl(x2, x2, 2),
        lsl(x3, x3, 2),

        // Set n loop counter
        mov(x9, nLoopIterations),

        // Row and column Pointer for A and B
        mov(x12, 0), // Columns of A
        mov(x13, 0), // Rows of B

        // Some constant values:
        // Jumping 4 rows in A | B      - (x25)
        mov(x25, 4*4), 

        // Jumping 4 columns in A       - (x26)
        lsl(x26, x2, 2),

        // Jumping 4 columns in B       - (x27)
        lsl(x27, x3, 2),

        // Set register with value 1
        fmovVec(v20, 1, s4), 
    });

    if ( nLoopIterations > 0)
    {
        // Start n loop (1 column)
        kernel.add_label("n_loop");
    
        if (mLoopIterations > 0)
        {
            internal_subkernels::decrementM4N4( kernel, mLoopIterations );
        }
    
        if (mLoopRemainder > 0)
        {
            switch (mLoopRemainder)
            {
            case 1:
                internal_subkernels::decrementM1N4( kernel );
                break;
            case 2:
                internal_subkernels::decrementM2N4( kernel );
                break;
            case 3:
                internal_subkernels::decrementM3N4( kernel );
                break;
            default:
                break;
            }
        }
        
        kernel.add_instr({
            // Restore positions
            mov(x4, x0),
            mov(x5, x1),
        
            // Update Columns of A
            add(x12, x12, x26, 0, 0),
        
            // Update Rows of B
            add(x13, x13, x25, 0, 0),
        
            // Apply the updates:
            add(x4, x4, x12, 0, 0),
            add(x5, x5, x13, 0, 0),
        
            // decrement n loop counter
            sub(x9, x9, 1, 0)
        });
        
        // check if loop counter is zero
        int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
        kernel.add_instr(base::cbnz(x9, -l_nLoopInstrCount * 4));
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
                    internal_subkernels::decrementM4N1( kernel, mLoopIterations );
                    break;
                case 2:
                    internal_subkernels::decrementM4N2( kernel, mLoopIterations );
                    break;
                case 3:
                    internal_subkernels::decrementM4N3( kernel, mLoopIterations );
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
                    internal_subkernels::decrementM1N1( kernel );
                }
                else if ( nLoopRemainder == 2 )
                {
                    internal_subkernels::decrementM1N2( kernel );
                }
                else if ( nLoopRemainder == 3 )
                {
                    internal_subkernels::decrementM1N3( kernel );
                }
                break;
            case 2:
                if ( nLoopRemainder == 1 )
                {
                    internal_subkernels::decrementM2N1( kernel );
                }
                else if ( nLoopRemainder == 2 )
                {
                    internal_subkernels::decrementM2N2( kernel );
                }
                else if ( nLoopRemainder == 3 )
                {
                    internal_subkernels::decrementM2N3( kernel );
                }
                break;
            case 3:
                if ( nLoopRemainder == 1 )
                {
                    internal_subkernels::decrementM3N1( kernel );
                }
                else if ( nLoopRemainder == 2 )
                {
                    internal_subkernels::decrementM3N2( kernel );
                }
                else if ( nLoopRemainder == 3 )
                {
                    internal_subkernels::decrementM3N3( kernel );
                }
                break;
            default:
                break;
            }
        }
    }

    kernel.add_instr({
        // Restore callee-saved registers
        simd_fp::ldpPost(v14, v15, sp, 16, d),
        simd_fp::ldpPost(v12, v13, sp, 16, d),
        simd_fp::ldpPost(v10, v11, sp, 16, d),
        simd_fp::ldpPost(v8, v9, sp, 16, d),

        ldpPost(x27, x28, sp, 16),
        ldpPost(x25, x26, sp, 16),
        ldpPost(x23, x24, sp, 16),
        ldpPost(x21, x22, sp, 16),
        ldpPost(x19, x20, sp, 16),

        // Restore stack pointer
        ldpPost(x29, x30, sp, 16),

        inst::ret()
    });

    kernel.write("decrement_trans_primitive.bin");
    kernel.set_kernel();
}

void mini_jit::kernels::unary::internal::decrementM4N4( mini_jit::Kernel &kernel,
                                                     int mLoopIterations )
{
    kernel.add_instr(mov(x6, mLoopIterations));

    kernel.add_label("m_4_n_4_loop");

    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 4x4 block of A (input matrix)
        ldr(v0, x7, 0, q),
        add(x7, x7, x2, 0, 0),
        ldr(v1, x7, 0, q),
        add(x7, x7, x2, 0, 0),
        ldr(v2, x7, 0, q),
        add(x7, x7, x2, 0, 0),
        ldr(v3, x7, 0, q),

        // Transpose 4x4 block
        // TRN
        trn1(v4, v0, v2, s4),
        trn1(v5, v1, v3, s4),
        trn2(v6, v0, v2, s4),
        trn2(v7, v1, v3, s4),

        // ZIP
        zip1(v8, v4, v5, s4),
        zip1(v9, v6, v7, s4),

        zip2(v10, v4, v5, s4),
        zip2(v11, v6, v7, s4),

        // Increment values
        fsubVec( v8,  v8, v20, s4),
        fsubVec( v9,  v9, v20, s4),
        fsubVec(v10, v10, v20, s4),
        fsubVec(v11, v11, v20, s4),

        // Store 4x4 Block of B
        str(v8, x8, 0, q),
        add(x8, x8, x3, 0, 0),
        str(v9, x8, 0, q),
        add(x8, x8, x3, 0, 0),
        str(v10, x8, 0, q),
        add(x8, x8, x3, 0, 0),
        str(v11, x8, 0, q),

        // Matrix A next 4 rows
        add(x4, x4, x25, 0, 0),

        // Matrix B next 4 columns
        add(x5, x5, x27, 0, 0),
        
        // decrement m loop counter
        sub(x6, x6, 1, 0)
    });

    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_4_n_4_loop");
    kernel.add_instr(base::cbnz(x6, -l_mLoopInstrCount * 4));
}

void mini_jit::kernels::unary::internal::decrementM3N4( mini_jit::Kernel &kernel )
{
    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 3x4 block of A (input matrix)
        mov(x17, x7),

        ldrPost(v0, x17, 8, d),
        ldr(v1, x17, 0, s),
        add(x7, x7, x2, 0, 0),
        mov(x17, x7),

        ldrPost(v2, x17, 8, d),
        ldr(v3, x17, 0, s),
        add(x7, x7, x2, 0, 0),
        mov(x17, x7),

        ldrPost(v4, x17, 8, d),
        ldr(v5, x17, 0, s),
        add(x7, x7, x2, 0, 0),
        mov(x17, x7),

        ldrPost(v6, x17, 8, d),
        ldr(v7, x17, 0, s),

        // Transpose 3x4 block
        // TRN (d)
        trn1(v8, v0, v4, s4),
        trn1(v9, v2, v6, s4),
        trn2(v10, v0, v4, s4),
        trn2(v11, v2, v6, s4),

        // TRN (s)
        trn1(v12, v1, v5, s2),
        trn1(v13, v3, v7, s2),

        // ZIP
        zip1(v14, v8, v9, s4),
        zip1(v15, v10, v11, s4),

        zip1(v18, v12, v13, s2),
        zip2(v19, v12, v13, s2),

        // Increment values
        fsubVec(v14, v14, v20, s4),
        fsubVec(v15, v15, v20, s4),
        fsubVec(v18, v18, v20, s2),
        fsubVec(v19, v19, v20, s2),

        // Store 3x4 Block of B
        str(v14, x8, 0, q),
        add(x8, x8, x3, 0, 0),

        str(v15, x8, 0, q),
        add(x8, x8, x3, 0, 0),

        strPost(v18, x8, 8, d),
        str(v19, x8, 0, d)
    });
}

void mini_jit::kernels::unary::internal::decrementM2N4( mini_jit::Kernel &kernel )
{
    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 2x4 block of A (input matrix)
        ldr(v0, x7, 0, d),
        add(x7, x7, x2, 0, 0),

        ldr(v1, x7, 0, d),
        add(x7, x7, x2, 0, 0),

        ldr(v2, x7, 0, d),
        add(x7, x7, x2, 0, 0),

        ldr(v3, x7, 0, d),

        // Transpose 2x4 block
        // TRN
        trn1(v4, v0, v2, s4),
        trn1(v5, v1, v3, s4),

        trn2(v6, v0, v2, s4),
        trn2(v7, v1, v3, s4),

        // ZIP
        zip1(v8, v4, v5, s4),
        zip1(v9, v6, v7, s4),

        // Increment values
        fsubVec(v8, v8, v20, s4),
        fsubVec(v9, v9, v20, s4),

        // Store 2x4 Block of B
        str(v8, x8, 0, q),
        add(x8, x8, x3, 0, 0),

        str(v9, x8, 0, q)
    });
}

void mini_jit::kernels::unary::internal::decrementM1N4( mini_jit::Kernel &kernel )
{
    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 1x4 block of A (input matrix)
        ldr(v0, x7, 0, s),
        add(x7, x7, x2, 0, 0),

        ldr(v1, x7, 0, s),
        add(x7, x7, x2, 0, 0),

        ldr(v2, x7, 0, s),
        add(x7, x7, x2, 0, 0),

        ldr(v3, x7, 0, s),

        // Transpose 1x4 block
        // TRN
        trn1(v4, v0, v2, s2),
        trn1(v5, v1, v3, s2),

        // ZIP
        zip1(v6, v4, v5, s4),

        // Increment values
        fsubVec(v6, v6, v20, s4),

        // Store 1x4 Block of B
        str(v6, x8, 0, q)
    });
}

void mini_jit::kernels::unary::internal::decrementM4N3( mini_jit::Kernel &kernel,
                                                       int mLoopIterations )
{
    kernel.add_instr(mov(x6, mLoopIterations));
    kernel.add_label("m_4_n_3_loop");

    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 4x3 block of A (input matrix)
        mov(x17, x7),

        ldrPost(v0, x17, 8, d),
        ldr(v1, x17, 0, d),
        add(x7, x7, x2, 0, 0),
        mov(x17, x7),

        ldrPost(v2, x17, 8, d),
        ldr(v3, x17, 0, d),
        add(x7, x7, x2, 0, 0),
        mov(x17, x7),

        ldrPost(v4, x17, 4, s),
        ldrPost(v5, x17, 4, s),
        ldrPost(v6, x17, 4, s),
        ldr(v7, x17, 0, s),

        // Transpose 4x3 matrix
        // TRN
        trn1(v8, v0, v2, s4),
        trn2(v9, v0, v2, s4),
        trn1(v10, v1, v3, s4),
        trn2(v11, v1, v3, s4),

        // Increment values
        fsubVec(v8, v8, v20, s2),
        fsubScalar(v4, v4, v20, s),
        fsubVec(v9, v9, v20, s2),
        fsubScalar(v5, v5, v20, s),
        fsubVec(v10, v10, v20, s2),
        fsubScalar(v6, v6, v20, s),
        fsubVec(v11, v11, v20, s2),
        fsubScalar(v7, v7, v20, s),

        // Store 4x3 Block of B
        mov(x17, x8),

        strPost(v8, x17, 8, d),
        str(v4, x17, 0, s),
        add(x8, x8, x3, 0, 0),
        mov(x17, x8),

        strPost(v9, x17, 8, d),
        str(v5, x17, 0, s),
        add(x8, x8, x3, 0, 0),
        mov(x17, x8),

        strPost(v10, x17, 8, d),
        str(v6, x17, 0, s),
        add(x8, x8, x3, 0, 0),
        mov(x17, x8),

        strPost(v11, x17, 8, d),
        str(v7, x17, 0, s),

        // Matrix A next 4 rows
        add(x4, x4, x25, 0, 0),

        // Matrix B next 1 columns
        add(x5, x5, x27, 0, 0),
        
        // decrement m loop counter
        sub(x6, x6, 1, 0)
    });
    
    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_4_n_3_loop");
    kernel.add_instr(base::cbnz(x6, -l_mLoopInstrCount * 4));
}

void mini_jit::kernels::unary::internal::decrementM4N2( mini_jit::Kernel &kernel,
                                                       int mLoopIterations )
{
    kernel.add_instr(mov(x6, mLoopIterations));
    kernel.add_label("m_4_n_2_loop");

    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 4x2 block of A (input matrix)
        mov(x17, x7),
        ldrPost(v0, x17, 8, d),
        ldr(v1, x17, 0, d),
        add(x7, x7, x2, 0, 0),
        mov(x17, x7),

        ldrPost(v2, x17, 8, d),
        ldr(v3, x17, 0, d),

        // Transpose 4x2 matrix
        // TRN
        trn1(v4, v0, v2, s4),
        trn2(v5, v0, v2, s4),

        trn1(v6, v1, v3, s4),
        trn2(v7, v1, v3, s4),

        // Increment values
        fsubVec(v4, v4, v20, s2),
        fsubVec(v5, v5, v20, s2),
        fsubVec(v6, v6, v20, s2),
        fsubVec(v7, v7, v20, s2),

        // Store 4x2 Block of B
        str(v4, x8, 0, d),
        add(x8, x8, x3, 0, 0),

        str(v5, x8, 0, d),
        add(x8, x8, x3, 0, 0),

        str(v6, x8, 0, d),
        add(x8, x8, x3, 0, 0),

        str(v7, x8, 0, d),

        // Matrix A next 4 rows
        add(x4, x4, x25, 0, 0),

        // Matrix B next 1 columns
        add(x5, x5, x27, 0, 0),
        
        // decrement m loop counter
        sub(x6, x6, 1, 0)
    });

    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_4_n_2_loop");
    kernel.add_instr(base::cbnz(x6, -l_mLoopInstrCount * 4));
}

void mini_jit::kernels::unary::internal::decrementM4N1( mini_jit::Kernel &kernel,
                                                       int mLoopIterations )
{
    kernel.add_instr(mov(x6, mLoopIterations));
    kernel.add_label("m_4_n_1_loop");

    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 4x1 block of A (input matrix)
        ldrPost(v0, x7, 4, s),
        ldrPost(v1, x7, 4, s),
        ldrPost(v2, x7, 4, s),
        ldr(v3, x7, 0, s),

        // Increment values
        fsubScalar(v0, v0, v20, s),
        fsubScalar(v1, v1, v20, s),
        fsubScalar(v2, v2, v20, s),
        fsubScalar(v3, v3, v20, s),

        // Store 4x1 Block of B
        str(v0, x8, 0, s),
        add(x8, x8, x3, 0, 0),

        str(v1, x8, 0, s),
        add(x8, x8, x3, 0, 0),

        str(v2, x8, 0, s),
        add(x8, x8, x3, 0, 0),

        str(v3, x8, 0, s),
        add(x8, x8, x3, 0, 0),

        // Matrix A next 4 rows
        add(x4, x4, x25, 0, 0),

        // Matrix B next 1 columns
        add(x5, x5, x27, 0, 0),
        
        // decrement m loop counter
        sub(x6, x6, 1, 0)
    });

    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_4_n_1_loop");
    kernel.add_instr(base::cbnz(x6, -l_mLoopInstrCount * 4));
}

void mini_jit::kernels::unary::internal::decrementM3N3( mini_jit::Kernel &kernel )
{
    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 3x3 block of A (input matrix)
        mov(x17, x7),

        ldrPost(v0, x17, 8, d),
        ldr(v1, x17, 0, s),
        add(x7, x7, x2, 0, 0),
        mov(x17, x7),

        ldrPost(v2, x17, 8, d),
        ldr(v3, x17, 0, s),
        add(x7, x7, x2, 0, 0),
        mov(x17, x7),

        ldrPost(v4, x17, 4, s),
        ldrPost(v5, x17, 4, s),
        ldr(v6, x17, 0, s),

        // Transpose 3x3 matrix
        // TRN
        trn1(v7, v0, v2, s4),
        trn2(v8, v0, v2, s4),

        // Increment values
        fsubVec(v7, v7, v20, s2),
        fsubScalar(v4, v4, v20, s),
        fsubVec(v8, v8, v20, s2),
        fsubScalar(v5, v5, v20, s),
        fsubScalar(v1, v1, v20, s),
        fsubScalar(v3, v3, v20, s),
        fsubScalar(v6, v6, v20, s),

        // Store 3x3 Block of B
        mov(x17, x8),

        strPost(v7, x17, 8, d),
        str(v4, x17, 0, s),
        add(x8, x8, x3, 0, 0),
        mov(x17, x8),

        strPost(v8, x17, 8, d),
        str(v5, x17, 0, s),
        add(x8, x8, x3, 0, 0),
        mov(x17, x8),

        strPost(v1, x17, 4, s),
        strPost(v3, x17, 4, s),
        str(v6, x17, 0, s)
    });
}

void mini_jit::kernels::unary::internal::decrementM3N2( mini_jit::Kernel &kernel )
{
    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 3x2 block of A (input matrix)
        mov(x17, x7),

        ldrPost(v0, x17, 8, d),
        ldr(v1, x17, 0, s),
        add(x7, x7, x2, 0, 0),
        mov(x17, x7),

        ldrPost(v2, x17, 8, d),
        ldr(v3, x17, 0, s),

        // Transpose 3x2 matrix
        // TRN
        trn1(v4, v0, v2, s4),
        trn2(v5, v0, v2, s4),

        // Increment values
        fsubVec(v4, v4, v20, s2),
        fsubVec(v5, v5, v20, s2),
        fsubScalar(v1, v1, v20, s),
        fsubScalar(v3, v3, v20, s),

        // Store 3x2 Block of B
        str(v4, x8, 0, d),
        add(x8, x8, x3, 0, 0),

        str(v5, x8, 0, d),
        add(x8, x8, x3, 0, 0),

        strPost(v1, x8, 4, s),
        str(v3, x8, 0, s)
    });
}

void mini_jit::kernels::unary::internal::decrementM3N1( mini_jit::Kernel &kernel )
{
    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 3x1 block of A (input matrix)
        ldrPost(v0, x7, 4, s),
        ldrPost(v1, x7, 4, s),
        ldr(v2, x7, 0, s),

        // Increment values
        fsubScalar(v0, v0, v20, s),
        fsubScalar(v1, v1, v20, s),
        fsubScalar(v2, v2, v20, s),

        // Store 3x2 Block of B
        str(v0, x8, 0, s),
        add(x8, x8, x3, 0, 0),

        str(v1, x8, 0, s),
        add(x8, x8, x3, 0, 0),

        str(v2, x8, 0, s)
    });
}

void mini_jit::kernels::unary::internal::decrementM2N3( mini_jit::Kernel &kernel )
{
    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 2x3 block of A (input matrix)
        mov(x17, x7),

        ldr(v0, x17, 0, d),
        add(x7, x7, x2, 0, 0),
        mov(x17, x7),

        ldr(v1, x17, 0, d),
        add(x7, x7, x2, 0, 0),
        mov(x17, x7),

        ldrPost(v2, x17, 4, s),
        ldr(v3, x17, 0, s),

        // Transpose 2x3 matrix
        // TRN
        trn1(v4, v0, v1, s2),
        trn2(v5, v0, v1, s2),

        // Increment values
        fsubVec(v4, v4, v20, s2),
        fsubScalar(v2, v2, v20, s),
        fsubVec(v5, v5, v20, s2),
        fsubScalar(v3, v3, v20, s),

        // Store 2x3 Block of B
        mov(x17, x8),

        strPost(v4, x17, 8, d),
        str(v2, x17, 0, s),
        add(x8, x8, x3, 0, 0),
        mov(x17, x8),

        strPost(v5, x17, 8, d),
        str(v3, x17, 0, s)
    });
}

void mini_jit::kernels::unary::internal::decrementM1N3( mini_jit::Kernel &kernel )
{
    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 1x3 block of A (input matrix)
        ldr(v0, x7, 0, s),
        add(x7, x7, x2, 0, 0),

        ldr(v1, x7, 0, s),
        add(x7, x7, x2, 0, 0),

        ldr(v2, x7, 0, s),

        // Increment values
        fsubScalar(v0, v0, v20, s),
        fsubScalar(v1, v1, v20, s),
        fsubScalar(v2, v2, v20, s),

        // Store 1x3 Block of B
        strPost(v0, x8, 4, s),
        strPost(v1, x8, 4, s),
        str(v2, x8, 0, s)
    });
}

void mini_jit::kernels::unary::internal::decrementM2N2( mini_jit::Kernel &kernel )
{
    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 2x3 block of A (input matrix)
        ldr(v0, x7, 0, d),
        add(x7, x7, x2, 0, 0),

        ldr(v1, x7, 0, d),

        // Transpose 2x3 matrix
        // TRN
        trn1(v2, v0, v1, s2),
        trn2(v3, v0, v1, s2),

        // Increment values
        fsubVec(v2, v2, v20, s2),
        fsubVec(v3, v3, v20, s2),

        // Store 2x3 Block of B
        str(v2, x8, 0, d),
        add(x8, x8, x3, 0, 0),
        
        str(v3, x8, 0, d)
    });
}

void mini_jit::kernels::unary::internal::decrementM2N1( mini_jit::Kernel &kernel )
{
    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 2x1 block of A (input matrix)
        ldrPost(v0, x7, 4, s),
        ldr(v1, x7, 0, s),

        // Increment values
        fsubScalar(v0, v0, v20, s),
        fsubScalar(v1, v1, v20, s),

        // Store 2x1 Block of B
        str(v0, x8, 0, s),
        add(x8, x8, x3, 0, 0),

        strPost(v1, x8, 0, s)
    });
}

void mini_jit::kernels::unary::internal::decrementM1N2( mini_jit::Kernel &kernel )
{
    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 1x2 block of A (input matrix)
        ldr(v0, x7, 0, s),
        add(x7, x7, x2, 0, 0),

        ldr(v1, x7, 0, s),

        // Increment values
        fsubScalar(v0, v0, v20, s),
        fsubScalar(v1, v1, v20, s),

        // Store 1x2 Block of B
        strPost(v0, x8, 4, s),
        str(v1, x8, 0, s)
    });
}

void mini_jit::kernels::unary::internal::decrementM1N1( mini_jit::Kernel &kernel )
{
    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 1x1 block of A (input matrix)
        ldr(v0, x7, 0, s),

        // Increment values
        fsubScalar(v0, v0, v20, s),

        // Store 1x1 Block of B
        str(v0, x8, 0, s)
    });
}
