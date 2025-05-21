#include "identity_primitive.h"
#include "Kernel.h"

#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
#include "instructions/all_instructions.h"

using simd_fp_t = mini_jit::registers::simd_fp_t;
using arr_spec_t = mini_jit::registers::arr_spec_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;

namespace inst = mini_jit::instructions;
namespace base = inst::base;
namespace simd_fp = inst::simd_fp;
namespace internal_subkernels = mini_jit::kernels::unary::internal;

void mini_jit::kernels::unary::identity(mini_jit::Kernel &kernel,
                                        int m,
                                        int n)
{
    // Prepare the kernel
    int mLoopIterations = m / 4;
    int mLoopRemainder = m % 4;

    int nLoopIterations = n / 4;
    int nLoopRemainder = n % 4;

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

    // Save base matrix pointer
    kernel.add_instr(base::mov(gpr_t::x4, gpr_t::x0)); // A
    kernel.add_instr(base::mov(gpr_t::x5, gpr_t::x1)); // B

    // Compute stride for A and B
    kernel.add_instr(base::lsl(gpr_t::x2, gpr_t::x2, 2));
    kernel.add_instr(base::lsl(gpr_t::x3, gpr_t::x3, 2));

    // Set n loop counter
    kernel.add_instr(base::mov(gpr_t::x9, nLoopIterations));

    // Row and column Pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x12, 0)); // Columns of A
    kernel.add_instr(base::mov(gpr_t::x13, 0)); // Rows of B

    // Some constand values:
    // Jumping 4 rows in A | B      - (x25)
    kernel.add_instr(base::mov(gpr_t::x25, 4*4)); 

    // Jumping 4 columns in A       - (x26)
    kernel.add_instr(base::mov(gpr_t::x23, 4));
    kernel.add_instr(base::mul(gpr_t::x23, gpr_t::x23, gpr_t::x2));
    kernel.add_instr(base::mov(gpr_t::x26, gpr_t::x23)); 

    // Jumping 4 columns in B       - (x27)
    kernel.add_instr(base::mov(gpr_t::x24, 4));
    kernel.add_instr(base::mul(gpr_t::x24, gpr_t::x24, gpr_t::x3));
    kernel.add_instr(base::mov(gpr_t::x27, gpr_t::x24));


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
        kernel.add_instr(base::mov(gpr_t::x4, gpr_t::x0));
        kernel.add_instr(base::mov(gpr_t::x5, gpr_t::x1));
    
        // Update Columns of A
        kernel.add_instr(base::add(gpr_t::x12, gpr_t::x12, gpr_t::x26, 0, 0));
    
        // Update Rows of B
        kernel.add_instr(base::add(gpr_t::x13, gpr_t::x13, gpr_t::x25, 0, 0));
    
        // Apply the updates:
        kernel.add_instr(base::add(gpr_t::x4, gpr_t::x4, gpr_t::x12, 0, 0));
        kernel.add_instr(base::add(gpr_t::x5, gpr_t::x5, gpr_t::x13, 0, 0));
    
        // decrement n loop counter
        kernel.add_instr(base::sub(gpr_t::x9, gpr_t::x9, 1, 0));
        // check if loop counter is zero
        int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
        kernel.add_instr(base::cbnz(gpr_t::x9, -l_nLoopInstrCount * 4));
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
    kernel.write("identity_primitive.bin");
    kernel.set_kernel();
}

void mini_jit::kernels::unary::internal::identityM4N4( mini_jit::Kernel &kernel,
                                                       int mLoopIterations )
{
    kernel.add_instr(base::mov(gpr_t::x6, mLoopIterations));

    kernel.add_label("m_4_n_4_loop");

    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 4x4 block of A (input matrix)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x7, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x7, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x7, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x7, 0, neon_size_spec_t::q));

    // Transpose 4x4 block
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v5, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v6, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v7, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));

    // ZIP
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v8, simd_fp_t::v4, simd_fp_t::v5, arr_spec_t::s4));
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v9, simd_fp_t::v6, simd_fp_t::v7, arr_spec_t::s4));

    kernel.add_instr(simd_fp::zip2(simd_fp_t::v10, simd_fp_t::v4, simd_fp_t::v5, arr_spec_t::s4));
    kernel.add_instr(simd_fp::zip2(simd_fp_t::v11, simd_fp_t::v6, simd_fp_t::v7, arr_spec_t::s4));

    // Store 4x4 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v8, gpr_t::x8, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v9, gpr_t::x8, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v10, gpr_t::x8, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v11, gpr_t::x8, 0, neon_size_spec_t::q));

    // Matrix A next 4 rows
    kernel.add_instr(base::add(gpr_t::x4, gpr_t::x4, gpr_t::x25, 0, 0));

    // Matrix B next 4 columns
    kernel.add_instr(base::add(gpr_t::x5, gpr_t::x5, gpr_t::x27, 0, 0));
    
    // decrement m loop counter
    kernel.add_instr(base::sub(gpr_t::x6, gpr_t::x6, 1, 0));
    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_4_n_4_loop");
    kernel.add_instr(base::cbnz(gpr_t::x6, -l_mLoopInstrCount * 4));
}

void mini_jit::kernels::unary::internal::identityM3N4( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 3x4 block of A (input matrix)
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x17, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v2, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x17, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v4, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v5, gpr_t::x17, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v6, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v7, gpr_t::x17, 0, neon_size_spec_t::s));

    // Transpose 3x4 block
    // TRN (d)
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v8, simd_fp_t::v0, simd_fp_t::v4, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v9, simd_fp_t::v2, simd_fp_t::v6, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v10, simd_fp_t::v0, simd_fp_t::v4, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v11, simd_fp_t::v2, simd_fp_t::v6, arr_spec_t::s4));

    // TRN (s)
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v12, simd_fp_t::v1, simd_fp_t::v5, arr_spec_t::s2));
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v13, simd_fp_t::v3, simd_fp_t::v7, arr_spec_t::s2));

    // ZIP
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v14, simd_fp_t::v8, simd_fp_t::v9, arr_spec_t::s4));
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v15, simd_fp_t::v10, simd_fp_t::v11, arr_spec_t::s4));

    kernel.add_instr(simd_fp::zip1(simd_fp_t::v18, simd_fp_t::v12, simd_fp_t::v13, arr_spec_t::s2));
    kernel.add_instr(simd_fp::zip2(simd_fp_t::v19, simd_fp_t::v12, simd_fp_t::v13, arr_spec_t::s2));

    // Store 3x4 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v14, gpr_t::x8, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v15, gpr_t::x8, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::strPost(simd_fp_t::v18, gpr_t::x8, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v19, gpr_t::x8, 0, neon_size_spec_t::q));
}

void mini_jit::kernels::unary::internal::identityM2N4( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 2x4 block of A (input matrix)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x7, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));

    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x7, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));

    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x7, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));

    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x7, 0, neon_size_spec_t::d));

    // Transpose 2x4 block
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v5, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));

    kernel.add_instr(simd_fp::trn2(simd_fp_t::v6, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v7, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));

    // ZIP
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v8, simd_fp_t::v4, simd_fp_t::v5, arr_spec_t::s4));
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v9, simd_fp_t::v6, simd_fp_t::v7, arr_spec_t::s4));

    // Store 2x4 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v8, gpr_t::x8, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v9, gpr_t::x8, 0, neon_size_spec_t::q));
}

void mini_jit::kernels::unary::internal::identityM1N4( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 1x4 block of A (input matrix)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x7, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));

    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x7, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));

    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x7, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));

    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x7, 0, neon_size_spec_t::s));

    // Transpose 1x4 block
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s2));
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v5, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s2));

    // ZIP
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v6, simd_fp_t::v4, simd_fp_t::v5, arr_spec_t::s4));

    // Store 1x4 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v6, gpr_t::x8, 0, neon_size_spec_t::q));
}

void mini_jit::kernels::unary::internal::identityM4N3( mini_jit::Kernel &kernel,
                                                       int mLoopIterations )
{
    kernel.add_instr(base::mov(gpr_t::x6, mLoopIterations));
    kernel.add_label("m_4_n_3_loop");


    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 4x3 block of A (input matrix)
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x17, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v2, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x17, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v4, gpr_t::x17, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v5, gpr_t::x17, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v6, gpr_t::x17, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v7, gpr_t::x17, 0, neon_size_spec_t::s));

    // Transpose 4x3 matrix
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v8, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v9, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v10, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v11, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));


    // Store 4x3 Block of B
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x8));

    kernel.add_instr(simd_fp::strPost(simd_fp_t::v8, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v4, gpr_t::x17, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x8));

    kernel.add_instr(simd_fp::strPost(simd_fp_t::v9, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v5, gpr_t::x17, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x8));

    kernel.add_instr(simd_fp::strPost(simd_fp_t::v10, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v6, gpr_t::x17, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x8));

    kernel.add_instr(simd_fp::strPost(simd_fp_t::v11, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v7, gpr_t::x17, 0, neon_size_spec_t::s));


    // Matrix A next 4 rows
    kernel.add_instr(base::add(gpr_t::x4, gpr_t::x4, gpr_t::x25, 0, 0));

    // Matrix B next 1 columns
    kernel.add_instr(base::add(gpr_t::x5, gpr_t::x5, gpr_t::x27, 0, 0));
    
    // decrement m loop counter
    kernel.add_instr(base::sub(gpr_t::x6, gpr_t::x6, 1, 0));
    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_4_n_3_loop");
    kernel.add_instr(base::cbnz(gpr_t::x6, -l_mLoopInstrCount * 4));
}

void mini_jit::kernels::unary::internal::identityM4N2( mini_jit::Kernel &kernel,
                                                       int mLoopIterations )
{
    kernel.add_instr(base::mov(gpr_t::x6, mLoopIterations));
    kernel.add_label("m_4_n_2_loop");


    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 4x2 block of A (input matrix)
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x17, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v2, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x17, 0, neon_size_spec_t::d));

    // Transpose 4x2 matrix
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v5, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));

    kernel.add_instr(simd_fp::trn1(simd_fp_t::v6, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v7, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));

    // Store 4x2 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v4, gpr_t::x8, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v5, gpr_t::x8, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v6, gpr_t::x8, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v7, gpr_t::x8, 0, neon_size_spec_t::d));


    // Matrix A next 4 rows
    kernel.add_instr(base::add(gpr_t::x4, gpr_t::x4, gpr_t::x25, 0, 0));

    // Matrix B next 1 columns
    kernel.add_instr(base::add(gpr_t::x5, gpr_t::x5, gpr_t::x27, 0, 0));
    
    // decrement m loop counter
    kernel.add_instr(base::sub(gpr_t::x6, gpr_t::x6, 1, 0));
    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_4_n_2_loop");
    kernel.add_instr(base::cbnz(gpr_t::x6, -l_mLoopInstrCount * 4));
}

void mini_jit::kernels::unary::internal::identityM4N1( mini_jit::Kernel &kernel,
                                                       int mLoopIterations )
{
    kernel.add_instr(base::mov(gpr_t::x6, mLoopIterations));
    kernel.add_label("m_4_n_1_loop");


    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 4x1 block of A (input matrix)
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x7, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v1, gpr_t::x7, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v2, gpr_t::x7, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x7, 0, neon_size_spec_t::s));

    // Store 4x1 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x8, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x8, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x8, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x8, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));


    // Matrix A next 4 rows
    kernel.add_instr(base::add(gpr_t::x4, gpr_t::x4, gpr_t::x25, 0, 0));

    // Matrix B next 1 columns
    kernel.add_instr(base::add(gpr_t::x5, gpr_t::x5, gpr_t::x27, 0, 0));
    
    // decrement m loop counter
    kernel.add_instr(base::sub(gpr_t::x6, gpr_t::x6, 1, 0));
    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_4_n_1_loop");
    kernel.add_instr(base::cbnz(gpr_t::x6, -l_mLoopInstrCount * 4));
}

void mini_jit::kernels::unary::internal::identityM3N3( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 3x3 block of A (input matrix)
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x17, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v2, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x17, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v4, gpr_t::x17, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v5, gpr_t::x17, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v6, gpr_t::x17, 0, neon_size_spec_t::s));

    // Transpose 3x3 matrix
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v7, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v8, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));

    // Store 3x3 Block of B
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x8));

    kernel.add_instr(simd_fp::strPost(simd_fp_t::v7, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v4, gpr_t::x17, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x8));

    kernel.add_instr(simd_fp::strPost(simd_fp_t::v8, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v5, gpr_t::x17, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x8));

    kernel.add_instr(simd_fp::strPost(simd_fp_t::v1, gpr_t::x17, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v3, gpr_t::x17, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::str(simd_fp_t::v6, gpr_t::x17, 0, neon_size_spec_t::s));
}

void mini_jit::kernels::unary::internal::identityM3N2( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 3x2 block of A (input matrix)
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x17, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v2, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x17, 0, neon_size_spec_t::s));

    // Transpose 3x2 matrix
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v5, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));

    // Store 3x2 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v4, gpr_t::x8, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v5, gpr_t::x8, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::strPost(simd_fp_t::v1, gpr_t::x8, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x8, 0, neon_size_spec_t::s));
}

void mini_jit::kernels::unary::internal::identityM3N1( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 3x2 block of A (input matrix)
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x7, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v1, gpr_t::x7, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x7, 0, neon_size_spec_t::s));

    // Store 3x2 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x8, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x8, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x8, 0, neon_size_spec_t::s));
}

void mini_jit::kernels::unary::internal::identityM2N3( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 2x3 block of A (input matrix)
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x17, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x17, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v2, gpr_t::x17, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x17, 0, neon_size_spec_t::s));

    // Transpose 2x3 matrix
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v1, arr_spec_t::s2));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v5, simd_fp_t::v0, simd_fp_t::v1, arr_spec_t::s2));

    // Store 2x3 Block of B
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x8));

    kernel.add_instr(simd_fp::strPost(simd_fp_t::v4, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x17, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x8));

    kernel.add_instr(simd_fp::strPost(simd_fp_t::v5, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x17, 0, neon_size_spec_t::s));
}

void mini_jit::kernels::unary::internal::identityM1N3( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 1x3 block of A (input matrix)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x7, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));

    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x7, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));

    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x7, 0, neon_size_spec_t::s));

    // Store 1x3 Block of B
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v0, gpr_t::x8, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v1, gpr_t::x8, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x8, 0, neon_size_spec_t::s));
}

void mini_jit::kernels::unary::internal::identityM2N2( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 2x3 block of A (input matrix)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x7, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));

    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x7, 0, neon_size_spec_t::d));

    // Transpose 2x3 matrix
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v2, simd_fp_t::v0, simd_fp_t::v1, arr_spec_t::s2));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v3, simd_fp_t::v0, simd_fp_t::v1, arr_spec_t::s2));

    // Store 2x3 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v2, gpr_t::x8, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));
    
    kernel.add_instr(simd_fp::str(simd_fp_t::v3, gpr_t::x8, 0, neon_size_spec_t::d));
}

void mini_jit::kernels::unary::internal::identityM2N1( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 2x1 block of A (input matrix)
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x7, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x7, 0, neon_size_spec_t::s));

    // Store 2x1 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x8, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::strPost(simd_fp_t::v1, gpr_t::x8, 0, neon_size_spec_t::s));
}

void mini_jit::kernels::unary::internal::identityM1N2( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 1x2 block of A (input matrix)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x7, 0, neon_size_spec_t::s));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));

    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x7, 0, neon_size_spec_t::s));

    // Store 1x2 Block of B
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v0, gpr_t::x8, 4, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x8, 0, neon_size_spec_t::s));
}

void mini_jit::kernels::unary::internal::identityM1N1( mini_jit::Kernel &kernel )
{
    // working pointer for A and B
    kernel.add_instr(base::mov(gpr_t::x7, gpr_t::x4));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x5));
    
    // Load 1x1 block of A (input matrix)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x7, 0, neon_size_spec_t::s));

    // Store 1x1 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v0, gpr_t::x8, 0, neon_size_spec_t::s));
}
