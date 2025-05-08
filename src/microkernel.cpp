#include "InstGen.h"
#include "Brgemm.h"
#include "Kernel.h"
#include <iostream>
#include <cstring>

using gpr_t = mini_jit::InstGen::gpr_t;
using simd_fp_t = mini_jit::InstGen::simd_fp_t;
using arr_spec_t = mini_jit::InstGen::arr_spec_t;

using dtype_t = mini_jit::Brgemm::dtype_t;


void matmul_16_6_1( float const * A,
                    float const * B,
                    float * C,
                    int64_t M,
                    int64_t N,
                    int64_t K,
                    int64_t br_stride_a,
                    int64_t br_stride_b )
{
    // std::cout << "matmul_16_6_1" << std::endl;

    mini_jit::Kernel l_kernel;
    mini_jit::InstGen l_gen;
    mini_jit::Brgemm l_brgemm;

    mini_jit::Brgemm::error_t l_res = l_brgemm.generate( 16, 6, 1, 4, 0, 0, 0, dtype_t::fp32 );

    if ( l_res != mini_jit::Brgemm::error_t::success )
    {
        std::cerr << "Failed to generate kernel" << std::endl;
        return;
    }

    // Generate the kernel

    // PCS
    l_kernel.add_instr( l_gen.base_ldp_pre(gpr_t::x29, gpr_t::x30, gpr_t::sp, -16) );
    l_kernel.add_instr( l_gen.mov_sp(gpr_t::x29, gpr_t::sp) );

    // Save callee-saved registers
    l_kernel.add_instr( l_gen.base_stp_pre(gpr_t::x19, gpr_t::x20, gpr_t::sp, -16) );
    l_kernel.add_instr( l_gen.base_stp_pre(gpr_t::x21, gpr_t::x22, gpr_t::sp, -16) );
    l_kernel.add_instr( l_gen.base_stp_pre(gpr_t::x23, gpr_t::x24, gpr_t::sp, -16) );
    l_kernel.add_instr( l_gen.base_stp_pre(gpr_t::x25, gpr_t::x26, gpr_t::sp, -16) );
    l_kernel.add_instr( l_gen.base_stp_pre(gpr_t::x27, gpr_t::x28, gpr_t::sp, -16) );

    l_kernel.add_instr( l_gen.neon_stp_pre(simd_fp_t::v8, simd_fp_t::v9, gpr_t::sp, -16, mini_jit::InstGen::neon_size_spec_t::d) );
    l_kernel.add_instr( l_gen.neon_stp_pre(simd_fp_t::v10, simd_fp_t::v11, gpr_t::sp, -16, mini_jit::InstGen::neon_size_spec_t::d) );
    l_kernel.add_instr( l_gen.neon_stp_pre(simd_fp_t::v12, simd_fp_t::v13, gpr_t::sp, -16, mini_jit::InstGen::neon_size_spec_t::d) );
    l_kernel.add_instr( l_gen.neon_stp_pre(simd_fp_t::v14, simd_fp_t::v15, gpr_t::sp, -16, mini_jit::InstGen::neon_size_spec_t::d) );

    // // Strides
    l_kernel.add_instr( l_gen.mov_imm(gpr_t::x6, 4) );
    l_kernel.add_instr( l_gen.mul_reg(gpr_t::x3, gpr_t::x3, gpr_t::x6) );
    l_kernel.add_instr( l_gen.mul_reg(gpr_t::x4, gpr_t::x4, gpr_t::x6) );
    l_kernel.add_instr( l_gen.mul_reg(gpr_t::x5, gpr_t::x5, gpr_t::x6) );

    // // Load Matrix A
    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x0, 0, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v2, simd_fp_t::v3, gpr_t::x0, 32, mini_jit::InstGen::neon_size_spec_t::q) );

    // // Load Matrix C
    l_kernel.add_instr( l_gen.mov_reg(gpr_t::x7, gpr_t::x2) );
    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v4, simd_fp_t::v5, gpr_t::x7, 0, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v6, simd_fp_t::v7, gpr_t::x7, 32, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v8, simd_fp_t::v9, gpr_t::x7, 0, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x7, 32, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v12, simd_fp_t::v13, gpr_t::x7, 0, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v14, simd_fp_t::v15, gpr_t::x7, 32, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v16, simd_fp_t::v17, gpr_t::x7, 0, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v18, simd_fp_t::v19, gpr_t::x7, 32, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v20, simd_fp_t::v21, gpr_t::x7, 0, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v22, simd_fp_t::v23, gpr_t::x7, 32, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x7, 0, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.neon_ldp_soff(simd_fp_t::v26, simd_fp_t::v27, gpr_t::x7, 32, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    // // Load Column of Matrix B
    l_kernel.add_instr( l_gen.mov_reg(gpr_t::x6, gpr_t::x1) );
    l_kernel.add_instr( l_gen.base_ldr_imm_uoff(gpr_t::w28, gpr_t::x6, 0) );
    l_kernel.add_instr( l_gen.neon_ldr_imm_uoff(simd_fp_t::v28, gpr_t::x6, 0, mini_jit::InstGen::neon_size_spec_t::s) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x6, gpr_t::x6, gpr_t::x4, 0, 0) );

    // // 1st Multiplication
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v28, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v5, simd_fp_t::v1, simd_fp_t::v28, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v6, simd_fp_t::v2, simd_fp_t::v28, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v7, simd_fp_t::v3, simd_fp_t::v28, arr_spec_t::s4) );

    // // Load Column of Matrix B
    l_kernel.add_instr( l_gen.neon_ldr_imm_uoff(simd_fp_t::v29, gpr_t::x6, 0, mini_jit::InstGen::neon_size_spec_t::s) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x6, gpr_t::x6, gpr_t::x4, 0, 0) );

    // // 2nd Multiplication
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v8, simd_fp_t::v0, simd_fp_t::v29, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v9, simd_fp_t::v1, simd_fp_t::v29, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v10, simd_fp_t::v2, simd_fp_t::v29, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v11, simd_fp_t::v3, simd_fp_t::v29, arr_spec_t::s4) );

    // // Load Column of Matrix B
    l_kernel.add_instr( l_gen.neon_ldr_imm_uoff(simd_fp_t::v30, gpr_t::x6, 0, mini_jit::InstGen::neon_size_spec_t::s) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x6, gpr_t::x6, gpr_t::x4, 0, 0) );

    // // 3rd Multiplication
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v12, simd_fp_t::v0, simd_fp_t::v30, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v13, simd_fp_t::v1, simd_fp_t::v30, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v14, simd_fp_t::v2, simd_fp_t::v30, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v15, simd_fp_t::v3, simd_fp_t::v30, arr_spec_t::s4) );

    // // Load Column of Matrix B
    l_kernel.add_instr( l_gen.neon_ldr_imm_uoff(simd_fp_t::v31, gpr_t::x6, 0, mini_jit::InstGen::neon_size_spec_t::s) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x6, gpr_t::x6, gpr_t::x4, 0, 0) );

    // // 4th Multiplication
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v16, simd_fp_t::v0, simd_fp_t::v31, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v17, simd_fp_t::v1, simd_fp_t::v31, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v18, simd_fp_t::v2, simd_fp_t::v31, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v19, simd_fp_t::v3, simd_fp_t::v31, arr_spec_t::s4) );

    // // Load Column of Matrix B
    l_kernel.add_instr( l_gen.neon_ldr_imm_uoff(simd_fp_t::v28, gpr_t::x6, 0, mini_jit::InstGen::neon_size_spec_t::s) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x6, gpr_t::x6, gpr_t::x4, 0, 0) );

    // // 5th Multiplication
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v20, simd_fp_t::v0, simd_fp_t::v28, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v21, simd_fp_t::v1, simd_fp_t::v28, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v22, simd_fp_t::v2, simd_fp_t::v28, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v23, simd_fp_t::v3, simd_fp_t::v28, arr_spec_t::s4) );

    // // Load Column of Matrix B
    l_kernel.add_instr( l_gen.neon_ldr_imm_uoff(simd_fp_t::v29, gpr_t::x6, 0, mini_jit::InstGen::neon_size_spec_t::s) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x6, gpr_t::x6, gpr_t::x4, 0, 0) );

    // // 6th Multiplication
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v24, simd_fp_t::v0, simd_fp_t::v29, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v25, simd_fp_t::v1, simd_fp_t::v29, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v26, simd_fp_t::v2, simd_fp_t::v29, arr_spec_t::s4) );
    l_kernel.add_instr( l_gen.neon_dp_fmla_vector(simd_fp_t::v27, simd_fp_t::v3, simd_fp_t::v29, arr_spec_t::s4) );

    // // Store Matrix C
    l_kernel.add_instr( l_gen.mov_reg(gpr_t::x7, gpr_t::x2) );
    l_kernel.add_instr( l_gen.neon_stp_soff(simd_fp_t::v4, simd_fp_t::v5, gpr_t::x7, 0, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.neon_stp_soff(simd_fp_t::v6, simd_fp_t::v7, gpr_t::x7, 32, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    l_kernel.add_instr( l_gen.neon_stp_soff(simd_fp_t::v8, simd_fp_t::v9, gpr_t::x7, 0, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.neon_stp_soff(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x7, 32, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    l_kernel.add_instr( l_gen.neon_stp_soff(simd_fp_t::v12, simd_fp_t::v13, gpr_t::x7, 0, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.neon_stp_soff(simd_fp_t::v14, simd_fp_t::v15, gpr_t::x7, 32, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    l_kernel.add_instr( l_gen.neon_stp_soff(simd_fp_t::v16, simd_fp_t::v17, gpr_t::x7, 0, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.neon_stp_soff(simd_fp_t::v18, simd_fp_t::v19, gpr_t::x7, 32, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    l_kernel.add_instr( l_gen.neon_stp_soff(simd_fp_t::v20, simd_fp_t::v21, gpr_t::x7, 0, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.neon_stp_soff(simd_fp_t::v22, simd_fp_t::v23, gpr_t::x7, 32, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.add_shifted_reg(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    l_kernel.add_instr( l_gen.neon_stp_soff(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x7, 0, mini_jit::InstGen::neon_size_spec_t::q) );
    l_kernel.add_instr( l_gen.neon_stp_soff(simd_fp_t::v26, simd_fp_t::v27, gpr_t::x7, 32, mini_jit::InstGen::neon_size_spec_t::q) );

    // // Restore callee-saved registers
    l_kernel.add_instr( l_gen.neon_ldp_post(simd_fp_t::v14, simd_fp_t::v15, gpr_t::sp, 16, mini_jit::InstGen::neon_size_spec_t::d) );
    l_kernel.add_instr( l_gen.neon_ldp_post(simd_fp_t::v12, simd_fp_t::v13, gpr_t::sp, 16, mini_jit::InstGen::neon_size_spec_t::d) );
    l_kernel.add_instr( l_gen.neon_ldp_post(simd_fp_t::v10, simd_fp_t::v11, gpr_t::sp, 16, mini_jit::InstGen::neon_size_spec_t::d) );
    l_kernel.add_instr( l_gen.neon_ldp_post(simd_fp_t::v8, simd_fp_t::v9, gpr_t::sp, 16, mini_jit::InstGen::neon_size_spec_t::d) );

    l_kernel.add_instr( l_gen.base_ldp_post(gpr_t::x27, gpr_t::x28, gpr_t::sp, 16) );
    l_kernel.add_instr( l_gen.base_ldp_post(gpr_t::x25, gpr_t::x26, gpr_t::sp, 16) );
    l_kernel.add_instr( l_gen.base_ldp_post(gpr_t::x23, gpr_t::x24, gpr_t::sp, 16) );
    l_kernel.add_instr( l_gen.base_ldp_post(gpr_t::x21, gpr_t::x22, gpr_t::sp, 16) );
    l_kernel.add_instr( l_gen.base_ldp_post(gpr_t::x19, gpr_t::x20, gpr_t::sp, 16) );

    // // Restore stack pointer
    l_kernel.add_instr( l_gen.base_ldp_post(gpr_t::x29, gpr_t::x30, gpr_t::sp, 16) );
    
    l_kernel.add_instr( l_gen.ret() );
    /*
    * END OF THE KERNEL
    */

    l_kernel.write( "matmul_16_6_1.bin" );
    l_kernel.set_kernel();

    // Execute the kernel
    // int64_t (* l_func)( float const *, float const *, float const *, int64_t, int64_t, int64_t, int64_t, int64_t ) = nullptr;
    // l_func = (int64_t (*)(float const *, float const *, float const *, int64_t, int64_t, int64_t, int64_t, int64_t )) l_gen.get_kernel();
    // l_func( A, B, C, M, N, K, 0, 0 );

    auto l_func = l_brgemm.get_kernel(); // returns kernel_t
    l_func( static_cast<void const *>(A),  // cast to match expected void const *
            static_cast<void const *>(B),  // cast to match expected void const *
            static_cast<void *>(C),        // cast to match expected void *
            K,  // ld_a: leading dimension of A
            N,  // ld_b: leading dimension of B
            N,  // ld_c: leading dimension of C
            0,  // br_stride_a
            0 );   // br_stride_b


    // Print result matrix C
    std::cout << "Matrix C (Result):\n";
    for (int i = 0; i < M; ++i) 
    {
        for (int j = 0; j < N; ++j) 
        {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}


int main()
{
    // Setup Matrix Multiplication
  const int M = 16;
  const int N = 6;
  const int K = 1;

  float A[M * K];
  float B[K * N];
  float C[M * N];

  // Initialize matrices
  for ( int i = 0; i < M * K; ++i )
  {
      A[i] = static_cast<float>( i );
  }
  for ( int j = 0; j < K * N; ++j )
  {
      B[j] = static_cast<float>( j );
  }
  std::memset( C, 0, sizeof( C ) );

  // Print matrix A
  std::cout << "Matrix A:" << std::endl;
  for ( int i = 0; i < M; ++i )
  {
      for ( int j = 0; j < K; ++j )
      {
          std::cout << A[i * K + j] << " ";
      }
      std::cout << "\n";
  }

  // Print matrix B
  std::cout << "Matrix B:" << std::endl;
  for ( int i = 0; i < K; ++i )
  {
      for ( int j = 0; j < N; ++j )
      {
          std::cout << B[i * N + j] << " ";
      }
      std::cout << "\n";
  }

  matmul_16_6_1( A, B, C, M, N, M, 0, 0 );

  return EXIT_SUCCESS;
}