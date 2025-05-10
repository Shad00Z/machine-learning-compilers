    .text
    .type matmul_64_64_64, %function
    .global matmul_64_64_64
    /*
    * Computes C+=AB for three matrices 
    * with the dimensions M=64, N=64, and K=64.
    *
    * @param x0 pointer to column-major matrix A.
    * @param x1 pointer to column-major matrix B.
    * @param x2 pointer to column-major matrix C.
    * @param x3 leading dimension of A.
    * @param x4 leading dimension of B.
    * @param x5 leading dimension of C.
    */
matmul_64_64_64:
// ------------------------------------------
// START PCS
// ------------------------------------------
    // save frame pointer and link register
    stp fp, lr, [sp, #-16]!
    // update frame pointer to current stack pointer
    mov fp, sp

    // save callee-saved registers
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
// ------------------------------------------
// END PCS
// ------------------------------------------

    // multiply strides with float size
    mov x6, #4
    mul x3, x3, x6 // lda
    mul x4, x4, x6 // ldb
    mul x5, x5, x6 // ldc

    mov x6, #4
    mul x22, x4, x6 // ldb * 4 columns
    mul x23, x5, x6 // ldc * 4 columns

    // set base matrix pointers
    mov x20, x1 // B
    mov x21, x2 // C

    // N loop counter
    mov x19, #16 // 64/4 = 16 blocks

_n_loop:

    // M loop counter
    mov x11, #4 // 64/16 = 4 blocks

    // set matrix pointers
    mov x7, x0 // A
    mov x8, x20 // B
    mov x9, x21 // C

_m_loop:
// ------------------------------------------
// START matmul_16_4_64
// ------------------------------------------

    // LOAD MATRIX C
    mov x12, x9
    // first column
    ldp q0, q1, [x12]
    ldp q2, q3, [x12, #32]
    // second column
    add x12, x12, x5
    ldp q4, q5, [x12]
    ldp q6, q7, [x12, #32]
    // third column
    add x12, x12, x5
    ldp q8, q9, [x12]
    ldp q10, q11, [x12, #32]
    // fourth column
    add x12, x12, x5
    ldp q12, q13, [x12]
    ldp q14, q15, [x12, #32]

    // K loop counter
    mov x14, #64
    // set start of A
    mov x15, x7
    // set start of B
    mov x16, x8
    // init row count of B
    mov x17, #0
_k_loop:
    // load column of A (16 values)
    ldp q24, q25, [x15]
    ldp q26, q27, [x15, #32]

    // B: COLUMN 0
    ldr s29, [x16]
    fmla v0.4s, v24.4s, v29.s[0]
    fmla v1.4s, v25.4s, v29.s[0]
    fmla v2.4s, v26.4s, v29.s[0]
    fmla v3.4s, v27.4s, v29.s[0]

    // B: COLUMN 1
    add x16, x16, x4
    ldr s29, [x16]
    fmla v4.4s, v24.4s, v29.s[0]
    fmla v5.4s, v25.4s, v29.s[0]
    fmla v6.4s, v26.4s, v29.s[0]
    fmla v7.4s, v27.4s, v29.s[0]

    // B: COLUMN 2
    add x16, x16, x4
    ldr s29, [x16]
    fmla v8.4s, v24.4s, v29.s[0]
    fmla v9.4s, v25.4s, v29.s[0]
    fmla v10.4s, v26.4s, v29.s[0]
    fmla v11.4s, v27.4s, v29.s[0]

    // B: COLUMN 3
    add x16, x16, x4
    ldr s29, [x16]
    fmla v12.4s, v24.4s, v29.s[0]
    fmla v13.4s, v25.4s, v29.s[0]
    fmla v14.4s, v26.4s, v29.s[0]
    fmla v15.4s, v27.4s, v29.s[0]

    // move to next column of A
    add x15, x15, x3
    // move to next row of B
    mov x16, x8
    add x17, x17, #4
    add x16, x16, x17

    // decrement loop counter
    sub x14, x14, #1
    // check if loop counter is zero
    cbnz x14, _k_loop
// END K LOOP

    // STORE MATRIX C
    mov x12, x9
    // first column
    stp q0, q1, [x12]
    stp q2, q3, [x12, #32]
    // second column
    add x12, x12, x5
    stp q4, q5, [x12]
    stp q6, q7, [x12, #32]
    // third column
    add x12, x12, x5
    stp q8, q9, [x12]
    stp q10, q11, [x12, #32]
    // fourth column
    add x12, x12, x5
    stp q12, q13, [x12]
    stp q14, q15, [x12, #32]

// ------------------------------------------
// END matmul_16_4_64
// ------------------------------------------

    // increase A and C pointers for next block
    // (jump 16 values)
    add x7, x7, #16*4
    add x9, x9, #16*4

    // decrement m loop counter
    sub x11, x11, #1
    // check if loop counter is zero
    cbnz x11, _m_loop
// END M LOOP

    // increase B and C pointers for next block
    // (jump 4 columns) 4*x4, 4*x5
    add x20, x20, x22
    add x21, x21, x23

    // decrement n loop counter
    sub x19, x19, #1
    // check if loop counter is zero
    cbnz x19, _n_loop
// END N LOOP

// ------------------------------------------
// START PCS
// ------------------------------------------
    // restore callee-saved registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    // restore frame pointer and link register
    ldp fp, lr, [sp], #16
// ------------------------------------------
// END PCS
// ------------------------------------------
    ret
