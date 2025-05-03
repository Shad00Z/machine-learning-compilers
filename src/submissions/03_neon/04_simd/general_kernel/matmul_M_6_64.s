    .text
    .type matmul_M_6_64, %function
    .global matmul_M_6_64
    /*
    * Computes C+=AB for three matrices 
    * with the dimensions M, N=6, and K=64.
    *
    * @param x0 pointer to column-major matrix A.
    * @param x1 pointer to column-major matrix B.
    * @param x2 pointer to column-major matrix C.
    * @param x3 leading dimension of A.
    * @param x4 leading dimension of B.
    * @param x5 leading dimension of C.
    */
matmul_M_6_64:
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

    // M mod 8
    and x10, x3, #7
    // M loop counter: M/8
    lsr x11, x3, #3

    // multiply strides with float size
    mov x6, #4
    mul x3, x3, x6 // lda
    mul x4, x4, x6 // ldb
    mul x5, x5, x6 // ldc

    // save base matrix pointers
    mov x7, x0 // A
    mov x8, x1 // B
    mov x9, x2 // C

    // save original k loop counter
    mov x19, #64

    // check if M < 8
    cmp x11, #0
    // if M < 8, skip the m loop
    beq start_trail

_m_loop:
// ------------------------------------------
// START matmul_8_6_64
// ------------------------------------------

    // LOAD MATRIX C
    mov x12, x9
    // first column
    ldp q0, q1, [x12]
    // second column
    add x12, x12, x5
    ldp q4, q5, [x12]
    // third column
    add x12, x12, x5
    ldp q8, q9, [x12]
    // fourth column
    add x12, x12, x5
    ldp q12, q13, [x12]
    // fifth column
    add x12, x12, x5
    ldp q16, q17, [x12]
    // sixth column
    add x12, x12, x5
    ldp q20, q21, [x12]

    // K loop counter
    mov x14, x19
    // set start of A
    mov x15, x7
    // set start of B
    mov x16, x8
    // init row count of B
    mov x17, #0
_k_loop:
    // load column of A (8 values)
    ldp q24, q25, [x15]

    // B: COLUMN 0
    ldr s29, [x16]
    fmla v0.4s, v24.4s, v29.s[0]
    fmla v1.4s, v25.4s, v29.s[0]

    // B: COLUMN 1
    add x16, x16, x4
    ldr s29, [x16]
    fmla v4.4s, v24.4s, v29.s[0]
    fmla v5.4s, v25.4s, v29.s[0]

    // B: COLUMN 2
    add x16, x16, x4
    ldr s29, [x16]
    fmla v8.4s, v24.4s, v29.s[0]
    fmla v9.4s, v25.4s, v29.s[0]

    // B: COLUMN 3
    add x16, x16, x4
    ldr s29, [x16]
    fmla v12.4s, v24.4s, v29.s[0]
    fmla v13.4s, v25.4s, v29.s[0]

    // B: COLUMN 4
    add x16, x16, x4
    ldr s29, [x16]
    fmla v16.4s, v24.4s, v29.s[0]
    fmla v17.4s, v25.4s, v29.s[0]

    // B: COLUMN 5
    add x16, x16, x4
    ldr s29, [x16]
    fmla v20.4s, v24.4s, v29.s[0]
    fmla v21.4s, v25.4s, v29.s[0]

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

    // STORE MATRIX C
    mov x12, x9
    // first column
    stp q0, q1, [x12]
    // second column
    add x12, x12, x5
    stp q4, q5, [x12]
    // third column
    add x12, x12, x5
    stp q8, q9, [x12]
    // fourth column
    add x12, x12, x5
    stp q12, q13, [x12]
    // fifth column
    add x12, x12, x5
    stp q16, q17, [x12]
    // sixth column
    add x12, x12, x5
    stp q20, q21, [x12]

// ------------------------------------------
// END matmul_8_6_64
// ------------------------------------------

    // increase A and C pointers for next block
    add x7, x7, #8*4
    add x9, x9, #8*4

    // decrement m loop counter
    sub x11, x11, #1
    // check if loop counter is zero
    cbnz x11, _m_loop

// ------------------------------------------
// START M mod 8 trail computation
// ------------------------------------------
start_trail:

    // Shared code for all cases
    // K loop counter
    mov x14, x19
    // set start of A
    mov x15, x7
    // set start of B
    mov x16, x8
    // init row count of B
    mov x17, #0

    adr x11, jump_table     // x11 = address of jump_table
    lsl x12, x10, #3        // x12 = index * 8
    ldr x13, [x11, x12]     // load relative offset
    add x13, x11, x13       // compute actual target address
    br x13                  // jump

jump_table:
    .8byte case_0 - jump_table
    .8byte case_1 - jump_table
    .8byte case_2 - jump_table
    .8byte case_3 - jump_table
    .8byte case_4 - jump_table
    .8byte case_5 - jump_table
    .8byte case_6 - jump_table
    .8byte case_7 - jump_table

// ------------------------------------------
// x10 == 0
// ------------------------------------------
case_0:
    B end_trail

// ------------------------------------------
// x10 == 1
// ------------------------------------------
case_1:
    // LOAD MATRIX C (1 value)
    mov x12, x9
    // first column
    ldr s0, [x12]
    // second column
    add x12, x12, x5
    ldr s1, [x12]
    // third column
    add x12, x12, x5
    ldr s2, [x12]
    // fourth column
    add x12, x12, x5
    ldr s3, [x12]
    // fifth column
    add x12, x12, x5
    ldr s4, [x12]
    // sixth column
    add x12, x12, x5
    ldr s5, [x12]

case_1_k_loop:
    // load column of A (1 value)
    ldr s24, [x15]

    // B: COLUMN 0
    ldr s29, [x16]
    fmadd s0, s24, s29, s0
    // B: COLUMN 1
    add x16, x16, x4
    ldr s29, [x16]
    fmadd s1, s24, s29, s1
    // B: COLUMN 2
    add x16, x16, x4
    ldr s29, [x16]
    fmadd s2, s24, s29, s2
    // B: COLUMN 3
    add x16, x16, x4
    ldr s29, [x16]
    fmadd s3, s24, s29, s3
    // B: COLUMN 4
    add x16, x16, x4
    ldr s29, [x16]
    fmadd s4, s24, s29, s4
    // B: COLUMN 5
    add x16, x16, x4
    ldr s29, [x16]
    fmadd s5, s24, s29, s5

    // move to next column of A
    add x15, x15, x3
    // move to next row of B
    mov x16, x8
    add x17, x17, #4
    add x16, x16, x17

    // decrement loop counter
    sub x14, x14, #1
    // check if loop counter is zero
    cbnz x14, case_1_k_loop

    // STORE MATRIX C
    mov x12, x9
    // first column
    str s0, [x12]
    // second column
    add x12, x12, x5
    str s1, [x12]
    // third column
    add x12, x12, x5
    str s2, [x12]
    // fourth column
    add x12, x12, x5
    str s3, [x12]
    // fifth column
    add x12, x12, x5
    str s4, [x12]
    // sixth column
    add x12, x12, x5
    str s5, [x12]

    B end_trail

// ------------------------------------------
// x10 == 2
// ------------------------------------------
case_2:
    // LOAD MATRIX C (2 values)
    mov x12, x9
    // first column
    ldr d0, [x12]
    // second column
    add x12, x12, x5
    ldr d1, [x12]
    // third column
    add x12, x12, x5
    ldr d2, [x12]
    // fourth column
    add x12, x12, x5
    ldr d3, [x12]
    // fifth column
    add x12, x12, x5
    ldr d4, [x12]
    // sixth column
    add x12, x12, x5
    ldr d5, [x12]

case_2_k_loop:
    // load column of A (2 values)
    ldr d24, [x15]

    // B: COLUMN 0
    ldr s29, [x16]
    fmla v0.2s, v24.2s, v29.s[0]
    // B: COLUMN 1
    add x16, x16, x4
    ldr s29, [x16]
    fmla v1.2s, v24.2s, v29.s[0]
    // B: COLUMN 2
    add x16, x16, x4
    ldr s29, [x16]
    fmla v2.2s, v24.2s, v29.s[0]
    // B: COLUMN 3
    add x16, x16, x4
    ldr s29, [x16]
    fmla v3.2s, v24.2s, v29.s[0]
    // B: COLUMN 4
    add x16, x16, x4
    ldr s29, [x16]
    fmla v4.2s, v24.2s, v29.s[0]
    // B: COLUMN 5
    add x16, x16, x4
    ldr s29, [x16]
    fmla v5.2s, v24.2s, v29.s[0]

    // move to next column of A
    add x15, x15, x3
    // move to next row of B
    mov x16, x8
    add x17, x17, #4
    add x16, x16, x17

    // decrement loop counter
    sub x14, x14, #1
    // check if loop counter is zero
    cbnz x14, case_2_k_loop

    // STORE MATRIX C
    mov x12, x9
    // first column
    str d0, [x12]
    // second column
    add x12, x12, x5
    str d1, [x12]
    // third column
    add x12, x12, x5
    str d2, [x12]
    // fourth column
    add x12, x12, x5
    str d3, [x12]
    // fifth column
    add x12, x12, x5
    str d4, [x12]
    // sixth column
    add x12, x12, x5
    str d5, [x12]

    B end_trail

// ------------------------------------------
// x10 == 3
// ------------------------------------------
case_3:
    // LOAD MATRIX C (3 values)
    mov x12, x9
    // first column
    mov x20, x12
    ld1 {v0.s}[0], [x20], #4
    ld1 {v0.s}[1], [x20], #4
    ld1 {v0.s}[2], [x20]
    mov  v0.s[3], wzr
    // second column
    add x12, x12, x5
    mov x20, x12
    ld1 {v1.s}[0], [x20], #4
    ld1 {v1.s}[1], [x20], #4
    ld1 {v1.s}[2], [x20]
    mov  v1.s[3], wzr
    // third column
    add x12, x12, x5
    mov x20, x12
    ld1 {v2.s}[0], [x20], #4
    ld1 {v2.s}[1], [x20], #4
    ld1 {v2.s}[2], [x20]
    mov  v2.s[3], wzr
    // fourth column
    add x12, x12, x5
    mov x20, x12
    ld1 {v3.s}[0], [x20], #4
    ld1 {v3.s}[1], [x20], #4
    ld1 {v3.s}[2], [x20]
    mov  v3.s[3], wzr
    // fifth column
    add x12, x12, x5
    mov x20, x12
    ld1 {v4.s}[0], [x20], #4
    ld1 {v4.s}[1], [x20], #4
    ld1 {v4.s}[2], [x20]
    mov  v4.s[3], wzr
    // sixth column
    add x12, x12, x5
    mov x20, x12
    ld1 {v5.s}[0], [x20], #4
    ld1 {v5.s}[1], [x20], #4
    ld1 {v5.s}[2], [x20]
    mov  v5.s[3], wzr

case_3_k_loop:
    // load column of A (3 values)
    mov x20, x15
    ld1 {v24.s}[0], [x20], #4
    ld1 {v24.s}[1], [x20], #4
    ld1 {v24.s}[2], [x20]
    mov  v24.s[3], wzr

    // B: COLUMN 0
    ldr s29, [x16]
    fmla v0.4s, v24.4s, v29.s[0]
    // B: COLUMN 1
    add x16, x16, x4
    ldr s29, [x16]
    fmla v1.4s, v24.4s, v29.s[0]
    // B: COLUMN 2
    add x16, x16, x4
    ldr s29, [x16]
    fmla v2.4s, v24.4s, v29.s[0]
    // B: COLUMN 3
    add x16, x16, x4
    ldr s29, [x16]
    fmla v3.4s, v24.4s, v29.s[0]
    // B: COLUMN 4
    add x16, x16, x4
    ldr s29, [x16]
    fmla v4.4s, v24.4s, v29.s[0]
    // B: COLUMN 5
    add x16, x16, x4
    ldr s29, [x16]
    fmla v5.4s, v24.4s, v29.s[0]

    // move to next column of A
    add x15, x15, x3
    // move to next row of B
    mov x16, x8
    add x17, x17, #4
    add x16, x16, x17

    // decrement loop counter
    sub x14, x14, #1
    // check if loop counter is zero
    cbnz x14, case_3_k_loop

    // STORE MATRIX C (3 values)
    mov x12, x9
    // first column
    mov x20, x12
    st1 {v0.s}[0], [x20], #4
    st1 {v0.s}[1], [x20], #4
    st1 {v0.s}[2], [x20]
    mov  v0.s[3], wzr
    // second column
    add x12, x12, x5
    mov x20, x12
    st1 {v1.s}[0], [x20], #4
    st1 {v1.s}[1], [x20], #4
    st1 {v1.s}[2], [x20]
    mov  v1.s[3], wzr
    // third column
    add x12, x12, x5
    mov x20, x12
    st1 {v2.s}[0], [x20], #4
    st1 {v2.s}[1], [x20], #4
    st1 {v2.s}[2], [x20]
    mov  v2.s[3], wzr
    // fourth column
    add x12, x12, x5
    mov x20, x12
    st1 {v3.s}[0], [x20], #4
    st1 {v3.s}[1], [x20], #4
    st1 {v3.s}[2], [x20]
    mov  v3.s[3], wzr
    // fifth column
    add x12, x12, x5
    mov x20, x12
    st1 {v4.s}[0], [x20], #4
    st1 {v4.s}[1], [x20], #4
    st1 {v4.s}[2], [x20]
    mov  v4.s[3], wzr
    // sixth column
    add x12, x12, x5
    mov x20, x12
    st1 {v5.s}[0], [x20], #4
    st1 {v5.s}[1], [x20], #4
    st1 {v5.s}[2], [x20]
    mov  v5.s[3], wzr

    B end_trail

// ------------------------------------------
// x10 == 4
// ------------------------------------------
case_4:
    // LOAD MATRIX C (4 values)
    mov x12, x9
    // first column
    ldr q0, [x12]
    // second column
    add x12, x12, x5
    ldr q1, [x12]
    // third column
    add x12, x12, x5
    ldr q2, [x12]
    // fourth column
    add x12, x12, x5
    ldr q3, [x12]
    // fifth column
    add x12, x12, x5
    ldr q4, [x12]
    // sixth column
    add x12, x12, x5
    ldr q5, [x12]

case_4_k_loop:
    // load column of A (4 values)
    ldr q24, [x15]

    // B: COLUMN 0
    ldr s29, [x16]
    fmla v0.4s, v24.4s, v29.s[0]
    // B: COLUMN 1
    add x16, x16, x4
    ldr s29, [x16]
    fmla v1.4s, v24.4s, v29.s[0]
    // B: COLUMN 2
    add x16, x16, x4
    ldr s29, [x16]
    fmla v2.4s, v24.4s, v29.s[0]
    // B: COLUMN 3
    add x16, x16, x4
    ldr s29, [x16]
    fmla v3.4s, v24.4s, v29.s[0]
    // B: COLUMN 4
    add x16, x16, x4
    ldr s29, [x16]
    fmla v4.4s, v24.4s, v29.s[0]
    // B: COLUMN 5
    add x16, x16, x4
    ldr s29, [x16]
    fmla v5.4s, v24.4s, v29.s[0]

    // move to next column of A
    add x15, x15, x3
    // move to next row of B
    mov x16, x8
    add x17, x17, #4
    add x16, x16, x17

    // decrement loop counter
    sub x14, x14, #1
    // check if loop counter is zero
    cbnz x14, case_4_k_loop

    // STORE MATRIX C
    mov x12, x9
    // first column
    str q0, [x12]
    // second column
    add x12, x12, x5
    str q1, [x12]
    // third column
    add x12, x12, x5
    str q2, [x12]
    // fourth column
    add x12, x12, x5
    str q3, [x12]
    // fifth column
    add x12, x12, x5
    str q4, [x12]
    // sixth column
    add x12, x12, x5
    str q5, [x12]

    B end_trail

// ------------------------------------------
// x10 == 5
// ------------------------------------------
case_5:
     // LOAD MATRIX C (5 values)
    mov x12, x9
    // first column
    ldr q0, [x12]
    ldr s1, [x12, #16]
    // second column
    add x12, x12, x5
    ldr q2, [x12]
    ldr s3, [x12, #16]
    // third column
    add x12, x12, x5
    ldr q4, [x12]
    ldr s5, [x12, #16]
    // fourth column
    add x12, x12, x5
    ldr q6, [x12]
    ldr s7, [x12, #16]
    // fifth column
    add x12, x12, x5
    ldr q8, [x12]
    ldr s9, [x12, #16]
    // sixth column
    add x12, x12, x5
    ldr q10, [x12]
    ldr s11, [x12, #16]

case_5_k_loop:
    // load column of A (5 values)
    ldr q24, [x15]
    ldr s25, [x15, #16]

    // B: COLUMN 0
    ldr s29, [x16]
    fmla v0.4s, v24.4s, v29.s[0]
    fmadd s1, s25, s29, s1 
    // B: COLUMN 1
    add x16, x16, x4
    ldr s29, [x16]
    fmla v2.4s, v24.4s, v29.s[0]
    fmadd s3, s25, s29, s3
    // B: COLUMN 2
    add x16, x16, x4
    ldr s29, [x16]
    fmla v4.4s, v24.4s, v29.s[0]
    fmadd s5, s25, s29, s5
    // B: COLUMN 3
    add x16, x16, x4
    ldr s29, [x16]
    fmla v6.4s, v24.4s, v29.s[0]
    fmadd s7, s25, s29, s7
    // B: COLUMN 4
    add x16, x16, x4
    ldr s29, [x16]
    fmla v8.4s, v24.4s, v29.s[0]
    fmadd s9, s25, s29, s9
    // B: COLUMN 5
    add x16, x16, x4
    ldr s29, [x16]
    fmla v10.4s, v24.4s, v29.s[0]
    fmadd s11, s25, s29, s11

    // move to next column of A
    add x15, x15, x3
    // move to next row of B
    mov x16, x8
    add x17, x17, #4
    add x16, x16, x17

    // decrement loop counter
    sub x14, x14, #1
    // check if loop counter is zero
    cbnz x14, case_5_k_loop

     // STORE MATRIX C (5 values)
    mov x12, x9
    // first column
    str q0, [x12]
    str s1, [x12, #16]
    // second column
    add x12, x12, x5
    str q2, [x12]
    str s3, [x12, #16]
    // third column
    add x12, x12, x5
    str q4, [x12]
    str s5, [x12, #16]
    // fourth column
    add x12, x12, x5
    str q6, [x12]
    str s7, [x12, #16]
    // fifth column
    add x12, x12, x5
    str q8, [x12]
    str s9, [x12, #16]
    // sixth column
    add x12, x12, x5
    str q10, [x12]
    str s11, [x12, #16]

    B end_trail

// ------------------------------------------
// x10 == 6
// ------------------------------------------
case_6:
     // LOAD MATRIX C (6 values)
    mov x12, x9
    // first column
    ldr q0, [x12]
    ldr d1, [x12, #16]
    // second column
    add x12, x12, x5
    ldr q2, [x12]
    ldr d3, [x12, #16]
    // third column
    add x12, x12, x5
    ldr q4, [x12]
    ldr d5, [x12, #16]
    // fourth column
    add x12, x12, x5
    ldr q6, [x12]
    ldr d7, [x12, #16]
    // fifth column
    add x12, x12, x5
    ldr q8, [x12]
    ldr d9, [x12, #16]
    // sixth column
    add x12, x12, x5
    ldr q10, [x12]
    ldr d11, [x12, #16]

case_6_k_loop:
    // load column of A (5 values)
    ldr q24, [x15]
    ldr d25, [x15, #16]

    // B: COLUMN 0
    ldr s29, [x16]
    fmla v0.4s, v24.4s, v29.s[0]
    fmla v1.2s, v25.2s, v29.s[0]
    // B: COLUMN 1
    add x16, x16, x4
    ldr s29, [x16]
    fmla v2.4s, v24.4s, v29.s[0]
    fmla v3.2s, v25.2s, v29.s[0]
    // B: COLUMN 2
    add x16, x16, x4
    ldr s29, [x16]
    fmla v4.4s, v24.4s, v29.s[0]
    fmla v5.2s, v25.2s, v29.s[0]
    // B: COLUMN 3
    add x16, x16, x4
    ldr s29, [x16]
    fmla v6.4s, v24.4s, v29.s[0]
    fmla v7.2s, v25.2s, v29.s[0]
    // B: COLUMN 4
    add x16, x16, x4
    ldr s29, [x16]
    fmla v8.4s, v24.4s, v29.s[0]
    fmla v9.2s, v25.2s, v29.s[0]
    // B: COLUMN 5
    add x16, x16, x4
    ldr s29, [x16]
    fmla v10.4s, v24.4s, v29.s[0]
    fmla v11.2s, v25.2s, v29.s[0]

    // move to next column of A
    add x15, x15, x3
    // move to next row of B
    mov x16, x8
    add x17, x17, #4
    add x16, x16, x17

    // decrement loop counter
    sub x14, x14, #1
    // check if loop counter is zero
    cbnz x14, case_6_k_loop

     // STORE MATRIX C (5 values)
    mov x12, x9
    // first column
    str q0, [x12]
    str d1, [x12, #16]
    // second column
    add x12, x12, x5
    str q2, [x12]
    str d3, [x12, #16]
    // third column
    add x12, x12, x5
    str q4, [x12]
    str d5, [x12, #16]
    // fourth column
    add x12, x12, x5
    str q6, [x12]
    str d7, [x12, #16]
    // fifth column
    add x12, x12, x5
    str q8, [x12]
    str d9, [x12, #16]
    // sixth column
    add x12, x12, x5
    str q10, [x12]
    str d11, [x12, #16]

    B end_trail

// ------------------------------------------
// x10 == 7
// ------------------------------------------
case_7:
    // LOAD MATRIX C (7 values)
    mov x12, x9
    // first column
    mov x20, x12
    ldr q0, [x20], #16
    ld1 {v1.s}[0], [x20], #4
    ld1 {v1.s}[1], [x20], #4
    ld1 {v1.s}[2], [x20]
    mov  v1.s[3], wzr
    // second column
    add x12, x12, x5
    mov x20, x12
    ldr q2, [x20], #16
    ld1 {v3.s}[0], [x20], #4
    ld1 {v3.s}[1], [x20], #4
    ld1 {v3.s}[2], [x20]
    mov  v3.s[3], wzr
    // third column
    add x12, x12, x5
    mov x20, x12
    ldr q4, [x20], #16
    ld1 {v5.s}[0], [x20], #4
    ld1 {v5.s}[1], [x20], #4
    ld1 {v5.s}[2], [x20]
    mov  v5.s[3], wzr
    // fourth column
    add x12, x12, x5
    mov x20, x12
    ldr q6, [x20], #16
    ld1 {v7.s}[0], [x20], #4
    ld1 {v7.s}[1], [x20], #4
    ld1 {v7.s}[2], [x20]
    mov  v7.s[3], wzr
    // fifth column
    add x12, x12, x5
    mov x20, x12
    ldr q8, [x20], #16
    ld1 {v9.s}[0], [x20], #4
    ld1 {v9.s}[1], [x20], #4
    ld1 {v9.s}[2], [x20]
    mov  v9.s[3], wzr
    // sixth column
    add x12, x12, x5
    mov x20, x12
    ldr q10, [x20], #16
    ld1 {v11.s}[0], [x20], #4
    ld1 {v11.s}[1], [x20], #4
    ld1 {v11.s}[2], [x20]
    mov  v11.s[3], wzr

case_7_k_loop:
    // load column of A (7 values)
    mov x20, x15
    ldr q24, [x20], #16
    ld1 {v25.s}[0], [x20], #4
    ld1 {v25.s}[1], [x20], #4
    ld1 {v25.s}[2], [x20]
    mov  v25.s[3], wzr

    // B: COLUMN 0
    ldr s29, [x16]
    fmla v0.4s, v24.4s, v29.s[0]
    fmla v1.4s, v25.4s, v29.s[0]
    // B: COLUMN 1
    add x16, x16, x4
    ldr s29, [x16]
    fmla v2.4s, v24.4s, v29.s[0]
    fmla v3.4s, v25.4s, v29.s[0]
    // B: COLUMN 2
    add x16, x16, x4
    ldr s29, [x16]
    fmla v4.4s, v24.4s, v29.s[0]
    fmla v5.4s, v25.4s, v29.s[0]
    // B: COLUMN 3
    add x16, x16, x4
    ldr s29, [x16]
    fmla v6.4s, v24.4s, v29.s[0]
    fmla v7.4s, v25.4s, v29.s[0]
    // B: COLUMN 4
    add x16, x16, x4
    ldr s29, [x16]
    fmla v8.4s, v24.4s, v29.s[0]
    fmla v9.4s, v25.4s, v29.s[0]
    // B: COLUMN 5
    add x16, x16, x4
    ldr s29, [x16]
    fmla v10.4s, v24.4s, v29.s[0]
    fmla v11.4s, v25.4s, v29.s[0]

    // move to next column of A
    add x15, x15, x3
    // move to next row of B
    mov x16, x8
    add x17, x17, #4
    add x16, x16, x17

    // decrement loop counter
    sub x14, x14, #1
    // check if loop counter is zero
    cbnz x14, case_7_k_loop

    // STORE MATRIX C (7 values)
    mov x12, x9
    // first column
    mov x20, x12
    str q0, [x20], #16
    st1 {v1.s}[0], [x20], #4
    st1 {v1.s}[1], [x20], #4
    st1 {v1.s}[2], [x20]
    mov  v1.s[3], wzr
    // second column
    add x12, x12, x5
    mov x20, x12
    str q2, [x20], #16
    st1 {v3.s}[0], [x20], #4
    st1 {v3.s}[1], [x20], #4
    st1 {v3.s}[2], [x20]
    mov  v3.s[3], wzr
    // third column
    add x12, x12, x5
    mov x20, x12
    str q4, [x20], #16
    st1 {v5.s}[0], [x20], #4
    st1 {v5.s}[1], [x20], #4
    st1 {v5.s}[2], [x20]
    mov  v5.s[3], wzr
    // fourth column
    add x12, x12, x5
    mov x20, x12
    str q6, [x20], #16
    st1 {v7.s}[0], [x20], #4
    st1 {v7.s}[1], [x20], #4
    st1 {v7.s}[2], [x20]
    mov  v7.s[3], wzr
    // fifth column
    add x12, x12, x5
    mov x20, x12
    str q8, [x20], #16
    st1 {v9.s}[0], [x20], #4
    st1 {v9.s}[1], [x20], #4
    st1 {v9.s}[2], [x20]
    mov  v9.s[3], wzr
    // sixth column
    add x12, x12, x5
    mov x20, x12
    str q10, [x20], #16
    st1 {v11.s}[0], [x20], #4
    st1 {v11.s}[1], [x20], #4
    st1 {v11.s}[2], [x20]
    mov  v11.s[3], wzr

    B end_trail


end_trail:

// ------------------------------------------
// END M mod 8 trail computation
// ------------------------------------------

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
