    .text
    .type v4_matmul_14_6_64, %function
    .global v4_matmul_14_6_64
    /*
    * Computes C+=AB for three matrices 
    * with the dimensions M=14, N=6, and K=64.
    * 
    * Idea:
    * Equal to Version 2 but with different loads
    * 
    * @param x0 pointer to column-major matrix A.
    * @param x1 pointer to column-major matrix B.
    * @param x2 pointer to column-major matrix C.
    * @param x3 leading dimension of A.
    * @param x4 leading dimension of B.
    * @param x5 leading dimension of C.
    */
v4_matmul_14_6_64:
// #################### START PCS ####################
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
// #################### END PCS ####################

    // multiply strides with float size
    mov x6, #4
    mul x3, x3, x6 // lda
    mul x4, x4, x6 // ldb
    mul x5, x5, x6 // ldc

    // LOAD MATRIX C
    mov x8, x2
    // first column
    ld1 {v0.4s-v3.4s},   [x8]
    add x8, x8, x5
    // second column
    ld1 {v4.4s-v7.4s},   [x8]
    add x8, x8, x5
    // third column
    ld1 {v8.4s-v11.4s},  [x8]
    add x8, x8, x5
    // fourth column
    ld1 {v12.4s-v15.4s}, [x8]
    add x8, x8, x5
    // fifth column
    ld1 {v16.4s-v19.4s}, [x8]
    add x8, x8, x5
    // sixth column
    ld1 {v20.4s-v23.4s}, [x8]

    //  K loop counter
    mov x6, #64
    // set start of A
    mov x7, x0
    // set start of B
    mov x8, x1
    // init row count of B
    mov x9, #0

_k1_loop:
    // load column of A
    ldp q24, q25, [x7] // 4 + 4 values
    ldr q26, [x7, #32] // 4 values
    ldr d27, [x7, #48] // 2 values

    // B: COLUMN 0
    ldr s29, [x8]
    fmla v0.4s, v24.4s, v29.s[0]
    fmla v1.4s, v25.4s, v29.s[0]
    fmla v2.4s, v26.4s, v29.s[0]
    fmla v3.2s, v27.2s, v29.s[0]
    // B: COLUMN 1
    add x8, x8, x4
    ldr s29, [x8]
    fmla v4.4s, v24.4s, v29.s[0]
    fmla v5.4s, v25.4s, v29.s[0]
    fmla v6.4s, v26.4s, v29.s[0]
    fmla v7.2s, v27.2s, v29.s[0]
    // B: COLUMN 2
    add x8, x8, x4
    ldr s29, [x8]
    fmla  v8.4s, v24.4s, v29.s[0]
    fmla  v9.4s, v25.4s, v29.s[0]
    fmla v10.4s, v26.4s, v29.s[0]
    fmla v11.2s, v27.2s, v29.s[0]
    // B: COLUMN 3
    add x8, x8, x4
    ldr s29, [x8]
    fmla v12.4s, v24.4s, v29.s[0]
    fmla v13.4s, v25.4s, v29.s[0]
    fmla v14.4s, v26.4s, v29.s[0]
    fmla v15.2s, v27.2s, v29.s[0]
    // B: COLUMN 4
    add x8, x8, x4
    ldr s29, [x8]
    fmla v16.4s, v24.4s, v29.s[0]
    fmla v17.4s, v25.4s, v29.s[0]
    fmla v18.4s, v26.4s, v29.s[0]
    fmla v19.2s, v27.2s, v29.s[0]
    // B: COLUMN 5
    add x8, x8, x4
    ldr s29, [x8]
    fmla v20.4s, v24.4s, v29.s[0]
    fmla v21.4s, v25.4s, v29.s[0]
    fmla v22.4s, v26.4s, v29.s[0]
    fmla v23.2s, v27.2s, v29.s[0]

    // move to next column of A
    add x7, x7, x3
    // move to next row of B
    mov x8, x1
    add x9, x9, #4
    add x8, x8, x9

    // decrement loop counter
    sub x6, x6, #1
    // check if loop counter is zero
    cbnz x6, _k1_loop


    // STORE MATRIX C
    mov x8, x2
    // first column
    st1 {v0.4s-v3.4s},   [x8]
    add x8, x8, x5
    // second column
    st1 {v4.4s-v7.4s},   [x8]
    add x8, x8, x5
    // third column
    st1 {v8.4s-v11.4s},  [x8]
    add x8, x8, x5
    // fourth column
    st1 {v12.4s-v15.4s}, [x8]
    add x8, x8, x5
    // fifth column
    st1 {v16.4s-v19.4s}, [x8]
    add x8, x8, x5
    // sixth column
    st1 {v20.4s-v23.4s}, [x8]

// #################### START PCS ####################
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
// #################### END PCS ####################
    ret
    