    .text
    .type v2_matmul_16_6_1, %function
    .global v2_matmul_16_6_1
    /*
    * Computes C+=AB for three matrices
    * Loads matrix A and C completely and matrix B
    * column by column.
    *
    * @param x0 pointer to column-major matrix A.
    * @param x1 pointer to column-major matrix B.
    * @param x2 pointer to column-major matrix C.
    * @param x3 leading dimension of A.
    * @param x4 leading dimension of B.
    * @param x5 leading dimension of C.
    */
v2_matmul_16_6_1:
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

    /*
     * Multiply strides with float size
     */ 
    mov x6, #4
    mul x3, x3, x6 // 16
    mul x4, x4, x6 // 1 
    mul x5, x5, x6 // 16

    /*
     * Load matrix A 
     */
    ldp q0, q1, [x0]
    ldp q2, q3, [x0, #32]

    /*
     * Load matrix C
     */
    mov x7, x2              // current column of C

    ldp q4, q5, [x7]
    ldp q6, q7, [x7, #32]
    add x7, x7, x5

    ldp q8,  q9,  [x7]
    ldp q10, q11, [x7, #32]
    add x7, x7, x5

    ldp q12, q13, [x7]
    ldp q14, q15, [x7, #32]
    add x7, x7, x5

    ldp q16, q17, [x7]
    ldp q18, q19, [x7, #32]
    add x7, x7, x5

    ldp q20, q21, [x7]
    ldp q22, q23, [x7, #32]
    add x7, x7, x5

    ldp q24, q25, [x7]
    ldp q26, q27, [x7, #32]

    /*
     * Load column of B (1 / 6)
     */
    mov x6, x1              // current column of B

    ldr s28, [x6]
    add x6, x6, x4
    
    /*
     * Multiply and accumulate (1 / 6)
     */ 
    fmla v4.4s, v0.4s, v28.s[0]
    fmla v5.4s, v1.4s, v28.s[0]
    fmla v6.4s, v2.4s, v28.s[0]
    fmla v7.4s, v3.4s, v28.s[0]

    /*
     * Load column of B (2 / 6)
     */
    ldr s29, [x6]
    add x6, x6, x4

    /*
     * Multiply and accumulate (2 / 6)
     */ 
    fmla v8.4s,  v0.4s, v29.s[0]
    fmla v9.4s,  v1.4s, v29.s[0]
    fmla v10.4s, v2.4s, v29.s[0]
    fmla v11.4s, v3.4s, v29.s[0]

    /*
     * Load column of B (3 / 6)
     */
    ldr s30, [x6]
    add x6, x6, x4

    /*
     * Multiply and accumulate (3 / 6)
     */ 
    fmla v12.4s, v0.4s, v30.s[0]
    fmla v13.4s, v1.4s, v30.s[0]
    fmla v14.4s, v2.4s, v30.s[0]
    fmla v15.4s, v3.4s, v30.s[0]

    /*
     * Load column of B (4 / 6)
     */
    ldr s31, [x6]
    add x6, x6, x4

    /*
     * Multiply and accumulate (4 / 6)
     */ 
    fmla v16.4s, v0.4s, v31.s[0]
    fmla v17.4s, v1.4s, v31.s[0]
    fmla v18.4s, v2.4s, v31.s[0]
    fmla v19.4s, v3.4s, v31.s[0]

    /*
     * Load column of B (5 / 6)
     */
    ldr s28, [x6]
    add x6, x6, x4

    /*
     * Multiply and accumulate (5 / 6)
     */ 
    fmla v20.4s, v0.4s, v28.s[0]
    fmla v21.4s, v1.4s, v28.s[0]
    fmla v22.4s, v2.4s, v28.s[0]
    fmla v23.4s, v3.4s, v28.s[0]

    /*
     * Load column of B (6 / 6)
     */
    ldr s29, [x6]
    add x6, x6, x4

    /*
     * Multiply and accumulate (6 / 6)
     */ 
    fmla v24.4s, v0.4s, v29.s[0]
    fmla v25.4s, v1.4s, v29.s[0]
    fmla v26.4s, v2.4s, v29.s[0]
    fmla v27.4s, v3.4s, v29.s[0]

    /*
     * Store results in C
     */
    mov x7, x2              // Start: column of C

    stp q4, q5, [x7]
    stp q6, q7, [x7, #32]
    add x7, x7, x5

    stp q8,  q9,  [x7]
    stp q10, q11, [x7, #32]
    add x7, x7, x5

    stp q12, q13, [x7]
    stp q14, q15, [x7, #32]
    add x7, x7, x5
    
    stp q16, q17, [x7]
    stp q18, q19, [x7, #32]
    add x7, x7, x5

    stp q20, q21, [x7]
    stp q22, q23, [x7, #32]
    add x7, x7, x5

    stp q24, q25, [x7]
    stp q26, q27, [x7, #32]
    add x7, x7, x5

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

    ret
