    .text
    .type v3_matmul_16_6_1, %function
    .global v3_matmul_16_6_1
    /*
    * Computes C+=AB for three matrices 
    * with the dimensions M=16, N=6, and K=1.
    *
    * @param x0 pointer to column-major matrix A.
    * @param x1 pointer to column-major matrix B.
    * @param x2 pointer to column-major matrix C.
    * @param x3 leading dimension of A.
    * @param x4 leading dimension of B.
    * @param x5 leading dimension of C.
    */
v3_matmul_16_6_1:
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
    mul x3, x3, x6
    mul x4, x4, x6
    mul x5, x5, x6

    /*
     * Load full matrix A
     */ 
    ldp q0, q1, [x0] // 4 + 4 values
    ldp q2, q3, [x0, #32] // 4 + 4 values

    /*
     * Current column of B 
     */
    mov x6, x1

    /*
     * Current column of C 
     */
    mov x7, x2


    /*
     * Matrix C: Column 0
     */
    // Load column of B
    ldr s8, [x6]

    // Load column of C
    ldp q4, q5, [x7]
    ldp q6, q7, [x7, #32]

    // Multiply and accumulate
    fmla v4.4s, v0.4s, v8.s[0]
    fmla v5.4s, v1.4s, v8.s[0]
    fmla v6.4s, v2.4s, v8.s[0]
    fmla v7.4s, v3.4s, v8.s[0]

    // Store column of C
    stp q4, q5, [x7]
    stp q6, q7, [x7, #32]

    /*
     * Matrix C: Column 1
     */
    // Increment current columns
    add x6, x6, x4
    add x7, x7, x5

    // Load column of B
    ldr s8, [x6]

    // Load column of C
    ldp q4, q5, [x7]
    ldp q6, q7, [x7, #32]

    // Multiply and accumulate
    fmla v4.4s, v0.4s, v8.s[0]
    fmla v5.4s, v1.4s, v8.s[0]
    fmla v6.4s, v2.4s, v8.s[0]
    fmla v7.4s, v3.4s, v8.s[0]

    // Store column of C
    stp q4, q5, [x7]
    stp q6, q7, [x7, #32]

    /*
     * Matrix C: Column 2
     */
    // Increment current columns
    add x6, x6, x4
    add x7, x7, x5

    // Load column of B
    ldr s8, [x6]

    // Load column of C
    ldp q4, q5, [x7]
    ldp q6, q7, [x7, #32]

    // Multiply and accumulate
    fmla v4.4s, v0.4s, v8.s[0]
    fmla v5.4s, v1.4s, v8.s[0]
    fmla v6.4s, v2.4s, v8.s[0]
    fmla v7.4s, v3.4s, v8.s[0]

    // Store column of C
    stp q4, q5, [x7]
    stp q6, q7, [x7, #32]

    /*
     * Matrix C: Column 3
     */
    // Increment current columns
    add x6, x6, x4
    add x7, x7, x5

    // Load column of B
    ldr s8, [x6]

    // Load column of C
    ldp q4, q5, [x7]
    ldp q6, q7, [x7, #32]

    // Multiply and accumulate
    fmla v4.4s, v0.4s, v8.s[0]
    fmla v5.4s, v1.4s, v8.s[0]
    fmla v6.4s, v2.4s, v8.s[0]
    fmla v7.4s, v3.4s, v8.s[0]

    // Store column of C
    stp q4, q5, [x7]
    stp q6, q7, [x7, #32]

    /*
     * Matrix C: Column 4
     */
    // Increment current columns
    add x6, x6, x4
    add x7, x7, x5

    // Load column of B
    ldr s8, [x6]

    // Load column of C
    ldp q4, q5, [x7]
    ldp q6, q7, [x7, #32]

    // Multiply and accumulate
    fmla v4.4s, v0.4s, v8.s[0]
    fmla v5.4s, v1.4s, v8.s[0]
    fmla v6.4s, v2.4s, v8.s[0]
    fmla v7.4s, v3.4s, v8.s[0]

    // Store column of C
    stp q4, q5, [x7]
    stp q6, q7, [x7, #32]

    /*
     * Matrix C: Column 5
     */
    // Increment current columns
    add x6, x6, x4
    add x7, x7, x5

    // Load column of B
    ldr s8, [x6]

    // Load column of C
    ldp q4, q5, [x7]
    ldp q6, q7, [x7, #32]

    // Multiply and accumulate
    fmla v4.4s, v0.4s, v8.s[0]
    fmla v5.4s, v1.4s, v8.s[0]
    fmla v6.4s, v2.4s, v8.s[0]
    fmla v7.4s, v3.4s, v8.s[0]
    
    // Store column of C
    stp q4, q5, [x7]
    stp q6, q7, [x7, #32]

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
