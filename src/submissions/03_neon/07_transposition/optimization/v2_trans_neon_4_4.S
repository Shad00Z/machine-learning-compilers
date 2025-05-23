    .text
    .type v2_trans_neon_4_4, %function
    .global v2_trans_neon_4_4
    /*
    * Transposition of matrix A
    *
    * @param x0 pointer to column-major matrix A.
    * @param x1 pointer to column-major matrix B.
    * @param x2 leading dimension of A.
    * @param x3 leading dimension of B.
    */
v2_trans_neon_4_4:
    // save frame pointer and link register
    stp fp, lr, [sp, #-16]!
    // update frame pointer to current stack pointer
    mov fp, sp

    // hold addresses to A and B in work registers
    mov x4, x0 // A
    mov x5, x1 // B

    // convert strides to bytes
    lsl x2, x2, #2 // stride of A
    lsl x3, x3, #2 // stride of B

    /*
    * Part 1:
    * Load 4x4 block of A.
    */
    ldr q0, [x4]
    add x4, x4, x2
    ldr q1, [x4]
    add x4, x4, x2
    ldr q2, [x4]
    add x4, x4, x2
    ldr q3, [x4]

    /*
    * Part 2:
    * Transpose 4x4 block.
    */
    trn1 v4.4s, v0.4s, v1.4s
    trn2 v5.4s, v0.4s, v1.4s
    trn1 v6.4s, v2.4s, v3.4s
    trn2 v7.4s, v2.4s, v3.4s

    zip1 v16.2d, v4.2d, v6.2d
    zip1 v17.2d, v5.2d, v7.2d
    zip2 v18.2d, v4.2d, v6.2d
    zip2 v19.2d, v5.2d, v7.2d

    /*
    * Part 3:
    * Store 4x4 block of A into B.
    */
    str q16, [x5]
    add x5, x5, x3
    str q17, [x5]
    add x5, x5, x3
    str q18, [x5]
    add x5, x5, x3
    str q19, [x5]

    // restore frame pointer and link register
    ldp fp, lr, [sp], #16

    ret
