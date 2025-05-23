    .text
    .type v2_trans_neon_8_8, %function
    .global v2_trans_neon_8_8
    /*
    * Transposition of matrix A
    *
    * @param x0 pointer to column-major matrix A.
    * @param x1 pointer to column-major matrix B.
    * @param x2 leading dimension of A.
    * @param x3 leading dimension of B.
    */
v2_trans_neon_8_8:
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

    // Constants
    mov x12, #0
    mov x13, #0

    mov x25, #16

    mov x23, #4
    mul x23, x23, x2
    mov x26, x23

    mov x24, #4
    mul x24, x24, x3
    mov x27, x24

    mov x9, #2 // n loop

n_loop:

    mov x6, #2 // m loop

m_loop:
    /*
     * Part 1:
     * Transpose 4x4 block.
     */
    mov x7, x4
    mov x8, x5

    ldr q0, [x7]
    add x7, x7, x2

    ldr q1, [x7]
    add x7, x7, x2

    ldr q2, [x7]
    add x7, x7, x2

    ldr q3, [x7]

    /*
    * Part 2.1:
    * Transpose 4x4 block.
    */
    trn1 v4.4s, v0.4s, v2.4s
    trn1 v5.4s, v1.4s, v3.4s

    trn2 v6.4s, v0.4s, v2.4s
    trn2 v7.4s, v1.4s, v3.4s

    /*
    * Part 2.2:
    * Transpose 4x4 block.
    */
    zip1 v16.4s, v4.4s, v5.4s
    zip1 v17.4s, v6.4s, v7.4s

    zip2 v18.4s, v4.4s, v5.4s
    zip2 v19.4s, v6.4s, v7.4s

    /*
    * Part 3:
    * Store 4x4 block of A into B.
    */
    str q16, [x8]
    add x8, x8, x3

    str q17, [x8]
    add x8, x8, x3

    str q18, [x8]
    add x8, x8, x3

    str q19, [x8]

    // Jump 4 rows in A
    add x4, x4, x25

    // Jump 4 columns in B
    add x5, x5, x27

    sub x6, x6, #1
    cbnz x6, m_loop


    // Restore Pointer for A and B
    mov x4, x0
    mov x5, x1

    add x12, x12, x26
    add x13, x13, x25

    add x4, x4, x12
    add x5, x5, x13

    sub x9, x9, #1
    cbnz x9, n_loop


    // restore frame pointer and link register
    ldp fp, lr, [sp], #16

    ret
