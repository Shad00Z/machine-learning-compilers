    .text
    .type trans_neon_8_8, %function
    .global trans_neon_8_8
    /*
    * Transposition of matrix A
    *
    * @param x0 pointer to column-major matrix A.
    * @param x1 pointer to column-major matrix B.
    * @param x2 leading dimension of A.
    * @param x3 leading dimension of B.
    */
trans_neon_8_8:
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

    // hold addresses to A and B in work registers
    mov x4, x0 // A
    mov x5, x1 // B

    // convert strides to bytes
    lsl x2, x2, #2 // stride of A
    lsl x3, x3, #2 // stride of B

    /*
    * Part 1.1:
    * Load 4x4 block of A (Left Top).
    */
    ldr q0, [x4]
    add x4, x4, x2
    ldr q1, [x4]
    add x4, x4, x2
    ldr q2, [x4]
    add x4, x4, x2
    ldr q3, [x4]

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
    zip1 v8.4s, v4.4s, v5.4s
    zip1 v9.4s, v6.4s, v7.4s

    zip2 v10.4s, v4.4s, v5.4s
    zip2 v11.4s, v6.4s, v7.4s

    /*
    * Part 3:
    * Store 4x4 block of A into B.
    */
    str q8, [x5]
    add x5, x5, x3
    str q9, [x5]
    add x5, x5, x3
    str q10, [x5]
    add x5, x5, x3
    str q11, [x5]


    /*
    * Part 1.2:
    * Load 4x4 block of A (Left bottom, top right).
    */
    mov x4, x0 // A
    mov x5, x1 // B

    add x4, x4, #16
    add x5, x5, #16

    ldr q0, [x4]
    add x4, x4, x2
    ldr q1, [x4]
    add x4, x4, x2
    ldr q2, [x4]
    add x4, x4, x2
    ldr q3, [x4]

    // Right top
    mov x4, x0 // A
    mov x5, x1 // B

    add x4, x4, #128
    add x5, x5, #128
    
    ldr q12, [x4]
    add x4, x4, x2
    ldr q13, [x4]
    add x4, x4, x2
    ldr q14, [x4]
    add x4, x4, x2
    ldr q15, [x4]

    /*
    * Part 2.1:
    * Transpose 4x4 block.
    */
    // Left Bottom
    trn1 v4.4s, v0.4s, v2.4s
    trn1 v5.4s, v1.4s, v3.4s

    trn2 v6.4s, v0.4s, v2.4s
    trn2 v7.4s, v1.4s, v3.4s

    // Right Top
    trn1 v16.4s, v12.4s, v14.4s
    trn1 v17.4s, v13.4s, v15.4s

    trn2 v18.4s, v12.4s, v14.4s
    trn2 v19.4s, v13.4s, v15.4s

    /*
    * Part 2.2:
    * Transpose 4x4 block.
    */
    // Left Bottom
    zip1 v8.4s, v4.4s, v5.4s    
    zip1 v9.4s, v6.4s, v7.4s    

    zip2 v10.4s, v4.4s, v5.4s   
    zip2 v11.4s, v6.4s, v7.4s   

    // Right Top
    zip1 v20.4s, v16.4s, v17.4s 
    zip1 v21.4s, v18.4s, v19.4s 

    zip2 v22.4s, v16.4s, v17.4s 
    zip2 v23.4s, v18.4s, v19.4s

    /*
    * Part 3:
    * Store 4x4 block of Submatrix A''' into A''.
    */
    // Left Bottom (values from right top)
    mov x5, x1
    add x5, x5, #16

    str q20, [x5]
    add x5, x5, x3
    str q21, [x5]
    add x5, x5, x3
    str q22, [x5]
    add x5, x5, x3
    str q23, [x5]

    // Right top (values from left bottom)
    mov x5, x1
    add x5, x5, #128

    str q8, [x5]
    add x5, x5, x3
    str q9, [x5]
    add x5, x5, x3
    str q10, [x5]
    add x5, x5, x3
    str q11, [x5]



    /*
    * Part 1.3:
    * Load 4x4 block of A (Right bottom).
    */
    mov x4, x0
    add x4, x4, #144

    ldr q0, [x4]
    add x4, x4, x2
    ldr q1, [x4]
    add x4, x4, x2
    ldr q2, [x4]
    add x4, x4, x2
    ldr q3, [x4]

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
    zip1 v8.4s, v4.4s, v5.4s
    zip1 v9.4s, v6.4s, v7.4s

    zip2 v10.4s, v4.4s, v5.4s
    zip2 v11.4s, v6.4s, v7.4s

    /*
    * Part 3:
    * Store 4x4 block of A into B.
    */
    mov x5, x1
    add x5, x5, #144

    str q8, [x5]
    add x5, x5, x3
    str q9, [x5]
    add x5, x5, x3
    str q10, [x5]
    add x5, x5, x3
    str q11, [x5]


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
