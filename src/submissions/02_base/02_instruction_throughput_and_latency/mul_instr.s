    .text
    .type mul_instr, %function
    .global mul_instr
    /*
    * Performs repeated integer multiplications to 
    * benchmark the throughput of the MUL instruction.
    *
    * @param x0: number of loop iterations.
    */
mul_instr:
    // procedure call standard
    stp	x29, x30, [sp, #-16]!
    mov	x29, sp

    // save callee-saved registers
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    // data to be added
    mov x27, #7
    mov x28, #59

    // consecutive MUL instructions 
loop:
    mul x1, x27, x28
    mul x2, x27, x28
    mul x3, x27, x28
    mul x4, x27, x28
    mul x5, x27, x28

    mul x6, x27, x28
    mul x7, x27, x28
    mul x8, x27, x28
    mul x9, x27, x28
    mul x10, x27, x28

    mul x11, x27, x28
    mul x12, x27, x28
    mul x13, x27, x28
    mul x14, x27, x28
    mul x15, x27, x28

    mul x16, x27, x28
    mul x17, x27, x28
    mul x19, x27, x28
    mul x20, x27, x28
    mul x21, x27, x28

    mul x22, x27, x28
    mul x23, x27, x28
    mul x24, x27, x28
    mul x25, x27, x28
    mul x26, x27, x28

    subs x0, x0, #1          // decrement loop counter
    b.ne loop                // branch if not zero

    // restore callee-saved registers
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    // procedure call standard
    ldp x29, x30, [sp], 16
    ret
