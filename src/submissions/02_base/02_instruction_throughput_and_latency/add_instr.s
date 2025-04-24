    .text
    .type add_instr, %function
    .global add_instr
    /*
    * Performs repeated integer additions to benchmark
    * the throughput of the ADD (shifted register) 
    * instruction.
    *
    * @param x0: number of loop iterations.
    */
add_instr:
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

// consecutive ADD instructions
loop:
    add x1, x27, x28
    add x2, x27, x28
    add x3, x27, x28
    add x4, x27, x28
    add x5, x27, x28

    add x6, x27, x28
    add x7, x27, x28
    add x8, x27, x28
    add x9, x27, x28
    add x10, x27, x28

    add x11, x27, x28
    add x12, x27, x28
    add x13, x27, x28
    add x14, x27, x28
    add x15, x27, x28

    add x16, x27, x28
    add x17, x27, x28
    add x19, x27, x28
    add x20, x27, x28
    add x21, x27, x28

    add x22, x27, x28
    add x23, x27, x28
    add x24, x27, x28
    add x25, x27, x28
    add x26, x27, x28

    subs x0, x0, #1          // decrement loop counter
    b.gt loop                // branch if greater than zero

    // restore callee-saved registers
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    // procedure call standard
    ldp x29, x30, [sp], 16
    ret
