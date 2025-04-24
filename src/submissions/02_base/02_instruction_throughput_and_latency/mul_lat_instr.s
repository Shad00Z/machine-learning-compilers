    .text
    .type mul_lat_instr, %function
    .global mul_lat_instr
    /*
    * Performs repeated dependent integer multiplications 
    * to benchmark the latency of the MUL instruction.
    *
    * @param x0: number of loop iterations.
    */
mul_lat_instr:
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

    // consecutive depdendent MUL instructions 
loop:
    // repeat the following block 5 times
    .rept 5
    mul x1, x1, x2
    mul x1, x1, x2
    mul x1, x1, x2
    mul x1, x1, x2
    mul x1, x1, x2
    .endr

    subs x0, x0, #1          // decrement loop counter
    b.ge loop                // branch if not zero

    // restore callee-saved registers
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    // procedure call standard
    ldp x29, x30, [sp], 16
    ret
