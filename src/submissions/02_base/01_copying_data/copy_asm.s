    .text
    .type copy_asm_0, %function
    .global copy_asm_0
    /*
    * Copies an array of 7 32-bit integers from one 
    * location to another.
    *
    * @param x0: address of the source array.
    * @param x1: address of the destination array.
    */
copy_asm_0:
    // b[0] = a[0]
    ldr w2, [x0]
    str w2, [x1]
    // b[1] = a[1]
    ldr w2, [x0, #4]
    str w2, [x1, #4]
    // b[2] = a[2]
    ldr w2, [x0, #8]
    str w2, [x1, #8]
    // b[3] = a[3]
    ldr w2, [x0, #12]
    str w2, [x1, #12]
    // b[4] = a[4]
    ldr w2, [x0, #16]
    str w2, [x1, #16]
    // b[5] = a[5]
    ldr w2, [x0, #20]
    str w2, [x1, #20]
    // b[6] = a[6]
    ldr w2, [x0, #24]
    str w2, [x1, #24]
    ret


    .text
    .type copy_asm_1, %function
    .global copy_asm_1
    /*
    * Copies an array of n 32-bit integers from one 
    * location to another.
    *
    * @param x0: number of elements to copy.
    * @param x1: address of the source array.
    * @param x2: address of the destination array.
    */
copy_asm_1:
    // number of elements copied
    mov x3, #0
    // byte offset for array
    mov x4, #0
loop:
    // b[i] = a[i]
    ldr w5, [x1, x4]
    str w5, [x2, x4]
    // increment the number of elements copied
    add x3, x3, #1
    // increment the byte offset
    add x4, x4, #4
    // check if we have copied n elements
    cmp x3, x0
    // if not, loop again
    blt loop
    ret
