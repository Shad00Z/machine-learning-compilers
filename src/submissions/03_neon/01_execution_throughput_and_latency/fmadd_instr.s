    .text
    .type fmadd_instr, %function
    .global fmadd_instr
fmadd_instr:
    // Procedure Call Standard
    stp	x29, x30, [sp, #-16]!
    mov	x29, sp

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

    // load 4 vectors (4 * 16 bytes) from register x1 into v0-v31 
    ldp s0, s1, [x1], #16
    ldp s2, s3, [x1], #16
    ldp s4, s5, [x1], #16
    ldp s6, s7, [x1], #16
    ldp s8, s9, [x1], #16
    ldp s10, s11, [x1], #16
    ldp s12, s13, [x1], #16
    ldp s14, s15, [x1], #16
    ldp s16, s17, [x1], #16
    ldp s18, s19, [x1], #16
    ldp s20, s21, [x1], #16
    ldp s22, s23, [x1], #16
    ldp s24, s25, [x1], #16
    ldp s26, s27, [x1], #16
    ldp s28, s29, [x1], #16
    ldp s30, s31, [x1], #16
    
loop:
    .rept 100
    fmadd  s0,  s8, s16, s24
    fmadd  s1,  s9, s17, s25
    fmadd  s2, s10, s18, s26
    fmadd  s3, s11, s19, s27
    fmadd  s4, s12, s20, s28

    fmadd  s5, s13, s21, s29
    fmadd  s6, s14, s22, s30
    fmadd  s7, s15, s23, s31
    fmadd  s8, s16, s24, s0
    fmadd  s9, s17, s25, s1

    fmadd s10, s18, s26, s2
    fmadd s11, s19, s27, s3
    fmadd s12, s20, s28, s4
    fmadd s13, s21, s29, s5
    fmadd s14, s22, s30, s6

    fmadd s15, s23, s31, s7
    fmadd s16, s24,  s0, s8
    fmadd s17, s25,  s1, s9
    fmadd s18, s26,  s2, s10
    fmadd s19, s27,  s3, s11

    fmadd s20, s28,  s4, s12
    fmadd s21, s29,  s5, s13
    fmadd s22, s30,  s6, s14
    fmadd s23, s31,  s7, s15
    fmadd s24,  s0,  s8, s16

    fmadd s25,  s1,  s9, s17
    fmadd s26,  s2, s10, s18
    fmadd s27,  s3, s11, s19
    fmadd s28,  s4, s12, s20
    fmadd s29,  s5, s13, s21

    fmadd s30,  s6, s14, s22
    fmadd s31,  s7, s15, s23
    .endr

    subs x0, x0, #1
    b.gt loop

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

    // Procedure Call Standard
    ldp x29, x30, [sp], 16
    ret
