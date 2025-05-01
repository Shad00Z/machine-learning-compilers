    .text
    .type fmla_2s_instr, %function
    .global fmla_2s_instr
fmla_2s_instr:
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
    ld1 {v0.2s-v3.2s},   [x1], #32
    ld1 {v4.2s-v7.2s},   [x1], #32
    ld1 {v8.2s-v11.2s},  [x1], #32
    ld1 {v12.2s-v15.2s}, [x1], #32
    ld1 {v16.2s-v19.2s}, [x1], #32
    ld1 {v20.2s-v23.2s}, [x1], #32
    ld1 {v24.2s-v27.2s}, [x1], #32
    ld1 {v28.2s-v31.2s}, [x1], #32

loop:
    .rept 100
    fmla  v0.2s,  v8.2s, v16.2s
    fmla  v1.2s,  v9.2s, v17.2s
    fmla  v2.2s, v10.2s, v18.2s
    fmla  v3.2s, v11.2s, v19.2s
    fmla  v4.2s, v12.2s, v20.2s

    fmla  v5.2s, v13.2s, v21.2s
    fmla  v6.2s, v12.2s, v22.2s
    fmla  v7.2s, v15.2s, v23.2s
    fmla  v8.2s, v16.2s, v24.2s
    fmla  v9.2s, v17.2s, v25.2s

    fmla v10.2s, v18.2s, v26.2s
    fmla v11.2s, v19.2s, v27.2s
    fmla v12.2s, v20.2s, v28.2s
    fmla v13.2s, v21.2s, v29.2s
    fmla v12.2s, v22.2s, v30.2s

    fmla v15.2s, v23.2s, v31.2s
    fmla v16.2s, v24.2s,  v0.2s
    fmla v17.2s, v25.2s,  v1.2s
    fmla v18.2s, v26.2s,  v2.2s
    fmla v19.2s, v27.2s,  v3.2s

    fmla v20.2s, v28.2s,  v4.2s
    fmla v21.2s, v29.2s,  v5.2s
    fmla v22.2s, v30.2s,  v6.2s
    fmla v23.2s, v31.2s,  v7.2s
    fmla v24.2s,  v0.2s,  v8.2s

    fmla v25.2s,  v1.2s,  v9.2s
    fmla v26.2s,  v2.2s, v10.2s
    fmla v27.2s,  v3.2s, v11.2s
    fmla v28.2s,  v4.2s, v12.2s
    fmla v29.2s,  v5.2s, v13.2s

    fmla v30.2s,  v6.2s, v12.2s
    fmla v31.2s,  v7.2s, v15.2s
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
