    .text
    .type fmla_4s_source_lat_instr, %function
    .global fmla_4s_source_lat_instr
fmla_4s_source_lat_instr:
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
    ld1 {v0.4s-v3.4s},   [x1], #64
    ld1 {v4.4s-v7.4s},   [x1], #64
    ld1 {v8.4s-v11.4s},  [x1], #64
    ld1 {v12.4s-v15.4s}, [x1], #64
    ld1 {v16.4s-v19.4s}, [x1], #64
    ld1 {v20.4s-v23.4s}, [x1], #64
    ld1 {v24.4s-v27.4s}, [x1], #64
    ld1 {v28.4s-v31.4s}, [x1], #64

loop:
    .rept 100
    fmla v0.4s, v0.4s,  v1.4s
    fmla v0.4s, v0.4s,  v2.4s
    fmla v0.4s, v0.4s,  v3.4s
    fmla v0.4s, v0.4s,  v4.4s

    fmla v0.4s, v0.4s,  v5.4s
    fmla v0.4s, v0.4s,  v6.4s
    fmla v0.4s, v0.4s,  v7.4s
    fmla v0.4s, v0.4s,  v8.4s

    fmla v0.4s, v0.4s,  v9.4s
    fmla v0.4s, v0.4s, v10.4s
    fmla v0.4s, v0.4s, v11.4s
    fmla v0.4s, v0.4s, v12.4s

    fmla v0.4s, v0.4s, v13.4s
    fmla v0.4s, v0.4s, v14.4s
    fmla v0.4s, v0.4s, v15.4s
    fmla v0.4s, v0.4s, v16.4s

    fmla v0.4s, v0.4s, v17.4s
    fmla v0.4s, v0.4s, v18.4s
    fmla v0.4s, v0.4s, v19.4s
    fmla v0.4s, v0.4s, v20.4s

    fmla v0.4s, v0.4s, v21.4s
    fmla v0.4s, v0.4s, v22.4s
    fmla v0.4s, v0.4s, v23.4s
    fmla v0.4s, v0.4s, v24.4s

    fmla v0.4s, v0.4s, v25.4s
    fmla v0.4s, v0.4s, v26.4s
    fmla v0.4s, v0.4s, v27.4s
    fmla v0.4s, v0.4s, v28.4s

    fmla v0.4s, v0.4s, v29.4s
    fmla v0.4s, v0.4s, v30.4s
    fmla v0.4s, v0.4s, v31.4s
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
