    .text
    .global test_asm
test_asm:
    ldp w0, w1, [x2], #8
    ldp w0, w1, [x2, #8]!
    ldp w0, w1, [x2, #8]
    ret