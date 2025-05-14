    .text
    .global test_asm
test_asm:

    fmadd s1, s25, s29, s1
    fmadd d1, d25, d29, d1

    ret