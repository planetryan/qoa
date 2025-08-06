// minimal RISC-V assembly stub
.text
.global asm_function
.align 4

asm_function:
    // function prologue
    addi sp, sp, -16
    sd ra, 8(sp)
    sd s0, 0(sp)
    addi s0, sp, 16
    
    // function body (no-op for stub)
    
    // Function epilogue
    ld ra, 8(sp)
    ld s0, 0(sp)
    addi sp, sp, 16
    ret
