// minimal POWER assembly stub
.text
.global asm_function
.align 4

asm_function:
    // function prologue
    mflr 0
    std 0, 16(1)
    stdu 1, -32(1)
    
    // function body (no-op for stub)
    
    // function epilogue
    addi 1, 1, 32
    ld 0, 16(1)
    mtlr 0
    blr
