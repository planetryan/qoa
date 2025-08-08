.section .data
    .align 8

    // constant for 1/sqrt(2)
    sqrt_half:      .double 0.70710678118654752440
    vec_sqrt_half:  .double 0.70710678118654752440

.section .text
    .global quantum_hadamard_transform
    .global quantum_pauli_x_gate
    .global quantum_pauli_z_gate
    .global quantum_phase_shift
    .global quantum_cnot_operation
    .global quantum_swap_gate
    .global quantum_toffoli_gate

// --- quantum_hadamard_transform ---

// input: r3 (state vector ptr), r4 (num qubits), r5 (target qubit)
quantum_hadamard_transform:
    // POWER ABI prologue
    mflr r0
    std r0, 16(r1) // save LR
    stdu r1, -160(r1) // allocate stack frame for saved registers

    // Save callee-saved GPRs (r14-r31)
    std r19, 144(r1)
    std r20, 136(r1)
    std r21, 128(r1)
    std r22, 120(r1)
    std r23, 112(r1)
    std r24, 104(r1)
    std r25, 96(r1)
    std r26, 88(r1)
    std r27, 80(r1)
    std r28, 72(r1)
    std r29, 64(r1)
    std r30, 56(r1)
    std r31, 48(r1)
    
    // Save callee-saved VSX registers (v20-v31)
    stxvd2x v20, r1, 40 
    stxvd2x v21, r1, 32 
    stxvd2x v22, r1, 24 
    stxvd2x v23, r1, 16 
    stxvd2x v24, r1, 8  
    stxvd2x v25, r1, 0  

    // move arguments to callee-saved registers
    mr r19, r3 // r19: state vector ptr
    mr r20, r4 // r20: num qubits
    mr r21, r5 // r21: target qubit

    // calculate total states (2^n)
    li r9, 1
    slw r22, r9, r20 // r22: 2^num_qubits

    // calculate stride for target qubit (2^target_qubit)
    slw r23, r9, r21 // r23: 2^target_qubit

    // load hadamard constant 1/sqrt(2) into a VSX register
    lfd f0, vec_sqrt_half@toc(r2)
    xvspltdp v15, f0 // broadcast to vector

    // loop through the state vector
    li r24, 0 // i = 0
.hadamard_loop:
    cmpw r24, r22
    bge .hadamard_done

    // if target bit is 0, process this pair
    srw r9, r24, r21 // shift i to get the target bit
    andi. r9, r9, 1 // isolate the bit
    bne .hadamard_skip // if bit is 1, skip

    // get index for |1> component (j = i + stride)
    add r25, r24, r23 // r25: j

    mulli r26, r24, 16 // r26: offset for state[i]
    mulli r27, r25, 16 // r27: offset for state[j]
    add r26, r19, r26 // r26: addr of state[i]
    add r27, r19, r27 // r27: addr of state[j]

    // load amplitudes as a vector of two doubles (q0 -> v0, q1 -> v1)
    lxvd2x v0, r26, 0 // v0 = state[i] (real, imag)
    lxvd2x v1, r27, 0 // v1 = state[j] (real, imag)

    xvmovdp v2, v0            // temp for 'a'
    xvadddp v0, v0, v1        // a + b
    xvmuldp v0, v0, v15       // (a + b) * constant

    xvsubdp v2, v2, v1        // a - b
    xvmuldp v2, v2, v15       // (a - b) * constant

    // store new amplitudes
    stxvd2x v0, r26, 0
    stxvd2x v2, r27, 0

.hadamard_skip:
    addi r24, r24, 1
    b .hadamard_loop

.hadamard_done:
    // restore regs
    lxvd2x v25, r1, 0
    lxvd2x v24, r1, 8
    lxvd2x v23, r1, 16
    lxvd2x v22, r1, 24
    lxvd2x v21, r1, 32
    lxvd2x v20, r1, 40
    
    ld r31, 48(r1)
    ld r30, 56(r1)
    ld r29, 64(r1)
    ld r28, 72(r1)
    ld r27, 80(r1)
    ld r26, 88(r1)
    ld r25, 96(r1)
    ld r24, 104(r1)
    ld r23, 112(r1)
    ld r22, 120(r1)
    ld r21, 128(r1)
    ld r20, 136(r1)
    ld r19, 144(r1)
    addi r1, r1, 160 // deallocate stack frame
    ld r0, 16(r1) // restore LR
    mtlr r0
    blr

// --- quantum_pauli_x_gate ---

// input: r3 (state vector ptr), r4 (num qubits), r5 (target qubit)
quantum_pauli_x_gate:
    mflr r0
    std r0, 16(r1) // save LR
    stdu r1, -160(r1) // allocate stack frame

    // Save callee-saved GPRs and VSX registers
    std r19, 144(r1)
    std r20, 136(r1)
    std r21, 128(r1)
    std r22, 120(r1)
    std r23, 112(r1)
    std r24, 104(r1)
    std r25, 96(r1)
    std r26, 88(r1)
    std r27, 80(r1)
    std r28, 72(r1)
    std r29, 64(r1)
    std r30, 56(r1)
    std r31, 48(r1)

    stxvd2x v20, r1, 40 
    stxvd2x v21, r1, 32 
    stxvd2x v22, r1, 24 
    stxvd2x v23, r1, 16 
    stxvd2x v24, r1, 8  
    stxvd2x v25, r1, 0  

    mr r19, r3 // r19: state vector ptr
    mr r20, r4 // r20: num qubits
    mr r21, r5 // r21: target qubit

    li r9, 1
    slw r22, r9, r20 // r22: 2^num_qubits
    slw r23, r9, r21 // r23: 2^target_qubit (stride)

    li r24, 0 // i = 0
.pauli_x_loop:
    cmpw r24, r22
    bge .pauli_x_done

    // if target bit is 0, swap it with its pair
    srw r9, r24, r21
    andi. r9, r9, 1
    bne .pauli_x_skip // if bit is 1, skip

    // get index for |1> component (j = i + stride)
    add r25, r24, r23

    // calculate addresses
    mulli r26, r24, 16
    mulli r27, r25, 16
    add r26, r19, r26 // addr of state[i]
    add r27, r19, r27 // addr of state[j]

    // load amplitudes
    lxvd2x v0, r26, 0 // q0 = state[i]
    lxvd2x v1, r27, 0 // q1 = state[j]

    // swap them
    stxvd2x v1, r26, 0 // store q1 at state[i]
    stxvd2x v0, r27, 0 // store q0 at state[j]

.pauli_x_skip:
    addi r24, r24, 1
    b .pauli_x_loop

.pauli_x_done:
    // restore regs
    lxvd2x v25, r1, 0
    lxvd2x v24, r1, 8
    lxvd2x v23, r1, 16
    lxvd2x v22, r1, 24
    lxvd2x v21, r1, 32
    lxvd2x v20, r1, 40
    
    ld r31, 48(r1)
    ld r30, 56(r1)
    ld r29, 64(r1)
    ld r28, 72(r1)
    ld r27, 80(r1)
    ld r26, 88(r1)
    ld r25, 96(r1)
    ld r24, 104(r1)
    ld r23, 112(r1)
    ld r22, 120(r1)
    ld r21, 128(r1)
    ld r20, 136(r1)
    ld r19, 144(r1)
    addi r1, r1, 160 // deallocate stack frame
    ld r0, 16(r1) // restore LR
    mtlr r0
    blr

// --- quantum_pauli_z_gate ---
// input: r3 (state vector ptr), r4 (num qubits), r5 (target qubit)
quantum_pauli_z_gate:
    mflr r0
    std r0, 16(r1) // save LR
    stdu r1, -160(r1) // allocate stack frame

    // Save callee-saved GPRs and VSX registers
    std r19, 144(r1)
    std r20, 136(r1)
    std r21, 128(r1)
    std r22, 120(r1)
    std r23, 112(r1)
    std r24, 104(r1)
    std r25, 96(r1)
    std r26, 88(r1)
    std r27, 80(r1)
    std r28, 72(r1)
    std r29, 64(r1)
    std r30, 56(r1)
    std r31, 48(r1)

    stxvd2x v20, r1, 40 
    stxvd2x v21, r1, 32 
    stxvd2x v22, r1, 24 
    stxvd2x v23, r1, 16 
    stxvd2x v24, r1, 8  
    stxvd2x v25, r1, 0  

    mr r19, r3 // r19: state vector ptr
    mr r20, r4 // r20: num qubits
    mr r21, r5 // r21: target qubit

    li r9, 1
    slw r22, r9, r20 // r22: 2^num_qubits
    slw r23, r9, r21 // r23: 2^target_qubit (mask)

    li r24, 0 // i = 0
.pauli_z_loop:
    cmpw r24, r22
    bge .pauli_z_done

    // if target bit is 1, negate the amplitude
    and. r9, r24, r23 // check if the target bit is set
    cmpi 0, r9, 0
    bne .pauli_z_process

.pauli_z_skip:
    addi r24, r24, 1
    b .pauli_z_loop

.pauli_z_process:
    // calculate address
    mulli r26, r24, 16
    add r26, r19, r26 // addr of state[i]
    
    // load amplitude
    lxvd2x v0, r26, 0 // q0 = state[i] (real, imag)
    
    // negate the amplitude (both real and imag parts)
    xvnegdp v0, v0 // v0 = -v0

    // store the new amplitude
    stxvd2x v0, r26, 0

    addi r24, r24, 1
    b .pauli_z_loop

.pauli_z_done:
    // restore regs
    lxvd2x v25, r1, 0
    lxvd2x v24, r1, 8
    lxvd2x v23, r1, 16
    lxvd2x v22, r1, 24
    lxvd2x v21, r1, 32
    lxvd2x v20, r1, 40
    
    ld r31, 48(r1)
    ld r30, 56(r1)
    ld r29, 64(r1)
    ld r28, 72(r1)
    ld r27, 80(r1)
    ld r26, 88(r1)
    ld r25, 96(r1)
    ld r24, 104(r1)
    ld r23, 112(r1)
    ld r22, 120(r1)
    ld r21, 128(r1)
    ld r20, 136(r1)
    ld r19, 144(r1)
    addi r1, r1, 160 // deallocate stack frame
    ld r0, 16(r1) // restore LR
    mtlr r0
    blr
