.data
    // align data for cache performance
    .align 8
    
    // constant for 1/sqrt(2)
    sqrt_half:      .double 0.70710678118654752440

.text
    .global quantum_hadamard_transform
    .global quantum_pauli_x_gate
    .global quantum_pauli_z_gate
    .global quantum_phase_shift
    .global quantum_cnot_operation
    .global quantum_swap_gate
    .global quantum_toffoli_gate

// --- quantum_hadamard_transform ---

// input: a0 (state vector ptr), a1 (num qubits), a2 (target qubit)
quantum_hadamard_transform:
    addi sp, sp, -16
    sd ra, 8(sp)    // save return address
    sd s0, 0(sp)    // save s0
    
    // move args to callee-saved registers
    mv s0, a0       // s0: state vector ptr
    mv s1, a1       // s1: num qubits
    mv s2, a2       // s2: target qubit

    // calculate total states (2^n)
    li t0, 1
    sll t1, t0, s1  // t1: 2^num_qubits

    // calculate stride for target qubit (2^target_qubit)
    sll t2, t0, s2  // t2: 2^target_qubit (stride)

    // load hadamard constant 1/sqrt(2) into a vector register
    // vsetvli configures vector registers to hold double-precision floats
    vsetvli zero, zero, e64, m1, ta, ma // vl=1 (or max), SEW=64
    la a3, sqrt_half
    fld ft0, 0(a3)
    vfmv.v.f v0, ft0 // broadcast scalar to vector register
    
    // loop through the state vector
    li t3, 0 // i = 0
.hadamard_loop:
    beq t3, t1, .hadamard_done // branch if i == total states

    // check if target bit is 0, process this pair
    srl t4, t3, s2   // shift i to get the target bit
    andi t4, t4, 1   // isolate the bit
    bne t4, zero, .hadamard_skip // if bit is 1, skip

    // get index for |1> component (j = i + stride)
    add t5, t3, t2   // t5: j

    // calculate addresses
    slli t6, t3, 4   // t6: offset for state[i] (i * 16 bytes)
    slli t7, t5, 4   // t7: offset for state[j]
    add t6, s0, t6   // t6: addr of state[i]
    add t7, s0, t7   // t7: addr of state[j]

    // load amplitudes as a vector of two doubles (complex number)
    vle64.v v1, (t6) // v1 = state[i] (real, imag)
    vle64.v v2, (t7) // v2 = state[j] (real, imag)

    // calculate new amplitudes using vector operations
    // v3 = a+b, v4 = a-b
    vfadd.vv v3, v1, v2
    vfsub.vv v4, v1, v2

    // v3 = (a+b) / sqrt(2), v4 = (a-b) / sqrt(2)
    vfmul.vv v3, v3, v0
    vfmul.vv v4, v4, v0

    // store new amplitudes
    vse64.v v3, (t6)
    vse64.v v4, (t7)

.hadamard_skip:
    addi t3, t3, 1
    j .hadamard_loop

.hadamard_done:
    // restore registers
    ld s0, 0(sp)
    ld ra, 8(sp)
    addi sp, sp, 16
    ret

// --- quantum_pauli_x_gate ---

// input: a0 (state vector ptr), a1 (num qubits), a2 (target qubit)
quantum_pauli_x_gate:
    
    addi sp, sp, -16
    sd ra, 8(sp)
    sd s0, 0(sp)
    
    mv s0, a0       // s0: state vector ptr
    mv s1, a1       // s1: num qubits
    mv s2, a2       // s2: target qubit

    li t0, 1
    sll t1, t0, s1  // t1: 2^num_qubits
    sll t2, t0, s2  // t2: 2^target_qubit (stride)

    li t3, 0 // i = 0
.pauli_x_loop:
    beq t3, t1, .pauli_x_done

    // check if target bit is 0
    srl t4, t3, s2
    andi t4, t4, 1
    bne t4, zero, .pauli_x_skip

    // get index for |1> component (j = i + stride)
    add t5, t3, t2

    // calculate addresses
    slli t6, t3, 4
    slli t7, t5, 4
    add t6, s0, t6   // addr of state[i]
    add t7, s0, t7   // addr of state[j]

    // load amplitudes
    vsetvli zero, zero, e64, m1, ta, ma
    vle64.v v1, (t6) // v1 = state[i]
    vle64.v v2, (t7) // v2 = state[j]

    // swap them
    vse64.v v2, (t6) // store v2 at state[i]
    vse64.v v1, (t7) // store v1 at state[j]

.pauli_x_skip:
    addi t3, t3, 1
    j .pauli_x_loop

.pauli_x_done:
    // restore registers
    ld s0, 0(sp)
    ld ra, 8(sp)
    addi sp, sp, 16
    ret

// --- quantum_pauli_z_gate ---
// input: a0 (state vector ptr), a1 (num qubits), a2 (target qubit)
quantum_pauli_z_gate:
    
    addi sp, sp, -16
    sd ra, 8(sp)
    sd s0, 0(sp)
    
    mv s0, a0       // s0: state vector ptr
    mv s1, a1       // s1: num qubits
    mv s2, a2       // s2: target qubit

    li t0, 1
    sll t1, t0, s1  // t1: 2^num_qubits
    sll t2, t0, s2  // t2: 2^target_qubit (mask)

    li t3, 0 // i = 0
.pauli_z_loop:
    beq t3, t1, .pauli_z_done

    // check if target bit is 1
    and t4, t3, t2 // check if the target bit is set
    beq t4, zero, .pauli_z_skip

    // calculate address
    slli t5, t3, 4
    add t5, s0, t5 // addr of state[i]
    
    // load amplitude
    vsetvli zero, zero, e64, m1, ta, ma
    vle64.v v1, (t5) // v1 = state[i] (real, imag)
    
    // negate the amplitude (both real and imag parts)
    vfneg.v v1, v1 // v1 = -v1

    // store the new amplitude
    vse64.v v1, (t5)

.pauli_z_skip:
    addi t3, t3, 1
    j .pauli_z_loop

.pauli_z_done:
    // restore registers
    ld s0, 0(sp)
    ld ra, 8(sp)
    addi sp, sp, 16
    ret
