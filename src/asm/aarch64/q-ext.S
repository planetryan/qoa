.section .data
    .align 3
    // constant for 1/sqrt(2)
    sqrt_half:      .double 0.70710678118654752440

.section .text
    .global quantum_hadamard_transform
    .global quantum_pauli_x_gate
    .global quantum_pauli_z_gate
    .global quantum_phase_shift
    .global quantum_cnot_operation
    .global quantum_swap_gate
    .global quantum_toffoli_gate

// --- quantum_hadamard_transform ---

quantum_hadamard_transform:
    stp x29, x30, [sp, #-16]!     // save fp, lr
    mov x29, sp                   // set new fp

    // save callee-saved regs
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!

    mov x19, x0                   // x19: state vector ptr
    mov x20, x1                   // x20: num qubits
    mov x21, x2                   // x21: target qubit

    mov x9, #1
    lsl x22, x9, x20              // x22: total states (2^n)

    // stride for target qubit
    lsl x23, x9, x21              // x23: 2^target_qubit

    // load hadamard constant 1/sqrt(2)
    ldr x9, =sqrt_half
    ldr d15, [x9]
    dup v15.2d, d15               // broadcast to vector

    // loop through the state vector
    mov x24, #0                   // i = 0
.hadamard_loop:
    cmp x24, x22
    b.ge .hadamard_done

    // if target bit is 0, process this pair
    lsr x9, x24, x21
    tst x9, #1
    b.ne .hadamard_skip

    // get index for |1> component
    add x10, x24, x23             // j = i + stride

    // calculate addresses
    add x11, x19, x24, lsl #4     // addr of state[i]
    add x12, x19, x10, lsl #4     // addr of state[j]

    // load amplitudes
    ldr q0, [x11]                 // q0 = state[i] (real, imag)
    ldr q1, [x12]                 // q1 = state[j] (real, imag)

    // calculate new amplitudes
    // new i = (a + b) / sqrt(2)
    // new j = (a - b) / sqrt(2)
    mov v2.16b, v0.16b            // temp for a
    fadd v0.2d, v0.2d, v1.2d      // a + b
    fmul v0.2d, v0.2d, v15.2d     // (a + b) * constant

    fsub v2.2d, v2.2d, v1.2d      // a - b
    fmul v2.2d, v2.2d, v15.2d     // (a - b) * constant

    // store new amplitudes
    str q0, [x11]
    str q2, [x12]

.hadamard_skip:
    add x24, x24, #1
    b .hadamard_loop

.hadamard_done:
    // restore regs
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// --- quantum_pauli_x_gate ---

quantum_pauli_x_gate:
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!

    mov x19, x0                   // x19: state vector ptr
    mov x20, x1                   // x20: num qubits
    mov x21, x2                   // x21: target qubit

    mov x9, #1
    lsl x22, x9, x20              // x22: 2^n
    lsl x23, x9, x21              // x23: 2^target_qubit (stride)

    mov x24, #0                   // i = 0
.pauli_x_loop:
    cmp x24, x22
    b.ge .pauli_x_done

    // if target bit is 0, process this pair once
    lsr x9, x24, x21
    tst x9, #1
    b.ne .pauli_x_skip

    add x10, x24, x23             // j = i + stride

    add x11, x19, x24, lsl #4     // addr of state[i]
    add x12, x19, x10, lsl #4     // addr of state[j]

    // swap complex amplitudes
    ldr q0, [x11]
    ldr q1, [x12]
    str q1, [x11]
    str q0, [x12]

.pauli_x_skip:
    add x24, x24, #1
    b .pauli_x_loop

.pauli_x_done:
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// --- quantum_pauli_z_gate ---

quantum_pauli_z_gate:
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!

    mov x19, x0                   // x19: state vector ptr
    mov x20, x1                   // x20: num qubits
    mov x21, x2                   // x21: target qubit

    mov x9, #1
    lsl x22, x9, x20              // x22: 2^n

    mov x24, #0                   // i = 0
.pauli_z_loop:
    cmp x24, x22
    b.ge .pauli_z_done

    // if target bit is 1, negate the amplitude
    lsr x9, x24, x21
    tst x9, #1
    b.eq .pauli_z_skip

    add x11, x19, x24, lsl #4     // addr of state[i]
    ldr q0, [x11]
    fneg v0.2d, v0.2d             // negate both real and imag
    str q0, [x11]

.pauli_z_skip:
    add x24, x24, #1
    b .pauli_z_loop

.pauli_z_done:
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// --- quantum_phase_shift ---

quantum_phase_shift:
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!

    mov x19, x0                   // x19: state vector
    mov x20, x1                   // x20: num qubits
    mov x21, x2                   // x21: target qubit
    fmov d8, d0                   // d8: phi

    // calculate e^(i*phi) = cos(phi) + i*sin(phi)
    // calls placeholder function for sincos
    fmov d0, d8
    bl get_sincos_from_phi
    fmov d9, d0                   // d9: cos(phi)
    fmov d10, d1                  // d10: sin(phi)

    mov x9, #1
    lsl x22, x9, x20              // x22: 2^n

    mov x24, #0                   // i = 0
.phase_shift_loop:
    cmp x24, x22
    b.ge .phase_shift_done

    // if target bit is 1, apply phase shift
    lsr x9, x24, x21
    tst x9, #1
    b.eq .phase_shift_skip

    // get amplitude c = c_real + i*c_imag
    add x11, x19, x24, lsl #4     // addr of state[i]
    ldr d0, [x11]                 // c_real
    ldr d1, [x11, #8]             // c_imag

    // calculate c * e^(i*phi)
    fmov d2, d0
    fmul d0, d0, d9               // c_real * cos(phi)
    fmov d3, d1
    fmul d3, d3, d10              // c_imag * sin(phi)
    fsub d0, d0, d3               // new real

    fmul d2, d2, d10              // c_real * sin(phi)
    fmul d1, d1, d9               // c_imag * cos(phi)
    fadd d1, d1, d2               // new imag

    // store back
    str d0, [x11]
    str d1, [x11, #8]

.phase_shift_skip:
    add x24, x24, #1
    b .phase_shift_loop

.phase_shift_done:
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// placeholder for a scalar sincos function
get_sincos_from_phi:
    // stub
    fmov d0, #1.0
    fmov d1, #0.0
    ret

// --- quantum_cnot_operation ---

quantum_cnot_operation:
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!

    mov x19, x0                   // x19: state vector
    mov x20, x1                   // x20: num qubits
    mov x21, x2                   // x21: control qubit
    mov x22, x3                   // x22: target qubit

    mov x9, #1
    lsl x23, x9, x20              // x23: 2^n

    // masks for control and target bits
    lsl x10, x9, x21              // x10: control mask
    lsl x11, x9, x22              // x11: target mask

    mov x24, #0                   // i = 0
.cnot_loop:
    cmp x24, x23
    b.ge .cnot_done

    // if control bit is 0, skip
    tst x24, x10
    b.eq .cnot_skip

    // process pair only when target bit is 0
    tst x24, x11
    b.ne .cnot_skip

    // index of paired state
    eor x12, x24, x11

    // addresses of the two states
    add x13, x19, x24, lsl #4
    add x14, x19, x12, lsl #4

    // swap the complex amplitudes
    ldr q0, [x13]
    ldr q1, [x14]
    str q1, [x13]
    str q0, [x14]

.cnot_skip:
    add x24, x24, #1
    b .cnot_loop

.cnot_done:
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// --- stubs for other gates ---

quantum_swap_gate:
    // stub: not implemented
    ret

quantum_toffoli_gate:
    // stub: not implemented
    ret
