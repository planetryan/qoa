.intel_syntax noprefix
.section .data
    .align 32
    # constant for 1/sqrt(2) used in hadamard transform
    sqrt_half:      .double 0.70710678118654752440

.section .text
    .global quantum_hadamard_transform
    .global quantum_pauli_x_gate
    .global quantum_pauli_z_gate
    .global quantum_phase_shift
    .global quantum_cnot_operation
    .global quantum_swap_gate
    .global quantum_toffoli_gate


quantum_hadamard_transform:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14
    push r15

    mov r12, rdi                # state vector pointer
    mov r13, rsi                # number of qubits
    mov r14, rdx                # target qubit

    mov rax, 1
    shl rax, r13                # total number of states (2^n)
    mov r15, rax                # save total states

    # calculate the stride for the target qubit
    mov rax, 1
    shl rax, r14                # 2^target_qubit
    mov rbx, rax                # rbx is the stride

    # load hadamard constant 1/sqrt(2)
    movsd xmm15, [rip + sqrt_half]

    # iterate through the state vector
    xor rcx, rcx                # loop counter i = 0
.hadamard_loop:
    cmp rcx, r15
    jge .hadamard_done

    # check if the i-th bit (corresponding to the target qubit) is 0
    mov rax, rcx
    shr rax, r14
    test rax, 1
    jnz .hadamard_skip           # if the target bit is 1, we've already processed this pair

    # calculate index for the |0> component: i
    # calculate index for the |1> component: j = i + stride
    mov rax, rcx
    add rax, rbx                # j = i + 2^target_qubit

    # indices are for complex numbers, so multiply by 16 (2 doubles per complex)
    lea rdi, [r12 + rcx * 16]    # address of state[i]
    lea rsi, [r12 + rax * 16]    # address of state[j]

    # load the amplitudes for state |...0...> and |...1...>
    movupd xmm0, [rdi]          # a = state[i] (real and imag)
    movupd xmm1, [rsi]          # b = state[j] (real and imag)

    # calculate new amplitudes
    # state'[i] = (a + b) / sqrt(2)
    # state'[j] = (a - b) / sqrt(2)
    movapd xmm2, xmm0           # temp for a
    addpd xmm0, xmm1            # a + b
    mulpd xmm0, xmm15           # (a + b) / sqrt(2)

    subpd xmm2, xmm1            # a - b
    mulpd xmm2, xmm15           # (a - b) / sqrt(2)

    # store new amplitudes back
    movupd [rdi], xmm0
    movupd [rsi], xmm2

.hadamard_skip:
    inc rcx
    jmp .hadamard_loop

.hadamard_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    ret

quantum_pauli_x_gate:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14
    push r15

    mov r12, rdi                # state vector pointer
    mov r13, rsi                # number of qubits
    mov r14, rdx                # target qubit

    mov rax, 1
    shl rax, r13                # 2^n
    mov r15, rax

    mov rax, 1
    shl rax, r14                # 2^target_qubit
    mov rbx, rax                # stride

    xor rcx, rcx
.pauli_x_loop:
    cmp rcx, r15
    jge .pauli_x_done

    mov rax, rcx
    shr rax, r14
    test rax, 1
    jnz .pauli_x_skip

    mov rax, rcx
    add rax, rbx

    lea rdi, [r12 + rcx * 16]
    lea rsi, [r12 + rax * 16]

    # swap the complex amplitudes
    movupd xmm0, [rdi]
    movupd xmm1, [rsi]
    movupd [rdi], xmm1
    movupd [rsi], xmm0

.pauli_x_skip:
    inc rcx
    jmp .pauli_x_loop

.pauli_x_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    ret

quantum_pauli_z_gate:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14
    push r15

    mov r12, rdi                # state vector pointer
    mov r13, rsi                # number of qubits
    mov r14, rdx                # target qubit

    mov rax, 1
    shl rax, r13                # 2^n
    mov r15, rax

    # create a negation mask: -1.0 for both real and imag parts
    pcmpeqd xmm14, xmm14        # all 1s
    psrldq xmm14, 8             # shift right to get 0x0000000000000000FFFFFFFFFFFFFFFF
    psllq xmm14, 63             # shift left to get sign bit 0x8000...
    movapd xmm15, xmm14         # copy sign bit to both double slots

    xor rcx, rcx
.pauli_z_loop:
    cmp rcx, r15
    jge .pauli_z_done

    # check if the target qubit bit is 1
    mov rax, rcx
    shr rax, r14
    test rax, 1
    jz .pauli_z_skip           # if bit is 0, do nothing

    # if bit is 1, negate the amplitude
    lea rdi, [r12 + rcx * 16]
    movupd xmm0, [rdi]
    xorpd xmm0, xmm15           # flip sign bits (negate)
    movupd [rdi], xmm0

.pauli_z_skip:
    inc rcx
    jmp .pauli_z_loop

.pauli_z_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    ret


quantum_phase_shift:
    push rbp
    mov rbp, rsp
    sub rsp, 16                 # space for saving xmm registers
    push r12
    push r13
    push r14
    push r15

    mov r12, rdi                # state vector
    mov r13, rsi                # num qubits
    mov r14, rdx                # target qubit
    movsd [rsp], xmm0           # save phi

    # calculate e^(i*phi) = cos(phi) + i*sin(phi)
    # we need a sincos function here. for now, let's assume we can get them.
    # placeholder for calling a sincos function.
    # call __svml_sincos (or similar)
    # let's assume xmm1 = cos(phi), xmm2 = sin(phi)
    call get_sincos_from_phi
    movsd xmm13, xmm1           # cos(phi)
    movsd xmm14, xmm2           # sin(phi)

    mov rax, 1
    shl rax, r13                # 2^n
    mov r15, rax

    xor rcx, rcx
.phase_shift_loop:
    cmp rcx, r15
    jge .phase_shift_done

    # if target bit is 1, apply the phase shift
    mov rax, rcx
    shr rax, r14
    test rax, 1
    jz .phase_shift_skip

    # get amplitude c = c_real + i*c_imag
    lea rdi, [r12 + rcx * 16]
    movsd xmm0, [rdi]           # c_real
    movsd xmm1, [rdi + 8]       # c_imag

    # calculate c * e^(i*phi) = (c_real + i*c_imag) * (cos(phi) + i*sin(phi))
    # real = c_real*cos(phi) - c_imag*sin(phi)
    # imag = c_real*sin(phi) + c_imag*cos(phi)
    movsd xmm2, xmm0            # copy c_real
    mulsd xmm0, xmm13           # c_real * cos(phi)
    movsd xmm3, xmm1            # copy c_imag
    mulsd xmm3, xmm14           # c_imag * sin(phi)
    subsd xmm0, xmm3            # new real part

    mulsd xmm2, xmm14           # c_real * sin(phi)
    mulsd xmm1, xmm13           # c_imag * cos(phi)
    addsd xmm1, xmm2            # new imag part

    # store back
    movsd [rdi], xmm0
    movsd [rdi + 8], xmm1

.phase_shift_skip:
    inc rcx
    jmp .phase_shift_loop

.phase_shift_done:
    pop r15
    pop r14
    pop r13
    pop r12
    add rsp, 16
    pop rbp
    ret

get_sincos_from_phi:
    # this is a placeholder. a real implementation would call a proper
    # sincos function like the one in main.s, but adapted for a single double.
    # input: xmm0 = phi
    # output: xmm1 = cos(phi), xmm2 = sin(phi)
    fld qword ptr [rsp]         # load phi onto fpu stack
    fsincos
    fstp qword ptr [rsp+8]      # store cos(phi)
    fstp qword ptr [rsp]        # store sin(phi)
    movsd xmm1, [rsp+8]         # load cos(phi) into xmm1
    movsd xmm2, [rsp]           # load sin(phi) into xmm2
    ret



quantum_cnot_operation:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14
    push r15
    push rbx

    mov r12, rdi                # state vector
    mov r13, rsi                # num qubits
    mov r14, rdx                # control qubit
    mov r15, rcx                # target qubit

    mov rax, 1
    shl rax, r13                # 2^n
    mov rdi, rax                # total states

    # create masks for control and target bits
    mov rax, 1
    shl rax, r14
    mov rbx, rax                # control mask

    mov rax, 1
    shl rax, r15
    mov r10, rax                # target mask

    xor rcx, rcx
.cnot_loop:
    cmp rcx, rdi
    jge .cnot_done

    # check if control bit is set in the current state index
    mov rax, rcx
    and rax, rbx
    jz .cnot_skip               # if control bit is 0, do nothing

    # if control bit is 1, we swap the target qubit's amplitudes
    # the paired state is found by flipping the target bit
    mov rax, rcx
    xor rax, r10                # index of paired state
    
    # only process pairs once (when target bit is 0)
    test rcx, r10
    jnz .cnot_skip

    # get addresses of the two states to swap
    lea r8, [r12 + rcx * 16]
    lea r9, [r12 + rax * 16]

    # swap the complex amplitudes
    movupd xmm0, [r8]
    movupd xmm1, [r9]
    movupd [r8], xmm1
    movupd [r9], xmm0

.cnot_skip:
    inc rcx
    jmp .cnot_loop

.cnot_done:
    pop rbx
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    ret



quantum_swap_gate:
    # stub: implementation is similar to cnot but swaps based on
    # states where the two qubits differ.
    ret


quantum_toffoli_gate:
    # stub: implementation would check if both control bits are 1
    # and then perform a pauli-x on the target qubit.
    ret
