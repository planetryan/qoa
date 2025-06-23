![qoa-logo](https://github.com/user-attachments/assets/5fdbae92-68f8-490e-8368-f3fd6c81a064)

# QOA
The Quantum Optical Assembly Programming Language

# Changelog

see changelog.md for detailed version history and isa updates.

i have created this guide for researchers, developers and students who will use qoa in practical applications. i have designed it to interact with quantum and/or optical systems in the same way classical risc syntax based assembly could manipulate electrons in transistors. i hope whoever reads this guide finds it useful.

sincerely, rayan

---

# Base Syntax / Operations Overview:

(not done yet, more operations might be added in the future)

## Quantum Operations

* **init qubit n**: initializes qubit n to state $|0\rangle$.
* **apply_gate g n**: applies a unitary gate g to qubit/mode n.
* **entangle n m**: entangles qubits/modes n and m.
* **measure n**: measures qubit/mode n, collapsing the quantum state.
* **phase_shift n value**: applies a phase shift to qubit/mode n by value radians.
* **reset n**: resets qubit/mode n to the ground state.
* **load n data**: loads classical or quantum data into qubit/mode n.
* **store n dest**: stores measurement result from qubit/mode n into classical memory at dest.
* **swap n m**: swaps the quantum or optical states between n and m.
* **entangle_bell n m**: creates a bell state between n and m.
* **apply_hadamard n**: applies a hadamard gate on qubit/mode n.
* **controlled_not control target**: executes a controlled not gate.
* **entangle_multi n1 n2 ... nN**: creates a multi-qubit entangled state across qubits or modes n1 through nN.
* **apply_rotation n axis angle**: applies rotation around specified axis (x, y, or z) on qubit/mode n by angle radians.
* **reset_all**: resets all qubits or modes in the system to the ground state.
* **entangle_swap n1 n2 n3 n4**: performs entanglement swapping operation among four qubits/modes.
* **error_correct n code**: invokes quantum error correction on qubit/mode n using specified code.
* **apply_phase_flip n**: applies a phase-flip (z) gate on qubit or mode n.
* **apply_bit_flip n**: applies a bit-flip (x) gate on qubit or mode n.
* **apply_t_gate n**: applies t gate ($\pi/8$ phase) on qubit or mode n.
* **apply_s_gate n**: applies s gate ($\pi/4$ phase) on qubit or mode n.
* **measure_in_basis n basis**: measures qubit/mode n in a specified basis.
* **decohore_protect n duration**: activates decoherence protection for qubit/mode n.
* **feedback_control n measurement_reg**: performs classical feedback control on qubit/mode n.
* **entangle_cluster n1 n2 ... nN**: generates a cluster state over qubits or modes.
* **apply_cphase control target angle**: controlled phase gate between control and target.
* **apply_qnd_measurement n dest**: quantum non-demolition measurement.
* **error_syndrome n code dest**: extracts error syndrome and stores in classical register.
* **controlled_swap control target1 target2**: controlled swap (fredkin) gate.
* **apply_feedforward_gate n control_reg**: applies a gate controlled by classical register.
* **apply_multi_qubit_rotation n1 n2 ... nN axis angles**: simultaneous rotations.
* **apply_measurement_basis_change n basis**: changes measurement basis dynamically.
* **controlled_phase_rotation control target angle**: conditional phase rotation.

## Optical Operations

* **photon_emit n**: emits a photon into optical mode n.
* **photon_detect n**: detects a photon in optical mode n.
* **photon_route n src dest**: routes photon from src to dest.
* **photon_count n dest**: counts detected photons in mode n.
* **apply_displacement n value**: displacement operation.
* **apply_squeezing n value**: squeezing operation.
* **measure_parity n**: measures parity in optical mode n.
* **photon_loss_simulate n rate duration**: simulates photon loss.
* **apply_kerr_nonlinearity n strength duration**: applies kerr nonlinearity.
* **time_delay n duration**: controlled time delay.
* **photon_bunching_control n enable**: toggles photon bunching control.
* **single_photon_source_on n**: activates single-photon source.
* **single_photon_source_off n**: deactivates single-photon source.
* **apply_linear_optical_transform matrix_addr src_modes dest_modes count**: linear optical transformation.
* **photon_detect_coincidence n1 n2 ... nN dest**: detects coincidence events.
* **apply_displacement_feedback n feedback_reg**: displacement based on feedback.
* **photon_detect_with_threshold n threshold dest**: conditional detection.
* **optical_switch_control n state**: controls optical switch.
* **apply_nonlinear_sigma n param**: custom nonlinear operation.
* **measure_with_delay n delay dest**: delayed measurement.
* **photon_loss_correction n code**: initiates loss error correction.
* **photon_emission_pattern n pattern duration**: emits photons in a pattern.
* **apply_squeezing_feedback n feedback_reg**: feedback-based squeezing.
* **apply_photon_subtraction n**: photon subtraction operation.
* **photon_addition n**: photon addition operation.

## Classical / Control flow Operations

* **wait cycles**: idles the system.
* **add n m dest**: classical addition.
* **sub n m dest**: classical subtraction.
* **and n m dest**: logical AND.
* **or n m dest**: logical OR.
* **xor n m dest**: logical XOR.
* **not n dest**: logical NOT.
* **jump label**: unconditional jump.
* **jump_if_zero n label**: conditional jump if zero.
* **jump_if_one n label**: conditional jump if one.
* **call label**: subroutine call.
* **return**: return from subroutine.
* **push n**: push to stack.
* **pop n**: pop from stack.
* **load_mem addr dest**: load from memory.
* **store_mem src addr**: store to memory.
* **load_classical src dest**: load from external classical source.
* **store_classical src dest**: store to external classical memory.
* **apply_measurement_basis_change n basis**: change measurement basis.
* **apply_feedforward_gate n control_reg**: dynamic gate application.

---

# Example code: Practical applications of QOA

## Example 1: three qubit quantum fourier transform

```
; QOA PROGRAM: QUANTUM FOURIER TRANSFORM WITH OPTICAL READOUT

INIT QUBIT Q0
INIT QUBIT Q1
INIT QUBIT Q2

LOAD Q0 0
LOAD Q1 1
LOAD Q2 0

APPLY_HADAMARD Q0
APPLY_HADAMARD Q1
APPLY_HADAMARD Q2

CONTROLLED_PHASE_ROTATION Q1 Q0 PI/2
CONTROLLED_PHASE_ROTATION Q2 Q0 PI/4
CONTROLLED_PHASE_ROTATION Q2 Q1 PI/2

SWAP Q0 Q2

MEASURE Q0
STORE Q0 R0
MEASURE Q1
STORE Q1 R1
MEASURE Q2
STORE Q2 R2

PHOTON_EMIT O0
PHOTON_EMIT O1

LOAD_CLASSICAL R0 O0
LOAD_CLASSICAL R1 O1

PHOTON_DETECT_COINCIDENCE O0 O1 O2 R3
STORE_CLASSICAL R3 0X1000

LOAD R4 1
AND R0 R1 R5
JUMP_IF_ONE R5 TRIGGER_CONTROL
TRIGGER_CONTROL:

HALT
```

## Example 2: optical network switch
```
; CONVERTS INCOMING OPTICAL SIGNALS INTO ELECTRICAL ONES BASED ON ROUTING LOGIC

LOAD R0 0
LOAD R1 1

PHOTON_EMIT O0
PHOTON_EMIT O1
PHOTON_EMIT O2

PHOTON_DETECT O0
PHOTON_DETECT O1
PHOTON_DETECT O2

ADD R1 0 R3
JUMP_IF_ZERO R3 ROUTE0
ADD R1 1 R3
JUMP_IF_ZERO R3 ROUTE1
ADD R1 2 R3
JUMP_IF_ZERO R3 ROUTE2
JUMP END

ROUTE0:
PHOTON_DETECT O0
STORE_CLASSICAL O0 0X1000
JUMP END

ROUTE1:
PHOTON_DETECT O1
STORE_CLASSICAL O1 0X1000
JUMP END

ROUTE2:
PHOTON_DETECT O2
STORE_CLASSICAL O2 0X1000
JUMP END

HALT
```
## Example 3: shor's algorithm

```
; INITIALIZE REGISTERS
INIT QUBIT Q0 - Q999
INIT QUBIT QTEMP0 - QTEMP999
INIT QUBIT QRES0 - QRES255

; LOAD TARGET RSA MODULUS FROM QUANTUM MEMORY
LOAD Q0 - Q255 0X2000 ; LOAD Q0-Q255 FROM MEMORY STARTING AT 0X2000

; PICK A RANDOM COPRIME 'A' < N (MODULUS), HARDCODED OR LOADED
LOAD Q256 - Q511 0X3000 ; LOAD Q256-Q511 FROM MEMORY STARTING AT 0X3000

; COMPUTE MODULAR EXPONENTIATION OF A0^X MOD N USING QUANTUM CIRCUITS
; THIS WOULD INVOLVE MANY INDIVIDUAL GATES AND ARITHMETIC OPERATIONS
; NO DIRECT ISA EQUIVALENT FOR COMPLEX QMOD_EXP, SO REPRESENTING ABSTRACTLY.
; APPLY_GATE MODULAR_EXPONENTIATION_CIRCUIT Q0 Q256 Q0 QTEMP0

; PERFORM QUANTUM FOURIER TRANSFORM TO EXTRACT PERIOD
; THIS WOULD INVOLVE MANY APPLY_HADAMARD AND CONTROLLED_PHASE_ROTATION GATES.
; APPLY_HADAMARD Q0
; CONTROLLED_PHASE_ROTATION Q1 Q0 PI/2
; ... ETC. ...

; MEASURE Q0 INTO CLASSICAL REGISTER TO EXTRACT PERIOD GUESS
MEASURE Q0 - Q999 ; MEASURE ALL QUBITS Q0-Q999
STORE Q0 - Q999 R0 - R999 ; STORE MEASUREMENTS FROM Q0-Q999 INTO R0-R999

; PERFORM CLASSICAL POST-PROCESSING (GCD ETC.) NOT SHOWN HERE
; CLASSICAL_GCD IS A HIGH-LEVEL CLASSICAL OPERATION, NOT IN QOA ISA.
; THIS WOULD BE DONE BY AN EXTERNAL CLASSICAL CONTROLLER OR A SEPARATE PROGRAM.

; FINAL QUANTUM CLEANUP (OPTIONAL)
RESET Q0 - Q999
RESET QTEMP0 - QTEMP999
RESET QRES0 - QRES255

HALT
```
### This is all for now. qoa is still being modeled and developed solely by me, and any practical applications would not be relevant until optical and quantum systems become more commonplace. as these technologies mature and gain wider adoption, qoa aims to provide a unified low-level assembly language capable of efficiently programming and controlling purely optical, purely quantum, or hybrid quantum-optical computing platforms. until then, qoa remains my theoretical passion project focused on laying the groundwork for future advancements in these future fields of computing.

## Thanks for reading! 
### -- rayan
