![qoa-logo](https://github.com/user-attachments/assets/5fdbae92-68f8-490e-8368-f3fd6c81a064)

# QOA  
**The Quantum Optical Assembly Programming Language**

---

## NOTE: EXTRA DOCUMENTATION CAN BE FOUND IN THE README FOLDER

## Changelog  
LATEST RELEASE NOTES: [HERE](<changelog/QOA v0.3.0.md>)

See `changelog` folder for more detailed version history and ISA updates.

---

I have created this guide for researchers, developers and students who will use QOA in practical applications. I have designed it to interact with quantum and/or optical systems in the same way classical RISC‑syntax based assembly could manipulate electrons in transistors. I hope whoever reads this guide finds it useful.

*Sincerely, Rayan*

## System Requirements:
- **CPU:** Processors with SIMD support (AVX-512/AVX2 recommended in x86_64 for best performance, NEON for AARCH64, and RVV 1.0 for RISC-V 64 bit).
- **CPU (continued):** Please try to run QOA on A CPU with as large as a cache as possible, AMD X3D cpus are a good option, I explained why [here](readme/QOA-system-requirements.md)
- **Memory:** Increased efficiency allows for simulation of larger quantum systems with the same memory footprint within 2^N doubling state requirements, so still exponential memory, but less compiler & executor overhead
- **Rust Compiler:** Requires Rust 1.55.0 or higher for portable SIMD support, crates may also need an update.
- **Rust Compiler Note:** USE `rustc 1.90.0-nightly (a7a1618e6 2025-07-22)` IF ISSUES ARISE! 

---

## Base Syntax / Operations Overview  
*(This list is comprehensive for QOA v0.3.0)*

### Quantum Operations
- **QINIT N**  
  Initializes the quantum state with N qubits, all set to the ∣0⟩ state.

- **H Q**  
  Applies a Hadamard gate to qubit Q.

- **APPLYBITFLIP Q**  
  Applies a Pauli‑X (bit‑flip) gate to qubit Q.

- **APPLYPHASEFLIP Q**  
  Applies a Pauli‑Z (phase‑flip) gate to qubit Q.

- **APPLYTGATE Q**  
  Applies a T gate (π/8 phase shift) to qubit Q.

- **APPLYSGATE Q**  
  Applies an S gate (π/4 phase shift) to qubit Q.

- **PHASESHIFT Q Angle**  
  Applies a phase shift to qubit Q by *Angle* radians.  
  Alias: `SETPHASE`.

- **RX Q Angle**  
  Rotation around the X‑axis on qubit Q by *Angle* radians.

- **RY Q Angle**  
  Rotation around the Y‑axis on qubit Q by *Angle* radians.

- **RZ Q Angle**  
  Rotation around the Z‑axis on qubit Q by *Angle* radians.

- **CONTROLLEDNOT C T**  
  CNOT gate with control qubit C and target qubit T.  
  Alias: `CNOT`.

- **CZ C T**  
  Controlled‑Z gate with control qubit C and target qubit T.

- **CPHASE C T Angle**  
  Controlled phase rotation between control C and target T by *Angle* radians.  
  Alias: `APPLYCPHASE`.

- **ENTANGLE C T**  
  Entangles qubits C and T (often implemented as a CNOT).

- **ENTANGLEBELL Q1 Q2**  
  Creates a Bell state between qubits Q1 and Q2.

- **ENTANGLEMULTI N Q1 ... QN**  
  Creates a multi‑qubit entangled state across N specified qubits.

- **ENTANGLECLUSTER N Q1 ... QN**  
  Generates a cluster state over N specified qubits.

- **ENTANGLESWAP Q1 Q2 Q3 Q4**  
  Performs entanglement swapping among four qubits.

- **ENTANGLESWAPMEASURE Q1 Q2 Q3 Q4 Label**  
  Entanglement swapping with measurement, jumping to *Label* based on outcome.

- **ENTANGLEWITHFB Q Label**  
  Entangles qubit Q with classical feedback, jumping to *Label*.

- **ENTANGLEDISTRIB Q Label**  
  Performs distributed entanglement on qubit Q, jumping to *Label*.

- **MEASURE Q**  
  Measures qubit Q, collapsing its quantum state and yielding a classical result.  
  Alias: `QMEAS`.

- **MEASUREINBASIS Q Label**  
  Measures qubit Q in a specified basis, jumping to *Label*.

- **RESET Q**  
  Resets qubit Q to the ground state ∣0⟩.

- **RESETALL**  
  Resets all qubits in the system to ∣0⟩.

- **MARKOBSERVED Q**  
  Marks qubit Q as observed (internal state tracking).

- **RELEASE Q**  
  Releases resources associated with qubit Q.

- **APPLYGATE Q GateID**  
  Applies a named unitary gate (e.g., “h”, “x”, “cz”) to qubit Q.

- **APPLYROTATION Q Angle**  
  General rotation to qubit Q by *Angle* (axis undefined).

- **APPLYMULTIQUBITROTATION AxisID N Q1 ... QN Angles**  
  Simultaneous rotations around a specified AxisID to N qubits with corresponding Angles.

- **APPLYKERRNONLIN Q Strength Duration**  
  Applies Kerr nonlinearity to qubit/mode Q with given Strength and Duration.

- **DECOHERENCEPROTECT Q Duration**  
  Activates decoherence protection for qubit Q for *Duration*.

- **BASISCHANGE Q Label**  
  Changes the measurement basis for qubit Q, jumping to *Label*.

- **APPLYNONLINEARPHASESHIFT Q Angle**  
  Nonlinear phase shift to qubit Q by *Angle*.

- **APPLYNONLINEARSI Q Param**  
  Custom nonlinear operation to qubit Q with *Param*.

- **QSTATETOMOGRAPHY Q Label**  
  Quantum state tomography on qubit Q, jumping to *Label*.

- **BELLSTATEVERIF Q1 Q2 Label**  
  Verifies Bell state between Q1 and Q2, jumping to *Label*.

- **QUANTUMZENOEFFECT Q Strength Duration**  
  Simulates Quantum Zeno Effect on qubit Q.

- **APPLYLINEAROPTICALTRANSFORM N_in N_out TransformID In_Modes Out_Modes**  
  Applies a linear optical transformation.

- **ERRORCORRECT Q CodeID**  
  Invokes quantum error correction on qubit Q using CodeID.

- **ERRORSYNDROME Q CodeID RegName**  
  Extracts error syndrome for qubit Q using CodeID, stores in *RegName*.

- **QNDMEASURE Q RegName**  
  Quantum non‑demolition measurement on qubit Q, stores result in *RegName*.

- **SWAP Q1 Q2**  
  Swaps the states of qubit Q1 and Q2.

---

### Optical Operations
- **PHOTONEMIT MODE**  
  Emits a photon into optical mode *MODE*.

- **PHOTONDETECT MODE**  
  Detects a photon in optical mode *MODE*.

- **PHOTONROUTE MODE FROM TO**  
  Routes a photon from *FROM* mode to *TO* mode.

- **PHOTONCOUNT MODE REGNAME**  
  Counts detected photons in mode *MODE*, stores count in *REGNAME*.

- **APPLYDISPLACEMENT MODE VALUE**  
  Applies a displacement to optical mode *MODE* by *VALUE*.

- **APPLYSQUEEZING MODE VALUE**  
  Applies a squeezing operation to optical mode *MODE* by *VALUE*.

- **MEASUREPARITY MODE**  
  Measures parity in optical mode *MODE*.

- **PHOTONLOSSSIMULATE MODE RATE DURATION**  
  Simulates photon loss in mode *MODE* with *RATE* over *DURATION*.

- **TIMEDELAY MODE DURATION**  
  Applies a controlled time delay to optical mode *MODE*.

- **PHOTONBUNCHINGCTL MODE ENABLE**  
  Toggles photon bunching control for mode *MODE* (`0` or `1`).

- **SINGLEPHOTONSOURCEON/OFF MODE**  
  Activates/deactivates a single‑photon source for mode *MODE*.

- **PHOTONDETECTCOINCIDENCE N MODE1 ... MODEN REGNAME**  
  Detects coincidence events among N modes, stores result in *REGNAME*.

- **APPLYDISPLACEMENTOP MODE ALPHA_RE ALPHA_IM**  
  Displacement operator with complex α (real, imag).

- **OPTICALSWITCHCONTROL MODE STATE**  
  Controls an optical switch on mode *MODE* to *STATE* (`0` or `1`).

- **MEASUREWITHDELAY MODE DELAY REGNAME**  
  Delayed measurement on mode *MODE*, stores in *REGNAME*.

- **PHOTONLOSSCORR MODE CodeID**  
  Photon loss error correction on mode *MODE* using *CodeID*.

- **PHOTONEMISSIONPATTERN MODE PatternID CYCLES**  
  Emits photons according to *PatternID* for *CYCLES*.

- **APPLYSQUEEZINGFEEDBACK MODE FeedbackRegName**  
  Applies squeezing feedback based on *FeedbackRegName*.

- **APPLYPHOTONSUBTRACTION MODE**  
  Photon subtraction operation on mode *MODE*.

- **PHOTONADDITION MODE**  
  Photon addition operation on mode *MODE*.

- **PNRDETECTION MODE**  
  Photon Number Resolving Detection on mode *MODE*.

- **SETOPTICALATTENUATION MODE VALUE**  
  Sets optical attenuation to *VALUE*.

- **DYNAMICPHASECOMP MODE VALUE**  
  Dynamic phase compensation by *VALUE*.

- **CROSSPHASEMOD MODE1 MODE2**  
  Cross‑phase modulation between modes.

- **OPTICALDELAYLINECTL MODE DURATION**  
  Controls an optical delay line.

- **OPTICALROUTING MODE1 MODE2**  
  Routes optical signal between modes.

- **SETPOS Q X Y**  
  Sets spatial position of qubit/mode Q to (X, Y).

- **SETWL Q Wavelength**  
  Sets wavelength of qubit/mode Q.

- **WLSHIFT Q Shift**  
  Shifts wavelength of qubit/mode Q by *Shift*.

- **MOVE Q DX DY**  
  Moves qubit/mode Q by (DX, DY).

---

### Classical / Control Flow Operations
- **HALT**  
  Stops program execution.

- **LOOPSTART Iterations** / **LOOPEND**  
  Begin/end a loop.

- **REGSET Reg Value**  
  Sets a classical register to a floating‑point value.

- **ADD / SUB / MUL / DIV DstReg Src1Reg Src2Reg**  
  Arithmetic operations.

- **COPY DstReg SrcReg**  
  Copy register value.

- **ANDBITS / ORBITS / XORBITS / NOTBITS**  
  Bitwise operations.

- **SHL / SHR DstReg SrcReg ShiftAmount**  
  Bit shifts.

- **CMP Reg1 Reg2**  
  Compare, set flags.

- **JUMP / JMP Label** / **JUMPABS / JMPABS Address**  
  Unconditional jumps.

- **JUMPIFZERO Reg Label**, **JUMPIFONE Reg Label**  
  Conditional jumps on register values.

- **IFEQ / IFNE / IFGT / IFLT Reg1 Reg2 Label**  
  Comparison-based jumps.

- **CALL / CALLADDR Label/Address**, **RETSUB**  
  Subroutine calls and returns.

- **PUSHREG / POPREG Reg**  
  Stack operations.

- **CHAROUT Q**  
  Measure qubit Q and print as character.

- **INPUT Reg**  
  Read character input.

- **GETTIME Reg**  
  Get current system time.

- **RAND Reg**, **SEEDRNG Seed**  
  Random number operations.

- **PRINTF FormatStringID NumRegs Reg1 ...**, **PRINT StringID**, **PRINTLN StringID**  
  Formatted / literal output.

- **VERBOSELOG Level MessageID**, **COMMENT MessageID**  
  Logging and comments.

- **BREAKPOINT**  
  Insert debug breakpoint.

- **EXITCODE Code**  
  Terminate with exit code.

- **STORECLASSICAL Reg Address**  
  Store register into classical memory.

---

## Example Code

### Example 1: Three‑Qubit Quantum Fourier Transform
```
; QOA PROGRAM: THREE QUBIT QUANTUM FOURIER TRANSFORM
QINIT 3        ; initialize 3 qubits (Q0, Q1, Q2)

; apply Hadamard to all qubits
H 0
H 1
H 2

; controlled phase rotations
CPHASE 1 0 1.5707963267948966  ; π/2
CPHASE 2 0 0.7853981633974483  ; π/4
CPHASE 2 1 1.5707963267948966  ; π/2

; swap qubits for standard QFT output
SWAP 0 2

; measure all qubits
MEASURE 0
MEASURE 1
MEASURE 2

HALT
```
### Example 2: Optical Network Switch

```
; QOA PROGRAM: OPTICAL NETWORK SWITCH
; this program simulates routing based on a classical control signal.
; it assumes photons are detected and their presence/absence is stored in registers.

; define classical registers for routing logic
REGSET R0 0.0 ; routing destination 0
REGSET R1 1.0 ; routing destination 1
REGSET R2 2.0 ; routing destination 2
REGSET R3 0.0 ; input signal (0, 1, or 2 to choose route)

; simulate incoming photons (emit for demo purposes)
PHOTONEMIT O0 ; incoming optical mode 0
PHOTONEMIT O1 ; incoming optical mode 1
PHOTONEMIT O2 ; incoming optical mode 2

; detect photons in incoming modes (results stored internally)
PHOTONDETECT O0
PHOTONDETECT O1
PHOTONDETECT O2

; load a classical value into R3 to simulate routing decision
; in a real scenario, this might come from an external sensor or control unit.
; for this example, let's hardcode R3 to simulate a choice.
REGSET R3 1.0 ; simulate choosing route 1

; route based on R3's value
; this uses a series of conditional jumps to simulate a switch.
CMP R3 R0 ; compare R3 with 0.0
IFEQ R3 R0 ROUTE0 ; if R3 == R0 (0.0), jump to ROUTE0

CMP R3 R1 ; compare R3 with 1.0
IFEQ R3 R1 ROUTE1 ; if R3 == R1 (1.0), jump to ROUTE1

CMP R3 R2 ; compare R3 with 2.0
IFEQ R3 R2 ROUTE2 ; if R3 == R2 (2.0), jump to ROUTE2

JUMP END_PROGRAM ; if no route matched, end program

; if route 0 is chosen, detect photon from O0 and store its presence
PHOTONDETECT O0 ; detect photon in optical mode 0
PHOTONCOUNT O0 R4 ; count photons in O0, store in R4
STORECLASSICAL R4 0X1000 ; store R4 (photon count) to classical memory at address 0x1000
JUMP END_PROGRAM

; if route 1 is chosen, detect photon from O1 and store its presence
PHOTONDETECT O1 ; detect photon in optical mode 1
PHOTONCOUNT O1 R4 ; count photons in O1, store in R4
STORECLASSICAL R4 0X1001 ; store R4 to classical memory at address 0x1001
JUMP END_PROGRAM

; if route 2 is chosen, detect photon from O2 and store its presence
PHOTONDETECT O2 ; detect photon in optical mode 2
PHOTONCOUNT O2 R4 ; count photons in O2, store in R4
STORECLASSICAL R4 0X1002 ; store R4 to classical memory at address 0x1002
JUMP END_PROGRAM

HALT

```

### Example 3: Shor's Algorithm

```
; QOA PROGRAM: SHOR'S ALGORITHM
; this program outlines the quantum steps of shor's algorithm for factoring a number n.
; it is highly conceptual and not executable for cryptographically relevant numbers
; due to immense qubit and gate requirements for modular exponentiation.

; parameters for factoring n=15 (l=4 bits).
; period register: 2l = 8 qubits (q0-q7).
; function register: l = 4 qubits (q8-q11).
; total qubits for this conceptual example: 12.
QINIT 12 ; initialize 12 qubits to |0...0>

; --- step 1: initialize period-finding register to superposition ---

; apply hadamard to all qubits in the period-finding register (q0-q7)
HAD 0
HAD 1
HAD 2
HAD 3
HAD 4
HAD 5
HAD 6
HAD 7

; function register (q8-q11) is already |0000> by default qinit

; --- step 2: modular exponentiation (u_a^x mod n) ---

; this is the most complex and resource-intensive part of shor's algorithm.
; it computes f(x) = a^x mod n in superposition, where 'a' is a randomly chosen base
; (e.g., a=7 for n=15).
; this involves applying controlled-u operations (u_a^k) to the function register,
; controlled by each qubit in the period register.
; each u_a^k operation is a reversible circuit for modular multiplication.
; implementing this for even n=15 requires decomposing complex arithmetic into
; thousands or millions of elementary qoa gates (cnots, toffolis, adders, multipliers).
; VERBOSELOG 0 "Conceptual Modular Exponentiation (a^x mod N) would be here."
; VERBOSELOG 0 "  Base 'a' = 7, Modulo 'N' = 15."
; VERBOSELOG 0 "  This involves controlled-U_a^k operations for k = 2^0, 2^1, ..., 2^7."
; VERBOSELOG 0 "  Each U_a^k is a complex reversible modular arithmetic circuit."
; VERBOSELOG 0 "  (This block represents immense complexity, not directly representable in QOA)"

; --- step 3: inverse quantum fourier transform (iqft) on period-finding register ---

; the iqft extracts the period 'r' from the superposed state in the period register.
; this is the reverse sequence of gates from the qft.
; VERBOSELOG 0 "Applying Inverse Quantum Fourier Transform (IQFT) on Period Register (Q0-Q7)"

; first, swap qubits to reverse the order from the qft output.
; this is crucial for the standard iqft implementation.
SWAP 0 7 ; swap Q0 and Q7
SWAP 1 6 ; swap Q1 and Q6
SWAP 2 5 ; swap Q2 and Q5
SWAP 3 4 ; swap Q3 and Q4

; now apply the iqft gates. angles are negative of qft angles.
; pi = 3.141592653589793
; pi/2 = 1.5707963267948966
; pi/4 = 0.7853981633974483
; pi/8 = 0.39269908169872414
; pi/16 = 0.19634954084936207
; pi/32 = 0.09817477042468103
; pi/64 = 0.04908738521234051
; pi/128 = 0.02454369260617025

; qubit 0 (now the most significant after swaps, originally Q7)
HAD 0
CPHASE 1 0 -1.5707963267948966 ; cp(-pi/2) q1-q0
CPHASE 2 0 -0.7853981633974483 ; cp(-pi/4) q2-q0
CPHASE 3 0 -0.39269908169872414 ; cp(-pi/8) q3-q0
CPHASE 4 0 -0.19634954084936207 ; cp(-pi/16) q4-q0
CPHASE 5 0 -0.09817477042468103 ; cp(-pi/32) q5-q0
CPHASE 6 0 -0.04908738521234051 ; cp(-pi/64) q6-q0
CPHASE 7 0 -0.02454369260617025 ; cp(-pi/128) q7-q0

; qubit 1 (originally Q6)
HAD 1
CPHASE 2 1 -1.5707963267948966 ; cp(-pi/2) q2-q1
CPHASE 3 1 -0.7853981633974483 ; cp(-pi/4) q3-q1
CPHASE 4 1 -0.39269908169872414 ; cp(-pi/8) q4-q1
CPHASE 5 1 -0.19634954084936207 ; cp(-pi/16) q5-q1
CPHASE 6 1 -0.09817477042468103 ; cp(-pi/32) q6-q1
CPHASE 7 1 -0.04908738521234051 ; cp(-pi/64) q7-q1

; qubit 2 (originally Q5)
HAD 2
CPHASE 3 2 -1.5707963267948966 ; cp(-pi/2) q3-q2
CPHASE 4 2 -0.7853981633974483 ; cp(-pi/4) q4-q2
CPHASE 5 2 -0.39269908169872414 ; cp(-pi/8) q5-q2
CPHASE 6 2 -0.19634954084936207 ; cp(-pi/16) q6-q2
CPHASE 7 2 -0.09817477042468103 ; cp(-pi/32) q7-q2

; qubit 3 (originally Q4)
HAD 3
CPHASE 4 3 -1.5707963267948966 ; cp(-pi/2) q4-q3
CPHASE 5 3 -0.7853981633974483 ; cp(-pi/4) q5-q3
CPHASE 6 3 -0.39269908169872414 ; cp(-pi/8) q6-q3
CPHASE 7 3 -0.19634954084936207 ; cp(-pi/16) q7-q3

; qubit 4 (originally Q3)
HAD 4
CPHASE 5 4 -1.5707963267948966 ; cp(-pi/2) q5-q4
CPHASE 6 4 -0.7853981633974483 ; cp(-pi/4) q6-q4
CPHASE 7 4 -0.39269908169872414 ; cp(-pi/8) q7-q4

; qubit 5 (originally Q2)
HAD 5
CPHASE 6 5 -1.5707963267948966 ; cp(-pi/2) q6-q5
CPHASE 7 5 -0.7853981633974483 ; cp(-pi/4) q7-q5

; qubit 6 (originally Q1)
HAD 6
CPHASE 7 6 -1.5707963267948966 ; cp(-pi/2) q7-q6

; qubit 7 (originally Q0)
HAD 7

; --- step 4: measurement of period-finding register ---

; measure the period-finding register (q0-q7)
MEAS 0
MEAS 1
MEAS 2
MEAS 3
MEAS 4
MEAS 5
MEAS 6
MEAS 7

; --- step 5: classical post-processing ---

; after measurement, a classical computer takes the measured value 'c'
; and uses the continued fractions algorithm to find the period 'r'.
; then, it calculates gcd(a^(r/2) +/- 1, n) to find factors of n.

; VERBOSELOG 0 "Classical Post-Processing: Use continued fractions to find period 'r'."
; VERBOSELOG 0 "Then calculate gcd(a^(r/2) +/- 1, N) to find factors of N."

HALT

```

## This is all for now, if you would like to follow development, please star the repository.
### Thanks for reading!
### -- Rayan (planetryan)