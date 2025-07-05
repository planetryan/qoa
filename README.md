![qoa-logo](https://github.com/user-attachments/assets/5fdbae92-68f8-490e-8368-f3fd6c81a064)

# QOA
The Quantum Optical Assembly Programming Language

# Changelog

see changelog folder for detailed version history and isa updates.

I have created this guide for researchers, developers and students who will use qoa in practical applications. I have designed it to interact with quantum and/or optical systems in the same way classical Risc syntax based assembly could manipulate electrons in transistors. I hope whoever reads this guide finds it useful.

Sincerely, Rayan

---

# Base Syntax / Operations Overview:

(this list is comprehensive for QOA v0.2.8)

## Quantum Operations

* **`QINIT <N>`**: Initializes the quantum state with `N` qubits, all set to the $|0\rangle$ state.
* **`H <Q>`**: Applies a Hadamard gate to qubit `Q`.
* **`APPLYBITFLIP <Q>`**: Applies a Pauli-X (bit-flip) gate to qubit `Q`.
* **`APPLYPHASEFLIP <Q>`**: Applies a Pauli-Z (phase-flip) gate to qubit `Q`.
* **`APPLYTGATE <Q>`**: Applies a T gate ($\pi/8$ phase shift) to qubit `Q`.
* **`APPLYSGATE <Q>`**: Applies an S gate ($\pi/4$ phase shift) to qubit `Q`.
* **`PHASESHIFT <Q> <Angle>`**: Applies a phase shift to qubit `Q` by `Angle` radians.
* **`SETPHASE <Q> <Angle>`**: Alias for `PHASESHIFT`.
* **`RX <Q> <Angle>`**: Applies a rotation around the X-axis on qubit `Q` by `Angle` radians.
* **`RY <Q> <Angle>`**: Applies a rotation around the Y-axis on qubit `Q` by `Angle` radians.
* **`RZ <Q> <Angle>`**: Applies a rotation around the Z-axis on qubit `Q` by `Angle` radians.
* **`CONTROLLEDNOT <C> <T>`**: Applies a CNOT gate with control qubit `C` and target qubit `T`.
* **`CNOT <C> <T>`**: Alias for `CONTROLLEDNOT`.
* **`CZ <C> <T>`**: Applies a Controlled-Z gate with control qubit `C` and target qubit `T`.
* **`CONTROLLEDPHASEROTATION <C> <T> <Angle>`**: Applies a controlled phase rotation between control `C` and target `T` by `Angle` radians.
* **`APPLYCPHASE <C> <T> <Angle>`**: Alias for `CONTROLLEDPHASEROTATION`.
* **`ENTANGLE <C> <T>`**: Entangles qubits `C` and `T` (often implemented as a CNOT).
* **`ENTANGLEBELL <Q1> <Q2>`**: Creates a Bell state between qubits `Q1` and `Q2`.
* **`ENTANGLEMULTI <N> <Q1> ... <QN>`**: Creates a multi-qubit entangled state across `N` specified qubits.
* **`ENTANGLECLUSTER <N> <Q1> ... <QN>`**: Generates a cluster state over `N` specified qubits.
* **`ENTANGLESWAP <Q1> <Q2> <Q3> <Q4>`**: Performs an entanglement swapping operation among four qubits.
* **`ENTANGLESWAPMEASURE <Q1> <Q2> <Q3> <Q4> <Label>`**: Performs entanglement swapping with measurement, jumping to `Label` based on outcome.
* **`ENTANGLEWITHFB <Q> <Label>`**: Entangles qubit `Q` with classical feedback, jumping to `Label`.
* **`ENTANGLEDISTRIB <Q> <Label>`**: Performs distributed entanglement on qubit `Q`, jumping to `Label`.
* **`MEASURE <Q>`**: Measures qubit `Q`, collapsing its quantum state and yielding a classical result.
* **`QMEAS <Q>`**: Alias for `MEASURE`.
* **`MEASUREINBASIS <Q> <Label>`**: Measures qubit `Q` in a specified basis, jumping to `Label`.
* **`RESET <Q>`**: Resets qubit `Q` to the ground state $|0\rangle$.
* **`RESETALL`**: Resets all qubits in the system to the ground state $|0\rangle$.
* **`MARKOBSERVED <Q>`**: Marks qubit `Q` as observed (for internal state tracking).
* **`RELEASE <Q>`**: Releases resources associated with qubit `Q`.
* **`APPLYGATE <Q> <Name>`**: Applies a named unitary gate (`"h"`, `"x"`, `"cz"`) to qubit `Q`.
* **`APPLYROTATION <Q> <Angle>`**: Applies a general rotation to qubit `Q` by `Angle` (axis undefined, use RX/RY/RZ for specific axes).
* **`APPLYMULTIQUBITROTATION <Axis> <N> <Q1> ... <QN> <Angles>`**: Applies simultaneous rotations around a specified `Axis` to `N` qubits with corresponding `Angles`.
* **`APPLYKERRNONLIN <Q> <Strength> <Duration>`**: Applies Kerr nonlinearity to qubit/mode `Q` with given `Strength` and `Duration`.
* **`DECOHERENCEPROTECT <Q> <Duration>`**: Activates decoherence protection for qubit `Q` for a `Duration`.
* **`BASISCHANGE <Q> <Label>`**: Changes the measurement basis for qubit `Q`, jumping to `Label`.
* **`APPLYNONLINEARPHASESHIFT <Q> <Angle>`**: Applies a nonlinear phase shift to qubit `Q` by `Angle`.
* **`APPLYNONLINEARSI <Q> <Param>`**: Applies a custom nonlinear operation to qubit `Q` with `Param`.
* **`QSTATETOMOGRAPHY <Q> <Label>`**: Performs quantum state tomography on qubit `Q`, jumping to `Label`.
* **`BELLSTATEVERIF <Q1> <Q2> <Label>`**: Verifies Bell state between `Q1` and `Q2`, jumping to `Label`.
* **`QUANTUMZENOEFFECT <Q> <Strength> <Duration>`**: Simulates Quantum Zeno Effect on qubit `Q`.
* **`APPLYLINEAROPTICALTRANSFORM <N_in> <N_out> <Name> <In_Modes> <Out_Modes>`**: Applies a linear optical transformation.
* **`ERRORCORRECT <Q> <CodeName>`**: Invokes quantum error correction on qubit `Q` using `CodeName`.
* **`ERRORSYNDROME <Q> <CodeName> <RegName>`**: Extracts error syndrome for qubit `Q` using `CodeName`, stores in `RegName`.
* **`QNDMEASURE <Q> <RegName>`**: Performs quantum non-demolition measurement on qubit `Q`, stores result in `RegName`.

## Optical Operations

* **`PHOTONEMIT <MODE>`**: Emits a photon into optical mode `Mode`.
* **`PHOTONDETECT <MODE>`**: Detects a photon in optical mode `Mode`.
* **`PHOTONROUTE <MODE> <FROM> <TO>`**: Routes a photon from `From` mode to `To` mode.
* **`PHOTONCOUNT <MODE> <REGNAME>`**: Counts detected photons in mode `Mode`, stores count in `RegName`.
* **`APPLYDISPLACEMENT <MODE> <VALUE>`**: Applies a displacement operation to optical mode `Mode` by `Value`.
* **`APPLYSQUEEZING <MODE> <VALUE>`**: Applies a squeezing operation to optical mode `Mode` by `Value`.
* **`MEASUREPARITY <MODE>`**: Measures parity in optical mode `Mode`.
* **`PHOTONLOSSSIMULATE <MODE> <RATE> <DURATION>`**: Simulates photon loss in mode `Mode` with `Rate` over `Duration`.
* **`TIMEDELAY <MODE> <DURATION>`**: Applies a controlled time delay to optical mode `Mode` for `Duration`.
* **`PHOTONBUNCHINGCTL <MODE> <ENABLE>`**: Toggles photon bunching control for mode `Mode` (`Enable` is 0 or 1).
* **`SINGLEPHOTONSOURCEON <MODE>`**: Activates a single-photon source for mode `Mode`.
* **`SINGLEPHOTONSOURCEOFF <MODE>`**: Deactivates a single-photon source for mode `Mode`.
* **`PHOTONDETECTCOINCIDENCE <N> <MODE1> ... <MODEN> <REGNAME>`**: Detects coincidence events among `N` specified modes, stores result in `RegName`.
* **`APPLYDISPLACEMENTOP <MODE> <ALPHA_RE> <ALPHA_IM>`**: Applies a displacement operator to mode `Mode` with complex `Alpha`.
* **`OPTICALSWITCHCONTROL <MODE> <STATE>`**: Controls an optical switch on mode `Mode` to `State` (0 or 1).
* **`MEASUREWITHDELAY <MODE> <DELAY> <REGNAME>`**: Performs a delayed measurement on mode `Mode` after `Delay`, stores result in `RegName`.
* **`PHOTONLOSSCORR <MODE> <CODENAME>`**: Initiates photon loss error correction for mode `Mode` using `CodeName`.
* **`PHOTONEMISSIONPATTERN <MODE> <PATTERNNAME> <CYCLES>`**: Emits photons in mode `Mode` according to `PatternName` for `Cycles`.
* **`APPLYSQUEEZINGFEEDBACK <MODE> <FEEDBACKREGNAME>`**: Applies squeezing feedback to mode `Mode` based on `FeedbackRegName`.
* **`APPLYPHOTONSUBTRACTION <MODE>`**: Performs a photon subtraction operation on mode `Mode`.
* **`PHOTONADDITION <MODE>`**: Performs a photon addition operation on mode `Mode`.
* **`PNRDETECTION <MODE>`**: Performs Photon Number Resolving Detection on mode `Mode`.
* **`SETOPTICALATTENUATION <MODE> <VALUE>`**: Sets optical attenuation for mode `Mode` to `Value`.
* **`DYNAMICPHASECOMP <MODE> <VALUE>`**: Applies dynamic phase compensation to mode `Mode` by `Value`.
* **`CROSSPHASEMOD <MODE1> <MODE2>`**: Applies cross-phase modulation between `Mode1` and `Mode2`.
* **`OPTICALDELAYLINECTL <MODE> <DURATION>`**: Controls an optical delay line for mode `Mode` for `Duration`.
* **`OPTICALROUTING <MODE1> <MODE2>`**: Routes optical signal between `Mode1` and `Mode2`.
* **`SETPOS <Q> <X> <Y>`**: Sets the spatial position of qubit/mode `Q` to `(X, Y)`.
* **`SETWL <Q> <Wavelength>`**: Sets the wavelength of qubit/mode `Q` to `Wavelength`.
* **`WLSHIFT <Q> <Shift>`**: Shifts the wavelength of qubit/mode `Q` by `Shift`.
* **`MOVE <Q> <DX> <DY>`**: Moves qubit/mode `Q` by `(DX, DY)`.

## Classical / Control flow Operations

* **`HALT`**: Stops program execution.
* **`LOOPSTART <Iterations>`**: Marks the beginning of a loop that repeats `Iterations` times.
* **`LOOPEND`**: Marks the end of a loop, returning control to `LOOPSTART` if iterations remain.
* **`REGSET <Reg> <Value>`**: Sets a classical register `Reg` to a floating-point `Value`.
* **`ADD <DstReg> <Src1Reg> <Src2Reg>`**: Adds the values of `Src1Reg` and `Src2Reg`, stores result in `DstReg`.
* **`SUB <DstReg> <Src1Reg> <Src2Reg>`**: Subtracts `Src2Reg` from `Src1Reg`, stores result in `DstReg`.
* **`MUL <DstReg> <Src1Reg> <Src2Reg>`**: Multiplies `Src1Reg` by `Src2Reg`, stores result in `DstReg`.
* **`DIV <DstReg> <Src1Reg> <Src2Reg>`**: Divides `Src1Reg` by `Src2Reg`, stores result in `DstReg`.
* **`COPY <DstReg> <SrcReg>`**: Copies the value from `SrcReg` to `DstReg`.
* **`ANDBITS <DstReg> <Src1Reg> <Src2Reg>`**: Performs bitwise AND on `Src1Reg` and `Src2Reg`, stores in `DstReg`.
* **`ORBITS <DstReg> <Src1Reg> <Src2Reg>`**: Performs bitwise OR on `Src1Reg` and `Src2Reg`, stores in `DstReg`.
* **`XORBITS <DstReg> <Src1Reg> <Src2Reg>`**: Performs bitwise XOR on `Src1Reg` and `Src2Reg`, stores in `DstReg`.
* **`NOTBITS <DstReg> <SrcReg>`**: Performs bitwise NOT on `SrcReg`, stores in `DstReg`.
* **`SHL <DstReg> <SrcReg> <ShiftAmount>`**: Performs bitwise Left Shift on `SrcReg` by `ShiftAmount`, stores in `DstReg`.
* **`SHR <DstReg> <SrcReg> <ShiftAmount>`**: Performs bitwise Right Shift on `SrcReg` by `ShiftAmount`, stores in `DstReg`.
* **`CMP <Reg1> <Reg2>`**: Compares values in `Reg1` and `Reg2`, setting internal flags.
* **`JUMP <Label>`**: Unconditionally jumps to the instruction at `Label`.
* **`JMP <Label>`**: Alias for `JUMP`.
* **`JUMPABS <Address>`**: Unconditionally jumps to the instruction at byte `Address`.
* **`JMPABS <Address>`**: Alias for `JUMPABS`.
* **`JUMPIFZERO <Reg> <Label>`**: Jumps to `Label` if the value in `Reg` is zero.
* **`IFEQ <Reg1> <Reg2> <Label>`**: Jumps to `Label` if `Reg1` equals `Reg2`.
* **`JUMPIFONE <Reg> <Label>`**: Jumps to `Label` if the value in `Reg` is one.
* **`IFNE <Reg1> <Reg2> <Label>`**: Jumps to `Label` if `Reg1` does not equal `Reg2`.
* **`IFGT <Reg1> <Reg2> <Label>`**: Jumps to `Label` if `Reg1` is greater than `Reg2`.
* **`IFLT <Reg1> <Reg2> <Label>`**: Jumps to `Label` if `Reg1` is less than `Reg2`.
* **`CALL <Label>`**: Calls a subroutine starting at `Label`, pushing current address to call stack.
* **`CALLADDR <Address>`**: Calls a subroutine at byte `Address`.
* **`RETSUB`**: Returns from the current subroutine, popping address from call stack.
* **`PUSHREG <Reg>`**: Pushes the value of `Reg` onto the classical stack.
* **`POPREG <Reg>`**: Pops a value from the classical stack into `Reg`.
* **`CHAROUT <Q>`**: Measures qubit `Q` and prints its classical result as a character.
* **`INPUT <Reg>`**: Reads a character from standard input and stores its ASCII value in `Reg`.
* **`GETTIME <Reg>`**: Gets the current system time (as a floating-point value) and stores it in `Reg`.
* **`RAND <Reg>`**: Generates a random floating-point number and stores it in `Reg`.
* **`SEEDRNG <Seed>`**: Seeds the random number generator with `Seed`.
* **`PRINTF <FormatString> <NumRegs> <Reg1> ...`**: Prints formatted output using `FormatString` and values from `NumRegs` registers.
* **`PRINT <String>`**: Prints the literal `String` to standard output.
* **`PRINTLN <String>`**: Prints the literal `String` to standard output, followed by a newline.
* **`VERBOSELOG <Level> <Message>`**: Logs a verbose `Message` with a specified `Level` (0-3).
* **`COMMENT <Message>`**: An inline comment, ignored by the compiler.
* **`BREAKPOINT`**: Inserts a debug breakpoint (behavior depends on debugger support).
* **`EXITCODE <Code>`**: Terminates the program with the specified exit `Code`.

---

# Example code: Practical applications of QOA

## Example 1: Three Qubit Quantum Fourier Transform

```
; QOA PROGRAM: THREE QUBIT QUANTUM FOURIER TRANSFORM

QINIT 3 ; initialize 3 qubits (Q0, Q1, Q2) to |000>

; apply hadamard to all qubits
H 0
H 1
H 2

; apply controlled phase rotations for the qft
; angles are in radians: pi/2, pi/4
CONTROLLEDPHASEROTATION 1 0 1.5707963267948966 ; cp(Q1, Q0, pi/2)
CONTROLLEDPHASEROTATION 2 0 0.7853981633974483 ; cp(Q2, Q0, pi/4)
CONTROLLEDPHASEROTATION 2 1 1.5707963267948966 ; cp(Q2, Q1, pi/2)

; swap qubits to reverse order for standard qft output
SWAP 0 2 ; swap Q0 and Q2

; measure the qubits to observe the transformed state
MEASURE 0
MEASURE 1
MEASURE 2

HALT
```

---

# Example 2: Optical Network Switch

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

ROUTE0:
; if route 0 is chosen, detect photon from O0 and store its presence
PHOTONDETECT O0 ; detect photon in optical mode 0
PHOTONCOUNT O0 R4 ; count photons in O0, store in R4
STORECLASSICAL R4 0X1000 ; store R4 (photon count) to classical memory at address 0x1000
JUMP END_PROGRAM

ROUTE1:
; if route 1 is chosen, detect photon from O1 and store its presence
PHOTONDETECT O1 ; detect photon in optical mode 1
PHOTONCOUNT O1 R4 ; count photons in O1, store in R4
STORECLASSICAL R4 0X1001 ; store R4 to classical memory at address 0x1001
JUMP END_PROGRAM

ROUTE2:
; if route 2 is chosen, detect photon from O2 and store its presence
PHOTONDETECT O2 ; detect photon in optical mode 2
PHOTONCOUNT O2 R4 ; count photons in O2, store in R4
STORECLASSICAL R4 0X1002 ; store R4 to classical memory at address 0x1002
JUMP END_PROGRAM

END_PROGRAM:
HALT
```

---

# Example 3: Shor's Algorithm

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
H 0
H 1
H 2
H 3
H 4
H 5
H 6
H 7

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
; this is impractical to generate or simulate in this format.
VERBOSELOG 0 "Conceptual Modular Exponentiation (a^x mod N) would be here."
VERBOSELOG 0 "  Base 'a' = 7, Modulo 'N' = 15."
VERBOSELOG 0 "  This involves controlled-U_a^k operations for k = 2^0, 2^1, ..., 2^7."
VERBOSELOG 0 "  Each U_a^k is a complex reversible modular arithmetic circuit."
VERBOSELOG 0 "  (This block represents immense complexity, not directly representable in QOA)"

; --- step 3: inverse quantum fourier transform (iqft) on period-finding register ---

; the iqft extracts the period 'r' from the superposed state in the period register.
; this is the reverse sequence of gates from the qft.
VERBOSELOG 0 "Applying Inverse Quantum Fourier Transform (IQFT) on Period Register (Q0-Q7)"

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
H 0
CONTROLLEDPHASEROTATION 1 0 -1.5707963267948966 ; cp(-pi/2) q1-q0
CONTROLLEDPHASEROTATION 2 0 -0.7853981633974483 ; cp(-pi/4) q2-q0
CONTROLLEDPHASEROTATION 3 0 -0.39269908169872414 ; cp(-pi/8) q3-q0
CONTROLLEDPHASEROTATION 4 0 -0.19634954084936207 ; cp(-pi/16) q4-q0
CONTROLLEDPHASEROTATION 5 0 -0.09817477042468103 ; cp(-pi/32) q5-q0
CONTROLLEDPHASEROTATION 6 0 -0.04908738521234051 ; cp(-pi/64) q6-q0
CONTROLLEDPHASEROTATION 7 0 -0.02454369260617025 ; cp(-pi/128) q7-q0

; qubit 1 (originally Q6)
H 1
CONTROLLEDPHASEROTATION 2 1 -1.5707963267948966 ; cp(-pi/2) q2-q1
CONTROLLEDPHASEROTATION 3 1 -0.7853981633974483 ; cp(-pi/4) q3-q1
CONTROLLEDPHASEROTATION 4 1 -0.39269908169872414 ; cp(-pi/8) q4-q1
CONTROLLEDPHASEROTATION 5 1 -0.19634954084936207 ; cp(-pi/16) q5-q1
CONTROLLEDPHASEROTATION 6 1 -0.09817477042468103 ; cp(-pi/32) q6-q1
CONTROLLEDPHASEROTATION 7 1 -0.04908738521234051 ; cp(-pi/64) q7-q1

; qubit 2 (originally Q5)
H 2
CONTROLLEDPHASEROTATION 3 2 -1.5707963267948966 ; cp(-pi/2) q3-q2
CONTROLLEDPHASEROTATION 4 2 -0.7853981633974483 ; cp(-pi/4) q4-q2
CONTROLLEDPHASEROTATION 5 2 -0.39269908169872414 ; cp(-pi/8) q5-q2
CONTROLLEDPHASEROTATION 6 2 -0.19634954084936207 ; cp(-pi/16) q6-q2
CONTROLLEDPHASEROTATION 7 2 -0.09817477042468103 ; cp(-pi/32) q7-q2

; qubit 3 (originally Q4)
H 3
CONTROLLEDPHASEROTATION 4 3 -1.5707963267948966 ; cp(-pi/2) q4-q3
CONTROLLEDPHASEROTATION 5 3 -0.7853981633974483 ; cp(-pi/4) q5-q3
CONTROLLEDPHASEROTATION 6 3 -0.39269908169872414 ; cp(-pi/8) q6-q3
CONTROLLEDPHASEROTATION 7 3 -0.19634954084936207 ; cp(-pi/16) q7-q3

; qubit 4 (originally Q3)
H 4
CONTROLLEDPHASEROTATION 5 4 -1.5707963267948966 ; cp(-pi/2) q5-q4
CONTROLLEDPHASEROTATION 6 4 -0.7853981633974483 ; cp(-pi/4) q6-q4
CONTROLLEDPHASEROTATION 7 4 -0.39269908169872414 ; cp(-pi/8) q7-q4

; qubit 5 (originally Q2)
H 5
CONTROLLEDPHASEROTATION 6 5 -1.5707963267948966 ; cp(-pi/2) q6-q5
CONTROLLEDPHASEROTATION 7 5 -0.7853981633974483 ; cp(-pi/4) q7-q5

; qubit 6 (originally Q1)
H 6
CONTROLLEDPHASEROTATION 7 6 -1.5707963267948966 ; cp(-pi/2) q7-q6

; qubit 7 (originally Q0)
H 7

; --- step 4: measurement of period-finding register ---

; measure the period-finding register (q0-q7)
MEASURE 0
MEASURE 1
MEASURE 2
MEASURE 3
MEASURE 4
MEASURE 5
MEASURE 6
MEASURE 7

; --- step 5: classical post-processing ---

; after measurement, a classical computer takes the measured value 'c'
; and uses the continued fractions algorithm to find the period 'r'.
; then, it calculates gcd(a^(r/2) +/- 1, n) to find factors of n.
VERBOSELOG 0 "Classical Post-Processing: Use continued fractions to find period 'r'."
VERBOSELOG 0 "Then calculate gcd(a^(r/2) +/- 1, N) to find factors of N."

HALT
```

### This is all for now. QOA is still being modeled and developed solely by me, and any practical applications would not be relevant until optical and quantum systems become more commonplace.
### As these technologies mature and gain wider adoption, QOA aims to provide a unified low-level assembly language capable of efficiently programming and controlling purely optical, purely quantum, or hybrid quantum-optical computing platforms.
### Until then, QOA remains my theoretical passion project focused on laying the groundwork for future advancements in these future fields of computing.

## Thanks for reading!

### -- Rayan
