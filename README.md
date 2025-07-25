![qoa-logo](https://github.com/user-attachments/assets/5fdbae92-68f8-490e-8368-f3fd6c81a064)

# QOA  
**The Quantum Optical Assembly Programming Language**

> **NOTE:** EXTRA DOCUMENTATION CAN BE FOUND IN THE README FOLDER

---

## Changelog

- **LATEST RELEASE NOTES:** [HERE](<changelog/QOA v0.3.1.md>)
- See `changelog/` folder for more detailed version history and ISA updates.

---

I have created this guide for researchers, developers and students who will use QOA in practical applications. I have designed it to interact with quantum and/or optical systems in the same way classical RISC‑syntax based assembly could manipulate electrons in transistors. I hope whoever reads this guide finds it useful.

*Sincerely,*  
**Rayan**

---

## System Requirements

- **CPU:**  
  - Processors with SIMD support (AVX‑512/AVX2 recommended on x86_64 for best performance, NEON for AARCH64, and RVV 1.0 for RISC‑V 64‑bit).  
  - Please try to run QOA on a CPU with as large a cache as possible (AMD X3D CPUs are a good option; see explanation in `README`).

- **Memory:**  
  - Increased efficiency allows for simulation of larger quantum systems with the same memory footprint within 2^N doubling state requirements -- still exponential memory, but less compiler & executor overhead.

- **Rust Compiler:**  
  - Requires **Rust 1.55.0** or higher for portable SIMD support; crates may also need an update.  
  - **Rust Compiler Note:** Use `rustc 1.90.0-nightly (a7a1618e6 2025-07-22)` **if issues arise**!
  - Rust edition `2024` used.

---

## Base Syntax / Operations Overview  
*(This list is for QOA v0.3.1)*

### Quantum Operations

| Instruction                                    | Description                                                                                 | Aliases                                             |
|-----------------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------------|
| `QINIT N`                                     | Initializes the quantum state with N qubits, all set to the \|0⟩ state.                   | QI, INITQUBIT, IQ, QINITQ                           |
| `H Q`                                         | Applies a Hadamard gate to qubit Q.                                                        | HAD, APPLYHADAMARD                                  |
| `APPLYBITFLIP Q`                              | Applies a Pauli‑X (bit‑flip) gate to qubit Q.                                              | X                                                   |
| `APPLYPHASEFLIP Q`                            | Applies a Pauli‑Z (phase‑flip) gate to qubit Q.                                            | Z                                                   |
| `APPLYTGATE Q`                                | Applies a T gate (π/8 phase shift) to qubit Q.                                             | T                                                   |
| `APPLYSGATE Q`                                | Applies an S gate (π/4 phase shift) to qubit Q.                                            | S                                                   |
| `PHASESHIFT Q Angle`                          | Applies a phase shift to qubit Q by *Angle* radians.                                       | P, SETPHASE, SETP, SPH                              |
| `RX Q Angle`                                  | Rotation around the X‑axis on qubit Q by *Angle* radians.                                  | —                                                   |
| `RY Q Angle`                                  | Rotation around the Y‑axis on qubit Q by *Angle* radians.                                  | —                                                   |
| `RZ Q Angle`                                  | Rotation around the Z‑axis on qubit Q by *Angle* radians.                                  | —                                                   |
| `PHASE Q Angle`                               | Applies a phase gate to qubit Q by *Angle* radians.                                        | PSE                                                 |
| `CONTROLLEDNOT C T`                           | CNOT gate with control qubit C and target qubit T.                                         | CNOT, CN                                            |
| `CZ C T`                                      | Controlled‑Z gate with control qubit C and target qubit T.                                 | —                                                   |
| `CPHASE C T Angle`                            | Controlled phase rotation between C and T by *Angle* radians.                              | CONTROLLEDPHASEROTATION, APPLYCPHASE, CPR           |
| `ENTANGLE C T`                                | Entangles qubits C and T (often implemented as a CNOT).                                    | —                                                   |
| `ENTANGLEBELL Q1 Q2`                          | Creates a Bell state between Q1 and Q2.                                                     | EBELL, EB                                           |
| `ENTANGLEMULTI Q1 ... QN`                     | Creates a multi‑qubit entangled state across specified qubits.                             | EMULTI, EM                                          |
| `ENTANGLECLUSTER Q1 ... QN`                   | Generates a cluster state over specified qubits.                                           | ECLUSTER, ECR                                       |
| `ENTANGLESWAP Q1 Q2 Q3 Q4`                    | Performs entanglement swapping among four qubits.                                          | ESWAP, ESP                                          |
| `ENTANGLESWAPMEASURE Q1 Q2 Q3 Q4 Label`       | Entanglement swapping with measurement, jumping to *Label* based on outcome.               | ESWAPM, ESM                                         |
| `ENTANGLEWITHCLASSICALFEEDBACK Q1 Q2 Signal`  | Entangles Q1 with Q2, with classical feedback from *Signal*.                               | EWCFB, ECFB                                         |
| `ENTANGLEDISTRIBUTED Q Node`                  | Performs distributed entanglement on qubit Q, involving *Node*.                            | EDIST, ED                                           |
| `MEASURE Q`                                   | Measures qubit Q, collapsing its quantum state and yielding a classical result.            | QMEAS, QM, MEAS, M                                  |
| `MEASUREINBASIS Q Basis`                     | Measures qubit Q in a specified *Basis*.                                                   | MEASB, MIB                                          |
| `RESET Q`                                     | Resets qubit Q to the ground state \|0⟩.                                                    | RST, RSTQ, QRESET, QR                               |
| `RESETALL`                                    | Resets all qubits in the system to \|0⟩.                                                   | RSTALL, RSA                                         |
| `MARKOBSERVED Q`                              | Marks qubit Q as observed (internal state tracking).                                       | MOBS, MO                                            |
| `RELEASE Q`                                   | Releases resources associated with qubit Q.                                                | REL, RL                                             |
| `APPLYGATE GateName Q`                        | Applies a named unitary gate (e.g., "h", "x", "cz") to qubit Q.                             | AGATE, AG                                           |
| `APPLYROTATION Q Axis Angle`                  | General rotation to qubit Q by *Angle* around Axis ('x', 'y', or 'z').                     | ROT, AR                                             |
| `APPLYMULTIQUBITROTATION Qs Axis Angles`      | Simultaneous rotations on multiple qubits with corresponding angles.                       | MROT, AMQR                                          |
| `APPLYKERRNONLINEARITY Q Strength Duration`   | Applies Kerr nonlinearity to qubit/mode Q.                                                 | AKNL                                                |
| `DECOHERENCEPROTECT Q Duration`               | Activates decoherence protection for qubit Q for *Duration*.                              | DPROT, DP                                           |
| `APPLYMEASUREMENTBASISCHANGE Q Basis`         | Changes the measurement basis for qubit Q to *Basis*.                                     | AMBC                                                |
| `APPLYNONLINEARPHASESHIFT Q Strength`         | Nonlinear phase shift to qubit Q by *Strength*.                                            | ANLPS, ANLP, ANLPH                                  |
| `APPLYNONLINEARSIGMA Q Strength`              | Custom nonlinear operation to qubit Q with *Strength*.                                     | ANLS, ANLSI                                         |
| `QUANTUMSTATETOMOGRAPHY Q Basis`              | Quantum state tomography on qubit Q in *Basis*.                                            | QST, QSTAT                                          |
| `BELLSTATEVERIFICATION Q1 Q2 ResultReg`      | Verifies Bell state between Q1 and Q2, stores result in *ResultReg*.                      | BSV, BSTATE                                         |
| `QUANTUMZENOEFFECT Q NumMeasurements IntervalCycles` | Simulates Quantum Zeno Effect on qubit Q.                                       | QZE, QZEN                                           |
| `APPLYLINEAROPTICALTRANSFORM Name InputQs OutputQs NumModes` | Applies a linear optical transformation.                                     | ALOT, ALOPT                                         |
| `ERRORCORRECT Q SyndromeType`                 | Invokes quantum error correction on qubit Q.                                              | ECORR, EC                                           |
| `ERRORSYNDROME Q SyndromeType ResultReg`      | Extracts error syndrome for qubit Q, stores in *ResultReg*.                               | ESYN, ES                                            |
| `APPLYQNDMEASUREMENT Q ResultReg`             | Quantum non‑demolition measurement on qubit Q.                                             | AQND, AQAD, AQNM                                    |
| `SWAP Q1 Q2`                                  | Swaps the states of qubit Q1 and Q2.                                                       | —                                                   |

### Optical Operations

| Instruction                                  | Description                                                                                   | Aliases                              |
|---------------------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------|
| `PHOTONEMIT Q`                              | Emits a photon from qubit/mode Q.                                                            | PEMIT, PE                            |
| `PHOTONDETECT Q`                            | Detects a photon at qubit/mode Q.                                                            | PDETECT, PD                          |
| `PHOTONROUTE Q FromPort ToPort`             | Routes a photon from port to port.                                                           | PROUTE, PR                           |
| `PHOTONCOUNT Q ResultReg`                   | Counts detected photons at Q, stores in *ResultReg*.                                         | PCOUNT, PC                           |
| `APPLYDISPLACEMENT Q Alpha`                 | Applies a displacement to Q by *Alpha*.                                                      | ADISP, AD                            |
| `APPLYSQUEEZING Q SqueezingFactor`          | Applies a squeezing operation to Q by *SqueezingFactor*.                                     | ASQ, AS                              |
| `MEASUREPARITY Q`                           | Measures parity of qubit/mode Q.                                                             | MPAR, MP                             |
| `PHOTONLOSSSIMULATE Q LossProbability Seed` | Simulates photon loss at Q with *LossProbability* and *Seed*.                                 | PLS, PLSIM                           |
| `TIMEDELAY Q Cycles`                        | Applies a controlled time delay to Q for *Cycles*.                                           | TDELAY, TD                           |
| `PHOTONBUNCHINGCONTROL Q Enable`            | Toggles photon bunching control for Q (true/false).                                          | PBUNCH, PBC                          |
| `SINGLEPHOTONSOURCEON Q`                    | Activates a single‑photon source for Q.                                                      | SPSON                                |
| `SINGLEPHOTONSOURCEOFF Q`                   | Deactivates a single‑photon source for Q.                                                    | SPSOFF                               |
| `PHOTONDETECTCOINCIDENCE Qs ResultReg`      | Detects coincidence events among multiple Qs, stores in *ResultReg*.                         | PDCOIN, PDC                          |
| `APPLYDISPLACEMENTOPERATOR Q Alpha Duration`| Displacement operator with *Alpha* and *Duration*.                                           | ADO, ADOP                            |
| `OPTICALSWITCHCONTROL Q State`              | Controls an optical switch on Q to *State* (true/false).                                     | OSC, OSW                             |
| `MEASUREWITHDELAY Q DelayCycles ResultReg`  | Delayed measurement on Q with *DelayCycles*, stores in *ResultReg*.                         | MWD, MWDEL                           |
| `PHOTONLOSSCORRECTION Q CorrectionReg`      | Photon loss error correction on Q using *CorrectionReg*.                                     | PLC, PLCOR                           |
| `PHOTONEMISSIONPATTERN Q PatternReg Cycles` | Emits photons according to *PatternReg* for *Cycles*.                                        | PEPAT, PEP                           |
| `APPLYSQUEEZINGFEEDBACK Q FeedbackReg`      | Applies squeezing feedback based on *FeedbackReg*.                                           | ASWF, ASF, ASFB                      |
| `APPLYPHOTONSUBTRACTION Q`                  | Photon subtraction operation on Q.                                                            | APSUB, APS                           |
| `PHOTONADDITION Q`                          | Photon addition operation on Q.                                                               | PADD, PA                             |
| `PHOTONNUMBERRESOLVINGDETECTION Q ResultReg`| Photon Number Resolving Detection on Q.                                                      | PNRD, PNR                            |
| `SETOPTICALATTENUATION Q Attenuation`       | Sets optical attenuation for Q to *Attenuation*.                                             | SOATT, SOA                           |
| `DYNAMICPHASECOMPENSATION Q Phase`          | Dynamic phase compensation for Q by *Phase*.                                                 | DPC, DPCMP                           |
| `CROSSPHASEMODULATION Q1 Q2 Strength`       | Cross‑phase modulation between Q1 and Q2 with *Strength*.                                    | CPM, CPMOD                           |
| `OPTICALDELAYLINECONTROL Q DelayCycles`     | Controls an optical delay line for Q for *DelayCycles*.                                      | ODLC, ODL                            |
| `OPTICALROUTING Q1 Q2`                      | Routes optical signal between Q1 and Q2.                                                      | OROUTE, OPTR                         |
| `SETPOS Q X Y`                              | Sets spatial position of Q to (*X*, *Y*).                                                    | SPOS, STP                            |
| `SETWL Q Wavelength`                        | Sets wavelength of Q to *Wavelength*.                                                        | SWL, SW                              |
| `WLSHIFT Q DeltaWavelength`                 | Shifts wavelength of Q by *DeltaWavelength*.                                                 | WLS, WLSH                            |
| `MOVE Q DX DY`                              | Moves qubit/mode Q by (*DX*, *DY*).                                                          | MOV, MV                              |

### Classical / Control Flow Operations

| Instruction                                | Description                                                                                                | Aliases                                     |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| `HALT`                                    | Stops program execution.                                                                                   | HLT                                         |
| `LOOPSTART Reg`                           | Begins a loop controlled by *Reg*.                                                                         | LSTART, LS                                  |
| `LOOPEND`                                 | Ends a loop.                                                                                                | LEND, LE                                    |
| `REGSET Reg Value`                        | Sets a classical register *Reg* to a floating‑point *Value*.                                               | RSET, RGST                                  |
| `ADD DstReg Src1Reg Src2Reg`              | Adds Src1Reg and Src2Reg, stores result in DstReg.                                                         | —                                           |
| `SUB DstReg Src1Reg Src2Reg`              | Subtracts Src2Reg from Src1Reg, stores result in DstReg.                                                    | —                                           |
| `MUL DstReg Src1Reg Src2Reg`              | Multiplies Src1Reg and Src2Reg, stores result in DstReg.                                                    | —                                           |
| `DIV DstReg Src1Reg Src2Reg`              | Divides Src1Reg by Src2Reg, stores result in DstReg.                                                        | —                                           |
| `REGADD DstReg Src1Reg Src2Reg`           | Adds two registers (dest, op1, op2).                                                                       | RADD, RGA                                   |
| `REGSUB DstReg Src1Reg Src2Reg`           | Subtracts two registers (dest, op1, op2).                                                                  | RSUB, RGS                                   |
| `REGCOPY DstReg SrcReg`                   | Copies value from SrcReg to DstReg.                                                                        | RCOPY, RC                                   |
| `ANDBITS DstReg Op1Reg Op2Reg`            | Performs bitwise AND on Op1Reg and Op2Reg, stores in DstReg.                                                | ANDB, AB                                    |
| `ORBITS DstReg Op1Reg Op2Reg`             | Performs bitwise OR on Op1Reg and Op2Reg, stores in DstReg.                                                 | ORB, OB                                     |
| `XORBITS DstReg Op1Reg Op2Reg`            | Performs bitwise XOR on Op1Reg and Op2Reg, stores in DstReg.                                                | XORB, XB                                    |
| `NOTBITS DstReg OpReg`                    | Performs bitwise NOT on OpReg, stores in DstReg.                                                           | NOTB, NB                                    |
| `SHL DstReg OpReg AmountReg`              | Shifts OpReg left by AmountReg bits, stores in DstReg.                                                     | —                                           |
| `SHR DstReg OpReg AmountReg`              | Shifts OpReg right by AmountReg bits, stores in DstReg.                                                    | —                                           |
| `CMP Reg1 Reg2`                           | Compares Reg1 and Reg2, setting internal flags.                                                            | —                                           |
| `JUMP Label`                              | Unconditional jump to *Label*.                                                                             | —                                           |
| `JMP Offset`                              | Unconditional relative jump by *Offset* (signed 64‑bit integer).                                             | —                                           |
| `JMPABS Address`                          | Unconditional absolute jump to *Address*.                                                                   | JMPA, JABS                                  |
| `JUMPIFZERO CondReg Label`                | Conditional jump to *Label* if CondReg is zero.                                                            | JIZ                                         |
| `JUMPIFONE CondReg Label`                 | Conditional jump to *Label* if CondReg is one.                                                             | JIO                                         |
| `IFEQ Reg1 Reg2 Offset`                   | If Reg1 == Reg2, jump relative by *Offset*.                                                                | IEQ                                         |
| `IFNE Reg1 Reg2 Offset`                   | If Reg1 != Reg2, jump relative by *Offset*.                                                                | INE                                         |
| `IFGT Reg1 Reg2 Offset`                   | If Reg1 > Reg2, jump relative by *Offset*.                                                                 | IGT                                         |
| `IFLT Reg1 Reg2 Offset`                   | If Reg1 < Reg2, jump relative by *Offset*.                                                                 | ILT                                         |
| `CALL Label`                              | Calls a subroutine at *Label*.                                                                             | CallLabel                                   |
| `CALLADDR Address`                        | Calls a subroutine at *Address*, pushes return address to stack.                                            | CADDR, CA                                   |
| `RETSUB`                                  | Returns from a subroutine, pops return address from stack.                                                  | RS                                          |
| `PUSH RegName`                            | Pushes classical register *RegName* value onto stack.                                                      | —                                           |
| `POP RegName`                             | Pops value from stack into classical register *RegName*.                                                   | —                                           |
| `LOAD Reg QubitVar`                       | Loads value from classical variable *QubitVar* into qubit register *Reg*.                                  | —                                           |
| `STORE Reg QubitVar`                      | Stores qubit measurement result from *Reg* into classical variable *QubitVar*.                              | —                                           |
| `LOADMEM Reg MemAddress`                  | Loads value from *MemAddress* into classical register *Reg*.                                               | LMEM, LM                                    |
| `STOREMEM Reg MemAddress`                 | Stores value from classical register *Reg* into *MemAddress*.                                              | SMEM, SM                                    |
| `LOADCLASSICAL Reg ClassicalVar`          | Loads value from classical variable *ClassicalVar* into classical register *Reg*.                           | LCL, LC                                     |
| `STORECLASSICAL Reg ClassicalVar`         | Stores value from classical register *Reg* into classical variable *ClassicalVar*.                          | SCL, SC                                     |
| `ALLOC RegAddr Size`                      | Allocates *Size* bytes of memory, stores start address in *RegAddr*.                                        | ALC                                         |
| `FREE Address`                            | Frees memory at *Address*.                                                                                 | FRE                                         |
| `CHARLOAD Reg CharValue`                  | Loads a character *CharValue* into *Reg*.                                                                  | CLOAD, CLD                                  |
| `CHAROUT Reg`                             | Outputs a character from *Reg*.                                                                            | COUT, CO                                    |
| `INPUT Reg`                               | Reads floating‑point value from stdin into *Reg*.                                                          | INP                                         |
| `GETTIME Reg`                             | Gets system timestamp into *Reg*.                                                                          | GTIME, GT                                   |
| `RAND Reg`                                | Generates a random number into *Reg*.                                                                      | RN                                          |
| `SEEDRNG Seed`                            | Seeds the random number generator with *Seed*.                                                             | SRNG                                        |
| `SQRT DstReg SrcReg`                      | Calculates square root of *SrcReg*, stores in *DstReg*.                                                    | SR                                          |
| `EXP DstReg SrcReg`                       | Calculates exponential of *SrcReg*, stores in *DstReg*.                                                    | —                                           |
| `LOG DstReg SrcReg`                       | Calculates logarithm of *SrcReg*, stores in *DstReg`.                                                      | —                                           |
| `PRINTF FormatString Regs...`             | C‑style formatted output using *FormatString* and register values.                                          | PF                                          |
| `PRINT String`                            | Prints a string literal *String*.                                                                          | —                                           |
| `PRINTLN String`                          | Prints a string literal *String* with a newline.                                                           | PLN                                         |
| `VERBOSELOG Q Message`                    | Logs verbose messages associated with qubit Q and *Message*.                                               | VLOG, VL                                    |
| `COMMENT Text`                            | Inline comment with *Text*.                                                                                | CMT, CM                                     |
| `BREAKPOINT`                              | Inserts a debug breakpoint.                                                                                | BP                                          |
| `EXITCODE Code`                           | Terminates program with *Code*.                                                                             | EXC, EX                                     |
| `BARRIER`                                 | Synchronization barrier.                                                                                   | BR                                          |
| `DUMPSTATE`                               | Outputs quantum amplitudes and phases.                                                                     | DSTATE, DS                                  |
| `DUMPREGS`                                | Outputs all register values.                                                                               | DREGS, DR                                   |

---

## Example Code

### Example 1: Three‑Qubit Quantum Fourier Transform
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

### Example 2: Optical Network Switch

```
; QOA PROGRAM: OPTICAL NETWORK SWITCH
; this program simulates routing based on a classical control signal.

; define classical registers for routing logic
REGSET R0 0.0 ; routing destination 0
REGSET R1 1.0 ; routing destination 1
REGSET R2 2.0 ; routing destination 2
REGSET R3 0.0 ; input signal (0, 1, or 2 to choose route)

; simulate incoming photons (emit for demo purposes)
PHOTONEMIT O0
PHOTONEMIT O1
PHOTONEMIT O2

; detect photons in incoming modes
PHOTONDETECT O0
PHOTONDETECT O1
PHOTONDETECT O2

; load a classical value into R3 to simulate routing decision
REGSET R3 1.0 ; simulate choosing route 1

; route based on R3's value
CMP R3 R0
IFEQ R3 R0 ROUTE0

CMP R3 R1
IFEQ R3 R1 ROUTE1

CMP R3 R2
IFEQ R3 R2 ROUTE2

JUMP END_PROGRAM

ROUTE0:
  PHOTONDETECT O0
  PHOTONCOUNT O0 R4
  STORECLASSICAL R4 0X1000
  JUMP END_PROGRAM

ROUTE1:
  PHOTONDETECT O1
  PHOTONCOUNT O1 R4
  STORECLASSICAL R4 0X1001
  JUMP END_PROGRAM

ROUTE2:
  PHOTONDETECT O2
  PHOTONCOUNT O2 R4
  STORECLASSICAL R4 0X1002
  JUMP END_PROGRAM

HALT

```

### Example 3: Shor’s Algorithm

```
; QOA PROGRAM: SHOR'S ALGORITHM
; outline for factoring n=15 (requires 12 qubits)

QINIT 12 ; initialize 12 qubits

; Step 1: superposition on period register (q0–q7)
HAD 0
HAD 1
HAD 2
HAD 3
HAD 4
HAD 5
HAD 6
HAD 7

; Step 2: modular exponentiation (conceptual)
VERBOSELOG 0 "Modular Exponentiation (a^x mod N) conceptual..."

; Step 3: inverse QFT on period register
SWAP 0 7
SWAP 1 6
SWAP 2 5
SWAP 3 4

HAD 0
CPHASE 1 0 -1.5707963267948966
...
HAD 7

; Step 4: measurement
MEAS 0
MEAS 1
MEAS 2
MEAS 3
MEAS 4
MEAS 5
MEAS 6
MEAS 7

; Step 5: classical post-processing (continued fractions, gcd, etc.)
VLOG 0 "Classical Post-Processing..."

HALT

```

### This is all for now. If you would like to follow development, please star the repository.

### Thanks for reading!
### *— Rayan (@planetryan)*