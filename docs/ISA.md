# QOA instruction reference

Generated from `src/isa.rs` by `qoa isa --markdown`. Do not edit by hand.

Mnemonics are case-insensitive. Every alias does exactly what its main name does.

## Operands

| Written as | Meaning |
|---|---|
| `q3` | qubit: q3 (or 3), or data[1] after QREG; ranges like q0..q3 where allowed |
| `o1` | optical mode: o1 (or 1) |
| `R2` | register: R2 (or 2), R0..R31 |
| `c0` / `c0..c3` | classical bit: c0, or a range like c0..c3 |
| `-> c0` / `-> R2` | result destination: c0 (classical bit) or R2 (register), optionally after -> |
| `c0` / `R2` | value: c0 (classical bit) or R2 (register) |
| `pi/2` | number or expression: 0.5, pi/2, -3*pi/4, sqrt(2)/2 |
| `8` / `R1` | count: whole number or register |
| `done` | label |
| `"text"` | text in double quotes |
| `X` | basis or axis: X, Y or Z |
| `"A"` | character: "A" or a character code like 65 |
| `data` | name: letters, digits and _, not an instruction name |

Instructions marked *simulator only* read information a real device cannot see without disturbing the state.

## Setup

| Instruction | Does | Aliases |
|---|---|---|
| `QINIT qubits` | Declare how many qubits the program uses (once). Without it, the count is the highest qubit index used plus one. | QI, INITQUBIT, IQ, QINITQ |
| `QMODE modes [cutoff]` | Declare optical modes, each holding 0..cutoff-1 photons (default cutoff 8). Probability pushed past the cutoff is reported after the run. | MODES |
| `QREG name size [first]` | Name qubits (optional): after QREG data 3, data[0]..data[2] (or data, all three) stand for the next 3 unnamed qubits, or for qubits from `first` if given. q0, q1, ... keep working. |  |
| `CREG name size [first]` | Name classical bits (optional), like QREG: CREG flags 2 makes flags[0], flags[1] and flags. |  |
| `OREG name size [first]` | Name optical modes (optional), like QREG: OREG arm 2 makes arm[0], arm[1] and arm. |  |

## Qubit gates

| Instruction | Does | Aliases |
|---|---|---|
| `H qubit` | Hadamard. Accepts a qubit range (q0..q3). | HAD, APPLYHADAMARD |
| `X qubit` | Pauli-X (bit flip). Accepts a qubit range (q0..q3). | APPLYBITFLIP |
| `Y qubit` | Pauli-Y. Accepts a qubit range (q0..q3). |  |
| `Z qubit` | Pauli-Z (phase flip). Accepts a qubit range (q0..q3). | APPLYPHASEFLIP |
| `S qubit` | S = diag(1, i). Accepts a qubit range (q0..q3). | APPLYSGATE |
| `SDG qubit` | S† = diag(1, -i). Accepts a qubit range (q0..q3). | SDAG |
| `T qubit` | T = diag(1, e^(i·pi/4)). Accepts a qubit range (q0..q3). | APPLYTGATE |
| `TDG qubit` | T† = diag(1, e^(-i·pi/4)). Accepts a qubit range (q0..q3). | TDAG |
| `SX qubit` | Square root of X. Accepts a qubit range (q0..q3). | SQRTX |
| `SXDG qubit` | Inverse of SX. Accepts a qubit range (q0..q3). | SXDAG |
| `PHASE qubit angle` | Phase gate diag(1, e^(i·angle)). Accepts a qubit range (q0..q3). | P, PHASESHIFT, SETPHASE, SETP, SPH, PSE, U1 |
| `RX qubit angle` | Rotation about X: exp(-i·angle·X/2). Accepts a qubit range (q0..q3). |  |
| `RY qubit angle` | Rotation about Y: exp(-i·angle·Y/2). Accepts a qubit range (q0..q3). |  |
| `RZ qubit angle` | Rotation about Z: exp(-i·angle·Z/2). Accepts a qubit range (q0..q3). |  |
| `U qubit theta phi lambda` | General single-qubit gate U(theta, phi, lambda), OpenQASM convention. Accepts a qubit range (q0..q3). | U3 |
| `ROT qubit axis angle` | Rotation about axis X, Y or Z (the same as RX, RY or RZ). Accepts a qubit range (q0..q3). | APPLYROTATION, AR |
| `MROT axis angle qubits...` | The same rotation on every listed qubit. | APPLYMULTIQUBITROTATION, AMQR |
| `GATE name qubit` | Apply a named single-qubit gate: h, x, y, z, s, sdg, t, tdg, sx, sxdg. | APPLYGATE, AGATE, AG |
| `CNOT control target` | Controlled-X: flip the target when the control is 1. | CX, CONTROLLEDNOT, CN, ENTANGLE |
| `CY control target` | Controlled-Y. |  |
| `CZ control target` | Controlled-Z: phase -1 on \|11>. |  |
| `CH control target` | Controlled-Hadamard. |  |
| `CSX control target` | Controlled square root of X. |  |
| `CU control target theta phi lambda` | Controlled U(theta, phi, lambda). | CU3 |
| `CPHASE control target angle` | Controlled phase: e^(i·angle) on \|11>. | CP, CONTROLLEDPHASEROTATION, APPLYCPHASE, CPR, CU1 |
| `CRX control target angle` | Controlled RX. |  |
| `CRY control target angle` | Controlled RY. |  |
| `CRZ control target angle` | Controlled RZ. |  |
| `SWAP a b` | Swap two qubits. |  |
| `CSWAP control a b` | Swap a and b when the control is 1 (Fredkin). | CONTROLLEDSWAP, CS, FREDKIN |
| `CCX control1 control2 target` | Toffoli: flip the target when both controls are 1. | TOFFOLI, CCNOT |
| `MCX controls... target` | Flip the target (the last qubit) when every other listed qubit is 1. |  |
| `MCZ qubits...` | Phase -1 on the basis states where every listed qubit is 1. |  |

## Entangling shorthands

| Instruction | Does | Aliases |
|---|---|---|
| `BELL a b` | Runs H a, then CNOT a b. From \|00> this gives (\|00> + \|11>)/sqrt(2). | ENTANGLEBELL, EBELL, EB |
| `GHZ qubits...` | Runs H on the first qubit, then CNOT from it to each of the others. From \|0..0> this gives a GHZ state. | ENTANGLEMULTI, EMULTI, EM |
| `CLUSTER qubits...` | Linear cluster state: H on every qubit, then CZ between each neighbouring pair. | ENTANGLECLUSTER, ECLUSTER, ECR |
| `ESWAP a b c d` | Entanglement swapping. With Bell pairs (a,b) and (c,d): Bell-measures b and c (into c<b> and c<c>), then corrects d, leaving (a,d) as a Bell pair. | ENTANGLESWAP, ESP |

## Qubit measurement and reset

| Instruction | Does | Aliases |
|---|---|---|
| `MEASURE qubit [-> to]` | Measure in the Z basis. The result goes to bit c<qubit> unless routed with -> cK or -> RK. Accepts a qubit range (q0..q3). | MEAS, M, QMEAS, QM, APPLYQNDMEASUREMENT, AQND, AQAD, AQNM |
| `MEASUREIN qubit basis [-> to]` | Measure in the X, Y or Z basis (rotates into Z, measures, rotates back). | MEASUREINBASIS, MEASB, MIB |
| `MEASUREALL` | Measure every qubit, each into its own bit c<qubit>. | MALL |
| `RESET qubit` | Put a qubit in \|0>: measure it, then flip it if it read 1. Accepts a qubit range (q0..q3). | RST, RSTQ, QRESET, QR |
| `RESETALL` | Every qubit to \|0> and every mode to vacuum. | RSTALL, RSA |
| `TOBASIS qubit basis` | Rotate so that a Z measurement reads the given basis (X: H; Y: SDG then H; Z: nothing). Accepts a qubit range (q0..q3). | APPLYMEASUREMENTBASISCHANGE, AMBC |

## Optical modes

| Instruction | Does | Aliases |
|---|---|---|
| `DISPLACE mode re [im]` | Displacement D(alpha), alpha = re + i·im. Turns vacuum into a coherent state with mean photon number \|alpha\|^2. | APPLYDISPLACEMENT, ADISP, AD |
| `DISPLACEREG mode alpha` | Displacement by the (real) value in a register: classical feed-forward. | APPLYDISPLACEMENTFEEDBACK |
| `SQUEEZE mode r [phi]` | Squeezing S(r·e^(i·phi)). With phi = 0 and r > 0 the x quadrature is squeezed. | APPLYSQUEEZING, ASQ, AS |
| `SQUEEZEREG mode r` | Squeezing by the value in a register: classical feed-forward. | APPLYSQUEEZINGFEEDBACK, ASWF, ASF, ASFB |
| `MODEPHASE mode angle` | Phase rotation e^(i·angle·n) of a mode. | DYNAMICPHASECOMPENSATION, DPC, DPCMP, PHASEROT |
| `KERR mode kappa [time]` | Self-Kerr nonlinearity e^(i·kappa·time·n^2) (time defaults to 1). | APPLYKERRNONLINEARITY, AKNL |
| `CROSSKERR a b kappa` | Cross-Kerr (cross-phase modulation) e^(i·kappa·n_a·n_b). | CROSSPHASEMODULATION, CPM, CPMOD |
| `BS a b theta [phi]` | Beamsplitter exp(theta·(e^(i·phi)·a·b† - e^(-i·phi)·a†·b)). theta = pi/4 is 50:50. | BEAMSPLITTER |
| `MODESWAP a b` | Exchange the contents of two modes (an ideal router). | OPTICALROUTING, OROUTE, OPTR |
| `ADDPHOTON mode` | Apply a† and renormalize. Post-selected: real photon addition only succeeds some of the time. | PHOTONADDITION, PADD, PA |
| `SUBPHOTON mode` | Apply a and renormalize (post-selected). Fails if the mode is certainly empty. | APPLYPHOTONSUBTRACTION, APSUB, APS |
| `LOSS mode probability` | Each photon in the mode is lost with this probability. One random outcome per run; averaged over shots it is the exact loss channel. | PHOTONLOSS, PHOTONLOSSSIMULATE, PLS, PLSIM, SETOPTICALATTENUATION, SOATT, SOA |
| `FOCK mode photons` | Discard what the mode held and prepare exactly that many photons. |  |
| `SINGLEPHOTON mode` | An ideal single-photon source: the same as FOCK mode 1. | SINGLEPHOTONSOURCEON, SPSON |
| `COUNT mode [-> to]` | Number-resolving photon detection. The count goes to the mode's readout unless routed with -> RK. | PHOTONCOUNT, PCOUNT, PC, PHOTONNUMBERRESOLVINGDETECTION, PNRD, PNR |
| `DETECT mode [-> to]` | Threshold (click) detection: 1 if any photon is present, else 0. | PHOTONDETECT, PDETECT, PD |
| `PARITY mode [-> to]` | Photon-number parity measurement: 0 for even, 1 for odd. | MEASUREPARITY, MPAR, MP |
| `COINCIDENCE modes... -> to` | Click-detect every listed mode; writes 1 if all of them clicked. | PHOTONDETECTCOINCIDENCE, PDCOIN, PDC |

## Qubit-mode coupling

| Instruction | Does | Aliases |
|---|---|---|
| `EMIT qubit mode` | Exchange \|1>\|0> and \|0>\|1> between a qubit and a mode: an excited emitter puts its excitation into the mode as a photon. Running it again absorbs the photon back. | PHOTONEMIT, PEMIT, PE, ABSORB |
| `DISPERSIVE qubit mode angle` | Dispersive coupling e^(i·angle·n) applied only where the qubit is 1. |  |

## Simulator-only inspection

| Instruction | Does | Aliases |
|---|---|---|
| `DUMPSTATE` | *Simulator only.* Print the exact state at this point. | DSTATE, DS |
| `EXPECT qubit basis [to]` | *Simulator only.* Exact <X>, <Y> or <Z> of a qubit, without disturbing it. Printed, or stored in a register. | QUANTUMSTATETOMOGRAPHY, QST, QSTAT |
| `BELLCHECK a b [to]` | *Simulator only.* Exact fidelity of qubits (a, b) with (\|00> + \|11>)/sqrt(2). | BELLSTATEVERIFICATION, BSV, BSTATE |
| `MEANPHOTON mode [to]` | *Simulator only.* Exact mean photon number <n> of a mode. | NMEAN |

## Classical registers

| Instruction | Does | Aliases |
|---|---|---|
| `REGSET reg value` | Set a register to a value. | SET, RSET, RGST |
| `REGCOPY dst src` | Copy one register into another. | MOV, RCOPY, RC |
| `ADD dst a b` | dst = a + b | REGADD, RADD, RGA |
| `SUB dst a b` | dst = a - b | REGSUB, RSUB, RGS |
| `MUL dst a b` | dst = a * b | REGMUL |
| `DIV dst a b` | dst = a / b (division by zero stops the program) | REGDIV |
| `ANDBITS dst a b` | Bitwise AND of the values as 64-bit integers. | AND, ANDB, AB |
| `ORBITS dst a b` | Bitwise OR of the values as 64-bit integers. | OR, ORB, OB |
| `XORBITS dst a b` | Bitwise XOR of the values as 64-bit integers. | XOR, XORB, XB |
| `NOTBITS dst src` | Bitwise NOT of the value as a 64-bit integer. | NOTB, NB |
| `SHL dst src amount` | Shift left by the amount in a register. |  |
| `SHR dst src amount` | Shift right by the amount in a register. |  |
| `SQRT dst src` | Square root. | SR |
| `EXP dst src` | e to the power of the value. |  |
| `LOG dst src` | Natural logarithm. | LN |
| `RAND reg` | A uniform random number in [0, 1). | RN |
| `SEEDRNG seed` | Reseed the random number generator. Measurements use it too, so every shot then repeats the same outcomes. | SRNG |
| `INPUT reg` | Read a number from standard input. | INP |
| `GETTIME reg` | Seconds since the Unix epoch. | GTIME, GT |
| `PUSH reg` | Push a register onto the stack. | PUSHREG |
| `POP reg` | Pop the stack into a register. | POPREG |
| `CHARLOAD reg char` | Load a character code into a register ("A" or 65). | CLOAD, CLD |
| `PACKBITS dst bits...` | Read classical bits as a whole number into a register. The first bit listed is the lowest; bits never written count as 0 (as in OpenQASM). |  |

## Control flow

| Instruction | Does | Aliases |
|---|---|---|
| `JUMP label` | Continue at a label. | JMP |
| `JUMPIFZERO value label` | Jump if the bit or register is 0. | JIZ, JZ |
| `JUMPIFONE value label` | Jump if the bit or register is 1. | JIO, JO |
| `IFEQ a b label` | Jump if a == b. | IEQ |
| `IFNE a b label` | Jump if a != b. | INE |
| `IFGT a b label` | Jump if a > b. | IGT |
| `IFLT a b label` | Jump if a < b. | ILT |
| `CALL label` | Call a subroutine at a label. | CALLLABEL |
| `RETSUB` | Return from a subroutine. | RET, RS |
| `LOOPSTART count` | Run the block up to the matching LOOPEND this many times (a number or a register). | LOOP, LSTART, LS |
| `LOOPEND` | End of a LOOPSTART block. | ENDLOOP, LEND, LE |
| `HALT` | Stop the program. | HLT, END |
| `BARRIER [qubits...]` | Does nothing in simulation. Marks a layer boundary and is kept for OpenQASM round trips. | BR |

## Output

| Instruction | Does | Aliases |
|---|---|---|
| `PRINT text` | Print text. |  |
| `PRINTLN [text]` | Print text and a newline. | PLN |
| `PRINTF format [regs...]` | Print text with each {} replaced by the next register's value. | PF |
| `CHAROUT reg` | Print a register's value as a character. | COUT, CO |
| `VERBOSELOG qubit text` | Print a message tagged with a qubit. | VLOG, VL |
| `DUMPREGS` | Print every register. | DREGS, DR |
| `COMMENT` | The rest of the line is a comment (like ;). | CMT, CM |

## Removed in 0.4

These were accepted by QOA 0.3 but did nothing, or had no defined meaning. Using them is now an error that explains what to use instead.

| Instruction | Why |
|---|---|
| `SETPOS` | position bookkeeping that never affected the quantum state |
| `SETWL` | wavelength bookkeeping that never affected the quantum state |
| `WLSHIFT` | wavelength bookkeeping that never affected the quantum state |
| `MOVE` | position bookkeeping that never affected the quantum state |
| `MARKOBSERVED` | it had no effect; a measured qubit is simply measured |
| `RELEASE` | it had no effect; a simulator has nothing to release |
| `DECOHERENCEPROTECT` | there is no physical model behind it; study decoherence with `qoa run --noise` |
| `ENTANGLESWAPMEASURE` | use ESWAP, then branch on the measured bits with JUMPIFONE |
| `ENTANGLEWITHCLASSICALFEEDBACK` | use gates plus JUMPIFONE/IFEQ for classical feedback |
| `ENTANGLEDISTRIBUTED` | networked entanglement between machines is not modelled |
| `APPLYNONLINEARPHASESHIFT` | it was never defined; use KERR or CROSSKERR |
| `APPLYNONLINEARSIGMA` | it was never defined; use KERR or CROSSKERR |
| `QUANTUMZENOEFFECT` | it needs continuous-time evolution, which QOA does not model; repeated MEASURE in a LOOPSTART shows the freezing |
| `APPLYLINEAROPTICALTRANSFORM` | build the network from BS and MODEPHASE; every linear optical unitary decomposes into them |
| `ERRORCORRECT` | no error-correcting code was ever specified; write the correction circuit with gates |
| `ERRORSYNDROME` | no code was ever specified; write the syndrome circuit with gates and MEASURE |
| `PHOTONROUTE` | ports were never defined; use MODESWAP or BS |
| `TIMEDELAY` | an ideal delay changes nothing; model a real one with LOSS and MODEPHASE |
| `OPTICALDELAYLINECONTROL` | an ideal delay changes nothing; model a real one with LOSS and MODEPHASE |
| `WAIT` | an ideal wait changes nothing; use `qoa run --noise` to study idle errors |
| `PHOTONBUNCHINGCONTROL` | bunching comes from interference (see BS), it is not a switch |
| `SINGLEPHOTONSOURCEOFF` | sources are instantaneous (SINGLEPHOTON); there is nothing to switch off |
| `APPLYDISPLACEMENTOPERATOR` | use DISPLACE |
| `OPTICALSWITCHCONTROL` | use MODESWAP to switch |
| `MEASUREWITHDELAY` | an ideal delay changes nothing; use MEASURE |
| `PHOTONLOSSCORRECTION` | no loss-correcting code was ever specified |
| `PHOTONEMISSIONPATTERN` | it was never defined; use EMIT inside LOOPSTART |
| `APPLYFEEDFORWARDGATE` | use JUMPIFONE or IFEQ to apply gates conditionally |
| `THERMALAVG` | not a quantum-circuit operation |
| `WKBFACTOR` | not a quantum-circuit operation |
| `CMP` | IFEQ, IFNE, IFGT and IFLT compare registers directly |
| `JMPABS` | byte addresses no longer exist; jump to a label |
| `CALLADDR` | byte addresses no longer exist; CALL a label |
| `LOAD` | memory was never implemented; use the 32 registers and PUSH/POP |
| `STORE` | memory was never implemented; use the 32 registers and PUSH/POP |
| `LOADMEM` | memory was never implemented; use the 32 registers and PUSH/POP |
| `STOREMEM` | memory was never implemented; use the 32 registers and PUSH/POP |
| `LOADCLASSICAL` | memory was never implemented; use the 32 registers and PUSH/POP |
| `STORECLASSICAL` | memory was never implemented; use the 32 registers and PUSH/POP |
| `ALLOC` | memory was never implemented; use the 32 registers and PUSH/POP |
| `FREE` | memory was never implemented; use the 32 registers and PUSH/POP |
| `BREAKPOINT` | use DUMPSTATE and DUMPREGS to look at a point in the program |
| `EXITCODE` | use HALT |
| `QUBITS` | use QINIT with the qubit count, e.g. QINIT 25 |
| `QGATE` | use GATE name qubit, or the gate's own instruction |
