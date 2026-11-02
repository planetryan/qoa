# QOA instruction reference

Generated from `src/isa.rs` by `qoa isa --markdown`. Do not edit by hand.

Mnemonics are case-insensitive. Every alias does exactly what its main name does.

## Short forms

From QOA 0.4.5 every instruction has a short name (the *Short* column), the preferred way to write it: `M q0` is `MEASURE q0`. Three more short forms, in plain QOA and in the glyph syntax, give the same program with fewer characters:

| Short | Long | Rule |
|---|---|---|
| `Mq0` | `MEASURE q0` | an instruction may touch its first operand when that operand is a plain register or range of its kind (`q3`, `o1`, `c0`, `R2`, `q0..q3`) |
| `M q0.c0` | `MEASURE q0 -> c0` | `.` routes a result, like `->` |
| `Mq0.c0` | `MEASURE q0 -> c0` | both together |

Short names that were not aliases before 0.4.5 (`D`, `K`, `N`, `E`, `C`, `J`, ...) never reserve a name: a register or constant called `L` or `C` keeps working. The long forms keep working everywhere.

## Operands

| Written as | Meaning |
|---|---|
| `q3` | qubit: q3 (or 3), or data[1] after QREG; ranges like q0..q3 where allowed |
| `o1` | optical mode: o1 (or 1) |
| `R2` | register: R2 (or 2), R0..R31 |
| `R2` / `(10)` | register or number: R2 (or 2), or a number in parentheses like (10), (pi/2) or (newline) |
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

| Instruction | Short | Does | Aliases |
|---|---|---|---|
| `QINIT qubits` | `QI` | Declare how many qubits the program uses (once). Without it, the count is the highest qubit index used plus one. | QI, INITQUBIT, IQ, QINITQ |
| `QMODE modes [cutoff]` | `QO` | Declare optical modes, each holding 0..cutoff-1 photons (default cutoff 8). Probability pushed past the cutoff is reported after the run. | MODES |
| `QREG name size [first]` | `QREG` | Name qubits (optional): after QREG data 3, data[0]..data[2] (or data, all three) stand for the next 3 unnamed qubits, or for qubits from `first` if given. q0, q1, ... keep working. |  |
| `CREG name size [first]` | `CREG` | Name classical bits (optional), like QREG: CREG flags 2 makes flags[0], flags[1] and flags. |  |
| `OREG name size [first]` | `OREG` | Name optical modes (optional), like QREG: OREG arm 2 makes arm[0], arm[1] and arm. |  |
| `RREG name size [first]` | `RREG` | Name classical registers (optional): after RREG total 2, total[0] and total[1] (or total, both) stand for the next 2 unnamed registers, or for registers from `first` if given. R0..R31 keep working. |  |
| `CONST name value` | `CONST` | Name a number: after CONST newline 10, newline can be used in any expression (RX q0 newline/4, ADD R1 R1 (newline)). A constant can use the constants defined above it. |  |

## Qubit gates

| Instruction | Short | Does | Aliases |
|---|---|---|---|
| `H qubit` | `H` | Hadamard. Accepts a qubit range (q0..q3). | HAD, APPLYHADAMARD |
| `X qubit` | `X` | Pauli-X (bit flip). Accepts a qubit range (q0..q3). | APPLYBITFLIP |
| `Y qubit` | `Y` | Pauli-Y. Accepts a qubit range (q0..q3). |  |
| `Z qubit` | `Z` | Pauli-Z (phase flip). Accepts a qubit range (q0..q3). | APPLYPHASEFLIP |
| `S qubit` | `S` | S = diag(1, i). Accepts a qubit range (q0..q3). | APPLYSGATE |
| `SDG qubit` | `SDG` | S† = diag(1, -i). Accepts a qubit range (q0..q3). | SDAG |
| `T qubit` | `T` | T = diag(1, e^(i·pi/4)). Accepts a qubit range (q0..q3). | APPLYTGATE |
| `TDG qubit` | `TDG` | T† = diag(1, e^(-i·pi/4)). Accepts a qubit range (q0..q3). | TDAG |
| `SX qubit` | `SX` | Square root of X. Accepts a qubit range (q0..q3). | SQRTX |
| `SXDG qubit` | `SXDG` | Inverse of SX. Accepts a qubit range (q0..q3). | SXDAG |
| `PHASE qubit angle` | `P` | Phase gate diag(1, e^(i·angle)). Accepts a qubit range (q0..q3). | P, PHASESHIFT, SETPHASE, SETP, SPH, PSE, U1 |
| `RX qubit angle` | `RX` | Rotation about X: exp(-i·angle·X/2). Accepts a qubit range (q0..q3). |  |
| `RY qubit angle` | `RY` | Rotation about Y: exp(-i·angle·Y/2). Accepts a qubit range (q0..q3). |  |
| `RZ qubit angle` | `RZ` | Rotation about Z: exp(-i·angle·Z/2). Accepts a qubit range (q0..q3). |  |
| `U qubit theta phi lambda` | `U` | General single-qubit gate U(theta, phi, lambda), OpenQASM convention. Accepts a qubit range (q0..q3). | U3 |
| `ROT qubit axis angle` | `ROT` | Rotation about axis X, Y or Z (the same as RX, RY or RZ). Accepts a qubit range (q0..q3). | APPLYROTATION, AR |
| `MROT axis angle qubits...` | `MROT` | The same rotation on every listed qubit. | APPLYMULTIQUBITROTATION, AMQR |
| `GATE name qubit` | `G` | Apply a named single-qubit gate: h, x, y, z, s, sdg, t, tdg, sx, sxdg. | APPLYGATE, AGATE, AG |
| `CNOT control target` | `CX` | Controlled-X: flip the target when the control is 1. | CX, CONTROLLEDNOT, CN, ENTANGLE |
| `CY control target` | `CY` | Controlled-Y. |  |
| `CZ control target` | `CZ` | Controlled-Z: phase -1 on \|11>. |  |
| `CH control target` | `CH` | Controlled-Hadamard. |  |
| `CSX control target` | `CSX` | Controlled square root of X. |  |
| `CU control target theta phi lambda` | `CU` | Controlled U(theta, phi, lambda). | CU3 |
| `CPHASE control target angle` | `CP` | Controlled phase: e^(i·angle) on \|11>. | CP, CONTROLLEDPHASEROTATION, APPLYCPHASE, CPR, CU1 |
| `CRX control target angle` | `CRX` | Controlled RX. |  |
| `CRY control target angle` | `CRY` | Controlled RY. |  |
| `CRZ control target angle` | `CRZ` | Controlled RZ. |  |
| `SWAP a b` | `SWAP` | Swap two qubits. |  |
| `CSWAP control a b` | `CS` | Swap a and b when the control is 1 (Fredkin). | CONTROLLEDSWAP, CS, FREDKIN |
| `CCX control1 control2 target` | `CCX` | Toffoli: flip the target when both controls are 1. | TOFFOLI, CCNOT |
| `MCX controls... target` | `MCX` | Flip the target (the last qubit) when every other listed qubit is 1. |  |
| `MCZ qubits...` | `MCZ` | Phase -1 on the basis states where every listed qubit is 1. |  |

## Entangling shorthands

| Instruction | Short | Does | Aliases |
|---|---|---|---|
| `BELL a b` | `B` | Runs H a, then CNOT a b. From \|00> this gives (\|00> + \|11>)/sqrt(2). | ENTANGLEBELL, EBELL, EB |
| `GHZ qubits...` | `GHZ` | Runs H on the first qubit, then CNOT from it to each of the others. From \|0..0> this gives a GHZ state. | ENTANGLEMULTI, EMULTI, EM |
| `CLUSTER qubits...` | `CL` | Linear cluster state: H on every qubit, then CZ between each neighbouring pair. | ENTANGLECLUSTER, ECLUSTER, ECR |
| `ESWAP a b c d` | `ESP` | Entanglement swapping. With Bell pairs (a,b) and (c,d): Bell-measures b and c (into c<b> and c<c>), then corrects d, leaving (a,d) as a Bell pair. | ENTANGLESWAP, ESP |

## Qubit measurement and reset

| Instruction | Short | Does | Aliases |
|---|---|---|---|
| `MEASURE qubit [-> to]` | `M` | Measure in the Z basis. The result goes to bit c<qubit> unless routed with -> cK or -> RK. Accepts a qubit range (q0..q3). | MEAS, M, QMEAS, QM, APPLYQNDMEASUREMENT, AQND, AQAD, AQNM |
| `MEASUREIN qubit basis [-> to]` | `MI` | Measure in the X, Y or Z basis (rotates into Z, measures, rotates back). | MEASUREINBASIS, MEASB, MIB |
| `MEASUREALL` | `MA` | Measure every qubit, each into its own bit c<qubit>. | MALL |
| `RESET qubit` | `RST` | Put a qubit in \|0>: measure it, then flip it if it read 1. Accepts a qubit range (q0..q3). | RST, RSTQ, QRESET, QR |
| `RESETALL` | `RA` | Every qubit to \|0> and every mode to vacuum. | RSTALL, RSA |
| `TOBASIS qubit basis` | `TB` | Rotate so that a Z measurement reads the given basis (X: H; Y: SDG then H; Z: nothing). Accepts a qubit range (q0..q3). | APPLYMEASUREMENTBASISCHANGE, AMBC |

## Optical modes

| Instruction | Short | Does | Aliases |
|---|---|---|---|
| `DISPLACE mode re [im]` | `D` | Displacement D(alpha), alpha = re + i·im. Turns vacuum into a coherent state with mean photon number \|alpha\|^2. | APPLYDISPLACEMENT, ADISP, AD |
| `DISPLACEREG mode alpha` | `DV` | Displacement by the (real) value in a register: classical feed-forward. | APPLYDISPLACEMENTFEEDBACK |
| `SQUEEZE mode r [phi]` | `SQ` | Squeezing S(r·e^(i·phi)). With phi = 0 and r > 0 the x quadrature is squeezed. | APPLYSQUEEZING, ASQ, AS |
| `SQUEEZEREG mode r` | `SQV` | Squeezing by the value in a register: classical feed-forward. | APPLYSQUEEZINGFEEDBACK, ASWF, ASF, ASFB |
| `MODEPHASE mode angle` | `PH` | Phase rotation e^(i·angle·n) of a mode. | DYNAMICPHASECOMPENSATION, DPC, DPCMP, PHASEROT |
| `KERR mode kappa [time]` | `K` | Self-Kerr nonlinearity e^(i·kappa·time·n^2) (time defaults to 1). | APPLYKERRNONLINEARITY, AKNL |
| `CROSSKERR a b kappa` | `XK` | Cross-Kerr (cross-phase modulation) e^(i·kappa·n_a·n_b). | CROSSPHASEMODULATION, CPM, CPMOD |
| `BS a b theta [phi]` | `BS` | Beamsplitter exp(theta·(e^(i·phi)·a·b† - e^(-i·phi)·a†·b)). theta = pi/4 is 50:50. | BEAMSPLITTER |
| `MODESWAP a b` | `MS` | Exchange the contents of two modes (an ideal router). | OPTICALROUTING, OROUTE, OPTR |
| `ADDPHOTON mode` | `PA` | Apply a† and renormalize. Post-selected: real photon addition only succeeds some of the time. | PHOTONADDITION, PADD, PA |
| `SUBPHOTON mode` | `PS` | Apply a and renormalize (post-selected). Fails if the mode is certainly empty. | APPLYPHOTONSUBTRACTION, APSUB, APS |
| `LOSS mode probability` | `L` | Each photon in the mode is lost with this probability. One random outcome per run; averaged over shots it is the exact loss channel. | PHOTONLOSS, PHOTONLOSSSIMULATE, PLS, PLSIM, SETOPTICALATTENUATION, SOATT, SOA |
| `FOCK mode photons` | `F` | Discard what the mode held and prepare exactly that many photons. |  |
| `SINGLEPHOTON mode` | `SP` | An ideal single-photon source: the same as FOCK mode 1. | SINGLEPHOTONSOURCEON, SPSON |
| `COUNT mode [-> to]` | `N` | Number-resolving photon detection. The count goes to the mode's readout unless routed with -> RK. | PHOTONCOUNT, PCOUNT, PC, PHOTONNUMBERRESOLVINGDETECTION, PNRD, PNR |
| `DETECT mode [-> to]` | `PD` | Threshold (click) detection: 1 if any photon is present, else 0. | PHOTONDETECT, PDETECT, PD |
| `PARITY mode [-> to]` | `MP` | Photon-number parity measurement: 0 for even, 1 for odd. | MEASUREPARITY, MPAR, MP |
| `COINCIDENCE modes... -> to` | `CD` | Click-detect every listed mode; writes 1 if all of them clicked. | PHOTONDETECTCOINCIDENCE, PDCOIN, PDC |

## Qubit-mode coupling

| Instruction | Short | Does | Aliases |
|---|---|---|---|
| `EMIT qubit mode` | `E` | Exchange \|1>\|0> and \|0>\|1> between a qubit and a mode: an excited emitter puts its excitation into the mode as a photon. Running it again absorbs the photon back. | PHOTONEMIT, PEMIT, PE, ABSORB |
| `DISPERSIVE qubit mode angle` | `DSP` | Dispersive coupling e^(i·angle·n) applied only where the qubit is 1. |  |

## Simulator-only inspection

| Instruction | Short | Does | Aliases |
|---|---|---|---|
| `DUMPSTATE` | `DS` | *Simulator only.* Print the exact state at this point. | DSTATE, DS |
| `EXPECT qubit basis [to]` | `EV` | *Simulator only.* Exact <X>, <Y> or <Z> of a qubit, without disturbing it. Printed, or stored in a register. | QUANTUMSTATETOMOGRAPHY, QST, QSTAT |
| `BELLCHECK a b [to]` | `BC` | *Simulator only.* Exact fidelity of qubits (a, b) with (\|00> + \|11>)/sqrt(2). | BELLSTATEVERIFICATION, BSV, BSTATE |
| `MEANPHOTON mode [to]` | `MN` | *Simulator only.* Exact mean photon number <n> of a mode. | NMEAN |

## Classical registers

| Instruction | Short | Does | Aliases |
|---|---|---|---|
| `REGSET reg value` | `SET` | Set a register to a value. | SET, RSET, RGST |
| `REGCOPY dst src` | `RC` | Copy one register into another. | MOV, RCOPY, RC |
| `ADD dst a b` | `ADD` | dst = a + b | REGADD, RADD, RGA |
| `SUB dst a b` | `SUB` | dst = a - b | REGSUB, RSUB, RGS |
| `MUL dst a b` | `MUL` | dst = a * b | REGMUL |
| `DIV dst a b` | `DIV` | dst = a / b (division by zero stops the program) | REGDIV |
| `ANDBITS dst a b` | `AND` | Bitwise AND of the values as 64-bit integers. | AND, ANDB, AB |
| `ORBITS dst a b` | `OR` | Bitwise OR of the values as 64-bit integers. | OR, ORB, OB |
| `XORBITS dst a b` | `XOR` | Bitwise XOR of the values as 64-bit integers. | XOR, XORB, XB |
| `NOTBITS dst src` | `NOT` | Bitwise NOT of the value as a 64-bit integer. | NOTB, NB |
| `SHL dst src amount` | `SHL` | Shift left by the amount in a register. |  |
| `SHR dst src amount` | `SHR` | Shift right by the amount in a register. |  |
| `SQRT dst src` | `SQRT` | Square root. | SR |
| `EXP dst src` | `EXP` | e to the power of the value. |  |
| `LOG dst src` | `LN` | Natural logarithm. | LN |
| `SIN dst src` | `SIN` | Sine of an angle in radians. |  |
| `COS dst src` | `COS` | Cosine of an angle in radians. |  |
| `TAN dst src` | `TAN` | Tangent of an angle in radians. |  |
| `POW dst a b` | `POW` | a to the power of b. |  |
| `RAND reg` | `RN` | A uniform random number in [0, 1). | RN |
| `SEEDRNG seed` | `SRNG` | Reseed the random number generator. Measurements use it too, so every shot then repeats the same outcomes. | SRNG |
| `INPUT reg` | `INP` | Read a number from standard input. | INP |
| `GETTIME reg` | `GT` | Seconds since the Unix epoch. | GTIME, GT |
| `PUSH reg` | `PUSH` | Push a register onto the stack. | PUSHREG |
| `POP reg` | `POP` | Pop the stack into a register. | POPREG |
| `CHARLOAD reg char` | `CLD` | Load a character code into a register ("A" or 65). | CLOAD, CLD |
| `PACKBITS dst bits...` | `PB` | Read classical bits as a whole number into a register. The first bit listed is the lowest; bits never written count as 0 (as in OpenQASM). |  |

## Memory, bytes and arguments

| Instruction | Short | Does | Aliases |
|---|---|---|---|
| `LOAD dst addr` | `LD` | Read the memory cell whose address is in addr. Memory is cells 0, 1, 2, ... (up to 2^32 of them), each holding one number; a cell never written holds 0. | LD |
| `STORE src addr` | `ST` | Write src into the memory cell whose address is in addr. | ST |
| `STORETEXT addr text` | `STX` | Write the bytes of the text (UTF-8) into memory from the address in addr, one byte per cell, then a 0; addr then holds the address after the 0. | STEXT |
| `GETC dst` | `GC` | Read the next byte of standard input (0..255), or -1 at the end of the input. | GETBYTE |
| `PUTC src` | `PUTC` | Write one byte (0..255) to the output exactly as it is, for binary output. | PUTBYTE |
| `STREAM stream` | `STREAM` | Send the output that follows to standard output (1) or to standard error (2). |  |
| `ARGC dst` | `ARGC` | How many arguments the program was given after -- (qoa run prog.qx -- a b). |  |
| `ARGBYTE dst arg index` | `ARGBYTE` | Byte number index of argument number arg (both counted from 0), or -1 past the end of the argument. |  |
| `GETLANG dst` | `GETLANG` | 1 when QOA is speaking Basque (QOA_LANG=eu), 0 for English, so a program can print its messages in the user's language. |  |

## Control flow

| Instruction | Short | Does | Aliases |
|---|---|---|---|
| `JUMP label` | `J` | Continue at a label. | JMP |
| `JUMPIFZERO value label` | `JZ` | Jump if the bit or register is 0. | JIZ, JZ |
| `JUMPIFONE value label` | `JO` | Jump if the bit or register is 1. | JIO, JO |
| `IFEQ a b label` | `IEQ` | Jump if a == b. | IEQ |
| `IFNE a b label` | `INE` | Jump if a != b. | INE |
| `IFGT a b label` | `IGT` | Jump if a > b. | IGT |
| `IFLT a b label` | `ILT` | Jump if a < b. | ILT |
| `CALL label` | `C` | Call a subroutine at a label. | CALLLABEL |
| `RETSUB` | `RET` | Return from a subroutine. | RET, RS |
| `LOOPSTART count` | `LS` | Run the block up to the matching LOOPEND this many times (a number or a register). | LOOP, LSTART, LS |
| `LOOPEND` | `LE` | End of a LOOPSTART block. | ENDLOOP, LEND, LE |
| `HALT` | `END` | Stop the program. | HLT, END |
| `EXIT status` | `EXIT` | Stop the program like HALT, with an exit status (0..255). The status of the first shot becomes the exit status of qoa run. |  |
| `BARRIER [qubits...]` | `BR` | Does nothing in simulation. Marks a layer boundary and is kept for OpenQASM round trips. | BR |

## Output

| Instruction | Short | Does | Aliases |
|---|---|---|---|
| `PRINT text` | `PRINT` | Print text. |  |
| `PRINTLN [text]` | `PLN` | Print text and a newline. | PLN |
| `PRINTF format [regs...]` | `PF` | Print text with each {} replaced by the next register's value. | PF |
| `CHAROUT reg` | `CO` | Print a register's value as a character. | COUT, CO |
| `VERBOSELOG qubit text` | `VL` | Print a message tagged with a qubit. | VLOG, VL |
| `DUMPREGS` | `DR` | Print every register. | DREGS, DR |
| `COMMENT` | `CM` | The rest of the line is a comment (like ;). | CMT, CM |

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
| `LOADMEM` | memory is now LOAD dst addr, with the address in a register |
| `STOREMEM` | memory is now STORE src addr, with the address in a register |
| `LOADCLASSICAL` | memory is now LOAD dst addr, with the address in a register |
| `STORECLASSICAL` | memory is now STORE src addr, with the address in a register |
| `ALLOC` | memory needs no allocation: LOAD and STORE use cells 0, 1, 2, ... directly |
| `FREE` | memory needs no freeing: LOAD and STORE use cells 0, 1, 2, ... directly |
| `BREAKPOINT` | use DUMPSTATE and DUMPREGS to look at a point in the program |
| `EXITCODE` | use HALT |
| `QUBITS` | use QINIT with the qubit count, e.g. QINIT 25 |
| `QGATE` | use GATE name qubit, or the gate's own instruction |
