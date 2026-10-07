# QOA glyph syntax (draft 0)

[Euskaraz: glyph.eu.md](glyph.eu.md)

**Status: a draft for review.** It is implemented and tested, so it can be tried, but every choice below is
a proposal. The open questions at the end are the decisions that are yours to make.

## Why

The plain QOA syntax is one instruction per line, spelled out: `CNOT q0 q1`. The concept sketch this draft
starts from packs a whole idea into one line:

```
H/R{3}[0xMEMORY_LOCATION?]; CC"V"%3$>#2PF"Measuring:" PR"R"%O$>#OPF"Result:" PF"O"%O$<<>>; END;
```

The glyph syntax keeps that density and stays just as direct about where things go. It is **not a second
language**: every glyph statement turns into ordinary QOA instructions, `qoa import file.qg` prints exactly
which ones, and errors point at the glyph that caused them.

```sh
qoa run examples/bell.qg        # run a glyph program
qoa import examples/bell.qg     # show the plain QOA it stands for
qoa check examples/bell.qg      # assemble and report, without running
```

## A first program

```
Q{2}; #C{2}; H/Q[0]; CX/Q[0],Q[1]; M/Q$>#C;
```

reads "two qubits Q, two bits C; H on Q[0]; CNOT from Q[0] to Q[1]; measure Q into C", and is the same
program as

```
QINIT 2
H q0
CNOT q0 q1
MEASURE q0 -> c0
MEASURE q1 -> c1
```

## Reading the concept

| concept | draft 0 reading |
|---|---|
| `H/R{3}` | `R{3}` declares a register of 3 qubits; `H/R` applies H to each of them |
| `[0xMEMORY_LOCATION?]` | `[k]` picks element k of a register (hexadecimal allowed: `R[0x2]`) |
| `$>#2` | `$>` routes a result ("goes into"); `#2` is classical register R2 |
| `PF"Measuring:"` | `PF` is already QOA's alias for PRINTF; text follows directly |
| `%O`, `%3` | `%` attaches a value to print (`PF"{}"%#2`); in a mode declaration, the cutoff (`~L{2}%6`) |
| `<<>>` | show the whole state (DUMPSTATE) |
| `;` | ends a statement, so a line can hold many |
| `END;` | stop (already QOA's alias for HALT) |
| `CC"V"%3`, `PR"R"%O` | not mapped yet: see the open questions |

[examples/concept.qg](../examples/concept.qg) is the concept written out as a working draft-0 program.

## Reference

### Statements and comments

- Every statement ends with `;`. Any number of statements fit on a line; newlines mean nothing.
- A line whose first character is `;` is a comment (as in the concept). `//` starts a comment anywhere.

### Registers

| glyph | meaning |
|---|---|
| `Q{4}` | 4 qubits named Q. Qubits are numbered in declaration order: the first register starts at q0 |
| `#C{4}` | 4 classical bits named C |
| `~L{2}%6` | 2 optical modes named L, each holding 0..5 photons (all mode registers share one cutoff; default 8) |

A declaration can stand alone (`Q{4};`) or appear at its first use (`H/Q{4};`). After that the name is
used without braces.

### Operands

| glyph | meaning | plain QOA |
|---|---|---|
| `Q` | every element of Q | `q0`, `q1`, ... |
| `Q[2]`, `Q[0x2]` | one element | `q2` |
| `Q[1..3]` | elements 1 to 3, inclusive | `q1`, `q2`, `q3` |
| `#C`, `#C[0]` | bits | `c0`, ... |
| `#2` | classical register 2 | `R2` |
| `(pi/4)`, `1.5` | a number or an expression | the same |
| `q3`, `o1`, `c0`, `R2` | plain QOA operands are accepted as they are | the same |

**Whole registers spread over elements.** `H/Q` is one H per qubit. With several operands, registers
must have the same size and go element by element, and single elements repeat: with `A{2}` and `B{2}`,
`CX/A,B` is `CNOT q0 q2` and `CNOT q1 q3`.

**A bare number is never an address.** In plain QOA `3` in a qubit slot means q3 and in a register slot
R3; in a glyph program that is an error, because `ADD/#1,#1,1` looks like "add one" but would add R1.
Write `#1`, `Q[1]` or `q1`.

### Instructions

| glyph | meaning | plain QOA |
|---|---|---|
| `H/Q[0]` | any QOA instruction (or alias) applied to operands | `H q0` |
| `RZ(pi/4)/Q[0]` | parameters in parentheses, after the operands in plain QOA | `RZ q0 pi/4` |
| `M/Q[0]$>#C[0]` | `$>` routes the result | `MEASURE q0 -> c0` |
| `COUNT/L[0]$>#3` | | `COUNT o0 -> R3` |
| `PF"x = {}\n"%#3` | print, `%` gives each `{}` its value | `PRINTF "x = {}\n" R3` |
| `PF"{}"%#C` | a bit register prints as a whole number | `PACKBITS R31 c0 c1` then `PRINTF "{}" R31` |
| `<<>>` | show the exact state | `DUMPSTATE` |
| `<<#>>` | show every register | `DUMPREGS` |
| `name:` | label | `name:` |
| `>name` | jump | `JUMP name` |
| `?x>name` | jump if bit or register x is 1 | `JUMPIFONE x name` |
| `!x>name` | jump if it is 0 | `JUMPIFZERO x name` |
| `*3{ ... }`, `*#2{ ... }` | repeat a block | `LOOPSTART 3` ... `LOOPEND` |
| `&name` | call a subroutine | `CALL name` |
| `^` | return from it | `RETSUB` |
| `` `any plain QOA` `` | escape: the text between backticks is used as it is | itself |
| `END` | stop | `HALT` |

Short names that plain QOA does not have: `M` for MEASURE and `CX` for CNOT. Every other instruction uses
its QOA name or alias (`qoa isa` lists them). Printing a bit register uses R31 (then R30, ... for more
values in the same statement) as scratch, so glyph programs should leave the top registers free.

## Examples

[examples/](../examples): `concept.qg`, `bell.qg`, `teleport.qg`, `hom.qg` (two emitters and a
beamsplitter), `retry.qg`. Each states its expected results in header comments, and the test suite
checks them; `tests/glyph.rs` also checks that the glyph programs give exactly the same counts as their
plain twins.

Teleportation, glyph and plain side by side:

```
Q{3}; #C{3};                                    QINIT 3
U(1.0, 0.5, -0.3)/Q[0];                         U q0 1.0 0.5 -0.3
BELL/Q[1],Q[2]; CX/Q[0],Q[1]; H/Q[0];           BELL q1 q2 / CNOT q0 q1 / H q0
M/Q[0..1]$>#C[0..1];                            MEASURE q0 -> c0 / MEASURE q1 -> c1
!#C[1]>x_gabe; X/Q[2]; x_gabe:;                 JUMPIFZERO c1 x_gabe / X q2 / x_gabe:
!#C[0]>z_gabe; Z/Q[2]; z_gabe:;                 JUMPIFZERO c0 z_gabe / Z q2 / z_gabe:
U(-1.0, 0.3, -0.5)/Q[2]; M/Q[2]$>#C[2];         U q2 -1.0 0.3 -0.5 / MEASURE q2 -> c2
```

## Open questions

1. **Separators.** The concept chains statements without `;` (`$>#2PF"Measuring:"`). Draft 0 requires `;`
   so that an error can always say exactly where a statement ended. Should a new instruction name end the
   previous statement on its own?
2. **`CC` and `PR`.** What did `CC"V"%3` and `PR"R"%O$>#O` stand for? (A guess: CC = copy classical, PR =
   print register.) They can become glyphs once their meaning is settled.
3. **`%`.** The concept uses `%3` and `%O` like counts or formats; draft 0 uses `%` only for print values
   and the mode cutoff. Is there a meaning you had in mind that should win?
4. **`[0x…]`.** Draft 0 treats it as an element index. Did you mean a raw address (a position in the state
   vector, or classical memory)?
5. **Names in plain QOA.** Named registers make the glyph form readable. Should plain QOA get them too
   (`QREG R 3` and `R[0]`)?
6. **The other direction.** `qoa export --to glyph` (plain QOA to glyph) is not written yet.
