# QOA glyph syntax

[Euskaraz: glyph.eu.md](glyph.eu.md)

**Status: experimental.** The syntax is implemented and tested, but it may still change in a later version;
any change will be listed in the release notes. The points that are not settled yet are at the end.

## Why

The plain QOA syntax is one instruction per line, spelled out: `CNOT q0 q1`. The glyph syntax started from a
one-line sketch that packs a whole idea into a single line:

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

## Reading the original sketch

| sketch | meaning |
|---|---|
| `H/R{3}` | `R{3}` declares a register of 3 qubits; `H/R` applies H to each of them |
| `[0xMEMORY_LOCATION?]` | right after a declaration, a location: `R{3}[0x4]` places R at q4..q6. After a name, an element: `R[0x2]` |
| `$>#2` | `$>` routes a result ("goes into"); `#2` is classical register R2 |
| `PF"Measuring:"` | `PF` is already QOA's alias for PRINTF; text follows directly |
| `%O`, `%3` | `%` starts a value and separates it from the next: `PF"{} {}"%#2%#3`, `CC[0x10]%3%#1`; in a mode declaration it starts the cutoff (`~L{2}%6`) |
| `<<>>` | show the whole state (DUMPSTATE) |
| `;` | ends a statement, so a line can hold many |
| `END;` | stop (already QOA's alias for HALT) |
| `CC"V"%3$>#2` | call classical: call routine V with the argument 3 and put its result in R2 (see [Classical calls](#classical-calls-cc)) |
| `PR"R"%O` | print with a label: `R = ` and the value (see [PR](#settled-in-042)) |

[examples/concept.qg](../examples/concept.qg) is that sketch written out as a working program.

## Reference

### Statements and comments

- Every statement ends with `;`. Any number of statements fit on a line; newlines mean nothing.
- A line whose first character is `;` is a comment (as in the original sketch). `//` starts a comment anywhere.

### Registers

| glyph | meaning |
|---|---|
| `Q{4}` | 4 qubits named Q. Qubits are numbered in declaration order: the first register starts at q0 |
| `Q{4}[0x4]` | the same, placed at a location: Q is q4..q7. Later registers without a location go after the highest one placed; overlapping registers are an error |
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

**A bare number is never an address.** In a glyph program a number is a number: `ADD/#1,#1,1` adds one,
`IFEQ/#1,10,done` compares R1 with ten, and `STORE/7,#3` stores seven at the address in R3 (in plain QOA
these are `ADD R1 R1 (1)`, `IFEQ R1 (10) done` and `STORE (7) R3`). Where only a qubit, a mode, a bit or a
register to write to fits, a bare number is an error, because plain QOA would read `3` there as q3 or R3.
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
| `PR"x"%#3` | print a label and values: `x = 3.5` | `PRINTF "x = {}\n" R3` |
| `<<>>` | show the exact state | `DUMPSTATE` |
| `<<#>>` | show every register | `DUMPREGS` |
| `name:` | label | `name:` |
| `0x10:` | a label that is an address, for `CC[0x10]` | `addr_0x10:` |
| `>name` | jump (to a name or an address: `>0x10`) | `JUMP name` |
| `?x>name` | jump if bit or register x is 1 | `JUMPIFONE x name` |
| `!x>name` | jump if it is 0 | `JUMPIFZERO x name` |
| `*3{ ... }`, `*#2{ ... }` | repeat a block | `LOOPSTART 3` ... `LOOPEND` |
| `&name` | call a subroutine (`&0x10` too) | `CALL name` |
| `CC[0x10]%a%b$>#2` | call classical: arguments on the stack, call, result into R2 | `PUSH`... `CALL addr_0x10` `POP R2` |
| `^` | return from it | `RETSUB` |
| `` `any plain QOA` `` | escape: the text between backticks is used as it is | itself |
| `END` | stop | `HALT` |

Short names that plain QOA does not have: `M` for MEASURE and `CX` for CNOT. Every other instruction uses
its QOA name or alias (`qoa isa` lists them). Printing a bit register uses R31 (then R30, ... for more
values in the same statement) as scratch, so glyph programs should leave the top registers free.

### Classical calls (CC)

`CC` hands values from the quantum program to a classical routine and gets one back, through a small
calling convention (ABI) on QOA's data stack, which is separate from the call stack:

1. the caller pushes the arguments (`%a%b`) last to first, so the routine pops the first argument first;
2. `CC[0x10]` calls the routine at address `0x10` (a label written `0x10:`), and `CC"name"` the one at a
   named label;
3. the routine pops its arguments (`POP/#1`), computes, pushes one result (`PUSH/#4`) and returns (`^`);
4. `$>#2` pops the result into R2. Without `$>`, the routine should push nothing.

An argument can be a number or expression (`%3`, `%(pi/2)`), a register (`%#1`) or bits (`%#C[0]`, or
`%#C` for a whole bit register as one number); qubits and modes have no classical value until they are
measured, so passing one is an error. Numbers and bits go through R31 on their way to the stack.

[examples/decoder.qg](../examples/decoder.qg) decodes a three-qubit repetition code: it measures the qubits
and calls a majority-vote routine at `0x10`:

```
M/Q$>#C; CC[0x10]%#C[0]%#C[1]%#C[2]$>#5; PF"decoded: {}\n"%#5; END;
0x10:; POP/#1; POP/#2; POP/#3; ADD/#4,#1,#2; ADD/#4,#4,#3; REGSET/#6,(1); SHR/#4,#4,#6; PUSH/#4; ^;
```

## From plain QOA to glyph

`qoa export --to glyph file.qoa` (or `--to-glyph`) writes any program as glyph syntax: from `.qoa`, `.qasm`,
`.qg` or a compiled `.qx`. Runs of the same gate become ranges (`H/Q[0..3]`, `M/Q[0..1]$>#C[0..1]`), jump
targets become address labels (`0x8:`), and the result is reassembled and compared with the original before
it is written. Options:

| option | effect |
|---|---|
| `--width N` | statements fill lines up to N characters (default 100; 0 = one statement per line) |
| `--names Q,C,L` | names of the qubit, bit and mode registers |
| `--long-names` | `MEASURE` and `CNOT` instead of `M` and `CX` |
| `--no-ranges` | one statement per element |
| `--hex` | indices in hexadecimal, like `Q[0x3]` |

## Examples

[examples/](../examples): `concept.qg`, `bell.qg`, `decoder.qg` (a classical call), `teleport.qg`,
`hom.qg` (two emitters and a beamsplitter), `retry.qg`. Each states its expected results in header comments, and the test suite
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

## Design choices

- **`CC`** is "call classical", a call to a memory location: `CC[0x10]` with the stack convention above.
- **`%`** is a logical starter and separator for values.
- **`[0x…]`** is a location right after a declaration, and an element index everywhere else.
- **Names in plain QOA** are available but never required: `QREG data 3`, `CREG flags 2`, `OREG arm 2`, then
  `data[1]` or `data`; `q0`, `c0` and `o0` keep working.
- **The other direction** exists: `qoa export --to glyph` (or `--to-glyph`) with the options above.

## Settled in 0.4.2

1. **Separators: `;` is required.** The original sketch chains statements without it (`$>#2PF"Measuring:"`).
   A new instruction name cannot end the previous statement on its own, because names are free: `PF` or `M`
   can be a register as well as an instruction, so `#PF` and `#P` followed by `F...` would read the same. With
   `;` every statement has a clear end, and an error can always say which statement it is in.
2. **`PR` prints with a label.** `PR"x"%#3` prints `x = 3`, and several values come out side by side
   (`PR"pair"%#1%#2` prints `pair = 1 2`). Values work as in `PF`: numbers, registers, bits, and bit registers
   as whole numbers. Qubits and modes have no value until they are measured, so they are an error. `PR"here"`
   alone prints just the label.
