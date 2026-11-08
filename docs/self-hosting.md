# QOA builds QOA

[Euskaraz: self-hosting.eu.md](self-hosting.eu.md)

QOA's assembler, the part that turns a `.qoa` text into a `.qx` binary, also exists written in QOA itself:
[src/qoa/asm.qoa](../src/qoa/asm.qoa). It runs on the QOA machine like any other QOA program, and it produces
exactly what `qoa compile` produces: the same bytes for every program, and for a program with mistakes the
same error messages, in the same order, word for word, in English and in Basque.

It also assembles itself. That is what "QOA builds QOA" means here: the copy of the assembler built by the
Rust assembler and the copy it builds of itself are identical, byte for byte.

## Using it

```sh
qoa compile --assembler src/qoa/asm.qoa program.qoa        # same result as: qoa compile program.qoa
qoa compile --self-hosted program.qoa                    # the same assembler, built into qoa itself
```

The assembler's source is `src/qoa/asm.qoa` in the repository, and from QOA 0.4.4 every release package
includes it too (in `tools/` until 0.4.7). `src/qoa/` holds the parts of QOA written in QOA, next to the
Rust sources, because they are compiled into the `qoa` binary; `tools/` holds helper scripts in Python and
shell. From QOA 0.4.7 the assembler is also built into the `qoa` binary: `--self-hosted` uses that copy,
so no file is needed, `qoa tools` lists the tools written in QOA that the binary carries, and
`qoa tools --write DIR` writes their sources out (`asm.qoa`, and `asm.qg` generated from it). A test checks
that the built-in copy is the file in `src/qoa/`, and that it gives the same bytes and errors as the Rust
assembler.

`--assembler` takes any QOA program, as source or as a compiled `.qx`. QOA runs it with the source file on
its input and the file name as its argument, and writes what it prints as the `.qx`. Its error messages go
straight to the terminal, and its exit status becomes `qoa`'s.

The same thing by hand, which shows that nothing hidden is involved:

```sh
qoa compile src/qoa/asm.qoa -o asm.qx                       # build the assembler (Rust does this once)
qoa run asm.qx --quiet -- program.qoa < program.qoa > program.qx
```

`--quiet` prints only what the program writes, with no report, and `qoa` exits with the program's `EXIT`
status. Everything after `--` is given to the program as arguments.

## The bootstrap

```sh
qoa compile src/qoa/asm.qoa -o stage1.qx                                   # built by Rust
qoa compile --assembler stage1.qx src/qoa/asm.qoa -o stage2.qx             # built by QOA
qoa compile --assembler stage2.qx src/qoa/asm.qoa -o stage3.qx             # built by the QOA-built QOA
cmp stage1.qx stage2.qx && cmp stage2.qx stage3.qx                       # all identical
```

The test suite runs exactly this on every build (`tests/selfhost.rs`).

## What is written in QOA, and what stays in Rust

| part | language | why |
|---|---|---|
| the assembler: lines, comments, tokens, labels, named registers, `q0..q3` ranges, every operand, expressions, decimal numbers, the binary format, every error message | **QOA** | it is a classical program that turns text into a list of operations, which a language can do for itself |
| the QOA machine: qubit states, photons, SIMD kernels, GPUs, MPI, noise | Rust | this is the numerical core; it needs 64-bit floating point at full machine speed, vector instructions and GPU drivers, which an interpreted program cannot provide |
| `sin`, `cos`, `tan`, `exp`, `ln`, `pow` | Rust (as instructions) | the same functions `qoa compile` uses, so both give the same bits |
| reading files, the command line | Rust | the operating system's side |
| the glyph syntax (`.qg`) and OpenQASM import | Rust, for now | see below |

The assembler is written in plain QOA, not in the glyph syntax. The glyph syntax is a compact notation for
circuits over registers of qubits and photons, and it turns into plain QOA. An assembler is classical code
(loops, tables, text), where plain QOA reads more clearly. Writing it in plain QOA also keeps the bootstrap
closed: it needs nothing but itself, while the glyph front end is still Rust. A glyph front end written in
QOA is possible future work.

## How it compares with Rust

Measured on the assembler's own source (12,000 lines) on a 4-core cloud machine, best of five runs:

| | Rust (`src/asm.rs`) | QOA (`src/qoa/asm.qoa`) |
|---|---|---|
| output | reference | identical bytes, identical messages |
| time | 30 ms | 0.95 s (about 30 times slower), of which about 0.35 s computes the SHA-256 of the 234 KB it writes |
| speed | about 400,000 lines per second | about 13,000 lines per second |
| size | about 3,000 lines for the assembler, expressions, the `.qx` writer, SHA-256 and seals, plus the instruction table | about 11,900 lines, plus about 300 lines of tables generated from Rust |

So Rust is about 30 times faster and about 4 times shorter for the same logic. (Before 0.4.5, without the
checksum, the QOA assembler took about 0.6 s on the same machine.) That is the cost of
assembly-style code on an interpreter: QOA has 32 number registers, numbered memory cells and jumps, and no
types, structures or strings. Every field access, comparison and constant is a separate instruction. In
return, the whole assembler is a QOA program, and anything that runs QOA can rebuild QOA's programs. A
13,000-lines-per-second assembler is still fast enough for any real QOA program: the largest program in the
repository is the assembler itself, and it builds in about a second.

## Why the two agree

- **Every program in the repository**: examples and all test programs, including the 58 that are meant to
  fail, are assembled by both in English and in Basque, comparing the exit status, the messages and the
  `.qx` bytes.
- **Generated programs**: about 7,000 accepted lines covering every instruction, alias and operand form, and
  3,000 random decimal numbers (long ones, tiny ones, halfway cases, subnormals) that test correct rounding;
  plus 165 programs full of mistakes that reach more than 50 different error messages, invalid UTF-8, Windows
  and old Mac line endings, and more than 30 errors at once.
- **Seals and checksums**: sealed programs of every length from 0 to 130 bytes (so every SHA-256 block
  boundary), with text outside ASCII, some changed after sealing and some with malformed seals. Every `.qx`
  either assembler writes ends with a SHA-256, so every comparison above also compares the two SHA-256
  implementations.
- **The fixed point** above.
- **The tables cannot drift**: the instruction table (names, aliases, operands, usage lines, the reasons for
  removed instructions) and the Unicode letter ranges inside `src/qoa/asm.qoa` are generated from `src/isa.rs`
  and from Rust's own character functions. A test fails if they differ.

A few places where matching Rust took care:

- **Decimal numbers** are rounded to the nearest 64-bit float exactly as Rust's parser does. Up to 15 digits
  with an exponent of at most 22, one exact multiplication or division does it. Anything else is divided out
  exactly with big integers and rounded half to even, which also covers subnormal numbers and literals with
  hundreds of digits.
- **Expressions** are evaluated in the same order as `src/expr.rs`, so `-0`, infinities in the middle of an
  expression (`exp(ln(0))` is 0) and `NaN` behave the same.
- **Text** is read as UTF-8 and rejected when invalid, with the same message as `qoa compile`; whitespace and
  letters follow Unicode as Rust defines them (`π` is pi, `U+3000` separates tokens).

## The instructions it needed

Writing an assembler in QOA needed a few classical instructions, added in QOA 0.4.3 and useful to any
program that works as a tool:

| instruction | does |
|---|---|
| `LOAD dst addr`, `STORE src addr` | memory: cells 0, 1, 2, ... (up to 2^32), each holding a number; unwritten cells read 0 |
| `STORETEXT addr "text"` | write a text's bytes into memory (for tables and messages) |
| `GETC dst`, `PUTC src` | read a byte from standard input (-1 at the end), write a byte as it is |
| `STREAM 1`, `STREAM 2` | send output to standard output or standard error |
| `ARGC dst`, `ARGBYTE dst arg i` | the program's arguments (`qoa run prog.qx -- a b`) |
| `GETLANG dst` | 1 when QOA speaks Basque, so a program can answer in the user's language |
| `EXIT n` | stop with an exit status |
| `SIN`, `COS`, `TAN`, `POW` | the remaining functions expressions use |

QOA 0.4.4 added numbers in parentheses (`ADD R1 R1 (1)`), `CONST` and `RREG`, which the assembler written in
QOA understands and uses too. [ISA.md](ISA.md) describes them all.

From 0.4.5 both assemblers write `.qx` format version 4, which ends with the SHA-256 of everything before it
(see [safety.md](safety.md#file-integrity)). The assembler written in QOA computes that SHA-256 itself, in QOA,
with the bit operations above, and checks `SEAL` lines on sealed sources the same way. Files in the older
formats (1 for 0.4.2, 2 with the 0.4.3 instructions, 3 with numbers in parentheses) still load.

## Limits

- It reads plain QOA (`.qoa`). Glyph (`.qg`) and OpenQASM (`.qasm`) sources still go through Rust.
- The same assembler also exists in the glyph syntax: [src/qoa/asm.qg](../src/qoa/asm.qg), generated from
  `src/qoa/asm.qoa` line by line with `qoa export --to glyph --keep-source` (every comment kept). A test checks
  that the two give the same program, byte for byte, so `qoa compile src/qoa/asm.qg` builds the same
  assembler. `src/qoa/asm.qoa` stays the file to edit.
- It keeps all its memory until it finishes (a few megabytes for its own source).
- `sin`, `cos`, `tan`, `exp`, `ln` and `pow` are the machine's, as in `qoa compile`. Both assemblers agree on
  the same machine; on a different machine both may round those few functions differently in the last bit.

## When an instruction is added to QOA

1. Add it to `src/isa.rs` and to `src/asm.rs` as usual.
2. Regenerate the tables: `QOA_REGENERATE=1 cargo test --test selfhost`.
3. In `src/qoa/asm.qoa`, give it a builder: a line in `buildtext` (name, builder number, parameter) and, if no
   existing builder fits, a new one in `build`.
4. `cargo test --test selfhost` compares both assemblers again; it fails until they agree.
