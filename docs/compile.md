# Compiling in stages

[Euskaraz: compile.eu.md](compile.eu.md)

`qoa compile` turns a program into its `.qx` bytecode. With `--emit` it can stop at any stage of the way
to machine code and write that stage out, so you can see what each step makes of your program, compare
stages, or debug one:

```sh
qoa compile program.qoa --emit qoa              # 1. canonical QOA text
qoa compile program.qoa --emit qx               # 2. the bytecode (the default)
qoa compile program.qoa --emit ir               # 3. basic blocks, each operation native or through QOA's machine
qoa compile program.qoa --emit asm --arch s390x # 4. assembly for one architecture
qoa compile program.qoa --emit obj              # 5. machine code: an object file (.o)
qoa compile program.qoa --emit so               # 6. a shared library (.so) ...
qoa run program.so                              #    ... that qoa runs, with the same results
```

Text stages (`qoa`, `ir`, `asm`) go to standard output unless you give `-o FILE`; `obj` and `so` are written
next to the source (`program.o`, `program.so`) or to `-o`. A `.qx` works as input too.

## Architectures

| `--arch` | processors | assembler syntax |
|---|---|---|
| `x86_64` | Intel and AMD 64-bit | AT&T |
| `aarch64` | ARM 64-bit (Graviton, Ampere, Apple chips under Linux, Raspberry Pi 4/5) | ARM |
| `riscv64` | RISC-V 64-bit (RV64GC) | RISC-V |
| `powerpc64le` | IBM POWER8 and newer, little-endian | POWER (ELFv2) |
| `s390x` | IBM Z (z10 and newer) | z/Architecture |
| `loongarch64` | Loongson LoongArch | LoongArch |

Without `--arch`, the stage is made for the machine you are on. The code is for ELF systems (Linux and
similar); macOS and Windows use other object formats, which are not written yet, so `--emit obj` and `so`
say so there, while `--emit asm --arch ...` works everywhere.

`obj` and `so` need a C compiler to assemble and link: `cc` for this machine, and for another architecture
`clang --target=...` (with `lld` for `so`) or that architecture's GCC (`aarch64-linux-gnu-gcc` and so on).
`--cc COMMAND` names one, for example `--cc "clang --target=riscv64-linux-gnu -fuse-ld=lld"`. If none is
found, QOA says which to install.

## What the native code does

The IR shows the plan. A few classical instructions become machine instructions: `SET`, `COPY`, `ADD`,
`SUB` and `MUL` on registers or numbers, and `JUMP`. Every other operation (gates, optics, measurements,
the rest of the classical instructions, input and output) is a call into QOA's machine, which runs it
exactly as the interpreter does and answers with the next operation. The native code goes straight on when
that is the next one in line, and otherwise looks up where to go in a table, so jumps, calls, returns and
loops all work.

```
; 8 operations in 5 blocks: 2 native (machine instructions), 6 through QOA's machine (step)
b0:  ; ops 0..1 -> b4
  0000  native   R0 = 5.0                  ; REGSET R0 5  (line 3)
  0001  step     1                         ; CALL addr_0x6  (line 4)
```

Add, subtract and multiply are plain IEEE 754 operations, so they give the same bits on all six
architectures as in the interpreter. Before each native instruction, the code tells the machine (`tick`), so
step counting, `--max-steps` and everything that depends on the position in the program are the same. The
result is that a native run gives exactly the interpreter's run: the same counts for the same seed, the same
registers, output, errors and line numbers. The tests check this for every example and test program, and
for QOA's own assembler written in QOA, which rebuilds its own bytecode byte for byte as native code; CI
runs them on all six architectures (five of them under qemu).

The library carries its program (the `.qx`, checked with its SHA-256 when it is loaded), so `qoa run
program.so` needs nothing else. Every `qoa run` option works with it: shots, seeds, noise, `--mpi`, GPUs.

## The calling convention

For anyone reading the assembly or calling it from elsewhere:

```c
// returns 0 when the program ends, 1 on an error (kept by the machine), 2 if code and program disagree
int64_t qoa_native_entry(void *machine, double regs[32],
                         int64_t (*step)(void *machine, uint64_t pc),   // next pc, -1 stop, -2 error
                         int64_t (*tick)(void *machine, uint64_t pc));  // 0, or -2 error
extern const uint8_t  qoa_native_qx[];      // the program's .qx
extern const uint64_t qoa_native_qx_len;
extern const uint32_t qoa_native_abi;       // 1
```

Every operation `N` has the label `.Lop_N`, and `.Ltab` is the table of their offsets. The code uses no
outside symbols, so the library is position-independent and loads anywhere.

## Safety

A `.so` is machine code: loading it runs it. Run only libraries you built, as you would any program. QOA
checks that a library was made by `qoa compile` for this version of the convention and that its program is
intact, but it cannot check machine code that was edited by hand. The Rust code that calls into the library
is the `native` module, one of the few allowed to use `unsafe` ([safety.md](safety.md)).
