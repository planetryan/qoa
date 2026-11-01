# Safety and robustness

[Euskaraz: safety.eu.md](safety.eu.md)

QOA is meant to be trusted with anything you give it: a program you wrote, a `.qx` file from somewhere else,
a damaged file, a typo, or a run far bigger than the machine. This page says what QOA promises, where its
limits are, how that is tested, and which small parts of the code still need Rust's `unsafe` and why.

## The promise

Whatever the input, QOA does one of two things: it runs the program, or it stops with a clear message that
says what is wrong, in English or in Basque. It does not crash, panic, hang without a limit, or fill the
memory until the system kills it. This covers:

- **Sources**: `.qoa`, `.qg` (glyph syntax) and `.qasm` (OpenQASM), including invalid UTF-8, random bytes,
  absurd numbers and nesting thousands of levels deep.
- **Compiled files** (`.qx`): a loaded program must be exactly what the assembler builds from its own text,
  checked operation by operation. A damaged or hand-edited file that names a register, qubit, mode, bit or
  jump target that does not exist, breaks a loop, or claims an absurd size is refused as corrupt, so the
  machine never sees it.
- **Running programs**: every resource a program can ask for is checked before it is used (see the limits
  below), and numbers that are not finite stop the program instead of quietly turning the state into NaN.
- **Output**: if the output cannot be written (a full disk), QOA stops and says so instead of losing it
  silently; if the reader goes away (`qoa ... | head`), QOA ends quietly like any command-line tool.

## What a program can reach

A QOA program runs in a sandbox. It can compute, read its standard input and the arguments given after `--`,
read the clock (`TIME`), and write to standard output and standard error. It cannot open files, use the
network, start other programs, read environment variables, or load code. So the ways a hostile program or
file could reach the machine are the ones QOA itself opens, and each is closed:

- **The terminal.** Terminals act on control sequences: a program could set the window title, write to the
  clipboard (OSC 52), move the cursor to fake output, or make the terminal answer a status request as if
  you had typed it. When QOA's output goes to a terminal, every control sequence that comes from a program
  or a file (program output, source lines in error messages, names) is shown as text (`^[]0;...`) instead of
  acting; only line breaks, tabs and colours pass. Output to a pipe or a file is left byte for byte as the
  program wrote it, so QOA programs still work as tools. A `.qx` whose program name contains control
  characters is refused as corrupt.
- **Compiled files.** A `.qx` is checked operation by operation before the machine sees it (see above).
- **Libraries.** QOA loads GPU drivers (`--gpu`) and MPI (`--mpi`) only when asked, and only from system
  locations: never from the current folder, where Windows and macOS would otherwise look and where a
  downloaded library could hide. `QOA_MPI_LIB` loads exactly the library it names; do not let others set it.
- **ffmpeg** (`qoa visual`). File names are passed to ffmpeg as plain files (`file:`), never as options or
  protocols, whatever their names (`-y.wav`, `concat:...`). Temporary files are created fresh, readable only
  by you, and never through an existing file or link.
- **Big or endless inputs.** Program files are read up to a size tied to the memory QOA may use (so
  `qoa run /dev/zero` stops at once), an `INPUT` line is at most 1 MiB, and recorded input counts toward the
  memory limit.

What stays outside QOA: ffmpeg's own handling of the media files you give it, GPU drivers, and the MPI
library.

## Limits

| what | limit | past it |
|---|---|---|
| memory for the state, optical operators, memory cells and results together | what QOA may use: 90% of the memory available when it starts, or `QOA_MAX_MEMORY` | a message with the size needed and the limit, before anything is allocated |
| qubits | 64 (and the state must fit in memory) | assembler error |
| optical modes | indices 0 to 63 (and the state must fit) | assembler error |
| classical bits | c0 to c4095 | assembler error |
| named registers (`QREG`, `CREG`, `OREG`, `RREG`, glyph `{n}`) | 1 to 4096 elements, first index 0 to 4095 | assembler error |
| expression nesting (parentheses, functions, signs, `^` chains) | 100 levels | assembler error, in both assemblers |
| steps of one shot | 100,000,000 (`--max-steps` changes it) | runtime error naming the limit |
| `PUSH` stack | 1,048,576 values | runtime error |
| nested `CALL`s | 100,000 | runtime error |
| memory cells (`LOAD`, `STORE`) | addresses 0 to 2^32 - 1, and the cells must fit in memory | runtime error |
| different results of a run | as many as fit in memory (about 1 KiB each) | runtime error, before the memory is full |
| light in a mode | must stay below the photon cutoff (`QMODE`) | runtime error when no probability is left below it |

Results are kept per different outcome, never per shot: a run of 100 million shots with four possible
outcomes needs a few kilobytes for its counts.

## How it is tested

- `tests/hostile.rs` runs on every build: 20,000 damaged `.qx` files (bit flips, cut files, huge lengths and
  indices, NaN and infinity in place of numbers), 10,000 garbled `.qoa`, `.qg` and `.qasm` programs, and
  programs nested 200,000 levels deep. Each one goes through loading, assembling, running (with small limits)
  and every export (QOA text, listing, json, dot, glyph, OpenQASM, IonQ). Any panic, abort or hang fails the
  test. Searching with other seeds (`QOA_HOSTILE_SEED`) found and fixed the problems listed in the 0.4.4
  release notes; seventeen seeds, about half a million inputs, now pass.
- Every program in `tests/programs/failures` must fail with its expected message, in both assemblers.
- The rest of the suite checks that what does run is right: every gate against an independent simulator,
  every vector kernel against a plain scalar version, the assembler written in QOA against the Rust one byte
  for byte, and 1, 2, 3, 4 and 5 MPI processes against one.

## Where `unsafe` is, and why

Rust's `unsafe` marks code whose memory safety the compiler cannot check by itself. It is forbidden in the
whole library (`#![deny(unsafe_code)]`) except in the modules below, each of which says why at its top, and
it is forbidden outright in the `qoa` command-line tool.

| module | why |
|---|---|
| `simd` | calling a vector kernel built for a processor feature (AVX-512, AVX2, NEON) after checking that the processor has it, and loading or storing one vector through a reference that covers exactly its memory; RISC-V vector kernels are in assembly, because stable Rust has no RVV support yet |
| `kernel`, `state` | worker threads each copy their own disjoint blocks of the state in and out through a shared pointer; every block's bounds are checked in release builds too, and test builds also check that no two threads ever touch the same block |
| `dd`, `quad` | calling sums and kernels built with FMA after checking that the processor has it |
| `mpi`, `opencl`, `vulkan`, `syslib` | loading and calling the MPI library and the GPU drivers, which are C libraries loaded at run time |

The x86 and ARM vector kernels were hand-written assembly until 0.4.4; they are now ordinary Rust on
bounds-checked slices, with bit-for-bit the same results and the same speed.

## Releases and dependencies

- **Verifiable downloads.** Every release file has a SHA-256 checksum (`SHA256SUMS`) and a signed build
  provenance: anyone can check that a file was built by this repository's release workflow from a given
  commit, with `gh attestation verify FILE --repo planetryan/qoa`.
- **Hardened binaries.** Linux binaries are position-independent with full RELRO and a non-executable stack,
  and the release workflow checks this for every Linux build; Windows binaries are built with Control Flow
  Guard.
- **Pinned build steps.** Every GitHub Action in CI and in the release workflow is pinned to an exact
  commit, so a moved or hijacked tag cannot change what builds QOA. Cargo builds use `--locked`.
- **Dependencies.** CI checks every dependency's exact version against the RustSec advisory database.
- **Continuous checks.** Besides the hostile-input test on every build, a nightly job runs it with 64 more
  seeds, and CI runs the whole suite with integer-overflow checks and debug assertions.

## What it does not promise

- A simulator answers questions about the model it simulates: a photon cutoff that is too small, or a noise
  model that does not describe the device, gives correct answers to the wrong question. QOA reports the
  truncation leakage and says when light leaves the cutoff entirely, but choosing the model is up to you.
- GPU drivers and MPI libraries are outside QOA. QOA checks what they report and compares GPU results with
  the CPU in its tests, but a faulty driver can still fail in its own way.
- Timing: a valid program can legitimately run for a very long time (a huge loop, an enormous state). The
  step limit bounds each shot, and the memory checks bound the size, but QOA does not guess how long you are
  willing to wait.
