![qoa-logo](https://github.com/user-attachments/assets/5fdbae92-68f8-490e-8368-f3fd6c81a064)

# QOA: Quantum Optical Assembly

QOA is an assembly language and a simulator for **qubits and light together**: qubits, optical modes
(photons), and the couplings between them, plus classical registers and control flow, in one program.

It runs on a classical CPU. Every result is computed exactly by simulating the quantum state, not measured
on quantum hardware. That makes it a tool for learning, prototyping and checking quantum-optical programs,
not a quantum computer.

[![ci](https://github.com/planetryan/qoa/actions/workflows/ci.yml/badge.svg)](https://github.com/planetryan/qoa/actions/workflows/ci.yml)

[Euskaraz: README.eu.md](README.eu.md) · [Instruction reference](docs/ISA.md) · [Release notes](<changelog/QOA v0.4.1.md>)

---

## What it does

- **Qubit circuits**: the standard gate set (H, X, CNOT, CZ, Toffoli, rotations, controlled gates, ...),
  measurement in any basis, reset, and mid-circuit measurement with classical feedback.
- **Optical modes**: photon-number (Fock) states with displacement, squeezing, beamsplitters, phase shifts,
  Kerr and cross-Kerr nonlinearities, photon addition and subtraction, loss, and number-resolving,
  threshold and parity detection.
- **Qubit-photon coupling**: an emitter qubit can put its excitation into a mode as a photon (`EMIT`), and a
  dispersive coupling (`DISPERSIVE`) lets a qubit control a mode's phase. This is the cavity-QED and
  quantum-network toolbox: heralded entanglement, Schrodinger cat states, qubit-photon entanglement.
- **Classical control**: 32 registers, arithmetic, loops, subroutines, branches on measurement results.
- **OpenQASM 2.0**: `qoa run circuit.qasm` runs OpenQASM directly (the whole language, including gate
  definitions and `if`), and `qoa export --to qasm` writes the qubit part of a program as OpenQASM 2.0.
  `--to ionq` writes IonQ's JSON format.
- **Two languages**: every message, error and help text is in English and Basque. `QOA_LANG=eu` or
  `QOA_LANG=en` chooses; otherwise the system locale decides.

What it does **not** do: run on quantum hardware by itself, simulate more than about 30 qubits (the state
vector doubles with every qubit; 28 qubits is 4 GiB), or model continuous-time physics (pulses, decoherence
rates). Noise is either a simple, documented model (`--noise`, `--loss`) or a real device's calibrated noise:
gate errors, T1/T2 relaxation and readout errors (`--noise-model`, [docs/noise.md](docs/noise.md)).

Where it is useful, where other tools are better, and what stands in the way of adopting it:
[docs/use-cases.md](docs/use-cases.md).

## Quick start

**Without building:** download a release from
[GitHub Releases](https://github.com/planetryan/qoa/releases). There is one archive per system, and
`qoa-<version>-all.zip` holds every system at once:

| system | processors | run |
|---|---|---|
| Windows 10 / 11 | x64, ARM64 | `qoa.exe run examples\bell.qoa`, or `run.cmd ...` / `.\run.ps1 ...` |
| Linux (glibc 2.35 or newer) | x86-64, ARM64, RISC-V 64, IBM POWER8 or newer (ppc64le), IBM Z (s390x), LoongArch 64 (glibc 2.36+) | `./qoa run examples/bell.qoa`, or `./run.sh ...` |
| Any Linux, static (musl), Android (Termux) | x86-64, ARM64, RISC-V 64 | `./qoa run examples/bell.qoa`, or `./run.sh ...` |

All builds are 64-bit: QOA computes in 64-bit floating point and needs a 64-bit address space for large
states.
| macOS | Intel, Apple Silicon | `./qoa run examples/bell.qoa`, or `./run.sh ...` |

The `run` scripts pick the right binary for the machine they run on. In a source checkout they build QOA
first. `SHA256SUMS` lists every file's checksum. The binaries are not signed: macOS may refuse to open a
downloaded one until you run `xattr -d com.apple.quarantine qoa`, and PowerShell may block `run.ps1`
unless you use `run.cmd` or `powershell -ExecutionPolicy Bypass -File run.ps1 ...`.

**From source**, with Rust 1.89 or newer (stable):

```sh
cargo build --release                              # or just ./run.sh (Windows: run.cmd), which builds and runs
./target/release/qoa run examples/bell.qoa
./target/release/qoa run examples/hom.qoa          # two photons on a beamsplitter
./target/release/qoa run examples/shor15.qoa       # Shor's algorithm factoring 15, start to finish
./target/release/qoa isa                           # every instruction
```

Output says plainly what ran and how:

```
program    examples/bell.qoa (4 operations)
system     2 qubit(s) = 4 amplitudes, 64 B
cpu        avx512 kernels, 4 thread(s)
noise      none: ideal gates, no photon loss
shots      1024 shot(s), sampled from one simulation (every measurement is at the end)
seed       7

outcome    c1..c0    count      share
           11          515      50.3%  ███████████████
           00          509      49.7%  ██████████████▉

time       0.5 ms
```

`--json` prints the same result as a record other programs can read.

## The language in one page

```
; comments start with ; (or //)
QINIT 3                     ; 3 qubits: q0, q1, q2
QMODE 2 6                   ; 2 optical modes, each holding 0..5 photons

H q0                        ; gates take typed operands: q3 is a qubit, o1 a mode, R2 a register, c0 a bit
CNOT q0 q1
RZ q2 pi/4                  ; angles are expressions: pi/4, -3*pi/2, sqrt(2)/2
H q0..q2                    ; a range applies a one-qubit instruction to each qubit

SINGLEPHOTON o0
BS o0 o1 pi/4               ; 50:50 beamsplitter
EMIT q1 o1                  ; qubit excitation -> photon

MEASURE q0 -> c0            ; results go where you route them: a bit (c0) or a register (R0)
COUNT o0 -> R1
JUMPIFONE c0 done           ; branch on a measured bit
X q2
done:
LOOPSTART 3                 ; loops, subroutines (CALL/RETSUB), arithmetic (ADD, MUL, ...)
  PRINTF "photons in o0: {}\n" R1
LOOPEND
```

Registers can be named if you like (`QREG data 2`, `CREG flags 2`, `OREG arm 2`, then `data[1]`, or `data`
for all of them); `q0`, `c0` and `o0` keep working. Mnemonics are case-insensitive, and the long names from
QOA 0.3 (`APPLYHADAMARD`, `PHOTONEMIT`, ...) still work as aliases. `qoa isa NAME` explains any instruction; [docs/ISA.md](docs/ISA.md) lists them all.
Errors point at the exact spot:

```
error: CNOT uses qubit 0 twice; the qubits must be different
  --> bell.qoa:2:9
   |
 2 | CNOT q0 q0
   |         ^^
```

### Glyph syntax (experimental)

The same language can also be written densely, many statements per line, in `.qg` files:

```
Q{2}; #C{2}; H/Q[0]; CX/Q[0],Q[1]; M/Q$>#C; PF"bell: {}\n"%#C;
```

`qoa import file.qg` shows the plain QOA it stands for, `qoa export --to glyph file.qoa` goes the other
way, and errors point at the glyph. `CC[0x10]%a%b$>#2` calls a classical routine at an address, with its
arguments on the stack. It is experimental and may still change: see [docs/glyph.md](docs/glyph.md).

### Looking inside a program

`qoa export --to <target>` writes a program in other forms, from `.qoa`, `.qg`, `.qasm` or a compiled `.qx`:

| target | what you get |
|---|---|
| `qoa` | canonical plain QOA: aliases resolved, ranges expanded, jump targets as `addr_0x..` labels |
| `listing` | the same, with every operation's address and source line; it still assembles |
| `json` | the program as data: resources, and each operation's address, line, text and structure |
| `dot` | the control-flow graph (basic blocks, jumps, loops, calls) for Graphviz: `qoa export --to dot f.qoa \| dot -Tsvg > f.svg` |
| `glyph` | glyph syntax (`--width`, `--names`, `--long-names`, `--no-ranges`, `--hex`) |
| `qasm`, `ionq` | for real hardware: [docs/hardware.md](docs/hardware.md) |

Every text export is reassembled and compared with the original before it is written, so a wrong export
is an error, never a silently different program.

### How shots are computed

- If every measurement comes at the end, QOA simulates once and samples all shots from that state.
- If a program measures mid-circuit, branches on results, or uses `--noise`/`--loss`/`--noise-model`, every
  shot is simulated from the start.
- If it measures nothing, QOA shows the exact final state (something only a simulator can do).

The report always says which of these happened.

### Photons and the cutoff

A mode stores photon numbers 0 to cutoff-1 (`QMODE modes cutoff`, default 8). Operations that create photons
can push probability past the cutoff. QOA measures that loss and reports it instead of hiding it; when it is
not negligible, the run warns and suggests a larger cutoff.

### Numbers and precision

The qubit state is kept in 64-bit floating point, and every sum over the state (norms, measurement
probabilities, sampling) in double-double, about 106 bits. The report's `numerics` line shows how far the
norm drifted. For more, `qoa run --precision quad` keeps qubit programs in 128-bit precision (double-double,
about 32 significant digits): amplitudes, gate matrices and measurement. It is 10 to 20 times slower. Details
in [docs/precision.md](docs/precision.md).

### Music through light

`qoa visual song.mp3 out.mp4` sends audio through a simulated quantum optical circuit: every sample becomes a
mode of light carrying a coherent state, the light passes through loss, squeezing, a Kerr medium or a
beamsplitter mesh, and is detected with its real quantum noise. The result comes out as audio, video or both
(`--audio-only`, `--video-only`, `--keep-audio`); the input is anything ffmpeg reads, or raw samples from
standard input. The video shows the light's quantum states as they happen: the exact Wigner function of the
brightest modes in every frame, at any resolution up to 4K60 and beyond. The light is computed exactly:
[docs/visual.md](docs/visual.md).

## Examples

All in [examples/](examples), each with comments explaining the physics, and each checked by the test
suite against what its comments claim:

| Program | Shows |
|---|---|
| `bell`, `ghz` | entanglement |
| `teleport` | teleportation with real classical feed-forward |
| `grover`, `grover-key` | Grover search; a toy key search (and why AES is out of reach) |
| `qpe`, `shor15` | phase estimation; Shor's algorithm factoring 15, classical part included |
| `hom`, `mach-zehnder` | two-photon interference; single-photon interferometer |
| `coherent`, `squeezed` | laser light (Poisson statistics); squeezed vacuum (only even photon numbers) |
| `emitter`, `cat` | qubit-photon entanglement; a Schrodinger cat state made with a qubit |
| `heralded-entanglement` | entangling two distant emitters by interfering their photons |
| `retry`, `sweep` | repeat-until-success; a classical loop around a quantum measurement |
| `bell.qasm` | OpenQASM run directly |
| `bell-torino.qasm` | Bell transpiled for IBM Torino; run it with that device's noise: `--noise-model examples/bell-torino.json` |
| `*.qg` | the glyph syntax: the original one-line sketch, Bell, teleportation, HOM with emitters, retry, and a repetition-code decoder that calls a classical routine (`CC`) |
| `kerr.rs` | light around a spinning black hole, with its Hawking pairs computed by QOA: `cargo run --release --example kerr -- --still kerr.png` ([docs/kerr.md](docs/kerr.md)) |

## Real hardware

The qubit part of a program exports to OpenQASM 2.0 or IonQ's JSON, including counted loops and
feed-forward on measured bits, and runs the same in Qiskit. Transpiled for IBM's Heron and Eagle devices
with their measured noise, small circuits keep 75-98% of the right answers and an 8-qubit Grover key search
drops to 30%. The linear-optics part matches Quandela's Perceval to 1e-15. With a free IBM Quantum account
(10 minutes of real-device time per 28 days), `tools/run_on_ibm.py examples/bell.qoa` runs a program on a
real quantum computer and compares the result with QOA's. Next to it, QOA predicts what the device will
return from the device's own calibration ([docs/noise.md](docs/noise.md)). Measurements, limits and details:
[docs/hardware.md](docs/hardware.md).

## Testing

```sh
cargo test --release                       # everything below except the limits
cargo test --release -- --ignored          # also the limits: 28 qubits (4 GiB), huge photon systems
```

- `tests/reference.rs` checks every gate against an independent, deliberately simple simulator.
- `tests/programs/` holds `.qoa` programs that state their expected results in header comments:
  sanity checks, intentional failures (every error message), stress tests, limits, and the old 0.3 programs.
- `tests/stress.rs` runs deep random circuits forward and backward, and checks boson sampling against
  matrix permanents.
- `tests/qasm.rs` compares every native OpenQASM gate with its official definition.

## Performance

A large simulation is limited by memory bandwidth, not arithmetic: every gate reads and writes the whole
state vector (28 qubits is 4 GiB). QOA does three things about it:

- **Hand-written SIMD assembly** for the inner loops: AVX-512 and AVX2 on x86-64, NEON on ARM, RVV 1.0 on
  RISC-V, picked at run time (`QOA_SIMD=scalar` turns it off). Every version is checked against a plain
  scalar one, on ARM and RISC-V under qemu too.
- **Gate fusion**: consecutive gates on the same qubit become one matrix, even with gates on other qubits in
  between.
- **Cache blocking**: a run of gates on the low qubits is applied block by block while each block sits in
  cache, so the run costs one pass over memory instead of one per gate.

On a 4-core Xeon (2.3 GHz), single gates now run at the memory's limit (about 75 GB/s at 27 qubits), and a
24-qubit, 355-gate circuit takes 0.9 s; Qiskit Aer takes 3.7 s on the same machine (5.6 s without its gate
fusion), and QOA 0.3.5 took 23 s. Details, measurements and how to reproduce them:
[docs/performance.md](docs/performance.md).

`qoa run --gpu` applies the qubit gates on a Vulkan GPU instead, in 64-bit floats, with the same idea
(many gates per pass over memory, in shared-memory tiles). It is verified against the CPU on a software
Vulkan driver, but has not been timed on a real GPU yet; what it should gain, and how to measure it:
[docs/gpu.md](docs/gpu.md).

## History

QOA 0.4 is a rewrite of the core. Version 0.3.5 is preserved on the
[`archive/v0.3.5-experimental`](https://github.com/planetryan/qoa/tree/archive/v0.3.5-experimental) branch.
See the [0.4.0 release notes](<changelog/QOA v0.4.0.md>) for what changed and why, and the
[0.4.1 release notes](<changelog/QOA v0.4.1.md>) for what came after.

## Contributing

How QOA is built, checked and written: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GPL-3.0, see [LICENSE](LICENSE).

*Ryan (@planetryan)*
