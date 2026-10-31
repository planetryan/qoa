# QOA on a GPU (Vulkan and OpenCL)

[Euskaraz: gpu.eu.md](gpu.eu.md)

**Short answer:** `qoa run --gpu` applies qubit gates on any GPU that has 64-bit floats, through Vulkan or
OpenCL. It is verified correct here on lavapipe (Mesa's software Vulkan) and on POCL (a CPU OpenCL). It has **not** been timed on a real GPU,
because the machine it was built on has none. Below: how it works, what it should gain on real
cards (worked out from published specifications, not measured), and how to measure it on yours.

## Using it

```sh
qoa run examples/grover-key.qoa --gpu
```

The report says which device ran the gates and what it did. This is a 22-qubit GHZ state on lavapipe:

```text
cpu        avx512 kernels, 4 thread(s)
gpu        llvmpipe (LLVM 20.1.2, 256 bits) (software Vulkan, runs on the CPU)
gpu work   1 batch(es) of qubit gates: 22 gates in 4 pass(es) over the state, 0.152 s computing; the state went up and back 1 time(s) (64.0 MiB each way, 0.132 s). measurements, optics and noise ran on the cpu
```

`--json` carries the same numbers under `"gpu"`, or `null` without `--gpu`. If no usable device is found,
`--gpu` stops with the reason instead of quietly running on the CPU:

```text
error: --gpu: no usable Vulkan device: Apple M2: no 64-bit floats in shaders (shaderFloat64); QOA does not compute in 32 bits
error: --gpu: the Vulkan loader found no driver for Vulkan 1.2 (ERROR_INCOMPATIBLE_DRIVER); install your GPU's Vulkan driver
error: --gpu: this QOA was built without GPU support (cargo feature `vulkan`)
```

The device needs Vulkan 1.2, `shaderFloat64` and `bufferDeviceAddress`. Desktop NVIDIA and AMD cards have
all three. Apple GPUs (through MoltenVK) and most phone GPUs have no 64-bit floats. Intel Arc has no 64-bit
hardware either, so where its driver offers 64-bit floats they are emulated and slow. With several
devices, a discrete GPU is preferred over an integrated one, and either over a software one.
Vulkan is loaded when the program runs, so building QOA needs no Vulkan SDK; `--no-default-features`
builds it without GPU support.

## OpenCL: AMD Instinct, NVIDIA, Intel

Some of the largest fp64 accelerators offer no Vulkan at all, among them AMD Instinct (the MI250X in
Frontier, the MI300A in El Capitan). For those, and for any device with an OpenCL driver, QOA runs the same
kernel written in OpenCL C:

```sh
qoa run circuit.qoa --gpu-api opencl     # or vulkan; --gpu alone tries Vulkan first, then OpenCL
```

- The device needs 64-bit floats (`cl_khr_fp64`) and 16 KiB of local memory. GPUs are preferred over
  accelerators, and either over a CPU implementation.
- `libOpenCL` is loaded when the program runs, so building needs no OpenCL SDK. On AMD it comes with ROCm,
  on NVIDIA with the driver, and POCL provides one for CPUs.
- The state goes to the device in up to 16 buffers, because OpenCL limits the size of each one.
- `tests/gpu.rs` runs the OpenCL device against the CPU on 80 random batches of up to 18 qubits. Every
  amplitude must match within 1e-10. CI runs it on POCL.

## Several GPUs

The way to use every GPU in a node, or in many nodes, is one MPI process per GPU:

```sh
mpirun -n 8 qoa run big.qoa --mpi --distribute --gpu       # 8 GPUs share one state
mpirun -n 8 qoa run noisy.qoa --mpi --gpu --shots 100000   # 8 GPUs share the shots
```

Each process takes the GPU numbered by its rank within the node: Open MPI, MPICH, Slurm, MVAPICH and
Cray launchers say which that is, and `QOA_GPU_DEVICE` overrides it. `QOA_GPU_DEVICE=cpu` keeps a
process on the CPU even with `--gpu`, so one job can use GPUs and CPU cores together through your
launcher's per-process settings (with Open MPI: `mpirun -n 4 -x QOA_GPU_DEVICE=0 qoa ... : -n 4 -x
QOA_GPU_DEVICE=cpu qoa ...`). With `--distribute`, every process
applies the gates on its own part of the state on its own GPU, and the processes exchange data only for
gates on the qubits that span them ([hpc.md](hpc.md)).

## How it works: the same squeeze

The GPU has the same problem as the CPU: a gate reads and writes the whole state and does very little
arithmetic per byte ([performance.md](performance.md)). The answer is the same too: apply many gates per
pass over memory.

- **Tiles in shared memory.** A work group of 256 threads loads a tile of 2^10 amplitudes (16 KiB, the
  size Vulkan guarantees on every device) into on-chip shared memory. It applies every gate of the stage
  there and writes the tile back, so a stage costs one pass over memory however many gates it holds. This
  is the GPU form of the CPU's cache blocks ([src/kernel.rs](../src/kernel.rs)).
- **Free choice of tile qubits.** A tile is any 10 qubits. The 5 lowest are always in it, so each read is
  512 contiguous bytes. The other 5 are the stage's target qubits, and any spare places go to the next-lowest
  qubits. The planner ([src/gpu.rs](../src/gpu.rs)) adds gates to a stage until a new target no longer
  fits. Controls do not need to be in the tile: controls outside it are checked once per tile against
  the tile's position.
- **Cheaper gate kinds.** Diagonal gates (phases, RZ, CZ) touch only each thread's own amplitudes, so
  consecutive diagonal gates need no barrier between them. An X (CNOT, Toffoli) swaps a pair with no
  arithmetic. A real matrix (H, RY) needs less than half the arithmetic of a complex one. Swaps exchange
  pairs. This matters more on a GPU than on the CPU (see below).
- **Exact arithmetic.** Everything is 64-bit floating point, the same as the CPU path. QOA never drops
  to 32 bits to go faster.
- **Any state size the card can hold.** The state is split into pieces of up to 1 GiB, each with its own
  buffer and device address, because many drivers limit a single buffer or allocation to 2-4 GiB. The
  kernel addresses up to 64 pieces and 2^31 amplitudes.
- **What stays on the CPU:** measurement, optical operations and noise. Each batch of gates sends the state
  to the card, applies the batch and brings the state back. A typical program (gates, then measurements)
  makes one round trip. A program that measures in the middle makes one round trip per stretch of gates,
  and so does every shot when shots are simulated one by one.

The kernel is one GLSL compute shader, [src/shaders/tile.comp](../src/shaders/tile.comp), compiled to
SPIR-V by `tools/shaders.sh`. The compiled `tile.spv` is checked in, so building QOA needs no shader
compiler, and a test fails if the source changed after the last compile. On the CPU the inner loops are
hand-written assembly, because there the choice of instructions is ours. On a GPU it is not. SPIR-V is an
intermediate form: the driver compiles it to the card's own instruction set (NVIDIA SASS, AMD RDNA) and
does the register allocation and scheduling itself, and Vulkan offers no way to give it machine code.
Hand-written SPIR-V would just be a harder-to-read version of the same input.

## Different juice: what limits a GPU

On the CPU, memory is the wall. On a GPU, 64-bit arithmetic can be the wall instead, because consumer
cards run it at 1/32 to 1/64 of their 32-bit speed. Each tile applies many gates per pass, so the
arithmetic per byte moved goes up, and on a gaming card it can overtake memory.

The example below is the 24-qubit, 355-gate circuit from [performance.md](performance.md) (5 layers of
H, a CNOT chain and RZ). Fusion turns its 355 gates into 259, and the planner groups them into **42
passes**, which is 42 x 512 MiB = 22.5 GB of memory traffic. Counting from the plan itself, it needs about
1,620 floating-point operations per amplitude:

- each RZ fused with the next layer's H is one complex matrix, 14 per amplitude;
- the CNOTs are swaps, 0;
- the first H layer is real, 6;
- the last RZ layer is diagonal, 6.

That is **27 GFLOP** in total. A card can be no faster than the slower of the two limits:

| card | 64-bit peak | memory | arithmetic time | memory time | at best |
|---|---|---|---|---|---|
| NVIDIA RTX 3060 | about 0.2 TFLOPS | 360 GB/s | 136 ms | 62 ms | ~140 ms, arithmetic-bound |
| NVIDIA RTX 4090 | about 1.3 TFLOPS | 1008 GB/s | 21 ms | 22 ms | ~22 ms, both limits at once |
| AMD RX 7900 XTX | about 1.9 TFLOPS | 960 GB/s | 14 ms | 23 ms | ~23 ms |
| NVIDIA A100 | 9.7 TFLOPS | 1.6 - 2.0 TB/s | 2.8 ms | 11 - 14 ms | ~12 ms, memory-bound |
| NVIDIA H100 SXM | 34 TFLOPS | 3.35 TB/s | 0.8 ms | 6.7 ms | ~7 ms, memory-bound |

The peaks are the manufacturers' published figures. **These are lower bounds, not measurements.**
Real kernels reach some fraction of peak, and the table ignores shared-memory bandwidth and barriers. The
same circuit takes 0.6 - 0.9 s on the 4-core CPU used throughout these docs, so a real GPU should be
roughly 5 to 100 times faster at the gates. Moving the state costs on top of that: 256 MiB each way over
PCIe 4.0 is about 20 - 40 ms per round trip, which is as much as the gates themselves on a fast card. Two
things follow:

- Without the cheaper gate kinds, the same circuit would need about 2,630 operations per amplitude (X and
  H as full complex matrices), and gaming cards would be arithmetic-bound at about 1.6 times the time.
  Removing arithmetic is worth as much on a consumer GPU as removing memory passes is on the CPU.
- On data-centre cards (A100, H100), memory is the limit again. There, bigger tiles would cut passes.
  Vulkan drivers typically allow 48 KiB of shared memory per work group on NVIDIA and 64 KiB on AMD, room
  for 2^11 or 2^12 amplitudes instead of 2^10.

## Measured here (lavapipe, a software GPU)

`cargo bench --bench gpu -- 20 24`, on the same 4-core Xeon. The "device" is the CPU running
LLVM-compiled shader code, so this checks the plumbing and the pass counts, **not GPU speed**. The last
column is QOA's own CPU path with the same gates:

| qubits | gates | passes | lavapipe computing | moving the state | QOA on the CPU |
|---|---|---|---|---|---|
| 20 | H on qubit 0 | 1 | 6.4 ms | 8.2 ms | 0.7 ms |
| 20 | H layer (20) | 3 | 33 ms | 7.1 ms | 4.7 ms |
| 20 | circuit (295) | 33 | 308 ms | 9.1 ms | 32 ms |
| 24 | H on qubit 0 | 1 | 56 ms | 171 ms | 9.7 ms |
| 24 | H layer (24) | 4 | 547 ms | 183 ms | 76 ms |
| 24 | circuit (355) | 42 | 5.6 s | 186 ms | 0.62 s |

Software Vulkan is about 9 times slower than QOA's hand-written AVX-512 kernels on the same cores. That is
expected, and it is the reason `--gpu` exists only to reach a real card. Never use it with lavapipe for
speed.

## How it was verified

- **The plan, without a GPU** (`tests/gpu.rs`). `gpu::emulate` repeats the shader's tile-by-tile index
  arithmetic in Rust. It is checked against a naive reference for 300 random gate batches: 1 to 14 qubits,
  with and without an optical mode (state lengths that are not powers of two), random controls inside and
  outside the tile, swaps, X, diagonal, real and complex gates.
- **The barriers, without a GPU.** `gpu::races` follows which thread touches which shared-memory element
  between two barriers, and fails if two threads touch the same element. Every stage in every random batch
  is checked. Removing the barriers is caught by this check and by nothing else, because lavapipe runs the
  256 threads of a work group in lockstep, where a missing barrier changes nothing.
- **The device against the CPU.** 120 random batches (1 to 16 qubits, with modes) and a 21-qubit state with
  200 gates, run twice to reuse the buffers. Every amplitude must match the CPU within 1e-10.
  Deliberately wrong shaders were checked to fail this: one with a swapped matrix element in the complex
  path, another in the real path.
- **Whole programs.** Every example and the stress programs give identical counts and probabilities with and
  without `--gpu`. A CLI test checks the report, the JSON and the error when no driver is present.

On a machine with a real GPU, run `QOA_REQUIRE_GPU=1 cargo test --release --test gpu`; without
`QOA_REQUIRE_GPU=1`, the device tests say why they were skipped. Then run `cargo bench --bench gpu`. The
first run on real hardware is what turns the table above from a projection into a measurement.

## What is not done yet

- **Keep the state on the card** across measurements: probabilities, collapse and sampling as GPU kernels,
  so mid-circuit measurement does not send the whole state back to the CPU.
- **Optical operations** (beamsplitters, displacement, squeezing, Kerr) on the GPU. They use the same
  slab structure as on the CPU, but are not written yet.
- **Bigger tiles** on cards with more shared memory, and overlapped transfers (two staging buffers).
- **Timing on real GPUs** (NVIDIA, AMD), which is the next measurement to make.
