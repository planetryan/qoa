# QOA performance

[Euskaraz: performance.eu.md](performance.eu.md)

What limits a state-vector simulator, what QOA does about it, how fast it is, and how to check every number
here on your own machine.

## The limit is memory, not arithmetic

An n-qubit state is 2^n complex numbers of 16 bytes: 24 qubits is 256 MiB, 27 qubits 2 GiB, 28 qubits
4 GiB. A one-qubit gate reads and writes all of it and does about 14 floating-point operations per
amplitude, roughly 0.4 operations per byte moved. Any modern core can do that arithmetic much faster than
memory can deliver the bytes, so once the state is bigger than the caches, **a gate costs one trip through
memory**, and the only ways to go faster are to make that trip at full speed and to make fewer trips.

`cargo bench --bench bandwidth` measures the machine's ceiling (scaling a vector in place, the same traffic
as a gate with no arithmetic) next to every gate kernel. On the 4-core Xeon (2.3 GHz, AVX-512, 2 MiB L2 per
core) used for all numbers below:

| state size | read + write, 4 threads | 1 thread |
|---|---|---|
| up to 128 MiB (23 qubits, in cache) | 167 - 187 GB/s | 43 - 73 GB/s |
| 512 MiB - 2 GiB (25 - 27 qubits, main memory) | 62 - 79 GB/s | 18 - 23 GB/s |

GB/s for a gate is 2 x state size / time (read and write), so it can be compared with this table directly.

## What QOA does

1. **Contiguous runs.** A gate on qubit q mixes amplitude pairs (i, i + 2^q). Those pairs come in contiguous
   runs, and controls select whole runs, so the kernels never decide element by element
   ([src/kernel.rs](../src/kernel.rs)).
2. **Hand-written SIMD assembly** for the three inner loops (2x2 matrix on paired runs, the same inside a
   vector register for qubits 0 and 1, and diagonal scaling), in [src/simd.rs](../src/simd.rs): AVX-512 and
   AVX2+FMA on x86-64, NEON on ARM, RVV 1.0 on RISC-V (vector-length agnostic). The best one is picked at run
   time; `QOA_SIMD=scalar|avx2|avx512|neon|rvv` forces one. Controls on qubits 0 and 1 become per-lane
   coefficients (identity where the control is off), so they cost nothing extra.
3. **Gate fusion.** Consecutive single-qubit gates on the same qubit become one matrix, even with gates on
   other qubits in between; products that come out as the identity are dropped.
4. **Cache blocking.** The VM queues gates instead of applying them one by one, until something needs the
   state (a measurement, an optical operation, a print). A run of gates is then applied in stages: gates on
   the low qubits are applied to 256 KiB blocks while each block sits in cache; gates that also touch up to
   four high qubits are applied to *gathered* blocks (16 KiB runs from the low qubits plus the high
   qubits). Either way a stage reads and writes memory once, however many gates it holds.
5. The report shows the most likely amplitudes with a parallel top-k selection instead of sorting the whole
   state (sorting 16 million amplitudes used to take longer than the simulation).

The report's `cpu` line says which kernels ran.

## Results

Single gates at 27 qubits (main memory, ceiling about 75 GB/s):

| gate | QOA 0.4.0 at commit 5750d4e | now |
|---|---|---|
| H on q0 | 19 GB/s | 66 GB/s |
| H on q1 | 28 GB/s | 67 GB/s |
| H on q3 | 38 GB/s | 61 GB/s |
| H on q13 | 58 GB/s | 81 GB/s |
| H on q26 | 46 GB/s | 74 - 80 GB/s |
| RX on q5 | 42 GB/s | 69 GB/s |
| RZ (diagonal) | 38 GB/s | 77 - 81 GB/s |
| CNOT q0 -> q13 | 85 GB/s | 79 - 84 GB/s |
| CZ q0, q26 (touches half the state) | 105 GB/s | 104 - 158 GB/s |
| SWAP q0, q26 | 56 GB/s | 61 - 68 GB/s |

Inside the cache, on one core (nanoseconds per amplitude for one gate, lower is better):

| kernels | ns per amplitude |
|---|---|
| AVX-512 | 0.35 - 0.45 |
| AVX2 | 0.55 - 0.60 |
| scalar Rust | 1.3 - 1.8 |

At 0.4 ns the AVX-512 kernel is close to this core's floating-point limit for a general complex 2x2
(8 vector operations and 2 shuffles per 8 amplitudes on 2 execution ports), so large stages are now limited
by arithmetic inside the cache and by bandwidth outside it.

Whole circuits:

| circuit | QOA 0.3.5 | QOA 0.4.0 (5750d4e) | now | Qiskit Aer 0.17 |
|---|---|---|---|---|
| 24 qubits, 5 layers of H, CNOT chain, RZ (355 gates) | 23.3 s | 4.1 s | 0.83 - 0.90 s | 3.7 s (5.6 s without fusion) |
| 27 qubits, one layer of 27 H gates | | 1.5 s (27 passes) | 0.68 s | |

All on the same machine; Aer through its Python API with `method='statevector'`, best of three runs after
a warm-up.

## Checking the numbers

```sh
cargo bench --bench bandwidth                  # ceiling and kernels at 20, 24, 27 qubits
cargo bench --bench bandwidth -- 23 28         # other sizes
QOA_SIMD=scalar cargo bench --bench bandwidth  # without the assembly
cargo bench --bench benchmark                  # criterion: kernels, optics, whole programs
```

Correctness of the fast paths:

- `tests/simd.rs` compares every assembly kernel available on the machine with the scalar version, over
  random run layouts and per-lane coefficients, and checks that nothing outside the runs is written.
- `tests/kernel.rs` compares batches (fusion, in-place and gathered blocks, with every block size, controls
  anywhere, swaps, states with optical modes) with a naive reference, and `tests/reference.rs` checks every
  gate against an independent simulator.
- `QOA_SIMD=avx2 cargo test` and `QOA_SIMD=scalar cargo test` run the whole suite on the other kernels.

ARM and RISC-V run under qemu (on Debian or Ubuntu: `qemu-user`, `gcc-aarch64-linux-gnu`,
`gcc-riscv64-linux-gnu`, and `rustup target add aarch64-unknown-linux-gnu riscv64gc-unknown-linux-gnu`):

```sh
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu"
cargo test --release --target aarch64-unknown-linux-gnu --test simd --test kernel --test reference

export CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc
export CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_RUNNER="qemu-riscv64 -cpu rv64,v=true,vlen=256 -L /usr/riscv64-linux-gnu"
cargo test --release --target riscv64gc-unknown-linux-gnu --test simd --test kernel --test reference
```

These pass for NEON, and for RVV with vector lengths 128, 256, 512 and 1024 bits; a RISC-V CPU without the V
extension falls back to the scalar kernels. qemu checks correctness, not speed: QOA has not been timed on
real ARM or RISC-V hardware.

## What is not done yet

- Gates whose matrix is real (H, RY) could skip half the arithmetic inside the cache.
- Optical-mode operations (beamsplitters, Kerr, emitters) still use plain Rust loops; programs with large
  photon spaces are slower than qubit circuits of the same size.
- A GPU backend (Vulkan) is being investigated separately.
