# QOA on clusters and supercomputers

*[Euskaraz](hpc.eu.md)*

QOA runs on a laptop, and it is meant to scale up to big machines. This page says what works across many
processes today, how to build and start it, and what each part costs.

## Shots over many processes (MPI)

A run with many shots can be spread over MPI processes, on one node or many. Every process simulates the
program, takes its share of the shots, and process 0 collects the counts and prints the result.

```sh
cargo build --release --features mpi          # needs an MPI library and its compiler wrapper (mpicc)
mpirun -n 64 ./target/release/qoa run circuit.qoa --mpi --shots 1000000
srun -n 64 ./target/release/qoa run circuit.qoa --mpi --shots 1000000      # under Slurm
```

- **Same answer for any number of processes.** Each shot has its own seed, so 1, 3 or 64 processes give
  exactly the same counts as a plain run. CI checks this with 3 OpenMPI processes on three programs, and the
  tests check 1, 2, 3 and 5 ranks on sampled runs, runs that simulate every shot, and noisy runs.
- **Errors do not hang the job.** If a shot fails on one process, every process still reports in, and
  process 0 prints the error and exits with a failure code.
- **Input and output.** Only process 0 prints. If the program reads input (`INPUT`), process 0 reads it
  and passes the values to the others.
- **Threads inside each process.** Each process uses all the cores it is given. With one process per node,
  use your launcher's binding options to give it the whole node (for example `srun --cpus-per-task`).

Every process holds the whole state, so this helps when there are many shots: noisy runs, runs with
measurements in the middle, or millions of samples. For a state too large for one node, see the next
section.

## One state over many processes

```sh
mpirun -n 64 ./target/release/qoa run big.qoa --mpi --distribute
```

With `--distribute` the processes share one state instead of each holding a copy. With 2^g processes, the
top g qubits choose the process, and each process holds 2^(n-g) amplitudes: 64 processes give each 1/64 of
the state. The number of processes must be a power of two.

- **Gates on the other qubits** run inside each process with the usual kernels (fusion, cache blocking,
  SIMD), on the process's own part.
- **Diagonal gates on the top qubits** (RZ, phases, CZ, T) need no communication: each process applies its
  own factor.
- **Other gates on the top qubits** exchange data between pairs of processes, 64 MiB at a time, so the extra
  memory is one chunk, not a second copy of the state.
- **Measurement** adds up partial sums in process order, and every process draws the same random numbers,
  so they all see the same outcome.
- **Sampling** finds which process holds each sampled amplitude, and only that process reads it.
- `EXPECT`, `BELLCHECK` and `DUMPSTATE` read the whole state. `DUMPSTATE` and the report's state table
  gather it to process 0 when it is up to 2^24 amplitudes; a larger state is reported as too large to show.

It gives the same results as one process: the tests check 1, 2, 4 and 8 processes for the final state
(within 1e-12), the counts, and the printed expectation values, and CI checks 2 and 4 OpenMPI processes.
Qubit programs only; optical modes, `--precision quad`, `--gpu` and device noise models say so instead of
running. Every shot runs on all processes together, so here more processes buy memory, not more shots at
once.

### Building against your system's MPI

The MPI part is a few lines of C ([src/mpi_shim.c](../src/mpi_shim.c)) compiled with your MPI's compiler
wrapper, so QOA works with whatever MPI the machine has (Open MPI, MPICH, Cray MPICH, Intel MPI).

| variable | what it does |
|---|---|
| `QOA_MPICC` (or `MPICC`) | the compiler wrapper, default `mpicc`; on Cray systems `QOA_MPICC=cc` |
| `QOA_MPI_LIBS` | link flags, if the wrapper cannot report them, e.g. `"-L/opt/mpi/lib -lmpi"` |

Without `--features mpi`, `qoa run --mpi` says how to build it instead of running on one process.

## How big a state can be

The state takes 16 bytes per amplitude (32 with `--precision quad`), so `n` qubits need 2^n × 16 bytes:
30 qubits need 16 GiB, 34 qubits 256 GiB. A run whose shots are sampled from one simulation needs 8 more
bytes per amplitude for the probabilities.

Before it allocates anything, QOA checks that the state fits in the memory it may use:

- `QOA_MAX_MEMORY`, if set (`512G`, `1.5T`, `64GiB` or a number of bytes);
- otherwise, on Linux, 90% of the memory available at start, or of the job's own limit when a job manager
  (Slurm and others, through cgroups) sets a smaller one;
- otherwise 64 GiB.

A state that does not fit is an error that says how much it needs and how much there is. It does not crash
the node. `qoa check` shows the same numbers without running anything.

## Not done yet

- **OpenCL.** The GPU path today uses Vulkan, which AMD Instinct accelerators (as in Frontier) do not offer.
  An OpenCL backend in 64-bit floating point would cover them, along with NVIDIA and Intel GPUs.
