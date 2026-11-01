# QOA on clusters and supercomputers

*[Euskaraz](hpc.eu.md)*

QOA runs on a laptop, and it is meant to scale up to big machines. This page says what works across many
processes today, how to build and start it, and what each part costs.

## Shots over many processes (MPI)

A run with many shots can be spread over MPI processes, on one node or many. Every process simulates the
program, takes its share of the shots, and process 0 collects the counts and prints the result.

```sh
mpirun -n 64 qoa run circuit.qoa --mpi --shots 1000000
srun -n 64 qoa run circuit.qoa --mpi --shots 1000000      # under Slurm
```

- **Same answer for any number of processes.** Each shot has its own seed, so 1, 3 or 64 processes give
  exactly the same counts as a plain run. CI checks this with 3 processes on three programs, with Open MPI
  and with MPICH, and the tests check 1, 2, 3 and 5 ranks on sampled runs, runs that simulate every shot,
  and noisy runs.
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
mpirun -n 64 qoa run big.qoa --mpi --distribute
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
(within 1e-12), the counts, and the printed expectation values, and CI checks 2 and 4 processes with Open
MPI and with MPICH.
Qubit programs only; optical modes, `--precision quad`, `--gpu` and device noise models say so instead of
running. Every shot runs on all processes together, so here more processes buy memory, not more shots at
once.

### Which MPI

Nothing has to be built for MPI. Every QOA release binary, and every `cargo build`, includes `--mpi`: QOA
loads the machine's MPI library when the run starts, the way it loads GPU drivers, so it uses whatever MPI
the cluster has and needs no C compiler or `mpicc` ([src/mpi.rs](../src/mpi.rs), all Rust).

MPI implementations do not share one binary interface, so QOA knows the two families explicitly:

- **Open MPI** (3.x to 5.x, `libmpi.so.40`),
- **the MPICH family**, which shares MPICH's interface: MPICH, Intel MPI, MVAPICH, Cray MPICH, ParaStation
  MPI and Microsoft MPI (`libmpi.so.12`, `libmpich.so.12`, `libmpi_cray.so.12`, `msmpi.dll`).

It picks the library the launcher belongs to (Open MPI's `mpirun` marks its processes) and asks the library
which implementation it is. A library outside these families is refused with a message, never guessed at.

| variable | what it does |
|---|---|
| `QOA_MPI_LIB` | the MPI library to load, by name or full path, e.g. `QOA_MPI_LIB=/opt/cray/pe/lib64/libmpi_cray.so.12` |

The library has to be the one the launcher belongs to: an MPICH program started by Open MPI's `mpirun`
works only through PMIx, and the reverse not at all. If no MPI library is found, `qoa run --mpi` says so
and stops. A static build (`*-musl`) cannot load libraries, so it has no `--mpi`.

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

## GPUs

`--gpu` works with both modes: one MPI process per GPU, each using the GPU its node rank points to. AMD
Instinct accelerators, which have no Vulkan, are reached through OpenCL (`--gpu-api opencl`). See
[gpu.md](gpu.md#several-gpus).

## Not done yet

- **Measurements on the GPU.** A measurement still brings each process's part of the state back to the CPU.
- **Optical modes across processes.** `--distribute` splits qubit programs only.
