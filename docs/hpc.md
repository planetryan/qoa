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
measurements in the middle, or millions of samples. Splitting one large state over many nodes is the next
step (see below).

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

- **One state over many nodes.** The state's top qubits would map to processes; gates on those qubits
  exchange halves of the state between pairs of processes. This is what simulations above one node's memory
  need (about 34 qubits on a 512 GiB node).
- **OpenCL.** The GPU path today uses Vulkan, which AMD Instinct accelerators (as in Frontier) do not offer.
  An OpenCL backend in 64-bit floating point would cover them, along with NVIDIA and Intel GPUs.
