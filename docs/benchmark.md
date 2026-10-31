# Measuring a machine: `qoa --benchmark`

*[Euskaraz](benchmark.eu.md)*

`qoa --benchmark` (or `--bm`, `qoa benchmark`, `qoa bench`, `qoa bm`) runs every part of QOA on fixed
problems and scores each one. It takes about half a minute; `--quick` uses smaller problems and takes about
ten seconds. `--json` prints the result for other programs, and `--no-gpu` skips the GPU part.

```text
machine    Intel(R) Xeon(R) Processor @ 2.80GHz (x86_64), 4 thread(s), avx512 kernels
scale      1000 = 4 cores of a 2.8 GHz Intel Xeon with AVX-512 (cloud VM), QOA 0.4.3; twice the score, twice as fast

  1071  memory bandwidth  36.4 GB/s
        a[i] = b[i] + 3·c[i] over three arrays (STREAM triad): what bounds every large simulation
        3 × 64 MiB
        best of 4: 0.006 s (median 0.006 s)
  ...
score      1004 (geometric mean of the 11 CPU items); GPU 1337
```

## The scale

Every item scores 1000 on the reference machine: 4 cores of a 2.8 GHz Intel Xeon with AVX-512, a cloud
virtual machine. Twice the score means twice as fast at that item. The total is the geometric mean of the
CPU items, as SPEC does it, so no single item decides it.

The GPU item has its own score, because not every machine has a GPU. 1000 there means the GPU runs the
circuit as fast as the reference machine's CPU, so a GPU scoring 10000 is ten times faster than that CPU.

`--quick` runs smaller problems, which fit in cache more often, so its scores are not comparable with a full
run's. Compare full runs with full runs.

## What each item measures

| item | what it does | unit |
|---|---|---|
| memory bandwidth | STREAM triad over three arrays: the limit of every large simulation | GB/s |
| assembler | parses, checks and assembles a 100,000-line generated program | thousands of lines per second |
| QOA machine | runs QOA's assembler written in QOA ([tools/asm.qoa](../tools/asm.qoa)) on its own 10,000-line source: the speed of the QOA machine's classical instructions. It also checks that the result is byte for byte what `qoa compile` writes | millions of QOA instructions per second |
| qubit gates, 20 and 24 qubits | a random circuit through the SIMD kernels, with fusion and cache blocking | billions of gate × amplitude updates per second |
| sampling | simulates a 20-qubit state once, then samples a million shots with exact cumulative sums | millions of shots per second |
| shots one by one | a program with a mid-circuit measurement, a branch and 1% noise, every shot from the start | thousands of shots per second |
| optical modes | displacement, squeezing, beamsplitters and Kerr on three modes of 24 photon numbers | operations per second |
| 128-bit gates | a random circuit in double-double | millions of gate × amplitude updates per second |
| exact sums | the norm of a 16-million-amplitude state in double-double | billions of amplitudes per second |
| light | squeezed light through loss and a Kerr medium, one exact homodyne shot per mode (`qoa visual`'s physics) | thousands of modes per second |
| Wigner panels | exact pixel-averaged Wigner functions of Kerr states, as the video draws them | panels per second |
| GPU | the 22-qubit circuit on the GPU, moving the state there and back included | billions of gate × amplitude updates per second |

Every problem uses fixed seeds, so every machine does exactly the same work. Each item runs three times
(two with `--quick`) and the best time counts; the report also shows the median, so a noisy machine shows
up as a gap between the two.

## What the score is not

It measures QOA's own work on this machine, not quantum computing in general, and not other simulators. Each
item is a real part of QOA and uses the same code a program does.
