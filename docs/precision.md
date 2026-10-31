# Numbers and precision

*[Euskaraz](precision.eu.md)*

A simulator's answer is only as good as its arithmetic. This page says what QOA computes in which precision,
and how to ask for more.

## By default: 64-bit amplitudes, exact sums

- **Amplitudes** are pairs of 64-bit floats (IEEE binary64, about 16 significant digits). The gate kernels
  use fused multiply-add where the processor has it, so each product is rounded once.
- **Sums over the state** (the norm, measurement probabilities, the cumulative distribution used for
  sampling, expectation values) are done in double-double: each running total is two 64-bit floats, about
  106 bits. A sum over 2^30 amplitudes is then as accurate as a single 64-bit value. It also does not depend
  on how many threads did the work.
- **The report's `numerics` line** shows |1 - ‖ψ‖²| at the end of the first shot. Every gate is unitary, so in
  exact arithmetic this would be 0; what is left is the rounding the run accumulated.

## `--precision quad`: 128 bits

```
qoa run circuit.qoa --precision quad
```

Every number in the qubit state is a double-double: two 64-bit floats whose sum carries about 106 bits,
roughly 32 significant digits. They are added and multiplied with error-free transformations (two-sum, and
two-product through fused multiply-add). That covers:

- the amplitudes;
- the gate matrices, with sines, cosines and square roots computed to 32 digits from the program's angles;
- measurement, the collapse that follows it, and every sum over the state.

One input stays 64-bit: the angles written in the program. The `.qx` format stores them as 64-bit floats,
so `pi/4` is already π/4 rounded to 16 digits before the simulation starts. QOA takes each angle as an exact
number and adds no error of its own.

What it buys, in `tests/quad.rs`:

| check | 64-bit | 128-bit |
|---|---|---|
| norm after 20,000 random gates on 10 qubits | off by 3e-13 | off by 4e-29 |
| sin, cos and √ against 50-digit references (including sin(12345.678)) | | within 1e-31 |
| 1000 H gates, then 8 T gates on another qubit | | back to the start within 1e-30 |
| same result on 1 thread and on many | yes | yes, bit for bit |

It runs on every 64-bit system QOA supports, using the fused multiply-add they all have (on x86-64 it is
detected at run time). It is slower: a 22-qubit, 400-gate random circuit takes 3.4 s against 0.22 s in 64
bits on 4 cores. Twice the memory per amplitude, and four times the arithmetic, are behind that.

### What stays 64-bit

`--precision quad` runs qubit programs. Programs with optical modes, `--gpu` and `--noise-model` are 64-bit
only for now, and QOA says so instead of silently mixing precisions.

## POWER and 128-bit hardware

POWER9 and later have IEEE binary128 arithmetic in hardware (113-bit significand). QOA's quad mode runs on
POWER today with double-double on POWER's fused multiply-add. Using the binary128 instructions directly is
planned.
