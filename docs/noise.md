# Device noise models

*[Euskaraz](noise.eu.md)*

`qoa run --noise-model FILE` runs a program with the noise of a real device: each native gate's
measured error, each qubit's T1 and T2, and each qubit's readout error, taken from the device's
calibration. The result is QOA's prediction of what that device returns, next to the exact (noiseless)
answer that QOA gives without the model.

```
python3 tools/run_on_ibm.py examples/grover.qoa --dry-run --save-model grover-torino
qoa run grover-torino.qasm --noise-model grover-torino.json
```

A ready example is included: Bell transpiled for IBM Torino, and that device's model.

```
qoa run examples/bell-torino.qasm --noise-model examples/bell-torino.json
```

## Where a model comes from

`tools/run_on_ibm.py` builds the model from the device's own calibration (Qiskit's `backend.target`, and
the readout probabilities in `backend.properties()`). It does this for the physical qubits the
transpiled circuit uses. It does this for every run, real or `--dry-run`, and the report gains a
"QOA + model" column:

```
outcome      QOA (exact)   QOA + model   device
101               94.5%         79.9%     80.5%
```

With `--save-model NAME` it also writes `NAME.json` (the model) and `NAME.qasm` (the transpiled circuit,
reduced to the qubits it uses), so the prediction can be rerun, changed or shared without Python.

## What the model contains

It is the standard device model, the same one Qiskit Aer builds with `NoiseModel.from_backend` and IBM's
device models use:

- **After every native gate** (`sx`, `x`, `rz`, `cz`, `reset`, ...), on the qubits it touched:
  1. **depolarizing error**: with probability *q*, a uniformly random non-identity Pauli on those qubits;
  2. **T1/T2 relaxation** over the gate's duration, on each of its qubits: amplitude damping with
     γ = 1 − e^(−t/T1), then dephasing that brings the coherence to exactly e^(−t/T2). T2 is capped at 2·T1
     as physics requires, and the temperature is 0.

  The calibration's gate error *e* is the error of both together. So *q* is solved from *e* after
  subtracting what relaxation alone causes:
  p = d·(e − (1 − F_r)) / (d·F_r − 1), q = p·(4ⁿ − 1)/4ⁿ, with d = 2ⁿ. Here F_r is the relaxation's
  average gate fidelity. If relaxation alone already explains *e*, there is no depolarizing part.
  `reset` gets relaxation only, and `rz` is virtual (error 0, duration 0).
- **On every measurement**: a readout error. A prepared 0 reads 1 with probability `p0m1`, and a prepared
  1 reads 0 with probability `p1m0`. These usually differ, because the qubit can decay while it is being
  read. Only the recorded bit changes, not the state.

A gate the device does not have, or has on other qubits, is an error, not a silent ideal gate. The same
goes for a macro such as `BELL`, a measurement in the X basis, or an optical operation. The message says
to transpile for the device first, which `tools/run_on_ibm.py` does.

## How exact the simulation is

With a model, every shot is simulated from the start as a quantum trajectory. Each channel is applied
through its exact unravelling. A depolarizing error picks its Pauli with the exact probabilities. An
amplitude-damping step jumps to |0⟩ with probability γ·P(1), and otherwise applies the exact no-jump
evolution diag(1, √(1−γ)), renormalized. Dephasing is a Z with probability (1 − f)/2. The distribution
of shots is therefore the channel's distribution itself, not an approximation of it. The only error is
the sampling error of a finite number of shots, plus 64-bit floating-point rounding.

This is checked three ways:

- **Exact formulas** (`tests/noise.rs`):
  - the depolarizing parameter equals Qiskit Aer's to 1e-12 on seven cases, including the clamped ones;
  - the relaxation reproduces e^(−t/T1) and e^(−t/T2) to 1e-15;
  - T1 decay, a Ramsey fringe (T2), asymmetric readout and two-qubit depolarizing each match their
    analytic value within 5σ.
- **Against Aer's exact density matrix**:
  - on Grover, QPE and GHZ transpiled for Torino, 400,000 QOA shots were compared with Aer's
    density-matrix probabilities under the same model, using a χ² test;
  - p-values: Grover 0.52, QPE 0.92, GHZ 0.37, so no detectable difference;
  - every gate channel was also compared as a superoperator with the IBM device model's own channel:
    the largest difference was 0.
- **In every `--dry-run`**:
  - the "device" column is Aer simulating the same calibration;
  - the report prints the distance between the two and how much of it sampling alone explains.

## What the model leaves out

The model holds what a calibration measures, and no more. A real device also has:

- decoherence while a qubit waits for others (idle time is not in the model);
- crosstalk between neighbouring qubits;
- leakage out of the qubit levels;
- coherent (systematic) gate errors;
- calibrations that drift over hours.

So the real device usually does somewhat worse than the model predicts. The gap between "QOA + model" and
"device" in a real run measures how much these effects matter for that circuit.

Not covered yet:

- circuits with classical branching in `tools/run_on_ibm.py`'s prediction;
- devices whose two-qubit gate is ECR (IBM's Eagle generation). QOA has no single instruction for ECR.
  Heron devices (CZ), which the free plan uses, are covered.

## File format

JSON. Times are in seconds, probabilities are between 0 and 1, and `qubits[i]` is QOA's `qi`:

```json
{
  "format": "qoa-noise-1",
  "device": "ibm_torino",
  "calibrated": "2026-10-29 12:00:00",
  "qubits": [
    {"physical": 92, "t1": 2.12e-4, "t2": 1.41e-4, "p0m1": 0.0093, "p1m0": 0.0088}
  ],
  "gates": [
    {"name": "sx", "qubits": [0], "error": 2.4e-4, "duration": 3.2e-8},
    {"name": "cz", "qubits": [0, 1], "error": 4.9e-3, "duration": 6.8e-8}
  ]
}
```

- `t1` or `t2` set to `null` means unknown, treated as infinite (no relaxation).
- Gate names are Qiskit's.
- `cz`, `cp`, `swap` and `rzz` are found in either qubit order.
- Unknown fields, out-of-range values, a qubit outside the list and duplicate gates are errors.
- Numbers are read exactly as written: the decimal of every double round-trips to the same bits.
