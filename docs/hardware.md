# Would QOA run on a real quantum computer?

[Euskaraz: hardware.eu.md](hardware.eu.md)

**Short answer: the qubit part, yes, through OpenQASM 2.0 or IonQ's JSON; the linear-optics part, yes in
principle, on photonic processors such as Quandela's; the qubit-photon coupling and the nonlinear optics,
not on any machine you can rent today.** No program was sent to real hardware for this page, because that
needs an account with a provider. Everything short of that was done and measured, and
`tools/run_on_ibm.py` takes the last step on IBM's free plan once you have an account (see the end).

## Qubit programs

QOA writes the qubit part of a program as OpenQASM 2.0 (`qoa export --to qasm`) or IonQ's JSON
(`qoa export --to ionq`). The exporter now also handles the two control-flow shapes OpenQASM 2.0 can say:
counted loops (unrolled) and forward jumps over a block of gates on one measured bit (each gate gets an
`if (cK==v)`, which is a dynamic circuit on IBM hardware). That covers teleportation and Grover search.

`tools/hardware_check.py` exports each example, runs it in Qiskit, transpiles it for two real IBM devices
(their published calibrations: `fake_torino`, a 133-qubit Heron, and `fake_sherbrooke`, a 127-qubit
Eagle) and simulates the result with each device's measured noise. From
[hardware-measurements.md](hardware-measurements.md):

| program | 2-qubit gates on Heron | QOA = Qiskit? (TVD) | right answers, ideal | on Heron (noise) | on Eagle (noise) |
|---|---|---|---|---|---|
| Bell pair | 1 | yes (0.001) | 100% | 98% | 98.5% |
| GHZ, 5 qubits | 5 | yes (0.001) | 100% | 90% | 89% |
| teleportation (feed-forward) | 2 | yes (0.003) | 100% | 98% | 98% |
| phase estimation, 4 qubits | 13 | yes (0.000) | 100% | 92% | 85% |
| Grover, 3 qubits | 37 | yes (0.004) | 94.5% | 80% | 75% |
| Grover key search, 8 qubits | 354 | yes (0.004) | 96% | 30% | 9.6% |

What this says:

- **The exported programs are the same programs.** QOA and Qiskit's simulator agree to within sampling
  noise on every one, and a QOA -> OpenQASM -> QOA round trip gives identical counts (tests/qasm.rs).
- **Small circuits run well on today's hardware**; the answer survives in 75-98% of shots.
- **The 8-qubit key search does not.** After transpiling to the device's native gates and connections it
  needs 354 two-qubit gates, and the noise leaves the right key in 30% of shots on Heron and 9.6% on Eagle,
  where guessing would give 6.25%. This is the real limit of current machines, not of QOA: error
  mitigation helps somewhat, error correction is what fixes it.
- **Not exportable yet:** programs that compute with registers (Shor's classical part, counters), loops
  whose count is in a register, and repeat-until-success loops (backward jumps). OpenQASM 3 can express
  all three (`while`, `if/else`, classical variables); an OpenQASM 3 exporter is the next step.

## Optical programs

OpenQASM has no optical modes, and no qubit machine can run them. Photonic processors can run the linear
part: single photons through beamsplitters and phases, counted at the end. `tools/hardware_check.py` takes a
QOA network, gets its single-photon transfer matrix from QOA, and runs the multi-photon input through
Perceval, the software for Quandela's photonic processors:

| network | photons | largest difference between QOA and Perceval |
|---|---|---|
| Hong-Ou-Mandel | 2 | 7e-16 |
| Mach-Zehnder, phase π/3 | 1 | 0 |
| 6-mode interferometer (boson sampling) | 3 | 7e-16 |

So the linear-optics subset of QOA is exactly what a photonic chip computes, and could be sent to one.
What has no hardware target yet:

- **qubit-photon coupling** (`EMIT`, `DISPERSIVE`): this is cavity and circuit QED, done in research labs,
  not offered as a cloud service;
- **nonlinear optics** (`KERR`, `CROSSKERR`) at the single-photon level, and **number-resolving detection
  with feed-forward** in the middle of a circuit: partly demonstrated in labs, not available to rent;
- **displacement and squeezing** exist on continuous-variable photonic machines, but as of this writing
  none is openly available as a cloud device (Xanadu's Borealis was for a time); check the providers before
  relying on this.

## Running on real hardware for free

**IBM Quantum's Open Plan** is the free way onto real qubits: a free account gets up to 10 minutes of QPU
runtime per rolling 28-day window, on IBM's current devices (IBM's documentation, checked October 2026;
the terms change, so check [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com) before relying on them).
A small circuit with a few thousand shots uses only seconds of that.

`tools/run_on_ibm.py` does the whole trip for the qubit part of a QOA program:

```sh
pip install qiskit qiskit-ibm-runtime qiskit-aer
cargo build --release
python3 tools/run_on_ibm.py examples/bell.qoa --dry-run     # first: a local model of an IBM device, no account
export QISKIT_IBM_TOKEN=...                                  # your API key from quantum.cloud.ibm.com
python3 tools/run_on_ibm.py examples/bell.qoa               # then the real device (asks before using QPU time)
python3 tools/run_on_ibm.py examples/grover.qoa --no-wait   # or submit and come back later:
python3 tools/run_on_ibm.py --job <id>
```

It exports the program to OpenQASM, compiles it for the least busy device that fits, says how many two-qubit
gates that takes, asks before spending QPU time, records the job in `target/ibm-jobs.json`, and when the
device answers prints its counts next to QOA's exact simulation, with the share of shots on the outcomes QOA
predicts, the distance between the two distributions, and the QPU seconds the job used. With `--dry-run`
the same path runs on a local model of an IBM device with its measured noise, which is how the numbers
above were made: Bell 97.8%, teleportation 97.4%, Grover 81.8% of shots on the right answers on the Torino
model. Programs the exporter refuses (register arithmetic, loops counted by a register, backward jumps)
cannot go yet; the script says which lines and why.

Every run also shows a third column, "QOA + model": QOA's own prediction for that device, simulated with
the device's calibration (gate errors, T1/T2, readout errors), as described in [noise.md](noise.md). On a
`--dry-run`, it matches Aer's simulation of the same calibration to sampling accuracy. On the real device,
the gap between the two columns is what the calibration does not capture: idle decoherence, crosstalk,
drift. `--save-model NAME` keeps the model and the transpiled circuit for `qoa run --noise-model`.

**Photons:** Quandela Cloud offers free accounts and runs Perceval programs on photonic processors
(Ascella, Belenos), but whether a free account includes QPU time, and how much, could not be checked from
here; check [cloud.quandela.com](https://cloud.quandela.com). The QOA side is ready: `tools/hardware_check.py`
already turns a QOA linear-optics network into the transfer matrix Perceval runs.

**IonQ and Amazon Braket** have free simulators, but their real devices are paid, outside research credit
programmes.

## Reproducing

```sh
cargo build --release
pip install qiskit qiskit-aer qiskit-ibm-runtime perceval-quandela
python3 tools/hardware_check.py --out docs/hardware-measurements.md
```
