# Would QOA run on a real quantum computer?

[Euskaraz: hardware.eu.md](hardware.eu.md)

**Short answer: the qubit part, yes, through OpenQASM 2.0, OpenQASM 3 or IonQ's JSON; the linear-optics part, yes in
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

## OpenQASM 3

`qoa export --to qasm3` writes OpenQASM 3.0 with the standard gate library (`stdgates.inc`). It covers
what the OpenQASM 2.0 export covers, and keeps three control-flow shapes as blocks instead of unrolling or
refusing them:

- counted loops, as `for uint i in [0:n-1] { ... }`;
- a forward jump on a measured bit, as `if (c[k] == v) { ... }`, and with a `JUMP` over a second block, as
  `if (...) { ... } else { ... }`;
- repeat-until-success (a backward jump on a measured bit, or a test at the top and a jump back), as
  `while (c[0] == false) { ... }`.

Gates that have no name in `stdgates.inc` use modifiers or equivalents: `inv @ sx`, `ctrl @ sx`,
`ctrl(3) @ x`, `ctrl(4) @ x`, `cp` for `cu1`, and `cu(θ,φ,λ,0)` for `cu3`. `qoa run` and `qoa import` read the
same subset back (bits, `if`/`else`, `while`, `for`, and the `inv`, `ctrl` and `pow` modifiers), so an
exported file also runs in QOA itself; a test checks that QOA -> OpenQASM 3 -> QOA gives the same counts,
shot for shot, and that OpenQASM 3 -> QOA -> OpenQASM 3 keeps the blocks.

Checked against Qiskit 2.5.2 and Aer 0.17.2 (`python3 tools/hardware_check.py --only qasm3`):

| program | blocks | QOA = Qiskit? (TVD) | IBM Torino model |
|---|---|---|---|
| teleportation | `if` | yes (0.003) | compiles |
| `examples/until.qoa`, repeat until a qubit reads 1 | `while` | yes (0.000) | refused: `while_loop` is not supported |
| `tests/programs/sanity/qasm3-blocks.qasm`, every block kind | `while`, `for`, `if`, `else` | yes (0.000) | refused: `while_loop` is not supported |

**What runs where.** The exported files are valid OpenQASM 3, and Qiskit loads and simulates every one.
IBM's devices take less. IBM's guide ([Execute dynamic circuits](https://quantum.cloud.ibm.com/docs/en/guides/execute-dynamic-circuits),
read in November 2026) says its new dynamic circuits are available on all backends, that "IBM Quantum
Compute Service currently only supports the conditional if statement", and lists these limits: "The for,
while, and switch instructions are not supported", nested conditionals are not allowed, reset and
measurement inside a conditional are not supported, arithmetic is not supported, and a condition compares
at most 32 bits. The model of IBM's Torino device in qiskit-ibm-runtime 0.51 agrees: it compiles
teleportation and an `if`/`else`, and refuses `while_loop`. IBM may change this, so check the guide before
relying on it. As of this writing:

- **`if` on a measured bit** (teleportation, feed-forward): runs on IBM, from OpenQASM 2.0 or 3. An `else`
  compiles for the Torino model, though IBM's guide names only the `if` statement.
- **Counted loops:** `tools/run_on_ibm.py` unrolls them before sending.
- **Repeat-until-success:** valid OpenQASM 3 and runs in Qiskit's simulators, but IBM's devices do not run
  `while` loops, nor the reset and measurement inside it; `tools/run_on_ibm.py` says so before spending QPU
  time, as it does for a block inside a block.
- **Still not exportable:** programs that compute with registers (Shor's classical part, counters, loops
  counted by a register, the attempt counter in `examples/retry.qoa`). OpenQASM 3 has integer variables,
  but Qiskit's OpenQASM 3 importer does not read them (Qiskit 2.5.2 rejects `uint` declarations) and IBM's
  devices do not run them, so the exporter refuses these programs, naming each line and why, instead of
  writing a file that nothing downstream can run.

## Optical programs

OpenQASM has no optical modes, and no qubit machine can run them. Photonic processors can run the linear
part: single photons through beamsplitters and phases, counted at the end. `tools/hardware_check.py` takes a
QOA network, gets its single-photon transfer matrix from QOA, and runs the multi-photon input through
Perceval, the software for Quandela's photonic processors:

| network | photons | largest difference between QOA and Perceval |
|---|---|---|
| Hong-Ou-Mandel | 2 | 7e-16 |
| Mach-Zehnder, phase π/3 | 1 | 3e-16 |
| 6-mode interferometer (boson sampling) | 3 | 7e-16 |

So the linear-optics subset of QOA is exactly what a photonic chip computes, and could be sent to one.
What has no hardware target yet:

- **qubit-photon coupling** (`EMIT`, `DISPERSIVE`): this is cavity and circuit QED, done in research labs,
  not offered as a cloud service;
- **nonlinear optics** (`KERR`, `CROSSKERR`) at the single-photon level, and **number-resolving detection
  with feed-forward** in the middle of a circuit: partly demonstrated in labs, not available to rent;
- **displacement and squeezing** exist on continuous-variable photonic machines, but as of this writing
  none is openly available as a cloud device (Xanadu's Borealis was, from its 2022 launch, through Xanadu
  Cloud and Amazon Braket; articles dated 2026 that describe that access appear to repeat the 2022
  announcement, and Xanadu's own pages did not confirm current access); check the providers before
  relying on this.

## Running on real hardware for free

**IBM Quantum's Open Plan** is the free way onto real qubits: a free account gets the equivalent of up to
10 minutes of QPU time (counted in IBM's resource units) per rolling 28-day window, on IBM's current
devices, and since 16 March 2026 active Open Plan users can opt in to the equivalent of 180 more minutes
over 12 months ([IBM: plans overview](https://quantum.cloud.ibm.com/docs/en/guides/plans-overview), read in
November 2026; the terms change, so check before relying on them). A small circuit with a few thousand
shots uses only seconds of that.

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
model. It sends OpenQASM 2.0 when the program fits it and OpenQASM 3 otherwise (with counted loops
unrolled). Programs with register arithmetic or loops counted by a register cannot go, and a
repeat-until-success loop is refused by IBM's current devices; the script says which lines and why.

Every run also shows a third column, "QOA + model": QOA's own prediction for that device, simulated with
the device's calibration (gate errors, T1/T2, readout errors), as described in [noise.md](noise.md). On a
`--dry-run`, it matches Aer's simulation of the same calibration to sampling accuracy. On the real device,
the gap between the two columns is what the calibration does not capture: idle decoherence, crosstalk,
drift. `--save-model NAME` keeps the model and the transpiled circuit for `qoa run --noise-model`.

**Photons:** Quandela runs Perceval programs on its photonic processors (Ascella, Belenos) through
Quandela Cloud, and sells time through Scaleway and OVHcloud; European researchers can apply for credits
through GENCI ([Quandela Cloud](https://www.quandela.com/cloud/), read in November 2026). Quandela's
public pages do not say whether a free account includes QPU time, or how much, so check with Quandela. The QOA side is ready: `tools/hardware_check.py`
already turns a QOA linear-optics network into the transfer matrix Perceval runs.

**IonQ and Amazon Braket** have free simulators, but their real devices are paid, outside research credit
programmes.

## Reproducing

```sh
cargo build --release
pip install qiskit qiskit-aer qiskit-ibm-runtime perceval-quandela
python3 tools/hardware_check.py --out docs/hardware-measurements.md
```
