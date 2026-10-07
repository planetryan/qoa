# Would QOA run on a real quantum computer?

[Euskaraz: hardware.eu.md](hardware.eu.md)

**Short answer: the qubit part, yes, through OpenQASM 2.0 or IonQ's JSON; the linear-optics part, yes in
principle, on photonic processors such as Quandela's; the qubit-photon coupling and the nonlinear optics,
not on any machine you can rent today.** No program was sent to real hardware for this page, because that
needs an account with a provider (see the end). Everything short of that was done and measured.

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

## Sending a program to real hardware

These need an account and a key; none was used here.

**IBM Quantum** (free tier available), OpenQASM 2.0 through Qiskit:

```python
from qiskit import qasm2, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

text = open("bell.qasm").read()                       # qoa export --to qasm examples/bell.qoa -o bell.qasm
qc = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
service = QiskitRuntimeService(channel="ibm_quantum_platform", token="YOUR_TOKEN")
backend = service.least_busy(operational=True, simulator=False)
job = SamplerV2(backend).run([transpile(qc, backend)], shots=4000)
print(job.result()[0].data)
```

**IonQ** (through their API or a cloud marketplace), with `qoa export --to ionq`: the JSON body goes
to IonQ's jobs endpoint with an API key, target `qpu.aria-1` or `simulator`.

**Quandela** (photonic), with Perceval's `RemoteProcessor` and a token from their cloud, using the transfer
matrix as in `tools/hardware_check.py`.

## Reproducing

```sh
cargo build --release
pip install qiskit qiskit-aer qiskit-ibm-runtime perceval-quandela
python3 tools/hardware_check.py --out docs/hardware-measurements.md
```
