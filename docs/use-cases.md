# What QOA is for, and what stands in its way

[Euskaraz: use-cases.eu.md](use-cases.eu.md)

QOA 0.4 is a small, exact simulator for qubits and photons in one state, driven by an assembly language with
real classical control. This page is about where that is useful today, where it is not, and what would
help or hinder people adopting it.

## What QOA is, in one paragraph

One state vector holds qubits and optical modes (photon numbers up to a cutoff), so a qubit can emit a
photon into a mode, a photon can shift a qubit's phase, and both can be measured with the result steering
the program. Every number it prints is computed exactly and checked: gates against an independent
simulator, beamsplitters against matrix permanents, exported programs against Qiskit, linear optics against
Quandela's Perceval. It reads and writes OpenQASM 2.0, writes IonQ JSON, prints a plain report or a JSON
record, speaks English and Basque, and runs at the memory's speed on x86-64, ARM and RISC-V. It stops at
around 28-30 qubits (or the equivalent number of amplitudes), like every exact state-vector simulator.

## Where it is useful

**1. Teaching quantum optics and quantum information together.**[^basque] Most courses split the two: circuit tools
(Qiskit, Cirq) have no photons, optics tools (QuTiP, Strawberry Fields, Perceval) have no qubit programs
with control flow. QOA's examples cover two-photon interference, a Mach-Zehnder interferometer, coherent
and squeezed light, a Schrödinger cat made with a qubit, emitter-photon entanglement and heralded
entanglement of two distant emitters, next to Bell, GHZ, teleportation, Grover and Shor's algorithm for 15,
all in one language, each with its expected results checked by the test suite. The report says what was
computed and how (sampled, simulated shot by shot, or exact), which is what a student needs to read results
correctly.

**2. Prototyping quantum-network protocols at small scale.** Heralded entanglement, entanglement swapping,
repeat-until-success and feed-forward corrections need qubits, photons, measurements and classical logic in
the same program. QOA runs these exactly with a handful of emitters and modes, which is enough to check that
a protocol's logic and its success probabilities are right before moving to a network simulator (NetSquid,
SeQUeNCe) that models timing and memory decay at scale.

**3. Planning small photonic experiments.** Before building a setup, a lab wants to know the expected
counts: Hong-Ou-Mandel visibility with some loss, the photon statistics of squeezed light, how many photons
a cat state needs, how a finite photon cutoff distorts the answer. QOA gives exact answers for a few modes
and a few photons, reports truncation instead of hiding it, and its linear-optics results agree with
Perceval to 1e-15.

**4. An executable specification for hybrid instruction sets.** Architectures that link qubit modules with
photons (photonic interconnects between ion or atom processors, emitter-based photonic machines) need an
instruction layer that talks about both. QOA is a concrete, tested example of such a layer, with a
binary format, a readable text form, a dense glyph form and an OpenQASM bridge. Compiler and architecture
research can use it as a reference to compile to or to check against.

**5. A reference to test other tools against.** Because its results are exact and cross-checked, QOA can be
the "known good" answer in another project's tests: export a circuit, run both, compare
(`tools/hardware_check.py` does exactly this with Qiskit and Perceval).

**6. Performance-engineering teaching.** `src/simd.rs` has the same three kernels in AVX-512, AVX2, NEON and
RVV assembly, each tested against a scalar version, and `cargo bench --bench bandwidth` shows them next to
the machine's memory ceiling. It is a compact, real example of memory-bound code, cache blocking and
per-ISA SIMD.

**7. Outreach and art.** `qoa visual` (audio to video) and the Kerr black-hole example (lensing plus Hawking
pairs whose statistics QOA computes) are ways to start a conversation about quantum physics with people who
would never open a circuit diagram.

[^basque]: QOA also works entirely in Basque (messages, errors, help and code comments), which as far as we
know no other quantum programming tool does. That can help Basque-medium courses and outreach, for example
around the BasQ programme and its IBM quantum computer in Donostia.

## Where it is not the right tool

| need | better tools | why |
|---|---|---|
| more than about 30 qubits | Qiskit Aer (MPS), cuQuantum, tensor-network simulators | an exact state vector doubles with every qubit |
| large Gaussian optics (many modes, squeezing) | Strawberry Fields, The Walrus | Gaussian states have compact descriptions; QOA stores every photon-number amplitude |
| realistic device noise (T1, T2, pulses, crosstalk) | Qiskit Aer noise models, Qiskit Dynamics, QuTiP | QOA's noise is a simple documented model |
| chemistry, optimisation, machine learning | Qiskit, PennyLane, CUDA-Q | QOA has no algorithm libraries |
| network timing and memory decay at scale | NetSquid, SeQUeNCe | QOA has no notion of time |
| running on hardware day to day | the vendors' SDKs | QOA exports to them, but has no job management |

## Adoption: what stands in the way, and what QOA has

People do not choose tools on merit alone. These are the forces that keep them on what they already use, and
where QOA stands on each:

| force | how hard it hits QOA | what QOA has | what would help most |
|---|---|---|---|
| **Python habit**: courses, notebooks and jobs assume `pip install` and Python | hard | a JSON report any language can read; OpenQASM in and out | a Python package (built with pyo3 and maturin) with `run(source)` returning the report as a dict, and conversions to and from Qiskit circuits |
| **Install friction** | medium | prebuilt binaries for 13 systems: Windows, macOS and Linux on x86-64 and ARM64, Linux on RISC-V, IBM POWER, IBM Z and LoongArch, and static builds that also run on Android | `cargo install qoa` from crates.io |
| **Ecosystem and backing**: IBM, Google, Xanadu, Quandela stand behind the big tools | hard, and not fixable | interoperability: QOA does not ask anyone to leave Qiskit or Perceval | QOA as a companion, not a replacement |
| **"Assembly" sounds hard** | medium | readable mnemonics, errors that point at the line with a suggestion, the glyph syntax for density | a five-minute tutorial that starts from OpenQASM people already know |
| **Trust**: is a small project correct? | medium | a strict test suite (reference simulator, permanents, Qiskit, Aer and Perceval cross-checks, every example's claims checked) run in public CI on every supported processor | independent users reporting what they found |
| **One maintainer**: will it be around next year? | medium | a small, readable codebase, documented design choices and a [contributing guide](../CONTRIBUTING.md) | issue templates and a few good first issues |
| **Licence**: GPL-3.0 keeps some companies from embedding it | low for teaching and research, real for industry | GPL keeps the project open | a licensing choice for the project; LGPL or a dual licence are the usual compromises |
| **Language**: most tools are English-only | works *for* QOA in Basque-medium teaching | complete Basque and English | Basque tutorials |

What works in QOA's favour is mostly that it is **different where the incumbents are thin**: qubits and
photons in one program, exact answers you can check, results that say how they were computed, and a second
language. It does not have to win against Qiskit to be worth using; it has to be the best tool for a few
specific jobs, and the cheapest switch in is that it already reads and writes OpenQASM.

## Roadmap

In order of value for the effort:

1. A Python package with a minimal API (run, report as a dict, Qiskit conversion) and three notebooks (HOM,
   heralded entanglement, the Kerr black hole) in English and Basque.
2. OpenQASM 3 export, so programs with loops, registers and repeat-until-success reach hardware.
3. Export to Quandela's Perceval, so the linear-optics part can run on photonic hardware.
4. Tutorials for Basque-medium courses.
5. A stable glyph syntax (its open points are in [glyph.md](glyph.md)) and a stable JSON report, the
   "classical ABI" other programs read results through.

Already done: prebuilt binaries and public CI (0.4.0, extended in 0.4.1), and noise models from real
devices' calibrations (0.4.1).
