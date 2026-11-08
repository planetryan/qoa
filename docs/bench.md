# A tabletop quantum-optics bench for QOA

[Euskaraz: bench.eu.md](bench.eu.md)

Cloud machines are one way to run QOA programs on real hardware ([hardware.md](hardware.md)). This page is
about the other way: a small quantum-optics bench of your own, built from parts that teaching labs use, on
which QOA's linear-optics instructions run on real single photons.

**No hardware runs all of QOA, and none ever will.** `DUMPSTATE`, `EXPECT`, `BELLCHECK` and `MEANPHOTON`
read the exact quantum state without measuring it, which only a simulator can do. `KERR` and `CROSSKERR`
at the single-photon level, `EMIT` and `DISPERSIVE` (a quantum emitter in a cavity), and deterministic
two-qubit gates on photons need research-lab equipment. What a bench can run is a well-defined subset, and
QOA checks programs against it (`qoa check --target`, below).

## Three tiers

| tier | what it is | QOA instructions it runs | rough cost |
|---|---|---|---|
| A | classical light: a laser, beamsplitters, phase shifters, photodiodes | `BS`, `PH`, `MS`, `LOSS`, `DISPLACE`; for programs with one photon, the detection probabilities are the same as for a single photon | low thousands of dollars |
| **B** | **real single photons**: a photon-pair source, single-photon detectors, coincidence counting | `SINGLEPHOTON` (heralded), `BS`, `PH`, `MS`, `LOSS`, `DETECT`, `COINCIDENCE`; two-photon interference (Hong-Ou-Mandel); polarisation qubits: `BELL` as the source, every one-qubit gate, `MEASURE` in X, Y or Z | tens of thousands |
| C | photon-number-resolving detectors (cryogenic), squeezed light with homodyne detection, fast switches | `COUNT`, `SQUEEZE`, `PARITY`, heralded `SUBPHOTON`, branching in the middle of a run | hundreds of thousands and up |

Tier A is not quantum (classical waves reproduce one-photon statistics, but not two-photon interference),
but it is the cheapest way to build and test the whole path from a `.qx` file to optics and back. Tier B is
genuinely quantum, and simple by 2026 standards: it is what undergraduate quantum-optics labs build.

## What tier B needs

Prices are list prices found in November 2026; they change, and vendors quote per order. Where no price
could be found, the table says so.

### The simplest route: a teaching kit

| item | what it gives | price |
|---|---|---|
| Thorlabs EDU-QOP1 (imperial) or EDU-QOP1/M (metric), Quantum Optics Educational Kit | a free-space photon-pair source with a type-I BBO crystal, and the parts for the course experiments, starting with Hanbury Brown-Twiss | $35,744.68 ([Thorlabs](https://www.thorlabs.com/item/edu-qop1)) |
| Thorlabs EDU-QOPA1(/M), Polarization-Entanglement Extension Kit | polarisation-entangled pairs (needs EDU-QOP1) | $4,800.18 ([Thorlabs](https://www.thorlabs.com/quantum-optics-educational-kits)) |
| or: qutools quED | a fiber-coupled source of polarisation-entangled pairs (405 nm pump, BBO crystals), two silicon single-photon avalanche detectors, a four-channel counter with coincidence logic | on request ([qutools](https://qutools.com/qued/)) |

Check with the vendor exactly which detectors and electronics a kit includes before ordering; the parts
list below is what the bench needs in total.

### Building it from parts

| part | what it is for | example | price |
|---|---|---|---|
| pump laser, 405 nm, tens of mW | makes the photon pairs (one 405 nm photon splits into two at 810 nm) | Thorlabs L405A1 diode, 175 mW (plus a mount with temperature control and a driver); or LP405C1, 30 mW, single-mode fiber output | $861.60 / $1,418.82 ([L405A1](https://www.thorlabs.com/item/L405A1), [LP405C1](https://www.thorlabs.com/item/LP405C1)) |
| nonlinear crystal | spontaneous parametric down-conversion: the pair source | type-I BBO, 1 mm (Thorlabs NLCQ1); two crossed ones for polarisation entanglement; or type-II BBO (NLCQ4); or a brighter PPKTP crystal (needs a temperature-controlled oven) | $652.70 / $674.10 / $2,685.00 ([BBO](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=16384), [PPKTP](https://www.thorlabs.com/ppktp-crystals-for-spontaneous-parametric-down-conversion-spdc)) |
| single-photon detectors, silicon SPAD, 2 to 4 | count photons at 810 nm (about 60% efficiency); 2 for heralding and Hong-Ou-Mandel, 4 for a full polarisation Bell test | Hamamatsu photon-counting modules; Excelitas SPCM, Thorlabs and Laser Components COUNT on quotation | about $2,800-3,700 each at Hamamatsu ([shop](https://shop.hamamatsu.com/collections/spad-single-photon-avalanche-diode)); others on request |
| coincidence counting | counts pairs of clicks within a few nanoseconds | a time tagger (Swabian Time Tagger 20: 8 inputs, 34 ps jitter, quoted at $11,300 in a 2024 teaching-lab parts list); or an FPGA coincidence counter built from the published teaching-lab designs | ~$11,300 ([arXiv:2407.15759](https://arxiv.org/html/2407.15759v1)); the FPGA counter of [Branning et al., Am. J. Phys. 77, 667 (2009)](https://www3.trincoll.edu/~dbrannin/Publications/AJP_77_667_(2009).pdf) costs under 5% of the traditional electronics |
| polarisation optics at 810 nm | the qubit gates and measurement bases: half- and quarter-wave plates, polarizing beamsplitter cubes; a 50:50 beamsplitter for Hong-Ou-Mandel | zero-order wave plates, PBS and NPBS cubes from any optics vendor | no price found here; typically a few hundred dollars each |
| filters and fiber coupling | remove the pump light (long-pass), select 810 nm (narrow band-pass), bring photons to the detectors | long-pass and 810/10 band-pass filters, fiber couplers, multimode fibers | no price found here |
| mechanics | holds everything still | an optical breadboard, kinematic and rotation mounts, posts | no price found here; often the largest hidden cost |
| motorized rotation mounts and a phase shifter | **what makes the bench programmable**: QOA sets wave-plate angles (one-qubit gates, measurement bases) and phases (`PH`) instead of a person turning knobs | motorized rotation mounts for the wave plates; a piezo mirror or a liquid-crystal retarder for phase | no price found here |
| safety | 405 nm laser goggles, a light-tight enclosure (the detectors need the dark) | | |

Order of magnitude: a minimal heralded-photon and Hong-Ou-Mandel bench (one laser, one crystal, two
detectors, an FPGA counter, optics) is in the low tens of thousands of dollars; a full polarisation Bell-test
bench with four detectors, a time tagger and motorized wave plates is around the price of the teaching kit.

## Which QOA programs run on it

On a tier-B bench, a program's modes are spatial paths of photons, and its qubits are the polarisations of
the two photons of a pair. `qoa check --target bench-b FILE` says whether a program fits, and names each
line that does not and why:

- `examples/mach-zehnder.qoa` (one heralded photon, two beamsplitters, a phase, click detectors) runs as it
  is.
- Hong-Ou-Mandel runs with click detectors: two heralded photons into a 50:50 beamsplitter, and the two
  output detectors never click together, which is the dip (`examples/hom-clicks.qoa`). `examples/hom.qoa` uses `COUNT`
  (number-resolving), which is tier C.
- A Bell test runs with `BELL q0 q1` as the source, any one-qubit gates (wave plates), and `MEASURE` in X,
  Y or Z on each photon (`examples/bell-photons.qoa`). `examples/bell.qoa` builds the pair with `H` and
  `CNOT`, which photons cannot do deterministically, so it is refused.
- The bench runs the same fixed sequence in every shot, so jumps, loops, calls and reading results in the
  middle of a run are refused (branching on a measurement needs fast switches, tier C).
- Probabilistic steps are honest: a heralded photon is there only when its partner clicked, detectors miss
  some photons and click in the dark, and QOA's simulation of the bench includes both.

`--target bench-a` checks the same way for a classical-light bench: one photon at most, one qubit (the
polarisation of the beam) at most, and nothing that needs two photons.

## Simulating your bench

A bench model (a json file, format `qoa-bench-1`) describes a bench's imperfections, and
`qoa run --noise-model bench.json program.qoa` simulates the program the way that bench would run it. The
program must pass the bench's profile first. [examples/bench-b.json](../examples/bench-b.json) has
illustrative numbers:

```sh
qoa run --noise-model examples/bench-b.json examples/bell-photons.qoa
```

| field | what it is | how to get it for your bench |
|---|---|---|
| `detector.efficiency` | probability that a photon reaching a detector makes it click | the detector's data sheet (silicon SPADs at 810 nm: about 0.6), or calibrate against a known source |
| `detector.dark` | probability of a click with no photon, within one coincidence window | the dark-count rate times the window, e.g. 100 per second × 10 ns = 1e-6 |
| `source.heralding` | probability that the heralded path really holds the photon (coupling and filter losses) | coincidences divided by herald counts, then divided by the detector efficiency |
| `source.two_photon` | probability of two photons instead of one; it grows with pump power | from a Hanbury Brown-Twiss measurement of the heralded photons |
| `source.visibility` | quality of the entangled pairs: the state is V\|Φ+⟩⟨Φ+\| + (1 − V) I/4 | the visibility of the polarisation-correlation fringes |
| `target` | `bench-b` (default) or `bench-a` | |

Every field is optional and defaults to ideal. The effects are applied in every shot, with their exact
probabilities: a photon reaching a detector is lost with probability 1 − efficiency; a detector with no
photon clicks with probability dark; a polarisation measurement (a polarizing beamsplitter and two
detectors) whose photon was lost, or where both detectors clicked, has no result. Its bit is left
unwritten and shows as `-` in the counts: those are the shots the bench's coincidence counter throws away,
and the shots with every bit written are what it keeps.
