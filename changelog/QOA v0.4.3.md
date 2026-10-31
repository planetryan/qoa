# QOA v0.4.3 Release Notes

**Status:** in progress, not released yet (euskaraz behean)

## New

* **`qoa --benchmark`** (also `--bm`, `qoa benchmark`, `qoa bench`, `qoa bm`): measures every part of QOA
  on fixed problems and scores each one. The parts are memory, the assembler, qubit gates at 20 and 24
  qubits, sampling, shots one by one, optical modes, 128-bit gates, exact sums, the light of `qoa visual`,
  Wigner panels and, when there is one, the GPU. 1000 is the reference machine and the total is the
  geometric mean. Each item shows its details, best and median times; `--quick`, `--json` and `--no-gpu`
  are available. See [docs/benchmark.md](../docs/benchmark.md).

---

# QOA v0.4.3 bertsio-oharrak

**Egoera:** lanean, oraindik argitaratu gabe

## Berria

* **`qoa --benchmark`** (baita `--bm`, `qoa benchmark`, `qoa bench`, `qoa bm` ere): QOA-ren zati guztiak
  problema finkoetan neurtzen ditu eta bakoitzari puntuazio bat ematen dio. Zatiak dira memoria,
  mihiztatzailea, qubit-ateak 20 eta 24 qubitetan, laginketa, shot-ak banan-banan, modu optikoak, 128
  biteko ateak, batura zehatzak, `qoa visual`-en argia, wigner panelak eta, badago, GPUa. 1000 erreferentziako
  makina da eta guztizkoa batez besteko geometrikoa. Atal bakoitzak bere xehetasunak, denbora onena eta
  mediana erakusten ditu; `--quick`, `--json` eta `--no-gpu` daude. Ikus
  [docs/benchmark.eu.md](../docs/benchmark.eu.md).
