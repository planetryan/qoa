# QOA v0.4.1 Release Notes

**Status:** released 2026-10-30 (euskaraz behean)

Builds on [0.4.0](<QOA v0.4.0.md>): real-device noise, exact sums, and two more systems.

## New

* **Device noise models**: `qoa run --noise-model FILE` runs a program with a real device's calibrated
  noise. After every native gate it applies that gate's measured error and its T1/T2 relaxation over the
  gate's duration, and each measurement gets that qubit's readout error.
  * It is the same model Qiskit Aer builds with `NoiseModel.from_backend`.
  * Each channel is simulated as an exact quantum trajectory, so the shots follow the channel's
    distribution with no approximation.
  * It matches Aer's exact density matrix on Grover, QPE and GHZ transpiled for IBM Torino (χ² p-values
    0.52, 0.92, 0.37 over 400,000 shots).
  * A gate the device does not have is an error, never a silent ideal gate.
  * See [docs/noise.md](../docs/noise.md).
* **Predictions for real devices**: `tools/run_on_ibm.py` builds the model from the device's calibration
  and adds a "QOA + model" column next to the device's counts.
  * `--save-model` keeps the model and the transpiled circuit.
  * `examples/bell-torino.qasm` with `examples/bell-torino.json` is a ready example.

* **More systems** (all 64-bit), from 7 builds to 13 (15 downloads with the all-platform bundle and checksums):
  * IBM POWER8 and newer (`powerpc64le-unknown-linux-gnu`), including POWER10 and POWER11.
  * IBM Z (`s390x-unknown-linux-gnu`), the first big-endian build.
  * LoongArch 64 (`loongarch64-unknown-linux-gnu`).
  * Static builds for any Linux, with no glibc needed (`x86_64`, `aarch64` and `riscv64gc` with musl). The
    ARM64 one runs on Android under Termux.
  * POWER, IBM Z and static ARM64 run the core test suites under emulation in CI. All give bit-identical
    results to x86 for the same seed.

## Changed

* **Exact sums over the state**:
  * Norms, measurement probabilities, the sampling distribution and expectation values are summed in
    double-double arithmetic (~106 bits, error-free transformations with FMA where the CPU has it).
  * They are correct to a 64-bit float's precision at any state size.
  * They give the same bits whatever the thread count; plain parallel sums differed in their last digits
    from run to run.
  * The report's new `numerics` line (and `norm_error` in json) shows how far the final state's norm has
    drifted from 1.
* `qoa export --help` lists every export format.
* The documentation is written for any reader: the glyph syntax is described as experimental with its open
  points, the use cases end with a public roadmap, and [CONTRIBUTING.md](../CONTRIBUTING.md) sets out how
  QOA is built, checked, written and released.
* JSON numbers are read with exact round-tripping.

---

# QOA v0.4.1 bertsio-oharrak

**Egoera:** 2026-10-30ean argitaratua

[0.4.0](<QOA v0.4.0.md>)-ren gainean: gailu errealen zarata, batura zehatzak eta bi sistema gehiago.

## Berria

* **Gailuen zarata-ereduak**: `qoa run --noise-model FITXATEGIA`-k programa bat benetako gailu baten zarata
  kalibratuarekin exekutatzen du. Ate natibo bakoitzaren ondoren ate horren errore neurtua eta T1/T2
  erlaxazioa aplikatzen ditu atearen iraupenean, eta neurketa bakoitzak qubit horren irakurketa-errorea du.
  * Qiskit aer-ek `NoiseModel.from_backend`-ekin eraikitzen duen eredu bera da.
  * Kanal bakoitza ibilbide kuantiko zehatz gisa simulatzen da, beraz shot-ek kanalaren banaketa
    jarraitzen dute, hurbilketarik gabe.
  * Aer-en dentsitate-matrize zehatzarekin bat dator Grover, QPE eta GHZ IBM Torino-rako transpilatuta
    (χ² p-balioak 0,52, 0,92, 0,37, 400.000 shot-ekin).
  * Gailuak ez duen ate bat errorea da, inoiz ez ate ideal isil bat.
  * Ikus [docs/noise.eu.md](../docs/noise.eu.md).
* **Gailu errealetarako aurreikuspenak**: `tools/run_on_ibm.py`-k eredua gailuaren kalibraziotik eraikitzen
  du eta "QOA + eredua" zutabe bat gehitzen du gailuaren kopuruen ondoan.
  * `--save-model`-ek eredua eta zirkuitu transpilatua gordetzen ditu.
  * `examples/bell-torino.qasm` `examples/bell-torino.json`-ekin adibide prest bat da.

* **Sistema gehiago** (denak 64 bitekoak), 7 bitarretik 13ra (15 deskarga, plataforma guztien paketea eta laburpenak barne):
  * IBM POWER8 eta berriagoak (`powerpc64le-unknown-linux-gnu`), POWER10 eta POWER11 barne.
  * IBM Z (`s390x-unknown-linux-gnu`), lehen big-endian bitarra.
  * LoongArch 64 (`loongarch64-unknown-linux-gnu`).
  * Edozein linux-erako bitar estatikoak, glibc-rik gabe (`x86_64`, `aarch64` eta `riscv64gc` musl-ekin).
    ARM64-ekoa Android-en ibiltzen da Termux-en.
  * POWER, IBM Z eta ARM64 estatikoak oinarrizko probak emulazioan exekutatzen dituzte CI-n. Denek x86-ren
    emaitza berberak ematen dituzte bitez bit hazi berarekin.

## Aldatua

* **Egoeraren gaineko batura zehatzak**:
  * Arauak, neurketa-probabilitateak, laginketa-banaketa eta itxarondako balioak double-double
    aritmetikan batzen dira (~106 bit, errorerik gabeko transformazioak, prozesadoreak FMA badu harekin).
  * 64 biteko koma higikorraren zehaztasunera arte zuzenak dira egoeraren tamaina edozein dela ere.
  * Bit berberak ematen dituzte hari-kopurua edozein dela ere; batura paralelo arruntak azken digituetan
    aldatzen ziren exekuzio batetik bestera.
  * Txostenaren `numerics` lerro berriak (eta json-eko `norm_error`-ek) amaierako egoeraren araua 1etik
    zenbat desbideratu den erakusten du.
* `qoa export --help`-ek esportazio-formatu guztiak zerrendatzen ditu.
* Dokumentazioa edozein irakurlerentzat idatzita dago: glifo-sintaxia esperimental gisa deskribatzen da bere
  puntu irekiekin, erabilera-kasuak ibilbide-orri publiko batekin amaitzen dira, eta
  [CONTRIBUTING.eu.md](../CONTRIBUTING.eu.md)-k QOA nola eraiki, egiaztatu, idatzi eta argitaratzen den
  azaltzen du.
* JSON zenbakiak zehaztasun osoz irakurtzen dira.
