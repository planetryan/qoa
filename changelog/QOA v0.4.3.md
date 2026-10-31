# QOA v0.4.3 Release Notes

**Status:** released 2026-10-31 (euskaraz behean)

Builds on [0.4.2](<QOA v0.4.2.md>): QOA's assembler written in QOA, which rebuilds itself identically, the
classical instructions that made it possible, and a benchmark of every part of QOA.

## New

* **QOA builds QOA.** QOA's assembler is now also written in QOA: [tools/asm.qoa](../tools/asm.qoa) runs on the
  QOA machine and turns a `.qoa` program into the same `.qx` bytes as `qoa compile`, with the same error
  messages word for word, in English and Basque. `qoa compile --assembler tools/asm.qoa program.qoa` uses it.
  It assembles itself, and the copy it builds is identical to the copy Rust builds. It is about 19 times
  slower than the Rust assembler. Tests compare the two on every program in the repository and on thousands
  of generated lines. See [docs/self-hosting.md](../docs/self-hosting.md).
* **Classical tools in QOA**: memory (`LOAD`, `STORE`, `STORETEXT`), raw bytes (`GETC`, `PUTC`), standard error
  (`STREAM 2`), program arguments (`ARGC`, `ARGBYTE`, after `qoa run prog -- a b`), the user's language
  (`GETLANG`), an exit status (`EXIT n`), and `SIN`, `COS`, `TAN`, `POW`. `qoa run --quiet` prints only the
  program's own output and exits with its status.
* **`qoa --benchmark`** (also `--bm`, `qoa benchmark`, `qoa bench`, `qoa bm`): measures every part of QOA
  on fixed problems and scores each one. The parts are memory, the assembler, the QOA machine itself
  (running the assembler written in QOA), qubit gates at 20 and 24 qubits, sampling, shots one by one,
  optical modes, 128-bit gates, exact sums, the light of `qoa visual`, Wigner panels and, when there is
  one, the GPU. 1000 is the reference machine and the total is the
  geometric mean. Each item shows its details, best and median times; `--quick`, `--json` and `--no-gpu`
  are available. See [docs/benchmark.md](../docs/benchmark.md).

## Changed

* Programs that use the new instructions are written as `.qx` format version 2. All other programs are
  still written as version 1, so QOA 0.4.2 reads them; QOA 0.4.3 reads both.
* `LOADMEM`, `STOREMEM`, `LOADCLASSICAL`, `STORECLASSICAL`, `ALLOC` and `FREE` (removed in 0.4) now point at
  `LOAD` and `STORE` in their error message. `POW`, `EXIT`, `STREAM`, `GETC` and the other new names can no
  longer name a register (`QREG exit 2`).
* For library users: `RunConfig` has `args` and `input` (code that fills it with `..RunConfig::default()` is
  unaffected), `RunOutput` has `exit_status`, `Op`, `UnOp` and `BinOp` have new variants at the end,
  `Program::FORMAT_VERSION` is 2 with `Program::format_version()` choosing per program, and `isa::Group` has
  `Memory`. `Spec::usage_in` gives the usage line in a chosen language.

## Fixed

* When a new register overlapped several existing ones, the error named one of them at random (it depended
  on hash order); it now names the first one declared.

---

# QOA v0.4.3 bertsio-oharrak

**Egoera:** 2026-10-31n argitaratua

[0.4.2](<QOA v0.4.2.md>)-ren gainean: QOAn idatzitako QOA-ren mihiztatzailea, bere burua berdin eraikitzen
duena, hori posible egin zuten agindu klasikoak, eta QOA-ren zati guztien errendimendu-proba.

## Berria

* **QOA-k QOA eraikitzen du.** QOA-ren mihiztatzailea QOAn idatzita ere badago orain:
  [tools/asm.qoa](../tools/asm.qoa) QOA makinan exekutatzen da eta `.qoa` programa bat `qoa compile`-ren `.qx`
  byte berberak bihurtzen du, errore-mezu berberekin hitzez hitz, ingelesez eta euskaraz.
  `qoa compile --assembler tools/asm.qoa programa.qoa`-k erabiltzen du. Bere burua mihiztatzen du, eta
  eraikitzen duen kopia rust-ek eraikitzen duenaren berdina da. Rust-en mihiztatzailea baino 19 aldiz
  motelagoa da gutxi gorabehera. Probek biak konparatzen dituzte biltegiko programa guztietan eta sortutako
  milaka lerrotan. Ikus [docs/self-hosting.eu.md](../docs/self-hosting.eu.md).
* **Tresna klasikoak QOAn**: memoria (`LOAD`, `STORE`, `STORETEXT`), byteak bere horretan (`GETC`, `PUTC`),
  errore-irteera (`STREAM 2`), programaren argumentuak (`ARGC`, `ARGBYTE`, `qoa run prog -- a b` ondoren),
  erabiltzailearen hizkuntza (`GETLANG`), irteera-egoera bat (`EXIT n`), eta `SIN`, `COS`, `TAN`, `POW`.
  `qoa run --quiet`-ek programaren irteera bakarrik inprimatzen du eta haren egoerarekin amaitzen da.
* **`qoa --benchmark`** (baita `--bm`, `qoa benchmark`, `qoa bench`, `qoa bm` ere): QOA-ren zati guztiak
  problema finkoetan neurtzen ditu eta bakoitzari puntuazio bat ematen dio. Zatiak dira memoria,
  mihiztatzailea, QOA makina bera (QOAn idatzitako mihiztatzailea exekutatzen), qubit-ateak 20 eta 24
  qubitetan, laginketa, shot-ak banan-banan, modu optikoak, 128 biteko ateak, batura zehatzak,
  `qoa visual`-en argia, wigner panelak eta, badago, GPUa. 1000 erreferentziako
  makina da eta guztizkoa batez besteko geometrikoa. Atal bakoitzak bere xehetasunak, denbora onena eta
  mediana erakusten ditu; `--quick`, `--json` eta `--no-gpu` daude. Ikus
  [docs/benchmark.eu.md](../docs/benchmark.eu.md).

## Aldatua

* Agindu berriak erabiltzen dituzten programak `.qx` formatuaren 2. bertsioan idazten dira. Beste programa
  guztiak 1. bertsioan idazten dira oraindik, eta QOA 0.4.2-k irakurtzen ditu; QOA 0.4.3-k biak irakurtzen ditu.
* `LOADMEM`, `STOREMEM`, `LOADCLASSICAL`, `STORECLASSICAL`, `ALLOC` eta `FREE` (0.4-n kenduak) `LOAD` eta
  `STORE`-ra bideratzen dute orain beren errore-mezuan. `POW`, `EXIT`, `STREAM`, `GETC` eta beste izen berriak
  ezin dira erregistro-izen izan (`QREG exit 2`).
* Liburutegiaren erabiltzaileentzat: `RunConfig`-ek `args` eta `input` ditu (`..RunConfig::default()`-rekin
  betetzen duen kodeari ez dio eragiten), `RunOutput`-ek `exit_status` du, `Op`, `UnOp` eta `BinOp`-ek aldaera
  berriak dituzte amaieran, `Program::FORMAT_VERSION` 2 da eta `Program::format_version()`-ek programa
  bakoitzarentzat aukeratzen du, eta `isa::Group`-ek `Memory` du. `Spec::usage_in`-ek erabilera-lerroa
  hizkuntza jakin batean ematen du.

## Konponduta

* Erregistro berri bat lehendik zeuden batzuekin gainjartzen zenean, erroreak horietako bat izendatzen zuen
  ausaz (hash-ordenaren menpe); orain adierazitako lehena izendatzen du.
