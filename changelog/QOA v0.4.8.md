# QOA v0.4.8 Release Notes

**Status:** in progress, not released yet (euskaraz behean)

## Added

* **[docs/bench.md](../docs/bench.md): a tabletop quantum-optics bench for QOA.** Three tiers (classical
  light; real single photons; number-resolving detectors and squeezing), which QOA instructions each can run,
  and what a single-photon bench needs, from a teaching kit or from parts, with the prices found in November
  2026 and their sources.
* **`qoa check --target bench-a|bench-b`** checks a program against such a bench. Each instruction the bench
  cannot run is an error with its line and the reason: simulator-only inspection, nonlinear optics, two-qubit
  gates on photons, number-resolving detection, jumps and loops (a bench runs a fixed sequence per shot),
  more photons, qubits or detectors than the bench has, a photon used after it was detected.
* **Bench models: `qoa run --noise-model bench.json`** with a file in the new `qoa-bench-1` format
  simulates a program the way a given bench would run it: detector efficiency and dark clicks, the heralded
  source's efficiency and two-photon emission, and the visibility of the entangled pairs, each applied with its
  exact probability in every shot. A polarisation measurement the bench would discard (photon lost, double
  click) leaves its bit unwritten (`-`). The program must pass the bench's profile.
  [examples/bench-b.json](../examples/bench-b.json) has illustrative numbers; docs/bench.md says how to measure
  each one on a real bench.
* **`examples/hom-clicks.qoa`** (Hong-Ou-Mandel with click detectors) and **`examples/bell-photons.qoa`**
  (a Bell test on photon polarisation, `BELL` as the pair source): both run on a single-photon bench.

## Changed

* **The parts of QOA written in QOA moved from `tools/` to `src/qoa/`** ([src/qoa/asm.qoa](../src/qoa/asm.qoa),
  [src/qoa/asm.qg](../src/qoa/asm.qg)): they are compiled into the `qoa` binary since 0.4.7, so they belong
  with the Rust sources. `tools/` keeps the helper scripts (Python, shell). Release packages carry them at the
  same path, `src/qoa/`, so `qoa compile --assembler src/qoa/asm.qoa` works in the repository and in a
  package alike.

---

# QOA v0.4.8 bertsio-oharrak

**Egoera:** lanean, oraindik argitaratu gabe

## Gehitua

* **[docs/bench.eu.md](../docs/bench.eu.md): mahai gaineko optika kuantikoko banku bat QOA-rentzat.** Hiru
  maila (argi klasikoa; fotoi bakar errealak; kopurua bereizten duten detektagailuak eta estutzea), maila
  bakoitzak zer QOA agindu exekuta ditzakeen, eta fotoi bakarreko banku batek zer behar duen, irakaskuntza-kit
  batetik edo piezaz pieza, 2026ko azaroan aurkitutako prezioekin eta haien iturriekin.
* **`qoa check --target bench-a|bench-b`**-k programa bat banku horren aurka egiaztatzen du. Bankuak exekutatu
  ezin duen agindu bakoitza errore bat da, bere lerroarekin eta arrazoiarekin: simulagailuko ikuskapena,
  optika ez-lineala, bi qubiteko ateak fotoiekin, kopurua bereizten duen detekzioa, jauziak eta begiztak
  (banku batek sekuentzia finko bat exekutatzen du shot bakoitzean), bankuak dituenak baino fotoi, qubit edo
  detektagailu gehiago, detektatu ondoren erabilitako fotoi bat.
* **Bench models: `qoa run --noise-model bench.json`** with a file in the new `qoa-bench-1` format
  simulates a program the way a given bench would run it: detector efficiency and dark clicks, the heralded
  source's efficiency and two-photon emission, and the visibility of the entangled pairs, each applied with its
  exact probability in every shot. A polarisation measurement the bench would discard (photon lost, double
  click) leaves its bit unwritten (`-`). The program must pass the bench's profile.
  [examples/bench-b.json](../examples/bench-b.json) has illustrative numbers; docs/bench.md says how to measure
  each one on a real bench.
* **Banku-ereduak: `qoa run --noise-model bankua.json`**-k, `qoa-bench-1` formatu berriko fitxategi batekin,
  programa bat banku jakin batek exekutatuko lukeen bezala simulatzen du: detektagailuen eraginkortasuna eta
  iluneko klikak, iturri iragarriaren eraginkortasuna eta bi fotoiko igorpena, eta bikote entrelazatuen
  ikusgarritasuna, bakoitza bere probabilitate zehatzarekin shot bakoitzean. Bankuak baztertuko lukeen
  polarizazio-neurketa batek (fotoia galduta, klik bikoitza) bita idatzi gabe uzten du (`-`). Programak
  bankuaren profila bete behar du. [examples/bench-b.json](../examples/bench-b.json)-k zenbaki ilustratiboak
  ditu; docs/bench.eu.md-k esaten du nola neurtu bakoitza benetako banku batean.
* **`examples/hom-clicks.qoa`** (Hong-Ou-Mandel klik-detektagailuekin) eta **`examples/bell-photons.qoa`**
  (Bell proba bat fotoien polarizazioarekin, `BELL` bikote-iturri gisa): biak fotoi bakarreko banku batean
  dabiltza.

## Aldatua

* **QOAn idatzitako QOA-ren zatiak `tools/`-etik `src/qoa/`-ra eraman dira**
  ([src/qoa/asm.qoa](../src/qoa/asm.qoa), [src/qoa/asm.qg](../src/qoa/asm.qg)): 0.4.7-tik `qoa` binarioan
  konpilatzen dira, beraz rust-en iturburuekin doaz. `tools/`-ek laguntza-scriptak gordetzen ditu (python,
  shell). Argitalpen-paketeek bide berean dakartzate, `src/qoa/`, beraz `qoa compile --assembler
  src/qoa/asm.qoa` berdin dabil biltegian eta pakete batean.
