# QOA v0.4.7 Release Notes

**Status:** in progress, not released yet (euskaraz behean)

## Added

* **`qoa export --to glyph --keep-source file.qoa`** converts a plain QOA source to glyphs line by line,
  keeping every comment, label, blank line and the layout. Each line becomes one statement in the shortest
  form (`Hq0`, `#2<=#1+3`, `!c0>again`, `PF"x"/#1`); a line glyphs cannot say stays as plain QOA between
  backticks. The result is assembled again and must be the same program, line numbers included.
* **The assembler written in QOA, in glyphs:** [tools/asm.qg](../tools/asm.qg) is `tools/asm.qoa` converted
  that way. A test checks that both give the same program byte for byte.
* **OpenQASM 3 export: `qoa export --to qasm3`.** Counted loops stay `for` loops, a forward jump on a
  measured bit becomes an `if` block (an `if`/`else` when a `JUMP` skips a second block), and
  repeat-until-success becomes a `while` loop. Gates missing from `stdgates.inc` use modifiers (`inv @ sx`,
  `ctrl(3) @ x`). Every exportable example gives the same distribution in Qiskit Aer as in QOA
  ([docs/hardware.md](../docs/hardware.md#openqasm-3)).
* **OpenQASM 3 import:** `qoa run` and `qoa import` read bits, `if`/`else` (and `else if`), `while`, `for`
  (when the loop variable is not used inside) and the `inv`, `ctrl`, `ctrl(n)` and `pow(k)` gate modifiers.
  QOA -> OpenQASM 3 -> QOA gives the same counts shot for shot. Blocks nested more than 64 deep are an error.
* **`examples/until.qoa`:** repeat-until-success on a measured bit alone, the shape OpenQASM 3 can say.
* **`tools/run_on_ibm.py`** sends OpenQASM 3 when OpenQASM 2.0 cannot hold the program, unrolls counted
  loops, and warns before spending QPU time when the program has a `while` loop or a nested block, which
  IBM's current devices do not run (IBM deprecated them in 2025). `tools/hardware_check.py` has an OpenQASM 3
  section (`--only qasm3`).

## Changed

* **Imported `if` on one measured bit** becomes a plain bit jump (`JUMPIFZERO`/`JUMPIFONE`) when the bit has
  surely been written, instead of packing the bits into a register. The program is the same, and it can now
  be exported to OpenQASM again. A bit that may not have been written yet is still read as 0, as OpenQASM
  says.
* **Errors for OpenQASM 3 features the importer does not read** now name classical types too (`uint`,
  `float`, `bool`, ...), instead of reporting an unknown register.

## Fixed

* `tools/hardware_check.py` crashed instead of reporting a program the exporter refuses.

---

# QOA v0.4.7 bertsio-oharrak

**Egoera:** lanean, oraindik argitaratu gabe

## Gehitua

* **`qoa export --to glyph --keep-source fitxategia.qoa`**-k QOA arrunteko iturburu bat glifora bihurtzen du
  lerroz lerro, iruzkin, etiketa eta lerro huts guztiak eta antolaketa gordeta. Lerro bakoitza adierazpen bat
  bihurtzen da forma laburrenean (`Hq0`, `#2<=#1+3`, `!c0>berriz`, `PF"x"/#1`); glifoek esan ezin duten lerro bat
  QOA arrunt gisa geratzen da ` artean. Emaitza berriz mihiztatzen da eta programa bera izan behar du,
  lerro-zenbakiak barne.
* **QOAn idatzitako mihiztatzailea, glifoetan:** [tools/asm.qg](../tools/asm.qg) horrela bihurtutako
  `tools/asm.qoa` da. Proba batek egiaztatzen du biek programa bera ematen dutela byte bat aldatu gabe.
* **OpenQASM 3 esportazioa: `qoa export --to qasm3`.** Kopuru finkoko begiztak `for` begizta gisa geratzen
  dira, neurtutako bit baten araberako aurrerako jauzi bat `if` bloke bihurtzen da (`if`/`else` bat `JUMP`
  batek bigarren bloke bat saltatzen duenean), eta arrakasta lortu arte errepikatzea `while` begizta bihurtzen
  da. `stdgates.inc`-n ez dauden ateek modifikatzaileak erabiltzen dituzte (`inv @ sx`, `ctrl(3) @ x`).
  Esporta daitekeen adibide orok banaketa bera ematen du qiskit Aer-en eta QOA-n
  ([docs/hardware.eu.md](../docs/hardware.eu.md#openqasm-3)).
* **OpenQASM 3 inportazioa:** `qoa run`-ek eta `qoa import`-ek bitak, `if`/`else` (eta `else if`), `while`,
  `for` (begizta-aldagaia barruan erabiltzen ez denean) eta `inv`, `ctrl`, `ctrl(n)` eta `pow(k)`
  ate-modifikatzaileak irakurtzen dituzte. QOA -> OpenQASM 3 -> QOA joan-etorriak kopuru berak ematen ditu,
  shot-ez shot. 64 maila baino sakonago habiaratutako blokeak errore bat dira.
* **`examples/until.qoa`:** errepikatu arrakasta lortu arte bit neurtu bakar batekin, OpenQASM 3-k esan
  dezakeen forma.
* **`tools/run_on_ibm.py`**-k OpenQASM 3 bidaltzen du OpenQASM 2.0-n programa sartzen ez denean, kopuru
  finkoko begiztak zabaltzen ditu, eta QPU denbora gastatu aurretik abisatzen du programak `while` begizta
  bat edo bloke habiaratu bat duenean, IBM-ren gaurko gailuek exekutatzen ez dituztenak (IBM-k 2025ean
  zaharkitutzat jo zituen). `tools/hardware_check.py`-k OpenQASM 3 atal bat du (`--only qasm3`).

## Aldatua

* **Inportatutako `if` bat bit neurtu bakar batekin** bit-jauzi arrunt bihurtzen da (`JUMPIFZERO`/
  `JUMPIFONE`) bita ziur idatzita dagoenean, bitak erregistro batean bildu beharrean. Programa bera da, eta
  orain berriz esporta daiteke OpenQASM-era. Agian oraindik idatzi gabe dagoen bit bat 0 gisa irakurtzen da
  oraindik ere, OpenQASM-ek dioen bezala.
* **Inportatzaileak irakurtzen ez dituen OpenQASM 3 ezaugarrien erroreek** mota klasikoak ere izendatzen
  dituzte orain (`uint`, `float`, `bool`, ...), erregistro ezezagun bat salatu beharrean.

## Konpondua

* `tools/hardware_check.py` hondatu egiten zen esportatzaileak baztertzen duen programa bat salatu beharrean.
