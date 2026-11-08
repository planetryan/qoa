# QOA v0.4.9 Release Notes

**Status:** in progress, not released yet (euskaraz behean)

## Added

* **The disassembler written in QOA:** [src/qoa/dis.qoa](../src/qoa/dis.qoa) reads a `.qx` and writes the
  plain QOA that `qoa export --to qoa --long-names` writes, byte for byte, and gives a damaged file the same
  error in both languages. It checks the header, version and SHA-256, decodes all 80 kinds of operation and
  rebuilds every 64-bit number exactly. [src/qoa/dis.qg](../src/qoa/dis.qg) is the same program in glyphs.
* **`qoa export --to qoa --long-names --self-hosted FILE`** uses it, built into the binary. With a source
  file, the built-in assembler written in QOA assembles it first, so text to binary to text runs in QOA.
  `qoa tools` lists it, `qoa tools --write DIR` writes it out, and release packages carry it in `src/qoa/`.
  The tests compare it with Rust on every program in the repository and on damaged files of every kind.

## Changed

* **The tools written in QOA are built into the binary as bytecode:** [src/qoa/asm.qx](../src/qoa/asm.qx)
  and [src/qoa/dis.qx](../src/qoa/dis.qx), built from their sources by the assembler written in QOA. `qoa`
  runs that bytecode directly, so the Rust assembler plays no part when `--self-hosted` assembles or
  disassembles. The tests check that each `.qx` is exactly what both assemblers build from its source, and
  that the assembler and the disassembler reproduce their own bytecode and text unchanged. `qoa tools --write`
  writes the `.qx` files too, and release packages carry them.

---

# QOA v0.4.9 bertsio-oharrak

**Egoera:** lanean, oraindik argitaratu gabe

## Gehitua

* **QOAn idatzitako desmihiztatzailea:** [src/qoa/dis.qoa](../src/qoa/dis.qoa)-k `.qx` bat irakurtzen du eta
  `qoa export --to qoa --long-names`-ek idazten duen QOA arrunta idazten du, bytez byte, eta fitxategi hondatu
  bati errore bera ematen dio bi hizkuntzetan. Goiburua, bertsioa eta SHA-256-a egiaztatzen ditu, 80 eragiketa
  mota guztiak deskodetzen ditu eta 64 biteko zenbaki bakoitza zehazki berreraikitzen du.
  [src/qoa/dis.qg](../src/qoa/dis.qg) programa bera da glifoetan.
* **`qoa export --to qoa --long-names --self-hosted FITXATEGIA`**-k hura erabiltzen du, binarioan sartuta.
  Iturburu-fitxategi batekin, QOAn idatzitako mihiztatzaile integratuak mihiztatzen du lehenik, beraz
  testutik bitarrera eta testura QOAn exekutatzen da. `qoa tools`-ek zerrendatzen du, `qoa tools --write
  KARPETA`-k idazten du, eta argitalpen-paketeek `src/qoa/`-n dakarte. Probek rust-ekin konparatzen dute
  biltegiko programa guztietan eta mota guztietako fitxategi hondatuetan.

## Aldatua

* **QOAn idatzitako tresnak bytecode gisa daude binarioan:** [src/qoa/asm.qx](../src/qoa/asm.qx) eta
  [src/qoa/dis.qx](../src/qoa/dis.qx), QOAn idatzitako mihiztatzaileak beren iturburuetatik eraikiak. `qoa`-k
  bytecode hori zuzenean exekutatzen du, beraz rust-en mihiztatzaileak ez du parterik `--self-hosted`-ek
  mihiztatzen edo desmihiztatzen duenean. Probek egiaztatzen dute `.qx` bakoitza bi mihiztatzaileek bere
  iturburutik eraikitzen dutena dela zehazki, eta mihiztatzaileak eta desmihiztatzaileak beren bytecode-a eta
  testua aldaketarik gabe berreraikitzen dituztela. `qoa tools --write`-k `.qx` fitxategiak ere idazten ditu,
  eta argitalpen-paketeek badakartzate.
