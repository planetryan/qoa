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

* **Compiling in stages, down to machine code: `qoa compile --emit qoa|qx|ir|asm|obj|so`.** Stop at any
  stage and write it, for analysis or debugging: canonical QOA text, the bytecode, an IR of basic blocks
  (each operation native or through QOA's machine), assembly for six architectures (`--arch x86_64`,
  `aarch64`, `riscv64`, `powerpc64le`, `s390x`, `loongarch64`), an object file, or a shared library.
  `qoa run program.so` runs the library: `SET`, `COPY`, `ADD`, `SUB`, `MUL` and `JUMP` are machine
  instructions, every other operation goes through QOA's machine, and the run is exactly the interpreter's
  (same counts for the same seed, same registers, output and errors). The tests check this on every example
  and test program and on QOA's own assembler written in QOA, and CI runs them on all six architectures,
  LoongArch now included. [docs/compile.md](../docs/compile.md) explains each stage and the calling
  convention.

## Changed

* **No fixed limit on how many qubits you simulate.** `qoa run` no longer refuses a state larger than the
  machine's memory: it warns first, saying how many qubits run well on this machine, then goes ahead, and
  the run may be very slow or be stopped by the system for lack of memory. Memory the system refuses is a
  clear error, not a crash. `qoa check` always says how many qubits run well here. `QOA_MAX_MEMORY` is now a
  firm limit you choose (a state above it is refused before anything is allocated) instead of a way to raise
  the old one. Only a state no 64-bit computer can address (from about 59 qubits on one machine) is refused
  outright, and such a program can still be checked and exported to hardware. README, docs/hpc.md and
  docs/safety.md explain it, with a table of memory against qubits.
* **Programs can have up to 65,536 qubits and 65,536 classical bits** (q0..q65535, c0..c65535), instead of 64
  qubits and 4,096 bits, so every qubit can still be measured into its own bit and programs for the largest
  quantum computers can be written, checked and exported. Named registers hold up to 65,536 elements. Both
  assemblers and the glyph syntax have the same limits and messages.
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

* **Urratsez urrats konpilatzea, makina-koderaino: `qoa compile --emit qoa|qx|ir|asm|obj|so`.** Edozein
  urratsetan gelditu eta hura idatzi, aztertzeko edo arazteko: QOA testu kanonikoa, bytecode-a, oinarrizko
  blokeen IR bat (eragiketa bakoitza natiboa ala QOA-ren makinaren bidez), sei arkitekturaren mihiztadura
  (`--arch x86_64`, `aarch64`, `riscv64`, `powerpc64le`, `s390x`, `loongarch64`), objektu-fitxategi bat, edo
  liburutegi partekatu bat. `qoa run programa.so`-k liburutegia exekutatzen du: `SET`, `COPY`, `ADD`, `SUB`,
  `MUL` eta `JUMP` makina-aginduak dira, gainerako eragiketa guztiak QOA-ren makinaren bidez doaz, eta
  exekuzioa interpretatzailearena da zehazki (kontaketa berberak hazi berarekin, erregistro, irteera eta
  errore berberak). Probek hau egiaztatzen dute adibide eta proba-programa guztietan eta QOAn idatzitako
  QOA-ren mihiztatzailean, eta ci-ak sei arkitekturetan exekutatzen ditu, LoongArch barne orain.
  [docs/compile.eu.md](../docs/compile.eu.md)-k urrats bakoitza eta dei-hitzarmena azaltzen ditu.

## Aldatua

* **Ez dago muga finkorik zenbat qubit simulatzen dituzun.** `qoa run`-ek jada ez du baztertzen makinaren
  memoria baino handiagoa den egoera bat: lehenik abisatzen du, makina honetan zenbat qubit dabiltzan ondo
  esanez, eta gero aurrera egiten du, eta exekuzioa oso motela izan daiteke edo sistemak memoria faltagatik
  geldiaraz dezake. Sistemak ukatzen duen memoria errore argi bat da, ez erortze bat. `qoa check`-ek beti
  esaten du zenbat qubit dabiltzan ondo hemen. `QOA_MAX_MEMORY` zuk aukeratzen duzun muga zorrotza da orain
  (horren gaineko egoera bat baztertu egiten da ezer erreserbatu aurretik), ez muga zaharra handitzeko modu
  bat. 64 biteko inongo ordenagailuk helbideratu ezin duen egoera bat bakarrik baztertzen da zuzenean (makina
  bakarrean 59 qubit ingurutik aurrera), eta programa hori egiaztatu eta hardwarera esporta daiteke hala ere.
  README-ak, docs/hpc.eu.md-k eta docs/safety.eu.md-k azaltzen dute, memoria eta qubitak alderatzen dituen
  taula batekin.
* **Programek 65.536 qubit eta 65.536 bit klasiko izan ditzakete** (q0..q65535, c0..c65535), 64 qubit eta
  4.096 biten ordez, beraz qubit bakoitza bere bitean neur daiteke oraindik eta ordenagailu kuantiko
  handienetarako programak idatzi, egiaztatu eta esporta daitezke. Izendun erregistroek 65.536 elementu arte
  dituzte. Bi mihiztatzaileek eta glifo-sintaxiak muga eta mezu berberak dituzte.
* **QOAn idatzitako tresnak bytecode gisa daude binarioan:** [src/qoa/asm.qx](../src/qoa/asm.qx) eta
  [src/qoa/dis.qx](../src/qoa/dis.qx), QOAn idatzitako mihiztatzaileak beren iturburuetatik eraikiak. `qoa`-k
  bytecode hori zuzenean exekutatzen du, beraz rust-en mihiztatzaileak ez du parterik `--self-hosted`-ek
  mihiztatzen edo desmihiztatzen duenean. Probek egiaztatzen dute `.qx` bakoitza bi mihiztatzaileek bere
  iturburutik eraikitzen dutena dela zehazki, eta mihiztatzaileak eta desmihiztatzaileak beren bytecode-a eta
  testua aldaketarik gabe berreraikitzen dituztela. `qoa tools --write`-k `.qx` fitxategiak ere idazten ditu,
  eta argitalpen-paketeek badakartzate.
