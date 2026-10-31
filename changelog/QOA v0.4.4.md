# QOA v0.4.4 Release Notes

**Status:** in progress, not released yet (euskaraz behean)

## Added

* **Numbers where a value goes.** An operand that reads a value (the inputs of `ADD`, `SUB`, `MUL`, `DIV`,
  the bit operations, `SQRT`, `EXP`, `LOG`, `SIN`, `COS`, `TAN`, `POW`, `REGCOPY`, the compare-and-jump
  instructions `IFEQ` `IFNE` `IFGT` `IFLT`, `LOAD`, `STORE`, `PUSH`, `PUTC`, `CHAROUT` and `ARGBYTE`) can be a
  number in parentheses instead of a register: `ADD R1 R1 (1)`, `IFLT R2 (10) loop`, `CHAROUT (65)`,
  `STORE (0.5) (100)`. Any expression works inside: `(pi/2)`, `(2^10)`. A bare number still means a register
  (`ADD R1 R1 1` adds R1), so every existing program keeps its meaning and its bytes.
* **`CONST name value`** names a number for the whole program: after `CONST newline 10`, `newline` works in
  every expression, as in `CHAROUT (newline)` or `RX q0 newline/4`. A constant can use the constants
  defined above it.
* **`RREG name size [first]`** names classical registers like `QREG` names qubits: after `RREG acc 2`,
  `acc[0]` and `acc[1]` are the next two unnamed registers, and `RREG i 1 8` makes `i` stand for R8. R0..R31
  keep working.
* In the **glyph syntax** a number in a value slot is that number: `ADD/#1,#1,1` adds one, and
  `IFEQ/#1,10,done` compares R1 with ten. Where only a qubit, a mode, a bit or a register to write to fits,
  a bare number is still an error.
* The assembler written in QOA (`tools/asm.qoa`) understands all of this and still produces exactly the bytes
  and messages of `qoa compile`. The test suite checks thousands of generated lines with the new forms.

## Changed

* A program that uses a number in parentheses is written as `.qx` format version 3. Programs without them
  stay version 1 or 2, so QOA 0.4.2 and 0.4.3 still run them; a version-3 file given to QOA 0.4.3 is reported
  as needing a newer QOA instead of as a damaged file.
* Messages about named classical registers write them as registers: `acc (R4..R5)`, not `acc (r4..r5)`.
* Classical instructions run about 10% faster than in QOA 0.4.3 (measured with the assembler written in QOA
  on its own source).
* Every release package and the all-platform bundle include `tools/asm.qoa`, QOA's assembler written in QOA,
  so `qoa compile --assembler tools/asm.qoa program.qoa` works with a downloaded binary alone. The release
  build checks on every system it can run that the packaged binary rebuilds the assembler identically.
* The "restore release" workflow republishes every release whose tag was deleted (GitHub turns such a
  release into a draft and lists it first), keeping the newest version as the latest release.

---

# QOA v0.4.4 bertsio-oharrak

**Egoera:** lanean, oraindik argitaratu gabe

## Gehitua

* **Zenbakiak balio bat doan lekuan.** Balio bat irakurtzen duen eragigai bat (`ADD`, `SUB`, `MUL`, `DIV`,
  bit-eragiketen, `SQRT`, `EXP`, `LOG`, `SIN`, `COS`, `TAN`, `POW` eta `REGCOPY`-ren sarrerak, `IFEQ` `IFNE`
  `IFGT` `IFLT` konparatu-eta-jauzi aginduak, `LOAD`, `STORE`, `PUSH`, `PUTC`, `CHAROUT` eta `ARGBYTE`)
  parentesi arteko zenbaki bat izan daiteke erregistro baten ordez: `ADD R1 R1 (1)`, `IFLT R2 (10) loop`,
  `CHAROUT (65)`, `STORE (0.5) (100)`. Barruan edozein adierazpenek balio du: `(pi/2)`, `(2^10)`. Zenbaki huts
  batek erregistro bat esan nahi du oraindik (`ADD R1 R1 1`-ek R1 gehitzen du), beraz lehendik dagoen
  programa orok bere esanahia eta bere byteak gordetzen ditu.
* **`CONST izena balioa`**-k zenbaki bati izena ematen dio programa osorako: `CONST newline 10` ondoren,
  `newline`-k edozein adierazpenetan balio du, `CHAROUT (newline)` edo `RX q0 newline/4` bezala. Konstante
  batek bere gainetik definitutakoak erabil ditzake.
* **`RREG izena tamaina [lehena]`**-k erregistro klasikoei izena ematen die, `QREG`-ek qubitei bezala:
  `RREG acc 2` ondoren, `acc[0]` eta `acc[1]` izenik gabeko hurrengo bi erregistroak dira, eta `RREG i 1 8`-k
  `i` R8 bihurtzen du. R0..R31-ek ere balio dute.
* **Glifo-sintaxian** balio-leku bateko zenbaki bat zenbaki hori da: `ADD/#1,#1,1`-ek bat gehitzen du, eta
  `IFEQ/#1,10,done`-k R1 hamarrekin konparatzen du. Qubit bat, modu bat, bit bat edo idazteko erregistro bat
  bakarrik egokitzen den lekuan, zenbaki huts bat errorea da oraindik.
* QOAn idatzitako mihiztatzaileak (`tools/asm.qoa`) hau guztia ulertzen du, eta `qoa compile`-ren byte eta
  mezu berberak sortzen ditu oraindik. Proba-multzoak forma berriak dituzten milaka lerro sortu egiaztatzen
  ditu.

## Aldatua

* Parentesi arteko zenbaki bat erabiltzen duen programa `.qx` formatuaren 3. bertsioan idazten da. Halakorik
  ez dutenak 1. edo 2. bertsioan geratzen dira, eta QOA 0.4.2-k eta 0.4.3-k exekutatzen dituzte oraindik;
  3. bertsioko fitxategi bat QOA 0.4.3-ri emanez gero, QOA berriago bat behar duela esaten du, fitxategi
  hondatu bat dela esan beharrean.
* Izendun erregistro klasikoei buruzko mezuek erregistro gisa idazten dituzte: `acc (R4..R5)`, ez
  `acc (r4..r5)`.
* Agindu klasikoak QOA 0.4.3-n baino % 10 inguru azkarrago dabiltza (QOAn idatzitako mihiztatzailearekin
  neurtuta, bere iturburuaren gainean).
* Argitalpen-pakete guztiek eta plataforma guztietako sortak `tools/asm.qoa` dakarte, QOAn idatzitako
  QOA-ren mihiztatzailea, `qoa compile --assembler tools/asm.qoa programa.qoa` deskargatutako bitar
  batekin bakarrik ibil dadin. Argitalpenaren eraikitzeak egiaztatzen du, exekuta dezakeen sistema
  guztietan, paketeko bitarrak mihiztatzailea berdin eraikitzen duela.
* "restore release" workflow-ak etiketa ezabatu zaien argitalpenak berriz argitaratzen ditu (GitHub-ek
  zirriborro bihurtzen ditu, eta lehenak erakusten ditu), bertsio berriena azken argitalpen gisa utzita.
