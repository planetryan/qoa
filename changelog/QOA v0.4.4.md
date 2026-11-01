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

## Safer

QOA now treats every input as possibly hostile, and the test suite proves it: on every build it feeds QOA
20,000 damaged `.qx` files and 10,000 garbled `.qoa`, `.qg` and `.qasm` programs, plus nesting 200,000 levels
deep, and requires a clear error (or a normal run) every time, never a crash. That search found and fixed:

* **Damaged `.qx` files are rejected.** A loaded program must be exactly what the assembler builds from its
  own text, checked operation by operation, so the machine never sees a register, qubit, mode, bit or jump
  target that does not exist, a broken loop, or an absurd size. Before, changing one byte of a `.qx` could
  make QOA try to allocate gigabytes and abort.
* **Optical operators check their memory before they are built.** A beam splitter on two modes with a cutoff
  of 200 needs about 80 GiB; QOA now says so instead of crashing. The cache of operator matrices is bounded,
  so a loop that displaces by a changing register value no longer fills memory.
* **Light that leaves the cutoff completely is an error.** Displacing a mode far beyond its photon cutoff
  (or adding a photon to a mode at its top level) used to leave an empty state and a result full of NaN; now
  QOA stops and says to raise the cutoff. Displacement and squeezing amounts from registers must be finite.
* **Large cutoffs are fast.** Building the displacement operator took time proportional to the cube of the
  cutoff instead of the square, and so did photon loss; reading a mode's photon distribution took the square
  of the cutoff with a single mode. A cutoff of 3,030 went from 159 s to 2 s, and a single-mode cutoff of
  half a million from more than a minute to under a second. The results are bit for bit the same.
* **Deep nesting is an error, not a stack overflow.** An expression may nest up to 100 levels (parentheses,
  functions, signs, `^` chains); deeper ones get a clear message in both assemblers, where 200,000 levels
  used to crash QOA.
* Glyph registers are limited to 4,096 elements like `QREG` (a 34-billion-bit `#C{...}` used to abort), and
  the messages for register sizes say the allowed range (1 to 4096, first index 0 to 4095).
* An OpenQASM gate definition with a `)` before its `(` or a `}` before its `{` is an error instead of a crash.
* `qoa ... | head` ends quietly like any command-line tool (status 141), and an output that cannot be written,
  such as a full disk, stops the program with a clear message instead of losing the output silently.
* **The x86 and ARM vector kernels are safe Rust.** AVX-512, AVX2 and NEON no longer use hand-written
  assembly with raw pointers: they are vector intrinsics on bounds-checked slices. They run the same
  instructions in the same order, so results are bit-for-bit the same (checked against the old assembly on
  x86 and, under qemu, on ARM), and at the same speed. The only `unsafe` left there is calling a kernel built
  for a processor feature after checking the processor has it, and loading or storing one vector through a
  reference that covers exactly its memory. RISC-V's RVV kernels stay in assembly, because stable Rust has
  no RVV support yet. `unsafe` is now forbidden in the whole crate except in the few modules that need it,
  each saying why, and in the `qoa` command-line tool it is forbidden outright.

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

## Seguruagoa

QOA-k sarrera guztiak gaiztoak izan daitezkeela kontuan hartzen du orain, eta proba-multzoak frogatzen du:
eraikitze bakoitzean 20.000 `.qx` fitxategi hondatu eta 10.000 `.qoa`, `.qg` eta `.qasm` programa nahasi
ematen dizkio, gehi 200.000 mailako habiaraketak, eta errore argi bat (edo exekuzio arrunt bat) eskatzen du
beti, inoiz ez huts egite bat. Bilaketa horrek hauek aurkitu eta konpondu zituen:

* **`.qx` fitxategi hondatuak baztertzen dira.** Kargatutako programa batek mihiztatzaileak bere testutik
  eraikitzen duena izan behar du zehazki, eragiketaz eragiketa egiaztatuta, eta beraz makinak ez du inoiz
  ikusten existitzen ez den erregistro, qubit, modu, bit edo jauzi-helbururik, begizta hautsirik edo neurri
  zentzugaberik. Lehen, `.qx` baten byte bakar bat aldatzeak QOA-k gigabyteak erreserbatzen saiatzea eta
  bertan behera uztea ekar zezakeen.
* **Eragile optikoek beren memoria egiaztatzen dute eraiki aurretik.** 200eko mugako bi modutako
  izpi-zatitzaile batek 80 GiB inguru behar ditu; QOA-k hala esaten du orain, huts egin beharrean.
  Eragile-matrizeen cache-a mugatuta dago, eta beraz erregistro aldakor batekin desplazatzen duen begizta
  batek ez du memoria betetzen.
* **Mugatik guztiz ateratzen den argia errorea da.** Modu bat bere fotoi-mugatik oso urrun desplazatzeak (edo
  goiko mailan dagoen modu bati fotoi bat gehitzeak) egoera huts bat eta NaN-ez betetako emaitza bat uzten
  zituen; orain QOA gelditu egiten da eta muga handitzeko esaten du. Erregistroetatik datozen desplazamendu
  eta estutze kopuruek finituak izan behar dute.
* **Muga handiak azkarrak dira.** Desplazamendu-eragilea eraikitzeak mugaren kuboaren araberako denbora
  behar zuen karratuarena beharrean, eta fotoi-galerak ere bai; modu baten fotoi-banaketa irakurtzeak
  mugaren karratua behar zuen modu bakarrarekin. 3.030eko muga bat 159 s-tik 2 s-ra pasatu zen, eta modu
  bakarreko milioi erdiko muga bat minutu bat baino gehiagotik segundo bat baino gutxiagora. Emaitzak bitez
  bit berdinak dira.
* **Habiaraketa sakona errorea da, ez pila-gainezkatze bat.** Adierazpen batek 100 maila izan ditzake
  (parentesiak, funtzioak, zeinuak, `^` kateak); sakonagoek mezu argi bat dute bi mihiztatzaileetan, eta
  200.000 mailak QOA erorarazten zuen lehen.
* Glifo-erregistroak 4.096 elementura mugatuta daude, `QREG` bezala (34.000 milioi biteko `#C{...}` batek
  bertan behera uzten zuen), eta erregistro-tamainen mezuek onartutako tartea esaten dute (1etik 4096ra,
  lehen indizea 0tik 4095era).
* `(` baino lehen `)` bat edo `{` baino lehen `}` bat duen OpenQASM ate-definizio bat errorea da, huts egite
  bat izan beharrean.
* `qoa ... | head` isilik amaitzen da edozein komando-lerroko tresna bezala (141 egoera), eta idatzi ezin
  den irteera batek (disko betea, adibidez) programa gelditzen du mezu argi batekin, irteera isilik galdu
  beharrean.
* **x86 eta ARM bektore-kernelak rust seguruak dira.** AVX-512, AVX2 eta NEON-ek ez dute gehiago erakusle
  gordinekin eskuz idatzitako mihiztadurarik erabiltzen: bektore-funtzio intrintsekoak dira, mugak
  egiaztatutako slice-en gainean. Agindu berberak exekutatzen dituzte ordena berean, beraz emaitzak bitez bit
  berdinak dira (mihiztadura zaharraren aurka egiaztatuta x86-n eta, qemu bidez, ARM-n), eta abiadura berean.
  Han geratzen den `unsafe` bakarra prozesadorearen ezaugarri baterako eraikitako kernel bati deitzea da,
  prozesadoreak ezaugarri hori duela egiaztatu ondoren, eta bektore bat kargatzea edo gordetzea haren
  memoria zehazki hartzen duen erreferentzia baten bidez. RISC-V-ren RVV kernelak mihiztaduran geratzen dira,
  rust egonkorrak oraindik ez baitu RVV-rik onartzen. `unsafe` debekatuta dago orain crate osoan, behar
  duten modulu gutxietan izan ezik (bakoitzak zergatik esaten du), eta `qoa` komando-lerroko tresnan
  erabat debekatuta dago.

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
