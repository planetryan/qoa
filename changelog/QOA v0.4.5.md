# QOA v0.4.5 Release Notes

**Status:** in progress, not released yet (euskaraz behean)

## Added

* **Every compiled file carries its own SHA-256.** `.qx` files are now format version 4: the same program
  bytes as before, followed by the SHA-256 of everything before them. QOA checks it before reading anything,
  so a file changed by even one bit (a bad disk, a cut download, a hand edit of an angle or a qubit number) is
  refused with a message that says so, instead of running as a different but still valid program. The check
  works with ordinary tools too: `head -c -32 f.qx | sha256sum` and `tail -c 32 f.qx | od -An -tx1` print
  the same digits. Files written by QOA 0.4.2 to 0.4.4 still load.
* **Sealed sources: `qoa seal FILE`.** It adds a last line, `SEAL sha256:...`, with the SHA-256 of every line
  above it (for `.qasm` a comment, `// SEAL sha256:...`, so other tools still read the file). QOA then
  refuses to run, check or compile the file if anything above the seal changed, and says how to seal it again
  if the change was yours. It works for `.qoa`, `.qg` and `.qasm`, and checks that the program assembles
  before sealing it. A `SEAL` line anywhere but last is an error that says where it belongs.
* **`QOA_REQUIRE_SEAL=1`** makes QOA refuse every source that is not sealed, for places where only approved
  programs should run (a seal line lost entirely would otherwise leave a plain, unsealed file).
* **Files are written whole or not at all.** `qoa compile`, `qoa export -o`, `qoa import -o` and `qoa seal`
  write a new file next to the target, flush it to the disk and only then put it in place: a full disk, a
  crash or a power cut leaves the previous file untouched.
* The assembler written in QOA (`tools/asm.qoa`) computes SHA-256 itself, in QOA, writes the same format 4
  files byte for byte, and checks seals with the same messages in English and Basque.
* **Short forms, preferred from now on.** Every instruction has a short name (`qoa isa` shows it first):
  `M` for MEASURE, `CX` for CNOT, `D` for DISPLACE, `SQ` for SQUEEZE, `K` for KERR, `N` for COUNT, `E` for
  EMIT, `B` for BELL, `SET` for REGSET, `LD`/`ST` for LOAD/STORE, `J` for JUMP, `C` for CALL, `RET`, `LS`/`LE`
  for loops, and so on. An instruction may touch its first operand when that is a plain register or range of
  its kind (`Hq0`, `SETR1 5`, `Mq0..q2`), and `.` routes a result like `->`: `MEASURE q0 -> c0` can be
  written `Mq0.c0`. The glyph syntax gets the same: `HQ[0]`, `CXQ[0],Q[1]`, `MQ.#C` (`.` for `$>`), and no
  `/` before an operand starting with `#` or `~` (`PUSH#1`, `ADD#4,#1,#2`). Both assemblers read all of it
  the same way. The long forms keep working everywhere, and every short form was an error before, so no
  program changes meaning; the new short names never reserve a name, so a register called `L` or `C` from
  an older program keeps working.
* `qoa export --to qoa`, `--to listing` and `--to glyph` write the short forms; `--long-names` writes the
  full ones (it now works for all three). Every export is still reassembled and compared before it is written.
* Every example, every test program and the assembler written in QOA are now written in the short forms,
  each checked to compile to exactly the same bytes as before (the assembler's source is 13% smaller).
* **Dense glyphs.** `??x{ ... }` runs a block if x is 1 and `!!x{ ... }` if it is 0; `#2<=#1+#3`,
  `#2<=2*pi/3`, `#2<=sqrt(#1)` and `#2<=#C` set a register with one operation (`+ - * / ^ & | << >>`) or
  function; `#2_+1` adds and `#2-=>1` subtracts; `LXP@(address)`, the large expandable pointer, reads or
  writes the memory cell at an address (`#2<=LXP@(100)`, `LXP@(#1)<=#2`); `name::{ ... }` names a block
  that `::name` writes out where it is used; and `""` repeats the previous instruction with new operands
  (`HQ[0]; ""Q[1];`). Each turns into ordinary QOA instructions (`qoa import` shows them), and each was an
  error before. The full rules are in docs/glyph.md.

## Fixed

* `KERR mode kappa time` with a kappa and a time whose product is too large for a 64-bit float was accepted
  and wrote a `.qx` that QOA would then refuse to load. Both assemblers now say "kappa × time is not a
  finite number". The hostile-input test now checks that every program that assembles also loads from its
  own `.qx`.

## Changed

* QOA reads and writes `.qx` files with its own code, and computes SHA-256 itself (checked against the NIST
  examples and `sha256sum`). The bincode library, which is no longer maintained, is gone; the program bytes
  are exactly what bincode wrote, checked against a file bincode itself made, so every older file still loads.
* QOA 0.4.4 and older cannot read format 4 files and say so clearly ("format version 4 is not supported");
  compile the source again with them if needed. Sources written in the new short forms need 0.4.5 too.
* The hostile-input test now also damages files whose checksum is recomputed afterwards, and files in the
  older format without one, so the reader itself keeps being tested behind the checksum.

---

# QOA v0.4.5 bertsio-oharrak

**Egoera:** lanean, oraindik argitaratu gabe

## Gehitua

* **Fitxategi konpilatu bakoitzak bere SHA-256-a du.** `.qx` fitxategiak formatuaren 4. bertsioa dira orain:
  lehengo programa-byte berberak, eta gero aurreko guztiaren SHA-256-a. QOA-k ezer irakurri aurretik
  egiaztatzen du, beraz bit bakar batean aldatutako fitxategi bat (disko txar bat, deskarga eten bat, angelu
  edo qubit-zenbaki baten eskuzko aldaketa) baztertu egiten da hala dioen mezu batekin, programa desberdin
  baina baliozko gisa exekutatu beharrean. Tresna arruntekin ere egiazta daiteke: `head -c -32 f.qx | sha256sum`-ek
  eta `tail -c 32 f.qx | od -An -tx1`-ek digitu berberak inprimatzen dituzte. QOA 0.4.2-tik 0.4.4-ra
  idatzitako fitxategiak kargatzen dira oraindik.
* **Iturburu zigilatuak: `qoa seal FITXATEGIA`.** Azken lerro bat gehitzen du, `SEAL sha256:...`, haren
  gaineko lerro guztien SHA-256-arekin (`.qasm`-en iruzkin bat, `// SEAL sha256:...`, beste tresnek
  fitxategia irakurtzen jarrai dezaten). Orduan QOA-k ez du fitxategia exekutatzen, egiaztatzen edo
  konpilatzen zigiluaren gainean zerbait aldatu bada, eta aldaketa zurea bada berriz nola zigilatu esaten du.
  `.qoa`, `.qg` eta `.qasm`-entzat balio du, eta programa mihiztatzen dela egiaztatzen du zigilatu aurretik.
  Azkena ez den lerro batean dagoen `SEAL` bat errore bat da, non egon behar duen esaten duena.
* **`QOA_REQUIRE_SEAL=1`**-ek zigilatuta ez dagoen iturburu oro baztertzen du, programa onartuak bakarrik
  exekutatu behar diren lekuetarako (osorik galdutako zigilu-lerro batek bestela fitxategi arrunt bat,
  zigilurik gabea, utziko luke).
* **Fitxategiak osorik idazten dira edo batere ez.** `qoa compile`-k, `qoa export -o`-k, `qoa import -o`-k eta
  `qoa seal`-ek fitxategi berri bat idazten dute helburuaren ondoan, diskora bidaltzen dute eta orduan bakarrik
  jartzen dute bere lekuan: disko bete batek, huts-egite batek edo itzalaldi batek aurreko fitxategia ukitu
  gabe uzten du.
* QOAn idatzitako mihiztatzaileak (`tools/asm.qoa`) SHA-256 berak kalkulatzen du, QOAn, 4. formatuko
  fitxategi berberak idazten ditu byte bat bera aldatu gabe, eta zigiluak egiaztatzen ditu mezu berberekin
  ingelesez eta euskaraz.
* **Forma laburrak, hemendik aurrera hobetsiak.** Agindu bakoitzak izen labur bat du (`qoa isa`-k lehenik
  erakusten du): `M` MEASURE-rentzat, `CX` CNOT-entzat, `D` DISPLACE-rentzat, `SQ` SQUEEZE-rentzat, `K`
  KERR-entzat, `N` COUNT-entzat, `E` EMIT-entzat, `B` BELL-entzat, `SET` REGSET-entzat, `LD`/`ST` LOAD/STORE-entzat,
  `J` JUMP-entzat, `C` CALL-entzat, `RET`, `LS`/`LE` begiztentzat, eta abar. Agindu batek bere lehen eragigaia
  uki dezake hura bere motako erregistro arrunt edo tarte bat denean (`Hq0`, `SETR1 5`, `Mq0..q2`), eta `.`-k
  emaitza bat bideratzen du `->` bezala: `MEASURE q0 -> c0` `Mq0.c0` idatz daiteke. Glifo-sintaxiak gauza bera
  du: `HQ[0]`, `CXQ[0],Q[1]`, `MQ.#C` (`.` `$>`-ren ordez), eta ez da `/` behar `#` edo `~`-rekin hasten den
  eragigai baten aurretik (`PUSH#1`, `ADD#4,#1,#2`). Mihiztatzaile biek modu berean irakurtzen dute dena.
  Forma luzeek leku guztietan funtzionatzen jarraitzen dute, eta forma labur guztiak errore bat ziren lehen,
  beraz ez du programa batek ere esanahia aldatzen; izen labur berriek ez dute inoiz izenik erreserbatzen,
  beraz programa zahar bateko `L` edo `C` izeneko erregistro batek funtzionatzen jarraitzen du.
* `qoa export --to qoa`-k, `--to listing`-ek eta `--to glyph`-ek forma laburrak idazten dituzte;
  `--long-names`-ek osoak (orain hiruentzat dabil). Esportazio bakoitza berriz mihiztatu eta konparatzen da
  idatzi aurretik, lehen bezala.
* Adibide guztiak, proba-programa guztiak eta QOAn idatzitako mihiztatzailea forma laburretan daude orain,
  bakoitza lehengo byte berberetara konpilatzen dela egiaztatuta (mihiztatzailearen iturburua %13 txikiagoa da).
* **Glifo trinkoak.** `??x{ ... }`-k bloke bat exekutatzen du x 1 bada eta `!!x{ ... }`-k 0 bada; `#2<=#1+#3`,
  `#2<=2*pi/3`, `#2<=sqrt(#1)` eta `#2<=#C`-k erregistro bati eragiketa (`+ - * / ^ & | << >>`) edo funtzio
  bakar batekin ematen diote balioa; `#2_+1`-ek gehitu eta `#2-=>1`-ek kentzen du; `LXP@(helbidea)`-k,
  erakusle handi hedagarriak, helbide bateko memoria-gelaxka irakurri edo idazten du (`#2<=LXP@(100)`,
  `LXP@(#1)<=#2`); `izena::{ ... }`-k `::izena`-k erabiltzen den lekuan idazten duen bloke bati izena ematen
  dio; eta `""`-k aurreko agindua errepikatzen du eragigai berriekin (`HQ[0]; ""Q[1];`). Bakoitza QOA agindu
  arrunt bihurtzen da (`qoa import`-ek erakusten ditu), eta bakoitza errore bat zen lehen. Arau osoak
  docs/glyph.eu.md-n daude.

## Konpondua

* Biderkadura 64 biteko koma higikorrerako handiegia duten kappa eta denbora batekin, `KERR modua kappa denbora`
  onartu egiten zen, eta gero QOA-k kargatzeari uko egingo zion `.qx` bat idazten zuen. Mihiztatzaile biek
  orain "kappa × denbora ez da zenbaki finitu bat" diote. Eraso-probak orain egiaztatzen du mihiztatzen den
  programa orok bere `.qx`-tik ere kargatzen duela.

## Aldatua

* QOA-k `.qx` fitxategiak bere kodearekin irakurri eta idazten ditu, eta SHA-256 berak kalkulatzen du (NIST-en
  adibideen eta `sha256sum`-en aurka egiaztatuta). bincode liburutegia, jada mantentzen ez dena, kendu da;
  programa-byteak bincode-k idazten zituenak dira zehazki, bincode-k berak egindako fitxategi baten aurka
  egiaztatuta, beraz fitxategi zahar guztiak kargatzen dira oraindik.
* QOA 0.4.4-k eta zaharragoek ezin dituzte 4. formatuko fitxategiak irakurri, eta argi esaten dute ("format
  version 4 is not supported"); behar izanez gero, konpilatu berriz iturburua haiekin. Forma labur berrietan
  idatzitako iturburuek ere 0.4.5 behar dute.
* Eraso-probak orain ondoren laburpena berriz kalkulatzen zaien fitxategiak ere hondatzen ditu, eta laburpenik
  gabeko formatu zaharreko fitxategiak, irakurlea bera laburpenaren atzean probatzen jarrai dadin.
