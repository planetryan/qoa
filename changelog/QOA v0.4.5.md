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

## Changed

* QOA reads and writes `.qx` files with its own code, and computes SHA-256 itself (checked against the NIST
  examples and `sha256sum`). The bincode library, which is no longer maintained, is gone; the program bytes
  are exactly what bincode wrote, checked against a file bincode itself made, so every older file still loads.
* QOA 0.4.4 and older cannot read format 4 files and say so clearly ("format version 4 is not supported");
  compile the source again with them if needed.
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

## Aldatua

* QOA-k `.qx` fitxategiak bere kodearekin irakurri eta idazten ditu, eta SHA-256 berak kalkulatzen du (NIST-en
  adibideen eta `sha256sum`-en aurka egiaztatuta). bincode liburutegia, jada mantentzen ez dena, kendu da;
  programa-byteak bincode-k idazten zituenak dira zehazki, bincode-k berak egindako fitxategi baten aurka
  egiaztatuta, beraz fitxategi zahar guztiak kargatzen dira oraindik.
* QOA 0.4.4-k eta zaharragoek ezin dituzte 4. formatuko fitxategiak irakurri, eta argi esaten dute ("format
  version 4 is not supported"); behar izanez gero, konpilatu berriz iturburua haiekin.
* Eraso-probak orain ondoren laburpena berriz kalkulatzen zaien fitxategiak ere hondatzen ditu, eta laburpenik
  gabeko formatu zaharreko fitxategiak, irakurlea bera laburpenaren atzean probatzen jarrai dadin.
