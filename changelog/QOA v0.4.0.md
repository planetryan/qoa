# QOA v0.4.0 Release Notes

**Status:** Rewrite of the core (euskaraz behean)

## Summary

QOA 0.4 rebuilds the assembler, the simulator and the command line so that every instruction does what it
says, every result is checked, and the program says plainly what it did. Version 0.3.5 is preserved on the
`archive/v0.3.5-experimental` branch.

## Why

A review of 0.3.5 found that it did not build on any toolchain, that CZ and controlled-phase gates gave wrong
results, that printed results had random noise added even with `--ideal`, that about 97 instructions
(including every optical one) had no effect, that measurement results were thrown away, and that the GPU,
distributed and assembly code was never called. 0.4 fixes all of these.

## New

* **One state for qubits and light**: qubits and optical modes (photon-number states) live in one state
  vector, so a qubit can emit a photon (`EMIT`), control a mode's phase (`DISPERSIVE`), and be entangled with
  light.
* **Real optics**: displacement, squeezing, beamsplitters, phase, Kerr and cross-Kerr, photon addition and
  subtraction, loss, and number-resolving, threshold, parity and coincidence detection. Truncation at the
  photon cutoff is measured and reported.
* **Classical control that works**: measurements route to classical bits or registers (`MEASURE q0 -> c0`),
  labels, conditional jumps on measured bits, counted loops (`LOOPSTART`/`LOOPEND`), subroutines.
* **Typed operands and ranges**: `q3`, `o1`, `R2`, `c0`, `H q0..q3`, angle expressions such as `pi/4`.
* **Errors that point at the problem**, with the line, the column and a suggestion. Instructions that never
  did anything now give an error explaining why and what to use instead.
* **A Kerr black hole**: `examples/kerr.rs` traces light through the metric of a fast-spinning black hole
  (shadow, lensed disk, Doppler shift) and computes its Hawking pairs with QOA's own squeezers and
  beamsplitter: their photon numbers follow Planck's law and always come in pairs. See
  [docs/kerr.md](../docs/kerr.md).
* **Exact optics**: displacement and squeezing matrices come from closed forms instead of a padded matrix
  exponential, faster and exact at cutoffs where the old method silently stopped short.
* **Glyph syntax (draft)**: the same instructions written densely, many per line, in `.qg` files
  (`Q{2}; H/Q[0]; CX/Q[0],Q[1]; M/Q$>#C;`), built from the original concept sketch. `qoa import` shows the
  plain QOA behind it and errors point at the glyph. A draft open for review: [docs/glyph.md](../docs/glyph.md).
* **OpenQASM 2.0**: the whole language can be imported or run directly (gate definitions, `if`, the full
  `qelib1.inc`), and the qubit part of a program can be exported. IonQ JSON export.
* **Honest output**: the report says how shots were computed (sampled from one simulation, simulated one by
  one, or the exact state), which seed was used, and what noise was applied. `--json` gives the same report as
  a record other programs can read.
* **Euskara**: all output, errors and help in Basque or English (`QOA_LANG=eu` / `QOA_LANG=en`). Code comments
  are in Basque.
* **Tests**: every gate against an independent reference simulator, OpenQASM gates against their official
  definitions, boson sampling against matrix permanents, deep circuits run forward and backward, 150+ test
  programs from sanity checks to 28 qubits.
* **Fast**: hand-written SIMD assembly (AVX-512, AVX2, NEON, RVV 1.0, picked at run time and each checked
  against a scalar version), gate fusion and cache blocking. Single gates run at the memory's speed; a
  24-qubit, 355-gate circuit takes 0.9 s, against 23 s in 0.3.5 and 3.7 s in Qiskit Aer on the same machine.
  See [docs/performance.md](../docs/performance.md).
* **Builds on stable Rust** (1.89+), and on x86-64, ARM and RISC-V.

## Removed

* GPU (Vulkan/CUDA/OpenCL), distributed rendering, MPI, and the hand-written `.S` assembly: none of it was
  reachable from QOA, and the assembly no longer assembled. It is preserved on the archive branch. The new
  SIMD assembly is in `src/simd.rs`, is called on every gate, and is tested.
* Example programs that claimed things they did not do (AES/RSA "decryption", an outline of Shor). They are
  replaced by programs that do what they say: a toy Grover key search, Shor's algorithm factoring 15 in full,
  and heralded entanglement of two emitters. The old programs are kept in `tests/programs/legacy`.
* The 0.3 binary format. Recompile `.qoa` sources with `qoa compile`.

## Notes

* `QINIT n` now declares the number of qubits once, as the 0.3 README described. Programs that used one
  `QINIT` per qubit get an error that says so.
* The `visual` command (audio spectrum video) is kept, and described honestly: it is an art tool, it does not
  visualize a quantum state. It was rewritten: frames go straight to ffmpeg instead of through PNG files on
  disk (3 s of 1080p audio used to leave 571 MB of PNGs behind), the same `--seed` gives the same video, and
  it builds on ARM and RISC-V. Bugs fixed on the way: at exactly 1920x1080 and 3840x2160 the pattern ignored
  the audio, the AVX-512 colour blend read the red channel for all three colours, the sine approximation was
  50% off for half its inputs, and the noise field repeated itself for negative coordinates. Rendering alone
  is about 5x faster; most of the remaining time is spent by the H.264 encoder (`--ffmpeg-flag=-preset
  --ffmpeg-flag=ultrafast` trades file size for speed).

**Thank you for using QOA.**

— *Ryan*

---

# QOA v0.4.0 bertsio-oharrak

**Egoera:** nukleoaren berridazketa

## Laburpena

QOA 0.4-k mihiztatzailea, simulagailua eta komando-lerroa berreraikitzen ditu: agindu bakoitzak dioena egiten
du, emaitza bakoitza egiaztatuta dago, eta programak argi esaten du zer egin duen. 0.3.5 bertsioa
`archive/v0.3.5-experimental` adarrean gordeta dago.

## Zergatik

0.3.5-en berrikuspen batek aurkitu zuen: ez zela inolako tresna-katerekin konpilatzen, CZ eta fase
kontrolatuko ateek emaitza okerrak ematen zituztela, inprimatutako emaitzei ausazko zarata gehitzen zitzaiela
`--ideal`-ekin ere, 97 agindu inguruk (optiko guztiak barne) ez zutela ezer egiten, neurketen emaitzak
baztertzen zirela, eta GPU, banatutako eta mihiztadura-kodea inoiz ez zela deitzen. 0.4-k horiek guztiak
konpontzen ditu.

## Berria

* **Egoera bakarra qubitentzat eta argiarentzat**: qubitak eta modu optikoak egoera-bektore bakarrean daude,
  beraz qubit batek fotoi bat igor dezake (`EMIT`), modu baten fasea kontrola dezake (`DISPERSIVE`) eta
  argiarekin entrelazatu daiteke.
* **Optika erreala**: desplazamendua, estutzea, izpi-zatitzaileak, fasea, Kerr eta Kerr gurutzatua,
  fotoi-gehiketa eta -kenketa, galera, eta kopurua bereizten duten, atalase-, paritate- eta
  koinzidentzia-detektagailuak. Fotoi-mugako trunkatzea neurtu eta jakinarazten da.
* **Benetan dabilen kontrol klasikoa**: neurketak bit klasikoetara edo erregistroetara bideratzen dira
  (`MEASURE q0 -> c0`), etiketak, neurtutako biten araberako jauziak, kopuru finkoko begiztak
  (`LOOPSTART`/`LOOPEND`), azpierrutinak.
* **Eragigai motadunak eta tarteak**: `q3`, `o1`, `R2`, `c0`, `H q0..q3`, `pi/4` bezalako angelu-adierazpenak.
* **Arazoa seinalatzen duten erroreak**, lerroarekin, zutabearekin eta iradokizun batekin. Inoiz ezer egin ez
  zuten aginduek orain errore bat ematen dute, zergatik eta zer erabili azalduz.
* **Kerr zulo beltz bat**: `examples/kerr.rs`-ek argia trazatzen du azkar biraka dagoen zulo beltz baten
  metrikan (itzala, lenteatutako diskoa, doppler efektua) eta bere hawking bikoteak QOA-ren estutzaile eta
  izpi-zatitzaileekin kalkulatzen ditu: haien fotoi-kopuruek planck-en legea jarraitzen dute eta beti
  bikoteka datoz. Ikus [docs/kerr.eu.md](../docs/kerr.eu.md).
* **Optika zehatza**: desplazamendu- eta estutze-matrizeak forma itxietatik datoz, matrize-esponentzial
  betegarridun baten ordez: azkarragoak, eta zehatzak metodo zaharra isilean laburra geratzen zen mugetan.
* **Glifo-sintaxia (zirriborroa)**: agindu berak modu trinkoan idatzita, lerro bakoitzean asko, `.qg`
  fitxategietan (`Q{2}; H/Q[0]; CX/Q[0],Q[1]; M/Q$>#C;`), jatorrizko kontzeptu-zirriborrotik abiatuta.
  `qoa import`-ek atzean dagoen QOA arrunta erakusten du eta erroreek glifoa seinalatzen dute. Berrikusteko
  zirriborro bat: [docs/glyph.eu.md](../docs/glyph.eu.md).
* **OpenQASM 2.0**: lengoaia osoa inporta edo zuzenean exekuta daiteke (`gate` definizioak, `if`, `qelib1.inc`
  osoa), eta programaren qubit-zatia esporta daiteke. IonQ JSON esportazioa.
* **Irteera zintzoa**: txostenak esaten du nola kalkulatu diren shot-ak (simulazio bakarretik laginduta,
  banan-banan simulatuta, edo egoera zehatza), zein hazi erabili den eta zein zarata aplikatu den. `--json`-ek
  txosten bera ematen du beste programek irakur dezaketen erregistro gisa.
* **Euskara**: irteera, errore eta laguntza guztiak euskaraz edo ingelesez (`QOA_LANG=eu` / `QOA_LANG=en`).
  Kodearen iruzkinak euskaraz daude.
* **Probak**: ate guztiak erreferentziazko simulagailu independente baten aurka, OpenQASM ateak haien
  definizio ofizialen aurka, boson-laginketa matrize-permanenteen aurka, zirkuitu sakonak aurrera eta atzera,
  150 proba-programa baino gehiago, zentzuzko egiaztapenetatik 28 qubitera arte.
* **Azkarra**: eskuz idatzitako SIMD mihiztadura (AVX-512, AVX2, NEON, RVV 1.0, exekuzio-garaian aukeratua
  eta bakoitza bertsio eskalar baten aurka egiaztatua), ate-fusioa eta cache-blokeak. Ate bakarrak memoriaren
  abiaduran dabiltza; 24 qubit eta 355 ateko zirkuitu batek 0.9 s behar ditu, 0.3.5-en 23 s eta Qiskit
  Aer-en 3.7 s makina berean. Ikus [docs/performance.eu.md](../docs/performance.eu.md).
* **Rust egonkorrean konpilatzen da** (1.89+), eta x86-64, ARM eta RISC-V-n.

## Kendua

* GPU (Vulkan/CUDA/OpenCL), banatutako errendatzea, MPI eta eskuz idatzitako `.S` mihiztadura: ez zen ezer
  QOA-tik iristen, eta mihiztadura jada ez zen mihiztatzen. Artxibo-adarrean gordeta dago. SIMD mihiztadura
  berria `src/simd.rs`-en dago, ate guztietan deitzen da, eta probatuta dago.
* Egiten ez zutena esaten zuten adibide-programak (AES/RSA "deszifratzea", Shor-en eskema bat). Esaten dutena
  egiten duten programek ordezkatzen dituzte: jostailu-gako baten grover bilaketa, Shor-en algoritmoa 15
  faktorizatzen osorik, eta bi igorleren entrelazamendu iragarria. Programa zaharrak
  `tests/programs/legacy`-n gordetzen dira.
* 0.3-ko formatu bitarra. Konpilatu berriz `.qoa` iturburuak `qoa compile`-rekin.

## Oharrak

* `QINIT n`-k orain qubit-kopurua adierazten du behin, 0.3-ko README-ak deskribatzen zuen bezala. Qubit
  bakoitzeko `QINIT` bat erabiltzen zuten programek hala dioen errore bat jasotzen dute.
* `visual` komandoa (audio-espektroaren bideoa) mantentzen da, eta zintzotasunez deskribatuta: arte-tresna
  bat da, ez du egoera kuantikorik bistaratzen. Berridatzi egin da: irudiak zuzenean doaz ffmpeg-era, diskoko
  PNG fitxategien bidez joan beharrean (1080p-ko 3 s-ko audio batek 571 MB PNG uzten zituen), `--seed` berak
  bideo bera ematen du, eta ARM eta RISC-V-n konpilatzen da. Bidean konpondutako akatsak: 1920x1080 eta
  3840x2160 bereizmenetan ereduak ez zion audioari kasurik egiten, AVX-512 kolore-nahasketak kolore
  guztietarako kanal gorria irakurtzen zuen, sinuaren hurbilketak % 50eko errorea zuen sarreren erdian, eta
  zarata-eremua errepikatu egiten zen koordenatu negatiboetan. Errendatzea bakarrik 5 aldiz azkarragoa da
  gutxi gorabehera; geratzen den denboraren zatirik handiena H.264 kodetzaileak hartzen du
  (`--ffmpeg-flag=-preset --ffmpeg-flag=ultrafast`-ek fitxategi-tamaina abiadurarekin trukatzen du).

**Eskerrik asko QOA erabiltzeagatik.**

— *Ryan*
