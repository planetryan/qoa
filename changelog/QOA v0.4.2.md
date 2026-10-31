# QOA v0.4.2 Release Notes

**Status:** in progress, not released yet (euskaraz behean)

## New

* **Music through light** (`qoa visual`): audio goes through a simulated quantum optical circuit and comes
  out as audio, video or both.
  * Input: anything ffmpeg reads, or raw samples from a file or standard input (`--raw`).
  * Output: processed audio and video (default), `--audio-only`, `--video-only`, or `--keep-audio` (the
    original sound with video from the processed light).
  * Each sample is a mode of light (orthonormal MDCT) carrying a coherent state; the circuit has a
    beamsplitter mesh, loss, squeezing and Kerr self-phase modulation. Detection is the exact mean, one
    homodyne shot per mode with real vacuum noise, or photon counting.
  * All of it is exact in every combination. That includes:
    * homodyne shots of the non-Gaussian state a Kerr medium leaves, sampled from their exact
      distribution (within 1e-12);
    * squeezed light through loss and a Kerr medium;
    * photon counting of squeezed light.
  * See [docs/visual.md](../docs/visual.md).
  * Decoding is in 64-bit floating point; `.wav` and `.flac` outputs are 24-bit by default.
* **The light's quantum states on video** (`qoa visual --phase-space row|full|off`). Every frame shows the
  exact Wigner function of the brightest modes before detection: coherent blobs, squeezed ellipses, Kerr
  crescents and their negative regions. The result that was measured is marked on each one. Each pixel is
  the exact average over its area, so nothing aliases. Any resolution and frame rate work, including 4K60,
  and ffmpeg options after `--` set the encoding.

* **128-bit precision** (`qoa run --precision quad`): qubit programs run with amplitudes, gate matrices,
  measurement and sums in double-double (about 32 significant digits) on every 64-bit system. Over 20,000
  gates on 10 qubits the norm stays within 4e-29, where 64-bit floats drift to 3e-13. Programs with optical
  modes, `--gpu` and device noise models stay 64-bit and say so. See [docs/precision.md](../docs/precision.md).
* **Kerr black hole, rendered properly** (`examples/kerr.rs`). The disk is a Page-Thorne relativistic thin
  disk, each ray shows the exact blackbody colour at its Doppler and gravitationally shifted temperature
  (CIE 1931), and each ray carries its travel time, so the disk's second image is an older view. Stars are
  lensed with their true magnification, and a galaxy behind the hole is lensed into rings. The image goes
  through camera glow and ACES tone mapping. It renders 1080p by default and any size up to 4K60 and beyond,
  and new tests check the flux, the travel times, star brightness and the colours.

* **Shots over MPI** (`qoa run --mpi`, build with `--features mpi`): shots are split between MPI processes
  on one node or many, and process 0 collects the counts. The counts are the same for any number of
  processes, and an error on one process stops the job cleanly. The MPI layer is a few lines of C built with
  the machine's own MPI wrapper. See [docs/hpc.md](../docs/hpc.md).
* **One state over many processes** (`qoa run --mpi --distribute`): for states larger than one node's
  memory. With 2^g processes each holds 1/2^g of the state. Gates on its own qubits run with the usual
  kernels; diagonal gates on the shared top qubits need no communication; other gates on them exchange
  64 MiB chunks between pairs of processes. Results match one process.
* **Memory-aware size limit:** the fixed limit of 2^32 amplitudes is gone. A state may use 90% of the
  memory available, or of the job's cgroup limit, or `QOA_MAX_MEMORY`. A state that does not fit is a clear
  error before anything is allocated.
* **Glyph syntax settled:** `;` stays required between statements, and `PR"x"%#3` prints a label and its
  values (`x = 3`). See [docs/glyph.md](../docs/glyph.md).

## Changed

* `qoa visual` no longer reads only `.wav` files, and its audio is no longer mixed to mono.

---

# QOA v0.4.2 bertsio-oharrak

**Egoera:** lanean, oraindik argitaratu gabe

## Berria

* **Musika argitik** (`qoa visual`): audioa zirkuitu optiko kuantiko simulatu batetik pasatzen da eta audio,
  bideo edo biak gisa ateratzen da.
  * Sarrera: ffmpeg-ek irakurtzen duen edozer, edo lagin gordinak fitxategi edo sarrera estandar batetik
    (`--raw`).
  * Irteera: audio eta bideo prozesatuak (lehenetsia), `--audio-only`, `--video-only`, edo `--keep-audio`
    (jatorrizko soinua, bideoa argi prozesatutik).
  * Lagin bakoitza egoera koherente bat daraman argi-modu bat da (MDCT ortonormala); zirkuituak
    habe-zatitzaile sare bat, galera, estutzea eta kerr fase-modulazio propioa ditu. Detekzioa batez besteko
    zehatza da, shot homodino bat modu bakoitzeko hutsaren benetako zaratarekin, edo fotoi-zenbaketa.
  * Dena zehatza da konbinazio guztietan. Barne daude:
    * kerr ingurune batek uzten duen egoera ez-gaussiarraren shot homodinoak, haien banaketa zehatzetik
      laginduta (1e-12-ren barruan);
    * argi estutua galeratik eta kerr ingurune batetik;
    * argi estutuaren fotoi-zenbaketa.
  * Ikus [docs/visual.eu.md](../docs/visual.eu.md).
  * Deskodetzea 64 biteko koma higikorrean egiten da; `.wav` eta `.flac` irteerak 24 bitekoak dira lehenetsiz.
* **Argiaren egoera kuantikoak bideoan** (`qoa visual --phase-space row|full|off`). Irudi bakoitzak modu
  distiratsuenen wigner funtzio zehatza erakusten du detekzioaren aurretik: orban koherenteak, elipse
  estutuak, kerr ilargi-erdiak eta haien eskualde negatiboak. Bakoitzean neurtutako emaitza markatzen da.
  Pixel bakoitza bere azaleraren batez besteko zehatza da, beraz ez dago aliasing-ik. Edozein bereizmen eta
  irudi-abiadura erabil daiteke, 4K60 barne, eta `--` ondorengo ffmpeg aukerek kodeketa erabakitzen dute.

* **128 biteko doitasuna** (`qoa run --precision quad`): qubit-programak anplitudeak, ateen matrizeak,
  neurketa eta baturak double-double-n dituztela exekutatzen dira (32 digitu esanguratsu inguru) 64 biteko
  sistema guztietan. 20.000 ate 10 qubitetan: araua 4e-29-ren barruan geratzen da, 64 biteko koma
  higikorrarekin 3e-13 desbideratzen den bitartean. Modu optikoak dituzten programak, `--gpu` eta gailuen
  zarata-ereduak 64 bitekoak dira oraindik, eta hala esaten dute. Ikus
  [docs/precision.eu.md](../docs/precision.eu.md).
* **Kerr zulo beltza, behar bezala errendatua** (`examples/kerr.rs`). Diskoa page-thorne disko mehe
  erlatibista da, izpi bakoitzak gorputz beltzaren kolore zehatza erakusten du bere tenperatura doppler eta
  grabitazioz desplazatuan (CIE 1931), eta izpi bakoitzak bere bidaia-denbora dakar, beraz diskoaren bigarren
  irudia ikuspegi zaharrago bat da. Izarrak beren benetako handitzearekin desbideratzen dira, eta zuloaren
  atzeko galaxia bat eraztunetan. Irudiak kamera-distira eta ACES tonu-mapaketa ditu. 1080p-n errendatzen du
  lehenetsiz, eta edozein tamainatan 4K60 eta gehiago, eta proba berriek fluxua, bidaia-denborak, izarren
  distira eta koloreak egiaztatzen dituzte.

* **Shot-ak MPI-n** (`qoa run --mpi`, konpilatu `--features mpi`-rekin): shot-ak MPI prozesuen artean
  banatzen dira, nodo batean edo askotan, eta 0. prozesuak zenbaketak biltzen ditu. Zenbaketak berdinak dira
  prozesu-kopurua edozein dela ere, eta prozesu bateko errore batek lana garbi gelditzen du. MPI geruza C lerro
  gutxi batzuk dira, makinaren MPI bilgarriarekin konpilatuak. Ikus [docs/hpc.eu.md](../docs/hpc.eu.md).
* **Egoera bakar bat prozesu askotan** (`qoa run --mpi --distribute`): nodo baten memoria baino handiagoak
  diren egoeretarako. 2^g prozesurekin bakoitzak egoeraren 1/2^g du. Bere qubitetako ateak ohiko kernelekin
  doaz; partekatutako goiko qubitetako ate diagonalek ez dute komunikaziorik behar; haietako beste ateek
  64 MiB-eko zatiak trukatzen dituzte prozesu-bikoteen artean. Emaitzak prozesu batekoak bezalakoak dira.
* **Memoriaren araberako tamaina-muga:** 2^32 anplitudeko muga finkoa desagertu da. Egoera batek erabilgarri
  dagoen memoriaren % 90 erabil dezake, edo lanaren cgroup mugarena, edo `QOA_MAX_MEMORY`. Sartzen ez den
  egoera bat errore argi bat da, ezer erreserbatu aurretik.
* **Glifo-sintaxia finkatuta:** `;` beharrezkoa da adierazpenen artean, eta `PR"x"%#3`-k etiketa bat eta
  haren balioak inprimatzen ditu (`x = 3`). Ikus [docs/glyph.eu.md](../docs/glyph.eu.md).

## Aldatua

* `qoa visual`-ek ez ditu `.wav` fitxategiak bakarrik irakurtzen, eta bere audioa ez da monora nahasten.
