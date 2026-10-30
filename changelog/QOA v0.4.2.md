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

## Aldatua

* `qoa visual`-ek ez ditu `.wav` fitxategiak bakarrik irakurtzen, eta bere audioa ez da monora nahasten.
