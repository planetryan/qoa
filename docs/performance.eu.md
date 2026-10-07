# QOA-ren errendimendua

[In English: performance.md](performance.md)

Zerk mugatzen duen egoera-bektoreko simulagailu bat, zer egiten duen QOA-k horren aurka, zein azkarra den, eta
nola egiaztatu hemengo zenbaki guztiak zure makinan.

## Muga memoria da, ez aritmetika

n qubiteko egoera bat 2^n zenbaki konplexu dira, 16 bytekoak: 24 qubit 256 MiB dira, 27 qubit 2 GiB, 28
qubit 4 GiB. Qubit bateko ate batek dena irakurri eta idazten du eta 14 eragiketa inguru egiten ditu
anplitude bakoitzeko, mugitutako byte bakoitzeko 0.4 eragiketa gutxi gorabehera. Edozein nukleo modernok
aritmetika hori memoriak byteak ekar ditzakeena baino askoz azkarrago egiten du, beraz egoera cachea baino
handiagoa denean, **ate batek memoria-bidaia bat kostatzen du**, eta azkarrago joateko bide bakarrak bidaia
hori abiadura osoan egitea eta bidaia gutxiago egitea dira.

`cargo bench --bench bandwidth`-ek makinaren muga neurtzen du (bektore bat tokian bertan eskalatzea, ate
batek aritmetikarik gabe egiten duen trafiko bera) ate-kernel bakoitzaren ondoan. Beheko zenbaki guztietarako
erabilitako 4 nukleoko Xeon-ean (2.3 GHz, AVX-512, 2 MiB L2 nukleo bakoitzeko):

| egoeraren tamaina | irakurri + idatzi, 4 hari | hari 1 |
|---|---|---|
| 128 MiB arte (23 qubit, cachean) | 167 - 187 GB/s | 43 - 73 GB/s |
| 512 MiB - 2 GiB (25 - 27 qubit, memoria nagusia) | 62 - 79 GB/s | 18 - 23 GB/s |

Ate baten GB/s = 2 x egoeraren tamaina / denbora (irakurri eta idatzi), beraz taula honekin zuzenean
konpara daiteke.

## Zer egiten duen QOA-k

1. **Lasterketa jarraituak.** q qubitaren ate batek (i, i + 2^q) anplitude-bikoteak nahasten ditu. Bikote
   horiek lasterketa jarraituetan daude, eta kontrolek lasterketa osoak hautatzen dituzte, beraz kernelek ez
   dute inoiz elementuz elementu erabakitzen ([src/kernel.rs](../src/kernel.rs)).
2. **Eskuz idatzitako SIMD mihiztadura** hiru barne-begiztetarako (2x2 matrizea lasterketa parekoetan, bera
   bektore-erregistro baten barruan 0. eta 1. qubitetarako, eta eskalatze diagonala),
   [src/simd.rs](../src/simd.rs)-en: AVX-512 eta AVX2+FMA x86-64-n, NEON ARM-n, RVV 1.0 RISC-V-n
   (bektore-luzerarekiko agnostikoa). Onena exekuzio-garaian aukeratzen da;
   `QOA_SIMD=scalar|avx2|avx512|neon|rvv`-k bat behartzen du. 0. eta 1. qubitetako kontrolak erreika
   bakoitzeko koefiziente bihurtzen dira (identitatea kontrola itzalita dagoenean), beraz ez dute ezer
   gehiago kostatzen.
3. **Ate-fusioa.** Qubit bereko ate jarraituak matrize bakar bihurtzen dira, tartean beste qubitetako ateak
   egon arren; identitate gisa ateratzen diren biderkadurak kentzen dira.
4. **Cache-blokeak.** Makina birtualak ateak ilaran jartzen ditu banan-banan aplikatu beharrean, zerbaitek
   egoera behar duen arte (neurketa bat, eragiketa optiko bat, inprimatze bat). Orduan ate-segida etapetan
   aplikatzen da: qubit baxuetako ateak 256 KiB-eko blokeei aplikatzen zaizkie bloke bakoitza cachean
   dagoen bitartean; gehienez lau qubit altu ere ukitzen dituzten ateak *bildutako* blokeei (qubit baxuetako
   16 KiB-eko lasterketak gehi qubit altuak). Bi kasuetan etapa batek memoria behin irakurri eta idazten du,
   dituen ate-kopurua edozein dela ere.
5. Txostenak anplitude probableenak top-k hautaketa paralelo batekin erakusten ditu, egoera osoa ordenatu
   beharrean (16 milioi anplitude ordenatzeak simulazioak baino gehiago irauten zuen).

Txostenaren `cpu` lerroak esaten du zein kernel exekutatu diren.

## Emaitzak

Ate bakarrak 27 qubitekin (memoria nagusia, muga 75 GB/s inguru):

| atea | QOA 0.4.0, 5750d4e commit-ean | orain |
|---|---|---|
| H q0-n | 19 GB/s | 66 GB/s |
| H q1-n | 28 GB/s | 67 GB/s |
| H q3-n | 38 GB/s | 61 GB/s |
| H q13-n | 58 GB/s | 81 GB/s |
| H q26-n | 46 GB/s | 74 - 80 GB/s |
| RX q5-n | 42 GB/s | 69 GB/s |
| RZ (diagonala) | 38 GB/s | 77 - 81 GB/s |
| CNOT q0 -> q13 | 85 GB/s | 79 - 84 GB/s |
| CZ q0, q26 (egoeraren erdia ukitzen du) | 105 GB/s | 104 - 158 GB/s |
| SWAP q0, q26 | 56 GB/s | 61 - 68 GB/s |

Cachearen barruan, nukleo batean (ate baten nanosegundoak anplitude bakoitzeko, gutxiago hobe):

| kernelak | ns anplitude bakoitzeko |
|---|---|
| AVX-512 | 0.35 - 0.45 |
| AVX2 | 0.55 - 0.60 |
| Rust eskalarra | 1.3 - 1.8 |

0.4 ns-rekin AVX-512 kernela nukleo honen koma mugikorreko mugatik hurbil dago 2x2 konplexu orokor baterako
(8 bektore-eragiketa eta 2 nahasketa 8 anplitudeko, 2 exekuzio-atakatan), beraz etapa handiak orain
aritmetikak mugatzen ditu cachearen barruan eta banda-zabalerak kanpoan.

Zirkuitu osoak:

| zirkuitua | QOA 0.3.5 | QOA 0.4.0 (5750d4e) | orain | Qiskit Aer 0.17 |
|---|---|---|---|---|
| 24 qubit, H, CNOT katea eta RZ-ren 5 geruza (355 ate) | 23.3 s | 4.1 s | 0.83 - 0.90 s | 3.7 s (5.6 s fusiorik gabe) |
| 27 qubit, 27 H ateko geruza bat | | 1.5 s (27 bidaia) | 0.68 s | |

Guztiak makina berean; Aer bere Python APIaren bidez, `method='statevector'`-ekin, beroketa baten ondoren
hiru exekuzioetatik onena.

## Zenbakiak egiaztatzea

```sh
cargo bench --bench bandwidth                  # muga eta kernelak 20, 24, 27 qubitekin
cargo bench --bench bandwidth -- 23 28         # beste tamaina batzuk
QOA_SIMD=scalar cargo bench --bench bandwidth  # mihiztadurarik gabe
cargo bench --bench benchmark                  # criterion: kernelak, optika, programa osoak
```

Bide azkarren zuzentasuna:

- `tests/simd.rs`-k makinan dagoen mihiztadurako kernel bakoitza bertsio eskalarrarekin konparatzen du,
  ausazko lasterketa-antolaketekin eta erreika bakoitzeko koefizienteekin, eta egiaztatzen du lasterketetatik
  kanpo ez dela ezer idazten.
- `tests/kernel.rs`-k sortak (fusioa, tokiko eta bildutako blokeak, bloke-tamaina guztiekin, kontrolak
  edonon, trukaketak, modu optikoak dituzten egoerak) erreferentzia inozo batekin konparatzen ditu, eta
  `tests/reference.rs`-k ate guztiak simulagailu independente baten aurka egiaztatzen ditu.
- `QOA_SIMD=avx2 cargo test` eta `QOA_SIMD=scalar cargo test`-ek proba-multzo osoa beste kernelekin
  exekutatzen dute.

ARM eta RISC-V qemu-n exekutatzen dira (Debian edo Ubuntu-n: `qemu-user`, `gcc-aarch64-linux-gnu`,
`gcc-riscv64-linux-gnu`, eta `rustup target add aarch64-unknown-linux-gnu riscv64gc-unknown-linux-gnu`):

```sh
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu"
cargo test --release --target aarch64-unknown-linux-gnu --test simd --test kernel --test reference

export CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc
export CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_RUNNER="qemu-riscv64 -cpu rv64,v=true,vlen=256 -L /usr/riscv64-linux-gnu"
cargo test --release --target riscv64gc-unknown-linux-gnu --test simd --test kernel --test reference
```

NEON-ekin pasatzen dira, eta RVV-rekin 128, 256, 512 eta 1024 biteko bektore-luzerekin; V hedapenik gabeko
RISC-V prozesadore batek kernel eskalarrak erabiltzen ditu. qemu-k zuzentasuna egiaztatzen du, ez
abiadura: QOA ez da neurtu ARM edo RISC-V hardware errealean.

## Oraindik egin gabe

- Matrize erreala duten ateek (H, RY) aritmetikaren erdia saihets lezakete cachearen barruan.
- Modu optikoen eragiketek (izpi-zatitzaileak, Kerr, igorleak) Rust begizta arruntak erabiltzen dituzte
  oraindik; fotoi-espazio handiak dituzten programak tamaina bereko qubit-zirkuituak baino motelagoak dira.
- GPU backend bat (Vulkan) bereiz aztertzen ari da.
