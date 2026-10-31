# QOA GPU batean (vulkan eta OpenCL)

[In English: gpu.md](gpu.md)

**Erantzun laburra:** `qoa run --gpu`-k qubit-ateak 64 biteko koma higikorra duen edozein GPUtan aplikatzen
ditu, vulkan edo OpenCL bidez. Hemen zuzena dela egiaztatu da lavapipe-n (mesa-ren software bidezko vulkan-a)
eta POCL-en (CPUrako OpenCL bat). **Ez** da
GPU erreal batean neurtu, eraiki den makinak ez duelako bat ere. Jarraian: nola dabilen, txartel errealetan
zenbat irabazi beharko lukeen (argitaratutako zehaztapenetatik kalkulatua, ez neurtua), eta zurean nola
neurtu.

## Erabiltzea

```sh
qoa run examples/grover-key.qoa --gpu
```

Txostenak esaten du zein gailuk exekutatu dituen ateak eta zer egin duen. Hau 22 qubiteko GHZ egoera bat da
lavapipe-n:

```text
cpu        avx512 kernelak, 4 hari
gpu        llvmpipe (LLVM 20.1.2, 256 bits) (software bidezko vulkan, CPUan exekutatzen da)
gpu lana   1 qubit-ate sorta: 22 ate egoeraren gaineko 4 pasalditan, 0.152 s kalkulatzen; egoera 1 aldiz igo eta jaitsi da (64.0 MiB norabide bakoitzean, 0.132 s). neurketak, optika eta zarata cpu-an exekutatu dira
```

`--json`-ek zenbaki berak ematen ditu `"gpu"` azpian, edo `null` `--gpu` gabe. Gailu erabilgarririk ez
badago, `--gpu` gelditu egiten da arrazoiarekin, isilean CPUan exekutatu beharrean:

```text
errorea: --gpu: ez dago vulkan gailu erabilgarririk: Apple M2: ez du 64 biteko koma higikorrik shader-etan (shaderFloat64); QOA ez da 32 bitetan kalkulatzen
errorea: --gpu: vulkan kargatzaileak ez du vulkan 1.2rako driverrik aurkitu (ERROR_INCOMPATIBLE_DRIVER); instalatu zure GPUaren vulkan driverra
errorea: --gpu: QOA hau GPU euskarririk gabe konpilatu da (cargo-ren `vulkan` ezaugarria)
```

Gailuak vulkan 1.2, `shaderFloat64` eta `bufferDeviceAddress` behar ditu. Mahaigaineko NVIDIA eta AMD
txartelek hirurak dituzte. Apple-ren GPUek (MoltenVK bidez) eta telefono gehienenek ez dute 64 biteko koma
higikorrik. Intel Arc-ek ere ez du 64 biteko hardwarerik, beraz bere driverrak 64 biteko koma higikorra
eskaintzen duenean emulatua eta motela da. Gailu bat baino gehiago badago, GPU diskretua integratuaren
aurretik dago, eta biak software bidezkoaren aurretik. Vulkan programa exekutatzean kargatzen da, beraz QOA
konpilatzeko ez da vulkan SDK-rik behar; `--no-default-features`-ek GPU euskarririk gabe konpilatzen du.

## OpenCL: AMD Instinct, NVIDIA, Intel

f64 azeleragailu handienetako batzuek ez dute vulkan-ik eskaintzen, besteak beste AMD Instinct-ek
(Frontier-eko MI250X, El Capitan-eko MI300A). Haientzat, eta OpenCL driverra duen edozein gailutarako,
QOA-k kernel bera exekutatzen du OpenCL C-n idatzita:

```sh
qoa run zirkuitua.qoa --gpu-api opencl     # edo vulkan; --gpu bakarrik idatzita vulkan lehenik, gero OpenCL
```

- Gailuak 64 biteko koma higikorra (`cl_khr_fp64`) eta 16 KiB tokiko memoria behar ditu. GPUak
  azeleragailuen gainetik nahiago dira, eta biak CPU inplementazio baten gainetik.
- `libOpenCL` programa exekutatzean kargatzen da, beraz konpilatzeko ez da OpenCL SDK-rik behar. AMD-n
  ROCm-ekin dator, NVIDIA-n driverrarekin, eta POCL-ek bat ematen du CPUetarako.
- Egoera gailura gehienez 16 buffertan doa, OpenCL-k bakoitzaren tamaina mugatzen duelako.
- `tests/gpu.rs`-k OpenCL gailua CPUaren aurka exekutatzen du 18 qubit arteko 80 ausazko sortatan. Anplitude
  bakoitzak 1e-10-ren barruan bat etorri behar du. CI-k POCL-en exekutatzen du.

## GPU batzuk

Nodo bateko, edo nodo askotako, GPU guztiak erabiltzeko modua MPI prozesu bat GPU bakoitzeko da:

```sh
mpirun -n 8 qoa run handia.qoa --mpi --distribute --gpu       # 8 GPUk egoera bakar bat partekatzen dute
mpirun -n 8 qoa run zaratatsua.qoa --mpi --gpu --shots 100000 # 8 GPUk shot-ak partekatzen dituzte
```

Prozesu bakoitzak nodoaren barruko bere rank-aren zenbakia duen GPUa hartzen du: Open MPI, MPICH, slurm,
MVAPICH eta Cray abiarazleek esaten dute zein den, eta `QOA_GPU_DEVICE`-k gainidazten du. `QOA_GPU_DEVICE=cpu`-k
prozesu bat CPUan mantentzen du `--gpu`-rekin ere, beraz lan batek GPUak eta CPU nukleoak batera erabil
ditzake abiarazlearen prozesu bakoitzeko ezarpenen bidez (Open MPI-rekin: `mpirun -n 4 -x QOA_GPU_DEVICE=0
qoa ... : -n 4 -x QOA_GPU_DEVICE=cpu qoa ...`). `--distribute`-rekin,
prozesu bakoitzak egoeraren bere zatiko ateak bere GPUan aplikatzen ditu, eta prozesuek datuak trukatzen
dituzte haiek zeharkatzen dituzten qubitetako ateetarako bakarrik ([hpc.eu.md](hpc.eu.md)).

## Nola dabilen: estutze bera

GPUak CPUaren arazo bera du: ate batek egoera osoa irakurri eta idazten du eta oso kalkulu gutxi egiten du
byte bakoitzeko ([performance.eu.md](performance.eu.md)). Erantzuna ere bera da: ate asko aplikatu memoriaren
gaineko pasaldi bakoitzean.

- **Lauzak memoria partekatuan.** 256 hariko lan-talde batek 2^10 anplitudeko lauza bat kargatzen du
  (16 KiB, vulkan-ek gailu guztietan bermatzen duen tamaina) txip barruko memoria partekatura. Etapako ate
  guztiak han aplikatzen ditu eta lauza itzultzen du, beraz etapa batek memoriaren gaineko pasaldi bat
  kostatzen du, zenbat ate dituen axola gabe. Hau CPUaren cache-blokeen GPU forma da
  ([src/kernel.rs](../src/kernel.rs)).
- **Lauzaren qubitak libreki aukeratuta.** Lauza edozein 10 qubit dira. 5 baxuenak beti daude barruan,
  beraz irakurketa bakoitza 512 byte jarraitu dira. Beste 5ak etapako helburu-qubitak dira, eta soberako
  lekuak hurrengo baxuenentzat dira. Planifikatzaileak ([src/gpu.rs](../src/gpu.rs)) ateak gehitzen dizkio
  etapari helburu berri bat sartzen ez den arte. Kontrolek ez dute lauzan egon behar: kanpokoak behin
  egiaztatzen dira lauza bakoitzeko, lauzaren posizioaren aurka.
- **Ate-mota merkeagoak.** Ate diagonalek (faseak, RZ, CZ) hari bakoitzaren anplitudeak bakarrik ukitzen
  dituzte, beraz ondoz ondoko ate diagonalek ez dute barrerarik behar haien artean. X batek (CNOT, toffoli)
  bikote bat trukatzen du kalkulurik gabe. Matrize erreal batek (H, RY) konplexu batek behar duen
  kalkuluaren erdia baino gutxiago behar du. Trukaketek bikoteak trukatzen dituzte. Hau garrantzitsuagoa da
  GPUan CPUan baino (ikus behean).
- **Kalkulu zehatza.** Dena 64 biteko koma higikorrean, CPUaren bidean bezala. QOA ez da inoiz 32 bitetara
  jaisten azkarrago joateko.
- **Txartelak har dezakeen edozein egoera-tamaina.** Egoera gehienez 1 GiB-eko zatietan banatzen da,
  bakoitza bere buffer eta gailu-helbidearekin, driver askok buffer edo esleipen bakar bat 2-4 GiB-era
  mugatzen dutelako. Kernelak gehienez 64 zati eta 2^31 anplitude helbideratzen ditu.
- **CPUan geratzen dena:** neurketa, eragiketa optikoak eta zarata. Ate-sorta bakoitzak egoera txartelera
  bidaltzen du, sorta aplikatzen du eta egoera itzultzen du. Programa arrunt batek (ateak, gero neurketak)
  joan-etorri bat egiten du. Erdian neurtzen duen programa batek joan-etorri bat egiten du ate-tarte
  bakoitzeko, eta baita shot bakoitzak ere shot-ak banan-banan simulatzen direnean.

Kernela GLSL kalkulu-shader bakar bat da, [src/shaders/tile.comp](../src/shaders/tile.comp), SPIR-V-ra
`tools/shaders.sh`-ek konpilatua. Konpilatutako `tile.spv` biltegian dago, beraz QOA konpilatzeko ez da
shader-konpiladorerik behar, eta proba batek huts egiten du iturburua azken konpilazioaren ondoren aldatu
bada. CPUan barneko begiztak eskuz idatzitako mihiztadura dira, han aginduen aukeraketa gurea delako. GPU
batean ez da. SPIR-V tarteko forma bat da: driverrak txartelaren agindu-multzo propiora konpilatzen du
(NVIDIA SASS, AMD RDNA) eta erregistroen esleipena eta antolaketa berak egiten ditu, eta vulkan-ek ez du
makina-kodea emateko modurik. Eskuz idatzitako SPIR-V sarrera beraren bertsio irakurtzen zailago bat
izango litzateke.

## Zuku desberdina: zerk mugatzen duen GPU bat

CPUan, memoria da horma. GPU batean, 64 biteko kalkulua izan daiteke horma, kontsumo-txartelek beren 32
biteko abiaduraren 1/32 eta 1/64 artean exekutatzen dutelako. Lauza bakoitzak ate asko aplikatzen ditu
pasaldi bakoitzean, beraz mugitutako byte bakoitzeko kalkulua igo egiten da, eta joko-txartel batean
memoria gaindi dezake.

Beheko adibidea [performance.eu.md](performance.eu.md)-ko 24 qubiteko 355 ateko zirkuitua da (5 geruza:
H, CNOT katea eta RZ). Fusioak 355 ateak 259 bihurtzen ditu, eta planifikatzaileak **42 pasalditan**
taldekatzen ditu, hau da, 42 x 512 MiB = 22.5 GB memoria-trafiko. Planetik bertatik zenbatuta, 1,620
koma higikorreko eragiketa inguru behar ditu anplitude bakoitzeko:

- RZ bakoitza hurrengo geruzaren H-rekin bateratuta matrize konplexu bakar bat da, 14 anplitude bakoitzeko;
- CNOTak trukaketak dira, 0;
- lehen H geruza erreala da, 6;
- azken RZ geruza diagonala da, 6.

Hori **27 GFLOP** dira guztira. Txartel bat ezin da bi mugetatik motelena baino azkarragoa izan:

| txartela | 64 biteko gailurra | memoria | kalkulu-denbora | memoria-denbora | onenean |
|---|---|---|---|---|---|
| NVIDIA RTX 3060 | 0.2 TFLOPS inguru | 360 GB/s | 136 ms | 62 ms | ~140 ms, kalkuluak mugatua |
| NVIDIA RTX 4090 | 1.3 TFLOPS inguru | 1008 GB/s | 21 ms | 22 ms | ~22 ms, bi mugak batera |
| AMD RX 7900 XTX | 1.9 TFLOPS inguru | 960 GB/s | 14 ms | 23 ms | ~23 ms |
| NVIDIA A100 | 9.7 TFLOPS | 1.6 - 2.0 TB/s | 2.8 ms | 11 - 14 ms | ~12 ms, memoriak mugatua |
| NVIDIA H100 SXM | 34 TFLOPS | 3.35 TB/s | 0.8 ms | 6.7 ms | ~7 ms, memoriak mugatua |

Gailurrak fabrikatzaileek argitaratutako zifrak dira. **Hauek beheko mugak dira, ez neurketak.** Kernel
errealek gailurraren zati bat lortzen dute, eta taulak ez ditu kontuan hartzen memoria partekatuaren
banda-zabalera eta barrerak. Zirkuitu berak 0.6 - 0.9 s behar ditu dokumentu hauetan erabilitako 4 nukleoko
CPUan, beraz GPU erreal batek 5 eta 100 aldiz artean azkarragoa izan beharko luke ateetan. Egoera mugitzea
gainera kostatzen da: 256 MiB norabide bakoitzean PCIe 4.0 bidez 20 - 40 ms inguru dira joan-etorri
bakoitzeko, txartel azkar batean ateak berak adina. Bi ondorio:

- Ate-mota merkeagorik gabe, zirkuitu berak 2,630 eragiketa inguru beharko lituzke anplitude bakoitzeko (X
  eta H matrize konplexu oso gisa), eta joko-txartelak kalkuluak mugatuak izango lirateke denboraren 1.6
  aldiz inguru. Kalkulua kentzeak kontsumo-GPU batean memoria-pasaldiak kentzeak CPUan adina balio du.
- Datu-zentroetako txarteletan (A100, H100), memoria da berriro muga. Han, lauza handiagoek pasaldiak
  murriztuko lituzkete. Vulkan driverrek normalean 48 KiB memoria partekatu uzten dute lan-talde bakoitzeko
  NVIDIAn eta 64 KiB AMDn, 2^10-en ordez 2^11 edo 2^12 anplituderako lekua.

## Hemen neurtua (lavapipe, software bidezko GPU bat)

`cargo bench --bench gpu -- 20 24`, 4 nukleoko xeon berean. "Gailua" LLVM-k konpilatutako shader-kodea
exekutatzen duen CPUa da, beraz honek hodiak eta pasaldi-kopuruak egiaztatzen ditu, **ez GPUaren
abiadura**. Azken zutabea QOA-ren CPU bide propioa da ate berekin:

| qubitak | ateak | pasaldiak | lavapipe kalkulatzen | egoera mugitzen | QOA CPUan |
|---|---|---|---|---|---|
| 20 | H 0 qubitean | 1 | 6.4 ms | 8.2 ms | 0.7 ms |
| 20 | H geruza (20) | 3 | 33 ms | 7.1 ms | 4.7 ms |
| 20 | zirkuitua (295) | 33 | 308 ms | 9.1 ms | 32 ms |
| 24 | H 0 qubitean | 1 | 56 ms | 171 ms | 9.7 ms |
| 24 | H geruza (24) | 4 | 547 ms | 183 ms | 76 ms |
| 24 | zirkuitua (355) | 42 | 5.6 s | 186 ms | 0.62 s |

Software bidezko vulkan-a QOA-ren eskuz idatzitako AVX-512 kernelak baino 9 aldiz inguru motelagoa da nukleo
berberetan. Hori espero zitekeen, eta horregatik dago `--gpu` txartel erreal batera iristeko bakarrik. Ez
erabili inoiz lavapipe-rekin abiadurarako.

## Nola egiaztatu den

- **Plana, GPUrik gabe** (`tests/gpu.rs`). `gpu::emulate`-k shader-aren lauzaz lauzako indize-kalkulua
  errepikatzen du rust-en. Erreferentzia inozo baten aurka egiaztatzen da 300 ausazko ate-sortarekin: 1etik
  14ra qubit, modu optiko batekin eta gabe (2ren berredura ez diren egoera-luzerak), ausazko kontrolak lauza
  barruan eta kanpoan, trukaketak, X, ate diagonal, erreal eta konplexuak.
- **Barrerak, GPUrik gabe.** `gpu::races`-ek jarraitzen du zein harik ukitzen duen memoria partekatuko zein
  elementu bi barreraren artean, eta huts egiten du bi harik elementu bera ukitzen badute. Ausazko sorta
  guztietako etapa guztiak egiaztatzen dira. Barrerak kentzea egiaztapen honek bakarrik harrapatzen du,
  lavapipe-k lan-talde baten 256 hariak urratsez urrats batera exekutatzen dituelako, eta han barrera falta
  batek ez du ezer aldatzen.
- **Gailua CPUaren aurka.** 120 ausazko sorta (1etik 16ra qubit, moduekin) eta 21 qubiteko egoera bat 200
  aterekin, bi aldiz exekutatua bufferrak berrerabiltzeko. Anplitude guztiek CPUarekin bat etorri behar dute
  1e-10 barruan. Nahita okerreko shader-ek huts egiten dutela egiaztatu da: bata matrize-elementu bat
  trukatuta bide konplexuan, bestea bide errealean.
- **Programa osoak.** Adibide guztiek eta estres-programek kopuru eta probabilitate berdin-berdinak ematen
  dituzte `--gpu`-rekin eta gabe. CLI proba batek txostena, JSONa eta driverrik ez dagoeneko errorea
  egiaztatzen ditu.

GPU erreal bat duen makina batean, exekutatu `QOA_REQUIRE_GPU=1 cargo test --release --test gpu`;
`QOA_REQUIRE_GPU=1` gabe, gailuaren probek zergatik saltatu diren esaten dute. Gero exekutatu `cargo bench
--bench gpu`. Hardware errealeko lehen exekuzioak bihurtzen du goiko taula aurreikuspen batetik neurketa.

## Oraindik egin gabe

- **Egoera txartelean mantendu** neurketetan zehar: probabilitateak, kolapsoa eta laginketa GPU kernel gisa,
  zirkuitu erdiko neurketak egoera osoa CPUra itzul ez dezan.
- **Eragiketa optikoak** (izpi-zatitzaileak, desplazamendua, estutzea, kerr) GPUan. CPUko xafla-egitura
  bera erabiltzen dute, baina ez daude idatzita oraindik.
- **Lauza handiagoak** memoria partekatu gehiago duten txarteletan, eta transferentzia gainjarriak (bi
  bitarteko buffer).
- **GPU errealetan neurtzea** (NVIDIA, AMD), egin beharreko hurrengo neurketa.
