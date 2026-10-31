# QOA klusterretan eta superkonputagailuetan

*[In English](hpc.md)*

QOA ordenagailu eramangarri batean dabil, eta makina handietara hazteko pentsatuta dago. Orri honek esaten du
zer dabilen gaur prozesu askotan, nola konpilatu eta abiarazi, eta zati bakoitzak zer kostatzen duen.

## Shot-ak prozesu askotan (MPI)

Shot askoko exekuzio bat MPI prozesuetan bana daiteke, nodo batean edo askotan. Prozesu bakoitzak programa
simulatzen du, bere shot-zatia hartzen du, eta 0. prozesuak zenbaketak biltzen ditu eta emaitza inprimatzen du.

```sh
cargo build --release --features mpi          # MPI liburutegi bat eta bere konpiladore-bilgarria (mpicc) behar ditu
mpirun -n 64 ./target/release/qoa run zirkuitua.qoa --mpi --shots 1000000
srun -n 64 ./target/release/qoa run zirkuitua.qoa --mpi --shots 1000000      # slurm-en azpian
```

- **Erantzun bera edozein prozesu-kopururekin.** Shot bakoitzak bere hazia du, beraz 1, 3 edo 64 prozesuk
  exekuzio arrunt batek bezalako zenbaketak ematen dituzte, zehazki. CI-k hiru programarekin egiaztatzen du
  3 OpenMPI prozesurekin, eta probek 1, 2, 3 eta 5 rank egiaztatzen dituzte exekuzio laginduetan, shot
  bakoitza simulatzen dutenetan eta zaratatsuetan.
- **Erroreek ez dute lana blokeatzen.** Shot batek prozesu batean huts egiten badu, prozesu guztiek ematen
  dute beren berri, eta 0. prozesuak errorea inprimatzen du eta huts-kode batekin amaitzen du.
- **Sarrera eta irteera.** 0. prozesuak bakarrik inprimatzen du. Programak sarrera irakurtzen badu
  (`INPUT`), 0. prozesuak irakurtzen du eta balioak besteei pasatzen dizkie.
- **Hariak prozesu bakoitzaren barruan.** Prozesu bakoitzak ematen zaizkion nukleo guztiak erabiltzen ditu.
  Nodo bakoitzeko prozesu bakarrarekin, erabili abiarazlearen lotze-aukerak nodo osoa emateko (adibidez
  `srun --cpus-per-task`).

Prozesu bakoitzak egoera osoa du, beraz honek shot asko daudenean laguntzen du: exekuzio zaratatsuak, erdian
neurketak dituztenak, edo milioika lagin. Nodo baterako handiegia den egoera baterako, ikus hurrengo atala.

## Egoera bakar bat prozesu askotan

```sh
mpirun -n 64 ./target/release/qoa run handia.qoa --mpi --distribute
```

`--distribute`-rekin prozesuek egoera bakar bat partekatzen dute, bakoitzak kopia bat izan ordez. 2^g
prozesurekin, goiko g qubitek prozesua aukeratzen dute, eta prozesu bakoitzak 2^(n-g) anplitude ditu: 64
prozesuk bakoitzari egoeraren 1/64 ematen diote. Prozesu-kopurua bi-ren berretura izan behar da.

- **Beste qubitetako ateak** prozesu bakoitzaren barruan doaz ohiko kernelekin (fusioa, cache-blokeak,
  SIMD), prozesuaren zatiaren gainean.
- **Goiko qubitetako ate diagonalek** (RZ, faseak, CZ, T) ez dute komunikaziorik behar: prozesu bakoitzak
  bere faktorea aplikatzen du.
- **Goiko qubitetako beste ateek** datuak trukatzen dituzte prozesu-bikoteen artean, 64 MiB aldi bakoitzean,
  beraz memoria gehigarria zati bat da, ez egoeraren bigarren kopia bat.
- **Neurketak** batura partzialak prozesuen ordenan batzen ditu, eta prozesu guztiek ausazko zenbaki berak
  ateratzen dituzte, beraz denek emaitza bera ikusten dute.
- **Laginketak** lagindutako anplitude bakoitza zein prozesuk duen aurkitzen du, eta prozesu horrek bakarrik
  irakurtzen du.
- `EXPECT`, `BELLCHECK` eta `DUMPSTATE`-k egoera osoa irakurtzen dute. `DUMPSTATE`-k eta txostenaren
  egoera-taulak 0. prozesuan biltzen dute 2^24 anplitude artekoa denean; handiagoa erakusteko handiegia dela
  esaten da.

Prozesu batek bezalako emaitzak ematen ditu: probek 1, 2, 4 eta 8 prozesu egiaztatzen dituzte amaierako
egoerarako (1e-12-ren barruan), zenbaketetarako eta inprimatutako itxarondako balioetarako, eta CI-k 2 eta 4
OpenMPI prozesu egiaztatzen ditu. qubit-programak bakarrik; modu optikoek, `--precision quad`-ek, `--gpu`-k
eta gailuen zarata-ereduek hala esaten dute exekutatu ordez. Shot bakoitza prozesu guztietan batera doa, beraz
hemen prozesu gehiagok memoria ematen dute, ez shot gehiago aldi berean.

### Zure sistemaren MPI-aren aurka konpilatzea

MPI zatia C lerro gutxi batzuk dira ([src/mpi_shim.c](../src/mpi_shim.c)), zure MPI-aren
konpiladore-bilgarriarekin konpilatuak, beraz QOA makinak duen MPI-arekin dabil (Open MPI, MPICH, Cray MPICH,
Intel MPI).

| aldagaia | zer egiten duen |
|---|---|
| `QOA_MPICC` (edo `MPICC`) | konpiladore-bilgarria, lehenetsia `mpicc`; Cray sistemetan `QOA_MPICC=cc` |
| `QOA_MPI_LIBS` | estekatze-aukerak, bilgarriak ezin baditu eman, adib. `"-L/opt/mpi/lib -lmpi"` |

`--features mpi` gabe, `qoa run --mpi`-k nola konpilatu esaten du, prozesu bakarrean exekutatu ordez.

## Egoera bat zein handia izan daitekeen

Egoerak 16 byte behar ditu anplitude bakoitzeko (32 `--precision quad`-ekin), beraz `n` qubitek 2^n × 16 byte
behar dituzte: 30 qubitek 16 GiB, 34 qubitek 256 GiB. Shot-ak simulazio bakar batetik laginkatzen dituen
exekuzio batek 8 byte gehiago behar ditu anplitude bakoitzeko, probabilitateetarako.

Ezer erreserbatu aurretik, QOA-k egiaztatzen du egoera erabil dezakeen memorian sartzen dela:

- `QOA_MAX_MEMORY`, ezarrita badago (`512G`, `1.5T`, `64GiB` edo byte-kopuru bat);
- bestela, linux-en, hasieran erabilgarri dagoen memoriaren % 90, edo lanaren beraren mugarena, lan-kudeatzaile
  batek (slurm eta beste batzuk, cgroup-en bidez) txikiagoa ezartzen duenean;
- bestela 64 GiB.

Sartzen ez den egoera bat errore bat da, zenbat behar duen eta zenbat dagoen esaten duena. Ez du nodoa
eroriarazten. `qoa check`-ek zenbaki berak erakusten ditu ezer exekutatu gabe.

## GPUak

`--gpu` bi moduekin dabil: MPI prozesu bat GPU bakoitzeko, bakoitzak bere nodo-rank-ak adierazten duen GPUa
erabiliz. AMD Instinct azeleragailuetara, vulkan-ik ez dutenez, OpenCL bidez iristen da (`--gpu-api opencl`).
Ikus [gpu.eu.md](gpu.eu.md#gpu-batzuk).

## Oraindik egin gabe

- **Neurketak GPUan.** Neurketa batek prozesu bakoitzaren egoera-zatia CPUra ekartzen du oraindik.
- **Modu optikoak prozesuen artean.** `--distribute`-k qubit-programak bakarrik banatzen ditu.
