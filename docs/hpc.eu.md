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
neurketak dituztenak, edo milioika lagin. Egoera handi bakar bat nodo askotan banatzea hurrengo urratsa da
(ikus behean).

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

## Oraindik egin gabe

- **Egoera bakar bat nodo askotan.** Egoeraren goiko qubitak prozesuei lotuko litzaizkieke; qubit horietako
  ateek egoeraren erdiak trukatuko lituzkete prozesu-bikoteen artean. Hori behar dute nodo baten memoria
  gainditzen duten simulazioek (34 qubit inguru 512 GiB-eko nodo batean).
- **OpenCL.** GPU bidea gaur vulkan-ekin dabil, eta AMD Instinct azeleragailuek (Frontier-en bezala) ez dute
  eskaintzen. 64 biteko koma higikorreko OpenCL motor batek haiek estaliko lituzke, NVIDIA eta Intel-en
  GPUekin batera.
