![qoa-logo](https://github.com/user-attachments/assets/5fdbae92-68f8-490e-8368-f3fd6c81a064)

# QOA: Quantum Optical Assembly

QOA **qubitentzat eta argiarentzat batera** egindako mihiztadura-lengoaia eta simulagailua da: qubitak, modu
optikoak (fotoiak) eta haien arteko akoplamenduak, erregistro klasikoekin eta kontrol-fluxuarekin, programa
bakarrean.

CPU klasiko batean exekutatzen da. Emaitza guztiak zehazki kalkulatzen dira egoera kuantikoa simulatuz, ez
dira hardware kuantikoan neurtzen. Beraz, programa kuanto-optikoak ikasteko, prototipatzeko eta egiaztatzeko
tresna bat da, ez ordenagailu kuantiko bat.

[![ci](https://github.com/planetryan/qoa/actions/workflows/ci.yml/badge.svg)](https://github.com/planetryan/qoa/actions/workflows/ci.yml)

[In English: README.md](README.md) · [Aginduen erreferentzia](docs/ISA.eu.md) · [Bertsio-oharrak](<changelog/QOA v0.4.1.md>)

---

## Zer egiten du

- **Qubit-zirkuituak**: ate estandarrak (H, X, CNOT, CZ, Toffoli, biraketak, ate kontrolatuak...), edozein
  oinarritako neurketa, berrezarpena, eta zirkuitu erdiko neurketa atzeraelikadura klasikoarekin.
- **Modu optikoak**: fotoi-kopuruaren (Fock) egoerak desplazamenduarekin, estutzearekin, izpi-zatitzaileekin,
  fase-aldaketekin, Kerr eta Kerr gurutzatu ez-lineartasunekin, fotoi-gehiketa eta -kenketarekin,
  galerarekin, eta kopurua bereizten duten, atalase- eta paritate-detektagailuekin.
- **Qubit-fotoi akoplamendua**: igorle-qubit batek bere kitzikapena fotoi gisa jar dezake modu batean
  (`EMIT`), eta akoplamendu dispertsiboak (`DISPERSIVE`) qubit bati moduaren fasea kontrolatzen uzten dio.
  Hori da kabitate-QED-aren eta sare kuantikoen tresna-kutxa: entrelazamendu iragarria, Schrödinger-en katu
  egoerak, qubit-fotoi entrelazamendua.
- **Kontrol klasikoa**: 32 erregistro, aritmetika, begiztak, azpierrutinak, neurketen araberako adarkatzea.
- **OpenQASM 2.0**: `qoa run zirkuitua.qasm`-k OpenQASM zuzenean exekutatzen du (lengoaia osoa, `gate`
  definizioak eta `if` barne), eta `qoa export --to qasm`-k programaren qubit-zatia OpenQASM 2.0 gisa
  idazten du. `--to ionq`-k IonQ-ren JSON formatua idazten du.
- **Bi hizkuntza**: mezu, errore eta laguntza-testu guztiak euskaraz eta ingelesez daude. `QOA_LANG=eu` edo
  `QOA_LANG=en` aukeratzeko; bestela sistemaren hizkuntzak erabakitzen du.

Zer **ez** duen egiten: bere kabuz hardware kuantikoan exekutatu, 30 qubit inguru baino gehiago simulatu
(egoera-bektorea bikoiztu egiten da qubit bakoitzeko; 28 qubit 4 GiB dira), edo denbora jarraituko fisika
modelatu (pultsuak, dekoherentzia-tasak). Zarata eredu sinple eta dokumentatu bat da (`--noise`, `--loss`),
edo benetako gailu baten zarata kalibratua: ate-erroreak, T1/T2 erlaxazioa eta irakurketa-erroreak
(`--noise-model`, [docs/noise.eu.md](docs/noise.eu.md)).

Non den baliagarria, non diren beste tresnak hobeak, eta zerk eragozten duen hartzea:
[docs/use-cases.eu.md](docs/use-cases.eu.md).

## Hasteko

**Konpilatu gabe:** jaitsi bertsio bat [GitHub Releases](https://github.com/planetryan/qoa/releases)-etik.
Artxibo bat dago sistema bakoitzeko, eta `qoa-<bertsioa>-all.zip`-ek sistema guztiak batera ditu:

| sistema | prozesadoreak | exekutatu |
|---|---|---|
| Windows 10 / 11 | x64, ARM64 | `qoa.exe run examples\bell.qoa`, edo `run.cmd ...` / `.\run.ps1 ...` |
| Linux (glibc 2.35 edo berriagoa) | x86-64, ARM64, RISC-V 64, IBM POWER8 edo berriagoa (ppc64le), IBM Z (s390x), LoongArch 64 (glibc 2.36+) | `./qoa run examples/bell.qoa`, edo `./run.sh ...` |
| Edozein linux, estatikoa (musl), Android (Termux) | x86-64, ARM64, RISC-V 64 | `./qoa run examples/bell.qoa`, edo `./run.sh ...` |

Bitar guztiak 64 bitekoak dira: QOA-k 64 biteko koma higikorrean kalkulatzen du, eta egoera handiek 64 biteko
helbide-espazioa behar dute.
| macOS | Intel, Apple Silicon | `./qoa run examples/bell.qoa`, edo `./run.sh ...` |

`run` script-ek exekutatzen diren makinarako bitar egokia aukeratzen dute. Iturburu-kopia batean QOA
konpilatzen dute lehenik. `SHA256SUMS`-ek fitxategi guztien laburpenak ditu. Bitarrak ez daude sinatuta:
macOS-ek jaitsitako bat irekitzeari uko egin diezaioke `xattr -d com.apple.quarantine qoa` exekutatu arte,
eta PowerShell-ek `run.ps1` blokea dezake, `run.cmd` edo `powershell -ExecutionPolicy Bypass -File run.ps1
...` erabiltzen ez baduzu.

**Iturburutik**, Rust 1.89 edo berriagoarekin (egonkorra):

```sh
cargo build --release                              # edo ./run.sh besterik ez (windows: run.cmd), konpilatu eta exekutatzen du
./target/release/qoa run examples/bell.qoa
./target/release/qoa run examples/hom.qoa          # bi fotoi izpi-zatitzaile batean
./target/release/qoa run examples/shor15.qoa       # shor-en algoritmoa 15 faktorizatzen, hasieratik amaierara
./target/release/qoa isa                           # agindu guztiak
```

Irteerak argi esaten du zer exekutatu den eta nola:

```
programa   examples/bell.qoa (4 eragiketa)
sistema    2 qubit = 4 anplitude, 64 B
cpu        avx512 kernelak, 4 hari
zarata     bat ere ez: ate idealak, fotoi-galerarik gabe
shots      1024 shot, simulazio bakarretik laginduta (neurketa guztiak amaieran daude)
hazia      7

emaitza    c1..c0  kopurua   ehunekoa
           11          515      50.3%  ███████████████
           00          509      49.7%  ██████████████▉

denbora    0.5 ms
```

`--json`-ek emaitza bera beste programek irakur dezaketen erregistro gisa inprimatzen du.

## Lengoaia orrialde batean

```
; iruzkinak ;-rekin hasten dira (edo //-rekin)
QINIT 3                     ; 3 qubit: q0, q1, q2
QMODE 2 6                   ; 2 modu optiko, bakoitzak 0..5 fotoi

H q0                        ; eragigaiek mota dute: q3 qubita, o1 modua, R2 erregistroa, c0 bita
CNOT q0 q1
RZ q2 pi/4                  ; angeluak adierazpenak dira: pi/4, -3*pi/2, sqrt(2)/2
H q0..q2                    ; tarte batek qubit bakarreko agindua qubit bakoitzari aplikatzen dio

SINGLEPHOTON o0
BS o0 o1 pi/4               ; 50:50 izpi-zatitzailea
EMIT q1 o1                  ; qubitaren kitzikapena -> fotoia

MEASURE q0 -> c0            ; emaitzak bideratzen dituzun lekura doaz: bit batera (c0) edo erregistro batera (R0)
COUNT o0 -> R1
JUMPIFONE c0 amaiera        ; neurtutako bit baten araberako jauzia
X q2
amaiera:
LOOPSTART 3                 ; begiztak, azpierrutinak (CALL/RETSUB), aritmetika (ADD, MUL, ...)
  PRINTF "fotoiak o0-n: {}\n" R1
LOOPEND
```

Erregistroei izena eman dakieke nahi izanez gero (`QREG data 2`, `CREG flags 2`, `OREG arm 2`, gero
`data[1]`, edo `data` guztientzat); `q0`, `c0` eta `o0` erabiltzen jarraitzen dute. Mnemonikoetan berdin dio
maiuskula edo minuskula, eta QOA 0.3-ko izen luzeek (`APPLYHADAMARD`, `PHOTONEMIT`...) ezizen gisa
funtzionatzen dute oraindik. `qoa isa IZENA`-k edozein agindu azaltzen du;
[docs/ISA.eu.md](docs/ISA.eu.md)-k guztiak zerrendatzen ditu. Erroreek leku zehatza seinalatzen dute:

```
errorea: CNOT-k 0. qubita bi aldiz erabiltzen du; qubitek desberdinak izan behar dute
  --> bell.qoa:2:9
   |
 2 | CNOT q0 q0
   |         ^^
```

### Glifo-sintaxia (esperimentala)

Lengoaia bera modu trinkoan ere idatz daiteke, lerro bakoitzean adierazpen askorekin, `.qg` fitxategietan:

```
Q{2}; #C{2}; H/Q[0]; CX/Q[0],Q[1]; M/Q$>#C; PF"bell: {}\n"%#C;
```

`qoa import fitxategia.qg`-k ordezkatzen duen QOA arrunta erakusten du, `qoa export --to glyph
fitxategia.qoa`-k alderantziz egiten du, eta erroreek glifoa seinalatzen dute. `CC[0x10]%a%b$>#2`-k helbide
bateko errutina klasiko bat deitzen du, argumentuak pilan dituela. Esperimentala da eta oraindik alda daiteke: ikus
[docs/glyph.eu.md](docs/glyph.eu.md).

### Programa baten barrura begiratzea

`qoa export --to <helburua>`-k programa bat beste forma batzuetan idazten du, `.qoa`, `.qg`, `.qasm` edo
konpilatutako `.qx` batetik:

| helburua | zer lortzen den |
|---|---|
| `qoa` | QOA arrunt kanonikoa: ezizenak ebatzita, tarteak zabalduta, jauzi-helburuak `addr_0x..` etiketa gisa |
| `listing` | bera, eragiketa bakoitzaren helbide eta iturburu-lerroarekin; mihiztatu daiteke oraindik |
| `json` | programa datu gisa: baliabideak, eta eragiketa bakoitzaren helbidea, lerroa, testua eta egitura |
| `dot` | kontrol-fluxuaren grafoa (oinarrizko blokeak, jauziak, begiztak, deiak) graphviz-erako: `qoa export --to dot f.qoa \| dot -Tsvg > f.svg` |
| `glyph` | glifo-sintaxia (`--width`, `--names`, `--long-names`, `--no-ranges`, `--hex`) |
| `qasm`, `ionq` | hardware errealerako: [docs/hardware.eu.md](docs/hardware.eu.md) |

Testu-esportazio bakoitza berriz mihiztatu eta jatorrizkoarekin konparatzen da idatzi aurretik, beraz
esportazio oker bat errore bat da, inoiz ez isilean desberdina den programa bat.

### Nola kalkulatzen diren shot-ak

- Neurketa guztiak amaieran badaude, QOA-k behin simulatzen du eta shot guztiak egoera horretatik lagintzen ditu.
- Programak zirkuitu erdian neurtzen badu, emaitzen arabera adarkatzen badu, edo `--noise`/`--loss`/`--noise-model`
  erabiltzen badira, shot bakoitza hasieratik simulatzen da.
- Ezer neurtzen ez badu, QOA-k amaierako egoera zehatza erakusten du (simulagailu batek bakarrik egin dezakeena).

Txostenak beti esaten du hauetako zein gertatu den.

### Fotoiak eta muga

Modu batek 0tik muga-1era arteko fotoi-kopuruak gordetzen ditu (`QMODE moduak muga`, lehenetsia 8). Fotoiak
sortzen dituzten eragiketek probabilitatea mugaz kanpora bultza dezakete. QOA-k galera hori neurtu eta
jakinarazten du, ezkutatu ordez; arbuiagarria ez denean, abisu bat ematen du eta muga handiago bat iradokitzen du.

### Zenbakiak eta doitasuna

Qubit-egoera 64 biteko koma higikorrean gordetzen da, eta egoeraren gaineko batura guztiak (arauak,
neurketa-probabilitateak, laginketa) double-double-n, 106 bit inguru. Txostenaren `zenbakiak` lerroak araua
zenbat desbideratu den erakusten du. Gehiagorako, `qoa run --precision quad`-ek qubit-programak 128 biteko
doitasunean gordetzen ditu (double-double, 32 digitu esanguratsu inguru): anplitudeak, ateen matrizeak eta
neurketa. 10 eta 20 aldiz artean motelagoa da. Xehetasunak: [docs/precision.eu.md](docs/precision.eu.md).

### Klusterrak eta superkonputagailuak

`qoa run --mpi`-k shot-ak MPI prozesuetan banatzen ditu (konpilatu `--features mpi`-rekin, abiarazi `mpirun`
edo `srun`-ekin), prozesu batek bezalako zenbaketekin, eta `--mpi --distribute`-k nodo bakarrerako handiegia den
egoera bat haien artean banatzen du. QOA-k egiaztatzen du egoera bat erabil dezakeen
memorian sartzen dela hasi aurretik (`QOA_MAX_MEMORY` aldatzeko). Ikus [docs/hpc.eu.md](docs/hpc.eu.md).

### Musika argitik

`qoa visual abestia.mp3 irteera.mp4`-k audioa zirkuitu optiko kuantiko simulatu batetik pasatzen du: lagin
bakoitza egoera koherente bat daraman argi-modu bat bihurtzen da, argiak galera, estutzea, kerr ingurune bat
edo habe-zatitzaile sare bat zeharkatzen ditu, eta bere benetako zarata kuantikoarekin detektatzen da.
Emaitza audio, bideo edo biak gisa ateratzen da (`--audio-only`, `--video-only`, `--keep-audio`); sarrera
ffmpeg-ek irakurtzen duen edozer da, edo sarrera estandarreko lagin gordinak. Bideoak argiaren egoera
kuantikoak erakusten ditu gertatzen diren unean: modu distiratsuenen wigner funtzio zehatza irudi bakoitzean,
edozein bereizmenetan, 4K60 eta gehiago. Argia zehazki kalkulatzen da: [docs/visual.eu.md](docs/visual.eu.md).

## Adibideak

Guztiak [examples/](examples)-en, bakoitza fisika azaltzen duten iruzkinekin, eta bakoitza proba-multzoak
egiaztatua bere iruzkinek diotenaren aurka:

| Programa | Erakusten du |
|---|---|
| `bell`, `ghz` | entrelazamendua |
| `teleport` | teleportazioa aurrera-elikadura klasiko errealarekin |
| `grover`, `grover-key` | grover-en bilaketa; jostailu-gako baten bilaketa (eta zergatik ez den aes-en eskura) |
| `qpe`, `shor15` | fasearen estimazioa; shor-en algoritmoa 15 faktorizatzen, zati klasikoa barne |
| `hom`, `mach-zehnder` | bi fotoiren interferentzia; fotoi bakarreko interferometroa |
| `coherent`, `squeezed` | laser-argia (poisson estatistika); huts estutua (fotoi-kopuru bikoitiak bakarrik) |
| `emitter`, `cat` | qubit-fotoi entrelazamendua; qubit batekin egindako Schrödinger-en katua |
| `heralded-entanglement` | urrutiko bi igorle entrelazatu haien fotoiak interferentzian jarriz |
| `retry`, `sweep` | errepikatu arrakasta lortu arte; neurketa kuantiko baten inguruko begizta klasikoa |
| `bell.qasm` | OpenQASM zuzenean exekutatuta |
| `bell-torino.qasm` | Bell IBM Torino-rako transpilatua; exekutatu gailu horren zaratarekin: `--noise-model examples/bell-torino.json` |
| `*.qg` | glifo-sintaxia: lerro bakarreko jatorrizko zirriborroa, bell, teleportazioa, HOM igorleekin, errepikatzea, eta errutina klasiko bat deitzen duen (`CC`) errepikapen-kode baten deskodetzailea |
| `kerr.rs` | argia biraka dagoen zulo beltz baten inguruan, bere hawking bikoteak QOA-k kalkulatuta: `cargo run --release --example kerr -- --still kerr.png` ([docs/kerr.eu.md](docs/kerr.eu.md)) |

## Hardware erreala

Programa baten qubit-zatia OpenQASM 2.0 edo IonQ-ren JSON gisa esportatzen da, kopuru finkoko begiztak
eta neurtutako biten araberako aurrera-elikadura barne, eta qiskit-en berdin exekutatzen da. IBM-ren Heron
eta Eagle gailuetarako transpilatuta eta haien zarata neurtuarekin, zirkuitu txikiek erantzun zuzenen % 75-98
gordetzen dute eta 8 qubiteko grover gako-bilaketa bat % 30era jaisten da. Optika linealeko zatia
Quandela-ren Perceval-ekin bat dator 1e-15-raino. IBM Quantum-en doako kontu batekin (gailu errealeko 10 minutu
28 egunean), `tools/run_on_ibm.py examples/bell.qoa`-k programa bat ordenagailu kuantiko erreal batean
exekutatzen du eta emaitza QOA-renarekin konparatzen du. Ondoan, QOA-k gailuak zer itzuliko duen aurreikusten
du gailuaren kalibraziotik bertatik ([docs/noise.eu.md](docs/noise.eu.md)). Neurketak, mugak eta xehetasunak:
[docs/hardware.eu.md](docs/hardware.eu.md).

## Probak

```sh
cargo test --release                       # behekoak guztiak, mugak izan ezik
cargo test --release -- --ignored          # mugak ere bai: 28 qubit (4 GiB), fotoi-sistema erraldoiak
```

- `tests/reference.rs`-k ate guztiak simulagailu independente eta nahita sinple baten aurka egiaztatzen ditu.
- `tests/programs/`-ek beren emaitza esperatuak goiburuko iruzkinetan dituzten `.qoa` programak ditu:
  zentzuzko egiaztapenak, nahita huts egiten dutenak (errore-mezu guztiak), estres-probak, mugak eta 0.3-ko
  programa zaharrak.
- `tests/stress.rs`-k ausazko zirkuitu sakonak aurrera eta atzera exekutatzen ditu, eta boson-laginketa
  matrize-permanenteen aurka egiaztatzen du.
- `tests/qasm.rs`-k OpenQASM ate berezko bakoitza bere definizio ofizialarekin konparatzen du.

## Errendimendua

Simulazio handi bat memoriaren banda-zabalerak mugatzen du, ez aritmetikak: ate bakoitzak egoera-bektore
osoa irakurri eta idazten du (28 qubit 4 GiB dira). QOA-k hiru gauza egiten ditu horren aurka:

- **Eskuz idatzitako SIMD mihiztadura** barne-begiztetarako: AVX-512 eta AVX2 x86-64-n, NEON ARM-n, RVV 1.0
  RISC-V-n, exekuzio-garaian aukeratuta (`QOA_SIMD=scalar`-ek desaktibatzen du). Bertsio bakoitza
  bertsio eskalar soil baten aurka egiaztatzen da, ARM eta RISC-V-n ere qemu bidez.
- **Ate-fusioa**: qubit bereko ate jarraituak matrize bakar bihurtzen dira, tartean beste qubitetako ateak
  egon arren.
- **Cache-blokeak**: qubit baxuetako ate-segida bat blokez bloke aplikatzen da bloke bakoitza cache-n
  dagoen bitartean, beraz segidak memoria behin zeharkatzen du, ate bakoitzeko behin beharrean.

4 nukleoko Xeon (2.3 GHz) batean, ate bakarrak memoriaren mugan dabiltza orain (75 GB/s inguru 27
qubitekin), eta 24 qubit eta 355 ateko zirkuitu batek 0.9 s behar ditu; Qiskit Aer-ek 3.7 s makina berean
(5.6 s bere ate-fusiorik gabe), eta QOA 0.3.5-ek 23 s. Xehetasunak, neurketak eta nola errepikatu:
[docs/performance.eu.md](docs/performance.eu.md).

`qoa run --gpu`-k qubit-ateak GPU batean aplikatzen ditu horren ordez, vulkan edo OpenCL bidez (AMD Instinct,
NVIDIA, Intel), 64 biteko koma higikorrean, ideia berarekin (ate asko memoriaren gaineko pasaldi bakoitzean,
memoria partekatuko lauzetan). MPI-rekin, prozesu bakoitzak bere GPUa erabiltzen du. CPUaren aurka egiaztatua
dago software bidezko vulkan driver batean eta POCL-en, baina oraindik ez da GPU erreal batean neurtu;
zer irabazi beharko lukeen, eta nola neurtu: [docs/gpu.eu.md](docs/gpu.eu.md).

## Historia

QOA 0.4 nukleoaren berridazketa da. 0.3.5 bertsioa
[`archive/v0.3.5-experimental`](https://github.com/planetryan/qoa/tree/archive/v0.3.5-experimental) adarrean
gordeta dago. Ikus [0.4.0 bertsio-oharrak](<changelog/QOA v0.4.0.md>) zer aldatu den eta zergatik jakiteko, eta
[0.4.1 bertsio-oharrak](<changelog/QOA v0.4.1.md>) ondoren etorri dena jakiteko.

## Laguntzea

QOA nola eraikitzen, egiaztatzen eta idazten den: [CONTRIBUTING.eu.md](CONTRIBUTING.eu.md).

## Lizentzia

GPL-3.0, ikus [LICENSE](LICENSE).

*Ryan (@planetryan)*
