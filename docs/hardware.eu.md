# QOA ordenagailu kuantiko erreal batean exekutatuko litzateke?

[In English: hardware.md](hardware.md)

**Erantzun laburra: qubit-zatia, bai, OpenQASM 2.0 edo IonQ-ren JSON bidez; optika linealeko zatia, bai
printzipioz, prozesadore fotonikoetan (Quandela-renak, adibidez); qubit-fotoi akoplamendua eta optika
ez-lineala, ez gaur alokatu daitekeen inolako makinatan.** Orrialde honetarako ez da programarik hardware
errealera bidali, hornitzaile batean kontu bat behar delako. Hori baino lehenagoko guztia egin eta neurtu da,
eta `tools/run_on_ibm.py`-k azken urratsa egiten du IBM-ren doako planean, kontu bat duzunean (ikus amaiera).

## Qubit-programak

QOA-k programa baten qubit-zatia OpenQASM 2.0 gisa (`qoa export --to qasm`) edo IonQ-ren JSON gisa (`qoa
export --to ionq`) idazten du. Esportatzaileak orain OpenQASM 2.0-k esan ditzakeen bi kontrol-fluxu formak
ere maneiatzen ditu: kopuru finkoko begiztak (zabalduta) eta neurtutako bit baten araberako aurrerako
jauziak ate-bloke baten gainetik (ate bakoitzak `if (cK==v)` bat jasotzen du, IBM hardwarean zirkuitu
dinamiko bat dena). Horrek teleportazioa eta grover-en bilaketa estaltzen ditu.

`tools/hardware_check.py`-k adibide bakoitza esportatzen du, qiskit-en exekutatzen du, IBM-ren bi gailu
errealetarako transpilatzen du (haien kalibrazio argitaratuak: `fake_torino`, 133 qubiteko Heron bat, eta
`fake_sherbrooke`, 127 qubiteko Eagle bat) eta emaitza gailu bakoitzaren zarata neurtuarekin simulatzen du.
[hardware-measurements.md](hardware-measurements.md)-tik:

| programa | 2 qubiteko ateak Heron-en | QOA = Qiskit? (TVD) | erantzun zuzenak, ideala | Heron-en (zarata) | Eagle-n (zarata) |
|---|---|---|---|---|---|
| bell bikotea | 1 | bai (0.001) | % 100 | % 98 | % 98.5 |
| GHZ, 5 qubit | 5 | bai (0.001) | % 100 | % 90 | % 89 |
| teleportazioa (aurrera-elikadura) | 2 | bai (0.003) | % 100 | % 98 | % 98 |
| fasearen estimazioa, 4 qubit | 13 | bai (0.000) | % 100 | % 92 | % 85 |
| grover, 3 qubit | 37 | bai (0.004) | % 94.5 | % 80 | % 75 |
| grover gako-bilaketa, 8 qubit | 354 | bai (0.004) | % 96 | % 30 | % 9.6 |

Honek esaten duena:

- **Esportatutako programak programa berak dira.** QOA eta qiskit-en simulagailua laginketa-zarataren
  barruan bat datoz guztietan, eta QOA -> OpenQASM -> QOA joan-etorri batek kopuru berdin-berdinak ematen
  ditu (tests/qasm.rs).
- **Zirkuitu txikiak ondo dabiltza gaurko hardwarean**; erantzunak shot-en % 75-98an irauten du.
- **8 qubiteko gako-bilaketa ez.** Gailuaren ate eta konexio natiboetara transpilatu ondoren 354 bi qubiteko
  ate behar ditu, eta zaratak gako zuzena shot-en % 30ean uzten du Heron-en eta % 9.6an Eagle-n, asmatzeak
  % 6.25 emango lukeen lekuan. Hau gaurko makinen benetako muga da, ez QOA-rena: erroreak arintzeak zerbait
  laguntzen du, erroreen zuzenketak konpontzen du.
- **Oraindik esportatu ezin dena:** erregistroekin kalkulatzen duten programak (shor-en zati klasikoa,
  kontagailuak), kopurua erregistro batean duten begiztak, eta arrakasta lortu arteko begiztak (atzerako
  jauziak). OpenQASM 3-k hirurak adieraz ditzake (`while`, `if/else`, aldagai klasikoak); OpenQASM 3
  esportatzaile bat da hurrengo urratsa.

## Programa optikoak

OpenQASM-ek ez du modu optikorik, eta qubit-makinek ezin dituzte exekutatu. Prozesadore fotonikoek zati
lineala exekuta dezakete: fotoi bakarrak izpi-zatitzaile eta faseetatik, amaieran kontatuta.
`tools/hardware_check.py`-k QOA sare bat hartzen du, bere fotoi bakarreko transferentzia-matrizea QOA-tik
lortzen du, eta fotoi anitzeko sarrera perceval-en bidez exekutatzen du, Quandela-ren prozesadore
fotonikoen softwarea:

| sarea | fotoiak | QOA eta Perceval-en arteko alde handiena |
|---|---|---|
| hong-ou-mandel | 2 | 7e-16 |
| mach-zehnder, π/3 fasea | 1 | 0 |
| 6 moduko interferometroa (boson-laginketa) | 3 | 7e-16 |

Beraz QOA-ren optika linealeko azpimultzoa txip fotoniko batek kalkulatzen duena da zehazki, eta batera
bidal liteke. Oraindik hardware-helbururik ez duena:

- **qubit-fotoi akoplamendua** (`EMIT`, `DISPERSIVE`): kabitate- eta zirkuitu-QED da, ikerketa-
  laborategietan egina, ez hodeiko zerbitzu gisa eskainia;
- **optika ez-lineala** (`KERR`, `CROSSKERR`) fotoi bakarreko mailan, eta **kopurua bereizten duen
  detekzioa aurrera-elikadurarekin** zirkuituaren erdian: laborategietan neurri batean frogatua, ez
  alokagarria;
- **desplazamendua eta estutzea** aldagai jarraituko makina fotonikoetan badaude, baina hau idaztean ez
  dago bat ere hodeiko gailu gisa irekian eskuragarri (Xanadu-ren Borealis izan zen denbora batez);
  egiaztatu hornitzaileekin horretan oinarritu aurretik.

## Hardware errealean doan exekutatzea

**IBM Quantum-en Open Plan** da qubit errealetara iristeko bide doakoa: doako kontu batek QPU exekuzio-denboraren
10 minutu arte ditu 28 eguneko leiho mugikor bakoitzean, IBM-ren gaurko gailuetan (IBM-ren dokumentazioa, 2026ko
urrian egiaztatua; baldintzak aldatzen dira, beraz egiaztatu [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com)
horretan oinarritu aurretik). Zirkuitu txiki batek, milaka shot-ekin, segundo gutxi batzuk erabiltzen ditu.

`tools/run_on_ibm.py`-k QOA programa baten qubit-zatiaren bidaia osoa egiten du:

```sh
pip install qiskit qiskit-ibm-runtime qiskit-aer
cargo build --release
python3 tools/run_on_ibm.py examples/bell.qoa --dry-run     # lehenik: IBM gailu baten modelo lokala, konturik gabe
export QISKIT_IBM_TOKEN=...                                  # zure API gakoa, quantum.cloud.ibm.com-etik
python3 tools/run_on_ibm.py examples/bell.qoa               # gero gailu erreala (galdetu egiten du QPU denbora erabili aurretik)
python3 tools/run_on_ibm.py examples/grover.qoa --no-wait   # edo bidali eta itzuli geroago:
python3 tools/run_on_ibm.py --job <id>
```

Programa OpenQASM-era esportatzen du, sartzen den gailu lanpetuena ez denerako konpilatzen du, zenbat bi
qubiteko ate behar dituen esaten du, galdetu egiten du QPU denbora gastatu aurretik, lana
`target/ibm-jobs.json`-en gordetzen du, eta gailuak erantzutean bere kopuruak QOA-ren simulazio zehatzaren
ondoan inprimatzen ditu, QOA-k aurreikusten dituen emaitzetako shot-en zatiarekin, bi banaketen arteko
distantziarekin eta lanak erabilitako QPU segundoekin. `--dry-run`-ekin bide bera IBM gailu baten modelo
lokalean exekutatzen da, bere zarata neurtuarekin, eta horrela egin dira goiko zenbakiak: bell % 97.8,
teleportazioa % 97.4, grover % 81.8 shot-ak erantzun zuzenetan, Torino modeloan. Esportatzaileak baztertzen
dituen programak (erregistro-aritmetika, erregistro batek zenbatutako begiztak, atzerako jauziak) ezin dira
oraindik bidali; script-ak zein lerro eta zergatik esaten du.

**Fotoiak:** Quandela Cloud-ek doako kontuak eskaintzen ditu eta Perceval programak prozesadore fotonikoetan
exekutatzen ditu (Ascella, Belenos), baina doako kontu batek QPU denbora duen, eta zenbat, ezin izan da hemendik
egiaztatu; egiaztatu [cloud.quandela.com](https://cloud.quandela.com). QOA-ren aldea prest dago:
`tools/hardware_check.py`-k dagoeneko QOA sare optiko lineal bat Perceval-ek exekutatzen duen
transferentzia-matrize bihurtzen du.

**IonQ eta Amazon Braket**-ek simulagailu doakoak dituzte, baina haien gailu errealak ordainpekoak dira,
ikerketa-kredituen programetatik kanpo.

## Errepikatzea

```sh
cargo build --release
pip install qiskit qiskit-aer qiskit-ibm-runtime perceval-quandela
python3 tools/hardware_check.py --out docs/hardware-measurements.md
```
