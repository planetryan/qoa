# Zertarako den QOA, eta zerk eragozten duen

[In English: use-cases.md](use-cases.md)

QOA 0.4 qubitentzat eta fotoientzat egoera bakarrean dagoen simulagailu txiki eta zehatz bat da, benetako
kontrol klasikoa duen mihiztadura-lengoaia batek gidatua. Orrialde honek esaten du non den baliagarria gaur,
non ez, eta zerk lagunduko edo eragotziko lukeen jendeak hartzea.

## Zer den QOA, paragrafo batean

Egoera-bektore bakar batek qubitak eta modu optikoak (fotoi-kopuruak muga bateraino) gordetzen ditu, beraz
qubit batek fotoi bat igor dezake modu batera, fotoi batek qubit baten fasea alda dezake, eta biak neur
daitezke emaitzak programa gidatzen duela. Inprimatzen duen zenbaki bakoitza zehazki kalkulatua eta
egiaztatua da: ateak simulagailu independente baten aurka, izpi-zatitzaileak matrize-permanenteen aurka,
esportatutako programak qiskit-en aurka, optika lineala Quandela-ren Perceval-en aurka. OpenQASM 2.0
irakurri eta idazten du, IonQ JSON idazten du, txosten arrunt bat edo JSON erregistro bat inprimatzen du,
euskaraz eta ingelesez hitz egiten du, eta memoriaren abiaduran dabil x86-64, ARM eta RISC-V-n. 28-30 qubit
inguruan gelditzen da (edo anplitude-kopuru baliokidean), egoera-bektoreko simulagailu zehatz guztiak bezala.

## Non den baliagarria

**1. Optika kuantikoa eta informazio kuantikoa batera irakastea.**[^basque] Ikastaro gehienek biak bereizten dituzte:
zirkuitu-tresnek (qiskit, cirq) ez dute fotoirik, optika-tresnek (QuTiP, Strawberry Fields, Perceval) ez
dute kontrol-fluxudun qubit-programarik. QOA-ren adibideek bi fotoiren interferentzia, mach-zehnder
interferometro bat, argi koherentea eta estutua, qubit batekin egindako schrödinger-en katu bat,
igorle-fotoi entrelazamendua eta urrutiko bi igorleren entrelazamendu iragarria hartzen dituzte, bell, GHZ,
teleportazio, grover eta shor-en (15) ondoan, lengoaia bakarrean, bakoitzaren emaitza esperatuak
proba-multzoak egiaztatuta. Txostenak esaten du zer kalkulatu den eta nola (laginduta, shot-ez shot
simulatuta, edo zehatza), ikasle batek emaitzak ondo irakurtzeko behar duena.

**2. Sare kuantikoen protokoloak eskala txikian prototipatzea.** Entrelazamendu iragarriak, entrelazamendu-
trukaketak, arrakasta lortu arteko errepikapenak eta aurrera-elikadurako zuzenketek qubitak, fotoiak,
neurketak eta logika klasikoa behar dituzte programa berean. QOA-k hauek zehazki exekutatzen ditu igorle
eta modu gutxi batzuekin, nahikoa protokolo baten logika eta arrakasta-probabilitateak zuzenak direla
egiaztatzeko, denbora eta memoriaren degradazioa eskalan modelatzen dituen sare-simulagailu batera
(NetSquid, SeQUeNCe) pasatu aurretik.

**3. Esperimentu fotoniko txikiak planifikatzea.** Muntaketa bat eraiki aurretik, laborategi batek
espero diren kontaketak jakin nahi ditu: hong-ou-mandel ikusgarritasuna galera batekin, argi estutuaren
fotoi-estatistikak, katu-egoera batek zenbat fotoi behar dituen, fotoi-muga finitu batek erantzuna nola
desitxuratzen duen. QOA-k erantzun zehatzak ematen ditu modu eta fotoi gutxi batzuetarako, trunkatzea
ezkutatu beharrean jakinarazten du, eta bere optika linealeko emaitzak Perceval-ekin bat datoz 1e-15-raino.

**4. Agindu-multzo hibridoen espezifikazio exekutagarri bat.** Qubit-moduluak fotoiekin lotzen dituzten
arkitekturek (ioi edo atomo prozesadoreen arteko lotura fotonikoak, igorleetan oinarritutako makina
fotonikoak) biei buruz hitz egiten duen agindu-geruza bat behar dute. QOA geruza horren adibide zehatz eta
probatu bat da, formatu bitar batekin, testu-forma irakurgarri batekin, glifo-forma trinko batekin eta
OpenQASM zubi batekin. Konpiladore- eta arkitektura-ikerketak erreferentzia gisa erabil dezake, harantz
konpilatzeko edo haren aurka egiaztatzeko.

**5. Beste tresnak probatzeko erreferentzia.** Bere emaitzak zehatzak eta gurutzatuta egiaztatuak direnez,
QOA beste proiektu baten probetako erantzun "ona" izan daiteke: zirkuitu bat esportatu, biak exekutatu,
konparatu (`tools/hardware_check.py`-k hori bera egiten du qiskit eta Perceval-ekin).

**6. Errendimendu-ingeniaritza irakastea.** `src/simd.rs`-ek hiru kernel berak ditu AVX-512, AVX2, NEON eta
RVV mihiztaduran, bakoitza bertsio eskalar baten aurka probatua, eta `cargo bench --bench bandwidth`-ek
makinaren memoria-mugaren ondoan erakusten ditu. Memoriak mugatutako kodearen, cache-blokeen eta ISA
bakoitzeko SIMD-aren adibide trinko eta erreal bat da.

**7. Dibulgazioa eta artea.** `qoa visual` (audiotik bideora) eta kerr zulo beltzaren adibidea (lenteo-efektua
gehi QOA-k estatistikak kalkulatzen dituen hawking bikoteak) fisika kuantikoari buruzko elkarrizketa bat
hasteko bideak dira, inoiz zirkuitu-diagrama bat irekiko ez lukeen jendearekin.

[^basque]: QOA osorik dago euskaraz ere (mezuak, erroreak, laguntza eta kodeko iruzkinak), eta guk dakigunez
beste programazio-tresna kuantiko batek ere ez du hori egiten. Euskarazko ikastaroei eta dibulgazioari
lagun diezaieke, adibidez BasQ programaren eta Donostiako IBM ordenagailu kuantikoaren inguruan.

## Non ez den tresna egokia

| beharra | tresna hobeak | zergatik |
|---|---|---|
| 30 qubit inguru baino gehiago | Qiskit Aer (MPS), cuQuantum, tentsore-sareko simulagailuak | egoera-bektore zehatz bat bikoiztu egiten da qubit bakoitzeko |
| optika gaussiar handia (modu asko, estutzea) | Strawberry Fields, The Walrus | egoera gaussiarrek deskribapen trinkoak dituzte; QOA-k fotoi-kopuruko anplitude guztiak gordetzen ditu |
| gailu-zarata errealista (T1, T2, pultsuak, gurutzaketa) | Qiskit Aer-en zarata-ereduak, Qiskit Dynamics, QuTiP | QOA-ren zarata eredu sinple eta dokumentatu bat da |
| kimika, optimizazioa, ikasketa automatikoa | Qiskit, PennyLane, CUDA-Q | QOA-k ez du algoritmo-liburutegirik |
| sareen denbora eta memoria-degradazioa eskalan | NetSquid, SeQUeNCe | QOA-k ez du denbora-nozioa |
| hardwarean egunero exekutatzea | hornitzaileen SDKak | QOA-k haietara esportatzen du, baina ez du lanen kudeaketarik |

## Hartzea: zerk eragozten duen, eta zer duen QOA-k

Jendeak ez ditu tresnak merituagatik bakarrik aukeratzen. Hauek dira dagoeneko erabiltzen dutenean
mantentzen dituzten indarrak, eta QOA bakoitzean non dagoen:

| indarra | QOA-ri zenbat eragiten dion | QOA-k duena | gehien lagunduko lukeena |
|---|---|---|---|
| **Python ohitura**: ikastaroek, koadernoek eta lanek `pip install` eta Python suposatzen dituzte | gogor | edozein lengoaiak irakur dezakeen JSON txosten bat; OpenQASM sarrera eta irteera | Python pakete bat (pyo3 eta maturin-ekin), `run(iturburua)`-k txostena hiztegi gisa itzultzen duela, eta qiskit-eko zirkuituetara eta haietatik bihurketak |
| **Instalatzeko marruskadura** | ertaina | 15 sistematarako bitar prestatuak: Windows, macOS eta Linux x86-64 eta ARM64-n, Linux RISC-V, IBM POWER, IBM Z eta LoongArch-en, eta Android-en ere ibiltzen diren bitar estatikoak | `cargo install qoa` crates.io-tik |
| **Ekosistema eta babesa**: IBM, Google, Xanadu, Quandela daude tresna handien atzean | gogor, eta ez konpongarria | elkarreragingarritasuna: QOA-k ez dio inori eskatzen qiskit edo Perceval uzteko | QOA lagun gisa, ez ordezko gisa |
| **"Mihiztadura" zaila dirudi** | ertaina | mnemoniko irakurgarriak, lerroa iradokizun batekin seinalatzen duten erroreak, glifo-sintaxia dentsitaterako | jendeak dagoeneko ezagutzen duen OpenQASM-etik hasten den bost minutuko tutorial bat |
| **Konfiantza**: proiektu txiki bat zuzena al da? | ertaina | proba-multzo zorrotz bat (erreferentziazko simulagailua, permanenteak, qiskit, aer eta Perceval egiaztapen gurutzatuak, adibide guztien baieztapenak egiaztatuak), CI publikoan prozesadore guztietan exekutatua | erabiltzaile independenteak aurkitu dutena jakinarazten |
| **Mantentzaile bakarra**: hemendik urtebetera egongo da? | ertaina | kode-oinarri txiki eta irakurgarri bat, diseinu-aukera dokumentatuak eta [ekarpen-gida](../CONTRIBUTING.eu.md) bat | arazo-txantiloiak eta hasteko arazo on batzuk |
| **Lizentzia**: GPL-3.0-k enpresa batzuk txertatzetik aldentzen ditu | baxua irakaskuntza eta ikerketarako, erreala industriarako | GPL-k proiektua irekita mantentzen du | proiektuaren lizentzia-aukera bat; LGPL edo lizentzia bikoitza dira ohiko konpromisoak |
| **Hizkuntza**: tresna gehienak ingelesez bakarrik daude | QOA-ren *alde* doa euskarazko irakaskuntzan | euskara eta ingelesa osorik | euskarazko tutorialak |

QOA-ren alde doana, batez ere, **lehendik daudenak mehe diren lekuan desberdina izatea** da: qubitak eta
fotoiak programa bakarrean, egiazta daitezkeen erantzun zehatzak, nola kalkulatu diren esaten duten
emaitzak, eta bigarren hizkuntza bat. Ez du qiskit-i irabazi behar erabiltzea merezi izateko; lan jakin
gutxi batzuetarako tresna onena izan behar du, eta sartzeko aldaketa merkeena da dagoeneko OpenQASM
irakurri eta idazten duela.

## Ibilbide-orria

Ahaleginaren araberako balioaren ordenan:

1. Python pakete bat API minimo batekin (exekutatu, txostena hiztegi gisa, qiskit bihurketa) eta hiru
   koaderno (HOM, entrelazamendu iragarria, kerr zulo beltza) ingelesez eta euskaraz.
2. OpenQASM 3 esportazioa, begiztak, erregistroak eta arrakasta lortu arteko errepikapenak dituzten
   programak hardwarera irits daitezen.
3. Quandela-ren Perceval-erako esportazioa, optika linealeko zatia hardware fotonikoan exekuta dadin.
4. Euskarazko ikastaroetarako tutorialak.
5. Glifo-sintaxi egonkor bat (finkatu gabeko puntuak [glyph.eu.md](glyph.eu.md)-n daude) eta JSON txosten
   egonkor bat, beste programek emaitzak irakurtzeko erabiltzen duten "ABI klasikoa".

Eginda dagoeneko: bitar prestatuak eta CI publikoa (0.4.0, 0.4.1-en zabaldua), eta gailu errealen
kalibrazioetatik ateratako zarata-ereduak (0.4.1).
