# Mahai gaineko optika kuantikoko banku bat QOA-rentzat

[In English: bench.md](bench.md)

Hodeiko makinak dira QOA programak hardware errealean exekutatzeko bide bat ([hardware.eu.md](hardware.eu.md)).
Orrialde hau beste bideari buruzkoa da: optika kuantikoko banku txiki bat, irakaskuntza-laborategiek
erabiltzen dituzten piezekin eraikia, non QOA-ren optika linealeko aginduak fotoi bakar errealekin
exekutatzen diren.

**Ez dago QOA osoa exekutatzen duen hardwarerik, eta ez da inoiz egongo.** `DUMPSTATE`, `EXPECT`,
`BELLCHECK` eta `MEANPHOTON` egoera kuantiko zehatza irakurtzen dute neurtu gabe, eta simulagailu batek
bakarrik egin dezake hori. `KERR` eta `CROSSKERR` fotoi bakarreko mailan, `EMIT` eta `DISPERSIVE` (igorle
kuantiko bat kabitate batean), eta bi qubiteko ate deterministak fotoiekin ikerketa-laborategiko ekipoa
behar dute. Banku batek exekuta dezakeena azpimultzo zehatz bat da, eta QOA-k programak harekin egiaztatzen
ditu (`qoa check --target`, behean).

## Hiru maila

| maila | zer den | exekutatzen dituen QOA aginduak | gutxi gorabeherako kostua |
|---|---|---|---|
| A | argi klasikoa: laser bat, izpi-zatitzaileak, fase-aldatzaileak, fotodiodoak | `BS`, `PH`, `MS`, `LOSS`, `DISPLACE`; fotoi bateko programetan, detekzio-probabilitateak fotoi bakar batenak dira | milaka dolar gutxi |
| **B** | **fotoi bakar errealak**: fotoi-bikoteen iturri bat, fotoi bakarreko detektagailuak, koinzidentzien zenbaketa | `SINGLEPHOTON` (iragarria), `BS`, `PH`, `MS`, `LOSS`, `DETECT`, `COINCIDENCE`; bi fotoien interferentzia (Hong-Ou-Mandel); polarizazio-qubitak: `BELL` iturri gisa, qubit bateko ate guztiak, `MEASURE` X, Y edo Z-n | hamarnaka mila |
| C | fotoi-kopurua bereizten duten detektagailuak (kriogenikoak), argi estutua detekzio homodinoarekin, etengailu azkarrak | `COUNT`, `SQUEEZE`, `PARITY`, `SUBPHOTON` iragarria, adarkatzea exekuzioaren erdian | ehunka mila eta gehiago |

A maila ez da kuantikoa (uhin klasikoek fotoi bateko estatistikak ematen dituzte, baina ez bi fotoien
interferentzia), baina `.qx` fitxategi batetik optikara eta atzera bide osoa eraiki eta probatzeko modurik
merkeena da. B maila benetan kuantikoa da, eta sinplea 2026ko neurrietan: graduko optika kuantikoko
laborategiek eraikitzen dutena da.

## B mailak zer behar duen

Prezioak 2026ko azaroan aurkitutako zerrenda-prezioak dira; aldatu egiten dira, eta hornitzaileek eskaera
bakoitzeko aurrekontua ematen dute. Preziorik aurkitu ez denean, taulak hala dio.

### Biderik errazena: irakaskuntza-kit bat

| pieza | zer ematen duen | prezioa |
|---|---|---|
| Thorlabs EDU-QOP1 (hazbeteak) edo EDU-QOP1/M (metrikoa), Quantum Optics Educational Kit | aire libreko fotoi-bikoteen iturri bat, I motako BBO kristal batekin, eta ikastaroaren esperimentuetarako piezak, Hanbury Brown-Twiss-etik hasita | $35,744.68 ([Thorlabs](https://www.thorlabs.com/item/edu-qop1)) |
| Thorlabs EDU-QOPA1(/M), Polarization-Entanglement Extension Kit | polarizazioan entrelazatutako bikoteak (EDU-QOP1 behar du) | $4,800.18 ([Thorlabs](https://www.thorlabs.com/quantum-optics-educational-kits)) |
| edo: qutools quED | polarizazioan entrelazatutako bikoteen iturri bat zuntzera akoplatua (405 nm-ko ponpa, BBO kristalak), fotoi bakarreko bi siliziozko lausotze-detektagailu, koinzidentzia-logika duen lau kanaleko kontagailu bat | eskatuta ([qutools](https://qutools.com/qued/)) |

Egiaztatu hornitzailearekin zer detektagailu eta elektronika dakartzan kit batek eskatu aurretik; beheko
pieza-zerrenda bankuak guztira behar duena da.

### Piezaz pieza eraikitzea

| pieza | zertarako | adibidea | prezioa |
|---|---|---|---|
| ponpa-laserra, 405 nm, hamarnaka mW | fotoi-bikoteak sortzen ditu (405 nm-ko fotoi bat 810 nm-ko bitan banatzen da) | Thorlabs L405A1 diodoa, 175 mW (gehi tenperatura-kontroldun euskarri bat eta elikagailu bat); edo LP405C1, 30 mW, modu bakarreko zuntz-irteera | $861.60 / $1,418.82 ([L405A1](https://www.thorlabs.com/item/L405A1), [LP405C1](https://www.thorlabs.com/item/LP405C1)) |
| kristal ez-lineala | beheranzko bihurketa parametriko espontaneoa: bikoteen iturria | I motako BBO, 1 mm (Thorlabs NLCQ1); bi gurutzatuak polarizazio-entrelazamendurako; edo II motako BBO (NLCQ4); edo PPKTP kristal distiratsuagoa (tenperatura-kontroldun labe bat behar du) | $652.70 / $674.10 / $2,685.00 ([BBO](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=16384), [PPKTP](https://www.thorlabs.com/ppktp-crystals-for-spontaneous-parametric-down-conversion-spdc)) |
| fotoi bakarreko detektagailuak, siliziozko SPAD, 2tik 4ra | 810 nm-ko fotoiak zenbatzen dituzte (% 60 inguruko eraginkortasuna); 2 iragarpenerako eta Hong-Ou-Mandel-erako, 4 polarizazioko Bell proba osorako | Hamamatsu-ren fotoi-zenbaketako moduluak; Excelitas SPCM, Thorlabs eta Laser Components COUNT aurrekontuz | 2,800-3,700 $ inguru bakoitza Hamamatsu-n ([denda](https://shop.hamamatsu.com/collections/spad-single-photon-avalanche-diode)); besteak eskatuta |
| koinzidentzien zenbaketa | nanosegundo gutxiren barruko klik-bikoteak zenbatzen ditu | denbora-etiketatzaile bat (Swabian Time Tagger 20: 8 sarrera, 34 ps-ko jitter-a, 11,300 $-tan aipatua 2024ko irakaskuntza-laborategi baten pieza-zerrendan); edo irakaskuntza-laborategien diseinu argitaratuekin eraikitako FPGA koinzidentzia-kontagailu bat | ~11,300 $ ([arXiv:2407.15759](https://arxiv.org/html/2407.15759v1)); [Branning et al., Am. J. Phys. 77, 667 (2009)](https://www3.trincoll.edu/~dbrannin/Publications/AJP_77_667_(2009).pdf)-ren FPGA kontagailuak elektronika tradizionalaren % 5 baino gutxiago balio du |
| polarizazio-optika 810 nm-n | qubit-ateak eta neurketa-oinarriak: uhin erdiko eta laurdeneko xaflak, kubo polarizatzaileak; 50:50 izpi-zatitzaile bat Hong-Ou-Mandel-erako | zero ordenako xaflak, PBS eta NPBS kuboak edozein optika-hornitzailerena | hemen ez da preziorik aurkitu; normalean ehunka dolar bakoitza |
| iragazkiak eta zuntz-akoplamendua | ponparen argia kendu (luzera luzeko iragazkia), 810 nm hautatu (banda estuko iragazkia), fotoiak detektagailuetara eraman | luzera luzeko eta 810/10 banda-iragazkiak, zuntz-akoplagailuak, modu anitzeko zuntzak | hemen ez da preziorik aurkitu |
| mekanika | dena geldi eusten du | mahai optiko bat, euskarri zinematikoak eta biraketakoak, zutoinak | hemen ez da preziorik aurkitu; askotan ezkutuko kosturik handiena |
| biraketa-euskarri motordunak eta fase-aldatzaile bat | **bankua programagarri egiten duena**: QOA-k xaflen angeluak (qubit bateko ateak, neurketa-oinarriak) eta faseak (`PH`) ezartzen ditu, pertsona batek botoiak biratu beharrean | xafletarako biraketa-euskarri motordunak; ispilu piezoelektriko bat edo kristal likidozko atzeratzaile bat fasetarako | hemen ez da preziorik aurkitu |
| segurtasuna | 405 nm-ko laser-betaurrekoak, argiaren aurkako itxitura bat (detektagailuek iluntasuna behar dute) | | |

Magnitude-ordena: fotoi iragarrietarako eta Hong-Ou-Mandel-erako banku txiki bat (laser bat, kristal bat,
bi detektagailu, FPGA kontagailu bat, optika) hamarnaka mila dolar gutxitan dago; lau detektagailu,
denbora-etiketatzaile bat eta xafla motordunak dituen polarizazioko Bell probako banku oso batek
irakaskuntza-kitaren prezio bera du gutxi gorabehera.

## Zer QOA programa dabiltzan bertan

B mailako banku batean, programa baten moduak fotoien bide espazialak dira, eta haren qubitak bikote
bateko bi fotoien polarizazioak. `qoa check --target bench-b FITXATEGIA`-k esaten du programa bat sartzen
den, eta sartzen ez den lerro bakoitza eta zergatia izendatzen ditu:

- `examples/mach-zehnder.qoa` (fotoi iragarri bat, bi izpi-zatitzaile, fase bat, klik-detektagailuak)
  dagoen bezala dabil.
- Hong-Ou-Mandel klik-detektagailuekin dabil: bi fotoi iragarri 50:50 izpi-zatitzaile batera, eta bi
  irteeretako detektagailuek ez dute inoiz batera klik egiten, eta hori da sakonunea
  (`examples/hom-clicks.qoa`). `examples/hom.qoa`-k
  `COUNT` erabiltzen du (kopurua bereizten du), C mailakoa dena.
- Bell proba bat `BELL q0 q1` iturri gisa erabiliz dabil, edozein qubit bateko aterekin (xaflak), eta
  `MEASURE` X, Y edo Z-n fotoi bakoitzean (`examples/bell-photons.qoa`). `examples/bell.qoa`-k bikotea `H`
  eta `CNOT`-ekin eraikitzen du, fotoiek modu deterministan egin ezin dutena, beraz baztertu egiten da.
- Bankuak sekuentzia finko bera exekutatzen du shot bakoitzean, beraz jauziak, begiztak, deiak eta
  emaitzak exekuzioaren erdian irakurtzea baztertu egiten dira (neurketa baten araberako adarkatzeak
  etengailu azkarrak behar ditu, C maila).
- Urrats probabilistikoak zintzoak dira: fotoi iragarri bat bere bikotekideak klik egin duenean bakarrik
  dago, detektagailuek fotoi batzuk galtzen dituzte eta ilunpean klik egiten dute, eta QOA-k bankuaren
  simulazioan biak sartzen ditu.

`--target bench-a`-k modu berean egiaztatzen du argi klasikoko banku baterako: fotoi bat gehienez, qubit
bat (izpiaren polarizazioa) gehienez, eta bi fotoi behar dituen ezer ez.

## Zure bankua simulatzea

Banku-eredu batek (json fitxategi bat, `qoa-bench-1` formatua) banku baten akatsak deskribatzen ditu, eta
`qoa run --noise-model bankua.json programa.qoa`-k programa banku horrek exekutatuko lukeen bezala
simulatzen du. Programak bankuaren profila bete behar du lehenik.
[examples/bench-b.json](../examples/bench-b.json)-k zenbaki ilustratiboak ditu:

```sh
qoa run --noise-model examples/bench-b.json examples/bell-photons.qoa
```

| eremua | zer den | nola lortu zure bankurako |
|---|---|---|
| `detector.efficiency` | detektagailu batera iristen den fotoi batek klik eragiteko probabilitatea | detektagailuaren datu-orria (siliziozko SPADak 810 nm-n: 0.6 inguru), edo kalibratu iturri ezagun baten aurka |
| `detector.dark` | fotoirik gabeko klik baten probabilitatea, koinzidentzia-leiho baten barruan | iluneko kontaketen tasa bider leihoa, adib. segundoko 100 × 10 ns = 1e-6 |
| `source.heralding` | bide iragarriak benetan fotoia izateko probabilitatea (akoplamendu- eta iragazki-galerak) | koinzidentziak zati iragarpen-kontaketak, eta gero zati detektagailuaren eraginkortasuna |
| `source.two_photon` | fotoi baten ordez bi izateko probabilitatea; ponparen potentziarekin hazten da | fotoi iragarrien Hanbury Brown-Twiss neurketa batetik |
| `source.visibility` | bikote entrelazatuen kalitatea: egoera V\|Φ+⟩⟨Φ+\| + (1 − V) I/4 da | polarizazio-korrelazioko franjen ikusgarritasuna |
| `target` | `bench-b` (lehenetsia) edo `bench-a` | |

Eremu guztiak aukerakoak dira, eta lehenetsia ideala da. Efektuak shot bakoitzean aplikatzen dira, beren
probabilitate zehatzekin: detektagailu batera iristen den fotoi bat galtzen da 1 − efficiency
probabilitatez; fotoirik gabeko detektagailu batek klik egiten du dark probabilitatez; polarizazio-neurketa
batek (izpi-zatitzaile polarizatzaile bat eta bi detektagailu) ez du emaitzarik bere fotoia galdu bada edo
bi detektagailuek klik egin badute. Haren bita idatzi gabe geratzen da eta `-` gisa agertzen da
kontaketetan: horiek dira bankuaren koinzidentzia-kontagailuak baztertzen dituen shot-ak, eta bit guztiak
idatzita dituztenak gordetzen dituenak.

## Zure bankuan exekutatzea: gailu-protokoloa

`qoa run --device KOMANDOA programa.qoa`-k programa bat banku batean exekutatzen du, simulatu beharrean.
QOA-k bankuaren kontrolagailu-programa abiarazten du (komandoaren hitzak zuriuneetan banatuta, shell-ik
gabe), lana haren sarrera estandarrean idazten du json lerro bakar gisa, eta json lerro bat irakurtzen du
kontaketekin, edozein exekuzio bezala jakinarazten dituenak. Programak bankuaren profila bete behar du
(`--target`, lehenetsia `bench-b`), eta simulazio batek bakarrik exekuta ditzakeen aginduak (inprimatzea,
adibidez) baztertu egiten dira. Protokoloak `qoa-device-1` du izena:

```
QOA -> {"protocol": "qoa-device-1", "job": {"program": "bell-photons.qoa", "target": "bench-b",
        "shots": 1000, "seed": 7, "qubits": 2, "modes": 0, "cutoff": 0, "bits": 2,
        "steps": [{"op": "pair", "qubits": [0, 1]},
                  {"op": "gate", "qubit": 0, "gate": "H", "params": [], "u": [1.5707963267948966, 0.0, 3.141592653589793]},
                  ...,
                  {"op": "measure", "qubit": 0, "basis": "Z", "bit": null}]}}
<- {"protocol": "qoa-device-1", "device": "nire bankua", "shots": 1000,
    "counts": [{"bits": "11", "readouts": [], "count": 180}, {"bits": "-0", "readouts": [], "count": 121}, ...]}
<- edo: {"protocol": "qoa-device-1", "error": "ponpa-laserra itzalita dago"}
```

| urratsa | bankuak zer egiten duen |
|---|---|
| `photon` `mode` | fotoi bakar iragarri bat bide horretara |
| `pair` `qubits` | bikote entrelazatuen iturria: bere bi fotoiak qubit horiek dira |
| `beamsplitter` `modes` `theta` `phi` | izpi-zatitzaile bat (QOA-ren `BS`) |
| `phase` `mode` `angle` | fase-aldaketa bat bide batean |
| `swap` `modes` | bi bide trukatu |
| `loss` `mode` `probability` | ahultzaile bat |
| `displace` `mode` `re` `im` | laser-pultsu bat bidera (bench-a) |
| `gate` `qubit` `gate` `params` `u` | qubit bateko ate bat fotoi baten polarizazioan: QOA-ren izena eta parametroak, eta ate bera U(θ, φ, λ) gisa (fase global batera arte) xaflak ezartzeko |
| `to_basis` `qubit` `basis` | X, Y edo Z neurketa-oinarrira biratu |
| `measure` `qubit` `basis` `bit` | polarizazio-neurketa bat bit batera (`null`: qubitaren bit propioa) |
| `detect` `mode` `bit` | klik-detektagailu bat (`null`: moduaren irakurketa) |
| `coincidence` `modes` `bit` | 1 zerrendako detektagailu guztiek klik egin badute |

Erantzuneko `bits` c(n-1)..c0 idazten da QOA-ren kontaketetan bezala, `-` emaitzarik gabeko bit batentzat,
eta `readouts`-ek modu bakoitzaren irakurketa du (`null` ez badago). QOA-k erantzuna egiaztatzen du: bit eta
modu kopuru zuzena, eta gailuak exekutatu dituela dioen shot-etara batzen diren kontaketak. Zenbakiak beren
digitu guztiekin bidaltzen dira, beraz bankuak jasotzen dituen angeluak programarenak dira zehazki.

`qoa bench-sim [--model bankua.json]` protokoloari simulazioz erantzuten dion ordezko kontrolagailu bat da,
bide osoa bankua egon aurretik eraiki eta probatu ahal izateko:

```sh
qoa run --device "qoa bench-sim --model examples/bench-b.json" examples/bell-photons.qoa
```

Hazi berarekin `qoa run --noise-model examples/bench-b.json`-en kontaketa berberak ematen ditu, eta hori
egiaztatzen dute probek. Benetako banku baten kontrolagailua programa txiki bat da (edozein lengoaiatan):
lana irakurtzen du, xaflak eta faseak ezartzen ditu, shot bakoitzeko klikak zenbatzen ditu, eta erantzuna
idazten du.
