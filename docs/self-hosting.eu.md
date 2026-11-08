# QOA-k QOA eraikitzen du

[In English: self-hosting.md](self-hosting.md)

QOA-ren mihiztatzailea, `.qoa` testu bat `.qx` bitar bihurtzen duen zatia, QOAn bertan idatzita ere badago:
[src/qoa/asm.qoa](../src/qoa/asm.qoa). QOA makinan exekutatzen da beste edozein QOA programa bezala, eta
`qoa compile`-k sortzen duen gauza bera sortzen du: programa guztietarako byte berberak, eta akatsak dituen
programa batentzat errore-mezu berberak, ordena berean, hitzez hitz, ingelesez eta euskaraz.

Bere burua ere mihiztatzen du. Horixe esan nahi du hemen "QOA-k QOA eraikitzen du": rust-en mihiztatzaileak
eraikitako mihiztatzailearen kopia eta hark bere buruaz eraikitzen duena berdinak dira, bytez byte.

## Erabilera

```sh
qoa compile --assembler src/qoa/asm.qoa programa.qoa       # emaitza bera: qoa compile programa.qoa
qoa compile --self-hosted programa.qoa                   # mihiztatzaile bera, qoa-n bertan sartua
```

Mihiztatzailearen iturburua biltegiko `src/qoa/asm.qoa` da, eta QOA 0.4.4-tik aurrera argitalpen-pakete
guztiek ere badakarte (`tools/`-en 0.4.7 arte). `src/qoa/`-k QOAn idatzitako QOA-ren zatiak ditu, rust-en
iturburuen ondoan, `qoa` binarioan konpilatzen direlako; `tools/`-ek python eta shell laguntza-scriptak
ditu. QOA 0.4.7-tik mihiztatzailea `qoa` binarioan ere sartuta dago: `--self-hosted`-ek kopia hori
erabiltzen du, beraz ez da fitxategirik behar, `qoa tools`-ek binarioak daramatzan QOAn idatzitako tresnak
zerrendatzen ditu, eta `qoa tools --write KARPETA`-k idazten ditu (`asm.qoa`, hartatik sortutako `asm.qg`,
eta `asm.qx`).

QOA 0.4.9-tik tresnak bytecode gisa daude binarioan: [src/qoa/asm.qx](../src/qoa/asm.qx) eta
[src/qoa/dis.qx](../src/qoa/dis.qx), QOAn idatzitako mihiztatzaileak beren iturburuetatik eraikitzen dituen
`.qx` fitxategiak. `qoa`-k bytecode hori bere horretan exekutatzen du, beraz rust-en mihiztatzaileak ez du
parterik tresna bat exekutatzean: `--self-hosted`-ekin zure programa mihiztatzen duen mihiztatzailea bera
QOAn idatzitako mihiztatzaileak mihiztatu zuen. Probek egiaztatzen dute `.qx` bakoitza bi mihiztatzaileek
bere iturburutik eraikitzen dutena dela zehazki (beraz bytecode-a ezin da dioen iturburutik aldendu),
mihiztatzaileak bere bytecode-a aldaketarik gabe berreraikitzen duela, eta desmihiztatzaileak tresna
bakoitzaren bytecode-a rust-ek idazten duen testu bihurtzen duela. Tresna baten iturburua aldatu ondoren,
haren bytecode-a `QOA_REGENERATE=1 cargo test --test selfhost`-ekin berreraikitzen da.

`--assembler`-ek edozein QOA programa hartzen du, iturburu gisa edo `.qx` konpilatu gisa. QOA-k exekutatzen
du iturburu-fitxategia sarreran eta fitxategi-izena argumentu gisa emanda, eta inprimatzen duena `.qx`
gisa idazten du. Haren errore-mezuak zuzenean terminalera doaz, eta haren irteera-egoera `qoa`-rena bihurtzen
da.

Gauza bera eskuz, ezkutuko ezer ez dagoela ikusteko:

```sh
qoa compile src/qoa/asm.qoa -o asm.qx                       # mihiztatzailea eraiki (rust-ek behin)
qoa run asm.qx --quiet -- programa.qoa < programa.qoa > programa.qx
```

`--quiet`-ek programak idazten duena bakarrik inprimatzen du, txostenik gabe, eta `qoa` programaren `EXIT`
egoerarekin amaitzen da. `--` ondoko guztia programari argumentu gisa ematen zaio.

## Abiaraztea

```sh
qoa compile src/qoa/asm.qoa -o stage1.qx                                   # rust-ek eraikia
qoa compile --assembler stage1.qx src/qoa/asm.qoa -o stage2.qx             # QOA-k eraikia
qoa compile --assembler stage2.qx src/qoa/asm.qoa -o stage3.qx             # QOA-k eraikitako QOA-k eraikia
cmp stage1.qx stage2.qx && cmp stage2.qx stage3.qx                       # guztiak berdinak
```

Proba-multzoak hauxe bera egiten du eraikitze bakoitzean (`tests/selfhost.rs`).

## Desmihiztatzailea, hau ere QOAn

Itzulerako bidea ere QOAn idatzita dago: [src/qoa/dis.qoa](../src/qoa/dis.qoa)-k `.qx` bat irakurtzen du eta
`qoa export --to qoa --long-names`-ek idazten duen QOA arrunta idazten du, bytez byte. Goiburua, bertsioa eta
SHA-256-a egiaztatzen ditu (QOAn, mihiztatzailearen SHA-256 errutinekin), 80 eragiketa motetako bakoitza
deskodetzen du, 64 biteko zenbaki bakoitza zehazki berreraikitzen du bere 8 byteetatik, eta `addr_0x..`
etiketak jauziak iristen diren lekuetan jartzen ditu. Fitxategi hondatu batek `qoa`-k ematen duen errore
bera jasotzen du, hitz berberekin eta bi hizkuntzetan.

```sh
qoa export --to qoa --long-names --self-hosted programa.qx      # binarioan sartutako kopia (src/qoa/dis.qx)
qoa export --to qoa --long-names --self-hosted programa.qoa     # lehenik QOA mihiztatzaileak mihiztatuta
qoa run src/qoa/dis.qoa --quiet -- programa.qx < programa.qx    # eskuz
```

Iturburu-fitxategi batekin, QOAn idatzitako mihiztatzaile integratuak mihiztatzen du lehenik, beraz
joan-etorri osoa (testua, bitarra, testua) QOAn exekutatzen da, eta hori egiten duten bi programak QOA-k
eraikitako bytecode-a dira. Probek rust-ekin konparatzen dute biltegiko
programa guztietan (mihiztatzailearen beraren 12.600 lerroak barne) eta mota guztietako fitxategi
hondatuetan. Oraingoz izen luzeak idazten ditu; forma laburrak (`Hq0`, `Mq0.c0`) dira hurrengo urratsa.
Mihiztatzaileak bezala, ez du emaitza berriz mihiztatzen programaren edukia egiaztatzeko, `qoa`-k `.qx` bat
kargatzean egiten duen bezala; formatua, laburpena eta aldaera-zenbaki guztiak egiaztatzen ditu.
Mihiztatzailearen bitarrarekin 0.3 s inguru behar ditu, rust-en 0.05 s-en aldean.

## Zer dagoen QOAn idatzita, eta zer geratzen den rust-en

| zatia | lengoaia | zergatik |
|---|---|---|
| mihiztatzailea: lerroak, iruzkinak, tokenak, etiketak, izendun erregistroak, `q0..q3` tarteak, eragigai guztiak, adierazpenak, zenbaki hamartarrak, formatu bitarra, errore-mezu guztiak | **QOA** | testua eragiketa-zerrenda bihurtzen duen programa klasiko bat da, lengoaia batek bere buruarentzat egin dezakeena |
| desmihiztatzailea: `.qx`-tik QOA arruntera izen luzeekin, laburpena, errore guztiak | **QOA** | mihiztatzailearen alderantzizkoa, joan-etorriak rust-ik behar ez dezan |
| QOA makina: qubit-egoerak, fotoiak, SIMD kernelak, GPUak, MPI, zarata | rust | zenbakizko muina da; 64 biteko koma higikorra behar du makinaren abiadura osoan, bektore-aginduak eta GPU kontrolatzaileak, eta programa interpretatu batek ezin ditu horiek eman |
| `sin`, `cos`, `tan`, `exp`, `ln`, `pow` | rust (agindu gisa) | `qoa compile`-k erabiltzen dituen funtzio berberak, biek bit berberak eman ditzaten |
| fitxategiak irakurtzea, komando-lerroa | rust | sistema eragilearen aldea |
| glifo-sintaxia (`.qg`) eta OpenQASM inportazioa | rust, oraingoz | ikus behean |

Mihiztatzailea QOA arruntean idatzita dago, ez glifo-sintaxian. Glifo-sintaxia qubit eta fotoi
erregistroen gaineko zirkuituetarako idazkera trinko bat da, eta QOA arrunt bihurtzen da. Mihiztatzaile bat
kode klasikoa da (begiztak, taulak, testua), eta hor QOA arrunta argiago irakurtzen da. QOA arruntean
idazteak abiaraztea itxita ere mantentzen du: bere burua besterik ez du behar, eta glifo-aurrealdea rust-en
dago oraindik. QOAn idatzitako glifo-aurrealde bat etorkizuneko lana izan daiteke.

## Nola konparatzen den rust-ekin

Mihiztatzailearen iturburu propioan neurtuta (12.000 lerro), 4 nukleoko hodeiko makina batean, bost
exekuzioetatik onena:

| | rust (`src/asm.rs`) | QOA (`src/qoa/asm.qoa`) |
|---|---|---|
| irteera | erreferentzia | byte berberak, mezu berberak |
| denbora | 30 ms | 0,95 s (30 aldiz motelago, gutxi gorabehera), horietatik 0,35 s inguru idazten dituen 234 KB-en SHA-256-a kalkulatzen |
| abiadura | 400.000 lerro segundoko inguru | 13.000 lerro segundoko inguru |
| tamaina | 3.000 lerro inguru mihiztatzailearentzat, adierazpenentzat, `.qx` idazlearentzat, SHA-256-arentzat eta zigiluentzat, gehi agindu-taula | 11.900 lerro inguru, gehi rust-etik sortutako 300 lerro inguru taula |

Beraz rust 30 aldiz azkarragoa da eta 4 aldiz laburragoa logika berarentzat. (0.4.5 aurretik, laburpenik
gabe, QOA-ren mihiztatzaileak 0,6 s inguru behar zituen makina berean.) Hori da interpretatzaile
batean mihiztatze-estiloko kodearen kostua: QOA-k 32 zenbaki-erregistro, memoria-gelaxka zenbakituak eta
jauziak ditu, eta ez mota, egitura edo testu-katerik. Eremu-irakurketa, konparazio eta konstante bakoitza
agindu bereizi bat da. Trukean, mihiztatzaile osoa QOA programa bat da, eta QOA exekutatzen duen edozerk
QOA-ren programak eraiki ditzake berriz. Segundoko 13.000 lerroko mihiztatzailea nahikoa azkarra da edozein
QOA programa errealentzat: biltegiko programa handiena mihiztatzailea bera da, eta segundo bat ingurutan
eraikitzen da.

## Zergatik datozen bat biak

- **Biltegiko programa guztiak**: adibideak eta proba-programa guztiak, huts egiteko egindako 58ak barne,
  biekin mihiztatzen dira ingelesez eta euskaraz, irteera-egoera, mezuak eta `.qx` byteak konparatuz.
- **Sortutako programak**: onartutako 7.000 lerro inguru agindu, ezizen eta eragigai-forma guztiekin, eta
  biribiltze zuzena probatzen duten ausazko 3.000 zenbaki hamartar (luzeak, txiki-txikiak, erdiko kasuak,
  azpinormalak); gehi akatsez betetako 165 programa, 50 errore-mezu desberdin baino gehiago, utf-8 baliogabea,
  Windows eta Mac zaharraren lerro-amaierak eta aldi berean 30 errore baino gehiago ukitzen dituztenak.
- **Zigiluak eta laburpenak**: 0tik 130 bytera arteko luzera guztietako programa zigilatuak (beraz SHA-256-aren
  bloke-muga guztiak), ascii ez den testuarekin, batzuk zigilatu ondoren aldatuak eta beste batzuk zigilu
  okerrekin. Mihiztatzaile batek zein besteak idatzitako `.qx` guztiak SHA-256 batekin amaitzen dira, beraz
  goiko konparazio guztiek SHA-256-aren bi inplementazioak ere konparatzen dituzte.
- Goiko **puntu finkoa**.
- **Taulak ezin dira aldendu**: `src/qoa/asm.qoa` barruko agindu-taula (izenak, ezizenak, eragigaiak,
  erabilera-lerroak, kendutako aginduen arrazoiak) eta unicode-ko letra-tarteak `src/isa.rs`-etik eta
  rust-en karaktere-funtzioetatik sortuak dira. Proba batek huts egiten du desberdinak badira.

Rust-ekin bat etortzeak arreta eskatu zuen leku batzuk:

- **Zenbaki hamartarrak** 64 biteko koma higikor hurbilenera biribiltzen dira rust-en irakurgailuak bezala.
  15 digitu arte eta gehienez 22ko berretzailearekin, biderketa edo zatiketa zehatz bakar batek egiten du.
  Gainerakoak zehazki zatitzen dira bigint-ekin eta erdia bikoitira biribiltzen, eta horrek zenbaki
  azpinormalak eta ehunka digituko literalak ere hartzen ditu.
- **Adierazpenak** `src/expr.rs`-en ordena berean ebaluatzen dira, beraz `-0`-k, adierazpen baten erdiko
  infinituek (`exp(ln(0))` 0 da) eta `NaN`-ek berdin jokatzen dute.
- **Testua** utf-8 gisa irakurtzen da eta baliogabea bada baztertu, `qoa compile`-ren mezu berarekin;
  hutsuneek eta letrek rust-ek definitzen duen unicode-ari jarraitzen diote (`π` pi da, `U+3000`-k tokenak
  bereizten ditu).

## Behar izan zituen aginduak

QOAn mihiztatzaile bat idazteko agindu klasiko batzuk behar ziren, QOA 0.4.3-n gehituak eta tresna gisa
lan egiten duen edozein programarentzat erabilgarriak:

| agindua | zer egiten du |
|---|---|
| `LOAD helmuga helbidea`, `STORE iturria helbidea` | memoria: 0, 1, 2, ... gelaxkak (2^32 arte), bakoitzean zenbaki bat; idatzi gabeek 0 dute |
| `STORETEXT helbidea "testua"` | testu baten byteak memorian idatzi (tauletarako eta mezuetarako) |
| `GETC helmuga`, `PUTC iturria` | byte bat irakurri sarrera estandarretik (-1 amaieran), byte bat idatzi bere horretan |
| `STREAM 1`, `STREAM 2` | irteera irteera estandarrera edo errore-irteerara bidali |
| `ARGC helmuga`, `ARGBYTE helmuga arg i` | programaren argumentuak (`qoa run prog.qx -- a b`) |
| `GETLANG helmuga` | 1 QOA euskaraz ari bada, programak erabiltzailearen hizkuntzan erantzun dezan |
| `EXIT n` | irteera-egoera batekin gelditu |
| `SIN`, `COS`, `TAN`, `POW` | adierazpenek erabiltzen dituzten gainerako funtzioak |

QOA 0.4.4-k parentesi arteko zenbakiak (`ADD R1 R1 (1)`), `CONST` eta `RREG` gehitu zituen, eta QOAn
idatzitako mihiztatzaileak ulertu eta erabili ere egiten ditu. [ISA.eu.md](ISA.eu.md)-k guztiak deskribatzen
ditu.

0.4.5etik mihiztatzaile biek `.qx` formatuaren 4. bertsioa idazten dute, aurreko guztiaren SHA-256-arekin
amaitzen dena (ikus [safety.eu.md](safety.eu.md#fitxategien-osotasuna)). QOAn idatzitako mihiztatzaileak
SHA-256 hori berak kalkulatzen du, QOAn, goiko bit-eragiketekin, eta iturburu zigilatuen `SEAL` lerroak modu
berean egiaztatzen ditu. Formatu zaharretako fitxategiak (1 0.4.2-rako, 2 0.4.3-ko aginduekin, 3 parentesi
arteko zenbakiekin) kargatzen dira oraindik.

## Mugak

- Mihiztatzaile bera glifo-sintaxian ere badago: [src/qoa/asm.qg](../src/qoa/asm.qg), `src/qoa/asm.qoa`-tik lerroz
  lerro sortua `qoa export --to glyph --keep-source`-rekin (iruzkin guztiak gordeta). Proba batek egiaztatzen du
  biek programa bera ematen dutela, byte bat aldatu gabe, beraz `qoa compile src/qoa/asm.qg`-k mihiztatzaile
  bera eraikitzen du. `src/qoa/asm.qoa` da editatu beharreko fitxategia.
- QOA arrunta irakurtzen du (`.qoa`). Glifo (`.qg`) eta OpenQASM (`.qasm`) iturburuak rust-etik pasatzen dira
  oraindik.
- Memoria guztia gordetzen du amaitu arte (megabyte gutxi batzuk bere iturburuarentzat).
- `sin`, `cos`, `tan`, `exp`, `ln` eta `pow` makinarenak dira, `qoa compile`-n bezala. Bi mihiztatzaileak bat
  datoz makina berean; beste makina batean biek biribil ditzakete funtzio gutxi horiek desberdin azken bitean.

## QOA-ri agindu bat gehitzen zaionean

1. Gehitu `src/isa.rs`-en eta `src/asm.rs`-en ohi bezala.
2. Sortu berriz taulak: `QOA_REGENERATE=1 cargo test --test selfhost`.
3. `src/qoa/asm.qoa`-n, eman eraikitzaile bat: lerro bat `buildtext`-en (izena, eraikitzaile-zenbakia,
   parametroa) eta, lehendik dagoen eraikitzaile bat egokitzen ez bada, berri bat `build`-en.
4. `cargo test --test selfhost`-ek bi mihiztatzaileak konparatzen ditu berriz; huts egiten du bat etorri arte.
