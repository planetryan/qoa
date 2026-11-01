# QOA-k QOA eraikitzen du

[In English: self-hosting.md](self-hosting.md)

QOA-ren mihiztatzailea, `.qoa` testu bat `.qx` bitar bihurtzen duen zatia, QOAn bertan idatzita ere badago:
[tools/asm.qoa](../tools/asm.qoa). QOA makinan exekutatzen da beste edozein QOA programa bezala, eta
`qoa compile`-k sortzen duen gauza bera sortzen du: programa guztietarako byte berberak, eta akatsak dituen
programa batentzat errore-mezu berberak, ordena berean, hitzez hitz, ingelesez eta euskaraz.

Bere burua ere mihiztatzen du. Horixe esan nahi du hemen "QOA-k QOA eraikitzen du": rust-en mihiztatzaileak
eraikitako mihiztatzailearen kopia eta hark bere buruaz eraikitzen duena berdinak dira, bytez byte.

## Erabilera

```sh
qoa compile --assembler tools/asm.qoa programa.qoa       # emaitza bera: qoa compile programa.qoa
```

Mihiztatzailearen iturburua biltegiko `tools/asm.qoa` da, eta QOA 0.4.4-tik aurrera argitalpen-pakete
guztiek ere badakarte.

`--assembler`-ek edozein QOA programa hartzen du, iturburu gisa edo `.qx` konpilatu gisa. QOA-k exekutatzen
du iturburu-fitxategia sarreran eta fitxategi-izena argumentu gisa emanda, eta inprimatzen duena `.qx`
gisa idazten du. Haren errore-mezuak zuzenean terminalera doaz, eta haren irteera-egoera `qoa`-rena bihurtzen
da.

Gauza bera eskuz, ezkutuko ezer ez dagoela ikusteko:

```sh
qoa compile tools/asm.qoa -o asm.qx                       # mihiztatzailea eraiki (rust-ek behin)
qoa run asm.qx --quiet -- programa.qoa < programa.qoa > programa.qx
```

`--quiet`-ek programak idazten duena bakarrik inprimatzen du, txostenik gabe, eta `qoa` programaren `EXIT`
egoerarekin amaitzen da. `--` ondoko guztia programari argumentu gisa ematen zaio.

## Abiaraztea

```sh
qoa compile tools/asm.qoa -o stage1.qx                                   # rust-ek eraikia
qoa compile --assembler stage1.qx tools/asm.qoa -o stage2.qx             # QOA-k eraikia
qoa compile --assembler stage2.qx tools/asm.qoa -o stage3.qx             # QOA-k eraikitako QOA-k eraikia
cmp stage1.qx stage2.qx && cmp stage2.qx stage3.qx                       # guztiak berdinak
```

Proba-multzoak hauxe bera egiten du eraikitze bakoitzean (`tests/selfhost.rs`).

## Zer dagoen QOAn idatzita, eta zer geratzen den rust-en

| zatia | lengoaia | zergatik |
|---|---|---|
| mihiztatzailea: lerroak, iruzkinak, tokenak, etiketak, izendun erregistroak, `q0..q3` tarteak, eragigai guztiak, adierazpenak, zenbaki hamartarrak, formatu bitarra, errore-mezu guztiak | **QOA** | testua eragiketa-zerrenda bihurtzen duen programa klasiko bat da, lengoaia batek bere buruarentzat egin dezakeena |
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

Mihiztatzailearen iturburu propioan neurtuta (11.000 lerro), 4 nukleoko hodeiko makina batean:

| | rust (`src/asm.rs`) | QOA (`tools/asm.qoa`) |
|---|---|---|
| irteera | erreferentzia | byte berberak, mezu berberak |
| denbora | 23 ms | 405 ms (17 aldiz motelago, gutxi gorabehera) |
| abiadura | 470.000 lerro segundoko inguru | 27.000 lerro segundoko inguru |
| tamaina | 2.700 lerro inguru mihiztatzailearentzat eta adierazpenentzat, gehi agindu-taula | 10.700 lerro inguru, gehi rust-etik sortutako 300 lerro inguru taula |

Beraz rust 17 aldiz azkarragoa da eta 4 aldiz laburragoa logika berarentzat. Hori da interpretatzaile
batean mihiztatze-estiloko kodearen kostua: QOA-k 32 zenbaki-erregistro, memoria-gelaxka zenbakituak eta
jauziak ditu, eta ez mota, egitura edo testu-katerik. Eremu-irakurketa, konparazio eta konstante bakoitza
agindu bereizi bat da. Trukean, mihiztatzaile osoa QOA programa bat da, eta QOA exekutatzen duen edozerk
QOA-ren programak eraiki ditzake berriz. Segundoko 27.000 lerroko mihiztatzailea nahikoa azkarra da edozein
QOA programa errealentzat: biltegiko programa handiena mihiztatzailea bera da, eta segundo erdi baino
gutxiagoan eraikitzen da.

## Zergatik datozen bat biak

- **Biltegiko programa guztiak**: adibideak eta proba-programa guztiak, huts egiteko egindako 58ak barne,
  biekin mihiztatzen dira ingelesez eta euskaraz, irteera-egoera, mezuak eta `.qx` byteak konparatuz.
- **Sortutako programak**: onartutako 7.000 lerro inguru agindu, ezizen eta eragigai-forma guztiekin, eta
  biribiltze zuzena probatzen duten ausazko 3.000 zenbaki hamartar (luzeak, txiki-txikiak, erdiko kasuak,
  azpinormalak); gehi akatsez betetako 165 programa, 50 errore-mezu desberdin baino gehiago, utf-8 baliogabea,
  Windows eta Mac zaharraren lerro-amaierak eta aldi berean 30 errore baino gehiago ukitzen dituztenak.
- Goiko **puntu finkoa**.
- **Taulak ezin dira aldendu**: `tools/asm.qoa` barruko agindu-taula (izenak, ezizenak, eragigaiak,
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

Horietako bat erabiltzen duen programa `.qx` formatuaren 2. bertsioan idazten da; beste programa guztiak 1.
bertsioan geratzen dira, eta QOA 0.4.2-k irakurtzen ditu oraindik. QOA 0.4.4-k parentesi arteko zenbakiak
(`ADD R1 R1 (1)`), `CONST` eta `RREG` gehitu zituen, eta QOAn idatzitako mihiztatzaileak ulertu eta erabili
ere egiten ditu; parentesi arteko zenbaki bat duen programa formatuaren 3. bertsioa da. [ISA.eu.md](ISA.eu.md)-k
guztiak deskribatzen ditu.

## Mugak

- QOA arrunta irakurtzen du (`.qoa`). Glifo (`.qg`) eta OpenQASM (`.qasm`) iturburuak rust-etik pasatzen dira
  oraindik.
- Memoria guztia gordetzen du amaitu arte (megabyte gutxi batzuk bere iturburuarentzat).
- `sin`, `cos`, `tan`, `exp`, `ln` eta `pow` makinarenak dira, `qoa compile`-n bezala. Bi mihiztatzaileak bat
  datoz makina berean; beste makina batean biek biribil ditzakete funtzio gutxi horiek desberdin azken bitean.

## QOA-ri agindu bat gehitzen zaionean

1. Gehitu `src/isa.rs`-en eta `src/asm.rs`-en ohi bezala.
2. Sortu berriz taulak: `QOA_REGENERATE=1 cargo test --test selfhost`.
3. `tools/asm.qoa`-n, eman eraikitzaile bat: lerro bat `buildtext`-en (izena, eraikitzaile-zenbakia,
   parametroa) eta, lehendik dagoen eraikitzaile bat egokitzen ez bada, berri bat `build`-en.
4. `cargo test --test selfhost`-ek bi mihiztatzaileak konparatzen ditu berriz; huts egiten du bat etorri arte.
