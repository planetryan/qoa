# QOA-ren glifo-sintaxia

[In English: glyph.md](glyph.md)

**Egoera: esperimentala.** Sintaxia inplementatuta eta probatuta dago, baina geroko bertsio batean alda
daiteke; aldaketa oro bertsio-oharretan agertuko da. Oraindik finkatu gabeko puntuak amaieran daude.

## Zergatik

QOA arruntaren sintaxia lerro bakoitzeko agindu bat da, osorik idatzita: `CNOT q0 q1`. Glifo-sintaxia lerro
bakarreko zirriborro batetik abiatu zen, ideia oso bat lerro bakarrean biltzen duena:

```
H/R{3}[0xMEMORY_LOCATION?]; CC"V"%3$>#2PF"Measuring:" PR"R"%O$>#OPF"Result:" PF"O"%O$<<>>; END;
```

Glifo-sintaxiak dentsitate hori gordetzen du, eta gauzak nora doazen hain zuzen esaten du. **Ez da bigarren
lengoaia bat**: glifo-adierazpen bakoitza QOA agindu arrunt bihurtzen da, `qoa import fitxategia.qg`-k
zehazki zein inprimatzen ditu, eta erroreek eragin dituen glifoa seinalatzen dute.

```sh
qoa run examples/bell.qg        # glifo-programa bat exekutatu
qoa import examples/bell.qg     # ordezkatzen duen QOA arrunta erakutsi
qoa check examples/bell.qg      # mihiztatu eta jakinarazi, exekutatu gabe
```

## Lehen programa bat

```
Q{2}; #C{2}; H/Q[0]; CX/Q[0],Q[1]; M/Q$>#C;
```

honela irakurtzen da: "bi qubit Q, bi bit C; H Q[0]-n; CNOT Q[0]-tik Q[1]-era; neurtu Q, C-ra", eta
programa hau bera da:

```
QINIT 2
H q0
CNOT q0 q1
MEASURE q0 -> c0
MEASURE q1 -> c1
```

## Jatorrizko zirriborroa irakurtzea

| zirriborroa | esanahia |
|---|---|
| `H/R{3}` | `R{3}`-k 3 qubiteko erregistro bat adierazten du; `H/R`-k H aplikatzen die bakoitzari |
| `[0xMEMORY_LOCATION?]` | adierazpen baten ondoren, kokapen bat: `R{3}[0x4]`-k R q4..q6-n jartzen du. Izen baten ondoren, elementu bat: `R[0x2]` |
| `$>#2` | `$>`-k emaitza bat bideratzen du ("hona doa"); `#2` R2 erregistro klasikoa da |
| `PF"Measuring:"` | `PF` QOA-n PRINTF-ren ezizena da jada; testua zuzenean dator |
| `%O`, `%3` | `%`-k balio bat hasten du eta hurrengotik bereizten du: `PF"{} {}"%#2%#3`, `CC[0x10]%3%#1`; modu-adierazpen batean muga hasten du (`~L{2}%6`) |
| `<<>>` | egoera osoa erakutsi (DUMPSTATE) |
| `;` | adierazpen bat amaitzen du, beraz lerro batek asko izan ditzake |
| `END;` | gelditu (QOA-n HALT-en ezizena da jada) |
| `CC"V"%3$>#2` | dei klasikoa: V errutina deitu 3 argumentuarekin eta emaitza R2-n utzi (ikus [Dei klasikoak](#dei-klasikoak-cc)) |
| `PR"R"%O` | etiketarekin inprimatu: `R = ` eta balioa (ikus [PR](#042-n-finkatua)) |

[examples/concept.qg](../examples/concept.qg) zirriborro hori bera da, programa erabilgarri gisa idatzita.

## Erreferentzia

### Adierazpenak eta iruzkinak

- Adierazpen bakoitza `;`-rekin amaitzen da. Lerro batean nahi adina adierazpen sartzen dira; lerro-jauziek
  ez dute ezer esan nahi.
- Lehen karakterea `;` duen lerroa iruzkina da (jatorrizko zirriborroan bezala). `//`-k edonon hasten du iruzkin bat.

### Erregistroak

| glifoa | esanahia |
|---|---|
| `Q{4}` | Q izeneko 4 qubit. Qubitak adierazpen-ordenan zenbakitzen dira: lehen erregistroa q0-n hasten da |
| `Q{4}[0x4]` | bera, kokapen batean: Q q4..q7 da. Kokapenik gabeko hurrengo erregistroak kokatutako altuenaren ondoren doaz; gainjartzen diren erregistroak errorea dira |
| `#C{4}` | C izeneko 4 bit klasiko |
| `~L{2}%6` | L izeneko 2 modu optiko, bakoitza 0..5 fotoirekin (modu-erregistro guztiek muga bera; lehenetsia 8) |

Adierazpen bat bakarrik egon daiteke (`Q{4};`) edo lehen erabileran agertu (`H/Q{4};`). Ondoren izena
giltzarik gabe erabiltzen da.

### Eragigaiak

| glifoa | esanahia | QOA arrunta |
|---|---|---|
| `Q` | Q-ren elementu guztiak | `q0`, `q1`, ... |
| `Q[2]`, `Q[0x2]` | elementu bat | `q2` |
| `Q[1..3]` | 1etik 3ra arteko elementuak, biak barne | `q1`, `q2`, `q3` |
| `#C`, `#C[0]` | bitak | `c0`, ... |
| `#2` | 2. erregistro klasikoa | `R2` |
| `(pi/4)`, `1.5` | zenbaki edo adierazpen bat | bera |
| `q3`, `o1`, `c0`, `R2` | QOA arrunteko eragigaiak dauden bezala onartzen dira | bera |

**Erregistro osoak elementuetan zabaltzen dira.** `H/Q` H bat da qubit bakoitzeko. Eragigai batzuekin,
erregistroek tamaina bera izan behar dute eta elementuz elementu doaz, eta elementu bakarrak errepikatu
egiten dira: `A{2}` eta `B{2}`-rekin, `CX/A,B` `CNOT q0 q2` eta `CNOT q1 q3` da.

**Zenbaki huts bat ez da inoiz helbide bat.** Glifo-programa batean zenbaki bat zenbaki bat da:
`ADD/#1,#1,1`-ek bat gehitzen du, `IFEQ/#1,10,done`-k R1 hamarrekin konparatzen du, eta `STORE/7,#3`-k
zazpi gordetzen du R3-ko helbidean (QOA arruntean `ADD R1 R1 (1)`, `IFEQ R1 (10) done` eta `STORE (7) R3`).
Qubit bat, modu bat, bit bat edo idazteko erregistro bat bakarrik egokitzen den lekuan, zenbaki huts bat
errorea da, QOA arruntak `3` q3 edo R3 gisa irakurriko lukeelako hor. Idatzi `#1`, `Q[1]` edo `q1`.

### Aginduak

| glifoa | esanahia | QOA arrunta |
|---|---|---|
| `H/Q[0]` | edozein QOA agindu (edo ezizen) eragigaiei aplikatuta | `H q0` |
| `RZ(pi/4)/Q[0]` | parametroak parentesi artean, QOA arruntean eragigaien ondoren | `RZ q0 pi/4` |
| `M/Q[0]$>#C[0]` | `$>`-k emaitza bideratzen du | `MEASURE q0 -> c0` |
| `COUNT/L[0]$>#3` | | `COUNT o0 -> R3` |
| `PF"x = {}\n"%#3` | inprimatu, `%`-k `{}` bakoitzari bere balioa ematen dio | `PRINTF "x = {}\n" R3` |
| `PF"{}"%#C` | bit-erregistro bat zenbaki oso gisa inprimatzen da | `PACKBITS R31 c0 c1` eta `PRINTF "{}" R31` |
| `PR"x"%#3` | etiketa bat eta balioak inprimatu: `x = 3.5` | `PRINTF "x = {}\n" R3` |
| `<<>>` | egoera zehatza erakutsi | `DUMPSTATE` |
| `<<#>>` | erregistro guztiak erakutsi | `DUMPREGS` |
| `izena:` | etiketa | `izena:` |
| `0x10:` | helbide bat den etiketa, `CC[0x10]`-rentzat | `addr_0x10:` |
| `>izena` | jauzia (izen edo helbide batera: `>0x10`) | `JUMP izena` |
| `?x>izena` | jauzi x bit edo erregistroa 1 bada | `JUMPIFONE x izena` |
| `!x>izena` | jauzi 0 bada | `JUMPIFZERO x izena` |
| `*3{ ... }`, `*#2{ ... }` | bloke bat errepikatu | `LOOPSTART 3` ... `LOOPEND` |
| `&izena` | azpierrutina bati deitu (`&0x10` ere bai) | `CALL izena` |
| `CC[0x10]%a%b$>#2` | dei klasikoa: argumentuak pilan, deia, emaitza R2-ra | `PUSH`... `CALL addr_0x10` `POP R2` |
| `^` | handik itzuli | `RETSUB` |
| `` `edozein QOA arrunt` `` | ihesa: atzera-komatxoen arteko testua dagoen bezala erabiltzen da | bera |
| `END` | gelditu | `HALT` |

QOA arruntak ez dituen izen laburrak: `M` MEASURE-rentzat eta `CX` CNOT-rentzat. Beste agindu guztiek beren
QOA izena edo ezizena erabiltzen dute (`qoa isa`-k zerrendatzen ditu). Bit-erregistro bat inprimatzeak R31
erabiltzen du (eta R30, ... adierazpen bereko balio gehiagorentzat) aldi baterako, beraz glifo-programek
goiko erregistroak libre utzi behar lituzkete.

### Dei klasikoak (CC)

`CC`-k programa kuantikoaren balioak errutina klasiko bati ematen dizkio eta bat itzultzen du, dei-
konbentzio (ABI) txiki baten bidez QOA-ren datu-pilan, dei-pilatik bereizia dena:

1. deitzaileak argumentuak (`%a%b`) pilan sartzen ditu azkenetik lehenera, beraz errutinak lehen argumentua
   ateratzen du lehenik;
2. `CC[0x10]`-k `0x10` helbideko errutina deitzen du (`0x10:` idatzitako etiketa bat), eta `CC"izena"`-k
   izendun etiketa batekoa;
3. errutinak bere argumentuak ateratzen ditu (`POP/#1`), kalkulatzen du, emaitza bat sartzen du (`PUSH/#4`)
   eta itzultzen da (`^`);
4. `$>#2`-k emaitza R2-ra ateratzen du. `$>` gabe, errutinak ez luke ezer sartu behar.

Argumentu bat zenbaki edo adierazpen bat (`%3`, `%(pi/2)`), erregistro bat (`%#1`) edo bitak (`%#C[0]`, edo
`%#C` bit-erregistro oso bat zenbaki bakar gisa) izan daiteke; qubitek eta moduek ez dute balio klasikorik
neurtu arte, beraz bat pasatzea errorea da. Zenbakiak eta bitak R31-etik pasatzen dira pilara bidean.

[examples/decoder.qg](../examples/decoder.qg)-k hiru qubiteko errepikapen-kode bat deskodetzen du: qubitak
neurtzen ditu eta `0x10`-ko gehiengo-bozketako errutina deitzen du:

```
M/Q$>#C; CC[0x10]%#C[0]%#C[1]%#C[2]$>#5; PF"deskodetua: {}\n"%#5; END;
0x10:; POP/#1; POP/#2; POP/#3; ADD/#4,#1,#2; ADD/#4,#4,#3; REGSET/#6,(1); SHR/#4,#4,#6; PUSH/#4; ^;
```

## QOA arruntetik glifora

`qoa export --to glyph fitxategia.qoa`-k (edo `--to-glyph`) edozein programa glifo-sintaxian idazten du:
`.qoa`, `.qasm`, `.qg` edo konpilatutako `.qx` batetik. Ate beraren segidak tarte bihurtzen dira (`H/Q[0..3]`,
`M/Q[0..1]$>#C[0..1]`), jauzi-helburuak helbide-etiketa (`0x8:`), eta emaitza berriz mihiztatu eta
jatorrizkoarekin konparatzen da idatzi aurretik. Aukerak:

| aukera | eragina |
|---|---|
| `--width N` | adierazpenek lerroak betetzen dituzte N karaktere arte (lehenetsia 100; 0 = adierazpen bat lerroko) |
| `--names Q,C,L` | qubit, bit eta modu erregistroen izenak |
| `--long-names` | `MEASURE` eta `CNOT`, `M` eta `CX` ordez |
| `--no-ranges` | adierazpen bat elementu bakoitzeko |
| `--hex` | indizeak hamaseitarrean, `Q[0x3]` bezala |

## Adibideak

[examples/](../examples): `concept.qg`, `bell.qg`, `decoder.qg` (dei klasiko bat), `teleport.qg`,
`hom.qg` (bi igorle eta izpi-zatitzaile bat), `retry.qg`. Bakoitzak bere emaitza esperatuak goiburuko iruzkinetan ditu, eta proba-multzoak
egiaztatzen ditu; `tests/glyph.rs`-k gainera egiaztatzen du glifo-programek beren biki arrunten kopuru
berdin-berdinak ematen dituztela.

Teleportazioa, glifoa eta arrunta alboz albo:

```
Q{3}; #C{3};                                    QINIT 3
U(1.0, 0.5, -0.3)/Q[0];                         U q0 1.0 0.5 -0.3
BELL/Q[1],Q[2]; CX/Q[0],Q[1]; H/Q[0];           BELL q1 q2 / CNOT q0 q1 / H q0
M/Q[0..1]$>#C[0..1];                            MEASURE q0 -> c0 / MEASURE q1 -> c1
!#C[1]>x_gabe; X/Q[2]; x_gabe:;                 JUMPIFZERO c1 x_gabe / X q2 / x_gabe:
!#C[0]>z_gabe; Z/Q[2]; z_gabe:;                 JUMPIFZERO c0 z_gabe / Z q2 / z_gabe:
U(-1.0, 0.3, -0.5)/Q[2]; M/Q[2]$>#C[2];         U q2 -1.0 0.3 -0.5 / MEASURE q2 -> c2
```

## Diseinu-aukerak

- **`CC`** "dei klasikoa" da, memoria-kokapen baterako deia: `CC[0x10]`, goiko pila-konbentzioarekin.
- **`%`** balioen hasiera eta bereizle logikoa da.
- **`[0x…]`** kokapen bat da adierazpen baten ondoren, eta elementu-indize bat beste leku guztietan.
- **Izenak QOA arruntean** erabilgarri daude baina ez dira inoiz derrigorrezkoak: `QREG data 3`,
  `CREG flags 2`, `OREG arm 2`, gero `data[1]` edo `data`; `q0`, `c0` eta `o0` erabiltzen jarraitzen dute.
- **Beste norabidea** badago: `qoa export --to glyph` (edo `--to-glyph`), goiko aukerekin.

## 0.4.2-n finkatua

1. **Bereizleak: `;` beharrezkoa da.** Jatorrizko zirriborroak adierazpenak hura gabe kateatzen ditu
   (`$>#2PF"Measuring:"`). Agindu-izen berri batek ezin du aurreko adierazpena bere kabuz amaitu, izenak libreak
   direlako: `PF` edo `M` erregistro bat izan daiteke agindu bat bezala, beraz `#PF` eta `#P` gero `F...`
   berdin irakurriko lirateke. `;`-rekin adierazpen bakoitzak amaiera argia du, eta errore batek beti esan
   dezake zein adierazpenetan dagoen.
2. **`PR`-k etiketarekin inprimatzen du.** `PR"x"%#3`-k `x = 3` inprimatzen du, eta balio batzuk elkarren
   ondoan ateratzen dira (`PR"bikotea"%#1%#2`-k `bikotea = 1 2`). Balioak `PF`-n bezala dabiltza: zenbakiak,
   erregistroak, bitak, eta bit-erregistroak zenbaki oso gisa. Qubitek eta moduek ez dute baliorik neurtu
   arte, beraz errore bat dira. `PR"hemen"` bakarrik idatzita etiketa bakarrik inprimatzen du.
