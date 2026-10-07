# QOA-ren glifo-sintaxia (0. zirriborroa)

[In English: glyph.md](glyph.md)

**Egoera: berrikusteko zirriborro bat.** Inplementatuta eta probatuta dago, beraz proba daiteke, baina
beheko aukera guztiak proposamenak dira. Amaierako galdera irekiak zuk hartu beharreko erabakiak dira.

## Zergatik

QOA arruntaren sintaxia lerro bakoitzeko agindu bat da, osorik idatzita: `CNOT q0 q1`. Zirriborro honen
abiapuntu den kontzeptu-zirriborroak ideia oso bat lerro bakarrean biltzen du:

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

## Kontzeptua irakurtzea

| kontzeptua | 0. zirriborroaren irakurketa |
|---|---|
| `H/R{3}` | `R{3}`-k 3 qubiteko erregistro bat adierazten du; `H/R`-k H aplikatzen die bakoitzari |
| `[0xMEMORY_LOCATION?]` | `[k]`-k erregistro baten k elementua hautatzen du (hamaseitarra onartzen da: `R[0x2]`) |
| `$>#2` | `$>`-k emaitza bat bideratzen du ("hona doa"); `#2` R2 erregistro klasikoa da |
| `PF"Measuring:"` | `PF` QOA-n PRINTF-ren ezizena da jada; testua zuzenean dator |
| `%O`, `%3` | `%`-k inprimatzeko balio bat eransten du (`PF"{}"%#2`); modu-adierazpen batean, muga (`~L{2}%6`) |
| `<<>>` | egoera osoa erakutsi (DUMPSTATE) |
| `;` | adierazpen bat amaitzen du, beraz lerro batek asko izan ditzake |
| `END;` | gelditu (QOA-n HALT-en ezizena da jada) |
| `CC"V"%3`, `PR"R"%O` | oraindik esleitu gabe: ikus galdera irekiak |

[examples/concept.qg](../examples/concept.qg) kontzeptua bera da, 0. zirriborroko programa erabilgarri gisa
idatzita.

## Erreferentzia

### Adierazpenak eta iruzkinak

- Adierazpen bakoitza `;`-rekin amaitzen da. Lerro batean nahi adina adierazpen sartzen dira; lerro-jauziek
  ez dute ezer esan nahi.
- Lehen karakterea `;` duen lerroa iruzkina da (kontzeptuan bezala). `//`-k edonon hasten du iruzkin bat.

### Erregistroak

| glifoa | esanahia |
|---|---|
| `Q{4}` | Q izeneko 4 qubit. Qubitak adierazpen-ordenan zenbakitzen dira: lehen erregistroa q0-n hasten da |
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

**Zenbaki huts bat ez da inoiz helbide bat.** QOA arruntean `3`-k q3 esan nahi du qubit-leku batean eta R3
erregistro-leku batean; glifo-programa batean errorea da, `ADD/#1,#1,1`-ek "bat gehitu" dirudielako baina
R1 gehituko lukeelako. Idatzi `#1`, `Q[1]` edo `q1`.

### Aginduak

| glifoa | esanahia | QOA arrunta |
|---|---|---|
| `H/Q[0]` | edozein QOA agindu (edo ezizen) eragigaiei aplikatuta | `H q0` |
| `RZ(pi/4)/Q[0]` | parametroak parentesi artean, QOA arruntean eragigaien ondoren | `RZ q0 pi/4` |
| `M/Q[0]$>#C[0]` | `$>`-k emaitza bideratzen du | `MEASURE q0 -> c0` |
| `COUNT/L[0]$>#3` | | `COUNT o0 -> R3` |
| `PF"x = {}\n"%#3` | inprimatu, `%`-k `{}` bakoitzari bere balioa ematen dio | `PRINTF "x = {}\n" R3` |
| `PF"{}"%#C` | bit-erregistro bat zenbaki oso gisa inprimatzen da | `PACKBITS R31 c0 c1` eta `PRINTF "{}" R31` |
| `<<>>` | egoera zehatza erakutsi | `DUMPSTATE` |
| `<<#>>` | erregistro guztiak erakutsi | `DUMPREGS` |
| `izena:` | etiketa | `izena:` |
| `>izena` | jauzia | `JUMP izena` |
| `?x>izena` | jauzi x bit edo erregistroa 1 bada | `JUMPIFONE x izena` |
| `!x>izena` | jauzi 0 bada | `JUMPIFZERO x izena` |
| `*3{ ... }`, `*#2{ ... }` | bloke bat errepikatu | `LOOPSTART 3` ... `LOOPEND` |
| `&izena` | azpierrutina bati deitu | `CALL izena` |
| `^` | handik itzuli | `RETSUB` |
| `` `edozein QOA arrunt` `` | ihesa: atzera-komatxoen arteko testua dagoen bezala erabiltzen da | bera |
| `END` | gelditu | `HALT` |

QOA arruntak ez dituen izen laburrak: `M` MEASURE-rentzat eta `CX` CNOT-rentzat. Beste agindu guztiek beren
QOA izena edo ezizena erabiltzen dute (`qoa isa`-k zerrendatzen ditu). Bit-erregistro bat inprimatzeak R31
erabiltzen du (eta R30, ... adierazpen bereko balio gehiagorentzat) aldi baterako, beraz glifo-programek
goiko erregistroak libre utzi behar lituzkete.

## Adibideak

[examples/](../examples): `concept.qg`, `bell.qg`, `teleport.qg`, `hom.qg` (bi igorle eta izpi-zatitzaile
bat), `retry.qg`. Bakoitzak bere emaitza esperatuak goiburuko iruzkinetan ditu, eta proba-multzoak
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

## Galdera irekiak

1. **Bereizleak.** Kontzeptuak adierazpenak `;` gabe kateatzen ditu (`$>#2PF"Measuring:"`). 0. zirriborroak
   `;` eskatzen du, errore batek beti zehazki esan dezan non amaitu zen adierazpen bat. Agindu-izen berri batek
   aurreko adierazpena bere kabuz amaitu beharko luke?
2. **`CC` eta `PR`.** Zer esan nahi zuten `CC"V"%3`-k eta `PR"R"%O$>#O`-k? (Asmakizun bat: CC = kopia
   klasikoa, PR = erregistroa inprimatu.) Esanahia finkatzen denean glifo bihur daitezke.
3. **`%`.** Kontzeptuak `%3` eta `%O` kopuru edo formatu gisa erabiltzen ditu; 0. zirriborroak `%`
   inprimatzeko balioetarako eta moduen mugarako bakarrik erabiltzen du. Lehentasuna izan beharko lukeen
   esanahiren bat zenuen buruan?
4. **`[0x…]`.** 0. zirriborroak elementu-indize gisa hartzen du. Helbide gordin bat esan nahi zenuen
   (egoera-bektoreko posizio bat, edo memoria klasikoa)?
5. **Izenak QOA arruntean.** Erregistro izendunek glifo-forma irakurgarri egiten dute. QOA arruntak ere
   izan beharko lituzke (`QREG R 3` eta `R[0]`)?
6. **Beste norabidea.** `qoa export --to glyph` (QOA arruntetik glifora) oraindik ez dago idatzita.
