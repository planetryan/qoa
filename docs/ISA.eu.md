# QOA aginduen erreferentzia

`src/isa.rs`-tik sortua `qoa isa --markdown` bidez. Ez editatu eskuz.

Mnemonikoetan berdin dio maiuskula edo minuskula. Ezizen bakoitzak bere izen nagusiak egiten duen gauza bera egiten du.

## Forma laburrak

QOA 0.4.5etik agindu bakoitzak izen labur bat du (*Laburra* zutabea), idazteko modu hobetsia: `M q0` `MEASURE q0` da. Beste hiru forma laburrek, QOA arruntean eta glifo-sintaxian, programa bera ematen dute karaktere gutxiagorekin:

| Laburra | Luzea | Araua |
|---|---|---|
| `Mq0` | `MEASURE q0` | agindu batek bere lehen eragigaia uki dezake, eragigai hori bere motako erregistro arrunt edo tarte bat denean (`q3`, `o1`, `c0`, `R2`, `q0..q3`) |
| `M q0.c0` | `MEASURE q0 -> c0` | `.`-k emaitza bat bideratzen du, `->` bezala |
| `Mq0.c0` | `MEASURE q0 -> c0` | biak batera |

0.4.5 aurretik ezizen ez ziren izen laburrek (`D`, `K`, `N`, `E`, `C`, `J`, ...) ez dute inoiz izenik erreserbatzen: `L` edo `C` izeneko erregistro edo konstante batek funtzionatzen jarraitzen du. Forma luzeek leku guztietan funtzionatzen jarraitzen dute.

## Eragigaiak

| Idazkera | Esanahia |
|---|---|
| `q3` | qubita: q3 (edo 3), edo data[1] QREG ondoren; q0..q3 bezalako tarteak onartzen diren lekuetan |
| `o1` | modu optikoa: o1 (edo 1) |
| `R2` | erregistroa: R2 (edo 2), R0..R31 |
| `R2` / `(10)` | erregistroa edo zenbakia: R2 (edo 2), edo parentesi arteko zenbaki bat, (10), (pi/2) edo (newline) bezala |
| `c0` / `c0..c3` | bit klasikoa: c0, edo c0..c3 bezalako tarte bat |
| `-> c0` / `-> R2` | emaitzaren helmuga: c0 (bit klasikoa) edo R2 (erregistroa), nahi bada -> ondoren |
| `c0` / `R2` | balioa: c0 (bit klasikoa) edo R2 (erregistroa) |
| `pi/2` | zenbakia edo adierazpena: 0.5, pi/2, -3*pi/4, sqrt(2)/2 |
| `8` / `R1` | kopurua: zenbaki osoa edo erregistroa |
| `done` | etiketa |
| `"text"` | testua komatxo bikoitzen artean |
| `X` | oinarria edo ardatza: X, Y edo Z |
| `"A"` | karakterea: "A" edo 65 bezalako karaktere-kode bat |
| `data` | izena: letrak, digituak eta _, ez agindu-izen bat |

*Simulagailuan bakarrik* markatutako aginduek benetako gailu batek egoera aldatu gabe ikusi ezin duen informazioa irakurtzen dute.

## Prestaketa

| Agindua | Laburra | Zer egiten du | Ezizenak |
|---|---|---|---|
| `QINIT qubitak` | `QI` | Programak zenbat qubit erabiltzen dituen adierazi (behin). Gabe, erabilitako qubit-indize handiena gehi bat da. | QI, INITQUBIT, IQ, QINITQ |
| `QMODE moduak [muga]` | `QO` | Modu optikoak adierazi, bakoitzak 0..muga-1 fotoi gordetzen dituela (muga lehenetsia 8). Mugaz kanpora doan probabilitatea exekuzioaren ondoren jakinarazten da. | MODES |
| `QREG izena size [first]` | `QREG` | Qubitei izena eman (aukerakoa): QREG data 3 ondoren, data[0]..data[2]-k (edo data-k, hirurak) izenik gabeko hurrengo 3 qubitak adierazten dituzte, edo `first`-etik hasitakoak emanez gero. q0, q1, ... erabiltzen jarraitzen dute. |  |
| `CREG izena size [first]` | `CREG` | Bit klasikoei izena eman (aukerakoa), QREG bezala: CREG flags 2-k flags[0], flags[1] eta flags sortzen ditu. |  |
| `OREG izena size [first]` | `OREG` | Modu optikoei izena eman (aukerakoa), QREG bezala: OREG arm 2-k arm[0], arm[1] eta arm sortzen ditu. |  |
| `RREG izena size [first]` | `RREG` | Erregistro klasikoei izena eman (aukerakoa): RREG total 2 ondoren, total[0] eta total[1] (edo total, biak) izenik gabeko hurrengo 2 erregistroak dira, edo `first`-etik aurrerakoak ematen bada. R0..R31-ek ere balio dute. |  |
| `CONST izena balioa` | `CONST` | Zenbaki bati izena eman: CONST newline 10 ondoren, newline edozein adierazpenetan erabil daiteke (RX q0 newline/4, ADD R1 R1 (newline)). Konstante batek bere gainetik definitutakoak erabil ditzake. |  |

## Qubit-ateak

| Agindua | Laburra | Zer egiten du | Ezizenak |
|---|---|---|---|
| `H qubita` | `H` | Hadamard. Qubit-tarte bat onartzen du (q0..q3). | HAD, APPLYHADAMARD |
| `X qubita` | `X` | Pauli-X (bit-iraulketa). Qubit-tarte bat onartzen du (q0..q3). | APPLYBITFLIP |
| `Y qubita` | `Y` | Pauli-Y. Qubit-tarte bat onartzen du (q0..q3). |  |
| `Z qubita` | `Z` | Pauli-Z (fase-iraulketa). Qubit-tarte bat onartzen du (q0..q3). | APPLYPHASEFLIP |
| `S qubita` | `S` | S = diag(1, i). Qubit-tarte bat onartzen du (q0..q3). | APPLYSGATE |
| `SDG qubita` | `SDG` | S† = diag(1, -i). Qubit-tarte bat onartzen du (q0..q3). | SDAG |
| `T qubita` | `T` | T = diag(1, e^(i·pi/4)). Qubit-tarte bat onartzen du (q0..q3). | APPLYTGATE |
| `TDG qubita` | `TDG` | T† = diag(1, e^(-i·pi/4)). Qubit-tarte bat onartzen du (q0..q3). | TDAG |
| `SX qubita` | `SX` | X-ren erro karratua. Qubit-tarte bat onartzen du (q0..q3). | SQRTX |
| `SXDG qubita` | `SXDG` | SX-ren alderantzizkoa. Qubit-tarte bat onartzen du (q0..q3). | SXDAG |
| `PHASE qubita angelua` | `P` | Fase-atea diag(1, e^(i·angelua)). Qubit-tarte bat onartzen du (q0..q3). | P, PHASESHIFT, SETPHASE, SETP, SPH, PSE, U1 |
| `RX qubita angelua` | `RX` | Biraketa X ardatzean: exp(-i·angelua·X/2). Qubit-tarte bat onartzen du (q0..q3). |  |
| `RY qubita angelua` | `RY` | Biraketa Y ardatzean: exp(-i·angelua·Y/2). Qubit-tarte bat onartzen du (q0..q3). |  |
| `RZ qubita angelua` | `RZ` | Biraketa Z ardatzean: exp(-i·angelua·Z/2). Qubit-tarte bat onartzen du (q0..q3). |  |
| `U qubita theta phi lambda` | `U` | Qubit bakarreko ate orokorra U(theta, phi, lambda), OpenQASM-en konbentzioa. Qubit-tarte bat onartzen du (q0..q3). | U3 |
| `ROT qubita ardatza angelua` | `ROT` | Biraketa X, Y edo Z ardatzean (RX, RY edo RZ bera). Qubit-tarte bat onartzen du (q0..q3). | APPLYROTATION, AR |
| `MROT ardatza angelua qubitak...` | `MROT` | Biraketa bera zerrendako qubit guztietan. | APPLYMULTIQUBITROTATION, AMQR |
| `GATE izena qubita` | `G` | Izendun qubit bakarreko atea aplikatu: h, x, y, z, s, sdg, t, tdg, sx, sxdg. | APPLYGATE, AGATE, AG |
| `CNOT kontrola helburua` | `CX` | X kontrolatua: helburua irauli kontrola 1 denean. | CX, CONTROLLEDNOT, CN, ENTANGLE |
| `CY kontrola helburua` | `CY` | Y kontrolatua. |  |
| `CZ kontrola helburua` | `CZ` | Z kontrolatua: -1 fasea \|11>-n. |  |
| `CH kontrola helburua` | `CH` | Hadamard kontrolatua. |  |
| `CSX kontrola helburua` | `CSX` | X-ren erro karratu kontrolatua. |  |
| `CU kontrola helburua theta phi lambda` | `CU` | U(theta, phi, lambda) kontrolatua. | CU3 |
| `CPHASE kontrola helburua angelua` | `CP` | Fase kontrolatua: e^(i·angelua) \|11>-n. | CP, CONTROLLEDPHASEROTATION, APPLYCPHASE, CPR, CU1 |
| `CRX kontrola helburua angelua` | `CRX` | RX kontrolatua. |  |
| `CRY kontrola helburua angelua` | `CRY` | RY kontrolatua. |  |
| `CRZ kontrola helburua angelua` | `CRZ` | RZ kontrolatua. |  |
| `SWAP a b` | `SWAP` | Bi qubit trukatu. |  |
| `CSWAP kontrola a b` | `CS` | a eta b trukatu kontrola 1 denean (Fredkin). | CONTROLLEDSWAP, CS, FREDKIN |
| `CCX kontrola1 kontrola2 helburua` | `CCX` | Toffoli: helburua irauli bi kontrolak 1 direnean. | TOFFOLI, CCNOT |
| `MCX kontrolak... helburua` | `MCX` | Helburua (azken qubita) irauli zerrendako beste qubit guztiak 1 direnean. |  |
| `MCZ qubitak...` | `MCZ` | -1 fasea zerrendako qubit guztiak 1 diren oinarri-egoeretan. |  |

## Entrelazamendu-laburdurak

| Agindua | Laburra | Zer egiten du | Ezizenak |
|---|---|---|---|
| `BELL a b` | `B` | H a exekutatzen du, gero CNOT a b. \|00>-tik (\|00> + \|11>)/sqrt(2) ematen du. | ENTANGLEBELL, EBELL, EB |
| `GHZ qubitak...` | `GHZ` | H lehen qubitean, gero CNOT hartatik beste guztietara. \|0..0>-tik GHZ egoera bat ematen du. | ENTANGLEMULTI, EMULTI, EM |
| `CLUSTER qubitak...` | `CL` | Cluster egoera lineala: H qubit guztietan, gero CZ ondoko bikote bakoitzean. | ENTANGLECLUSTER, ECLUSTER, ECR |
| `ESWAP a b c d` | `ESP` | Entrelazamendu-trukaketa. (a,b) eta (c,d) bell bikoteekin: b eta c bell-neurtzen ditu (c<b> eta c<c>-ra), gero d zuzentzen du, eta (a,d) bell bikote bihurtzen da. | ENTANGLESWAP, ESP |

## Qubit-neurketa eta berrezarpena

| Agindua | Laburra | Zer egiten du | Ezizenak |
|---|---|---|---|
| `MEASURE qubita [-> helmuga]` | `M` | Z oinarrian neurtu. Emaitza c<qubit> bitera doa, -> cK edo -> RK bidez beste nonbait bideratzen ez bada. Qubit-tarte bat onartzen du (q0..q3). | MEAS, M, QMEAS, QM, APPLYQNDMEASUREMENT, AQND, AQAD, AQNM |
| `MEASUREIN qubita oinarria [-> helmuga]` | `MI` | X, Y edo Z oinarrian neurtu (Z-ra biratu, neurtu, atzera biratu). | MEASUREINBASIS, MEASB, MIB |
| `MEASUREALL` | `MA` | Qubit guztiak neurtu, bakoitza bere c<qubit> bitera. | MALL |
| `RESET qubita` | `RST` | Qubit bat \|0>-n jarri: neurtu, eta 1 irakurri badu irauli. Qubit-tarte bat onartzen du (q0..q3). | RST, RSTQ, QRESET, QR |
| `RESETALL` | `RA` | Qubit guztiak \|0>-ra eta modu guztiak hutsera. | RSTALL, RSA |
| `TOBASIS qubita oinarria` | `TB` | Biratu Z neurketa batek oinarri hori irakur dezan (X: H; Y: SDG eta gero H; Z: ezer ez). Qubit-tarte bat onartzen du (q0..q3). | APPLYMEASUREMENTBASISCHANGE, AMBC |

## Modu optikoak

| Agindua | Laburra | Zer egiten du | Ezizenak |
|---|---|---|---|
| `DISPLACE modua re [im]` | `D` | Desplazamendua D(alpha), alpha = re + i·im. Hutsa egoera koherente bihurtzen du, \|alpha\|^2 batez besteko fotoiekin. | APPLYDISPLACEMENT, ADISP, AD |
| `DISPLACEREG modua alpha` | `DV` | Erregistro bateko balioaren (erreala) desplazamendua: aurrera-elikadura klasikoa. | APPLYDISPLACEMENTFEEDBACK |
| `SQUEEZE modua r [phi]` | `SQ` | Estutzea S(r·e^(i·phi)). phi = 0 eta r > 0 direnean x koadratura estutzen da. | APPLYSQUEEZING, ASQ, AS |
| `SQUEEZEREG modua r` | `SQV` | Erregistro bateko balioaren estutzea: aurrera-elikadura klasikoa. | APPLYSQUEEZINGFEEDBACK, ASWF, ASF, ASFB |
| `MODEPHASE modua angelua` | `PH` | Modu baten fase-biraketa e^(i·angelua·n). | DYNAMICPHASECOMPENSATION, DPC, DPCMP, PHASEROT |
| `KERR modua kappa [denbora]` | `K` | Kerr ez-lineartasuna e^(i·kappa·denbora·n^2) (denbora lehenetsia 1). | APPLYKERRNONLINEARITY, AKNL |
| `CROSSKERR a b kappa` | `XK` | Kerr gurutzatua (fase-modulazio gurutzatua) e^(i·kappa·n_a·n_b). | CROSSPHASEMODULATION, CPM, CPMOD |
| `BS a b theta [phi]` | `BS` | Izpi-zatitzailea exp(theta·(e^(i·phi)·a·b† - e^(-i·phi)·a†·b)). theta = pi/4 50:50 da. | BEAMSPLITTER |
| `MODESWAP a b` | `MS` | Bi moduren edukiak trukatu (bideratzaile ideal bat). | OPTICALROUTING, OROUTE, OPTR |
| `ADDPHOTON modua` | `PA` | a† aplikatu eta normalizatu. Post-hautatua: benetako fotoi-gehiketak batzuetan bakarrik lortzen du. | PHOTONADDITION, PADD, PA |
| `SUBPHOTON modua` | `PS` | a aplikatu eta normalizatu (post-hautatua). Huts egiten du modua ziur hutsik badago. | APPLYPHOTONSUBTRACTION, APSUB, APS |
| `LOSS modua probabilitatea` | `L` | Moduko fotoi bakoitza probabilitate honekin galtzen da. Ausazko emaitza bat exekuzio bakoitzeko; shot-en batez bestekoa galera-kanal zehatza da. | PHOTONLOSS, PHOTONLOSSSIMULATE, PLS, PLSIM, SETOPTICALATTENUATION, SOATT, SOA |
| `FOCK modua fotoiak` | `F` | Moduak zuena baztertu eta zehazki fotoi-kopuru hori prestatu. |  |
| `SINGLEPHOTON modua` | `SP` | Fotoi bakarreko iturri ideala: FOCK modua 1 bera. | SINGLEPHOTONSOURCEON, SPSON |
| `COUNT modua [-> helmuga]` | `N` | Fotoi-kopurua bereizten duen detekzioa. Kopurua moduaren irakurketara doa, -> RK bidez bideratzen ez bada. | PHOTONCOUNT, PCOUNT, PC, PHOTONNUMBERRESOLVINGDETECTION, PNRD, PNR |
| `DETECT modua [-> helmuga]` | `PD` | Atalase-detekzioa (klik): 1 fotoiren bat badago, bestela 0. | PHOTONDETECT, PDETECT, PD |
| `PARITY modua [-> helmuga]` | `MP` | Fotoi-kopuruaren paritatearen neurketa: 0 bikoitia, 1 bakoitia. | MEASUREPARITY, MPAR, MP |
| `COINCIDENCE moduak... -> helmuga` | `CD` | Zerrendako modu guztiak klik-detektatu; 1 idazten du guztiek klik egin badute. | PHOTONDETECTCOINCIDENCE, PDCOIN, PDC |

## Qubit-modu akoplamendua

| Agindua | Laburra | Zer egiten du | Ezizenak |
|---|---|---|---|
| `EMIT qubita modua` | `E` | \|1>\|0> eta \|0>\|1> trukatu qubit baten eta modu baten artean: igorle kitzikatu batek bere kitzikapena fotoi gisa jartzen du moduan. Berriz exekutatuta fotoia xurgatzen du. | PHOTONEMIT, PEMIT, PE, ABSORB |
| `DISPERSIVE qubita modua angelua` | `DSP` | Akoplamendu dispertsiboa e^(i·angelua·n), qubita 1 den lekuetan bakarrik. |  |

## Ikuskapena (simulagailuan bakarrik)

| Agindua | Laburra | Zer egiten du | Ezizenak |
|---|---|---|---|
| `DUMPSTATE` | `DS` | *Simulagailuan bakarrik.* Puntu honetako egoera zehatza inprimatu. | DSTATE, DS |
| `EXPECT qubita oinarria [helmuga]` | `EV` | *Simulagailuan bakarrik.* Qubit baten <X>, <Y> edo <Z> zehatza, hura aldatu gabe. Inprimatuta, edo erregistro batean gordeta. | QUANTUMSTATETOMOGRAPHY, QST, QSTAT |
| `BELLCHECK a b [helmuga]` | `BC` | *Simulagailuan bakarrik.* (a, b) qubiten fideltasun zehatza (\|00> + \|11>)/sqrt(2)-rekin. | BELLSTATEVERIFICATION, BSV, BSTATE |
| `MEANPHOTON modua [helmuga]` | `MN` | *Simulagailuan bakarrik.* Modu baten batez besteko fotoi-kopuru zehatza <n>. | NMEAN |

## Erregistro klasikoak

| Agindua | Laburra | Zer egiten du | Ezizenak |
|---|---|---|---|
| `REGSET erregistroa balioa` | `SET` | Erregistro bati balio bat eman. | SET, RSET, RGST |
| `REGCOPY dst src` | `RC` | Erregistro bat beste batera kopiatu. | MOV, RCOPY, RC |
| `ADD dst a b` | `ADD` | dst = a + b | REGADD, RADD, RGA |
| `SUB dst a b` | `SUB` | dst = a - b | REGSUB, RSUB, RGS |
| `MUL dst a b` | `MUL` | dst = a * b | REGMUL |
| `DIV dst a b` | `DIV` | dst = a / b (zeroz zatitzeak programa gelditzen du) | REGDIV |
| `ANDBITS dst a b` | `AND` | Balioen bitez biteko AND, 64 biteko oso gisa. | AND, ANDB, AB |
| `ORBITS dst a b` | `OR` | Balioen bitez biteko OR, 64 biteko oso gisa. | OR, ORB, OB |
| `XORBITS dst a b` | `XOR` | Balioen bitez biteko XOR, 64 biteko oso gisa. | XOR, XORB, XB |
| `NOTBITS dst src` | `NOT` | Balioaren bitez biteko NOT, 64 biteko oso gisa. | NOTB, NB |
| `SHL dst src kopurua` | `SHL` | Ezkerrera desplazatu erregistro bateko kopuruan. |  |
| `SHR dst src kopurua` | `SHR` | Eskuinera desplazatu erregistro bateko kopuruan. |  |
| `SQRT dst src` | `SQRT` | Erro karratua. | SR |
| `EXP dst src` | `EXP` | e balioaren berretzailera. |  |
| `LOG dst src` | `LN` | Logaritmo nepertarra. | LN |
| `SIN dst src` | `SIN` | Angelu baten sinua, radianetan. |  |
| `COS dst src` | `COS` | Angelu baten kosinua, radianetan. |  |
| `TAN dst src` | `TAN` | Angelu baten tangentea, radianetan. |  |
| `POW dst a b` | `POW` | a b-ren berretzailera. |  |
| `RAND erregistroa` | `RN` | Ausazko zenbaki uniforme bat [0, 1) tartean. | RN |
| `SEEDRNG hazia` | `SRNG` | Ausazko zenbaki-sorgailuari hazi berria eman. Neurketek ere hura erabiltzen dute, beraz shot guztiek emaitza berak errepikatuko dituzte. | SRNG |
| `INPUT erregistroa` | `INP` | Zenbaki bat irakurri sarrera estandarretik. | INP |
| `GETTIME erregistroa` | `GT` | Unix garaitik igarotako segundoak. | GTIME, GT |
| `PUSH erregistroa` | `PUSH` | Erregistro bat pilan sartu. | PUSHREG |
| `POP erregistroa` | `POP` | Pilatik atera erregistro batera. | POPREG |
| `CHARLOAD erregistroa karakterea` | `CLD` | Karaktere-kode bat erregistro batean kargatu ("A" edo 65). | CLOAD, CLD |
| `PACKBITS dst bitak...` | `PB` | Bit klasikoak zenbaki oso gisa irakurri erregistro batera. Zerrendako lehen bita baxuena da; inoiz idatzi gabeko bitak 0 dira (OpenQASM-en bezala). |  |

## Memoria, byteak eta argumentuak

| Agindua | Laburra | Zer egiten du | Ezizenak |
|---|---|---|---|
| `LOAD dst addr` | `LD` | addr-ek duen helbideko memoria-gelaxka irakurri. Memoria 0, 1, 2, ... gelaxkak dira (2^32 arte), bakoitzean zenbaki bat; inoiz idatzi gabeko gelaxka batek 0 du. | LD |
| `STORE src addr` | `ST` | src addr-ek duen helbideko memoria-gelaxkan idatzi. | ST |
| `STORETEXT addr testua` | `STX` | Testuaren byteak (UTF-8) memorian idatzi addr-eko helbidetik aurrera, gelaxka bakoitzean byte bat, eta gero 0 bat; ondoren addr-ek 0aren ondorengo helbidea du. | STEXT |
| `GETC dst` | `GC` | Sarrera estandarraren hurrengo bytea irakurri (0..255), edo -1 sarreraren amaieran. | GETBYTE |
| `PUTC src` | `PUTC` | Byte bat (0..255) irteerara idatzi bere horretan, irteera bitarrerako. | PUTBYTE |
| `STREAM stream` | `STREAM` | Hurrengo irteera irteera estandarrera (1) edo errore-irteerara (2) bidali. |  |
| `ARGC dst` | `ARGC` | Programari -- ondoren zenbat argumentu eman zaizkion (qoa run prog.qx -- a b). |  |
| `ARGBYTE dst arg index` | `ARGBYTE` | arg zenbakidun argumentuaren index zenbakidun bytea (biak 0tik zenbatuta), edo -1 argumentuaren amaieratik aurrera. |  |
| `GETLANG dst` | `GETLANG` | 1 QOA euskaraz ari bada (QOA_LANG=eu), 0 ingelesez, programak bere mezuak erabiltzailearen hizkuntzan inprimatu ditzan. |  |

## Kontrol-fluxua

| Agindua | Laburra | Zer egiten du | Ezizenak |
|---|---|---|---|
| `JUMP etiketa` | `J` | Etiketa batean jarraitu. | JMP |
| `JUMPIFZERO balioa etiketa` | `JZ` | Jauzi bita edo erregistroa 0 bada. | JIZ, JZ |
| `JUMPIFONE balioa etiketa` | `JO` | Jauzi bita edo erregistroa 1 bada. | JIO, JO |
| `IFEQ a b etiketa` | `IEQ` | Jauzi a == b bada. | IEQ |
| `IFNE a b etiketa` | `INE` | Jauzi a != b bada. | INE |
| `IFGT a b etiketa` | `IGT` | Jauzi a > b bada. | IGT |
| `IFLT a b etiketa` | `ILT` | Jauzi a < b bada. | ILT |
| `CALL etiketa` | `C` | Etiketa bateko azpierrutina deitu. | CALLLABEL |
| `RETSUB` | `RET` | Azpierrutinatik itzuli. | RET, RS |
| `LOOPSTART kopurua` | `LS` | Dagokion LOOPEND-erainoko blokea hainbat aldiz exekutatu (zenbaki bat edo erregistro bat). | LOOP, LSTART, LS |
| `LOOPEND` | `LE` | LOOPSTART bloke baten amaiera. | ENDLOOP, LEND, LE |
| `HALT` | `END` | Programa gelditu. | HLT, END |
| `EXIT status` | `EXIT` | Programa gelditu HALT bezala, irteera-egoera batekin (0..255). Lehen shot-aren egoera qoa run-en irteera-egoera bihurtzen da. |  |
| `BARRIER [qubitak...]` | `BR` | Simulazioan ez du ezer egiten. Geruza-muga bat markatzen du eta OpenQASM-erako joan-etorrietan gordetzen da. | BR |

## Irteera

| Agindua | Laburra | Zer egiten du | Ezizenak |
|---|---|---|---|
| `PRINT testua` | `PRINT` | Testua inprimatu. |  |
| `PRINTLN [testua]` | `PLN` | Testua eta lerro-jauzi bat inprimatu. | PLN |
| `PRINTF formatua [erregistroak...]` | `PF` | Testua inprimatu, {} bakoitza hurrengo erregistroaren balioarekin ordezkatuta. | PF |
| `CHAROUT erregistroa` | `CO` | Erregistro baten balioa karaktere gisa inprimatu. | COUT, CO |
| `VERBOSELOG qubita testua` | `VL` | Qubit batekin etiketatutako mezu bat inprimatu. | VLOG, VL |
| `DUMPREGS` | `DR` | Erregistro guztiak inprimatu. | DREGS, DR |
| `COMMENT` | `CM` | Lerroaren gainerakoa iruzkina da (; bezala). | CMT, CM |

## 0.4-n kenduak

QOA 0.3-k onartzen zituen, baina ez zuten ezer egiten edo ez zuten esanahi zehatzik. Orain erabiltzea errore bat da, eta zer erabili azaltzen du.

| Agindua | Zergatik |
|---|---|
| `SETPOS` | egoera kuantikoan inoiz eraginik izan ez zuen posizio-erregistroa |
| `SETWL` | egoera kuantikoan inoiz eraginik izan ez zuen uhin-luzera erregistroa |
| `WLSHIFT` | egoera kuantikoan inoiz eraginik izan ez zuen uhin-luzera erregistroa |
| `MOVE` | egoera kuantikoan inoiz eraginik izan ez zuen posizio-erregistroa |
| `MARKOBSERVED` | ez zuen eraginik; neurtutako qubit bat neurtuta dago, besterik ez |
| `RELEASE` | ez zuen eraginik; simulagailu batek ez du ezer askatzeko |
| `DECOHERENCEPROTECT` | ez dago eredu fisikorik atzean; dekoherentzia aztertzeko erabili `qoa run --noise` |
| `ENTANGLESWAPMEASURE` | erabili ESWAP, eta gero adarkatu neurtutako bitekin JUMPIFONE bidez |
| `ENTANGLEWITHCLASSICALFEEDBACK` | erabili ateak eta JUMPIFONE/IFEQ atzeraelikadura klasikorako |
| `ENTANGLEDISTRIBUTED` | makinen arteko sareko entrelazamendua ez dago modelatuta |
| `APPLYNONLINEARPHASESHIFT` | ez zen inoiz definitu; erabili KERR edo CROSSKERR |
| `APPLYNONLINEARSIGMA` | ez zen inoiz definitu; erabili KERR edo CROSSKERR |
| `QUANTUMZENOEFFECT` | denbora jarraituko bilakaera behar du, eta QOA-k ez du modelatzen; LOOPSTART baten barruko MEASURE errepikatuek izozte bera erakusten dute |
| `APPLYLINEAROPTICALTRANSFORM` | eraiki sarea BS eta MODEPHASE erabiliz; unitario optiko lineal oro haietan deskonposatzen da |
| `ERRORCORRECT` | ez zen inoiz errore-zuzentzeko koderik zehaztu; idatzi zuzenketa-zirkuitua ateekin |
| `ERRORSYNDROME` | ez zen inoiz koderik zehaztu; idatzi sindrome-zirkuitua ateekin eta MEASURE-rekin |
| `PHOTONROUTE` | atakak ez ziren inoiz definitu; erabili MODESWAP edo BS |
| `TIMEDELAY` | atzerapen ideal batek ez du ezer aldatzen; benetakoa modelatzeko erabili LOSS eta MODEPHASE |
| `OPTICALDELAYLINECONTROL` | atzerapen ideal batek ez du ezer aldatzen; benetakoa modelatzeko erabili LOSS eta MODEPHASE |
| `WAIT` | itxaronaldi ideal batek ez du ezer aldatzen; geldirik dauden erroreak aztertzeko erabili `qoa run --noise` |
| `PHOTONBUNCHINGCONTROL` | multzokatzea interferentziatik dator (ikus BS), ez da etengailu bat |
| `SINGLEPHOTONSOURCEOFF` | iturriak berehalakoak dira (SINGLEPHOTON); ez dago ezer itzaltzeko |
| `APPLYDISPLACEMENTOPERATOR` | erabili DISPLACE |
| `OPTICALSWITCHCONTROL` | erabili MODESWAP aldatzeko |
| `MEASUREWITHDELAY` | atzerapen ideal batek ez du ezer aldatzen; erabili MEASURE |
| `PHOTONLOSSCORRECTION` | ez zen inoiz galera zuzentzeko koderik zehaztu |
| `PHOTONEMISSIONPATTERN` | ez zen inoiz definitu; erabili EMIT LOOPSTART baten barruan |
| `APPLYFEEDFORWARDGATE` | erabili JUMPIFONE edo IFEQ ateak baldintzapean aplikatzeko |
| `THERMALAVG` | ez da zirkuitu kuantikoko eragiketa bat |
| `WKBFACTOR` | ez da zirkuitu kuantikoko eragiketa bat |
| `CMP` | IFEQ, IFNE, IFGT eta IFLT aginduek erregistroak zuzenean konparatzen dituzte |
| `JMPABS` | byte-helbideak jada ez daude; jauzi etiketa batera |
| `CALLADDR` | byte-helbideak jada ez daude; erabili CALL etiketa batekin |
| `LOADMEM` | memoria orain LOAD helmuga helbidea da, helbidea erregistro batean |
| `STOREMEM` | memoria orain STORE iturria helbidea da, helbidea erregistro batean |
| `LOADCLASSICAL` | memoria orain LOAD helmuga helbidea da, helbidea erregistro batean |
| `STORECLASSICAL` | memoria orain STORE iturria helbidea da, helbidea erregistro batean |
| `ALLOC` | memoriak ez du esleipenik behar: LOAD eta STORE 0, 1, 2, ... gelaxkak zuzenean erabiltzen dituzte |
| `FREE` | memoria ez da askatu behar: LOAD eta STORE 0, 1, 2, ... gelaxkak zuzenean erabiltzen dituzte |
| `BREAKPOINT` | erabili DUMPSTATE eta DUMPREGS programaren puntu bat ikusteko |
| `EXITCODE` | erabili HALT |
| `QUBITS` | erabili QINIT qubit-kopuruarekin, adib. QINIT 25 |
| `QGATE` | erabili GATE izena qubita, edo atearen agindu propioa |
