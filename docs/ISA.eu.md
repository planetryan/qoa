# QOA aginduen erreferentzia

`src/isa.rs`-tik sortua `qoa isa --markdown` bidez. Ez editatu eskuz.

Mnemonikoetan berdin dio maiuskula edo minuskula. Ezizen bakoitzak bere izen nagusiak egiten duen gauza bera egiten du.

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

| Agindua | Zer egiten du | Ezizenak |
|---|---|---|
| `QINIT qubitak` | Programak zenbat qubit erabiltzen dituen adierazi (behin). Gabe, erabilitako qubit-indize handiena gehi bat da. | QI, INITQUBIT, IQ, QINITQ |
| `QMODE moduak [muga]` | Modu optikoak adierazi, bakoitzak 0..muga-1 fotoi gordetzen dituela (muga lehenetsia 8). Mugaz kanpora doan probabilitatea exekuzioaren ondoren jakinarazten da. | MODES |
| `QREG izena size [first]` | Qubitei izena eman (aukerakoa): QREG data 3 ondoren, data[0]..data[2]-k (edo data-k, hirurak) izenik gabeko hurrengo 3 qubitak adierazten dituzte, edo `first`-etik hasitakoak emanez gero. q0, q1, ... erabiltzen jarraitzen dute. |  |
| `CREG izena size [first]` | Bit klasikoei izena eman (aukerakoa), QREG bezala: CREG flags 2-k flags[0], flags[1] eta flags sortzen ditu. |  |
| `OREG izena size [first]` | Modu optikoei izena eman (aukerakoa), QREG bezala: OREG arm 2-k arm[0], arm[1] eta arm sortzen ditu. |  |
| `RREG izena size [first]` | Erregistro klasikoei izena eman (aukerakoa): RREG total 2 ondoren, total[0] eta total[1] (edo total, biak) izenik gabeko hurrengo 2 erregistroak dira, edo `first`-etik aurrerakoak ematen bada. R0..R31-ek ere balio dute. |  |
| `CONST izena balioa` | Zenbaki bati izena eman: CONST newline 10 ondoren, newline edozein adierazpenetan erabil daiteke (RX q0 newline/4, ADD R1 R1 (newline)). Konstante batek bere gainetik definitutakoak erabil ditzake. |  |

## Qubit-ateak

| Agindua | Zer egiten du | Ezizenak |
|---|---|---|
| `H qubita` | Hadamard. Qubit-tarte bat onartzen du (q0..q3). | HAD, APPLYHADAMARD |
| `X qubita` | Pauli-X (bit-iraulketa). Qubit-tarte bat onartzen du (q0..q3). | APPLYBITFLIP |
| `Y qubita` | Pauli-Y. Qubit-tarte bat onartzen du (q0..q3). |  |
| `Z qubita` | Pauli-Z (fase-iraulketa). Qubit-tarte bat onartzen du (q0..q3). | APPLYPHASEFLIP |
| `S qubita` | S = diag(1, i). Qubit-tarte bat onartzen du (q0..q3). | APPLYSGATE |
| `SDG qubita` | S† = diag(1, -i). Qubit-tarte bat onartzen du (q0..q3). | SDAG |
| `T qubita` | T = diag(1, e^(i·pi/4)). Qubit-tarte bat onartzen du (q0..q3). | APPLYTGATE |
| `TDG qubita` | T† = diag(1, e^(-i·pi/4)). Qubit-tarte bat onartzen du (q0..q3). | TDAG |
| `SX qubita` | X-ren erro karratua. Qubit-tarte bat onartzen du (q0..q3). | SQRTX |
| `SXDG qubita` | SX-ren alderantzizkoa. Qubit-tarte bat onartzen du (q0..q3). | SXDAG |
| `PHASE qubita angelua` | Fase-atea diag(1, e^(i·angelua)). Qubit-tarte bat onartzen du (q0..q3). | P, PHASESHIFT, SETPHASE, SETP, SPH, PSE, U1 |
| `RX qubita angelua` | Biraketa X ardatzean: exp(-i·angelua·X/2). Qubit-tarte bat onartzen du (q0..q3). |  |
| `RY qubita angelua` | Biraketa Y ardatzean: exp(-i·angelua·Y/2). Qubit-tarte bat onartzen du (q0..q3). |  |
| `RZ qubita angelua` | Biraketa Z ardatzean: exp(-i·angelua·Z/2). Qubit-tarte bat onartzen du (q0..q3). |  |
| `U qubita theta phi lambda` | Qubit bakarreko ate orokorra U(theta, phi, lambda), OpenQASM-en konbentzioa. Qubit-tarte bat onartzen du (q0..q3). | U3 |
| `ROT qubita ardatza angelua` | Biraketa X, Y edo Z ardatzean (RX, RY edo RZ bera). Qubit-tarte bat onartzen du (q0..q3). | APPLYROTATION, AR |
| `MROT ardatza angelua qubitak...` | Biraketa bera zerrendako qubit guztietan. | APPLYMULTIQUBITROTATION, AMQR |
| `GATE izena qubita` | Izendun qubit bakarreko atea aplikatu: h, x, y, z, s, sdg, t, tdg, sx, sxdg. | APPLYGATE, AGATE, AG |
| `CNOT kontrola helburua` | X kontrolatua: helburua irauli kontrola 1 denean. | CX, CONTROLLEDNOT, CN, ENTANGLE |
| `CY kontrola helburua` | Y kontrolatua. |  |
| `CZ kontrola helburua` | Z kontrolatua: -1 fasea \|11>-n. |  |
| `CH kontrola helburua` | Hadamard kontrolatua. |  |
| `CSX kontrola helburua` | X-ren erro karratu kontrolatua. |  |
| `CU kontrola helburua theta phi lambda` | U(theta, phi, lambda) kontrolatua. | CU3 |
| `CPHASE kontrola helburua angelua` | Fase kontrolatua: e^(i·angelua) \|11>-n. | CP, CONTROLLEDPHASEROTATION, APPLYCPHASE, CPR, CU1 |
| `CRX kontrola helburua angelua` | RX kontrolatua. |  |
| `CRY kontrola helburua angelua` | RY kontrolatua. |  |
| `CRZ kontrola helburua angelua` | RZ kontrolatua. |  |
| `SWAP a b` | Bi qubit trukatu. |  |
| `CSWAP kontrola a b` | a eta b trukatu kontrola 1 denean (Fredkin). | CONTROLLEDSWAP, CS, FREDKIN |
| `CCX kontrola1 kontrola2 helburua` | Toffoli: helburua irauli bi kontrolak 1 direnean. | TOFFOLI, CCNOT |
| `MCX kontrolak... helburua` | Helburua (azken qubita) irauli zerrendako beste qubit guztiak 1 direnean. |  |
| `MCZ qubitak...` | -1 fasea zerrendako qubit guztiak 1 diren oinarri-egoeretan. |  |

## Entrelazamendu-laburdurak

| Agindua | Zer egiten du | Ezizenak |
|---|---|---|
| `BELL a b` | H a exekutatzen du, gero CNOT a b. \|00>-tik (\|00> + \|11>)/sqrt(2) ematen du. | ENTANGLEBELL, EBELL, EB |
| `GHZ qubitak...` | H lehen qubitean, gero CNOT hartatik beste guztietara. \|0..0>-tik GHZ egoera bat ematen du. | ENTANGLEMULTI, EMULTI, EM |
| `CLUSTER qubitak...` | Cluster egoera lineala: H qubit guztietan, gero CZ ondoko bikote bakoitzean. | ENTANGLECLUSTER, ECLUSTER, ECR |
| `ESWAP a b c d` | Entrelazamendu-trukaketa. (a,b) eta (c,d) bell bikoteekin: b eta c bell-neurtzen ditu (c<b> eta c<c>-ra), gero d zuzentzen du, eta (a,d) bell bikote bihurtzen da. | ENTANGLESWAP, ESP |

## Qubit-neurketa eta berrezarpena

| Agindua | Zer egiten du | Ezizenak |
|---|---|---|
| `MEASURE qubita [-> helmuga]` | Z oinarrian neurtu. Emaitza c<qubit> bitera doa, -> cK edo -> RK bidez beste nonbait bideratzen ez bada. Qubit-tarte bat onartzen du (q0..q3). | MEAS, M, QMEAS, QM, APPLYQNDMEASUREMENT, AQND, AQAD, AQNM |
| `MEASUREIN qubita oinarria [-> helmuga]` | X, Y edo Z oinarrian neurtu (Z-ra biratu, neurtu, atzera biratu). | MEASUREINBASIS, MEASB, MIB |
| `MEASUREALL` | Qubit guztiak neurtu, bakoitza bere c<qubit> bitera. | MALL |
| `RESET qubita` | Qubit bat \|0>-n jarri: neurtu, eta 1 irakurri badu irauli. Qubit-tarte bat onartzen du (q0..q3). | RST, RSTQ, QRESET, QR |
| `RESETALL` | Qubit guztiak \|0>-ra eta modu guztiak hutsera. | RSTALL, RSA |
| `TOBASIS qubita oinarria` | Biratu Z neurketa batek oinarri hori irakur dezan (X: H; Y: SDG eta gero H; Z: ezer ez). Qubit-tarte bat onartzen du (q0..q3). | APPLYMEASUREMENTBASISCHANGE, AMBC |

## Modu optikoak

| Agindua | Zer egiten du | Ezizenak |
|---|---|---|
| `DISPLACE modua re [im]` | Desplazamendua D(alpha), alpha = re + i·im. Hutsa egoera koherente bihurtzen du, \|alpha\|^2 batez besteko fotoiekin. | APPLYDISPLACEMENT, ADISP, AD |
| `DISPLACEREG modua alpha` | Erregistro bateko balioaren (erreala) desplazamendua: aurrera-elikadura klasikoa. | APPLYDISPLACEMENTFEEDBACK |
| `SQUEEZE modua r [phi]` | Estutzea S(r·e^(i·phi)). phi = 0 eta r > 0 direnean x koadratura estutzen da. | APPLYSQUEEZING, ASQ, AS |
| `SQUEEZEREG modua r` | Erregistro bateko balioaren estutzea: aurrera-elikadura klasikoa. | APPLYSQUEEZINGFEEDBACK, ASWF, ASF, ASFB |
| `MODEPHASE modua angelua` | Modu baten fase-biraketa e^(i·angelua·n). | DYNAMICPHASECOMPENSATION, DPC, DPCMP, PHASEROT |
| `KERR modua kappa [denbora]` | Kerr ez-lineartasuna e^(i·kappa·denbora·n^2) (denbora lehenetsia 1). | APPLYKERRNONLINEARITY, AKNL |
| `CROSSKERR a b kappa` | Kerr gurutzatua (fase-modulazio gurutzatua) e^(i·kappa·n_a·n_b). | CROSSPHASEMODULATION, CPM, CPMOD |
| `BS a b theta [phi]` | Izpi-zatitzailea exp(theta·(e^(i·phi)·a·b† - e^(-i·phi)·a†·b)). theta = pi/4 50:50 da. | BEAMSPLITTER |
| `MODESWAP a b` | Bi moduren edukiak trukatu (bideratzaile ideal bat). | OPTICALROUTING, OROUTE, OPTR |
| `ADDPHOTON modua` | a† aplikatu eta normalizatu. Post-hautatua: benetako fotoi-gehiketak batzuetan bakarrik lortzen du. | PHOTONADDITION, PADD, PA |
| `SUBPHOTON modua` | a aplikatu eta normalizatu (post-hautatua). Huts egiten du modua ziur hutsik badago. | APPLYPHOTONSUBTRACTION, APSUB, APS |
| `LOSS modua probabilitatea` | Moduko fotoi bakoitza probabilitate honekin galtzen da. Ausazko emaitza bat exekuzio bakoitzeko; shot-en batez bestekoa galera-kanal zehatza da. | PHOTONLOSS, PHOTONLOSSSIMULATE, PLS, PLSIM, SETOPTICALATTENUATION, SOATT, SOA |
| `FOCK modua fotoiak` | Moduak zuena baztertu eta zehazki fotoi-kopuru hori prestatu. |  |
| `SINGLEPHOTON modua` | Fotoi bakarreko iturri ideala: FOCK modua 1 bera. | SINGLEPHOTONSOURCEON, SPSON |
| `COUNT modua [-> helmuga]` | Fotoi-kopurua bereizten duen detekzioa. Kopurua moduaren irakurketara doa, -> RK bidez bideratzen ez bada. | PHOTONCOUNT, PCOUNT, PC, PHOTONNUMBERRESOLVINGDETECTION, PNRD, PNR |
| `DETECT modua [-> helmuga]` | Atalase-detekzioa (klik): 1 fotoiren bat badago, bestela 0. | PHOTONDETECT, PDETECT, PD |
| `PARITY modua [-> helmuga]` | Fotoi-kopuruaren paritatearen neurketa: 0 bikoitia, 1 bakoitia. | MEASUREPARITY, MPAR, MP |
| `COINCIDENCE moduak... -> helmuga` | Zerrendako modu guztiak klik-detektatu; 1 idazten du guztiek klik egin badute. | PHOTONDETECTCOINCIDENCE, PDCOIN, PDC |

## Qubit-modu akoplamendua

| Agindua | Zer egiten du | Ezizenak |
|---|---|---|
| `EMIT qubita modua` | \|1>\|0> eta \|0>\|1> trukatu qubit baten eta modu baten artean: igorle kitzikatu batek bere kitzikapena fotoi gisa jartzen du moduan. Berriz exekutatuta fotoia xurgatzen du. | PHOTONEMIT, PEMIT, PE, ABSORB |
| `DISPERSIVE qubita modua angelua` | Akoplamendu dispertsiboa e^(i·angelua·n), qubita 1 den lekuetan bakarrik. |  |

## Ikuskapena (simulagailuan bakarrik)

| Agindua | Zer egiten du | Ezizenak |
|---|---|---|
| `DUMPSTATE` | *Simulagailuan bakarrik.* Puntu honetako egoera zehatza inprimatu. | DSTATE, DS |
| `EXPECT qubita oinarria [helmuga]` | *Simulagailuan bakarrik.* Qubit baten <X>, <Y> edo <Z> zehatza, hura aldatu gabe. Inprimatuta, edo erregistro batean gordeta. | QUANTUMSTATETOMOGRAPHY, QST, QSTAT |
| `BELLCHECK a b [helmuga]` | *Simulagailuan bakarrik.* (a, b) qubiten fideltasun zehatza (\|00> + \|11>)/sqrt(2)-rekin. | BELLSTATEVERIFICATION, BSV, BSTATE |
| `MEANPHOTON modua [helmuga]` | *Simulagailuan bakarrik.* Modu baten batez besteko fotoi-kopuru zehatza <n>. | NMEAN |

## Erregistro klasikoak

| Agindua | Zer egiten du | Ezizenak |
|---|---|---|
| `REGSET erregistroa balioa` | Erregistro bati balio bat eman. | SET, RSET, RGST |
| `REGCOPY dst src` | Erregistro bat beste batera kopiatu. | MOV, RCOPY, RC |
| `ADD dst a b` | dst = a + b | REGADD, RADD, RGA |
| `SUB dst a b` | dst = a - b | REGSUB, RSUB, RGS |
| `MUL dst a b` | dst = a * b | REGMUL |
| `DIV dst a b` | dst = a / b (zeroz zatitzeak programa gelditzen du) | REGDIV |
| `ANDBITS dst a b` | Balioen bitez biteko AND, 64 biteko oso gisa. | AND, ANDB, AB |
| `ORBITS dst a b` | Balioen bitez biteko OR, 64 biteko oso gisa. | OR, ORB, OB |
| `XORBITS dst a b` | Balioen bitez biteko XOR, 64 biteko oso gisa. | XOR, XORB, XB |
| `NOTBITS dst src` | Balioaren bitez biteko NOT, 64 biteko oso gisa. | NOTB, NB |
| `SHL dst src kopurua` | Ezkerrera desplazatu erregistro bateko kopuruan. |  |
| `SHR dst src kopurua` | Eskuinera desplazatu erregistro bateko kopuruan. |  |
| `SQRT dst src` | Erro karratua. | SR |
| `EXP dst src` | e balioaren berretzailera. |  |
| `LOG dst src` | Logaritmo nepertarra. | LN |
| `SIN dst src` | Angelu baten sinua, radianetan. |  |
| `COS dst src` | Angelu baten kosinua, radianetan. |  |
| `TAN dst src` | Angelu baten tangentea, radianetan. |  |
| `POW dst a b` | a b-ren berretzailera. |  |
| `RAND erregistroa` | Ausazko zenbaki uniforme bat [0, 1) tartean. | RN |
| `SEEDRNG hazia` | Ausazko zenbaki-sorgailuari hazi berria eman. Neurketek ere hura erabiltzen dute, beraz shot guztiek emaitza berak errepikatuko dituzte. | SRNG |
| `INPUT erregistroa` | Zenbaki bat irakurri sarrera estandarretik. | INP |
| `GETTIME erregistroa` | Unix garaitik igarotako segundoak. | GTIME, GT |
| `PUSH erregistroa` | Erregistro bat pilan sartu. | PUSHREG |
| `POP erregistroa` | Pilatik atera erregistro batera. | POPREG |
| `CHARLOAD erregistroa karakterea` | Karaktere-kode bat erregistro batean kargatu ("A" edo 65). | CLOAD, CLD |
| `PACKBITS dst bitak...` | Bit klasikoak zenbaki oso gisa irakurri erregistro batera. Zerrendako lehen bita baxuena da; inoiz idatzi gabeko bitak 0 dira (OpenQASM-en bezala). |  |

## Memoria, byteak eta argumentuak

| Agindua | Zer egiten du | Ezizenak |
|---|---|---|
| `LOAD dst addr` | addr-ek duen helbideko memoria-gelaxka irakurri. Memoria 0, 1, 2, ... gelaxkak dira (2^32 arte), bakoitzean zenbaki bat; inoiz idatzi gabeko gelaxka batek 0 du. | LD |
| `STORE src addr` | src addr-ek duen helbideko memoria-gelaxkan idatzi. | ST |
| `STORETEXT addr testua` | Testuaren byteak (UTF-8) memorian idatzi addr-eko helbidetik aurrera, gelaxka bakoitzean byte bat, eta gero 0 bat; ondoren addr-ek 0aren ondorengo helbidea du. | STEXT |
| `GETC dst` | Sarrera estandarraren hurrengo bytea irakurri (0..255), edo -1 sarreraren amaieran. | GETBYTE |
| `PUTC src` | Byte bat (0..255) irteerara idatzi bere horretan, irteera bitarrerako. | PUTBYTE |
| `STREAM stream` | Hurrengo irteera irteera estandarrera (1) edo errore-irteerara (2) bidali. |  |
| `ARGC dst` | Programari -- ondoren zenbat argumentu eman zaizkion (qoa run prog.qx -- a b). |  |
| `ARGBYTE dst arg index` | arg zenbakidun argumentuaren index zenbakidun bytea (biak 0tik zenbatuta), edo -1 argumentuaren amaieratik aurrera. |  |
| `GETLANG dst` | 1 QOA euskaraz ari bada (QOA_LANG=eu), 0 ingelesez, programak bere mezuak erabiltzailearen hizkuntzan inprimatu ditzan. |  |

## Kontrol-fluxua

| Agindua | Zer egiten du | Ezizenak |
|---|---|---|
| `JUMP etiketa` | Etiketa batean jarraitu. | JMP |
| `JUMPIFZERO balioa etiketa` | Jauzi bita edo erregistroa 0 bada. | JIZ, JZ |
| `JUMPIFONE balioa etiketa` | Jauzi bita edo erregistroa 1 bada. | JIO, JO |
| `IFEQ a b etiketa` | Jauzi a == b bada. | IEQ |
| `IFNE a b etiketa` | Jauzi a != b bada. | INE |
| `IFGT a b etiketa` | Jauzi a > b bada. | IGT |
| `IFLT a b etiketa` | Jauzi a < b bada. | ILT |
| `CALL etiketa` | Etiketa bateko azpierrutina deitu. | CALLLABEL |
| `RETSUB` | Azpierrutinatik itzuli. | RET, RS |
| `LOOPSTART kopurua` | Dagokion LOOPEND-erainoko blokea hainbat aldiz exekutatu (zenbaki bat edo erregistro bat). | LOOP, LSTART, LS |
| `LOOPEND` | LOOPSTART bloke baten amaiera. | ENDLOOP, LEND, LE |
| `HALT` | Programa gelditu. | HLT, END |
| `EXIT status` | Programa gelditu HALT bezala, irteera-egoera batekin (0..255). Lehen shot-aren egoera qoa run-en irteera-egoera bihurtzen da. |  |
| `BARRIER [qubitak...]` | Simulazioan ez du ezer egiten. Geruza-muga bat markatzen du eta OpenQASM-erako joan-etorrietan gordetzen da. | BR |

## Irteera

| Agindua | Zer egiten du | Ezizenak |
|---|---|---|
| `PRINT testua` | Testua inprimatu. |  |
| `PRINTLN [testua]` | Testua eta lerro-jauzi bat inprimatu. | PLN |
| `PRINTF formatua [erregistroak...]` | Testua inprimatu, {} bakoitza hurrengo erregistroaren balioarekin ordezkatuta. | PF |
| `CHAROUT erregistroa` | Erregistro baten balioa karaktere gisa inprimatu. | COUT, CO |
| `VERBOSELOG qubita testua` | Qubit batekin etiketatutako mezu bat inprimatu. | VLOG, VL |
| `DUMPREGS` | Erregistro guztiak inprimatu. | DREGS, DR |
| `COMMENT` | Lerroaren gainerakoa iruzkina da (; bezala). | CMT, CM |

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
