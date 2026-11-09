# Segurtasuna eta sendotasuna

[In English: safety.md](safety.md)

QOA ematen diozun edozertan fidatzeko egina dago: zuk idatzitako programa bat, beste nonbaitetik datorren
`.qx` fitxategi bat, fitxategi hondatu bat, idazketa-akats bat, edo makina baino askoz handiagoa den exekuzio
bat. Orri honek esaten du QOA-k zer agintzen duen, non dauden haren mugak, nola probatzen den, eta kodearen
zein zati txikik behar duten oraindik rust-en `unsafe`, eta zergatik.

## Agintzen duena

Sarrera edozein dela ere, QOA-k bi gauzetako bat egiten du: programa exekutatzen du, edo gelditu egiten da
zer dagoen gaizki esaten duen mezu argi batekin, ingelesez edo euskaraz. Ez da erortzen, ez du panic
egiten, ez da mugarik gabe geratzen, eta ez du memoria betetzen sistemak hil arte zuri lehenik esan gabe. Hau
hartzen du:

- **Iturburuak**: `.qoa`, `.qg` (glifo-sintaxia) eta `.qasm` (OpenQASM), utf-8 baliogabea, ausazko byteak,
  zenbaki zentzugabeak eta milaka mailako habiaraketak barne.
- **Fitxategi konpilatuak** (`.qx`): fitxategi bakoitza bere gainerako byte guztien SHA-256-arekin amaitzen
  da, beraz edonon aldatutako bit bakar bat ere baztertu egiten da ezer irakurri aurretik (ikus
  [Fitxategien osotasuna](#fitxategien-osotasuna)). Horretaz gain, kargatutako programa batek mihiztatzaileak
  bere testutik eraikitzen duena izan behar du zehazki, eragiketaz eragiketa egiaztatuta: existitzen ez den
  erregistro, qubit, modu, bit edo jauzi-helburu bat aipatzen duen, begizta bat hausten duen, edo neurri
  zentzugabe bat adierazten duen fitxategi bat hondatu gisa baztertzen da, eta makinak ez du inoiz ikusten.
- **Exekutatzen ari diren programak**: programa batek eska dezakeen baliabide bakoitza erabili aurretik
  egiaztatzen da (ikus behean mugak), eta finituak ez diren zenbakiek programa gelditzen dute, egoera
  isilik NaN bihurtu beharrean.
- **Egoera handiak**: QOA-k nahita saiatzen uzten dizun gauza bakarra makinaren memoria baino handiagoa den
  egoera kuantiko bat da, zuk bakarrik dakizulako exekuzio motel batek merezi duen. `qoa run`-ek hasi aurretik
  abisatzen du, makina honetan zenbat qubit dabiltzan ondo esanez, eta gero aurrera egiten du; exekuzioa oso
  motela izan daiteke orduan, edo sistemak memoria faltagatik geldiaraz dezake. Sistemak ukatzen duen memoria
  errore argi bat da, ez erortze bat. Ezarri `QOA_MAX_MEMORY` egoera horiek baztertzeko, adibidez makina
  partekatu batean edo zuk idatzi ez dituzun programetarako: orduan ez da ezer erreserbatzen muga horretatik
  gora.
- **Irteera**: irteera ezin bada idatzi (disko betea), QOA gelditu egiten da eta hala esaten du, isilik
  galdu beharrean; irakurlea joaten bada (`qoa ... | head`), QOA isilik amaitzen da edozein komando-lerroko
  tresna bezala.

## Fitxategien osotasuna

Programa bat datorren fitxategia bezain fidagarria da. QOA-k bere fitxategiak urrats guztietan egiaztatzen
ditu:

- **Fitxategi konpilatuek laburpen bat dute.** 0.4.5etik `.qx` bakoitza bere aurreko byte guztien
  SHA-256-arekin amaitzen da. Ezer irakurri aurretik, QOA-k berriz kalkulatzen du; disko-errore batek,
  deskarga eten batek edo eskuzko aldaketa batek bit bakar bat aldatu badu (ate baten angelua, qubit-zenbaki
  bat, programaren izena), fitxategia baztertu egiten da, hala dioen mezu batekin. Laburpenik gabe aldaketa
  horrek programa baliozko bat izaten jarrai lezake, isilik beste zerbait kalkulatuz. Edonork egiazta dezake
  `.qx` bat tresna arruntekin: `head -c -32 f.qx | sha256sum`-ek eta `tail -c 32 f.qx | od -An -tx1`-ek
  digitu berberak ematen dituzte. QOA 0.4.2-tik 0.4.4-ra idatzitako fitxategiak (laburpenik gabe) kargatzen
  dira oraindik.
- **Iturburuak zigila daitezke.** `qoa seal programa.qoa`-k azken lerro bat gehitzen du, `SEAL sha256:...`,
  haren gaineko lerro guztien SHA-256-arekin (`.qasm`-en iruzkin bat da, `// SEAL sha256:...`, beste tresnek
  fitxategia irakurtzen jarrai dezaten). Hortik aurrera QOA-k ez du fitxategia exekutatzen, egiaztatzen edo
  konpilatzen zigiluaren gainean zerbait aldatu bada, eta aldaketa zurea bada berriz nola zigilatu esaten du.
  Zigiluak `.qoa`, `.qg` eta `.qasm`-entzat balio du, mihiztatzaile biek egiaztatzen dute (Rust-ekoak eta QOAn
  idatzitakoak), eta aukerakoa da: zigilurik gabeko fitxategiek lehen bezala funtzionatzen dute.
- **Zigiluak eskatzea.** Osorik galdutako zigilu-lerro batek (haren gainean moztutako fitxategi batek)
  zigilurik gabeko fitxategi bat uzten du. Iturburu zigilatuak bakarrik exekutatu behar diren lekuan, ezarri
  `QOA_REQUIRE_SEAL=1`: orduan QOA-k zigilurik gabeko iturburu guztiak baztertzen ditu.
- **Fitxategiak osorik idazten dira edo batere ez.** `qoa compile`-k, `qoa export -o`-k, `qoa import -o`-k eta
  `qoa seal`-ek helburuaren ondoan fitxategi berri batean idazten dute, diskora bidaltzen dute, eta orduan
  bakarrik jartzen dute bere lekuan. Disko bete batek, huts-egite batek edo itzalaldi batek aurreko
  fitxategia bere horretan uzten du, inoiz ez erdi idatzita.

## Programa batek zer irits dezakeen

QOA programa bat isolatuta exekutatzen da. Kalkulatu dezake, bere sarrera estandarra eta `--` ondoren
emandako argumentuak irakurri, ordularia irakurri (`TIME`), eta irteera estandarrean eta errore-irteeran
idatzi. Ezin du fitxategirik ireki, sarea erabili, beste programarik abiarazi, ingurune-aldagairik irakurri,
edo koderik kargatu. Beraz programa edo fitxategi gaizto batek makinara iristeko dituen bideak QOA-k berak
irekitzen dituenak dira, eta bakoitza itxita dago:

- **Terminala.** Terminalek kontrol-sekuentziei erantzuten diete: programa batek leihoaren izenburua alda
  lezake, arbelean idatzi (OSC 52), kurtsorea mugitu irteera faltsutzeko, edo terminalari egoera-galdera
  bati erantzunarazi zuk idatzi bazenu bezala. QOA-ren irteera terminal batera doanean, programa edo
  fitxategi batetik datorren kontrol-sekuentzia oro (programaren irteera, errore-mezuetako iturburu-lerroak,
  izenak) testu gisa erakusten da (`^[]0;...`) eragin beharrean; lerro-jauziak, tabulazioak eta koloreak
  bakarrik pasatzen dira. Hodi edo fitxategi baterako irteera programak idatzi bezala uzten da, bytez byte,
  QOA programek tresna gisa lanean jarrai dezaten. Programa-izenean kontrol-karaktereak dituen `.qx` bat
  hondatu gisa baztertzen da.
- **Fitxategi konpilatuak.** `.qx` bat eragiketaz eragiketa egiaztatzen da makinak ikusi aurretik (ikus
  goian).
- **Liburutegiak.** QOA-k GPU kontrolatzaileak (`--gpu`) eta MPI (`--mpi`) eskatzen zaionean bakarrik
  kargatzen ditu, eta sistemaren lekuetatik bakarrik: inoiz ez uneko karpetatik, Windows-ek eta macOS-ek
  bestela begiratuko luketen eta deskargatutako liburutegi bat ezkuta daitekeen lekutik. `QOA_MPI_LIB`-ek
  izendatzen duen liburutegia kargatzen du zehazki; ez utzi beste inori ezartzen.
- **ffmpeg** (`qoa visual`). Fitxategi-izenak fitxategi arrunt gisa pasatzen zaizkio ffmpeg-i (`file:`),
  inoiz ez aukera edo protokolo gisa, izena edozein dela ere (`-y.wav`, `concat:...`). Aldi baterako
  fitxategiak berriak sortzen dira, zuk bakarrik irakur ditzakezu, eta inoiz ez lehendik dagoen fitxategi
  edo esteka baten bidez.
- **Sarrera handiak edo amaigabeak.** Programa-fitxategiak QOA-k erabil dezakeen memoriari lotutako tamaina
  bateraino irakurtzen dira (beraz `qoa run /dev/zero` berehala gelditzen da), `INPUT` lerro bat 1 MiB-ekoa
  da gehienez, eta grabatutako sarrera memoria-mugan zenbatzen da.

QOA-tik kanpo geratzen dena: ffmpeg-ek ematen dizkiozun multimedia-fitxategiak nola tratatzen dituen, GPU
kontrolatzaileak, eta MPI liburutegia.

## Mugak

| zer | muga | gainditzean |
|---|---|---|
| egoera kuantikoaren memoria | sartzen dena: QOA hastean erabilgarri dagoen memoriaren % 90 (`qoa check`-ek esaten du zenbat qubit diren) | abisu bat exekuzioaren aurretik, eta gero aurrera; `QOA_MAX_MEMORY` ezarrita, behar den tamaina eta muga esaten dituen mezu bat, ezer erreserbatu aurretik |
| 64 biteko inongo ordenagailuk helbideratu ezin duen egoera (2^59 anplitudetik aurrera, 8 EiB inguru) | inon ezin da gorde | mezu bat ezer erreserbatu aurretik; programa egiaztatu eta esporta daiteke hala ere |
| eragile optikoetarako, memoria-gelaxketarako eta emaitzetarako memoria | QOA hastean erabilgarri dagoen memoriaren % 90, edo `QOA_MAX_MEMORY` | behar den tamaina eta muga esaten dituen mezu bat, ezer erreserbatu aurretik |
| qubitak | q0-tik q65535-era, bit klasiko bakoitzeko bat (zenbat simulatzen diren ondo memoriaren araberakoa da, goian) | mihiztatzailearen errorea |
| modu optikoak | 0tik 63ra arteko indizeak (eta egoerak sartu behar du) | mihiztatzailearen errorea |
| bit klasikoak | c0-tik c65535-era | mihiztatzailearen errorea |
| izendun erregistroak (`QREG`, `CREG`, `OREG`, `RREG`, glifoen `{n}`) | 1etik 65536ra elementu, lehen indizea 0tik 65535era | mihiztatzailearen errorea |
| adierazpenen habiaraketa (parentesiak, funtzioak, zeinuak, `^` kateak) | 100 maila | mihiztatzailearen errorea, bi mihiztatzaileetan |
| shot baten urratsak | 100.000.000 (`--max-steps`-ek aldatzen du) | muga aipatzen duen exekuzio-errorea |
| `PUSH` pila | 1.048.576 balio | exekuzio-errorea |
| `CALL` habiaratuak | 100.000 | exekuzio-errorea |
| memoria-gelaxkak (`LOAD`, `STORE`) | 0tik 2^32 - 1era arteko helbideak, eta gelaxkek memorian sartu behar dute | exekuzio-errorea |
| exekuzio baten emaitza desberdinak | memorian sartzen direnak (1 KiB inguru bakoitza) | exekuzio-errorea, memoria bete aurretik |
| modu bateko argia | fotoi-mugaren azpian geratu behar da (`QMODE`) | exekuzio-errorea haren azpian probabilitaterik geratzen ez denean |

Emaitzak emaitza desberdin bakoitzeko gordetzen dira, inoiz ez shot bakoitzeko: lau emaitza posible dituen
100 milioi shot-eko exekuzio batek kilobyte gutxi batzuk behar ditu bere zenbaketetarako.

## Nola probatzen den

- `tests/hostile.rs` eraikitze bakoitzean exekutatzen da: 20.000 `.qx` fitxategi hondatu (bit aldatuak,
  moztutako fitxategiak, luzera eta indize erraldoiak, NaN eta infinitua zenbakien ordez; herena laburpena
  kaltearen ondoren berriz kalkulatuta, eta beste herena laburpenik gabeko formatu zaharrean, irakurlea bera
  ere proba dadin), 10.000 `.qoa`,
  `.qg` eta `.qasm` programa nahasi, eta 200.000 mailako habiaraketak dituzten programak. Bakoitza kargatu,
  mihiztatu, exekutatu (muga txikiekin) eta esportazio guztietatik pasatzen da (QOA testua, zerrenda, json,
  dot, glifoak, OpenQASM, IonQ), `QOA_MAX_MEMORY` ezarrita, tamaina nahasi batek ezin dezan gutxi baino
  gehiago erreserbatu. Edozein panic, bertan behera uzte edo geldialdik huts eginarazten du proba.
  Beste hazi batzuekin bilatzeak (`QOA_HOSTILE_SEED`) 0.4.4-ko bertsio-oharretan zerrendatutako arazoak
  aurkitu eta konpondu zituen; hamazazpi hazi, milioi erdi sarrera inguru, igarotzen dira orain.
- `tests/programs/failures`-eko programa bakoitzak espero den mezuarekin huts egin behar du, bi
  mihiztatzaileetan.
- Gainerako proba-multzoak egiaztatzen du exekutatzen dena zuzena dela: ate bakoitza simulagailu
  independente baten aurka, bektore-kernel bakoitza bertsio eskalar soil baten aurka, QOAn idatzitako
  mihiztatzailea rust-ekoaren aurka bytez byte, eta 1, 2, 3, 4 eta 5 MPI prozesu baten aurka.

## Non dagoen `unsafe`, eta zergatik

Rust-en `unsafe`-k konpiladoreak berak bere memoria-segurtasuna egiazta ezin duen kodea markatzen du.
Liburutegi osoan debekatuta dago (`#![deny(unsafe_code)]`), beheko moduluetan izan ezik, eta bakoitzak bere
hasieran esaten du zergatik; eta `qoa` komando-lerroko tresnan erabat debekatuta dago.

| modulua | zergatik |
|---|---|
| `simd` | prozesadorearen ezaugarri baterako (AVX-512, AVX2, NEON) eraikitako bektore-kernel bati deitzea, prozesadoreak ezaugarri hori duela egiaztatu ondoren, eta bektore bat kargatzea edo gordetzea haren memoria zehazki hartzen duen erreferentzia baten bidez; RISC-V-ren bektore-kernelak mihiztaduran daude, rust egonkorrak oraindik ez baitu RVV-rik onartzen |
| `kernel`, `state` | lan-hari bakoitzak egoeraren bere bloke disjuntuak kopiatzen ditu barrura eta kanpora erakusle partekatu baten bidez; bloke bakoitzaren mugak argitalpenean ere egiaztatzen dira, eta proba-eraikitzeek egiaztatzen dute bi harik ez dutela inoiz bloke bera ukitzen |
| `dd`, `quad` | FMA-rekin eraikitako baturei eta kernelei deitzea, prozesadoreak FMA duela egiaztatu ondoren |
| `mpi`, `opencl`, `vulkan`, `syslib` | MPI liburutegia eta GPU kontrolatzaileak kargatzea eta haiei deitzea, exekuzioan kargatutako C liburutegiak baitira |
| `native` | `qoa compile --emit so`-k egindako liburutegi bat kargatzea eta haren makina-kodeari deitzea, hark makinari deitzen dion bitartean; kode natiboak erregistroen bere kopia du, dei bakoitzaren inguruan sinkronizatuta, beraz rust erreferentzia batek ere ez du memoriarik partekatzen harekin ([compile.eu.md](compile.eu.md)) |

x86 eta ARM bektore-kernelak eskuz idatzitako mihiztadura ziren 0.4.4 arte; orain mugak egiaztatutako
slice-en gaineko rust arrunta dira, emaitza berberekin bitez bit eta abiadura berean.

## Argitalpenak eta menpekotasunak

- **Deskarga egiaztagarriak.** Argitalpeneko fitxategi bakoitzak SHA-256 laburpen bat (`SHA256SUMS`) eta
  eraikitze-jatorri sinatu bat ditu: edonork egiazta dezake fitxategi bat biltegi honen argitalpen-workflow-ak
  eraiki zuela commit jakin batetik, `gh attestation verify FITXATEGIA --repo planetryan/qoa`-rekin.
- **Bitar babestuak.** Linux bitarrak posizioarekiko independenteak dira, RELRO osoarekin eta pila
  ez-exekutagarriarekin, eta argitalpen-workflow-ak egiaztatzen du linux eraikitze bakoitzean; Windows
  bitarrak Control Flow Guard-ekin eraikitzen dira.
- **Eraikitze-urrats finkatuak.** CI-ko eta argitalpen-workflow-ko GitHub Action guztiak commit zehatz bati
  lotuta daude, beraz mugitutako edo bahitutako etiketa batek ezin du aldatu QOA zerk eraikitzen duen. Cargo
  eraikitzeek `--locked` erabiltzen dute.
- **Menpekotasunak.** CI-k menpekotasun bakoitzaren bertsio zehatza RustSec-en aholku-datu-basearen aurka
  egiaztatzen du. QOA-k bere `.qx` formatua berak irakurri eta idazten du eta SHA-256 berak kalkulatzen du
  (NIST-en adibideen eta `sha256sum`-en aurka egiaztatuta), beraz fitxategi-formatuak ez du kanpoko
  liburutegirik behar.
- **Etengabeko egiaztapenak.** Eraikitze bakoitzeko eraso-probaz gain, gaueko lan batek beste 64 hazirekin
  exekutatzen du, eta CI-k proba-multzo osoa zenbaki osoen gainezkatze-egiaztapenekin eta debug
  baieztapenekin exekutatzen du.

## Agintzen ez duena

- Simulagailu batek simulatzen duen ereduari buruzko galderak erantzuten ditu: txikiegia den fotoi-muga
  batek, edo gailua deskribatzen ez duen zarata-eredu batek, galdera okerrari erantzun zuzenak ematen
  dizkio. QOA-k trunkatze-ihesa jakinarazten du eta argia mugatik guztiz ateratzen denean hala esaten du,
  baina eredua aukeratzea zure esku dago.
- GPU kontrolatzaileak eta MPI liburutegiak QOA-tik kanpo daude. QOA-k haiek jakinarazten dutena
  egiaztatzen du eta GPUaren emaitzak CPUarenekin konparatzen ditu bere probetan, baina kontrolatzaile
  akastun batek bere erara huts egin dezake oraindik.
- Denbora: baliozko programa bat oso denbora luzean exekuta daiteke zilegiki (begizta erraldoi bat, egoera
  izugarri bat). Urrats-mugak shot bakoitza mugatzen du, eta memoria-egiaztapenek tamaina, baina QOA-k ez du
  asmatzen zenbat itxaron nahi duzun.
