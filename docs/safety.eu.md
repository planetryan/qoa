# Segurtasuna eta sendotasuna

[In English: safety.md](safety.md)

QOA ematen diozun edozertan fidatzeko egina dago: zuk idatzitako programa bat, beste nonbaitetik datorren
`.qx` fitxategi bat, fitxategi hondatu bat, idazketa-akats bat, edo makina baino askoz handiagoa den exekuzio
bat. Orri honek esaten du QOA-k zer agintzen duen, non dauden haren mugak, nola probatzen den, eta kodearen
zein zati txikik behar duten oraindik rust-en `unsafe`, eta zergatik.

## Agintzen duena

Sarrera edozein dela ere, QOA-k bi gauzetako bat egiten du: programa exekutatzen du, edo gelditu egiten da
zer dagoen gaizki esaten duen mezu argi batekin, ingelesez edo euskaraz. Ez da erortzen, ez du panic
egiten, ez da mugarik gabe geratzen, eta ez du memoria betetzen sistemak hil arte. Hau hartzen du:

- **Iturburuak**: `.qoa`, `.qg` (glifo-sintaxia) eta `.qasm` (OpenQASM), utf-8 baliogabea, ausazko byteak,
  zenbaki zentzugabeak eta milaka mailako habiaraketak barne.
- **Fitxategi konpilatuak** (`.qx`): kargatutako programa batek mihiztatzaileak bere testutik eraikitzen
  duena izan behar du zehazki, eragiketaz eragiketa egiaztatuta. Existitzen ez den erregistro, qubit, modu,
  bit edo jauzi-helburu bat aipatzen duen, begizta bat hausten duen, edo neurri zentzugabe bat adierazten duen
  fitxategi hondatu edo eskuz aldatu bat hondatu gisa baztertzen da, eta makinak ez du inoiz ikusten.
- **Exekutatzen ari diren programak**: programa batek eska dezakeen baliabide bakoitza erabili aurretik
  egiaztatzen da (ikus behean mugak), eta finituak ez diren zenbakiek programa gelditzen dute, egoera
  isilik NaN bihurtu beharrean.
- **Irteera**: irteera ezin bada idatzi (disko betea), QOA gelditu egiten da eta hala esaten du, isilik
  galdu beharrean; irakurlea joaten bada (`qoa ... | head`), QOA isilik amaitzen da edozein komando-lerroko
  tresna bezala.

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
| egoerarako, eragile optikoetarako, memoria-gelaxketarako eta emaitzetarako memoria, guztira | QOA-k erabil dezakeena: hastean erabilgarri dagoen memoriaren % 90, edo `QOA_MAX_MEMORY` | behar den tamaina eta muga esaten dituen mezu bat, ezer erreserbatu aurretik |
| qubitak | 64 (eta egoerak memorian sartu behar du) | mihiztatzailearen errorea |
| modu optikoak | 0tik 63ra arteko indizeak (eta egoerak sartu behar du) | mihiztatzailearen errorea |
| bit klasikoak | c0-tik c4095-era | mihiztatzailearen errorea |
| izendun erregistroak (`QREG`, `CREG`, `OREG`, `RREG`, glifoen `{n}`) | 1etik 4096ra elementu, lehen indizea 0tik 4095era | mihiztatzailearen errorea |
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
  moztutako fitxategiak, luzera eta indize erraldoiak, NaN eta infinitua zenbakien ordez), 10.000 `.qoa`,
  `.qg` eta `.qasm` programa nahasi, eta 200.000 mailako habiaraketak dituzten programak. Bakoitza kargatu,
  mihiztatu, exekutatu (muga txikiekin) eta esportazio guztietatik pasatzen da (QOA testua, zerrenda, json,
  dot, glifoak, OpenQASM, IonQ). Edozein panic, bertan behera uzte edo geldialdik huts eginarazten du proba.
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
  egiaztatzen du.
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
