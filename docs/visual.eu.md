# Musika argitik: `qoa visual`

*[In English](visual.md)*

`qoa visual`-ek audioa zirkuitu optiko kuantiko simulatu batetik pasatzen du eta emaitza audio, bideo edo
biak gisa idazten du. Argia zehazki kalkulatzen da. Bideoa soinu prozesatuari jarraitzen dion artea da.

```
qoa visual abestia.mp3 irteera.mp4                       # bideoa audio prozesatuarekin
qoa visual abestia.flac irteera.flac --audio-only        # audio prozesatua bakarrik
qoa visual abestia.mp3 irteera.mp4 --keep-audio          # jatorrizko audioa, bideoa argi prozesatutik
qoa visual abestia.mp3 irteera.mp4 --video-only          # bideoa, audio-pistarik gabe
ffmpeg -i abestia.mp3 -f s16le -ac 1 -ar 48000 - | qoa visual - irteera.wav --raw s16le --rate 48000 --channels 1 --audio-only
qoa visual abestia.flac irteera.mkv --resolution 3840x2160 --fps 60 --phase-space full -- -c:v libx265 -crf 14
```

Sarrera ffmpeg-ek irakur dezakeen edozer da (mp3, flac, wav, ogg, bideo baten audioa...), edo lagin
gordinak fitxategi batetik edo sarrera estandarretik (`-`), `--raw FORMATUA --rate HZ --channels N`-rekin.
Irteeraren luzapenak edukiontzia aukeratzen du. Lehenetsiak: 24 bit `.wav` eta `.flac`-en, FLAC `.mkv`
barruan, AAC 320 kb/s-tan `.mp4` barruan. `--` ondorengo guztia ffmpeg-era doa bere horretan.

## Zer gertatzen zaion soinuari

1. **Deskodetzea.** ffmpeg-ek sarrera 64 biteko koma higikorrera deskodetzen du, kanal guztiak gordeta.
2. **Moduak.** Kanal bakoitza modu optikoetan banatzen da transformazio gainjarri ortonormal batekin: MDCT
   bat leiho sinusoidalarekin eta % 50 gainjartzearekin, `--block` modu bloke bakoitzeko (lehenetsia 1024).
   Lagin bakoitza modu bat da, eta moduak ortonormalak direnez, argiaren fotoi-kopurua soinuaren energia da.
3. **Argia.** Modu bakoitzak egoera koherente bat darama, |α⟩. Eskala osoko tonu batek `--photons` fotoi
   ditu lagin bakoitzeko (lehenetsia 100.000). Musika X koadraturan doa, tokiko osziladorea fase-erreferentzia
   dela.
4. **Zirkuitua**, ordena honetan:
   - `--mix RAD`: ondoko banden arteko habe-zatitzaile sare bat;
   - `--loss L`: argiaren L zati bat galtzen da. Detekzioaren ondoko irabazi klasiko batek ozentasuna
     berreskuratzen du, beraz galera seinale-zarata erlazio okerrago gisa entzuten da;
   - `--squeeze DB`: iturria X-n estututa dago, beraz koadratura neurtua tiro-zarata baino isilagoa da,
     uhin grabitazionalen detektagailuetan bezala;
   - `--kerr RAD`: kerr fase-modulazio propioa, U = exp(−iκ n(n−1)). Modu distiratsuenak RAD radian
     hartzen ditu, isilagoek gutxiago: anplitudearen araberako fase batek soinua okertzen du. Haren batez
     besteko eremua kolapsatu egiten da fasea zabaltzean.
5. **Detekzioa** (`--detect`):
   - `homodyne` (lehenetsia): X-ren neurketa bat modu bakoitzeko, hutsaren benetako zaratarekin. Fotoi
     gutxiagok xuxurla gehiago ematen du. Zarataren potentzia lagin bakoitzeko zehazki eskala osoaren
     1/(8·fotoiak) da, estutze eta galeren aurretik.
   - `mean`: ⟨x⟩ itxarondako balio zehatza, shot infinituren batez bestekoa balitz bezala. Efekturik gabe
     soinua aldatu gabe itzultzen da, 1e-12-ren barruan.
   - `count`: fotoi-zenbaketa, shot bat modu bakoitzeko, poisson estatistika zehatzarekin. Zenbaketak ezin
     du ikurra ikusi, eta batez besteko eremutik hartzen da.
6. **Berriz soinura** alderantzizko transformazioarekin.

## Zenbat zehatza den

Argian ez da ezer hurbiltzen:

- Errotazioek eta galerek egoera koherenteak egoera koherente bihurtzen dituzte.
- Homodino estutuaren zarata gaussiarra da, forma itxian.
- Kerr-en batez besteko eremua α·exp(|α|²(e^(−2iκ) − 1)) da. `tests/light.rs`-k fotoi-kopuruaren oinarriko
  batura zuzen batekin egiaztatzen du, 1e-12-raino.
- Kerr-ek fotoi-estatistika poisson uzten du, eta zehazki lagintzen da, edozein batez bestekorekin
  milaka milioiraino, hörmann-en baztertze eraldatuarekin eta log-probabilitate zehatz batekin.
- Kerr ingurune baten ondoren, shot homodino bat egoera ez-gaussiarraren banaketa zehatzetik ateratzen da:
  - Kerr fasea, κn(n−1), zehazki banatzen da egoera koherentearen biraketa batean eta axola duten
    fotoi-kopuruen gaineko chirp koadratiko batean.
  - Chirp-aren fourier-koefizienteek egoera egoera koherente biratuen gainjarketa bihurtzen dute. Leiho-tamaina
    bakoitzeko fft bat modu guztiek partekatzen dute, eta koefizienteak haien batura 1e-12 azpitik dagoen
    bitartean bakarrik ezabatzen dira.
  - Gainjarketa hori egoeraren puntu klasikora desplazatutako sistema batera eramaten da, non
    fotoi-kopuruaren garapen labur bat den. Garapena koefizienteak anplitudean 1e-17 azpitik daudenean gelditzen
    da.
  - Shot-a banaketa metatu zehatzaren alderantzizkoa da. Haren gai ez-diagonalak hermite funtzioen
    wronskiar bat dira, eta diagonalak eskailera-eragileen errepikapen batetik datoz.
  - `tests/light.rs`-k dentsitatea fotoi-kopuruaren oinarriko batura zuzen batekin egiaztatzen du 1e-12-raino
    (1e-14 inguruan bat dator), eta laginen batez bestekoa eta bariantza haien forma itxiekin.
  - Shot batek 0,15 ms inguru behar ditu nukleo batean moduaren kerr fasea txikia denean (modu gehienak), eta
    1,5 eta 2,5 ms artean handia denean (modu distiratsuenak `--kerr 2`-rekin). Moduak paraleloan doaz nukleo
    guztietan.
- Zarata kanal eta bloke bakoitzeko hazi bidezko sorgailu batetik dator, beraz hazi berak irteera bera
  ematen du edozein hari-kopururekin.

Argi estutua zehatza da konbinazio guztietan:

- **Galera.** Galeratik pasatu den argi estutua egoera gaussiar nahasia da. Zehazki idazten da P
  koadraturan zarata gaussiar klasikoz desplazatutako egoera estutu puru gisa, beraz shot bakoitzak zarata
  hori ateratzen du eta gero egoera puru batekin lan egiten du.
- **Zenbaketa.** Egoera estutu desplazatu baten fotoi-kopuruak haren funtzio karakteristikoaren,
  ⟨e^{iφn̂}⟩-ren, fft-tik datoz, eta funtzio horrek forma itxia du (integral gaussiar bat). Poisson-ekin
  egiaztatzen dira 1e-14-raino, eta fotoi-kopuruaren errepikapen zuzen batekin 1e-13-raino.
- **Kerr ingurune batetik.** Garapen bera erabiltzen da egoera koherente estutu biratuekin. Haien anplitudeek
  hiru gaiko errepikapen bati jarraitzen diote, eskala bereizian gordeta, ezer azpigainezka ez dadin.
  Dentsitatea fotoi-kopuruaren oinarriko batura zuzen batekin bat dator 1e-12-raino.
- **Batez besteko eremua.** ⟨e^{iφn̂}a⟩-k forma itxia du egoera koherente estutu baterako, eta galeraren
  zarataren gaineko batez bestekoa integral gaussiar itxia da. Kalkulu zuzenarekin bat dator 1e-12-raino.
- **`--detect mean`-ekin eta kerr-ik gabe,** estutzeak ez du eraginik: zarata bakarrik aldatzen du, eta
  batez bestekoak hori kentzen du. Txostenak hala dio.

## Bideoa

Bideoak bi geruza ditu. Edozein bereizmen eta irudi-abiadura erabil daiteke (`--resolution 3840x2160 --fps 60`
4K60-rako), eta `--` ondorengo guztia ffmpeg-era doa, beraz kodetzailea, kalitatea eta edukiontzia zuk
aukeratzen dituzu.

**Artea.** Irudi bakoitzak soinu prozesatuaren ozentasuna, zentroide espektrala, banden energiak eta
transienteak hartzen ditu, eta zarata-eremu bat, interferentzia-eredu bat, keinuak eta espektro-barra bat
gidatzen ditu. Argiaren irteerari jarraitzen dio, beraz zarata kuantikoa eta kerr distortsioa irudian ere
ikusten dira.

**Argiaren egoera kuantikoak** (`--phase-space`). Irudi bakoitzak une horretako modu distiratsuenak
erakusten ditu, detekzioaren aurretik dauden bezala. Bakoitza bere wigner funtzio gisa marrazten da: bi
koadraturen, X (horizontala) eta P (bertikala), gaineko probabilitate-mapa baten bertsio kuantikoa:

![Galeratik eta kerr ingurune batetik pasatutako argi estutuaren zortzi modu, bakoitza zenbatutako fotoi-kopuruarekin](images/phase-space.jpg)

- **Koloreak.** Laranjatik zurira, egoera dagoen tokia da. Urdinak balio negatiboak markatzen ditu, egoera
  klasiko batek ezin dituenak; kerr ingurunea nahikoa indartsua denean agertzen dira.
- **Eraztun txikiak**, izkinan, hutsaren zarataren tamaina du, eskala gisa. Egoera koherente bat tamaina
  horretako orban biribil bat da, argi estutua elipse bat, eta kerr argia ilargi-erdi bihurtzen da.
- **Marka berdea** neurtutakoa da: marra bertikal bat emaitza homodinoan, marra eten bat batez besteko
  zehatzean, edo arku bat zenbatutako fotoi-kopuruan. Arku hori jatorriaren inguruko √(2n+1) erradioko
  zirkuluaren zati bat da.
- **Antolaketa.** `row`-ek (lehenetsia) 4 panel jartzen ditu artearen gainean, `full`-ek irudia 8 panel
  handiri ematen die, eta `off`-ek itzaltzen ditu.

Panelak kalkulatuta daude, ez marraztuta:

- Kerr ingurunerik gabe egoera gaussiarra da, eta forma itxian marrazten da.
- Kerr-ekin, shot homodinoek erabiltzen duten garapen zehatz beretik dator.
- Estutzeak, galerak eta kerr-ek batera egoera nahasi bat ematen dute. Galeraren zarataren gainean batez
  bestekoa egiten da, irudia aldatzeari utzi arte (bere gailurraren 1e-3-ra) bere burua fintzen duen
  erregela batekin, gehienez 705 egoera pururekin.
- Pixel bakoitza W-ren batez bestekoa da pixel horretan, beraz pixel bat baino xehetasun finagoak ezin dira
  eredu faltsu bihurtu (aliasing). `tests/light.rs`-k integrazio zuzenarekin egiaztatzen du, 1e-5-raino.
  Panel bakoitzak 1 probabilitatea eta egoeraren batez besteko eremu zehatza dituela ere egiaztatzen du.

Kerr ingurunerik gabe panelek ia ez dute ezer kostatzen. Kerr-ekin, eta batez ere estutzea, galera eta kerr
batera daudenean, bideoaren zatirik geldoena dira: 4 nukleotan 720p-ko irudi batek 0,4 eta 0,7 s artean
behar ditu orduan.

Irudiak paraleloan marrazten dira ordena finko batean, beraz hazi berak bideo bera ematen du.

## Hardware erreala

Zirkuitua aldagai jarraituko prozesadore fotoniko batek egiten duena da: argi koherente eta estutua,
habe-zatitzaile sareak, kerr ingurunea eta detekzio homodinoa edo fotoi-kopuruarena. Gaur ez dago hodeiko
gailu publikorik audio-abiadurako egoera koherenteak hartzen dituenik, beraz `qoa visual`-ek bat simulatzen du.
