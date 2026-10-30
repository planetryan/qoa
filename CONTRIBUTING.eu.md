# QOA-n laguntzea

*[In English: CONTRIBUTING.md](CONTRIBUTING.md)*

Eskerrik asko laguntzeagatik. Orri honek QOA nola eraikitzen, egiaztatzen eta idazten den azaltzen du,
aldaketa batek bat egin dezan.

## Eraiki eta egiaztatu

```
cargo build --release
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

CI-k gauza bera exekutatzen du Linux, Windows eta macOS-en, eta oinarrizko probak qemu-n ARM64, RISC-V
(128 eta 512 bektore-luzerak), IBM POWER10, IBM Z (big-endian) eta ARM64 estatikoan. Aldaketa bat prest dago
dena berde dagoenean.

## Kodeak betetzen dituen arauak

- **Ezerk ez du itxurarik egiten.** Inplementatu gabeko agindu edo aukera bat hala dela esaten duen errore
  bat da, inoiz ez isilean ezer egiten ez duen zerbait edo zehatz gisa aurkeztutako hurbilketa bat. Emaitzek
  nola kalkulatu diren esaten dute.
- **Zehaztasuna abiaduraren aurretik.** Fisika 64 biteko koma higikorrean edo hobean kalkulatzen da; egoeraren
  gaineko baturak double-double aritmetikan (`src/dd.rs`). Zehaztasun txikiagoko hardware-ak irudiak marraz
  ditzake, inoiz ez emaitzak kalkulatu.
- **Bi hizkuntza.** Mezu, errore eta laguntza-testu guztiak ingelesez eta euskaraz daude, `t!` eta `ts!`
  makroen bidez (`src/lang.rs`); `QOA_LANG=eu` edo `QOA_LANG=en` bat aukeratzen du.
- **Iruzkinak euskaraz**, minuskulaz, `//`-rekin (`///` edo `//!` gabe). `tests/style.rs`-k hori egiaztatzen
  du biltegiko iturburu-fitxategi guztietan.
- **Baieztapen oro probatuta dago.** Adibide baten iruzkinek zer inprimatu behar duen esaten dute, eta
  `tests/programs.rs`-k egiaztatzen du; dokumentazioko zenbakiak proba batetik edo `tools/`-eko script batetik
  datoz.

## Dokumentazioa eta bertsio-oharrak idaztea

Dokumentazioa, bertsio-oharrak eta commit-mezuak irakurtzen dituen edonorentzat dira, orain edo urte batzuk
geroago:

- Azaldu zer aldatu den eta zergatik, irakurle batek egiazta dezakeen moduan: komandoa, emaitza, frogatzen
  duen proba.
- Idatzi inolako eztabaidatan egon ez den norbaitentzat: ez aipatu elkarrizketa pribaturik, zeregin-zerrenda
  pertsonalik edo nork zer erabaki zuen. Diseinuko puntu irekiak puntu ireki gisa idazten dira, ez pertsona
  bati egindako galdera gisa.
- Erabili bertsioak eta datak, ez "orain", "duela gutxi" edo "berria".
- Mantendu ingelesa eta euskara batera: `izena.md` eta `izena.eu.md`; bertsio-oharrak
  `changelog/QOA vX.Y.Z.md` dira, lehenik ingelesez eta behean euskaraz.
- Esan argi zer ez dagoen egina edo zer ez den onartzen.

## Argitaratzea

1. Ezarri bertsioa `Cargo.toml`-en eta idatzi `changelog/QOA vX.Y.Z.md`.
2. Itxaron commit-ean CI berdea izan arte, eta eraman `main` hara.
3. Exekutatu `release` workflow-a `main`-en, `release` eremuan `vX.Y.Z` jarrita (Actions fitxatik edo GitHub
   aplikaziotik). Sistema guztietarako konpilatzen du, paketea Linux, Windows eta macOS-en probatzen du,
   commit-a etiketatzen du eta argitalpena argitaratzen du changelog-a ohar gisa erabilita.
