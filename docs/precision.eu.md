# Zenbakiak eta doitasuna

*[In English](precision.md)*

Simulagailu baten erantzuna bere aritmetika bezain ona da. Orri honek esaten du QOA-k zer kalkulatzen duen
zein doitasunetan, eta nola eskatu gehiago.

## Lehenetsita: 64 biteko anplitudeak, batura zehatzak

- **Anplitudeak** 64 biteko koma higikorreko bikoteak dira (IEEE binary64, 16 digitu esanguratsu inguru).
  Ateen kernelek biderketa-batuketa fusionatua erabiltzen dute prozesadoreak duenean, beraz biderkadura
  bakoitza behin biribiltzen da.
- **Egoeraren gaineko baturak** (araua, neurketa-probabilitateak, laginketarako banaketa metatua,
  itxarondako balioak) double-double-n egiten dira: batura bakoitza 64 biteko bi zenbaki dira, 106 bit
  inguru. 2^30 anplitudeko batura bat 64 biteko balio bakar bat bezain zehatza da orduan. Ez dago lana egin
  duten hari-kopuruaren menpe ere.
- **Txostenaren `zenbakiak` lerroak** |1 - ‖ψ‖²| erakusten du lehen shot-aren amaieran. Ate guztiak
  unitarioak dira, beraz aritmetika zehatzean 0 izango litzateke; geratzen dena exekuzioak metatutako
  biribiltzea da.

## `--precision quad`: 128 bit

```
qoa run zirkuitua.qoa --precision quad
```

Qubit-egoerako zenbaki bakoitza double-double bat da: 64 biteko bi zenbaki, haien baturak 106 bit inguru
dituela, 32 digitu esanguratsu gutxi gorabehera. Errorerik gabeko transformazioekin batzen eta biderkatzen
dira (two-sum, eta two-product biderketa-batuketa fusionatuaren bidez). Hauek hartzen ditu:

- anplitudeak;
- ateen matrizeak, sinuak, kosinuak eta erro karratuak 32 digitura kalkulatuta programaren angeluetatik;
- neurketa, ondorengo kolapsoa eta egoeraren gaineko batura guztiak.

Sarrera bat 64 bitekoa da oraindik: programan idatzitako angeluak. `.qx` formatuak 64 biteko zenbaki gisa
gordetzen ditu, beraz `pi/4` π/4 da jada 16 digitura biribilduta simulazioa hasi aurretik. QOA-k angelu
bakoitza zenbaki zehatz gisa hartzen du, eta ez dio errorerik gehitzen.

Zer ematen duen, `tests/quad.rs`-en:

| egiaztapena | 64 bit | 128 bit |
|---|---|---|
| araua 10 qubitetan 20.000 ate ausazkoren ondoren | 3e-13 desbideratuta | 4e-29 desbideratuta |
| sin, cos eta √ 50 digituko erreferentzien aurka (sin(12345.678) barne) | | 1e-31-en barruan |
| 1000 H ate, gero 8 T ate beste qubit batean | | hasierara itzultzen da 1e-30-en barruan |
| emaitza bera hari batean eta askotan | bai | bai, bitez bit |

QOA-k onartzen dituen 64 biteko sistema guztietan dabil, guztiek duten biderketa-batuketa fusionatuarekin
(x86-64-n exekuzioan detektatzen da). Motelagoa da: 22 qubiteko eta 400 ateko zirkuitu ausazko batek 3,4 s
behar ditu, 64 bitetan 0,22 s-ren aurrean, 4 nukleotan. Anplitude bakoitzeko memoria bikoitza eta lau aldiz
aritmetika gehiago daude horren atzean.

### Zer den 64 bitekoa oraindik

`--precision quad`-ek qubit-programak exekutatzen ditu. Modu optikoak dituzten programak, `--gpu` eta
`--noise-model` 64 bitekoak dira oraingoz, eta QOA-k hala esaten du doitasunak isilean nahastu ordez.

## POWER eta 128 biteko hardware-a

POWER9-k eta ondorengoek IEEE binary128 aritmetika dute hardware-an: 113 biteko esanguratsua, zuzen
biribildua. QOA-ren quad moduak double-double erabiltzen du POWER-en, POWER-en biderketa-batuketa
fusionatuaren gainean, beste toki guztietan bezala. Arrazoiak:

- **Erantzun bera toki guztietan.** Double-double zuzen biribildutako 64 biteko eragiketa berberekin
  eraikitzen da sistema guztietan, beraz exekuzio batek bit berak ematen ditu POWER, x86-64, ARM eta RISC-V-n.
  binary128 POWER-en bakarrik erabiltzeak POWER-en emaitzak besteenetatik desberdinduko lituzke azken
  digituetan.
- **POWER8-n ere bai.** POWER8-k ez du binary128 unitaterik; double-double han ere dabil.
- **Aldea 7 bit da.** binary128-k 113 bit ditu eta double-double-k 106 inguru, biak 32 digitu esanguratsu
  ingurukoak.

POWER-erako binary128 motor bat aukera gisa gehi liteke, IEEE binary128 emaitza zehatzak nahi badira.
