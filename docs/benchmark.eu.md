# Makina bat neurtzea: `qoa --benchmark`

*[In English](benchmark.md)*

`qoa --benchmark`-ek (edo `--bm`, `qoa benchmark`, `qoa bench`, `qoa bm`) QOA-ren zati guztiak problema
finkoetan exekutatzen ditu eta bakoitzari puntuazio bat ematen dio. Minutu erdi inguru behar du; `--quick`-ek
problema txikiagoak erabiltzen ditu eta hamar segundo inguru behar ditu. `--json`-ek emaitza beste
programentzat inprimatzen du, eta `--no-gpu`-k GPU zatia saltatzen du.

```text
makina     Intel(R) Xeon(R) Processor @ 2.80GHz (x86_64), 4 hari, avx512 kernelak
eskala     1000 = 4 cores of a 2.8 GHz Intel Xeon with AVX-512 (cloud VM), QOA 0.4.3; puntuazio bikoitza, bi aldiz azkarrago

  1071  memoriaren banda-zabalera  36.4 GB/s
        a[i] = b[i] + 3·c[i] hiru array-tan (STREAM triad): simulazio handi guztiak mugatzen dituena
        3 × 64 MiB
        4tik onena: 0.006 s (mediana 0.006 s)
  ...
puntuazioa 1004 (11 CPU atalen batez besteko geometrikoa); GPUa 1337
```

## Eskala

Atal bakoitzak 1000 puntu ditu erreferentziako makinan: 2,8 GHz-eko Intel Xeon baten 4 nukleo AVX-512-rekin,
hodeiko makina birtual bat. Puntuazio bikoitzak atal horretan bi aldiz azkarrago esan nahi du. Guztizkoa CPU
atalen batez besteko geometrikoa da, SPEC-ek egiten duen bezala, atal bakar batek ez dezan erabaki.

GPU atalak bere puntuazioa du, makina guztiek ez baitute GPUrik. Han 1000-k esan nahi du GPUak zirkuitua
erreferentziako makinaren CPUa bezain azkar exekutatzen duela, beraz 10000 puntu dituen GPU bat CPU hori baino
hamar aldiz azkarragoa da.

`--quick`-ek problema txikiagoak exekutatzen ditu, maizago sartzen direnak cache-n, beraz haren puntuazioak ez
dira proba osoarenekin konparagarriak. Konparatu proba osoak proba osoekin.

## Atal bakoitzak zer neurtzen duen

| atala | zer egiten duen | unitatea |
|---|---|---|
| memoriaren banda-zabalera | STREAM triad hiru array-tan: simulazio handi guztien muga | GB/s |
| mihiztatzailea | 100.000 lerroko programa sortu bat analizatu, egiaztatu eta mihiztatzen du | milaka lerro segundoko |
| QOA makina | QOAn idatzitako QOA-ren mihiztatzailea ([tools/asm.qoa](../tools/asm.qoa)) exekutatzen du bere 10.000 lerroko iturburuarekin: QOA makinaren agindu klasikoen abiadura. Emaitza `qoa compile`-k idazten duena bytez byte dela ere egiaztatzen du | milioi QOA agindu segundoko |
| qubit-ateak, 20 eta 24 qubit | zirkuitu ausazko bat SIMD kernelen bidez, fusio eta cache-blokeekin | mila milioika ate × anplitude eguneraketa segundoko |
| laginketa | 20 qubiteko egoera bat behin simulatzen du, gero milioi bat shot laginkatzen ditu batura metatu zehatzekin | milioika shot segundoko |
| shot-ak banan-banan | erdiko neurketa, adarkatze bat eta % 1 zarata dituen programa bat, shot bakoitza hasieratik | milaka shot segundoko |
| modu optikoak | desplazamendua, estutzea, habe-zatitzaileak eta kerr 24 fotoi-kopuruko hiru modutan | eragiketak segundoko |
| 128 biteko ateak | zirkuitu ausazko bat double-double-n | milioika ate × anplitude eguneraketa segundoko |
| batura zehatzak | 16 milioi anplitudeko egoera baten araua double-double-n | mila milioika anplitude segundoko |
| argia | argi estutua galeratik eta kerr ingurune batetik, shot homodino zehatz bat modu bakoitzeko (`qoa visual`-en fisika) | milaka modu segundoko |
| wigner panelak | kerr egoeren wigner funtzio zehatzak pixelka batez bestekotuta, bideoak marrazten dituen bezala | panelak segundoko |
| GPUa | 22 qubiteko zirkuitua GPUan, egoera hara eta hona mugitzea barne | mila milioika ate × anplitude eguneraketa segundoko |

Problema guztiek hazi finkoak erabiltzen dituzte, beraz makina guztiek lan bera egiten dute zehazki. Atal
bakoitza hiru aldiz exekutatzen da (bi `--quick`-ekin) eta denbora onena kontatzen da; txostenak mediana ere
erakusten du, beraz makina zaratatsu bat bien arteko aldean ikusten da.

## Puntuazioa zer ez den

QOA-ren lan propioa neurtzen du makina honetan, ez konputazio kuantikoa oro har, eta ez beste simulagailuak.
Atal bakoitza QOA-ren benetako zati bat da eta programa batek erabiltzen duen kode bera erabiltzen du.
