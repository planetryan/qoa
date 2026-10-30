# Gailuen zarata-ereduak

*[In English](noise.md)*

`qoa run --noise-model FITXATEGIA`-k programa bat benetako gailu baten zaratarekin exekutatzen du. Gailuaren
kalibraziotik hartzen ditu ate natibo bakoitzaren errore neurtua, qubit bakoitzaren T1 eta T2, eta qubit
bakoitzaren irakurketa-errorea. Emaitza gailu horrek zer itzuliko duen QOA-ren aurreikuspena da. Ereduarik
gabe QOA-k ematen duen erantzun zehatzaren (zaratarik gabekoaren) ondoan agertzen da.

```
python3 tools/run_on_ibm.py examples/grover.qoa --dry-run --save-model grover-torino
qoa run grover-torino.qasm --noise-model grover-torino.json
```

Adibide bat prest dago: Bell, IBM Torino-rako transpilatua, eta gailu horren eredua.

```
qoa run examples/bell-torino.qasm --noise-model examples/bell-torino.json
```

## Nondik dator eredu bat

`tools/run_on_ibm.py`-k eredua gailuaren kalibraziotik bertatik eraikitzen du (qiskit-en `backend.target`,
eta `backend.properties()`-eko irakurketa-probabilitateak). Zirkuitu transpilatuak erabiltzen dituen qubit
fisikoetarako egiten du. Exekuzio guztietan egiten du, errealetan zein `--dry-run`-ekoetan, eta txostenak
"QOA + eredua" zutabe bat gehitzen du:

```
emaitza      QOA (zehatza)  QOA + eredua  gailua
101               94.5%         79.9%     80.5%
```

`--save-model IZENA`-rekin `IZENA.json` (eredua) eta `IZENA.qasm` (zirkuitu transpilatua, erabiltzen dituen
qubitetara murriztuta) ere idazten ditu. Horrela aurreikuspena berriz exekutatu, aldatu edo partekatu
daiteke, python-ik gabe.

## Zer dauka ereduak

Gailu-eredu estandarra da, qiskit aer-ek `NoiseModel.from_backend`-ekin eraikitzen duen bera, eta IBM-ren
gailu-modeloek erabiltzen dutena:

- **Ate natibo bakoitzaren ondoren** (`sx`, `x`, `rz`, `cz`, `reset`, ...), ukitu dituen qubitetan:
  1. **depolarizazio-errorea**: *q* probabilitatez, identitatea ez den ausazko pauli bat (uniformea) qubit
     horietan;
  2. **T1/T2 erlaxazioa** atearen iraupenean, haren qubit bakoitzean. Lehenik anplitude-moteltzea,
     γ = 1 − e^(−t/T1). Gero desfasea, koherentzia zehazki e^(−t/T2) izan dadin. T2 2·T1-era mugatzen
     da, fisikak eskatzen duen bezala, eta tenperatura 0 da.

  Kalibrazioaren ate-errorea (*e*) bien batera neurtutako errorea da. Beraz *q* *e*-tik askatzen da,
  erlaxazioak berak eragiten duena kendu ondoren:
  p = d·(e − (1 − F_r)) / (d·F_r − 1), q = p·(4ⁿ − 1)/4ⁿ, d = 2ⁿ izanik. F_r erlaxazioaren batez besteko
  fideltasuna da. Erlaxazioak berak *e* osoa azaltzen badu, ez dago depolarizaziorik. `reset`-ek
  erlaxazioa bakarrik du, eta `rz` birtuala da (errorea 0, iraupena 0).
- **Neurketa bakoitzean**: irakurketa-errorea. Prestatutako 0 bat 1 irakurtzen da `p0m1` probabilitatez,
  eta prestatutako 1 bat 0 `p1m0` probabilitatez. Normalean ezberdinak dira, qubita irakurtzen ari den
  bitartean erlaxa daitekeelako. Gordetako bita bakarrik aldatzen da, ez egoera.

Gailuak ez duen ate bat, edo beste qubit batzuetan duena, errorea da, ez ate ideal isil bat. Gauza bera
gertatzen da `BELL` bezalako makro batekin, X oinarriko neurketa batekin edo eragiketa optiko batekin.
Mezuak dio lehenik gailurako transpilatu behar dela, eta `tools/run_on_ibm.py`-k hori egiten du.

## Zenbat zehatza den simulazioa

Eredu batekin, shot bakoitza hasieratik simulatzen da ibilbide kuantiko gisa. Kanal bakoitza bere desegite
zehatzaren bidez aplikatzen da. Depolarizazio-errore batek bere pauli-a probabilitate zehatzekin aukeratzen
du. Anplitude-moteltzeko urrats batek |0⟩-ra jauzi egiten du γ·P(1) probabilitatez; bestela, jauzirik
gabeko eboluzio zehatza aplikatzen du, diag(1, √(1−γ)), normalizatuta. Desfasea Z bat da (1 − f)/2
probabilitatez. Beraz shot-en banaketa kanalaren banaketa bera da, ez haren hurbilketa bat. Errore
bakarrak shot kopuru mugatuaren laginketa-errorea eta 64 biteko koma higikorraren biribiltzea dira.

Hiru eratara egiaztatzen da:

- **Formula zehatzak** (`tests/noise.rs`):
  - depolarizazio-parametroa qiskit aer-enaren berdina da 1e-12-ra arte, zazpi kasutan (mugatuak barne);
  - erlaxazioak e^(−t/T1) eta e^(−t/T2) ematen ditu 1e-15-era arte;
  - T1 jaitsiera, ramsey franja (T2), irakurketa asimetrikoa eta bi qubiteko depolarizazioa bakoitza
    bere balio analitikoarekin bat dator, 5σ-ren barruan.
- **Aer-en dentsitate-matrize zehatzaren aurka**:
  - Grover, QPE eta GHZ Torino-rako transpilatuta, QOA-ren 400.000 shot aer-en dentsitate-matrizeko
    probabilitateekin konparatu ziren eredu berarekin, χ² proba batekin;
  - p-balioak: Grover 0,52, QPE 0,92, GHZ 0,37, beraz alde detektagarririk ez;
  - ate-kanal bakoitza ere superoperadore gisa konparatu zen IBM-ren gailu-modeloaren kanal berarekin:
    alderik handiena 0 izan zen.
- **`--dry-run` bakoitzean**:
  - "gailua" zutabea aer da, kalibrazio bera simulatzen;
  - txostenak bien arteko distantzia inprimatzen du, eta horren zenbat azaltzen duen laginketak berak.

## Zer uzten du kanpoan ereduak

Ereduak kalibrazio batek neurtzen duena dauka, eta ez gehiago. Benetako gailu batek hauek ere baditu:

- dekoherentzia qubit bat besteen zain dagoen bitartean (itxaronaldiak ez daude ereduan);
- ondoko qubiten arteko diafonia;
- qubit-mailetatik kanporako ihesa;
- ate-errore koherenteak (sistematikoak);
- orduetan zehar aldatzen diren kalibrazioak.

Beraz benetako gailuak ereduak baino zertxobait okerrago egiten du normalean. Exekuzio erreal batean
"QOA + eredua" eta "gailua" zutabeen arteko aldeak neurtzen du efektu horiek zenbat axola duten zirkuitu
horretarako.

Oraindik ez dago:

- adarkatze klasikoa duten zirkuituak `tools/run_on_ibm.py`-ren aurreikuspenean;
- bi qubiteko atea ECR duten gailuak (IBM-ren Eagle belaunaldia). QOA-k ez du agindu bakarrik ECR-rako.
  Heron gailuak (CZ), doako planak erabiltzen dituenak, badaude.

## Fitxategi-formatua

JSON. Denborak segundotan daude, probabilitateak 0 eta 1 artean, eta `qubits[i]` QOA-ren `qi` da:

```json
{
  "format": "qoa-noise-1",
  "device": "ibm_torino",
  "calibrated": "2026-10-29 12:00:00",
  "qubits": [
    {"physical": 92, "t1": 2.12e-4, "t2": 1.41e-4, "p0m1": 0.0093, "p1m0": 0.0088}
  ],
  "gates": [
    {"name": "sx", "qubits": [0], "error": 2.4e-4, "duration": 3.2e-8},
    {"name": "cz", "qubits": [0, 1], "error": 4.9e-3, "duration": 6.8e-8}
  ]
}
```

- `t1` edo `t2` `null` bada ezezaguna da, eta mugagabetzat hartzen da (erlaxaziorik gabe).
- Ate-izenak qiskit-enak dira.
- `cz`, `cp`, `swap` eta `rzz` bi qubit-ordenetan bilatzen dira.
- Eremu ezezagunak, tartetik kanpoko balioak, zerrendatik kanpoko qubit bat eta ate bikoiztuak erroreak dira.
- Zenbakiak idatzita dauden bezala irakurtzen dira: double bakoitzaren hamartarrak bit berberak ematen ditu.
