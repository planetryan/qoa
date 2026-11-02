# QOA v0.4.6 Release Notes

**Status:** in progress, not released yet (euskaraz behean)

## Added

* **Glyphs as dense as they can be.** The `;` between statements can be left out wherever the boundary is
  clear: only a statement ending in a letter, digit or `_` followed by one starting with one needs it, and an
  assignment followed by a statement that starts with an operator. Bits and modes with `#` or `~` in front
  take an index without brackets (`#C0`, `#C0..2`, `~L1`). `Q{2}#C{2}HQ[0]CXQ[0],Q[1]MQ.#C` is a whole Bell
  program. (`Q0` cannot mean `Q[0]`: in plain QOA it is already qubit 0, which is a different qubit when Q
  is not the first register.)
* **`qoa export --to glyph --dense`** writes any program that way, with classical instructions as
  assignments (`#2<=#1+3`, `#2_+1`, `#2<=LXP@(100)`) and parameters without spaces. Like every export it is
  reassembled and compared before it is written. The assembler written in QOA comes out at 152 KB, against
  214 KB in plain QOA with the long names.
* Instructions whose text follows their other operands (STORETEXT, VERBOSELOG) can be written that way in
  glyphs: `STX#1"abc"`, `VL/Q[0]"note"`.

## Fixed

* A program that declares more classical bits than it uses (`CREG out 33` with only c0..c2 measured) wrote a
  `.qx` that QOA then refused as corrupt, and could not be exported; the text forms now carry the bit count
  (`CREG bits 33`). Found by the hostile-input test's new check that every program loads from its own `.qx`.
* `LOSS` on a mode with a large photon cutoff did work proportional to the square of the cutoff even when
  only a few photon numbers were occupied: `FOCK o0 4` then `LOSS o0 1` with a cutoff of 262,149 took 36
  seconds. It now skips photon numbers with no probability (exactly the same result) and takes 2.
* The glyph export could not write a STORETEXT whose text contains a backtick (the assembler written in QOA
  has some); it now uses the form above instead of a backtick escape.
* A register name where a statement should start (`CX/Q[0] Q[1]`, a comma forgotten) now says so: "Q is a
  register, not an instruction; is a , missing before it?"

---

# QOA v0.4.6 bertsio-oharrak

**Egoera:** lanean, oraindik argitaratu gabe

## Gehitua

* **Glifoak ahalik eta trinkoenak.** Adierazpenen arteko `;` utz daiteke muga argia den lekuetan: letra,
  digitu edo `_` batekin amaitzen den adierazpen batek, batekin hasten den baten aurretik, behar du bakarrik,
  eta esleipen batek eragile batekin hasten den adierazpen baten aurretik. `#` edo `~` aurrean duten bitek eta
  moduek kortxeterik gabeko indizea hartzen dute (`#C0`, `#C0..2`, `~L1`). `Q{2}#C{2}HQ[0]CXQ[0],Q[1]MQ.#C`
  Bell programa oso bat da. (`Q0`-k ezin du `Q[0]` esan nahi: QOA arruntean 0 qubita da jada, eta beste qubit
  bat da Q lehen erregistroa ez denean.)
* **`qoa export --to glyph --dense`**-ek edozein programa horrela idazten du, agindu klasikoak esleipen gisa
  (`#2<=#1+3`, `#2_+1`, `#2<=LXP@(100)`) eta parametroak hutsunerik gabe. Esportazio guztiak bezala, berriz
  mihiztatu eta konparatzen da idatzi aurretik. QOAn idatzitako mihiztatzailea 152 KB-tan geratzen da, QOA
  arruntean izen luzeekin 214 KB-ren aurrean.
* Testua beste eragigaien ondoren duten aginduak (STORETEXT, VERBOSELOG) horrela idatz daitezke glifoetan:
  `STX#1"abc"`, `VL/Q[0]"oharra"`.

## Konpondua

* Erabiltzen dituenak baino bit klasiko gehiago adierazten zituen programa batek (`CREG out 33`, c0..c2 bakarrik
  neurtuta) QOA-k gero hondatutzat jotzen zuen `.qx` bat idazten zuen, eta ezin zen esportatu; testu-formek
  bit-kopurua daramate orain (`CREG bits 33`). Eraso-probaren egiaztapen berriak aurkitua: programa orok bere
  `.qx`-tik kargatu behar du.
* Fotoi-muga handiko modu batean `LOSS`-ek mugaren karratuaren proportzioko lana egiten zuen fotoi-kopuru
  gutxi batzuk bakarrik okupatuta egon arren: `FOCK o0 4` eta gero `LOSS o0 1`, 262.149ko mugarekin, 36 segundo
  behar zituen. Orain probabilitaterik gabeko fotoi-kopuruak saltatzen ditu (emaitza zehazki bera) eta 2 behar ditu.
* Glifo-esportazioak ezin zuen ` bat zuen testuko STORETEXT bat idatzi (QOAn idatzitako mihiztatzaileak
  badu); orain goiko forma erabiltzen du, ` arteko ihesaren ordez.
* Adierazpen bat hasi beharko litzatekeen lekuan erregistro-izen bat (`CX/Q[0] Q[1]`, koma bat ahaztuta)
  orain hala esaten du: "Q erregistro bat da, ez agindu bat; koma bat falta al da haren aurretik?"
