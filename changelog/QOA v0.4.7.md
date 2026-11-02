# QOA v0.4.7 Release Notes

**Status:** in progress, not released yet (euskaraz behean)

## Added

* **`qoa export --to glyph --keep-source file.qoa`** converts a plain QOA source to glyphs line by line,
  keeping every comment, label, blank line and the layout. Each line becomes one statement in the shortest
  form (`Hq0`, `#2<=#1+3`, `!c0>again`, `PF"x"/#1`); a line glyphs cannot say stays as plain QOA between
  backticks. The result is assembled again and must be the same program, line numbers included.
* **The assembler written in QOA, in glyphs:** [tools/asm.qg](../tools/asm.qg) is `tools/asm.qoa` converted
  that way. A test checks that both give the same program byte for byte.

---

# QOA v0.4.7 bertsio-oharrak

**Egoera:** lanean, oraindik argitaratu gabe

## Gehitua

* **`qoa export --to glyph --keep-source fitxategia.qoa`**-k QOA arrunteko iturburu bat glifora bihurtzen du
  lerroz lerro, iruzkin, etiketa eta lerro huts guztiak eta antolaketa gordeta. Lerro bakoitza adierazpen bat
  bihurtzen da forma laburrenean (`Hq0`, `#2<=#1+3`, `!c0>berriz`, `PF"x"/#1`); glifoek esan ezin duten lerro bat
  QOA arrunt gisa geratzen da ` artean. Emaitza berriz mihiztatzen da eta programa bera izan behar du,
  lerro-zenbakiak barne.
* **QOAn idatzitako mihiztatzailea, glifoetan:** [tools/asm.qg](../tools/asm.qg) horrela bihurtutako
  `tools/asm.qoa` da. Proba batek egiaztatzen du biek programa bera ematen dutela byte bat aldatu gabe.
