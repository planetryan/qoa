# QOA v0.4.4 Release Notes

**Status:** in progress, not released yet (euskaraz behean)

## Changed

* Every release package and the all-platform bundle include `tools/asm.qoa`, QOA's assembler written in QOA,
  so `qoa compile --assembler tools/asm.qoa program.qoa` works with a downloaded binary alone. The release
  build checks on every system it can run that the packaged binary rebuilds the assembler identically.

---

# QOA v0.4.4 bertsio-oharrak

**Egoera:** lanean, oraindik argitaratu gabe

## Aldatua

* Argitalpen-pakete guztiek eta plataforma guztietako sortak `tools/asm.qoa` dakarte, QOAn idatzitako
  QOA-ren mihiztatzailea, `qoa compile --assembler tools/asm.qoa programa.qoa` deskargatutako bitar
  batekin bakarrik ibil dadin. Argitalpenaren eraikitzeak egiaztatzen du, exekuta dezakeen sistema
  guztietan, paketeko bitarrak mihiztatzailea berdin eraikitzen duela.
