# QOA v0.4.8 Release Notes

**Status:** in progress, not released yet (euskaraz behean)

## Changed

* **The parts of QOA written in QOA moved from `tools/` to `src/qoa/`** ([src/qoa/asm.qoa](../src/qoa/asm.qoa),
  [src/qoa/asm.qg](../src/qoa/asm.qg)): they are compiled into the `qoa` binary since 0.4.7, so they belong
  with the Rust sources. `tools/` keeps the helper scripts (Python, shell). Release packages carry them at the
  same path, `src/qoa/`, so `qoa compile --assembler src/qoa/asm.qoa` works in the repository and in a
  package alike.

---

# QOA v0.4.8 bertsio-oharrak

**Egoera:** lanean, oraindik argitaratu gabe

## Aldatua

* **QOAn idatzitako QOA-ren zatiak `tools/`-etik `src/qoa/`-ra eraman dira**
  ([src/qoa/asm.qoa](../src/qoa/asm.qoa), [src/qoa/asm.qg](../src/qoa/asm.qg)): 0.4.7-tik `qoa` binarioan
  konpilatzen dira, beraz rust-en iturburuekin doaz. `tools/`-ek laguntza-scriptak gordetzen ditu (python,
  shell). Argitalpen-paketeek bide berean dakartzate, `src/qoa/`, beraz `qoa compile --assembler
  src/qoa/asm.qoa` berdin dabil biltegian eta pakete batean.
