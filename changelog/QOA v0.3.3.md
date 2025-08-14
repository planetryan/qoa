# QOA v0.3.3 Release Notes

**Release Date:** 08/06/2025

**Status:** Stable Release

## Summary

QOA v0.3.3 adds expanded quantum computing math support, adds helpful debugging utilities, and extends compatibility to POWER ISA. Yes, even your POWER10 or 11 beast is supported.

## New Features

### GPU support / hybrid rendering

* I am working on a GPU kernel for Hybrid GPU / CPU rendering. Not ready yet unfortuately, but in development.

### Quantum Math Libraries

* **Added QOA Math Modules** - Initial math libraries for quantum computing, including:
  - Complex number algebra (pretty optimized I believe).
  - Matrix and vector operations tailored for QOA's gate logic.
  - Some support for quantum state math simulations.

  > Keep in note I havent finished the math library 100%, so if something is broken, let me know!

* **`/asm` Introduced** — Folder for internal math libraries used by QOA, aiming to eventually replace all external math dependencies. (primarly Intel MKL)

### Debugging Tools

* **Debug ASM** - debug asm output and build script for non x86 platforms

### Power ISA Support

* **POWER Architecture Support** — Partial support added for PowerPC64/VSX (mostly for POWER 9 or later). Includes:
  - Conditional compilation for endianness.
  - SIMD support detection.
  - Early-stage optimizations using VSX where possible.

## Migration Guide

* You *may* need to recompile QOA with updated math settings.
* If you're targeting PowerPC/POWER, ensure your Rust toolchain is set up with `powerpc64le-unknown-linux-gnu`.

## Notes

* Full MKL removal isn't done *yet* - but im getting closer. QMath (QOA math library) will soon become the default math backend.
* Expect a minor bump soon (v0.3.4) focused on some  non x86-64 optimizations.

**Thank you for using QOA.**

— *Ryan*

