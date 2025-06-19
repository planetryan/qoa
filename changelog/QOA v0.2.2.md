# QOA v0.2.2 Release Notes

Release Date: 18/06/2025
Status: Stable Pre Release

## Summary

QOA v0.2.2 is small updated stable release focused on improving instruction correctness and stability. This version addresses a issue with the regset opcode and includes minor internal changes to support future extensions.

# Changes

## Things I Fixed:

- Corrected a bug in the regset opcode that caused incorrect register assignment during instruction parsing and execution.
- Resolved inconsistencies between parsed assembly and binary output in programs that use regset.

## Things I Improved:

- Refined binary encoding logic for .qexe, .oexe, and .qoexe outputs where regset is used.
- Internal cleanup in the register handling code to prepare for unified register typing in future versions.

## Migration of old qoa source files:

- No breaking changes in this release.
- Programs using regset should be recompiled with v0.2.2 to avoid incorrect behavior.
- Existing binaries remain compatible but may not reflect fixed semantics without recompilation.
