# QOA v0.2.8 Release Notes

**Release Date:** 04/07/2025  
**Status:** Pre Release

## Summary
QOA v0.2.8 significantly boosts simulation performance for larger qubit counts through extensive parallelization, alongside crucial bug fixes and improved handling of quantum state operations.

## Changes

### Things I Added:
- **Added `rayon` library integration:** Enabled parallel processing for core quantum state manipulations, including amplitude updates, noise application, and normalization.

### Things I Improved:
- **Improved performance for quantum operations:** Applied parallel iterators (`par_iter_mut`, `into_par_iter().map().collect()`, `par_iter().map().sum()`) across various quantum gate applications (Hadamard, X, Y, Z, CNOT, RX, RY, RZ), noise application, and normalization, leading to substantial speedups for higher qubit counts.
- **Enhanced `run` command qubit handling:** The `--qubit` flag now more robustly initializes the quantum state with the specified qubit count, overriding inferred values.
- **Refined amplitude display logic:** The `print_amplitudes` function now uses parallel iteration for collecting and sorting amplitudes, improving efficiency when displaying top N results.
- **Improved `QInit` instruction behavior:** `QInit` now correctly preserves the existing noise configuration when re-initializing the quantum state.
- **Enhanced `apply_depolarizing_noise`:** Updated to ensure thread-safe random number generation within parallel loops by correctly managing the RNG instance.
- **Added a bunch of flags and such related to QOA computing.** You can check them out when running `qoa flags` or `cargo r flags` or `cargo r help` for help. 

### Things I Fixed:
- **Fixed General slower Qubit calculation preformance**
- **Fixed `unused mut` warning:** Resolved the persistent warning related to `temp_rng` in `src/runtime/quantum_state.rs`.

## Migration of old QOA source files:
- **Recompiling Required:** For all source files, including an updating of Syntax, if Possible.

## Notes
QOA v0.2.8 focuses on making larger quantum simulations more efficient and stable, paving the way for more complex quantum algorithm implementation.

### Thank you for using QOA!

#### - Rayan