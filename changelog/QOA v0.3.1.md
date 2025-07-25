# QOA v0.3.1 Release Notes

**Release Date:** 07/24/2025  
**Status:** Stable Release  

## Summary  
QOA v0.3.1 adds upon v0.3.0 by introducing benchmarking capabilities, further expanding instruction shortening, and implementing additional general improvements and optimizations across QOA.

## New Features  

### Benchmarking  
- **Integrated Benchmarking:** Added Criterion.rs benchmarks to measure the performance of core quantum gate operations and simulate system stress (CPU and I/O load). This allows for systematic performance tracking and optimization.  
- **Configurable Benchmarks:** Benchmarks are configurable for different qubit counts and can be run with flexible Criterion settings for detailed analysis.

### Instruction Shortening  
- **Expanded Instruction Shortcuts:** Expanded the set of short instruction aliases for various quantum operations and classical control flow, saving your fingers from typing so much.
  - Examples include `RSTQ` for `RESET`, `SPH` for `SETPHASE`, `JABS` for `JMPABS`, and many more.

### General Improvements and Optimizations  
- **Continued Performance Refinements:** Ongoing optimizations to existing vectorized quantum operations.
- **Code Clarity:** Minor refactorings and adjustments to improve code readability and maintainability.  
- **Robustness Improvements:** Small fixes and enhancements to increase the overall stability and reliability of QOA.

## Migration Guide  
- Code recompilation and/or updating may be required in some cases due to instruction alias additions.  
- Existing quantum circuit simulations should continue to function, potentially seeing minor performance gains from underlying optimizations.  
- To utilize the new benchmarking features, refer to the `benchmark.rs` and `run-benchmark.sh` (or `run-benchmark.ps1`) files. in the `/scripts` folder.

## Notes  
- Further improvement planned from my TODO list.

---

**Thank you for using QOA!**  
— *Rayan*
