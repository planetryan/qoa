# QOA v0.3.0 Release Notes

**Release Date:** 07/23/2025  
**Status:** Stable Release

## Summary
QOA v0.3.0 represents a major update focused on significant performance improvements through SIMD (Single Instruction Multiple Data) vectorization and parallel processing.
This release dramatically increases the efficiency of quantum circuit simulation, especially for larger qubit systems.

## Major Features

### Instruction shortening
- I have added short instruction shortcuts, for example, `APPLYNONLINERSIGMA` is `ANLS`, and so forth. This should allow for shorter development time, but requires acryonym knowlege.

### Vectorization & SIMD Support:
- **Full SIMD Implementation:** Added comprehensive SIMD support using Rust's portable SIMD feature for quantum operations, enabling CPU architecture-specific optimizations.
- **Vectorized Quantum Gates:** Implemented vectorized versions of all core quantum gates:
  - Single-qubit gates: Hadamard, X, Y, Z, T, S, phase shifts, and rotations (Rx, Ry, Rz)
  - Two-qubit gates: CNOT, CZ, SWAP, controlled phase rotations
  - Multi-qubit operations: controlled SWAP (Fredkin gate), Toffoli gates etc.
- **Optimized Reset Operations:** Added vectorized qubit reset operations with `apply_reset_vectorized` and `apply_reset_all_vectorized` for efficient state reinitialization.
- **Custom Unitary Support:** Added vectorized implementation for applying custom unitary matrices to quantum states.

### Parallel Processing:
- **Rayon Integration:** Leveraged Rayon parallel iterators throughout the codebase for multi-threaded execution.
- **Parallel State Vector Operations:** Implemented SIMD parallel processing for:
  - State vector normalization
  - Amplitude probability calculations
  - State validation checks
  - Quantum gate application

### Performance Improvements:
- **Reduced Memory Bandwidth:** Optimized algorithms to minimize memory transfers during quantum operations.
- **Cache-Friendly Algorithms:** Restructured quantum operations to improve cache utilization.
- **Improved Scaling:** Better performance scaling with increasing qubit count, allowing simulation of larger quantum systems.
- **Reduced Overhead:** Minimized computational overhead in repeated quantum operations.

### Architectural Enhancements:
- **Modular Vectorization Layer:** Created a dedicated `vectorization.rs` file that abstracts platform-specific optimizations.
- **SIMD Optimizations:** Added specific optimizations for x86_64 architecture with AVX-512/AVX2/AVX support. Also including ARM Neon and RISC-V RVV 1.0
- **Portable SIMD Fallback:** Implemented fallback to Rust's portable SIMD when platform-specific optimizations are unavailable.

## Technical Improvements:
- **Comprehensive Test Suite:** Added extensive tests for all vectorized quantum operations to ensure correctness.
- **RNG Improvements:** Enhanced random number generation for measurement and noise simulation with better thread safety.
- **Improved Error Handling:** Better error propagation and handling in vectorized operations.
- **Memory Optimizations:** Reduced unnecessary allocations during quantum operations.

## New Instructions:
- Improved and added extra support for efficient execution of control flow instructions including conditional jumps, subroutine calls, and arithmetic operations.
- Improved noise simulation capabilities with configurable noise models.

## Migration Guide:
- Code recompilation and/or updating absolutely required in some cases.
- Existing quantum circuit simulations should see automatic performance improvements without code changes.
- For advanced users implementing custom gates: review the `vectorization.rs` for optimal performance.

## System Requirements:
- **CPU:** Processors with SIMD support (AVX-512/AVX2 recommended in x86_64 for best performance, NEON for AARCH64, and RVV 1.0 for RISC-V 64 bit).
- **Memory:** Increased efficiency allows for simulation of larger quantum systems with the same memory footprint within 2^N doubling state requirements, so still exponential memory, but less compiler & executor overhead
- **Rust Compiler:** Requires Rust 1.55.0 or higher for portable SIMD support, crates may also need an update.
- **Rust Compiler Note:** USE `rustc 1.90.0-nightly (a7a1618e6 2025-07-22)` IF ISSUES ARISE! 

## Notes
- Further architecture-specific optimizations planned for future releases.

### Thank you for using QOA!

#### - Rayan