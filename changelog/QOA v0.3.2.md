# QOA v0.3.2 Release Notes

**Release Date:** 07/27/2025

**Status:** Stable Release

## Summary

QOA v0.3.2 builds upon previous versions by introducing significant optimization to the visualizer using SIMD, alongside continued improvements in some Unix & PowerShell scripts.

## New Features

### General Improvements and Optimizations

* **`Build.rs`** Added `Build.rs` For native & optimized builds.

* **Visualizer Optimization with SIMD:** Implemented SIMD optimizations for the quantum visualizer.

* **Improved Scripts** Improved the UNIX & PowerShell Scripts.

* **Code Clarity:** Minor refactorings and adjustments to improve code readability and maintainability.

## Migration Guide

* Code recompilation and/or updating may be required in some cases due to minor syntax changes, especially related to Quantum Visualizers

* potentially seeing minor performance gains from underlying optimizations, particularly in the visualizer, ffmpeg encoding not affected in terms of preformance.

* To utilize the new scripts refer to the `/scripts` folder.

## Notes

* Further improvement planned from my TODO list.

**Thank you for using QOA!** 
— *Rayan*