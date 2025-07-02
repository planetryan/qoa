# QOA v0.2.7 Release Notes

**Release Date:** 02/07/2025  
**Status:** Pre Release

## Summary
QOA v0.2.7 delivers enhanced visualizer experience, improved audio reactivity, more robust color cycling, and additional QOA compiler, and executor improvements.

## Changes

### Things I Added:
- **Added bass-reactive visualization:** Quantum Visualizer now responds more strongly to low frequencies for a more immersive audiovisual experience.
- **Added persistent and smooth 360° color cycling:** Visual output now reliably cycles through the entire hue spectrum at a user-friendly pace.
- **Added improved handling for hue wraparound in color logic.**
- **Added more detailed error messages and warnings for missing or unsupported audio formats.**
- **Added internal code comments and documentation for easier customization**

### Things I Improved:
- **Improved HSV-to-RGB color mapping** to ensure full spectrum color transitions without color "sticking" or loss of vividness.
- **Improved argument parsing for visualizer settings**, making it easier to fine-tune FPS, resolution, and spectrum direction via command-line.
- **Improved quantum noise and measurement logic** for greater realism and smoother visual transitions.
- **Improved performance for large input files and high frame count renders.**
- **Refactored code for better maintainability and modularity.**

### Things I Fixed:
- **Fixed issue where color cycling could get stuck in yellow/green ranges.**
- **Fixed warnings about unused variables and improved variable naming consistency.**
- **Fixed bug with quantum measurement randomness not resetting as expected.**

## Migration of old QOA source files:
- **Recompiling Required For all source files, Including an updating of Syntax, if Possible.**

## Notes
QOA v0.2.7 making quantum visualization accessible, with a focus on user customization.

### Thank you for using QOA!

#### - Rayan