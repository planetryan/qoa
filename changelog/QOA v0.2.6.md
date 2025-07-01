# QOA v0.2.6 Release Notes

**Release Date:** 01/07/2025
**Status:** Pre Release

## Summary
QOA v0.2.6 introduces a video file visualizer using QOA, making visualization of quantum native files and other file types easier with FFmpeg

## Changes

### Things I Added:
- **added automatic parsing and application of ffmpeg arguments (`-r` for framerate, `-s` or `-video_size` for resolution) from the command line, allowing dynamic adjustment of output video properties.**
- **added QOA Visualizer with `flags` updated as well as `help` to improve usage for new users**

### Things I Improved:
- **improved some things in `main.rs`, such as some optimizations, adding full `FFmpeg` support args in `visualizer.rs`, among other optimizations for preformance and size**

## Migration of old QOA source files:
- **old source files not affected, recompiling reccomended though**

## This may not seem like much, but it definetly was a nice thing to work on especially since it was on my Todo list for some time
### Thank you for using QOA & QOA Visualizer!

#### - Rayan