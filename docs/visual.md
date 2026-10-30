# Music through light: `qoa visual`

*[Euskaraz](visual.eu.md)*

`qoa visual` sends audio through a simulated quantum optical circuit and writes the result as audio, video
or both. The light is computed exactly. The video is art that follows the processed sound.

```
qoa visual song.mp3 out.mp4                          # video with the processed audio
qoa visual song.flac out.flac --audio-only           # processed audio only
qoa visual song.mp3 out.mp4 --keep-audio             # original audio, video from the processed light
qoa visual song.mp3 out.mp4 --video-only             # video, no audio track
ffmpeg -i song.mp3 -f s16le -ac 1 -ar 48000 - | qoa visual - out.wav --raw s16le --rate 48000 --channels 1 --audio-only
```

The input is anything ffmpeg can read (mp3, flac, wav, ogg, the audio of a video...), or raw samples from a
file or from standard input (`-`) with `--raw FORMAT --rate HZ --channels N`. The output's extension picks
the container. Defaults: 24-bit for `.wav` and `.flac`, FLAC inside `.mkv`, AAC at 320 kb/s inside `.mp4`.
Anything after `--` goes to ffmpeg as it is.

## What happens to the sound

1. **Decoding.** ffmpeg decodes the input to 64-bit floating point, every channel kept.
2. **Modes.** Each channel is split into optical modes with an orthonormal lapped transform: an MDCT with a
   sine window and 50% overlap, `--block` modes per block (default 1024). Every sample is one mode, and
   because the modes are orthonormal, the light's photon number is the sound's energy.
3. **Light.** Each mode carries a coherent state |α⟩. A full-scale tone has `--photons` photons per sample
   (default 100,000). The music rides on the X quadrature, with the local oscillator as the phase reference.
4. **Circuit**, in this order:
   - `--mix RAD`: a mesh of beamsplitters between neighbouring bands;
   - `--loss L`: a fraction L of the light is lost. A classical gain after detection restores the loudness,
     so loss is heard as a worse signal-to-noise ratio;
   - `--squeeze DB`: the source is squeezed in X, so the measured quadrature is quieter than shot noise, as
     in gravitational-wave detectors;
   - `--kerr RAD`: Kerr self-phase modulation, U = exp(−iκ n(n−1)). The brightest mode picks up RAD
     radians, quieter modes less: an amplitude-dependent phase that bends the sound. Its mean field
     collapses as the phase spreads.
5. **Detection** (`--detect`):
   - `homodyne` (default): one measurement of X per mode, with the real vacuum noise. Fewer photons give
     more hiss. The noise power per sample is exactly 1/(8·photons) of full scale, before squeezing and
     loss.
   - `mean`: the exact expectation value ⟨x⟩, as if averaged over infinitely many shots. With no effects
     the sound comes back unchanged to within 1e-12.
   - `count`: photon counting, one shot per mode, with exact Poisson statistics. Counting cannot see the
     sign, which is taken from the mean field.
6. **Back to sound** with the inverse transform.

## How exact it is

Nothing in the light is approximated:

- Rotations and loss turn coherent states into coherent states.
- The squeezed homodyne noise is Gaussian in closed form.
- The Kerr mean field is α·exp(|α|²(e^(−2iκ) − 1)). `tests/light.rs` checks it against a direct sum in the
  photon-number basis, to 1e-12.
- Kerr leaves the photon statistics Poisson, and they are sampled exactly, for any mean up to billions,
  with Hörmann's transformed rejection and an exact log-probability.
- After a Kerr medium, a homodyne shot is drawn from the exact distribution of the non-Gaussian state:
  - The Kerr phase κn(n−1) is split exactly into a rotation of the coherent state and a quadratic chirp
    over the photon numbers that matter.
  - The chirp's Fourier coefficients turn the state into a superposition of rotated coherent states. One
    FFT per window size is shared by every mode, and coefficients are dropped only while their total stays
    below 1e-12.
  - That superposition is moved into a frame displaced to the state's classical point, where it is a short
    photon-number expansion. The expansion stops when its coefficients fall below 1e-17 in amplitude.
  - The shot comes from inverting the exact cumulative distribution. Its off-diagonal terms are a
    Wronskian of Hermite functions, and its diagonal terms come from a ladder-operator recurrence.
  - `tests/light.rs` checks the density against a direct sum in the photon-number basis to 1e-12 (it agrees
    to about 1e-14), and the samples' mean and variance against their closed forms.
  - A shot costs about 0.15 ms on one core when the mode's Kerr phase is small (most modes), and 1.5 to
    2.5 ms when it is large (the brightest modes with `--kerr 2`). Modes run in parallel on every core.
- The noise comes from a seeded generator per channel and block, so the same seed gives the same output on
  any number of threads.

Two combinations are not implemented yet, and are errors that say so: squeezed light through a Kerr medium
(`--squeeze` with `--kerr`), and counting squeezed light (`--squeeze` with `--detect count`).

## The video

The video is art. Each frame takes the processed sound's loudness, spectral centroid, band energies and
transients, and drives a noise field, an interference pattern, flicker and a spectrum bar. It follows the
light's output, so quantum noise and Kerr distortion show up in the picture. A layer that draws the modes'
phase-space states is planned for 0.4.2. Frames are drawn in parallel in a fixed order, so the same seed
gives the same video.

## Real hardware

The circuit is what a continuous-variable photonic processor does: coherent and squeezed light, beamsplitter
meshes, Kerr media and homodyne or photon-number detection. No public cloud device takes audio-rate coherent
states today, so `qoa visual` simulates one.
