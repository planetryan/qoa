# Light around a Kerr black hole

[Euskaraz: kerr.eu.md](kerr.eu.md)

![A Kerr black hole with spin 0.99, seen from 80 degrees, rendered by examples/kerr.rs](images/kerr.jpg)

`examples/kerr.rs` renders light bending around a small, fast-spinning black hole (spin a = 0.99 M), and
uses QOA to compute the quantum statistics of the pairs such a hole creates from the vacuum at its horizon.

```sh
cargo run --release --example kerr -- --still kerr.png --out kerr.mp4
cargo run --release --example kerr -- --size 3840x2160 --fps 60 --out kerr-4k.mp4 -- -c:v libx265 -crf 16
cargo run --release --example kerr -- --help      # spin, inclination, distance, field of view, disk, length...
```

It has two parts, and they are different kinds of physics.

## 1. Where the light goes: general relativity (classical)

Photons follow null geodesics of the Kerr metric. For every pixel (nine per pixel by default, each at a
fixed random spot inside its ninth of the pixel) the program traces one backwards from the camera:

- **Equations.** The Hamiltonian H = ½ g^μν p_μ p_ν in Boyer-Lindquist coordinates, with its derivatives
  written out analytically, integrated in 64-bit floating point with an adaptive Cash-Karp Runge-Kutta
  method. The energy E and the angular momentum L are constants; r, θ, p_r and p_θ evolve, and so does the
  coordinate time t, which gives each ray's travel time.
- **Camera.** A zero-angular-momentum observer at r = 40 M (`--distance`), with a 36-degree vertical field
  of view (`--fov`). It turns with the frame dragging, so it sees no rotation of its own.
- **Ends.** A ray ends on the disk, in the hole, or at 5000 M in the sky, where what is left of its bending
  is about 4e-4 radians.

The rays are traced once; every frame then paints the disk at its own moment. A 1080p frame with 9 rays per
pixel takes about 3 minutes of tracing on 4 cores, then about 1.5 seconds per frame.

### What the picture shows

| part | how it is computed | physics or choice |
|---|---|---|
| shadow, photon ring, the far side lifted over the top, the image of the underside | the geodesics | physics |
| disk temperature | Page-Thorne thin disk: the relativistic flux with no torque at the innermost stable orbit; the gas is hottest at r ≈ 2 M | physics; the peak (`--temperature`, 25,000 K) is the scene's choice |
| disk colour and brightness | a blackbody at g·T, exactly: I_ν/ν³ is conserved along a ray, so a blackbody stays a blackbody at the shifted temperature. g combines the Doppler shift and the gravitational redshift. The spectrum is integrated over the CIE 1931 colour matching functions into sRGB | physics |
| when you see each part | each ray carries its travel time, so the disk is drawn as it was when that light left it; the second image is older than the first | physics |
| stars | positions and brightness lensed: each pixel's footprint on the sky gives the magnification, so a star's light is conserved, brighter where the lens magnifies and fainter where it squeezes | the lensing is physics; the star field is a scene |
| galaxy | a band of stars and gas with dust lanes, behind the hole, lensed into rings | scene |
| disk texture | log-normal brightness changes carried by the orbiting gas and sheared by differential rotation | artistic |
| glow and tone | a two-scale camera glow that keeps the total light, ACES tone mapping, sRGB, dithering | camera |

The physics is tested (`cargo test --release --example kerr`):

| check | result |
|---|---|
| non-spinning hole: photons are captured below impact parameter 3√3 M | yes, within 1% |
| weak field: deflection 4M/b + 15π/4 (M/b)² at b = 300 M | within 1% |
| spinning hole (a = 0.9): equatorial capture edges at b = 2.8444 M (with the spin) and 6.8323 M (against it), computed independently from the radial potential | within 0.01 M |
| H = 0 and the Carter constant stay constant along off-equator rays | to 1e-6 |
| Page-Thorne flux against the relativistic integral it comes from, for spins 0, 0.5 and 0.99 | to 1e-6 |
| travel time of a radial ray against Schwarzschild's Δr + 2M ln((r - 2M)/(r0 - 2M)) | to 1e-7 |
| a star's light summed over the pixels it covers | its flux, to 1e-3 |
| colours: 6500 K is white, 3000 K red, 30,000 K blue; brightness rises with T and becomes ∝ T (Rayleigh-Jeans) | yes |

## 2. Energy in and out of existence: Hawking pairs (quantum, computed with QOA)

Near the horizon the vacuum produces pairs: one quantum escapes as Hawking radiation, its partner falls in
carrying negative energy, and the hole loses mass. For one frequency ω and azimuthal number m, the pair
is a two-mode squeezed vacuum with tanh² r = e^(-x), x = (ω - mΩ_H)/T_H.

QOA builds that state from its own instructions, with nothing special for black holes:

```
QMODE 2 <cutoff>
SQUEEZE o0 r
SQUEEZE o1 r pi
BS o0 o1 pi/4          ; two single-mode squeezers on a 50:50 beamsplitter = a two-mode squeezer
```

and the program checks what QOA computes against the physics:

```
hawking    temperature T = κ/2π = 0.00984 (in units of 1/M). ...
                  ω   m       x          r     <n> QOA  <n> Planck     unpaired
             0.0098   0    1.00    0.70341    0.581977    0.581977      1.3e-13
             0.0148   0    1.50    0.51311    0.287217    0.287217      1.9e-14
             0.0197   0    2.00    0.38597    0.156518    0.156518      2.6e-15
             0.0295   0    3.00    0.22695    0.052396    0.052396      5.1e-17
             0.4436   1    1.00    0.70341    0.581977    0.581977      1.3e-13
             0.4535   1    2.00    0.38597    0.156518    0.156518      2.6e-15
check      QOA's mean photon numbers match Planck's 1/(e^x - 1) to 8.8e-12 ...
```

- The escaping quantum's photon number is thermal, with mean 1/(e^x - 1): Planck's law, the spectrum of a
  black body at the Hawking temperature, comes out of nothing more than squeezing and a beamsplitter.
- The escaping and infalling photon numbers are always equal ("unpaired" is the probability that they
  differ): every quantum that comes out has a partner that went in.
- Modes with m = 1 feel the rotation through the chemical potential mΩ_H. For 0 < ω < mΩ_H a spinning hole
  amplifies waves instead and loses rotational energy (superradiance); that regime is not modelled here.

In the video, the number of pairs in each frame is drawn from the distributions QOA computed. Where they
appear (on the edge of the shadow) and how they move (one outwards, one inwards) is illustration.

## Scale, honestly

The disk's default peak of 25,000 K keeps its colours visible. Real disks are hotter: about 10^5 K around a
supermassive hole and 10^7 K (X-rays) around a stellar one. Above about 10^5 K a blackbody's visible colour
hardly changes (it is in its Rayleigh-Jeans tail), so such a disk would look the same blue-white, only
brighter.

Hawking's temperature is about 1.2e23 K divided by the hole's mass in kilograms. A hole of the Sun's mass
sits at about 6e-8 K, far colder than the cosmic microwave background; only a very light hole (1e11 kg, the
mass of a mountain) reaches 1e12 K. The flashes in the video are therefore greatly exaggerated; the shapes,
the redshifts and the statistics are not.

A naming coincidence: QOA's `KERR` instruction is the optical Kerr effect (John Kerr, 1875), unrelated to the
Kerr black hole (Roy Kerr, 1963).
