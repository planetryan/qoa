# Light around a Kerr black hole

[Euskaraz: kerr.eu.md](kerr.eu.md)

![A Kerr black hole with spin 0.99, seen from 80 degrees, rendered by examples/kerr.rs](images/kerr.png)

`examples/kerr.rs` renders light bending around a small, fast-spinning black hole (spin a = 0.99 M), and
uses QOA to compute the quantum statistics of the pairs such a hole creates from the vacuum at its horizon.

```sh
cargo run --release --example kerr -- --still kerr.png --out kerr.mp4
cargo run --release --example kerr -- --help      # spin, inclination, size, length, seed
```

It has two parts, and they are different kinds of physics.

## 1. Where the light goes: general relativity (classical)

Photons follow null geodesics of the Kerr metric. For every pixel (four per pixel by default) the
program traces one backwards from the camera:

- **Equations.** The Hamiltonian H = ½ g^μν p_μ p_ν in Boyer-Lindquist coordinates, with its derivatives
  written out analytically, integrated with an adaptive Cash-Karp Runge-Kutta method. The energy E and
  the angular momentum L are constants; r, θ, p_r and p_θ evolve.
- **Camera.** A zero-angular-momentum observer at r = 40 M (an observer that turns with the frame dragging,
  so it sees no rotation of its own).
- **Disk.** A thin disk from the innermost stable circular orbit (r = 1.45 M for a = 0.99) to 14 M, its gas
  on circular Keplerian orbits Ω = 1/(r^{3/2} + a). The colour comes from the frequency ratio g = 1/(u^t (E -
  Ω L)), which combines the Doppler shift and the gravitational redshift: the side moving towards the camera
  is brighter and bluer (intensity ∝ g⁴). The temperature profile and the swirls are an artistic model.
- **Sky.** Stars and a faint 15-degree grid, so the lensing of the background is visible.

What comes out of the equations, not from drawing: the shadow, its flattened and shifted shape (the hole's
spin drags light on one side), the image of the far side of the disk lifted over the top, and the second
image of its underside below.

The integrator is tested (`cargo test --release --example kerr`):

| check | result |
|---|---|
| non-spinning hole: photons are captured below impact parameter 3√3 M | yes, within 1% |
| weak field: deflection 4M/b + 15π/4 (M/b)² at b = 300 M | within 1% |
| spinning hole (a = 0.9): equatorial capture edges at b = 2.8444 M (with the spin) and 6.8323 M (against it), computed independently from the radial potential | within 0.01 M |
| H = 0 and the Carter constant stay constant along off-equator rays | to 1e-6 |

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

Hawking's temperature is about 1.2e23 K divided by the hole's mass in kilograms. A hole of the Sun's mass
sits at about 6e-8 K, far colder than the cosmic microwave background; only a very light hole (1e11 kg, the
mass of a mountain) reaches 1e12 K. The flashes in the video are therefore greatly exaggerated; the shapes,
the redshifts and the statistics are not.

A naming coincidence: QOA's `KERR` instruction is the optical Kerr effect (John Kerr, 1875), unrelated to the
Kerr black hole (Roy Kerr, 1963).
