// fitxategi honek kerr zulo beltz txiki eta biziki biraka dabilen baten inguruko argia simulatzen du, bi zatitan:
//
//   1. argi-izpiak kerr metrikan (erlatibitate orokorra, fisika klasikoa): pixel bakoitzeko geodesiko nulu bat
//      kameratik atzera trazatzen da. hortik ateratzen dira zuloaren itzala, fotoi-eraztuna, akrezio-diskoa
//      (doppler efektuarekin eta gorriranzko desplazamendu grabitazionalarekin) eta atzeko zeruaren lenteo-efektua.
//   2. hawking erradiazioa, eredu kuantiko analogo gisa, QOA-rekin: horizontean hutsetik bikoteak sortzen dira
//      (bat ihes egiten du, bestea erortzen da energia negatiboarekin). maiztasun bakoitzeko, bikotea bi moduko
//      estutzea da, QOA-ren SQUEEZE eta BS aginduekin eraikia, eta QOA-k kalkulatzen ditu bere estatistika
//      zehatzak: fotoi-kopuruak beti berdinak bi moduetan, eta batez bestekoa planck-en 1/(e^(ω/T) - 1).
//
//   cargo run --release --example kerr -- --out kerr.mp4 --still kerr.png
//
// unitateak: G = c = M = 1 (zuloaren masa). bideoko hawking bikoteen kopurua QOA-ren banaketatik lagintzen da;
// pantailako kokapena irudizkoa da. txostenak esaten du zer den fisika eta zer irudikapena.

use clap::Parser;
use qoa::report::Style;
use qoa::{t, ts};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::f64::consts::{FRAC_PI_2, PI};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[derive(Parser, Debug)]
#[command(about = ts!(
    "Light around a small, fast-spinning Kerr black hole, and its Hawking pairs computed with QOA",
    "Argia kerr zulo beltz txiki eta azkar biraka dabilen baten inguruan, eta bere hawking bikoteak QOA-rekin kalkulatuta"
))]
struct Args {
    #[arg(long, default_value_t = 0.99, help = ts!("spin a/M, from 0 to below 1", "biraketa a/M, 0tik 1era (1 gabe)"))]
    spin: f64,
    #[arg(long, default_value_t = 80.0, help = ts!("camera inclination from the spin axis, degrees", "kameraren inklinazioa biraketa-ardatzetik, graduetan"))]
    inclination: f64,
    #[arg(long, default_value = "640x360", help = ts!("frame size", "irudiaren tamaina"))]
    size: String,
    #[arg(long, default_value_t = 6.0, help = ts!("video length in seconds", "bideoaren iraupena segundotan"))]
    seconds: f64,
    #[arg(long, default_value_t = 30, help = ts!("frames per second", "irudi segundoko"))]
    fps: u32,
    #[arg(long, default_value_t = 2, help = ts!("rays per pixel side (2 = 4 rays per pixel, smoother edges)", "izpiak pixel-alde bakoitzeko (2 = 4 izpi pixel bakoitzeko, ertz leunagoak)"))]
    samples: usize,
    #[arg(long, default_value_t = 7, help = ts!("random seed for the Hawking pairs", "hawking bikoteen ausazko hazia"))]
    seed: u64,
    #[arg(long, help = ts!("video output (mp4, needs ffmpeg)", "bideo-irteera (mp4, ffmpeg behar du)"))]
    out: Option<PathBuf>,
    #[arg(long, help = ts!("still image output (png with ffmpeg, otherwise ppm)", "irudi finkoa (png ffmpeg-ekin, bestela ppm)"))]
    still: Option<PathBuf>,
}

// ------------------------------------------------------------------ kerr metrika

#[derive(Clone, Copy, Debug)]
struct Kerr {
    a: f64,
}

impl Kerr {
    // kanpoko eta barneko horizonteak
    fn r_plus(&self) -> f64 {
        1.0 + (1.0 - self.a * self.a).sqrt()
    }
    fn r_minus(&self) -> f64 {
        1.0 - (1.0 - self.a * self.a).sqrt()
    }
    // horizontearen abiadura angeluarra eta gainazal-grabitatea
    fn omega_h(&self) -> f64 {
        self.a / (self.r_plus().powi(2) + self.a * self.a)
    }
    fn kappa(&self) -> f64 {
        (self.r_plus() - self.r_minus()) / (2.0 * (self.r_plus().powi(2) + self.a * self.a))
    }
    // hawking tenperatura, 1/M unitateetan
    fn temperature(&self) -> f64 {
        self.kappa() / (2.0 * PI)
    }
    // barneko orbita zirkular egonkorra (prograda): bardeen, press eta teukolsky
    fn isco(&self) -> f64 {
        let a = self.a;
        let z1 = 1.0 + (1.0 - a * a).cbrt() * ((1.0 + a).cbrt() + (1.0 - a).cbrt());
        let z2 = (3.0 * a * a + z1 * z1).sqrt();
        3.0 + z2 - ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt()
    }
    // boyer-lindquist metrikaren osagai kobarianteak (t, φ blokea eta r, θ)
    fn metric(&self, r: f64, th: f64) -> (f64, f64, f64, f64, f64) {
        let a = self.a;
        let (s, c) = (th.sin(), th.cos());
        let sigma = r * r + a * a * c * c;
        let delta = r * r - 2.0 * r + a * a;
        let big_a = (r * r + a * a).powi(2) - a * a * delta * s * s;
        let gtt = -(1.0 - 2.0 * r / sigma);
        let gtp = -2.0 * a * r * s * s / sigma;
        let gpp = big_a * s * s / sigma;
        (gtt, gtp, gpp, sigma / delta, sigma)
    }
}

// fotoi baten egoera: (r, θ, φ, p_r, p_θ); E = -p_t eta L = p_φ konstanteak dira
type Ray = [f64; 5];

// hamiltondarra H = N / (2Σ), N = Δ p_r² + p_θ² - ((r²+a²)E - aL)²/Δ + (L - aE sin²θ)²/sin²θ.
// fotoi batentzat H = 0. ekuazioak: dx/dλ = ∂H/∂p, dp/dλ = -∂H/∂x (deribatu analitikoak)
fn derivs(k: &Kerr, e: f64, l: f64, y: &Ray) -> Ray {
    let a = k.a;
    let [r, th, _, pr, pth] = *y;
    let (s, c) = (th.sin(), th.cos());
    let s = if s.abs() < 1e-9 {
        1e-9f64.copysign(s)
    } else {
        s
    };
    let s2 = s * s;
    let sigma = r * r + a * a * c * c;
    let delta = r * r - 2.0 * r + a * a;
    let w = (r * r + a * a) * e - a * l;
    let u = l - a * e * s2;
    let n = delta * pr * pr + pth * pth - w * w / delta + u * u / s2;
    let ddelta = 2.0 * r - 2.0;
    let dn_dr =
        ddelta * pr * pr - (2.0 * w * 2.0 * r * e * delta - w * w * ddelta) / (delta * delta);
    let dn_dth = 2.0 * u * (-2.0 * a * e * s * c) / s2 - 2.0 * u * u * c / (s2 * s);
    let dsig_dr = 2.0 * r;
    let dsig_dth = -2.0 * a * a * c * s;
    let two_sig = 2.0 * sigma;
    let dh_dr = dn_dr / two_sig - n * dsig_dr / (two_sig * sigma);
    let dh_dth = dn_dth / two_sig - n * dsig_dth / (two_sig * sigma);
    let dphi = (a * w / delta + l / s2 - a * e) / sigma;
    [delta * pr / sigma, pth / sigma, dphi, -dh_dr, -dh_dth]
}

// hamiltondarra bera eta carter-en konstantea Q (probetarako)
#[cfg(test)]
fn hamiltonian(k: &Kerr, e: f64, l: f64, y: &Ray) -> f64 {
    let a = k.a;
    let [r, th, _, pr, pth] = *y;
    let (s, c) = (th.sin(), th.cos());
    let sigma = r * r + a * a * c * c;
    let delta = r * r - 2.0 * r + a * a;
    let w = (r * r + a * a) * e - a * l;
    let u = l - a * e * s * s;
    (delta * pr * pr + pth * pth - w * w / delta + u * u / (s * s)) / (2.0 * sigma)
}

#[cfg(test)]
fn carter(k: &Kerr, e: f64, l: f64, y: &Ray) -> f64 {
    let (s, c) = (y[1].sin(), y[1].cos());
    y[4] * y[4] + c * c * (l * l / (s * s) - k.a * k.a * e * e)
}

fn add(y: &Ray, k: &Ray, h: f64) -> Ray {
    std::array::from_fn(|i| y[i] + h * k[i])
}

// cash-karp runge-kutta 4(5) urrats bat: (emaitza, errore-estimazioa)
fn rk45(kerr: &Kerr, e: f64, l: f64, y: &Ray, h: f64) -> (Ray, f64) {
    let f = |y: &Ray| derivs(kerr, e, l, y);
    let k1 = f(y);
    let k2 = f(&add(y, &k1, h * 0.2));
    let y3: Ray = std::array::from_fn(|i| y[i] + h * (3.0 / 40.0 * k1[i] + 9.0 / 40.0 * k2[i]));
    let k3 = f(&y3);
    let y4: Ray = std::array::from_fn(|i| y[i] + h * (0.3 * k1[i] - 0.9 * k2[i] + 1.2 * k3[i]));
    let k4 = f(&y4);
    let y5: Ray = std::array::from_fn(|i| {
        y[i] + h * (-11.0 / 54.0 * k1[i] + 2.5 * k2[i] - 70.0 / 27.0 * k3[i] + 35.0 / 27.0 * k4[i])
    });
    let k5 = f(&y5);
    let y6: Ray = std::array::from_fn(|i| {
        y[i] + h
            * (1631.0 / 55296.0 * k1[i]
                + 175.0 / 512.0 * k2[i]
                + 575.0 / 13824.0 * k3[i]
                + 44275.0 / 110592.0 * k4[i]
                + 253.0 / 4096.0 * k5[i])
    });
    let k6 = f(&y6);
    let five: Ray = std::array::from_fn(|i| {
        y[i] + h
            * (37.0 / 378.0 * k1[i]
                + 250.0 / 621.0 * k3[i]
                + 125.0 / 594.0 * k4[i]
                + 512.0 / 1771.0 * k6[i])
    });
    let four: Ray = std::array::from_fn(|i| {
        y[i] + h
            * (2825.0 / 27648.0 * k1[i]
                + 18575.0 / 48384.0 * k3[i]
                + 13525.0 / 55296.0 * k4[i]
                + 277.0 / 14336.0 * k5[i]
                + 0.25 * k6[i])
    });
    // errore erlatiboa, egoeraren eskalarekin. φ kanpoan: beste ezer ez da haren menpe, eta horizontetik hurbil
    // (boyer-lindquist koordenatuetan) mugarik gabe biratzen da, urratsak alferrik txikituz
    let err = [0, 1, 3, 4]
        .iter()
        .map(|&i| (five[i] - four[i]).abs() / (y[i].abs() + (h * k1[i]).abs() + 1e-3))
        .fold(0.0, f64::max);
    (five, err)
}

// izpi baten amaiera
#[derive(Clone, Copy, Debug, PartialEq)]
enum Hit {
    // zerura ihes: norabide asintotikoa
    Sky { th: f64, phi: f64 },
    // akrezio-diskoa: erradioa, azimuta eta maiztasun-erlazioa g = ν_ikusia / ν_igorria
    Disk { r: f64, phi: f64, g: f64 },
    // horizontea zeharkatu du
    Hole,
}

struct Scene {
    kerr: Kerr,
    r_cam: f64,
    disk_in: f64,
    disk_out: f64,
}

impl Scene {
    // izpi bat jarraitu (denboran atzera, momentuak alderantzikatuta) zerura, diskora edo zulora iritsi arte.
    // `e` eta `l` fotoi fisikoarenak dira (kamerara iristen dena), gorriranzko desplazamendurako
    fn trace(&self, e: f64, l: f64, start: Ray) -> (Hit, Ray) {
        let k = &self.kerr;
        let (ed, ld) = (-e, -l);
        let mut y = [start[0], start[1], start[2], -start[3], -start[4]];
        let mut h = 0.5;
        // harrapatzea: horizontetik 0.02-ra (fotoi-orbita guztiak hortik kanpo daude, a < 1 denean)
        let r_capture = k.r_plus() + 0.02;
        const TOL: f64 = 1e-7;
        for _ in 0..50_000 {
            let (next, err) = rk45(k, ed, ld, &y, h);
            if err > TOL {
                h *= (0.9 * (TOL / err).powf(0.25)).max(0.1);
                continue;
            }
            // diskoa: θ-k π/2 zeharkatzen du urrats honetan. ebaki-puntua urrats-tamainaren bisekzioz finkatzen da
            // (interpolazio linealak urrats handietan erradioan erroreak uzten ditu, eraztun-itxurako marrak)
            let (d0, d1) = (y[1] - FRAC_PI_2, next[1] - FRAC_PI_2);
            if d0 * d1 < 0.0 {
                let (mut lo, mut hi) = (0.0, h);
                let mut at = next;
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    let (p, _) = rk45(k, ed, ld, &y, mid);
                    if (p[1] - FRAC_PI_2) * d0 > 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                        at = p;
                    }
                    if hi - lo < 1e-9 * h.abs().max(1e-12) {
                        break;
                    }
                }
                let r = at[0];
                if r >= self.disk_in && r <= self.disk_out {
                    return (
                        Hit::Disk {
                            r,
                            phi: at[2],
                            g: self.redshift(r, e, l),
                        },
                        at,
                    );
                }
            }
            y = next;
            if y[0] < r_capture {
                return (Hit::Hole, y);
            }
            // ihesa: kamera baino urrunago eta kanpora (trazatzeko norabidean dr/dλ = Δ p_r / Σ > 0)
            if y[0] > self.r_cam * 1.5 && y[3] > 0.0 {
                return (
                    Hit::Sky {
                        th: y[1],
                        phi: y[2],
                    },
                    y,
                );
            }
            h = (h * (0.9 * (TOL / err.max(1e-16)).powf(0.2)).min(4.0)).min(0.2 * y[0]);
        }
        (Hit::Hole, y)
    }

    // diskoko gasa orbita zirkular keplertarretan (Ω = 1/(r^1.5 + a)); g = 1 / (u^t (E - Ω L))
    fn redshift(&self, r: f64, e: f64, l: f64) -> f64 {
        let k = &self.kerr;
        let omega = 1.0 / (r.powf(1.5) + k.a);
        let (gtt, gtp, gpp, _, _) = k.metric(r, FRAC_PI_2);
        let ut = 1.0 / (-(gtt + 2.0 * omega * gtp + omega * omega * gpp)).sqrt();
        1.0 / (ut * (e - omega * l))
    }

    // kameraren pixel baten norabidea -> fotoi fisikoaren E, L eta hasierako egoera. kamera behatzaile
    // ez-birakari batena da (ZAMO) r_cam-en, zulora begira
    fn camera_ray(&self, th_cam: f64, x: f64, y: f64) -> (f64, f64, Ray) {
        let k = &self.kerr;
        let (r, th, a) = (self.r_cam, th_cam, k.a);
        let s = th.sin();
        let sigma = r * r + a * a * th.cos().powi(2);
        let delta = r * r - 2.0 * r + a * a;
        let big_a = (r * r + a * a).powi(2) - a * a * delta * s * s;
        let alpha = (sigma * delta / big_a).sqrt();
        let omega = 2.0 * a * r / big_a;
        // begirada -r̂ norabidean; pantailaren x φ̂ da, y -θ̂ (iparra gora)
        let norm = (1.0 + x * x + y * y).sqrt();
        let (nr, nth, nph) = (-1.0 / norm, -y / norm, x / norm);
        // fotoi fisikoa kamerara dator: tokiko momentua (1, -n)
        let (pr_l, pth_l, pph_l) = (-nr, -nth, -nph);
        let pt = 1.0 / alpha;
        let pphi = omega / alpha + pph_l * (sigma / big_a).sqrt() / s;
        let pr_up = pr_l * (delta / sigma).sqrt();
        let pth_up = pth_l / sigma.sqrt();
        let (gtt, gtp, gpp, grr, gthth) = k.metric(r, th);
        let e = -(gtt * pt + gtp * pphi);
        let l = gtp * pt + gpp * pphi;
        (e, l, [r, th, 0.0, grr * pr_up, gthth * pth_up])
    }
}

// ------------------------------------------------------------------ irudia

// gorputz beltzaren kolorea (tanner helland-en hurbilketa, 1000-40000 K)
fn blackbody(kelvin: f64) -> [f64; 3] {
    let t = kelvin.clamp(1000.0, 40000.0) / 100.0;
    let r = if t <= 66.0 {
        1.0
    } else {
        (1.292936 * (t - 60.0).powf(-0.1332047)).clamp(0.0, 1.0)
    };
    let g = if t <= 66.0 {
        (0.3900816 * t.ln() - 0.6318414).clamp(0.0, 1.0)
    } else {
        (1.1298909 * (t - 60.0).powf(-0.0755148)).clamp(0.0, 1.0)
    };
    let b = if t >= 66.0 {
        1.0
    } else if t <= 19.0 {
        0.0
    } else {
        (0.5432068 * (t - 10.0).ln() - 1.1962541).clamp(0.0, 1.0)
    };
    [r, g, b]
}

fn hash(x: i64, y: i64) -> f64 {
    let mut z = (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (y as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

// zerua: izar ausazkoak eta 15 graduko sare ahul bat (lenteo-efektua ikusteko)
fn sky(th: f64, phi: f64) -> [f64; 3] {
    let (u, v) = (phi.rem_euclid(2.0 * PI) * 180.0, th.cos() * 180.0);
    let (cx, cy) = (u.floor() as i64, v.floor() as i64);
    let h = hash(cx, cy);
    let mut c = [0.01, 0.012, 0.02];
    if h > 0.985 {
        let (fx, fy) = (u - cx as f64 - 0.5, v - cy as f64 - 0.5);
        let b = (-(fx * fx + fy * fy) * 30.0).exp() * (h - 0.985) / 0.015 * 3.0;
        let col = blackbody(3000.0 + 9000.0 * hash(cy, cx));
        c = [c[0] + b * col[0], c[1] + b * col[1], c[2] + b * col[2]];
    }
    // sare-lerroak leunki (gradutan neurtutako distantziaren gaussiar bat)
    let dist = |deg: f64| (deg.rem_euclid(15.0) - 7.5).abs() - 7.5;
    let g = (-(dist(phi.to_degrees()) / 0.25).powi(2)).exp()
        + (-(dist(th.to_degrees()) / 0.25).powi(2)).exp();
    let g = g.min(1.0) * 0.04;
    [c[0] + g * 0.6, c[1] + g * 0.8, c[2] + g * 1.3]
}

// diskoaren kolorea: tenperatura-profila T ∝ r^(-3/4) (1 - √(r_in/r))^(1/4), ikusitako tenperatura g T,
// intentsitatea g⁴ (liouville), eta azimutean biratzen den egitura bat (biraketa diferentziala)
fn disk(scene: &Scene, r: f64, phi: f64, g: f64, time: f64) -> [f64; 3] {
    let rin = scene.disk_in;
    let profile = (r / rin).powf(-0.75) * (1.0 - (rin / r).sqrt()).max(0.0).powf(0.25);
    let omega = 1.0 / (r.powf(1.5) + scene.kerr.a);
    let p = phi - omega * time;
    // egitura: besoak eta zurrunbilo leunak (lerro meheak ez, aliasing-ik ez izateko)
    let texture = 0.75 + 0.15 * (3.0 * p + 2.5 * r.ln()).sin() + 0.1 * (5.0 * p - 1.3 * r).sin();
    let kelvin = 4200.0 * g * profile / 0.5;
    let col = blackbody(kelvin);
    let intensity = 2.2 * g.powi(4) * profile.powi(4) / 0.0625 * texture;
    [col[0] * intensity, col[1] * intensity, col[2] * intensity]
}

// ------------------------------------------------------------------ hawking bikoteak QOA-rekin

struct PairMode {
    omega: f64,
    m: i32,
    x: f64,
    squeeze: f64,
    // QOA-k kalkulatutako fotoi-kopuruaren banaketa (bikotearen ihes egiten duen kidea)
    dist: Vec<f64>,
    mean: f64,
    planck: f64,
    // P(n_ihes ≠ n_bikotekidea): bi moduko estutze zehatzean 0
    unpaired: f64,
    leakage: f64,
}

// bikote bat maiztasun bakoitzeko: tanh² r = e^(-x), x = (ω - mΩ_H)/T_H. bi moduko estutzea S(r) ⊗ S(-r) eta
// 50:50 izpi-zatitzaile batekin eraikitzen da (hutsetik), QOA programa arrunt gisa
fn pair_mode(omega: f64, m: i32, kerr: &Kerr) -> PairMode {
    let x = (omega - m as f64 * kerr.omega_h()) / kerr.temperature();
    let squeeze = (-x / 2.0).exp().atanh();
    // fotoi-muga: tarteko modu bakarreko egoera estutuek buztan luzea dute (tanh(r)^N, e^(-xN/2) inguru),
    // azken bikote-egoerak baino luzeagoa; 50/x-k 1e-10 azpitik uzten du
    let cutoff = ((50.0 / x).ceil() as usize + 6).clamp(8, 120);
    let src = format!(
        "QMODE 2 {}\nSQUEEZE o0 {}\nSQUEEZE o1 {} pi\nBS o0 o1 pi/4\n",
        cutoff, squeeze, squeeze
    );
    let prog = qoa::asm::assemble(&src, "hawking").expect("hawking programa");
    let cfg = qoa::vm::RunConfig {
        shots: 1,
        ..Default::default()
    };
    let out = qoa::vm::run(&prog, &cfg, &mut std::io::sink()).expect("hawking exekuzioa");
    let st = &out.final_state;
    let dist = st.photon_distribution(0);
    let unpaired: f64 = (0..st.len())
        .filter(|&i| st.photons(i, 0) != st.photons(i, 1))
        .map(|i| st.amplitudes()[i].norm_sqr())
        .sum();
    PairMode {
        omega,
        m,
        x,
        squeeze,
        mean: st.mean_photons(0),
        planck: 1.0 / (x.exp() - 1.0),
        unpaired,
        leakage: out.leakage,
        dist,
    }
}

fn sample(dist: &[f64], r: f64) -> usize {
    let mut acc = 0.0;
    for (n, p) in dist.iter().enumerate() {
        acc += p;
        if r < acc {
            return n;
        }
    }
    dist.len() - 1
}

// bikote baten irudikapena: kide bat argitsua kanpora doa, bestea gorria barrura (energia negatiboa)
struct Flash {
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
    born: f64,
}

fn main() {
    let args = Args::parse();
    if !(0.0..1.0).contains(&args.spin) {
        eprintln!(
            "{}",
            ts!(
                "--spin must be at least 0 and below 1",
                "--spin-ek 0 edo handiagoa eta 1 baino txikiagoa izan behar du"
            )
        );
        std::process::exit(2);
    }
    let Some((w, h)) = args
        .size
        .split_once('x')
        .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
    else {
        eprintln!(
            "{}",
            t!(
                "size '{}' should look like 640x360",
                "'{}' tamainak 640x360 itxura izan behar du",
                args.size
            )
        );
        std::process::exit(2);
    };
    let kerr = Kerr { a: args.spin };
    let scene = Scene {
        kerr,
        r_cam: 40.0,
        disk_in: kerr.isco(),
        disk_out: 14.0,
    };
    let style = Style::detect();
    let line = |label: &str, text: String| println!("{}  {}", style.label(label), text);

    line(
        ts!("hole", "zuloa"),
        t!(
            "mass M = 1, spin a = {}: horizons r+ = {:.4} and r- = {:.4}, horizon rotates at Ω = {:.4}, disk from the ISCO r = {:.3} to {}",
            "M = 1 masa, a = {} biraketa: horizonteak r+ = {:.4} eta r- = {:.4}, horizontearen biraketa Ω = {:.4}, diskoa ISCO r = {:.3}-tik {}-raino",
            kerr.a,
            kerr.r_plus(),
            kerr.r_minus(),
            kerr.omega_h(),
            scene.disk_in,
            scene.disk_out
        ),
    );

    // 1. hawking bikoteak, QOA-rekin
    let t_h = kerr.temperature();
    line(
        ts!("hawking", "hawking"),
        t!(
            "temperature T = κ/2π = {:.5} (in units of 1/M). Each mode below is a pair made from vacuum by two-mode squeezing (SQUEEZE, SQUEEZE, BS) and computed exactly by QOA",
            "tenperatura T = κ/2π = {:.5} (1/M unitateetan). Beheko modu bakoitza hutsetik bi moduko estutzearekin (SQUEEZE, SQUEEZE, BS) sortutako bikote bat da, QOA-k zehazki kalkulatua",
            t_h
        ),
    );
    let mut modes = Vec::new();
    for (x, m) in [(1.0, 0), (1.5, 0), (2.0, 0), (3.0, 0), (1.0, 1), (2.0, 1)] {
        let omega = x * t_h + m as f64 * kerr.omega_h();
        modes.push(pair_mode(omega, m, &kerr));
    }
    println!(
        "           {:>8}  {:>2}  {:>6}  {:>9}  {:>10}  {:>10}  {:>11}",
        "ω",
        "m",
        "x",
        "r",
        ts!("<n> QOA", "<n> QOA"),
        "<n> Planck",
        ts!("unpaired", "bikotegabe")
    );
    for p in &modes {
        println!(
            "           {:>8.4}  {:>2}  {:>6.2}  {:>9.5}  {:>10.6}  {:>10.6}  {:>11.1e}",
            p.omega, p.m, p.x, p.squeeze, p.mean, p.planck, p.unpaired
        );
    }
    let worst = modes
        .iter()
        .map(|p| (p.mean - p.planck).abs() / p.planck)
        .fold(0.0, f64::max);
    let leak = modes.iter().map(|p| p.leakage).fold(0.0, f64::max);
    line(
        ts!("check", "proba"),
        t!(
            "QOA's mean photon numbers match Planck's 1/(e^x - 1) to {:.1e} (relative, photon cutoff lost at most {:.1e}); escaping and infalling photon numbers are always equal. Modes with m = 1 feel the hole's rotation: x = (ω - mΩ)/T. For 0 < ω < mΩ the hole amplifies waves instead (superradiance), not modelled here",
            "QOA-ren batez besteko fotoi-kopuruak planck-en 1/(e^x - 1)-ekin bat datoz {:.1e}-raino (erlatiboa, fotoi-mugak gehienez {:.1e} galdu du); ihes egiten duten eta erortzen diren fotoien kopuruak beti berdinak dira. m = 1 moduek zuloaren biraketa sentitzen dute: x = (ω - mΩ)/T. 0 < ω < mΩ denean zuloak uhinak anplifikatzen ditu (superradiantzia), hemen modelatu gabe",
            worst,
            leak
        ),
    );
    line(
        ts!("scale", "eskala"),
        ts!(
            "real holes are cold: Hawking's temperature is about 1.2e23 K divided by the mass in kg, so a hole of the Sun's mass sits at about 6e-8 K and only a very light one (1e11 kg, about a mountain) reaches 1e12 K. The video's flashes are not to scale",
            "benetako zuloak hotzak dira: hawking-en tenperatura 1.2e23 K da gutxi gorabehera masarekin (kg) zatituta, beraz eguzkiaren masako zulo bat 6e-8 K-an dago eta oso arin bat bakarrik (1e11 kg, mendi bat inguru) iristen da 1e12 K-ra. Bideoko distirak ez daude eskalan"
        )
        .to_string(),
    );

    if args.out.is_none() && args.still.is_none() {
        line(
            ts!("image", "irudia"),
            ts!(
                "no --out or --still given: nothing rendered",
                "ez da --out edo --still eman: ez da ezer errendatu"
            )
            .to_string(),
        );
        return;
    }

    // 2. izpiak: pixel bakoitzeko geodesiko bat, behin (kamera finkoa; diskoa eta bikoteak denboran aldatzen dira)
    let started = std::time::Instant::now();
    let th_cam = args.inclination.to_radians();
    let fov = 0.62_f64;
    let scale = (fov / 2.0).tan() / (h as f64 / 2.0);
    // pixel bakoitzeko ss x ss izpi (azpilagin-sare uniformea); hits[pixel * ss² + k]
    let ss = args.samples.max(1);
    let hits: Vec<Hit> = (0..w * h * ss * ss)
        .into_par_iter()
        .map(|i| {
            let (pix, k) = (i / (ss * ss), i % (ss * ss));
            let (sx, sy) = (
                ((k % ss) as f64 + 0.5) / ss as f64,
                ((k / ss) as f64 + 0.5) / ss as f64,
            );
            let (px, py) = (
                (pix % w) as f64 + sx - w as f64 / 2.0,
                h as f64 / 2.0 - (pix / w) as f64 - sy,
            );
            let (e, l, start) = scene.camera_ray(th_cam, px * scale, py * scale);
            scene.trace(e, l, start).0
        })
        .collect();
    let per = ss * ss;
    let pixel_hits = |pix: usize| &hits[pix * per..(pix + 1) * per];
    let holes = hits.iter().filter(|h| **h == Hit::Hole).count() / per;
    let disk_px = hits
        .iter()
        .filter(|h| matches!(h, Hit::Disk { .. }))
        .count()
        / per;
    line(
        ts!("rays", "izpiak"),
        t!(
            "{} light rays traced backwards through the Kerr metric in {:.1} s: {:.1}% end in the hole (its shadow), {:.1}% on the disk, the rest in the sky",
            "{} argi-izpi kerr metrikan atzera trazatuak {:.1} s-tan: % {:.1} zuloan amaitzen dira (bere itzala), % {:.1} diskoan, gainerakoak zeruan",
            hits.len(),
            started.elapsed().as_secs_f64(),
            100.0 * holes as f64 / (w * h) as f64,
            100.0 * disk_px as f64 / (w * h) as f64
        ),
    );

    // itzalaren ertza: bikoteak hor agertzen dira (irudikapena)
    let edge: Vec<(usize, usize)> = (1..h - 1)
        .flat_map(|y| (1..w - 1).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            let hole = |x: usize, y: usize| pixel_hits(y * w + x).iter().all(|h| *h == Hit::Hole);
            hole(x, y)
                && [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .any(|(dx, dy)| !hole((x as i64 + dx) as usize, (y as i64 + dy) as usize))
        })
        .collect();

    let frames = ((args.seconds * args.fps as f64).round() as usize).max(1);
    let dt = 1.0 / args.fps as f64;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut flashes: Vec<Flash> = Vec::new();
    let mut total_pairs = 0usize;
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let render = |time: f64, flashes: &[Flash]| -> Vec<u8> {
        let mut px = vec![0u8; w * h * 3];
        px.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                // azpilagin guztien batez bestekoa
                let mut c = [0.0; 3];
                for hit in pixel_hits(y * w + x) {
                    let s = match *hit {
                        Hit::Sky { th, phi } => sky(th, phi),
                        Hit::Disk { r, phi, g } => disk(&scene, r, phi, g, time * 12.0),
                        Hit::Hole => [0.0; 3],
                    };
                    for k in 0..3 {
                        c[k] += s[k] / per as f64;
                    }
                }
                for f in flashes {
                    let age = time - f.born;
                    let life = 0.8;
                    if !(0.0..life).contains(&age) {
                        continue;
                    }
                    let fade = 1.0 - age / life;
                    for (sign, col) in [(1.0, [0.8, 0.9, 1.6]), (-0.6, [1.4, 0.25, 0.1])] {
                        let (fx, fy) = (f.x + sign * f.dx * age, f.y + sign * f.dy * age);
                        let d2 = (x as f64 - fx).powi(2) + (y as f64 - fy).powi(2);
                        let b = 2.0 * fade * (-d2 / 2.0).exp();
                        c = [c[0] + b * col[0], c[1] + b * col[1], c[2] + b * col[2]];
                    }
                }
                for (k, v) in c.iter().enumerate() {
                    // tonu-mapaketa
                    row[x * 3 + k] = (255.0 * (1.0 - (-1.6 * v).exp())).round() as u8;
                }
            }
        });
        px
    };

    let mut ffmpeg = args.out.as_ref().map(|out| {
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
            ])
            .arg(format!("{}x{}", w, h))
            .args([
                "-framerate",
                &args.fps.to_string(),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(out)
            .stdin(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| {
                eprintln!(
                    "{}",
                    t!("cannot start ffmpeg: {}", "ezin da ffmpeg abiarazi: {}", e)
                );
                std::process::exit(1)
            })
    });
    let mut first = None;
    for f in 0..frames {
        let time = f as f64 * dt;
        // irudi honetako bikote berriak: modu bakoitzeko kopurua QOA-ren banaketatik
        for m in &modes {
            let n = sample(&m.dist, rng.random::<f64>());
            for _ in 0..n {
                if edge.is_empty() || rng.random::<f64>() > 0.25 {
                    continue;
                }
                let (x, y) = edge[rng.random_range(0..edge.len())];
                let (ux, uy) = (x as f64 - cx, y as f64 - cy);
                let norm = (ux * ux + uy * uy).sqrt().max(1.0);
                let speed = 40.0 * (1.0 + m.omega / (t_h * 4.0));
                flashes.push(Flash {
                    x: x as f64,
                    y: y as f64,
                    dx: speed * ux / norm,
                    dy: speed * uy / norm,
                    born: time,
                });
                total_pairs += 1;
            }
        }
        flashes.retain(|fl| time - fl.born < 0.8);
        let px = render(time, &flashes);
        if let Some(child) = ffmpeg.as_mut() {
            child
                .stdin
                .as_mut()
                .unwrap()
                .write_all(&px)
                .expect("ffmpeg");
        }
        if first.is_none() && f >= frames / 2 {
            first = Some(px);
        }
    }
    if let Some(mut child) = ffmpeg {
        drop(child.stdin.take());
        let status = child.wait().expect("ffmpeg");
        if !status.success() {
            eprintln!(
                "{}",
                t!("ffmpeg failed ({})", "ffmpeg-ek huts egin du ({})", status)
            );
            std::process::exit(1);
        }
        line(
            ts!("video", "bideoa"),
            t!(
                "{}: {} frames, {} Hawking pairs drawn (counts from QOA's distributions; positions on the shadow's edge are illustrative)",
                "{}: {} irudi, {} hawking bikote marraztuta (kopuruak QOA-ren banaketetatik; itzalaren ertzeko kokapenak irudizkoak dira)",
                args.out.as_ref().unwrap().display(),
                frames,
                total_pairs
            ),
        );
    }
    if let (Some(still), Some(px)) = (args.still.as_ref(), first) {
        let is_png = still
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("png"));
        let written = if is_png {
            Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                ])
                .arg(format!("{}x{}", w, h))
                .args(["-i", "-", "-frames:v", "1"])
                .arg(still)
                .stdin(Stdio::piped())
                .spawn()
                .and_then(|mut c| {
                    c.stdin.take().unwrap().write_all(&px)?;
                    c.wait()
                })
                .is_ok_and(|s| s.success())
        } else {
            let mut data = format!("P6\n{} {}\n255\n", w, h).into_bytes();
            data.extend_from_slice(&px);
            std::fs::write(still, data).is_ok()
        };
        if written {
            line(ts!("image", "irudia"), still.display().to_string());
        } else {
            eprintln!(
                "{}",
                t!(
                    "could not write {}",
                    "ezin izan da {} idatzi",
                    still.display()
                )
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // urrunetik datorren izpi ekuatorial bat, b talka-parametroarekin, zulorantz
    fn incoming(k: &Kerr, b: f64, r0: f64) -> (f64, f64, Ray) {
        let (e, l) = (1.0, b);
        let a = k.a;
        let delta = r0 * r0 - 2.0 * r0 + a * a;
        let w = (r0 * r0 + a * a) * e - a * l;
        let u = l - a * e;
        // H = 0 -> Δ p_r² = w²/Δ - u²  (θ = π/2, p_θ = 0)
        let pr = -((w * w / delta - u * u) / delta).sqrt();
        (e, l, [r0, FRAC_PI_2, 0.0, pr, 0.0])
    }

    // izpia aurrera jarraitu (fisikoki), ihes edo harrapatze arte
    fn follow(k: &Kerr, e: f64, l: f64, mut y: Ray, r_max: f64) -> (bool, Ray, f64, f64) {
        let (h0, q0) = (hamiltonian(k, e, l, &y), carter(k, e, l, &y));
        let mut h = 1.0;
        let mut worst_h: f64 = h0.abs();
        for _ in 0..1_000_000 {
            let (next, err) = rk45(k, e, l, &y, h);
            if err > 1e-10 {
                h *= 0.5;
                continue;
            }
            y = next;
            worst_h = worst_h.max(hamiltonian(k, e, l, &y).abs());
            if y[0] < k.r_plus() + 0.02 {
                return (true, y, worst_h, (carter(k, e, l, &y) - q0).abs());
            }
            if y[0] > r_max && y[3] > 0.0 {
                return (false, y, worst_h, (carter(k, e, l, &y) - q0).abs());
            }
            h = (h * 1.5).min(0.05 * y[0]);
        }
        panic!("ez da amaitu");
    }

    #[test]
    fn schwarzschild_photon_capture_threshold_is_3_sqrt_3() {
        let k = Kerr { a: 0.0 };
        let bc = 27f64.sqrt();
        for (b, captured) in [(bc * 0.99, true), (bc * 1.01, false)] {
            let (e, l, y) = incoming(&k, b, 500.0);
            assert_eq!(follow(&k, e, l, y, 600.0).0, captured, "b = {}", b);
        }
    }

    #[test]
    fn weak_field_deflection_is_4m_over_b() {
        // urrutiko izpia: desbideratzea 4M/b + 15π/4 (M/b)²; distantzia finituaren zuzenketarekin
        let k = Kerr { a: 0.0 };
        let (b, r0) = (300.0, 2.0e5);
        let (e, l, y) = incoming(&k, b, r0);
        let (captured, end, _, _) = follow(&k, e, l, y, r0);
        assert!(!captured);
        let straight = PI - (b / r0).asin() - (b / end[0]).asin();
        let deflection = end[2] - straight;
        let want = 4.0 / b + 15.0 * PI / 4.0 / (b * b);
        assert!(
            (deflection - want).abs() < 0.01 * want,
            "{} vs {}",
            deflection,
            want
        );
    }

    #[test]
    fn hamiltonian_and_carter_constant_are_conserved() {
        // kerr biraka, ekuatoretik kanpoko izpia: H = 0 eta Q konstanteak geratzen dira
        let k = Kerr { a: 0.9 };
        let scene = Scene {
            kerr: k,
            r_cam: 40.0,
            disk_in: 100.0,
            disk_out: 101.0,
        };
        for (x, y) in [(0.05, 0.1), (-0.12, 0.03), (0.2, -0.2)] {
            let (e, l, start) = scene.camera_ray(1.2, x, y);
            assert!(hamiltonian(&k, e, l, &start).abs() < 1e-12);
            let (_, end, worst, dq) = follow(
                &k,
                -e,
                -l,
                [start[0], start[1], start[2], -start[3], -start[4]],
                60.0,
            );
            let q = carter(&k, e, l, &start).abs().max(1.0);
            assert!(worst < 1e-6, "H {}", worst);
            assert!(dq / q < 1e-6, "Q {}", dq);
            assert!(end[0].is_finite());
        }
    }

    #[test]
    fn kerr_shadow_is_shifted_by_the_spin() {
        // ekuatoreko itzala: biraketaren norabidean (prograda) harrapatze-muga txikiagoa da
        let k = Kerr { a: 0.9 };
        let edge = |sign: f64| {
            let (mut lo, mut hi) = (1.0, 10.0);
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                let (e, l, y) = incoming(&k, sign * mid, 500.0);
                if follow(&k, e, l, y, 600.0).0 {
                    lo = mid
                } else {
                    hi = mid
                }
            }
            lo
        };
        // a = 0.9-rako mugak, R(r) = ((r² + a²) - a b)² - Δ (b - a)² potentzialetik modu independentean
        // kalkulatuak: prograda 2.8444, retrograda 6.8323
        let (pro, retro) = (edge(1.0), edge(-1.0));
        assert!(
            (pro - 2.8444).abs() < 0.01 && (retro - 6.8323).abs() < 0.01,
            "{} {}",
            pro,
            retro
        );
    }

    #[test]
    fn hawking_pairs_from_qoa_follow_planck() {
        let k = Kerr { a: 0.99 };
        for x in [1.0, 2.0, 4.0] {
            let p = pair_mode(x * k.temperature(), 0, &k);
            assert!(
                (p.mean - p.planck).abs() / p.planck < 1e-4,
                "x={} {} {}",
                x,
                p.mean,
                p.planck
            );
            assert!(p.unpaired < 1e-10, "{}", p.unpaired);
        }
    }

    #[test]
    fn hole_numbers() {
        let k = Kerr { a: 0.0 };
        assert!((k.r_plus() - 2.0).abs() < 1e-12 && (k.isco() - 6.0).abs() < 1e-9);
        // schwarzschild-en tenperatura 1/(8π M)
        assert!((k.temperature() - 1.0 / (8.0 * PI)).abs() < 1e-12);
        let k = Kerr { a: 0.99 };
        assert!((k.isco() - 1.4545).abs() < 1e-3, "{}", k.isco());
    }
}
