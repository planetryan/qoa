// fitxategi honek kerr zulo beltz txiki eta biziki biraka dabilen baten inguruko argia errendatzen du, bi zatitan:
//
//   1. argi-izpiak kerr metrikan (erlatibitate orokorra, fisika klasikoa): pixel bakoitzeko hainbat geodesiko
//      nulu kameratik atzera trazatzen dira (f64). hortik ateratzen dira zuloaren itzala, fotoi-eraztuna,
//      diskoaren irudi guztiak eta zeruaren lenteo-efektua. diskoa page-thorne disko mehe bat da (fluxu
//      erlatibista zehatza); ikusten den kolorea g·T tenperaturako gorputz beltza da zehazki (liouville),
//      CIE 1931 funtzioekin kalkulatua; izpi bakoitzak bere atzerapena dakar, beraz irudi sekundarioek diskoa
//      lehenagoko une batean erakusten dute. izarren distira lenteo-efektuaren benetako handitzearekin doa
//      (pixelaren aztarna zeruan). turbulentzia, esne-bidea eta kameraren distira (bloom) artistikoak dira
//   2. hawking erradiazioa, eredu kuantiko analogo gisa, QOA-rekin: horizontean hutsetik bikoteak sortzen dira
//      (bat ihes egiten du, bestea erortzen da energia negatiboarekin). maiztasun bakoitzeko, bikotea bi moduko
//      estutzea da, QOA-ren SQUEEZE eta BS aginduekin eraikia, eta QOA-k kalkulatzen ditu bere estatistika
//      zehatzak: fotoi-kopuruak beti berdinak bi moduetan, eta batez bestekoa planck-en 1/(e^(ω/T) - 1).
//
//   cargo run --release --example kerr -- --out kerr.mp4 --still kerr.png
//
// unitateak: G = c = M = 1 (zuloaren masa). geodesikoak behin trazatzen dira (kamera finkoa) eta irudi
// bakoitzak diskoa bere denboran margotzen du. gordetako diskoaren balioak (r, φ, atzerapena, tenperatura)
// marrazteko bakarrik dira, f32-n (8 biteko irteerarako nahikoa); fisika guztia f64-n kalkulatzen da.
// bideoko hawking bikoteen kopurua QOA-ren banaketatik lagintzen da; pantailako kokapena irudizkoa da.

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
    #[arg(long, default_value_t = 40.0, help = ts!("camera distance from the hole, in units of its mass M", "kameraren distantzia zulotik, haren masaren (M) unitateetan"))]
    distance: f64,
    #[arg(long, default_value_t = 36.0, help = ts!("vertical field of view, degrees", "ikus-eremu bertikala, graduetan"))]
    fov: f64,
    #[arg(long, default_value_t = 16.0, help = ts!("outer edge of the disk, in M", "diskoaren kanpoko ertza, M-tan"))]
    disk_out: f64,
    #[arg(long, default_value_t = 25000.0, help = ts!("hottest temperature in the disk, kelvin (set by the hole's mass and accretion rate)", "diskoko tenperatura beroena, kelvinetan (zuloaren masak eta akrezio-tasak erabakitzen dute)"))]
    temperature: f64,
    #[arg(long, default_value = "1920x1080", help = ts!("frame size, e.g. 1920x1080 or 3840x2160", "irudiaren tamaina, adib. 1920x1080 edo 3840x2160"))]
    size: String,
    #[arg(long, default_value_t = 8.0, help = ts!("video length in seconds", "bideoaren iraupena segundotan"))]
    seconds: f64,
    #[arg(long, default_value_t = 30, help = ts!("frames per second", "irudi segundoko"))]
    fps: u32,
    #[arg(long, default_value_t = 15.0, help = ts!("hole time per second of video, in M", "zuloaren denbora bideoko segundo bakoitzeko, M-tan"))]
    speed: f64,
    #[arg(long, default_value_t = 3, help = ts!("rays per pixel side (3 = 9 rays per pixel)", "izpiak pixel-alde bakoitzeko (3 = 9 izpi pixel bakoitzeko)"))]
    samples: usize,
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true, help = ts!("exposure change in stops", "esposizio-aldaketa, urratsetan (stops)"))]
    exposure: f64,
    #[arg(long, default_value_t = 7, help = ts!("random seed (stars, disk turbulence, Hawking pairs)", "ausazko hazia (izarrak, diskoaren turbulentzia, hawking bikoteak)"))]
    seed: u64,
    #[arg(long, help = ts!("video output (needs ffmpeg)", "bideo-irteera (ffmpeg behar du)"))]
    out: Option<PathBuf>,
    #[arg(long, help = ts!("still image output (png with ffmpeg, otherwise ppm)", "irudi finkoa (png ffmpeg-ekin, bestela ppm)"))]
    still: Option<PathBuf>,
    #[arg(last = true, allow_hyphen_values = true, help = ts!("arguments passed to ffmpeg after --", "-- ondoren ffmpeg-i pasatzen zaizkion argumentuak"))]
    ffmpeg_args: Vec<String>,
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

// fotoi baten egoera: (r, θ, φ, p_r, p_θ, t); E = -p_t eta L = p_φ konstanteak dira. t koordenatu-denbora
// da: izpi bakoitzaren atzerapena (argiak behar duen denbora) ematen du
type Ray = [f64; 6];

// hamiltondarra H = N / (2Σ), N = Δ p_r² + p_θ² - ((r²+a²)E - aL)²/Δ + (L - aE sin²θ)²/sin²θ.
// fotoi batentzat H = 0. ekuazioak: dx/dλ = ∂H/∂p, dp/dλ = -∂H/∂x (deribatu analitikoak); dt/dλ = -∂H/∂E =
// (W (r²+a²)/Δ + a U)/Σ
fn derivs(k: &Kerr, e: f64, l: f64, y: &Ray) -> Ray {
    let a = k.a;
    let [r, th, _, pr, pth, _] = *y;
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
    let dt = (w * (r * r + a * a) / delta + a * u) / sigma;
    [delta * pr / sigma, pth / sigma, dphi, -dh_dr, -dh_dth, dt]
}

// hamiltondarra bera eta carter-en konstantea Q (probetarako)
#[cfg(test)]
fn hamiltonian(k: &Kerr, e: f64, l: f64, y: &Ray) -> f64 {
    let a = k.a;
    let [r, th, _, pr, pth, _] = *y;
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
    // errore erlatiboa, egoeraren eskalarekin. φ eta t kanpoan: beste ezer ez da haien menpe, eta horizontetik
    // hurbil (boyer-lindquist koordenatuetan) mugarik gabe hazten dira, urratsak alferrik txikituz
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
    Sky {
        th: f64,
        phi: f64,
    },
    // akrezio-diskoa: erradioa, azimuta, maiztasun-erlazioa g = ν_ikusia / ν_igorria eta atzerapena (argiak
    // diskotik kamerara behar duen koordenatu-denbora)
    Disk {
        r: f64,
        phi: f64,
        g: f64,
        delay: f64,
    },
    // horizontea zeharkatu du
    Hole,
}

// zerua: hemendik aurrera izpiaren norabidea ez da ia aldatzen
const R_SKY: f64 = 5000.0;

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
        let mut y = [start[0], start[1], start[2], -start[3], -start[4], 0.0];
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
                            delay: -at[5],
                        },
                        at,
                    );
                }
            }
            y = next;
            if y[0] < r_capture {
                return (Hit::Hole, y);
            }
            // ihesa: urrun (5000 M, geratzen den desbideratzea 2M/r ≈ 4e-4 rad) eta kanpora (trazatzeko norabidean
            // dr/dλ = Δ p_r / Σ > 0)
            if y[0] > R_SKY && y[3] > 0.0 {
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
        (e, l, [r, th, 0.0, grr * pr_up, gthth * pth_up, 0.0])
    }
}

// ------------------------------------------------------------------ koloreak

// CIE 1931 2°-ko kolore-funtzioak: wyman, sloan eta shirley-ren (2013) lobulu anitzeko doikuntza (% 1 inguru)
fn cie(lambda: f64) -> [f64; 3] {
    let g = |mu: f64, s1: f64, s2: f64| {
        let t = (lambda - mu) / if lambda < mu { s1 } else { s2 };
        (-0.5 * t * t).exp()
    };
    [
        1.056 * g(599.8, 37.9, 31.0) + 0.362 * g(442.0, 16.0, 26.7) - 0.065 * g(501.1, 20.4, 26.2),
        0.821 * g(568.8, 46.9, 40.5) + 0.286 * g(530.9, 16.3, 31.1),
        1.217 * g(437.0, 11.8, 36.0) + 0.681 * g(459.0, 26.0, 13.8),
    ]
}

// planck-en B_λ(T), unitate erlatiboetan (2hc² gabe), λ nanometrotan
fn planck(lambda: f64, kelvin: f64) -> f64 {
    // hc/k, nm·K
    const C2: f64 = 1.438_776_877e7;
    1.0 / (lambda.powi(5) * (C2 / (lambda * kelvin)).exp_m1())
}

fn luminance(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

// sRGB-tik kanpoko kolore bat argitasun berean desaturatu, osagai negatiborik gabe
fn gamut(c: [f64; 3]) -> [f64; 3] {
    let y = luminance(c);
    let m = c.iter().copied().fold(f64::MAX, f64::min);
    if m >= 0.0 || y <= 0.0 {
        return c.map(|v| v.max(0.0));
    }
    let s = y / (y - m);
    c.map(|v| (y + (v - y) * s).max(0.0))
}

// gorputz beltzaren kolorea sRGB linealean: B(λ, T) CIE-ren funtzioekin integratuta (380-780 nm, 1 nm), XYZ →
// sRGB (D65). argitasuna fisikoa da (6500 K: Y = 1), beraz disko beroagoa distiratsuagoa da berez. taula log T-n
// 300 K-tik 1e8 K-ra; gorago rayleigh-jeans (kolore bera, argitasuna ∝ T)
struct Palette {
    lo: f64,
    step: f64,
    rgb: Vec<[f64; 3]>,
}

impl Palette {
    fn new() -> Palette {
        let xyz = |t: f64| {
            let mut s = [0.0; 3];
            for l in 380..=780 {
                let (b, c) = (planck(l as f64, t), cie(l as f64));
                for k in 0..3 {
                    s[k] += b * c[k];
                }
            }
            s
        };
        let norm = 1.0 / xyz(6500.0)[1];
        let (lo, hi, n) = (300f64.ln(), 1e8f64.ln(), 4096);
        let step = (hi - lo) / (n - 1) as f64;
        let rgb = (0..n)
            .into_par_iter()
            .map(|i| {
                let [x, y, z] = xyz((lo + i as f64 * step).exp()).map(|v| v * norm);
                gamut([
                    3.2406 * x - 1.5372 * y - 0.4986 * z,
                    -0.9689 * x + 1.8758 * y + 0.0415 * z,
                    0.0557 * x - 0.2040 * y + 1.0570 * z,
                ])
            })
            .collect();
        Palette { lo, step, rgb }
    }

    fn at(&self, kelvin: f64) -> [f64; 3] {
        if kelvin.is_nan() || kelvin <= 300.0 {
            return [0.0; 3];
        }
        let u = (kelvin.ln() - self.lo) / self.step;
        let last = self.rgb.len() - 1;
        if u >= last as f64 {
            let k = kelvin / (self.lo + last as f64 * self.step).exp();
            return self.rgb[last].map(|v| v * k);
        }
        let (i, f) = (u as usize, u.fract());
        std::array::from_fn(|k| self.rgb[i][k] * (1.0 - f) + self.rgb[i + 1][k] * f)
    }
}

// ------------------------------------------------------------------ diskoa

// page-thorne disko meheko fluxua (M = Ṁ = 1), barneko ertzean momenturik gabe, x = √r:
// F = 3/(8π x⁴ (x³ - 3x + 2a)) · [x - x0 - (3/2) a ln(x/x0) - Σ_i 3(x_i - a)²/(x_i Π_{j≠i}(x_i - x_j)) ln((x - x_i)/(x0 - x_i))]
// x_i: x³ - 3x + 2a = 0-ren erroak
fn page_thorne(a: f64, r_in: f64, r: f64) -> f64 {
    let (x, x0) = (r.sqrt(), r_in.sqrt());
    let ac = a.acos();
    let xs = [
        2.0 * ((ac - PI) / 3.0).cos(),
        2.0 * ((ac + PI) / 3.0).cos(),
        -2.0 * (ac / 3.0).cos(),
    ];
    let mut bracket = x - x0 - 1.5 * a * (x / x0).ln();
    for i in 0..3 {
        let (xi, xj, xk) = (xs[i], xs[(i + 1) % 3], xs[(i + 2) % 3]);
        bracket -=
            3.0 * (xi - a).powi(2) / (xi * (xi - xj) * (xi - xk)) * ((x - xi) / (x0 - xi)).ln();
    }
    3.0 / (8.0 * PI) * bracket / (x.powi(4) * (x.powi(3) - 3.0 * x + 2.0 * a))
}

// turbulentziaren belaunaldi baten bizitza (M): eredua sortu, biraketa diferentzialak luzatzen du, eta hurrengoak
// ordezkatzen du (bi belaunaldi aldi berean, pisuen karratuen batura 1)
const LIFE: f64 = 60.0;

struct Disk {
    a: f64,
    r_in: f64,
    r_out: f64,
    f_max: f64,
    r_peak: f64,
    t_peak: f64,
    seed: u64,
}

impl Disk {
    fn new(a: f64, r_in: f64, r_out: f64, t_peak: f64, seed: u64) -> Disk {
        let (f_max, r_peak) = (0..20_000)
            .map(|i| {
                let r = r_in * (r_out / r_in).powf(i as f64 / 19_999.0);
                (page_thorne(a, r_in, r), r)
            })
            .fold((0.0, r_in), |m, x| if x.0 > m.0 { x } else { m });
        Disk {
            a,
            r_in,
            r_out,
            f_max,
            r_peak,
            t_peak,
            seed,
        }
    }

    // gasaren tenperatura: σT⁴ = F
    fn temperature(&self, r: f64) -> f64 {
        self.t_peak * (page_thorne(self.a, self.r_in, r).max(0.0) / self.f_max).powf(0.25)
    }

    // kanpoko ertza leun (azken % 12a): eszenaren aukera, ez fisika
    fn edge(&self, r: f64) -> f64 {
        let u = ((self.r_out - r) / (0.12 * self.r_out)).clamp(0.0, 1.0);
        u * u * (3.0 - 2.0 * u)
    }

    // fluxuaren aldakuntza turbulentoa, log-normala (batez bestekoa 1), gasarekin batera orbitan: belaunaldi
    // bakoitza (r, φ - Ω(r)(t - t_j)) puntuan, beraz biraketa diferentzialak besoetan luzatzen du
    fn turbulence(&self, r: f64, phi: f64, t: f64) -> f64 {
        const SIGMA: f64 = 0.35;
        let omega = 1.0 / (r.powf(1.5) + self.a);
        let s = t / (LIFE / 2.0);
        let k = s.floor();
        let w = s - k;
        let generation = |j: f64| {
            let p = phi - omega * (t - j * LIFE / 2.0);
            fbm(
                [2.5 * p.cos(), 2.5 * p.sin(), 5.0 * r.ln()],
                4.0,
                self.seed ^ (j as i64 as u64).wrapping_mul(0x9e37_79b9),
            )
        };
        let (sw, cw) = (FRAC_PI_2 * w).sin_cos();
        let n = sw * generation(k) + cw * generation(k - 1.0);
        (SIGMA * n - SIGMA * SIGMA / 2.0).exp()
    }
}

// ------------------------------------------------------------------ zarata

fn hash3(i: i64, j: i64, k: i64, seed: u64) -> u64 {
    let mut z = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (j as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ (k as u64).wrapping_mul(0x94d0_49bb_1331_11eb)
        ^ seed.wrapping_mul(0xd6e8_feb8_6659_fd93);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn unit(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

// perlin-en gradiente-zarata 3D-n (desbideratzea 0.27)
fn noise3(p: [f64; 3], seed: u64) -> f64 {
    const G: [[f64; 3]; 12] = [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, -1.0],
    ];
    let i = p.map(|v| v.floor());
    let f: [f64; 3] = std::array::from_fn(|k| p[k] - i[k]);
    let u = f.map(|t| t * t * t * (t * (t * 6.0 - 15.0) + 10.0));
    let mut acc = 0.0;
    for c in 0..8 {
        let o = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
        let g = G[(hash3(
            i[0] as i64 + o[0] as i64,
            i[1] as i64 + o[1] as i64,
            i[2] as i64 + o[2] as i64,
            seed,
        ) % 12) as usize];
        let mut d = 0.0;
        let mut w = 1.0;
        for k in 0..3 {
            d += g[k] * (f[k] - o[k] as f64);
            w *= if o[k] == 1 { u[k] } else { 1.0 - u[k] };
        }
        acc += w * d;
    }
    acc
}

// zarata fraktala, desbideratze unitarioa gutxi gorabehera; `octaves` zatikia izan daiteke (azkena leunki sartzen
// da), xehetasuna pixelaren aztarnara mugatzeko
fn fbm(p: [f64; 3], octaves: f64, seed: u64) -> f64 {
    let (mut sum, mut norm, mut amp, mut freq) = (0.0, 0.0, 1.0, 1.0);
    let mut o = 0;
    while (o as f64) < octaves {
        let w = (octaves - o as f64).min(1.0);
        sum += w * amp * noise3(p.map(|v| v * freq), seed.wrapping_add(o as u64 * 7919));
        norm += (w * amp).powi(2);
        amp *= 0.5;
        freq *= 2.0;
        o += 1;
    }
    sum / (0.27 * norm.sqrt().max(1e-9))
}

// ------------------------------------------------------------------ zerua

fn dir(th: f64, phi: f64) -> [f64; 3] {
    [th.sin() * phi.cos(), th.sin() * phi.sin(), th.cos()]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    a.map(|v| v / n)
}

// izar bat: norabidea eta fluxua kolorearekin (pantailako unitateetan · sr)
struct Star {
    d: [f64; 3],
    rgb: [f64; 3],
}

// zeru urruna: izarrak (z = cos θ eta φ saretan, bilaketarako) eta esne-bide bat. zerua eszena bat da, ez katalogo
// bat; haren lenteo-efektua, ordea, zehatza da
struct Sky {
    stars: Vec<Star>,
    nz: usize,
    nphi: usize,
    cells: Vec<Vec<u32>>,
    // izar-fluxuaren dentsitatea (fluxua/sr) sare lodi batean, aztarna oso handietarako
    mean: Vec<[f64; 3]>,
    pole: [f64; 3],
    center: [f64; 3],
    side: [f64; 3],
}

const MZ: usize = 45;
const MPHI: usize = 90;

impl Sky {
    fn new(seed: u64, palette: &Palette) -> Sky {
        // esne-bidea zuloaren atzetik pasatzen da, 30° okertuta: haren irudiak lenteo-efektua erakusten du
        let behind = dir(1.745, PI);
        let (th_hat, ph_hat) = ([0.1736, 0.0, -0.9848], [0.0, -1.0, 0.0]);
        let along: [f64; 3] = std::array::from_fn(|k| 0.866 * ph_hat[k] + 0.5 * th_hat[k]);
        let pole = normalize(cross(behind, along));
        let center = normalize(std::array::from_fn(|k| behind[k] + 0.3 * along[k]));
        let side = cross(pole, center);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed_57a2);
        let stars = (0..80_000)
            .map(|_| {
                // % 55 esne-bidean (latitudea laplace, 7°), gainerakoak uniformeki
                let d = if rng.random::<f64>() < 0.55 {
                    let l = (rng.random::<f64>() * 2.0 - 1.0) * PI;
                    let u: f64 = rng.random::<f64>().max(1e-12);
                    let b = -0.12 * u.ln() * if rng.random::<bool>() { 1.0 } else { -1.0 };
                    let (sb, cb) = b.sin_cos();
                    let (sl, cl) = l.sin_cos();
                    std::array::from_fn(|k| cb * (cl * center[k] + sl * side[k]) + sb * pole[k])
                } else {
                    let z = rng.random::<f64>() * 2.0 - 1.0;
                    dir(z.acos(), rng.random::<f64>() * 2.0 * PI)
                };
                // fluxuak: N(>F) ∝ F^(-3/2); tenperaturak gehienak hotzak
                let u: f64 = rng.random::<f64>().max(1e-9);
                let flux = 0.05 * u.powf(-2.0 / 3.0).min(400.0);
                let v: f64 = rng.random();
                let kelvin = 2800.0 + 9000.0 * v * v + 22_000.0 * v.powi(8);
                let c = palette.at(kelvin);
                let y = luminance(c);
                Star {
                    d,
                    rgb: c.map(|x| x / y * flux),
                }
            })
            .collect();
        Sky::with_stars(stars, pole, center, side)
    }

    fn with_stars(stars: Vec<Star>, pole: [f64; 3], center: [f64; 3], side: [f64; 3]) -> Sky {
        let (nz, nphi) = (180, 360);
        let mut cells = vec![Vec::new(); nz * nphi];
        let mut mean = vec![[0.0; 3]; MZ * MPHI];
        for (i, s) in stars.iter().enumerate() {
            let (iz, ip) = Self::cell(s.d, nz, nphi);
            cells[iz * nphi + ip].push(i as u32);
            let (mz, mp) = Self::cell(s.d, MZ, MPHI);
            for (m, v) in mean[mz * MPHI + mp].iter_mut().zip(s.rgb) {
                *m += v;
            }
        }
        // gelaxka lodi bakoitzaren solido-angelua 4π/(MZ·MPHI); 3x3 leundu (φ-n biribila)
        let area = 4.0 * PI / (MZ * MPHI) as f64;
        let smooth = (0..MZ * MPHI)
            .map(|i| {
                let (z, p) = (i / MPHI, i % MPHI);
                let mut acc = [0.0; 3];
                let mut n = 0.0;
                for dz in -1i64..=1 {
                    let zz = z as i64 + dz;
                    if !(0..MZ as i64).contains(&zz) {
                        continue;
                    }
                    for dp in -1i64..=1 {
                        let pp = (p as i64 + dp).rem_euclid(MPHI as i64) as usize;
                        let m = mean[zz as usize * MPHI + pp];
                        for k in 0..3 {
                            acc[k] += m[k];
                        }
                        n += 1.0;
                    }
                }
                acc.map(|v| v / (n * area))
            })
            .collect();
        Sky {
            stars,
            nz,
            nphi,
            cells,
            mean: smooth,
            pole,
            center,
            side,
        }
    }

    fn cell(d: [f64; 3], nz: usize, nphi: usize) -> (usize, usize) {
        let z = ((d[2] + 1.0) / 2.0 * nz as f64)
            .floor()
            .clamp(0.0, nz as f64 - 1.0) as usize;
        let p = d[1].atan2(d[0]).rem_euclid(2.0 * PI);
        let p = (p / (2.0 * PI) * nphi as f64)
            .floor()
            .min(nphi as f64 - 1.0) as usize;
        (z, p)
    }

    fn mean_at(&self, d: [f64; 3]) -> [f64; 3] {
        let (z, p) = Self::cell(d, MZ, MPHI);
        self.mean[z * MPHI + p]
    }

    // izarren argia pixel batean: pixel-azalera · Σ F N(norabidea; aztarna), aztarna pixelaren irudia zeruan
    // (kobariantza `cov`, `c` inguruko planoan, e1 eta e2 ardatzekin). handitutako eskualdeetan aztarna txikia
    // da eta izarra distiratsuagoa, txikitutakoetan alderantziz: fluxua gordetzen da. aztarna oso handia denean
    // (2.5° baino gehiago), batez besteko dentsitatea
    fn stars_at(
        &self,
        c: [f64; 3],
        e: [[f64; 3]; 2],
        cov: [[f64; 2]; 2],
        pix_area: f64,
    ) -> [f64; 3] {
        let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
        let tr = cov[0][0] + cov[1][1];
        let big = (tr / 2.0 + ((tr / 2.0).powi(2) - det).max(0.0).sqrt()).sqrt();
        let (lo, hi) = (1.5f64.to_radians(), 2.5f64.to_radians());
        let mean = self.mean_at(c).map(|v| v * pix_area);
        if big >= hi || det <= 0.0 {
            return mean;
        }
        let inv = [
            [cov[1][1] / det, -cov[0][1] / det],
            [-cov[1][0] / det, cov[0][0] / det],
        ];
        let norm = pix_area / (2.0 * PI * det.sqrt());
        let reach = 4.5 * big;
        let cos_reach = reach.cos();
        let mut acc = [0.0; 3];
        let z0 = ((c[2] - reach + 1.0) / 2.0 * self.nz as f64)
            .floor()
            .max(0.0) as usize;
        let z1 = (((c[2] + reach + 1.0) / 2.0 * self.nz as f64).floor() as usize).min(self.nz - 1);
        let edge_z = (c[2].abs() + reach).min(1.0);
        let sin_min = (1.0 - edge_z * edge_z).sqrt();
        let p0 = c[1].atan2(c[0]).rem_euclid(2.0 * PI);
        let (pa, pb) = if sin_min <= reach * 1.01 || reach / sin_min >= PI {
            (0i64, self.nphi as i64 - 1)
        } else {
            let span = reach / sin_min;
            (
                ((p0 - span) / (2.0 * PI) * self.nphi as f64).floor() as i64,
                ((p0 + span) / (2.0 * PI) * self.nphi as f64).floor() as i64,
            )
        };
        for iz in z0..=z1 {
            for ip in pa..=pb {
                let ip = ip.rem_euclid(self.nphi as i64) as usize;
                for &i in &self.cells[iz * self.nphi + ip] {
                    let s = &self.stars[i as usize];
                    let cd = dot(s.d, c);
                    if cd < cos_reach {
                        continue;
                    }
                    let (u, v) = (dot(s.d, e[0]) / cd, dot(s.d, e[1]) / cd);
                    let q =
                        u * (inv[0][0] * u + inv[0][1] * v) + v * (inv[1][0] * u + inv[1][1] * v);
                    if q < 40.0 {
                        let w = norm * (-0.5 * q).exp();
                        for (a, v) in acc.iter_mut().zip(s.rgb) {
                            *a += w * v;
                        }
                    }
                }
            }
        }
        if big <= lo {
            return acc;
        }
        let f = (big - lo) / (hi - lo);
        std::array::from_fn(|k| acc[k] * (1.0 - f) + mean[k] * f)
    }

    // esne-bidea: disko mehe eta lodi bat, erdiko puntu bat, eta hauts-ildoak; xehetasuna aztarnara mugatuta
    // (`sigma`, radianetan) aliasing-ik ez izateko
    fn diffuse(&self, d: [f64; 3], sigma: f64) -> [f64; 3] {
        let b = dot(d, self.pole).clamp(-1.0, 1.0).asin();
        let l = dot(d, self.side).atan2(dot(d, self.center));
        let octaves = ((1.0 / (6.0 * sigma.max(1e-6))).log2() + 1.0).clamp(1.0, 7.0);
        let n1 = fbm(d.map(|v| v * 3.0), octaves, 101);
        let n2 = fbm(d.map(|v| v * 7.0), (octaves - 1.0).max(1.0), 202);
        let band = 0.6 * (-0.5 * (b / 0.09).powi(2)).exp() + 0.4 * (-b.abs() / 0.25).exp();
        let clumps = (0.7 + 0.45 * n1).max(0.0);
        let bulge = (-(l * l + 3.0 * b * b) / (2.0 * 0.28 * 0.28)).exp();
        let tau =
            2.2 * (-0.5 * (b / 0.035).powi(2)).exp() * (0.55 + 0.5 * n2).max(0.0) * (1.0 + bulge);
        let dust = (-tau).exp();
        let glow: [f64; 3] = std::array::from_fn(|k| {
            [0.30, 0.34, 0.42][k] * band * clumps + [0.55, 0.42, 0.28][k] * bulge * 0.6
        });
        std::array::from_fn(|k| 0.14 * glow[k] * dust + [0.0015, 0.0017, 0.0024][k])
    }
}

// pixel baten aztarna zeruan: azpilaginen (pixeleko posizioa, norabidea) multzotik, karratu txikienen bidez
// J = ∂(u, v)/∂(x, y) (radianak pixeleko, planoa batez besteko norabidearen inguruan). itzultzen du (erdia, ardatzak,
// J Jᵀ/(2π)) edo None aztarna zabalegia bada (kaustiko baten gainean)
#[allow(clippy::type_complexity)]
fn footprint(
    samples: &[([f64; 2], [f64; 3])],
    pix: f64,
) -> Option<([f64; 3], [[f64; 3]; 2], [[f64; 2]; 2])> {
    let mut c = [0.0; 3];
    for (_, d) in samples {
        for k in 0..3 {
            c[k] += d[k];
        }
    }
    let c = normalize(c);
    if samples.iter().any(|(_, d)| dot(*d, c) < 0.9) {
        return None;
    }
    let up = if c[2].abs() < 0.99 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let e1 = normalize(cross(up, c));
    let e2 = cross(c, e1);
    let flat = [[pix * pix / (2.0 * PI), 0.0], [0.0, pix * pix / (2.0 * PI)]];
    if samples.len() < 3 {
        return Some((c, [e1, e2], flat));
    }
    // [1, x, y] diseinua, karratu txikienak bi helburuetarako
    let mut m = [[0.0; 3]; 3];
    let mut bu = [0.0; 3];
    let mut bv = [0.0; 3];
    for ([x, y], d) in samples {
        let row = [1.0, *x, *y];
        let cd = dot(*d, c);
        let (u, v) = (dot(*d, e1) / cd, dot(*d, e2) / cd);
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += row[i] * row[j];
            }
            bu[i] += row[i] * u;
            bv[i] += row[i] * v;
        }
    }
    let det3 = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let dm = det3(&m);
    if dm.abs() < 1e-9 * (samples.len() as f64).powi(3) {
        return Some((c, [e1, e2], flat));
    }
    let solve = |b: &[f64; 3]| -> [f64; 3] {
        std::array::from_fn(|col| {
            let mut mc = m;
            for r in 0..3 {
                mc[r][col] = b[r];
            }
            det3(&mc) / dm
        })
    };
    let (su, sv) = (solve(&bu), solve(&bv));
    let j = [[su[1], su[2]], [sv[1], sv[2]]];
    let jj = |a: usize, b: usize| (j[a][0] * j[b][0] + j[a][1] * j[b][1]) / (2.0 * PI);
    Some((c, [e1, e2], [[jj(0, 0), jj(0, 1)], [jj(1, 0), jj(1, 1)]]))
}

// ------------------------------------------------------------------ irudia: distira eta tonuak

// lausotze gaussiarra, hiru kutxa-pasaldirekin (σ pixeletan), lerroz lerro eta zutabez zutabe
fn blur(img: &[[f64; 3]], w: usize, h: usize, sigma: f64) -> Vec<[f64; 3]> {
    let r = (((4.0 * sigma * sigma + 1.0).sqrt() - 1.0) / 2.0)
        .round()
        .max(1.0) as i64;
    let pass = |src: &[[f64; 3]], n: usize, len: usize| -> Vec<[f64; 3]> {
        let mut out = vec![[0.0; 3]; n * len];
        out.par_chunks_mut(len).enumerate().for_each(|(row, dst)| {
            let line = &src[row * len..(row + 1) * len];
            let at = |i: i64| line[i.clamp(0, len as i64 - 1) as usize];
            let mut acc = [0.0; 3];
            for i in -r..=r {
                let v = at(i);
                for k in 0..3 {
                    acc[k] += v[k];
                }
            }
            let inv = 1.0 / (2 * r + 1) as f64;
            for (x, d) in dst.iter_mut().enumerate() {
                *d = acc.map(|v| v * inv);
                let (add, sub) = (at(x as i64 + r + 1), at(x as i64 - r));
                for k in 0..3 {
                    acc[k] += add[k] - sub[k];
                }
            }
        });
        out
    };
    let transpose = |src: &[[f64; 3]], w: usize, h: usize| -> Vec<[f64; 3]> {
        let mut out = vec![[0.0; 3]; w * h];
        out.par_chunks_mut(h).enumerate().for_each(|(x, col)| {
            for (y, d) in col.iter_mut().enumerate() {
                *d = src[y * w + x];
            }
        });
        out
    };
    let mut a = img.to_vec();
    for _ in 0..3 {
        a = pass(&a, h, w);
    }
    let mut t = transpose(&a, w, h);
    for _ in 0..3 {
        t = pass(&t, w, h);
    }
    transpose(&t, h, w)
}

// ACES tonu-mapaketa (stephen hill-en RRT + ODT doikuntza), sRGB lineala → [0, 1]
fn aces(c: [f64; 3]) -> [f64; 3] {
    const IN: [[f64; 3]; 3] = [
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777],
    ];
    const OUT: [[f64; 3]; 3] = [
        [1.60475, -0.53108, -0.07367],
        [-0.10208, 1.10813, -0.00605],
        [-0.00327, -0.07276, 1.07602],
    ];
    let v: [f64; 3] = std::array::from_fn(|i| IN[i][0] * c[0] + IN[i][1] * c[1] + IN[i][2] * c[2]);
    let v = v.map(|x| {
        (x * (x + 0.024_578_6) - 0.000_090_537) / (x * (0.983_729 * x + 0.432_951) + 0.238_081)
    });
    std::array::from_fn(|i| {
        (OUT[i][0] * v[0] + OUT[i][1] * v[1] + OUT[i][2] * v[2]).clamp(0.0, 1.0)
    })
}

fn srgb(v: f64) -> f64 {
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

// HDR irudia → 8 biteko sRGB: kameraren distira (energia gordetzen duen bi eskalako PSF bat), ACES, sRGB eta
// dithering-a (banden kontra)
fn finish(hdr: &[[f64; 3]], w: usize, h: usize, frame: u64) -> Vec<u8> {
    let near = blur(hdr, w, h, 0.0025 * h as f64);
    let far = blur(hdr, w, h, 0.02 * h as f64);
    let mut px = vec![0u8; w * h * 3];
    px.par_chunks_mut(3).enumerate().for_each(|(i, out)| {
        let c: [f64; 3] =
            std::array::from_fn(|k| 0.86 * hdr[i][k] + 0.10 * near[i][k] + 0.04 * far[i][k]);
        let m = aces(c);
        for k in 0..3 {
            let dither = unit(hash3(i as i64, k as i64, frame as i64, 0xd17)) - 0.5;
            out[k] = (255.0 * srgb(m[k]) + dither).round().clamp(0.0, 255.0) as u8;
        }
    });
    px
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

// diskoaren izpi bat marrazteko (ikus goiko oharra: f32 marrazketarako bakarrik)
#[derive(Clone, Copy)]
struct DiskHit {
    r: f32,
    phi: f32,
    delay: f32,
    // ikusitako oinarrizko tenperatura g·T(r), eta pisua (ertza / azpilagin-kopurua)
    kelvin: f32,
    weight: f32,
}

// pixel baten emaitza trazatu ondoren
struct Pixel {
    sky: [f64; 3],
    hole: bool,
    disk: Vec<DiskHit>,
}

fn write_png(path: &std::path::Path, w: usize, h: usize, px: &[u8]) -> bool {
    let is_png = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("png"));
    if is_png {
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
            .arg(path)
            .stdin(Stdio::piped())
            .spawn()
            .and_then(|mut c| {
                c.stdin.take().unwrap().write_all(px)?;
                c.wait()
            })
            .is_ok_and(|s| s.success())
    } else {
        let mut data = format!("P6\n{} {}\n255\n", w, h).into_bytes();
        data.extend_from_slice(px);
        std::fs::write(path, data).is_ok()
    }
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
        .filter(|&(w, h)| w >= 16 && h >= 16)
    else {
        eprintln!(
            "{}",
            t!(
                "size '{}' should look like 1920x1080",
                "'{}' tamainak 1920x1080 itxura izan behar du",
                args.size
            )
        );
        std::process::exit(2);
    };
    let kerr = Kerr { a: args.spin };
    if args.distance < 6.0 || args.disk_out <= kerr.isco() || args.disk_out >= args.distance {
        eprintln!(
            "{}",
            t!(
                "the camera must be at least 6 M away and the disk must end between the ISCO ({:.3} M) and the camera",
                "kamerak gutxienez 6 M-ra egon behar du eta diskoa ISCO-ren ({:.3} M) eta kameraren artean amaitu behar da",
                kerr.isco()
            )
        );
        std::process::exit(2);
    }
    let scene = Scene {
        kerr,
        r_cam: args.distance,
        disk_in: kerr.isco(),
        disk_out: args.disk_out,
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

    // 2. izpiak: pixel bakoitzeko ss × ss geodesiko (estratifikatuak, dardara finko batekin), behin (kamera
    // finkoa); zerua hemen bertan kalkulatzen da (estatikoa da), diskoa irudi bakoitzean
    let started = std::time::Instant::now();
    let palette = Palette::new();
    let disk = Disk::new(
        kerr.a,
        scene.disk_in,
        scene.disk_out,
        args.temperature,
        args.seed,
    );
    let sky = Sky::new(args.seed, &palette);
    let th_cam = args.inclination.to_radians();
    let scale = (args.fov.to_radians() / 2.0).tan() / (h as f64 / 2.0);
    let pix_area = scale * scale;
    let ss = args.samples.max(1);
    let per = ss * ss;
    let pixels: Vec<Pixel> = (0..w * h)
        .into_par_iter()
        .map(|pix| {
            let mut sky_samples = Vec::new();
            let mut hits = Vec::new();
            let mut holes = 0;
            for k in 0..per {
                let jx = unit(hash3(pix as i64, k as i64, 1, args.seed));
                let jy = unit(hash3(pix as i64, k as i64, 2, args.seed));
                let (sx, sy) = (
                    ((k % ss) as f64 + jx) / ss as f64,
                    ((k / ss) as f64 + jy) / ss as f64,
                );
                let (px, py) = (
                    (pix % w) as f64 + sx - w as f64 / 2.0,
                    h as f64 / 2.0 - (pix / w) as f64 - sy,
                );
                let (e, l, start) = scene.camera_ray(th_cam, px * scale, py * scale);
                match scene.trace(e, l, start).0 {
                    Hit::Sky { th, phi } => sky_samples.push(([sx, sy], dir(th, phi), 1.0 / e)),
                    Hit::Disk { r, phi, g, delay } => hits.push(DiskHit {
                        r: r as f32,
                        phi: phi.rem_euclid(2.0 * PI) as f32,
                        delay: delay as f32,
                        kelvin: (g * disk.temperature(r)) as f32,
                        weight: (disk.edge(r) / per as f64) as f32,
                    }),
                    Hit::Hole => holes += 1,
                }
            }
            // zerua: izarrak pixelaren aztarnarekin, esne-bidea azpilagin bakoitzean (xehetasuna aztarnara
            // mugatuta). g = 1/E: kamerak (ZAMO) zeruko argia apur bat urdinera desplazatuta ikusten du, I ∝ g⁴
            let mut col = [0.0; 3];
            if !sky_samples.is_empty() {
                let frac = sky_samples.len() as f64 / per as f64;
                let pts: Vec<([f64; 2], [f64; 3])> =
                    sky_samples.iter().map(|s| (s.0, s.1)).collect();
                let g4 = (sky_samples.iter().map(|s| s.2).sum::<f64>() / sky_samples.len() as f64)
                    .powi(4);
                let (stars, sigma) = match footprint(&pts, scale) {
                    Some((c, e, cov)) => {
                        let tr = cov[0][0] + cov[1][1];
                        let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
                        // izar puntualen PSF-a: σ = 0.5 pixel
                        let psf = 0.25 * pix_area;
                        let cov = [[cov[0][0] + psf, cov[0][1]], [cov[1][0], cov[1][1] + psf]];
                        let big = (tr / 2.0 + ((tr / 2.0).powi(2) - det).max(0.0).sqrt()).sqrt();
                        (sky.stars_at(c, e, cov, pix_area), big * (2.0 * PI).sqrt())
                    }
                    None => (sky.mean_at(pts[0].1).map(|v| v * pix_area), 0.5),
                };
                let mut diffuse = [0.0; 3];
                for (_, d) in &pts {
                    let v = sky.diffuse(*d, sigma);
                    for k in 0..3 {
                        diffuse[k] += v[k] / pts.len() as f64;
                    }
                }
                col = std::array::from_fn(|k| frac * g4 * (stars[k] + diffuse[k]));
            }
            Pixel {
                sky: col,
                hole: holes == per,
                disk: hits,
            }
        })
        .collect();
    let rays = w * h * per;
    let hole_px = pixels.iter().filter(|p| p.hole).count();
    let disk_rays: usize = pixels.iter().map(|p| p.disk.len()).sum();
    line(
        ts!("rays", "izpiak"),
        t!(
            "{} light rays traced backwards through the Kerr metric in {:.1} s ({} per pixel): {:.1}% of the pixels are the hole's shadow, {:.1}% of the rays end on the disk, the rest in the sky",
            "{} argi-izpi kerr metrikan atzera trazatuak {:.1} s-tan ({} pixel bakoitzeko): pixelen % {:.1} zuloaren itzala da, izpien % {:.1} diskoan amaitzen dira, gainerakoak zeruan",
            rays,
            started.elapsed().as_secs_f64(),
            per,
            100.0 * hole_px as f64 / (w * h) as f64,
            100.0 * disk_rays as f64 / rays as f64
        ),
    );
    line(
        ts!("disk", "diskoa"),
        t!(
            "Page-Thorne thin disk: the gas is hottest ({:.0} K) at r = {:.2} M and cools outwards; each ray shows a blackbody at g·T exactly (Doppler and gravitational shift, CIE 1931 colours), seen as it was when that ray left it",
            "page-thorne disko mehea: gasa beroena ({:.0} K) r = {:.2} M-n dago eta kanpora hozten da; izpi bakoitzak g·T-ko gorputz beltza erakusten du zehazki (doppler eta desplazamendu grabitazionala, CIE 1931 koloreak), izpia irten zen unean zegoen bezala",
            disk.t_peak,
            disk.r_peak
        ),
    );
    line(
        ts!("sky", "zerua"),
        t!(
            "{} stars and a galaxy, lensed: each star's brightness follows its true magnification (the pixel's footprint on the sky). The sky is a scene, shown brighter than it would be next to a real disk; the disk's turbulence and the camera glow are artistic",
            "{} izar eta galaxia bat, lenteo-efektuarekin: izar bakoitzaren distirak bere benetako handitzea jarraitzen du (pixelaren aztarna zeruan). zerua eszena bat da, benetako disko baten ondoan izango litzatekeena baino distiratsuagoa; diskoaren turbulentzia eta kameraren distira artistikoak dira",
            sky.stars.len()
        ),
    );

    // diskoa denbora batean: izpi bakoitza t + atzerapena unean (atzerapena negatiboa: lehenago igorria)
    let disk_light = |time: f64| -> Vec<[f64; 3]> {
        pixels
            .par_iter()
            .map(|p| {
                let mut c = [0.0; 3];
                for hit in &p.disk {
                    let m = disk.turbulence(hit.r as f64, hit.phi as f64, time - hit.delay as f64);
                    let rgb = palette.at(hit.kelvin as f64 * m.powf(0.25));
                    for k in 0..3 {
                        c[k] += hit.weight as f64 * rgb[k];
                    }
                }
                c
            })
            .collect()
    };
    // esposizioa behin (dirdirarik gabe): diskoaren argitasunaren 99.5 pertzentila → 1
    let mut lum: Vec<f64> = disk_light(0.0)
        .iter()
        .map(|c| luminance(*c))
        .filter(|&y| y > 0.0)
        .collect();
    lum.sort_by(f64::total_cmp);
    let p995 = lum
        .get(lum.len() * 995 / 1000)
        .copied()
        .unwrap_or(1.0)
        .max(1e-30);
    let exposure = 1.0 / p995 * 2f64.powf(args.exposure);

    // itzalaren ertza: bikoteak hor agertzen dira (irudikapena)
    let edge: Vec<(usize, usize)> = (1..h - 1)
        .flat_map(|y| (1..w - 1).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            let hole = |x: usize, y: usize| pixels[y * w + x].hole;
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
    let unit_px = h as f64 / 360.0;
    let render = |time: f64, flashes: &[Flash], frame: u64| -> Vec<u8> {
        let light = disk_light(time * args.speed);
        let mut hdr: Vec<[f64; 3]> = pixels
            .iter()
            .zip(&light)
            .map(|(p, d)| std::array::from_fn(|k| p.sky[k] + exposure * d[k]))
            .collect();
        // bikoteak (pantailako unitateetan)
        hdr.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for f in flashes {
                let age = time - f.born;
                let life = 0.8;
                if !(0.0..life).contains(&age) {
                    continue;
                }
                let fade = 1.0 - age / life;
                for (sign, col) in [(1.0, [0.8, 1.0, 1.7]), (-0.6, [1.6, 0.25, 0.08])] {
                    let (fx, fy) = (f.x + sign * f.dx * age, f.y + sign * f.dy * age);
                    let s2 = (0.7 * unit_px).powi(2);
                    if (y as f64 - fy).abs() > 5.0 * unit_px {
                        continue;
                    }
                    let x0 = (fx - 5.0 * unit_px).floor().max(0.0) as usize;
                    let x1 = ((fx + 5.0 * unit_px).ceil() as usize).min(w - 1);
                    for (x, c) in row.iter_mut().enumerate().take(x1 + 1).skip(x0) {
                        let d2 = (x as f64 - fx).powi(2) + (y as f64 - fy).powi(2);
                        let b = 2.0 * fade * (-d2 / (2.0 * s2)).exp();
                        for k in 0..3 {
                            c[k] += b * col[k];
                        }
                    }
                }
            }
        });
        finish(&hdr, w, h, frame)
    };

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
                "slow",
                "-crf",
                "16",
                "-pix_fmt",
                "yuv420p",
            ])
            .args(&args.ffmpeg_args)
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
    let frames = if ffmpeg.is_some() {
        frames
    } else {
        frames / 2 + 1
    };
    let mut still = None;
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
                let speed = 40.0 * unit_px * (1.0 + m.omega / (t_h * 4.0));
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
        // bideorik gabe, erdiko irudia bakarrik errendatzen da (bikoteak hara arte lagintzen dira)
        if ffmpeg.is_none() && f + 1 < frames {
            continue;
        }
        let px = render(time, &flashes, f as u64);
        if let Some(child) = ffmpeg.as_mut() {
            child
                .stdin
                .as_mut()
                .unwrap()
                .write_all(&px)
                .expect("ffmpeg");
        }
        if still.is_none() && (f >= frames / 2 || f + 1 == frames) {
            still = Some(px);
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
                "{}: {} frames in {:.0} s, {} Hawking pairs drawn (counts from QOA's distributions; positions on the shadow's edge are illustrative)",
                "{}: {} irudi {:.0} s-tan, {} hawking bikote marraztuta (kopuruak QOA-ren banaketetatik; itzalaren ertzeko kokapenak irudizkoak dira)",
                args.out.as_ref().unwrap().display(),
                frames,
                started.elapsed().as_secs_f64(),
                total_pairs
            ),
        );
    }
    if let (Some(path), Some(px)) = (args.still.as_ref(), still) {
        if write_png(path, w, h, &px) {
            line(ts!("image", "irudia"), path.display().to_string());
        } else {
            eprintln!(
                "{}",
                t!(
                    "could not write {}",
                    "ezin izan da {} idatzi",
                    path.display()
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
        (e, l, [r0, FRAC_PI_2, 0.0, pr, 0.0, 0.0])
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
                [start[0], start[1], start[2], -start[3], -start[4], 0.0],
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
    fn page_thorne_flux_matches_the_relativistic_integral() {
        // F(r) = -Ω,r / (4π √-g (E - ΩL)²) ∫_{r_in}^r (E - ΩL) L,r dr, √-g = r (ekuatorean), orbita zirkular
        // prograden E, L eta Ω-rekin; integrala simpson-ekin, deribatuak zenbakizko
        for a in [0.0, 0.5, 0.99] {
            let k = Kerr { a };
            let r_in = k.isco();
            let orbit = |r: f64| {
                let x = r.sqrt();
                let den = r.powf(0.75) * (r.powf(1.5) - 3.0 * x + 2.0 * a).sqrt();
                let e = (r.powf(1.5) - 2.0 * x + a) / den;
                let l = (r * r - 2.0 * a * x + a * a) / den;
                (e, l, 1.0 / (r.powf(1.5) + a))
            };
            let d = |f: &dyn Fn(f64) -> f64, r: f64| (f(r + 1e-5) - f(r - 1e-5)) / 2e-5;
            let integral = |r: f64| {
                let n = 4000;
                let hh = (r - r_in) / n as f64;
                (0..=n)
                    .map(|i| {
                        let x = r_in + i as f64 * hh;
                        let (e, l, om) = orbit(x);
                        let w = if i == 0 || i == n {
                            1.0
                        } else if i % 2 == 1 {
                            4.0
                        } else {
                            2.0
                        };
                        w * (e - om * l) * d(&|y| orbit(y).1, x)
                    })
                    .sum::<f64>()
                    * hh
                    / 3.0
            };
            for r in [r_in * 1.3, r_in * 2.0, 10.0, 30.0] {
                let (e, l, om) = orbit(r);
                let numeric =
                    -d(&|y| orbit(y).2, r) / (4.0 * PI * r * (e - om * l).powi(2)) * integral(r);
                let closed = page_thorne(a, r_in, r);
                assert!(
                    (closed - numeric).abs() < 1e-6 * numeric.abs(),
                    "a {} r {}: {} {}",
                    a,
                    r,
                    closed,
                    numeric
                );
            }
            assert!(page_thorne(a, r_in, r_in).abs() < 1e-15);
        }
    }

    #[test]
    fn blackbody_colours_are_physical() {
        let p = Palette::new();
        // 6500 K ia zuria eta Y = 1 (normalizazioa); hotzagoa gorriagoa, beroagoa urdinagoa
        let w = p.at(6500.0);
        assert!((luminance(w) - 1.0).abs() < 1e-3, "{:?}", w);
        let (lo, hi) = (
            w.iter().copied().fold(f64::MAX, f64::min),
            w.iter().copied().fold(0.0, f64::max),
        );
        assert!(hi / lo < 1.2, "{:?}", w);
        let c = p.at(3000.0);
        assert!(c[0] > c[1] && c[1] > c[2], "{:?}", c);
        let b = p.at(30000.0);
        assert!(b[2] > b[1] && b[1] > b[0], "{:?}", b);
        // argitasuna T-rekin hazten da, eta T handietan T-ren proportzionala (rayleigh-jeans)
        let mut prev = 0.0;
        for t in [500.0, 1000.0, 3000.0, 6500.0, 2e4, 1e6, 1e9] {
            let y = luminance(p.at(t));
            assert!(y > prev, "{} {}", t, y);
            prev = y;
        }
        let ratio = luminance(p.at(2e9)) / luminance(p.at(1e9));
        assert!((ratio - 2.0).abs() < 1e-6, "{}", ratio);
    }

    #[test]
    fn light_travel_time_matches_schwarzschild() {
        // izpi erradial bat kanpora r = 10-etik: Δt = Δr + 2 ln((r - 2)/(r0 - 2))
        let k = Kerr { a: 0.0 };
        let r0 = 10.0;
        let pr = r0 * r0 / (r0 * r0 - 2.0 * r0);
        let mut y: Ray = [r0, FRAC_PI_2, 0.0, pr, 0.0, 0.0];
        let mut h = 0.01;
        while y[0] < 40.0 {
            let (next, err) = rk45(&k, 1.0, 0.0, &y, h);
            if err > 1e-10 {
                h *= 0.5;
                continue;
            }
            y = next;
            h = (h * 1.5).min(0.5);
        }
        let want = (y[0] - r0) + 2.0 * ((y[0] - 2.0) / (r0 - 2.0)).ln();
        assert!((y[5] - want).abs() < 1e-7 * want, "{} {}", y[5], want);
    }

    #[test]
    fn lensed_stars_keep_their_flux() {
        // lenterik gabe (J = pix·I): izar baten argia pixel guztietan batuta bere fluxua da
        let star = Star {
            d: dir(1.0, 0.5),
            rgb: [2.0, 1.0, 0.5],
        };
        let sky = Sky::with_stars(
            vec![star],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        let pix: f64 = 2e-4;
        let c0 = dir(1.0, 0.5);
        let up = [0.0, 0.0, 1.0];
        let e1 = normalize(cross(up, c0));
        let e2 = cross(c0, e1);
        let mut total = [0.0; 3];
        for i in -12..=12 {
            for j in -12..=12 {
                // pixel bakoitzaren erdia planoan, eta bere aztarna (J Jᵀ/2π + PSF)
                let (u, v) = (i as f64 * pix + 0.3 * pix, j as f64 * pix - 0.2 * pix);
                let c = normalize(std::array::from_fn(|k| c0[k] + u * e1[k] + v * e2[k]));
                let f1 = normalize(cross(up, c));
                let f2 = cross(c, f1);
                let var = pix * pix / (2.0 * PI) + 0.25 * pix * pix;
                let got = sky.stars_at(c, [f1, f2], [[var, 0.0], [0.0, var]], pix * pix);
                for k in 0..3 {
                    total[k] += got[k];
                }
            }
        }
        for (k, want) in [2.0, 1.0, 0.5].iter().enumerate() {
            assert!((total[k] - want).abs() < 1e-3 * want, "{:?}", total);
        }
    }

    #[test]
    fn the_footprint_recovers_the_sky_mapping() {
        // azpilaginak J lineal batekin mapatuta: footprint-ek J Jᵀ/(2π) itzultzen du
        let c0 = dir(1.2, 2.0);
        let e1 = normalize(cross([0.0, 0.0, 1.0], c0));
        let e2 = cross(c0, e1);
        let j = [[3e-4, 1e-4], [-0.5e-4, 2e-4]];
        let samples: Vec<([f64; 2], [f64; 3])> = (0..9)
            .map(|k| {
                let (x, y) = (((k % 3) as f64 + 0.37) / 3.0, ((k / 3) as f64 + 0.61) / 3.0);
                let (u, v) = (
                    j[0][0] * (x - 0.5) + j[0][1] * (y - 0.5),
                    j[1][0] * (x - 0.5) + j[1][1] * (y - 0.5),
                );
                (
                    [x, y],
                    normalize(std::array::from_fn(|i| c0[i] + u * e1[i] + v * e2[i])),
                )
            })
            .collect();
        let (_, _, cov) = footprint(&samples, 2e-4).unwrap();
        let want = |a: usize, b: usize| (j[a][0] * j[b][0] + j[a][1] * j[b][1]) / (2.0 * PI);
        for (a, b) in [(0, 0), (0, 1), (1, 1)] {
            assert!(
                (cov[a][b] - want(a, b)).abs() < 1e-3 * want(0, 0),
                "{:?}",
                cov
            );
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
