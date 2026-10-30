// fitxategi honek audio-seinale bat argi gisa garraiatzen du zirkuitu optiko kuantiko batetik (`qoa visual`):
//
//   1. seinalea modu optikoetan banatzen da mdct ortonormal batekin (leiho sinusoidala, % 50 gainjarrita,
//      princen-bradley): kanal bakoitzeko lagin bakoitza modu bat da, eta modu-funtzioak ortonormalak dira,
//      beraz argiaren fotoi-kopurua seinalearen energia da (parseval). efekturik gabe seinalea bera itzultzen
//      da, 1e-15 inguruko errorearekin
//   2. modu bakoitza egoera koherente bat da, |α>, α = c·√(2P): P fotoi lagin bakoitzeko eskala osoko sinu
//      batentzat (`photons`). seinalea X koadraturan dago (tokiko osziladorearen fasea erreferentzia da)
//   3. zirkuitua, ordena honetan: (a) ondoko banden arteko habe-zatitzaile sarea (`mix`, errotazio errealak),
//      (b) galera (`loss`), (c) iturriaren estutzea X-n (`squeeze`, dB), (d) kerr efektua (`kerr`, U =
//      exp(-iκ n(n-1))), (e) detekzioa: batez bestekoa zehazki (`mean`), homodinoa X-n (`homodyne`: shot bat
//      modu bakoitzeko, benetako zarata kuantikoarekin) edo fotoi-zenbaketa (`count`)
//   4. mdct alderantzizkoa, gainjarrita gehituta
//
// kalkulu guztiak zehatzak dira: egoera koherenteak errotazio eta galeren ondoren koherenteak dira, egoera
// gaussiarrek forma itxia dute, kerr-en batez besteko eremua α·exp(|α|²(e^{-2iκ} - 1)) da zehazki, eta
// fotoi-zenbaketak poisson banaketa zehatza lagintzen du (kerr-ek ez du fotoi-kopurua aldatzen). konbinazio
// batek kalkulu zehatzik ez badu (kerr homodinoarekin: egoera ez-gaussiar distiratsu bat modu bakoitzeko,
// segundoko 44.000 aldiz), errore bat da, ez hurbilketa bat.

use crate::t;
use num_complex::Complex64 as C;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

// --------------------------------------------------------------------- mdct ortonormala

pub struct Mdct {
    // koefiziente-kopurua irudi bakoitzeko (urratsa); irudiaren luzera 2n da
    n: usize,
    window: Vec<f64>,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
    // e^{-iπ m/(2n)}, m = 0..2n
    pre: Vec<C>,
    // e^{-iπ n0 (k+1/2)/n}, k = 0..n
    post: Vec<C>,
    // e^{iπ n0 k/n}, k = 0..n
    ipre: Vec<C>,
    // e^{iπ (m+n0)/(2n)}, m = 0..2n
    ipost: Vec<C>,
    scale: f64,
}

impl Mdct {
    pub fn new(n: usize) -> Mdct {
        assert!(n >= 2 && n.is_multiple_of(2), "mdct-ren tamaina bikoitia");
        let two = 2 * n;
        let nf = n as f64;
        let n0 = nf / 2.0 + 0.5;
        let cis = |x: f64| {
            let (s, c) = (PI * x).sin_cos();
            C::new(c, s)
        };
        let mut planner = FftPlanner::new();
        Mdct {
            n,
            // sin(π(m + 1/2)/(2n)): w[m]² + w[m+n]² = 1 eta simetrikoa
            window: (0..two)
                .map(|m| (PI * (m as f64 + 0.5) / two as f64).sin())
                .collect(),
            fft: planner.plan_fft_forward(two),
            ifft: planner.plan_fft_inverse(two),
            pre: (0..two).map(|m| cis(-(m as f64) / two as f64)).collect(),
            post: (0..n).map(|k| cis(-n0 * (k as f64 + 0.5) / nf)).collect(),
            ipre: (0..n).map(|k| cis(n0 * k as f64 / nf)).collect(),
            ipost: (0..two)
                .map(|m| cis((m as f64 + n0) / two as f64))
                .collect(),
            scale: (2.0 / nf).sqrt(),
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    // 2n lagin -> n koefiziente: X[k] = √(2/n) Σ w[m] x[m] cos(π/n (m + n0)(k + 1/2))
    pub fn forward(&self, frame: &[f64], out: &mut [f64]) {
        let two = 2 * self.n;
        let mut buf: Vec<C> = (0..two)
            .map(|m| self.pre[m] * (frame[m] * self.window[m]))
            .collect();
        self.fft.process(&mut buf);
        for k in 0..self.n {
            out[k] = self.scale * (self.post[k] * buf[k]).re;
        }
    }

    // n koefiziente -> 2n lagin leihoarekin, gainjarrita gehitzeko
    pub fn inverse(&self, coeffs: &[f64], out: &mut [f64]) {
        let two = 2 * self.n;
        let mut buf = vec![C::new(0.0, 0.0); two];
        for k in 0..self.n {
            buf[k] = self.ipre[k] * coeffs[k];
        }
        self.ifft.process(&mut buf);
        for m in 0..two {
            out[m] = self.scale * self.window[m] * (self.ipost[m] * buf[m]).re;
        }
    }
}

// seinale bat irudietan: hasieran n zero, amaieran nahikoa, lagin bakoitza bi irudiren barruan egon dadin
pub fn analyze(mdct: &Mdct, signal: &[f64]) -> Vec<Vec<f64>> {
    let n = mdct.n;
    let frames = signal.len().div_ceil(n) + 1;
    let mut padded = vec![0.0; (frames + 1) * n];
    padded[n..n + signal.len()].copy_from_slice(signal);
    (0..frames)
        .into_par_iter()
        .map(|f| {
            let mut out = vec![0.0; n];
            mdct.forward(&padded[f * n..f * n + 2 * n], &mut out);
            out
        })
        .collect()
}

// `analyze`-ren alderantzizkoa: `len` laginetako seinalea
pub fn synthesize(mdct: &Mdct, frames: &[Vec<f64>], len: usize) -> Vec<f64> {
    let n = mdct.n;
    let parts: Vec<Vec<f64>> = frames
        .par_iter()
        .map(|c| {
            let mut out = vec![0.0; 2 * n];
            mdct.inverse(c, &mut out);
            out
        })
        .collect();
    let mut acc = vec![0.0; (frames.len() + 1) * n];
    for (f, part) in parts.iter().enumerate() {
        for (a, p) in acc[f * n..f * n + 2 * n].iter_mut().zip(part) {
            *a += p;
        }
    }
    acc[n..n + len].to_vec()
}

// --------------------------------------------------------------------- zirkuitua

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detect {
    // ⟨x⟩ zehatza: zaratarik gabe, shot infinituren batez bestekoa
    Mean,
    // X koadraturaren neurketa bat modu bakoitzeko
    Homodyne,
    // fotoi-kopuruaren neurketa bat modu bakoitzeko (ikurra batez besteko eremutik)
    Count,
}

#[derive(Clone, Debug)]
pub struct Light {
    // eskala osoko sinu baten fotoi-kopurua lagin bakoitzeko
    pub photons: f64,
    // galdutako zatia, [0, 1)
    pub loss: f64,
    // iturriaren estutzea X koadraturan, dB-tan (zarata-murrizketa)
    pub squeeze_db: f64,
    // kerr fasea modu distiratsuenean (radianak): 2κ|α|²_max
    pub kerr: f64,
    // ondoko banden arteko habe-zatitzaileen angelua (radianak)
    pub mix: f64,
    pub detect: Detect,
}

impl Default for Light {
    fn default() -> Self {
        Light {
            photons: 1e5,
            loss: 0.0,
            squeeze_db: 0.0,
            kerr: 0.0,
            mix: 0.0,
            detect: Detect::Homodyne,
        }
    }
}

// argiaren laburpena txostenerako
#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub modes: usize,
    // fotoi guztiak, galeren ondoren
    pub photons: f64,
    // modu distiratsuenaren fotoi-kopurua
    pub brightest: f64,
    // κ, fotoi-pare bakoitzeko kerr fasea
    pub kappa: f64,
}

impl Light {
    // balioak eta konbinazioak: zehazki kalkula ezin dena errore bat da
    pub fn check(&self) -> Result<(), String> {
        if !(self.photons > 0.0 && self.photons.is_finite()) {
            return Err(t!(
                "--photons must be a positive number, not {}",
                "--photons zenbaki positibo bat izan behar da, ez {}",
                self.photons
            ));
        }
        if !(0.0..1.0).contains(&self.loss) {
            return Err(t!(
                "--loss is the fraction of light lost, from 0 up to (not including) 1, not {}",
                "--loss galdutako argiaren zatia da, 0tik 1era (1 gabe), ez {}",
                self.loss
            ));
        }
        if !(0.0..=30.0).contains(&self.squeeze_db) {
            return Err(t!(
                "--squeeze is in dB of noise reduction, from 0 to 30, not {}",
                "--squeeze zarata-murrizketaren dB-ak dira, 0tik 30era, ez {}",
                self.squeeze_db
            ));
        }
        if !self.kerr.is_finite() || !self.mix.is_finite() {
            return Err(ts_err());
        }
        match self.detect {
            Detect::Mean if self.squeeze_db > 0.0 => Err(t!(
                "--squeeze changes only the quantum noise, and --detect mean has none: use --detect homodyne to hear it",
                "--squeeze-k zarata kuantikoa bakarrik aldatzen du, eta --detect mean-ek ez du zaratarik: erabili --detect homodyne entzuteko"
            )),
            Detect::Homodyne if self.kerr != 0.0 => Err(t!(
                "--kerr with --detect homodyne has no exact computation at audio rates: every mode would be a bright non-Gaussian state to sample (about 44,000 per second). Use --detect mean (exact mean field) or --detect count (exact photon statistics)",
                "--kerr eta --detect homodyne batera ezin dira zehazki kalkulatu audio-abiaduran: modu bakoitza lagindu beharreko egoera ez-gaussiar distiratsu bat litzateke (44.000 inguru segundoko). Erabili --detect mean (batez besteko eremu zehatza) edo --detect count (fotoi-estatistika zehatza)"
            )),
            Detect::Count if self.squeeze_db > 0.0 => Err(t!(
                "photon counting of squeezed light (--squeeze with --detect count) is not implemented yet",
                "argi estutuaren fotoi-zenbaketa (--squeeze eta --detect count) ez dago oraindik"
            )),
            _ => Ok(()),
        }
    }

    // kanal bakoitzaren irudiak (mdct koefizienteak) tokian prozesatu. `seed`-ek zarata erabakitzen du: hazi
    // berak emaitza bera ematen du hari-kopurua edozein dela ere (irudi bakoitzak bere rng-a)
    pub fn process(&self, channels: &mut [Vec<Vec<f64>>], seed: u64) -> Summary {
        let s = (2.0 * self.photons).sqrt();
        let t = 1.0 - self.loss;
        if self.mix != 0.0 {
            let (sn, cs) = self.mix.sin_cos();
            channels
                .par_iter_mut()
                .flat_map(|ch| ch.par_iter_mut())
                .for_each(|frame| mesh(frame, cs, sn));
        }
        // fotoiak galeren ondoren: |α|² = t·(s·c)²
        let (total, brightest) = channels
            .par_iter()
            .flat_map(|ch| ch.par_iter())
            .map(|frame| {
                let mut sum = crate::dd::Dd::ZERO;
                let mut max: f64 = 0.0;
                for &c in frame {
                    let a2 = t * (s * c) * (s * c);
                    sum.add(a2);
                    max = max.max(a2);
                }
                (sum, max)
            })
            .reduce(
                || (crate::dd::Dd::ZERO, 0.0),
                |a, b| (a.0.plus(b.0), a.1.max(b.1)),
            );
        let kappa = if self.kerr != 0.0 && brightest > 0.0 {
            self.kerr / (2.0 * brightest)
        } else {
            0.0
        };
        // homodinoaren bariantza X-n (hutsa = 1/2): estutzea galeraren aurretik
        let squeezed = 10f64.powf(-self.squeeze_db / 10.0);
        let variance = t * squeezed / 2.0 + (1.0 - t) / 2.0;
        let gain = 1.0 / (t.sqrt() * s);
        let modes = channels.iter().map(|ch| ch.len()).sum::<usize>()
            * channels
                .first()
                .and_then(|c| c.first())
                .map_or(0, |f| f.len());
        for (ci, ch) in channels.iter_mut().enumerate() {
            ch.par_iter_mut().enumerate().for_each(|(fi, frame)| {
                let mut rng = frame_rng(seed, ci as u64, fi as u64);
                let mut spare: Option<f64> = None;
                for c in frame.iter_mut() {
                    let alpha = t.sqrt() * s * *c;
                    let field = if kappa != 0.0 {
                        kerr_mean(alpha, kappa)
                    } else {
                        alpha
                    };
                    *c = gain
                        * match self.detect {
                            Detect::Mean => field,
                            Detect::Homodyne => {
                                let g = match spare.take() {
                                    Some(g) => g,
                                    None => {
                                        let (a, b) = normal_pair(&mut rng);
                                        spare = Some(b);
                                        a
                                    }
                                };
                                // x = √2 α + √V g; α-ren unitateetan x/√2
                                alpha + (variance / 2.0).sqrt() * g
                            }
                            Detect::Count => {
                                let n = poisson(&mut rng, alpha * alpha) as f64;
                                n.sqrt() * if field < 0.0 { -1.0 } else { 1.0 }
                            }
                        };
                }
            });
        }
        Summary {
            modes,
            photons: total.value(),
            brightest,
            kappa,
        }
    }
}

fn ts_err() -> String {
    t!(
        "--kerr and --mix must be finite numbers",
        "--kerr eta --mix zenbaki finituak izan behar dira"
    )
}

// ondoko moduen arteko habe-zatitzaileak, adreilu-horma gisa: lehenik (0,1), (2,3)..., gero (1,2), (3,4)...
// errotazio errealak dira: egoera koherenteak koherente geratzen dira (α → Oα), zehazki
fn mesh(frame: &mut [f64], cs: f64, sn: f64) {
    for start in [0, 1] {
        let mut k = start;
        while k + 1 < frame.len() {
            let (a, b) = (frame[k], frame[k + 1]);
            frame[k] = cs * a - sn * b;
            frame[k + 1] = sn * a + cs * b;
            k += 2;
        }
    }
}

// kerr-en ondoren egoera koherente baten batez besteko eremuaren zati erreala: U = exp(-iκ n(n-1)) bada,
// U†aU = e^{-2iκn} a eta ⟨a⟩ = α exp(|α|²(e^{-2iκ} - 1)). e^{-2iκ} - 1 = -2 sin²κ - i sin 2κ (ezeztapenik gabe)
pub fn kerr_mean(alpha: f64, kappa: f64) -> f64 {
    let n = alpha * alpha;
    let sk = kappa.sin();
    let decay = (-2.0 * n * sk * sk).exp();
    let phase = n * (2.0 * kappa).sin();
    alpha * decay * phase.cos()
}

// irudi baten rng-a, haziaren, kanalaren eta irudiaren arabera (splitmix64)
fn frame_rng(seed: u64, channel: u64, frame: u64) -> StdRng {
    let mut z = seed
        ^ channel.wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ frame.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    StdRng::seed_from_u64(z ^ (z >> 31))
}

// bi normal estandar independente (box-muller), (0, 1] tarteko uniformeekin
pub fn normal_pair(rng: &mut StdRng) -> (f64, f64) {
    let u1 = 1.0 - rng.random::<f64>();
    let u2 = rng.random::<f64>();
    let r = (-2.0 * u1.ln()).sqrt();
    let (s, c) = (2.0 * PI * u2).sin_cos();
    (r * c, r * s)
}

// ln(1 + d) - d, ezeztapenik gabe d txikia denean
fn log1pmx(d: f64) -> f64 {
    if d.abs() < 0.1 {
        // -d²/2 + d³/3 - d⁴/4 + ...: 0.1^24/24 < 1e-25
        let mut term = d;
        let mut sum = 0.0;
        for j in 2..26 {
            term *= -d;
            sum += term / j as f64;
        }
        sum
    } else {
        d.ln_1p() - d
    }
}

// stirling-en zuzenketa: ln k! - (k ln k - k + ½ ln(2πk)), k >= 20 denean 1e-17 baino zehatzago
fn stirling_tail(k: f64) -> f64 {
    let k2 = k * k;
    (1.0 / 12.0 - (1.0 / 360.0 - (1.0 / 1260.0 - 1.0 / (1680.0 * k2)) / k2) / k2) / k
}

// ln P(k) poisson(λ) banaketan, λ eta k handiak direnean ere zehatza (ez dira zenbaki handiak kentzen)
pub fn poisson_log_pmf(k: u64, lambda: f64) -> f64 {
    if k == 0 {
        return -lambda;
    }
    let kf = k as f64;
    if k < 20 {
        let ln_fact: f64 = (2..=k).map(|j| (j as f64).ln()).sum();
        return -lambda + kf * lambda.ln() - ln_fact;
    }
    kf * log1pmx((lambda - kf) / kf) - 0.5 * (2.0 * PI * kf).ln() - stirling_tail(kf)
}

// poisson(λ) lagin zehatz bat: λ < 10 alderantzizko metodoz, bestela hörmann-en PTRS (1993) baztertze
// zehatza, onarpen-proba `poisson_log_pmf` zehatzarekin
pub fn poisson(rng: &mut StdRng, lambda: f64) -> u64 {
    if lambda <= 0.0 {
        return 0;
    }
    if lambda < 10.0 {
        let mut p = (-lambda).exp();
        let mut cum = p;
        let u = rng.random::<f64>();
        let mut k = 0u64;
        while u > cum && k < 1000 {
            k += 1;
            p *= lambda / k as f64;
            cum += p;
        }
        return k;
    }
    let slam = lambda.sqrt();
    let b = 0.931 + 2.53 * slam;
    let a = -0.059 + 0.02483 * b;
    let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = rng.random::<f64>() - 0.5;
        let v = rng.random::<f64>();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + lambda + 0.43).floor();
        if us >= 0.07 && v <= vr {
            return k as u64;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        if (v * inv_alpha / (a / (us * us) + b)).ln() <= poisson_log_pmf(k as u64, lambda) {
            return k as u64;
        }
    }
}
