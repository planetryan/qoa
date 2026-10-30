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
// fotoi-zenbaketak poisson banaketa zehatza lagintzen du (kerr-ek ez du fotoi-kopurua aldatzen), eta kerr-en
// ondoko homodinoak egoera ez-gaussiarraren X banaketa zehatza lagintzen du (`KerrSampler`). konbinazio batek
// kalkulu zehatzik ez badu, errore bat da, ez hurbilketa bat.

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
        Ok(())
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
        // galeraren ondoko argi estutua gaussiar nahasia da: X-n `variance` duen egoera estutu purua, P-n zarata
        // klasiko gaussiar batekin desplazatuta (vp - 1/(4 vx)), zehazki. u = Im α-ren zarataren desbideratzea
        let squeezing = self.squeeze_db > 0.0;
        let r_pure = -0.5 * (2.0 * variance).ln();
        let vp = t / squeezed / 2.0 + (1.0 - t) / 2.0;
        let s_u = ((vp - 1.0 / (4.0 * variance)).max(0.0) / 2.0).sqrt();
        let gain = 1.0 / (t.sqrt() * s);
        // kerr-en ondoko homodinoa: taulak behin, modu guztientzat
        let sampler =
            (self.detect == Detect::Homodyne && kappa != 0.0).then(|| KerrSampler::new(kappa));
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
                    // batez besteko eremua: kerr-ik gabe α; kerr-ekin forma itxian (estutuarekin, galeraren
                    // zarataren gaineko batez bestekoa barne)
                    let field = if kappa == 0.0 {
                        alpha
                    } else if squeezing {
                        kerr_mean_mixed(alpha, r_pure, s_u, kappa).re
                    } else {
                        kerr_mean(alpha, kappa)
                    };
                    // nahastearen osagai puru bat: α + iu, u ~ N(0, s_u²)
                    let pure = |rng: &mut StdRng| {
                        let u = if s_u > 0.0 {
                            s_u * normal_pair(rng).0
                        } else {
                            0.0
                        };
                        C::new(alpha, u)
                    };
                    *c = gain
                        * match self.detect {
                            Detect::Mean => field,
                            Detect::Homodyne if sampler.is_some() && squeezing => {
                                let state = Pure::lab(pure(&mut rng), r_pure);
                                sampler.as_ref().unwrap().sample_pure(&state, &mut rng)
                                    / 2f64.sqrt()
                            }
                            Detect::Homodyne if sampler.is_some() => {
                                let x = sampler.as_ref().unwrap().sample(alpha, &mut rng);
                                x / 2f64.sqrt()
                            }
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
                                let n = if squeezing {
                                    count_pure(pure(&mut rng), r_pure, &mut rng) as f64
                                } else {
                                    poisson(&mut rng, alpha * alpha) as f64
                                };
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

// --------------------------------------------------------------------- kerr eta homodinoa

// kerr-en ondoko egoera, U = exp(-iκ n(n-1)) |α>, zehazki. n(n-1) = (n-c0)² + (2c0-1)(n-c0) + c0(c0-1)
// denez (c0 ≈ |α|² osoa), egoera biraketa bat (θ0 = κ(2c0-1)) eta chirp koadratiko bat da: e^{-iκ(n-c0)²}.
// chirp-a fotoi-kopuruaren leiho batean (M luzera, ertzetan erf-leun batez itzalita, leihoaren erdian 1
// zehazki) bere fourier-koefizienteekin idazten da, eta e^{i n θ_j} faktore bakoitza egoera koherente baten
// biraketa da: ψ = Σ_j C_j |a e^{iφ_j}>, φ_j = 2πj/M - θ0. koefizienteak M eta κ-ren menpe bakarrik daude,
// beraz M tamaina bakoitzeko fft bat egiten da behin, modu guztientzat.
//
// zehaztasuna: leihotik kanpo poisson amplitudeak < 1e-16 (12σ + 30); ezabatutako koefizienteen Σ|F_j| <= 1e-12,
// beraz egoeraren errorea <= 1e-12 eta laginen banaketarena (bariazio osoa) <= 3e-12. distiratsuenetan,
// faseak (√2·a·x·Δsin) 1e6 radianetaraino iristen dira: sarrerako 64 biteko a eta x-ek berek ematen dute
// haien doitasuna (1e-10 rad inguru), eta algoritmoak ez du hori baino gehiago galtzen.
//
// laginketa: P(x) = |Σ C_j φ_j(x)|², φ_j(x) = π^{-1/4} e^{-(x-μ_j)²/2} e^{iΦ_j(x)}, μ_j = √2 a cos φ_j.
// cauchy-schwarz-en bidez P(x) <= S·Σ |C_j| e^{-(x-μ_j)²}/√π (S = Σ|C_j|), gaussiarren nahaste bat:
// j aukeratu |C_j|/S probabilitatez, x ~ N(μ_j, 1/2), eta onartu P(x)/(S·Σ|C_j| e^{-(x-μ_j)²}) probabilitatez.
// leihoaren tamaina M ez da 2-ren berredura bat: oktaba bakoitzeko 16 tamaina, alferrik gutxi galtzeko.

const TWO_PI_HI: f64 = std::f64::consts::TAU;
const TWO_PI_LO: f64 = 2.449_293_598_294_706_4e-16;
// ezabatutako koefizienteen |F| batura handiena (egoeraren errore-muga); fft-aren biribiltze-zarata (1e-18
// inguru koefiziente bakoitzeko) ezabatzeko adina
const DROPPED: f64 = 1e-12;

// κ·q mod 2π, double-double-n (q osoa eta zehatza)
fn phase_mod(kappa: f64, q: f64) -> f64 {
    let (p, e) = crate::dd::two_prod::<false>(kappa, q);
    let k = (p / TWO_PI_HI).round();
    let (a, ae) = crate::dd::two_prod::<false>(k, TWO_PI_HI);
    ((p - a) - ae) - k * TWO_PI_LO + e
}

// itzala: laukizuzen bat gaussiar diskretu batekin konboluzionatuta (aurrizki-baturekin): leuna (haren
// espektroa gaussiar gisa jaisten da) eta erdian 1 zehazki. `sw` gaussiarraren zabalera da, eta itzala 0 da
// mutur bakoitzetik 7·sw-ra (e^{-49})
fn taper(m: usize, sw: f64) -> Vec<f64> {
    let half = (7.0 * sw).ceil() as i64;
    let mut prefix = Vec::with_capacity(2 * half as usize + 2);
    let mut acc = crate::dd::Dd::ZERO;
    prefix.push(0.0);
    for k in -half..=half {
        let t = k as f64 / sw;
        acc.add((-t * t).exp());
        prefix.push(acc.value());
    }
    let total = acc.value();
    // cdf(d) = Σ_{k <= d} G(k) / Σ G
    let cdf = |d: i64| -> f64 {
        if d < -half {
            0.0
        } else if d >= half {
            1.0
        } else {
            prefix[(d + half + 1) as usize] / total
        }
    };
    let (m1, m2) = (half, m as i64 - 1 - half);
    (0..m as i64).map(|i| cdf(i - m1) - cdf(i - m2)).collect()
}

// M tamaina bakoitzeko koefiziente esanguratsuak (j zeinuduna, F_j)
type Table = Arc<Vec<(i64, C)>>;

pub struct KerrSampler {
    kappa: f64,
    // behar direnean kalkulatuta
    tables: std::sync::Mutex<std::collections::HashMap<usize, Table>>,
}

impl KerrSampler {
    pub fn new(kappa: f64) -> KerrSampler {
        KerrSampler {
            kappa,
            tables: Default::default(),
        }
    }

    // leihoaren tamaina: euskarria (n̄ ± 12σ ± 31) itzalaren zati lauaren barruan ([0.21M, 0.79M] inguru),
    // oktaba bakoitzeko 16 tamainatara biribilduta, taulak berrerabiltzeko
    fn size(n_bar: f64) -> usize {
        Self::size_w(12.0 * n_bar.sqrt() + 31.0)
    }

    // `w`: euskarriaren erdi-zabalera fotoi-kopuruan
    fn size_w(w: f64) -> usize {
        let need = (2.0 * w / 0.56).ceil().max(64.0) as usize;
        let step = (need.next_power_of_two() / 32).max(2);
        need.div_ceil(step) * step
    }

    fn table_for(&self, m: usize) -> Table {
        if let Some(t) = self.tables.lock().unwrap().get(&m) {
            return t.clone();
        }
        // blokeoa askatuta kalkulatu
        let t = Arc::new(Self::table(self.kappa, m));
        self.tables.lock().unwrap().entry(m).or_insert(t).clone()
    }

    fn table(kappa: f64, m: usize) -> Vec<(i64, C)> {
        let h = (m / 2) as i64;
        let tp = taper(m, m as f64 / 64.0);
        let mut buf: Vec<C> = (0..m)
            .map(|i| {
                let taper = tp[i];
                let d = i as i64 - h;
                let ph = phase_mod(kappa, (d * d) as f64);
                let (sn, cs) = ph.sin_cos();
                C::new(cs, -sn) * taper
            })
            .collect();
        FftPlanner::new().plan_fft_forward(m).process(&mut buf);
        let inv = 1.0 / m as f64;
        let mut all: Vec<(i64, C)> = buf
            .iter()
            .enumerate()
            .map(|(j, f)| {
                let js = if j < m / 2 {
                    j as i64
                } else {
                    j as i64 - m as i64
                };
                (js, f * inv)
            })
            .collect();
        // txikienetatik hasita ezabatu, beren |F| batura DROPPED-era iritsi arte
        all.sort_by(|a, b| a.1.norm().total_cmp(&b.1.norm()));
        let mut dropped = 0.0;
        let mut cut = 0;
        for (i, (_, f)) in all.iter().enumerate() {
            if dropped + f.norm() > DROPPED {
                cut = i;
                break;
            }
            dropped += f.norm();
            cut = i + 1;
        }
        all.split_off(cut)
    }

    // modu baten egoera fock oinarri desplazatuan: ψ = D(β) Σ_k d_k |k>, β = a e^{iφ_r} (egoera klasikoaren
    // puntua, osagai erreferentziala). osagai bakoitza D(-β)|a e^{iφ_j}> = e^{i a² sin δ_j}|γ_j> da, γ_j = a(e^{iδ_j} - 1),
    // eta <k|γ> = e^{-|γ|²/2} γ^k/√k!. kerr-en egoera trinkoa da fotoi-kopuru txikitan (kerr fasearen araberakoa),
    // beraz k-ren batura koefizienteak arbuiagarriak direnean gelditzen da. itzultzen du (x0 = √2 Re β, d_k)
    fn displaced(&self, a: f64) -> (f64, Vec<C>) {
        let n_bar = a * a;
        let m = Self::size(n_bar) as i64;
        let table = self.table_for(m as usize);
        let c0 = n_bar.round() as i64;
        let n_lo = c0 - m / 2;
        // θ0 = κ(2c0 - 1) mod 2π; erreferentzia: chirp-aren erdia (j = 0), hau da egoeraren angelu klasikoa φ_r = -θ0
        // (koefiziente handiena ez: chirp-aren espektroaren ertzetan fresnel-uhinak handiagoak izan daitezke)
        let theta0 = phase_mod(self.kappa, (2 * c0 - 1) as f64);
        let j_r = 0i64;
        let phi_r =
            (TWO_PI_HI * j_r as f64 / m as f64 - theta0) + TWO_PI_LO * j_r as f64 / m as f64;
        let (a2, a2e) = crate::dd::two_prod::<false>(a, a);
        // (koefizientea, γ, lehen k esanguratsua)
        let comps: Vec<(C, C, usize)> = table
            .iter()
            .map(|&(j, f)| {
                // C_j = F_j e^{-i 2π j n_lo / M}, osoko aritmetikaz murriztuta
                let r = ((j as i128 * n_lo as i128).rem_euclid(m as i128)) as f64;
                let (sn, cs) = (TWO_PI_HI * r / m as f64).sin_cos();
                let d = TWO_PI_HI * (j - j_r) as f64 / m as f64;
                let (sh, ch) = (d / 2.0).sin_cos();
                // γ = a(e^{iδ} - 1) = 2ia sin(δ/2) e^{iδ/2}
                let g = C::new(-2.0 * a * sh * sh, 2.0 * a * sh * ch);
                // e^{i a² sin δ}, a² double-double-n
                let ph = phase_mod2(a2, a2e, d.sin());
                let cj = f * C::new(cs, -sn) * C::from_polar(1.0, ph);
                // |γ| handiekin e^{-|γ|²/2} ia azpinormala da (bit gutxi): log-ean hasi k = (|γ| - 20)²-n, non
                // |<k|γ>| ≈ e^{-400} (normala, eta lehenagoko guztiak arbuiagarriak)
                let start = if g.norm_sqr() < 1200.0 {
                    0
                } else {
                    ((g.norm() - 20.0).powi(2) as usize).max(1)
                };
                (cj, g, start)
            })
            .collect();
        // osagai bakoitzaren egoera: t = <k|γ>, eta noiz amaitzen den (poisson-en goiko isatsa, |γ|² + 20|γ| + 60)
        let n = comps.len();
        let c: Vec<C> = comps.iter().map(|x| x.0).collect();
        let g: Vec<C> = comps.iter().map(|x| x.1).collect();
        let start: Vec<usize> = comps.iter().map(|x| x.2).collect();
        let end: Vec<usize> = g
            .iter()
            .map(|g| (g.norm_sqr() + 20.0 * g.norm() + 60.0) as usize)
            .collect();
        let mut t: Vec<C> = (0..n)
            .map(|i| {
                if start[i] == 0 {
                    C::new((-g[i].norm_sqr() / 2.0).exp(), 0.0)
                } else {
                    C::new(0.0, 0.0)
                }
            })
            .collect();
        // aktibo dauden osagaiak (hasita eta amaitu gabe)
        let mut active: Vec<usize> = (0..n).filter(|&i| start[i] == 0).collect();
        let mut pending: Vec<usize> = (0..n).filter(|&i| start[i] > 0).collect();
        pending.sort_by_key(|&i| std::cmp::Reverse(start[i]));
        let mut d = Vec::new();
        let mut quiet = 0;
        let mut rot = C::new(1.0, 0.0);
        let step = C::from_polar(1.0, phi_r);
        for k in 0..100_000usize {
            if k > 0 {
                let inv = 1.0 / (k as f64).sqrt();
                for &i in &active {
                    t[i] = t[i] * g[i] * inv;
                }
                // e^{ikφ_r}: 64 urratsero zehazki berriz kalkulatuta, biribiltzea ez metatzeko
                rot = if k % 64 == 0 {
                    C::from_polar(1.0, (k as f64 * phi_r) % std::f64::consts::TAU)
                } else {
                    rot * step
                };
            }
            // k-n hasten direnak: log-ean, |<k|γ>|² = poisson(k; |γ|²)
            while let Some(&i) = pending.last() {
                if start[i] > k {
                    break;
                }
                pending.pop();
                let mag = (poisson_log_pmf(k as u64, g[i].norm_sqr()) / 2.0).exp();
                t[i] = C::from_polar(mag, (k as f64 * g[i].arg()) % std::f64::consts::TAU);
                active.push(i);
            }
            active.retain(|&i| k <= end[i]);
            let mut sum = C::new(0.0, 0.0);
            let mut scale = 0.0;
            for &i in &active {
                let term = c[i] * t[i];
                sum += term;
                scale += term.re.abs() + term.im.abs();
            }
            let dk = sum * rot;
            // zero numerikoa: 1e-17 azpitik, edo baturaren biribiltze-zarataren azpitik (osagai urrunek elkar
            // ezeztatzen dute, eta haien hondarra ez da egoera)
            let floor = 1e-17f64.max(1e-11 * scale);
            quiet = if dk.norm() < floor { quiet + 1 } else { 0 };
            d.push(dk);
            // isatsa: 40 koefiziente jarraian zero numerikoak (probabilitatearen irizpide bat ez litzateke nahikoa:
            // 1e-12 probabilitate = 1e-6 anplitude)
            if k > 40 && quiet >= 40 {
                break;
            }
        }
        (2f64.sqrt() * a * phi_r.cos(), d)
    }

    // X-ren banaketa metatu zehatza eta dentsitatea x puntuetan (α >= 0), garapen bakarrarekin
    pub fn cdf_and_density(&self, alpha: f64, xs: &[f64]) -> Vec<(f64, f64)> {
        let (x0, d) = self.displaced(alpha.abs());
        xs.iter().map(|x| cdf_density(&d, x - x0)).collect()
    }

    // X koadraturaren dentsitatea (hutsa: bariantza 1/2), α >= 0 edo < 0
    pub fn density(&self, alpha: f64, x: f64) -> f64 {
        let a = alpha.abs();
        let x = if alpha < 0.0 { -x } else { x };
        if a == 0.0 {
            return (-x * x).exp() / PI.sqrt();
        }
        let (x0, d) = self.displaced(a);
        cdf_density(&d, x - x0).1
    }

    // X-ren lagin bat kerr-en ondoko egoeran: banaketa metatu zehatzaren alderantzizkoa
    pub fn sample(&self, alpha: f64, rng: &mut StdRng) -> f64 {
        let a = alpha.abs();
        let sign = if alpha < 0.0 { -1.0 } else { 1.0 };
        if a == 0.0 {
            return normal_pair(rng).0 / 2f64.sqrt();
        }
        let (x0, d) = self.displaced(a);
        sign * invert(&d, x0, rng)
    }

    // X-ren lagin bat kerr-en ondoren egoera estutu puru batean: R(ψ) D(a) S(r e^{iθ}) |0>
    pub fn sample_pure(&self, p: &Pure, rng: &mut StdRng) -> f64 {
        if p.r == 0.0 && p.psi == 0.0 {
            return self.sample(p.a, rng);
        }
        let (beta, d) = self.displaced_pure(p);
        invert(&d, 2f64.sqrt() * beta.re, rng)
    }

    // X-ren dentsitatea egoera estutu puru baterako (probetarako)
    pub fn density_pure(&self, p: &Pure, x: f64) -> f64 {
        let (beta, d) = self.displaced_pure(p);
        cdf_density(&d, x - 2f64.sqrt() * beta.re).1
    }

    // `displaced`-en bertsio orokorra egoera estutu baterako: osagaiak R(φ_j) D(a) S(r e^{iθ}) |0> dira, eta
    // D(-β)-ren ondoren <k|γ_j, ζ_j> hiru gaiko errepikapenez (eskala bereizita, azpigainezkarik gabe):
    // c_0 = e^{-|γ|²/2 - γ̄² e^{iθ'} tanh r / 2} / √cosh r, c_{k+1} = (Γ c_k - e^{iθ'} tanh r √k c_{k-1}) / √(k+1),
    // Γ = (γ cosh r + γ̄ e^{iθ'} sinh r)/cosh r. itzultzen du (β, d_k)
    fn displaced_pure(&self, p: &Pure) -> (C, Vec<C>) {
        let (a, r) = (p.a, p.r);
        let (n_bar, std, tail) = number_stats(a, r, p.theta);
        let m = Self::size_w(12.0 * std + 31.0 + tail) as i64;
        let table = self.table_for(m as usize);
        let c0 = n_bar.round() as i64;
        let n_lo = c0 - m / 2;
        let theta0 = phase_mod(self.kappa, (2 * c0 - 1) as f64);
        let phi_r = -theta0;
        let (a2, a2e) = crate::dd::two_prod::<false>(a, a);
        let (ch, tr) = (r.cosh(), r.tanh());
        let half_ln_ch = 0.5 * ch.ln();
        // osagai bakoitza: koefizientea, Γ, B = e^{iθ'} tanh r, eskala, eta errepikapenaren bi balioak
        struct Comp {
            c: C,
            big: C,
            b: C,
            scale: f64,
            factor: f64,
            prev: C,
            cur: C,
        }
        let mut comps: Vec<Comp> = table
            .iter()
            .map(|&(j, f)| {
                let rr = ((j as i128 * n_lo as i128).rem_euclid(m as i128)) as f64;
                let (sn, cs) = (TWO_PI_HI * rr / m as f64).sin_cos();
                let d = TWO_PI_HI * j as f64 / m as f64;
                let (sh2, ch2) = (d / 2.0).sin_cos();
                let g = C::new(-2.0 * a * sh2 * sh2, 2.0 * a * sh2 * ch2);
                let ph = phase_mod2(a2, a2e, d.sin());
                let c = f * C::new(cs, -sn) * C::from_polar(1.0, ph);
                let e = C::from_polar(1.0, p.theta + 2.0 * d);
                let big = g + g.conj() * e * tr;
                let ln0 = -half_ln_ch - g.norm_sqr() / 2.0 - g.conj() * g.conj() * e * tr / 2.0;
                Comp {
                    c,
                    big,
                    b: e * tr,
                    scale: ln0.re,
                    factor: if ln0.re > -740.0 { ln0.re.exp() } else { 0.0 },
                    prev: C::new(0.0, 0.0),
                    cur: C::from_polar(1.0, ln0.im),
                }
            })
            .collect();
        let rot_angle = phi_r + p.psi;
        let mut d = Vec::new();
        let mut quiet = 0;
        for k in 0..200_000usize {
            let mut sum = C::new(0.0, 0.0);
            let mut sc = 0.0;
            for x in &comps {
                if x.factor != 0.0 {
                    let term = x.c * x.cur * x.factor;
                    sum += term;
                    sc += term.re.abs() + term.im.abs();
                }
            }
            let dk = sum * C::from_polar(1.0, (k as f64 * rot_angle) % std::f64::consts::TAU);
            let floor = 1e-17f64.max(1e-11 * sc);
            quiet = if dk.norm() < floor { quiet + 1 } else { 0 };
            d.push(dk);
            if k > 40 && quiet >= 40 {
                break;
            }
            let (sk, inv) = ((k as f64).sqrt(), 1.0 / ((k + 1) as f64).sqrt());
            for x in comps.iter_mut() {
                let next = (x.big * x.cur - x.b * sk * x.prev) * inv;
                x.prev = x.cur;
                x.cur = next;
                let mag = next.norm();
                if mag > 1e100 || (mag < 1e-100 && mag > 0.0) {
                    x.cur /= mag;
                    x.prev /= mag;
                    x.scale += mag.ln();
                    x.factor = if x.scale > -740.0 { x.scale.exp() } else { 0.0 };
                }
            }
        }
        (C::from_polar(a, rot_angle), d)
    }
}

// banaketa metatu zehatzaren alderantzizkoa: x0 + y, F(y) = u·Σ|d_k|²
fn invert(d: &[C], x0: f64, rng: &mut StdRng) -> f64 {
    let total: f64 = crate::dd::sum(d.iter().map(|c| c.norm_sqr()));
    let target = rng.random::<f64>() * total;
    // hermite funtzioak √(2K+1)-tik aurrera erortzen dira; +12 tarte
    let reach = (2.0 * d.len() as f64 + 3.0).sqrt() + 12.0;
    let (mut lo, mut hi) = (-reach, reach);
    let mut y = 0.0;
    for _ in 0..200 {
        let (f, p) = cdf_density(d, y);
        let err = f - target;
        if err > 0.0 {
            hi = y;
        } else {
            lo = y;
        }
        if hi - lo < 1e-13 * (1.0 + y.abs()) || err.abs() < 1e-16 {
            break;
        }
        // newton, tartetik irtetzen bada erdibitu
        let step = if p > 0.0 { y - err / p } else { f64::NAN };
        y = if step > lo && step < hi {
            step
        } else {
            (lo + hi) / 2.0
        };
    }
    x0 + y
}

// egoera estutu puru bat: R(ψ) D(a) S(r e^{iθ}) |0>, a >= 0
#[derive(Clone, Copy, Debug)]
pub struct Pure {
    pub a: f64,
    pub r: f64,
    pub theta: f64,
    pub psi: f64,
}

impl Pure {
    // laborategiko egoera D(α) S(r) |0> (X estutua), α konplexua
    pub fn lab(alpha: C, r: f64) -> Pure {
        let psi = alpha.arg();
        Pure {
            a: alpha.norm(),
            r,
            theta: -2.0 * psi,
            psi,
        }
    }
}

// D(a) S(r e^{iθ}) |0>-ren fotoi-kopurua: batez bestekoa, desbideratzea eta isats estutuaren luzera
// (anplitudea tanh(r)^{n/2} < 1e-17)
fn number_stats(a: f64, r: f64, theta: f64) -> (f64, f64, f64) {
    let (sh, ch) = (r.sinh(), r.cosh());
    let (s2, c2) = (theta / 2.0).sin_cos();
    let var =
        a * a * ((-2.0 * r).exp() * c2 * c2 + (2.0 * r).exp() * s2 * s2) + 2.0 * sh * sh * ch * ch;
    let tail = if r > 0.0 {
        2.0 * 39.2 / -(r.tanh().ln())
    } else {
        0.0
    };
    (a * a + sh * sh, var.sqrt(), tail)
}

// ln <ψ|e^{iφn}|ψ>, ψ = D(α) S(r e^{iθ}) |0> (bargmann-en integral gaussiarretik), |α|² sin φ fasea gabe
// (itzulitako bigarren balioa, bere zehaztasunean murrizteko):
// -ln cosh r - ½ ln(1 - t₂t̄₁) - |δ|²/2 - t₂δ̄²/2 - t̄₁(δ + t₂δ̄)²/(2(1 - t₂t̄₁)), t₁ = e^{iθ} tanh r,
// t₂ = e^{i(θ+2φ)} tanh r, δ = α(e^{iφ} - 1)
fn ln_chi(alpha: C, r: f64, theta: f64, phi: f64) -> (C, f64) {
    let tr = r.tanh();
    let t1 = C::from_polar(tr, theta);
    let t2 = C::from_polar(tr, theta + 2.0 * phi);
    // e^{iφ} - 1 = 2i sin(φ/2) e^{iφ/2}, ezeztapenik gabe
    let (sh, ch) = (phi / 2.0).sin_cos();
    let em1 = C::new(-2.0 * sh * sh, 2.0 * sh * ch);
    let delta = alpha * em1;
    let den = 1.0 - t2 * t1.conj();
    let dc = delta.conj();
    let q = -r.cosh().ln()
        - 0.5 * den.ln()
        - delta.norm_sqr() / 2.0
        - t2 * dc * dc / 2.0
        - t1.conj() * (delta + t2 * dc) * (delta + t2 * dc) / (2.0 * den);
    (q, alpha.norm_sqr() * phi.sin())
}

// <ψ|e^{iφn} a|ψ> = χ(φ)·(Γ - e^{i(θ+φ)} tanh r Γ̄)/(cosh r - sinh r tanh r e^{2iφ}), Γ = α cosh r + ᾱ e^{iθ} sinh r
// (a ψ = (Γ - e^{iθ} sinh r a†)ψ/cosh r eta <e^{iφn}a†> = e^{iφ} conj <e^{-iφn}a>-tik). kerr-en batez besteko
// eremua φ = -2κ-n da: <U†aU> = <e^{-2iκn} a>
fn mean_factor(alpha: C, r: f64, theta: f64, phi: f64) -> C {
    let (sh, ch) = (r.sinh(), r.cosh());
    let e = C::from_polar(1.0, theta);
    let big = alpha * ch + alpha.conj() * e * sh;
    (big - C::from_polar(r.tanh(), theta + phi) * big.conj())
        / (ch - C::from_polar(sh * r.tanh(), 2.0 * phi))
}

// kerr-en batez besteko eremua <U†aU> egoera puru batean: D(α) S(r) |0>
pub fn kerr_mean_pure(alpha: C, r: f64, kappa: f64) -> C {
    let phi = -2.0 * kappa;
    let (q, big) = ln_chi(alpha, r, 0.0, phi);
    (q + C::new(0.0, big)).exp() * mean_factor(alpha, r, 0.0, phi)
}

// kerr-en batez besteko eremua (α unitateetan) argi estutu eta galdu baterako: egoera nahastea D(a + iu) S(r)|0>
// u ~ N(0, s²) gainean. ln χ zehazki koadratikoa da u-n eta faktorea lineala, beraz batez bestekoa integral
// gaussiar itxia da: e^{q0 + q1²/(4A)} √(π/A)/(√(2π) s) (g0 + g1 q1/(2A)), A = 1/(2s²) - q2
pub fn kerr_mean_mixed(a: f64, r: f64, s: f64, kappa: f64) -> C {
    let phi = -2.0 * kappa;
    let at = |u: f64| {
        let alpha = C::new(a, u);
        let (q, big) = ln_chi(alpha, r, 0.0, phi);
        // |α|² sin φ = a² sin φ + u² sin φ: a²-ren zatia konstantea da
        (
            q + C::new(0.0, big - a * a * phi.sin()),
            mean_factor(alpha, r, 0.0, phi),
        )
    };
    let big0 = a * a * phi.sin();
    let (r0, g0) = at(0.0);
    if s == 0.0 {
        return C::from_polar(1.0, big0) * r0.exp() * g0;
    }
    let ((rp, gp), (rm, gm)) = (at(s), at(-s));
    let q1 = (rp - rm) / (2.0 * s);
    let q2 = ((rp + rm) / 2.0 - r0) / (s * s);
    let g1 = (gp - gm) / (2.0 * s);
    let big_a = C::new(1.0 / (2.0 * s * s), 0.0) - q2;
    let pref = (C::new(PI, 0.0) / big_a).sqrt() / ((2.0 * PI).sqrt() * s);
    C::from_polar(1.0, big0)
        * (r0 + q1 * q1 / (4.0 * big_a)).exp()
        * pref
        * (g0 + g1 * q1 / (2.0 * big_a))
}

// D(α) S(r) |0>-ren fotoi-kopuruaren banaketa zehatza: P(n) = (1/M) Σ_k χ(2πk/M) e^{-i2πkn/M} leiho batean
// (euskarria ± 12σ + isatsa); leihotik kanpoko probabilitatea < 1e-34 denez, aliasing-a arbuiagarria da.
// itzultzen du (lehen n, probabilitateak)
pub fn number_distribution(alpha: C, r: f64) -> (i64, Vec<f64>) {
    let theta_rel = -2.0 * alpha.arg();
    let (n_bar, std, tail) = number_stats(alpha.norm(), r, theta_rel);
    let w = 12.0 * std + 31.0 + tail;
    let m = ((2.0 * w).ceil() as usize).max(64).next_power_of_two();
    let n_lo = n_bar.round() as i64 - (m / 2) as i64;
    let (a2, a2e) = crate::dd::two_prod::<false>(alpha.norm(), alpha.norm());
    let mut buf: Vec<C> = (0..m)
        .map(|k| {
            let phi = TWO_PI_HI * k as f64 / m as f64;
            let (q, _) = ln_chi(alpha, r, 0.0, phi);
            let big = phase_mod2(a2, a2e, phi.sin());
            // e^{-iφ n_lo}, osoko aritmetikaz
            let shift = ((k as i128 * n_lo as i128).rem_euclid(m as i128)) as f64;
            let ph = big - TWO_PI_HI * shift / m as f64;
            q.exp() * C::from_polar(1.0, ph)
        })
        .collect();
    FftPlanner::new().plan_fft_forward(m).process(&mut buf);
    let inv = 1.0 / m as f64;
    let probs = (0..m)
        .map(|j| {
            if n_lo + (j as i64) < 0 {
                0.0
            } else {
                (buf[j].re * inv).max(0.0)
            }
        })
        .collect();
    (n_lo, probs)
}

// fotoi-zenbaketa bat D(α) S(r) |0>-n: banaketa zehatzaren alderantzizkoa
pub fn count_pure(alpha: C, r: f64, rng: &mut StdRng) -> u64 {
    let (n_lo, probs) = number_distribution(alpha, r);
    let mut cum = Vec::with_capacity(probs.len());
    let mut acc = crate::dd::Dd::ZERO;
    for &p in &probs {
        acc.add(p);
        cum.push(acc.value());
    }
    let u = rng.random::<f64>() * acc.value();
    let j = cum.partition_point(|&c| c <= u).min(probs.len() - 1);
    (n_lo + j as i64).max(0) as u64
}

// κ·q mod 2π, q = hi + lo (double-double) bider s
fn phase_mod2(hi: f64, lo: f64, s: f64) -> f64 {
    let (p, e) = crate::dd::two_prod::<false>(hi, s);
    let e = e + lo * s;
    let k = (p / TWO_PI_HI).round();
    let (a, ae) = crate::dd::two_prod::<false>(k, TWO_PI_HI);
    ((p - a) - ae) - k * TWO_PI_LO + e
}

// erfc zehatza: |x| < 2 taylor seriez (erf), bestela zatiki jarraituz (lentz)
pub fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 2.0 {
        // erf(x) = 2/√π Σ (-1)^n x^{2n+1} / (n! (2n+1))
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        for n in 1..80 {
            term *= -x2 / n as f64;
            let add = term / (2 * n + 1) as f64;
            sum += add;
            if add.abs() < 1e-18 * sum.abs() {
                break;
            }
        }
        return 1.0 - 2.0 / PI.sqrt() * sum;
    }
    // erfc(x) = e^{-x²}/√π · 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))
    let tiny = 1e-300;
    let mut f = x;
    let mut c = x;
    let mut dd = 0.0;
    for n in 1..300 {
        let an = n as f64 / 2.0;
        dd = x + an * dd;
        dd = if dd.abs() < tiny { tiny } else { dd };
        c = x + an / c;
        c = if c.abs() < tiny { tiny } else { c };
        dd = 1.0 / dd;
        let delta = c * dd;
        f *= delta;
        if (delta - 1.0).abs() < 1e-17 {
            break;
        }
    }
    (-x * x).exp() / (PI.sqrt() * f)
}

// F(y) = ∫_{-∞}^y |Σ d_k ψ_k|² eta P(y) = |Σ d_k ψ_k(y)|², ψ_k hermite funtzio normalizatuak. k != l denean
// ∫ψ_kψ_l = (ψ_k'ψ_l - ψ_kψ_l')/(2(l-k)) (wronskiarra, ψ'' = (y² - 2k - 1)ψ-tik); diagonala errepikapenez:
// J_kk = J_{k-1,k-1} + (√(k-1) J_{k-2,k} - √(k+1) J_{k-1,k+1} - √2 ψ_{k-1}ψ_k)/√k, J_00 = erfc(-y)/2
fn cdf_density(d: &[C], y: f64) -> (f64, f64) {
    let kk = d.len();
    // ψ_0..ψ_{K+1}
    let mut psi = Vec::with_capacity(kk + 2);
    psi.push(PI.powf(-0.25) * (-y * y / 2.0).exp());
    psi.push(2f64.sqrt() * y * psi[0]);
    for n in 1..=kk {
        let next = (2.0 / (n + 1) as f64).sqrt() * y * psi[n]
            - (n as f64 / (n + 1) as f64).sqrt() * psi[n - 1];
        psi.push(next);
    }
    // ψ_k' = (√k ψ_{k-1} - √(k+1) ψ_{k+1})/√2
    let dpsi: Vec<f64> = (0..=kk)
        .map(|k| {
            let down = if k > 0 {
                (k as f64).sqrt() * psi[k - 1]
            } else {
                0.0
            };
            (down - ((k + 1) as f64).sqrt() * psi[k + 1]) / 2f64.sqrt()
        })
        .collect();
    let j_off =
        |k: usize, l: usize| (dpsi[k] * psi[l] - psi[k] * dpsi[l]) / (2.0 * (l as f64 - k as f64));
    // diagonala
    let mut jd = Vec::with_capacity(kk);
    jd.push(erfc(-y) / 2.0);
    for k in 1..kk {
        let a = if k >= 2 {
            ((k - 1) as f64).sqrt() * j_off(k - 2, k)
        } else {
            0.0
        };
        let b = ((k + 1) as f64).sqrt() * j_off(k - 1, k + 1);
        let c = 2f64.sqrt() * psi[k - 1] * psi[k];
        jd.push(jd[k - 1] + (a - b - c) / (k as f64).sqrt());
    }
    let mut cdf = crate::dd::Dd::ZERO;
    let mut amp = C::new(0.0, 0.0);
    for k in 0..kk {
        amp += d[k] * psi[k];
        cdf.add(d[k].norm_sqr() * jd[k]);
    }
    // ez-diagonala: Σ_{k≠l} Re(d_k d̄_l) J_kl = Re Σ_k A_k Σ_{l≠k} B̄_l/(l-k), A_k = d_k ψ_k', B_k = d_k ψ_k
    // (hilbert-en transformazio diskretua: K handietan fft bidez, K log K)
    let a: Vec<C> = (0..kk).map(|k| d[k] * dpsi[k]).collect();
    let b: Vec<C> = (0..kk).map(|k| (d[k] * psi[k]).conj()).collect();
    let h = hilbert(&b);
    for k in 0..kk {
        cdf.add((a[k] * h[k]).re);
    }
    (cdf.value(), amp.norm_sqr())
}

// h_k = Σ_{l≠k} b_l/(l-k): zuzenean K txikietan, bestela korrelazio gisa fft-z (2K luzerako zero-betetzea)
fn hilbert(b: &[C]) -> Vec<C> {
    let kk = b.len();
    if kk <= 256 {
        return (0..kk)
            .map(|k| {
                let mut s = C::new(0.0, 0.0);
                for (l, bl) in b.iter().enumerate() {
                    if l != k {
                        s += bl / (l as f64 - k as f64);
                    }
                }
                s
            })
            .collect();
    }
    // h_k = Σ_m b_{k+m} r(m), r(m) = 1/m (m≠0): korrelazioa = konboluzioa b-rekin eta r(-m)-rekin
    let n = (2 * kk).next_power_of_two();
    let mut planner = FftPlanner::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);
    let mut fb: Vec<C> = (0..n)
        .map(|i| if i < kk { b[i] } else { C::new(0.0, 0.0) })
        .collect();
    // kernel-a: g(j) = r(-j) = -1/j, j zeinuduna (-K..K) n moduluan
    let mut fg = vec![C::new(0.0, 0.0); n];
    for j in 1..kk as i64 {
        fg[j as usize] = C::new(-1.0 / j as f64, 0.0);
        fg[n - j as usize] = C::new(1.0 / j as f64, 0.0);
    }
    fwd.process(&mut fb);
    fwd.process(&mut fg);
    for (x, y) in fb.iter_mut().zip(&fg) {
        *x *= y;
    }
    inv.process(&mut fb);
    // (b ⊛ g)(k) = Σ_l b_l g(k-l) = Σ_l b_l (-1/(k-l)) = Σ_l b_l/(l-k)
    let scale = 1.0 / n as f64;
    (0..kk).map(|k| fb[k] * scale).collect()
}
