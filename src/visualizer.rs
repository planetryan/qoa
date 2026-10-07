// fitxategi honek `qoa visual` egiten du: audio-fitxategi bat (.wav) bideo bihurtzen du ffmpeg-ekin.
// arte-tresna bat da, ez du egoera kuantikorik bistaratzen. qoa-ren simulagailua ausazko bit-iturri gisa
// bakarrik erabiltzen da (ikus `BitSource`).
//
// irudi bakoitzeko:
//   1. audio-zatiaren ezaugarriak (fft): ozentasuna, zentroidea, banden energia, transienteak...
//   2. ezaugarriak parametro bisual bihurtu (`Look`)
//   3. errendatu: perlin zarata + interferentzia + keinuak + desplazamendua, eta espektroa barra gisa
//   4. pixel gordinak zuzenean ffmpeg-en sarrerara (stdin), tarteko png fitxategirik gabe
// ezaugarriak ordenan kalkulatzen dira (aurreko irudiaren araberakoak dira) eta errendatzea paraleloa da
// banda horizontaletan, ordena finko batean; beraz hazi berak bideo bera ematen du edozein hari-kopururekin.

use indicatif::{ProgressBar, ProgressStyle};
use qoa::program::Gate;
use qoa::report::Style;
use qoa::state::State;
use qoa::{t, ts};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use realfft::RealFftPlanner;
use std::f64::consts::{FRAC_PI_2, PI};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumDirection {
    None,
    Ltr,
    Rtl,
}

pub struct Options<'a> {
    pub input: &'a Path,
    pub output: &'a Path,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub direction: SpectrumDirection,
    pub seed: u64,
    pub ffmpeg_flags: &'a [String],
    pub ffmpeg_args: &'a [String],
}

// --- audioa ---

// wav-a irakurri eta mono bihurtu, [-1, 1] tartean. itzultzen du: laginak, lagin-maiztasuna, kanal-kopurua
pub fn decode_wav(path: &Path) -> Result<(Vec<f32>, u32, u16), String> {
    let is_wav = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
    if !is_wav {
        return Err(t!(
            "'{}' is not a .wav file; convert it first, e.g. ffmpeg -i song.mp3 song.wav",
            "'{}' ez da .wav fitxategi bat; bihurtu lehenago, adib. ffmpeg -i abestia.mp3 abestia.wav",
            path.display()
        ));
    }
    let unreadable = |e: hound::Error| {
        t!(
            "cannot read '{}': {}",
            "ezin da '{}' irakurri: {}",
            path.display(),
            e
        )
    };
    let mut reader = hound::WavReader::open(path).map_err(unreadable)?;
    let spec = reader.spec();
    // osoko laginak bit-sakoneraren arabera eskalatu (8, 16, 24 edo 32 bit)
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect()
        }
    }
    .map_err(unreadable)?;
    Ok((
        mix_to_mono(&samples, spec.channels as usize),
        spec.sample_rate,
        spec.channels,
    ))
}

// kanal guztien batez bestekoa, lagin-multzo bakoitzeko
pub fn mix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let inv = 1.0 / channels as f32;
    interleaved
        .chunks_exact(channels)
        .map(|f| f.iter().sum::<f32>() * inv)
        .collect()
}

// irudi bateko audio-ezaugarriak
#[derive(Debug, Clone, Default)]
pub struct Features {
    pub rms: f64,
    pub centroid_hz: f64,
    pub low: f64,        // 0-200 Hz arteko energia
    pub mid: f64,        // 200-2000 Hz
    pub high: f64,       // 2000-20000 Hz
    pub rolloff_hz: f64, // energiaren % 85 maiztasun honen azpian dago
    pub transient: f64,  // ozentasunaren aldaketa aurreko iruditik, [0, 1]
}

// banda baten energia: [low_hz, high_hz] arteko bin-en magnitudeen karratuen batura (muturrak barne)
pub fn band_energy(mags: &[f64], bin_hz: f64, low_hz: f64, high_hz: f64) -> f64 {
    if mags.is_empty() || bin_hz <= 0.0 || low_hz >= high_hz {
        return 0.0;
    }
    let start = (low_hz / bin_hz).floor().max(0.0) as usize;
    let end = ((high_hz / bin_hz).ceil() as usize).min(mags.len() - 1);
    if start > end {
        return 0.0;
    }
    mags[start..=end].iter().map(|m| m * m).sum()
}

// `frame`: irudi honen laginak; `mags`: haien fft-aren magnitudeak; `bin_hz`: bin baten zabalera
pub fn features(frame: &[f32], mags: &[f64], bin_hz: f64, prev_rms: f64) -> Features {
    let rms = if frame.is_empty() {
        0.0
    } else {
        (frame.iter().map(|&s| s as f64 * s as f64).sum::<f64>() / frame.len() as f64).sqrt()
    };
    let total: f64 = mags.iter().sum();
    let weighted: f64 = mags
        .iter()
        .enumerate()
        .map(|(i, m)| i as f64 * bin_hz * m)
        .sum();
    let mut cumulative = 0.0;
    let mut rolloff = 0;
    for (i, m) in mags.iter().enumerate() {
        cumulative += m;
        if cumulative >= 0.85 * total {
            rolloff = i;
            break;
        }
    }
    Features {
        rms,
        centroid_hz: if total > 0.0 { weighted / total } else { 0.0 },
        low: band_energy(mags, bin_hz, 0.0, 200.0),
        mid: band_energy(mags, bin_hz, 200.0, 2000.0),
        high: band_energy(mags, bin_hz, 2000.0, 20000.0),
        rolloff_hz: rolloff as f64 * bin_hz,
        transient: ((rms - prev_rms).abs() * 50.0).min(1.0),
    }
}

// --- bit-iturria ---

const QUBITS: usize = 16;
const HOLD_FRAMES: usize = 30;

// ausazko bitak simulatutako qubitetatik: qubit bakoitza |+> egoeran prestatzen da (H) eta neurtu egiten
// da, beraz bit bakoitza 50/50 da. neurtu ondoren egoera kolapsatuta dago eta bit berak ematen ditu berriz
// prestatu arte (`HOLD_FRAMES` irudiro). emaitzak sasi-ausazko sorgailu batek erabakitzen ditu (simulagailu
// batean beti bezala): ez da benetako ausazkotasun kuantikoa.
struct BitSource {
    state: State,
    rng: StdRng,
    frame: usize,
}

impl BitSource {
    fn new(seed: u64) -> Self {
        BitSource {
            state: Self::prepare(),
            rng: StdRng::seed_from_u64(seed),
            frame: 0,
        }
    }

    // |+...+> prestatu: qubit bakoitzean H
    fn prepare() -> State {
        let mut state = State::new(QUBITS, 0, 2).expect("16 qubit");
        for q in 0..QUBITS {
            state.apply_1q(q, Gate::H.matrix(), 0);
        }
        state
    }

    // irudi honetako bitak, bit bat qubit bakoitzeko
    fn next(&mut self) -> u64 {
        if self.frame.is_multiple_of(HOLD_FRAMES) {
            self.state = Self::prepare();
        }
        self.frame += 1;
        let mut bits = 0u64;
        for q in 0..QUBITS {
            let r: f64 = self.rng.random();
            if self.state.measure_qubit(q, r) == 1 {
                bits |= 1 << q;
            }
        }
        bits
    }
}

// --- matematika azkarra ---

// bhaskara i-ren sinu-hurbilketa, [0, π] tartean definitua eta beste erdirako ikurra aldatuta.
// errore absolutu maximoa 0.0017 ingurukoa da
#[inline(always)]
pub fn fast_sin(x: f64) -> f64 {
    // [0, 2π) tartera eraman libm-ko deirik gabe (x86-64 oinarrizkoan `floor` eta `%` funtzio-deiak
    // dira, `as i64` berriz instrukzio bakarra) eta adarrik gabe: pixel bakoitzean deitzen da eta
    // ausazko adarrek iragarpen-hutsegiteak eragiten dituzte
    let turns = x * (0.5 / PI);
    let mut k = turns as i64 as f64;
    k -= (k > turns) as u8 as f64;
    let mut y = (turns - k) * (2.0 * PI);
    let upper = (y > PI) as u8 as f64;
    y -= upper * PI;
    let p = y * (PI - y);
    (1.0 - 2.0 * upper) * 16.0 * p / (5.0 * PI * PI - 4.0 * p)
}

// sinua eta kosinua |x| <= π denean, taylor polinomioekin (x^13 eta x^14 arte) horner eran. koefizienteak
// konstanteak dira: `x / 6.0` idatziz gero konpilatzaileak zatiketa bat egiten du pixel bakoitzean (biderketa
// bat baino 5-10 aldiz motelagoa), ez baitu `x * (1/6)` emaitza bera ematen. errorea 3e-5 baino txikiagoa
// da π-n eta 3e-8 baino txikiagoa |x| <= 2 denean (pixelen desplazamendua)
#[inline(always)]
fn sin_cos_small(x: f64) -> (f64, f64) {
    const S: [f64; 6] = [
        -1.0 / 6.0,
        1.0 / 120.0,
        -1.0 / 5040.0,
        1.0 / 362880.0,
        -1.0 / 39916800.0,
        1.0 / 6227020800.0,
    ];
    const C: [f64; 7] = [
        -1.0 / 2.0,
        1.0 / 24.0,
        -1.0 / 720.0,
        1.0 / 40320.0,
        -1.0 / 3628800.0,
        1.0 / 479001600.0,
        -1.0 / 87178291200.0,
    ];
    let x2 = x * x;
    let s =
        x * (1.0 + x2 * (S[0] + x2 * (S[1] + x2 * (S[2] + x2 * (S[3] + x2 * (S[4] + x2 * S[5]))))));
    let c = 1.0
        + x2 * (C[0]
            + x2 * (C[1] + x2 * (C[2] + x2 * (C[3] + x2 * (C[4] + x2 * (C[5] + x2 * C[6]))))));
    (s, c)
}

// hsv -> rgb, h graduetan [0, 360), s eta v [0, 1] tartean
pub fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [f64; 3] {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

// pixel bakoitzeko ausazko zenbaki deterministikoa [0, 1) tartean (splitmix64-ren nahasketa)
#[inline(always)]
fn pixel_random(seed: u64, frame: u64, x: u32, y: u32) -> f64 {
    let mut z = seed
        ^ frame.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ ((y as u64) << 32 | x as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

// --- perlin zarata ---

// ardatz bateko gelaxka: sare-puntua (0..255), barneko posizioa eta haren leunketa
#[derive(Clone, Copy)]
struct Cell {
    i: usize,
    f: f64,
    fade: f64,
}

impl Cell {
    fn at(v: f64) -> Cell {
        let fl = v.floor();
        let f = v - fl;
        Cell {
            i: (fl as i64 & 255) as usize,
            f,
            fade: f * f * f * (f * (f * 6.0 - 15.0) + 10.0),
        }
    }
}

// perlin zarata 2d klasikoa (ken perlin, 2002): permutazio-taula hazi batetik nahastuta, 8 gradiente.
// balioak [-1, 1] tartean daude eta 0 dira sare-puntuetan
pub struct Perlin {
    p: [u8; 512],
}

impl Perlin {
    pub fn new(seed: u64) -> Perlin {
        let mut table: Vec<u8> = (0..=255).collect();
        table.shuffle(&mut StdRng::seed_from_u64(seed));
        let mut p = [0u8; 512];
        for (i, v) in p.iter_mut().enumerate() {
            *v = table[i & 255];
        }
        Perlin { p }
    }

    // 8 gradienteak: (±1, ±1), (±1, 0), (0, ±1). taula batetik, adarrik gabe
    #[inline(always)]
    fn grad(hash: u8, x: f64, y: f64) -> f64 {
        const G: [(f64, f64); 8] = [
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
        ];
        let (gx, gy) = G[(hash & 7) as usize];
        gx * x + gy * y
    }

    #[inline(always)]
    fn lerp(t: f64, a: f64, b: f64) -> f64 {
        a + t * (b - a)
    }

    #[inline(always)]
    fn at_cells(&self, cx: Cell, cy: Cell) -> f64 {
        let p = &self.p;
        let a = p[cx.i] as usize + cy.i;
        let b = p[cx.i + 1] as usize + cy.i;
        let n00 = Self::grad(p[a], cx.f, cy.f);
        let n10 = Self::grad(p[b], cx.f - 1.0, cy.f);
        let n01 = Self::grad(p[a + 1], cx.f, cy.f - 1.0);
        let n11 = Self::grad(p[b + 1], cx.f - 1.0, cy.f - 1.0);
        Self::lerp(
            cy.fade,
            Self::lerp(cx.fade, n00, n10),
            Self::lerp(cx.fade, n01, n11),
        )
    }

    #[cfg(test)]
    fn at(&self, x: f64, y: f64) -> f64 {
        self.at_cells(Cell::at(x), Cell::at(y))
    }
}

// --- parametro bisualak ---

// irudi baten parametroak, audio-ezaugarrietatik eta biten iturritik kalkulatuak
#[derive(Debug, Clone, Default)]
struct Look {
    // oinarrizko distira (1etik gora; pixel bakoitzean mugatzen da)
    brightness: f64,
    // irudiaren kolorea v = 1 denean; pixel bakoitzak bere distirarekin biderkatzen du
    rgb: [f64; 3],
    // zarata-ereduaren dentsitatea
    density: f64,
    // zarata-eremuaren desplazamendua (bitetatik eta kaosetik)
    offset: f64,
    // pixel bakoitzeko ausazko distiraren pisua (transienteekin)
    flicker: f64,
    // sinu-sareta baten pisua
    interference: f64,
    // koordenatuen kurbadura
    flow: f64,
    // pixelen desplazamendua, 20 pixeleraino
    depth: f64,
}

// espektroaren zentroidea [0, 1) tartean (bin-indizea / bin-kopurua), kolorerako
fn spectrum_centroid01(mags: &[f64]) -> f64 {
    let total: f64 = mags.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let inv = 1.0 / mags.len() as f64;
    mags.iter()
        .enumerate()
        .map(|(i, m)| i as f64 * inv * m)
        .sum::<f64>()
        / total
}

impl Look {
    // `hue`: kolore-zikloa, irudi bakoitzean 0.01 gradu aurreratzen da
    fn new(
        f: &Features,
        mags: &[f64],
        bits: u64,
        frame: usize,
        frames: usize,
        hue: &mut f64,
    ) -> Look {
        let unit = |v: f64| v.clamp(0.0, 1.0);
        let rms = unit(f.rms);
        let centroid = f.centroid_hz.clamp(0.0, 5000.0) / 5000.0;
        let rolloff = f.rolloff_hz.clamp(0.0, 20000.0) / 20000.0;
        let transient = unit(f.transient);
        let bass = unit(f.low / (f.low + f.mid + f.high + 1e-6));

        // bitetatik: batekoen proportzioa eta haren araberako "zarata" [-1, 1] tartean
        let ones = bits.count_ones() as f64 / QUBITS as f64;
        let noise = (ones * 0.8 + (frame as f64 / frames as f64 * 0.1).sin() * 0.2) * 2.0 - 1.0;

        *hue = (*hue + 0.01) % 360.0;
        let h = (*hue + spectrum_centroid01(mags) * 180.0 + (1.0 - noise.abs()) * 90.0) % 360.0;
        let s = (0.7 + ones * 0.3).min(1.0);
        let chaos = (f.high / (f.low + 1.0)).min(10.0) / 10.0;
        Look {
            brightness: 1.0 + bass + rms * 0.3,
            rgb: hsv_to_rgb(h, s, 1.0),
            density: 4.0 + bass * 8.0 + rms * 1.5,
            offset: (noise * 1_000_000.0).abs().floor() / 1000.0 + chaos * 5.0,
            flicker: transient * (1.0 + bass),
            interference: (rolloff + centroid) / 2.0,
            flow: rms * centroid,
            depth: transient,
        }
    }
}

// --- errendatzea ---

// banda bakoitzak 64 errenkada ditu; pixel bat 20 errenkadaraino mugi daiteke (`Look::depth`), beraz
// banda batek bere ertzetatik 24 errenkadara arte idazten du. bandak bi fasetan prozesatzen dira
// (bikoitiak, gero bakoitiak): fase bereko bandek ez dute errenkada bera ukitzen, 64 >= 2 * 24 delako
const BAND: usize = 64;
const MARGIN: usize = 24;

// ardatz bateko aurrekalkulua: zarataren gelaxka eta interferentziaren uhina, koordenatu bakoitzeko.
// `phase` 0 da x ardatzerako eta π/2 y ardatzerako (kosinua)
fn axis(n: u32, look: &Look, phase: f64) -> Vec<(Cell, f64)> {
    let half = n as f64 / 2.0;
    (0..n)
        .map(|i| {
            let mut c = (i as f64 - half) / n as f64;
            c += fast_sin(c + look.flow * 0.1 + phase) * 0.05;
            (
                Cell::at(c * look.density * 10.0 + look.offset),
                fast_sin(c * 10.0 + phase),
            )
        })
        .collect()
}

struct Window<'a> {
    first_row: usize, // `buf`-en lehen errenkada irudian
    rows: std::ops::Range<usize>,
    buf: &'a mut [u8],
}

// fase bateko leihoak: banda bakoitzaren errenkadak eta ertzak, elkarren artean gainjarri gabe
fn windows(pixels: &mut [u8], row_bytes: usize, height: usize, phase: usize) -> Vec<Window<'_>> {
    let mut out = Vec::new();
    let mut rest = pixels;
    let mut rest_row = 0;
    let bands = height.div_ceil(BAND);
    for k in (phase..bands).step_by(2) {
        let lo = (k * BAND).saturating_sub(MARGIN);
        let hi = ((k + 1) * BAND + MARGIN).min(height);
        let (_, tail) = std::mem::take(&mut rest).split_at_mut((lo - rest_row) * row_bytes);
        let (buf, tail) = tail.split_at_mut((hi - lo) * row_bytes);
        rest = tail;
        rest_row = hi;
        out.push(Window {
            first_row: lo,
            rows: k * BAND..((k + 1) * BAND).min(height),
            buf,
        });
    }
    out
}

// zarata-eremua banda batean: pixel bakoitzak bere kolorea kalkulatzen du eta desplazatutako lekuan
// nahasten du (% 50) lehendik dagoenarekin
#[allow(clippy::too_many_arguments)]
fn render_band(
    w: Window,
    look: &Look,
    perlin: &Perlin,
    cols: &[(Cell, f64)],
    rows: &[(Cell, f64)],
    width: usize,
    frame: u64,
    seed: u64,
) {
    let row_bytes = width * 3;
    let lo = w.first_row;
    let hi = lo + w.buf.len() / row_bytes;
    let rgb = look.rgb.map(|c| c * 255.0);
    for y in w.rows.clone() {
        let (cy, wave_y) = rows[y];
        for (x, &(cx, wave_x)) in cols.iter().enumerate() {
            let n = perlin.at_cells(cx, cy);
            let value = (n + wave_x * wave_y * look.interference).clamp(-1.0, 1.0);
            let mut v = look.brightness * (value + 1.0) / 2.0 * (1.0 - look.flicker);
            if look.flicker > 0.0 {
                v += look.flicker * pixel_random(seed, frame, x as u32, y as u32);
            }
            let v = v.clamp(0.0, 1.0);
            // desplazamendua (sin 2n, cos 2n) norabidean; |n| <= 1. koordenatu negatiboak 0an geratzen dira
            let (sin, cos) = sin_cos_small(n * 2.0);
            let tx = (x as f64 + look.depth * 20.0 * sin) as usize;
            let ty = (y as f64 + look.depth * 20.0 * cos) as usize;
            if tx >= width || ty < lo || ty >= hi {
                continue;
            }
            let at = (ty - lo) * row_bytes + tx * 3;
            for (old, c) in w.buf[at..at + 3].iter_mut().zip(rgb) {
                *old = ((*old as u16 + (v * c) as u8 as u16) >> 1) as u8;
            }
        }
    }
}

// espektroaren barren altuera zutabe bakoitzeko (0: barrarik ez)
fn bar_heights(mags: &[f64], width: usize, height: usize) -> Vec<usize> {
    let max_h = (height / 4) as f64;
    let bar_h = |m: f64| (m * 500.0).min(max_h) as usize;
    let mut heights = vec![0; width];
    if mags.is_empty() {
        return heights;
    }
    let bar_w = width as f64 / mags.len() as f64;
    if bar_w >= 1.0 {
        // bin bakoitzak bere barra, `bar_w` zabalekoa (osoko zatira biribildua)
        for (i, &m) in mags.iter().enumerate() {
            let x0 = (i as f64 * bar_w) as usize;
            for h in heights.iter_mut().skip(x0).take(bar_w as usize) {
                *h = bar_h(m);
            }
        }
    } else {
        // bin gehiago zutabeak baino: zutabe bakoitzak bere bin-en maximoa
        for (x, h) in heights.iter_mut().enumerate() {
            let a = x * mags.len() / width;
            let b = ((x + 1) * mags.len() / width).max(a + 1);
            *h = bar_h(mags[a..b].iter().copied().fold(0.0, f64::max));
        }
    }
    heights
}

// espektroa barra gorri gisa (% 80 gorria): ltr behean ezkerretik, rtl goian eskuinetik
fn render_spectrum(
    pixels: &mut [u8],
    mags: &[f64],
    direction: SpectrumDirection,
    width: usize,
    height: usize,
) {
    if direction == SpectrumDirection::None {
        return;
    }
    let mut heights = bar_heights(mags, width, height);
    if direction == SpectrumDirection::Rtl {
        heights.reverse();
    }
    pixels
        .par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, &h) in heights.iter().enumerate() {
                let covered = match direction {
                    SpectrumDirection::Ltr => y + h >= height && h > 0,
                    _ => y < h,
                };
                if covered {
                    for (old, red) in row[x * 3..x * 3 + 3].iter_mut().zip([255.0f32, 0.0, 0.0]) {
                        *old = (*old as f32 * 0.2 + red * 0.8) as u8;
                    }
                }
            }
        });
}

// irudi oso bat `pixels`-en (rgb24, zabalera * altuera * 3 byte)
#[allow(clippy::too_many_arguments)]
fn render(
    pixels: &mut [u8],
    look: &Look,
    perlin: &Perlin,
    mags: &[f64],
    direction: SpectrumDirection,
    width: usize,
    height: usize,
    frame: u64,
    seed: u64,
) {
    pixels.fill(0);
    let cols = axis(width as u32, look, 0.0);
    let rows = axis(height as u32, look, FRAC_PI_2);
    for phase in 0..2 {
        windows(pixels, width * 3, height, phase)
            .into_par_iter()
            .for_each(|w| render_band(w, look, perlin, &cols, &rows, width, frame, seed));
    }
    render_spectrum(pixels, mags, direction, width, height);
}

// --- bideoa ---

// irudiak sortu eta ffmpeg-i pasatu. irudien kalkulua eta ffmpeg-en idazketa paraleloan doaz
pub fn render_video(o: &Options) -> Result<(), String> {
    let started = Instant::now();
    let (width, height) = (o.width as usize, o.height as usize);
    if width == 0 || height == 0 || o.fps == 0 {
        return Err(ts!(
            "width, height and fps must be above zero",
            "zabalerak, altuerak eta fps-k zero baino handiagoak izan behar dute"
        )
        .into());
    }
    let (audio, rate, channels) = decode_wav(o.input)?;
    let per_frame = (rate / o.fps) as usize;
    if per_frame == 0 || audio.is_empty() {
        return Err(t!(
            "'{}' has no audio to render (or fps is above the sample rate)",
            "'{}'-k ez du errendatzeko audiorik (edo fps lagin-maiztasuna baino handiagoa da)",
            o.input.display()
        ));
    }
    let frames = audio.len().div_ceil(per_frame);

    let style = Style::detect();
    let line = |label: &str, text: String| println!("{}  {}", style.label(label), text);
    line(
        ts!("input", "sarrera"),
        t!(
            "{}: {:.2} s, {} Hz, {} channel(s) mixed to mono",
            "{}: {:.2} s, {} Hz, {} kanal monora nahastuta",
            o.input.display(),
            audio.len() as f64 / rate as f64,
            rate,
            channels
        ),
    );
    line(
        ts!("video", "bideoa"),
        t!(
            "{}: {}x{}, {} frames/s, {} frames, H.264 (libx264 crf 18) with AAC audio",
            "{}: {}x{}, {} irudi/s, {} irudi, H.264 (libx264 crf 18) AAC audioarekin",
            o.output.display(),
            width,
            height,
            o.fps,
            frames
        ),
    );
    line(
        ts!("seed", "hazia"),
        t!(
            "{} (same seed and audio, same video)",
            "{} (hazi eta audio berak, bideo bera)",
            o.seed
        ),
    );
    line(
        ts!("note", "oharra"),
        ts!(
            "an art tool: the picture follows the audio, and 16 simulated qubits supply random bits. It does not show a quantum state",
            "arte-tresna bat: irudiak audioari jarraitzen dio, eta simulatutako 16 qubitek ausazko bitak ematen dituzte. Ez du egoera kuantikorik erakusten"
        )
        .to_string(),
    );

    let mut args: Vec<String> = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    args.push(format!("{}x{}", width, height));
    args.extend([
        "-framerate".into(),
        o.fps.to_string(),
        "-i".into(),
        "-".into(),
        "-i".into(),
    ]);
    args.push(o.input.to_string_lossy().into_owned());
    for a in [
        "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
    ] {
        args.push(a.into());
    }
    // erabiltzailearen aukerak azkenean, lehenetsiak gainidazteko
    args.extend(o.ffmpeg_flags.iter().cloned());
    args.extend(o.ffmpeg_args.iter().cloned());
    args.push(o.output.to_string_lossy().into_owned());

    let mut child = Command::new("ffmpeg")
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ts!(
                    "ffmpeg was not found; install it (https://ffmpeg.org) and try again",
                    "ez da ffmpeg aurkitu; instalatu (https://ffmpeg.org) eta saiatu berriz"
                )
                .to_string()
            } else {
                t!("cannot start ffmpeg: {}", "ezin da ffmpeg abiarazi: {}", e)
            }
        })?;

    // idazle-haria: irudiak ffmpeg-i pasatzen dizkio eta bufferrak itzultzen ditu berriz erabiltzeko
    let mut stdin = child.stdin.take().expect("stdin");
    let (to_writer, from_main) = mpsc::sync_channel::<Vec<u8>>(2);
    let (to_main, recycled) = mpsc::channel::<Vec<u8>>();
    let writer = std::thread::spawn(move || -> std::io::Result<()> {
        for buf in from_main {
            stdin.write_all(&buf)?;
            let _ = to_main.send(buf);
        }
        Ok(())
    });

    let progress = ProgressBar::new(frames as u64);
    progress.set_style(
        ProgressStyle::with_template(ts!(
            "{elapsed_precise} {bar:40} {pos}/{len} frames, {per_sec}, {eta} left",
            "{elapsed_precise} {bar:40} {pos}/{len} irudi, {per_sec}, {eta} falta"
        ))
        .expect("template")
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );

    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(per_frame);
    let mut fft_in = fft.make_input_vec();
    let mut fft_out = fft.make_output_vec();
    let bin_hz = rate as f64 / per_frame as f64;
    let perlin = Perlin::new(o.seed);
    let mut bits = BitSource::new(o.seed);
    let (mut prev_rms, mut hue) = (0.0, 0.0);
    let mut sent = 0;
    for i in 0..frames {
        let chunk = &audio[i * per_frame..((i + 1) * per_frame).min(audio.len())];
        fft_in.fill(0.0);
        fft_in[..chunk.len()].copy_from_slice(chunk);
        fft.process(&mut fft_in, &mut fft_out).expect("fft");
        let mags: Vec<f64> = fft_out.iter().map(|c| c.norm() as f64).collect();
        let feats = features(chunk, &mags, bin_hz, prev_rms);
        prev_rms = feats.rms;
        let look = Look::new(&feats, &mags, bits.next(), i, frames, &mut hue);

        let mut buf = recycled
            .try_recv()
            .unwrap_or_else(|_| vec![0u8; width * height * 3]);
        render(
            &mut buf,
            &look,
            &perlin,
            &mags,
            o.direction,
            width,
            height,
            i as u64,
            o.seed,
        );
        // ffmpeg amaitu bada (adib. aukera oker bat), idazle-haria gelditu da: ondoren jakinaraziko da
        if to_writer.send(buf).is_err() {
            break;
        }
        sent += 1;
        progress.inc(1);
    }
    drop(to_writer);
    progress.finish_and_clear();
    let written = writer.join().expect("writer thread");
    let status = child
        .wait()
        .map_err(|e| t!("ffmpeg: {}", "ffmpeg: {}", e))?;
    if !status.success() {
        return Err(t!(
            "ffmpeg failed ({}); its message is above",
            "ffmpeg-ek huts egin du ({}); haren mezua goian dago",
            status
        ));
    }
    if let Err(e) = written {
        return Err(t!(
            "could not send frames to ffmpeg: {}",
            "ezin izan dira irudiak ffmpeg-i bidali: {}",
            e
        ));
    }
    let secs = started.elapsed().as_secs_f64();
    line(
        ts!("time", "denbora"),
        t!(
            "{:.1} s ({:.1} frames/s)",
            "{:.1} s ({:.1} irudi/s)",
            secs,
            sent as f64 / secs
        ),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_sin_is_close_to_sin_everywhere() {
        let mut worst: f64 = 0.0;
        for i in -20000..20000 {
            let x = i as f64 * 0.001;
            worst = worst.max((fast_sin(x) - x.sin()).abs());
        }
        assert!(worst < 0.002, "errore maximoa {}", worst);
    }

    #[test]
    fn small_sin_cos_are_accurate() {
        for i in -3141..=3141 {
            let x = i as f64 * 0.001;
            let (s, c) = sin_cos_small(x);
            assert!(
                (s - x.sin()).abs() < 3e-5 && (c - x.cos()).abs() < 3e-5,
                "{}",
                x
            );
            if x.abs() <= 2.0 {
                assert!(
                    (s - x.sin()).abs() < 3e-8 && (c - x.cos()).abs() < 3e-8,
                    "{}",
                    x
                );
            }
        }
    }

    #[test]
    fn hsv_primaries() {
        let close = |a: [f64; 3], b: [f64; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12);
        assert!(close(hsv_to_rgb(0.0, 1.0, 1.0), [1.0, 0.0, 0.0]));
        assert!(close(hsv_to_rgb(120.0, 1.0, 1.0), [0.0, 1.0, 0.0]));
        assert!(close(hsv_to_rgb(240.0, 1.0, 1.0), [0.0, 0.0, 1.0]));
        assert!(close(hsv_to_rgb(60.0, 1.0, 0.5), [0.5, 0.5, 0.0]));
        assert!(close(hsv_to_rgb(300.0, 0.0, 0.7), [0.7, 0.7, 0.7]));
        assert!(close(hsv_to_rgb(-60.0, 1.0, 1.0), [1.0, 0.0, 1.0]));
        // v-rekiko lineala da: errendatzeak kolorea behin kalkulatzen du eta pixel bakoitzean eskalatzen du
        for h in [10.0, 75.0, 130.0, 200.0, 290.0, 350.0] {
            let one = hsv_to_rgb(h, 0.8, 1.0);
            let part = hsv_to_rgb(h, 0.8, 0.37);
            assert!(close(part, one.map(|c| c * 0.37)));
        }
    }

    #[test]
    fn perlin_is_zero_on_the_lattice_bounded_and_continuous() {
        let p = Perlin::new(3);
        for x in -5..5 {
            for y in -5..5 {
                assert_eq!(p.at(x as f64, y as f64), 0.0);
            }
        }
        let mut seen = (f64::MAX, f64::MIN);
        for i in 0..20000 {
            let (x, y) = (i as f64 * 0.0137 - 100.0, i as f64 * 0.0291 - 200.0);
            let v = p.at(x, y);
            assert!(v.abs() <= 1.0);
            assert!((p.at(x + 1e-6, y) - v).abs() < 1e-4);
            seen = (seen.0.min(v), seen.1.max(v));
        }
        // zarata da, ez konstante bat
        assert!(seen.0 < -0.3 && seen.1 > 0.3, "{:?}", seen);
        // hazi desberdinak, eremu desberdinak
        assert_ne!(Perlin::new(4).at(0.5, 0.5), p.at(0.5, 0.5));
    }

    #[test]
    fn stereo_and_multichannel_mix_to_the_mean() {
        assert_eq!(mix_to_mono(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
        let three = mix_to_mono(&[0.3, 0.3, 0.6, 1.0, 0.0, -1.0], 3);
        assert!(three.len() == 2 && (three[0] - 0.4).abs() < 1e-6 && three[1].abs() < 1e-6);
        assert_eq!(mix_to_mono(&[0.1, 0.2], 1), vec![0.1, 0.2]);
    }

    #[test]
    fn features_of_a_pure_tone() {
        // 1000 Hz-ko sinua, anplitudea 0.5: rms 0.5/√2, zentroidea 1000 Hz inguruan
        let rate = 48000.0;
        let n = 4800;
        let tone: Vec<f32> = (0..n)
            .map(|i| (0.5 * (2.0 * PI * 1000.0 * i as f64 / rate).sin()) as f32)
            .collect();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n);
        let mut input = tone.clone();
        let mut out = fft.make_output_vec();
        fft.process(&mut input, &mut out).unwrap();
        let mags: Vec<f64> = out.iter().map(|c| c.norm() as f64).collect();
        let f = features(&tone, &mags, rate / n as f64, 0.0);
        assert!((f.rms - 0.5 / 2f64.sqrt()).abs() < 1e-4, "{}", f.rms);
        assert!((f.centroid_hz - 1000.0).abs() < 1.0, "{}", f.centroid_hz);
        assert!(f.mid > 0.99 * (f.low + f.mid + f.high));
        assert!((f.transient - 1.0).abs() < 1e-12);
        let quiet = features(&tone, &mags, rate / n as f64, f.rms);
        assert_eq!(quiet.transient, 0.0);
    }

    #[test]
    fn band_energy_edges() {
        let mags = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(band_energy(&mags, 10.0, 0.0, 10.0), 1.0 + 4.0);
        assert_eq!(band_energy(&mags, 10.0, 15.0, 1000.0), 4.0 + 9.0 + 16.0);
        assert_eq!(band_energy(&mags, 10.0, 50.0, 60.0), 0.0);
        assert_eq!(band_energy(&mags, 10.0, 20.0, 20.0), 0.0);
        assert_eq!(band_energy(&[], 10.0, 0.0, 10.0), 0.0);
    }

    #[test]
    fn bits_hold_then_change() {
        let mut src = BitSource::new(5);
        let first: Vec<u64> = (0..HOLD_FRAMES).map(|_| src.next()).collect();
        assert!(first.iter().all(|&b| b == first[0]));
        let later: Vec<u64> = (0..10 * HOLD_FRAMES).map(|_| src.next()).collect();
        assert!(later.iter().any(|&b| b != first[0]));
        // bitak 50/50 dira: 11 multzotan, batekoen batez bestekoa 8 inguruan
        let mut all = vec![first[0]];
        all.extend(later.iter().step_by(HOLD_FRAMES));
        let mean = all.iter().map(|b| b.count_ones() as f64).sum::<f64>() / all.len() as f64;
        assert!((mean - 8.0).abs() < 2.5, "{}", mean);
    }

    fn sample_look() -> Look {
        Look {
            brightness: 1.4,
            rgb: hsv_to_rgb(20.0, 0.8, 1.0),
            density: 7.0,
            offset: 412.345,
            flicker: 0.3,
            interference: 0.4,
            flow: 0.2,
            depth: 0.9,
        }
    }

    fn render_with_threads(threads: usize, width: usize, height: usize) -> Vec<u8> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mags: Vec<f64> = (0..300)
            .map(|i| (i as f64 * 0.37).sin().abs() * 0.4)
            .collect();
        let mut px = vec![0u8; width * height * 3];
        pool.install(|| {
            render(
                &mut px,
                &sample_look(),
                &Perlin::new(9),
                &mags,
                SpectrumDirection::Ltr,
                width,
                height,
                17,
                9,
            )
        });
        px
    }

    #[test]
    fn rendering_does_not_depend_on_thread_count() {
        // 300 errenkada: 5 banda, bi faseak eta azken banda erdia probatzen dira
        let one = render_with_threads(1, 320, 300);
        let four = render_with_threads(4, 320, 300);
        assert!(one == four);
        assert!(one.iter().any(|&b| b > 0));
    }

    // erreferentzia: banda eta leihorik gabe, pixel bakoitza ordenan (bikoitiak, gero bakoitiak)
    fn render_reference(
        look: &Look,
        perlin: &Perlin,
        width: usize,
        height: usize,
        frame: u64,
        seed: u64,
    ) -> Vec<u8> {
        let mut px = vec![0u8; width * height * 3];
        let cols = axis(width as u32, look, 0.0);
        let rows = axis(height as u32, look, FRAC_PI_2);
        let bands = height.div_ceil(BAND);
        for phase in 0..2 {
            for k in (phase..bands).step_by(2) {
                let w = Window {
                    first_row: 0,
                    rows: k * BAND..((k + 1) * BAND).min(height),
                    buf: &mut px,
                };
                render_band(w, look, perlin, &cols, &rows, width, frame, seed);
            }
        }
        px
    }

    #[test]
    fn windows_lose_no_pixel_writes() {
        // irudi osoa leiho bakarra balitz bezala errendatu eta konparatu: ertzek ez dute idazketarik galtzen
        let (w, h) = (200, 333);
        let look = sample_look();
        let perlin = Perlin::new(2);
        let reference = render_reference(&look, &perlin, w, h, 3, 4);
        let mut px = vec![0u8; w * h * 3];
        for phase in 0..2 {
            for win in windows(&mut px, w * 3, h, phase) {
                render_band(
                    win,
                    &look,
                    &perlin,
                    &axis(w as u32, &look, 0.0),
                    &axis(h as u32, &look, FRAC_PI_2),
                    w,
                    3,
                    4,
                );
            }
        }
        assert!(px == reference);
    }

    #[test]
    fn spectrum_bars_cover_the_right_pixels() {
        let (w, h) = (100, 40);
        // 4 bin, 25 pixeleko barrak; altuerak 500 * m, gehienez h / 4 = 10
        let mags = [0.004, 0.0, 1.0, 0.012];
        assert_eq!(
            bar_heights(&mags, w, h)[..],
            [vec![2; 25], vec![0; 25], vec![10; 25], vec![6; 25]].concat()[..]
        );
        let mut px = vec![0u8; w * h * 3];
        render_spectrum(&mut px, &mags, SpectrumDirection::Ltr, w, h);
        let red = |x: usize, y: usize| px[(y * w + x) * 3];
        assert_eq!(red(0, h - 1), 204);
        assert_eq!(red(0, h - 2), 204);
        assert_eq!(red(0, h - 3), 0);
        assert_eq!(red(30, h - 1), 0);
        assert_eq!(red(60, h - 10), 204);
        let mut px = vec![0u8; w * h * 3];
        render_spectrum(&mut px, &mags, SpectrumDirection::Rtl, w, h);
        assert_eq!(px[(w - 1) * 3], 204);
        assert_eq!(px[(2 * w - 1) * 3], 204);
        assert_eq!(px[(2 * w + w - 1) * 3], 0);
        // zutabeak bin-ak baino gutxiago direnean ere barrak ikusten dira
        let many: Vec<f64> = (0..1000)
            .map(|i| if i == 500 { 1.0 } else { 0.0 })
            .collect();
        let hs = bar_heights(&many, 100, 40);
        assert_eq!(hs[50], 10);
        assert_eq!(hs.iter().filter(|&&x| x > 0).count(), 1);
    }

    // errendatze-abiadura bakarrik (ffmpeg gabe): cargo test --release --bin qoa -- --ignored --nocapture
    #[test]
    #[ignore]
    fn render_speed_1080p() {
        let (w, h) = (1920, 1080);
        let look = sample_look();
        let perlin = Perlin::new(1);
        let mags: Vec<f64> = (0..368)
            .map(|i| (i as f64 * 0.37).sin().abs() * 0.4)
            .collect();
        let mut px = vec![0u8; w * h * 3];
        render(
            &mut px,
            &look,
            &perlin,
            &mags,
            SpectrumDirection::Ltr,
            w,
            h,
            0,
            1,
        );
        let n = 60;
        let start = Instant::now();
        for f in 0..n {
            render(
                &mut px,
                &look,
                &perlin,
                &mags,
                SpectrumDirection::Ltr,
                w,
                h,
                f,
                1,
            );
        }
        let per = start.elapsed().as_secs_f64() / n as f64;
        println!(
            "1080p: {:.2} ms/irudi, {:.1} irudi/s, {:.1} ns/pixel (hari guztiak)",
            per * 1e3,
            1.0 / per,
            per * 1e9 / (w * h) as f64
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let start = Instant::now();
        pool.install(|| {
            render(
                &mut px,
                &look,
                &perlin,
                &mags,
                SpectrumDirection::Ltr,
                w,
                h,
                0,
                1,
            )
        });
        let one = start.elapsed().as_secs_f64();
        println!(
            "1080p hari batean: {:.2} ms/irudi, {:.1} ns/pixel",
            one * 1e3,
            one * 1e9 / (w * h) as f64
        );
    }
}
