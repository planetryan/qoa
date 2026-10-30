// fitxategi honek `qoa visual` egiten du: audio bat (ffmpeg-ek irakur dezakeen edozein fitxategi, edo lagin
// gordinak fitxategi edo stdin batetik) argi gisa garraiatzen du zirkuitu optiko kuantiko simulatu batetik
// (src/light.rs) eta emaitza audio, bideo edo biak gisa idazten du ffmpeg-ekin:
//
//   1. deskodetu: ffmpeg-ek laginak 64 biteko koma higikorrean ematen ditu (pcm_f64le), kanal guztiak
//   2. argia: kanal bakoitza mdct moduetan, zirkuitua eta detekzioa zehazki, eta berriz seinale (src/light.rs)
//   3. bideoa: irudi bakoitzeko audio prozesatuaren ezaugarriak (fft): ozentasuna, zentroidea, banden
//      energia, transienteak...; parametro bisual bihurtu (`Look`) eta errendatu: perlin zarata +
//      interferentzia + keinuak + desplazamendua, eta espektroa barra gisa. pixel gordinak zuzenean ffmpeg-en
//      sarrerara. irudiak arte dira, ez fisika: 32 biteko kalkulua nahikoa da haietarako
//   4. fase-espazioa: irudi bakoitzean, une horretako modu distiratsuenen egoera kuantikoa detekzioaren
//      aurretik, haien wigner funtzio zehatza (src/light.rs, `Light::phase_space`), eta modu bakoitzean
//      neurtutakoa (homodinoaren x-a edo fotoi-kopuruaren eraztuna). fisika da: 64 bitekoa
//   5. irteera: bideoa eta audio prozesatua, audioa bakarrik, bideoa bakarrik, edo bideoa jatorrizko audioarekin
//
// ezaugarriak ordenan kalkulatzen dira (aurreko irudiaren araberakoak dira) eta errendatzea paraleloa da
// banda horizontaletan, ordena finko batean; beraz hazi berak emaitza bera ematen du edozein hari-kopururekin.

use indicatif::{ProgressBar, ProgressStyle};
use qoa::light::{self, KerrSampler, Light, Mdct, Panel};
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

// zer idazten den
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    // bideoa eta audio prozesatua
    Both,
    // audio prozesatua bakarrik
    AudioOnly,
    // bideoa bakarrik, audiorik gabe
    VideoOnly,
    // bideo prozesatua jatorrizko audioarekin
    KeepAudio,
}

// fase-espazioko panelak: lerro bat artearen gainean, irudi osoa, edo ezer ez
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseSpace {
    Row,
    Full,
    Off,
}

impl PhaseSpace {
    // modu-kopurua irudi bakoitzeko
    fn picks(self) -> usize {
        match self {
            PhaseSpace::Row => 4,
            PhaseSpace::Full => 8,
            PhaseSpace::Off => 0,
        }
    }
}

// lagin gordinak: ffmpeg-en lagin-formatua (u8, s16le, f32le, ...), maiztasuna eta kanalak
#[derive(Debug, Clone)]
pub struct Raw {
    pub format: String,
    pub rate: u32,
    pub channels: u32,
}

pub struct Options<'a> {
    // fitxategia, edo `-` stdin-erako
    pub input: &'a Path,
    pub output: &'a Path,
    pub raw: Option<Raw>,
    pub mode: Mode,
    pub light: Light,
    // mdct-ren tamaina: modu-kopurua irudi eta kanal bakoitzeko
    pub block: usize,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub phase_space: PhaseSpace,
    pub direction: SpectrumDirection,
    pub seed: u64,
    pub ffmpeg_flags: &'a [String],
    pub ffmpeg_args: &'a [String],
}

// --- audioa ---

pub struct Audio {
    // kanal bakoitzaren laginak, [-1, 1] eskalan
    pub channels: Vec<Vec<f64>>,
    pub rate: u32,
}

fn ffmpeg_missing(tool: &str, e: std::io::Error) -> String {
    if e.kind() == std::io::ErrorKind::NotFound {
        t!(
            "{} was not found; install ffmpeg (https://ffmpeg.org) and try again",
            "ez da {} aurkitu; instalatu ffmpeg (https://ffmpeg.org) eta saiatu berriz",
            tool
        )
    } else {
        t!("cannot start {}: {}", "ezin da {} abiarazi: {}", tool, e)
    }
}

// aldi baterako fitxategi bat, erabili ondoren ezabatzen dena
struct TempFile(std::path::PathBuf);

impl TempFile {
    fn new(tag: &str) -> TempFile {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        TempFile(std::env::temp_dir().join(format!("qoa-{}-{}-{}", std::process::id(), nanos, tag)))
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// ffprobe: lehen audio-korrontearen lagin-maiztasuna eta kanal-kopurua
fn probe(path: &Path) -> Result<(u32, u32), String> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .map_err(|e| ffmpeg_missing("ffprobe", e))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mut parts = text.trim().split(',');
    let rate = parts.next().and_then(|v| v.trim().parse::<u32>().ok());
    let channels = parts.next().and_then(|v| v.trim().parse::<u32>().ok());
    match (out.status.success(), rate, channels) {
        (true, Some(r), Some(c)) if r > 0 && c > 0 => Ok((r, c)),
        _ => Err(t!(
            "'{}' has no audio that ffmpeg can read{}",
            "'{}'-k ez du ffmpeg-ek irakur dezakeen audiorik{}",
            path.display(),
            {
                let e = String::from_utf8_lossy(&out.stderr);
                if e.trim().is_empty() {
                    String::new()
                } else {
                    format!(" ({})", e.trim())
                }
            }
        )),
    }
}

// audioa deskodetu 64 biteko laginetara. fitxategi batetik edo stdin-etik (`-`); gordina bada, `raw`-ek dio
// nola irakurri
pub fn decode_audio(input: &Path, raw: Option<&Raw>) -> Result<Audio, String> {
    let stdin = input.as_os_str() == "-";
    // edukiontzi bat stdin-etik: lehenik fitxategi batera, ffprobe-k aztertu ahal izateko
    let mut kept: Option<TempFile> = None;
    let source: std::path::PathBuf = if stdin && raw.is_none() {
        let tmp = TempFile::new("input");
        let mut f = std::fs::File::create(&tmp.0)
            .map_err(|e| t!("cannot buffer stdin: {}", "ezin da stdin gorde: {}", e))?;
        std::io::copy(&mut std::io::stdin().lock(), &mut f)
            .map_err(|e| t!("cannot read stdin: {}", "ezin da stdin irakurri: {}", e))?;
        let p = tmp.0.clone();
        kept = Some(tmp);
        p
    } else {
        if !stdin && !input.exists() {
            return Err(t!(
                "cannot read '{}': no such file",
                "ezin da '{}' irakurri: ez dago fitxategi hori",
                input.display()
            ));
        }
        input.to_path_buf()
    };
    let (rate, channels) = match raw {
        Some(r) => (r.rate, r.channels),
        None => probe(&source)?,
    };
    let mut args: Vec<String> = ["-hide_banner", "-loglevel", "error", "-nostdin"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    if let Some(r) = raw {
        // stdin gordina zuzenean ffmpeg-era
        if stdin {
            args.retain(|a| a != "-nostdin");
        }
        args.extend([
            "-f".into(),
            r.format.clone(),
            "-ar".into(),
            r.rate.to_string(),
            "-ac".into(),
            r.channels.to_string(),
        ]);
    }
    args.extend(["-i".into(), source.to_string_lossy().into_owned()]);
    args.extend(
        ["-map", "0:a:0", "-f", "f64le", "-c:a", "pcm_f64le", "-"]
            .iter()
            .map(|s| s.to_string()),
    );
    let out = Command::new("ffmpeg")
        .args(&args)
        .stdin(if stdin && raw.is_some() {
            Stdio::inherit()
        } else {
            Stdio::null()
        })
        .output()
        .map_err(|e| ffmpeg_missing("ffmpeg", e))?;
    drop(kept);
    if !out.status.success() {
        return Err(t!(
            "ffmpeg could not decode '{}': {}",
            "ffmpeg-ek ezin izan du '{}' deskodetu: {}",
            input.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let samples: Vec<f64> = out
        .stdout
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    let c = channels as usize;
    let mut chans = vec![Vec::with_capacity(samples.len() / c); c];
    for frame in samples.chunks_exact(c) {
        for (k, v) in frame.iter().enumerate() {
            chans[k].push(*v);
        }
    }
    Ok(Audio {
        channels: chans,
        rate,
    })
}

// kanal guztien batez bestekoa (bideoaren ezaugarrietarako, 32 bitean)
pub fn mono(channels: &[Vec<f64>]) -> Vec<f32> {
    let n = channels.first().map_or(0, |c| c.len());
    let inv = 1.0 / channels.len().max(1) as f64;
    (0..n)
        .map(|i| (channels.iter().map(|c| c[i]).sum::<f64>() * inv) as f32)
        .collect()
}

// kanalak tartekatuta, f64 little-endian byteetan (ffmpeg-en f64le)
fn interleaved_bytes(channels: &[Vec<f64>]) -> Vec<u8> {
    let n = channels.first().map_or(0, |c| c.len());
    let mut out = Vec::with_capacity(n * channels.len() * 8);
    for i in 0..n {
        for c in channels {
            out.extend_from_slice(&c[i].to_le_bytes());
        }
    }
    out
}

// fase-espazioan erakusten den modu bat: kanala, bin-a, eta koefizientea detekzioaren aurretik (sarearen
// ondoren) eta ondoren
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probe {
    pub channel: usize,
    pub bin: usize,
    pub before: f64,
    pub after: f64,
}

// bloke bateko `picks` modu distiratsuenak: lehenik gailurrak (|c| bere bi auzokideak baino handiagoa,
// espektroko lerro desberdinak), gero gainerakoak |c|-ren arabera (isiltasunean, hutsa)
fn brightest(frames: &[&[f64]], picks: usize) -> Vec<(usize, usize)> {
    let mut peaks = Vec::new();
    let mut rest = Vec::new();
    for (ch, f) in frames.iter().enumerate() {
        for k in 0..f.len() {
            let a = f[k].abs();
            let left = if k > 0 { f[k - 1].abs() } else { 0.0 };
            let right = if k + 1 < f.len() { f[k + 1].abs() } else { 0.0 };
            if a > 0.0 && a >= left && a > right {
                peaks.push((a, ch, k));
            } else {
                rest.push((a, ch, k));
            }
        }
    }
    let order = |x: &(f64, usize, usize), y: &(f64, usize, usize)| {
        y.0.total_cmp(&x.0).then((x.1, x.2).cmp(&(y.1, y.2)))
    };
    peaks.sort_by(order);
    rest.sort_by(order);
    peaks
        .into_iter()
        .chain(rest)
        .take(picks)
        .map(|(_, ch, k)| (ch, k))
        .collect()
}

// argia kanal guztietan: mdct, zirkuitua, alderantzizkoa. `picks` > 0 bada, bloke bakoitzeko modu
// distiratsuenak ere bai (fase-espaziorako)
pub fn apply_light(
    audio: &Audio,
    light: &Light,
    block: usize,
    seed: u64,
    picks: usize,
) -> (Vec<Vec<f64>>, light::Summary, Vec<Vec<Probe>>) {
    let mdct = Mdct::new(block);
    let mut frames: Vec<Vec<Vec<f64>>> = audio
        .channels
        .iter()
        .map(|c| light::analyze(&mdct, c))
        .collect();
    let mut probes: Vec<Vec<Probe>> = Vec::new();
    if picks > 0 {
        let mut before = frames.clone();
        light.apply_mesh(&mut before);
        let blocks = before.first().map_or(0, |c| c.len());
        probes = (0..blocks)
            .into_par_iter()
            .map(|f| {
                let cols: Vec<&[f64]> = before.iter().map(|c| &c[f][..]).collect();
                brightest(&cols, picks)
                    .into_iter()
                    .map(|(channel, bin)| Probe {
                        channel,
                        bin,
                        before: before[channel][f][bin],
                        after: 0.0,
                    })
                    .collect()
            })
            .collect();
    }
    let summary = light.process(&mut frames, seed);
    for (f, list) in probes.iter_mut().enumerate() {
        for p in list.iter_mut() {
            p.after = frames[p.channel][f][p.bin];
        }
    }
    let out = frames
        .iter()
        .zip(&audio.channels)
        .map(|(f, c)| light::synthesize(&mdct, f, c.len()))
        .collect();
    (out, summary, probes)
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

// --- fase-espazioa ---

// panelen kokapena (goiko ezkerreko izkinak) eta tamaina; None txikiegiak badira
fn panel_layout(
    mode: PhaseSpace,
    width: usize,
    height: usize,
) -> Option<(Vec<(usize, usize)>, usize)> {
    let (cells, size) = match mode {
        PhaseSpace::Off => return None,
        PhaseSpace::Row => {
            let size = (height as f64 * 0.22).round() as usize;
            let gap = (size / 8).max(4);
            let count = 4.min(width.saturating_sub(gap) / (size + gap).max(1));
            (
                (0..count)
                    .map(|k| (gap + k * (size + gap), gap))
                    .collect::<Vec<_>>(),
                size,
            )
        }
        PhaseSpace::Full => {
            let (cols, rows) = if width >= height { (4, 2) } else { (2, 4) };
            let gap = (width.min(height) / 40).max(4);
            let size = ((width.saturating_sub((cols + 1) * gap)) / cols)
                .min(height.saturating_sub((rows + 1) * gap) / rows);
            let x0 = (width - (cols * size + (cols - 1) * gap)) / 2;
            let y0 = (height - (rows * size + (rows - 1) * gap)) / 2;
            (
                (0..rows)
                    .flat_map(|r| {
                        (0..cols).map(move |c| (x0 + c * (size + gap), y0 + r * (size + gap)))
                    })
                    .collect(),
                size,
            )
        }
    };
    (size >= 24 && !cells.is_empty()).then_some((cells, size))
}

// wigner-en kolorea, v = W / max|W| ∈ [-1, 1]: positiboa beltzetik laranjara eta zurira, negatiboa (egoera
// ez-klasikoa) beltzetik urdinera eta ziana-zurira
fn wigner_color(v: f64) -> [f64; 3] {
    let t = v.abs().min(1.0).powf(0.7);
    let (mid, top) = if v >= 0.0 {
        ([1.0, 0.42, 0.08], [1.0, 0.96, 0.85])
    } else {
        ([0.1, 0.4, 1.0], [0.75, 0.95, 1.0])
    };
    let mut out = [0.0; 3];
    for c in 0..3 {
        out[c] = if t < 0.6 {
            mid[c] * t / 0.6
        } else {
            mid[c] + (top[c] - mid[c]) * (t - 0.6) / 0.4
        };
    }
    out
}

// W pixel batean: panela pantailaren tamainan kalkulatuta badago, zuzenean; bestela interpolazio bilineala
fn panel_value(panel: &Panel, size: usize, px: usize, py: usize) -> f64 {
    let n = panel.n;
    if n == size {
        return panel.w[py * n + px];
    }
    let f = n as f64 / size as f64;
    let (x, y) = ((px as f64 + 0.5) * f - 0.5, (py as f64 + 0.5) * f - 0.5);
    let (x0, y0) = (x.floor().max(0.0), y.floor().max(0.0));
    let (ix, iy) = ((x0 as usize).min(n - 1), (y0 as usize).min(n - 1));
    let (jx, jy) = ((ix + 1).min(n - 1), (iy + 1).min(n - 1));
    let (fx, fy) = ((x - x0).clamp(0.0, 1.0), (y - y0).clamp(0.0, 1.0));
    let row = |r: usize| panel.w[r * n + ix] * (1.0 - fx) + panel.w[r * n + jx] * fx;
    row(iy) * (1.0 - fy) + row(jy) * fy
}

// panelak irudian: wigner funtzioa, hutsaren eraztuna eskala gisa (1σ, beheko ezkerrean) eta neurtutakoa
// (homodinoa: x-ren marra; batez bestekoa: marra etena; zenbaketa: √(2n+1) erradioko eraztuna)
#[allow(clippy::too_many_arguments)]
fn draw_panels(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    mode: PhaseSpace,
    cells: &[(usize, usize)],
    size: usize,
    panels: &[(Panel, Probe)],
    light: &Light,
) {
    if mode == PhaseSpace::Full {
        pixels
            .par_iter_mut()
            .for_each(|v| *v = (*v as f32 * 0.35) as u8);
    }
    let scale = light.amplitude_scale();
    for ((x0, y0), (panel, probe)) in cells.iter().zip(panels) {
        let top = panel.w.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let h = 2.0 * panel.half / size as f64;
        let left = panel.center.0 - panel.half;
        let upper = panel.center.1 + panel.half;
        let after = probe.after * scale;
        let line_x =
            (light.detect != light::Detect::Count).then(|| (2f64.sqrt() * after - left) / h);
        let ring =
            (light.detect == light::Detect::Count).then(|| (2.0 * after * after + 1.0).sqrt());
        let vac = std::f64::consts::FRAC_1_SQRT_2 / h;
        let pad = (size / 24) as f64;
        let vac_c = (pad + vac.max(1.5), size as f64 - pad - vac.max(1.5));
        let border = (size / 240).max(1);
        let dashed = light.detect == light::Detect::Mean;
        pixels[y0 * width * 3..(y0 + size).min(height) * width * 3]
            .par_chunks_mut(width * 3)
            .enumerate()
            .for_each(|(py, row)| {
                for px in 0..size.min(width - x0) {
                    let o = (x0 + px) * 3;
                    let w = panel_value(panel, size, px, py);
                    let c = wigner_color(if top > 0.0 { w / top } else { 0.0 });
                    let mut rgb = [0.0; 3];
                    for k in 0..3 {
                        rgb[k] = row[o + k] as f64 / 255.0 * 0.18 + c[k];
                    }
                    let mut over = |color: [f64; 3], a: f64| {
                        for k in 0..3 {
                            rgb[k] = rgb[k] * (1.0 - a) + color[k] * a;
                        }
                    };
                    let (fx, fy) = (px as f64 + 0.5, py as f64 + 0.5);
                    // neurtutakoa
                    if let Some(lx) = line_x {
                        let a = (1.0 - (fx - lx).abs() / (border as f64 + 0.5)).max(0.0);
                        if a > 0.0 && (!dashed || (py / 6) % 2 == 0) {
                            over([0.45, 1.0, 0.55], 0.85 * a);
                        }
                    }
                    if let Some(radius) = ring {
                        let (x, p) = (left + fx * h, upper - fy * h);
                        let a = (1.0
                            - ((x * x + p * p).sqrt() - radius).abs()
                                / (h * (border as f64 + 0.5)))
                            .max(0.0);
                        if a > 0.0 {
                            over([0.45, 1.0, 0.55], 0.85 * a);
                        }
                    }
                    // hutsaren eraztuna
                    let dv = ((fx - vac_c.0).powi(2) + (fy - vac_c.1).powi(2)).sqrt();
                    let a = if vac >= 1.5 {
                        (1.0 - (dv - vac).abs() / 0.75).max(0.0)
                    } else {
                        (1.0 - (dv - 1.0).max(0.0)).max(0.0)
                    };
                    if a > 0.0 {
                        over([0.85, 0.85, 0.85], 0.8 * a);
                    }
                    // ertza
                    if px < border || py < border || px + border >= size || py + border >= size {
                        over([0.6, 0.6, 0.6], 0.6);
                    }
                    for k in 0..3 {
                        row[o + k] = (rgb[k].clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                }
            });
    }
}

// --- bideoa ---

fn label_detect(d: light::Detect) -> String {
    match d {
        light::Detect::Mean => t!(
            "exact mean field (no quantum noise: the average of infinitely many shots)",
            "batez besteko eremu zehatza (zarata kuantikorik gabe: shot infinituren batez bestekoa)"
        ),
        light::Detect::Homodyne => t!(
            "homodyne, one shot per mode (real vacuum noise)",
            "homodinoa, shot bat modu bakoitzeko (hutsaren benetako zarata)"
        ),
        light::Detect::Count => t!(
            "photon counting, one shot per mode (exact Poisson statistics; the sign comes from the mean field)",
            "fotoi-zenbaketa, shot bat modu bakoitzeko (poisson estatistika zehatza; ikurra batez besteko eremutik)"
        ),
    }
}

// audio-kodeketa lehenetsia irteerako luzapenaren arabera (erabiltzailearen aukerek gainidazten dute)
fn audio_codec(output: &Path, video: bool) -> Vec<&'static str> {
    let ext = output
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match (ext.as_str(), video) {
        ("wav", _) => vec!["-c:a", "pcm_s24le"],
        ("flac", _) | ("mkv", true) => vec!["-c:a", "flac", "-sample_fmt", "s32"],
        ("webm", true) => vec!["-c:a", "libopus", "-b:a", "256k"],
        (_, true) => vec!["-c:a", "aac", "-b:a", "320k"],
        _ => vec![],
    }
}

// sarrera irakurri, argitik pasatu eta irteera idatzi (audioa, bideoa edo biak)
pub fn run(o: &Options) -> Result<(), String> {
    let started = Instant::now();
    let (width, height) = (o.width as usize, o.height as usize);
    let video = o.mode != Mode::AudioOnly;
    if video && (width == 0 || height == 0 || o.fps == 0) {
        return Err(ts!(
            "width, height and fps must be above zero",
            "zabalerak, altuerak eta fps-k zero baino handiagoak izan behar dute"
        )
        .into());
    }
    if o.block < 16 || !o.block.is_multiple_of(2) {
        return Err(t!(
            "--block must be an even number of 16 or more, not {}",
            "--block 16 edo gehiagoko zenbaki bikoitia izan behar da, ez {}",
            o.block
        ));
    }
    o.light.check()?;
    let audio = decode_audio(o.input, o.raw.as_ref())?;
    let len = audio.channels.first().map_or(0, |c| c.len());
    if len == 0 {
        return Err(t!(
            "'{}' has no audio samples",
            "'{}'-k ez du audio-laginik",
            o.input.display()
        ));
    }
    let rate = audio.rate;

    let style = Style::detect();
    let line = |label: &str, text: String| println!("{}  {}", style.label(label), text);
    line(
        ts!("input", "sarrera"),
        t!(
            "{}: {:.2} s, {} Hz, {} channel(s), decoded to 64-bit floats",
            "{}: {:.2} s, {} Hz, {} kanal, 64 biteko koma higikorrera deskodetua",
            o.input.display(),
            len as f64 / rate as f64,
            rate,
            audio.channels.len()
        ),
    );

    // argia: bideoak beti audio prozesatua jarraitzen du; audioa prozesatua da --keep-audio gabe
    let picks = if video { o.phase_space.picks() } else { 0 };
    let (processed, summary, probes) = apply_light(&audio, &o.light, o.block, o.seed, picks);
    let mut effects = Vec::new();
    if o.light.mix != 0.0 {
        effects.push(t!(
            "beamsplitter mesh {} rad",
            "habe-zatitzaile sarea {} rad",
            o.light.mix
        ));
    }
    if o.light.loss > 0.0 {
        effects.push(t!("loss {}", "galera {}", o.light.loss));
    }
    if o.light.squeeze_db > 0.0 {
        effects.push(t!("squeezed {} dB", "estutua {} dB", o.light.squeeze_db));
    }
    if o.light.kerr != 0.0 {
        effects.push(t!(
            "Kerr {} rad at the brightest mode (κ = {:.3e})",
            "kerr {} rad modu distiratsuenean (κ = {:.3e})",
            o.light.kerr,
            summary.kappa
        ));
    }
    line(
        ts!("light", "argia"),
        t!(
            "{} modes ({} per block and channel), {:.3e} photons in all, {:.3e} in the brightest mode; {} photons per sample at full scale",
            "{} modu ({} bloke eta kanal bakoitzeko), {:.3e} fotoi guztira, {:.3e} modu distiratsuenean; {} fotoi lagin bakoitzeko eskala osoan",
            summary.modes,
            o.block,
            summary.photons,
            summary.brightest,
            o.light.photons
        ),
    );
    line(
        ts!("circuit", "zirkuitua"),
        if effects.is_empty() {
            ts!("no optical effects", "efektu optikorik gabe").to_string()
        } else {
            effects.join(", ")
        },
    );
    line(ts!("detection", "detekzioa"), label_detect(o.light.detect));
    if picks > 0 {
        line(
            ts!("phase space", "fase-espazioa"),
            t!(
                "the exact Wigner function of the {} brightest modes in every frame, before detection, with what was measured",
                "irudi bakoitzean {} modu distiratsuenen wigner funtzio zehatza, detekzioaren aurretik, neurtutakoarekin",
                picks
            ),
        );
    }
    if o.light.detect == light::Detect::Mean && o.light.squeeze_db > 0.0 && o.light.kerr == 0.0 {
        line(
            ts!("note", "oharra"),
            ts!(
                "squeezing changes only the quantum noise, which --detect mean averages away: here it has no effect",
                "estutzeak zarata kuantikoa bakarrik aldatzen du, eta --detect mean-ek hori batez bestekoan ezabatzen du: hemen ez du eraginik"
            )
            .to_string(),
        );
    }

    let audio_out: &[Vec<f64>] = match o.mode {
        Mode::KeepAudio => &audio.channels,
        _ => &processed,
    };
    let frames = len.div_ceil((rate / o.fps.max(1)).max(1) as usize);
    let what = match o.mode {
        Mode::Both => t!(
            "{}x{} video with the processed audio",
            "{}x{} bideoa audio prozesatuarekin",
            width,
            height
        ),
        Mode::AudioOnly => ts!("processed audio only", "audio prozesatua bakarrik").to_string(),
        Mode::VideoOnly => t!(
            "{}x{} video, no audio",
            "{}x{} bideoa, audiorik gabe",
            width,
            height
        ),
        Mode::KeepAudio => t!(
            "{}x{} video with the original audio",
            "{}x{} bideoa jatorrizko audioarekin",
            width,
            height
        ),
    };
    line(
        ts!("output", "irteera"),
        if video {
            t!(
                "{}: {}, {} frames/s, {} frames",
                "{}: {}, {} irudi/s, {} irudi",
                o.output.display(),
                what,
                o.fps,
                frames
            )
        } else {
            format!("{}: {}", o.output.display(), what)
        },
    );
    line(
        ts!("seed", "hazia"),
        t!(
            "{} (same seed and input, same output)",
            "{} (hazi eta sarrera berak, irteera bera)",
            o.seed
        ),
    );

    // ffmpeg-en argumentuak
    let mut args: Vec<String> = ["-hide_banner", "-loglevel", "error", "-y"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let pcm_input = |args: &mut Vec<String>, src: &str| {
        args.extend([
            "-f".into(),
            "f64le".into(),
            "-ar".into(),
            rate.to_string(),
            "-ac".into(),
            audio.channels.len().to_string(),
            "-i".into(),
            src.into(),
        ]);
    };
    // bideoarekin, audioa aldi baterako fitxategi batetik (stdin irudientzat da)
    let audio_file = TempFile::new("audio.f64");
    if video {
        args.extend([
            "-f".into(),
            "rawvideo".into(),
            "-pix_fmt".into(),
            "rgb24".into(),
            "-s".into(),
        ]);
        args.push(format!("{}x{}", width, height));
        args.extend([
            "-framerate".into(),
            o.fps.to_string(),
            "-i".into(),
            "-".into(),
        ]);
        if o.mode != Mode::VideoOnly {
            std::fs::write(&audio_file.0, interleaved_bytes(audio_out)).map_err(|e| {
                t!(
                    "cannot write a temporary file: {}",
                    "ezin da aldi baterako fitxategi bat idatzi: {}",
                    e
                )
            })?;
            pcm_input(&mut args, &audio_file.0.to_string_lossy());
            args.extend(
                ["-map", "0:v:0", "-map", "1:a:0"]
                    .iter()
                    .map(|s| s.to_string()),
            );
            args.extend(audio_codec(o.output, true).iter().map(|s| s.to_string()));
        } else {
            args.push("-an".into());
        }
        args.extend(
            [
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            ]
            .iter()
            .map(|s| s.to_string()),
        );
    } else {
        pcm_input(&mut args, "-");
        args.extend(audio_codec(o.output, false).iter().map(|s| s.to_string()));
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
        .map_err(|e| ffmpeg_missing("ffmpeg", e))?;
    let mut stdin = child.stdin.take().expect("ffmpeg-en sarrera");

    let mut sent = 0usize;
    let written: std::io::Result<()> = if !video {
        // sarrera itxi: ffmpeg-ek amaiera ikusi arte itxaroten du
        let r = stdin.write_all(&interleaved_bytes(audio_out));
        drop(stdin);
        r
    } else {
        // idazle-haria: irudiak ffmpeg-i pasatzen dizkio eta bufferrak itzultzen ditu berriz erabiltzeko
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
            .expect("aurrerapen-barraren txantiloia")
            .progress_chars("█▉▊▋▌▍▎▏ "),
        );
        // bideoak argiak prozesatutako seinalea jarraitzen du
        let signal = mono(&processed);
        let per_frame = (rate / o.fps) as usize;
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(per_frame.max(1));
        let mut fft_in = fft.make_input_vec();
        let mut fft_out = fft.make_output_vec();
        let bin_hz = rate as f64 / per_frame.max(1) as f64;
        let perlin = Perlin::new(o.seed);
        let mut bits = BitSource::new(o.seed);
        let (mut prev_rms, mut hue) = (0.0, 0.0);
        // fase-espazioa: panelak bloke bakoitzeko behin (irudi jarraituek bloke bera izan dezakete)
        let layout = panel_layout(o.phase_space, width, height);
        let sampler = (summary.kappa != 0.0).then(|| KerrSampler::new(summary.kappa));
        let mut shown: Option<(usize, Vec<(Panel, Probe)>)> = None;
        for i in 0..frames {
            let chunk = &signal[(i * per_frame).min(len)..((i + 1) * per_frame).min(len)];
            fft_in.fill(0.0);
            fft_in[..chunk.len()].copy_from_slice(chunk);
            fft.process(&mut fft_in, &mut fft_out)
                .expect("fft-aren tamainak");
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
            if let Some((cells, size)) = &layout {
                // irudiaren erdiko laginaren blokea (f blokea f·N laginean dago zentratuta)
                let at = (i as f64 + 0.5) / o.fps as f64 * rate as f64;
                let f =
                    ((at / o.block as f64).round() as usize).min(probes.len().saturating_sub(1));
                if shown.as_ref().is_none_or(|s| s.0 != f) && !probes.is_empty() {
                    let n = (*size).min(720);
                    let scale = o.light.amplitude_scale();
                    let panels = probes[f]
                        .par_iter()
                        .take(cells.len())
                        .map(|p| {
                            (
                                o.light.phase_space(p.before * scale, sampler.as_ref(), n),
                                *p,
                            )
                        })
                        .collect();
                    shown = Some((f, panels));
                }
                if let Some((_, panels)) = &shown {
                    draw_panels(
                        &mut buf,
                        width,
                        height,
                        o.phase_space,
                        cells,
                        *size,
                        panels,
                        &o.light,
                    );
                }
            }
            // ffmpeg amaitu bada (adib. aukera oker bat), idazle-haria gelditu da: ondoren jakinaraziko da
            if to_writer.send(buf).is_err() {
                break;
            }
            sent += 1;
            progress.inc(1);
        }
        drop(to_writer);
        progress.finish_and_clear();
        writer.join().expect("idazle-haria")
    };
    let status = child
        .wait()
        .map_err(|e| t!("ffmpeg: {}", "ffmpeg: {}", e))?;
    drop(audio_file);
    if !status.success() {
        return Err(t!(
            "ffmpeg failed ({}); its message is above",
            "ffmpeg-ek huts egin du ({}); haren mezua goian dago",
            status
        ));
    }
    if let Err(e) = written {
        return Err(t!(
            "could not send data to ffmpeg: {}",
            "ezin izan dira datuak ffmpeg-i bidali: {}",
            e
        ));
    }
    let secs = started.elapsed().as_secs_f64();
    line(
        ts!("time", "denbora"),
        if video {
            t!(
                "{:.1} s ({:.1} frames/s)",
                "{:.1} s ({:.1} irudi/s)",
                secs,
                sent as f64 / secs
            )
        } else {
            t!("{:.1} s", "{:.1} s", secs)
        },
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
        assert_eq!(mono(&[vec![1.0, 0.5], vec![0.0, 0.5]]), vec![0.5, 0.5]);
        let three = mono(&[vec![0.3, 1.0], vec![0.3, 0.0], vec![0.6, -1.0]]);
        assert!(three.len() == 2 && (three[0] - 0.4).abs() < 1e-6 && three[1].abs() < 1e-6);
        assert_eq!(mono(&[vec![0.1, 0.2]]), vec![0.1, 0.2]);
        // tartekatuta: l0 r0 l1 r1
        let bytes = interleaved_bytes(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let back: Vec<f64> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect();
        assert_eq!(back, vec![1.0, 3.0, 2.0, 4.0]);
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

    #[test]
    fn panels_fit_in_the_frame_without_overlapping() {
        for (w, h) in [
            (1920, 1080),
            (3840, 2160),
            (1280, 720),
            (1080, 1920),
            (320, 180),
            (500, 500),
        ] {
            for mode in [PhaseSpace::Row, PhaseSpace::Full] {
                let Some((cells, size)) = panel_layout(mode, w, h) else {
                    continue;
                };
                assert!(cells.len() <= mode.picks());
                for (i, &(x, y)) in cells.iter().enumerate() {
                    assert!(x + size <= w && y + size <= h, "{}x{} {:?}", w, h, mode);
                    for &(x2, y2) in &cells[i + 1..] {
                        assert!(
                            x + size <= x2 || x2 + size <= x || y + size <= y2 || y2 + size <= y
                        );
                    }
                }
            }
        }
        assert_eq!(
            panel_layout(PhaseSpace::Row, 1920, 1080).unwrap().0.len(),
            4
        );
        assert_eq!(
            panel_layout(PhaseSpace::Full, 3840, 2160).unwrap().0.len(),
            8
        );
        assert!(panel_layout(PhaseSpace::Row, 96, 54).is_none());
        assert!(panel_layout(PhaseSpace::Off, 1920, 1080).is_none());
    }

    #[test]
    fn the_brightest_modes_are_spectral_peaks_first() {
        let a = [0.1, 0.9, 0.8, 0.2, 0.0, -0.5, 0.0];
        let b = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0];
        // gailurrak: a[1] 0.9, a[5] 0.5, b[2] 0.3; gero gainerakoak: a[2] 0.8...
        assert_eq!(
            brightest(&[&a, &b], 4),
            vec![(0, 1), (0, 5), (1, 2), (0, 2)]
        );
        // isiltasuna: hutsa, bin-en ordenan
        assert_eq!(brightest(&[&[0.0; 4]], 2), vec![(0, 0), (0, 1)]);
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
