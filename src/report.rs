// fitxategi honek exekuzio baten emaitza testu edo json bihurtzen du.
// `Report` da emaitza-erregistroa (json-en irteera bera, programa klasikoek irakurtzeko);
// `render` funtzioak terminalerako bertsioa idazten du. itxura aldatzeko, hemen bakarrik aldatu.
// json-eko gakoak beti ingelesez daude (programek irakurtzen dituzte), testua hizkuntzaren arabera.

use crate::program::Program;
use crate::state::{State, format_bytes};
use crate::vm::{Method, Record, RunConfig, RunOutput, fmt_value};
use crate::{t, ts};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use serde::Serialize;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Serialize, Debug)]
pub struct Report {
    pub qoa_version: String,
    pub program: String,
    pub operations: usize,
    pub qubits: u32,
    pub modes: u32,
    pub cutoff: u32,
    pub cutoff_declared: bool,
    pub amplitudes: u128,
    pub memory_bytes: u128,
    pub noise: f64,
    pub loss: f64,
    // `--noise-model`: gailua eta haren kalibrazioa (bestela null)
    pub noise_model: Option<ModelInfo>,
    pub method: Method,
    pub shots: u64,
    pub seed: u64,
    pub seed_given: bool,
    pub elapsed_ms: f64,
    // erabilitako kernelak (avx512, avx2, neon, rvv edo scalar) eta hari-kopurua
    pub kernels: String,
    pub threads: usize,
    // MPI prozesuak (`--mpi`), shot-ak haien artean banatuta; 1 bestela
    pub ranks: usize,
    // egoera bera prozesuen artean banatuta (`--distribute`)
    pub distributed: bool,
    // `--gpu`: gailua eta zer egin duen (bestela null)
    pub gpu: Option<crate::gpu::Stats>,
    pub steps_first_shot: u64,
    // bitak: c[n-1]..c[0] ordenan, '-' idatzi gabe; moduak: o0, o1, ... (null idatzi gabe)
    pub counts: Vec<CountEntry>,
    pub truncation_leakage: f64,
    // |1 - ‖ψ‖²| lehen shot-aren amaieran: biribiltzeak metatutako errorearen behe-muga
    pub norm_error: f64,
    // qubit-egoeraren doitasuna: "double" (64 bit) edo "quad" (128 bit, double-double)
    pub precision: crate::vm::Precision,
    pub registers_first_shot: Vec<f64>,
    pub state: Option<Vec<AmpEntry>>,
    pub warnings: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct ModelInfo {
    pub device: String,
    pub calibrated: String,
    pub qubits: usize,
    pub gates: usize,
    // qoa-ren qubit bakoitzaren gailuko zenbakia (null ezezaguna bada)
    pub physical_qubits: Vec<Option<u32>>,
}

#[derive(Serialize, Debug)]
pub struct CountEntry {
    pub bits: String,
    pub modes: Vec<Option<u32>>,
    pub count: u64,
    pub share: f64,
}

#[derive(Serialize, Debug)]
pub struct AmpEntry {
    pub basis: String,
    pub re: f64,
    pub im: f64,
    pub probability: f64,
}

pub struct ReportInput<'a> {
    pub program: &'a Program,
    pub config: &'a RunConfig,
    pub seed_given: bool,
    pub shots_given: bool,
    pub output: &'a RunOutput,
    pub elapsed: std::time::Duration,
    pub show_state: bool,
    pub top: usize,
}

pub fn build(input: ReportInput) -> Report {
    let p = input.program;
    let out = input.output;
    let dim = p.state_dim().unwrap_or(0);
    let total = out.shots as f64;
    let counts = out
        .counts
        .iter()
        .map(|(rec, n)| CountEntry {
            bits: bits_string(rec),
            modes: rec.readouts.clone(),
            count: *n,
            share: *n as f64 / total,
        })
        .collect();
    let mut warnings = Vec::new();
    if out.leakage > 1e-6 {
        warnings.push(t!(
            "the photon cutoff ({}) lost up to {:.2e} of the probability, so results are approximate; raise it with QMODE {} {}",
            "fotoi-mugak ({}) probabilitatearen {:.2e} arte galdu du, beraz emaitzak hurbilketak dira; handitu QMODE {} {} bidez",
            p.cutoff,
            out.leakage,
            p.n_modes,
            p.cutoff * 2
        ));
    }
    if out.method == Method::StateOnly && input.shots_given {
        warnings.push(t!(
            "--shots has no effect: the program measures nothing. Add MEASURE or MEASUREALL to get counts",
            "--shots-ek ez du eraginik: programak ez du ezer neurtzen. Gehitu MEASURE edo MEASUREALL kopuruak lortzeko"
        ));
    }
    // qubit edo modurik ez badago, ez dago egoerarik erakusteko
    let has_system = p.n_qubits > 0 || p.n_modes > 0;
    let want_state = has_system && (input.show_state || out.method == Method::StateOnly);
    // egoera banatu handi bat ez da biltzen (src/dist.rs): orduan ez dago erakusteko
    let whole = p.state_dim() == Some(out.final_state.len() as u128);
    if want_state && !whole {
        warnings.push(t!(
            "the final state is spread over {} processes and too large to gather and show",
            "amaierako egoera {} prozesutan banatuta dago eta handiegia da bildu eta erakusteko",
            input.config.team.as_ref().map_or(1, |t| t.size())
        ));
    }
    let state = (want_state && whole).then(|| amp_entries(&out.final_state, input.top).0);
    Report {
        qoa_version: env!("CARGO_PKG_VERSION").to_string(),
        program: p.name.clone(),
        operations: p.ops.len(),
        qubits: p.n_qubits,
        modes: p.n_modes,
        cutoff: p.cutoff,
        cutoff_declared: p.cutoff_declared,
        amplitudes: dim,
        memory_bytes: dim * 16,
        noise: input.config.noise,
        loss: input.config.loss,
        noise_model: input.config.model.as_ref().map(|m| ModelInfo {
            device: m.device.clone(),
            calibrated: m.calibrated.clone(),
            qubits: m.qubits.len(),
            gates: m.gate_count(),
            physical_qubits: m
                .qubits
                .iter()
                .take(p.n_qubits as usize)
                .map(|q| q.physical)
                .collect(),
        }),
        method: out.method,
        shots: out.shots,
        seed: input.config.seed,
        seed_given: input.seed_given,
        elapsed_ms: input.elapsed.as_secs_f64() * 1000.0,
        kernels: crate::simd::active().name().to_string(),
        threads: rayon::current_num_threads(),
        ranks: input.config.team.as_ref().map_or(1, |t| t.size()),
        distributed: input.config.distribute,
        gpu: input.config.gpu.as_ref().map(|g| g.stats()),
        steps_first_shot: out.steps,
        counts,
        truncation_leakage: out.leakage,
        norm_error: out.norm_error,
        precision: input.config.precision,
        registers_first_shot: out.regs.to_vec(),
        state,
        warnings,
    }
}

// bitak c[n-1]..c[0] ordenan (qiskit bezala: c0 eskuinean), '-' idatzi ez bada
pub fn bits_string(rec: &Record) -> String {
    rec.bits
        .iter()
        .rev()
        .map(|b| match b {
            Some(0) => '0',
            Some(_) => '1',
            None => '-',
        })
        .collect()
}

// oinarri-egoeraren etiketa: |q[n-1]..q0> eta moduak badaude |q..; o0 o1 ...>
pub fn basis_label(state: &State, idx: usize) -> String {
    let n = state.n_qubits();
    let mut s = String::from("|");
    for q in (0..n).rev() {
        s.push(if (idx >> q) & 1 == 1 { '1' } else { '0' });
    }
    if state.n_modes() > 0 {
        if n > 0 {
            s.push_str("; ");
        }
        let parts: Vec<String> = (0..state.n_modes())
            .map(|o| state.photons(idx, o).to_string())
            .collect();
        s.push_str(&parts.join(" "));
    }
    s.push('>');
    s
}

fn basis_header(state: &State) -> String {
    let n = state.n_qubits();
    let mut s = String::from("|");
    if n > 0 {
        if n <= 4 {
            for q in (0..n).rev() {
                s.push_str(&format!("q{}", q));
            }
        } else {
            s.push_str(&format!("q{}..q0", n - 1));
        }
    }
    if state.n_modes() > 0 {
        if n > 0 {
            s.push_str("; ");
        }
        let parts: Vec<String> = (0..state.n_modes()).map(|o| format!("o{}", o)).collect();
        s.push_str(&parts.join(" "));
    }
    s.push('>');
    s
}

// probabilitate handieneko `top` anplitudeak (0 = guztiak), 1e-12 baino handiagoak
// anplitude baten sailkapena: probabilitate handiena lehenik, berdinketan indize txikiena
#[derive(PartialEq)]
struct Ranked(f64, usize);

impl Eq for Ranked {}

impl Ord for Ranked {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0).then(o.1.cmp(&self.1))
    }
}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

// zero ez diren anplitudeetatik `top` handienak (0: guztiak), ordenatuta, eta zero ez direnen kopurua.
// egoera handietan zati bakoitzak bere `top` onenak gordetzen ditu (gutxieneko pila batean) paraleloan;
// ez da bektore osoa ordenatzen (24 qubitekin 16 milioi elementu)
fn top_amplitudes(amps: &[C], top: usize) -> (Vec<usize>, usize) {
    let local = |base: usize, chunk: &[C]| {
        let mut heap = BinaryHeap::new();
        let mut nonzero = 0;
        for (j, a) in chunk.iter().enumerate() {
            let p = a.norm_sqr();
            if p <= 1e-12 {
                continue;
            }
            nonzero += 1;
            let r = Ranked(p, base + j);
            if top == 0 || heap.len() < top {
                heap.push(Reverse(r));
            } else if r > heap.peek().unwrap().0 {
                heap.pop();
                heap.push(Reverse(r));
            }
        }
        (heap.into_vec(), nonzero)
    };
    let chunk = 1 << 16;
    let (mut found, nonzero) = amps
        .par_chunks(chunk)
        .enumerate()
        .map(|(k, c)| local(k * chunk, c))
        .reduce(
            || (Vec::new(), 0),
            |mut a, b| {
                a.0.extend(b.0);
                (a.0, a.1 + b.1)
            },
        );
    // Reverse-n ordena goranzkoa = sailkapenean beheranzkoa: onena lehenik
    found.sort();
    if top > 0 {
        found.truncate(top);
    }
    (found.into_iter().map(|Reverse(r)| r.1).collect(), nonzero)
}

fn amp_entries(state: &State, top: usize) -> (Vec<AmpEntry>, usize) {
    let amps = state.amplitudes();
    let (idx, nonzero) = top_amplitudes(amps, top);
    let entries = idx
        .into_iter()
        .map(|i| AmpEntry {
            basis: basis_label(state, i),
            re: amps[i].re,
            im: amps[i].im,
            probability: amps[i].norm_sqr(),
        })
        .collect();
    (entries, nonzero)
}

// DUMPSTATE-k erabiltzen duen taula
pub fn state_table(state: &State, top: usize) -> String {
    let (entries, nonzero) = amp_entries(state, top);
    let header = basis_header(state);
    let width = entries
        .iter()
        .map(|e| e.basis.len())
        .chain(std::iter::once(header.len()))
        .max()
        .unwrap_or(4);
    let mut s = format!(
        "  {:<w$}   {:>22}   {}\n",
        header,
        ts!("amplitude", "anplitudea"),
        ts!("probability", "probabilitatea"),
        w = width
    );
    for e in &entries {
        s.push_str(&format!(
            "  {:<w$}   {:>+10.6}{:>+11.6}i   {:.6}\n",
            e.basis,
            e.re,
            e.im,
            e.probability,
            w = width
        ));
    }
    if nonzero > entries.len() {
        s.push_str(&t!(
            "  ({} more nonzero amplitudes not shown)\n",
            "  (zero ez diren beste {} anplitude ez dira erakusten)\n",
            nonzero - entries.len()
        ));
    }
    s
}

// terminalerako estiloa: koloreak terminal bada eta NO_COLOR ez badago
pub struct Style {
    pub color: bool,
}

impl Style {
    pub fn detect() -> Style {
        use std::io::IsTerminal;
        Style {
            color: std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none(),
        }
    }

    pub fn plain() -> Style {
        Style { color: false }
    }

    // etiketa ezkerrean, zabalera finkoan, ilun
    pub fn label(&self, s: &str) -> String {
        let padded = format!("{:<9}", s);
        if self.color {
            format!("\x1b[2m{}\x1b[0m", padded)
        } else {
            padded
        }
    }

    fn warn(&self, s: &str) -> String {
        let padded = format!("{:<9}", s);
        if self.color {
            format!("\x1b[33m{}\x1b[0m", padded)
        } else {
            padded
        }
    }
}

const INDENT: &str = "           ";

fn bar(share: f64, width: usize) -> String {
    let cells = share * width as f64;
    let full = cells.floor() as usize;
    let rem = cells - full as f64;
    let parts = ['▏', '▎', '▍', '▌', '▋', '▊', '▉'];
    let mut s = "█".repeat(full);
    let k = (rem * 8.0).floor() as usize;
    if k > 0 && full < width {
        s.push(parts[k - 1]);
    }
    s
}

// terminalerako txostena: lerro bakoitzak gauza bat esaten du, etiketa ezkerrean
pub fn render(r: &Report, style: &Style) -> String {
    let mut s = String::new();
    let mut line = |label: &str, text: String| {
        s.push_str(&format!("{}  {}\n", style.label(label), text));
    };

    line(
        ts!("program", "programa"),
        t!(
            "{} ({} operations)",
            "{} ({} eragiketa)",
            r.program,
            r.operations
        ),
    );
    let mut system = t!("{} qubit(s)", "{} qubit", r.qubits);
    if r.modes > 0 {
        system.push_str(&if r.cutoff_declared {
            t!(
                ", {} mode(s) of 0..{} photons",
                ", {} modu, 0..{} fotoikoak",
                r.modes,
                r.cutoff - 1
            )
        } else {
            t!(
                ", {} mode(s) of 0..{} photons (default cutoff)",
                ", {} modu, 0..{} fotoikoak (muga lehenetsia)",
                r.modes,
                r.cutoff - 1
            )
        });
    }
    system.push_str(&t!(
        " = {} amplitudes, {}",
        " = {} anplitude, {}",
        r.amplitudes,
        format_bytes(r.memory_bytes)
    ));
    line(ts!("system", "sistema"), system);
    let ranks = if r.distributed {
        t!(
            " in each of {} MPI processes, the state split between them",
            " {} MPI prozesu bakoitzean, egoera haien artean banatuta",
            r.ranks
        )
    } else if r.ranks > 1 {
        t!(
            " in each of {} MPI processes (shots split between them)",
            " {} MPI prozesu bakoitzean (shot-ak haien artean banatuta)",
            r.ranks
        )
    } else {
        String::new()
    };
    line(
        ts!("cpu", "cpu"),
        if r.precision == crate::vm::Precision::Quad {
            t!(
                "128-bit engine (double-double on fused multiply-add), {} thread(s){}",
                "128 biteko motorra (double-double biderketa-batuketa fusionatuaren gainean), {} hari{}",
                r.threads,
                ranks
            )
        } else {
            t!(
                "{} kernels, {} thread(s){}",
                "{} kernelak, {} hari{}",
                r.kernels,
                r.threads,
                ranks
            )
        },
    );
    // zenbakiak: ateak f64-n (FMA-rekin simd kerneletan), baturak double-double-n
    let fma = if r.kernels == "scalar" {
        String::new()
    } else {
        t!(
            ", fused multiply-add in the qubit-gate kernels",
            ", biderketa-batuketa fusionatua qubit-ateen kerneletan"
        )
    };
    line(
        ts!("numerics", "zenbakiak"),
        if r.precision == crate::vm::Precision::Quad {
            t!(
                "128-bit (double-double, ~32 digits): amplitudes, gate matrices, measurement and sums; the program's angles are its 64-bit values, taken as exact; norm off by {:.1e} at the end",
                "128 bit (double-double, ~32 digitu): anplitudeak, ateen matrizeak, neurketa eta baturak; programaren angeluak bere 64 biteko balioak dira, zehatz gisa hartuta; araua {:.1e} desbideratuta amaieran",
                r.norm_error
            )
        } else {
            t!(
                "64-bit floats{}; sums over the state in double-double (~106 bits); norm off by {:.1e} at the end",
                "64 biteko koma higikorra{}; egoeraren gaineko baturak double-double-n (~106 bit); araua {:.1e} desbideratuta amaieran",
                fma,
                r.norm_error
            )
        },
    );
    if let Some(g) = &r.gpu {
        line(
            ts!("gpu", "gpu"),
            format!("{} ({})", g.device, g.device_type),
        );
        let work = if g.batches == 0 {
            t!(
                "nothing ran on it: the program applied no qubit gates",
                "ez da ezer exekutatu bertan: programak ez du qubit-aterik aplikatu"
            )
        } else {
            t!(
                "{} batch(es) of qubit gates: {} gates in {} pass(es) over the state, {:.3} s computing; the state went up and back {} time(s) ({} each way, {:.3} s). measurements, optics and noise ran on the cpu",
                "{} qubit-ate sorta: {} ate egoeraren gaineko {} pasalditan, {:.3} s kalkulatzen; egoera {} aldiz igo eta jaitsi da ({} norabide bakoitzean, {:.3} s). neurketak, optika eta zarata cpu-an exekutatu dira",
                g.batches,
                g.gates,
                g.passes,
                g.compute_seconds,
                g.batches,
                crate::state::format_bytes(g.bytes_up as u128),
                g.transfer_seconds
            )
        };
        line(ts!("gpu work", "gpu lana"), work);
    }
    let noise = match (r.noise > 0.0, r.loss > 0.0) {
        _ if r.noise_model.is_some() => {
            let m = r.noise_model.as_ref().unwrap();
            let physical: Vec<String> = m
                .physical_qubits
                .iter()
                .enumerate()
                .map(|(i, p)| match p {
                    Some(p) => format!("q{}={}", i, p),
                    None => format!("q{}=?", i),
                })
                .collect();
            let when = if m.calibrated.is_empty() {
                String::new()
            } else {
                t!(", calibrated {}", ", kalibratua {}", m.calibrated)
            };
            t!(
                "device model {}{}: depolarizing error and T1/T2 relaxation after each native gate, readout errors on measurement (physical qubits {})",
                "{} gailuaren eredua{}: depolarizazio-errorea eta T1/T2 erlaxazioa ate natibo bakoitzaren ondoren, irakurketa-erroreak neurketan (qubit fisikoak {})",
                m.device,
                when,
                physical.join(" ")
            )
        }
        (false, false) => t!(
            "none: ideal gates, no photon loss",
            "bat ere ez: ate idealak, fotoi-galerarik gabe"
        ),
        (true, false) => t!(
            "depolarizing, p = {} per qubit per gate",
            "depolarizazioa, p = {} qubit eta ate bakoitzeko",
            r.noise
        ),
        (false, true) => t!(
            "photon loss, {} per mode per optical operation",
            "fotoi-galera, {} modu eta eragiketa optiko bakoitzeko",
            r.loss
        ),
        (true, true) => t!(
            "depolarizing p = {} per qubit per gate; photon loss {} per mode per optical operation",
            "depolarizazioa p = {} qubit eta ate bakoitzeko; fotoi-galera {} modu eta eragiketa optiko bakoitzeko",
            r.noise,
            r.loss
        ),
    };
    line(ts!("noise", "zarata"), noise);
    let method = match r.method {
        Method::Sampled => t!(
            "{} shot(s), sampled from one simulation (every measurement is at the end)",
            "{} shot, simulazio bakarretik laginduta (neurketa guztiak amaieran daude)",
            r.shots
        ),
        Method::EveryShot => t!(
            "{} shot(s), each simulated from the start (mid-circuit measurement, branching or noise)",
            "{} shot, bakoitza hasieratik simulatuta (erdiko neurketa, adarkatzea edo zarata)",
            r.shots
        ),
        Method::StateOnly if r.qubits == 0 && r.modes == 0 => t!(
            "1 run; the program has no qubits or modes (classical only)",
            "exekuzio 1; programak ez du qubit edo modurik (klasikoa bakarrik)"
        ),
        Method::StateOnly => t!(
            "1 run; the program measures nothing, so the exact final state is shown",
            "exekuzio 1; programak ez du ezer neurtzen, beraz amaierako egoera zehatza erakusten da"
        ),
    };
    line("shots", method);
    if r.method != Method::StateOnly || r.seed_given {
        let seed = if r.seed_given {
            format!("{}", r.seed)
        } else {
            t!(
                "{} (random; repeat this run with --seed {})",
                "{} (ausazkoa; exekuzio hau errepikatzeko --seed {})",
                r.seed,
                r.seed
            )
        };
        line(ts!("seed", "hazia"), seed);
    }
    s.push('\n');

    if r.method != Method::StateOnly {
        render_counts(&mut s, r, style);
    }

    if let Some(state) = &r.state {
        let title = if r.method == Method::StateOnly {
            t!(
                "final state, exact (a real device could only sample this)",
                "amaierako egoera, zehatza (benetako gailu batek lagindu besterik ezin luke egin)"
            )
        } else {
            t!(
                "state at the end of shot 1, exact (simulator only)",
                "1. shot-aren amaierako egoera, zehatza (simulagailuan bakarrik)"
            )
        };
        s.push_str(&format!(
            "{}  {}\n",
            style.label(ts!("state", "egoera")),
            title
        ));
        let w = state
            .iter()
            .map(|e| e.basis.len())
            .max()
            .unwrap_or(4)
            .max(6);
        for e in state {
            s.push_str(&format!(
                "{}{:<w$}  {:>+10.6}{:>+11.6}i   p = {:.6}\n",
                INDENT,
                e.basis,
                e.re,
                e.im,
                e.probability,
                w = w
            ));
        }
        if state.is_empty() {
            s.push_str(&format!(
                "{}{}\n",
                INDENT,
                ts!("(no qubits or modes)", "(ez dago qubit edo modurik)")
            ));
        }
        s.push('\n');
    }

    for w in &r.warnings {
        s.push_str(&format!(
            "{}  {}\n",
            style.warn(ts!("warning", "abisua")),
            w
        ));
    }
    if r.modes > 0 && r.truncation_leakage > 0.0 && r.truncation_leakage <= 1e-6 {
        s.push_str(&format!(
            "{}  {}\n",
            style.label(ts!("cutoff", "muga")),
            t!(
                "the photon cutoff lost at most {:.1e} of the probability (negligible)",
                "fotoi-mugak probabilitatearen {:.1e} galdu du gehienez (arbuiagarria)",
                r.truncation_leakage
            )
        ));
    }
    let regs: Vec<String> = r
        .registers_first_shot
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0.0)
        .map(|(i, v)| format!("R{}={}", i, fmt_value(*v)))
        .collect();
    if !regs.is_empty() {
        let shot = if r.shots > 1 {
            ts!(" (shot 1)", " (1. shot-a)")
        } else {
            ""
        };
        s.push_str(&format!(
            "{}  {}{}\n",
            style.label(ts!("regs", "erreg.")),
            regs.join(" "),
            shot
        ));
    }
    s.push_str(&format!(
        "{}  {:.1} ms\n",
        style.label(ts!("time", "denbora")),
        r.elapsed_ms
    ));
    s
}

fn render_counts(s: &mut String, r: &Report, style: &Style) {
    if r.counts.is_empty() {
        return;
    }
    let any_bits = r.counts.iter().any(|c| c.bits.chars().any(|b| b != '-'));
    let any_modes = r.counts.iter().any(|c| c.modes.iter().any(|m| m.is_some()));
    if !any_bits && !any_modes {
        s.push_str(&format!(
            "{}  {}\n\n",
            style.label(ts!("outcome", "emaitza")),
            t!(
                "nothing was written to a classical bit or mode readout (measurements went to registers only)",
                "ez da ezer idatzi bit klasiko batean edo modu-irakurketa batean (neurketak erregistroetara bakarrik joan dira)"
            )
        ));
        return;
    }
    let n_bits = r.counts[0].bits.len();
    // goiburua: zein bit eta zein modu
    let mut header = String::new();
    if any_bits {
        header.push_str(&if n_bits <= 1 {
            "c0".to_string()
        } else {
            format!("c{}..c0", n_bits - 1)
        });
    }
    if any_modes {
        if !header.is_empty() {
            header.push_str("  ");
        }
        header.push_str(ts!("photons per mode", "fotoiak moduko"));
    }
    let keys: Vec<String> = r
        .counts
        .iter()
        .map(|c| {
            let mut k = String::new();
            if any_bits {
                k.push_str(&c.bits);
            }
            if any_modes {
                if !k.is_empty() {
                    k.push_str("  ");
                }
                let m: Vec<String> = c
                    .modes
                    .iter()
                    .enumerate()
                    .filter_map(|(o, v)| v.map(|n| format!("o{}={}", o, n)))
                    .collect();
                k.push_str(&m.join(" "));
            }
            k
        })
        .collect();
    let w = keys
        .iter()
        .map(|k| k.chars().count())
        .max()
        .unwrap_or(1)
        .max(header.chars().count());
    let count_w = r.shots.to_string().len().max(7);
    s.push_str(&format!(
        "{}  {:<w$}  {:>cw$}   {:>8}\n",
        style.label(ts!("outcome", "emaitza")),
        header,
        ts!("count", "kopurua"),
        ts!("share", "ehunekoa"),
        w = w,
        cw = count_w
    ));
    let shown = 32usize;
    for (k, c) in keys.iter().zip(&r.counts).take(shown) {
        s.push_str(&format!(
            "{}{:<w$}  {:>cw$}   {:>7.1}%  {}\n",
            INDENT,
            k,
            c.count,
            c.share * 100.0,
            bar(c.share, 30),
            w = w,
            cw = count_w
        ));
    }
    if r.counts.len() > shown {
        let rest: u64 = r.counts[shown..].iter().map(|c| c.count).sum();
        s.push_str(&format!(
            "{}{}\n",
            INDENT,
            t!(
                "({} more outcomes, {} shots between them)",
                "(beste {} emaitza, {} shot guztira)",
                r.counts.len() - shown,
                rest
            )
        ));
    }
    s.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_amplitudes_match_a_full_sort() {
        // berdinketak barne (probabilitate bereko bikoteak): indize txikiena lehenik
        let amps: Vec<C> = (0..200_000u64)
            .map(|i| {
                let x = ((i.wrapping_mul(2654435761) >> 7) % 1000) as f64 / 1000.0;
                if i % 7 == 0 {
                    C::new(0.0, 0.0)
                } else {
                    C::new(x, -x / 3.0)
                }
            })
            .collect();
        let mut all: Vec<usize> = (0..amps.len())
            .filter(|&i| amps[i].norm_sqr() > 1e-12)
            .collect();
        all.sort_by(|&a, &b| {
            amps[b]
                .norm_sqr()
                .total_cmp(&amps[a].norm_sqr())
                .then(a.cmp(&b))
        });
        for top in [1, 16, 1000] {
            let (got, nonzero) = top_amplitudes(&amps, top);
            assert_eq!(nonzero, all.len());
            assert_eq!(got, all[..top].to_vec(), "top {}", top);
        }
        let (got, _) = top_amplitudes(&amps, 0);
        assert_eq!(got, all);
    }
}
