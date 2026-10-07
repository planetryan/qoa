// fitxategi honek exekuzio baten emaitza testu edo json bihurtzen du.
// `Report` da emaitza-erregistroa (json-en irteera bera, programa klasikoek irakurtzeko);
// `render` funtzioak terminalerako bertsioa idazten du. itxura aldatzeko, hemen bakarrik aldatu.
// json-eko gakoak beti ingelesez daude (programek irakurtzen dituzte), testua hizkuntzaren arabera.

use crate::program::Program;
use crate::state::{State, format_bytes};
use crate::vm::{Method, Record, RunConfig, RunOutput, fmt_value};
use crate::{t, ts};
use serde::Serialize;

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
    pub method: Method,
    pub shots: u64,
    pub seed: u64,
    pub seed_given: bool,
    pub elapsed_ms: f64,
    pub steps_first_shot: u64,
    // bitak: c[n-1]..c[0] ordenan, '-' idatzi gabe; moduak: o0, o1, ... (null idatzi gabe)
    pub counts: Vec<CountEntry>,
    pub truncation_leakage: f64,
    pub registers_first_shot: Vec<f64>,
    pub state: Option<Vec<AmpEntry>>,
    pub warnings: Vec<String>,
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
    let state = want_state.then(|| amp_entries(&out.final_state, input.top));
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
        method: out.method,
        shots: out.shots,
        seed: input.config.seed,
        seed_given: input.seed_given,
        elapsed_ms: input.elapsed.as_secs_f64() * 1000.0,
        steps_first_shot: out.steps,
        counts,
        truncation_leakage: out.leakage,
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
fn amp_entries(state: &State, top: usize) -> Vec<AmpEntry> {
    let amps = state.amplitudes();
    let mut idx: Vec<usize> = (0..state.len())
        .filter(|&i| amps[i].norm_sqr() > 1e-12)
        .collect();
    idx.sort_by(|&a, &b| {
        let (pa, pb) = (amps[a].norm_sqr(), amps[b].norm_sqr());
        pb.partial_cmp(&pa).unwrap().then(a.cmp(&b))
    });
    if top > 0 {
        idx.truncate(top);
    }
    idx.into_iter()
        .map(|i| AmpEntry {
            basis: basis_label(state, i),
            re: amps[i].re,
            im: amps[i].im,
            probability: amps[i].norm_sqr(),
        })
        .collect()
}

// DUMPSTATE-k erabiltzen duen taula
pub fn state_table(state: &State, top: usize) -> String {
    let entries = amp_entries(state, top);
    let nonzero = state
        .amplitudes()
        .iter()
        .filter(|a| a.norm_sqr() > 1e-12)
        .count();
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
    fn label(&self, s: &str) -> String {
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
    let noise = match (r.noise > 0.0, r.loss > 0.0) {
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
