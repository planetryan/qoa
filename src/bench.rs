// fitxategi honek QOA-ren errendimendu-proba osoa egiten du (`qoa --benchmark`, `--bm`): makinaren memoria,
// mihiztatzailea, qubit-ateak, laginketa, shot-ak, modu optikoak, 128 biteko motorra, batura zehatzak, argia
// (`qoa visual`-en fisika), wigner panelak eta, badago, GPUa.
//
// atal bakoitzak lan finko bat egiten du (hazi finkoekin, beraz lan bera makina guztietan), hainbat aldiz, eta
// onena hartzen da. puntuazioa: 1000 erreferentziako makinan (ikus REFERENCE); bikoitza, bi aldiz azkarrago.
// guztizkoa CPU atalen batez besteko geometrikoa da (SPEC bezala), atal bakar batek ez dezan dena erabaki.
// GPUa bere puntuazioarekin doa, ez baitago makina guztietan.

use crate::kernel::{self, QGate};
use crate::light::{self, Detect, KerrSampler, Light, Mdct};
use crate::program::Gate;
use crate::quad::{self, QuadGate, QuadState};
use crate::state::State;
use crate::vm::{self, RunConfig};
use crate::{t, ts};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use std::time::Instant;

// erreferentziako makina: 1000 puntu atal bakoitzean
pub const REFERENCE: &str = "4 cores of a 2.8 GHz Intel Xeon with AVX-512 (cloud VM), QOA 0.4.3";

#[derive(Clone, Debug, serde::Serialize)]
pub struct Item {
    pub key: &'static str,
    pub name: String,
    pub what: String,
    // emaitza eta bere unitatea (handiagoa hobea)
    pub value: f64,
    pub unit: &'static str,
    // 1000 = erreferentzia
    pub score: f64,
    // denborak segundotan: onena eta mediana, eta exekuzio-kopurua
    pub best_s: f64,
    pub median_s: f64,
    pub runs: usize,
    pub details: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Report {
    pub qoa_version: String,
    pub quick: bool,
    pub cpu: String,
    pub arch: String,
    pub threads: usize,
    pub kernels: String,
    pub reference: String,
    pub score: f64,
    pub items: Vec<Item>,
    pub gpu: Option<Item>,
    pub gpu_note: Option<String>,
    pub seconds: f64,
}

// lan bat `runs` aldiz: (onena, mediana), segundotan
fn time(runs: usize, mut f: impl FnMut()) -> (f64, f64) {
    let mut t: Vec<f64> = (0..runs)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64()
        })
        .collect();
    t.sort_by(f64::total_cmp);
    (t[0], t[t.len() / 2])
}

// sasi-ausazko sorgailu finkoa (lan bera makina guztietan)
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, m: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % m
    }

    fn angle(&mut self) -> f64 {
        self.next(1_000_000) as f64 / 1_000_000.0 * std::f64::consts::TAU
    }
}

// ausazko qubit-zirkuitu bat (H, RX, RZ, CNOT, CZ, T), kernelentzat
fn circuit(n: usize, depth: usize, seed: u64) -> Vec<QGate> {
    let mut r = Lcg(seed);
    (0..depth)
        .map(|_| {
            let q = r.next(n as u64) as usize;
            let c = (q + 1 + r.next(n as u64 - 1) as usize) % n;
            let (g, controls) = match r.next(6) {
                0 => (Gate::H, 0),
                1 => (Gate::RX(r.angle()), 0),
                2 => (Gate::RZ(r.angle()), 0),
                3 => (Gate::X, 1 << c),
                4 => (Gate::Z, 1 << c),
                _ => (Gate::T, 0),
            };
            QGate::Mat {
                q,
                m: g.matrix(),
                controls,
            }
        })
        .collect()
}

struct Spec {
    key: &'static str,
    name: String,
    what: String,
    unit: &'static str,
    // erreferentziako balioa (1000 puntu)
    reference: f64,
}

fn item(
    spec: Spec,
    work: f64,
    (best, median): (f64, f64),
    runs: usize,
    details: Vec<String>,
) -> Item {
    let value = work / best.max(1e-12);
    Item {
        key: spec.key,
        name: spec.name,
        what: spec.what,
        value,
        unit: spec.unit,
        score: 1000.0 * value / spec.reference,
        best_s: best,
        median_s: median,
        runs,
        details,
    }
}

fn memory(quick: bool, runs: usize) -> Item {
    let n = if quick { 1 << 21 } else { 1 << 23 };
    let b = vec![1.0f64; n];
    let c = vec![2.0f64; n];
    let mut a = vec![0.0f64; n];
    let times = time(runs + 1, || {
        a.par_chunks_mut(1 << 14)
            .zip(b.par_chunks(1 << 14))
            .zip(c.par_chunks(1 << 14))
            .for_each(|((a, b), c)| {
                for ((x, y), z) in a.iter_mut().zip(b).zip(c) {
                    *x = y + 3.0 * z;
                }
            });
    });
    std::hint::black_box(&a);
    item(
        Spec {
            key: "memory",
            name: ts!("memory bandwidth", "memoriaren banda-zabalera").into(),
            what: ts!(
                "a[i] = b[i] + 3·c[i] over three arrays (STREAM triad): what bounds every large simulation",
                "a[i] = b[i] + 3·c[i] hiru array-tan (STREAM triad): simulazio handi guztiak mugatzen dituena"
            )
            .into(),
            unit: "GB/s",
            reference: 34.0,
        },
        (3 * n * 8) as f64 / 1e9,
        times,
        runs + 1,
        vec![t!(
            "3 × {} MiB",
            "3 × {} MiB",
            (n * 8) >> 20
        )],
    )
}

fn assembler(quick: bool, runs: usize) -> Item {
    let lines = if quick { 20_000 } else { 100_000 };
    let mut r = Lcg(7);
    let mut src = String::from("QINIT 16\nQMODE 2 6\n");
    for k in 0..lines {
        let q = r.next(16);
        src.push_str(&match k % 7 {
            0 => format!("H q{}\n", q),
            1 => format!("RX q{} {:.6}\n", q, r.angle()),
            2 => format!("CNOT q{} q{}\n", q, (q + 1) % 16),
            3 => format!("REGSET R{} {}\n", q, k),
            4 => format!("ADD R1 R{} R2 ; sum\n", q),
            5 => format!("BS o0 o1 pi/{}\n", 2 + q),
            _ => format!("CRZ q{} q{} pi/3\n", q, (q + 3) % 16),
        });
    }
    let mut n_ops = 0;
    let times = time(runs, || {
        let p = crate::asm::assemble(&src, "bench.qoa").expect("benchmark programa");
        n_ops = p.ops.len();
    });
    item(
        Spec {
            key: "assembler",
            name: ts!("assembler", "mihiztatzailea").into(),
            what: ts!(
                "parse, check and assemble a generated program",
                "sortutako programa bat analizatu, egiaztatu eta mihiztatu"
            )
            .into(),
            unit: "k lines/s",
            reference: 530.0,
        },
        lines as f64 / 1000.0,
        times,
        runs,
        vec![t!(
            "{} lines, {} operations",
            "{} lerro, {} eragiketa",
            lines,
            n_ops
        )],
    )
}

fn gates(n: usize, depth: usize, runs: usize, reference: f64, key: &'static str) -> Item {
    let gates = circuit(n, depth, 11 + n as u64);
    let mut amps = vec![C::new(0.0, 0.0); 1 << n];
    amps[0] = C::new(1.0, 0.0);
    let times = time(runs, || kernel::apply_gates(&mut amps, n, &gates));
    std::hint::black_box(&amps);
    let updates = (depth as f64) * (1u64 << n) as f64;
    item(
        Spec {
            key,
            name: t!("qubit gates, {} qubits", "qubit-ateak, {} qubit", n),
            what: ts!(
                "a random circuit through the SIMD kernels, with fusion and cache blocking",
                "zirkuitu ausazko bat SIMD kernelen bidez, fusio eta cache-blokeekin"
            )
            .into(),
            unit: "G amp-gates/s",
            reference,
        },
        updates / 1e9,
        times,
        runs,
        vec![
            t!(
                "{} gates on {} amplitudes ({})",
                "{} ate {} anplitudetan ({})",
                depth,
                1u64 << n,
                crate::state::format_bytes(16u128 << n)
            ),
            t!("{} kernels", "{} kernelak", crate::simd::active().name()),
        ],
    )
}

fn sampling(quick: bool, runs: usize) -> Item {
    let n = 20;
    let shots = if quick { 100_000 } else { 1_000_000 };
    let mut src = format!("QINIT {}\n", n);
    for q in 0..n {
        src.push_str(&format!("RY q{} {}\n", q, 0.3 + 0.1 * q as f64));
    }
    src.push_str("MEASUREALL\n");
    let prog = crate::asm::assemble(&src, "bench.qoa").expect("benchmark programa");
    let cfg = RunConfig {
        shots,
        seed: 1,
        ..RunConfig::default()
    };
    let times = time(runs, || {
        vm::run(&prog, &cfg, &mut std::io::sink()).expect("benchmark exekuzioa");
    });
    item(
        Spec {
            key: "sampling",
            name: ts!("sampling", "laginketa").into(),
            what: ts!(
                "simulate once, then sample every shot from the state (exact cumulative sums)",
                "behin simulatu, gero shot guztiak egoeratik laginkatu (batura metatu zehatzak)"
            )
            .into(),
            unit: "M shots/s",
            reference: 0.87,
        },
        shots as f64 / 1e6,
        times,
        runs,
        vec![t!(
            "{} shots of {} qubits",
            "{} shot {} qubitekin",
            shots,
            n
        )],
    )
}

fn every_shot(quick: bool, runs: usize) -> Item {
    let shots = if quick { 4_000 } else { 20_000 };
    let src = "QINIT 10\nH q0\nCNOT q0 q1\nMEASURE q0 -> c0\nJUMPIFZERO c0 skip\nX q2\nskip:\nRY q3 0.7\nCRZ q3 q4 pi/3\nH q5\nMEASUREALL\n";
    let prog = crate::asm::assemble(src, "bench.qoa").expect("benchmark programa");
    let cfg = RunConfig {
        shots,
        seed: 2,
        noise: 0.01,
        ..RunConfig::default()
    };
    let times = time(runs, || {
        vm::run(&prog, &cfg, &mut std::io::sink()).expect("benchmark exekuzioa");
    });
    item(
        Spec {
            key: "shots",
            name: ts!("shots one by one", "shot-ak banan-banan").into(),
            what: ts!(
                "a program with a mid-circuit measurement, a branch and noise: every shot simulated from the start",
                "erdiko neurketa, adarkatze bat eta zarata dituen programa bat: shot bakoitza hasieratik simulatua"
            )
            .into(),
            unit: "k shots/s",
            reference: 42.0,
        },
        shots as f64 / 1e3,
        times,
        runs,
        vec![t!(
            "{} shots of 10 qubits, 1% noise",
            "{} shot 10 qubitekin, % 1 zarata",
            shots
        )],
    )
}

fn optics(quick: bool, runs: usize) -> Item {
    let reps = if quick { 20 } else { 100 };
    let src = format!(
        "QMODE 3 24\nLOOPSTART {}\nDISPLACE o0 0.4\nSQUEEZE o1 0.2\nBS o0 o1 pi/4\nKERR o2 0.1\nBS o1 o2 pi/5 0.3\nMODEPHASE o0 0.2\nLOOPEND\n",
        reps
    );
    let prog = crate::asm::assemble(&src, "bench.qoa").expect("benchmark programa");
    let cfg = RunConfig {
        shots: 1,
        ..RunConfig::default()
    };
    let times = time(runs, || {
        vm::run(&prog, &cfg, &mut std::io::sink()).expect("benchmark exekuzioa");
    });
    item(
        Spec {
            key: "optics",
            name: ts!("optical modes", "modu optikoak").into(),
            what: ts!(
                "displacement, squeezing, beamsplitters and Kerr on three modes in the photon-number basis",
                "desplazamendua, estutzea, habe-zatitzaileak eta kerr hiru modutan fotoi-kopuruaren oinarrian"
            )
            .into(),
            unit: "ops/s",
            reference: 900.0,
        },
        (reps * 6) as f64,
        times,
        runs,
        vec![t!(
            "{} operations, 3 modes × 24 photon numbers ({} amplitudes)",
            "{} eragiketa, 3 modu × 24 fotoi-kopuru ({} anplitude)",
            reps * 6,
            24 * 24 * 24
        )],
    )
}

fn quad_gates(quick: bool, runs: usize) -> Item {
    let n = if quick { 14 } else { 18 };
    let depth = 200;
    let gates: Vec<QuadGate> = circuit(n, depth, 5)
        .into_iter()
        .map(|g| QuadGate::from_f64(&g))
        .collect();
    let mut s = QuadState::new(n).expect("128 biteko egoera");
    let times = time(runs, || s.apply(&gates));
    std::hint::black_box(&s);
    let _ = quad::frac_1_sqrt_2();
    item(
        Spec {
            key: "quad",
            name: ts!("128-bit gates", "128 biteko ateak").into(),
            what: ts!(
                "the same kind of circuit in double-double (about 32 digits)",
                "zirkuitu mota bera double-double-n (32 digitu inguru)"
            )
            .into(),
            unit: "M amp-gates/s",
            reference: 360.0,
        },
        depth as f64 * (1u64 << n) as f64 / 1e6,
        times,
        runs,
        vec![t!(
            "{} gates on {} amplitudes",
            "{} ate {} anplitudetan",
            depth,
            1u64 << n
        )],
    )
}

fn exact_sums(quick: bool, runs: usize) -> Item {
    let n = if quick { 20 } else { 24 };
    let mut amps = vec![C::new(0.0, 0.0); 1 << n];
    amps.par_iter_mut().enumerate().for_each(|(i, a)| {
        *a = C::new(
            ((i * 7919) % 1000) as f64 * 1e-4,
            ((i * 104729) % 997) as f64 * 1e-4,
        )
    });
    let state = State::from_amplitudes(n, 0, 2, amps).expect("egoera");
    let mut norm = 0.0;
    let times = time(runs, || norm = state.norm_sqr());
    std::hint::black_box(norm);
    item(
        Spec {
            key: "sums",
            name: ts!("exact sums", "batura zehatzak").into(),
            what: ts!(
                "the state's norm in double-double (~106 bits), as every measurement uses",
                "egoeraren araua double-double-n (~106 bit), neurketa guztiek erabiltzen duten bezala"
            )
            .into(),
            unit: "G amps/s",
            reference: 0.40,
        },
        (1u64 << n) as f64 / 1e9,
        times,
        runs,
        vec![t!("{} amplitudes", "{} anplitude", 1u64 << n)],
    )
}

fn light_item(quick: bool, runs: usize) -> Item {
    let samples = if quick { 3_000 } else { 24_000 };
    let mut r = Lcg(3);
    let signal: Vec<f64> = (0..samples)
        .map(|i| 0.4 * (i as f64 * 0.05).sin() + 0.05 * (r.next(1000) as f64 / 500.0 - 1.0))
        .collect();
    let mdct = Mdct::new(1024);
    let light = Light {
        photons: 1e5,
        loss: 0.1,
        squeeze_db: 3.0,
        kerr: 1.0,
        mix: 0.05,
        detect: Detect::Homodyne,
    };
    let frames = light::analyze(&mdct, &signal);
    let modes = frames.len() * 1024;
    let times = time(runs, || {
        let mut ch = vec![frames.clone()];
        light.process(&mut ch, 9);
    });
    item(
        Spec {
            key: "light",
            name: ts!("light (qoa visual)", "argia (qoa visual)").into(),
            what: ts!(
                "squeezed light through loss and a Kerr medium, one exact homodyne shot per mode",
                "argi estutua galeratik eta kerr ingurune batetik, shot homodino zehatz bat modu bakoitzeko"
            )
            .into(),
            unit: "k modes/s",
            reference: 3.4,
        },
        modes as f64 / 1e3,
        times,
        runs,
        vec![t!(
            "{} modes ({} samples of audio)",
            "{} modu ({} audio-lagin)",
            modes,
            samples
        )],
    )
}

fn wigner(quick: bool, runs: usize) -> Item {
    let count = if quick { 4 } else { 10 };
    let light = Light {
        kerr: 2.0,
        ..Light::default()
    };
    let kappa = 2.0 / (2.0 * 2e5);
    let sampler = KerrSampler::new(kappa);
    let times = time(runs, || {
        for k in 0..count {
            // modu ahuletik distiratsuenera (√(2·10⁵) ≈ 447), bideoan bezala
            let alpha = 447.0 * ((k + 1) as f64 / count as f64).sqrt();
            std::hint::black_box(light.phase_space(alpha, Some(&sampler), 192));
        }
    });
    item(
        Spec {
            key: "wigner",
            name: ts!("Wigner panels", "wigner panelak").into(),
            what: ts!(
                "exact pixel-averaged Wigner functions of Kerr states, as in the video",
                "kerr egoeren wigner funtzio zehatzak pixelka batez bestekotuta, bideoan bezala"
            )
            .into(),
            unit: "panels/s",
            reference: 230.0,
        },
        count as f64,
        times,
        runs,
        vec![t!(
            "{} panels of 192 × 192",
            "{} panel 192 × 192koak",
            count
        )],
    )
}

fn gpu_item(quick: bool, runs: usize) -> Result<Item, String> {
    let gpu = crate::gpu::Gpu::open()?;
    let n = if quick { 18 } else { 22 };
    let depth = 200;
    let gates = circuit(n, depth, 13);
    let mut amps = vec![C::new(0.0, 0.0); 1 << n];
    amps[0] = C::new(1.0, 0.0);
    let mut failed = None;
    let times = time(runs, || {
        if let Err(e) = gpu.apply(&mut amps, n, &gates) {
            failed.get_or_insert(e);
        }
    });
    if let Some(e) = failed {
        return Err(e);
    }
    let s = gpu.stats();
    Ok(item(
        Spec {
            key: "gpu",
            name: t!("GPU: {}", "GPUa: {}", s.device),
            what: ts!(
                "the same random circuit on the GPU, including moving the state there and back",
                "zirkuitu ausazko bera GPUan, egoera hara eta hona mugitzea barne"
            )
            .into(),
            unit: "G amp-gates/s",
            // 1000 = erreferentziako makinaren CPUa zirkuitu berarekin: puntuazioa CPUarekiko abiadura da
            reference: 8.0,
        },
        depth as f64 * (1u64 << n) as f64 / 1e9,
        times,
        runs,
        vec![
            t!(
                "{} gates on {} amplitudes, {}",
                "{} ate {} anplitudetan, {}",
                depth,
                1u64 << n,
                s.device_type
            ),
            t!(
                "{:.3} s moving the state, {:.3} s computing (all runs)",
                "{:.3} s egoera mugitzen, {:.3} s kalkulatzen (exekuzio guztiak)",
                s.transfer_seconds,
                s.compute_seconds
            ),
        ],
    ))
}

fn cpu_name() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| {
                    l.starts_with("model name") || l.starts_with("cpu\t") || l.starts_with("Model")
                })
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| ts!("unknown", "ezezaguna").into())
}

// proba osoa; `progress` atal bakoitzaren izenarekin deitzen da hasi aurretik
pub fn run(quick: bool, gpu: bool, mut progress: impl FnMut(&str)) -> Report {
    let started = Instant::now();
    let runs = if quick { 2 } else { 3 };
    let mut items = Vec::new();
    let mut step = |name: &str, f: &dyn Fn() -> Item, items: &mut Vec<Item>| {
        progress(name);
        items.push(f());
    };
    step("memory", &|| memory(quick, runs), &mut items);
    step("assembler", &|| assembler(quick, runs), &mut items);
    step(
        "gates 20",
        &|| gates(20, if quick { 100 } else { 400 }, runs, 8.0, "gates20"),
        &mut items,
    );
    if !quick {
        step(
            "gates 24",
            &|| gates(24, 200, runs, 8.4, "gates24"),
            &mut items,
        );
    }
    step("sampling", &|| sampling(quick, runs), &mut items);
    step("shots", &|| every_shot(quick, runs), &mut items);
    step("optics", &|| optics(quick, runs), &mut items);
    step("quad", &|| quad_gates(quick, runs), &mut items);
    step("sums", &|| exact_sums(quick, runs), &mut items);
    step("light", &|| light_item(quick, runs), &mut items);
    step("wigner", &|| wigner(quick, runs), &mut items);
    let (gpu_item, gpu_note) = if gpu {
        progress("gpu");
        match gpu_item(quick, runs) {
            Ok(i) => (Some(i), None),
            Err(e) => (None, Some(e)),
        }
    } else {
        (
            None,
            Some(ts!("not run (--no-gpu)", "ez da exekutatu (--no-gpu)").into()),
        )
    };
    let score =
        (items.iter().map(|i| i.score.max(1e-9).ln()).sum::<f64>() / items.len() as f64).exp();
    Report {
        qoa_version: env!("CARGO_PKG_VERSION").to_string(),
        quick,
        cpu: cpu_name(),
        arch: std::env::consts::ARCH.to_string(),
        threads: rayon::current_num_threads(),
        kernels: crate::simd::active().name().to_string(),
        reference: REFERENCE.to_string(),
        score,
        items,
        gpu: gpu_item,
        gpu_note,
        seconds: started.elapsed().as_secs_f64(),
    }
}

// zenbaki bat 3-4 digitu esanguratsurekin
fn num(v: f64) -> String {
    if v >= 100.0 {
        format!("{:.0}", v)
    } else if v >= 10.0 {
        format!("{:.1}", v)
    } else {
        format!("{:.2}", v)
    }
}

// txostena testu gisa, xehetasun guztiekin
pub fn render(r: &Report, style: &crate::report::Style) -> String {
    let mut s = String::new();
    let line = |label: &str, text: String| format!("{}  {}\n", style.label(label), text);
    s += &line(
        ts!("machine", "makina"),
        t!(
            "{} ({}), {} thread(s), {} kernels",
            "{} ({}), {} hari, {} kernelak",
            r.cpu,
            r.arch,
            r.threads,
            r.kernels
        ),
    );
    s += &line(
        ts!("scale", "eskala"),
        t!(
            "1000 = {}; twice the score, twice as fast",
            "1000 = {}; puntuazio bikoitza, bi aldiz azkarrago",
            r.reference
        ),
    );
    if r.quick {
        s += &line(
            ts!("mode", "modua"),
            ts!(
                "quick: smaller problems, so scores lean on cache more than a full run",
                "azkarra: problema txikiagoak, beraz puntuazioak cache-ren menpe gehiago daude proba osoan baino"
            )
            .into(),
        );
    }
    s.push('\n');
    let all: Vec<&Item> = r.items.iter().chain(r.gpu.as_ref()).collect();
    for i in all {
        s += &format!(
            "{:>6}  {}  {} {}\n",
            num(i.score),
            style.label(&i.name),
            num(i.value),
            i.unit
        );
        s += &format!("        {}\n", i.what);
        for d in &i.details {
            s += &format!("        {}\n", d);
        }
        s += &format!(
            "        {}\n",
            t!(
                "best of {}: {:.3} s (median {:.3} s)",
                "{}tik onena: {:.3} s (mediana {:.3} s)",
                i.runs,
                i.best_s,
                i.median_s
            )
        );
    }
    if let Some(note) = &r.gpu_note {
        s += &format!(
            "{:>6}  {}  {}\n",
            "-",
            style.label(ts!("GPU", "GPUa")),
            t!("skipped: {}", "saltatua: {}", note)
        );
    }
    s.push('\n');
    s += &line(
        ts!("score", "puntuazioa"),
        t!(
            "{} (geometric mean of the {} CPU items){}",
            "{} ({} CPU atalen batez besteko geometrikoa){}",
            num(r.score),
            r.items.len(),
            r.gpu
                .as_ref()
                .map(|g| t!("; GPU {}", "; GPUa {}", num(g.score)))
                .unwrap_or_default()
        ),
    );
    s += &line(ts!("time", "denbora"), format!("{:.1} s", r.seconds));
    s
}
