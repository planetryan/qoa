// fitxategi honek programa exekutatzen du (makina birtuala) eta shot-ak kudeatzen ditu.
//
// shot bat = programaren exekuzio oso bat. hiru modu daude, eta txostenak beti esaten du zein
// erabili den:
//   - sampled: neurketa guztiak amaieran badaude eta ez badago ausazkotasunik aurretik, behin
//     simulatzen da eta shot guztiak egoera horretatik lagintzen dira (zehatza eta azkarra)
//   - every shot: bestela, shot bakoitza hasieratik exekutatzen da (zarata edo gailu-eredu batekin beti)
//   - state only: programak ez badu ezer neurtzen, behin exekutatzen da eta egoera zehatza erakusten da

use crate::fock::{self, Mat};
use crate::gpu::Gpu;
use crate::kernel::QGate;
use crate::noise::{GateNoise, Native, NoiseModel, Relax};
use crate::program::{Basis, BinOp, Cmp, Count, Dest, Gate, NUM_REGS, Op, Program, Source, UnOp};
use crate::state::State;
use crate::{t, ts};
use num_complex::Complex64 as C;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

// exekuzio-aukerak
#[derive(Clone, Debug)]
pub struct RunConfig {
    pub shots: u64,
    pub seed: u64,
    // qubit-ate bakoitzaren ondoren, ukitutako qubit bakoitzean pauli errore baten probabilitatea
    pub noise: f64,
    // eragiketa optiko bakoitzaren ondoren, fotoi bakoitza galtzeko probabilitatea
    pub loss: f64,
    pub max_steps: u64,
    // qubit-ate sortak GPU honetan (`qoa run --gpu`); bestela CPUan
    pub gpu: Option<Arc<Gpu>>,
    // benetako gailu baten zarata-eredua (`qoa run --noise-model`, ikus src/noise.rs): ate natiboak
    // bakarrik, bakoitza bere errore, erlaxazio eta irakurketa-errorearekin
    pub model: Option<Arc<NoiseModel>>,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            shots: 1024,
            seed: 0,
            noise: 0.0,
            loss: 0.0,
            max_steps: 100_000_000,
            gpu: None,
            model: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    Sampled,
    EveryShot,
    StateOnly,
}

// shot bateko emaitza klasikoa: bit klasikoak eta moduen irakurketak (idatzi gabeak None)
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Record {
    pub bits: Vec<Option<u8>>,
    pub readouts: Vec<Option<u32>>,
}

#[derive(Debug)]
pub struct RunOutput {
    pub method: Method,
    pub shots: u64,
    // emaitza bakoitza eta zenbat aldiz agertu den, gehienetik gutxienera
    pub counts: Vec<(Record, u64)>,
    // lehen shot-aren amaierako egoera
    pub final_state: State,
    // lehen shot-aren amaierako erregistroak
    pub regs: [f64; NUM_REGS],
    // shot guztietako trunkatze-ihes handiena
    pub leakage: f64,
    // lehen shot-ak exekutatutako urratsak
    pub steps: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeError {
    pub line: u32,
    pub message: String,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line > 0 {
            write!(
                f,
                "{}",
                t!("line {}: {}", "{}. lerroa: {}", self.line, self.message)
            )
        } else {
            write!(f, "{}", self.message)
        }
    }
}

// zain dauden qubit-ateen sorta handiena: begizta luze batek ez du memoria mugarik gabe betetzen
const MAX_PENDING: usize = 4096;

const X_MATRIX: [C; 4] = [
    C::new(0.0, 0.0),
    C::new(1.0, 0.0),
    C::new(1.0, 0.0),
    C::new(0.0, 0.0),
];

// matrize optikoak behin kalkulatu eta shot guztietan berrerabili
// gakoa: (mota, lehen parametroaren bitak, bigarrenarenak, muga)
type MatrixKey = (u8, u64, u64, usize);

#[derive(Default)]
pub struct MatrixCache {
    map: Mutex<HashMap<MatrixKey, Arc<Mat>>>,
}

impl MatrixCache {
    fn get(&self, kind: u8, a: f64, b: f64, d: usize, make: impl FnOnce() -> Mat) -> Arc<Mat> {
        let key = (kind, a.to_bits(), b.to_bits(), d);
        if let Some(m) = self.map.lock().unwrap().get(&key) {
            return m.clone();
        }
        // blokeoa askatuta kalkulatu, beste hariak ez gelditzeko
        let m = Arc::new(make());
        self.map.lock().unwrap().insert(key, m.clone());
        m
    }
}

// shot bakoitzaren rng-a: hazitik eta shot-zenbakitik, hari-ordena edozein dela ere berdina
fn shot_rng(seed: u64, shot: u64) -> StdRng {
    // splitmix64
    let mut z = seed.wrapping_add(shot.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    StdRng::seed_from_u64(z ^ (z >> 31))
}

// sarrerak: lehen shot-ak stdin-etik irakurtzen ditu eta gordetzen; besteek errepikatzen dituzte
enum Inputs<'a> {
    Record(&'a mut Vec<f64>),
    Replay(&'a [f64], usize),
}

struct Machine<'a> {
    prog: &'a Program,
    // egoera eta oraindik aplikatu gabeko qubit-ateak: ateak sortan aplikatzen dira (fusioa eta
    // cache-blokeak, ikus src/kernel.rs) egoera irakurri edo beste eragiketa bat behar den arte.
    // `state` ez da zuzenean erabiltzen: `st()`-k lehenik sorta aplikatzen du
    state: State,
    pending: Vec<QGate>,
    regs: [f64; NUM_REGS],
    bits: Vec<Option<u8>>,
    readouts: Vec<Option<u32>>,
    stack: Vec<f64>,
    calls: Vec<usize>,
    loops: Vec<(usize, u64)>,
    rng: StdRng,
    cfg: &'a RunConfig,
    cache: &'a MatrixCache,
    out: &'a mut dyn Write,
    inputs: Inputs<'a>,
    steps: u64,
    // laginketarako: pc honetara iristean probabilitateak gorde
    capture_at: Option<usize>,
    captured: Option<Vec<f64>>,
    // GPUak huts egin badu: exekuzioa hurrengo urratsean errore honekin gelditzen da
    gpu_error: Option<String>,
}

pub fn run(
    prog: &Program,
    cfg: &RunConfig,
    out: &mut dyn Write,
) -> Result<RunOutput, RuntimeError> {
    let cache = MatrixCache::default();
    if let Some(m) = &cfg.model
        && prog.n_qubits as usize > m.qubits.len()
    {
        return Err(RuntimeError {
            line: 0,
            message: t!(
                "the program uses {} qubits, but the device model ({}) describes only {}",
                "programak {} qubit erabiltzen ditu, baina gailu-ereduak ({}) {} bakarrik deskribatzen ditu",
                prog.n_qubits,
                m.device,
                m.qubits.len()
            ),
        });
    }
    let has_readout = prog.has_readout();
    let ideal = cfg.noise == 0.0 && cfg.loss == 0.0 && cfg.model.is_none();
    let split = if has_readout && ideal {
        prog.sampling_split()
    } else {
        None
    };
    let method = if !has_readout {
        Method::StateOnly
    } else if split.is_some() {
        Method::Sampled
    } else {
        Method::EveryShot
    };
    let shots = if method == Method::StateOnly {
        1
    } else {
        cfg.shots.max(1)
    };

    // lehen shot-a: irteera erakusten du eta sarrerak gordetzen ditu
    let mut recorded_inputs = Vec::new();
    let mut first = Machine::new(
        prog,
        cfg,
        &cache,
        out,
        Inputs::Record(&mut recorded_inputs),
        shot_rng(cfg.seed, 0),
    )?;
    first.capture_at = split;
    first.run()?;
    let first_record = first.record();
    let first_leakage = first.state.leakage;
    let steps = first.steps;
    let captured = first.captured.take();
    let regs = first.regs;
    let final_state = std::mem::replace(&mut first.state, State::new(0, 0, 2).unwrap());
    drop(first);

    let mut counts: HashMap<Record, u64> = HashMap::new();
    *counts.entry(first_record).or_insert(0) += 1;
    let mut leakage = first_leakage;

    if shots > 1 {
        match method {
            Method::Sampled => {
                let probs = captured.expect("probabilitateak lehen irakurketan gorde dira");
                let split = split.unwrap();
                // banaketa metatua, bilaketa bitarrerako
                // banaketa metatua double-double-n: azken gaiak ere f64 baten zehaztasunarekin
                let mut cumulative = probs;
                let mut acc = crate::dd::Dd::ZERO;
                for p in cumulative.iter_mut() {
                    acc.add(*p);
                    *p = acc.value();
                }
                let total = acc.value();
                let tail = &prog.ops[split..];
                let sampled: HashMap<Record, u64> = (1..shots)
                    .into_par_iter()
                    .fold(HashMap::new, |mut m, shot| {
                        let mut rng = shot_rng(cfg.seed, shot);
                        let r: f64 = rng.random::<f64>() * total;
                        let idx = cumulative
                            .partition_point(|&c| c <= r)
                            .min(cumulative.len() - 1);
                        *m.entry(record_from_index(prog, &final_state, tail, idx))
                            .or_insert(0) += 1;
                        m
                    })
                    .reduce(HashMap::new, merge_counts);
                counts = merge_counts(counts, sampled);
            }
            Method::EveryShot => {
                let results: Vec<Result<(Record, f64), RuntimeError>> = (1..shots)
                    .into_par_iter()
                    .map(|shot| {
                        let mut sink = std::io::sink();
                        let mut m = Machine::new(
                            prog,
                            cfg,
                            &cache,
                            &mut sink,
                            Inputs::Replay(&recorded_inputs, 0),
                            shot_rng(cfg.seed, shot),
                        )?;
                        m.run()?;
                        Ok((m.record(), m.state.leakage))
                    })
                    .collect();
                for r in results {
                    let (rec, leak) = r?;
                    *counts.entry(rec).or_insert(0) += 1;
                    leakage = leakage.max(leak);
                }
            }
            Method::StateOnly => unreachable!(),
        }
    }

    let mut counts: Vec<(Record, u64)> = counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok(RunOutput {
        method,
        shots,
        counts,
        final_state,
        regs,
        leakage,
        steps,
    })
}

fn merge_counts(mut a: HashMap<Record, u64>, b: HashMap<Record, u64>) -> HashMap<Record, u64> {
    for (k, v) in b {
        *a.entry(k).or_insert(0) += v;
    }
    a
}

// laginketa: oinarri-egoera baten indizetik amaierako neurketen emaitzak atera
fn record_from_index(prog: &Program, state: &State, tail: &[Op], idx: usize) -> Record {
    let mut rec = Record {
        bits: vec![None; prog.n_bits as usize],
        readouts: vec![None; prog.n_modes as usize],
    };
    let photons = |o: u32| state.photons(idx, o as usize) as u32;
    for op in tail {
        match op {
            Op::Measure { q, dest, .. } => {
                let v = State::bit(idx, *q as usize);
                match dest {
                    Dest::Own => rec.bits[*q as usize] = Some(v),
                    Dest::Bit(k) => rec.bits[*k as usize] = Some(v),
                    Dest::Reg(_) => {}
                }
            }
            Op::MeasureAll => {
                for q in 0..prog.n_qubits as usize {
                    rec.bits[q] = Some(State::bit(idx, q));
                }
            }
            Op::PhotonCount { o, dest } => {
                if *dest == Dest::Own {
                    rec.readouts[*o as usize] = Some(photons(*o));
                }
            }
            Op::PhotonDetect { o, dest } | Op::Parity { o, dest } => {
                let v = match op {
                    Op::PhotonDetect { .. } => (photons(*o) > 0) as u8,
                    _ => (photons(*o) % 2) as u8,
                };
                match dest {
                    Dest::Own => rec.readouts[*o as usize] = Some(v as u32),
                    Dest::Bit(k) => rec.bits[*k as usize] = Some(v),
                    Dest::Reg(_) => {}
                }
            }
            Op::Coincidence { modes, dest } => {
                let v = modes.iter().all(|m| photons(*m) > 0) as u8;
                if let Dest::Bit(k) = dest {
                    rec.bits[*k as usize] = Some(v);
                }
            }
            _ => {}
        }
    }
    rec
}

impl<'a> Machine<'a> {
    fn new(
        prog: &'a Program,
        cfg: &'a RunConfig,
        cache: &'a MatrixCache,
        out: &'a mut dyn Write,
        inputs: Inputs<'a>,
        rng: StdRng,
    ) -> Result<Machine<'a>, RuntimeError> {
        let state = State::new(
            prog.n_qubits as usize,
            prog.n_modes as usize,
            prog.cutoff as usize,
        )
        .map_err(|message| RuntimeError { line: 0, message })?;
        Ok(Machine {
            prog,
            state,
            pending: Vec::new(),
            regs: [0.0; NUM_REGS],
            bits: vec![None; prog.n_bits as usize],
            readouts: vec![None; prog.n_modes as usize],
            stack: Vec::new(),
            calls: Vec::new(),
            loops: Vec::new(),
            rng,
            cfg,
            cache,
            out,
            inputs,
            steps: 0,
            capture_at: None,
            captured: None,
            gpu_error: None,
        })
    }

    fn record(&self) -> Record {
        Record {
            bits: self.bits.clone(),
            readouts: self.readouts.clone(),
        }
    }

    fn run(&mut self) -> Result<(), RuntimeError> {
        let mut pc = 0usize;
        while pc < self.prog.ops.len() {
            if Some(pc) == self.capture_at && self.captured.is_none() {
                self.captured = Some(self.st().probabilities());
            }
            self.steps += 1;
            if self.steps > self.cfg.max_steps {
                return Err(self.err(pc, t!(
                    "stopped after {} steps; if the program really needs more, raise --max-steps",
                    "{} urratsen ondoren gelditua; programak benetan gehiago behar baditu, handitu --max-steps",
                    self.cfg.max_steps)));
            }
            // gailu-eredu batekin: eragiketa natiboa dela egiaztatu exekutatu aurretik, eta haren zarata
            // aplikatu ondoren
            let device = match self.cfg.model.as_deref() {
                Some(model) => self.device_gate(pc, model)?,
                None => None,
            };
            let flow = self.step(pc)?;
            if let Some((noise, qubits)) = device {
                self.gate_noise(noise, &qubits);
            }
            if let Some(e) = self.gpu_error.take() {
                return Err(self.err(pc, e));
            }
            match flow {
                Flow::Next => pc += 1,
                Flow::Goto(t) => pc = t,
                Flow::Stop => break,
            }
        }
        self.flush();
        match self.gpu_error.take() {
            Some(e) => Err(RuntimeError {
                line: 0,
                message: e,
            }),
            None => Ok(()),
        }
    }

    fn err(&self, pc: usize, message: impl Into<String>) -> RuntimeError {
        RuntimeError {
            line: self.prog.line_of(pc),
            message: message.into(),
        }
    }

    fn uniform(&mut self) -> f64 {
        self.rng.random::<f64>()
    }

    // ------------------------------------------------------------------ qubitak

    fn gate(&mut self, q: u32, gate: Gate, controls: usize) {
        if gate == Gate::X {
            self.queue(QGate::Mat {
                q: q as usize,
                m: X_MATRIX,
                controls,
            });
        } else {
            self.queue(QGate::Mat {
                q: q as usize,
                m: gate.matrix(),
                controls,
            });
        }
    }

    // qubit-ate bat sortara; sorta luzeegia bada, aplikatu
    fn queue(&mut self, g: QGate) {
        self.pending.push(g);
        if self.pending.len() >= MAX_PENDING {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        match &self.cfg.gpu {
            Some(gpu) => {
                if let Err(e) = self.state.apply_gates_on(gpu, &self.pending) {
                    self.gpu_error.get_or_insert(e);
                }
            }
            None => self.state.apply_gates(&self.pending),
        }
        self.pending.clear();
    }

    // egoera, zain dauden ateak aplikatu ondoren
    fn st(&mut self) -> &mut State {
        self.flush();
        &mut self.state
    }

    fn mask(qs: &[u32]) -> usize {
        qs.iter().fold(0usize, |m, q| m | (1usize << q))
    }

    fn measure_z(&mut self, q: u32) -> u8 {
        let r = self.uniform();
        self.st().measure_qubit(q as usize, r)
    }

    // neurketa X edo Y oinarrian: z-ra biratu, neurtu, itzuli
    fn measure_basis(&mut self, q: u32, basis: Basis) -> u8 {
        self.rotate_into(q, basis);
        let v = self.measure_z(q);
        self.rotate_back(q, basis);
        v
    }

    fn rotate_into(&mut self, q: u32, basis: Basis) {
        match basis {
            Basis::X => self.gate(q, Gate::H, 0),
            Basis::Y => {
                self.gate(q, Gate::Sdg, 0);
                self.gate(q, Gate::H, 0);
            }
            Basis::Z => {}
        }
    }

    fn rotate_back(&mut self, q: u32, basis: Basis) {
        match basis {
            Basis::X => self.gate(q, Gate::H, 0),
            Basis::Y => {
                self.gate(q, Gate::H, 0);
                self.gate(q, Gate::S, 0);
            }
            Basis::Z => {}
        }
    }

    // zarata: ukitutako qubit bakoitzean, p probabilitatez X, Y edo Z ausazkoa (depolarizazioa)
    fn depolarize(&mut self, qubits: &[u32]) {
        let p = self.cfg.noise;
        if p <= 0.0 {
            return;
        }
        for &q in qubits {
            if self.uniform() < p {
                let g = match self.rng.random_range(0..3) {
                    0 => Gate::X,
                    1 => Gate::Y,
                    _ => Gate::Z,
                };
                self.gate(q, g, 0);
            }
        }
    }

    // gailu-eredua: eragiketa hau gailuan egin daitekeen. ate natibo bat bada, haren zarata eta qubitak
    fn device_gate(
        &self,
        pc: usize,
        model: &'a NoiseModel,
    ) -> Result<Option<(&'a GateNoise, Vec<u32>)>, RuntimeError> {
        match crate::noise::native(&self.prog.ops[pc]) {
            Native::Free | Native::Readout => Ok(None),
            Native::Unsupported(why) => Err(self.err(pc, t!(
                "{}; with a device model ({}) a program can use only the device's native gates: transpile it for the device first (tools/run_on_ibm.py does)",
                "{}; gailu-eredu batekin ({}) programak gailuaren ate natiboak bakarrik erabil ditzake: lehenik gailurako transpilatu (tools/run_on_ibm.py-k egiten du)",
                why, model.device))),
            Native::Gate(name, qubits) => match model.gate(&name, &qubits) {
                Some(g) => Ok(Some((g, qubits))),
                None => {
                    let on: Vec<String> = qubits.iter().map(|q| format!("q{}", q)).collect();
                    Err(self.err(pc, t!(
                        "the device model ({}) has no '{}' on {}: the device can't run it directly (its native gates: {}). transpile the program for the device first (tools/run_on_ibm.py does)",
                        "gailu-ereduak ({}) ez du '{}' aterik {} qubitetan: gailuak ezin du zuzenean egin (bere ate natiboak: {}). lehenik programa gailurako transpilatu (tools/run_on_ibm.py-k egiten du)",
                        model.device, name, on.join(" "), model.gate_names().join(", "))))
                }
            },
        }
    }

    // ate natibo baten zarata, aer-ek bezala: lehenik depolarizazioa (identitatea ez den n qubiteko
    // pauli bat, uniformeki), gero qubit bakoitzaren T1/T2 erlaxazioa atearen iraupenean
    fn gate_noise(&mut self, g: &GateNoise, qubits: &[u32]) {
        if g.pauli > 0.0 && self.uniform() < g.pauli {
            let k = self.rng.random_range(1..1u32 << (2 * qubits.len()));
            for (i, &q) in qubits.iter().enumerate() {
                match (k >> (2 * i)) & 3 {
                    1 => self.gate(q, Gate::X, 0),
                    2 => self.gate(q, Gate::Y, 0),
                    3 => self.gate(q, Gate::Z, 0),
                    _ => {}
                }
            }
        }
        for (r, &q) in g.relax.iter().zip(qubits) {
            self.relax(q, *r);
        }
    }

    // T1/T2 kanala ibilbide zehatz gisa. anplitude-moteltzea: jauzia (|1> -> |0>) gamma·P1 probabilitatez;
    // bestela diag(1, √(1-gamma)) eboluzioa, normalizatuta. gero desfasea: Z bat z probabilitatez
    fn relax(&mut self, q: u32, r: Relax) {
        if r.gamma > 0.0 {
            let u = self.uniform();
            let st = self.st();
            let p1 = st.prob_one(q as usize).clamp(0.0, 1.0);
            if u < r.gamma * p1 {
                st.collapse_qubit(q as usize, 1, p1);
                self.gate(q, Gate::X, 0);
            } else {
                let s = 1.0 / (1.0 - r.gamma * p1).sqrt();
                self.queue(QGate::Mat {
                    q: q as usize,
                    m: [
                        C::new(s, 0.0),
                        C::new(0.0, 0.0),
                        C::new(0.0, 0.0),
                        C::new(s * (1.0 - r.gamma).sqrt(), 0.0),
                    ],
                    controls: 0,
                });
            }
        }
        if r.z > 0.0 && self.uniform() < r.z {
            self.gate(q, Gate::Z, 0);
        }
    }

    // gailu-eredua: neurketaren emaitza klasikoa irakurketa-errorearekin (egoera kuantikoa ez da aldatzen)
    fn readout(&mut self, q: u32, v: u8) -> u8 {
        let cfg = self.cfg;
        let Some(model) = cfg.model.as_deref() else {
            return v;
        };
        let qn = &model.qubits[q as usize];
        let flip = if v == 0 { qn.p0m1 } else { qn.p1m0 };
        if flip > 0.0 && self.uniform() < flip {
            1 - v
        } else {
            v
        }
    }

    fn lose_photons(&mut self, modes: &[u32]) {
        let g = self.cfg.loss;
        if g <= 0.0 {
            return;
        }
        for &o in modes {
            let r = self.uniform();
            self.st().photon_loss(o as usize, g, r);
        }
    }

    // emaitza helmugan idatzi; `own` neurtutako objektuaren irakurketa da
    fn write_bit(&mut self, dest: Dest, own_bit: Option<u32>, own_mode: Option<u32>, v: u32) {
        match dest {
            Dest::Own => {
                if let Some(q) = own_bit {
                    self.bits[q as usize] = Some(v as u8);
                } else if let Some(o) = own_mode {
                    self.readouts[o as usize] = Some(v);
                }
            }
            Dest::Bit(k) => self.bits[k as usize] = Some(v as u8),
            Dest::Reg(r) => self.regs[r as usize] = v as f64,
        }
    }

    // ------------------------------------------------------------------ moduak

    fn mode_matrix(&mut self, o: u32, kind: u8, a: f64, b: f64) {
        let d = self.state.cutoff();
        let m = self.cache.get(kind, a, b, d, || match kind {
            1 => fock::displacement(C::new(a, b), d),
            _ => fock::squeezing(a, b, d),
        });
        self.st().apply_mode_matrix(o as usize, &m);
    }

    fn print(&mut self, s: &str) {
        let _ = self.out.write_all(s.as_bytes());
    }

    fn value_of(&self, pc: usize, src: Source) -> Result<f64, RuntimeError> {
        match src {
            Source::Reg(r) => Ok(self.regs[r as usize]),
            Source::Bit(k) => match self.bits.get(k as usize).copied().flatten() {
                Some(v) => Ok(v as f64),
                None => Err(self.err(
                    pc,
                    t!(
                        "c{} is read before anything was written to it",
                        "c{} irakurri da ezer idatzi aurretik",
                        k
                    ),
                )),
            },
        }
    }

    // ------------------------------------------------------------------ urrats bat

    fn step(&mut self, pc: usize) -> Result<Flow, RuntimeError> {
        let prog = self.prog;
        let op = &prog.ops[pc];
        match op {
            Op::Gate { q, gate } => {
                self.gate(*q, *gate, 0);
                self.depolarize(&[*q]);
            }
            Op::Controlled { controls, q, gate } => {
                self.gate(*q, *gate, Self::mask(controls));
                self.depolarize(&op.gate_qubits());
            }
            Op::PhaseAll { qubits, theta } => {
                self.queue(QGate::phase(Self::mask(qubits), *theta));
                self.depolarize(qubits);
            }
            Op::Swap { a, b, controls } => {
                self.queue(QGate::Swap {
                    a: *a as usize,
                    b: *b as usize,
                    controls: Self::mask(controls),
                });
                self.depolarize(&op.gate_qubits());
            }
            Op::Bell { a, b } => {
                self.gate(*a, Gate::H, 0);
                self.gate(*b, Gate::X, 1 << a);
                self.depolarize(&[*a, *b]);
            }
            Op::Ghz { qubits } => {
                let first = qubits[0];
                self.gate(first, Gate::H, 0);
                for &q in &qubits[1..] {
                    self.gate(q, Gate::X, 1 << first);
                }
                self.depolarize(qubits);
            }
            Op::Cluster { qubits } => {
                for &q in qubits {
                    self.gate(q, Gate::H, 0);
                }
                for w in qubits.windows(2) {
                    self.queue(QGate::phase(Self::mask(w), std::f64::consts::PI));
                }
                self.depolarize(qubits);
            }
            Op::EntangleSwap { a: _, b, c, d } => {
                // bell-neurketa b eta c gainean, gero d zuzendu
                let (b, c, d) = (*b, *c, *d);
                self.gate(c, Gate::X, 1 << b);
                self.gate(b, Gate::H, 0);
                let mb = self.measure_z(b);
                let mc = self.measure_z(c);
                self.bits[b as usize] = Some(mb);
                self.bits[c as usize] = Some(mc);
                if mc == 1 {
                    self.gate(d, Gate::X, 0);
                }
                if mb == 1 {
                    self.gate(d, Gate::Z, 0);
                }
            }
            Op::Measure { q, basis, dest } => {
                let v = self.measure_basis(*q, *basis);
                let v = self.readout(*q, v);
                self.write_bit(*dest, Some(*q), None, v as u32);
            }
            Op::MeasureAll => {
                for q in 0..self.prog.n_qubits {
                    let v = self.measure_z(q);
                    let v = self.readout(q, v);
                    self.bits[q as usize] = Some(v);
                }
            }
            Op::Reset { q } => {
                if self.measure_z(*q) == 1 {
                    self.gate(*q, Gate::X, 0);
                }
            }
            Op::ResetAll => {
                // zain dauden ateak ez dira aplikatu behar: egoera osoa berrezartzen da
                self.pending.clear();
                let leak = self.state.leakage;
                self.state = State::new(self.state.n_qubits(), self.state.n_modes(), self.state.cutoff())
                    .expect("lehengo tamaina bera");
                self.state.leakage = leak;
            }
            Op::ToBasis { q, basis } => self.rotate_into(*q, *basis),

            Op::Emit { q, o } => {
                self.st().emit(*q as usize, *o as usize);
                self.depolarize(&[*q]);
                self.lose_photons(&[*o]);
            }
            Op::PhotonCount { o, dest } => {
                let r = self.uniform();
                let n = self.st().measure_photons(*o as usize, r) as u32;
                match dest {
                    Dest::Reg(reg) => self.regs[*reg as usize] = n as f64,
                    _ => self.readouts[*o as usize] = Some(n),
                }
            }
            Op::PhotonDetect { o, dest } => {
                let r = self.uniform();
                let v = self.st().measure_click(*o as usize, r);
                self.write_bit(*dest, None, Some(*o), v as u32);
            }
            Op::Parity { o, dest } => {
                let r = self.uniform();
                let v = self.st().measure_parity(*o as usize, r);
                self.write_bit(*dest, None, Some(*o), v as u32);
            }
            Op::Coincidence { modes, dest } => {
                let mut all = 1u32;
                for &m in modes {
                    let r = self.uniform();
                    if self.st().measure_click(m as usize, r) == 0 {
                        all = 0;
                    }
                }
                self.write_bit(*dest, None, None, all);
            }
            Op::Displace { o, re, im } => {
                self.mode_matrix(*o, 1, *re, *im);
                self.lose_photons(&[*o]);
            }
            Op::DisplaceReg { o, r } => {
                let alpha = self.regs[*r as usize];
                self.mode_matrix(*o, 1, alpha, 0.0);
                self.lose_photons(&[*o]);
            }
            Op::Squeeze { o, r, phi } => {
                self.mode_matrix(*o, 2, *r, *phi);
                self.lose_photons(&[*o]);
            }
            Op::SqueezeReg { o, reg } => {
                let r = self.regs[*reg as usize];
                self.mode_matrix(*o, 2, r, 0.0);
                self.lose_photons(&[*o]);
            }
            Op::ModePhase { o, theta } => {
                let f: Vec<C> = (0..self.state.cutoff())
                    .map(|n| C::from_polar(1.0, theta * n as f64))
                    .collect();
                self.st().apply_mode_diagonal(*o as usize, &f);
                self.lose_photons(&[*o]);
            }
            Op::Kerr { o, kappa } => {
                let f: Vec<C> = (0..self.state.cutoff())
                    .map(|n| C::from_polar(1.0, kappa * (n * n) as f64))
                    .collect();
                self.st().apply_mode_diagonal(*o as usize, &f);
                self.lose_photons(&[*o]);
            }
            Op::CrossKerr { a, b, kappa } => {
                let d = self.state.cutoff();
                let f: Vec<C> = (0..d * d)
                    .map(|i| C::from_polar(1.0, kappa * ((i / d) * (i % d)) as f64))
                    .collect();
                self.st().apply_two_mode_diagonal(*a as usize, *b as usize, &f);
                self.lose_photons(&[*a, *b]);
            }
            Op::BeamSplitter { a, b, theta, phi } => {
                let d = self.state.cutoff();
                let (t, p) = (*theta, *phi);
                let m = self.cache.get(3, t, p, d, || fock::beamsplitter(t, p, d));
                self.st().apply_two_mode_matrix(*a as usize, *b as usize, &m);
                self.lose_photons(&[*a, *b]);
            }
            Op::ModeSwap { a, b } => {
                self.st().swap_modes(*a as usize, *b as usize);
                self.lose_photons(&[*a, *b]);
            }
            Op::AddPhoton { o } => {
                let o = *o as usize;
                let dist = self.st().photon_distribution(o);
                let d = dist.len();
                let expected = crate::dd::sum(dist.iter().enumerate().map(|(n, p)| (n + 1) as f64 * p));
                let dropped = d as f64 * dist[d - 1];
                self.st().ladder(o, true);
                self.st().normalize();
                let leak = dropped / expected;
                if leak > 1e-13 {
                    self.st().leakage += leak;
                }
                self.lose_photons(&[o as u32]);
            }
            Op::SubPhoton { o } => {
                let mean = self.st().mean_photons(*o as usize);
                if mean < 1e-14 {
                    return Err(self.err(pc, t!(
                        "SUBPHOTON on o{}, which holds no photons: removing a photon from vacuum never succeeds",
                        "SUBPHOTON o{}-n, fotoirik ez duena: hutsetik fotoi bat kentzeak ez du inoiz lortzen",
                        o)));
                }
                self.st().ladder(*o as usize, false);
                self.st().normalize();
                self.lose_photons(&[*o]);
            }
            Op::Loss { o, gamma } => {
                let r = self.uniform();
                self.st().photon_loss(*o as usize, *gamma, r);
            }
            Op::Fock { o, n } => {
                // modua neurtu (hau da, haren edukia baztertu) eta n fotoira eraman
                let r = self.uniform();
                let m = self.st().measure_photons(*o as usize, r);
                self.st().move_photons(*o as usize, m, *n as usize);
            }
            Op::Dispersive { q, o, theta } => {
                self.st().dispersive(*q as usize, *o as usize, *theta);
                self.depolarize(&[*q]);
                self.lose_photons(&[*o]);
            }

            Op::DumpState => {
                let text = crate::report::state_table(self.st(), 16);
                self.print(&t!("state at line {}:\n{}", "egoera {}. lerroan:\n{}", self.prog.line_of(pc), text));
            }
            Op::DumpRegs => {
                let mut s = t!("registers:", "erregistroak:");
                for (i, v) in self.regs.iter().enumerate() {
                    if *v != 0.0 {
                        s.push_str(&format!(" R{}={}", i, fmt_value(*v)));
                    }
                }
                if self.regs.iter().all(|v| *v == 0.0) {
                    s.push_str(ts!(" all zero", " guztiak zero"));
                }
                s.push('\n');
                self.print(&s);
            }
            Op::Expect { q, basis, dest } => {
                let b = self.st().bloch(*q as usize);
                let v = match basis {
                    Basis::X => b[0],
                    Basis::Y => b[1],
                    Basis::Z => b[2],
                };
                match dest {
                    Some(r) => self.regs[*r as usize] = v,
                    None => self.print(&t!("<{}> of q{} = {:.6}\n", "q{1}-ren <{0}> = {2:.6}\n", basis.name(), q, v)),
                }
            }
            Op::BellCheck { a, b, dest } => {
                let f = self.st().bell_fidelity(*a as usize, *b as usize);
                match dest {
                    Some(r) => self.regs[*r as usize] = f,
                    None => self.print(&t!("Bell fidelity of q{} q{} = {:.6}\n", "q{} q{} qubiten bell-fideltasuna = {:.6}\n", a, b, f)),
                }
            }
            Op::MeanPhoton { o, dest } => {
                let n = self.st().mean_photons(*o as usize);
                match dest {
                    Some(r) => self.regs[*r as usize] = n,
                    None => self.print(&t!("<n> of o{} = {:.6}\n", "o{}-ren <n> = {:.6}\n", o, n)),
                }
            }

            Op::Set { r, value } => self.regs[*r as usize] = *value,
            Op::Copy { dst, src } => self.regs[*dst as usize] = self.regs[*src as usize],
            Op::Bin { op, dst, a, b } => {
                let (x, y) = (self.regs[*a as usize], self.regs[*b as usize]);
                let v = match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    BinOp::Mul => x * y,
                    BinOp::Div => {
                        if y == 0.0 {
                            return Err(self.err(pc, t!("division by zero (R{} is 0)", "zeroz zatitzea (R{} 0 da)", b)));
                        }
                        x / y
                    }
                    BinOp::And => ((x as i64) & (y as i64)) as f64,
                    BinOp::Or => ((x as i64) | (y as i64)) as f64,
                    BinOp::Xor => ((x as i64) ^ (y as i64)) as f64,
                    BinOp::Shl => ((x as i64).wrapping_shl(y.clamp(0.0, 63.0) as u32)) as f64,
                    BinOp::Shr => ((x as i64).wrapping_shr(y.clamp(0.0, 63.0) as u32)) as f64,
                };
                self.regs[*dst as usize] = v;
            }
            Op::Un { op, dst, src } => {
                let x = self.regs[*src as usize];
                let v = match op {
                    UnOp::Sqrt => {
                        if x < 0.0 {
                            return Err(self.err(pc, t!("square root of a negative number ({})", "zenbaki negatibo baten erro karratua ({})", fmt_value(x))));
                        }
                        x.sqrt()
                    }
                    UnOp::Exp => x.exp(),
                    UnOp::Log => {
                        if x <= 0.0 {
                            return Err(self.err(pc, t!("logarithm of a number that is not positive ({})", "positiboa ez den zenbaki baten logaritmoa ({})", fmt_value(x))));
                        }
                        x.ln()
                    }
                    UnOp::Not => (!(x as i64)) as f64,
                };
                self.regs[*dst as usize] = v;
            }
            Op::Rand { r } => self.regs[*r as usize] = self.uniform(),
            Op::Seed { seed } => self.rng = StdRng::seed_from_u64(*seed),
            Op::Input { r } => {
                let v = match &mut self.inputs {
                    Inputs::Record(list) => {
                        let mut line = String::new();
                        let _ = self.out.flush();
                        std::io::stdin().lock().read_line(&mut line).ok();
                        let v = line.trim().parse::<f64>().map_err(|_| RuntimeError {
                            line: self.prog.line_of(pc),
                            message: t!("INPUT expected a number, got '{}'", "INPUT-ek zenbaki bat espero zuen, baina '{}' jaso du", line.trim()),
                        })?;
                        list.push(v);
                        v
                    }
                    Inputs::Replay(list, i) => {
                        let v = list.get(*i).copied().unwrap_or(0.0);
                        *i += 1;
                        v
                    }
                };
                self.regs[*r as usize] = v;
            }
            Op::Time { r } => {
                self.regs[*r as usize] = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0)
            }
            Op::Push { r } => {
                if self.stack.len() >= 1 << 20 {
                    return Err(self.err(pc, t!("stack overflow (more than a million values pushed)", "pila gainezka (milioi bat balio baino gehiago sartuta)")));
                }
                self.stack.push(self.regs[*r as usize]);
            }
            Op::Pop { r } => match self.stack.pop() {
                Some(v) => self.regs[*r as usize] = v,
                None => return Err(self.err(pc, t!("POP on an empty stack", "POP pila huts batean"))),
            },
            Op::PackBits { dst, bits } => {
                let v = bits.iter().enumerate().fold(0u64, |acc, (i, k)| {
                    let b = self.bits.get(*k as usize).copied().flatten().unwrap_or(0) as u64;
                    acc | (b << i)
                });
                self.regs[*dst as usize] = v as f64;
            }

            Op::Jump { to } => return Ok(Flow::Goto(*to)),
            Op::JumpIf { src, value, to } => {
                if self.value_of(pc, *src)? == *value {
                    return Ok(Flow::Goto(*to));
                }
            }
            Op::Branch { cmp, a, b, to } => {
                let (x, y) = (self.regs[*a as usize], self.regs[*b as usize]);
                let take = match cmp {
                    Cmp::Eq => x == y,
                    Cmp::Ne => x != y,
                    Cmp::Gt => x > y,
                    Cmp::Lt => x < y,
                };
                if take {
                    return Ok(Flow::Goto(*to));
                }
            }
            Op::Call { to } => {
                if self.calls.len() >= 100_000 {
                    return Err(self.err(pc, t!("too many nested CALLs (more than 100000)", "CALL habiaratu gehiegi (100000 baino gehiago)")));
                }
                self.calls.push(pc + 1);
                return Ok(Flow::Goto(*to));
            }
            Op::Return => match self.calls.pop() {
                Some(back) => return Ok(Flow::Goto(back)),
                None => return Err(self.err(pc, t!("RETSUB without a CALL", "RETSUB CALL gabe"))),
            },
            Op::LoopStart { count, end } => {
                let n = match count {
                    Count::Lit(n) => *n,
                    Count::Reg(r) => {
                        let v = self.regs[*r as usize];
                        if v < 0.0 || v.fract() != 0.0 {
                            return Err(self.err(pc, t!(
                                "a loop count must be a whole number of 0 or more; R{} holds {}",
                                "begizta-kopuruak 0 edo gehiagoko zenbaki osoa izan behar du; R{}-k {} du",
                                r, fmt_value(v))));
                        }
                        v as u64
                    }
                };
                if n == 0 {
                    return Ok(Flow::Goto(*end));
                }
                if self.loops.len() >= 10_000 {
                    return Err(self.err(pc, t!("too many nested loops", "begizta habiaratu gehiegi")));
                }
                self.loops.push((pc, n));
            }
            Op::LoopEnd { start } => {
                match self.loops.last_mut() {
                    Some((s, remaining)) if *s == *start => {
                        *remaining -= 1;
                        if *remaining > 0 {
                            return Ok(Flow::Goto(start + 1));
                        }
                        self.loops.pop();
                    }
                    _ => return Err(self.err(pc, t!(
                        "LOOPEND reached without running its LOOPSTART (did a jump enter the loop?)",
                        "LOOPEND-era iritsi da bere LOOPSTART exekutatu gabe (jauzi batek begiztan sartu al du?)"))),
                }
            }
            Op::Halt => return Ok(Flow::Stop),
            Op::Barrier => {}
            Op::Print { text, newline } => {
                let s = if *newline { format!("{}\n", text) } else { text.clone() };
                self.print(&s);
            }
            Op::Printf { format, regs } => {
                let mut s = String::new();
                let mut parts = format.split("{}");
                if let Some(first) = parts.next() {
                    s.push_str(first);
                }
                for (part, r) in parts.zip(regs) {
                    s.push_str(&fmt_value(self.regs[*r as usize]));
                    s.push_str(part);
                }
                self.print(&s);
            }
            Op::CharOut { r } => {
                let v = self.regs[*r as usize];
                match char::from_u32(v as u32).filter(|_| v >= 0.0 && v.fract() == 0.0) {
                    Some(c) => self.print(&c.to_string()),
                    None => return Err(self.err(pc, t!("R{} holds {}, which is not a character code", "R{}-k {} du, eta hori ez da karaktere-kode bat", r, fmt_value(v)))),
                }
            }
            Op::Log { q, text } => self.print(&format!("[q{}] {}\n", q, text)),
        }
        Ok(Flow::Next)
    }
}

enum Flow {
    Next,
    Goto(usize),
    Stop,
}

// zenbaki osoak hamartarrik gabe, besteak bere horretan
pub fn fmt_value(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}
