// fitxategi honek programa exekutatzen du (makina birtuala) eta shot-ak kudeatzen ditu.
//
// shot bat = programaren exekuzio oso bat. hiru modu daude, eta txostenak beti esaten du zein
// erabili den:
//   - sampled: neurketa guztiak amaieran badaude eta ez badago ausazkotasunik aurretik, behin
//     simulatzen da eta shot guztiak egoera horretatik lagintzen dira (zehatza eta azkarra)
//   - every shot: bestela, shot bakoitza hasieratik exekutatzen da (zarata edo gailu-eredu batekin beti)
//   - state only: programak ez badu ezer neurtzen, behin exekutatzen da eta egoera zehatza erakusten da

use crate::dist::DistState;
use crate::fock::{self, Mat};
use crate::gpu::Gpu;
use crate::kernel::QGate;
use crate::noise::{GateNoise, Native, NoiseModel, Relax};
use crate::program::{Basis, BinOp, Cmp, Count, Dest, Gate, NUM_REGS, Op, Program, Source, UnOp};
use crate::quad::{self, QuadGate, QuadState};
use crate::state::State;
use crate::team::{self, Team};
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
    // qubit-egoeraren doitasuna: 64 bit (lehenetsia) edo 128 bit (`--precision quad`, src/quad.rs)
    pub precision: Precision,
    // prozesu-taldea (`--mpi`): shot-ak rank-en artean banatzen dira eta 0. rank-ean biltzen dira
    pub team: Option<Arc<dyn Team>>,
    // `--distribute`: egoera bakarra taldeko prozesuen artean banatuta (src/dist.rs), shot-ak denek batera
    pub distribute: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    #[default]
    Double,
    Quad,
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
            precision: Precision::Double,
            team: None,
            distribute: false,
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
    // |1 - ‖ψ‖²| lehen shot-aren amaieran, egoeraren doitasun osoan (128 bitekoa quad moduan)
    pub norm_error: f64,
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
    // 128 biteko modua: qubit-egoera eta haren ate-sorta (orduan `state` hutsik dago)
    quad: Option<QuadState>,
    // egoera banatua (`--distribute`): ateak `pending`-etik, orduan `state` hutsik dago
    dist: Option<DistState>,
    // laginketarako egoera banatuan: rank bakoitzaren probabilitate-batura
    captured_totals: Option<Vec<f64>>,
    pending_quad: Vec<QuadGate>,
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
    if cfg.distribute {
        let why = if cfg.team.is_none() {
            Some(ts!("it needs --mpi", "--mpi behar du"))
        } else if prog.n_modes > 0 {
            Some(ts!(
                "the program uses optical modes",
                "programak modu optikoak erabiltzen ditu"
            ))
        } else if cfg.precision == Precision::Quad {
            Some(ts!(
                "it runs in 64 bits, not with --precision quad",
                "64 bitetan dabil, ez --precision quad-ekin"
            ))
        } else if cfg.model.is_some() {
            Some(ts!(
                "device noise models are not distributed",
                "gailuen zarata-ereduak ez daude banatuta"
            ))
        } else {
            None
        };
        if let Some(why) = why {
            return Err(RuntimeError {
                line: 0,
                message: t!(
                    "--distribute runs qubit programs over MPI processes: {}",
                    "--distribute-k qubit-programak MPI prozesuetan exekutatzen ditu: {}",
                    why
                ),
            });
        }
    }
    if cfg.precision == Precision::Quad {
        let why = if prog.n_modes > 0 {
            Some(ts!(
                "the program uses optical modes",
                "programak modu optikoak erabiltzen ditu"
            ))
        } else if cfg.gpu.is_some() {
            Some(ts!(
                "--gpu computes in 64 bits",
                "--gpu-k 64 bitetan kalkulatzen du"
            ))
        } else if cfg.model.is_some() {
            Some(ts!(
                "device noise models run in 64 bits",
                "gailuen zarata-ereduak 64 bitetan dabiltza"
            ))
        } else {
            None
        };
        if let Some(why) = why {
            return Err(RuntimeError {
                line: 0,
                message: t!(
                    "--precision quad (128-bit) runs qubit programs only: {}",
                    "--precision quad-ek (128 bit) qubit-programak bakarrik exekutatzen ditu: {}",
                    why
                ),
            });
        }
    }
    // egoera sortzean huts egin dezakeena orain, edozein komunikazioren aurretik: rank guztiek berdin kalkulatzen
    // dute, beraz denek huts egiten dute batera eta inor ez da zain geratzen
    let precheck = if cfg.precision == Precision::Quad {
        let dim = 1u128.checked_shl(prog.n_qubits).unwrap_or(u128::MAX);
        crate::state::check_memory(
            &t!(
                "{} qubits in 128-bit precision: {} amplitudes",
                "{} qubit 128 biteko doitasunean: {} anplitude",
                prog.n_qubits,
                dim
            ),
            dim.saturating_mul(32),
        )
    } else if cfg.distribute {
        DistState::check(
            prog.n_qubits as usize,
            cfg.team.as_ref().map_or(1, |t| t.size()),
        )
        .map(|_| ())
    } else {
        State::check(
            prog.n_qubits as usize,
            prog.n_modes as usize,
            prog.cutoff as usize,
        )
        .map(|_| ())
    };
    precheck.map_err(|message| RuntimeError { line: 0, message })?;
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
    // laginketak probabilitate-zerrenda bat behar du egoeraren ondoan (8 byte anplitude bakoitzeko)
    if method == Method::Sampled
        && let Some(dim) = prog.state_dim()
    {
        let per = if cfg.precision == Precision::Quad {
            40
        } else {
            24
        };
        let dim = if cfg.distribute {
            dim / cfg.team.as_ref().map_or(1, |t| t.size() as u128)
        } else {
            dim
        };
        crate::state::check_memory(
            &t!(
                "sampling {} amplitudes (the state and its probabilities)",
                "{} anplitude laginkatzeak (egoera eta haren probabilitateak)",
                dim
            ),
            dim.saturating_mul(per),
        )
        .map_err(|message| RuntimeError { line: 0, message })?;
    }
    let shots = if method == Method::StateOnly {
        1
    } else {
        cfg.shots.max(1)
    };

    // lehen shot-a: irteera erakusten du eta sarrerak gordetzen ditu
    // taldean: rank guztiek lehen shot-a exekutatzen dute (egoera eta probabilitateak behar dituzte), baina
    // 0.ak bakarrik idazten du irteera, eta haren sarrerak besteei zabaltzen zaizkie
    let team = cfg.team.as_deref();
    let (rank, ranks) = team.map_or((0, 1), |t| (t.rank(), t.size()));
    let mut recorded_inputs = Vec::new();
    // shot-ak banatuta: 0.ak lehen shot-a amaitzean zabaltzen ditu sarrerak. egoera banatuan, berriz, prozesu
    // guztiak batera exekutatzen dira, eta INPUT bakoitza unean bertan zabaltzen da (ikus Op::Input)
    let spread_inputs = team.is_some() && !cfg.distribute;
    let mut replay = Vec::new();
    if let Some(t) = team
        && spread_inputs
        && rank != 0
    {
        replay = f64s(&t.broadcast(Vec::new()));
    }
    let mut sink = std::io::sink();
    let first_out: &mut dyn Write = if rank == 0 { out } else { &mut sink };
    let mut first = Machine::new(
        prog,
        cfg,
        &cache,
        first_out,
        if rank == 0 || cfg.distribute {
            Inputs::Record(&mut recorded_inputs)
        } else {
            Inputs::Replay(&replay, 0)
        },
        shot_rng(cfg.seed, 0),
    )?;
    first.capture_at = split;
    let first_result = first.run();
    if let Some(t) = team
        && spread_inputs
        && rank == 0
    {
        let inputs = match &first.inputs {
            Inputs::Record(list) => list.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Inputs::Replay(..) => Vec::new(),
        };
        t.broadcast(inputs);
    }
    first_result?;
    let first_record = first.record();
    let first_leakage = first.state.leakage;
    let steps = first.steps;
    let captured = first.captured.take();
    let regs = first.regs;
    let captured_totals = first.captured_totals.take();
    let local_bits = first.dist.as_ref().map_or(0, |d| d.local_bits());
    let (final_state, norm_error) = match (first.quad.take(), first.dist.take()) {
        (Some(q), _) => (q.to_state(), q.norm_error()),
        (None, Some(d)) => {
            let e = d.norm_error();
            (
                d.gather().unwrap_or_else(|| State::new(0, 0, 2).unwrap()),
                e,
            )
        }
        (None, None) => {
            let s = std::mem::replace(&mut first.state, State::new(0, 0, 2).unwrap());
            let e = s.norm_error();
            (s, e)
        }
    };
    drop(first);

    let mut counts: HashMap<Record, u64> = HashMap::new();
    if rank == 0 {
        *counts.entry(first_record).or_insert(0) += 1;
    }
    let mut leakage = first_leakage;
    // rank honen shot-ak: [lo, hi), 1..shots tartea zati jarraituetan banatuta. shot bakoitzak bere hazia
    // du, beraz emaitza ez dago rank-kopuruaren menpe
    // (egoera banatuan, prozesu guztiek shot guztiak egiten dituzte batera)
    let (lo, hi) = if cfg.distribute {
        (1, shots)
    } else {
        (
            1 + (shots - 1) * rank as u64 / ranks as u64,
            1 + (shots - 1) * (rank as u64 + 1) / ranks as u64,
        )
    };
    let mut failed: Option<RuntimeError> = None;

    if shots > 1 {
        match method {
            Method::Sampled if cfg.distribute => {
                // egoera banatua: r rank bakoitzaren tartean bilatzen da (rank-en baturak ordenan); jabeak
                // bakarrik kalkulatzen du emaitza, eta zenbaketak 0.ean biltzen dira
                let probs = captured.expect("probabilitateak lehen irakurketan gorde dira");
                let totals = captured_totals.expect("rank-en baturak");
                let mut offsets = Vec::with_capacity(totals.len() + 1);
                let mut acc = crate::dd::Dd::ZERO;
                offsets.push(0.0);
                for t in &totals {
                    acc.add(*t);
                    offsets.push(acc.value());
                }
                let total = acc.value();
                let mut cumulative = probs;
                let mut run = crate::dd::Dd::ZERO;
                for p in cumulative.iter_mut() {
                    run.add(*p);
                    *p = run.value();
                }
                let tail = &prog.ops[split.unwrap()..];
                let sampled: HashMap<Record, u64> = (lo..hi)
                    .into_par_iter()
                    .fold(HashMap::new, |mut m, shot| {
                        let mut rng = shot_rng(cfg.seed, shot);
                        let r: f64 = rng.random::<f64>() * total;
                        let owner = offsets[1..].partition_point(|&o| o <= r).min(ranks - 1);
                        if owner == rank {
                            let idx = cumulative
                                .partition_point(|&c| c <= r - offsets[rank])
                                .min(cumulative.len() - 1);
                            let global = (rank << local_bits) | idx;
                            *m.entry(record_from_index(prog, &final_state, tail, global))
                                .or_insert(0) += 1;
                        }
                        m
                    })
                    .reduce(HashMap::new, merge_counts);
                counts = merge_counts(counts, sampled);
            }
            Method::EveryShot if cfg.distribute => {
                // shot bakoitza prozesu guztien artean, bata bestearen atzetik (sarrerak prozesu guztiek
                // gorde dituzte lehen shot-ean)
                let shared_inputs: &[f64] = &recorded_inputs;
                for shot in lo..hi {
                    let mut sink = std::io::sink();
                    let r = Machine::new(
                        prog,
                        cfg,
                        &cache,
                        &mut sink,
                        Inputs::Replay(shared_inputs, 0),
                        shot_rng(cfg.seed, shot),
                    )
                    .and_then(|mut m| {
                        m.run()?;
                        Ok(m.record())
                    });
                    match r {
                        Ok(rec) => *counts.entry(rec).or_insert(0) += 1,
                        Err(e) => {
                            failed.get_or_insert(e);
                            break;
                        }
                    }
                }
            }
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
                let sampled: HashMap<Record, u64> = (lo..hi)
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
                // 0. rank-ak irakurritako sarrerak; besteek zabalkundetik jaso dituzte
                let shared_inputs: &[f64] = if rank == 0 { &recorded_inputs } else { &replay };
                let results: Vec<Result<(Record, f64), RuntimeError>> = (lo..hi)
                    .into_par_iter()
                    .map(|shot| {
                        let mut sink = std::io::sink();
                        let mut m = Machine::new(
                            prog,
                            cfg,
                            &cache,
                            &mut sink,
                            Inputs::Replay(shared_inputs, 0),
                            shot_rng(cfg.seed, shot),
                        )?;
                        m.run()?;
                        Ok((m.record(), m.state.leakage))
                    })
                    .collect();
                for r in results {
                    match r {
                        Ok((rec, leak)) => {
                            *counts.entry(rec).or_insert(0) += 1;
                            leakage = leakage.max(leak);
                        }
                        Err(e) => {
                            failed.get_or_insert(e);
                        }
                    }
                }
            }
            Method::StateOnly => unreachable!(),
        }
    }

    // taldean: rank guztiek beren zenbaketak (edo errorea) 0.ari bidaltzen dizkiote, beti, inor zain gera ez dadin
    // (egoera banatuan shot bakoitza prozesu guztiek egiten dute, eta 0.ak ditu zenbaketa guztiak, laginketan izan ezik)
    let gathered = !cfg.distribute || method == Method::Sampled;
    if let Some(t) = team
        && ranks > 1
        && gathered
    {
        let mine = encode_part(&counts, leakage, failed.as_ref());
        if let Some(all) = t.gather(mine) {
            for part in &all[1..] {
                let (c, leak, err) = decode_part(part);
                if let Some(e) = err {
                    failed.get_or_insert(e);
                }
                counts = merge_counts(counts, c);
                leakage = leakage.max(leak);
            }
        }
    }
    if let Some(e) = failed {
        return Err(e);
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
        norm_error,
    })
}

fn f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

// rank baten emaitza bytetan: ihesa, errorea (lerroa eta mezua, edo ezer ez) eta zenbaketak (bitak:
// 0, 1 edo 255 idatzi gabe; irakurketak: u32, u32::MAX idatzi gabe)
fn encode_part(counts: &HashMap<Record, u64>, leakage: f64, err: Option<&RuntimeError>) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(leakage.to_le_bytes());
    let msg = err.map(|e| (e.line, e.message.as_bytes().to_vec()));
    out.push(msg.is_some() as u8);
    if let Some((line, m)) = &msg {
        out.extend((*line as u64).to_le_bytes());
        out.extend(team::encode_list(std::slice::from_ref(m)));
    }
    for (rec, n) in counts {
        let mut item = Vec::new();
        item.extend((rec.bits.len() as u32).to_le_bytes());
        item.extend(rec.bits.iter().map(|b| b.unwrap_or(255)));
        item.extend((rec.readouts.len() as u32).to_le_bytes());
        for r in &rec.readouts {
            item.extend(r.unwrap_or(u32::MAX).to_le_bytes());
        }
        item.extend(n.to_le_bytes());
        out.extend(team::encode_list(std::slice::from_ref(&item)));
    }
    out
}

fn decode_part(bytes: &[u8]) -> (HashMap<Record, u64>, f64, Option<RuntimeError>) {
    let leakage = f64::from_le_bytes(bytes[..8].try_into().unwrap());
    let mut rest = &bytes[9..];
    let mut err = None;
    if bytes[8] == 1 {
        let line = u64::from_le_bytes(rest[..8].try_into().unwrap()) as u32;
        let n = u64::from_le_bytes(rest[8..16].try_into().unwrap()) as usize;
        let message = String::from_utf8_lossy(&rest[16..16 + n]).into_owned();
        err = Some(RuntimeError { line, message });
        rest = &rest[16 + n..];
    }
    let mut counts = HashMap::new();
    for item in team::decode_list(rest) {
        let u32_at = |i: usize| u32::from_le_bytes(item[i..i + 4].try_into().unwrap());
        let nb = u32_at(0) as usize;
        let bits = item[4..4 + nb]
            .iter()
            .map(|&b| (b != 255).then_some(b))
            .collect();
        let nr = u32_at(4 + nb) as usize;
        let readouts = (0..nr)
            .map(|k| u32_at(8 + nb + 4 * k))
            .map(|v| (v != u32::MAX).then_some(v))
            .collect();
        let at = 8 + nb + 4 * nr;
        let n = u64::from_le_bytes(item[at..at + 8].try_into().unwrap());
        *counts.entry(Record { bits, readouts }).or_insert(0) += n;
    }
    (counts, leakage, err)
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
        let to_err = |message| RuntimeError { line: 0, message };
        let mut dist = None;
        let (state, quad) = if cfg.precision == Precision::Quad {
            (
                State::new(0, 0, 2).map_err(to_err)?,
                Some(QuadState::new(prog.n_qubits as usize).map_err(to_err)?),
            )
        } else if cfg.distribute {
            let team = cfg
                .team
                .clone()
                .expect("--distribute --mpi-rekin egiaztatuta");
            dist = Some(
                DistState::new(prog.n_qubits as usize, team, cfg.gpu.clone()).map_err(to_err)?,
            );
            (State::new(0, 0, 2).map_err(to_err)?, None)
        } else {
            let state = State::new(
                prog.n_qubits as usize,
                prog.n_modes as usize,
                prog.cutoff as usize,
            )
            .map_err(to_err)?;
            (state, None)
        };
        Ok(Machine {
            prog,
            state,
            pending: Vec::new(),
            quad,
            pending_quad: Vec::new(),
            dist,
            captured_totals: None,
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
                self.flush();
                if let Some(d) = &self.dist {
                    let (local, totals) = d.local_probabilities();
                    self.captured = Some(local);
                    self.captured_totals = Some(totals);
                } else {
                    self.captured = Some(match self.qd() {
                        Some(q) => q.probabilities(),
                        None => self.st().probabilities(),
                    });
                }
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
        if self.quad.is_some() {
            self.queue_quad(QuadGate::Mat {
                q: q as usize,
                m: quad::matrix(gate),
                controls,
            });
            return;
        }
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

    // qubit-ate bat sortara; sorta luzeegia bada, aplikatu. 128 biteko moduan, f64 sarrerak zehatz gisa
    fn queue(&mut self, g: QGate) {
        if self.quad.is_some() {
            self.queue_quad(QuadGate::from_f64(&g));
            return;
        }
        self.pending.push(g);
        if self.pending.len() >= MAX_PENDING {
            self.flush();
        }
    }

    fn queue_quad(&mut self, g: QuadGate) {
        self.pending_quad.push(g);
        if self.pending_quad.len() >= MAX_PENDING {
            self.flush();
        }
    }

    // `mask`-eko qubit guztiak 1 direnean e^{iθ} fasea (128 bitetan quad moduan)
    fn phase(&mut self, mask: usize, theta: f64) {
        if self.quad.is_some() {
            self.queue_quad(QuadGate::phase(mask, theta));
        } else {
            self.queue(QGate::phase(mask, theta));
        }
    }

    fn flush(&mut self) {
        if let Some(q) = self.quad.as_mut() {
            q.apply(&self.pending_quad);
            self.pending_quad.clear();
        }
        if let Some(d) = self.dist.as_mut() {
            d.apply(&self.pending);
            self.pending.clear();
            if let Some(e) = d.take_error() {
                self.gpu_error.get_or_insert(e);
            }
            return;
        }
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

    // 128 biteko egoera (quad moduan), zain dauden ateak aplikatu ondoren
    fn qd(&mut self) -> Option<&mut QuadState> {
        self.flush();
        self.quad.as_mut()
    }

    fn mask(qs: &[u32]) -> usize {
        qs.iter().fold(0usize, |m, q| m | (1usize << q))
    }

    fn measure_z(&mut self, q: u32) -> u8 {
        let r = self.uniform();
        self.flush();
        if let Some(d) = self.dist.as_mut() {
            return d.measure_qubit(q as usize, r);
        }
        match self.qd() {
            Some(s) => s.measure_qubit(q as usize, r),
            None => self.st().measure_qubit(q as usize, r),
        }
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
                self.phase(Self::mask(qubits), *theta);
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
                    self.phase(Self::mask(w), std::f64::consts::PI);
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
                self.pending_quad.clear();
                if let Some(q) = self.quad.as_mut() {
                    *q = QuadState::new(q.n_qubits()).expect("lehengo tamaina bera");
                }
                if let Some(d) = self.dist.as_mut() {
                    d.reset();
                }
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
                self.flush();
                let text = if let Some(d) = &self.dist {
                    match d.gather() {
                        Some(s) => crate::report::state_table(&s, 16),
                        // (0.ak bakarrik inprimatzen du; hor, None egoera handiegia delako da)
                        None => t!(
                            "  ({} amplitudes spread over {} processes: too large to show)\n",
                            "  ({} anplitude {} prozesutan banatuta: handiegia erakusteko)\n",
                            1u128 << d.n_qubits(),
                            d.ranks()
                        ),
                    }
                } else {
                    let snapshot = self.qd().map(|q| q.to_state());
                    match &snapshot {
                        Some(s) => crate::report::state_table(s, 16),
                        None => crate::report::state_table(self.st(), 16),
                    }
                };
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
                self.flush();
                let b = match (&self.dist, self.quad.as_ref()) {
                    (Some(d), _) => d.bloch(*q as usize),
                    (None, Some(s)) => s.bloch(*q as usize),
                    (None, None) => self.st().bloch(*q as usize),
                };
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
                self.flush();
                let f = match (&self.dist, self.quad.as_ref()) {
                    (Some(d), _) => d.bell_fidelity(*a as usize, *b as usize),
                    (None, Some(s)) => s.bell_fidelity(*a as usize, *b as usize),
                    (None, None) => self.st().bell_fidelity(*a as usize, *b as usize),
                };
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
                // egoera banatuan, 0. prozesuak irakurtzen du eta besteei bidaltzen die unean bertan
                let distributed = self.dist.is_some();
                let team = self.cfg.team.clone();
                let v = match &mut self.inputs {
                    Inputs::Record(list)
                        if distributed && team.as_ref().is_some_and(|t| t.rank() != 0) =>
                    {
                        let t = team.as_ref().unwrap();
                        let v = f64s(&t.broadcast(Vec::new()))
                            .first()
                            .copied()
                            .unwrap_or(0.0);
                        list.push(v);
                        v
                    }
                    Inputs::Record(list) => {
                        let mut line = String::new();
                        let _ = self.out.flush();
                        std::io::stdin().lock().read_line(&mut line).ok();
                        let v = line.trim().parse::<f64>().map_err(|_| RuntimeError {
                            line: self.prog.line_of(pc),
                            message: t!("INPUT expected a number, got '{}'", "INPUT-ek zenbaki bat espero zuen, baina '{}' jaso du", line.trim()),
                        })?;
                        list.push(v);
                        if distributed && let Some(t) = &team {
                            t.broadcast(v.to_le_bytes().to_vec());
                        }
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
