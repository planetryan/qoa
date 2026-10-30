// fitxategi honek gailu baten zarata-eredua kargatzen du (`qoa run --noise-model FITXATEGIA`): benetako
// gailu baten kalibrazioak (ibm-ren `backend.target`-etik, ikus tools/run_on_ibm.py), eta simulagailuak
// shot bakoitzean aplikatzen ditu, qiskit aer-en `NoiseModel.from_backend`-ek bezala:
//
//   ate natibo bakoitzaren ondoren: lehenik depolarizazioa, gero T1/T2 erlaxazioa atearen iraupenean
//   (ateko qubit bakoitzean, tenperatura 0, T2 <= 2·T1-ra mugatuta)
//   neurketa bakoitzean: irakurketa-errorea (0 -> 1 p0m1 probabilitatez, 1 -> 0 p1m0 probabilitatez)
//
// depolarizazio-parametroa ateko errorea (e = 1 - batez besteko fideltasuna) eta erlaxazioak berak
// eragiten duena kontuan hartuta kalkulatzen da, biak batera neurtutako errorea izan daitezen.
// kanalak ibilbide kuantiko zehatz gisa aplikatzen dira (jauziak eta jauzirik gabeko eboluzioa, bere
// probabilitate zehatzekin): shot-en banaketa kanal osoarena da, hurbilketarik gabe.
//
// ereduan ez dagoen ate bat (edo qubit-pare bat) exekuzio-errorea da: gailuak ezin du zuzenean egin,
// eta programa lehenik gailurako transpilatu behar da.
//
// formatua (json, unitateak segundotan):
//
//   {"format": "qoa-noise-1", "device": "ibm_torino", "calibrated": "...",
//    "qubits": [{"physical": 52, "t1": 1.6e-4, "t2": 1.1e-4, "p0m1": 0.012, "p1m0": 0.021}, ...],
//    "gates": [{"name": "sx", "qubits": [0], "error": 2.4e-4, "duration": 3.2e-8}, ...]}
//
// qubits[i] QOA-ren qi qubita da (physical: gailuko zenbakia, informaziorako). t1/t2 null: mugagabea.

use crate::program::{Basis, Gate, Op};
use crate::t;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

pub const FORMAT: &str = "qoa-noise-1";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModel {
    format: String,
    #[serde(default)]
    device: String,
    #[serde(default)]
    calibrated: String,
    qubits: Vec<RawQubit>,
    gates: Vec<RawGate>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawQubit {
    #[serde(default)]
    physical: Option<u32>,
    t1: Option<f64>,
    t2: Option<f64>,
    p0m1: f64,
    p1m0: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGate {
    name: String,
    qubits: Vec<u32>,
    error: f64,
    duration: f64,
}

#[derive(Debug, Clone)]
pub struct NoiseModel {
    pub device: String,
    pub calibrated: String,
    pub qubits: Vec<QubitNoise>,
    gates: HashMap<(String, Vec<u32>), GateNoise>,
}

#[derive(Debug, Clone)]
pub struct QubitNoise {
    pub physical: Option<u32>,
    // segundotan; mugagabea ezezaguna bada. t2 dagoeneko 2·t1-ra mugatuta
    pub t1: f64,
    pub t2: f64,
    // irakurketa-erroreak: 0 prestatuta 1 irakurtzea, eta alderantziz
    pub p0m1: f64,
    pub p1m0: f64,
}

#[derive(Debug, Clone)]
pub struct GateNoise {
    pub error: f64,
    pub duration: f64,
    // identitatea ez den n qubiteko pauli bat (uniformeki) aplikatzeko probabilitatea
    pub pauli: f64,
    // ateko qubit bakoitzaren erlaxazioa, atearen qubit-ordenan
    pub relax: Vec<Relax>,
}

// qubit bateko erlaxazioa ate baten iraupenean: anplitude-moteltzea (gamma) eta gero Z bat z
// probabilitatez (desfasea), batera T1/T2 kanal zehatza
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Relax {
    pub gamma: f64,
    pub z: f64,
}

// eragiketa batek gailuan zer den
#[derive(Debug, Clone, PartialEq)]
pub enum Native {
    // ez du qubitik ukitzen (klasikoa, kontrol-fluxua, barrera, simulagailuaren ikuskapena)
    Free,
    // z oinarriko neurketa: irakurketa-errorea bakarrik
    Readout,
    // ate natibo bat (qiskit-en izenarekin) qubit hauetan
    Gate(String, Vec<u32>),
    // gailu batek ezin du zuzenean egin; arrazoia
    Unsupported(String),
}

impl NoiseModel {
    pub fn load(path: &Path) -> Result<NoiseModel, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        NoiseModel::parse(&text).map_err(|e| format!("{}: {}", path.display(), e))
    }

    pub fn parse(text: &str) -> Result<NoiseModel, String> {
        let raw: RawModel = serde_json::from_str(text).map_err(|e| {
            t!(
                "not a QOA noise model ({}): {}",
                "ez da QOA zarata-eredu bat ({}): {}",
                FORMAT,
                e
            )
        })?;
        if raw.format != FORMAT {
            return Err(t!(
                "format is '{}'; this QOA reads '{}'",
                "formatua '{}' da; QOA honek '{}' irakurtzen du",
                raw.format,
                FORMAT
            ));
        }
        let mut qubits = Vec::with_capacity(raw.qubits.len());
        for (i, q) in raw.qubits.iter().enumerate() {
            let time = |v: Option<f64>, what: &str| match v {
                None => Ok(f64::INFINITY),
                Some(t) if t > 0.0 && t.is_finite() => Ok(t),
                Some(t) => Err(t!(
                    "qubit {}: {} must be a positive time in seconds (or null if unknown), not {}",
                    "{}. qubita: {}-k denbora positibo bat izan behar du segundotan (edo null ezezaguna bada), ez {}",
                    i,
                    what,
                    t
                )),
            };
            let t1 = time(q.t1, "t1")?;
            // aer-ek bezala: T2 > 2·T1 ez da fisikoa, eta 2·T1-ra mugatzen da
            let t2 = time(q.t2, "t2")?.min(2.0 * t1);
            for (what, p) in [("p0m1", q.p0m1), ("p1m0", q.p1m0)] {
                if !(0.0..=1.0).contains(&p) {
                    return Err(t!(
                        "qubit {}: {} must be a probability between 0 and 1, not {}",
                        "{}. qubita: {}-k 0 eta 1 arteko probabilitate bat izan behar du, ez {}",
                        i,
                        what,
                        p
                    ));
                }
            }
            qubits.push(QubitNoise {
                physical: q.physical,
                t1,
                t2,
                p0m1: q.p0m1,
                p1m0: q.p1m0,
            });
        }
        let mut gates = HashMap::new();
        for g in &raw.gates {
            let at = || format!("{} {:?}", g.name, g.qubits);
            if g.qubits.is_empty() || g.qubits.len() > 3 {
                return Err(t!(
                    "gate {}: a native gate acts on 1 to 3 qubits",
                    "{} atea: ate natibo batek 1etik 3ra qubit ukitzen ditu",
                    at()
                ));
            }
            for (k, &q) in g.qubits.iter().enumerate() {
                if q as usize >= qubits.len() {
                    return Err(t!(
                        "gate {}: qubit {} is not in the model's {} qubits",
                        "{} atea: {} qubita ez dago ereduaren {} qubitetan",
                        at(),
                        q,
                        qubits.len()
                    ));
                }
                if g.qubits[..k].contains(&q) {
                    return Err(t!(
                        "gate {}: the same qubit twice",
                        "{} atea: qubit bera bi aldiz",
                        at()
                    ));
                }
            }
            if !(0.0..=1.0).contains(&g.error) {
                return Err(t!(
                    "gate {}: error must be between 0 and 1, not {}",
                    "{} atea: errorea 0 eta 1 artean egon behar da, ez {}",
                    at(),
                    g.error
                ));
            }
            if !(g.duration >= 0.0 && g.duration.is_finite()) {
                return Err(t!(
                    "gate {}: duration must be 0 or more seconds, not {}",
                    "{} atea: iraupenak 0 segundo edo gehiago izan behar du, ez {}",
                    at(),
                    g.duration
                ));
            }
            let relax: Vec<Relax> = g
                .qubits
                .iter()
                .map(|&q| relaxation(qubits[q as usize].t1, qubits[q as usize].t2, g.duration))
                .collect();
            let noise = GateNoise {
                error: g.error,
                duration: g.duration,
                pauli: pauli_probability(g.error, &relax, g.duration),
                relax,
            };
            let key = (g.name.clone(), g.qubits.clone());
            if gates.insert(key, noise).is_some() {
                return Err(t!(
                    "gate {} is listed twice",
                    "{} atea bi aldiz dago zerrendan",
                    at()
                ));
            }
        }
        Ok(NoiseModel {
            device: raw.device,
            calibrated: raw.calibrated,
            qubits,
            gates,
        })
    }

    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    // ate baten zarata qubit hauetan. ate simetrikoak (cz, cp, swap, rzz) beste ordenan ere bilatzen
    // dira: fisikoki ate bera dira
    pub fn gate(&self, name: &str, qubits: &[u32]) -> Option<&GateNoise> {
        let key = (name.to_string(), qubits.to_vec());
        if let Some(g) = self.gates.get(&key) {
            return Some(g);
        }
        if matches!(name, "cz" | "cp" | "swap" | "rzz") && qubits.len() == 2 {
            return self
                .gates
                .get(&(name.to_string(), vec![qubits[1], qubits[0]]));
        }
        None
    }

    // eredu honetako ate-izenak, ordenatuta (mezuetarako)
    pub fn gate_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.gates.keys().map(|(n, _)| n.clone()).collect();
        v.sort();
        v.dedup();
        v
    }
}

// qubit bateko T1/T2 kanala `time` iraupenean, anplitude-moteltze gisa (gamma = 1 - e^{-t/T1}) eta
// ondoren desfase gisa: koherentzia e^{-t/T2} izan dadin, moteltzeak √(1-gamma) uzten du eta
// desfaseak f = e^{-t/T2} / √(1-gamma) faktorea gehitzen du, hau da, Z bat (1 - f)/2 probabilitatez
pub fn relaxation(t1: f64, t2: f64, time: f64) -> Relax {
    if time <= 0.0 {
        return Relax { gamma: 0.0, z: 0.0 };
    }
    let gamma = -(-time / t1).exp_m1();
    let keep = 1.0 - gamma;
    let z = if keep > 0.0 {
        // t2 <= 2·t1 denez, f <= 1; e^{-t/T2 + t/(2T1)} zuzenean, zatiketarik gabe
        let f = (-time / t2 + time / (2.0 * t1)).exp().min(1.0);
        (1.0 - f) / 2.0
    } else {
        // egoera osoa |0>-ra eraman da: koherentziarik ez dago
        0.0
    };
    Relax { gamma, z }
}

// erlaxazio-kanalaren prozesu-fideltasuna qubit batean: (1 + (1-gamma) + 2c) / 4, c = koherentzia
fn process_fidelity(r: &Relax) -> f64 {
    let c = (1.0 - r.gamma).sqrt() * (1.0 - 2.0 * r.z);
    (2.0 - r.gamma + 2.0 * c) / 4.0
}

// aer-en `_device_depolarizing_error`: ateko errorea e = 1 - F (batez besteko fideltasuna) erlaxazioaren
// eta depolarizazioaren konposizioarena da; depolarizazioaren p parametroa askatu, eta identitatea ez
// den pauli baten probabilitatea itzuli (p·(4^n - 1)/4^n)
pub fn pauli_probability(error: f64, relax: &[Relax], duration: f64) -> f64 {
    let n = relax.len() as i32;
    let d = 2f64.powi(n);
    let e = error.min(d / (d + 1.0));
    let relax_fid = if duration > 0.0 {
        let pro: f64 = relax.iter().map(process_fidelity).product();
        (d * pro + 1.0) / (d + 1.0)
    } else {
        1.0
    };
    let relax_infid = 1.0 - relax_fid;
    if e <= relax_infid {
        return 0.0;
    }
    let four = 4f64.powi(n);
    let p = (d * (e - relax_infid) / (d * relax_fid - 1.0)).min(four / (four - 1.0));
    p * (four - 1.0) / four
}

// eragiketa bat gailuan: ate natiboa (qiskit-en izenarekin), neurketa, ezer ez, edo ezinezkoa
pub fn native(op: &Op) -> Native {
    let unsupported = |what: String| {
        Native::Unsupported(t!(
            "{} is not a native operation of a qubit device",
            "{} ez da qubit-gailu baten eragiketa natiboa",
            what
        ))
    };
    // hainbat ate agindu bakarrean
    let mac = |name: &str| {
        unsupported(t!(
            "{} (several gates in one instruction)",
            "{} (hainbat ate agindu bakarrean)",
            name
        ))
    };
    match op {
        Op::Gate { q, gate } => Native::Gate(gate_name(gate).to_string(), vec![*q]),
        Op::Controlled { controls, gate, .. } => {
            let name = match (controls.len(), gate) {
                (1, Gate::U(..)) => "cu3".to_string(),
                (1, g) => format!("c{}", gate_name(g)),
                (2, Gate::X) => "ccx".to_string(),
                (n, g) => {
                    return unsupported(t!(
                        "a {}-controlled {}",
                        "{} kontroldun {}",
                        n,
                        gate_name(g)
                    ));
                }
            };
            Native::Gate(name, op.gate_qubits())
        }
        Op::PhaseAll { qubits, theta } => {
            let is_pi = *theta == std::f64::consts::PI;
            let name = match qubits.len() {
                1 => "p",
                2 if is_pi => "cz",
                2 => "cp",
                3 if is_pi => "ccz",
                n => {
                    return unsupported(t!("a phase on {} qubits", "{} qubiteko fasea", n));
                }
            };
            Native::Gate(name.to_string(), qubits.clone())
        }
        Op::Swap { controls, .. } => match controls.len() {
            0 => Native::Gate("swap".into(), op.gate_qubits()),
            1 => Native::Gate("cswap".into(), op.gate_qubits()),
            _ => unsupported(t!("a multi-controlled swap", "kontrol anitzeko trukaketa")),
        },
        Op::Reset { q } => Native::Gate("reset".into(), vec![*q]),
        Op::Measure { basis, .. } => match basis {
            Basis::Z => Native::Readout,
            b => unsupported(t!(
                "MEASURE in the {} basis (a device measures Z)",
                "MEASURE {} oinarrian (gailu batek Z neurtzen du)",
                b.name()
            )),
        },
        Op::MeasureAll => Native::Readout,
        Op::Bell { .. } => mac("BELL"),
        Op::Ghz { .. } => mac("GHZ"),
        Op::Cluster { .. } => mac("CLUSTER"),
        Op::EntangleSwap { .. } => mac("ESWAP"),
        Op::ResetAll => mac("RESETALL"),
        Op::ToBasis { .. } => mac("TOBASIS"),
        Op::Emit { .. }
        | Op::PhotonCount { .. }
        | Op::PhotonDetect { .. }
        | Op::Parity { .. }
        | Op::Coincidence { .. }
        | Op::Displace { .. }
        | Op::DisplaceReg { .. }
        | Op::Squeeze { .. }
        | Op::SqueezeReg { .. }
        | Op::ModePhase { .. }
        | Op::Kerr { .. }
        | Op::CrossKerr { .. }
        | Op::BeamSplitter { .. }
        | Op::ModeSwap { .. }
        | Op::AddPhoton { .. }
        | Op::SubPhoton { .. }
        | Op::Loss { .. }
        | Op::Fock { .. }
        | Op::Dispersive { .. } => unsupported(t!("an optical operation", "eragiketa optiko bat")),
        Op::DumpState
        | Op::DumpRegs
        | Op::Expect { .. }
        | Op::BellCheck { .. }
        | Op::MeanPhoton { .. }
        | Op::Set { .. }
        | Op::Copy { .. }
        | Op::Bin { .. }
        | Op::Un { .. }
        | Op::Rand { .. }
        | Op::Seed { .. }
        | Op::Input { .. }
        | Op::Time { .. }
        | Op::Push { .. }
        | Op::Pop { .. }
        | Op::PackBits { .. }
        | Op::Jump { .. }
        | Op::JumpIf { .. }
        | Op::Branch { .. }
        | Op::Call { .. }
        | Op::Return
        | Op::LoopStart { .. }
        | Op::LoopEnd { .. }
        | Op::Halt
        | Op::Barrier
        | Op::Print { .. }
        | Op::Printf { .. }
        | Op::CharOut { .. }
        | Op::Log { .. } => Native::Free,
    }
}

// qiskit-en ate-izenak (ibm-ren target-ek erabiltzen dituztenak)
fn gate_name(g: &Gate) -> &'static str {
    match g {
        Gate::U(..) => "u",
        g => g.qasm_name(),
    }
}
