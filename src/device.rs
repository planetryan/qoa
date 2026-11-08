// fitxategi honek QOA-ren gailu-protokoloa du ("qoa-device-1", docs/bench.md): QOA-k programa baten optika
// banku baten kontrolagailura bidaltzen du, eta kontrolagailuak kontaketak itzultzen ditu. biak json lerro bat
// dira, kontrolagailuaren sarrera eta irteera estandarretan (`qoa run --device KOMANDOA`):
//
//   QOA -> {"protocol": "qoa-device-1", "job": {"program": "...", "target": "bench-b", "shots": 1000,
//           "seed": 7, "qubits": 2, "modes": 0, "cutoff": 0, "bits": 2, "steps": [...]}}
//   <- {"protocol": "qoa-device-1", "device": "...", "shots": 1000,
//       "counts": [{"bits": "1-", "readouts": [], "count": 12}, ...]}
//   edo <- {"protocol": "qoa-device-1", "error": "..."}
//
// urratsak (steps), ordenan: photon (fotoi iragarri bat moduan), beamsplitter, phase, swap, loss, displace,
// detect (klik bat), coincidence, pair (bikote entrelazatua, BELL), gate (qubit bateko ate bat: izena eta
// parametroak, eta U(θ, φ, λ) angeluak xafletarako), to_basis, measure ("bit": null qubitaren bit propioa
// da, c<qubit>; detect-en, moduaren irakurketa). "bits" testua c(n-1)..c0 da, eta `-`
// emaitzarik gabeko bit bat; "readouts" moduen irakurketak (null irakurri gabe). kontrolagailu batek urrats
// ezagun bat egin ezin badu, errore bat itzultzen du, ez beste zerbait.
//
// `qoa bench-sim`-ek protokoloa simulazioz erantzuten du (banku-eredu batekin, aukeran): hardwarerik gabe
// garatzeko, eta proben erreferentzia gisa.

use crate::noise::BenchModel;
use crate::program::{Basis, Dest, Gate, Op, Program};
use crate::t;
use crate::target::Target;
use crate::vm::{Method, Record, RunConfig, RunOutput, RuntimeError};
use serde_json::{Value, json};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
use std::io::{BufRead, Write};

pub const PROTOCOL: &str = "qoa-device-1";

fn basis_name(b: Basis) -> &'static str {
    b.name()
}

// qubit bateko ate bat: izena, parametroak eta U(θ, φ, λ) (fase globala alde batera)
fn gate_step(q: u32, g: Gate) -> Value {
    let (name, params, u): (&str, Vec<f64>, [f64; 3]) = match g {
        Gate::H => ("H", vec![], [FRAC_PI_2, 0.0, PI]),
        Gate::X => ("X", vec![], [PI, 0.0, PI]),
        Gate::Y => ("Y", vec![], [PI, FRAC_PI_2, FRAC_PI_2]),
        Gate::Z => ("Z", vec![], [0.0, 0.0, PI]),
        Gate::S => ("S", vec![], [0.0, 0.0, FRAC_PI_2]),
        Gate::Sdg => ("SDG", vec![], [0.0, 0.0, -FRAC_PI_2]),
        Gate::T => ("T", vec![], [0.0, 0.0, FRAC_PI_4]),
        Gate::Tdg => ("TDG", vec![], [0.0, 0.0, -FRAC_PI_4]),
        Gate::SX => ("SX", vec![], [FRAC_PI_2, -FRAC_PI_2, FRAC_PI_2]),
        Gate::SXdg => ("SXDG", vec![], [-FRAC_PI_2, -FRAC_PI_2, FRAC_PI_2]),
        Gate::RX(a) => ("RX", vec![a], [a, -FRAC_PI_2, FRAC_PI_2]),
        Gate::RY(a) => ("RY", vec![a], [a, 0.0, 0.0]),
        Gate::RZ(a) => ("RZ", vec![a], [0.0, 0.0, a]),
        Gate::P(a) => ("PHASE", vec![a], [0.0, 0.0, a]),
        Gate::U(a, b, c) => ("U", vec![a, b, c], [a, b, c]),
    };
    json!({"op": "gate", "qubit": q, "gate": name, "params": params, "u": u})
}

// helburu-bita: zenbaki bat, edo null eragiketaren berezko helburuarentzat (qubitaren bita, moduaren
// irakurketa). erregistroak profilak baztertzen ditu
fn bit_of(dest: Dest) -> Value {
    match dest {
        Dest::Bit(k) => json!(k),
        Dest::Own | Dest::Reg(_) => Value::Null,
    }
}

// programaren lana, bankuaren profila egiaztatu ondoren
pub fn job(p: &Program, target: Target, shots: u64, seed: u64) -> Result<Value, String> {
    let problems = crate::target::check(p, target);
    let mut errors: Vec<String> = problems.iter().map(|e| e.to_string()).collect();
    let mut steps = Vec::new();
    for (pc, op) in p.ops.iter().enumerate() {
        let step = match op {
            Op::Fock { o, .. } => json!({"op": "photon", "mode": o}),
            Op::BeamSplitter { a, b, theta, phi } => {
                json!({"op": "beamsplitter", "modes": [a, b], "theta": theta, "phi": phi})
            }
            Op::ModePhase { o, theta } => json!({"op": "phase", "mode": o, "angle": theta}),
            Op::ModeSwap { a, b } => json!({"op": "swap", "modes": [a, b]}),
            Op::Loss { o, gamma } => json!({"op": "loss", "mode": o, "probability": gamma}),
            Op::Displace { o, re, im } => json!({"op": "displace", "mode": o, "re": re, "im": im}),
            Op::PhotonDetect { o, dest } => {
                json!({"op": "detect", "mode": o, "bit": bit_of(*dest)})
            }
            Op::Coincidence { modes, dest } => {
                json!({"op": "coincidence", "modes": modes, "bit": bit_of(*dest)})
            }
            Op::Bell { a, b } => json!({"op": "pair", "qubits": [a, b]}),
            Op::Gate { q, gate } => gate_step(*q, *gate),
            Op::ToBasis { q, basis } => {
                json!({"op": "to_basis", "qubit": q, "basis": basis_name(*basis)})
            }
            // "bit": null on qubitaren bit propioa da, c<qubit>
            Op::Measure { q, basis, dest } => json!({"op": "measure", "qubit": q,
                "basis": basis_name(*basis), "bit": bit_of(*dest)}),
            Op::MeasureAll => {
                for q in 0..p.n_qubits {
                    steps.push(json!({"op": "measure", "qubit": q, "basis": "Z", "bit": q}));
                }
                continue;
            }
            Op::Barrier | Op::Halt => continue,
            other => {
                if problems.is_empty() {
                    errors.push(t!(
                        "line {}: {} runs in a simulation only: a device run sends the optics to the bench and reads back counts",
                        "{}. lerroa: {} simulazio batean bakarrik dabil: gailu-exekuzio batek optika bankura bidaltzen du eta kontaketak irakurtzen ditu",
                        p.line_of(pc),
                        crate::disasm::op_name(other).unwrap_or("?")
                    ));
                }
                continue;
            }
        };
        steps.push(step);
    }
    if !errors.is_empty() {
        return Err(t!(
            "cannot run on {} (docs/bench.md):\n  {}",
            "ezin da {}-n exekutatu (docs/bench.eu.md):\n  {}",
            target.name(),
            errors.join("\n  ")
        ));
    }
    Ok(json!({"protocol": PROTOCOL, "job": {
        "program": p.name, "target": target.name(), "shots": shots, "seed": seed,
        "qubits": p.n_qubits, "modes": p.n_modes,
        "cutoff": if p.n_modes > 0 { p.cutoff } else { 0 }, "bits": p.n_bits,
        "steps": steps,
    }}))
}

fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value, String> {
    v.get(key).ok_or_else(|| {
        t!(
            "the message has no '{}' field",
            "mezuak ez du '{}' eremurik",
            key
        )
    })
}

fn uint(v: &Value, key: &str) -> Result<u64, String> {
    field(v, key)?.as_u64().ok_or_else(|| {
        t!(
            "'{}' must be a whole number",
            "'{}'-k zenbaki osoa izan behar du",
            key
        )
    })
}

fn num(v: &Value, key: &str) -> Result<f64, String> {
    field(v, key)?.as_f64().ok_or_else(|| {
        t!(
            "'{}' must be a number",
            "'{}'-k zenbaki bat izan behar du",
            key
        )
    })
}

fn modes2(v: &Value) -> Result<(u64, u64), String> {
    let m = field(v, "modes")?.as_array().filter(|a| a.len() == 2);
    let m = m.ok_or_else(|| {
        t!(
            "'modes' must list two modes",
            "'modes'-k bi modu zerrendatu behar ditu"
        )
    })?;
    match (m[0].as_u64(), m[1].as_u64()) {
        (Some(a), Some(b)) => Ok((a, b)),
        _ => Err(t!(
            "'modes' must list two modes",
            "'modes'-k bi modu zerrendatu behar ditu"
        )),
    }
}

fn arrow(v: &Value) -> String {
    match v.get("bit").and_then(Value::as_u64) {
        Some(k) => format!(" -> c{}", k),
        None => String::new(),
    }
}

// lan bat QOA iturburu gisa (simulagailuak exekutatzen duena): urrats bakoitza agindu bat da, ordena berean
pub fn job_source(job: &Value) -> Result<String, String> {
    let mut out = Vec::new();
    let qubits = uint(job, "qubits")?;
    let modes = uint(job, "modes")?;
    if qubits > 0 {
        out.push(format!("QINIT {}", qubits));
    }
    if modes > 0 {
        out.push(format!("QMODE {} {}", modes, uint(job, "cutoff")?));
    }
    let bits = uint(job, "bits")?;
    if bits > 0 {
        out.push(format!("CREG bits {}", bits));
    }
    let steps = field(job, "steps")?.as_array().ok_or_else(|| {
        t!(
            "'steps' must be a list",
            "'steps'-k zerrenda bat izan behar du"
        )
    })?;
    for s in steps {
        let op = field(s, "op")?.as_str().unwrap_or("");
        let line = match op {
            "photon" => format!("SINGLEPHOTON o{}", uint(s, "mode")?),
            "beamsplitter" => {
                let (a, b) = modes2(s)?;
                format!("BS o{} o{} {} {}", a, b, num(s, "theta")?, num(s, "phi")?)
            }
            "phase" => format!("MODEPHASE o{} {}", uint(s, "mode")?, num(s, "angle")?),
            "swap" => {
                let (a, b) = modes2(s)?;
                format!("MODESWAP o{} o{}", a, b)
            }
            "loss" => format!("LOSS o{} {}", uint(s, "mode")?, num(s, "probability")?),
            "displace" => format!(
                "DISPLACE o{} {} {}",
                uint(s, "mode")?,
                num(s, "re")?,
                num(s, "im")?
            ),
            "detect" => format!("DETECT o{}{}", uint(s, "mode")?, arrow(s)),
            "coincidence" => {
                let ms: Vec<String> = field(s, "modes")?
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_u64)
                            .map(|m| format!("o{}", m))
                            .collect()
                    })
                    .unwrap_or_default();
                format!("COINCIDENCE {}{}", ms.join(" "), arrow(s))
            }
            "pair" => {
                let q = field(s, "qubits")?.as_array().filter(|a| a.len() == 2);
                let q = q.ok_or_else(|| {
                    t!(
                        "'qubits' must list two qubits",
                        "'qubits'-k bi qubit zerrendatu behar ditu"
                    )
                })?;
                format!(
                    "BELL q{} q{}",
                    q[0].as_u64().unwrap_or(0),
                    q[1].as_u64().unwrap_or(0)
                )
            }
            "gate" => {
                let name = field(s, "gate")?.as_str().unwrap_or("");
                let params: Vec<String> = field(s, "params")?
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_f64)
                            .map(|x| format!("{}", x))
                            .collect()
                    })
                    .unwrap_or_default();
                let known = [
                    "H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "SX", "SXDG", "RX", "RY", "RZ",
                    "PHASE", "U",
                ];
                if !known.contains(&name) {
                    return Err(t!("unknown gate '{}'", "'{}' ate ezezaguna", name));
                }
                let mut l = format!("{} q{}", name, uint(s, "qubit")?);
                for p in params {
                    l.push(' ');
                    l.push_str(&p);
                }
                l
            }
            "to_basis" => format!(
                "TOBASIS q{} {}",
                uint(s, "qubit")?,
                field(s, "basis")?.as_str().unwrap_or("?")
            ),
            "measure" => {
                let basis = field(s, "basis")?.as_str().unwrap_or("?");
                let q = uint(s, "qubit")?;
                if basis == "Z" {
                    format!("MEASURE q{}{}", q, arrow(s))
                } else {
                    format!("MEASUREIN q{} {}{}", q, basis, arrow(s))
                }
            }
            other => {
                return Err(t!("unknown step '{}'", "'{}' urrats ezezaguna", other));
            }
        };
        out.push(line);
    }
    Ok(out.join("\n") + "\n")
}

fn bits_text(r: &Record) -> String {
    r.bits
        .iter()
        .rev()
        .map(|b| match b {
            Some(1) => '1',
            Some(_) => '0',
            None => '-',
        })
        .collect()
}

// simulazioaren erantzuna protokoloaren formatuan
fn reply(device: &str, out: &RunOutput) -> Value {
    let counts: Vec<Value> = out
        .counts
        .iter()
        .map(|(r, n)| json!({"bits": bits_text(r), "readouts": r.readouts, "count": n}))
        .collect();
    json!({"protocol": PROTOCOL, "device": device, "shots": out.shots, "counts": counts})
}

fn error_reply(message: &str) -> Value {
    json!({"protocol": PROTOCOL, "error": message})
}

// lan bat simulatu (bench-sim-ek erabiltzen du)
pub fn simulate(message: &Value, model: Option<&BenchModel>) -> Value {
    let attempt = || -> Result<Value, String> {
        if message.get("protocol").and_then(Value::as_str) != Some(PROTOCOL) {
            return Err(t!("not a {} message", "ez da {} mezu bat", PROTOCOL));
        }
        let job = field(message, "job")?;
        let src = job_source(job)?;
        let name = field(job, "program")?.as_str().unwrap_or("job").to_string();
        let p =
            crate::asm::assemble(&src, &name).map_err(|d| crate::asm::render(&d, &src, &name))?;
        let cfg = RunConfig {
            shots: uint(job, "shots")?,
            seed: job.get("seed").and_then(Value::as_u64).unwrap_or(0),
            bench: model.map(|m| std::sync::Arc::new(m.clone())),
            ..RunConfig::default()
        };
        let out = crate::vm::run(&p, &cfg, &mut std::io::sink()).map_err(|e| e.to_string())?;
        let device = model.map_or_else(
            || t!("ideal simulated bench", "banku simulatu ideala"),
            |m| t!("simulated {}", "{} simulatua", m.device),
        );
        Ok(reply(&device, &out))
    };
    attempt().unwrap_or_else(|e| error_reply(&e))
}

// `qoa bench-sim`: lerro bakoitza lan bat, erantzun bakoitza lerro bat
pub fn serve(
    model: Option<&BenchModel>,
    input: impl BufRead,
    mut output: impl Write,
) -> Result<(), String> {
    for line in input.lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let answer = match serde_json::from_str::<Value>(&line) {
            Ok(v) => simulate(&v, model),
            Err(e) => error_reply(&t!("not json: {}", "ez da json: {}", e)),
        };
        writeln!(output, "{}", answer).map_err(|e| e.to_string())?;
        output.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

// gailuaren erantzuna: haren izena, shot-kopurua eta kontaketak
pub type Reply = (String, u64, Vec<(Record, u64)>);

// kontrolagailuaren erantzuna kontaketa bihurtu
pub fn read_reply(p: &Program, v: &Value) -> Result<Reply, String> {
    if v.get("protocol").and_then(Value::as_str) != Some(PROTOCOL) {
        return Err(t!(
            "the device did not answer in the {} protocol",
            "gailuak ez du {} protokoloan erantzun",
            PROTOCOL
        ));
    }
    if let Some(e) = v.get("error") {
        return Err(t!(
            "the device says: {}",
            "gailuak dio: {}",
            e.as_str().unwrap_or("?")
        ));
    }
    let device = v
        .get("device")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_string();
    let shots = uint(v, "shots")?;
    let list = field(v, "counts")?.as_array().ok_or_else(|| {
        t!(
            "'counts' must be a list",
            "'counts'-k zerrenda bat izan behar du"
        )
    })?;
    let mut counts = Vec::new();
    let mut total = 0u64;
    for c in list {
        let text = field(c, "bits")?.as_str().unwrap_or("");
        if text.chars().count() != p.n_bits as usize || text.chars().any(|ch| !"01-".contains(ch)) {
            return Err(t!(
                "the device returned bits '{}', but the program has {} bit(s) (each 0, 1 or -)",
                "gailuak '{}' bitak itzuli ditu, baina programak {} bit ditu (bakoitza 0, 1 edo -)",
                text,
                p.n_bits
            ));
        }
        let bits: Vec<Option<u8>> = text
            .chars()
            .rev()
            .map(|ch| match ch {
                '1' => Some(1),
                '0' => Some(0),
                _ => None,
            })
            .collect();
        let readouts: Vec<Option<u32>> = match c.get("readouts") {
            Some(Value::Array(a)) => a.iter().map(|x| x.as_u64().map(|n| n as u32)).collect(),
            _ => vec![None; p.n_modes as usize],
        };
        if readouts.len() != p.n_modes as usize {
            return Err(t!(
                "the device returned {} mode reading(s), but the program has {} mode(s)",
                "gailuak {} modu-irakurketa itzuli ditu, baina programak {} modu ditu",
                readouts.len(),
                p.n_modes
            ));
        }
        let n = uint(c, "count")?;
        total += n;
        counts.push((Record { bits, readouts }, n));
    }
    if total != shots {
        return Err(t!(
            "the device's counts add up to {}, but it says it ran {} shot(s)",
            "gailuaren kontaketen batura {} da, baina {} shot exekutatu dituela dio",
            total,
            shots
        ));
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok((device, shots, counts))
}

// `qoa run --device KOMANDOA`: kontrolagailua abiarazi, lana bidali, erantzuna irakurri
pub fn run(
    p: &Program,
    cfg: &RunConfig,
    command: &str,
    target: Target,
) -> Result<RunOutput, RuntimeError> {
    let err = |message: String| RuntimeError { line: 0, message };
    let job = job(p, target, cfg.shots, cfg.seed).map_err(err)?;
    let argv: Vec<&str> = command.split_whitespace().collect();
    let Some((program, args)) = argv.split_first() else {
        return Err(err(t!(
            "--device needs a command",
            "--device-k komando bat behar du"
        )));
    };
    let mut child = std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| {
            err(t!(
                "cannot start the device controller '{}': {}",
                "ezin da '{}' gailu-kontrolagailua abiarazi: {}",
                program,
                e
            ))
        })?;
    {
        let mut stdin = child.stdin.take().expect("sarrera piped da");
        writeln!(stdin, "{}", job).map_err(|e| {
            err(t!(
                "cannot send the job to the device: {}",
                "ezin zaio lana gailuari bidali: {}",
                e
            ))
        })?;
    }
    let stdout = child.stdout.take().expect("irteera piped da");
    let mut line = String::new();
    std::io::BufReader::new(stdout)
        .read_line(&mut line)
        .map_err(|e| {
            err(t!(
                "cannot read the device's answer: {}",
                "ezin da gailuaren erantzuna irakurri: {}",
                e
            ))
        })?;
    let _ = child.wait();
    if line.trim().is_empty() {
        return Err(err(t!(
            "the device controller '{}' gave no answer",
            "'{}' gailu-kontrolagailuak ez du erantzunik eman",
            program
        )));
    }
    let v: Value = serde_json::from_str(&line).map_err(|e| {
        err(t!(
            "the device's answer is not json: {}",
            "gailuaren erantzuna ez da json: {}",
            e
        ))
    })?;
    let (device, shots, counts) = read_reply(p, &v).map_err(err)?;
    let final_state = crate::state::State::new(0, 0, 1).map_err(err)?;
    Ok(RunOutput {
        method: Method::Device,
        shots,
        counts,
        final_state,
        regs: [0.0; crate::program::NUM_REGS],
        leakage: 0.0,
        steps: 0,
        norm_error: 0.0,
        exit_status: None,
        device: Some(device),
    })
}
