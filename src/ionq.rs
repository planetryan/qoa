// fitxategi honek qoa programa ionq-ren json formatura esportatzen du (ionq.circuit.v0).
// ionq-k qubit guztiak neurtzen ditu amaieran, beraz programaren neurketak amaieran egon behar
// dute eta ez dira idazten. modu optikoak, kontrol-fluxua eta erdiko neurketak ezin dira adierazi.
// oharra: formatua ionq-ren dokumentazioaren arabera idatzita dago; ez da ionq-ren api errealaren
// aurka probatu.

use crate::program::{Gate, Op, Program};
use crate::qasm::{ExportError, describe};
use crate::t;
use serde_json::{Value, json};

pub fn from_program(p: &Program, shots: u64) -> Result<String, Vec<ExportError>> {
    let mut errors = Vec::new();
    let mut circuit: Vec<Value> = Vec::new();
    if p.n_modes > 0 {
        errors.push(ExportError {
            line: 0,
            message: t!("the program uses {} optical mode(s); IonQ hardware has no optical modes", "programak {} modu optiko erabiltzen ditu; IonQ-ren hardwareak ez du modu optikorik", p.n_modes),
        });
    }
    // neurketak amaieran bakarrik
    let mut measured = false;
    for (pc, op) in p.ops.iter().enumerate() {
        let line = p.line_of(pc);
        let mut fail = |m: String| errors.push(ExportError { line, message: m });
        if measured
            && op.touches_quantum_state()
            && !matches!(op, Op::Measure { .. } | Op::MeasureAll)
        {
            fail(t!(
                "IonQ measures every qubit once at the end; gates after a measurement cannot be sent",
                "IonQ-k qubit guztiak behin neurtzen ditu amaieran; neurketa baten ondorengo ateak ezin dira bidali"
            ));
            continue;
        }
        match op {
            Op::Gate { q, gate } => circuit.extend(gate_json(gate, *q, &[])),
            Op::Controlled { controls, q, gate } => match gate {
                Gate::X | Gate::Y | Gate::Z | Gate::H | Gate::RX(_) | Gate::RY(_) | Gate::RZ(_) => {
                    circuit.extend(gate_json(gate, *q, controls))
                }
                _ => fail(t!(
                    "a controlled {} has no IonQ gate",
                    "{} kontrolatuak ez du IonQ aterik",
                    gate.qasm_name()
                )),
            },
            Op::PhaseAll { qubits, theta } => {
                if (*theta - std::f64::consts::PI).abs() < 1e-15 && qubits.len() >= 2 {
                    let (t, cs) = qubits.split_last().unwrap();
                    circuit.push(json!({"gate": "z", "controls": cs, "target": t}));
                } else if qubits.len() == 2 {
                    // cp(θ) = rz(θ/2) kontrolean + kontrolatutako rz(θ) (fase globala alde batera)
                    circuit
                        .push(json!({"gate": "rz", "target": qubits[0], "rotation": theta / 2.0}));
                    circuit.push(json!({"gate": "rz", "controls": [qubits[0]], "target": qubits[1], "rotation": theta}));
                } else if qubits.len() == 1 {
                    circuit.push(json!({"gate": "rz", "target": qubits[0], "rotation": theta}));
                } else {
                    fail(t!(
                        "a phase on {} qubits at once has no IonQ gate",
                        "{} qubiteko aldibereko faseak ez du IonQ aterik",
                        qubits.len()
                    ));
                }
            }
            Op::Swap { a, b, controls } => {
                if controls.is_empty() {
                    circuit.push(json!({"gate": "swap", "targets": [a, b]}));
                } else {
                    fail(t!(
                        "a controlled swap has no IonQ gate",
                        "trukaketa kontrolatuak ez du IonQ aterik"
                    ));
                }
            }
            Op::Bell { a, b } => {
                circuit.push(json!({"gate": "h", "target": a}));
                circuit.push(json!({"gate": "cnot", "control": a, "target": b}));
            }
            Op::Ghz { qubits } => {
                circuit.push(json!({"gate": "h", "target": qubits[0]}));
                for t in &qubits[1..] {
                    circuit.push(json!({"gate": "cnot", "control": qubits[0], "target": t}));
                }
            }
            Op::Cluster { qubits } => {
                for t in qubits {
                    circuit.push(json!({"gate": "h", "target": t}));
                }
                for w in qubits.windows(2) {
                    circuit.push(json!({"gate": "z", "controls": [w[0]], "target": w[1]}));
                }
            }
            Op::Measure { basis, .. } => {
                if *basis != crate::program::Basis::Z {
                    fail(t!(
                        "only Z-basis measurements can be sent to IonQ (add the rotation as gates)",
                        "Z oinarriko neurketak bakarrik bidal daitezke IonQ-ra (gehitu biraketa ate gisa)"
                    ));
                }
                measured = true;
            }
            Op::MeasureAll => measured = true,
            Op::Barrier => {}
            Op::Halt if pc + 1 == p.ops.len() => {}
            Op::Print { .. }
            | Op::Printf { .. }
            | Op::Log { .. }
            | Op::DumpState
            | Op::DumpRegs => {}
            other => fail(t!(
                "{} cannot be sent to IonQ",
                "{} ezin da IonQ-ra bidali",
                describe(other)
            )),
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    let doc = json!({
        "name": p.name,
        "target": "simulator",
        "shots": shots,
        "input": {
            "format": "ionq.circuit.v0",
            "qubits": p.n_qubits,
            "circuit": circuit,
        }
    });
    Ok(serde_json::to_string_pretty(&doc).expect("json serializes"))
}

// ionq-ren ate-izenak: si = s†, ti = t†, v = √x, vi = √x†; u ateak rz·ry·rz bihurtzen dira
fn gate_json(g: &Gate, target: u32, controls: &[u32]) -> Vec<Value> {
    let with = |mut v: Value| {
        if !controls.is_empty() {
            v["controls"] = json!(controls);
        }
        v
    };
    let simple = |name: &str| vec![with(json!({"gate": name, "target": target}))];
    let rot = |name: &str, a: f64| with(json!({"gate": name, "target": target, "rotation": a}));
    match g {
        Gate::H => simple("h"),
        Gate::X => simple("x"),
        Gate::Y => simple("y"),
        Gate::Z => simple("z"),
        Gate::S => simple("s"),
        Gate::Sdg => simple("si"),
        Gate::T => simple("t"),
        Gate::Tdg => simple("ti"),
        Gate::SX => simple("v"),
        Gate::SXdg => simple("vi"),
        Gate::RX(a) => vec![rot("rx", *a)],
        Gate::RY(a) => vec![rot("ry", *a)],
        Gate::RZ(a) | Gate::P(a) => vec![rot("rz", *a)],
        // u(θ, φ, λ) = rz(φ)·ry(θ)·rz(λ) fase global batera arte: lehenik rz(λ)
        Gate::U(t, ph, l) => vec![rot("rz", *l), rot("ry", *t), rot("rz", *ph)],
    }
}
