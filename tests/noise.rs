// fitxategi honek gailuen zarata-ereduak egiaztatzen ditu (src/noise.rs, `qoa run --noise-model`):
// depolarizazio-parametroa aer-en balio berberak direla (qiskit-aer 0.17-rekin kalkulatuak), erlaxazioaren
// formulak T1/T2 definizioak betetzen dituela, eta kanal bakoitzak shot-etan emaitza zehatza ematen duela
// (T1 jaitsiera, ramsey T2, irakurketa, bi qubiteko depolarizazioa), 5 sigmako tartean hazi finkoekin.
// gainera: formatu okerrak eta gailuak egin ezin dituen eragiketak errore argiak dira.

use qoa::noise::{NoiseModel, pauli_probability, relaxation};
use qoa::vm::{Method, RunConfig, RunOutput, RuntimeError};
use std::sync::Arc;

fn model(qubits: &str, gates: &str) -> String {
    format!(
        r#"{{"format": "qoa-noise-1", "device": "test", "qubits": [{}], "gates": [{}]}}"#,
        qubits, gates
    )
}

fn run(src: &str, model_json: &str, shots: u64) -> Result<RunOutput, RuntimeError> {
    let p = qoa::asm::assemble(src, "t.qoa").unwrap();
    let cfg = RunConfig {
        shots,
        seed: 11,
        model: Some(Arc::new(NoiseModel::parse(model_json).unwrap())),
        ..RunConfig::default()
    };
    qoa::vm::run(&p, &cfg, &mut std::io::sink())
}

// bitak irakurtzeko: `want` bezalako bit-balioak dituzten shot-en zatia
fn share(out: &RunOutput, want: &[(usize, u8)]) -> f64 {
    let hit: u64 = out
        .counts
        .iter()
        .filter(|(r, _)| want.iter().all(|&(k, v)| r.bits[k] == Some(v)))
        .map(|(_, n)| n)
        .sum();
    hit as f64 / out.shots as f64
}

fn close(got: f64, want: f64, shots: u64) {
    let sigma = (want * (1.0 - want) / shots as f64).sqrt().max(1e-9);
    assert!(
        (got - want).abs() < 5.0 * sigma,
        "{} ≠ {} (5σ = {})",
        got,
        want,
        5.0 * sigma
    );
}

// (errorea, iraupena, qubit bakoitzaren (t1, t2), aer-en emaitza)
type Case = (f64, f64, &'static [(f64, f64)], f64);

#[test]
fn depolarizing_parameter_matches_aer() {
    // (errorea, iraupena, [(t1, t2)]) => identitatea ez den pauli baten probabilitatea, aer-en
    // `_device_depolarizing_error`-etik (1 - prozesu-fideltasuna)
    let cases: [Case; 7] = [
        (
            0.004854875406855308,
            6.8e-8,
            &[(2.24e-4, 3.1e-4), (1.5e-4, 0.9e-4)],
            0.005396089108455904,
        ),
        (2.4e-4, 3.2e-8, &[(1.9e-4, 1.2e-4)], 0.00018462590731160855),
        (
            0.02,
            6.8e-8,
            &[(5e-6, 7e-6), (8e-6, 3e-6)],
            0.003660569384137391,
        ),
        // erlaxazioak berak errorea baino gehiago ematen du: depolarizaziorik ez
        (1e-5, 3.2e-8, &[(2e-5, 1e-5)], 0.0),
        // muga fisikoetara mugatuta
        (0.9, 0.0, &[(1e-4, 1e-4)], 1.0),
        (0.3, 0.0, &[(1e-4, 1e-4), (1e-4, 1e-4)], 0.375),
        (0.5, 1e-6, &[(2e-6, 9e-6)], 0.75),
    ];
    for (error, duration, qubits, want) in cases {
        let relax: Vec<_> = qubits
            .iter()
            .map(|&(t1, t2)| relaxation(t1, f64::min(t2, 2.0 * t1), duration))
            .collect();
        let got = pauli_probability(error, &relax, duration);
        assert!(
            (got - want).abs() <= 1e-12 * want.max(1e-3),
            "{:?}: {} ≠ {}",
            (error, duration, qubits),
            got,
            want
        );
    }
}

#[test]
fn relaxation_gives_the_t1_and_t2_decay() {
    for (t1, t2, t) in [
        (1e-4, 5e-5, 3e-8),
        (1e-4, 2e-4, 1e-5),
        (2e-6, 1e-6, 5e-6),
        (1e-4, 1.5e-4, 0.0),
    ] {
        let r = relaxation(t1, t2, t);
        // |1>-en populazioa e^{-t/T1}, koherentzia e^{-t/T2}
        assert!((1.0 - r.gamma - (-t / t1).exp()).abs() < 1e-15);
        let coherence = (1.0 - r.gamma).sqrt() * (1.0 - 2.0 * r.z);
        assert!(
            (coherence - (-t / t2).exp()).abs() < 1e-15,
            "{} {} {}",
            t1,
            t2,
            t
        );
        assert!((0.0..=0.5).contains(&r.z));
    }
}

#[test]
fn amplitude_damping_matches_t1() {
    // |1> eta |+> iraupen batean: P(1) = e^{-t/T1} eta e^{-t/T1}/2
    let m = model(
        r#"{"t1": 1e-6, "t2": null, "p0m1": 0, "p1m0": 0}"#,
        r#"{"name": "x", "qubits": [0], "error": 0, "duration": 0},
           {"name": "sx", "qubits": [0], "error": 0, "duration": 0},
           {"name": "rz", "qubits": [0], "error": 0, "duration": 5e-7}"#,
    );
    let shots = 100_000;
    let decay = (-0.5f64).exp();
    let out = run("QINIT 1\nX q0\nRZ q0 0\nMEASURE q0\n", &m, shots).unwrap();
    close(share(&out, &[(0, 1)]), decay, shots);
    let out = run("QINIT 1\nSX q0\nRZ q0 0\nMEASURE q0\n", &m, shots).unwrap();
    close(share(&out, &[(0, 1)]), decay / 2.0, shots);
}

#[test]
fn ramsey_fringe_matches_t2() {
    // SX, itxaron, SX: P(1) = (1 + e^{-t/T2}) / 2, T1-ren menpekotasunik gabe
    let m = model(
        r#"{"t1": 1e-5, "t2": 2e-6, "p0m1": 0, "p1m0": 0}"#,
        r#"{"name": "sx", "qubits": [0], "error": 0, "duration": 0},
           {"name": "rz", "qubits": [0], "error": 0, "duration": 1e-6}"#,
    );
    let shots = 100_000;
    let out = run("QINIT 1\nSX q0\nRZ q0 0\nSX q0\nMEASURE q0\n", &m, shots).unwrap();
    assert_eq!(out.method, Method::EveryShot);
    close(share(&out, &[(0, 1)]), (1.0 + (-0.5f64).exp()) / 2.0, shots);
}

#[test]
fn readout_errors_flip_the_recorded_bit() {
    let m = model(
        r#"{"t1": null, "t2": null, "p0m1": 0.1, "p1m0": 0.25},
           {"t1": null, "t2": null, "p0m1": 0.1, "p1m0": 0.25}"#,
        r#"{"name": "x", "qubits": [1], "error": 0, "duration": 0}"#,
    );
    let shots = 100_000;
    for src in [
        "QINIT 2\nX q1\nMEASURE q0\nMEASURE q1\n",
        "QINIT 2\nX q1\nMEASUREALL\n",
    ] {
        let out = run(src, &m, shots).unwrap();
        close(share(&out, &[(0, 1)]), 0.1, shots);
        close(share(&out, &[(1, 0)]), 0.25, shots);
    }
}

#[test]
fn two_qubit_depolarizing_spreads_evenly() {
    // cz-ren errorea e iraupenik gabe |00>-n: 15 pauli-etatik 12k bitak aldatzen dituzte, eta
    // P(01) = P(10) = P(11) = e/3. eredua [1, 0] ordenan dago: cz simetrikoa da
    let m = model(
        r#"{"t1": null, "t2": null, "p0m1": 0, "p1m0": 0},
           {"t1": null, "t2": null, "p0m1": 0, "p1m0": 0}"#,
        r#"{"name": "cz", "qubits": [1, 0], "error": 0.2, "duration": 0}"#,
    );
    let shots = 200_000;
    let out = run("QINIT 2\nCZ q0 q1\nMEASURE q0\nMEASURE q1\n", &m, shots).unwrap();
    for (b0, b1) in [(1, 0), (0, 1), (1, 1)] {
        close(share(&out, &[(0, b0), (1, b1)]), 0.2 / 3.0, shots);
    }
}

#[test]
fn operations_a_device_cannot_run_are_errors() {
    let m = model(
        r#"{"t1": null, "t2": null, "p0m1": 0, "p1m0": 0},
           {"t1": null, "t2": null, "p0m1": 0, "p1m0": 0}"#,
        r#"{"name": "sx", "qubits": [0], "error": 0, "duration": 0}"#,
    );
    let message = |src: &str| run(src, &m, 10).unwrap_err().to_string();
    let e = message("QINIT 1\nH q0\nMEASURE q0\n");
    assert!(
        e.contains("no 'h' on q0") && e.contains("transpile"),
        "{}",
        e
    );
    let e = message("QINIT 2\nSX q1\nMEASURE q1\n");
    assert!(e.contains("no 'sx' on q1") && e.contains("sx"), "{}", e);
    let e = message("QINIT 1\nMEASUREIN q0 X\n");
    assert!(e.contains("Z"), "{}", e);
    let e = message("QINIT 2\nBELL q0 q1\nMEASUREALL\n");
    assert!(e.contains("BELL"), "{}", e);
    let e = message("QINIT 3\nSX q0\nMEASURE q0\n");
    assert!(e.contains("3 qubits") && e.contains("only 2"), "{}", e);
    let e = message("QINIT 1\nQMODE 1 4\nDISPLACE o0 0.5 0\nMEASURE q0\n");
    assert!(e.contains("optical"), "{}", e);
}

#[test]
fn malformed_models_are_rejected() {
    let q = r#"{"t1": 1e-4, "t2": 5e-4, "p0m1": 0, "p1m0": 0}"#;
    let ok = model(q, "");
    let parsed = NoiseModel::parse(&ok).unwrap();
    // T2 > 2·T1 ez da fisikoa: aer-ek bezala mugatzen da
    assert_eq!(parsed.qubits[0].t2, 2e-4);
    for (bad, why) in [
        (ok.replace("qoa-noise-1", "qoa-noise-9"), "qoa-noise-9"),
        (ok.replace("1e-4", "-1e-4"), "t1"),
        (ok.replace(r#""p0m1": 0"#, r#""p0m1": 2"#), "p0m1"),
        (ok.replace(r#""p0m1""#, r#""p0m2""#), "p0m2"),
        (
            model(
                q,
                r#"{"name": "x", "qubits": [3], "error": 0, "duration": 0}"#,
            ),
            "qubit 3",
        ),
        (
            model(
                q,
                r#"{"name": "x", "qubits": [0], "error": 0, "duration": 0},
                   {"name": "x", "qubits": [0], "error": 0, "duration": 0}"#,
            ),
            "twice",
        ),
        (
            model(
                q,
                r#"{"name": "x", "qubits": [0], "error": 1.5, "duration": 0}"#,
            ),
            "error",
        ),
    ] {
        let e = NoiseModel::parse(&bad).unwrap_err();
        assert!(e.contains(why), "{}: {}", why, e);
    }
}
