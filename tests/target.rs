// fitxategi honek hardware-profilak probatzen ditu (`qoa check --target`, src/target.rs, docs/bench.md):
// banku batek exekuta dezakeena onartzen da, eta ezin duen eragiketa bakoitzak bere lerroa eta arrazoia ditu

use qoa::lang::{self, Lang};
use qoa::target::{Target, check};

fn problems(src: &str, t: Target) -> Vec<String> {
    lang::set(Lang::En);
    let p = qoa::asm::assemble(src, "t")
        .unwrap_or_else(|d| panic!("{}", qoa::asm::render(&d, src, "t")));
    check(&p, t).iter().map(|e| e.to_string()).collect()
}

#[test]
fn bench_programs_pass_and_the_rest_say_why() {
    lang::set(Lang::En);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // b mailako bankuan dabiltzan adibideak, eta bakoitzaren lehen arrazoia gainerakoentzat
    let runs_b = [
        "mach-zehnder.qoa",
        "hom-clicks.qoa",
        "bell-photons.qoa",
        "hello.qoa",
    ];
    for e in std::fs::read_dir(root.join("examples")).unwrap() {
        let path = e.unwrap().path();
        if path.extension().is_none_or(|x| x != "qoa") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let src = std::fs::read_to_string(&path).unwrap();
        let p = qoa::asm::assemble(&src, &name).unwrap();
        let found = check(&p, Target::BenchB);
        if runs_b.contains(&name.as_str()) {
            assert!(found.is_empty(), "{}: {:?}", name, found);
        } else {
            assert!(!found.is_empty(), "{} runs on bench-b?", name);
        }
    }
}

#[test]
fn each_refusal_names_the_line_and_the_reason() {
    let b = Target::BenchB;
    let a = Target::BenchA;
    let cases: [(&str, Target, &str); 16] = [
        (
            "QI 1\nHq0\nDUMPSTATE\n",
            b,
            "line 3: DUMPSTATE reads the exact quantum state",
        ),
        (
            "QO 1 3\nKo0 0.1\n",
            b,
            "line 2: KERR: a nonlinearity at the single-photon level",
        ),
        (
            "QO 1 3\nSPo0\nNo0\n",
            b,
            "line 3: COUNT: counting photons needs number-resolving",
        ),
        (
            "QI 2\nHq0\nCXq0 q1\n",
            b,
            "line 3: CNOT: two-qubit gates on photons",
        ),
        (
            "QI 1\nagain:\nHq0\nJ again\n",
            b,
            "line 4: JUMP: the bench runs the same fixed sequence",
        ),
        ("QI 1\nMq0\nHq0\n", b, "line 3: q0 was measured already"),
        (
            "QI 2\nHq0\nBq0 q1\n",
            b,
            "line 3: BELL is the photon-pair source on bench-b",
        ),
        (
            "QO 3 3\nSPo0\nSPo1\nSPo2\n",
            b,
            "line 4: bench-b makes 2 photon(s) per shot at most",
        ),
        ("QO 1 3\nDo0 0.5\n", b, "line 2: DISPLACE needs a laser"),
        (
            "QO 2 3\nSPo0\nCDo0 o1.c0\n",
            a,
            "line 3: COINCIDENCE needs two photons",
        ),
        (
            "QI 1\nQO 1 3\nHq0\nSPo0\n",
            b,
            "qubits are photon polarisations and the modes are photon paths",
        ),
        (
            "QO 1 3\nFo0 2\n",
            b,
            "line 2: FOCK 2: the source makes single photons",
        ),
        (
            "QO 5 3\nPDo0\nPDo1\nPDo2\nPDo3\nPDo4\n",
            b,
            "the program reads 5 detectors; bench-b has 4",
        ),
        (
            "QI 2\nBq0 q1\n",
            a,
            "line 2: BELL needs an entangled photon pair",
        ),
        (
            "QI 3\nHq0\n",
            b,
            "the program has 3 qubits; bench-b has 2 at most",
        ),
        (
            "QI 1\nRSTq0\n",
            b,
            "line 2: RESET: a photon cannot be put back",
        ),
    ];
    for (src, t, want) in cases {
        let got = problems(src, t);
        assert!(
            got.iter().any(|g| g.contains(want)),
            "{:?} on {}: {:?}",
            src,
            t.name(),
            got
        );
    }
    // eta onartzen direnak
    for (src, t) in [
        ("QO 1 3\nDo0 0.5\nPDo0\n", a),
        ("QI 1\nRYq0 0.3\nMI q0 X\n", a),
        ("QO 2 3\nSPo0\nSPo1\nBSo0 o1 pi/4\nCDo0 o1.c0\n", b),
        ("QI 2\nBq0 q1\nRYq0 0.4\nTBq1 X\nMq0\nMq1\n", b),
    ] {
        assert!(
            problems(src, t).is_empty(),
            "{:?} on {}: {:?}",
            src,
            t.name(),
            problems(src, t)
        );
    }
}

#[test]
fn the_cli_refuses_with_lines_in_both_languages() {
    let qoa = env!("CARGO_BIN_EXE_qoa");
    let run = |lang: &str, args: &[&str]| {
        std::process::Command::new(qoa)
            .args(args)
            .env("QOA_LANG", lang)
            .output()
            .unwrap()
    };
    let o = run("en", &["check", "--target", "bench-b", "examples/bell.qoa"]);
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("cannot run on bench-b") && err.contains("line 7: CNOT"),
        "{}",
        err
    );
    let o = run("eu", &["check", "--target", "bench-b", "examples/bell.qoa"]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("ezin da bench-b-n exekutatu") && err.contains("7. lerroa: CNOT"),
        "{}",
        err
    );
    let o = run(
        "en",
        &["check", "--target", "bench-b", "examples/bell-photons.qoa"],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stdout).contains("runs on bench-b"));
    let o = run("en", &["check", "--target", "nope", "examples/bell.qoa"]);
    assert!(
        String::from_utf8_lossy(&o.stderr)
            .contains("unknown target 'nope' (known: bench-a, bench-b)")
    );
}

// banku-eredua (src/noise.rs, "qoa-bench-1"): banaketak forma itxiko formulekin bat datoz
fn bench_run(file: &str, json: &str, shots: u64) -> Vec<(qoa::vm::Record, u64)> {
    lang::set(Lang::En);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(root.join(file)).unwrap();
    let p = qoa::asm::assemble(&src, file).unwrap();
    let cfg = qoa::vm::RunConfig {
        shots,
        seed: 11,
        bench: Some(std::sync::Arc::new(
            qoa::noise::BenchModel::parse(json).unwrap(),
        )),
        ..Default::default()
    };
    qoa::vm::run(&p, &cfg, &mut std::io::sink()).unwrap().counts
}

fn share(counts: &[(qoa::vm::Record, u64)], f: impl Fn(&qoa::vm::Record) -> bool) -> f64 {
    let total: u64 = counts.iter().map(|(_, n)| n).sum();
    counts
        .iter()
        .filter(|(r, _)| f(r))
        .map(|(_, n)| *n)
        .sum::<u64>() as f64
        / total as f64
}

#[test]
fn bench_models_give_the_closed_form_shares() {
    let shots = 100_000;
    // ±4 desbideratze estandar shot-kopuru honetarako
    let tol = 4.0 * (0.25 / shots as f64).sqrt();
    // mach-zehnder: fotoia h probabilitatez, detektatua η probabilitatez; π/3 fasearekin o1-era 3/4
    let (eta, h) = (0.6, 0.5);
    let json = format!(
        r#"{{"format": "qoa-bench-1", "detector": {{"efficiency": {}}}, "source": {{"heralding": {}}}}}"#,
        eta, h
    );
    let c = bench_run("examples/mach-zehnder.qoa", &json, shots);
    let o =
        |r: &qoa::vm::Record, a: u32, b: u32| r.readouts[0] == Some(a) && r.readouts[1] == Some(b);
    for (a, b, want) in [
        (0, 0, 1.0 - h * eta),
        (0, 1, h * eta * 0.75),
        (1, 0, h * eta * 0.25),
    ] {
        let got = share(&c, |r| o(r, a, b));
        assert!(
            (got - want).abs() < tol,
            "o0={} o1={}: {} vs {}",
            a,
            b,
            got,
            want
        );
    }
    // bell: bi emaitzak η² probabilitatez; haien artean, berdinak (1 + V)/2 (werner egoera)
    let (eta, v) = (0.7, 0.8);
    let json = format!(
        r#"{{"format": "qoa-bench-1", "detector": {{"efficiency": {}}}, "source": {{"visibility": {}}}}}"#,
        eta, v
    );
    let c = bench_run("examples/bell-photons.qoa", &json, shots);
    let both = share(&c, |r| r.bits.iter().all(|b| b.is_some()));
    assert!((both - eta * eta).abs() < tol, "{} vs {}", both, eta * eta);
    let equal = share(&c, |r| {
        r.bits.iter().all(|b| b.is_some()) && r.bits[0] == r.bits[1]
    }) / both;
    assert!(
        (equal - (1.0 + v) / 2.0).abs() < 2.0 * tol,
        "{} vs {}",
        equal,
        (1.0 + v) / 2.0
    );
    // klik ilunak: fotoia beti iristen da, eta beste detektagailuaren klik ilun batek emaitza baztertzen du
    let d = 0.1;
    let json = format!(
        r#"{{"format": "qoa-bench-1", "detector": {{"dark": {}}}}}"#,
        d
    );
    let c = bench_run("examples/bell-photons.qoa", &json, shots);
    let first = share(&c, |r| r.bits[0].is_none());
    assert!((first - d).abs() < tol, "{} vs {}", first, d);
    // hong-ou-mandel bi fotoiko igorpenarekin: |2,1> sarrerak bi irteerak batera piztu ditzake
    let json = r#"{"format": "qoa-bench-1", "source": {"two_photon": 0.2}}"#;
    let c = bench_run("examples/hom-clicks.qoa", json, 20_000);
    assert!(share(&c, |r| o(r, 1, 1)) > 0.02);
    let c = bench_run(
        "examples/hom-clicks.qoa",
        r#"{"format": "qoa-bench-1"}"#,
        20_000,
    );
    assert_eq!(share(&c, |r| o(r, 1, 1)), 0.0);
}

#[test]
fn bench_models_refuse_what_the_bench_cannot_do() {
    lang::set(Lang::En);
    let run = |src: &str, json: &str| {
        let p = qoa::asm::assemble(src, "t").unwrap();
        let cfg = qoa::vm::RunConfig {
            bench: Some(std::sync::Arc::new(
                qoa::noise::BenchModel::parse(json).unwrap(),
            )),
            ..Default::default()
        };
        qoa::vm::run(&p, &cfg, &mut std::io::sink())
            .unwrap_err()
            .message
    };
    let e = run(
        "QI 2\nHq0\nCXq0 q1\nMq0\nMq1\n",
        r#"{"format": "qoa-bench-1"}"#,
    );
    assert!(
        e.contains("describes a bench-b bench") && e.contains("line 3: CNOT"),
        "{}",
        e
    );
    let e = run(
        "QO 1 2\nSPo0\nPDo0\n",
        r#"{"format": "qoa-bench-1", "source": {"two_photon": 0.1}}"#,
    );
    assert!(e.contains("photon cutoff of at least 3"), "{}", e);
    for (json, want) in [
        (
            r#"{"format": "qoa-bench-1", "detector": {"efficiency": 1.5}}"#,
            "detector.efficiency is 1.5",
        ),
        (
            r#"{"format": "qoa-bench-1", "target": "bench-z"}"#,
            "unknown target 'bench-z'",
        ),
        (
            r#"{"format": "qoa-bench-1", "detector": {"eff": 0.5}}"#,
            "unknown field",
        ),
        (
            r#"{"format": "qoa-bench-2"}"#,
            "bench model format 'qoa-bench-2' is not supported",
        ),
    ] {
        let e = qoa::noise::BenchModel::parse(json).unwrap_err();
        assert!(e.contains(want), "{}: {}", json, e);
    }
}

// gailu-protokoloa (src/device.rs): lana eta haren iturburua programa bera dira, eta bench-sim-en bidezko
// exekuzioak simulazio zuzenaren kontaketa berberak ematen ditu
#[test]
fn device_jobs_round_trip_to_the_same_program() {
    lang::set(Lang::En);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for (file, target) in [
        ("examples/mach-zehnder.qoa", Target::BenchB),
        ("examples/hom-clicks.qoa", Target::BenchB),
        ("examples/bell-photons.qoa", Target::BenchB),
        ("examples/mach-zehnder.qoa", Target::BenchA),
    ] {
        let src = std::fs::read_to_string(root.join(file)).unwrap();
        let p = qoa::asm::assemble(&src, file).unwrap();
        let job = qoa::device::job(&p, target, 100, 3).unwrap();
        let back_src = qoa::device::job_source(&job["job"]).unwrap();
        let back = qoa::asm::assemble(&back_src, file).unwrap();
        assert_eq!(p.ops, back.ops, "{}\n{}", file, back_src);
        assert_eq!(
            (p.n_qubits, p.n_modes, p.n_bits),
            (back.n_qubits, back.n_modes, back.n_bits),
            "{}",
            file
        );
    }
    // gate bakoitzaren U(θ, φ, λ) angeluak: xafla bidezko bankuentzat, eta izena simulagailuarentzat
    let p = qoa::asm::assemble("QI 1\nRXq0 0.3\nSXq0\nTq0\nMq0\n", "t").unwrap();
    let job = qoa::device::job(&p, Target::BenchB, 10, 1).unwrap();
    let steps = job["job"]["steps"].as_array().unwrap();
    assert_eq!(steps[0]["gate"], "RX");
    assert_eq!(steps[0]["u"][0], 0.3);
    assert_eq!(steps[1]["gate"], "SX");
    assert_eq!(steps[3]["op"], "measure");
}

#[test]
fn a_device_run_through_bench_sim_matches_the_simulation() {
    let qoa = env!("CARGO_BIN_EXE_qoa");
    let run = |args: &[&str]| {
        let o = std::process::Command::new(qoa)
            .args(args)
            .env("QOA_LANG", "en")
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
        v["counts"].clone()
    };
    let controller = format!("{} bench-sim --model examples/bench-b.json", qoa);
    for file in ["examples/bell-photons.qoa", "examples/hom-clicks.qoa"] {
        let common = ["run", file, "--shots", "3000", "--seed", "8", "--json"];
        let mut dev: Vec<&str> = common.to_vec();
        dev.extend(["--device", controller.as_str()]);
        let mut sim: Vec<&str> = common.to_vec();
        sim.extend(["--noise-model", "examples/bench-b.json"]);
        assert_eq!(run(&dev), run(&sim), "{}", file);
    }
}

#[test]
fn device_answers_are_checked() {
    lang::set(Lang::En);
    let p = qoa::asm::assemble("QI 2\nBq0 q1\nMq0\nMq1\n", "t").unwrap();
    let reply =
        |v: serde_json::Value| qoa::device::read_reply(&p, &v).map(|(_, n, c)| (n, c.len()));
    let ok = serde_json::json!({"protocol": "qoa-device-1", "device": "d", "shots": 3,
        "counts": [{"bits": "11", "readouts": [], "count": 2}, {"bits": "-0", "readouts": [], "count": 1}]});
    assert_eq!(reply(ok), Ok((3, 2)));
    for (v, want) in [
        (
            serde_json::json!({"protocol": "qoa-device-1", "error": "laser off"}),
            "the device says: laser off",
        ),
        (
            serde_json::json!({"protocol": "other"}),
            "did not answer in the qoa-device-1 protocol",
        ),
        (
            serde_json::json!({"protocol": "qoa-device-1", "shots": 1, "counts": [{"bits": "1", "count": 1}]}),
            "returned bits '1', but the program has 2 bit(s)",
        ),
        (
            serde_json::json!({"protocol": "qoa-device-1", "shots": 5, "counts": [{"bits": "11", "count": 1}]}),
            "counts add up to 1, but it says it ran 5 shot(s)",
        ),
    ] {
        let e = reply(v).unwrap_err();
        assert!(e.contains(want), "{}", e);
    }
    // bench-sim: lerro txarrek errore-erantzun bat dute, ez dute prozesua eteten
    let mut out = Vec::new();
    let input = b"not json\n{\"protocol\": \"qoa-device-1\", \"job\": {\"program\": \"x\", \"qubits\": 1, \"modes\": 0, \"bits\": 1, \"shots\": 1, \"steps\": [{\"op\": \"teleport\"}]}}\n";
    qoa::device::serve(None, &input[..], &mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "{}", text);
    assert!(
        lines[0].contains("not json") && lines[1].contains("unknown step 'teleport'"),
        "{}",
        text
    );
}
