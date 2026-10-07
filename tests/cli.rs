// fitxategi honek `qoa` programa bitarra bera probatzen du: komandoak, irteera-kodeak, bi hizkuntzak,
// json irteera eta docs/ISA.md fitxategiak sortzailearekin bat egiten duten.

use std::path::Path;
use std::process::{Command, Output};

fn qoa(lang: &str, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_qoa"))
        .args(args)
        .env("QOA_LANG", lang)
        .env("NO_COLOR", "1")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("qoa runs")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    dir.join(name)
}

#[test]
fn run_prints_a_direct_report() {
    let o = qoa("en", &["run", "examples/bell.qoa", "--seed", "1"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let out = stdout(&o);
    for want in [
        "program",
        "system",
        "2 qubit(s) = 4 amplitudes",
        "noise",
        "shots",
        "seed",
        "outcome",
        "c1..c0",
        "00",
        "11",
        "time",
    ] {
        assert!(out.contains(want), "missing {:?} in:\n{}", want, out);
    }
}

#[test]
fn run_speaks_basque() {
    let o = qoa("eu", &["run", "examples/bell.qoa", "--seed", "1"]);
    assert!(o.status.success());
    let out = stdout(&o);
    for want in [
        "programa", "sistema", "zarata", "hazia", "emaitza", "kopurua", "ehunekoa", "denbora",
    ] {
        assert!(out.contains(want), "missing {:?} in:\n{}", want, out);
    }
}

#[test]
fn errors_exit_with_failure_and_point_at_the_line() {
    let bad = scratch("bad.qoa");
    std::fs::write(&bad, "QINIT 2\nCNOT q0 q0\n").unwrap();
    let o = qoa("en", &["run", bad.to_str().unwrap()]);
    assert!(!o.status.success());
    let err = stderr(&o);
    assert!(err.contains("error: CNOT uses qubit 0 twice"), "{}", err);
    assert!(err.contains("bad.qoa:2:9"), "{}", err);
    let o = qoa("eu", &["run", bad.to_str().unwrap()]);
    assert!(
        stderr(&o).contains("errorea: CNOT-k 0. qubita bi aldiz erabiltzen du"),
        "{}",
        stderr(&o)
    );
}

#[test]
fn same_seed_same_results() {
    let a = stdout(&qoa(
        "en",
        &["run", "examples/grover.qoa", "--seed", "99", "--json"],
    ));
    let b = stdout(&qoa(
        "en",
        &["run", "examples/grover.qoa", "--seed", "99", "--json"],
    ));
    let ca: serde_json::Value = serde_json::from_str(&a).unwrap();
    let cb: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(ca["counts"], cb["counts"]);
}

#[test]
fn json_is_the_result_record() {
    let o = qoa(
        "en",
        &[
            "run",
            "examples/hom.qoa",
            "--seed",
            "3",
            "--shots",
            "500",
            "--json",
        ],
    );
    assert!(o.status.success(), "{}", stderr(&o));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["shots"], 500);
    assert_eq!(v["modes"], 2);
    let total: u64 = v["counts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["count"].as_u64().unwrap())
        .sum();
    assert_eq!(total, 500);
    for c in v["counts"].as_array().unwrap() {
        // hong-ou-mandel: (1,1) inoiz ez
        assert_ne!(c["modes"], serde_json::json!([1, 1]));
    }
}

#[test]
fn compile_then_run_the_binary() {
    let out = scratch("teleport.qx");
    let o = qoa(
        "en",
        &[
            "compile",
            "examples/teleport.qoa",
            "-o",
            out.to_str().unwrap(),
        ],
    );
    assert!(o.status.success(), "{}", stderr(&o));
    let a = stdout(&qoa(
        "en",
        &["run", out.to_str().unwrap(), "--seed", "5", "--json"],
    ));
    let b = stdout(&qoa(
        "en",
        &["run", "examples/teleport.qoa", "--seed", "5", "--json"],
    ));
    let va: serde_json::Value = serde_json::from_str(&a).unwrap();
    let vb: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(va["counts"], vb["counts"]);
}

#[test]
fn old_binaries_are_refused_with_an_explanation() {
    let old = scratch("old.qx");
    std::fs::write(&old, b"QEXE\x01\x05\x00").unwrap();
    let o = qoa("en", &["run", old.to_str().unwrap()]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("built by QOA 0.3.x"), "{}", stderr(&o));
}

#[test]
fn check_reports_the_resources() {
    let o = qoa("en", &["check", "examples/cat.qoa"]);
    assert!(o.status.success());
    let out = stdout(&o);
    assert!(out.contains("assembles without errors"), "{}", out);
    assert!(
        out.contains("1 qubit(s), 1 mode(s) of 0..23 photons"),
        "{}",
        out
    );
}

#[test]
fn qasm_files_run_directly_and_import_and_export_work() {
    let o = qoa("en", &["run", "examples/bell.qasm", "--seed", "2"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let o = qoa("en", &["import", "examples/bell.qasm"]);
    assert!(stdout(&o).contains("CNOT q0 q1"), "{}", stdout(&o));
    let o = qoa("en", &["export", "examples/ghz.qoa", "--to", "qasm"]);
    assert!(stdout(&o).contains("OPENQASM 2.0;"), "{}", stdout(&o));
    let o = qoa("en", &["export", "examples/hom.qoa", "--to", "qasm"]);
    assert!(!o.status.success());
    assert!(
        stderr(&o).contains("cannot be exported to OpenQASM 2.0"),
        "{}",
        stderr(&o)
    );
    let o = qoa("en", &["export", "examples/bell.qoa", "--to", "ionq"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["input"]["qubits"], 2);
}

#[test]
fn isa_describes_instructions_in_both_languages() {
    let o = qoa("en", &["isa", "cx"]);
    assert!(
        stdout(&o).contains("usage      CNOT control target"),
        "{}",
        stdout(&o)
    );
    let o = qoa("eu", &["isa", "cx"]);
    assert!(
        stdout(&o).contains("erabilera  CNOT kontrola helburua"),
        "{}",
        stdout(&o)
    );
    let o = qoa("en", &["isa", "setpos"]);
    assert!(stdout(&o).contains("removed in QOA 0.4"), "{}", stdout(&o));
}

#[test]
fn generated_docs_are_up_to_date() {
    // docs/ISA.md eta docs/ISA.eu.md `qoa isa --markdown`-ren irteera berdinak izan behar dira
    for (lang, file) in [("en", "docs/ISA.md"), ("eu", "docs/ISA.eu.md")] {
        let generated = stdout(&qoa(lang, &["isa", "--markdown"]));
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(file);
        let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            generated == on_disk,
            "{} is stale; regenerate it with: QOA_LANG={} cargo run -- isa --markdown > {}",
            file,
            lang,
            file
        );
    }
}

#[test]
fn bad_arguments_are_rejected() {
    let o = qoa("en", &["run", "examples/bell.qoa", "--noise", "1.5"]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("not a probability"), "{}", stderr(&o));
    let o = qoa("en", &["run", "does/not/exist.qoa"]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("cannot read"), "{}", stderr(&o));
}
