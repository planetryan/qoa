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

// probetarako wav txiki bat: 0.5 s, 8000 Hz, estereo, 16 bit; tonu bat eta kolpe batzuk
fn write_wav(path: &Path) {
    let rate = 8000u32;
    let n = rate / 2;
    let mut data = Vec::new();
    for i in 0..n {
        let t = i as f64 / rate as f64;
        let beat = if i % 1000 < 80 { 0.5 } else { 0.0 };
        let l = 0.4 * (2.0 * std::f64::consts::PI * 440.0 * t).sin() + beat;
        let r = 0.4 * (2.0 * std::f64::consts::PI * 660.0 * t).sin() - beat;
        for v in [l, r] {
            data.extend_from_slice(&((v * 32767.0) as i16).to_le_bytes());
        }
    }
    let mut f = Vec::new();
    f.extend_from_slice(b"RIFF");
    f.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    f.extend_from_slice(b"WAVEfmt ");
    f.extend_from_slice(&16u32.to_le_bytes());
    f.extend_from_slice(&1u16.to_le_bytes()); // pcm
    f.extend_from_slice(&2u16.to_le_bytes()); // kanalak
    f.extend_from_slice(&rate.to_le_bytes());
    f.extend_from_slice(&(rate * 4).to_le_bytes());
    f.extend_from_slice(&4u16.to_le_bytes());
    f.extend_from_slice(&16u16.to_le_bytes());
    f.extend_from_slice(b"data");
    f.extend_from_slice(&(data.len() as u32).to_le_bytes());
    f.extend_from_slice(&data);
    std::fs::write(path, f).unwrap();
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success())
}

#[test]
fn visual_explains_bad_input() {
    let o = qoa("en", &["visual", "examples/bell.qoa", "out.mp4"]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("is not a .wav file"), "{}", stderr(&o));
    let o = qoa("eu", &["visual", "ez-dago.wav", "out.mp4"]);
    assert!(!o.status.success());
    assert!(
        stderr(&o).contains("ezin da 'ez-dago.wav' irakurri"),
        "{}",
        stderr(&o)
    );
    let wav = scratch("visual-bad.wav");
    write_wav(&wav);
    let o = qoa(
        "en",
        &[
            "visual",
            wav.to_str().unwrap(),
            "out.mp4",
            "--resolution",
            "wide",
        ],
    );
    assert!(!o.status.success());
    assert!(
        stderr(&o).contains("should look like 1920x1080"),
        "{}",
        stderr(&o)
    );
}

// irudi gordinak fitxategi batera (kodetu gabe), bideoa pixelez pixel konparatzeko
fn render_raw(wav: &Path, out: &Path, seed: &str) -> Vec<u8> {
    let o = qoa(
        "en",
        &[
            "visual",
            wav.to_str().unwrap(),
            out.to_str().unwrap(),
            "--resolution",
            "96x54",
            "--fps",
            "20",
            "--ltr",
            "--seed",
            seed,
            "--",
            "-an",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
        ],
    );
    assert!(o.status.success(), "{}{}", stdout(&o), stderr(&o));
    let report = stdout(&o);
    for want in [
        "input",
        "0.50 s, 8000 Hz, 2 channel(s) mixed to mono",
        "video",
        "96x54, 20 frames/s, 10 frames",
        "seed",
        "note",
        "time",
    ] {
        assert!(report.contains(want), "missing {:?} in:\n{}", want, report);
    }
    std::fs::read(out).unwrap()
}

#[test]
fn visual_renders_the_same_video_for_the_same_seed() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg ez dago; proba hau saltatzen da");
        return;
    }
    let wav = scratch("visual.wav");
    write_wav(&wav);
    let a = render_raw(&wav, &scratch("visual-a.rgb"), "5");
    let b = render_raw(&wav, &scratch("visual-b.rgb"), "5");
    let c = render_raw(&wav, &scratch("visual-c.rgb"), "6");
    // 0.5 s * 20 irudi/s = 10 irudi, bakoitza 96 * 54 * 3 byte
    assert_eq!(a.len(), 10 * 96 * 54 * 3);
    assert!(a == b, "hazi berak bideo bera eman behar du");
    assert!(a != c, "hazi desberdinak bideo desberdina eman behar du");
    // espektroa: 440 Hz-ko tonua 22. bin-ean dago (20 Hz bin bakoitzeko); 201 bin 96 zutabetan, beraz
    // 10. zutabean, eta barra gorri betea da (altuera 54 / 4 = 13 pixel)
    let frame = &a[..96 * 54 * 3];
    let px = |x: usize, y: usize| &frame[(y * 96 + x) * 3..(y * 96 + x) * 3 + 3];
    for y in 41..54 {
        assert!(
            px(10, y)[0] >= 204 && px(10, y)[1] < 52,
            "{} {:?}",
            y,
            px(10, y)
        );
    }
    assert!(px(10, 40)[0] < 204 || px(10, 40)[1] > 51);

    // mp4 bat ere bai, kodetzaile lehenetsiekin
    let mp4 = scratch("visual.mp4");
    let o = qoa(
        "eu",
        &[
            "visual",
            wav.to_str().unwrap(),
            mp4.to_str().unwrap(),
            "--resolution",
            "64x36",
            "--fps",
            "10",
            "--seed",
            "1",
        ],
    );
    assert!(o.status.success(), "{}{}", stdout(&o), stderr(&o));
    assert!(stdout(&o).contains("monora nahastuta") && stdout(&o).contains("arte-tresna"));
    assert!(std::fs::metadata(&mp4).unwrap().len() > 1000);
}

#[test]
fn clap_help_and_errors_speak_basque_too() {
    let o = qoa("eu", &["visual", "--help"]);
    assert!(o.status.success());
    let out = stdout(&o);
    for want in [
        "Erabilera:",
        "Argumentuak:",
        "Aukerak:",
        "[lehenetsia: 60]",
        "Laguntza inprimatu",
    ] {
        assert!(out.contains(want), "missing {:?} in:\n{}", want, out);
    }
    for unwanted in ["Usage:", "Options:", "[default:", "Print help"] {
        assert!(!out.contains(unwanted), "{:?} in:\n{}", unwanted, out);
    }
    let o = qoa("eu", &["run", "examples/bell.qoa", "--shots", "abc"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(
        stderr(&o).contains("errorea: balio baliogabea 'abc'")
            && stderr(&o).contains("ez da zenbaki oso bat"),
        "{}",
        stderr(&o)
    );
    // ingelesez clap-en testuak bere horretan
    let o = qoa("en", &["visual", "--help"]);
    assert!(stdout(&o).contains("Usage:") && stdout(&o).contains("[default: 60]"));
}

#[test]
fn glyph_files_run_import_and_point_errors_at_the_glyph() {
    let o = qoa("en", &["run", "examples/bell.qg", "--seed", "1"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(stdout(&o).contains("examples/bell.qg") && stdout(&o).contains("11"));
    let o = qoa("en", &["import", "examples/bell.qg"]);
    assert!(o.status.success());
    assert_eq!(
        stdout(&o),
        "QINIT 2\nH q0\nCNOT q0 q1\nMEASURE q0 -> c0\nMEASURE q1 -> c1\n"
    );
    let bad = scratch("bad.qg");
    std::fs::write(&bad, "Q{2};\nH/Q[0]; FROB/Q[1];\n").unwrap();
    let o = qoa("eu", &["check", bad.to_str().unwrap()]);
    assert!(!o.status.success());
    let err = stderr(&o);
    assert!(
        err.contains("bad.qg:2:9") && err.contains("FROB/Q[1]"),
        "{}",
        err
    );
}
