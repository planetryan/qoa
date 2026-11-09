// fitxategi honek `qoa compile --emit` probatzen du: urrats bakoitza idazten dela, arkitektura guztien
// mihiztadura mihiztatzen dela, eta kode natiboak (liburutegi partekatu gisa) programa guztiak
// interpretatzaileak bezala exekutatzen dituela, hazi berarekin: kontaketa, erregistro, urrats, irteera eta
// errore berberak. ci-ak qemu-n exekutatzen du arkitektura bakoitzean (QOA_NATIVE_CC-k arkitektura horren
// C konpilatzailea dio)

use qoa::compile::Arch;
use qoa::program::Program;
use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn scratch(name: &str) -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn qoa(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_qoa"))
        .args(args)
        .env("QOA_LANG", "en")
        .env("NO_COLOR", "1")
        .current_dir(root())
        .output()
        .expect("qoa runs")
}

// probatzeko programak: adibideak, sanity eta stress (mihiztatzen diren failures-ak ere: exekuzio-erroreak)
fn programs() -> Vec<(String, Program)> {
    let mut out = Vec::new();
    for dir in [
        "examples",
        "tests/programs/sanity",
        "tests/programs/failures",
        "tests/programs/stress",
    ] {
        let mut files: Vec<PathBuf> = std::fs::read_dir(root().join(dir))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "qoa"))
            .collect();
        files.sort();
        for f in files {
            let src = std::fs::read_to_string(&f).unwrap();
            let name = f.strip_prefix(root()).unwrap().display().to_string();
            if let Ok(p) = qoa::asm::assemble(&src, &name) {
                out.push((name, p));
            }
        }
    }
    assert!(out.len() > 80, "{}", out.len());
    out
}

#[test]
fn every_stage_is_written() {
    let dir = scratch("stages");
    let _ = std::fs::create_dir_all(&dir);
    let f = "tests/programs/sanity/call-ret.qoa";
    // testuzko urratsak irteerara
    let o = qoa(&["compile", f, "--emit", "qoa"]);
    assert!(o.status.success());
    let text = String::from_utf8_lossy(&o.stdout).into_owned();
    assert!(text.contains("C addr_0x6"), "{}", text);
    let o = qoa(&["compile", f, "--emit", "ir"]);
    let ir = String::from_utf8_lossy(&o.stdout).into_owned();
    assert!(o.status.success());
    assert!(ir.contains("func qoa_native_entry"), "{}", ir);
    assert!(ir.contains("native"), "{}", ir);
    assert!(ir.contains("step"), "{}", ir);
    for arch in Arch::ALL {
        let o = qoa(&["compile", f, "--emit", "asm", "--arch", arch.name()]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let asm = String::from_utf8_lossy(&o.stdout).into_owned();
        assert!(asm.contains("qoa_native_entry:"), "{}", arch.name());
        assert!(asm.contains("qoa_native_qx:"), "{}", arch.name());
    }
    // qx urratsa: bytecode berbera
    let qx = dir.join("call-ret.qx");
    let o = qoa(&["compile", f, "--emit", "qx", "-o", qx.to_str().unwrap()]);
    assert!(o.status.success());
    let p = Program::from_bytes(&std::fs::read(&qx).unwrap()).unwrap();
    assert_eq!(qoa::compile::ir(&p).unwrap(), ir);
    // akatsak: --arch urrats okerrean, arkitektura ezezaguna, .so beste agindu batean
    let o = qoa(&["compile", f, "--emit", "ir", "--arch", "x86_64"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--arch is for"));
    let o = qoa(&["compile", f, "--emit", "asm", "--arch", "vax"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown architecture 'vax'"));
    let so = dir.join("x.so");
    std::fs::write(&so, b"not a library").unwrap();
    let o = qoa(&["export", so.to_str().unwrap(), "--to", "qoa"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("is native code"));
}

// arkitektura guztien mihiztadura objektu-fitxategi bihurtzen da (clang --target-ekin, linux-en). clang gabe
// proba saltatu egiten da, QOA_REQUIRE_CROSS_ASM=1 ezarrita ez badago (ci-ak ezartzen du)
#[cfg(target_os = "linux")]
#[test]
fn every_architecture_assembles() {
    let clang = Command::new("clang").arg("--version").output().is_ok();
    if !clang {
        assert!(
            std::env::var_os("QOA_REQUIRE_CROSS_ASM").is_none(),
            "QOA_REQUIRE_CROSS_ASM is set but clang is missing"
        );
        eprintln!("clang not found: every_architecture_assembles skipped");
        return;
    }
    let dir = scratch("cross");
    let _ = std::fs::create_dir_all(&dir);
    let tool = qoa::tools::find("asm").unwrap().program().unwrap();
    let mut progs = vec![("asm".to_string(), tool)];
    progs.extend(programs().into_iter().take(12));
    for arch in Arch::ALL {
        for (i, (name, p)) in progs.iter().enumerate() {
            let s = dir.join(format!("{}-{}.s", arch.name(), i));
            let o = dir.join(format!("{}-{}.o", arch.name(), i));
            std::fs::write(&s, qoa::compile::asm(p, arch).unwrap()).unwrap();
            let r = Command::new("clang")
                .arg(format!("--target={}", arch.triple()))
                .args(["-c", "-o"])
                .arg(&o)
                .arg(&s)
                .output()
                .unwrap();
            assert!(
                r.status.success(),
                "{} {}: {}",
                arch.name(),
                name,
                String::from_utf8_lossy(&r.stderr)
            );
        }
    }
}

#[cfg(all(target_os = "linux", not(target_env = "musl"), feature = "native"))]
mod run {
    use super::*;
    use qoa::lang::{self, Lang};
    use qoa::vm::{self, RunConfig};

    fn host() -> Arch {
        Arch::host().expect("linux on one of the six architectures")
    }

    // programaren liburutegia, makina honetako C konpilatzailearekin (QOA_NATIVE_CC, lehenetsia cc)
    fn build(name: &str, p: &Program) -> PathBuf {
        let safe: String = name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let s = scratch(&format!("native-{}.s", safe));
        let so = scratch(&format!("native-{}.so", safe));
        std::fs::write(&s, qoa::compile::asm(p, host()).unwrap()).unwrap();
        let cc = std::env::var("QOA_NATIVE_CC")
            .ok()
            .filter(|c| !c.trim().is_empty())
            .unwrap_or_else(|| "cc".into());
        let mut parts = cc.split_whitespace();
        let r = Command::new(parts.next().unwrap())
            .args(parts)
            .args(["-shared", "-nostdlib", "-o"])
            .arg(&so)
            .arg(&s)
            .output()
            .unwrap_or_else(|e| panic!("{}: {}", cc, e));
        assert!(
            r.status.success(),
            "{}: {}",
            name,
            String::from_utf8_lossy(&r.stderr)
        );
        so
    }

    type Outcome = Result<(Vec<(qoa::vm::Record, u64)>, Vec<u64>, u64, Option<u8>), (u32, String)>;

    fn outcome(p: &Program, cfg: &RunConfig) -> (Outcome, Vec<u8>) {
        let mut out = Vec::new();
        let r = vm::run(p, cfg, &mut out)
            .map(|o| {
                (
                    o.counts,
                    o.regs.iter().map(|r| r.to_bits()).collect(),
                    o.steps,
                    o.exit_status,
                )
            })
            .map_err(|e| (e.line, e.message));
        (r, out)
    }

    #[test]
    fn native_code_runs_every_program_like_the_interpreter() {
        lang::set(Lang::En);
        let mut n = 0;
        for (name, p) in programs() {
            let so = build(&name, &p);
            let loaded = qoa::native::load(&so).unwrap_or_else(|e| panic!("{}: {}", name, e));
            assert_eq!(loaded.program, p, "{}", name);
            let cfg = RunConfig {
                shots: 64,
                seed: 3,
                max_steps: 20_000_000,
                ..RunConfig::default()
            };
            let native = RunConfig {
                native: Some(loaded.entry),
                ..cfg.clone()
            };
            assert_eq!(outcome(&p, &native), outcome(&p, &cfg), "{}", name);
            n += 1;
        }
        assert!(n > 80, "{}", n);
    }

    // QOAn idatzitako mihiztatzailea kode natibo gisa: bere iturburua mihiztatzen du, eta .qx-a rust-ena da
    #[test]
    fn the_assembler_written_in_qoa_runs_as_native_code() {
        let tool = qoa::tools::find("asm").unwrap();
        let p = tool.program().unwrap();
        let loaded = qoa::native::load(&build("asm", &p)).unwrap();
        let cfg = RunConfig {
            shots: 1,
            max_steps: u64::MAX,
            args: vec![tool.path()],
            input: Some(std::sync::Arc::new(tool.source.as_bytes().to_vec())),
            native: Some(loaded.entry),
            ..RunConfig::default()
        };
        let mut out = Vec::new();
        let r = vm::run(&p, &cfg, &mut out).unwrap();
        assert_eq!(r.exit_status.unwrap_or(0), 0);
        assert_eq!(out, tool.binary);
    }

    #[test]
    fn qoa_run_runs_a_shared_library() {
        let so = scratch("cli-bell.so");
        let o = qoa(&[
            "compile",
            "examples/bell.qoa",
            "--emit",
            "so",
            "-o",
            so.to_str().unwrap(),
        ]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let native = qoa(&["run", so.to_str().unwrap(), "--seed", "5", "--json"]);
        let interp = qoa(&["run", "examples/bell.qoa", "--seed", "5", "--json"]);
        assert!(
            native.status.success(),
            "{}",
            String::from_utf8_lossy(&native.stderr)
        );
        let counts = |o: &std::process::Output| {
            let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
            v["counts"].clone()
        };
        assert_eq!(counts(&native), counts(&interp));
        // .so ez den fitxategi bat: errore argia
        let bad = scratch("not-native.so");
        std::fs::write(&bad, b"\x7fELF but not really").unwrap();
        let o = qoa(&["run", bad.to_str().unwrap()]);
        assert!(!o.status.success());
        assert!(
            String::from_utf8_lossy(&o.stderr).contains("cannot load"),
            "{}",
            String::from_utf8_lossy(&o.stderr)
        );
    }
}
