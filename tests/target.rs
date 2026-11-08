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
