// fitxategi honek .qoa programak exekutatzen ditu eta haien goiburuko itxaropenak egiaztatzen ditu.
//
// programa bakoitzak `; expect-...` iruzkinetan esaten du zer gertatu behar den, beraz proba berri
// bat gehitzeko nahikoa da .qoa fitxategi bat idaztea karpeta egokian, rust idatzi gabe:
//
//   ; shots: 4000                    exekuzio-kopurua (lehenetsia 2000)
//   ; seed: 7                        hazia (lehenetsia 1)
//   ; max-steps: 1000                urrats-muga (amaigabeko begiztak probatzeko)
//   ; expect-error: testua           mihiztatze- edo exekuzio-errore batek testu hau izan behar du
//   ; expect-output: testua          programaren irteerak testu hau izan behar du
//   ; expect-only: 00 11             emaitza hauek bakarrik ager daitezke (gakoak, hutsunez bereizita)
//   ; expect-share: 101 0.945 0.02   emaitza, espero den zatia, tolerantzia
//   ; expect-prob: |11> 0.5          amaierako egoeran oinarri-egoera baten probabilitatea (1e-9 tolerantzia)
//   ; expect-leakage-below: 1e-6     fotoi-mugaren ihesa muga honen azpitik
//
// emaitza-gakoak: idatzitako bitak (c[n-1]..c0, '-' idatzi gabe), eta moduak badaude, "o0=1 o1=0".
// adibidez "11", "-0 o0=1", "o0=0 o1=2". zuriuneak '_' gisa idatz daitezke: "-0_o0=1".
// karpetak: tests/programs/{sanity,failures,stress}, examples, eta legacy (0.3 programa zaharrak).

use qoa::lang::{self, Lang};
use qoa::vm::{self, RunConfig};
use std::path::{Path, PathBuf};

#[derive(Default, Debug)]
struct Expect {
    shots: Option<u64>,
    seed: Option<u64>,
    max_steps: Option<u64>,
    error: Option<String>,
    output: Vec<String>,
    only: Option<Vec<String>>,
    shares: Vec<(String, f64, f64)>,
    probs: Vec<(String, f64)>,
    leakage_below: Option<f64>,
}

fn parse_expect(src: &str) -> Expect {
    let mut e = Expect::default();
    for line in src.lines() {
        // .qoa eta .qg fitxategietan `;`, .qasm fitxategietan `//`
        let t = line.trim();
        let Some(rest) = t.strip_prefix(';').or_else(|| t.strip_prefix("//")) else {
            continue;
        };
        let Some((key, value)) = rest.trim().split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "shots" => e.shots = value.parse().ok(),
            "seed" => e.seed = value.parse().ok(),
            "max-steps" => e.max_steps = value.parse().ok(),
            "expect-error" => e.error = Some(value.to_string()),
            "expect-output" => e.output.push(value.to_string()),
            "expect-only" => {
                e.only = Some(
                    value
                        .split_whitespace()
                        .map(|k| k.replace('_', " "))
                        .collect(),
                )
            }
            "expect-share" => {
                let p: Vec<&str> = value.split_whitespace().collect();
                e.shares.push((
                    p[0].replace('_', " "),
                    p[1].parse().unwrap(),
                    p[2].parse().unwrap(),
                ));
            }
            "expect-prob" => {
                let (basis, p) = value.rsplit_once(' ').unwrap();
                e.probs.push((basis.trim().to_string(), p.parse().unwrap()));
            }
            "expect-leakage-below" => e.leakage_below = value.parse().ok(),
            _ => {}
        }
    }
    e
}

// emaitza-gakoa, txostenaren antzera
fn key(rec: &vm::Record) -> String {
    let bits = qoa::report::bits_string(rec);
    let any_bits = rec.bits.iter().any(|b| b.is_some());
    let modes: Vec<String> = rec
        .readouts
        .iter()
        .enumerate()
        .filter_map(|(o, v)| v.map(|n| format!("o{}={}", o, n)))
        .collect();
    match (any_bits, modes.is_empty()) {
        (true, true) => bits,
        (true, false) => format!("{} {}", bits, modes.join(" ")),
        (false, _) => modes.join(" "),
    }
}

// programa bat exekutatu eta itxaropenak egiaztatu; huts eginez gero, zergatik itzultzen du
fn check(path: &Path) -> Result<(), String> {
    let src = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let name = path.display().to_string();
    let e = parse_expect(&src);
    let assembled = if path.extension().is_some_and(|x| x == "qasm") {
        qoa::qasm::to_qoa(&src, &name)
            .map_err(|err| err.to_string())
            .and_then(|q| {
                qoa::asm::assemble(&q, &name).map_err(|d| qoa::asm::render(&d, &q, &name))
            })
    } else if path.extension().is_some_and(|x| x == "qg") {
        qoa::glyph::assemble(&src, &name).map_err(|d| qoa::asm::render(&d, &src, &name))
    } else {
        qoa::asm::assemble(&src, &name).map_err(|d| qoa::asm::render(&d, &src, &name))
    };
    let program = match (assembled, &e.error) {
        (Ok(p), _) => p,
        (Err(msg), Some(want)) if msg.contains(want.as_str()) => return Ok(()),
        (Err(msg), Some(want)) => {
            return Err(format!(
                "expected an error containing {:?}, got:\n{}",
                want, msg
            ));
        }
        (Err(msg), None) => return Err(format!("did not assemble:\n{}", msg)),
    };
    let cfg = RunConfig {
        shots: e.shots.unwrap_or(2000),
        seed: e.seed.unwrap_or(1),
        max_steps: e.max_steps.unwrap_or(100_000_000),
        ..RunConfig::default()
    };
    let mut out_bytes = Vec::new();
    let out = match (vm::run(&program, &cfg, &mut out_bytes), &e.error) {
        (Ok(_), Some(want)) => {
            return Err(format!(
                "expected an error containing {:?}, but it ran",
                want
            ));
        }
        (Err(err), Some(want)) if err.to_string().contains(want.as_str()) => return Ok(()),
        (Err(err), Some(want)) => {
            return Err(format!(
                "expected an error containing {:?}, got: {}",
                want, err
            ));
        }
        (Err(err), None) => return Err(format!("runtime error: {}", err)),
        (Ok(o), None) => o,
    };
    let printed = String::from_utf8_lossy(&out_bytes);
    for want in &e.output {
        let want = want.replace("\\n", "\n");
        if !printed.contains(&want) {
            return Err(format!(
                "output does not contain {:?}; output was:\n{}",
                want, printed
            ));
        }
    }
    let total = out.shots as f64;
    let counts: Vec<(String, u64)> = out.counts.iter().map(|(r, n)| (key(r), *n)).collect();
    if let Some(only) = &e.only {
        for (k, n) in &counts {
            if !only.contains(k) {
                return Err(format!(
                    "outcome {:?} appeared {} times but only {:?} are allowed; counts: {:?}",
                    k, n, only, counts
                ));
            }
        }
    }
    for (k, share, tol) in &e.shares {
        let got = counts.iter().find(|c| &c.0 == k).map_or(0, |c| c.1) as f64 / total;
        if (got - share).abs() > *tol {
            return Err(format!(
                "outcome {:?}: share {} but expected {} ± {}; counts: {:?}",
                k, got, share, tol, counts
            ));
        }
    }
    for (basis, p) in &e.probs {
        let st = &out.final_state;
        let got: f64 = (0..st.len())
            .filter(|&i| &qoa::report::basis_label(st, i) == basis)
            .map(|i| st.amplitudes()[i].norm_sqr())
            .sum();
        if (got - p).abs() > 1e-9 {
            return Err(format!(
                "probability of {} is {} but expected {}",
                basis, got, p
            ));
        }
    }
    if let Some(limit) = e.leakage_below
        && out.leakage >= limit
    {
        return Err(format!(
            "truncation leakage {:e} is not below {:e}",
            out.leakage, limit
        ));
    }
    Ok(())
}

fn files_in(dir: &str) -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(dir);
    let mut v: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", root.display(), e))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|x| x == "qoa" || x == "qasm" || x == "qg")
        })
        .collect();
    v.sort();
    v
}

// karpeta bateko programa guztiak egiaztatu eta huts egin dutenak batera jakinarazi
fn check_dir(dir: &str) {
    lang::set(Lang::En);
    let files = files_in(dir);
    assert!(!files.is_empty(), "ez dago programarik {}-n", dir);
    let mut failures = Vec::new();
    for f in &files {
        if let Err(why) = check(f) {
            failures.push(format!("{}: {}", f.display(), why));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} programs failed:\n\n{}",
        failures.len(),
        files.len(),
        failures.join("\n\n")
    );
}

#[test]
fn sanity_programs() {
    check_dir("tests/programs/sanity");
}

#[test]
fn failure_programs_fail_with_the_right_message() {
    check_dir("tests/programs/failures");
}

#[test]
fn stress_programs() {
    check_dir("tests/programs/stress");
}

// mugetako programak: minutuak eta ram asko behar dituzte, beraz `--ignored` bidez bakarrik
#[test]
#[ignore]
fn limit_programs() {
    check_dir("tests/programs/limits");
}

#[test]
fn examples_do_what_their_comments_say() {
    check_dir("examples");
}

// 0.3 programa zaharrak: bakoitzak exekutatu behar du, edo 0.4-ko errore argi batekin huts egin.
// inoiz ez isilean ezer egin gabe (hori zen 0.3-ren arazoa).
#[test]
fn legacy_programs_run_or_explain_what_changed() {
    lang::set(Lang::En);
    let expected: &[(&str, Option<&str>)] = &[
        ("aes-decryption-16.qoa", Some("qubit 16 does not exist")),
        ("aes-decryption-22.qoa", Some("qubit 24 does not exist")),
        ("aes-decryption.qoa", None),
        ("fusion-sim-0.2.6.qoa", Some("QINIT appears more than once")),
        ("fusion-sim.qoa", Some("QINIT appears more than once")),
        ("grover.qoa", Some("QINIT appears more than once")),
        ("hello_world.qoa", None),
        ("light.qoa", Some("SETPOS was removed in QOA 0.4")),
        ("loop.qoa", Some("QINIT declared no qubits")),
        ("rsa-decryption.qoa", None),
        ("shor.qoa", None),
        (
            "teleport-25q-IonQ-compatible.qoa",
            Some("QUBITS was removed in QOA 0.4"),
        ),
        ("teleport.qoa", Some("QINIT appears more than once")),
        ("test.qoa", Some("QUBITS was removed in QOA 0.4")),
    ];
    let files = files_in("tests/programs/legacy");
    assert_eq!(
        files.len(),
        expected.len(),
        "a legacy program was added or removed; update this table"
    );
    for (name, want) in expected {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/programs/legacy")
            .join(name);
        let src = std::fs::read_to_string(&path).unwrap();
        let result = qoa::asm::assemble(&src, name);
        match (result, want) {
            (Ok(p), None) => {
                let cfg = RunConfig {
                    shots: 16,
                    seed: 1,
                    ..RunConfig::default()
                };
                vm::run(&p, &cfg, &mut std::io::sink())
                    .unwrap_or_else(|e| panic!("{} failed at runtime: {}", name, e));
            }
            (Err(d), Some(w)) => {
                let text = qoa::asm::render(&d, &src, name);
                assert!(
                    text.contains(w),
                    "{}: {:?} espero zen hemen:\n{}",
                    name,
                    w,
                    text
                );
                // erroreek kausa erakutsi behar dute, ez kaskada bat
                assert!(
                    d.len() <= 31,
                    "{}: {} errore, kaskada bat da",
                    name,
                    d.len()
                );
            }
            (Ok(_), Some(w)) => panic!(
                "{} mihiztatu da, baina {:?}-rekin huts egin behar zuen",
                name, w
            ),
            (Err(d), None) => panic!(
                "{} should assemble:\n{}",
                name,
                qoa::asm::render(&d, &src, name)
            ),
        }
    }
}
