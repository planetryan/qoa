// fitxategi honek QOA-ren sarrera guztiak datu gaiztoekin probatzen ditu: .qx fitxategi aldatuak (bitak,
// byteak, luzerak, moztuak), eta .qoa, .qg eta .qasm iturburu nahasiak. arau bakarra: QOA-k ez du inoiz
// huts egiten panic batekin; errore-mezu bat ematen du edo programa exekutatzen du. exekuzioak mugatuak
// dira (urrats, shot eta memoria gutxi), proba azkarra izan dadin.

use qoa::program::Program;
use qoa::vm::{self, RunConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

fn sources() -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut dirs = vec![Path::new("examples").to_path_buf()];
    while let Some(d) = dirs.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                dirs.push(p);
            } else if let Ok(text) = std::fs::read_to_string(&p) {
                out.push((p.display().to_string(), text));
            }
        }
    }
    let mut dirs = vec![Path::new("tests/programs").to_path_buf()];
    while let Some(d) = dirs.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                dirs.push(p);
            } else if let Ok(text) = std::fs::read_to_string(&p) {
                out.push((p.display().to_string(), text));
            }
        }
    }
    out.sort();
    out
}

fn ext(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or("")
}

// iturburu bat bere motaren arabera mihiztatu (errorea ere ondo dago)
fn assemble(name: &str, text: &str) -> Option<Program> {
    match ext(name) {
        "qoa" => qoa::asm::assemble(text, name).ok(),
        "qg" => qoa::glyph::assemble(text, name).ok(),
        "qasm" => qoa::qasm::to_qoa(text, name)
            .ok()
            .and_then(|q| qoa::asm::assemble(&q, name).ok()),
        _ => None,
    }
}

fn cfg() -> RunConfig {
    RunConfig {
        shots: 2,
        max_steps: 2_000,
        // INPUT eta GETC-k hau irakurtzen dute, ez terminala
        input: Some(std::sync::Arc::new(b"3\n-1.5\nabc\n\xff".to_vec())),
        args: vec!["a".into(), "\u{e9}".into()],
        ..RunConfig::default()
    }
}

// programa batekin egin daitekeen guztia: exekutatu eta esportatu
fn exercise(p: &Program) {
    let _ = vm::run(p, &cfg(), &mut std::io::sink());
    let _ = qoa::disasm::to_qoa(p);
    let _ = qoa::disasm::listing(p);
    let _ = qoa::disasm::to_json(p);
    let _ = qoa::disasm::to_dot(p);
    let _ = qoa::disasm::to_glyph(p, &Default::default());
    let _ = qoa::qasm::from_program(p);
    let _ = qoa::ionq::from_program(p, 2);
    let _ = p.to_bytes();
}

fn mutate(rng: &mut StdRng, bytes: &[u8]) -> Vec<u8> {
    let mut b = bytes.to_vec();
    for _ in 0..rng.random_range(1..6) {
        if b.is_empty() {
            break;
        }
        let i = rng.random_range(0..b.len());
        match rng.random_range(0..9) {
            0 => b[i] ^= 1 << rng.random_range(0..8),
            1 => b[i] = 0,
            2 => b[i] = 0xff,
            3 => b[i] = rng.random(),
            // luzera edo indize bat (u32 edo u64) handi bihurtu
            4 => {
                let end = (i + 4).min(b.len());
                b[i..end].fill(0xff);
            }
            5 => {
                let end = (i + 8).min(b.len());
                b[i..end]
                    .copy_from_slice(&[0x00, 0x00, 0x00, 0x80, 0xff, 0xff, 0xff, 0x7f][..end - i]);
            }
            6 => b.truncate(i),
            7 => {
                let n = rng.random_range(1..16);
                let junk: Vec<u8> = (0..n).map(|_| rng.random()).collect();
                b.splice(i..i, junk);
            }
            // f64 bat NaN, infinitu edo erraldoi bihurtu
            _ => {
                let v: f64 =
                    [f64::NAN, f64::INFINITY, -1e300, 4.5e15, -0.0][rng.random_range(0..5)];
                for (k, x) in v.to_le_bytes().iter().enumerate() {
                    if i + k < b.len() {
                        b[i + k] = *x;
                    }
                }
            }
        }
    }
    b
}

fn mangle(rng: &mut StdRng, text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    let junk = [
        '(', ')', '[', ']', '{', '}', ';', '"', '\\', '#', '$', '>', '<', '%', '/', '.', ',', '-',
        '9', '0', 'q', 'R', 'x', 'π', '\u{3000}', '\n', '\0', '\u{feff}', 'é', '*', '^', '?', '!',
    ];
    for _ in 0..rng.random_range(1..8) {
        if chars.is_empty() {
            chars.push('x');
        }
        let i = rng.random_range(0..chars.len());
        match rng.random_range(0..5) {
            0 => chars[i] = junk[rng.random_range(0..junk.len())],
            1 => {
                chars.remove(i);
            }
            2 => chars.insert(i, junk[rng.random_range(0..junk.len())]),
            3 => {
                // zenbaki erraldoi bat
                let big = [
                    "99999999999999999999",
                    "4294967296",
                    "1e308",
                    "-1",
                    "0x7fffffff",
                ][rng.random_range(0..5)];
                for (k, c) in big.chars().enumerate() {
                    chars.insert(i + k, c);
                }
            }
            _ => {
                // lerro bat bikoiztu
                let line: String = chars[i..].iter().take_while(|&&c| c != '\n').collect();
                for (k, c) in line.chars().enumerate() {
                    chars.insert(i + k, c);
                }
            }
        }
    }
    chars.into_iter().collect()
}

// QOA_HOSTILE_LOG=fitxategia: uneko kasuaren izena han, proba bat gelditzen bada zein den jakiteko
static WHERE: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

fn check(what: &str, f: impl FnOnce()) -> Option<String> {
    if let Ok(log) = std::env::var("QOA_HOSTILE_LOG") {
        let _ = std::fs::write(log, what);
    }
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(()) => None,
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            let at = WHERE.lock().unwrap_or_else(|e| e.into_inner()).clone();
            Some(format!("{}: {} (at {})", what, msg, at))
        }
    }
}

#[test]
fn hostile_inputs_give_errors_not_crashes() {
    // memoria gutxi: programa aldatu batek ezin du makina bete
    // segurtasuna: proba-fitxategi honek proba bakarra du, beraz beste harik ez du ingurunea irakurtzen
    unsafe { std::env::set_var("QOA_MAX_MEMORY", "16M") };
    qoa::lang::set(qoa::lang::Lang::En);
    // panic-aren lekua gorde (fitxategia:lerroa), mezuan erakusteko
    std::panic::set_hook(Box::new(|info| {
        if let Some(l) = info.location() {
            *WHERE.lock().unwrap_or_else(|e| e.into_inner()) = format!("{}:{}", l.file(), l.line());
        }
    }));
    // QOA_HOSTILE_SEED=n: beste hazi batekin (beste kasu batzuk), eskuz bilatzeko
    let seed = std::env::var("QOA_HOSTILE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xbad_c0de);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut problems = Vec::new();
    let srcs = sources();
    let mut binaries = Vec::new();
    for (name, text) in &srcs {
        if let Some(p) = problems_or(&mut problems, name, || assemble(name, text)) {
            binaries.push((name.clone(), p.to_bytes()));
        }
    }
    assert!(
        binaries.len() > 100,
        "programa gutxiegi: {}",
        binaries.len()
    );
    // .qx aldatuak
    let mut loaded = 0;
    for round in 0..20000 {
        let (name, bytes) = &binaries[round % binaries.len()];
        let bad = mutate(&mut rng, bytes);
        let what = format!("{} (aldaketa {})", name, round);
        // QOA_HOSTILE_DUMP="n fitxategia": n. aldaketa (edo "sn": n. nahasketa) fitxategi horretan idatzi,
        // eskuz aztertzeko
        if let Ok(dump) = std::env::var("QOA_HOSTILE_DUMP")
            && let Some((n, file)) = dump.split_once(' ')
            && n == round.to_string()
        {
            std::fs::write(file, &bad).unwrap();
        }
        if let Some(e) = check(&what, || {
            if let Ok(p) = Program::from_bytes(&bad) {
                loaded += 1;
                exercise(&p);
            }
        }) {
            problems.push(e);
        }
    }
    // iturburu nahasiak, mota bakoitzeko mihiztatzailearekin
    for round in 0..10000 {
        let (name, text) = &srcs[round % srcs.len()];
        let bad = mangle(&mut rng, text);
        let what = format!("{} (nahasketa {})", name, round);
        if let Ok(dump) = std::env::var("QOA_HOSTILE_DUMP")
            && let Some((n, file)) = dump.split_once(' ')
            && n == format!("s{}", round)
        {
            std::fs::write(file, &bad).unwrap();
        }
        if let Some(e) = check(&what, || {
            if let Some(p) = assemble(name, &bad) {
                exercise(&p);
            }
            let _ = qoa::glyph::expand(&bad);
            let _ = qoa::expr::eval(&bad);
        }) {
            problems.push(e);
        }
    }
    // habiaraketa erraldoiak: pila gainezkatzeak ezin dira harrapatu (prozesua amaitzen dute), beraz
    // hemen iristeak berak frogatzen du ez dagoela halakorik
    let n = 200_000;
    let deep = [
        (
            "deep.qoa",
            format!("QINIT 1\nRX q0 {}1{}\n", "(".repeat(n), ")".repeat(n)),
        ),
        ("deep.qoa", format!("QINIT 1\nRX q0 {}1\n", "-".repeat(n))),
        ("deep.qoa", format!("QINIT 1\nRX q0 {}1\n", "2^".repeat(n))),
        (
            "deep.qoa",
            format!("QINIT 1\nRX q0 {}1{}\n", "sqrt(".repeat(n), ")".repeat(n)),
        ),
        (
            "deep.qoa",
            format!(
                "QINIT 1\n{}H q0\n{}",
                "LOOPSTART 2\n".repeat(n),
                "LOOPEND\n".repeat(n)
            ),
        ),
        (
            "deep.qg",
            format!("Q{{1}}; {}H/Q;{}\n", "*2{ ".repeat(n), " }".repeat(n)),
        ),
        (
            "deep.qg",
            format!("Q{{1}}; RX({}1{})/Q;\n", "(".repeat(n), ")".repeat(n)),
        ),
        (
            "deep.qasm",
            format!(
                "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nrx({}1{}) q[0];\n",
                "(".repeat(n),
                ")".repeat(n)
            ),
        ),
    ];
    for (name, text) in &deep {
        if let Some(e) = check(name, || {
            if let Some(p) = assemble(name, text) {
                let _ = qoa::disasm::to_qoa(&p);
            }
        }) {
            problems.push(e);
        }
    }
    let _ = std::panic::take_hook();
    problems.dedup();
    assert!(
        problems.is_empty(),
        "{} panic ({} .qx aldatu kargatuta):\n{}",
        problems.len(),
        loaded,
        problems
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn problems_or<T>(
    problems: &mut Vec<String>,
    what: &str,
    f: impl FnOnce() -> Option<T>,
) -> Option<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            problems.push(format!("{}: panic mihiztatzean", what));
            None
        }
    }
}
