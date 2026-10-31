// fitxategi honek tools/asm.qoa probatzen du: QOAn idatzitako QOA-ren mihiztatzailea. egiaztatzen du haren
// taulak src/isa.rs-ekin bat datozela, programa guztiak `qoa compile`-k bezala mihiztatzen dituela (byte
// berberak), errore berberak ematen dituela (testu berberarekin, bi hizkuntzetan), eta bere burua
// mihiztatzen duela: rust-ek eraikitako mihiztatzaileak eta berak eraikitakoak byte berberak dituzte.
//
// taulak berriz sortzeko (src/isa.rs aldatu ondoren): QOA_REGENERATE=1 cargo test --test selfhost

use qoa::isa::{SPECS, Status};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

const ASM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/asm.qoa");

// isatext-en erregistroak: izena|ezizenak|eragigaiak|hedapena|kendua|arrazoia en|eu|erabilera en|eu
fn isa_block() -> String {
    let mut out = String::new();
    for s in SPECS {
        let (removed, en, eu) = match s.status {
            Status::Removed(en, eu) => ("1", en, eu),
            _ => ("0", "", ""),
        };
        let fields = [
            s.name.to_string(),
            s.aliases.join(" "),
            s.operands.to_string(),
            if s.broadcast { "1" } else { "0" }.to_string(),
            removed.to_string(),
            en.to_string(),
            eu.to_string(),
            s.usage_in(false),
            s.usage_in(true),
        ];
        for f in &fields {
            // STORETEXT-en testu soil gisa idazten dira: ez komatxorik, ez alderantzizko barrarik, ez '|'
            assert!(
                !f.contains(['"', '\\', '|']),
                "{}: {:?} ezin da taulan idatzi",
                s.name,
                f
            );
        }
        out.push_str(&format!("STORETEXT R1 \"{}\"\n", fields.join("|")));
    }
    out
}

// unicode-ko tarteak (ascii ez direnak), `char::is_alphabetic` edo `char::is_alphanumeric` bezala
fn unicode_block(f: fn(char) -> bool) -> String {
    let mut ranges = Vec::new();
    let mut start: Option<u32> = None;
    for u in 0x80..=0x11_0000u32 {
        let yes = u < 0x11_0000 && char::from_u32(u).is_some_and(f);
        match (yes, start) {
            (true, None) => start = Some(u),
            (false, Some(a)) => {
                ranges.push((a, u - 1));
                start = None;
            }
            _ => {}
        }
    }
    let mut out = String::new();
    for chunk in ranges.chunks(12) {
        let items: Vec<String> = chunk
            .iter()
            .map(|(a, b)| format!("{:x}-{:x}", a, b))
            .collect();
        out.push_str(&format!("STORETEXT R1 \"{}\"\n", items.join(" ")));
    }
    out
}

// `; @NAME-begin` eta `; @NAME-end` arteko lerroak ordezkatu
fn replace_block(src: &str, name: &str, body: &str) -> String {
    let begin = format!("; @{}-begin\n", name);
    let end = format!("; @{}-end\n", name);
    let a = src.find(&begin).expect("hasiera-marka") + begin.len();
    let b = src.find(&end).expect("amaiera-marka");
    format!("{}{}{}", &src[..a], body, &src[b..])
}

#[test]
fn tables_match_the_rust_side() {
    let src = std::fs::read_to_string(ASM).unwrap();
    let mut want = replace_block(&src, "isa", &isa_block());
    want = replace_block(&want, "alpha", &unicode_block(char::is_alphabetic));
    want = replace_block(&want, "alnum", &unicode_block(char::is_alphanumeric));
    if want != src {
        if std::env::var_os("QOA_REGENERATE").is_some() {
            std::fs::write(ASM, &want).unwrap();
        } else {
            panic!(
                "tools/asm.qoa-ren taulak ez datoz bat src/isa.rs-ekin edo rust-en unicode-arekin; \
                 sortu berriz: QOA_REGENERATE=1 cargo test --test selfhost"
            );
        }
    }
}

fn qoa(lang: &str, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_qoa"))
        .args(args)
        .env("QOA_LANG", lang)
        .env("NO_COLOR", "1")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("qoa runs")
}

fn scratch(name: &str) -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join(name)
}

// tools/asm.qoa rust-en mihiztatzailearekin eraikita, behin
fn stage1() -> &'static Path {
    static QX: OnceLock<PathBuf> = OnceLock::new();
    QX.get_or_init(|| {
        let out = scratch("asm-stage1.qx");
        let o = qoa("en", &["compile", ASM, "-o", out.to_str().unwrap()]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        out
    })
}

// fitxategi bat bi mihiztatzaileekin: irteera-kodea, stdout, stderr eta .qx byteak berdinak
fn compare(lang: &str, file: &Path, out: &Path) -> Result<(), String> {
    let f = file.to_str().unwrap();
    let o = out.to_str().unwrap();
    let _ = std::fs::remove_file(out);
    let rust = qoa(lang, &["compile", f, "-o", o]);
    let want = std::fs::read(out).ok();
    let _ = std::fs::remove_file(out);
    let asm = stage1().to_str().unwrap();
    let mine = qoa(lang, &["compile", "--assembler", asm, f, "-o", o]);
    let got = std::fs::read(out).ok();
    let text = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
    if rust.status.code() != mine.status.code() {
        return Err(format!(
            "{} ({}): irteera-kodea {:?} / {:?}\n--- rust:\n{}\n--- qoa:\n{}",
            f,
            lang,
            rust.status.code(),
            mine.status.code(),
            text(&rust.stderr),
            text(&mine.stderr)
        ));
    }
    if rust.stderr != mine.stderr {
        return Err(format!(
            "{} ({}): mezuak desberdinak\n--- rust:\n{}\n--- qoa:\n{}",
            f,
            lang,
            text(&rust.stderr),
            text(&mine.stderr)
        ));
    }
    if rust.stdout != mine.stdout || want != got {
        return Err(format!("{} ({}): .qx desberdina", f, lang));
    }
    Ok(())
}

fn programs() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for dir in [
        "examples",
        "tests/programs/sanity",
        "tests/programs/failures",
        "tests/programs/stress",
        "tests/programs/limits",
        "tests/programs/legacy",
    ] {
        for e in std::fs::read_dir(root.join(dir)).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_some_and(|x| x == "qoa") {
                files.push(Path::new(dir).join(p.file_name().unwrap()));
            }
        }
    }
    files.sort();
    files.push(PathBuf::from("tools/asm.qoa"));
    files
}

#[test]
fn every_program_assembles_to_the_same_bytes_and_errors() {
    let out = scratch("every.qx");
    let mut bad = Vec::new();
    for f in programs() {
        for lang in ["en", "eu"] {
            if let Err(e) = compare(lang, &f, &out) {
                bad.push(e);
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n\n"));
}

#[test]
fn qoa_builds_qoa_a_fixed_point() {
    // 1. rust-ek eraiki tools/asm.qoa; 2. horrek bere burua; 3. bigarrenak berriz: hiruak berdinak
    let one = std::fs::read(stage1()).unwrap();
    let two_path = scratch("asm-stage2.qx");
    let o = qoa(
        "en",
        &[
            "compile",
            "--assembler",
            stage1().to_str().unwrap(),
            ASM,
            "-o",
            two_path.to_str().unwrap(),
        ],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let two = std::fs::read(&two_path).unwrap();
    assert!(
        one == two,
        "QOAk eraikitako mihiztatzailea ez da rust-ek eraikitakoa"
    );
    let three_path = scratch("asm-stage3.qx");
    let o = qoa(
        "en",
        &[
            "compile",
            "--assembler",
            two_path.to_str().unwrap(),
            ASM,
            "-o",
            three_path.to_str().unwrap(),
        ],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(two == std::fs::read(&three_path).unwrap());
}

// ---- sortutako programak: agindu guztiak, eragigai onekin eta txarrekin ----

const EXPRESSIONS: &[&str] = &[
    "0.5",
    "pi/2",
    "-3*pi/4",
    "(1+2)/sqrt(2)",
    "2^3^2",
    "1e-3",
    "1.5E+4",
    ".5",
    "5.",
    "π",
    "tau",
    "e",
    "-0",
    "ln(1)",
    "exp(ln(0))",
    "1/ln(0)",
    "sqrt(-1)^0",
    "sin(1e22)",
    "cos(-0)",
    "tan(1)",
    "+-+1",
    "2*-3",
    "-(0)",
    "0.1000000000000000055511151231257827021181583404541015625",
    "0.1000000000000000055511151231257827021181583404541015624",
    "2.2250738585072011e-308",
    "2.2250738585072014e-308",
    "4.9406564584124654e-324",
    "2.4703282292062327e-324",
    "2.4703282292062328e-324",
    "1.7976931348623157e308",
    "1.7976931348623158e308",
    "1.7976931348623159e308",
    "9007199254740993",
    "9007199254740995",
    "123456789012345678901234567890",
    "0.000000000000000000000000000001234",
    "1e400",
    "1e-400",
    "0e99999",
    "00000.000001e00000000000000000000000000000000006",
    "1e00000000000000000000000000001",
    "8.98846567431158e307*2",
    "1/0",
    "1/(1-1)",
    "1..2",
    ".",
    "1.2.3",
    "2e",
    "foo",
    "sqrt 2",
    "sqrt",
    "(1",
    "1)",
    "Π",
    "π2",
    "x×y",
    "2*×",
    "é",
    "1+",
    "SQRT(4)",
    "Ln(2)",
    "PI",
    "ее",
    "_",
    "1 2",
    "(1\u{3000}+\u{a0}2)",
    "3!",
    "2^",
    "--1",
    "e1",
    "1e+",
];

fn long_number(rng: &mut StdRng) -> String {
    let digits: String = (0..rng.random_range(16..900))
        .map(|_| char::from(b'0' + rng.random_range(0..10u8)))
        .collect();
    let point = rng.random_range(0..digits.len());
    format!(
        "{}.{}e{}",
        &digits[..point],
        &digits[point..],
        rng.random_range(-700..400)
    )
}

fn operand(rng: &mut StdRng, kind: char) -> String {
    let bad = rng.random_bool(0.12);
    match kind {
        'q' => match rng.random_range(0..12) {
            0 if bad => "q99".into(),
            1 if bad => "o1".into(),
            2 if bad => "q1..q0".into(),
            3 => "q0..q2".into(),
            4 => "+2".into(),
            5 if bad => "data[7]".into(),
            6 => "data[1]".into(),
            7 => "data".into(),
            _ => format!("q{}", rng.random_range(0..4)),
        },
        'o' => match rng.random_range(0..8) {
            0 if bad => "o70".into(),
            1 if bad => "q0".into(),
            2 => "1".into(),
            3 => "o0..o1".into(),
            _ => format!("o{}", rng.random_range(0..3)),
        },
        'r' => match rng.random_range(0..8) {
            0 if bad => "R32".into(),
            1 if bad => "x".into(),
            2 => "7".into(),
            _ => format!("R{}", rng.random_range(0..32)),
        },
        'c' => match rng.random_range(0..8) {
            0 if bad => "3".into(),
            1 => "c0..c2".into(),
            2 if bad => "c5000".into(),
            3 => "bits[1]".into(),
            _ => format!("c{}", rng.random_range(0..8)),
        },
        'd' => match rng.random_range(0..6) {
            0 => format!("-> c{}", rng.random_range(0..4)),
            1 => format!("R{}", rng.random_range(0..32)),
            2 if bad => "-> q1".into(),
            3 => "-> c0..c2".into(),
            _ => format!("-> R{}", rng.random_range(0..32)),
        },
        'v' => {
            if rng.random_bool(0.5) {
                "c1".into()
            } else {
                "R4".into()
            }
        }
        'f' => {
            if rng.random_bool(0.1) {
                long_number(rng)
            } else {
                EXPRESSIONS[rng.random_range(0..EXPRESSIONS.len())].replace("\\u{3000}", "\u{3000}")
            }
        }
        'i' => match rng.random_range(0..8) {
            0 => "0x1F".into(),
            1 => "18446744073709551615".into(),
            2 if bad => "18446744073709551616".into(),
            3 if bad => "-1".into(),
            4 => "+7".into(),
            _ => rng.random_range(0..9).to_string(),
        },
        'n' => match rng.random_range(0..5) {
            0 => "R3".into(),
            1 if bad => "R40".into(),
            2 if bad => "abc".into(),
            _ => rng.random_range(0..4).to_string(),
        },
        'l' => match rng.random_range(0..6) {
            0 if bad => "missing".into(),
            1 if bad => "9x".into(),
            2 => "L.two".into(),
            _ => "l1".into(),
        },
        's' => match rng.random_range(0..6) {
            0 => "\"tab\\there {} \\\" q\\\\ \\x\"".into(),
            1 if bad => "bare".into(),
            2 => "\"ünï {} cödé\"".into(),
            _ => "\"text {}\"".into(),
        },
        'b' => ["X", "y", "Z", "W", "XY"][rng.random_range(0..if bad { 5 } else { 3 })].into(),
        'g' => ["h", "X", "not", "sdag", "Tdg", "sx", "cx", "zz"]
            [rng.random_range(0..if bad { 8 } else { 6 })]
        .into(),
        'k' => ["\"A\"", "65", "0x41", "\"é\"", "\"AB\"", "1114112"]
            [rng.random_range(0..if bad { 6 } else { 4 })]
        .into(),
        'w' => {
            ["data", "q3", "pi", "H", "x y"][rng.random_range(0..if bad { 5 } else { 1 })].into()
        }
        _ => "?".into(),
    }
}

fn instruction(rng: &mut StdRng) -> String {
    let live: Vec<_> = SPECS.iter().filter(|s| !s.is_removed()).collect();
    let spec = if rng.random_bool(0.04) {
        // kendutako bat, edo idazketa-akats bat
        let removed: Vec<_> = SPECS.iter().filter(|s| s.is_removed()).collect();
        return match rng.random_range(0..3) {
            0 => removed[rng.random_range(0..removed.len())].name.to_string(),
            1 => "MEASUR q0".into(),
            _ => "FROBNICATE q0".into(),
        };
    } else {
        live[rng.random_range(0..live.len())]
    };
    let mut name = spec.name.to_string();
    if !spec.aliases.is_empty() && rng.random_bool(0.3) {
        name = spec.aliases[rng.random_range(0..spec.aliases.len())].to_string();
    }
    if rng.random_bool(0.2) {
        name = name.to_lowercase();
    }
    let mut out = name;
    for op in spec.operand_specs() {
        if op.optional && rng.random_bool(0.4) {
            continue;
        }
        let reps = match op.repeat {
            qoa::isa::Repeat::One => 1,
            _ => rng.random_range(if op.optional { 0 } else { 1 }..4),
        };
        for _ in 0..reps {
            out.push(if rng.random_bool(0.1) { ',' } else { ' ' });
            out.push_str(&operand(rng, op.kind));
        }
    }
    if rng.random_bool(0.03) {
        out.push_str(" extra");
    }
    if rng.random_bool(0.02) {
        out.push_str(" -> ->");
    }
    out
}

fn program(rng: &mut StdRng) -> String {
    let mut lines = Vec::new();
    if rng.random_bool(0.6) {
        lines.push(format!(
            "QINIT {}",
            ["4", "2", "0", "65", "x"][rng.random_range(0..5)]
        ));
    }
    if rng.random_bool(0.5) {
        lines.push(
            ["QMODE 3", "QMODE 3 4", "QMODE 2 1", "QMODE", "QMODE 3 6 9"][rng.random_range(0..5)]
                .into(),
        );
    }
    if rng.random_bool(0.5) {
        lines.push(
            [
                "QREG data 3 1",
                "QREG data 3",
                "QREG data 9",
                "QREG pi 2",
                "QREG q3 1",
                "QREG data 0",
            ][rng.random_range(0..6)]
            .into(),
        );
        if rng.random_bool(0.5) {
            lines.push(
                ["CREG bits 3", "OREG light 2", "QREG more 2 2"][rng.random_range(0..3)].into(),
            );
        }
    }
    for _ in 0..rng.random_range(3..18) {
        let line = match rng.random_range(0..20) {
            0 => "l1:".to_string(),
            1 => format!("L.two: {}", instruction(rng)),
            2 => "; a comment # with // marks".into(),
            3 => String::new(),
            4 => "LOOPSTART 2".into(),
            5 => "LOOPEND".into(),
            6 => format!("\t{}\t; trailing", instruction(rng)),
            7 => format!("{} // more", instruction(rng)),
            8 => format!("\u{3000}{}\u{a0}", instruction(rng)),
            9 => "PRINT \"a;b#c//d\" # real comment".into(),
            10 => "l1:".into(),
            _ => instruction(rng),
        };
        lines.push(line);
    }
    let ending = ["\n", "\r\n", "\n", "\r"][rng.random_range(0..4)];
    let mut text = lines.join(ending);
    if rng.random_bool(0.5) {
        text.push_str(ending);
    }
    text
}

#[test]
fn generated_programs_assemble_the_same() {
    let mut rng = StdRng::seed_from_u64(0x51f_4057);
    let out = scratch("gen.qx");
    let mut bad = Vec::new();
    let mut cases: Vec<Vec<u8>> = (0..160).map(|_| program(&mut rng).into_bytes()).collect();
    // utf-8 baliogabea, 30 errore baino gehiago, eta lerro hutsak bakarrik
    cases.push(b"H q0\n\xff\xfe\n".to_vec());
    cases.push(b"H q0\n\xed\xa0\x80\n".to_vec());
    cases.push(
        (0..45)
            .map(|i| format!("BOGUS{} q0\n", i))
            .collect::<String>()
            .into_bytes(),
    );
    cases.push(b"\n\n\n".to_vec());
    cases.push(Vec::new());
    for (i, case) in cases.iter().enumerate() {
        let file = scratch(&format!("gen-{}.qoa", i));
        std::fs::write(&file, case).unwrap();
        let lang = if i % 3 == 0 { "eu" } else { "en" };
        if let Err(e) = compare(lang, &file, &out) {
            bad.push(e);
        }
    }
    assert!(
        bad.is_empty(),
        "{} desberdin:\n{}",
        bad.len(),
        bad.join("\n\n")
    );
}

// zenbaki hamartar bat edozein formatan: digitu asko, puntua edonon, berretzaile handiak
fn decimal(rng: &mut StdRng) -> String {
    let n = match rng.random_range(0..4) {
        0 => rng.random_range(1..8),
        1 => rng.random_range(8..20),
        2 => rng.random_range(15..40),
        _ => rng.random_range(40..120),
    };
    let mut digits: String = (0..n)
        .map(|_| char::from(b'0' + rng.random_range(0..10u8)))
        .collect();
    if rng.random_bool(0.3) {
        // erdiko kasuak: ...5000..., eta zero luzeak
        digits.push('5');
        digits.extend(std::iter::repeat_n('0', rng.random_range(0..30)));
    }
    let point = rng.random_range(0..=digits.len());
    let mut s = format!("{}.{}", &digits[..point], &digits[point..]);
    if s.starts_with('.') && rng.random_bool(0.5) {
        s.insert(0, '0');
    }
    if s == "." {
        s = "0.5".into();
    }
    if rng.random_bool(0.7) {
        let e: i32 = rng.random_range(-345..320);
        s.push_str(&format!(
            "{}{}",
            if rng.random_bool(0.5) { "e" } else { "E" },
            e
        ));
    }
    s
}

#[test]
fn generated_valid_programs_assemble_the_same() {
    let header = "QINIT 6\nQMODE 3 4\nQREG data 3 1\nCREG bits 3\nOREG light 2\nl1:\nL.two:\n";
    let mut rng = StdRng::seed_from_u64(0x0a5_0a5);
    let mut kept = Vec::new();
    // mihiztatzaile batek onartzen dituen lerroak bakarrik, gero fitxategi handietan elkarrekin
    for _ in 0..6000 {
        let line = instruction(&mut rng);
        if qoa::asm::assemble(&format!("{}{}\n", header, line), "x.qoa").is_ok() {
            kept.push(line);
        }
    }
    for _ in 0..3000 {
        let line = format!("REGSET R{} {}", rng.random_range(0..32), decimal(&mut rng));
        if qoa::asm::assemble(&format!("{}{}\n", header, line), "x.qoa").is_ok() {
            kept.push(line);
        }
    }
    for _ in 0..40 {
        kept.push(format!(
            "LOOPSTART {}\n{}\nLOOPSTART R{}\n{}\nLOOPEND\nLOOPEND",
            rng.random_range(0..5),
            "H q0",
            rng.random_range(0..32),
            "PRINTLN \"inner\""
        ));
    }
    assert!(
        kept.len() > 4000,
        "onartutako lerro gutxiegi: {}",
        kept.len()
    );
    let out = scratch("valid.qx");
    let mut bad = Vec::new();
    for (i, chunk) in kept.chunks(700).enumerate() {
        let file = scratch(&format!("valid-{}.qoa", i));
        std::fs::write(&file, format!("{}{}\n", header, chunk.join("\n"))).unwrap();
        let lang = if i % 2 == 0 { "en" } else { "eu" };
        if let Err(e) = compare(lang, &file, &out) {
            bad.push(e);
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n\n"));
}
