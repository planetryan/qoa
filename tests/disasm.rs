// fitxategi honek src/disasm.rs egiaztatzen du: programa guztiak (adibideak eta tests/programs) QOA kanoniko,
// zerrenda eta glifo gisa idatzita (glifoaren aukera guztiekin) berriz mihiztatzen direla programa berera,
// .qx bitar batetik ere; json-a irakurgarria dela eta eragiketa guztiak dituela; eta dot grafoaren ertz
// guztiek existitzen diren blokeetara jotzen dutela.

use qoa::disasm::{self, GlyphOptions};
use qoa::program::Program;
use std::path::Path;

// iturburu-fitxategi bat, CLI-k bezala (.qoa, .qg, .qasm); mihiztatzen ez direnak (nahita) saltatzen dira
fn load(path: &Path) -> Option<Program> {
    let src = std::fs::read_to_string(path).ok()?;
    let name = path.display().to_string();
    match path.extension()?.to_str()? {
        "qoa" => qoa::asm::assemble(&src, &name).ok(),
        "qg" => qoa::glyph::assemble(&src, &name).ok(),
        "qasm" => {
            let text = qoa::qasm::to_qoa(&src, &name).ok()?;
            qoa::asm::assemble(&text, &name).ok()
        }
        _ => None,
    }
}

fn programs() -> Vec<Program> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut out = Vec::new();
    for dir in [
        "examples",
        "tests/programs/sanity",
        "tests/programs/stress",
        "tests/programs/limits",
        "tests/programs/legacy",
    ] {
        let mut files: Vec<_> = std::fs::read_dir(root.join(dir))
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .collect();
        files.sort();
        out.extend(files.iter().filter_map(|f| load(f)));
    }
    assert!(out.len() >= 50, "{} programa bakarrik", out.len());
    out
}

fn reassemble(p: &Program, text: &str) -> Program {
    qoa::asm::assemble(text, &p.name).unwrap_or_else(|d| panic!("{}: {:?}\n{}", p.name, d, text))
}

#[test]
fn canonical_qoa_and_listings_reassemble_to_the_same_program() {
    for p in programs() {
        for text in [disasm::to_qoa(&p), disasm::listing(&p)] {
            let text = text.unwrap_or_else(|e| panic!("{}: {}", p.name, e));
            let back = reassemble(&p, &text);
            assert_eq!(back.ops, p.ops, "{}", p.name);
            assert_eq!(
                (back.n_qubits, back.n_modes, back.n_bits),
                (p.n_qubits, p.n_modes, p.n_bits)
            );
        }
        // .qx bitar batetik (iturbururik gabe) ere
        let bin = Program::from_bytes(&p.to_bytes()).unwrap();
        assert_eq!(disasm::to_qoa(&bin), disasm::to_qoa(&p));
    }
}

#[test]
fn glyph_exports_reassemble_with_every_option() {
    let variants = [
        GlyphOptions::default(),
        GlyphOptions {
            width: 0,
            ..GlyphOptions::default()
        },
        GlyphOptions {
            ranges: false,
            hex: true,
            ..GlyphOptions::default()
        },
        GlyphOptions {
            long_names: true,
            width: 40,
            qubits: "data".into(),
            bits: "flags".into(),
            modes: "arm".into(),
            ..GlyphOptions::default()
        },
    ];
    for p in programs() {
        for o in &variants {
            let text = disasm::to_glyph(&p, o).unwrap_or_else(|e| panic!("{}: {}", p.name, e));
            let back = qoa::glyph::assemble(&text, &p.name)
                .unwrap_or_else(|d| panic!("{}: {:?}\n{}", p.name, d, text));
            assert_eq!(back.ops, p.ops, "{}\n{}", p.name, text);
            if o.width > 0 {
                let widest = text.lines().map(|l| l.chars().count()).max().unwrap_or(0);
                let one_long = text.lines().any(|l| !l.contains("; ") && l.len() > o.width);
                assert!(
                    widest <= o.width || one_long,
                    "{}: {} > {}",
                    p.name,
                    widest,
                    o.width
                );
            }
        }
    }
}

#[test]
fn glyph_export_collapses_runs_into_ranges() {
    let p = qoa::asm::assemble(
        "QINIT 4\nH q0..q3\nMEASURE q0 -> c0\nMEASURE q1 -> c1\nCNOT q0 q2\nCNOT q1 q3\nH q0\nH q0\n",
        "t.qoa",
    )
    .unwrap();
    let text = disasm::to_glyph(&p, &GlyphOptions::default()).unwrap();
    assert!(text.contains("H/Q[0..3];"), "{}", text);
    assert!(text.contains("M/Q[0..1]$>#C[0..1];"), "{}", text);
    assert!(text.contains("CX/Q[0..1],Q[2..3];"), "{}", text);
    // bi H berdin ez dira tarte bat
    assert!(text.contains("H/Q[0]; H/Q[0];"), "{}", text);
    let hex = GlyphOptions {
        hex: true,
        ..GlyphOptions::default()
    };
    assert!(
        disasm::to_glyph(&p, &hex)
            .unwrap()
            .contains("H/Q[0x0..0x3];")
    );
    let bad = GlyphOptions {
        qubits: "X".into(),
        ..GlyphOptions::default()
    };
    assert!(disasm::to_glyph(&p, &bad).is_err());
}

#[test]
fn json_lists_every_operation() {
    for p in programs() {
        let v: serde_json::Value = serde_json::from_str(&disasm::to_json(&p).unwrap()).unwrap();
        let ops = v["ops"].as_array().unwrap();
        assert_eq!(ops.len(), p.ops.len(), "{}", p.name);
        for (a, op) in ops.iter().enumerate() {
            assert_eq!(op["address"], a);
            assert!(op["text"].as_str().is_some_and(|t| !t.is_empty()));
        }
    }
}

#[test]
fn dot_edges_point_at_existing_blocks() {
    for p in programs() {
        let dot = disasm::to_dot(&p).unwrap();
        assert!(dot.starts_with("digraph") && dot.trim_end().ends_with('}'));
        // nodoa: `bN [label=...]`; ertza: `bN -> bM [...]` (etiketen barruan ere ager daiteke ->)
        let words = |l: &str| {
            l.split_whitespace()
                .take(3)
                .map(String::from)
                .collect::<Vec<_>>()
        };
        let nodes: Vec<String> = dot
            .lines()
            .map(words)
            .filter(|w| w.len() > 1 && w[1].starts_with('['))
            .map(|w| w[0].clone())
            .collect();
        assert_eq!(
            !p.ops.is_empty(),
            nodes.contains(&"b0".to_string()),
            "{}",
            p.name
        );
        for w in dot
            .lines()
            .map(words)
            .filter(|w| w.len() > 2 && w[1] == "->")
        {
            assert!(
                nodes.contains(&w[2]),
                "{}: {} ez dago\n{}",
                p.name,
                w[2],
                dot
            );
        }
    }
}
