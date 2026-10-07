// fitxategi honek proiektuaren estiloa zaintzen du, karpeta guztietan:
//   - ez dago `///` edo `//!` dokumentazio-iruzkinik (iruzkin arruntak `//` dira)
//   - iruzkinak euskaraz daude (.rs fitxategietan `//`, .qoa fitxategietan `;`)
// ingelesa detektatzeko: iruzkin batek ingelesezko funtzio-hitz ohiko bi edo gehiago baditu, ingelesa da.

use std::path::{Path, PathBuf};

// oraindik itzuli gabeak (bakoitza bere zereginarekin); hutsik egon behar luke azkenean
const NOT_YET: &[&str] = &["src/visualizer.rs"];

const ENGLISH: &[&str] = &[
    "the", "and", "is", "are", "this", "that", "with", "for", "of", "to", "we", "it", "be", "from", "not",
    "which", "when", "should", "will", "can", "into", "only", "each", "every", "if", "or", "an", "by", "its",
];

fn files(root: &Path, exts: &[&str], out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(root).unwrap().flatten() {
        let p = entry.path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name == "target" || name.starts_with('.') || name == "legacy" {
            continue;
        }
        if p.is_dir() {
            files(&p, exts, out);
        } else if p.extension().is_some_and(|e| exts.contains(&e.to_str().unwrap())) {
            out.push(p);
        }
    }
}

fn looks_english(comment: &str) -> bool {
    // `kodea` atzera-komatxoen artean dago eta ez da kontuan hartzen
    let prose: String = comment.split('`').step_by(2).collect::<Vec<_>>().join(" ");
    let words: Vec<String> = prose
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect();
    let hits: std::collections::HashSet<&str> = ENGLISH.iter().copied().filter(|e| words.iter().any(|w| w == e)).collect();
    hits.len() >= 2
}

// lerro bateko iruzkina (testu-kateen barrukoak kontuan hartu gabe)
fn comment_of<'a>(line: &'a str, marker: &str) -> Option<&'a str> {
    let mut in_str = false;
    let bytes = line.as_bytes();
    let m = marker.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' && (i == 0 || bytes[i - 1] != b'\\') {
            in_str = !in_str;
        } else if !in_str && bytes[i..].starts_with(m) {
            return Some(&line[i + m.len()..]);
        }
        i += 1;
    }
    None
}

#[test]
fn comments_follow_the_project_style() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut all = Vec::new();
    files(root, &["rs", "qoa"], &mut all);
    let mut problems = Vec::new();
    for path in all {
        let rel = path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        if NOT_YET.contains(&rel.as_str()) {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let is_rs = rel.ends_with(".rs");
        for (n, line) in text.lines().enumerate() {
            let t = line.trim_start();
            if is_rs && (t.starts_with("///") || t.starts_with("//!")) {
                problems.push(format!("{}:{}: doc comment (use //)", rel, n + 1));
                continue;
            }
            let marker = if is_rs { "//" } else { ";" };
            if let Some(c) = comment_of(line, marker) {
                // probetako itxaropen-goiburuak eta expect lerroak ez dira iruzkin arruntak
                let c = c.trim();
                if c.starts_with("expect") || c.starts_with("shots:") || c.starts_with("seed:") || c.starts_with("max-steps:") {
                    continue;
                }
                if looks_english(c) {
                    problems.push(format!("{}:{}: English comment: {}", rel, n + 1, c));
                }
            }
        }
    }
    assert!(problems.is_empty(), "{} style problem(s):\n{}", problems.len(), problems.join("\n"));
}
