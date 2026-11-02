// fitxategi honek QOA arrunteko iturburu bat glifo-sintaxira bihurtzen du lerroz lerro, iruzkinak, etiketak eta
// lerro hutsak gordeta (`qoa export --to glyph --keep-source`). lerro bakoitza adierazpen bat bihurtzen da forma
// laburrenean (`Hq0`, `#2<=#1+3`, `!c0>salto`, `PF"x"/#1`); glifo-formak adierazten ez duen lerro bat bere horretan
// geratzen da ` artean. emaitza berriz mihiztatzen da eta jatorrizkoaren programa bera izan behar du (eragiketak,
// lerroak, qubit, modu eta bitak): ez bada, desberdin dagoen lerroa ` artean jartzen da, eta berriz.

use crate::isa;
use crate::program::Program;
use crate::t;
use std::collections::HashMap;

// etiketa bat glifoan: puntuak `_` bihurtuta (glifo-izenek letrak, digituak eta _ dituzte)
fn label_name(l: &str) -> String {
    l.replace('.', "_")
}

fn is_reg(t: &str) -> bool {
    t.len() > 1 && t.starts_with(['R', 'r']) && t[1..].bytes().all(|b| b.is_ascii_digit())
}

fn plain_reg(t: &str) -> bool {
    t.len() > 1
        && t.starts_with(['q', 'Q', 'c', 'C', 'o', 'O'])
        && t[1..].bytes().all(|b| b.is_ascii_digit())
}

fn simple_number(t: &str) -> bool {
    t.parse::<f64>().is_ok_and(|v| v.is_finite()) && !t.contains(['i', 'I', 'n', 'N'])
}

fn is_name(t: &str) -> bool {
    t.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

// eragigai bat glifoan: erregistroak #n, (n) eta adierazpenak parentesi artean, gainerakoak bere horretan
fn operand(t: &str, labels: &HashMap<String, String>) -> Option<String> {
    if is_reg(t) {
        return Some(format!("#{}", &t[1..]));
    }
    if let Some(l) = labels.get(&t.to_ascii_lowercase()) {
        return Some(l.clone());
    }
    if plain_reg(t) || simple_number(t) || is_name(t) {
        return Some(t.to_string());
    }
    if t.starts_with('(') && t.ends_with(')') {
        return Some(t.to_string());
    }
    // adierazpen bat parentesi artean (glifoaren parentesiek ; eta " ez dituzte onartzen)
    if t.starts_with('"') || t.contains(['`', ';', '"', '\n']) {
        return None;
    }
    Some(format!("({})", t))
}

// esleipen baten eragigaia: erregistroak #n, zenbakiak bere horretan, adierazpenak parentesi artean
fn value(t: &str) -> Option<String> {
    if is_reg(t) {
        return Some(format!("#{}", &t[1..]));
    }
    let inner = t
        .strip_prefix('(')
        .and_then(|x| x.strip_suffix(')'))
        .unwrap_or(t);
    if simple_number(inner) && !inner.contains(['e', 'E', '+']) {
        return Some(inner.to_string());
    }
    if inner.contains(['`', ';', '"', '\n']) {
        return None;
    }
    Some(format!("({})", inner))
}

// lerro baten kodea (iruzkinik gabe) glifo-adierazpen bat
fn statement(toks: &[String], labels: &HashMap<String, String>) -> Option<String> {
    let (lead, toks) = match toks.first() {
        Some(t) if t.ends_with(':') => {
            let l = label_name(t.trim_end_matches(':'));
            (format!("{}:", l), &toks[1..])
        }
        _ => (String::new(), toks),
    };
    let Some(head) = toks.first() else {
        return Some(lead);
    };
    let spec = isa::lookup(head)?;
    if spec.is_removed() {
        return None;
    }
    let args = &toks[1..];
    let label = |t: &String| labels.get(&t.to_ascii_lowercase()).cloned();
    let body = match (spec.name, args) {
        ("JUMP", [l]) => format!(">{}", label(l)?),
        ("CALL", [l]) => format!("&{}", label(l)?),
        ("JUMPIFZERO", [v, l]) => format!("!{}>{}", operand(v, labels)?, label(l)?),
        ("JUMPIFONE", [v, l]) => format!("?{}>{}", operand(v, labels)?, label(l)?),
        ("RETSUB", []) => "^".into(),
        ("LOOPSTART", [n]) if n.bytes().all(|b| b.is_ascii_digit()) || is_reg(n) => {
            format!("*{}{{", operand(n, labels)?)
        }
        ("LOOPEND", []) => "}".into(),
        ("DUMPSTATE", []) => "<<>>".into(),
        ("DUMPREGS", []) => "<<#>>".into(),
        ("REGSET", [d, v]) if is_reg(d) => format!("#{}<={}", &d[1..], value(v)?),
        ("REGCOPY", [d, s]) if is_reg(d) && is_reg(s) => format!("#{}<=#{}", &d[1..], &s[1..]),
        ("LOAD", [d, a]) if is_reg(d) => {
            format!("#{}<=LXP@({})", &d[1..], value(a)?.trim_matches(['(', ')']))
        }
        ("STORE", [s, a]) => format!(
            "LXP@({})<={}",
            value(a)?.trim_matches(['(', ')']),
            value(s)?
        ),
        (name, [d, a, b]) if is_reg(d) && (is_reg(a) || is_reg(b)) => {
            let sym = match name {
                "ADD" => "+",
                "SUB" => "-",
                "MUL" => "*",
                "DIV" => "/",
                "POW" => "^",
                "ANDBITS" => "&",
                "ORBITS" => "|",
                "SHL" => "<<",
                "SHR" => ">>",
                _ => return generic(spec, args, labels).map(|s| format!("{}{}", lead, s)),
            };
            if a.eq_ignore_ascii_case(d) && name == "ADD" {
                format!("#{}_+{}", &d[1..], value(b)?)
            } else if a.eq_ignore_ascii_case(d) && name == "SUB" {
                format!("#{}-=>{}", &d[1..], value(b)?)
            } else {
                format!("#{}<={}{}{}", &d[1..], value(a)?, sym, value(b)?)
            }
        }
        _ => generic(spec, args, labels)?,
    };
    Some(format!("{}{}", lead, body))
}

// agindu bat bere izen laburrarekin: testuak, eragigaiak (lehena itsatsita ahal denean) eta helburua `.`-rekin
fn generic(spec: &isa::Spec, args: &[String], labels: &HashMap<String, String>) -> Option<String> {
    let name = isa::short(spec);
    let (ops, dest) = match args.iter().position(|t| t == "->") {
        Some(k) if k + 2 == args.len() => (&args[..k], Some(&args[k + 1])),
        Some(_) => return None,
        None => (args, None),
    };
    let lead_texts = ops.iter().take_while(|t| t.starts_with('"')).count();
    let mut s = String::from(name);
    for t in &ops[..lead_texts] {
        s.push_str(t);
    }
    let rest = &ops[lead_texts..];
    let trailing = rest.iter().position(|t| t.starts_with('"'));
    let (vals, after) = match trailing {
        Some(k) if rest[k..].iter().all(|t| t.starts_with('"')) && lead_texts == 0 => {
            (&rest[..k], &rest[k..])
        }
        Some(_) => return None,
        None => (rest, &rest[rest.len()..]),
    };
    if !vals.is_empty() {
        let items: Vec<String> = vals
            .iter()
            .map(|t| operand(t, labels))
            .collect::<Option<_>>()?;
        let first = &items[0];
        let glued = lead_texts == 0 && plain_reg(first) && crate::glyph::glues(name, first, &[]);
        if first.starts_with('#') && lead_texts == 0 || glued {
            s.push_str(&items.join(","));
        } else {
            s.push('/');
            s.push_str(&items.join(","));
        }
    }
    for t in after {
        s.push_str(t);
    }
    if let Some(d) = dest {
        s.push('.');
        s.push_str(&operand(d, labels)?);
    }
    Some(s)
}

// QOA arrunteko iturburu oso bat glifora, lerroz lerro. `verbatim`: ` artean utzi beharreko lerroak (0tik)
fn convert(source: &str, verbatim: &[bool], labels: &HashMap<String, String>) -> Option<String> {
    let mut out = String::new();
    for (n, line) in source.lines().enumerate() {
        let (toks, code_end) = crate::asm::line_tokens(line)?;
        let code = &line[..code_end];
        let comment = &line[code_end..];
        let indent: String = line
            .chars()
            .take_while(|c| *c == ' ' || *c == '\t')
            .collect();
        let comment_glyph = if comment.is_empty() {
            String::new()
        } else {
            // `;` edo `#` iruzkinak: lerroaren hasieran `;`, kodearen ondoren `//`
            let text = comment
                .strip_prefix("//")
                .or_else(|| comment.strip_prefix(';'))
                .or_else(|| comment.strip_prefix('#'))
                .unwrap_or(comment);
            if toks.is_empty() {
                format!(";{}", text)
            } else {
                format!("// {}", text.trim_start())
            }
        };
        if toks.is_empty() {
            out.push_str(&indent);
            out.push_str(&comment_glyph);
            out.push('\n');
            continue;
        }
        let stmt = if verbatim[n] {
            None
        } else {
            statement(&toks, labels)
        };
        let stmt = match stmt {
            Some(s) => s,
            None => {
                // bere horretan, etiketa-izenak glifoarenak bezala
                let toks2: Vec<String> = toks
                    .iter()
                    .map(|t| match t.strip_suffix(':') {
                        Some(l) if labels.contains_key(&l.to_ascii_lowercase()) => {
                            format!("{}:", label_name(l))
                        }
                        _ => labels
                            .get(&t.to_ascii_lowercase())
                            .cloned()
                            .unwrap_or_else(|| t.clone()),
                    })
                    .collect();
                let plain = toks2.join(" ");
                if plain.contains('`') {
                    return None;
                }
                let _ = code;
                format!("`{}`", plain)
            }
        };
        out.push_str(&indent);
        out.push_str(&stmt);
        if !stmt.ends_with(['{', '}', ':']) {
            out.push(';');
        }
        if !comment_glyph.is_empty() {
            out.push(' ');
            out.push_str(&comment_glyph);
        }
        out.push('\n');
    }
    Some(out)
}

// bi programa berdinak diren (izena izan ezik); ez badira, lehen desberdintasunaren iturburu-lerroa
fn first_difference(a: &Program, b: &Program) -> Result<(), Option<u32>> {
    let header =
        (a.n_qubits, a.n_modes, a.n_bits, a.cutoff) == (b.n_qubits, b.n_modes, b.n_bits, b.cutoff);
    for k in 0..a.ops.len().max(b.ops.len()) {
        if a.ops.get(k) != b.ops.get(k) || a.lines.get(k) != b.lines.get(k) {
            return Err(a.lines.get(k).or(b.lines.get(k)).copied());
        }
    }
    if header { Ok(()) } else { Err(None) }
}

// QOA arruntetik glifora, iturburua gordeta. emaitza jatorrizkoaren programa bera dela egiaztatzen du
pub fn qoa_to_glyph(source: &str, name: &str) -> Result<String, String> {
    // zigilua iturburuari dagokio: glifo-fitxategia bere aldetik zigila daiteke (`qoa seal`)
    let source = match crate::seal::find(source.as_bytes(), crate::seal::Style::Plain) {
        Some(f) => &source[..f.start],
        None => source,
    };
    let original = crate::asm::assemble(source, name).map_err(|d| {
        t!(
            "{} does not assemble: {}",
            "{} ez da mihiztatzen: {}",
            name,
            d.first().map_or(String::new(), |d| d.message.clone())
        )
    })?;
    // etiketak (minuskulaz, mihiztatzaileak bezala) eta haien izen glifoan; bi etiketa izen berera bihurtzen
    // badira, ezin da
    let mut labels = HashMap::new();
    let mut seen = HashMap::new();
    for line in source.lines() {
        if let Some((toks, _)) = crate::asm::line_tokens(line)
            && let Some(l) = toks.first().and_then(|t| t.strip_suffix(':'))
        {
            let g = label_name(l);
            if let Some(prev) = seen.insert(g.to_ascii_lowercase(), l.to_string())
                && !prev.eq_ignore_ascii_case(l)
            {
                return Err(t!(
                    "the labels {} and {} would have the same name in glyphs",
                    "{} eta {} etiketek izen bera izango lukete glifoetan",
                    prev,
                    l
                ));
            }
            labels.insert(l.to_ascii_lowercase(), g);
        }
    }
    let n = source.lines().count();
    let mut verbatim = vec![false; n];
    let fail = || {
        t!(
            "{} could not be written as glyphs; this is a QOA bug",
            "{} ezin izan da glifo gisa idatzi; QOA-ren akats bat da",
            name
        )
    };
    for _ in 0..=n {
        let text = convert(source, &verbatim, &labels).ok_or_else(fail)?;
        let back = crate::glyph::assemble(&text, name);
        let bad_line = match back {
            Ok(p) => match first_difference(&original, &p) {
                Ok(()) => return Ok(text),
                Err(line) => line,
            },
            Err(d) => d.first().map(|d| d.line),
        };
        match bad_line.and_then(|l| l.checked_sub(1)).map(|l| l as usize) {
            Some(l) if l < n && !verbatim[l] => verbatim[l] = true,
            _ => return Err(fail()),
        }
    }
    Err(fail())
}
