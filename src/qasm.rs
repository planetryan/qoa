// fitxategi honek openqasm inportatzen (qoa testura) eta esportatzen (openqasm 2.0-ra) du.
//
// inportazioa: openqasm 2.0 osoa: qreg/creg, qelib1.inc-ren ate guztiak (qiskit-en bertsio
// zabaldua barne), `gate` definizioak (lerroan zabaltzen dira), `if (c == n)`, measure, reset eta
// barrier. `opaque` ateek ez dute definiziorik, beraz ezin dira simulatu, eta hala esaten da.
// openqasm 3-tik oinarrizko zatia ere onartzen da (qubit[n], bit[n], c = measure q).
//
// esportazioa: qubit-zatia; modu optikoak, begiztak eta erregistroak ezin dira openqasm 2.0-n
// adierazi, eta errorea ematen dute lerroa adieraziz.

use crate::expr;
use crate::program::{Basis, Count, Dest, Gate, Op, Program, Source};
use crate::t;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub struct QasmError {
    pub line: u32,
    pub message: String,
}

impl std::fmt::Display for QasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            t!("line {}: {}", "{}. lerroa: {}", self.line, self.message)
        )
    }
}

fn qerr(line: u32, message: impl Into<String>) -> QasmError {
    QasmError {
        line,
        message: message.into(),
    }
}

// iruzkinak kendu, lerro-zenbakiak mantenduz (/* */ barruko lerro-jauziak gordetzen dira)
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut in_str = false;
    while i < chars.len() {
        let c = chars[i];
        if c == '"' {
            in_str = !in_str;
            out.push(c);
            i += 1;
        } else if !in_str && c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if !in_str && c == '/' && chars.get(i + 1) == Some(&'*') {
            i += 2;
            while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                if chars[i] == '\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i += 2;
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

// adierazpenetan banatu: goi-mailako ; batek edo bloke baten } itxierak amaitzen dute.
// adierazpen bakoitzarekin haren hasierako lerroa itzultzen da.
fn split_statements(src: &str) -> Result<Vec<(String, u32)>, QasmError> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut line = 1u32;
    let mut start = 1u32;
    let mut depth = 0i32;
    for c in src.chars() {
        if cur.trim().is_empty() && !c.is_whitespace() {
            start = line;
        }
        match c {
            '{' => {
                depth += 1;
                cur.push(c);
            }
            '}' => {
                depth -= 1;
                cur.push(c);
                if depth < 0 {
                    return Err(qerr(
                        line,
                        t!("a } without a matching {", "} bat { parekorik gabe"),
                    ));
                }
                if depth == 0 {
                    out.push((cur.trim().to_string(), start));
                    cur.clear();
                }
            }
            ';' if depth == 0 => {
                if !cur.trim().is_empty() {
                    out.push((cur.trim().to_string(), start));
                }
                cur.clear();
            }
            _ => cur.push(c),
        }
        if c == '\n' {
            line += 1;
        }
    }
    if depth > 0 {
        return Err(qerr(
            start,
            t!("a { is never closed", "{ bat ez da inoiz ixten"),
        ));
    }
    if !cur.trim().is_empty() {
        return Err(qerr(
            start,
            t!(
                "the last statement is missing its ;",
                "azken adierazpenari ; falta zaio"
            ),
        ));
    }
    Ok(out)
}

// erregistro baten kokapena qubit/bit lau zerrendan
#[derive(Clone, Copy)]
struct Reg {
    offset: u32,
    size: u32,
}

// argumentu bat: erregistro osoa edo elementu bakarra
enum Target {
    One(u32),
    All(u32, u32),
}

// erabiltzaileak definitutako atea (gate izena(params) qargs { gorputza })
#[derive(Clone)]
struct GateDef {
    params: Vec<String>,
    qargs: Vec<String>,
    body: Vec<(String, u32)>,
}

struct Importer {
    qregs: HashMap<String, Reg>,
    cregs: HashMap<String, Reg>,
    n_qubits: u32,
    n_bits: u32,
    lines: Vec<String>,
    gates: HashMap<String, GateDef>,
    opaque: HashSet<String>,
    if_count: u32,
    // openqasm 3-ren bloke habiaratuen sakonera
    depth: u32,
    // puntu honetara iristen den bide guztietan idatzita dauden bitak. OpenQASM-en bitak 0 dira hasieran, QOA-n
    // irakurtzea errore bat da idatzi aurretik: idatzi gabeko bitak PACKBITS-ekin irakurtzen dira (0 gisa)
    written: HashSet<u32>,
}

// bloke habiaratuen muga: sarrera gaizto batek pila ez agortzeko
const BLOCK_DEPTH_MAX: u32 = 64;

// adierazpenaren lehen hitza (gako-hitzak ezagutzeko)
fn keyword(s: &str) -> &str {
    s.split(|c: char| c.is_whitespace() || c == '(' || c == '{')
        .next()
        .unwrap_or("")
}

// openqasm testua qoa testu bihurtzen du
pub fn to_qoa(src: &str, name: &str) -> Result<String, QasmError> {
    // `// SEAL sha256:...` azken lerroa bada, egiaztatu
    let src = match crate::seal::check(src.as_bytes(), crate::seal::Style::Comment, name) {
        Ok(Some(start)) => &src[..start],
        Ok(None) => src,
        Err(b) => {
            return Err(QasmError {
                line: b.line,
                message: b.message,
            });
        }
    };
    let clean = strip_comments(src);
    let mut imp = Importer {
        qregs: HashMap::new(),
        cregs: HashMap::new(),
        n_qubits: 0,
        n_bits: 0,
        lines: Vec::new(),
        gates: HashMap::new(),
        opaque: HashSet::new(),
        if_count: 0,
        depth: 0,
        written: HashSet::new(),
    };
    imp.block(&split_statements(&clean)?)?;
    let mut out = String::new();
    out.push_str(&t!(
        "; imported from {} by qoa {}\n",
        "; {} fitxategitik inportatua, qoa {}\n",
        name,
        env!("CARGO_PKG_VERSION")
    ));
    out.push_str(&format!("QINIT {}\n", imp.n_qubits));
    for l in &imp.lines {
        out.push_str(l);
        out.push('\n');
    }
    Ok(out)
}

impl Importer {
    // adierazpen-zerrenda bat; `if (...) { }`-ren ondoren datozen `else`-ak harekin batera
    fn block(&mut self, stmts: &[(String, u32)]) -> Result<(), QasmError> {
        let mut i = 0;
        while i < stmts.len() {
            let (s, line) = (&stmts[i].0, stmts[i].1);
            match keyword(s) {
                "if" if braced(s) => {
                    let mut j = i + 1;
                    while j < stmts.len() && keyword(&stmts[j].0) == "else" {
                        j += 1;
                    }
                    self.if_block(s, line, &stmts[i + 1..j])?;
                    i = j;
                }
                "else" => {
                    return Err(qerr(
                        line,
                        t!(
                            "else without an if {{ }} block before it",
                            "else bat aurretik if {{ }} blokerik gabe"
                        ),
                    ));
                }
                "while" => {
                    self.while_block(s, line)?;
                    i += 1;
                }
                "for" => {
                    self.for_block(s, line)?;
                    i += 1;
                }
                _ => {
                    self.statement(s, line)?;
                    i += 1;
                }
            }
        }
        Ok(())
    }

    // `{ ... }` gorputza: haren adierazpenak, lerro-zenbaki errealekin
    fn body(&mut self, s: &str, open: usize, line: u32) -> Result<(), QasmError> {
        if !s.ends_with('}') {
            return Err(qerr(
                line,
                t!(
                    "a {{ block must end with }}",
                    "{{ bloke batek }}-rekin amaitu behar du"
                ),
            ));
        }
        self.depth += 1;
        if self.depth > BLOCK_DEPTH_MAX {
            return Err(qerr(
                line,
                t!(
                    "blocks are nested more than {} deep",
                    "blokeak {} maila baino sakonago daude habiaratuta",
                    BLOCK_DEPTH_MAX
                ),
            ));
        }
        let first = line + s[..=open].matches('\n').count() as u32;
        let inner: Vec<(String, u32)> = split_statements(&s[open + 1..s.len() - 1])
            .map_err(|e| qerr(e.line + first - 1, e.message))?
            .into_iter()
            .map(|(t, l)| (t, l + first - 1))
            .collect();
        let r = self.block(&inner);
        self.depth -= 1;
        r
    }

    // `(baldintza)`: bitak eta balioa. `c[0] == true`, `c == 3`, `c[0] != 1`, `c[0]`, `!c[0]`
    fn condition(&self, cond: &str, line: u32) -> Result<(String, u64, bool), QasmError> {
        let cond = cond.trim();
        let (lhs, value, equal) = if let Some((l, r)) = cond.split_once("==") {
            (l, self.cond_value(r, line)?, true)
        } else if let Some((l, r)) = cond.split_once("!=") {
            (l, self.cond_value(r, line)?, false)
        } else if let Some(l) = cond.strip_prefix('!') {
            (l, 0, true)
        } else {
            (cond, 1, true)
        };
        let bits = match self.target(lhs.trim(), false, line)? {
            Target::One(b) => format!("c{}", b),
            Target::All(o, 1) => format!("c{}", o),
            Target::All(o, n) => format!("c{}..c{}", o, o + n - 1),
        };
        Ok((bits, value, equal))
    }

    fn cond_value(&self, r: &str, line: u32) -> Result<u64, QasmError> {
        match r.trim() {
            "true" => Ok(1),
            "false" => Ok(0),
            v => v.parse().map_err(|_| {
                qerr(
                    line,
                    t!(
                        "a condition compares bits with a whole number, true or false, found '{}'",
                        "baldintza batek bitak zenbaki oso batekin, true-rekin edo false-rekin konparatzen ditu, baina '{}' dago",
                        v
                    ),
                )
            }),
        }
    }

    // baldintza ez bada betetzen, `label`-era salto
    fn jump_unless(&mut self, cond: &str, label: &str, line: u32) -> Result<(), QasmError> {
        let (bits, value, equal) = self.condition(cond, line)?;
        self.skip_unless(&bits, value, equal, label);
        Ok(())
    }

    // `bits == value` (edo `!=`) ez bada, `label`-era salto. bit bakarra 0 edo 1-ekin: JUMPIFZERO/JUMPIFONE
    // zuzenean (erregistrorik gabe, beraz OpenQASM-era berriz esportatzen da); bestela bitak R31-en
    fn skip_unless(&mut self, bits: &str, value: u64, equal: bool, label: &str) {
        let one = bits
            .strip_prefix('c')
            .and_then(|k| k.parse::<u32>().ok())
            .filter(|k| self.written.contains(k));
        if one.is_some() && value <= 1 {
            let want = if equal { value } else { 1 - value };
            let op = if want == 1 { "JUMPIFZERO" } else { "JUMPIFONE" };
            self.lines.push(format!("{} {} {}", op, bits, label));
            return;
        }
        self.lines.push(format!("PACKBITS R31 {}", bits));
        self.lines.push(format!("REGSET R30 {}", value));
        self.lines.push(format!(
            "{} R31 R30 {}",
            if equal { "IFNE" } else { "IFEQ" },
            label
        ));
    }

    // `kw (baldintza) {`: baldintza eta `{`-ren kokapena
    fn head<'a>(&self, s: &'a str, kw: &str, line: u32) -> Result<(&'a str, usize), QasmError> {
        let open = s.find('(').ok_or_else(|| {
            qerr(
                line,
                t!(
                    "expected {} (condition) {{",
                    "{} (baldintza) {{ espero zen",
                    kw
                ),
            )
        })?;
        let close = matching_paren(s, open).ok_or_else(|| {
            qerr(
                line,
                t!(
                    "missing ) after the {} condition",
                    ") falta da {} baldintzaren ondoren",
                    kw
                ),
            )
        })?;
        let brace = close
            + 1
            + s[close + 1..].find('{').ok_or_else(|| {
                qerr(
                    line,
                    t!(
                        "expected {{ after the {} condition",
                        "{{ espero zen {} baldintzaren ondoren",
                        kw
                    ),
                )
            })?;
        if !s[close + 1..brace].trim().is_empty() {
            return Err(qerr(
                line,
                t!(
                    "expected {{ after the {} condition",
                    "{{ espero zen {} baldintzaren ondoren",
                    kw
                ),
            ));
        }
        Ok((&s[open + 1..close], brace))
    }

    // if (b) { A } else if (b2) { B } else { C }
    fn if_block(&mut self, s: &str, line: u32, elses: &[(String, u32)]) -> Result<(), QasmError> {
        let (cond, brace) = self.head(s, "if", line)?;
        self.if_count += 1;
        let k = self.if_count;
        let skip = format!("qasm_else_{}", k);
        let done = format!("qasm_done_{}", k);
        self.jump_unless(cond, &skip, line)?;
        let before = self.written.clone();
        self.body(s, brace, line)?;
        let then = std::mem::replace(&mut self.written, before.clone());
        if elses.is_empty() {
            self.lines.push(format!("{}:", skip));
            self.written = before;
            return Ok(());
        }
        self.lines.push(format!("JUMP {}", done));
        self.lines.push(format!("{}:", skip));
        let (e, eline) = (&elses[0].0, elses[0].1);
        let rest = e["else".len()..].trim_start();
        if keyword(rest) == "if" {
            // else if: gainerako else-ak katean
            let chain: Vec<(String, u32)> = std::iter::once((rest.to_string(), eline))
                .chain(elses[1..].iter().cloned())
                .collect();
            self.depth += 1;
            if self.depth > BLOCK_DEPTH_MAX {
                return Err(qerr(
                    eline,
                    t!(
                        "blocks are nested more than {} deep",
                        "blokeak {} maila baino sakonago daude habiaratuta",
                        BLOCK_DEPTH_MAX
                    ),
                ));
            }
            let r = self.block(&chain);
            self.depth -= 1;
            r?;
        } else {
            if elses.len() > 1 {
                return Err(qerr(
                    elses[1].1,
                    t!("else after an else", "else bat beste else baten ondoren"),
                ));
            }
            if !rest.starts_with('{') {
                return Err(qerr(
                    eline,
                    t!("expected else {{ ... }}", "else {{ ... }} espero zen"),
                ));
            }
            let open = e.len() - rest.len();
            self.body(e, open, eline)?;
        }
        // bi adarretan idatzitakoak bakarrik daude ziur idatzita
        self.written = self.written.intersection(&then).copied().collect();
        self.lines.push(format!("{}:", done));
        Ok(())
    }

    // while (b) { A }: baldintza hasieran, gorputzaren ondoren berriz
    fn while_block(&mut self, s: &str, line: u32) -> Result<(), QasmError> {
        let (cond, brace) = self.head(s, "while", line)?;
        self.if_count += 1;
        let top = format!("qasm_while_{}", self.if_count);
        let end = format!("qasm_wend_{}", self.if_count);
        self.lines.push(format!("{}:", top));
        // baldintza lehen aldiz eta gorputzaren ondoren irakurtzen da: aurretik idatzitakoekin bakarrik
        self.jump_unless(cond, &end, line)?;
        let before = self.written.clone();
        self.body(s, brace, line)?;
        self.written = before;
        self.lines.push(format!("JUMP {}", top));
        self.lines.push(format!("{}:", end));
        Ok(())
    }

    // for uint i in [a:b] { A } (edo [a:urratsa:b], edo {1, 2, 3}): LOOPSTART kopuruarekin. aldagaia gorputzean
    // erabiltzea ez da onartzen (QOA-ren begiztek ez dute kontagailurik)
    fn for_block(&mut self, s: &str, line: u32) -> Result<(), QasmError> {
        let bad = || {
            qerr(
                line,
                t!(
                    "expected for [type] name in [start:end] {{ ... }}",
                    "for [mota] izena in [hasiera:amaiera] {{ ... }} espero zen"
                ),
            )
        };
        let rest = s["for".len()..].trim_start();
        let (before, after) = rest.split_once(" in ").ok_or_else(bad)?;
        let var = before.split_whitespace().last().ok_or_else(bad)?;
        let after = after.trim_start();
        let (count, brace) = if let Some(r) = after.strip_prefix('[') {
            let close = r.find(']').ok_or_else(bad)?;
            let parts: Vec<&str> = r[..close].split(':').collect();
            let num = |t: &str| -> Result<i64, QasmError> {
                t.trim().parse::<i64>().map_err(|_| {
                    qerr(
                        line,
                        t!(
                            "a for range must use whole numbers, found '{}'",
                            "for barruti batek zenbaki osoak erabili behar ditu, baina '{}' dago",
                            t.trim()
                        ),
                    )
                })
            };
            let (a, step, b) = match parts.as_slice() {
                [a, b] => (num(a)?, 1, num(b)?),
                [a, st, b] => (num(a)?, num(st)?, num(b)?),
                _ => return Err(bad()),
            };
            if step == 0 {
                return Err(qerr(
                    line,
                    t!(
                        "a for range step cannot be 0",
                        "for barruti baten urratsa ezin da 0 izan"
                    ),
                ));
            }
            // [a:b] bi muturrak barne (i128: muturrak edozein i64 izan daitezke)
            let (a, step, b) = (a as i128, step as i128, b as i128);
            let n = if (step > 0 && b < a) || (step < 0 && b > a) {
                0
            } else {
                ((b - a) / step + 1).min(u64::MAX as i128) as u64
            };
            (n, s.len() - after.len() + 1 + close + 1)
        } else if let Some(r) = after.strip_prefix('{') {
            let close = r.find('}').ok_or_else(bad)?;
            let n = r[..close]
                .split(',')
                .filter(|x| !x.trim().is_empty())
                .count() as u64;
            (n, s.len() - after.len() + 1 + close + 1)
        } else {
            return Err(bad());
        };
        // barrutiaren ondoren `{` bakarrik
        let gap = s[brace..].len() - s[brace..].trim_start().len();
        let brace = brace + gap;
        if !s[brace..].starts_with('{') {
            return Err(bad());
        }
        let body = &s[brace..];
        if body
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|w| w == var)
        {
            return Err(qerr(
                line,
                t!(
                    "the loop variable '{}' is used inside the loop; QOA loops have no counter, so this is not supported",
                    "'{}' begizta-aldagaia begiztaren barruan erabiltzen da; QOA-ren begiztek ez dute kontagailurik, beraz ez da onartzen",
                    var
                ),
            ));
        }
        if count > u32::MAX as u64 {
            return Err(qerr(
                line,
                t!(
                    "the loop runs too many times",
                    "begizta gehiegitan exekutatzen da"
                ),
            ));
        }
        if count == 0 {
            return Ok(());
        }
        self.lines.push(format!("LOOPSTART {}", count));
        self.body(s, brace, line)?;
        self.lines.push("LOOPEND".to_string());
        Ok(())
    }

    fn statement(&mut self, s: &str, line: u32) -> Result<(), QasmError> {
        if s.is_empty() {
            return Ok(());
        }
        let first = s
            .split(|c: char| c.is_whitespace() || c == '(')
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        // `uint[4]`, `float[64]`: mota-izena kortxeteen aurretik
        let kw = first.split('[').next().unwrap_or("");
        match kw {
            "openqasm" | "include" => return Ok(()),
            "gate" => return self.define(s, line),
            "opaque" => {
                let name = s[6..]
                    .trim()
                    .split(|c: char| c.is_whitespace() || c == '(')
                    .next()
                    .unwrap_or("")
                    .to_string();
                self.opaque.insert(name);
                return Ok(());
            }
            "if" => return self.if_statement(s, line),
            "def" | "defcal" | "cal" | "const" | "input" | "output" | "let" | "box" | "switch"
            | "extern" | "gphase" | "int" | "uint" | "float" | "angle" | "bool" | "complex"
            | "duration" | "stretch" | "break" | "continue" | "return" | "end" => {
                return Err(qerr(
                    line,
                    t!(
                        "'{}' is an OpenQASM 3 feature the importer does not support (OpenQASM 2.0 is fully supported; of OpenQASM 3, bits, if/else, while and for on measured bits, and gate modifiers)",
                        "'{}' OpenQASM 3-ren ezaugarri bat da, eta inportatzaileak ez du onartzen (OpenQASM 2.0 osorik onartzen da; OpenQASM 3-tik, bitak, if/else, while eta for bit neurtuekin, eta ate-modifikatzaileak)",
                        kw
                    ),
                ));
            }
            _ => {}
        }
        // ate-modifikatzaileak (inv @, ctrl @, ctrl(n) @, pow(k) @): deiaren aurretik
        let (mods, s) = split_modifiers(s, line)?;
        if s.contains('{') {
            return Err(qerr(
                line,
                t!("cannot import '{}'", "ezin da '{}' inportatu", s),
            ));
        }

        // erregistro-deklarazioak
        if let Some(rest) = s.strip_prefix("qreg").or_else(|| s.strip_prefix("creg")) {
            let quantum = s.starts_with("qreg");
            let (name, size) = parse_decl_v2(rest).ok_or_else(|| {
                qerr(
                    line,
                    t!(
                        "cannot read declaration '{}'",
                        "ezin da '{}' deklarazioa irakurri",
                        s
                    ),
                )
            })?;
            return self.declare(quantum, name, size, line);
        }
        if first == "qubit"
            || first == "bit"
            || first.starts_with("qubit[")
            || first.starts_with("bit[")
        {
            let quantum = s.starts_with("qubit");
            let rest = s
                .trim_start_matches("qubit")
                .trim_start_matches("bit")
                .trim();
            let (size, name) = if let Some(r) = rest.strip_prefix('[') {
                let (n, after) = r.split_once(']').ok_or_else(|| {
                    qerr(
                        line,
                        t!("missing ] in declaration", "] falta da deklarazioan"),
                    )
                })?;
                (
                    n.trim()
                        .parse::<u32>()
                        .map_err(|_| qerr(line, t!("bad size '{}'", "'{}' tamaina okerra", n)))?,
                    after.trim(),
                )
            } else {
                (1, rest)
            };
            if name.contains('=') {
                return Err(qerr(
                    line,
                    t!(
                        "declaring and assigning in one statement is not supported; declare first",
                        "adierazpen bakarrean deklaratu eta esleitzea ez da onartzen; deklaratu lehenik"
                    ),
                ));
            }
            return self.declare(quantum, name.to_string(), size, line);
        }

        // neurketak
        if let Some(rest) = s.strip_prefix("measure") {
            let (q, c) = rest.split_once("->").ok_or_else(|| {
                qerr(
                    line,
                    t!("expected measure q -> c", "measure q -> c espero zen"),
                )
            })?;
            return self.measure(q.trim(), c.trim(), line);
        }
        if let Some((c, q)) = s.split_once('=') {
            if let Some(q) = q.trim().strip_prefix("measure") {
                return self.measure(q.trim(), c.trim(), line);
            }
            return Err(qerr(
                line,
                t!("cannot import '{}'", "ezin da '{}' inportatu", s),
            ));
        }
        if let Some(rest) = s.strip_prefix("reset") {
            for q in self.expand_one(rest.trim(), line)? {
                self.lines.push(format!("RESET q{}", q));
            }
            return Ok(());
        }
        if let Some(rest) = s.strip_prefix("barrier") {
            let mut qs = Vec::new();
            for a in split_args(rest) {
                qs.extend(self.expand_one(&a, line)?);
            }
            self.lines.push(barrier_line(&qs));
            return Ok(());
        }

        // ate-deia
        let (gname, ptexts, args) = parse_call(s).map_err(|m| qerr(line, m))?;
        let mut params = Vec::new();
        for p in &ptexts {
            params.push(
                expr::eval(p).map_err(|e| {
                    qerr(line, t!("parameter '{}': {}", "'{}' parametroa: {}", p, e))
                })?,
            );
        }
        let (gname, params, reps) = modified(&mods, gname, params, line)?;
        let targets: Vec<Target> = split_args(&args)
            .iter()
            .map(|a| self.target(a, true, line))
            .collect::<Result<_, _>>()?;
        // hedapena: erregistro osoak elementuz elementu
        let width = targets.iter().find_map(|t| match t {
            Target::All(_, n) => Some(*n),
            _ => None,
        });
        if let Some(w) = width
            && targets
                .iter()
                .any(|t| matches!(t, Target::All(_, n) if *n != w))
        {
            return Err(qerr(
                line,
                t!(
                    "registers of different sizes in one gate call",
                    "tamaina desberdineko erregistroak ate-dei berean"
                ),
            ));
        }
        for k in 0..width.unwrap_or(1) {
            let qs: Vec<u32> = targets
                .iter()
                .map(|t| match t {
                    Target::One(q) => *q,
                    Target::All(o, _) => o + k,
                })
                .collect();
            for _ in 0..reps {
                self.apply(&gname, &params, &qs, line, 0)?;
            }
        }
        Ok(())
    }

    // gate izena(a, b) x, y { ... }
    fn define(&mut self, s: &str, line: u32) -> Result<(), QasmError> {
        let open = s.find('{').ok_or_else(|| {
            qerr(
                line,
                t!(
                    "a gate definition needs a { body }",
                    "ate-definizio batek { gorputz } bat behar du"
                ),
            )
        })?;
        // gorputza { eta azken } artean; } {-ren aurretik badago, ez dago gorputzik
        let close = s.rfind('}').filter(|&c| c > open).ok_or_else(|| {
            qerr(
                line,
                t!(
                    "a gate definition needs a { body }",
                    "ate-definizio batek { gorputz } bat behar du"
                ),
            )
        })?;
        let header = s[4..open].trim();
        let (name, params, qargs) = match header.find('(') {
            Some(p) => {
                // ( -ren ondoko lehen ), ez aurretik dagoen bat
                let pc = header[p..].find(')').map(|k| p + k).ok_or_else(|| {
                    qerr(
                        line,
                        t!("missing ) in gate definition", ") falta da ate-definizioan"),
                    )
                })?;
                (
                    header[..p].trim().to_string(),
                    split_args(&header[p + 1..pc]),
                    split_args(&header[pc + 1..]),
                )
            }
            None => {
                let (n, rest) = header
                    .split_once(char::is_whitespace)
                    .unwrap_or((header, ""));
                (n.trim().to_string(), Vec::new(), split_args(rest))
            }
        };
        if name.is_empty() || qargs.is_empty() {
            return Err(qerr(
                line,
                t!(
                    "a gate definition needs a name and at least one qubit argument",
                    "ate-definizio batek izen bat eta gutxienez qubit-argumentu bat behar ditu"
                ),
            ));
        }
        // gorputzeko adierazpenak, bakoitza bere lerroarekin
        let body_src = &s[open + 1..close];
        let body_line = line + s[..open].matches('\n').count() as u32;
        let mut body = Vec::new();
        let mut offset = 0u32;
        for piece in body_src.split(';') {
            let lead = piece.len() - piece.trim_start().len();
            let here = body_line + offset + piece[..lead].matches('\n').count() as u32;
            offset += piece.matches('\n').count() as u32;
            if !piece.trim().is_empty() {
                body.push((piece.trim().to_string(), here));
            }
        }
        if native_arity(&name).is_some() {
            // qelib1.inc txertatuta dagoenean gertatzen da: qoa-ren ate berezkoa erabiltzen da
            self.lines.push(t!("; gate {} is defined in the source; QOA's built-in {} is used (same matrix)", "; {} atea iturburuan definituta dago; QOA-ren berezko {} erabiltzen da (matrize bera)", name, name));
            return Ok(());
        }
        self.gates.insert(
            name,
            GateDef {
                params,
                qargs,
                body,
            },
        );
        Ok(())
    }

    // if (c == n) agindua
    fn if_statement(&mut self, s: &str, line: u32) -> Result<(), QasmError> {
        let open = s.find('(').ok_or_else(|| {
            qerr(
                line,
                t!(
                    "expected if (creg == value)",
                    "if (creg == balioa) espero zen"
                ),
            )
        })?;
        let close = matching_paren(s, open).ok_or_else(|| {
            qerr(
                line,
                t!(
                    "missing ) after the if condition",
                    ") falta da if baldintzaren ondoren"
                ),
            )
        })?;
        let cond = &s[open + 1..close];
        let rest = s[close + 1..].trim();
        let (lhs, rhs) = cond.split_once("==").ok_or_else(|| {
            qerr(
                line,
                t!(
                    "only == conditions are supported in if",
                    "if-en == baldintzak bakarrik onartzen dira"
                ),
            )
        })?;
        let value: u64 = rhs.trim().parse().map_err(|_| {
            qerr(
                line,
                t!(
                    "the if value must be a whole number, found '{}'",
                    "if-en balioak zenbaki osoa izan behar du, baina '{}' dago",
                    rhs.trim()
                ),
            )
        })?;
        let bits = match self.target(lhs.trim(), false, line)? {
            Target::One(b) => format!("c{}", b),
            Target::All(o, 1) => format!("c{}", o),
            Target::All(o, n) => format!("c{}..c{}", o, o + n - 1),
        };
        self.if_count += 1;
        let label = format!("qasm_skip_{}", self.if_count);
        self.lines.push(format!("; if ({}) {}", cond.trim(), rest));
        self.skip_unless(&bits, value, true, &label);
        let before = self.written.clone();
        self.statement(rest, line)?;
        self.written = before;
        self.lines.push(format!("{}:", label));
        Ok(())
    }

    fn declare(
        &mut self,
        quantum: bool,
        name: String,
        size: u32,
        line: u32,
    ) -> Result<(), QasmError> {
        let name = name.trim().to_string();
        if self.qregs.contains_key(&name) || self.cregs.contains_key(&name) {
            return Err(qerr(
                line,
                t!(
                    "register '{}' is declared twice",
                    "'{}' erregistroa bi aldiz deklaratuta dago",
                    name
                ),
            ));
        }
        if quantum {
            self.qregs.insert(
                name,
                Reg {
                    offset: self.n_qubits,
                    size,
                },
            );
            self.n_qubits += size;
        } else {
            self.cregs.insert(
                name,
                Reg {
                    offset: self.n_bits,
                    size,
                },
            );
            self.n_bits += size;
        }
        Ok(())
    }

    fn target(&self, arg: &str, quantum: bool, line: u32) -> Result<Target, QasmError> {
        let regs = if quantum { &self.qregs } else { &self.cregs };
        let arg = arg.trim();
        if let Some((name, rest)) = arg.split_once('[') {
            let idx_text = rest
                .strip_suffix(']')
                .ok_or_else(|| qerr(line, t!("missing ] in '{}'", "] falta da '{}'-n", arg)))?;
            let reg = regs.get(name.trim()).ok_or_else(|| {
                qerr(
                    line,
                    t!(
                        "unknown register '{}'",
                        "'{}' erregistro ezezaguna",
                        name.trim()
                    ),
                )
            })?;
            let idx: u32 = idx_text
                .trim()
                .parse()
                .map_err(|_| qerr(line, t!("bad index '{}'", "'{}' indize okerra", idx_text)))?;
            if idx >= reg.size {
                return Err(qerr(
                    line,
                    t!(
                        "{}[{}] is out of range (size {})",
                        "{}[{}] tartetik kanpo dago ({} tamaina)",
                        name.trim(),
                        idx,
                        reg.size
                    ),
                ));
            }
            Ok(Target::One(reg.offset + idx))
        } else {
            let reg = regs.get(arg).ok_or_else(|| {
                qerr(
                    line,
                    t!("unknown register '{}'", "'{}' erregistro ezezaguna", arg),
                )
            })?;
            Ok(Target::All(reg.offset, reg.size))
        }
    }

    fn expand_one(&self, arg: &str, line: u32) -> Result<Vec<u32>, QasmError> {
        Ok(match self.target(arg, true, line)? {
            Target::One(q) => vec![q],
            Target::All(o, n) => (o..o + n).collect(),
        })
    }

    fn measure(&mut self, q: &str, c: &str, line: u32) -> Result<(), QasmError> {
        match (self.target(q, true, line)?, self.target(c, false, line)?) {
            (Target::One(q), Target::One(c)) => {
                self.lines.push(format!("MEASURE q{} -> c{}", q, c));
                self.written.insert(c);
            }
            (Target::All(qo, qn), Target::All(co, cn)) if qn == cn => {
                for k in 0..qn {
                    self.lines
                        .push(format!("MEASURE q{} -> c{}", qo + k, co + k));
                    self.written.insert(co + k);
                }
            }
            _ => {
                return Err(qerr(
                    line,
                    t!(
                        "measure needs matching sizes on both sides",
                        "measure-k tamaina bera behar du bi aldeetan"
                    ),
                ));
            }
        }
        Ok(())
    }

    // ate bat qubit zehatzetan: berezkoa bada qoa lerroak, definitua bada gorputza zabaldu
    fn apply(
        &mut self,
        name: &str,
        p: &[f64],
        qs: &[u32],
        line: u32,
        depth: u32,
    ) -> Result<(), QasmError> {
        if depth > 64 {
            return Err(qerr(
                line,
                t!(
                    "gate '{}' expands more than 64 levels deep (recursive definition?)",
                    "'{}' atea 64 maila baino sakonago zabaltzen da (definizio errekurtsiboa?)",
                    name
                ),
            ));
        }
        if let Some((nparams, nqubits)) = native_arity(name) {
            if p.len() != nparams {
                return Err(qerr(
                    line,
                    t!(
                        "{} takes {} parameter(s), found {}",
                        "{}-k {} parametro hartzen ditu, baina {} daude",
                        name,
                        nparams,
                        p.len()
                    ),
                ));
            }
            if qs.len() != nqubits {
                return Err(qerr(
                    line,
                    t!(
                        "{} acts on {} qubit(s), found {}",
                        "{} {} qubitetan aritzen da, baina {} daude",
                        name,
                        nqubits,
                        qs.len()
                    ),
                ));
            }
            for k in 0..qs.len() {
                if qs[..k].contains(&qs[k]) {
                    return Err(qerr(
                        line,
                        t!(
                            "{} uses qubit {} twice",
                            "{}-k {}. qubita bi aldiz erabiltzen du",
                            name,
                            qs[k]
                        ),
                    ));
                }
            }
            for l in native_lines(name, p, qs) {
                self.lines.push(l);
            }
            return Ok(());
        }
        if let Some(def) = self.gates.get(name).cloned() {
            if p.len() != def.params.len() {
                return Err(qerr(
                    line,
                    t!(
                        "{} takes {} parameter(s), found {}",
                        "{}-k {} parametro hartzen ditu, baina {} daude",
                        name,
                        def.params.len(),
                        p.len()
                    ),
                ));
            }
            if qs.len() != def.qargs.len() {
                return Err(qerr(
                    line,
                    t!(
                        "{} acts on {} qubit(s), found {}",
                        "{} {} qubitetan aritzen da, baina {} daude",
                        name,
                        def.qargs.len(),
                        qs.len()
                    ),
                ));
            }
            let env: HashMap<String, f64> =
                def.params.iter().cloned().zip(p.iter().copied()).collect();
            let qmap: HashMap<&str, u32> = def
                .qargs
                .iter()
                .map(|s| s.as_str())
                .zip(qs.iter().copied())
                .collect();
            for (stmt, sline) in &def.body {
                if let Some(rest) = stmt.strip_prefix("barrier") {
                    let mut mapped = Vec::new();
                    for a in split_args(rest) {
                        mapped.push(*qmap.get(a.as_str()).ok_or_else(|| {
                            qerr(
                                *sline,
                                t!(
                                    "'{}' is not an argument of gate {}",
                                    "'{}' ez da {} atearen argumentu bat",
                                    a,
                                    name
                                ),
                            )
                        })?);
                    }
                    self.lines.push(barrier_line(&mapped));
                    continue;
                }
                let (inner, ptexts, args) = parse_call(stmt).map_err(|m| qerr(*sline, m))?;
                let mut params = Vec::new();
                for t in &ptexts {
                    params.push(expr::eval_with(t, &env).map_err(|e| {
                        qerr(
                            *sline,
                            t!("parameter '{}': {}", "'{}' parametroa: {}", t, e),
                        )
                    })?);
                }
                let mut mapped = Vec::new();
                for a in split_args(&args) {
                    mapped.push(*qmap.get(a.as_str()).ok_or_else(|| {
                        qerr(*sline, t!("'{}' is not an argument of gate {} (gate bodies use argument names, not register indices)", "'{}' ez da {} atearen argumentu bat (ate-gorputzek argumentu-izenak erabiltzen dituzte, ez erregistro-indizeak)", a, name))
                    })?);
                }
                self.apply(&inner, &params, &mapped, *sline, depth + 1)?;
            }
            return Ok(());
        }
        if self.opaque.contains(name) {
            return Err(qerr(
                line,
                t!(
                    "gate '{}' is declared opaque: it has no definition, so it cannot be simulated",
                    "'{}' atea opaque gisa deklaratuta dago: ez du definiziorik, beraz ezin da simulatu",
                    name
                ),
            ));
        }
        Err(qerr(
            line,
            t!(
                "unknown gate '{}' (not in qelib1.inc and not defined in the file)",
                "'{}' ate ezezaguna (ez dago qelib1.inc-n eta ez dago fitxategian definituta)",
                name
            ),
        ))
    }
}

// `if (...) { ... }` (blokea) ala `if (...) ate;` (OpenQASM 2.0)
fn braced(s: &str) -> bool {
    s.find('(')
        .and_then(|open| matching_paren(s, open))
        .is_some_and(|close| s[close + 1..].trim_start().starts_with('{'))
}

// ate-modifikatzaile bat
#[derive(Clone, Copy)]
enum Modifier {
    Inv,
    Ctrl(u32),
    Pow(i64),
}

// `inv @ ctrl(2) @ x q[0], q[1], q[2]` -> ([Inv, Ctrl(2)], "x q[0], q[1], q[2]")
fn split_modifiers(s: &str, line: u32) -> Result<(Vec<Modifier>, &str), QasmError> {
    let mut mods = Vec::new();
    let mut rest = s;
    while let Some(at) = rest.find('@') {
        let m = rest[..at].trim();
        let arg = |m: &str| -> Option<i64> {
            let inner = m.split_once('(')?.1.strip_suffix(')')?;
            expr::eval(inner)
                .ok()
                .filter(|v| v.fract() == 0.0 && v.abs() <= 1e6)
                .map(|v| v as i64)
        };
        mods.push(match m.split('(').next().unwrap_or("").trim() {
            "inv" if m == "inv" => Modifier::Inv,
            "ctrl" if m == "ctrl" => Modifier::Ctrl(1),
            "ctrl" => match arg(m) {
                Some(n) if (1..=4).contains(&n) => Modifier::Ctrl(n as u32),
                _ => {
                    return Err(qerr(
                        line,
                        t!(
                            "'{}': ctrl(n) takes 1 to 4 controls",
                            "'{}': ctrl(n)-k 1etik 4ra kontrol hartzen ditu",
                            m
                        ),
                    ));
                }
            },
            "pow" => match arg(m) {
                Some(k) => Modifier::Pow(k),
                None => {
                    return Err(qerr(
                        line,
                        t!(
                            "'{}': only whole-number powers are supported",
                            "'{}': berretzaile osoak bakarrik onartzen dira",
                            m
                        ),
                    ));
                }
            },
            _ => {
                return Err(qerr(
                    line,
                    t!(
                        "gate modifier '{}' is not supported (inv, ctrl, ctrl(n) and pow(k) are)",
                        "'{}' ate-modifikatzailea ez da onartzen (inv, ctrl, ctrl(n) eta pow(k) bai)",
                        m
                    ),
                ));
            }
        });
        rest = rest[at + 1..].trim_start();
    }
    Ok((mods, rest))
}

// modifikatzaileak ate berezko batean: barrukoenetik (eskuinetik) kanpokoenera. (izena, parametroak, errepikapenak)
fn modified(
    mods: &[Modifier],
    name: String,
    params: Vec<f64>,
    line: u32,
) -> Result<(String, Vec<f64>, u64), QasmError> {
    let mut name = name;
    let mut params = params;
    let mut reps: i64 = 1;
    for m in mods.iter().rev() {
        let next = match *m {
            Modifier::Inv => {
                reps = -reps;
                Some((name.clone(), params.clone()))
            }
            Modifier::Pow(k) => {
                reps = reps
                    .checked_mul(k)
                    .filter(|r| r.abs() <= 1_000_000)
                    .ok_or_else(|| {
                        qerr(
                            line,
                            t!(
                                "the gate power is too large",
                                "atearen berretzailea handiegia da"
                            ),
                        )
                    })?;
                Some((name.clone(), params.clone()))
            }
            // ctrl @ (U^k) = (ctrl @ U)^k, beraz errepikapenak eta alderantzikatzea bukaeran
            Modifier::Ctrl(n) => (0..n).try_fold((name.clone(), params.clone()), |(g, p), _| {
                controlled(&g, &p)
            }),
        };
        let (g, p) = next.ok_or_else(|| {
            qerr(
                line,
                t!(
                    "the importer has no controlled form of '{}'",
                    "inportatzaileak ez du '{}'-ren forma kontrolaturik",
                    name
                ),
            )
        })?;
        name = g;
        params = p;
    }
    if reps < 0 {
        let (g, p) = inverse(&name, &params).ok_or_else(|| {
            qerr(
                line,
                t!(
                    "the importer has no inverse of '{}'",
                    "inportatzaileak ez du '{}'-ren alderantzizkorik",
                    name
                ),
            )
        })?;
        name = g;
        params = p;
    }
    Ok((name, params, reps.unsigned_abs()))
}

// ate berezko baten alderantzizkoa
fn inverse(name: &str, p: &[f64]) -> Option<(String, Vec<f64>)> {
    if native_arity(name).is_some_and(|(n, _)| n != p.len()) {
        // parametro-kopuru okerra: apply-k esango du zer dagoen gaizki
        return Some((name.to_string(), p.to_vec()));
    }
    let neg: Vec<f64> = p.iter().map(|x| -x).collect();
    let same = |n: &str| Some((n.to_string(), p.to_vec()));
    match name {
        "id" | "u0" | "x" | "y" | "z" | "h" | "cx" | "CX" | "cy" | "cz" | "ch" | "swap" | "ccx"
        | "cswap" | "c3x" | "c4x" => same(name),
        "s" => same("sdg"),
        "sdg" => same("s"),
        "t" => same("tdg"),
        "tdg" => same("t"),
        "sx" => same("sxdg"),
        "sxdg" => same("sx"),
        "rx" | "ry" | "rz" | "p" | "phase" | "u1" | "crx" | "cry" | "crz" | "cp" | "cphase"
        | "cu1" | "rxx" | "rzz" => Some((name.to_string(), neg)),
        // U(θ,φ,λ)^-1 = U(-θ,-λ,-φ)
        "u" | "u3" | "U" => Some(("u3".to_string(), vec![-p[0], -p[2], -p[1]])),
        "cu3" => Some(("cu3".to_string(), vec![-p[0], -p[2], -p[1]])),
        "cu" => Some(("cu".to_string(), vec![-p[0], -p[2], -p[1], -p[3]])),
        // u2(φ,λ) = U(π/2,φ,λ)
        "u2" => Some((
            "u3".to_string(),
            vec![-std::f64::consts::FRAC_PI_2, -p[1], -p[0]],
        )),
        _ => None,
    }
}

// ate berezko baten bertsio kontrolatua (kontrola lehen qubita)
fn controlled(name: &str, p: &[f64]) -> Option<(String, Vec<f64>)> {
    let g = match name {
        "x" => "cx",
        "y" => "cy",
        "z" => "cz",
        "h" => "ch",
        "sx" => "csx",
        "p" | "phase" | "u1" => "cp",
        "rx" => "crx",
        "ry" => "cry",
        "rz" => "crz",
        "u" | "u3" | "U" => "cu3",
        "swap" => "cswap",
        "cx" | "CX" => "ccx",
        "ccx" => "c3x",
        "c3x" => "c4x",
        _ => return None,
    };
    Some((g.to_string(), p.to_vec()))
}

// qelib1.inc-ko ateak (jatorrizkoa eta qiskit-en bertsio zabaldua): (parametroak, qubitak)
fn native_arity(name: &str) -> Option<(usize, usize)> {
    Some(match name {
        "id" | "x" | "y" | "z" | "h" | "s" | "sdg" | "t" | "tdg" | "sx" | "sxdg" => (0, 1),
        "u0" | "rx" | "ry" | "rz" | "p" | "u1" | "phase" => (1, 1),
        "u2" => (2, 1),
        "u3" | "u" | "U" => (3, 1),
        "cx" | "CX" | "cy" | "cz" | "ch" | "csx" | "swap" => (0, 2),
        "cp" | "cu1" | "cphase" | "crx" | "cry" | "crz" | "rxx" | "rzz" => (1, 2),
        "cu3" => (3, 2),
        "cu" => (4, 2),
        "ccx" | "cswap" | "rccx" => (0, 3),
        "c3x" => (0, 4),
        "c4x" => (0, 5),
        _ => return None,
    })
}

// ate berezko bat qoa lerro bihurtu (zenbakiak doitasun osoz)
fn native_lines(name: &str, p: &[f64], qs: &[u32]) -> Vec<String> {
    let q = |i: usize| format!("q{}", qs[i]);
    let n = |x: f64| format!("{}", x);
    let one = |m: &str| vec![format!("{} {}", m, q(0))];
    match name {
        "id" | "u0" => vec![],
        "x" => one("X"),
        "y" => one("Y"),
        "z" => one("Z"),
        "h" => one("H"),
        "s" => one("S"),
        "sdg" => one("SDG"),
        "t" => one("T"),
        "tdg" => one("TDG"),
        "sx" => one("SX"),
        "sxdg" => one("SXDG"),
        "rx" => vec![format!("RX {} {}", q(0), n(p[0]))],
        "ry" => vec![format!("RY {} {}", q(0), n(p[0]))],
        "rz" => vec![format!("RZ {} {}", q(0), n(p[0]))],
        "p" | "u1" | "phase" => vec![format!("PHASE {} {}", q(0), n(p[0]))],
        "u2" => vec![format!("U {} pi/2 {} {}", q(0), n(p[0]), n(p[1]))],
        "u3" | "u" | "U" => vec![format!("U {} {} {} {}", q(0), n(p[0]), n(p[1]), n(p[2]))],
        "cx" | "CX" => vec![format!("CNOT {} {}", q(0), q(1))],
        "cy" => vec![format!("CY {} {}", q(0), q(1))],
        "cz" => vec![format!("CZ {} {}", q(0), q(1))],
        "ch" => vec![format!("CH {} {}", q(0), q(1))],
        "csx" => vec![format!("CSX {} {}", q(0), q(1))],
        "swap" => vec![format!("SWAP {} {}", q(0), q(1))],
        "cp" | "cu1" | "cphase" => vec![format!("CPHASE {} {} {}", q(0), q(1), n(p[0]))],
        "crx" => vec![format!("CRX {} {} {}", q(0), q(1), n(p[0]))],
        "cry" => vec![format!("CRY {} {} {}", q(0), q(1), n(p[0]))],
        "crz" => vec![format!("CRZ {} {} {}", q(0), q(1), n(p[0]))],
        "cu3" => vec![format!(
            "CU {} {} {} {} {}",
            q(0),
            q(1),
            n(p[0]),
            n(p[1]),
            n(p[2])
        )],
        // qiskit-en cu: e^(i·gamma) faseko kontrolatutako U; fasea kontrol-qubitean
        "cu" => vec![
            format!("PHASE {} {}", q(0), n(p[3])),
            format!("CU {} {} {} {} {}", q(0), q(1), n(p[0]), n(p[1]), n(p[2])),
        ],
        // rzz(θ) = exp(-iθ/2 Z⊗Z) = cx; rz(θ); cx
        "rzz" => vec![
            format!("CNOT {} {}", q(0), q(1)),
            format!("RZ {} {}", q(1), n(p[0])),
            format!("CNOT {} {}", q(0), q(1)),
        ],
        // rxx(θ) = exp(-iθ/2 X⊗X): h oinarrian rzz bera
        "rxx" => vec![
            format!("H {}", q(0)),
            format!("H {}", q(1)),
            format!("CNOT {} {}", q(0), q(1)),
            format!("RZ {} {}", q(1), n(p[0])),
            format!("CNOT {} {}", q(0), q(1)),
            format!("H {}", q(0)),
            format!("H {}", q(1)),
        ],
        "ccx" => vec![format!("CCX {} {} {}", q(0), q(1), q(2))],
        "cswap" => vec![format!("CSWAP {} {} {}", q(0), q(1), q(2))],
        // erlatibo-faseko toffoli, qelib1.inc-ko definizio bera
        "rccx" => vec![
            format!("U {} pi/2 0 pi", q(2)),
            format!("PHASE {} pi/4", q(2)),
            format!("CNOT {} {}", q(1), q(2)),
            format!("PHASE {} -pi/4", q(2)),
            format!("CNOT {} {}", q(0), q(2)),
            format!("PHASE {} pi/4", q(2)),
            format!("CNOT {} {}", q(1), q(2)),
            format!("PHASE {} -pi/4", q(2)),
            format!("U {} pi/2 0 pi", q(2)),
        ],
        "c3x" => vec![format!("MCX {} {} {} {}", q(0), q(1), q(2), q(3))],
        "c4x" => vec![format!("MCX {} {} {} {} {}", q(0), q(1), q(2), q(3), q(4))],
        _ => unreachable!("native_arity eta native_lines ez datoz bat {}-n", name),
    }
}

fn barrier_line(qs: &[u32]) -> String {
    let list: Vec<String> = qs.iter().map(|q| format!("q{}", q)).collect();
    format!("BARRIER {}", list.join(" ")).trim_end().to_string()
}

// izena(p1, p2) arg1, arg2 → (izena, parametroak, argumentuak)
fn parse_call(s: &str) -> Result<(String, Vec<String>, String), String> {
    let s = s.trim();
    let name_end = s
        .find(|c: char| c.is_whitespace() || c == '(')
        .ok_or_else(|| t!("cannot import '{}'", "ezin da '{}' inportatu", s))?;
    let name = s[..name_end].to_string();
    let rest = s[name_end..].trim_start();
    if let Some(after_open) = rest.strip_prefix('(') {
        let open = s.len() - rest.len();
        let close = matching_paren(s, open).ok_or(t!("missing )", ") falta da"))?;
        let _ = after_open;
        let params = split_params(&s[open + 1..close]);
        Ok((name, params, s[close + 1..].trim().to_string()))
    } else {
        Ok((name, Vec::new(), rest.to_string()))
    }
}

fn parse_decl_v2(rest: &str) -> Option<(String, u32)> {
    let (name, r) = rest.split_once('[')?;
    let n = r.trim().strip_suffix(']')?.trim().parse().ok()?;
    Some((name.trim().to_string(), n))
}

fn matching_paren(s: &str, open: usize) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in s.char_indices().skip_while(|(i, _)| *i < open) {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_args(s: &str) -> Vec<String> {
    s.split(',')
        .map(|a| a.trim().to_string())
        .filter(|a| !a.is_empty())
        .collect()
}

// parametroak komaz banatu, parentesi barrukoak kontuan hartuz
fn split_params(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

// ---------------------------------------------------------------- esportazioa

#[derive(Debug, Clone, PartialEq)]
pub struct ExportError {
    pub line: u32,
    pub message: String,
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line > 0 {
            write!(
                f,
                "{}",
                t!("line {}: {}", "{}. lerroa: {}", self.line, self.message)
            )
        } else {
            write!(f, "{}", self.message)
        }
    }
}

// qoa programa openqasm 2.0 bihurtzen du (qelib1.inc ateak erabiliz)
// esportaziorako programa laua: (baldintza, pc) zerrenda. OpenQASM 2.0-k ez du jauzirik ez begiztarik, baina
// bi egitura ohiko zehazki adieraz daitezke:
//   - kopuru literaleko begiztak (LOOPSTART 3 ... LOOPEND): gorputza n aldiz errepikatuta
//   - aurrerako jauzi bat bit bakar baten arabera, agindu zuzenen bloke baten gainetik (JUMPIFZERO c1 x / X q2 /
//     x:): blokeko agindu bakoitza `if (c1==1)`-ekin
// beste kontrol-fluxua (atzerako jauziak, erregistroak, deiak) ez da hemen ukitzen eta esportatzean errorea ematen du
type Cond = Option<(u32, u8)>;

const UNROLL_MAX: usize = 100_000;

// barruan kontrol-fluxurik ez duen eta kanpotik erdira jauzirik jasotzen ez duen tartea
fn straight_block(p: &Program, from: usize, to: usize) -> bool {
    let inner = |t: usize| t > from && t < to;
    p.ops[from..to].iter().all(|op| {
        !matches!(
            op,
            Op::Jump { .. }
                | Op::JumpIf { .. }
                | Op::Branch { .. }
                | Op::BranchV { .. }
                | Op::Call { .. }
                | Op::Return
                | Op::LoopStart { .. }
                | Op::LoopEnd { .. }
                | Op::Halt
        )
    }) && !p.ops.iter().any(|op| match op {
        Op::Jump { to: t }
        | Op::JumpIf { to: t, .. }
        | Op::Branch { to: t, .. }
        | Op::BranchV { to: t, .. }
        | Op::Call { to: t } => inner(*t),
        Op::LoopStart { end, .. } => inner(*end),
        Op::LoopEnd { start } => inner(*start),
        _ => false,
    })
}

// begizta baten gorputzera (`start` LOOPSTART da, `le` LOOPEND) kanpotik ez da jauzirik sartzen eta handik ez
// da ateratzen
fn closed_loop(p: &Program, start: usize, le: usize) -> bool {
    let target = |op: &Op| op.jump_target();
    let jumps_in = p.ops.iter().enumerate().any(|(pc, op)| {
        (pc < start || pc > le) && target(op).is_some_and(|t| t > start && t <= le)
    });
    let jumps_out = p.ops[start + 1..le].iter().any(|op| {
        matches!(op, Op::Halt | Op::Return) || target(op).is_some_and(|t| t <= start || t > le)
    });
    !jumps_in && !jumps_out
}

fn flatten(
    p: &Program,
    from: usize,
    to: usize,
    cond: Cond,
    out: &mut Vec<(Cond, usize)>,
) -> Result<(), usize> {
    let mut pc = from;
    while pc < to {
        match &p.ops[pc] {
            // `end` LOOPEND-aren ondorengoa da; LOOPEND bera end - 1 da
            Op::LoopStart {
                count: Count::Lit(n),
                end,
            } if closed_loop(p, pc, end - 1) => {
                for _ in 0..*n {
                    flatten(p, pc + 1, end - 1, cond, out)?;
                    if out.len() > UNROLL_MAX {
                        return Err(pc);
                    }
                }
                pc = *end;
            }
            Op::JumpIf {
                src: Source::Bit(k),
                value,
                to: t,
            } if cond.is_none() && *t > pc && *t <= to && straight_block(p, pc + 1, *t) => {
                // bita `value` denean salto egiten da, beraz blokea bitaren beste balioarekin exekutatzen da
                let run = if *value == 0.0 { 1 } else { 0 };
                flatten(p, pc + 1, *t, Some((*k, run)), out)?;
                pc = *t;
            }
            _ => {
                out.push((cond, pc));
                pc += 1;
            }
        }
    }
    Ok(())
}

// OpenQASM 3-rako programa egituratua: eragiketak, eta blokeen hasierak eta amaierak (`for`, `if`, `while`)
enum Item {
    Op(usize),
    Open(String),
    Close,
}

// atzerako jauzi bat bit baten arabera (`L: ... JUMPIFZERO c0 L`): L-tik hasten den do-while bat. (jauziaren pc,
// bita, jauzia egiten duen balioa)
fn back_jump(p: &Program, from: usize, to: usize) -> Option<(usize, u32, f64)> {
    (from + 1..to).find_map(|j| match &p.ops[j] {
        Op::JumpIf {
            src: Source::Bit(k),
            value,
            to: t,
        } if *t == from && closed_region(p, from, j) => Some((j, *k, *value)),
        _ => None,
    })
}

// [from, j] tartera kanpotik ez da jauzirik sartzen (from-era izan ezik) eta handik ez da ateratzen (j-ko
// atzerako jauzia izan ezik)
fn closed_region(p: &Program, from: usize, j: usize) -> bool {
    let inside = |t: usize| t > from && t <= j;
    let jumps_in = p
        .ops
        .iter()
        .enumerate()
        .any(|(pc, op)| (pc < from || pc > j) && op.jump_target().is_some_and(inside))
        || p.ops
            .iter()
            .enumerate()
            .any(|(pc, op)| pc != j && (pc < from || pc > j) && op.jump_target() == Some(from));
    let jumps_out = p.ops[from..j].iter().any(|op| {
        matches!(op, Op::Halt | Op::Return) || op.jump_target().is_some_and(|t| t < from || t > j)
    });
    !jumps_in && !jumps_out
}

// [a, b) blokea itxita: kanpotik ez da haren barrura jauzirik egiten (a barne), eta barrutik ez da kanpora
// ateratzen, b-ra izan ezik (blokearen amaierara: barruko if batek bertara salta dezake; flatten3-k egituratzen
// ez badu, jauzia esportatu ezin den eragiketa gisa geratzen da, inoiz ez programa desberdin gisa)
fn closed_block(p: &Program, a: usize, b: usize) -> bool {
    closed_block_but(p, a, b, None)
}

// closed_block bezala, baina `entry` pc-ko jauziak a-ra sar daiteke (if/else-ren else adarra)
fn closed_block_but(p: &Program, a: usize, b: usize, entry: Option<usize>) -> bool {
    let inside = |t: usize| t >= a && t < b;
    let jumps_in = p.ops.iter().enumerate().any(|(pc, op)| {
        !inside(pc)
            && op
                .jump_target()
                .is_some_and(|t| inside(t) && !(Some(pc) == entry && t == a))
    });
    let jumps_out = p.ops[a..b].iter().any(|op| {
        matches!(op, Op::Halt | Op::Return)
            || op.jump_target().is_some_and(|t| !inside(t) && t != b)
    });
    !jumps_in && !jumps_out
}

// x-ra [lo, x) tartetik bakarrik salta daiteke (blokearen amaierako JUMP-a: kanpotik ez da bertara sartzen)
fn entered_only_from(p: &Program, x: usize, lo: usize) -> bool {
    !p.ops
        .iter()
        .enumerate()
        .any(|(pc, op)| !(lo..x).contains(&pc) && op.jump_target() == Some(x))
}

// if/else-ren amaiera: then adarraren azken JUMP-aren helburua
fn else_end(p: &Program, t: usize) -> usize {
    match p.ops[t - 1] {
        Op::Jump { to } => to,
        _ => t,
    }
}

fn flatten3(
    p: &Program,
    from: usize,
    to: usize,
    depth: usize,
    out: &mut Vec<Item>,
) -> Result<(), usize> {
    let mut pc = from;
    while pc < to {
        if let Some((j, k, value)) = back_jump(p, pc, to) {
            // do-while: gorputza behin, eta gero bita jauziaren balioa den bitartean
            let body_start = out.len();
            flatten3(p, pc, j, depth + 1, out)?;
            let body: Vec<Item> = out[body_start..]
                .iter()
                .map(|i| match i {
                    Item::Op(x) => Item::Op(*x),
                    Item::Open(s) => Item::Open(s.clone()),
                    Item::Close => Item::Close,
                })
                .collect();
            out.push(Item::Open(format!(
                "while (c[{}] == {}) {{",
                k,
                value != 0.0
            )));
            out.extend(body);
            out.push(Item::Close);
            if out.len() > UNROLL_MAX {
                return Err(pc);
            }
            pc = j + 1;
            continue;
        }
        match &p.ops[pc] {
            // `goia: JUMPIF c v amaiera; A; JUMP goia; amaiera:`: while (c != v) { A }
            Op::JumpIf {
                src: Source::Bit(k),
                value,
                to: t,
            } if *t > pc + 1
                && *t <= to
                && matches!(p.ops[*t - 1], Op::Jump { to: d } if d == pc)
                && closed_block(p, pc + 1, *t - 1)
                && entered_only_from(p, *t - 1, pc + 1)
                && !p
                    .ops
                    .iter()
                    .enumerate()
                    .any(|(j, op)| j != *t - 1 && op.jump_target() == Some(pc)) =>
            {
                out.push(Item::Open(format!(
                    "while (c[{}] == {}) {{",
                    k,
                    *value == 0.0
                )));
                flatten3(p, pc + 1, *t - 1, depth + 1, out)?;
                out.push(Item::Close);
                pc = *t;
            }
            Op::LoopStart {
                count: Count::Lit(n),
                end,
            } if closed_loop(p, pc, end - 1) => {
                if *n > 0 {
                    out.push(Item::Open(format!(
                        "for uint i{} in [0:{}] {{",
                        depth,
                        n - 1
                    )));
                    flatten3(p, pc + 1, end - 1, depth + 1, out)?;
                    out.push(Item::Close);
                }
                pc = *end;
            }
            Op::JumpIf {
                src: Source::Bit(k),
                value,
                to: t,
            } if *t > pc + 1
                && *t <= to
                && matches!(p.ops[*t - 1], Op::Jump { to: d } if d > *t && d <= to)
                && closed_block(p, pc + 1, *t - 1)
                && entered_only_from(p, *t - 1, pc + 1)
                && closed_block_but(p, *t, else_end(p, *t), Some(pc)) =>
            {
                // if/else: `JUMPIF c v else; A; JUMP done; else: B; done:`
                let d = else_end(p, *t);
                out.push(Item::Open(format!("if (c[{}] == {}) {{", k, *value == 0.0)));
                flatten3(p, pc + 1, *t - 1, depth + 1, out)?;
                out.push(Item::Close);
                out.push(Item::Open("else {".to_string()));
                flatten3(p, *t, d, depth + 1, out)?;
                out.push(Item::Close);
                pc = d;
            }
            Op::JumpIf {
                src: Source::Bit(k),
                value,
                to: t,
            } if *t > pc && *t <= to && closed_block(p, pc + 1, *t) => {
                // bita `value` denean salto egiten da, beraz blokea bitaren beste balioarekin exekutatzen da
                out.push(Item::Open(format!("if (c[{}] == {}) {{", k, *value == 0.0)));
                flatten3(p, pc + 1, *t, depth + 1, out)?;
                out.push(Item::Close);
                pc = *t;
            }
            _ => {
                out.push(Item::Op(pc));
                pc += 1;
            }
        }
    }
    Ok(())
}

// OpenQASM 3.0 (stdgates.inc): OpenQASM 2.0-k adierazten duena, gehi `for` begiztak (zabaldu gabe), `if`
// blokeak eta `while` begiztak bit neurtu baten arabera (arrakasta arte errepikatu). QOA-ren erregistroek ez dute
// baliokiderik qiskit-ek eta IBM-ren gailuek onartzen dutenean, beraz haiekin kalkulatzen duten programak ez
pub fn from_program_v3(p: &Program) -> Result<String, Vec<ExportError>> {
    from_program_in(p, true)
}

pub fn from_program(p: &Program) -> Result<String, Vec<ExportError>> {
    from_program_in(p, false)
}

// qelib1.inc-ko izenak OpenQASM 3-ren stdgates.inc-n: aldatzaileak (inv @, ctrl @) edo baliokideak
fn v3_line(l: &str) -> String {
    let swaps: [(&str, &str); 6] = [
        ("sxdg ", "inv @ sx "),
        ("csx ", "ctrl @ sx "),
        ("cu1(", "cp("),
        ("c3x ", "ctrl(3) @ x "),
        ("c4x ", "ctrl(4) @ x "),
        ("cu3(", "cu("),
    ];
    for (a, b) in swaps {
        if let Some(rest) = l.strip_prefix(a) {
            if a == "cu3(" {
                // cu3(θ,φ,λ) = cu(θ,φ,λ,0)
                if let Some(close) = rest.find(')') {
                    return format!("cu({},0){}", &rest[..close], &rest[close + 1..]);
                }
            }
            return format!("{}{}", b, rest);
        }
    }
    l.to_string()
}

fn from_program_in(p: &Program, v3: bool) -> Result<String, Vec<ExportError>> {
    let version = if v3 { "OpenQASM 3" } else { "OpenQASM 2.0" };
    let mut errors = Vec::new();
    let mut body = Vec::new();
    let mut uses_bits = false;
    let q = |i: u32| format!("q[{}]", i);
    let fmt = |x: f64| format!("{}", x);
    if p.n_modes > 0 {
        errors.push(ExportError {
            line: 0,
            message: t!(
                "the program uses {} optical mode(s); OpenQASM has no optical modes",
                "programak {} modu optiko erabiltzen ditu; OpenQASM-ek ez du modu optikorik",
                p.n_modes
            ),
        });
    }
    let mut flat = Vec::new();
    let mut items = Vec::new();
    let flattened = if v3 {
        flatten3(p, 0, p.ops.len(), 0, &mut items)
    } else {
        flatten(p, 0, p.ops.len(), None, &mut flat)
    };
    if v3 {
        // eragiketak lerroak bihurtzeko begiztak (cond, pc) behar du; blokeak beren lekuan
        flat = items
            .iter()
            .filter_map(|i| match i {
                Item::Op(pc) => Some((None, *pc)),
                _ => None,
            })
            .collect();
    }
    if let Err(pc) = flattened {
        errors.push(ExportError {
            line: p.line_of(pc),
            message: if v3 {
                // do-while-ak gorputza bi aldiz idazten du (behin, eta gero `while`-aren barruan)
                t!(
                    "writing this repeat-until loop as a while loop would make more than {} lines",
                    "errepikatu-arte begizta hau while begizta gisa idazteak {} lerro baino gehiago sortuko lituzke",
                    UNROLL_MAX
                )
            } else {
                t!(
                    "unrolling this loop would make more than {} instructions",
                    "begizta hau zabaltzeak {} agindu baino gehiago sortuko lituzke",
                    UNROLL_MAX
                )
            },
        });
        return Err(errors);
    }
    // baldintzak badaude, bit bakoitza bere erregistroan (OpenQASM 2.0-k erregistro osoa konparatzen du)
    let split_bits = !v3 && flat.iter().any(|(c, _)| c.is_some());
    // v3: eragiketa bakoitzaren lerroak, gero blokeekin batera ordenatzeko
    let mut op_lines: HashMap<usize, Vec<String>> = HashMap::new();
    let bit = |k: u32| {
        if split_bits {
            format!("c{}[0]", k)
        } else {
            format!("c[{}]", k)
        }
    };
    for &(cond, pc) in &flat {
        if v3 && op_lines.contains_key(&pc) {
            continue;
        }
        let op = &p.ops[pc];
        let line = p.line_of(pc);
        let before = body.len();
        let mut fail = |m: String| errors.push(ExportError { line, message: m });
        match op {
            Op::Gate { q: t, gate } => body.push(gate_line(gate, &q(*t))),
            Op::Controlled {
                controls,
                q: t,
                gate,
            } => match (controls.len(), gate) {
                (1, Gate::X) => body.push(format!("cx {},{};", q(controls[0]), q(*t))),
                (1, Gate::Y) => body.push(format!("cy {},{};", q(controls[0]), q(*t))),
                (1, Gate::H) => body.push(format!("ch {},{};", q(controls[0]), q(*t))),
                (1, Gate::SX) => body.push(format!("csx {},{};", q(controls[0]), q(*t))),
                (1, Gate::RX(a)) => {
                    body.push(format!("crx({}) {},{};", fmt(*a), q(controls[0]), q(*t)))
                }
                (1, Gate::RY(a)) => {
                    body.push(format!("cry({}) {},{};", fmt(*a), q(controls[0]), q(*t)))
                }
                (1, Gate::RZ(a)) => {
                    body.push(format!("crz({}) {},{};", fmt(*a), q(controls[0]), q(*t)))
                }
                (1, Gate::U(a, b, c)) => body.push(format!(
                    "cu3({},{},{}) {},{};",
                    fmt(*a),
                    fmt(*b),
                    fmt(*c),
                    q(controls[0]),
                    q(*t)
                )),
                (2, Gate::X) => body.push(format!(
                    "ccx {},{},{};",
                    q(controls[0]),
                    q(controls[1]),
                    q(*t)
                )),
                (3, Gate::X) => body.push(format!(
                    "c3x {},{},{},{};",
                    q(controls[0]),
                    q(controls[1]),
                    q(controls[2]),
                    q(*t)
                )),
                (4, Gate::X) => body.push(format!(
                    "c4x {},{},{},{},{};",
                    q(controls[0]),
                    q(controls[1]),
                    q(controls[2]),
                    q(controls[3]),
                    q(*t)
                )),
                _ => fail(t!(
                    "a {}-controlled {} has no qelib1.inc gate",
                    "{} kontroldun {} ateak ez du baliokiderik qelib1.inc-n",
                    controls.len(),
                    gate.qasm_name()
                )),
            },
            Op::PhaseAll { qubits, theta } => {
                let is_pi = (*theta - std::f64::consts::PI).abs() < 1e-15;
                match qubits.len() {
                    1 => body.push(format!("u1({}) {};", fmt(*theta), q(qubits[0]))),
                    2 if is_pi => body.push(format!("cz {},{};", q(qubits[0]), q(qubits[1]))),
                    2 => body.push(format!(
                        "cu1({}) {},{};",
                        fmt(*theta),
                        q(qubits[0]),
                        q(qubits[1])
                    )),
                    3..=5 if is_pi => {
                        // -1 fasea qubit guztiak 1 direnean = H azkenean; x kontrol anitzekin; H
                        let last = *qubits.last().unwrap();
                        let ctrls: Vec<String> =
                            qubits[..qubits.len() - 1].iter().map(|c| q(*c)).collect();
                        let gate = match qubits.len() {
                            3 => "ccx",
                            4 => "c3x",
                            _ => "c4x",
                        };
                        body.push(format!("h {};", q(last)));
                        body.push(format!("{} {},{};", gate, ctrls.join(","), q(last)));
                        body.push(format!("h {};", q(last)));
                    }
                    n => fail(t!(
                        "a phase on {} qubits at once has no qelib1.inc gate",
                        "{} qubiteko aldibereko faseak ez du baliokiderik qelib1.inc-n",
                        n
                    )),
                }
            }
            Op::Swap { a, b, controls } => match controls.len() {
                0 => body.push(format!("swap {},{};", q(*a), q(*b))),
                1 => body.push(format!("cswap {},{},{};", q(controls[0]), q(*a), q(*b))),
                _ => fail(t!(
                    "a multi-controlled swap has no qelib1.inc gate",
                    "kontrol anitzeko trukaketak ez du baliokiderik qelib1.inc-n"
                )),
            },
            Op::Bell { a, b } => {
                body.push(format!("h {};", q(*a)));
                body.push(format!("cx {},{};", q(*a), q(*b)));
            }
            Op::Ghz { qubits } => {
                body.push(format!("h {};", q(qubits[0])));
                for t in &qubits[1..] {
                    body.push(format!("cx {},{};", q(qubits[0]), q(*t)));
                }
            }
            Op::Cluster { qubits } => {
                for t in qubits {
                    body.push(format!("h {};", q(*t)));
                }
                for w in qubits.windows(2) {
                    body.push(format!("cz {},{};", q(w[0]), q(w[1])));
                }
            }
            Op::Measure { q: t, basis, dest } => {
                let b = match dest {
                    Dest::Own => *t,
                    Dest::Bit(k) => *k,
                    Dest::Reg(_) => {
                        fail(t!(
                            "a measurement into a register cannot be expressed in {}",
                            "erregistro baterako neurketa ezin da {}-n adierazi",
                            version
                        ));
                        continue;
                    }
                };
                uses_bits = true;
                match basis {
                    Basis::X => body.push(format!("h {};", q(*t))),
                    Basis::Y => {
                        body.push(format!("sdg {};", q(*t)));
                        body.push(format!("h {};", q(*t)));
                    }
                    Basis::Z => {}
                }
                body.push(format!("measure {} -> {};", q(*t), bit(b)));
                match basis {
                    Basis::X => body.push(format!("h {};", q(*t))),
                    Basis::Y => {
                        body.push(format!("h {};", q(*t)));
                        body.push(format!("s {};", q(*t)));
                    }
                    Basis::Z => {}
                }
            }
            Op::MeasureAll => {
                uses_bits = true;
                for t in 0..p.n_qubits {
                    body.push(format!("measure {} -> {};", q(t), bit(t)));
                }
            }
            Op::Reset { q: t } => body.push(format!("reset {};", q(*t))),
            Op::ResetAll => {
                for t in 0..p.n_qubits {
                    body.push(format!("reset {};", q(t)));
                }
            }
            Op::ToBasis { q: t, basis } => match basis {
                Basis::X => body.push(format!("h {};", q(*t))),
                Basis::Y => {
                    body.push(format!("sdg {};", q(*t)));
                    body.push(format!("h {};", q(*t)));
                }
                Basis::Z => {}
            },
            Op::Barrier => body.push("barrier q;".to_string()),
            Op::Halt if pc + 1 == p.ops.len() => {}
            // irteera bakarrik egiten dutenak: iruzkin gisa uzten dira
            Op::Print { .. }
            | Op::Printf { .. }
            | Op::Log { .. }
            | Op::DumpState
            | Op::DumpRegs
            | Op::Expect { dest: None, .. }
            | Op::BellCheck { dest: None, .. }
            | Op::MeanPhoton { dest: None, .. } => {
                body.push(t!(
                    "// dropped: output-only instruction on line {}",
                    "// kendua: {}. lerroko irteera-agindua",
                    line
                ));
            }
            other => fail(t!(
                "{} cannot be expressed in {}",
                "{} ezin da {}-n adierazi",
                describe(other),
                version
            )),
        }
        if v3 {
            let lines: Vec<String> = body.drain(before..).map(|l| v3_line(&l)).collect();
            op_lines.insert(pc, lines);
            continue;
        }
        // baldintzapeko blokea: lerro bakoitza `if`-ekin (iruzkinak izan ezik)
        if let Some((k, v)) = cond {
            for l in body[before..].iter_mut().filter(|l| !l.starts_with("//")) {
                *l = format!("if(c{}=={}) {}", k, v, l);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    let mut out = String::new();
    out.push_str(&t!(
        "// exported from {} by qoa {}\n",
        "// {} fitxategitik esportatua, qoa {}\n",
        p.name,
        env!("CARGO_PKG_VERSION")
    ));
    if v3 {
        out.push_str("OPENQASM 3.0;\ninclude \"stdgates.inc\";\n");
        out.push_str(&format!("qubit[{}] q;\n", p.n_qubits.max(1)));
        out.push_str(&format!("bit[{}] c;\n", p.n_bits.max(1)));
        let mut depth = 0usize;
        for i in &items {
            match i {
                Item::Op(pc) => {
                    for l in op_lines.get(pc).into_iter().flatten() {
                        out.push_str(&"  ".repeat(depth));
                        out.push_str(l);
                        out.push('\n');
                    }
                }
                Item::Open(s) => {
                    out.push_str(&"  ".repeat(depth));
                    out.push_str(s);
                    out.push('\n');
                    depth += 1;
                }
                Item::Close => {
                    depth = depth.saturating_sub(1);
                    out.push_str(&"  ".repeat(depth));
                    out.push_str("}\n");
                }
            }
        }
        return Ok(out);
    }
    out.push_str("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n");
    out.push_str(&format!("qreg q[{}];\n", p.n_qubits.max(1)));
    if split_bits {
        for k in 0..p.n_bits.max(1) {
            out.push_str(&format!("creg c{}[1];\n", k));
        }
    } else if uses_bits {
        out.push_str(&format!("creg c[{}];\n", p.n_bits.max(1)));
    }
    for l in body {
        out.push_str(&l);
        out.push('\n');
    }
    Ok(out)
}

fn gate_line(g: &Gate, target: &str) -> String {
    let params = g.params();
    let name = match g {
        // jatorrizko qelib1.inc-n p ez dago, u1 bai
        Gate::P(_) => "u1",
        _ => g.qasm_name(),
    };
    if params.is_empty() {
        format!("{} {};", name, target)
    } else {
        let ps: Vec<String> = params.iter().map(|x| format!("{}", x)).collect();
        format!("{}({}) {};", name, ps.join(","), target)
    }
}

// errore-mezuetarako izen irakurgarria
pub fn describe(op: &Op) -> String {
    let s = format!("{:?}", op);
    let name = s
        .split([' ', '{', '('])
        .next()
        .unwrap_or("operation")
        .to_string();
    match op {
        Op::Emit { .. } | Op::Dispersive { .. } => t!(
            "{} (qubit-mode coupling)",
            "{} (qubit-modu akoplamendua)",
            name
        ),
        _ if !op.optical_modes().is_empty() => t!(
            "{} (optical mode operation)",
            "{} (modu optikoko eragiketa)",
            name
        ),
        Op::PhotonCount { .. }
        | Op::PhotonDetect { .. }
        | Op::Parity { .. }
        | Op::Coincidence { .. } => {
            t!("{} (photon detection)", "{} (fotoi-detekzioa)", name)
        }
        Op::Jump { .. }
        | Op::JumpIf { .. }
        | Op::Branch { .. }
        | Op::BranchV { .. }
        | Op::Call { .. }
        | Op::Return
        | Op::LoopStart { .. }
        | Op::LoopEnd { .. }
        | Op::Halt => t!("{} (control flow)", "{} (kontrol-fluxua)", name),
        _ => t!(
            "{} (classical register operation)",
            "{} (erregistro klasikoko eragiketa)",
            name
        ),
    }
}
