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
use crate::program::{Basis, Dest, Gate, Op, Program};
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
}

// openqasm testua qoa testu bihurtzen du
pub fn to_qoa(src: &str, name: &str) -> Result<String, QasmError> {
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
    };
    for (stmt, line) in split_statements(&clean)? {
        imp.statement(&stmt, line)?;
    }
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
    fn statement(&mut self, s: &str, line: u32) -> Result<(), QasmError> {
        if s.is_empty() {
            return Ok(());
        }
        let first = s
            .split(|c: char| c.is_whitespace() || c == '(')
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        match first.as_str() {
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
            "def" | "defcal" | "cal" | "for" | "while" | "const" | "input" | "output" | "let"
            | "box" | "switch" | "extern" | "gphase" => {
                return Err(qerr(
                    line,
                    t!(
                        "'{}' is an OpenQASM 3 feature the importer does not support (OpenQASM 2.0 is fully supported)",
                        "'{}' OpenQASM 3-ren ezaugarri bat da, eta inportatzaileak ez du onartzen (OpenQASM 2.0 osorik onartzen da)",
                        first
                    ),
                ));
            }
            _ => {}
        }
        if s.contains('@') {
            return Err(qerr(
                line,
                t!(
                    "gate modifiers (ctrl @, inv @, pow @) are OpenQASM 3 features the importer does not support",
                    "ate-modifikatzaileak (ctrl @, inv @, pow @) OpenQASM 3-ren ezaugarriak dira, eta inportatzaileak ez ditu onartzen"
                ),
            ));
        }
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
            self.apply(&gname, &params, &qs, line, 0)?;
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
        let close = s.rfind('}').ok_or_else(|| {
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
                let pc = header.find(')').ok_or_else(|| {
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
        self.lines.push(format!("PACKBITS R31 {}", bits));
        self.lines.push(format!("REGSET R30 {}", value));
        self.lines.push(format!("IFNE R31 R30 {}", label));
        self.statement(rest, line)?;
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
                self.lines.push(format!("MEASURE q{} -> c{}", q, c))
            }
            (Target::All(qo, qn), Target::All(co, cn)) if qn == cn => {
                for k in 0..qn {
                    self.lines
                        .push(format!("MEASURE q{} -> c{}", qo + k, co + k));
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
        _ => unreachable!("native_arity and native_lines disagree on {}", name),
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
pub fn from_program(p: &Program) -> Result<String, Vec<ExportError>> {
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
    for (pc, op) in p.ops.iter().enumerate() {
        let line = p.line_of(pc);
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
                let bit = match dest {
                    Dest::Own => *t,
                    Dest::Bit(k) => *k,
                    Dest::Reg(_) => {
                        fail(t!(
                            "a measurement into a register cannot be expressed in OpenQASM 2.0",
                            "erregistro baterako neurketa ezin da OpenQASM 2.0-n adierazi"
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
                body.push(format!("measure {} -> c[{}];", q(*t), bit));
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
                    body.push(format!("measure {} -> c[{}];", q(t), t));
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
                "{} cannot be expressed in OpenQASM 2.0",
                "{} ezin da OpenQASM 2.0-n adierazi",
                describe(other)
            )),
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
    out.push_str("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n");
    out.push_str(&format!("qreg q[{}];\n", p.n_qubits.max(1)));
    if uses_bits {
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
