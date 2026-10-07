// fitxategi honek .qoa testua programa bihurtzen du (mihiztatzailea).
// lerroz lerro irakurtzen du, agindu bakoitza isa.rs taulan bilatzen du, eragigaiak egiaztatzen
// ditu eta akats bakoitzak lerroa eta zutabea adierazten ditu.

use crate::expr;
use crate::isa::{self, OperandSpec, Repeat, Spec};
use crate::program::{
    Basis, BinOp, Cmp, Count, DEFAULT_CUTOFF, Dest, Gate, NUM_REGS, Op, Program, Source, UnOp,
};
use crate::{t, ts};
use std::collections::HashMap;

// errore bat, iturburuko kokapen zehatzarekin
#[derive(Clone, Debug, PartialEq)]
pub struct Diagnostic {
    pub line: u32,
    pub col: u32,
    pub len: u32,
    pub message: String,
    pub help: Option<String>,
}

impl Diagnostic {
    fn at(tok: &Token, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            line: tok.line,
            col: tok.col,
            len: tok.text.chars().count().max(1) as u32,
            message: message.into(),
            help: None,
        }
    }

    fn help(mut self, help: impl Into<String>) -> Diagnostic {
        self.help = Some(help.into());
        self
    }
}

// rustc-en antzeko formatuan inprimatzen ditu erroreak:
//   error: ...
//     --> fitxategia:lerroa:zutabea
//      |
//    3 | CNOT q0 q0
//      |         ^^
pub fn render(diags: &[Diagnostic], source: &str, name: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = String::new();
    for d in diags {
        out.push_str(&format!("{}: {}\n", ts!("error", "errorea"), d.message));
        out.push_str(&format!("  --> {}:{}:{}\n", name, d.line, d.col));
        if let Some(text) = lines.get((d.line as usize).wrapping_sub(1)) {
            let num = d.line.to_string();
            let pad = " ".repeat(num.len());
            out.push_str(&format!(" {} |\n", pad));
            out.push_str(&format!(" {} | {}\n", num, text));
            let caret_pad: String = text
                .chars()
                .take((d.col as usize).saturating_sub(1))
                .map(|c| if c == '\t' { '\t' } else { ' ' })
                .collect();
            out.push_str(&format!(
                " {} | {}{}\n",
                pad,
                caret_pad,
                "^".repeat(d.len as usize)
            ));
        }
        if let Some(h) = &d.help {
            out.push_str(&format!("  = {}: {}\n", ts!("help", "laguntza"), h));
        }
        out.push('\n');
    }
    out
}

// lerro bateko token bat (testua eta haren kokapena)
#[derive(Clone, Debug)]
struct Token {
    text: String,
    line: u32,
    col: u32,
}

// iruzkina kendu (; // edo #), testu-kateen barruan izan ezik
fn strip_comment(line: &str) -> &str {
    let mut in_str = false;
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'"' {
            in_str = !in_str;
        } else if !in_str {
            if c == b';' || c == b'#' {
                return &line[..i];
            }
            if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                return &line[..i];
            }
        }
        i += 1;
    }
    line
}

// lerroa tokenetan banatu: hutsuneek eta komek bereizten dituzte, "testuak" eta (parentesiak)
// osorik geratzen dira, eta -> token bereizia da
fn tokenize(line: &str, line_no: u32) -> Result<Vec<Token>, Diagnostic> {
    let chars: Vec<char> = line.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() || c == ',' {
            i += 1;
            continue;
        }
        let start = i;
        if c == '"' {
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\\' {
                    i += 1;
                }
                i += 1;
            }
            if i >= chars.len() {
                return Err(Diagnostic {
                    line: line_no,
                    col: start as u32 + 1,
                    len: 1,
                    message: t!(
                        "text is missing its closing \"",
                        "testuari amaierako \" falta zaio"
                    ),
                    help: None,
                });
            }
            i += 1;
        } else if c == '-' && i + 1 < chars.len() && chars[i + 1] == '>' {
            i += 2;
        } else {
            let mut depth = 0i32;
            while i < chars.len() {
                let ch = chars[i];
                if ch == '(' {
                    depth += 1;
                } else if ch == ')' {
                    depth -= 1;
                } else if depth == 0
                    && (ch.is_whitespace()
                        || ch == ','
                        || (ch == '-' && i + 1 < chars.len() && chars[i + 1] == '>'))
                {
                    break;
                }
                i += 1;
            }
        }
        toks.push(Token {
            text: chars[start..i].iter().collect(),
            line: line_no,
            col: start as u32 + 1,
        });
    }
    Ok(toks)
}

fn is_label_name(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' || c == '.' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

// eragigai bat, motaren arabera irakurrita
#[derive(Clone, Debug)]
enum Arg {
    Index(u32),
    Indices(Vec<u32>),
    Dest(Dest),
    Source(Source),
    Float(f64),
    Int(u64),
    Count(Count),
    Label(String),
    Text(String),
    Basis(Basis),
    Name(String),
    Missing,
}

// mihiztatzailearen egoera fitxategi oso batean zehar
struct Asm {
    diags: Vec<Diagnostic>,
    ops: Vec<Op>,
    lines: Vec<u32>,
    // jauzi-helburuak: (agindu-indizea, etiketa, tokena)
    fixups: Vec<(usize, String, Token)>,
    labels: HashMap<String, (usize, Token)>,
    loop_stack: Vec<(usize, Token)>,
    declared_qubits: Option<u32>,
    declared_modes: Option<(u32, u32, bool)>,
    max_qubit: Option<u32>,
    max_mode: Option<u32>,
    max_bit: Option<u32>,
    qinit_repeated: bool,
    qinit_token: Option<Token>,
    // qubit-muga gainditu da: errore bat bakarrik, kaskadarik gabe
    qubit_limit_reported: bool,
    // izendun erregistroak (QREG, CREG, OREG) eta mota bakoitzeko hurrengo elementu librea
    named: HashMap<String, Named>,
    named_next: HashMap<char, u32>,
    // lehen pasaldian ikusitako etiketa-izenak (minuskulaz): izen hutsak ez dira erregistro gisa irakurtzen
    label_names: std::collections::HashSet<String>,
}

// izendun erregistro bat: mota ('q', 'c', 'o'), lehen elementua, tamaina eta adierazpenaren tokena
struct Named {
    kind: char,
    start: u32,
    size: u32,
    tok: Token,
}

// testu osoa programa bihurtu; akatsik badago, guztiak itzultzen ditu
pub fn assemble(source: &str, name: &str) -> Result<Program, Vec<Diagnostic>> {
    let mut a = Asm {
        diags: Vec::new(),
        ops: Vec::new(),
        lines: Vec::new(),
        fixups: Vec::new(),
        labels: HashMap::new(),
        loop_stack: Vec::new(),
        declared_qubits: None,
        declared_modes: None,
        max_qubit: None,
        max_mode: None,
        max_bit: None,
        qinit_repeated: false,
        qinit_token: None,
        qubit_limit_reported: false,
        named: HashMap::new(),
        named_next: HashMap::new(),
        label_names: std::collections::HashSet::new(),
    };

    // lehen pasaldia: QINIT eta QMODE bilatu, gero indizeak berehala egiaztatu ahal izateko
    for (n, raw) in source.lines().enumerate() {
        let line_no = n as u32 + 1;
        let Ok(toks) = tokenize(strip_comment(raw), line_no) else {
            continue;
        };
        if let Some(name) = toks.first().and_then(|t| t.text.strip_suffix(':')) {
            a.label_names.insert(name.to_ascii_lowercase());
        }
        let toks = strip_label(&toks);
        let Some(first) = toks.first() else { continue };
        let Some(spec) = isa::lookup(&first.text) else {
            continue;
        };
        if spec.name == "QINIT" || spec.name == "QMODE" {
            a.declaration(spec, first, &toks[1..]);
        }
        if matches!(spec.name, "QREG" | "CREG" | "OREG") {
            a.register(spec, first, &toks[1..]);
        }
    }
    a.registers_fit();

    // bigarren pasaldia: agindu guztiak
    for (n, raw) in source.lines().enumerate() {
        let line_no = n as u32 + 1;
        let toks = match tokenize(strip_comment(raw), line_no) {
            Ok(t) => t,
            Err(d) => {
                a.diags.push(d);
                continue;
            }
        };
        if toks.is_empty() {
            continue;
        }
        let mut rest: &[Token] = &toks;
        // etiketa: "izena:" lerroaren hasieran (agindu bat jarrai dezake)
        if let Some(name) = rest[0].text.strip_suffix(':') {
            if is_label_name(name) {
                let key = name.to_ascii_lowercase();
                if let Some((_, prev)) = a.labels.get(&key) {
                    let prev_line = prev.line;
                    a.diags.push(
                        Diagnostic::at(
                            &rest[0],
                            t!(
                                "label '{}' is defined twice",
                                "'{}' etiketa bi aldiz definituta dago",
                                name
                            ),
                        )
                        .help(t!(
                            "it was first defined on line {}",
                            "lehen aldiz {}. lerroan definitu zen",
                            prev_line
                        )),
                    );
                } else {
                    a.labels.insert(key, (a.ops.len(), rest[0].clone()));
                }
                rest = &rest[1..];
            } else {
                a.diags.push(
                    Diagnostic::at(&rest[0], t!("'{}' is not a valid label name", "'{}' ez da etiketa-izen baliozkoa", name)).help(t!(
                        "labels start with a letter or _ and contain letters, digits, _ or .",
                        "etiketak letra batekin edo _-rekin hasten dira, eta letrak, digituak, _ edo . dituzte"
                    )),
                );
                continue;
            }
        }
        if rest.is_empty() {
            continue;
        }
        if let Some(renamed) = a.rename(rest) {
            a.line(&renamed);
        }
    }

    // erregistro-izen bat ezin da etiketa bat ere izan (`JUMP data` anbiguoa litzateke)
    for (name, r) in &a.named {
        if a.labels.contains_key(&name.to_ascii_lowercase()) {
            a.diags.push(Diagnostic::at(
                &r.tok,
                t!(
                    "'{}' is both a register and a label; rename one of them",
                    "'{}' erregistroa eta etiketa da aldi berean; aldatu baten izena",
                    name
                ),
            ));
        }
    }

    // itxi gabeko begiztak
    for (_, tok) in std::mem::take(&mut a.loop_stack) {
        a.diags.push(Diagnostic::at(
            &tok,
            t!(
                "LOOPSTART has no matching LOOPEND",
                "LOOPSTART-ek ez du LOOPEND parekorik"
            ),
        ));
    }

    // jauzi-helburuak ebatzi
    for (idx, label, tok) in std::mem::take(&mut a.fixups) {
        match a.labels.get(&label.to_ascii_lowercase()) {
            Some((target, _)) => set_target(&mut a.ops[idx], *target),
            None => a.diags.push(Diagnostic::at(
                &tok,
                t!(
                    "no label named '{}'",
                    "ez dago '{}' izeneko etiketarik",
                    label
                ),
            )),
        }
    }

    if !a.diags.is_empty() {
        a.diags.sort_by_key(|d| (d.line, d.col));
        // gehienez 30 errore; gainerakoak lerro batean laburbildu
        const SHOWN: usize = 30;
        if a.diags.len() > SHOWN {
            let rest = a.diags.len() - SHOWN;
            let last_line = a.diags[SHOWN].line;
            a.diags.truncate(SHOWN);
            a.diags.push(Diagnostic {
                line: last_line,
                col: 1,
                len: 0,
                message: t!(
                    "... and {} more error(s) not shown",
                    "... eta erakusten ez diren beste {} errore",
                    rest
                ),
                help: None,
            });
        }
        return Err(a.diags);
    }

    let n_qubits = a
        .declared_qubits
        .unwrap_or_else(|| a.max_qubit.map_or(0, |m| m + 1));
    let (n_modes, cutoff, cutoff_declared) = a
        .declared_modes
        .unwrap_or_else(|| (a.max_mode.map_or(0, |m| m + 1), DEFAULT_CUTOFF, false));
    let n_bits = n_qubits.max(a.max_bit.map_or(0, |m| m + 1));
    Ok(Program {
        name: name.to_string(),
        n_qubits,
        n_modes,
        cutoff,
        cutoff_declared,
        n_bits,
        ops: a.ops,
        lines: a.lines,
    })
}

// lerroaren hasierako etiketa saltatu (lehen pasaldirako)
fn strip_label(toks: &[Token]) -> Vec<Token> {
    match toks.first() {
        Some(t) if t.text.ends_with(':') => toks[1..].to_vec(),
        _ => toks.to_vec(),
    }
}

fn set_target(op: &mut Op, target: usize) {
    match op {
        Op::Jump { to } | Op::JumpIf { to, .. } | Op::Branch { to, .. } | Op::Call { to } => {
            *to = target
        }
        _ => unreachable!("jauziek bakarrik dituzte etiketa-zuzenketak"),
    }
}

impl Asm {
    fn error(&mut self, d: Diagnostic) {
        self.diags.push(d);
    }

    fn emit(&mut self, op: Op, line: u32) -> usize {
        self.ops.push(op);
        self.lines.push(line);
        self.ops.len() - 1
    }

    // QINIT eta QMODE (lehen pasaldian bakarrik)
    fn declaration(&mut self, spec: &Spec, head: &Token, args: &[Token]) {
        let parse_int = |t: &Token| -> Option<u32> { t.text.parse::<u32>().ok() };
        if spec.name == "QINIT" {
            if self.declared_qubits.is_some() || self.qinit_repeated {
                if !self.qinit_repeated {
                    self.error(Diagnostic::at(head, t!("QINIT appears more than once", "QINIT behin baino gehiagotan agertzen da")).help(t!(
                        "since QOA 0.4, QINIT takes the qubit count once (QINIT 3); it no longer allocates one qubit per line",
                        "QOA 0.4-tik aurrera, QINIT-ek qubit-kopurua hartzen du behin (QINIT 3); jada ez du lerro bakoitzeko qubit bat sortzen"
                    )));
                }
                // deklarazioa ez da fidagarria: kopurua erabileratik ondorioztatu, errore-kaskadarik gabe
                self.qinit_repeated = true;
                self.declared_qubits = None;
                return;
            }
            match args {
                [t] => match parse_int(t) {
                    Some(n) if n <= 64 => {
                        self.declared_qubits = Some(n);
                        self.qinit_token = Some(t.clone());
                    }
                    Some(_) => self.error(Diagnostic::at(
                        t,
                        t!(
                            "more than 64 qubits cannot be simulated as a state vector",
                            "64 qubit baino gehiago ezin dira egoera-bektore gisa simulatu"
                        ),
                    )),
                    None => self.error(Diagnostic::at(
                        t,
                        t!(
                            "expected a qubit count, found '{}'",
                            "qubit-kopuru bat espero zen, baina '{}' dago",
                            t.text
                        ),
                    )),
                },
                _ => self.error(
                    Diagnostic::at(
                        head,
                        t!(
                            "QINIT takes one number: the qubit count",
                            "QINIT-ek zenbaki bat hartzen du: qubit-kopurua"
                        ),
                    )
                    .help(t!("for example: QINIT 3", "adibidez: QINIT 3")),
                ),
            }
        } else {
            if self.declared_modes.is_some() {
                self.error(Diagnostic::at(
                    head,
                    t!(
                        "QMODE appears more than once",
                        "QMODE behin baino gehiagotan agertzen da"
                    ),
                ));
                return;
            }
            let usage = || {
                t!(
                    "QMODE takes the mode count and an optional cutoff",
                    "QMODE-k modu-kopurua eta aukerako muga bat hartzen ditu"
                )
            };
            let example = || t!("for example: QMODE 2 6", "adibidez: QMODE 2 6");
            match args {
                [m] | [m, _] => {
                    let modes = parse_int(m);
                    let cutoff = match args.get(1) {
                        Some(c) => parse_int(c).map(|v| (v, true)),
                        None => Some((DEFAULT_CUTOFF, false)),
                    };
                    match (modes, cutoff) {
                        (Some(m), Some((c, declared))) if c >= 2 => {
                            self.declared_modes = Some((m, c, declared))
                        }
                        (Some(_), Some(_)) => self.error(Diagnostic::at(
                            &args[1],
                            t!(
                                "the cutoff must be at least 2 (room for 0 and 1 photons)",
                                "mugak gutxienez 2 izan behar du (0 eta 1 fotoirentzako lekua)"
                            ),
                        )),
                        _ => self.error(Diagnostic::at(head, usage()).help(example())),
                    }
                }
                _ => self.error(Diagnostic::at(head, usage()).help(example())),
            }
        }
    }

    // QREG / CREG / OREG izena tamaina [lehena]
    fn register(&mut self, spec: &Spec, head: &Token, args: &[Token]) {
        let kind = match spec.name {
            "QREG" => 'q',
            "CREG" => 'c',
            _ => 'o',
        };
        let usage = || {
            t!(
                "{0} takes a name, a size and an optional first index, like {0} data 3",
                "{0}-k izen bat, tamaina bat eta aukerako lehen indize bat hartzen ditu, {0} data 3 bezala",
                spec.name
            )
        };
        let (name_tok, size_tok, first_tok) = match args {
            [n, s] => (n, s, None),
            [n, s, f] => (n, s, Some(f)),
            _ => {
                self.error(Diagnostic::at(head, usage()));
                return;
            }
        };
        let name = name_tok.text.as_str();
        let shaped = name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        let operand_like = name.len() > 1
            && name.starts_with(['q', 'Q', 'o', 'O', 'c', 'C', 'r', 'R'])
            && name[1..].chars().all(|c| c.is_ascii_digit());
        let reserved = ["pi", "tau", "e", "sqrt", "sin", "cos", "tan", "exp", "ln"]
            .contains(&name.to_ascii_lowercase().as_str());
        if !shaped || operand_like || reserved || isa::lookup(name).is_some() {
            self.error(Diagnostic::at(
                name_tok,
                t!(
                    "'{}' cannot name a register: use letters, digits and _, not an instruction, an operand like q3, or a constant like pi",
                    "'{}' ezin da erregistro-izena izan: erabili letrak, digituak eta _, ez agindu bat, q3 bezalako eragigai bat edo pi bezalako konstante bat",
                    name
                ),
            ));
            return;
        }
        if self.named.contains_key(name) {
            self.error(Diagnostic::at(
                name_tok,
                t!(
                    "register '{}' is declared twice",
                    "'{}' erregistroa bi aldiz adierazita dago",
                    name
                ),
            ));
            return;
        }
        let size = parse_int(&size_tok.text).filter(|n| (1..=4096).contains(n));
        let Some(size) = size.map(|n| n as u32) else {
            self.error(Diagnostic::at(
                size_tok,
                t!(
                    "expected a size of at least 1, found '{}'",
                    "gutxienez 1eko tamaina espero zen, baina '{}' dago",
                    size_tok.text
                ),
            ));
            return;
        };
        let next = *self.named_next.get(&kind).unwrap_or(&0);
        let start = match first_tok {
            None => next,
            Some(f) => match parse_int(&f.text).filter(|&n| n < 4096) {
                Some(n) => n as u32,
                None => {
                    self.error(Diagnostic::at(
                        f,
                        t!(
                            "expected a first index, found '{}'",
                            "lehen indize bat espero zen, baina '{}' dago",
                            f.text
                        ),
                    ));
                    return;
                }
            },
        };
        if let Some((other, r)) = self
            .named
            .iter()
            .find(|(_, r)| r.kind == kind && start < r.start + r.size && r.start < start + size)
        {
            let msg = t!(
                "{} ({}{}..{}{}) overlaps {} ({}{}..{}{})",
                "{} ({}{}..{}{}) {}-rekin gainjartzen da ({}{}..{}{})",
                name,
                kind,
                start,
                kind,
                start + size - 1,
                other,
                kind,
                r.start,
                kind,
                r.start + r.size - 1
            );
            self.error(Diagnostic::at(name_tok, msg));
            return;
        }
        self.named_next.insert(kind, next.max(start + size));
        let last = start + size - 1;
        let max = match kind {
            'q' => &mut self.max_qubit,
            'c' => &mut self.max_bit,
            _ => &mut self.max_mode,
        };
        *max = Some(max.map_or(last, |m| m.max(last)));
        self.named.insert(
            name.to_string(),
            Named {
                kind,
                start,
                size,
                tok: name_tok.clone(),
            },
        );
    }

    // izendun erregistroak QINIT eta QMODE-k adierazitakoaren barruan
    fn registers_fit(&mut self) {
        let mut problems = Vec::new();
        for (name, r) in &self.named {
            let limit = match r.kind {
                'q' => self.declared_qubits,
                'o' => self.declared_modes.map(|m| m.0),
                _ => None,
            };
            if let Some(n) = limit
                && r.start + r.size > n
            {
                let what = if r.kind == 'q' { "QINIT" } else { "QMODE" };
                problems.push(Diagnostic::at(
                    &r.tok,
                    t!(
                        "{} needs {}{}..{}{}, but {} declares {}",
                        "{}-k {}{}..{}{} behar ditu, baina {}-k {} adierazten ditu",
                        name,
                        r.kind,
                        r.start,
                        r.kind,
                        r.start + r.size - 1,
                        what,
                        n
                    ),
                ));
            }
        }
        problems.into_iter().for_each(|d| self.error(d));
    }

    // izendun erregistroak indize arrunt bihurtu: data[1] -> q4, data -> q3..q5 (mnemonikoa eta testuak ez)
    // errore bat badago, None: lerroa ez da gehiago aztertzen (errore-kaskadarik ez)
    fn rename(&mut self, toks: &[Token]) -> Option<Vec<Token>> {
        if self.named.is_empty() {
            return Some(toks.to_vec());
        }
        let mut ok = true;
        let mut out = Vec::with_capacity(toks.len());
        for (i, t) in toks.iter().enumerate() {
            let (base, inner) = match t.text.split_once('[') {
                Some((b, rest)) => match rest.strip_suffix(']') {
                    Some(inner) => (b, Some(inner)),
                    None => (t.text.as_str(), None),
                },
                None => (t.text.as_str(), None),
            };
            let label = inner.is_none() && self.label_names.contains(&base.to_ascii_lowercase());
            let reg = (i > 0 && !t.text.starts_with('"') && !label)
                .then(|| self.named.get(base))
                .flatten();
            let Some(r) = reg else {
                out.push(t.clone());
                continue;
            };
            let (kind, start, size) = (r.kind, r.start, r.size);
            let range = match inner {
                None => Some((0, size - 1)),
                Some(s) => match s.split_once("..") {
                    Some((a, b)) => parse_int(a).zip(parse_int(b)),
                    None => parse_int(s).map(|k| (k, k)),
                }
                .map(|(a, b)| (a as u32, b as u32)),
            };
            match range {
                Some((a, b)) if a <= b && b < size => {
                    let text = if a == b {
                        format!("{}{}", kind, start + a)
                    } else {
                        format!("{}{}..{}{}", kind, start + a, kind, start + b)
                    };
                    out.push(Token { text, ..t.clone() });
                }
                Some((_, b)) if b >= size => {
                    self.error(Diagnostic::at(
                        t,
                        t!(
                            "{} has {} element(s); index {} is past the end",
                            "{}-k {} elementu ditu; {} indizea amaieratik kanpo dago",
                            base,
                            size,
                            b
                        ),
                    ));
                    ok = false;
                }
                _ => {
                    self.error(Diagnostic::at(
                        t,
                        t!(
                            "expected an index like {0}[1] or {0}[0..2]",
                            "{0}[1] edo {0}[0..2] bezalako indize bat espero zen",
                            base
                        ),
                    ));
                    ok = false;
                }
            }
        }
        ok.then_some(out)
    }

    // agindu-lerro bat (etiketa kenduta)
    fn line(&mut self, toks: &[Token]) {
        let head = &toks[0];
        let Some(spec) = isa::lookup(&head.text) else {
            let mut d = Diagnostic::at(
                head,
                t!(
                    "unknown instruction '{}'",
                    "'{}' agindu ezezaguna",
                    head.text
                ),
            );
            if let Some(s) = suggest(&head.text) {
                d = d.help(t!("did you mean {}?", "{} esan nahi zenuen?", s));
            } else {
                d = d.help(t!(
                    "`qoa isa` lists every instruction",
                    "`qoa isa`-k agindu guztiak zerrendatzen ditu"
                ));
            }
            self.error(d);
            return;
        };
        if let Some(why) = spec.removed_reason() {
            self.error(
                Diagnostic::at(
                    head,
                    t!(
                        "{} was removed in QOA 0.4",
                        "{} QOA 0.4-n kendu zen",
                        spec.name
                    ),
                )
                .help(why.to_string()),
            );
            return;
        }
        // deklarazioak lehen pasaldian landu dira
        if matches!(
            spec.name,
            "QINIT" | "QMODE" | "QREG" | "CREG" | "OREG" | "COMMENT"
        ) {
            return;
        }
        let args = &toks[1..];

        // hedapena: `H q0..q3` lau H bihurtzen da
        if spec.broadcast
            && let Some(first) = args.first()
            && first.text.contains("..")
        {
            let Some(list) = self.index_list(first, 'q') else {
                return;
            };
            // helmuga ere tartea bada (MEASURE q0..q2 -> c0..c2), bikoteka lotzen dira
            let dest_range = args
                .iter()
                .position(|t| t.text.contains("..") && t.col != first.col);
            let dest_list = match dest_range {
                Some(i) => {
                    let tk = args[i].clone();
                    let Some(l) = self.dest_list(&tk) else { return };
                    if l.len() != list.len() {
                        self.error(Diagnostic::at(
                            &tk,
                            t!(
                                "this range has {} entries but the qubit range has {}",
                                "tarte honek {} elementu ditu, baina qubit-tarteak {}",
                                l.len(),
                                list.len()
                            ),
                        ));
                        return;
                    }
                    Some((i, l))
                }
                None => None,
            };
            for (k, q) in list.iter().enumerate() {
                let mut expanded: Vec<Token> = args.to_vec();
                expanded[0] = Token {
                    text: format!("q{}", q),
                    ..first.clone()
                };
                if let Some((i, l)) = &dest_list {
                    expanded[*i] = Token {
                        text: l[k].clone(),
                        ..args[*i].clone()
                    };
                }
                self.instruction(spec, head, &expanded);
            }
            return;
        }
        self.instruction(spec, head, args);
    }

    // eragigaiak sinaduraren arabera banatu eta agindua sortu
    fn instruction(&mut self, spec: &'static Spec, head: &Token, args: &[Token]) {
        let specs = spec.operand_specs();
        let Some(groups) = self.match_operands(spec, head, &specs, args) else {
            return;
        };
        let mut values = Vec::with_capacity(specs.len());
        for (os, toks) in specs.iter().zip(&groups) {
            match self.parse_operand(spec, os, toks) {
                Some(v) => values.push(v),
                None => return,
            }
        }
        self.build(spec, head, &groups, values, head.line);
    }

    // tokenak sinadurako elementuekin lotu. elementu bakoitzak token-zerrenda bat jasotzen du
    // (hutsa aukerakoa falta bada, bat baino gehiago errepikatzen bada)
    fn match_operands(
        &mut self,
        spec: &Spec,
        head: &Token,
        specs: &[OperandSpec],
        args: &[Token],
    ) -> Option<Vec<Vec<Token>>> {
        // -> geziak helmuga baten aurretik bakarrik joan daitezke
        let mut toks: Vec<Token> = Vec::new();
        let mut arrow_before = Vec::new();
        let mut pending_arrow: Option<Token> = None;
        for tk in args {
            if tk.text == "->" {
                if pending_arrow.is_some() {
                    self.error(Diagnostic::at(
                        tk,
                        t!("two arrows in a row", "bi gezi jarraian"),
                    ));
                    return None;
                }
                pending_arrow = Some(tk.clone());
            } else {
                arrow_before.push(pending_arrow.take());
                toks.push(tk.clone());
            }
        }
        if let Some(arrow) = pending_arrow {
            self.error(Diagnostic::at(
                &arrow,
                t!(
                    "the arrow is not followed by a destination",
                    "geziaren ondoren ez dago helmugarik"
                ),
            ));
            return None;
        }

        let usage = t!("usage: {}", "erabilera: {}", spec.usage());
        let variadic = specs.iter().position(|s| s.repeat != Repeat::One);
        let mut groups: Vec<Vec<Token>> = vec![Vec::new(); specs.len()];
        match variadic {
            None => {
                let required = specs.iter().filter(|s| !s.optional).count();
                if toks.len() < required || toks.len() > specs.len() {
                    let expected = if required == specs.len() {
                        required.to_string()
                    } else {
                        format!("{}-{}", required, specs.len())
                    };
                    let at = toks.get(specs.len()).unwrap_or(head);
                    let msg = if toks.len() < required {
                        t!(
                            "too few operands for {}: expected {}, found {}",
                            "eragigai gutxiegi {}-rentzat: {} espero ziren, {} daude",
                            spec.name,
                            expected,
                            toks.len()
                        )
                    } else {
                        t!(
                            "too many operands for {}: expected {}, found {}",
                            "eragigai gehiegi {}-rentzat: {} espero ziren, {} daude",
                            spec.name,
                            expected,
                            toks.len()
                        )
                    };
                    self.error(Diagnostic::at(at, msg).help(usage));
                    return None;
                }
                for (i, tk) in toks.iter().enumerate() {
                    groups[i].push(tk.clone());
                }
            }
            Some(v) => {
                let after = specs.len() - v - 1;
                let min = v
                    + after
                    + if specs[v].repeat == Repeat::OneOrMore {
                        1
                    } else {
                        0
                    };
                if toks.len() < min {
                    self.error(
                        Diagnostic::at(
                            toks.last().unwrap_or(head),
                            t!(
                                "too few operands for {}",
                                "eragigai gutxiegi {}-rentzat",
                                spec.name
                            ),
                        )
                        .help(usage),
                    );
                    return None;
                }
                for i in 0..v {
                    groups[i].push(toks[i].clone());
                }
                let var_end = toks.len() - after;
                for tk in &toks[v..var_end] {
                    groups[v].push(tk.clone());
                }
                for k in 0..after {
                    groups[v + 1 + k].push(toks[var_end + k].clone());
                }
            }
        }
        // gezia helmugaren aurretik bakarrik
        let mut pos = 0;
        for (i, g) in groups.iter().enumerate() {
            for _ in g {
                if let Some(arrow) = &arrow_before[pos]
                    && specs[i].kind != 'd'
                {
                    self.error(Diagnostic::at(arrow, t!(
                            "-> can only come before a result destination; {} takes none here",
                            "-> emaitzaren helmuga baten aurretik bakarrik joan daiteke; {}-k ez du hemen helmugarik hartzen",
                            spec.name
                        )).help(usage));
                    return None;
                }
                pos += 1;
            }
        }
        Some(groups)
    }

    fn parse_operand(&mut self, spec: &Spec, os: &OperandSpec, toks: &[Token]) -> Option<Arg> {
        if toks.is_empty() {
            return Some(Arg::Missing);
        }
        if os.repeat != Repeat::One {
            // zerrenda: tarteak zabaldu
            let mut all = Vec::new();
            match os.kind {
                'q' | 'o' | 'c' => {
                    for tk in toks {
                        all.extend(self.index_list(tk, os.kind)?);
                    }
                    return Some(Arg::Indices(all));
                }
                'r' => {
                    for tk in toks {
                        all.push(self.index(tk, 'r')?);
                    }
                    return Some(Arg::Indices(all));
                }
                _ => unreachable!("{}: errepikapena q, o, c eta r-rentzat bakarrik", spec.name),
            }
        }
        let tk = &toks[0];
        let name = os.name;
        Some(match os.kind {
            'q' | 'o' | 'r' | 'c' => {
                if tk.text.contains("..") {
                    self.error(Diagnostic::at(
                        tk,
                        t!(
                            "{} does not take a range here",
                            "{}-k ez du hemen tarterik hartzen",
                            spec.name
                        ),
                    ));
                    return None;
                }
                Arg::Index(self.index(tk, os.kind)?)
            }
            'd' => Arg::Dest(self.dest(tk)?),
            'v' => match self.dest(tk)? {
                Dest::Bit(b) => Arg::Source(Source::Bit(b)),
                Dest::Reg(r) => Arg::Source(Source::Reg(r)),
                Dest::Own => unreachable!(),
            },
            'f' => match expr::eval(&tk.text) {
                Ok(v) => Arg::Float(v),
                Err(e) => {
                    self.error(Diagnostic::at(
                        tk,
                        t!(
                            "expected a number for {}: {}",
                            "zenbaki bat espero zen {}-rentzat: {}",
                            name,
                            e
                        ),
                    ));
                    return None;
                }
            },
            'i' => match parse_int(&tk.text) {
                Some(v) => Arg::Int(v),
                None => {
                    self.error(Diagnostic::at(
                        tk,
                        t!(
                            "expected a whole number for {}, found '{}'",
                            "zenbaki oso bat espero zen {}-rentzat, baina '{}' dago",
                            name,
                            tk.text
                        ),
                    ));
                    return None;
                }
            },
            'n' => {
                if tk.text.starts_with(['R', 'r']) {
                    Arg::Count(Count::Reg(self.index(tk, 'r')?))
                } else {
                    match parse_int(&tk.text) {
                        Some(v) => Arg::Count(Count::Lit(v)),
                        None => {
                            self.error(Diagnostic::at(tk, t!(
                                "expected a count (a number or a register like R1), found '{}'",
                                "kopuru bat espero zen (zenbaki bat edo R1 bezalako erregistro bat), baina '{}' dago",
                                tk.text
                            )));
                            return None;
                        }
                    }
                }
            }
            'l' => {
                if !is_label_name(&tk.text) {
                    self.error(Diagnostic::at(
                        tk,
                        t!(
                            "'{}' is not a label name",
                            "'{}' ez da etiketa-izen bat",
                            tk.text
                        ),
                    ));
                    return None;
                }
                Arg::Label(tk.text.clone())
            }
            's' => Arg::Text(self.string(tk)?),
            'k' => {
                if tk.text.starts_with('"') {
                    let s = self.string(tk)?;
                    let mut cs = s.chars();
                    match (cs.next(), cs.next()) {
                        (Some(c), None) => Arg::Int(c as u64),
                        _ => {
                            self.error(Diagnostic::at(
                                tk,
                                t!(
                                    "expected exactly one character",
                                    "karaktere bakar bat espero zen"
                                ),
                            ));
                            return None;
                        }
                    }
                } else {
                    match parse_int(&tk.text) {
                        Some(v) if v <= 0x10FFFF => Arg::Int(v),
                        _ => {
                            self.error(Diagnostic::at(tk, t!(
                                "expected a character (\"A\") or a character code, found '{}'",
                                "karaktere bat (\"A\") edo karaktere-kode bat espero zen, baina '{}' dago",
                                tk.text
                            )));
                            return None;
                        }
                    }
                }
            }
            'b' => match tk.text.to_ascii_uppercase().as_str() {
                "X" => Arg::Basis(Basis::X),
                "Y" => Arg::Basis(Basis::Y),
                "Z" => Arg::Basis(Basis::Z),
                _ => {
                    self.error(Diagnostic::at(
                        tk,
                        t!(
                            "expected X, Y or Z for {}, found '{}'",
                            "X, Y edo Z espero zen {}-rentzat, baina '{}' dago",
                            name,
                            tk.text
                        ),
                    ));
                    return None;
                }
            },
            'g' => Arg::Name(tk.text.clone()),
            _ => unreachable!(),
        })
    }

    // q3 / o1 / R2 / c0 edo zenbaki hutsa (bitek aurrizkia behar dute); mugak egiaztatzen dira
    fn index(&mut self, tk: &Token, kind: char) -> Option<u32> {
        let text = tk.text.as_str();
        let prefixes: &[char] = match kind {
            'q' => &['q', 'Q'],
            'o' => &['o', 'O'],
            'c' => &['c', 'C'],
            _ => &['r', 'R'],
        };
        let digits = text.strip_prefix(prefixes).unwrap_or(text);
        let other_prefix = text
            .chars()
            .next()
            .filter(|c| c.is_ascii_alphabetic() && !prefixes.contains(c));
        // bit klasikoek aurrizkia behar dute (c0), zenbaki hutsa anbiguoa litzatekeelako
        let bare_bit = kind == 'c' && !text.starts_with(prefixes);
        let parsed = if other_prefix.is_none() && !bare_bit {
            digits.parse::<u32>().ok()
        } else {
            None
        };
        let Some(n) = parsed else {
            let (msg, hint) = match kind {
                'q' => (
                    t!(
                        "expected a qubit, found '{}'",
                        "qubit bat espero zen, baina '{}' dago",
                        text
                    ),
                    t!(
                        "write a qubit as q3 (or just 3)",
                        "idatzi qubit bat q3 gisa (edo 3 besterik ez)"
                    ),
                ),
                'o' => (
                    t!(
                        "expected an optical mode, found '{}'",
                        "modu optiko bat espero zen, baina '{}' dago",
                        text
                    ),
                    t!(
                        "write an optical mode as o1 (or just 1)",
                        "idatzi modu optiko bat o1 gisa (edo 1 besterik ez)"
                    ),
                ),
                'c' => (
                    t!(
                        "expected a classical bit, found '{}'",
                        "bit klasiko bat espero zen, baina '{}' dago",
                        text
                    ),
                    t!(
                        "write a classical bit as c0",
                        "idatzi bit klasiko bat c0 gisa"
                    ),
                ),
                _ => (
                    t!(
                        "expected a register, found '{}'",
                        "erregistro bat espero zen, baina '{}' dago",
                        text
                    ),
                    t!(
                        "write a register as R2 (or just 2)",
                        "idatzi erregistro bat R2 gisa (edo 2 besterik ez)"
                    ),
                ),
            };
            self.error(Diagnostic::at(tk, msg).help(hint));
            return None;
        };
        match kind {
            'q' => {
                if let Some(limit) = self.declared_qubits {
                    if n >= limit {
                        if !self.qubit_limit_reported {
                            self.qubit_limit_reported = true;
                            let msg = if limit == 0 {
                                t!(
                                    "qubit {} does not exist: QINIT declared no qubits",
                                    "{}. qubita ez da existitzen: QINIT-ek ez du qubitik adierazi",
                                    n
                                )
                            } else {
                                t!(
                                    "qubit {} does not exist: QINIT declared {} qubit(s) (q0..q{})",
                                    "{}. qubita ez da existitzen: QINIT-ek {} qubit adierazi ditu (q0..q{})",
                                    n,
                                    limit,
                                    limit - 1
                                )
                            };
                            let line = self.qinit_token.as_ref().map_or(0, |q| q.line);
                            self.error(Diagnostic::at(tk, msg).help(t!(
                                "QINIT (line {}) takes the number of qubits; this program needs at least {}. Later uses of missing qubits are not reported again",
                                "QINIT-ek ({}. lerroa) qubit-kopurua hartzen du; programa honek gutxienez {} behar ditu. Falta diren qubiten hurrengo erabilerak ez dira berriz jakinarazten",
                                line,
                                n + 1
                            )));
                        }
                        return None;
                    }
                } else if n >= 64 {
                    self.error(Diagnostic::at(tk, t!(
                        "qubit index too large (a state vector cannot hold more than 64 qubits)",
                        "qubit-indize handiegia (egoera-bektore batek ezin ditu 64 qubit baino gehiago gorde)"
                    )));
                    return None;
                }
                self.max_qubit = Some(self.max_qubit.map_or(n, |m| m.max(n)));
            }
            'o' => {
                if let Some((limit, _, _)) = self.declared_modes {
                    if n >= limit {
                        self.error(Diagnostic::at(
                            tk,
                            t!(
                                "mode {} does not exist: QMODE declared {} mode(s)",
                                "{}. modua ez da existitzen: QMODE-k {} modu adierazi ditu",
                                n,
                                limit
                            ),
                        ));
                        return None;
                    }
                } else if n >= 64 {
                    self.error(Diagnostic::at(
                        tk,
                        t!("mode index too large", "modu-indize handiegia"),
                    ));
                    return None;
                }
                self.max_mode = Some(self.max_mode.map_or(n, |m| m.max(n)));
            }
            'c' => {
                if n >= 4096 {
                    self.error(Diagnostic::at(
                        tk,
                        t!(
                            "classical bit index too large (c0..c4095)",
                            "bit klasikoaren indize handiegia (c0..c4095)"
                        ),
                    ));
                    return None;
                }
                self.max_bit = Some(self.max_bit.map_or(n, |m| m.max(n)));
            }
            _ => {
                if n as usize >= NUM_REGS {
                    self.error(Diagnostic::at(
                        tk,
                        t!(
                            "register R{} does not exist (R0..R{})",
                            "R{} erregistroa ez da existitzen (R0..R{})",
                            n,
                            NUM_REGS - 1
                        ),
                    ));
                    return None;
                }
            }
        }
        Some(n)
    }

    // q0..q3 tartea edo indize bakarra
    fn index_list(&mut self, tk: &Token, kind: char) -> Option<Vec<u32>> {
        if let Some((a, b)) = tk.text.split_once("..") {
            let ta = Token {
                text: a.to_string(),
                ..tk.clone()
            };
            // amaieran aurrizkia aukerakoa da: q0..3 eta q0..q3 berdinak dira
            let b_text = match a.chars().next() {
                Some(p)
                    if p.is_ascii_alphabetic() && b.starts_with(|c: char| c.is_ascii_digit()) =>
                {
                    format!("{}{}", p, b)
                }
                _ => b.to_string(),
            };
            let tb = Token {
                text: b_text,
                ..tk.clone()
            };
            let lo = self.index(&ta, kind)?;
            let hi = self.index(&tb, kind)?;
            if hi < lo {
                self.error(Diagnostic::at(
                    tk,
                    t!(
                        "a range must count upwards, like q0..q3",
                        "tarte batek gorantz zenbatu behar du, q0..q3 bezala"
                    ),
                ));
                return None;
            }
            Some((lo..=hi).collect())
        } else {
            Some(vec![self.index(tk, kind)?])
        }
    }

    // helmuga-tartea (c0..c2 edo R0..R2), testu gisa itzulita hedapenerako
    fn dest_list(&mut self, tk: &Token) -> Option<Vec<String>> {
        let (a, b) = tk.text.split_once("..").unwrap();
        let prefix = a.chars().next().unwrap_or(' ');
        let bad = |me: &mut Asm| {
            me.error(Diagnostic::at(
                tk,
                t!(
                    "a destination range is written c0..c3 or R0..R3",
                    "helmuga-tarte bat c0..c3 edo R0..R3 gisa idazten da"
                ),
            ));
        };
        if !matches!(prefix, 'c' | 'C' | 'r' | 'R') {
            bad(self);
            return None;
        }
        let lo = a[1..].parse::<u32>().ok();
        let hi = b
            .trim_start_matches(['c', 'C', 'r', 'R'])
            .parse::<u32>()
            .ok();
        match (lo, hi) {
            (Some(lo), Some(hi)) if hi >= lo => {
                Some((lo..=hi).map(|i| format!("{}{}", prefix, i)).collect())
            }
            _ => {
                bad(self);
                None
            }
        }
    }

    // c3 (bit klasikoa) edo R2 (erregistroa)
    fn dest(&mut self, tk: &Token) -> Option<Dest> {
        let text = tk.text.as_str();
        if text.starts_with(['c', 'C']) {
            return Some(Dest::Bit(self.index(tk, 'c')?));
        }
        if text.starts_with(['r', 'R']) {
            return Some(Dest::Reg(self.index(tk, 'r')?));
        }
        self.error(
            Diagnostic::at(
                tk,
                t!(
                    "expected a classical bit or register, found '{}'",
                    "bit klasiko bat edo erregistro bat espero zen, baina '{}' dago",
                    text
                ),
            )
            .help(t!(
                "write a classical bit as c0 or a register as R2",
                "idatzi bit klasiko bat c0 gisa edo erregistro bat R2 gisa"
            )),
        );
        None
    }

    fn string(&mut self, tk: &Token) -> Option<String> {
        let Some(inner) = tk.text.strip_prefix('"').and_then(|s| s.strip_suffix('"')) else {
            self.error(Diagnostic::at(
                tk,
                t!(
                    "expected text in double quotes, found '{}'",
                    "komatxo bikoitzen arteko testua espero zen, baina '{}' dago",
                    tk.text
                ),
            ));
            return None;
        };
        let mut out = String::new();
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some(other) => {
                        out.push('\\');
                        out.push(other);
                    }
                    None => out.push('\\'),
                }
            } else {
                out.push(c);
            }
        }
        Some(out)
    }

    // balioztatutako eragigaietatik Op sortu
    fn build(&mut self, spec: &Spec, head: &Token, groups: &[Vec<Token>], v: Vec<Arg>, line: u32) {
        // laguntzaileak: eragigaiak motaren arabera atera
        let idx = |i: usize| match &v[i] {
            Arg::Index(n) => *n,
            other => panic!(
                "{}",
                t!(
                    "operand {} of {} is {:?}, not an index (QOA bug)",
                    "{1}-ren {0}. eragigaia {2:?} da, ez indize bat (QOA-ren akatsa)",
                    i,
                    spec.name,
                    other
                )
            ),
        };
        let list = |i: usize| match &v[i] {
            Arg::Indices(l) => l.clone(),
            Arg::Missing => Vec::new(),
            other => panic!(
                "{}",
                t!(
                    "operand {} of {} is {:?}, not a list (QOA bug)",
                    "{1}-ren {0}. eragigaia {2:?} da, ez zerrenda bat (QOA-ren akatsa)",
                    i,
                    spec.name,
                    other
                )
            ),
        };
        let float = |i: usize, default: f64| match &v[i] {
            Arg::Float(f) => *f,
            Arg::Missing => default,
            other => panic!(
                "{}",
                t!(
                    "operand {} of {} is {:?}, not a number (QOA bug)",
                    "{1}-ren {0}. eragigaia {2:?} da, ez zenbaki bat (QOA-ren akatsa)",
                    i,
                    spec.name,
                    other
                )
            ),
        };
        let int = |i: usize| match &v[i] {
            Arg::Int(n) => *n,
            other => panic!(
                "{}",
                t!(
                    "operand {} of {} is {:?}, not an integer (QOA bug)",
                    "{1}-ren {0}. eragigaia {2:?} da, ez zenbaki oso bat (QOA-ren akatsa)",
                    i,
                    spec.name,
                    other
                )
            ),
        };
        let dest = |i: usize| match &v[i] {
            Arg::Dest(d) => *d,
            Arg::Missing => Dest::Own,
            other => panic!(
                "{}",
                t!(
                    "operand {} of {} is {:?}, not a destination (QOA bug)",
                    "{1}-ren {0}. eragigaia {2:?} da, ez helmuga bat (QOA-ren akatsa)",
                    i,
                    spec.name,
                    other
                )
            ),
        };
        let opt_reg = |i: usize| match &v[i] {
            Arg::Index(n) => Some(*n),
            Arg::Missing => None,
            other => panic!(
                "{}",
                t!(
                    "operand {} of {} is {:?}, not a register (QOA bug)",
                    "{1}-ren {0}. eragigaia {2:?} da, ez erregistro bat (QOA-ren akatsa)",
                    i,
                    spec.name,
                    other
                )
            ),
        };
        let basis = |i: usize| match &v[i] {
            Arg::Basis(b) => *b,
            other => panic!(
                "{}",
                t!(
                    "operand {} of {} is {:?}, not a basis (QOA bug)",
                    "{1}-ren {0}. eragigaia {2:?} da, ez oinarri bat (QOA-ren akatsa)",
                    i,
                    spec.name,
                    other
                )
            ),
        };
        let text = |i: usize| match &v[i] {
            Arg::Text(s) => s.clone(),
            Arg::Missing => String::new(),
            other => panic!(
                "{}",
                t!(
                    "operand {} of {} is {:?}, not text (QOA bug)",
                    "{1}-ren {0}. eragigaia {2:?} da, ez testua (QOA-ren akatsa)",
                    i,
                    spec.name,
                    other
                )
            ),
        };
        let label = |i: usize| match &v[i] {
            Arg::Label(l) => l.clone(),
            other => panic!(
                "{}",
                t!(
                    "operand {} of {} is {:?}, not a label (QOA bug)",
                    "{1}-ren {0}. eragigaia {2:?} da, ez etiketa bat (QOA-ren akatsa)",
                    i,
                    spec.name,
                    other
                )
            ),
        };
        let tok = |i: usize| groups[i].first().cloned().unwrap_or_else(|| head.clone());

        // qubit/modu desberdinak behar dituzten aginduetarako egiaztapena
        let distinct = |me: &mut Asm, items: &[(u32, usize)], qubits: bool| -> bool {
            for (k, (a, ia)) in items.iter().enumerate() {
                for (b, _) in &items[..k] {
                    if a == b {
                        let msg = if qubits {
                            t!(
                                "{} uses qubit {} twice; the qubits must be different",
                                "{}-k {}. qubita bi aldiz erabiltzen du; qubitek desberdinak izan behar dute",
                                spec.name,
                                a
                            )
                        } else {
                            t!(
                                "{} uses mode {} twice; the modes must be different",
                                "{}-k {}. modua bi aldiz erabiltzen du; moduek desberdinak izan behar dute",
                                spec.name,
                                a
                            )
                        };
                        me.error(Diagnostic::at(&tok(*ia), msg));
                        return false;
                    }
                }
            }
            true
        };
        let all = |qs: &[u32], i: usize| qs.iter().map(|q| (*q, i)).collect::<Vec<_>>();

        let g1 = |q: u32, gate: Gate| Op::Gate { q, gate };
        let op = match spec.name {
            "H" => g1(idx(0), Gate::H),
            "X" => g1(idx(0), Gate::X),
            "Y" => g1(idx(0), Gate::Y),
            "Z" => g1(idx(0), Gate::Z),
            "S" => g1(idx(0), Gate::S),
            "SDG" => g1(idx(0), Gate::Sdg),
            "T" => g1(idx(0), Gate::T),
            "TDG" => g1(idx(0), Gate::Tdg),
            "SX" => g1(idx(0), Gate::SX),
            "SXDG" => g1(idx(0), Gate::SXdg),
            "PHASE" => g1(idx(0), Gate::P(float(1, 0.0))),
            "RX" => g1(idx(0), Gate::RX(float(1, 0.0))),
            "RY" => g1(idx(0), Gate::RY(float(1, 0.0))),
            "RZ" => g1(idx(0), Gate::RZ(float(1, 0.0))),
            "U" => g1(idx(0), Gate::U(float(1, 0.0), float(2, 0.0), float(3, 0.0))),
            "ROT" => g1(idx(0), rotation(basis(1), float(2, 0.0))),
            "MROT" => {
                let qs = list(2);
                if !distinct(self, &all(&qs, 2), true) {
                    return;
                }
                let g = rotation(basis(0), float(1, 0.0));
                for q in qs {
                    self.emit(Op::Gate { q, gate: g }, line);
                }
                return;
            }
            "GATE" => {
                let name = match &v[0] {
                    Arg::Name(n) => n.clone(),
                    _ => unreachable!(),
                };
                match Gate::from_name(&name) {
                    Some(g) => g1(idx(1), g),
                    None => {
                        self.error(Diagnostic::at(&tok(0), t!("unknown gate name '{}'", "'{}' ate-izen ezezaguna", name)).help(t!(
                            "GATE takes one of: h x y z s sdg t tdg sx sxdg (two-qubit gates have their own instructions)",
                            "GATE-k hauetako bat hartzen du: h x y z s sdg t tdg sx sxdg (bi qubiteko ateek beren agindu propioak dituzte)"
                        )));
                        return;
                    }
                }
            }
            "CNOT" | "CY" | "CH" | "CSX" | "CU" | "CRX" | "CRY" | "CRZ" => {
                let (c, tq) = (idx(0), idx(1));
                if !distinct(self, &[(c, 0), (tq, 1)], true) {
                    return;
                }
                let gate = match spec.name {
                    "CNOT" => Gate::X,
                    "CY" => Gate::Y,
                    "CH" => Gate::H,
                    "CSX" => Gate::SX,
                    "CU" => Gate::U(float(2, 0.0), float(3, 0.0), float(4, 0.0)),
                    "CRX" => Gate::RX(float(2, 0.0)),
                    "CRY" => Gate::RY(float(2, 0.0)),
                    _ => Gate::RZ(float(2, 0.0)),
                };
                Op::Controlled {
                    controls: vec![c],
                    q: tq,
                    gate,
                }
            }
            "CZ" | "CPHASE" => {
                let (c, tq) = (idx(0), idx(1));
                if !distinct(self, &[(c, 0), (tq, 1)], true) {
                    return;
                }
                let theta = if spec.name == "CZ" {
                    std::f64::consts::PI
                } else {
                    float(2, 0.0)
                };
                Op::PhaseAll {
                    qubits: vec![c, tq],
                    theta,
                }
            }
            "SWAP" => {
                let (a, b) = (idx(0), idx(1));
                if !distinct(self, &[(a, 0), (b, 1)], true) {
                    return;
                }
                Op::Swap {
                    a,
                    b,
                    controls: vec![],
                }
            }
            "CSWAP" => {
                let (c, a, b) = (idx(0), idx(1), idx(2));
                if !distinct(self, &[(c, 0), (a, 1), (b, 2)], true) {
                    return;
                }
                Op::Swap {
                    a,
                    b,
                    controls: vec![c],
                }
            }
            "CCX" => {
                let (c1, c2, tq) = (idx(0), idx(1), idx(2));
                if !distinct(self, &[(c1, 0), (c2, 1), (tq, 2)], true) {
                    return;
                }
                Op::Controlled {
                    controls: vec![c1, c2],
                    q: tq,
                    gate: Gate::X,
                }
            }
            "MCX" => {
                let controls = list(0);
                let tq = idx(1);
                let mut items = all(&controls, 0);
                items.push((tq, 1));
                if !distinct(self, &items, true) {
                    return;
                }
                Op::Controlled {
                    controls,
                    q: tq,
                    gate: Gate::X,
                }
            }
            "MCZ" => {
                let qs = list(0);
                if !distinct(self, &all(&qs, 0), true) {
                    return;
                }
                Op::PhaseAll {
                    qubits: qs,
                    theta: std::f64::consts::PI,
                }
            }
            "BELL" => {
                let (a, b) = (idx(0), idx(1));
                if !distinct(self, &[(a, 0), (b, 1)], true) {
                    return;
                }
                Op::Bell { a, b }
            }
            "GHZ" | "CLUSTER" => {
                let qs = list(0);
                if !distinct(self, &all(&qs, 0), true) {
                    return;
                }
                if spec.name == "GHZ" {
                    Op::Ghz { qubits: qs }
                } else {
                    Op::Cluster { qubits: qs }
                }
            }
            "ESWAP" => {
                let (a, b, c, d) = (idx(0), idx(1), idx(2), idx(3));
                if !distinct(self, &[(a, 0), (b, 1), (c, 2), (d, 3)], true) {
                    return;
                }
                Op::EntangleSwap { a, b, c, d }
            }
            "MEASURE" => Op::Measure {
                q: idx(0),
                basis: Basis::Z,
                dest: dest(1),
            },
            "MEASUREIN" => Op::Measure {
                q: idx(0),
                basis: basis(1),
                dest: dest(2),
            },
            "MEASUREALL" => Op::MeasureAll,
            "RESET" => Op::Reset { q: idx(0) },
            "RESETALL" => Op::ResetAll,
            "TOBASIS" => Op::ToBasis {
                q: idx(0),
                basis: basis(1),
            },
            "DISPLACE" => Op::Displace {
                o: idx(0),
                re: float(1, 0.0),
                im: float(2, 0.0),
            },
            "DISPLACEREG" => Op::DisplaceReg {
                o: idx(0),
                r: idx(1),
            },
            "SQUEEZE" => Op::Squeeze {
                o: idx(0),
                r: float(1, 0.0),
                phi: float(2, 0.0),
            },
            "SQUEEZEREG" => Op::SqueezeReg {
                o: idx(0),
                reg: idx(1),
            },
            "MODEPHASE" => Op::ModePhase {
                o: idx(0),
                theta: float(1, 0.0),
            },
            "KERR" => Op::Kerr {
                o: idx(0),
                kappa: float(1, 0.0) * float(2, 1.0),
            },
            "CROSSKERR" | "BS" | "MODESWAP" => {
                let (a, b) = (idx(0), idx(1));
                if !distinct(self, &[(a, 0), (b, 1)], false) {
                    return;
                }
                match spec.name {
                    "CROSSKERR" => Op::CrossKerr {
                        a,
                        b,
                        kappa: float(2, 0.0),
                    },
                    "BS" => Op::BeamSplitter {
                        a,
                        b,
                        theta: float(2, 0.0),
                        phi: float(3, 0.0),
                    },
                    _ => Op::ModeSwap { a, b },
                }
            }
            "ADDPHOTON" => Op::AddPhoton { o: idx(0) },
            "SUBPHOTON" => Op::SubPhoton { o: idx(0) },
            "LOSS" => {
                let gamma = float(1, 0.0);
                if !(0.0..=1.0).contains(&gamma) {
                    self.error(Diagnostic::at(
                        &tok(1),
                        t!(
                            "a loss probability must be between 0 and 1, found {}",
                            "galera-probabilitate batek 0 eta 1 artean egon behar du, baina {} da",
                            gamma
                        ),
                    ));
                    return;
                }
                Op::Loss { o: idx(0), gamma }
            }
            "FOCK" => {
                let n = int(1);
                let (cutoff, declared) = match self.declared_modes {
                    Some((_, c, d)) => (c, d),
                    None => (DEFAULT_CUTOFF, false),
                };
                if n >= cutoff as u64 {
                    let help = if declared {
                        t!("raise the cutoff with QMODE", "handitu muga QMODE bidez")
                    } else {
                        t!(
                            "declare a larger cutoff, e.g. QMODE 1 16",
                            "adierazi muga handiago bat, adib. QMODE 1 16"
                        )
                    };
                    self.error(
                        Diagnostic::at(
                            &tok(1),
                            t!(
                                "{} photons do not fit: the cutoff is {} (0..{} photons)",
                                "{} fotoi ez dira sartzen: muga {} da (0..{} fotoi)",
                                n,
                                cutoff,
                                cutoff - 1
                            ),
                        )
                        .help(help),
                    );
                    return;
                }
                Op::Fock {
                    o: idx(0),
                    n: n as u32,
                }
            }
            "SINGLEPHOTON" => Op::Fock { o: idx(0), n: 1 },
            "COUNT" => {
                let d = dest(1);
                if let Dest::Bit(_) = d {
                    self.error(Diagnostic::at(&tok(1), t!(
                        "a photon count can be more than 1, so it cannot go to a classical bit",
                        "fotoi-kopurua 1 baino handiagoa izan daiteke, beraz ezin da bit klasiko batera joan"
                    )).help(t!("route it to a register, e.g. COUNT o0 -> R1", "bideratu erregistro batera, adib. COUNT o0 -> R1")));
                    return;
                }
                Op::PhotonCount { o: idx(0), dest: d }
            }
            "DETECT" => Op::PhotonDetect {
                o: idx(0),
                dest: dest(1),
            },
            "PARITY" => Op::Parity {
                o: idx(0),
                dest: dest(1),
            },
            "COINCIDENCE" => {
                let modes = list(0);
                if !distinct(self, &all(&modes, 0), false) {
                    return;
                }
                Op::Coincidence {
                    modes,
                    dest: dest(1),
                }
            }
            "EMIT" => Op::Emit {
                q: idx(0),
                o: idx(1),
            },
            "DISPERSIVE" => Op::Dispersive {
                q: idx(0),
                o: idx(1),
                theta: float(2, 0.0),
            },
            "DUMPSTATE" => Op::DumpState,
            "EXPECT" => Op::Expect {
                q: idx(0),
                basis: basis(1),
                dest: opt_reg(2),
            },
            "BELLCHECK" => {
                let (a, b) = (idx(0), idx(1));
                if !distinct(self, &[(a, 0), (b, 1)], true) {
                    return;
                }
                Op::BellCheck {
                    a,
                    b,
                    dest: opt_reg(2),
                }
            }
            "MEANPHOTON" => Op::MeanPhoton {
                o: idx(0),
                dest: opt_reg(1),
            },
            "REGSET" => Op::Set {
                r: idx(0),
                value: float(1, 0.0),
            },
            "REGCOPY" => Op::Copy {
                dst: idx(0),
                src: idx(1),
            },
            "ADD" | "SUB" | "MUL" | "DIV" | "ANDBITS" | "ORBITS" | "XORBITS" | "SHL" | "SHR" => {
                let op = match spec.name {
                    "ADD" => BinOp::Add,
                    "SUB" => BinOp::Sub,
                    "MUL" => BinOp::Mul,
                    "DIV" => BinOp::Div,
                    "ANDBITS" => BinOp::And,
                    "ORBITS" => BinOp::Or,
                    "XORBITS" => BinOp::Xor,
                    "SHL" => BinOp::Shl,
                    _ => BinOp::Shr,
                };
                Op::Bin {
                    op,
                    dst: idx(0),
                    a: idx(1),
                    b: idx(2),
                }
            }
            "NOTBITS" | "SQRT" | "EXP" | "LOG" => {
                let op = match spec.name {
                    "NOTBITS" => UnOp::Not,
                    "SQRT" => UnOp::Sqrt,
                    "EXP" => UnOp::Exp,
                    _ => UnOp::Log,
                };
                Op::Un {
                    op,
                    dst: idx(0),
                    src: idx(1),
                }
            }
            "RAND" => Op::Rand { r: idx(0) },
            "SEEDRNG" => Op::Seed { seed: int(0) },
            "INPUT" => Op::Input { r: idx(0) },
            "GETTIME" => Op::Time { r: idx(0) },
            "PUSH" => Op::Push { r: idx(0) },
            "POP" => Op::Pop { r: idx(0) },
            "CHARLOAD" => Op::Set {
                r: idx(0),
                value: int(1) as f64,
            },
            "PACKBITS" => {
                let bits = list(1);
                if bits.len() > 52 {
                    self.error(Diagnostic::at(
                        &tok(1),
                        t!(
                            "at most 52 bits fit exactly in a register",
                            "gehienez 52 bit sartzen dira zehazki erregistro batean"
                        ),
                    ));
                    return;
                }
                Op::PackBits { dst: idx(0), bits }
            }
            "JUMP" | "CALL" => {
                let op = if spec.name == "JUMP" {
                    Op::Jump { to: usize::MAX }
                } else {
                    Op::Call { to: usize::MAX }
                };
                let i = self.emit(op, line);
                self.fixups.push((i, label(0), tok(0)));
                return;
            }
            "JUMPIFZERO" | "JUMPIFONE" => {
                let src = match &v[0] {
                    Arg::Source(s) => *s,
                    _ => unreachable!(),
                };
                let value = if spec.name == "JUMPIFZERO" { 0.0 } else { 1.0 };
                let i = self.emit(
                    Op::JumpIf {
                        src,
                        value,
                        to: usize::MAX,
                    },
                    line,
                );
                self.fixups.push((i, label(1), tok(1)));
                return;
            }
            "IFEQ" | "IFNE" | "IFGT" | "IFLT" => {
                let cmp = match spec.name {
                    "IFEQ" => Cmp::Eq,
                    "IFNE" => Cmp::Ne,
                    "IFGT" => Cmp::Gt,
                    _ => Cmp::Lt,
                };
                let i = self.emit(
                    Op::Branch {
                        cmp,
                        a: idx(0),
                        b: idx(1),
                        to: usize::MAX,
                    },
                    line,
                );
                self.fixups.push((i, label(2), tok(2)));
                return;
            }
            "RETSUB" => Op::Return,
            "LOOPSTART" => {
                let count = match &v[0] {
                    Arg::Count(c) => *c,
                    _ => unreachable!(),
                };
                let i = self.emit(
                    Op::LoopStart {
                        count,
                        end: usize::MAX,
                    },
                    line,
                );
                self.loop_stack.push((i, head.clone()));
                return;
            }
            "LOOPEND" => {
                let Some((start, _)) = self.loop_stack.pop() else {
                    self.error(Diagnostic::at(
                        head,
                        t!("LOOPEND without a LOOPSTART", "LOOPEND LOOPSTART gabe"),
                    ));
                    return;
                };
                let end = self.emit(Op::LoopEnd { start }, line);
                if let Op::LoopStart { end: e, .. } = &mut self.ops[start] {
                    *e = end + 1;
                }
                return;
            }
            "HALT" => Op::Halt,
            "BARRIER" => Op::Barrier,
            "PRINT" => Op::Print {
                text: text(0),
                newline: false,
            },
            "PRINTLN" => Op::Print {
                text: text(0),
                newline: true,
            },
            "PRINTF" => {
                let format = text(0);
                let regs = list(1);
                let slots = format.matches("{}").count();
                if slots != regs.len() {
                    self.error(Diagnostic::at(
                        &tok(0),
                        t!(
                            "the format has {} {{}} slot(s) but {} register(s) are given",
                            "formatuak {} {{}} hutsune ditu, baina {} erregistro eman dira",
                            slots,
                            regs.len()
                        ),
                    ));
                    return;
                }
                Op::Printf { format, regs }
            }
            "CHAROUT" => Op::CharOut { r: idx(0) },
            "VERBOSELOG" => Op::Log {
                q: idx(0),
                text: text(1),
            },
            "DUMPREGS" => Op::DumpRegs,
            other => unreachable!("{} aginduak ez du eraikitzailerik", other),
        };
        self.emit(op, line);
    }
}

fn rotation(axis: Basis, angle: f64) -> Gate {
    match axis {
        Basis::X => Gate::RX(angle),
        Basis::Y => Gate::RY(angle),
        Basis::Z => Gate::RZ(angle),
    }
}

fn parse_int(s: &str) -> Option<u64> {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>().ok()
    }
}

// idazketa-akats txikietarako iradokizuna (editatze-distantzia txikiena)
fn suggest(word: &str) -> Option<&'static str> {
    let w = word.to_ascii_uppercase();
    // hitz laburretan akats bakarra onartzen da, bestela edozer litzateke "antzekoa"
    let limit = if w.len() <= 4 { 1 } else { 2 };
    let mut best: Option<(&'static str, usize)> = None;
    for s in isa::SPECS.iter().filter(|s| !s.is_removed()) {
        for cand in std::iter::once(&s.name).chain(s.aliases.iter()) {
            let d = edit_distance(&w, cand);
            if d <= limit && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((s.name, d));
            }
        }
    }
    best.map(|(n, _)| n)
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[b.len()]
}
