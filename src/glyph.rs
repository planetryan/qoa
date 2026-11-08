// fitxategi honek glifo-sintaxia (.qg, 0. zirriborroa) QOA iturburu arrunt bihurtzen du. glifo-forma ez da
// lengoaia desberdin bat: QOA bera da, trinkoago idatzia, lerro bakoitzean agindu askorekin. ikus docs/glyph.md.
//
//   R{3}; #C{3}; ~L{2}%6;          erregistroak: 3 qubit, 3 bit, 2 modu optiko (6 mailakoak)
//   H/R; CNOT/R[0],R[1];           agindua/eragigaiak; erregistro osoa = elementu bakoitzeko
//   RZ(pi/4)/R[2];                 parametroak parentesi artean
//   MEASURE/R$>#C;                 $> = emaitza hona doa (->)
//   PF"emaitza: {}"%#C;            % = balio hau inprimatu
//   PR"x"%#3;                      izena eta balioa: `x = 3` (PRINTF "x = {}\n" R3)
//   <<>>;  *3{ X/R[0]; }  ?#C[0]>amaiera;  amaiera:;  END;
//
// itzulpenak sarrerako adierazpen bakoitza QOA lerro bat edo gehiago bihurtzen du, eta lerro bakoitzak
// bere jatorria gordetzen du: mihiztatzailearen erroreek glifo-iturburuko adierazpena seinalatzen dute.

use crate::asm::{self, Diagnostic};
use crate::program::Program;
use crate::{t, ts};
use std::collections::HashMap;

// lehenetsitako fotoi-muga, `~L{n}` muga gabe idazten denean (QMODE-ren lehenetsi bera)
const DEFAULT_CUTOFF: usize = 8;

// bit-erregistroak zenbaki gisa inprimatzeko erabiltzen diren erregistroak, R31-etik behera
const SCRATCH_TOP: usize = 31;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Qubits,
    Bits,
    Modes,
}

impl Kind {
    fn prefix(self) -> &'static str {
        match self {
            Kind::Qubits => "q",
            Kind::Bits => "c",
            Kind::Modes => "o",
        }
    }
}

#[derive(Clone, Debug)]
struct Register {
    kind: Kind,
    start: usize,
    size: usize,
}

// iturburuko posizio bat (1etik hasita)
#[derive(Clone, Copy, Debug, Default)]
struct Pos {
    line: u32,
    col: u32,
}

// sortutako QOA lerro baten jatorria glifo-iturburuan
#[derive(Clone, Copy, Debug)]
struct Origin {
    pos: Pos,
    len: u32,
}

pub struct Expanded {
    // QOA iturburu arrunta
    pub source: String,
    origins: Vec<Origin>,
}

impl Expanded {
    // QOA lerroko diagnostiko bat glifo-iturburuko adierazpenera
    fn remap(&self, d: Diagnostic) -> Diagnostic {
        match self.origins.get((d.line as usize).wrapping_sub(1)) {
            Some(o) => Diagnostic {
                line: o.pos.line,
                col: o.pos.col,
                len: o.len,
                message: d.message,
                help: d.help,
            },
            None => d,
        }
    }
}

pub fn assemble(src: &str, name: &str) -> Result<Program, Vec<Diagnostic>> {
    let ex = expand_file(src, name)?;
    let mut prog = asm::assemble(&ex.source, name)
        .map_err(|ds| ds.into_iter().map(|d| ex.remap(d)).collect::<Vec<_>>())?;
    // exekuzio-garaiko mezuek (erroreak, DUMPSTATE) glifo-iturburuko lerroa aipa dezaten
    for l in prog.lines.iter_mut() {
        if let Some(o) = ex.origins.get((*l as usize).wrapping_sub(1)) {
            *l = o.pos.line;
        }
    }
    Ok(prog)
}

// eragigai bat, erregistroak ebatzi aurretik
#[derive(Clone, Debug)]
enum Operand {
    // izendun erregistroa, aukerako tartearekin (barne-mugak)
    Named {
        name: String,
        range: Option<(usize, usize)>,
        pos: Pos,
    },
    // #k: R<k> erregistro klasikoa
    Reg(usize),
    // beste edozer (q3, c1, pi/4, 5...): bere horretan
    Raw(String),
    // parentesi arteko balioa, `(pi/2)`: parentesirik gabe, erregistro edo zenbaki baten lekuan izan ezik
    Value(String),
}

#[derive(Clone, Debug)]
enum Stmt {
    // agindu bat: izena, testuak, eragigaiak, parametroak, % argumentuak, $> helburuak
    Op {
        name: String,
        texts: Vec<String>,
        operands: Vec<Operand>,
        params: Vec<String>,
        args: Vec<Operand>,
        dests: Vec<Operand>,
        // 0.4.6: eragigaien ondorengo testuak (`STX#1"testua"`, `VL/Q[0]"oharra"`), aginduaren ordenan
        after: Vec<String>,
    },
    Label(String),
    Jump(String),
    JumpIf {
        one: bool,
        value: Operand,
        label: String,
    },
    Call(String),
    // dei klasikoa (ABIa): argumentuak datu-pilan, eskuinetik ezkerrera (deitutakoak lehen argumentua
    // ateratzen du lehenik), deia, eta emaitza pilatik helmugara. dei-pila bereizia da
    ClassicalCall {
        target: String,
        args: Vec<Operand>,
        dest: Option<Operand>,
    },
    Return,
    LoopStart(String),
    // `}`: begizta, baldintzazko bloke edo bloke izendun baten amaiera
    LoopEnd,
    // 0.4.5: `??x{ ... }` (x 1 bada) eta `!!x{ ... }` (x 0 bada)
    IfStart {
        one: bool,
        value: Operand,
    },
    // 0.4.5: `izena::{ ... }` bloke izendun bat, eta `::izena` haren lekuan
    Define(String),
    Use(String),
    // adierazpen batek sortzen dituen hainbat agindu, ordenan
    Many(Vec<Stmt>),
    Verbatim(String),
    // erregistro-adierazpen hutsa (ez du QOA lerrorik sortzen)
    Nothing,
}

struct Parser {
    chars: Vec<char>,
    i: usize,
    line: u32,
    col: u32,
    regs: HashMap<String, Register>,
    order: Vec<String>,
    counts: [usize; 3],
    cutoff: Option<(usize, Pos)>,
    diags: Vec<Diagnostic>,
    // `""`-k errepikatzen duen azken agindua: izena, parametroak, testuak
    last_op: Option<(String, Vec<String>, Vec<String>)>,
    // ??x{ eta !!x{-ren balioan `{` blokea da, ez erregistro-adierazpen bat
    no_declare: bool,
}

fn diag(pos: Pos, len: u32, message: String) -> Diagnostic {
    Diagnostic {
        line: pos.line,
        col: pos.col,
        len: len.max(1),
        message,
        help: None,
    }
}

impl Parser {
    fn new(src: &str) -> Parser {
        Parser {
            chars: src.chars().collect(),
            i: 0,
            line: 1,
            col: 1,
            regs: HashMap::new(),
            last_op: None,
            no_declare: false,
            order: Vec::new(),
            counts: [0; 3],
            cutoff: None,
            diags: Vec::new(),
        }
    }

    fn pos(&self) -> Pos {
        Pos {
            line: self.line,
            col: self.col,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }

    fn peek_at(&self, k: usize) -> Option<char> {
        self.chars.get(self.i + k).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.i).copied()?;
        self.i += 1;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn starts_with(&self, s: &str) -> bool {
        s.chars()
            .enumerate()
            .all(|(k, c)| self.peek_at(k) == Some(c))
    }

    // hutsuneak eta iruzkinak: `//` lerro-amaierara arte, eta `;`-z hasten den lerro bat
    fn skip_blank(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => {
                    self.bump();
                }
                Some('/') if self.peek_at(1) == Some('/') => self.skip_line(),
                Some(';') if self.at_line_start() => self.skip_line(),
                _ => return,
            }
        }
    }

    fn at_line_start(&self) -> bool {
        self.chars[..self.i]
            .iter()
            .rev()
            .take_while(|c| **c != '\n')
            .all(|c| c.is_whitespace())
    }

    fn skip_line(&mut self) {
        while let Some(c) = self.peek() {
            if c == '\n' {
                return;
            }
            self.bump();
        }
    }

    fn skip_spaces(&mut self) {
        while matches!(self.peek(), Some(c) if c == ' ' || c == '\t') {
            self.bump();
        }
    }

    fn ident(&mut self) -> Option<String> {
        let c = self.peek()?;
        if !(c.is_ascii_alphabetic() || c == '_') {
            return None;
        }
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        Some(s)
    }

    fn number(&mut self) -> Option<usize> {
        let start = self.i;
        let text: String = if self.starts_with("0x") || self.starts_with("0X") {
            self.bump();
            self.bump();
            let mut s = String::new();
            while let Some(c) = self.peek().filter(|c| c.is_ascii_hexdigit()) {
                s.push(c);
                self.bump();
            }
            return usize::from_str_radix(&s, 16).ok().or_else(|| {
                self.i = start;
                None
            });
        } else {
            let mut s = String::new();
            while let Some(c) = self.peek().filter(|c| c.is_ascii_digit()) {
                s.push(c);
                self.bump();
            }
            s
        };
        text.parse().ok()
    }

    fn error(&mut self, pos: Pos, len: u32, message: String) {
        self.diags.push(diag(pos, len, message));
    }

    // adierazpenaren amaierara (`;`) salto egin, errore baten ondoren jarraitzeko
    fn recover(&mut self) {
        while let Some(c) = self.peek() {
            self.bump();
            if c == ';' {
                return;
            }
        }
    }

    fn string(&mut self) -> Option<String> {
        let start = self.pos();
        self.bump();
        let mut s = String::from("\"");
        loop {
            match self.bump() {
                Some('"') => {
                    s.push('"');
                    return Some(s);
                }
                Some('\\') => {
                    s.push('\\');
                    if let Some(c) = self.bump() {
                        s.push(c);
                    }
                }
                Some('\n') | None => {
                    self.error(
                        start,
                        1,
                        ts!(
                            "this text is never closed with \"",
                            "testu hau ez da inoiz \"-rekin ixten"
                        )
                        .into(),
                    );
                    return None;
                }
                Some(c) => s.push(c),
            }
        }
    }

    // parentesi arteko adierazpena (habiaratuak barne), parentesirik gabe
    fn paren(&mut self) -> Option<String> {
        let start = self.pos();
        self.bump();
        let mut depth = 1;
        let mut s = String::new();
        while let Some(c) = self.bump() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(s.trim().to_string());
                    }
                }
                ';' | '\n' => break,
                _ => {}
            }
            s.push(c);
        }
        self.error(
            start,
            1,
            ts!("this ( is never closed", "( hau ez da inoiz ixten").into(),
        );
        None
    }

    // erregistro bat adierazi: `at` emanez gero elementu horretan hasten da (`R{3}[0x4]`: q4..q6), bestela
    // mota bereko aurreko erregistroen ondoren
    fn declare(
        &mut self,
        name: &str,
        kind: Kind,
        size: usize,
        at: Option<usize>,
        pos: Pos,
    ) -> bool {
        if let Some(r) = self.regs.get(name) {
            if r.kind != kind || r.size != size || at.is_some_and(|a| a != r.start) {
                self.error(
                    pos,
                    name.len() as u32,
                    t!(
                        "{} is already declared with a different size or kind",
                        "{} lehendik adierazita dago beste tamaina edo mota batekin",
                        name
                    ),
                );
                return false;
            }
            return true;
        }
        if size == 0 {
            self.error(
                pos,
                name.len() as u32,
                t!(
                    "{} needs at least one element",
                    "{}-k elementu bat behar du gutxienez",
                    name
                ),
            );
            return false;
        }
        // QOA arruntaren QREG/CREG/OREG-en muga berak: 65536 elementu, eta lehena 0..65535
        if size > crate::program::MAX_BITS as usize {
            self.error(
                pos,
                name.len() as u32,
                t!(
                    "{} has {} elements, more than the 65536 a register can hold",
                    "{}-k {} elementu ditu, erregistro batek izan ditzakeen 65536 baino gehiago",
                    name,
                    size
                ),
            );
            return false;
        }
        if at.is_some_and(|a| a >= crate::program::MAX_BITS as usize) {
            self.error(
                pos,
                name.len() as u32,
                t!(
                    "{} starts at {}, but a register starts at 0 to 65535",
                    "{} {}-n hasten da, baina erregistro bat 0 eta 65535 artean hasten da",
                    name,
                    at.unwrap_or(0)
                ),
            );
            return false;
        }
        let k = kind as usize;
        let start = at.unwrap_or(self.counts[k]);
        let clash = self
            .regs
            .iter()
            .find(|(_, r)| r.kind == kind && start < r.start + r.size && r.start < start + size);
        if let Some((other, r)) = clash {
            let msg = t!(
                "{} at {}..{} overlaps {} ({}..{})",
                "{} ({}..{}) {}-rekin gainjartzen da ({}..{})",
                name,
                start,
                start + size - 1,
                other,
                r.start,
                r.start + r.size - 1
            );
            self.error(pos, name.len() as u32, msg);
            return false;
        }
        self.regs
            .insert(name.to_string(), Register { kind, start, size });
        self.order.push(name.to_string());
        self.counts[k] = self.counts[k].max(start + size);
        true
    }

    // adierazpenaren ondoko `[0x4]`: erregistroaren kokapena (lehen elementua), zenbaki bakarra
    fn placement(&mut self) -> Option<Option<usize>> {
        if self.peek() != Some('[') {
            return Some(None);
        }
        let pos = self.pos();
        let (a, b) = self.index()?;
        if a != b {
            self.error(
                pos,
                1,
                ts!(
                    "a register's location is one number, like [0x4]",
                    "erregistro baten kokapena zenbaki bakarra da, [0x4] bezala"
                )
                .into(),
            );
            return None;
        }
        Some(Some(a))
    }

    // `{n}` tamaina (eta modu-erregistroetan aukerako `%c` muga)
    fn size(&mut self, kind: Kind) -> Option<usize> {
        let pos = self.pos();
        self.bump();
        let n = self.number();
        if self.peek() != Some('}') || n.is_none() {
            self.error(
                pos,
                1,
                ts!(
                    "expected a size like {3}",
                    "{3} bezalako tamaina bat espero zen"
                )
                .into(),
            );
            return None;
        }
        self.bump();
        if kind == Kind::Modes && self.peek() == Some('%') {
            let cpos = self.pos();
            self.bump();
            match self.number() {
                Some(c) => match self.cutoff {
                    Some((old, _)) if old != c => self.error(
                        cpos,
                        1,
                        t!(
                            "every mode register must use the same cutoff (an earlier one uses {})",
                            "modu-erregistro guztiek muga bera erabili behar dute (aurreko batek {} erabiltzen du)",
                            old
                        ),
                    ),
                    _ => self.cutoff = Some((c, cpos)),
                },
                None => self.error(cpos, 1, ts!("expected a cutoff after %", "muga bat espero zen % ondoren").into()),
            }
        }
        n
    }

    // `[k]` edo `[a..b]`
    fn index(&mut self) -> Option<(usize, usize)> {
        let pos = self.pos();
        self.bump();
        let a = self.number();
        let b = if self.starts_with("..") {
            self.bump();
            self.bump();
            self.number()
        } else {
            a
        };
        match (a, b, self.peek()) {
            (Some(a), Some(b), Some(']')) if a <= b => {
                self.bump();
                Some((a, b))
            }
            _ => {
                self.error(
                    pos,
                    1,
                    ts!(
                        "expected an index like [2] or [0..3]",
                        "[2] edo [0..3] bezalako indize bat espero zen"
                    )
                    .into(),
                );
                None
            }
        }
    }

    fn operand(&mut self) -> Option<Operand> {
        self.skip_spaces();
        let pos = self.pos();
        match self.peek()? {
            '#' | '~' => {
                let kind = if self.bump() == Some('#') {
                    Kind::Bits
                } else {
                    Kind::Modes
                };
                if kind == Kind::Bits && self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    return self.number().map(Operand::Reg);
                }
                let Some(name) = self.ident() else {
                    self.error(
                        pos,
                        1,
                        ts!(
                            "expected a name after # or ~",
                            "izen bat espero zen # edo ~ ondoren"
                        )
                        .into(),
                    );
                    return None;
                };
                self.named(name, Some(kind), pos)
            }
            '(' => self.paren().map(Operand::Value),
            c if c.is_ascii_alphabetic() || c == '_' => {
                let name = self.ident()?;
                self.named(name, None, pos)
            }
            c if c.is_ascii_digit() || c == '-' || c == '.' => {
                let mut s = String::new();
                while let Some(c) = self
                    .peek()
                    .filter(|c| c.is_ascii_alphanumeric() || "-.+".contains(*c))
                {
                    s.push(c);
                    self.bump();
                }
                Some(Operand::Raw(s))
            }
            _ => {
                self.error(
                    pos,
                    1,
                    ts!("expected an operand", "eragigai bat espero zen").into(),
                );
                None
            }
        }
    }

    fn named(&mut self, name: String, kind: Option<Kind>, pos: Pos) -> Option<Operand> {
        if self.peek() == Some('{') && !self.no_declare {
            let size = self.size(kind.unwrap_or(Kind::Qubits))?;
            let at = self.placement()?;
            if !self.declare(&name, kind.unwrap_or(Kind::Qubits), size, at, pos) {
                return None;
            }
        }
        // 0.4.6: # edo ~ duten izenetan indizea kortxeterik gabe: `#C0` = `#C[0]`, `~L0..3` = `~L[0..3]`
        // (lehen errore bat zen: C0 ez dago adierazita). `Q0` ez: QOA arruntean q0 qubita da jada
        if kind.is_some()
            && !self.regs.contains_key(&name)
            && self.peek() != Some('[')
            && let Some((base, a)) = split_index(&name, &self.regs)
        {
            let b = if self.starts_with("..") && self.peek_at(2).is_some_and(|c| c.is_ascii_digit())
            {
                self.bump();
                self.bump();
                self.number()?
            } else {
                a
            };
            if a > b {
                self.error(
                    pos,
                    name.len() as u32,
                    ts!(
                        "expected an index like #C2 or #C0..3",
                        "#C2 edo #C0..3 bezalako indize bat espero zen"
                    )
                    .into(),
                );
                return None;
            }
            return Some(Operand::Named {
                name: base,
                range: Some((a, b)),
                pos,
            });
        }
        let range = if self.peek() == Some('[') {
            Some(self.index()?)
        } else {
            None
        };
        if !self.regs.contains_key(&name) && kind.is_none() && range.is_none() {
            // q3, c1, o0, R2, pi...: QOA arruntaren eragigaiak, bere horretan
            return Some(Operand::Raw(name));
        }
        Some(Operand::Named { name, range, pos })
    }

    // itsatsitako agindu bat: izena ez bada agindu bat, aurrizki luzeena agindu bat bada eta gainerakoa deklaratutako
    // erregistro bat edo erregistro arrunt bat (q0, c1, o2, R3), bi zati: (agindua, lehen eragigaia). eragigaiak
    // huts egiten badu (tarte okerra), Some(None)
    fn unglue(&mut self, name: String, pos: Pos) -> (String, Option<Option<Operand>>) {
        if crate::isa::lookup(mnemonic(&name)).is_some() || name == "PR" || !name.is_ascii() {
            return (name, None);
        }
        for k in (1..name.len()).rev() {
            let (head, rest) = name.split_at(k);
            if crate::isa::lookup(mnemonic(head)).is_none() && head != "PR" {
                continue;
            }
            let plain = {
                let mut c = rest.chars();
                c.next().is_some_and(|l| "qQcCoOrR".contains(l))
                    && rest.len() > 1
                    && c.all(|d| d.is_ascii_digit())
            };
            if self.regs.contains_key(rest) || plain {
                let at = Pos {
                    line: pos.line,
                    col: pos.col + k as u32,
                };
                let op = self.named(rest.to_string(), None, at);
                return (head.to_string(), Some(op));
            }
        }
        (name, None)
    }

    // aginduak testuen ondoren eragigairik hartzen duen (ezezaguna bada, bai: errorea lehen bezala)
    fn takes_operands(&self, name: &str, texts: usize) -> bool {
        match crate::isa::lookup(mnemonic(name)) {
            Some(spec) => spec
                .operand_specs()
                .iter()
                .skip(texts)
                .any(|o| o.kind != 's'),
            None => name != "PR",
        }
    }

    // lehen eragigaia jada irakurrita: gainerakoak komaz bereizita
    fn more_operands(&mut self, first: Operand) -> Option<Vec<Operand>> {
        let mut out = vec![first];
        loop {
            self.skip_spaces();
            if self.peek() != Some(',') {
                return Some(out);
            }
            self.bump();
            out.push(self.operand()?);
        }
    }

    fn operands(&mut self) -> Option<Vec<Operand>> {
        let mut out = vec![self.operand()?];
        loop {
            self.skip_spaces();
            if self.peek() != Some(',') {
                return Some(out);
            }
            self.bump();
            out.push(self.operand()?);
        }
    }

    fn label_name(&mut self) -> Option<String> {
        self.skip_spaces();
        let pos = self.pos();
        if self.peek().is_some_and(|c| c.is_ascii_digit()) {
            let n = self.number();
            if n.is_none() {
                self.error(
                    pos,
                    1,
                    ts!(
                        "expected an address like 0x10",
                        "0x10 bezalako helbide bat espero zen"
                    )
                    .into(),
                );
            }
            return n.map(address);
        }
        let name = self.ident();
        if name.is_none() {
            self.error(
                pos,
                1,
                ts!("expected a label name", "etiketa-izen bat espero zen").into(),
            );
        }
        name
    }

    fn statement(&mut self) -> Option<Stmt> {
        let pos = self.pos();
        if self.peek() == Some('`') {
            self.bump();
            let mut s = String::new();
            loop {
                match self.bump() {
                    Some('`') => return Some(Stmt::Verbatim(s)),
                    Some('\n') | None => {
                        self.error(
                            pos,
                            1,
                            ts!("this ` is never closed", "` hau ez da inoiz ixten").into(),
                        );
                        return None;
                    }
                    Some(c) => s.push(c),
                }
            }
        }
        if self.starts_with("<<>>") {
            (0..4).for_each(|_| {
                self.bump();
            });
            return Some(Stmt::Op {
                name: "DUMPSTATE".into(),
                texts: vec![],
                operands: vec![],
                params: vec![],
                args: vec![],
                dests: vec![],
                after: vec![],
            });
        }
        if self.starts_with("<<#>>") {
            (0..5).for_each(|_| {
                self.bump();
            });
            return Some(Stmt::Op {
                name: "DUMPREGS".into(),
                texts: vec![],
                operands: vec![],
                params: vec![],
                args: vec![],
                dests: vec![],
                after: vec![],
            });
        }
        match self.peek()? {
            '*' => {
                self.bump();
                let count = match self.operand()? {
                    Operand::Reg(k) => format!("R{}", k),
                    Operand::Raw(s) | Operand::Value(s) => s,
                    Operand::Named { pos, .. } => {
                        self.error(
                            pos,
                            1,
                            ts!(
                                "a loop count is a number or a register like #2",
                                "begizta-kopurua zenbaki bat edo #2 bezalako erregistro bat da"
                            )
                            .into(),
                        );
                        return None;
                    }
                };
                self.skip_spaces();
                if self.peek() != Some('{') {
                    let p = self.pos();
                    self.error(
                        p,
                        1,
                        ts!(
                            "expected { after the loop count",
                            "{ espero zen begizta-kopuruaren ondoren"
                        )
                        .into(),
                    );
                    return None;
                }
                self.bump();
                Some(Stmt::LoopStart(count))
            }
            '}' => {
                self.bump();
                Some(Stmt::LoopEnd)
            }
            '>' => {
                self.bump();
                Some(Stmt::Jump(self.label_name()?))
            }
            '?' | '!' if self.peek_at(1) == self.peek() => {
                // 0.4.5: `??x{ ... }` x 1 bada, `!!x{ ... }` x 0 bada
                let one = self.bump() == Some('?');
                self.bump();
                self.no_declare = true;
                let value = self.operand();
                self.no_declare = false;
                let value = value?;
                self.skip_spaces();
                if self.peek() != Some('{') {
                    let p = self.pos();
                    self.error(
                        p,
                        1,
                        ts!(
                            "expected { and the statements to run",
                            "{ eta exekutatzeko adierazpenak espero ziren"
                        )
                        .into(),
                    );
                    return None;
                }
                self.bump();
                Some(Stmt::IfStart { one, value })
            }
            '#' if self.peek_at(1).is_some_and(|c| c.is_ascii_digit()) => self.assignment(),
            ':' if self.peek_at(1) == Some(':') => {
                self.bump();
                self.bump();
                let Some(name) = self.ident() else {
                    self.error(
                        pos,
                        2,
                        ts!(
                            "expected the name of a block after ::",
                            "bloke baten izena espero zen :: ondoren"
                        )
                        .into(),
                    );
                    return None;
                };
                Some(Stmt::Use(name))
            }
            '"' if self.peek_at(1) == Some('"') => {
                // 0.4.5: `""` aurreko agindua berriz, eragigai berriekin
                self.bump();
                self.bump();
                let Some((name, params, texts)) = self.last_op.clone() else {
                    self.error(
                        pos,
                        2,
                        ts!(
                            "\"\" repeats the instruction before it, and there is none",
                            "\"\"-k aurreko agindua errepikatzen du, eta ez dago bat ere"
                        )
                        .into(),
                    );
                    return None;
                };
                let glued = match self.peek() {
                    Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                        let at = self.pos();
                        let reg = self.ident()?;
                        Some(self.named(reg, None, at))
                    }
                    _ => None,
                };
                self.op_tail(name, params, texts, glued)
            }
            '?' | '!' => {
                let one = self.bump() == Some('?');
                let value = self.operand()?;
                self.skip_spaces();
                if self.peek() != Some('>') {
                    let p = self.pos();
                    self.error(
                        p,
                        1,
                        ts!("expected > and a label", "> eta etiketa bat espero ziren").into(),
                    );
                    return None;
                }
                self.bump();
                Some(Stmt::JumpIf {
                    one,
                    value,
                    label: self.label_name()?,
                })
            }
            '&' => {
                self.bump();
                Some(Stmt::Call(self.label_name()?))
            }
            '^' => {
                self.bump();
                Some(Stmt::Return)
            }
            '#' | '~' => {
                // adierazpen hutsa: #C{3}, ~L{2}%6
                match self.operand()? {
                    Operand::Named { range: None, .. } => Some(Stmt::Nothing),
                    _ => {
                        self.error(
                            pos,
                            1,
                            ts!(
                                "expected a declaration like #C{3} or ~L{2}%6",
                                "#C{3} edo ~L{2}%6 bezalako adierazpen bat espero zen"
                            )
                            .into(),
                        );
                        None
                    }
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => self.op_or_label(pos),
            // helbide bat etiketa gisa: `0x10:` (CC[0x10] bidez deitzen da)
            c if c.is_ascii_digit() => {
                let n = self.number();
                if let (Some(n), Some(':')) = (n, self.peek()) {
                    self.bump();
                    return Some(Stmt::Label(address(n)));
                }
                self.error(
                    pos,
                    1,
                    ts!(
                        "a number starts a statement only as an address label, like 0x10:",
                        "zenbaki batek helbide-etiketa gisa bakarrik hasten du adierazpen bat, 0x10: bezala"
                    )
                    .into(),
                );
                None
            }
            _ => {
                self.error(
                    pos,
                    1,
                    ts!(
                        "this does not start a statement",
                        "honek ez du adierazpen bat hasten"
                    )
                    .into(),
                );
                None
            }
        }
    }

    fn op_or_label(&mut self, pos: Pos) -> Option<Stmt> {
        let name = self.ident()?;
        match self.peek() {
            Some(':') if self.peek_at(1) != Some(':') => {
                self.bump();
                return Some(Stmt::Label(name));
            }
            Some('{') => {
                let size = self.size(Kind::Qubits)?;
                let at = self.placement()?;
                self.declare(&name, Kind::Qubits, size, at, pos);
                return Some(Stmt::Nothing);
            }
            _ => {}
        }
        if name == "CC" {
            return self.classical_call();
        }
        // adierazitako erregistro bat adierazpen baten hasieran: ziurrenik koma bat falta da (`CX/Q[0] Q[1]`)
        if self.regs.contains_key(&name)
            && crate::isa::lookup(mnemonic(&name)).is_none()
            && !self.starts_with("::")
        {
            self.error(
                pos,
                name.len() as u32,
                t!(
                    "{} is a register, not an instruction; is a , missing before it?",
                    "{} erregistro bat da, ez agindu bat; koma bat falta al da haren aurretik?",
                    name
                ),
            );
            return None;
        }
        if self.starts_with("::") {
            // 0.4.5: `izena::{ ... }` bloke izendun bat
            self.bump();
            self.bump();
            self.skip_spaces();
            if self.peek() != Some('{') {
                let p = self.pos();
                self.error(
                    p,
                    1,
                    ts!(
                        "expected { and the statements of the block",
                        "{ eta blokearen adierazpenak espero ziren"
                    )
                    .into(),
                );
                return None;
            }
            self.bump();
            return Some(Stmt::Define(name));
        }
        if name == "LXP" && self.peek() == Some('@') {
            // 0.4.5: `LXP@(helbidea)<=balioa`: memoria-gelaxka batean idatzi
            let addr = self.pointer()?;
            self.skip_spaces();
            if !self.starts_with("<=") {
                let p = self.pos();
                self.error(
                    p,
                    1,
                    ts!(
                        "expected <= and the value to store",
                        "<= eta gordetzeko balioa espero ziren"
                    )
                    .into(),
                );
                return None;
            }
            self.bump();
            self.bump();
            let value = self.operand()?;
            return Some(Self::op("STORE", vec![value, addr]));
        }
        // 0.4.5: agindua bere lehen eragigaiari itsatsita (`HQ`, `CXA,B`, `Mq0`)
        let (name, glued) = self.unglue(name, pos);
        let mut params = Vec::new();
        if glued.is_none() && self.peek() == Some('(') {
            let p = self.paren()?;
            params.extend(split_params(&p));
        }
        let mut texts = Vec::new();
        while glued.is_none() && self.peek() == Some('"') {
            texts.push(self.string()?);
        }
        self.op_tail(name, params, texts, glued)
    }

    // agindu baten gainerakoa, izena, parametroak eta testuak irakurrita: eragigaiak, % argumentuak eta helburua
    fn op_tail(
        &mut self,
        name: String,
        params: Vec<String>,
        texts: Vec<String>,
        glued: Option<Option<Operand>>,
    ) -> Option<Stmt> {
        let mut operands = Vec::new();
        self.skip_spaces();
        if let Some(first) = glued {
            operands = self.more_operands(first?)?;
        } else if self.peek() == Some('/') {
            self.bump();
            operands = self.operands()?;
        } else if matches!(self.peek(), Some('#' | '~')) && self.takes_operands(&name, texts.len())
        {
            // 0.4.5: # edo ~ eragigai baten aurretik `/` gabe (`PUSH#1`, `ADD#4,#1,#2`). eragigairik hartzen ez
            // duen agindu baten ondoren (`RA`, `PRINT"x"`), # hurrengo adierazpena da (0.4.6, `;` gabe)
            operands = self.operands()?;
        }
        // 0.4.6: aginduaren testua eragigaien ondoren doanean (STORETEXT, VERBOSELOG): `STX#1"testua"`
        let mut after = Vec::new();
        if !operands.is_empty() && texts.is_empty() && self.text_after(&name) {
            while self.peek() == Some('"') {
                after.push(self.string()?);
            }
        }
        let mut args = Vec::new();
        let mut dests = Vec::new();
        loop {
            self.skip_spaces();
            if self.peek() == Some('%') {
                self.bump();
                args.push(self.operand()?);
            } else if self.starts_with("$>") {
                self.bump();
                self.bump();
                dests = self.operands()?;
            } else if self.peek() == Some('.')
                && self.peek_at(1) != Some('.')
                && !operands.is_empty()
            {
                // 0.4.5: `.` `$>`-ren ordez (`M/Q.#C`, `Mq0.c0`)
                self.bump();
                dests = self.operands()?;
            } else {
                break;
            }
        }
        let name = name.to_ascii_uppercase();
        self.last_op = Some((name.clone(), params.clone(), texts.clone()));
        Some(Stmt::Op {
            name,
            texts,
            operands,
            params,
            args,
            dests,
            after,
        })
    }

    // aginduaren testua beste eragigai baten ondoren doan (STORETEXT helbidea testua, VERBOSELOG qubita testua)
    fn text_after(&self, name: &str) -> bool {
        crate::isa::lookup(mnemonic(name)).is_some_and(|spec| {
            let k = spec.operand_specs();
            k.len() > 1 && k.last().is_some_and(|o| o.kind == 's') && k[0].kind != 's'
        })
    }

    // uneko posiziotik hasten den adierazpen numeriko oso baten luzera (karaktereetan), 0 osoa ez bada:
    // zenbakiak, izenak, funtzio-deiak, parentesiak, + - * / ^ eta hutsuneak; bi eragigai jarraian (`2 pi`,
    // `)H`) adierazpenaren amaiera dira
    fn expr_len(&self) -> usize {
        let c = &self.chars[self.i..];
        let (mut j, mut depth, mut operand) = (0usize, 0i32, true);
        let mut end = 0;
        while j < c.len() {
            let ch = c[j];
            if ch == ' ' || ch == '\t' {
                j += 1;
                continue;
            }
            if operand {
                if ch.is_ascii_digit() || ch == '.' {
                    while j < c.len() && (c[j].is_ascii_digit() || c[j] == '.') {
                        j += 1;
                    }
                    if j < c.len() && matches!(c[j], 'e' | 'E') {
                        let sign = j + 1 < c.len() && matches!(c[j + 1], '+' | '-');
                        let d = j + 1 + usize::from(sign);
                        if d < c.len() && c[d].is_ascii_digit() {
                            j = d;
                            while j < c.len() && c[j].is_ascii_digit() {
                                j += 1;
                            }
                        }
                    }
                    operand = false;
                } else if ch.is_ascii_alphabetic() || ch == '_' {
                    while j < c.len() && (c[j].is_ascii_alphanumeric() || c[j] == '_') {
                        j += 1;
                    }
                    if j < c.len() && c[j] == '(' {
                        depth += 1;
                        j += 1;
                    } else {
                        operand = false;
                    }
                } else if ch == '(' {
                    depth += 1;
                    j += 1;
                } else if ch == '+' || ch == '-' {
                    j += 1;
                } else {
                    break;
                }
            } else if "+-*/^".contains(ch) {
                operand = true;
                j += 1;
            } else if ch == ')' && depth > 0 {
                depth -= 1;
                j += 1;
            } else {
                break;
            }
            if !operand && depth == 0 {
                end = j;
            }
        }
        end
    }

    // esleipen baten eragigaia: zenbakiak zorrotz irakurrita (`2+1` ez da zenbaki bat: 2 eta gero + eragilea),
    // gainerakoa operand() bezala
    fn assign_operand(&mut self) -> Option<Operand> {
        self.skip_spaces();
        let starts_number = match self.peek() {
            Some(c) if c.is_ascii_digit() || c == '.' => true,
            Some('-') => self
                .peek_at(1)
                .is_some_and(|c| c.is_ascii_digit() || c == '.'),
            _ => false,
        };
        if !starts_number {
            return self.operand();
        }
        let mut t = String::new();
        if self.peek() == Some('-') {
            t.push('-');
            self.bump();
        }
        if self.starts_with("0x") || self.starts_with("0X") {
            t.push_str("0x");
            self.bump();
            self.bump();
            let pos = self.pos();
            let mut digits = String::new();
            while let Some(c) = self.peek().filter(|c| c.is_ascii_hexdigit()) {
                digits.push(c);
                self.bump();
            }
            // QOA arruntaren adierazpenek hamartarra irakurtzen dute
            let Ok(v) = u64::from_str_radix(&digits, 16) else {
                self.error(
                    pos,
                    1,
                    ts!(
                        "expected a hexadecimal number after 0x",
                        "zenbaki hamaseitar bat espero zen 0x ondoren"
                    )
                    .into(),
                );
                return None;
            };
            let neg = t.starts_with('-');
            return Some(Operand::Raw(format!("{}{}", if neg { "-" } else { "" }, v)));
        }
        while let Some(c) = self.peek().filter(|c| c.is_ascii_digit() || *c == '.') {
            t.push(c);
            self.bump();
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            let sign = matches!(self.peek_at(1), Some('+' | '-'));
            let digit_at = if sign { 2 } else { 1 };
            if self.peek_at(digit_at).is_some_and(|c| c.is_ascii_digit()) {
                for _ in 0..digit_at {
                    t.push(self.bump().expect("e"));
                }
                while let Some(c) = self.peek().filter(|c| c.is_ascii_digit()) {
                    t.push(c);
                    self.bump();
                }
            }
        }
        Some(Operand::Raw(t))
    }

    // esleipen baten eragigaiak: bitak (`#C[0]`, `#C`, `c0`) zenbaki bihurtzen dira aldi baterako erregistro batean
    // (R31, gero R30), PF-k bezala; gainerakoak bere horretan
    fn computed(&mut self, name: &str, d: Operand, srcs: Vec<Operand>) -> Option<Stmt> {
        let mut before = Vec::new();
        let mut ops = vec![d];
        let mut scratch = 31;
        for a in srcs {
            let bits = matches!(&a, Operand::Named { .. })
                || matches!(&a, Operand::Raw(r) if r.len() > 1 && r.starts_with(['c', 'C']) && r[1..].bytes().all(|b| b.is_ascii_digit()));
            if bits {
                match elements(&self.regs, &a) {
                    Ok(els) => {
                        before.push(Stmt::Verbatim(format!(
                            "PACKBITS R{} {}",
                            scratch,
                            els.join(" ")
                        )));
                        ops.push(Operand::Reg(scratch));
                        scratch -= 1;
                    }
                    Err(e) => {
                        self.diags.push(e);
                        return None;
                    }
                }
            } else {
                ops.push(a);
            }
        }
        let op = Self::op(name, ops);
        Some(if before.is_empty() {
            op
        } else {
            before.push(op);
            Stmt::Many(before)
        })
    }

    fn op(name: &str, operands: Vec<Operand>) -> Stmt {
        Stmt::Op {
            name: name.into(),
            texts: vec![],
            operands,
            params: vec![],
            args: vec![],
            dests: vec![],
            after: vec![],
        }
    }

    // `@(helbidea)`: memoria-gelaxka baten helbidea (zenbaki, adierazpen edo erregistro bat)
    fn pointer(&mut self) -> Option<Operand> {
        let pos = self.pos();
        self.bump();
        if self.peek() != Some('(') {
            self.error(
                pos,
                1,
                ts!(
                    "a memory cell is written LXP@(address), like LXP@(100) or LXP@(#2)",
                    "memoria-gelaxka bat LXP@(helbidea) idazten da, LXP@(100) edo LXP@(#2) bezala"
                )
                .into(),
            );
            return None;
        }
        let inner = self.paren()?;
        let inner = inner.trim();
        // erregistro bat: #2 edo R2
        if let Some(k) = inner
            .strip_prefix('#')
            .or_else(|| inner.strip_prefix(['R', 'r']))
            .and_then(|d| d.parse::<usize>().ok())
        {
            return Some(Operand::Reg(k));
        }
        Some(Operand::Value(inner.to_string()))
    }

    // 0.4.5: `#2<=balioa`, `#2<=a+b` (+ - * / ^ & | << >>), `#2<=sqrt(a)` (sqrt exp ln sin cos tan not),
    // `#2<=LXP@(helbidea)`, `#2_+a` (gehitu) eta `#2-=>a` (kendu)
    fn assignment(&mut self) -> Option<Stmt> {
        self.bump();
        let dst = self.number()?;
        let d = Operand::Reg(dst);
        self.skip_spaces();
        if self.starts_with("_+") || self.starts_with("-=>") {
            let add = self.starts_with("_+");
            let n = if add { 2 } else { 3 };
            (0..n).for_each(|_| {
                self.bump();
            });
            let v = self.assign_operand()?;
            return self.computed(if add { "ADD" } else { "SUB" }, d.clone(), vec![d, v]);
        }
        if !self.starts_with("<=") {
            let p = self.pos();
            self.error(
                p,
                1,
                ts!(
                    "after a register, expected <= (set it), _+ (add to it) or -=> (subtract from it)",
                    "erregistro baten ondoren, <= (ezarri), _+ (gehitu) edo -=> (kendu) espero zen"
                )
                .into(),
            );
            return None;
        }
        self.bump();
        self.bump();
        self.skip_spaces();
        // erregistrorik ez duen eskuin aldea adierazpen konstante oso bat da: `#7<=2*pi/3` -> REGSET R7 2*pi/3.
        // adierazpena bere gramatikak uzten duen arte irakurtzen da, `;` gabe hurrengo adierazpena hasten den lekuan
        // gelditzeko (`#7<=pi/2#8<=1`)
        let rest: String = self.chars[self.i..self.i + self.expr_len()]
            .iter()
            .collect();
        let register_like = rest
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|w| {
                w.len() > 1
                    && w.starts_with(['R', 'r', 'c', 'C'])
                    && w[1..].bytes().all(|b| b.is_ascii_digit())
            });
        let after = self.chars[self.i + rest.chars().count()..]
            .iter()
            .copied()
            .find(|c| *c != ' ' && *c != '\t');
        let constant = !rest.trim().is_empty()
            && !matches!(after, Some('&' | '|' | '<' | '>' | '@' | '['))
            && !rest.contains(['#', '@', '~', '"'])
            && !register_like
            && !self.regs.keys().any(|k| {
                rest.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .any(|w| w == k)
            });
        if constant {
            let n = rest.chars().count();
            (0..n).for_each(|_| {
                self.bump();
            });
            return Some(Self::op(
                "REGSET",
                vec![d, Operand::Value(rest.trim().to_string())],
            ));
        }
        // LXP@(helbidea)
        if self.starts_with("LXP@") {
            (0..3).for_each(|_| {
                self.bump();
            });
            let addr = self.pointer()?;
            return Some(Self::op("LOAD", vec![d, addr]));
        }
        // funtzio bat: sqrt(a)
        for (f, op) in [
            ("sqrt", "SQRT"),
            ("exp", "EXP"),
            ("ln", "LOG"),
            ("sin", "SIN"),
            ("cos", "COS"),
            ("tan", "TAN"),
            ("not", "NOTBITS"),
        ] {
            let call = format!("{}(", f);
            if self.starts_with(&call) {
                (0..f.len()).for_each(|_| {
                    self.bump();
                });
                let inner = self.paren()?;
                let inner = inner.trim();
                // erregistro bat edo balio bat; adierazpen konstante osoa (sqrt(2)) balio bat da
                let v = match inner
                    .strip_prefix('#')
                    .or_else(|| inner.strip_prefix(['R', 'r']))
                    .and_then(|d| d.parse::<usize>().ok())
                {
                    Some(k) => Operand::Reg(k),
                    None => Operand::Value(inner.to_string()),
                };
                return Some(Self::op(op, vec![d, v]));
            }
        }
        let a = self.assign_operand()?;
        self.skip_spaces();
        let ops: [(&str, &str); 9] = [
            ("<<", "SHL"),
            (">>", "SHR"),
            ("+", "ADD"),
            ("-", "SUB"),
            ("*", "MUL"),
            ("/", "DIV"),
            ("^", "POW"),
            ("&", "ANDBITS"),
            ("|", "ORBITS"),
        ];
        for (sym, op) in ops {
            if self.starts_with(sym) {
                (0..sym.len()).for_each(|_| {
                    self.bump();
                });
                let b = self.assign_operand()?;
                self.skip_spaces();
                if self.peek().is_some_and(|c| "+-*/^&|<>".contains(c)) {
                    let p = self.pos();
                    self.error(
                        p,
                        1,
                        ts!(
                            "an assignment takes one operator; split it, like #1<=#2*2; #1<=#1+#3;",
                            "esleipen batek eragile bakarra hartzen du; banatu, #1<=#2*2; #1<=#1+#3; bezala"
                        )
                        .into(),
                    );
                    return None;
                }
                return self.computed(op, d, vec![a, b]);
            }
        }
        // balio bakarra: erregistroa kopiatu, bitak zenbaki gisa, edo zenbaki bat
        Some(match &a {
            Operand::Reg(_) => Self::op("REGCOPY", vec![d, a]),
            Operand::Raw(r)
                if r.len() > 1
                    && r.starts_with(['R', 'r'])
                    && r[1..].bytes().all(|b| b.is_ascii_digit()) =>
            {
                Self::op("REGCOPY", vec![d, a])
            }
            Operand::Named { .. } | Operand::Raw(_)
                if matches!(&a, Operand::Named { .. })
                    || matches!(&a, Operand::Raw(r) if r.starts_with(['c', 'C']) && r[1..].bytes().all(|b| b.is_ascii_digit()) && r.len() > 1) =>
            {
                match elements(&self.regs, &a) {
                    Ok(els) => Stmt::Verbatim(format!("PACKBITS R{} {}", dst, els.join(" "))),
                    Err(e) => {
                        self.diags.push(e);
                        return None;
                    }
                }
            }
            _ => Self::op("REGSET", vec![d, a]),
        })
    }

    // `CC[0x10]%a%b$>#2` edo `CC"izena"%a$>#2`: dei klasikoa (ikus Stmt::ClassicalCall)
    fn classical_call(&mut self) -> Option<Stmt> {
        self.skip_spaces();
        let pos = self.pos();
        let target = match self.peek() {
            Some('[') => {
                let (a, b) = self.index()?;
                if a != b {
                    self.error(
                        pos,
                        1,
                        ts!(
                            "CC calls one address, like [0x10]",
                            "CC-k helbide bakarra deitzen du, [0x10] bezala"
                        )
                        .into(),
                    );
                    return None;
                }
                address(a)
            }
            Some('"') => {
                let s = self.string()?;
                let name = s.trim_matches('"').to_string();
                let ok = name
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                    && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
                if !ok {
                    self.error(
                        pos,
                        name.len() as u32 + 2,
                        t!(
                            "\"{}\" is not a label name",
                            "\"{}\" ez da etiketa-izen bat",
                            name
                        ),
                    );
                    return None;
                }
                name
            }
            _ => {
                self.error(
                    pos,
                    1,
                    ts!(
                        "CC needs what to call: an address like [0x10] or a label like \"name\"",
                        "CC-k zer deitu behar du: [0x10] bezalako helbide bat edo \"izena\" bezalako etiketa bat"
                    )
                    .into(),
                );
                return None;
            }
        };
        let mut args = Vec::new();
        let mut dest = None;
        loop {
            self.skip_spaces();
            if self.peek() == Some('%') {
                self.bump();
                args.push(self.operand()?);
            } else if self.starts_with("$>") {
                self.bump();
                self.bump();
                dest = Some(self.operand()?);
            } else {
                break;
            }
        }
        Some(Stmt::ClassicalCall { target, args, dest })
    }

    // sarrera osoa: adierazpenak `;`-z amaituta
    fn parse(&mut self) -> Vec<(Stmt, Pos, u32)> {
        let mut out = Vec::new();
        loop {
            self.skip_blank();
            if self.peek().is_none() {
                return out;
            }
            let pos = self.pos();
            let start = self.i;
            match self.statement() {
                Some(stmt) => {
                    // adierazpenaren azken karakterea (hutsuneak kontuan hartu gabe)
                    let last = self.chars[..self.i]
                        .iter()
                        .rev()
                        .copied()
                        .find(|c| !c.is_whitespace());
                    self.skip_spaces();
                    let len = (self.i - start) as u32;
                    // 0.4.6: `;` utz daiteke adierazpenen arteko muga argia denean: aurrekoa letra, digitu edo _
                    // batekin amaitzen bada eta hurrengoa batekin hasten bada izan ezik (`#PF` ezin baita
                    // `#P` eta `F...`-tik bereizi)
                    let ident =
                        |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
                    let next_starts = {
                        let save = (self.i, self.line, self.col);
                        self.skip_blank();
                        let next = self.peek();
                        (self.i, self.line, self.col) = save;
                        next
                    };
                    match self.peek() {
                        Some(';') => {
                            self.bump();
                            out.push((stmt, pos, len));
                        }
                        _ if next_starts.is_none()
                            || (self.peek() != Some(';')
                                && !(ident(last) && ident(next_starts))) =>
                        {
                            out.push((stmt, pos, len))
                        }
                        // bloke baten amaiera `}` ondoren `;` gabe ere onartzen da
                        _ if matches!(
                            stmt,
                            Stmt::LoopStart(_)
                                | Stmt::LoopEnd
                                | Stmt::IfStart { .. }
                                | Stmt::Define(_)
                        ) =>
                        {
                            out.push((stmt, pos, len))
                        }
                        _ => {
                            let p = self.pos();
                            self.error(
                                p,
                                1,
                                ts!(
                                    "expected ; at the end of the statement",
                                    "; espero zen adierazpenaren amaieran"
                                )
                                .into(),
                            );
                            self.recover();
                        }
                    }
                }
                None => {
                    if self.diags.is_empty() {
                        self.error(
                            pos,
                            1,
                            ts!(
                                "could not read this statement",
                                "ezin izan da adierazpen hau irakurri"
                            )
                            .into(),
                        );
                    }
                    self.recover();
                }
            }
        }
    }
}

// parametroak komaz bereizita, parentesi barruko komak kontuan hartu gabe
fn split_params(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

// helbide baten etiketa-izena QOA arruntean: 0x10 -> addr_0x10
fn address(n: usize) -> String {
    format!("addr_0x{:x}", n)
}

// CC-ren argumentu bat datu-pilara: erregistroak bere horretan, bitak zenbaki gisa (PACKBITS), zenbakiak
// REGSET bidez. R31 aldi baterakoa da
fn push_lines(
    regs: &HashMap<String, Register>,
    a: &Operand,
    pos: Pos,
    len: u32,
) -> Result<Vec<String>, Diagnostic> {
    let scratch = format!("R{}", SCRATCH_TOP);
    let quantum = || {
        diag(
            pos,
            len,
            ts!(
                "CC passes numbers, registers and bits; a qubit or mode has no classical value until it is measured",
                "CC-k zenbakiak, erregistroak eta bitak pasatzen ditu; qubit edo modu batek ez du balio klasikorik neurtu arte"
            )
            .into(),
        )
    };
    match a {
        Operand::Reg(k) => Ok(vec![format!("PUSH R{}", k)]),
        Operand::Named { name, .. } => match regs.get(name).map(|r| r.kind) {
            Some(Kind::Bits) => {
                let bits = elements(regs, a)?;
                Ok(vec![
                    format!("PACKBITS {} {}", scratch, bits.join(" ")),
                    format!("PUSH {}", scratch),
                ])
            }
            Some(_) => Err(quantum()),
            None => elements(regs, a).map(|_| vec![]),
        },
        Operand::Raw(s) | Operand::Value(s) => {
            let lower = s.to_ascii_lowercase();
            let indexed = |p: char| {
                lower
                    .strip_prefix(p)
                    .is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
            };
            if indexed('r') {
                Ok(vec![format!("PUSH {}", s)])
            } else if indexed('c') {
                Ok(vec![
                    format!("PACKBITS {} {}", scratch, s),
                    format!("PUSH {}", scratch),
                ])
            } else if indexed('q') || indexed('o') {
                Err(quantum())
            } else {
                Ok(vec![
                    format!("REGSET {} {}", scratch, s),
                    format!("PUSH {}", scratch),
                ])
            }
        }
    }
}

// glifo-izen laburrak (QOA-k berak ez dituenak)
// `C12` -> (`C`, 12), `C` adierazitako erregistro bat bada (aurrizki adierazi luzeena) eta gainerakoa digituak
fn split_index(name: &str, regs: &HashMap<String, Register>) -> Option<(String, usize)> {
    (1..name.len()).find_map(|k| {
        let (base, digits) = name.split_at(name.len() - k);
        (regs.contains_key(base) && digits.bytes().all(|b| b.is_ascii_digit()))
            .then(|| digits.parse().ok().map(|n| (base.to_string(), n)))
            .flatten()
    })
}

// esportatzailearentzat: `buru` eta `izena` itsatsita (`HQ`) berriz `buru` + `izena` gisa irakurtzen den, `izenak`
// deklaratuta daudenean (Parser::unglue-ren arau berberak)
pub(crate) fn glues(head: &str, ident: &str, declared: &[&str]) -> bool {
    let joined = format!("{}{}", head, ident);
    if ident.is_empty()
        || !joined.is_ascii()
        || crate::isa::lookup(mnemonic(&joined)).is_some()
        || joined == "PR"
    {
        return false;
    }
    for k in (1..joined.len()).rev() {
        let (h, rest) = joined.split_at(k);
        if crate::isa::lookup(mnemonic(h)).is_none() && h != "PR" {
            continue;
        }
        let plain = {
            let mut c = rest.chars();
            c.next().is_some_and(|l| "qQcCoOrR".contains(l))
                && rest.len() > 1
                && c.all(|d| d.is_ascii_digit())
        };
        if declared.contains(&rest) || plain {
            return k == head.len();
        }
    }
    false
}

fn mnemonic(name: &str) -> &str {
    match name {
        "M" => "MEASURE",
        "CX" => "CNOT",
        _ => name,
    }
}

// fitxategi oso bat: zigilua (badago) egiaztatu eta haren gainekoa hedatu
pub fn expand_file(src: &str, name: &str) -> Result<Expanded, Vec<Diagnostic>> {
    expand(asm::unseal(src, name, crate::seal::Style::Plain)?)
}

pub fn expand(src: &str) -> Result<Expanded, Vec<Diagnostic>> {
    let mut p = Parser::new(src);
    let stmts = p.parse();
    if !p.diags.is_empty() {
        return Err(p.diags);
    }
    let mut lines: Vec<(String, Origin)> = Vec::new();
    let head = Origin {
        pos: Pos { line: 1, col: 1 },
        len: 1,
    };
    if p.counts[Kind::Qubits as usize] > 0 {
        lines.push((format!("QINIT {}", p.counts[Kind::Qubits as usize]), head));
    }
    if p.counts[Kind::Modes as usize] > 0 {
        let c = p.cutoff.map_or(DEFAULT_CUTOFF, |c| c.0);
        lines.push((
            format!("QMODE {} {}", p.counts[Kind::Modes as usize], c),
            head,
        ));
    }
    let mut cx = Emit {
        regs: &p.regs,
        lines,
        diags: Vec::new(),
        blocks: Vec::new(),
        defined: HashMap::new(),
        ifs: 0,
        budget: MAX_EXPANSION,
    };
    cx.stmts(&stmts);
    if !cx.blocks.is_empty() {
        cx.diags.push(diag(
            Pos { line: 1, col: 1 },
            1,
            ts!(
                "a block { is never closed with }",
                "bloke baten { ez da inoiz }-rekin ixten"
            )
            .into(),
        ));
    }
    let (lines, diags) = (cx.lines, cx.diags);
    if !diags.is_empty() {
        return Err(diags);
    }
    let source = lines
        .iter()
        .map(|(l, _)| l.as_str())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    Ok(Expanded {
        source,
        origins: lines.into_iter().map(|(_, o)| o).collect(),
    })
}

// bloke izendunen hedapenaren muga (adierazpenak guztira): bloke batek beste bat bi aldiz erabiltzen badu,
// eta horrek beste bat..., hedapena esponentziala izan daiteke
const MAX_EXPANSION: usize = 1_000_000;

#[derive(Clone)]
enum Block {
    Loop,
    // baldintzazko blokea: amaierako etiketa
    If(String),
}

// adierazpenak QOA lerro bihurtzea
struct Emit<'a> {
    regs: &'a HashMap<String, Register>,
    lines: Vec<(String, Origin)>,
    diags: Vec<Diagnostic>,
    blocks: Vec<Block>,
    defined: HashMap<String, Vec<(Stmt, Pos, u32)>>,
    // ??/!! blokeen kontagailua, etiketa bakarretarako
    ifs: usize,
    budget: usize,
}

impl Emit<'_> {
    fn stmts(&mut self, stmts: &[(Stmt, Pos, u32)]) {
        let mut k = 0;
        while k < stmts.len() {
            let (stmt, pos, len) = &stmts[k];
            k += 1;
            if self.budget == 0 {
                self.diags.push(diag(
                    *pos,
                    *len,
                    t!(
                        "the named blocks expand to more than {} statements",
                        "bloke izendunak {} adierazpen baino gehiagotan hedatzen dira",
                        MAX_EXPANSION
                    ),
                ));
                return;
            }
            self.budget -= 1;
            if let Stmt::Define(name) = stmt {
                // blokearen gorputza: dagokion `}` arte, barruko blokeak kontuan hartuta
                let mut depth = 1usize;
                let start = k;
                while k < stmts.len() {
                    match stmts[k].0 {
                        Stmt::LoopStart(_) | Stmt::IfStart { .. } | Stmt::Define(_) => depth += 1,
                        Stmt::LoopEnd => depth -= 1,
                        _ => {}
                    }
                    k += 1;
                    if depth == 0 {
                        break;
                    }
                }
                if depth > 0 {
                    self.diags.push(diag(
                        *pos,
                        *len,
                        t!(
                            "the block {}:: is never closed with }}",
                            "{}:: blokea ez da inoiz }}-rekin ixten",
                            name
                        ),
                    ));
                    return;
                }
                if self.defined.contains_key(name) {
                    self.diags.push(diag(
                        *pos,
                        *len,
                        t!(
                            "the block {} is defined twice",
                            "{} blokea bi aldiz definituta dago",
                            name
                        ),
                    ));
                }
                self.defined
                    .insert(name.clone(), stmts[start..k - 1].to_vec());
                continue;
            }
            self.stmt(stmt, *pos, *len);
        }
    }

    fn stmt(&mut self, stmt: &Stmt, pos: Pos, len: u32) {
        let origin = Origin { pos, len };
        let regs = self.regs;
        let lines = &mut self.lines;
        let mut emit = |s: String| lines.push((s, origin));
        match stmt {
            Stmt::Nothing | Stmt::Define(_) => {}
            Stmt::Label(l) => emit(format!("{}:", l)),
            Stmt::Jump(l) => emit(format!("JUMP {}", l)),
            Stmt::Call(l) => emit(format!("CALL {}", l)),
            Stmt::ClassicalCall { target, args, dest } => {
                for a in args.iter().rev() {
                    match push_lines(regs, a, pos, len) {
                        Ok(v) => v.into_iter().for_each(&mut emit),
                        Err(d) => self.diags.push(d),
                    }
                }
                emit(format!("CALL {}", target));
                match dest {
                    None => {}
                    Some(Operand::Reg(k)) => emit(format!("POP R{}", k)),
                    Some(Operand::Raw(r))
                        if r.starts_with(['R', 'r']) && r[1..].parse::<usize>().is_ok() =>
                    {
                        emit(format!("POP {}", r))
                    }
                    Some(_) => self.diags.push(diag(
                        pos,
                        len,
                        ts!(
                            "the result of CC goes to a register, like $>#2",
                            "CC-ren emaitza erregistro batera doa, $>#2 bezala"
                        )
                        .into(),
                    )),
                }
            }
            Stmt::Return => emit("RETSUB".into()),
            Stmt::Verbatim(s) => emit(s.clone()),
            Stmt::Many(v) => {
                for st in v {
                    self.stmt(st, pos, len);
                }
            }
            Stmt::LoopStart(n) => {
                self.blocks.push(Block::Loop);
                emit(format!("LOOPSTART {}", n))
            }
            Stmt::IfStart { one, value } => match elements(regs, value) {
                Ok(v) if v.len() == 1 => {
                    self.ifs += 1;
                    let (run, end) = (
                        format!("glyph.if{}.run", self.ifs),
                        format!("glyph.if{}.end", self.ifs),
                    );
                    if v[0].starts_with(['c', 'C']) {
                        // bit bat 0 edo 1 da: jauzi bakarra
                        let skip = if *one { "JUMPIFZERO" } else { "JUMPIFONE" };
                        emit(format!("{} {} {}", skip, v[0], end));
                    } else {
                        let run_if = if *one { "JUMPIFONE" } else { "JUMPIFZERO" };
                        emit(format!("{} {} {}", run_if, v[0], run));
                        emit(format!("JUMP {}", end));
                        emit(format!("{}:", run));
                    }
                    self.blocks.push(Block::If(end));
                }
                Ok(_) => self.diags.push(diag(
                    pos,
                    len,
                    ts!(
                        "?? and !! test one bit or register, not a whole register",
                        "??-k eta !!-k bit edo erregistro bakarra egiaztatzen dute, ez erregistro oso bat"
                    )
                    .into(),
                )),
                Err(d) => self.diags.push(d),
            },
            Stmt::LoopEnd => match self.blocks.pop() {
                None => self.diags.push(diag(
                    pos,
                    1,
                    ts!("this } closes no block", "} honek ez du blokerik ixten").into(),
                )),
                Some(Block::Loop) => emit("LOOPEND".into()),
                Some(Block::If(end)) => emit(format!("{}:", end)),
            },
            Stmt::Use(name) => match self.defined.get(name).cloned() {
                Some(body) => self.stmts(&body),
                None => self.diags.push(diag(
                    pos,
                    len,
                    t!(
                        "::{} uses a block that is not defined above (write {}::{{ ... }} first)",
                        "::{}-k goian definituta ez dagoen bloke bat erabiltzen du (idatzi {}::{{ ... }} lehenik)",
                        name,
                        name
                    ),
                )),
            },
            Stmt::JumpIf { one, value, label } => match elements(regs, value) {
                Ok(v) if v.len() == 1 => emit(format!(
                    "{} {} {}",
                    if *one { "JUMPIFONE" } else { "JUMPIFZERO" },
                    v[0],
                    label
                )),
                Ok(_) => self.diags.push(diag(
                    pos,
                    len,
                    ts!(
                        "a jump tests one bit or register, not a whole register",
                        "jauzi batek bit edo erregistro bakarra egiaztatzen du, ez erregistro oso bat"
                    )
                    .into(),
                )),
                Err(d) => self.diags.push(d),
            },
            Stmt::Op {
                name,
                texts,
                operands,
                params,
                args,
                dests,
                after,
            } => match op_lines(
                regs, name, texts, operands, params, args, dests, pos, len,
            ) {
                // eragigaien ondorengo testuak lerro bakoitzaren amaieran
                Ok(v) => v
                    .into_iter()
                    .map(|l| {
                        after
                            .iter()
                            .fold(l, |acc, t| format!("{} {}", acc, t))
                    })
                    .for_each(emit),
                Err(d) => self.diags.push(d),
            },
        }
    }
}

// eragigai baten QOA elementuak: erregistro osoa -> elementu bakoitza
fn elements(regs: &HashMap<String, Register>, op: &Operand) -> Result<Vec<String>, Diagnostic> {
    match op {
        Operand::Raw(s) | Operand::Value(s) => Ok(vec![s.clone()]),
        Operand::Reg(k) => Ok(vec![format!("R{}", k)]),
        Operand::Named { name, range, pos } => {
            let Some(r) = regs.get(name) else {
                return Err(diag(
                    *pos,
                    name.len() as u32,
                    t!(
                        "{} is not declared; declare it with {}{{n}}",
                        "{} ez dago adierazita; adierazi {}{{n}} bidez",
                        name,
                        name
                    ),
                ));
            };
            let (a, b) = range.unwrap_or((0, r.size - 1));
            if b >= r.size {
                return Err(diag(
                    *pos,
                    name.len() as u32,
                    t!(
                        "{} has {} element(s); index {} is past the end",
                        "{}-k {} elementu ditu; {} indizea amaieratik kanpo dago",
                        name,
                        r.size,
                        b
                    ),
                ));
            }
            Ok((a..=b)
                .map(|k| format!("{}{}", r.kind.prefix(), r.start + k))
                .collect())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn op_lines(
    regs: &HashMap<String, Register>,
    name: &str,
    texts: &[String],
    operands: &[Operand],
    params: &[String],
    args: &[Operand],
    dests: &[Operand],
    pos: Pos,
    len: u32,
) -> Result<Vec<String>, Diagnostic> {
    // `PR"izena"%a%b`: izena eta balioak inprimatu (`izena = a b` eta lerro-jauzia), PRINTF bidez
    let label;
    let (name, texts) = if name == "PR" {
        let wrong = |m: String| Err(diag(pos, len, m));
        if texts.len() != 1 || !operands.is_empty() || !dests.is_empty() {
            return wrong(
                ts!(
                    "PR takes one label and the values to print: PR\"x\"%#3",
                    "PR-k etiketa bat eta inprimatzeko balioak hartzen ditu: PR\"x\"%#3"
                )
                .into(),
            );
        }
        let text = texts[0].trim_matches('"');
        if text.contains('{') || text.contains('}') {
            return wrong(
                ts!(
                    "a PR label cannot contain { or }; use PF for a format of your own",
                    "PR etiketa batek ezin du { edo } eduki; erabili PF zure formatu baterako"
                )
                .into(),
            );
        }
        label = if args.is_empty() {
            vec![format!("\"{}\\n\"", text)]
        } else {
            vec![format!(
                "\"{} = {}\\n\"",
                text,
                vec!["{}"; args.len()].join(" ")
            )]
        };
        ("PRINTF", &label[..])
    } else {
        (name, texts)
    };
    // erregistro edo zenbaki baten lekuan (x) zenbakiak, konstanteak eta parentesi arteko balioak `(..)` QOA
    // arruntean: `ADD/#1,#1,1` -> `ADD R1 R1 (1)`, `ADD/#1,#1,newline` -> `ADD R1 R1 (newline)` (CONST
    // newline 10 ondoren). QOA arrunteko erregistroak (`R2`) bere horretan
    let specs = crate::isa::lookup(mnemonic(name)).map(|s| s.operand_specs());
    let kind_at = |k: usize| {
        specs.as_ref().map_or(' ', |sp| {
            sp.get(texts.len() + k)
                .or(sp.last())
                .map_or(' ', |s| s.kind)
        })
    };
    let ops: Vec<Vec<String>> = operands
        .iter()
        .enumerate()
        .map(|(k, o)| {
            let els = elements(regs, o)?;
            let value = |e: String| format!("({})", e);
            Ok(match (kind_at(k), o) {
                ('x', Operand::Value(_)) => els.into_iter().map(value).collect(),
                ('x', Operand::Raw(_)) => els
                    .into_iter()
                    .map(|e| {
                        let register = e.len() > 1
                            && e.starts_with(['R', 'r'])
                            && e[1..].bytes().all(|b| b.is_ascii_digit());
                        if register { e } else { value(e) }
                    })
                    .collect(),
                _ => els,
            })
        })
        .collect::<Result<_, Diagnostic>>()?;
    let dst: Vec<Vec<String>> = dests
        .iter()
        .map(|o| elements(regs, o))
        .collect::<Result<_, _>>()?;
    // hedapena: elementu bat baino gehiagoko zerrenda guztiek luzera bera
    let width = ops
        .iter()
        .chain(&dst)
        .map(Vec::len)
        .filter(|&n| n > 1)
        .max()
        .unwrap_or(1);
    if ops
        .iter()
        .chain(&dst)
        .any(|v| v.len() > 1 && v.len() != width)
    {
        return Err(diag(
            pos,
            len,
            ts!(
                "registers used together must have the same size (each element goes with the matching one)",
                "batera erabilitako erregistroek tamaina bera izan behar dute (elementu bakoitza bere parekoarekin doa)"
            )
            .into(),
        ));
    }
    // zenbaki hutsak qubit, modu, erregistro edo bit baten lekuan: QOA arruntean `3`-k q3 edo R3 esan
    // nahi du, baina glifo-forman zenbaki bat zenbaki bat da. anbiguoa denez, errorea
    if let Some(spec) = crate::isa::lookup(mnemonic(name)) {
        let specs = spec.operand_specs();
        for (k, o) in operands.iter().enumerate() {
            let at = texts.len() + k;
            let kind = specs.get(at).or(specs.last()).map_or(' ', |s| s.kind);
            if let Operand::Raw(text) | Operand::Value(text) = o
                && "qorcv".contains(kind)
                && text.parse::<u64>().is_ok()
            {
                return Err(diag(
                    pos,
                    len,
                    t!(
                        "{0} in operand {2} of {1} would be read as an address, not a number: write #{0} for register R{0}, or q{0} / o{0} / c{0} for a qubit, mode or bit (a number goes into a register with REGSET first)",
                        "{0} {1}-ren {2}. eragigaian helbide gisa irakurriko litzateke, ez zenbaki gisa: idatzi #{0} R{0} erregistrorako, edo q{0} / o{0} / c{0} qubit, modu edo bit baterako (zenbaki bat lehenik REGSET-ekin sartzen da erregistro batean)",
                        text,
                        mnemonic(name),
                        k + 1
                    ),
                ));
            }
        }
    }
    let pick = |v: &Vec<String>, k: usize| {
        if v.len() == 1 {
            v[0].clone()
        } else {
            v[k].clone()
        }
    };
    let mut out = Vec::new();
    // % argumentuak: bit-erregistroak zenbaki bihurtu PACKBITS-ekin (R31, R30, ...)
    let mut printed = Vec::new();
    for (k, a) in args.iter().enumerate() {
        let els = elements(regs, a)?;
        let is_bits = matches!(a, Operand::Named { name, .. } if regs.get(name).is_some_and(|r| r.kind == Kind::Bits));
        if is_bits {
            let r = format!("R{}", SCRATCH_TOP - k);
            out.push(format!("PACKBITS {} {}", r, els.join(" ")));
            printed.push(r);
        } else if els.len() == 1 {
            printed.push(els[0].clone());
        } else {
            return Err(diag(
                pos,
                len,
                ts!(
                    "only bit registers can be printed whole with %",
                    "bit-erregistroak bakarrik inprimatu daitezke osorik %-rekin"
                )
                .into(),
            ));
        }
    }
    for k in 0..width {
        let mut line = String::from(mnemonic(name));
        for t in texts {
            line.push(' ');
            line.push_str(t);
        }
        for v in &ops {
            line.push(' ');
            line.push_str(&pick(v, k));
        }
        for p in params {
            line.push(' ');
            line.push_str(p);
        }
        for r in &printed {
            line.push(' ');
            line.push_str(r);
        }
        if !dst.is_empty() {
            line.push_str(" ->");
            for v in &dst {
                line.push(' ');
                line.push_str(&pick(v, k));
            }
        }
        out.push(line);
    }
    Ok(out)
}
