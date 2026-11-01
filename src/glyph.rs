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
    let ex = expand(src)?;
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

#[derive(Debug)]
enum Stmt {
    // agindu bat: izena, testuak, eragigaiak, parametroak, % argumentuak, $> helburuak
    Op {
        name: String,
        texts: Vec<String>,
        operands: Vec<Operand>,
        params: Vec<String>,
        args: Vec<Operand>,
        dests: Vec<Operand>,
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
    LoopEnd,
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
        // QOA arruntaren QREG/CREG/OREG-en muga berak: 4096 elementu, eta lehena 0..4095
        if size > 4096 {
            self.error(
                pos,
                name.len() as u32,
                t!(
                    "{} has {} elements, more than the 4096 a register can hold",
                    "{}-k {} elementu ditu, erregistro batek izan ditzakeen 4096 baino gehiago",
                    name,
                    size
                ),
            );
            return false;
        }
        if at.is_some_and(|a| a > 4095) {
            self.error(
                pos,
                name.len() as u32,
                t!(
                    "{} starts at {}, but a register starts at 0 to 4095",
                    "{} {}-n hasten da, baina erregistro bat 0 eta 4095 artean hasten da",
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
        if self.peek() == Some('{') {
            let size = self.size(kind.unwrap_or(Kind::Qubits))?;
            let at = self.placement()?;
            if !self.declare(&name, kind.unwrap_or(Kind::Qubits), size, at, pos) {
                return None;
            }
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
            Some(':') => {
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
        let mut params = Vec::new();
        if self.peek() == Some('(') {
            let p = self.paren()?;
            params.extend(split_params(&p));
        }
        let mut texts = Vec::new();
        while self.peek() == Some('"') {
            texts.push(self.string()?);
        }
        let mut operands = Vec::new();
        self.skip_spaces();
        if self.peek() == Some('/') {
            self.bump();
            operands = self.operands()?;
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
            } else {
                break;
            }
        }
        Some(Stmt::Op {
            name: name.to_ascii_uppercase(),
            texts,
            operands,
            params,
            args,
            dests,
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
                    self.skip_spaces();
                    let len = (self.i - start) as u32;
                    match self.peek() {
                        Some(';') => {
                            self.bump();
                            out.push((stmt, pos, len));
                        }
                        // bloke baten amaiera `}` ondoren `;` gabe ere onartzen da
                        _ if matches!(stmt, Stmt::LoopStart(_) | Stmt::LoopEnd) => {
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
fn mnemonic(name: &str) -> &str {
    match name {
        "M" => "MEASURE",
        "CX" => "CNOT",
        _ => name,
    }
}

pub fn expand(src: &str) -> Result<Expanded, Vec<Diagnostic>> {
    let mut p = Parser::new(src);
    let stmts = p.parse();
    if !p.diags.is_empty() {
        return Err(p.diags);
    }
    let mut lines: Vec<(String, Origin)> = Vec::new();
    let mut diags = Vec::new();
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
    let mut depth = 0usize;
    for (stmt, pos, len) in stmts {
        let origin = Origin { pos, len };
        let mut emit = |s: String| lines.push((s, origin));
        match stmt {
            Stmt::Nothing => {}
            Stmt::Label(l) => emit(format!("{}:", l)),
            Stmt::Jump(l) => emit(format!("JUMP {}", l)),
            Stmt::Call(l) => emit(format!("CALL {}", l)),
            Stmt::ClassicalCall { target, args, dest } => {
                for a in args.iter().rev() {
                    match push_lines(&p.regs, a, pos, len) {
                        Ok(v) => v.into_iter().for_each(&mut emit),
                        Err(d) => diags.push(d),
                    }
                }
                emit(format!("CALL {}", target));
                match dest {
                    None => {}
                    Some(Operand::Reg(k)) => emit(format!("POP R{}", k)),
                    Some(Operand::Raw(r)) if r.starts_with(['R', 'r']) && r[1..].parse::<usize>().is_ok() => {
                        emit(format!("POP {}", r))
                    }
                    Some(_) => diags.push(diag(
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
            Stmt::Verbatim(s) => emit(s),
            Stmt::LoopStart(n) => {
                depth += 1;
                emit(format!("LOOPSTART {}", n))
            }
            Stmt::LoopEnd => {
                if depth == 0 {
                    diags.push(diag(pos, 1, ts!("this } closes no loop", "} honek ez du begiztarik ixten").into()));
                } else {
                    depth -= 1;
                    emit("LOOPEND".into())
                }
            }
            Stmt::JumpIf { one, value, label } => match elements(&p.regs, &value) {
                Ok(v) if v.len() == 1 => emit(format!("{} {} {}", if one { "JUMPIFONE" } else { "JUMPIFZERO" }, v[0], label)),
                Ok(_) => diags.push(diag(pos, len, ts!("a jump tests one bit or register, not a whole register", "jauzi batek bit edo erregistro bakarra egiaztatzen du, ez erregistro oso bat").into())),
                Err(d) => diags.push(d),
            },
            Stmt::Op { name, texts, operands, params, args, dests } => {
                match op_lines(&p.regs, &name, &texts, &operands, &params, &args, &dests, pos, len) {
                    Ok(v) => v.into_iter().for_each(emit),
                    Err(d) => diags.push(d),
                }
            }
        }
    }
    if depth > 0 {
        diags.push(diag(
            Pos { line: 1, col: 1 },
            1,
            ts!(
                "a loop { is never closed with }",
                "begizta baten { ez da inoiz }-rekin ixten"
            )
            .into(),
        ));
    }
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
