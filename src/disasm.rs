// fitxategi honek programa mihiztatu bat (`Program`) beste forma batzuetara idazten du, analisirako eta
// arazketarako (`qoa export --to ...`):
//
//   qoa       QOA arrunt kanonikoa (desmihiztatzea): ezizenak izen nagusira, tarteak zabalduta, jauzi-
//             helburuak `addr_0x..` etiketekin. .qx bitar batetik ere badabil
//   listing   bera, eragiketa bakoitzaren helbidea (0x..) eta iturburu-lerroa iruzkin gisa; berriz
//             mihiztatu daiteke
//   json      programa datu gisa: baliabideak eta eragiketa bakoitza (helbidea, lerroa, testua, egitura)
//   dot       kontrol-fluxuaren grafoa (graphviz): oinarrizko blokeak eta jauziak, begiztak, deiak
//   glyph     glifo-sintaxia (docs/glyph.md), aukerekin (`GlyphOptions`)
//
// desmihiztatzea mihiztatzailearen (src/asm.rs) alderantzizkoa da: lerro bakoitza berriz mihiztatuta Op
// bera ematen du. `qoa`, `listing` eta `glyph` irteerek hori bera egiaztatzen dute idatzi aurretik (berriz
// mihiztatu eta konparatu), eta tests/disasm.rs-ek programa guztiekin. jauzi-helburuen helbideak glifo-
// sintaxiaren CC[0x..] eta `0x..:` helbide berberak dira.

use crate::program::{BinOp, Cmp, Count, Dest, Gate, Op, Program, Source, UnOp, Val};
use crate::{t, ts};
use std::collections::BTreeSet;
use std::f64::consts::PI;

// eragigai bat, motaren arabera (QOA arruntean eta glifoan modu desberdinean idazten dira)
#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Q(u32),
    O(u32),
    C(u32),
    R(u32),
    // zenbakia, oinarria (X/Y/Z) edo beste hitz bat, bere horretan
    Word(String),
    Label(usize),
    Text(String),
    // erregistro baten lekuko zenbakia: `(1)` (QOA arruntean eta glifoan berdin)
    Imm(f64),
}

// agindu bat: izena, eragigaiak ordenan eta aukerako helmuga (-> ondoren)
#[derive(Clone, Debug)]
struct Instr {
    name: &'static str,
    toks: Vec<Tok>,
    dest: Option<Tok>,
}

// helbide baten etiketa (glifoaren `0x..:` bera)
pub fn label(addr: usize) -> String {
    format!("addr_0x{:x}", addr)
}

// zenbaki bat berriz irakurtzean bit berak emateko moduan (rust-en idazkera laburrena, beti zehatza)
fn num(x: f64) -> Tok {
    Tok::Word(format!("{}", x))
}

// erregistro bat edo zenbaki bat (Val)
fn val(v: Val) -> Tok {
    match v {
        Val::Reg(r) => Tok::R(r),
        Val::Num(x) => Tok::Imm(x),
    }
}

fn binop_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "ADD",
        BinOp::Sub => "SUB",
        BinOp::Mul => "MUL",
        BinOp::Div => "DIV",
        BinOp::And => "ANDBITS",
        BinOp::Or => "ORBITS",
        BinOp::Xor => "XORBITS",
        BinOp::Shl => "SHL",
        BinOp::Shr => "SHR",
        BinOp::Pow => "POW",
    }
}

fn unop_name(op: UnOp) -> &'static str {
    match op {
        UnOp::Not => "NOTBITS",
        UnOp::Sqrt => "SQRT",
        UnOp::Exp => "EXP",
        UnOp::Log => "LOG",
        UnOp::Sin => "SIN",
        UnOp::Cos => "COS",
        UnOp::Tan => "TAN",
    }
}

fn cmp_name(cmp: Cmp) -> &'static str {
    match cmp {
        Cmp::Eq => "IFEQ",
        Cmp::Ne => "IFNE",
        Cmp::Gt => "IFGT",
        Cmp::Lt => "IFLT",
    }
}

fn escape(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn dest_tok(d: Dest) -> Option<Tok> {
    match d {
        Dest::Own => None,
        Dest::Bit(k) => Some(Tok::C(k)),
        Dest::Reg(k) => Some(Tok::R(k)),
    }
}

fn source_tok(s: Source) -> Tok {
    match s {
        Source::Reg(k) => Tok::R(k),
        Source::Bit(k) => Tok::C(k),
    }
}

fn qs(v: &[u32]) -> Vec<Tok> {
    v.iter().map(|&q| Tok::Q(q)).collect()
}

fn unexpressible(addr: usize, what: &str) -> String {
    t!(
        "operation 0x{:x} ({}) has no QOA text form",
        "0x{:x} eragiketak ({}) ez du QOA testu-formarik",
        addr,
        what
    )
}

// Op bat agindu bihurtu (asm.rs-eko `build`-en alderantzizkoa)
fn instr(op: &Op, addr: usize) -> Result<Instr, String> {
    let i = |name, toks| Instr {
        name,
        toks,
        dest: None,
    };
    let with = |name, toks, dest| Instr { name, toks, dest };
    let basis = |b: crate::program::Basis| Tok::Word(b.name().into());
    Ok(match op {
        Op::Gate { q, gate } => {
            let q = Tok::Q(*q);
            match *gate {
                Gate::H => i("H", vec![q]),
                Gate::X => i("X", vec![q]),
                Gate::Y => i("Y", vec![q]),
                Gate::Z => i("Z", vec![q]),
                Gate::S => i("S", vec![q]),
                Gate::Sdg => i("SDG", vec![q]),
                Gate::T => i("T", vec![q]),
                Gate::Tdg => i("TDG", vec![q]),
                Gate::SX => i("SX", vec![q]),
                Gate::SXdg => i("SXDG", vec![q]),
                Gate::RX(t) => i("RX", vec![q, num(t)]),
                Gate::RY(t) => i("RY", vec![q, num(t)]),
                Gate::RZ(t) => i("RZ", vec![q, num(t)]),
                Gate::P(t) => i("PHASE", vec![q, num(t)]),
                Gate::U(a, b, c) => i("U", vec![q, num(a), num(b), num(c)]),
            }
        }
        Op::Controlled { controls, q, gate } => {
            let t = Tok::Q(*q);
            match (controls.as_slice(), *gate) {
                ([c], g) => {
                    let c = Tok::Q(*c);
                    match g {
                        Gate::X => i("CNOT", vec![c, t]),
                        Gate::Y => i("CY", vec![c, t]),
                        Gate::H => i("CH", vec![c, t]),
                        Gate::SX => i("CSX", vec![c, t]),
                        Gate::U(a, b, d) => i("CU", vec![c, t, num(a), num(b), num(d)]),
                        Gate::RX(a) => i("CRX", vec![c, t, num(a)]),
                        Gate::RY(a) => i("CRY", vec![c, t, num(a)]),
                        Gate::RZ(a) => i("CRZ", vec![c, t, num(a)]),
                        _ => {
                            return Err(unexpressible(
                                addr,
                                ts!("controlled gate", "ate kontrolatua"),
                            ));
                        }
                    }
                }
                ([a, b], Gate::X) => i("CCX", vec![Tok::Q(*a), Tok::Q(*b), t]),
                (cs, Gate::X) if !cs.is_empty() => {
                    let mut v = qs(cs);
                    v.push(t);
                    i("MCX", v)
                }
                _ => {
                    return Err(unexpressible(
                        addr,
                        ts!("controlled gate", "ate kontrolatua"),
                    ));
                }
            }
        }
        Op::PhaseAll { qubits, theta } => match qubits.as_slice() {
            [a, b] if *theta == PI => i("CZ", vec![Tok::Q(*a), Tok::Q(*b)]),
            [a, b] => i("CPHASE", vec![Tok::Q(*a), Tok::Q(*b), num(*theta)]),
            v if *theta == PI && !v.is_empty() => i("MCZ", qs(v)),
            _ => {
                return Err(unexpressible(
                    addr,
                    ts!("multi-qubit phase", "qubit anitzeko fasea"),
                ));
            }
        },
        Op::Swap { a, b, controls } => match controls.as_slice() {
            [] => i("SWAP", vec![Tok::Q(*a), Tok::Q(*b)]),
            [c] => i("CSWAP", vec![Tok::Q(*c), Tok::Q(*a), Tok::Q(*b)]),
            _ => {
                return Err(unexpressible(
                    addr,
                    ts!("multi-controlled swap", "kontrol anitzeko trukaketa"),
                ));
            }
        },
        Op::Bell { a, b } => i("BELL", vec![Tok::Q(*a), Tok::Q(*b)]),
        Op::Ghz { qubits } => i("GHZ", qs(qubits)),
        Op::Cluster { qubits } => i("CLUSTER", qs(qubits)),
        Op::EntangleSwap { a, b, c, d } => i("ESWAP", qs(&[*a, *b, *c, *d])),
        Op::Measure { q, basis: b, dest } => {
            if *b == crate::program::Basis::Z {
                with("MEASURE", vec![Tok::Q(*q)], dest_tok(*dest))
            } else {
                with("MEASUREIN", vec![Tok::Q(*q), basis(*b)], dest_tok(*dest))
            }
        }
        Op::MeasureAll => i("MEASUREALL", vec![]),
        Op::Reset { q } => i("RESET", vec![Tok::Q(*q)]),
        Op::ResetAll => i("RESETALL", vec![]),
        Op::ToBasis { q, basis: b } => i("TOBASIS", vec![Tok::Q(*q), basis(*b)]),
        Op::Emit { q, o } => i("EMIT", vec![Tok::Q(*q), Tok::O(*o)]),
        Op::PhotonCount { o, dest } => with("COUNT", vec![Tok::O(*o)], dest_tok(*dest)),
        Op::PhotonDetect { o, dest } => with("DETECT", vec![Tok::O(*o)], dest_tok(*dest)),
        Op::Parity { o, dest } => with("PARITY", vec![Tok::O(*o)], dest_tok(*dest)),
        Op::Coincidence { modes, dest } => with(
            "COINCIDENCE",
            modes.iter().map(|&o| Tok::O(o)).collect(),
            dest_tok(*dest),
        ),
        Op::Displace { o, re, im } => i("DISPLACE", vec![Tok::O(*o), num(*re), num(*im)]),
        Op::DisplaceReg { o, r } => i("DISPLACEREG", vec![Tok::O(*o), Tok::R(*r)]),
        Op::Squeeze { o, r, phi } => i("SQUEEZE", vec![Tok::O(*o), num(*r), num(*phi)]),
        Op::SqueezeReg { o, reg } => i("SQUEEZEREG", vec![Tok::O(*o), Tok::R(*reg)]),
        Op::ModePhase { o, theta } => i("MODEPHASE", vec![Tok::O(*o), num(*theta)]),
        Op::Kerr { o, kappa } => i("KERR", vec![Tok::O(*o), num(*kappa)]),
        Op::CrossKerr { a, b, kappa } => i("CROSSKERR", vec![Tok::O(*a), Tok::O(*b), num(*kappa)]),
        Op::BeamSplitter { a, b, theta, phi } => {
            i("BS", vec![Tok::O(*a), Tok::O(*b), num(*theta), num(*phi)])
        }
        Op::ModeSwap { a, b } => i("MODESWAP", vec![Tok::O(*a), Tok::O(*b)]),
        Op::AddPhoton { o } => i("ADDPHOTON", vec![Tok::O(*o)]),
        Op::SubPhoton { o } => i("SUBPHOTON", vec![Tok::O(*o)]),
        Op::Loss { o, gamma } => i("LOSS", vec![Tok::O(*o), num(*gamma)]),
        Op::Fock { o, n } => i("FOCK", vec![Tok::O(*o), Tok::Word(n.to_string())]),
        Op::Dispersive { q, o, theta } => {
            i("DISPERSIVE", vec![Tok::Q(*q), Tok::O(*o), num(*theta)])
        }
        Op::DumpState => i("DUMPSTATE", vec![]),
        Op::DumpRegs => i("DUMPREGS", vec![]),
        Op::Expect { q, basis: b, dest } => {
            let mut v = vec![Tok::Q(*q), basis(*b)];
            v.extend(dest.map(Tok::R));
            i("EXPECT", v)
        }
        Op::BellCheck { a, b, dest } => {
            let mut v = vec![Tok::Q(*a), Tok::Q(*b)];
            v.extend(dest.map(Tok::R));
            i("BELLCHECK", v)
        }
        Op::MeanPhoton { o, dest } => {
            let mut v = vec![Tok::O(*o)];
            v.extend(dest.map(Tok::R));
            i("MEANPHOTON", v)
        }
        Op::Set { r, value } => i("REGSET", vec![Tok::R(*r), num(*value)]),
        Op::Copy { dst, src } => i("REGCOPY", vec![Tok::R(*dst), Tok::R(*src)]),
        Op::Bin { op, dst, a, b } => i(binop_name(*op), vec![Tok::R(*dst), Tok::R(*a), Tok::R(*b)]),
        Op::Un { op, dst, src } => i(unop_name(*op), vec![Tok::R(*dst), Tok::R(*src)]),
        Op::Rand { r } => i("RAND", vec![Tok::R(*r)]),
        Op::Seed { seed } => i("SEEDRNG", vec![Tok::Word(seed.to_string())]),
        Op::Input { r } => i("INPUT", vec![Tok::R(*r)]),
        Op::Time { r } => i("GETTIME", vec![Tok::R(*r)]),
        Op::Push { r } => i("PUSH", vec![Tok::R(*r)]),
        Op::Pop { r } => i("POP", vec![Tok::R(*r)]),
        Op::PackBits { dst, bits } => {
            let mut v = vec![Tok::R(*dst)];
            v.extend(bits.iter().map(|&c| Tok::C(c)));
            i("PACKBITS", v)
        }
        Op::Jump { to } => i("JUMP", vec![Tok::Label(*to)]),
        Op::JumpIf { src, value, to } => {
            let name = if *value == 0.0 {
                "JUMPIFZERO"
            } else if *value == 1.0 {
                "JUMPIFONE"
            } else {
                return Err(unexpressible(
                    addr,
                    ts!("jump on a value", "balio baten araberako jauzia"),
                ));
            };
            i(name, vec![source_tok(*src), Tok::Label(*to)])
        }
        Op::Branch { cmp, a, b, to } => i(
            cmp_name(*cmp),
            vec![Tok::R(*a), Tok::R(*b), Tok::Label(*to)],
        ),
        Op::Call { to } => i("CALL", vec![Tok::Label(*to)]),
        Op::Return => i("RETSUB", vec![]),
        Op::LoopStart { count, .. } => match count {
            Count::Lit(n) => i("LOOPSTART", vec![Tok::Word(n.to_string())]),
            Count::Reg(r) => i("LOOPSTART", vec![Tok::R(*r)]),
        },
        Op::LoopEnd { .. } => i("LOOPEND", vec![]),
        Op::Halt => i("HALT", vec![]),
        Op::Barrier => i("BARRIER", vec![]),
        Op::Print { text, newline } => i(
            if *newline { "PRINTLN" } else { "PRINT" },
            vec![Tok::Text(text.clone())],
        ),
        Op::Printf { format, regs } => {
            let mut v = vec![Tok::Text(format.clone())];
            v.extend(regs.iter().map(|&r| Tok::R(r)));
            i("PRINTF", v)
        }
        Op::CharOut { r } => i("CHAROUT", vec![Tok::R(*r)]),
        Op::Log { q, text } => i("VERBOSELOG", vec![Tok::Q(*q), Tok::Text(text.clone())]),
        Op::Load { dst, addr } => i("LOAD", vec![Tok::R(*dst), Tok::R(*addr)]),
        Op::Store { src, addr } => i("STORE", vec![Tok::R(*src), Tok::R(*addr)]),
        Op::StoreText { addr, text } => {
            i("STORETEXT", vec![Tok::R(*addr), Tok::Text(text.clone())])
        }
        Op::GetByte { r } => i("GETC", vec![Tok::R(*r)]),
        Op::PutByte { r } => i("PUTC", vec![Tok::R(*r)]),
        Op::Stream { stderr } => i(
            "STREAM",
            vec![Tok::Word(if *stderr { "2" } else { "1" }.to_string())],
        ),
        Op::Exit { status } => i("EXIT", vec![Tok::Word(status.to_string())]),
        Op::GetLang { r } => i("GETLANG", vec![Tok::R(*r)]),
        Op::ArgCount { r } => i("ARGC", vec![Tok::R(*r)]),
        Op::ArgByte { dst, arg, index } => {
            i("ARGBYTE", vec![Tok::R(*dst), Tok::R(*arg), Tok::R(*index)])
        }
        Op::BinV { op, dst, a, b } => i(binop_name(*op), vec![Tok::R(*dst), val(*a), val(*b)]),
        Op::UnV { op, dst, src } => i(unop_name(*op), vec![Tok::R(*dst), val(*src)]),
        Op::BranchV { cmp, a, b, to } => i(cmp_name(*cmp), vec![val(*a), val(*b), Tok::Label(*to)]),
        Op::LoadV { dst, addr } => i("LOAD", vec![Tok::R(*dst), val(*addr)]),
        Op::StoreV { src, addr } => i("STORE", vec![val(*src), val(*addr)]),
        Op::PutByteV { v } => i("PUTC", vec![val(*v)]),
        Op::CharOutV { v } => i("CHAROUT", vec![val(*v)]),
        Op::PushV { v } => i("PUSH", vec![val(*v)]),
        Op::ArgByteV { dst, arg, index } => {
            i("ARGBYTE", vec![Tok::R(*dst), val(*arg), val(*index)])
        }
    })
}

fn plain_tok(t: &Tok) -> String {
    match t {
        Tok::Q(k) => format!("q{}", k),
        Tok::O(k) => format!("o{}", k),
        Tok::C(k) => format!("c{}", k),
        Tok::R(k) => format!("R{}", k),
        Tok::Imm(x) => format!("({})", x),
        Tok::Word(w) => w.clone(),
        Tok::Label(a) => label(*a),
        Tok::Text(s) => escape(s),
    }
}

fn plain(ins: &Instr) -> String {
    let mut s = String::from(ins.name);
    for t in &ins.toks {
        s.push(' ');
        s.push_str(&plain_tok(t));
    }
    if let Some(d) = &ins.dest {
        s.push_str(" -> ");
        s.push_str(&plain_tok(d));
    }
    s
}

// jauzi eta dei guztien helburuak
fn targets(p: &Program) -> BTreeSet<usize> {
    p.ops.iter().filter_map(Op::jump_target).collect()
}

fn header(p: &Program) -> Vec<String> {
    let mut out = Vec::new();
    if p.n_qubits > 0 {
        out.push(format!("QINIT {}", p.n_qubits));
    }
    if p.n_modes > 0 {
        out.push(format!("QMODE {} {}", p.n_modes, p.cutoff));
    }
    out
}

fn instrs(p: &Program) -> Result<Vec<Instr>, String> {
    p.ops
        .iter()
        .enumerate()
        .map(|(a, op)| instr(op, a))
        .collect()
}

// berriz mihiztatu eta konparatu: idazten dena programa bera dela bermatzeko
fn same_program(a: &Program, b: &Program) -> Result<(), String> {
    let fields = (a.n_qubits, a.n_modes, a.n_bits) == (b.n_qubits, b.n_modes, b.n_bits)
        && (a.n_modes == 0 || a.cutoff == b.cutoff);
    if !fields {
        return Err(t!(
            "the export describes a different system ({} qubits, {} modes, {} bits instead of {}, {}, {}); this is a QOA bug",
            "esportazioak beste sistema bat deskribatzen du ({} qubit, {} modu, {} bit, {}, {}, {} beharrean); QOA-ren akats bat da",
            b.n_qubits,
            b.n_modes,
            b.n_bits,
            a.n_qubits,
            a.n_modes,
            a.n_bits
        ));
    }
    if let Some(k) = (0..a.ops.len().max(b.ops.len())).find(|&k| a.ops.get(k) != b.ops.get(k)) {
        return Err(t!(
            "the export does not reassemble to the same program (first difference at 0x{:x}: {:?} instead of {:?}); this is a QOA bug",
            "esportazioa ez da programa berera mihiztatzen (lehen aldea 0x{:x}-n: {:?}, {:?} beharrean); QOA-ren akats bat da",
            k,
            b.ops.get(k),
            a.ops.get(k)
        ));
    }
    Ok(())
}

fn check_plain(p: &Program, text: &str) -> Result<(), String> {
    let back = crate::asm::assemble(text, &p.name).map_err(|d| {
        t!(
            "the export does not reassemble ({}); this is a QOA bug",
            "esportazioa ez da berriz mihiztatzen ({}); QOA-ren akats bat da",
            d.first().map_or(String::new(), |d| d.message.clone())
        )
    })?;
    same_program(p, &back)
}

// .qx fitxategi batetik kargatutako programa bat onartu aurretik: QOA-k bere testu-formatik sortzen duen
// programa bera izan behar du (eragiketak, qubitak, moduak, fotoi-muga eta bitak). horrela makinak ez du
// inoiz ikusten mihiztatzaileak onartuko ez lukeen ezer: erregistro, qubit, modu, bit edo jauzi-helburu
// baliogaberik, begizta okerrik edo neurri erraldoirik, fitxategia nahita aldatuta egonda ere
pub fn check_loaded(p: &Program) -> Result<(), String> {
    // izena mezuetan eta txostenean erakusten da: kontrol-karaktererik gabe (terminal-sekuentziak)
    if p.name.chars().any(|c| c.is_control()) || p.name.len() > 4096 {
        return Err(t!(
            "its program name contains control characters or is too long",
            "bere programa-izenak kontrol-karaktereak ditu edo luzeegia da"
        ));
    }
    let ins = instrs(p)?;
    let labels = targets(p);
    let mut out = header(p);
    // testuaren lerro bakoitza zein eragiketatik datorren, mezuetarako
    let mut op_of_line: Vec<Option<usize>> = vec![None; out.len()];
    for (a, i) in ins.iter().enumerate() {
        if labels.contains(&a) {
            out.push(format!("{}:", label(a)));
            op_of_line.push(None);
        }
        out.push(plain(i));
        op_of_line.push(Some(a));
    }
    if labels.contains(&ins.len()) {
        out.push(format!("{}:", label(ins.len())));
    }
    let text = out.join("\n") + "\n";
    let back = crate::asm::assemble(&text, &p.name).map_err(|d| {
        let d = &d[0];
        let line = (d.line as usize).checked_sub(1);
        match line.and_then(|l| op_of_line.get(l).copied().flatten()) {
            Some(a) => t!(
                "operation 0x{:x} (`{}`) is not valid: {}",
                "0x{:x} eragiketa (`{}`) ez da baliozkoa: {}",
                a,
                out[line.unwrap_or(0)],
                d.message
            ),
            None => d.message.clone(),
        }
    })?;
    let system = (p.n_qubits, p.n_modes, p.n_bits) == (back.n_qubits, back.n_modes, back.n_bits)
        && (p.n_modes == 0 || p.cutoff == back.cutoff);
    if !system {
        return Err(t!(
            "it declares {} qubit(s), {} mode(s) and {} bit(s), but its operations make {}, {} and {}",
            "{} qubit, {} modu eta {} bit adierazten ditu, baina bere eragiketek {}, {} eta {} egiten dituzte",
            p.n_qubits,
            p.n_modes,
            p.n_bits,
            back.n_qubits,
            back.n_modes,
            back.n_bits
        ));
    }
    if let Some(k) = (0..p.ops.len().max(back.ops.len())).find(|&k| p.ops.get(k) != back.ops.get(k))
    {
        return Err(t!(
            "operation 0x{:x} is not what QOA builds from its own text (`{}`)",
            "0x{:x} eragiketa ez da QOA-k bere testutik eraikitzen duena (`{}`)",
            k,
            ins.get(k).map(plain).unwrap_or_default()
        ));
    }
    Ok(())
}

// QOA arrunt kanonikoa
pub fn to_qoa(p: &Program) -> Result<String, String> {
    let ins = instrs(p)?;
    let labels = targets(p);
    let mut out = header(p);
    for (a, i) in ins.iter().enumerate() {
        if labels.contains(&a) {
            out.push(format!("{}:", label(a)));
        }
        out.push(plain(i));
    }
    if labels.contains(&ins.len()) {
        out.push(format!("{}:", label(ins.len())));
    }
    let text = out.join("\n") + "\n";
    check_plain(p, &text)?;
    Ok(text)
}

// zerrenda: QOA kanonikoa, eragiketa bakoitzaren helbidea eta iturburu-lerroa iruzkinetan
pub fn listing(p: &Program) -> Result<String, String> {
    let ins = instrs(p)?;
    let labels = targets(p);
    let mut out = vec![
        format!("; {}", p.name),
        t!(
            "; {} qubit(s), {} mode(s) (cutoff {}), {} classical bit(s); {} operation(s)",
            "; {} qubit, {} modu ({} muga), {} bit klasiko; {} eragiketa",
            p.n_qubits,
            p.n_modes,
            p.cutoff,
            p.n_bits,
            p.ops.len()
        ),
        t!(
            "; each line: instruction ; address  source line",
            "; lerro bakoitza: agindua ; helbidea  iturburu-lerroa"
        ),
    ];
    out.extend(header(p));
    let width = ins
        .iter()
        .map(|i| plain(i).len())
        .max()
        .unwrap_or(0)
        .min(56);
    for (a, i) in ins.iter().enumerate() {
        if labels.contains(&a) {
            out.push(format!("{}:", label(a)));
        }
        let line = p.lines.get(a).copied().unwrap_or(0);
        out.push(format!(
            "{:<width$}  ; 0x{:04x}  {}",
            plain(i),
            a,
            if line == 0 {
                ts!("generated", "sortua").to_string()
            } else {
                t!("line {}", "{}. lerroa", line)
            },
            width = width
        ));
    }
    if labels.contains(&ins.len()) {
        out.push(format!("{}:", label(ins.len())));
    }
    let text = out.join("\n") + "\n";
    check_plain(p, &text)?;
    Ok(text)
}

// programa json gisa
pub fn to_json(p: &Program) -> Result<String, String> {
    let ins = instrs(p)?;
    let ops: Vec<serde_json::Value> = p
        .ops
        .iter()
        .enumerate()
        .map(|(a, op)| {
            serde_json::json!({
                "address": a,
                "line": p.lines.get(a).copied().unwrap_or(0),
                "text": plain(&ins[a]),
                "op": op,
            })
        })
        .collect();
    let doc = serde_json::json!({
        "program": p.name,
        "qubits": p.n_qubits,
        "modes": p.n_modes,
        "cutoff": p.cutoff,
        "bits": p.n_bits,
        "amplitudes": p.state_dim().map(|d| d.to_string()),
        "ops": ops,
    });
    serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())
}

// kontrol-fluxuaren grafoa: oinarrizko blokeak (jauzi-helburuetan eta jauzien ondoren mozten dira)
pub fn to_dot(p: &Program) -> Result<String, String> {
    let ins = instrs(p)?;
    let n = p.ops.len();
    let mut leaders: BTreeSet<usize> = targets(p);
    leaders.insert(0);
    for (a, op) in p.ops.iter().enumerate() {
        match op {
            Op::Jump { .. }
            | Op::JumpIf { .. }
            | Op::Branch { .. }
            | Op::BranchV { .. }
            | Op::Call { .. }
            | Op::Return
            | Op::Halt
            | Op::LoopEnd { .. } => {
                leaders.insert(a + 1);
            }
            Op::LoopStart { end, .. } => {
                leaders.insert(a + 1);
                leaders.insert(*end);
            }
            _ => {}
        }
    }
    let starts: Vec<usize> = leaders.into_iter().filter(|&a| a < n).collect();
    let block_of = |a: usize| -> String {
        if a >= n {
            "end".into()
        } else {
            let k = starts.partition_point(|&s| s <= a) - 1;
            format!("b{:x}", starts[k])
        }
    };
    let quote = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let mut out = vec![
        format!("digraph \"{}\" {{", quote(&p.name)),
        "  node [shape=box, fontname=monospace, fontsize=10];".into(),
        "  end [shape=oval, label=\"end\"];".into(),
    ];
    for (k, &s) in starts.iter().enumerate() {
        let e = starts.get(k + 1).copied().unwrap_or(n);
        let mut lines = vec![format!("0x{:04x}", s)];
        for (a, i) in ins.iter().enumerate().take(e).skip(s) {
            if a - s == 12 {
                lines.push(t!("... {} more", "... beste {}", e - a));
                break;
            }
            lines.push(plain(i));
        }
        let body: Vec<String> = lines.iter().map(|l| quote(l)).collect();
        out.push(format!("  b{:x} [label=\"{}\\l\"];", s, body.join("\\l")));
        let last = e - 1;
        let edge = |to: usize, label: &str, style: &str| {
            format!(
                "  b{:x} -> {} [label=\"{}\"{}];",
                s,
                block_of(to),
                label,
                style
            )
        };
        match &p.ops[last] {
            Op::Jump { to } => out.push(edge(*to, "", "")),
            Op::JumpIf { src, value, to } => {
                let what = match src {
                    Source::Reg(k) => format!("R{}", k),
                    Source::Bit(k) => format!("c{}", k),
                };
                out.push(edge(*to, &format!("{} = {}", what, value), ""));
                out.push(edge(e, ts!("else", "bestela"), ", style=dashed"));
            }
            Op::Branch { to, .. } | Op::BranchV { to, .. } => {
                out.push(edge(*to, ts!("if", "baldin"), ""));
                out.push(edge(e, ts!("else", "bestela"), ", style=dashed"));
            }
            Op::Call { to } => {
                out.push(edge(*to, ts!("call", "deia"), ", style=bold"));
                out.push(edge(
                    e,
                    ts!("after return", "itzuli ondoren"),
                    ", style=dotted",
                ));
            }
            Op::Return => out.push(format!(
                "  b{:x} -> end [label=\"{}\", style=dotted];",
                s,
                ts!("return to the caller", "deitzailera itzuli")
            )),
            Op::Halt => out.push(edge(n, "", "")),
            Op::LoopStart { end, .. } => {
                out.push(edge(e, ts!("loop", "begizta"), ""));
                out.push(edge(*end, ts!("count 0", "0 aldiz"), ", style=dashed"));
            }
            Op::LoopEnd { start } => {
                out.push(edge(start + 1, ts!("again", "berriz"), ""));
                out.push(edge(e, ts!("done", "amaituta"), ", style=dashed"));
            }
            _ => out.push(edge(e, "", "")),
        }
    }
    out.push("}".into());
    Ok(out.join("\n") + "\n")
}

// glifo-esportazioaren aukerak (`qoa export --to glyph`)
#[derive(Clone, Debug)]
pub struct GlyphOptions {
    // lerro baten zabalera gehienezkoa (0: adierazpen bat lerroko)
    pub width: usize,
    // qubit, bit eta modu erregistroen izenak
    pub qubits: String,
    pub bits: String,
    pub modes: String,
    // M eta CX ordez MEASURE eta CNOT
    pub long_names: bool,
    // ondoz ondoko ate berdinak tarte batean bildu (H/Q[0..3])
    pub ranges: bool,
    // indizeak hamaseitarrean ([0x3])
    pub hex: bool,
}

impl Default for GlyphOptions {
    fn default() -> Self {
        GlyphOptions {
            width: 100,
            qubits: "Q".into(),
            bits: "C".into(),
            modes: "L".into(),
            long_names: false,
            ranges: true,
            hex: false,
        }
    }
}

// glifo-adierazpen bat tarteak bildu aurretik: izena, testuak, eragigaiak, parametroak, % argumentuak,
// helmuga; edo forma berezi bat (etiketa, jauzia...) testu gisa
#[derive(Clone, Debug, PartialEq)]
enum Stmt {
    Op {
        name: String,
        texts: Vec<String>,
        operands: Vec<Tok>,
        params: Vec<String>,
        args: Vec<Tok>,
        dest: Option<Tok>,
    },
    Label(usize),
    Raw(String),
}

fn is_register(t: &Tok) -> bool {
    matches!(t, Tok::Q(_) | Tok::O(_) | Tok::C(_) | Tok::R(_))
}

fn glyph_stmt(ins: &Instr, o: &GlyphOptions) -> Stmt {
    let name = match ins.name {
        "MEASURE" if !o.long_names => "M",
        "CNOT" if !o.long_names => "CX",
        n => n,
    }
    .to_string();
    let label_tok = |t: &Tok| match t {
        Tok::Label(a) => format!("0x{:x}", a),
        _ => String::new(),
    };
    let reg_src = |t: &Tok| glyph_tok(t, o);
    match (ins.name, ins.toks.as_slice()) {
        ("JUMP", [l]) => return Stmt::Raw(format!(">{}", label_tok(l))),
        ("CALL", [l]) => return Stmt::Raw(format!("&{}", label_tok(l))),
        ("JUMPIFONE", [v, l]) => return Stmt::Raw(format!("?{}>{}", reg_src(v), label_tok(l))),
        ("JUMPIFZERO", [v, l]) => return Stmt::Raw(format!("!{}>{}", reg_src(v), label_tok(l))),
        ("RETSUB", []) => return Stmt::Raw("^".into()),
        ("LOOPSTART", [n]) => return Stmt::Raw(format!("*{}{{", reg_src(n))),
        ("LOOPEND", []) => return Stmt::Raw("}".into()),
        ("DUMPSTATE", []) => return Stmt::Raw("<<>>".into()),
        ("DUMPREGS", []) => return Stmt::Raw("<<#>>".into()),
        ("HALT", []) => return Stmt::Raw("END".into()),
        _ => {}
    }
    let lead = ins
        .toks
        .iter()
        .take_while(|t| matches!(t, Tok::Text(_)))
        .count();
    let rest = &ins.toks[lead..];
    if rest.iter().any(|t| matches!(t, Tok::Text(_))) {
        // testua eragigaien ondoren (VERBOSELOG): glifo-formak ez du hori adierazten, QOA arrunta bere horretan
        return Stmt::Raw(format!("`{}`", plain(ins)));
    }
    let texts = ins.toks[..lead]
        .iter()
        .map(|t| match t {
            Tok::Text(s) => escape(s),
            _ => unreachable!(),
        })
        .collect();
    if ins.name == "PRINTF" {
        return Stmt::Op {
            name: "PF".into(),
            texts,
            operands: vec![],
            params: vec![],
            args: rest.to_vec(),
            dest: None,
        };
    }
    // erregistroak lehenik eta zenbakiak amaieran: `RZ(0.5)/Q[0]`; bestela dena eragigai gisa, ordenan
    let regs = rest.iter().take_while(|t| is_register(t)).count();
    let (operands, params) = if rest[regs..].iter().all(|t| matches!(t, Tok::Word(_))) && regs > 0 {
        let params = rest[regs..]
            .iter()
            .map(|t| match t {
                Tok::Word(w) => w.clone(),
                _ => unreachable!(),
            })
            .collect();
        (rest[..regs].to_vec(), params)
    } else {
        (rest.to_vec(), vec![])
    };
    Stmt::Op {
        name,
        texts,
        operands,
        params,
        args: vec![],
        dest: ins.dest.clone(),
    }
}

fn index(k: u32, o: &GlyphOptions) -> String {
    if o.hex {
        format!("0x{:x}", k)
    } else {
        k.to_string()
    }
}

fn glyph_tok(t: &Tok, o: &GlyphOptions) -> String {
    match t {
        Tok::Q(k) => format!("{}[{}]", o.qubits, index(*k, o)),
        Tok::O(k) => format!("{}[{}]", o.modes, index(*k, o)),
        Tok::C(k) => format!("#{}[{}]", o.bits, index(*k, o)),
        Tok::R(k) => format!("#{}", k),
        // glifoan zenbaki huts bat erregistro edo zenbaki baten lekuan balio bat da: `ADD/#1,#1,1`
        Tok::Imm(x) => format!("{}", x),
        Tok::Label(a) => label(*a),
        Tok::Word(w) => {
            if w.starts_with(|c: char| c.is_ascii_digit() || c == '-' || c == '.')
                && w.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-.+".contains(c))
                || w.chars().all(|c| c.is_ascii_alphabetic())
            {
                w.clone()
            } else {
                format!("({})", w)
            }
        }
        Tok::Text(s) => escape(s),
    }
}

// elementu-zerrenda bat (tarte batean bildua): erregistro bera, indizeak ondoz ondo
fn glyph_list(items: &[Tok], o: &GlyphOptions) -> String {
    let first = &items[0];
    if items.len() == 1 || items.iter().all(|t| t == first) {
        return glyph_tok(first, o);
    }
    let (a, b) = match (first, &items[items.len() - 1]) {
        (Tok::Q(a), Tok::Q(b)) | (Tok::O(a), Tok::O(b)) | (Tok::C(a), Tok::C(b)) => (*a, *b),
        _ => unreachable!(),
    };
    let reg = match first {
        Tok::Q(_) => o.qubits.clone(),
        Tok::O(_) => o.modes.clone(),
        _ => format!("#{}", o.bits),
    };
    format!("{}[{}..{}]", reg, index(a, o), index(b, o))
}

// bi tokenek tarte bat jarraitzen duten: erregistro bera eta hurrengo indizea
fn next_of(a: &Tok, b: &Tok) -> bool {
    matches!((a, b), (Tok::Q(x), Tok::Q(y)) | (Tok::O(x), Tok::O(y)) | (Tok::C(x), Tok::C(y)) if *y == x + 1)
}

// ondoz ondoko adierazpenak tarte batean bil daitezkeen: izen, testu eta parametro berak, eta eragigai
// bakoitza finko edo urrats bateko aurrerapena (lehen biek erabakitzen dute zein)
fn joins(group: &[Vec<Tok>], next: &[Tok]) -> bool {
    let first = &group[0];
    if first.len() != next.len() {
        return false;
    }
    (0..first.len()).all(|k| {
        let last = &group[group.len() - 1][k];
        if group.len() == 1 {
            last == &next[k] || next_of(last, &next[k])
        } else if group[1][k] == first[k] {
            next[k] == first[k]
        } else {
            next_of(last, &next[k])
        }
    }) && (0..first.len()).any(|k| first[k] != next[k])
}

pub fn to_glyph(p: &Program, o: &GlyphOptions) -> Result<String, String> {
    for name in [&o.qubits, &o.bits, &o.modes] {
        let ok = name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !ok || ["X", "Y", "Z", "END"].contains(&name.as_str()) {
            return Err(t!(
                "'{}' cannot be a register name (letters, digits and _, not X, Y, Z or END)",
                "'{}' ezin da erregistro-izena izan (letrak, digituak eta _, ez X, Y, Z edo END)",
                name
            ));
        }
    }
    if o.qubits == o.bits || o.qubits == o.modes || o.bits == o.modes {
        return Err(t!(
            "the qubit, bit and mode registers need different names",
            "qubit, bit eta modu erregistroek izen desberdinak behar dituzte"
        ));
    }
    let ins = instrs(p)?;
    let labels = targets(p);
    let mut stmts: Vec<Stmt> = Vec::new();
    for (a, i) in ins.iter().enumerate() {
        if labels.contains(&a) {
            stmts.push(Stmt::Label(a));
        }
        stmts.push(glyph_stmt(i, o));
    }
    if labels.contains(&ins.len()) {
        stmts.push(Stmt::Label(ins.len()));
    }

    // tarteak: ondoz ondoko Op adierazpenak, forma berekoak
    let mut text_stmts: Vec<(String, bool)> = Vec::new();
    let mut k = 0;
    while k < stmts.len() {
        match &stmts[k] {
            Stmt::Label(a) => {
                text_stmts.push((format!("0x{:x}:;", a), true));
                k += 1;
            }
            Stmt::Raw(s) => {
                let block = s.ends_with('{') || s == "}";
                text_stmts.push((if block { s.clone() } else { format!("{};", s) }, false));
                k += 1;
            }
            Stmt::Op {
                name,
                texts,
                operands,
                params,
                args,
                dest,
            } => {
                let row = |s: &Stmt| -> Option<Vec<Tok>> {
                    match s {
                        Stmt::Op {
                            name: n,
                            texts: t,
                            operands: ops,
                            params: ps,
                            args: a,
                            dest: d,
                        } if n == name
                            && t == texts
                            && ps == params
                            && a == args
                            && a.is_empty() =>
                        {
                            let mut v = ops.clone();
                            v.extend(d.clone());
                            (d.is_some() == dest.is_some()).then_some(v)
                        }
                        _ => None,
                    }
                };
                let mut group = vec![row(&stmts[k]).unwrap_or_default()];
                let mut j = k + 1;
                if o.ranges && args.is_empty() && !group[0].is_empty() {
                    while let Some(next) = stmts.get(j).and_then(row) {
                        if !joins(&group, &next) {
                            break;
                        }
                        group.push(next);
                        j += 1;
                    }
                }
                let n_ops = operands.len();
                let column =
                    |c: usize| -> Vec<Tok> { group.iter().map(|r| r[c].clone()).collect() };
                let mut s = name.clone();
                if !params.is_empty() {
                    s.push_str(&format!("({})", params.join(", ")));
                }
                for t in texts {
                    s.push_str(t);
                }
                if n_ops > 0 {
                    s.push('/');
                    let cols: Vec<String> = (0..n_ops).map(|c| glyph_list(&column(c), o)).collect();
                    s.push_str(&cols.join(","));
                }
                for a in args {
                    s.push('%');
                    s.push_str(&glyph_tok(a, o));
                }
                if dest.is_some() {
                    s.push_str("$>");
                    s.push_str(&glyph_list(&column(n_ops), o));
                }
                s.push(';');
                text_stmts.push((s, false));
                k = j;
            }
        }
    }

    let mut decls = Vec::new();
    if p.n_qubits > 0 {
        decls.push(format!("{}{{{}}};", o.qubits, p.n_qubits));
    }
    if p.n_bits > 0 {
        decls.push(format!("#{}{{{}}};", o.bits, p.n_bits));
    }
    if p.n_modes > 0 {
        decls.push(format!("~{}{{{}}}%{};", o.modes, p.n_modes, p.cutoff));
    }
    let mut lines = vec![format!("// {}", p.name)];
    if !decls.is_empty() {
        lines.push(decls.join(" "));
    }
    let mut cur = String::new();
    for (s, label_line) in text_stmts {
        let fits = o.width > 0 && cur.len() + 1 + s.len() <= o.width;
        if !cur.is_empty() && (label_line || !fits) {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(&s);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    let text = lines.join("\n") + "\n";
    let back = crate::glyph::assemble(&text, &p.name).map_err(|d| {
        t!(
            "the glyph export does not assemble ({}); this is a QOA bug",
            "glifo-esportazioa ez da mihiztatzen ({}); QOA-ren akats bat da",
            d.first().map_or(String::new(), |d| d.message.clone())
        )
    })?;
    same_program(p, &back)?;
    Ok(text)
}
