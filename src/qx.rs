// fitxategi honek .qx formatu bitarra idazten eta irakurtzen du, kanpoko liburutegirik gabe.
//
//   "QOAX"  u16 bertsioa (little-endian)  programa  [SHA-256]
//
// programaren byteak 0.4.4 arte bincode 1-ek idazten zituen berberak dira, eta formatu horretan geratzen
// dira, fitxategi zahar guztiak irakurtzeko: zenbaki osoak little-endian eta luzera finkoan (u32 4 byte,
// u64 eta usize 8), f64 bere 8 byteak, bool byte bat (0 edo 1), Option byte bat (0 ez dago, 1 badago) eta
// gero balioa, enum bat u32 aldaera-zenbakia (deklarazio-ordenan, 0tik) eta gero haren eremuak, zerrenda
// eta testu bat u64 luzera eta gero elementuak (testua utf-8).
//
// 4. bertsiotik (QOA 0.4.5) fitxategiak 32 byteko SHA-256 batekin amaitzen dira, aurreko byte guztiena:
// byte bakar bat aldatzen bada (disko txar bat, deskarga eten bat, eskuz aldatua), QOA-k ez du kargatzen.
// tresna arruntekin ere egiazta daiteke: `head -c -32 f.qx | sha256sum` eta `tail -c 32 f.qx | xxd -p`.

use crate::program::*;
use crate::t;

pub type Res<T> = Result<T, String>;

pub struct W(pub Vec<u8>);

pub struct R<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> R<'a> {
    pub fn new(data: &'a [u8]) -> R<'a> {
        R { data, pos: 0 }
    }

    pub fn left(&self) -> usize {
        self.data.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        if self.left() < n {
            return Err(t!(
                "the file ends in the middle of the program",
                "fitxategia programaren erdian amaitzen da"
            ));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn arr<const N: usize>(&mut self) -> Res<[u8; N]> {
        Ok(self.take(N)?.try_into().expect("luzera zuzena"))
    }

    // zerrenda baten luzera: gutxienez `min` byte elementu bakoitzeko, beraz fitxategian sartu behar da
    // (luzera erraldoi batek ez du memoria erreserbatzen)
    fn len(&mut self, min: usize) -> Res<usize> {
        let n = u64::get(self)?;
        if n > (self.left() / min.max(1)) as u64 {
            return Err(t!(
                "a list of {} items does not fit in the rest of the file",
                "{} elementuko zerrenda bat ez da fitxategiaren gainerakoan sartzen",
                n
            ));
        }
        Ok(n as usize)
    }

    fn tag(&mut self, what: &str, n: u32) -> Res<u32> {
        let v = u32::get(self)?;
        if v >= n {
            return Err(t!(
                "unknown {} (number {})",
                "{} ezezaguna ({}. zenbakia)",
                what,
                v
            ));
        }
        Ok(v)
    }
}

pub trait Codec: Sized {
    fn put(&self, w: &mut W);
    fn get(r: &mut R) -> Res<Self>;
    // elementu batek behar duen byte-kopuru txikiena (zerrenden luzera egiaztatzeko)
    const MIN: usize;
}

macro_rules! num {
    ($($t:ty),*) => {$(
        impl Codec for $t {
            fn put(&self, w: &mut W) {
                w.0.extend_from_slice(&self.to_le_bytes());
            }
            fn get(r: &mut R) -> Res<Self> {
                Ok(<$t>::from_le_bytes(r.arr()?))
            }
            const MIN: usize = std::mem::size_of::<$t>();
        }
    )*};
}
num!(u8, u16, u32, u64, f64);

impl Codec for usize {
    fn put(&self, w: &mut W) {
        (*self as u64).put(w)
    }
    fn get(r: &mut R) -> Res<Self> {
        let v = u64::get(r)?;
        usize::try_from(v).map_err(|_| {
            t!(
                "the position {} is too large",
                "{} posizioa handiegia da",
                v
            )
        })
    }
    const MIN: usize = 8;
}

impl Codec for bool {
    fn put(&self, w: &mut W) {
        w.0.push(*self as u8)
    }
    fn get(r: &mut R) -> Res<Self> {
        match u8::get(r)? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(t!(
                "{} where only 0 or 1 fits",
                "{} 0 edo 1 bakarrik doan lekuan",
                v
            )),
        }
    }
    const MIN: usize = 1;
}

impl Codec for String {
    fn put(&self, w: &mut W) {
        self.len().put(w);
        w.0.extend_from_slice(self.as_bytes());
    }
    fn get(r: &mut R) -> Res<Self> {
        let n = r.len(1)?;
        let b = r.take(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| {
            t!(
                "a text that is not valid UTF-8",
                "utf-8 baliozkoa ez den testu bat"
            )
        })
    }
    const MIN: usize = 8;
}

impl<T: Codec> Codec for Vec<T> {
    fn put(&self, w: &mut W) {
        self.len().put(w);
        for x in self {
            x.put(w);
        }
    }
    fn get(r: &mut R) -> Res<Self> {
        let n = r.len(T::MIN)?;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(T::get(r)?);
        }
        Ok(v)
    }
    const MIN: usize = 8;
}

impl<T: Codec> Codec for Option<T> {
    fn put(&self, w: &mut W) {
        match self {
            None => w.0.push(0),
            Some(x) => {
                w.0.push(1);
                x.put(w);
            }
        }
    }
    fn get(r: &mut R) -> Res<Self> {
        match u8::get(r)? {
            0 => Ok(None),
            1 => Ok(Some(T::get(r)?)),
            v => Err(t!(
                "{} where only 0 or 1 fits",
                "{} 0 edo 1 bakarrik doan lekuan",
                v
            )),
        }
    }
    const MIN: usize = 1;
}

// enum bat: `aldaera(eremuak)` zerrenda batetik, ordenan (zenbakia = posizioa). `put`-en match-a osoa da,
// beraz aldaera berri bat hemen idatzi gabe ez da konpilatzen
macro_rules! enum_codec {
    ($ty:ident, $what:expr, $min:expr; $($var:ident $({ $($f:ident),* })? $(( $($p:ident),* ))?;)*) => {
        impl Codec for $ty {
            #[allow(unused_assignments, unused_variables)]
            fn put(&self, w: &mut W) {
                let mut i = 0u32;
                $(
                    #[allow(unused_parens)]
                    if let $ty::$var $({ $($f),* })? $(( $($p),* ))? = self {
                        i.put(w);
                        $($($f.put(w);)*)?
                        $($($p.put(w);)*)?
                        return;
                    }
                    i += 1;
                )*
                unreachable!("aldaera guztiak zerrendan daude");
            }
            #[allow(unused_assignments)]
            fn get(r: &mut R) -> Res<Self> {
                const N: u32 = [$(stringify!($var)),*].len() as u32;
                let tag = r.tag(&$what, N)?;
                let mut i = 0u32;
                $(
                    if tag == i {
                        return Ok($ty::$var $({ $($f: Codec::get(r)?),* })? $(( $({ let _ = stringify!($p); Codec::get(r)? }),* ))?);
                    }
                    i += 1;
                )*
                unreachable!("tag < N")
            }
            const MIN: usize = $min;
        }
    };
}

// `put`-en `if let` kateak ez du egiaztatzen aldaera guztiak daudela; hau bai (konpilazioan)
fn _every_op_listed(op: &Op) {
    match op {
        Op::Gate { .. }
        | Op::Controlled { .. }
        | Op::PhaseAll { .. }
        | Op::Swap { .. }
        | Op::Bell { .. }
        | Op::Ghz { .. }
        | Op::Cluster { .. }
        | Op::EntangleSwap { .. }
        | Op::Measure { .. }
        | Op::MeasureAll
        | Op::Reset { .. }
        | Op::ResetAll
        | Op::ToBasis { .. }
        | Op::Emit { .. }
        | Op::PhotonCount { .. }
        | Op::PhotonDetect { .. }
        | Op::Parity { .. }
        | Op::Coincidence { .. }
        | Op::Displace { .. }
        | Op::DisplaceReg { .. }
        | Op::Squeeze { .. }
        | Op::SqueezeReg { .. }
        | Op::ModePhase { .. }
        | Op::Kerr { .. }
        | Op::CrossKerr { .. }
        | Op::BeamSplitter { .. }
        | Op::ModeSwap { .. }
        | Op::AddPhoton { .. }
        | Op::SubPhoton { .. }
        | Op::Loss { .. }
        | Op::Fock { .. }
        | Op::Dispersive { .. }
        | Op::DumpState
        | Op::DumpRegs
        | Op::Expect { .. }
        | Op::BellCheck { .. }
        | Op::MeanPhoton { .. }
        | Op::Set { .. }
        | Op::Copy { .. }
        | Op::Bin { .. }
        | Op::Un { .. }
        | Op::Rand { .. }
        | Op::Seed { .. }
        | Op::Input { .. }
        | Op::Time { .. }
        | Op::Push { .. }
        | Op::Pop { .. }
        | Op::PackBits { .. }
        | Op::Jump { .. }
        | Op::JumpIf { .. }
        | Op::Branch { .. }
        | Op::Call { .. }
        | Op::Return
        | Op::LoopStart { .. }
        | Op::LoopEnd { .. }
        | Op::Halt
        | Op::Barrier
        | Op::Print { .. }
        | Op::Printf { .. }
        | Op::CharOut { .. }
        | Op::Log { .. }
        | Op::Load { .. }
        | Op::Store { .. }
        | Op::StoreText { .. }
        | Op::GetByte { .. }
        | Op::PutByte { .. }
        | Op::Stream { .. }
        | Op::Exit { .. }
        | Op::GetLang { .. }
        | Op::ArgCount { .. }
        | Op::ArgByte { .. }
        | Op::BinV { .. }
        | Op::UnV { .. }
        | Op::BranchV { .. }
        | Op::LoadV { .. }
        | Op::StoreV { .. }
        | Op::PutByteV { .. }
        | Op::CharOutV { .. }
        | Op::PushV { .. }
        | Op::ArgByteV { .. } => {}
    }
}

enum_codec!(Gate, t!("gate", "ate"), 4;
    H; X; Y; Z; S; Sdg; T; Tdg; SX; SXdg;
    RX(a); RY(a); RZ(a); P(a); U(a, b, c);
);
enum_codec!(Basis, t!("basis", "oinarri"), 4; X; Y; Z;);
enum_codec!(Dest, t!("destination", "helburu"), 4; Own; Bit(a); Reg(a););
enum_codec!(Source, t!("source", "iturri"), 4; Reg(a); Bit(a););
enum_codec!(Count, t!("count", "kopuru"), 4; Lit(a); Reg(a););
enum_codec!(Val, t!("value", "balio"), 4; Reg(a); Num(a););
enum_codec!(BinOp, t!("operation", "eragiketa"), 4; Add; Sub; Mul; Div; And; Or; Xor; Shl; Shr; Pow;);
enum_codec!(UnOp, t!("operation", "eragiketa"), 4; Sqrt; Exp; Log; Not; Sin; Cos; Tan;);
enum_codec!(Cmp, t!("comparison", "konparazio"), 4; Eq; Ne; Gt; Lt;);

enum_codec!(Op, t!("operation", "eragiketa"), 4;
    Gate { q, gate };
    Controlled { controls, q, gate };
    PhaseAll { qubits, theta };
    Swap { a, b, controls };
    Bell { a, b };
    Ghz { qubits };
    Cluster { qubits };
    EntangleSwap { a, b, c, d };
    Measure { q, basis, dest };
    MeasureAll;
    Reset { q };
    ResetAll;
    ToBasis { q, basis };
    Emit { q, o };
    PhotonCount { o, dest };
    PhotonDetect { o, dest };
    Parity { o, dest };
    Coincidence { modes, dest };
    Displace { o, re, im };
    DisplaceReg { o, r };
    Squeeze { o, r, phi };
    SqueezeReg { o, reg };
    ModePhase { o, theta };
    Kerr { o, kappa };
    CrossKerr { a, b, kappa };
    BeamSplitter { a, b, theta, phi };
    ModeSwap { a, b };
    AddPhoton { o };
    SubPhoton { o };
    Loss { o, gamma };
    Fock { o, n };
    Dispersive { q, o, theta };
    DumpState;
    DumpRegs;
    Expect { q, basis, dest };
    BellCheck { a, b, dest };
    MeanPhoton { o, dest };
    Set { r, value };
    Copy { dst, src };
    Bin { op, dst, a, b };
    Un { op, dst, src };
    Rand { r };
    Seed { seed };
    Input { r };
    Time { r };
    Push { r };
    Pop { r };
    PackBits { dst, bits };
    Jump { to };
    JumpIf { src, value, to };
    Branch { cmp, a, b, to };
    Call { to };
    Return;
    LoopStart { count, end };
    LoopEnd { start };
    Halt;
    Barrier;
    Print { text, newline };
    Printf { format, regs };
    CharOut { r };
    Log { q, text };
    Load { dst, addr };
    Store { src, addr };
    StoreText { addr, text };
    GetByte { r };
    PutByte { r };
    Stream { stderr };
    Exit { status };
    GetLang { r };
    ArgCount { r };
    ArgByte { dst, arg, index };
    BinV { op, dst, a, b };
    UnV { op, dst, src };
    BranchV { cmp, a, b, to };
    LoadV { dst, addr };
    StoreV { src, addr };
    PutByteV { v };
    CharOutV { v };
    PushV { v };
    ArgByteV { dst, arg, index };
);

impl Codec for Program {
    fn put(&self, w: &mut W) {
        self.name.put(w);
        self.n_qubits.put(w);
        self.n_modes.put(w);
        self.cutoff.put(w);
        self.cutoff_declared.put(w);
        self.n_bits.put(w);
        self.ops.put(w);
        self.lines.put(w);
    }
    fn get(r: &mut R) -> Res<Self> {
        Ok(Program {
            name: Codec::get(r)?,
            n_qubits: Codec::get(r)?,
            n_modes: Codec::get(r)?,
            cutoff: Codec::get(r)?,
            cutoff_declared: Codec::get(r)?,
            n_bits: Codec::get(r)?,
            ops: Codec::get(r)?,
            lines: Codec::get(r)?,
        })
    }
    const MIN: usize = 8;
}

// programaren byteak (goibururik eta laburpenik gabe)
pub fn body(p: &Program) -> Vec<u8> {
    let mut w = W(Vec::new());
    p.put(&mut w);
    w.0
}

// programaren byteak irakurri; ondoren ezer ez da geratu behar
pub fn read_body(data: &[u8]) -> Res<Program> {
    let mut r = R::new(data);
    let p = Program::get(&mut r)?;
    if r.left() > 0 {
        return Err(t!(
            "{} unexpected bytes after the program",
            "{} byte espero gabe programaren ondoren",
            r.left()
        ));
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    // programa guztiak: idatzi, irakurri, eta byte berberak berriz
    #[test]
    fn every_program_round_trips() {
        let mut n = 0;
        for dir in [
            "examples",
            "tests/programs/sanity",
            "tests/programs/stress",
            "src/qoa",
        ] {
            let Ok(rd) = std::fs::read_dir(dir) else {
                continue;
            };
            for e in rd {
                let path = e.unwrap().path();
                let text = std::fs::read_to_string(&path).unwrap_or_default();
                let name = path.to_string_lossy().to_string();
                let p = if name.ends_with(".qoa") {
                    crate::asm::assemble(&text, &name).ok()
                } else if name.ends_with(".qg") {
                    crate::glyph::assemble(&text, &name).ok()
                } else if name.ends_with(".qasm") {
                    crate::qasm::to_qoa(&text, &name)
                        .ok()
                        .and_then(|q| crate::asm::assemble(&q, &name).ok())
                } else {
                    continue;
                };
                let Some(p) = p else { continue };
                let b = body(&p);
                let back = read_body(&b).unwrap();
                assert_eq!(back, p, "{}", name);
                assert_eq!(body(&back), b, "{}", name);
                n += 1;
            }
        }
        assert!(n > 50, "{} programs", n);
    }

    // aldaera guztiak, balio bitxiekin (NaN, infinitua, -0, testu ez-ascii, zerrenda hutsak)
    pub(crate) fn every_op() -> Vec<Op> {
        use crate::program::Gate as G;
        use Op::*;
        let g = [
            G::H,
            G::X,
            G::Y,
            G::Z,
            G::S,
            G::Sdg,
            G::T,
            G::Tdg,
            G::SX,
            G::SXdg,
            G::RX(-0.0),
            G::RY(f64::NAN),
            G::RZ(f64::INFINITY),
            G::P(1e-300),
            G::U(1.0, -2.5, 3.25),
        ];
        let d = [Dest::Own, Dest::Bit(7), Dest::Reg(31)];
        let v = [Val::Reg(3), Val::Num(-1.5), Val::Num(f64::NEG_INFINITY)];
        let mut ops: Vec<Op> = g.iter().map(|&gate| Op::Gate { q: 63, gate }).collect();
        ops.extend(g.iter().map(|&gate| Controlled {
            controls: vec![1, 2, 3],
            q: 0,
            gate,
        }));
        for (i, &dest) in d.iter().enumerate() {
            let basis = [Basis::X, Basis::Y, Basis::Z][i];
            ops.push(Measure { q: 2, basis, dest });
            ops.push(PhotonCount { o: 1, dest });
            ops.push(PhotonDetect { o: 2, dest });
            ops.push(Parity { o: 3, dest });
            ops.push(Coincidence {
                modes: vec![],
                dest,
            });
            ops.push(ToBasis { q: 4, basis });
            ops.push(Expect {
                q: 1,
                basis,
                dest: [None, Some(0), Some(31)][i],
            });
        }
        ops.extend([
            PhaseAll {
                qubits: vec![0, 1],
                theta: 0.25,
            },
            Swap {
                a: 1,
                b: 2,
                controls: vec![],
            },
            Bell { a: 0, b: 1 },
            Ghz {
                qubits: vec![0, 1, 2],
            },
            Cluster { qubits: vec![3] },
            EntangleSwap {
                a: 0,
                b: 1,
                c: 2,
                d: 3,
            },
            MeasureAll,
            Reset { q: 5 },
            ResetAll,
            Emit { q: 0, o: 1 },
            Displace {
                o: 0,
                re: 0.5,
                im: -0.5,
            },
            DisplaceReg { o: 1, r: 2 },
            Squeeze {
                o: 0,
                r: 0.1,
                phi: 0.2,
            },
            SqueezeReg { o: 0, reg: 4 },
            ModePhase { o: 0, theta: 3.0 },
            Kerr { o: 1, kappa: 0.01 },
            CrossKerr {
                a: 0,
                b: 1,
                kappa: 0.02,
            },
            BeamSplitter {
                a: 0,
                b: 1,
                theta: 0.7,
                phi: 0.0,
            },
            ModeSwap { a: 1, b: 0 },
            AddPhoton { o: 0 },
            SubPhoton { o: 1 },
            Loss { o: 0, gamma: 0.3 },
            Fock { o: 1, n: 5 },
            Dispersive {
                q: 0,
                o: 0,
                theta: 0.4,
            },
            DumpState,
            DumpRegs,
            BellCheck {
                a: 0,
                b: 1,
                dest: Some(2),
            },
            MeanPhoton { o: 0, dest: None },
            Set { r: 1, value: 2.5 },
            Copy { dst: 1, src: 2 },
            Rand { r: 3 },
            Seed { seed: u64::MAX },
            Input { r: 4 },
            Time { r: 5 },
            Push { r: 6 },
            Pop { r: 7 },
            PackBits {
                dst: 8,
                bits: vec![0, 4095],
            },
            Jump { to: 0 },
            JumpIf {
                src: Source::Reg(1),
                value: 1.0,
                to: 3,
            },
            JumpIf {
                src: Source::Bit(2),
                value: 0.0,
                to: 4,
            },
            Call { to: 5 },
            Return,
            LoopStart {
                count: Count::Lit(10),
                end: 7,
            },
            LoopStart {
                count: Count::Reg(3),
                end: 8,
            },
            LoopEnd { start: 1 },
            Halt,
            Barrier,
            Print {
                text: "kaixo π 🎵\n\"".into(),
                newline: true,
            },
            Print {
                text: String::new(),
                newline: false,
            },
            Printf {
                format: "{} {}".into(),
                regs: vec![1, 2],
            },
            CharOut { r: 0 },
            Log {
                q: 0,
                text: "ñ".into(),
            },
            Load { dst: 1, addr: 2 },
            Store { src: 3, addr: 4 },
            StoreText {
                addr: 5,
                text: "abc".into(),
            },
            GetByte { r: 1 },
            PutByte { r: 2 },
            Stream { stderr: true },
            Stream { stderr: false },
            Exit { status: 255 },
            GetLang { r: 1 },
            ArgCount { r: 2 },
            ArgByte {
                dst: 1,
                arg: 2,
                index: 3,
            },
        ]);
        for op in [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::Div,
            BinOp::And,
            BinOp::Or,
            BinOp::Xor,
            BinOp::Shl,
            BinOp::Shr,
            BinOp::Pow,
        ] {
            ops.push(Bin {
                op,
                dst: 1,
                a: 2,
                b: 3,
            });
            ops.push(BinV {
                op,
                dst: 1,
                a: v[0],
                b: v[1],
            });
        }
        for op in [
            UnOp::Sqrt,
            UnOp::Exp,
            UnOp::Log,
            UnOp::Not,
            UnOp::Sin,
            UnOp::Cos,
            UnOp::Tan,
        ] {
            ops.push(Un { op, dst: 1, src: 2 });
            ops.push(UnV {
                op,
                dst: 1,
                src: v[2],
            });
        }
        for cmp in [Cmp::Eq, Cmp::Ne, Cmp::Gt, Cmp::Lt] {
            ops.push(Branch {
                cmp,
                a: 1,
                b: 2,
                to: 0,
            });
            ops.push(BranchV {
                cmp,
                a: v[1],
                b: v[0],
                to: 1,
            });
        }
        ops.extend([
            LoadV { dst: 1, addr: v[1] },
            StoreV {
                src: v[0],
                addr: v[1],
            },
            PutByteV { v: v[1] },
            CharOutV { v: v[0] },
            PushV { v: v[2] },
            ArgByteV {
                dst: 1,
                arg: v[0],
                index: v[1],
            },
        ]);
        ops
    }

    fn every_op_program() -> Program {
        let ops = every_op();
        Program {
            name: "proba π.qoa".into(),
            n_qubits: 64,
            n_modes: 3,
            cutoff: 9,
            cutoff_declared: true,
            n_bits: 4096,
            lines: (0..ops.len() as u32).collect(),
            ops,
        }
    }

    // tests/data/every-op-v3.qx bincode 1-ek idatzi zuen (QOA 0.4.4): aldaera guztiak, balio bitxiekin.
    // idazle eta irakurle honek byte berberak eman eta irakurri behar ditu, fitxategi zaharrak irakurtzeko
    #[test]
    fn every_operation_as_bincode_wrote_it() {
        let old = std::fs::read("tests/data/every-op-v3.qx").unwrap();
        assert_eq!(&old[..6], b"QOAX\x03\x00");
        let p = every_op_program();
        assert_eq!(body(&p), old[6..]);
        // NaN ez da bere buruaren berdina: byteak alderatu
        assert_eq!(body(&read_body(&old[6..]).unwrap()), old[6..]);
    }

    // QOA 0.4.4-k benetan idatzitako fitxategiak (1. eta 3. bertsioak, laburpenik gabe) kargatzen dira, eta
    // iturburutik orain eraikitako programa bera dira
    #[test]
    fn files_written_by_0_4_4_still_load() {
        for (file, src) in [
            ("tests/data/bell-0.4.4.qx", "examples/bell.qoa"),
            (
                "tests/data/conveniences-0.4.4.qx",
                "tests/programs/sanity/conveniences.qoa",
            ),
        ] {
            let old = std::fs::read(file).unwrap();
            let p = Program::from_bytes(&old).unwrap();
            let text = std::fs::read_to_string(src).unwrap();
            let mut now = crate::asm::assemble(&text, src).unwrap();
            now.name = p.name.clone();
            assert_eq!(p, now, "{}", file);
            assert_eq!(body(&p), old[6..], "{}", file);
        }
    }

    #[test]
    fn version_4_ends_with_the_sha256_of_everything_before() {
        let p = crate::asm::assemble("H q0\nCNOT q0 q1\nMEASURE q0 -> c0\n", "x.qoa").unwrap();
        let b = p.to_bytes();
        assert_eq!(&b[..6], b"QOAX\x04\x00");
        let (head, sum) = b.split_at(b.len() - 32);
        assert_eq!(sum, crate::sha256::digest(head));
        assert_eq!(&head[6..], body(&p));
        assert_eq!(Program::from_bytes(&b).unwrap(), p);
        // byte bakar bat aldatuta, edozein lekutan: ez da kargatzen
        for i in 0..b.len() {
            for bit in [0x01u8, 0x80] {
                let mut bad = b.clone();
                bad[i] ^= bit;
                assert!(
                    Program::from_bytes(&bad).is_err(),
                    "byte {} bit {:x}",
                    i,
                    bit
                );
            }
        }
        // moztuta edo luzatuta ere ez
        for n in 0..b.len() {
            assert!(Program::from_bytes(&b[..n]).is_err(), "{} byte", n);
        }
        let mut long = b.clone();
        long.push(0);
        assert!(Program::from_bytes(&long).is_err());
    }

    #[test]
    fn hostile_lengths_do_not_allocate() {
        // luzera erraldoiak: errore bat, memoria erreserbatu gabe
        let mut w = W(Vec::new());
        "x".to_string().put(&mut w);
        for v in [0u32, 0, 8, 0] {
            v.put(&mut w);
        }
        false.put(&mut w);
        0u32.put(&mut w);
        u64::MAX.put(&mut w);
        let e = read_body(&w.0).unwrap_err();
        assert!(e.contains("does not fit"), "{}", e);
    }
}
