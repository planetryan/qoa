// fitxategi honek qoa programa mihiztatuaren forma definitzen du: vm-ak exekutatzen dituen
// eragiketen zerrenda laua.
//
// aurrealde guztiek (qoa iturburua, openqasm inportazioa, etorkizuneko glifo-sintaxia) `Program` bat
// sortzen dute. `qoa compile`-k .qx fitxategi batean idazten du: "QOAX" byteak, u16 formatu-bertsioa,
// eta gero programa bincode-rekin serializatuta.

use crate::t;
use num_complex::Complex64 as C;
use serde::{Deserialize, Serialize};
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

// erregistro klasikoen kopurua: R0..R31
pub const NUM_REGS: usize = 32;

// qubit bakarreko atea
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Gate {
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
    T,
    Tdg,
    SX,
    SXdg,
    RX(f64),
    RY(f64),
    RZ(f64),
    // diag(1, e^{iθ})
    P(f64),
    // qubit bakarreko ate orokorra U(θ, φ, λ) (openqasm-en konbentzioa)
    U(f64, f64, f64),
}

impl Gate {
    // 2x2 matrize unitarioa, errenkadaka: [m00, m01, m10, m11]
    pub fn matrix(&self) -> [C; 4] {
        let z = C::new(0.0, 0.0);
        let one = C::new(1.0, 0.0);
        let i = C::new(0.0, 1.0);
        let r = FRAC_1_SQRT_2;
        match *self {
            Gate::H => [
                C::new(r, 0.0),
                C::new(r, 0.0),
                C::new(r, 0.0),
                C::new(-r, 0.0),
            ],
            Gate::X => [z, one, one, z],
            Gate::Y => [z, -i, i, z],
            Gate::Z => [one, z, z, -one],
            Gate::S => [one, z, z, i],
            Gate::Sdg => [one, z, z, -i],
            Gate::T => [one, z, z, C::from_polar(1.0, FRAC_PI_4)],
            Gate::Tdg => [one, z, z, C::from_polar(1.0, -FRAC_PI_4)],
            Gate::SX => [
                C::new(0.5, 0.5),
                C::new(0.5, -0.5),
                C::new(0.5, -0.5),
                C::new(0.5, 0.5),
            ],
            Gate::SXdg => [
                C::new(0.5, -0.5),
                C::new(0.5, 0.5),
                C::new(0.5, 0.5),
                C::new(0.5, -0.5),
            ],
            Gate::RX(t) => {
                let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
                [
                    C::new(c, 0.0),
                    C::new(0.0, -s),
                    C::new(0.0, -s),
                    C::new(c, 0.0),
                ]
            }
            Gate::RY(t) => {
                let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
                [
                    C::new(c, 0.0),
                    C::new(-s, 0.0),
                    C::new(s, 0.0),
                    C::new(c, 0.0),
                ]
            }
            Gate::RZ(t) => [
                C::from_polar(1.0, -t / 2.0),
                z,
                z,
                C::from_polar(1.0, t / 2.0),
            ],
            Gate::P(t) => [one, z, z, C::from_polar(1.0, t)],
            Gate::U(theta, phi, lambda) => {
                let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
                [
                    C::new(c, 0.0),
                    -C::from_polar(s, lambda),
                    C::from_polar(s, phi),
                    C::from_polar(c, phi + lambda),
                ]
            }
        }
    }

    // izena minuskulaz, openqasm-ek erabiltzen duen bezala
    pub fn qasm_name(&self) -> &'static str {
        match self {
            Gate::H => "h",
            Gate::X => "x",
            Gate::Y => "y",
            Gate::Z => "z",
            Gate::S => "s",
            Gate::Sdg => "sdg",
            Gate::T => "t",
            Gate::Tdg => "tdg",
            Gate::SX => "sx",
            Gate::SXdg => "sxdg",
            Gate::RX(_) => "rx",
            Gate::RY(_) => "ry",
            Gate::RZ(_) => "rz",
            Gate::P(_) => "p",
            Gate::U(..) => "u3",
        }
    }

    pub fn params(&self) -> Vec<f64> {
        match *self {
            Gate::RX(t) | Gate::RY(t) | Gate::RZ(t) | Gate::P(t) => vec![t],
            Gate::U(a, b, c) => vec![a, b, c],
            _ => vec![],
        }
    }

    // `GATE izena q` aginduak onartzen dituen izenak
    pub fn from_name(name: &str) -> Option<Gate> {
        Some(match name.to_ascii_lowercase().as_str() {
            "h" => Gate::H,
            "x" | "not" => Gate::X,
            "y" => Gate::Y,
            "z" => Gate::Z,
            "s" => Gate::S,
            "sdg" | "sdag" => Gate::Sdg,
            "t" => Gate::T,
            "tdg" | "tdag" => Gate::Tdg,
            "sx" => Gate::SX,
            "sxdg" => Gate::SXdg,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Basis {
    X,
    Y,
    Z,
}

impl Basis {
    pub fn name(&self) -> &'static str {
        match self {
            Basis::X => "X",
            Basis::Y => "Y",
            Basis::Z => "Z",
        }
    }
}

// neurketa baten emaitza nora idazten den
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dest {
    // neurtutakoaren irakurketa propioa: c<q> bita q qubitarentzat, o moduaren detektagailua
    Own,
    // cK bit klasikoa
    Bit(u32),
    // RK erregistroa
    Reg(u32),
}

// jauzi batek aztertzen duen balioa: erregistro bat edo bit klasiko bat
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    Reg(u32),
    Bit(u32),
}

// begizta-kopurua: literala edo erregistro baten balioa begiztan sartzean
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Count {
    Lit(u64),
    Reg(u32),
}

// erregistro bat edo zenbaki bat (`ADD R1 R1 (1)`: parentesi arteko zenbakia erregistro baten lekuan).
// 0.4.3 arteko eragiketek erregistroak bakarrik dituzte; zenbaki bat duten forma berriak (BinV, BranchV, ...)
// formatuaren 2. bertsioan daude, eta mihiztatzaileak forma zaharra idazten du eragigai guztiak erregistroak
// direnean, programa zaharrek byte berberak izan ditzaten
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Val {
    Reg(u32),
    Num(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    // 0.4.3: a^b (formatu bitarraren 2. bertsioa)
    Pow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnOp {
    Sqrt,
    Exp,
    Log,
    Not,
    // 0.4.3: trigonometria, radianetan (formatu bitarraren 2. bertsioa)
    Sin,
    Cos,
    Tan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cmp {
    Eq,
    Ne,
    Gt,
    Lt,
}

// eragiketa exekutagarri bat. qubitak, moduak eta erregistroak indizeak dira; jauzi-helburuak
// `Program::ops`-eko indizeak dira.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Op {
    // ---- qubit-ateak ----
    Gate {
        q: u32,
        gate: Gate,
    },
    // `gate` q-ri aplikatu kontrol-qubit guztiak 1 direnean (CNOT, CY, CH, CRX.., CCX, MCX)
    Controlled {
        controls: Vec<u32>,
        q: u32,
        gate: Gate,
    },
    // e^{iθ}-z biderkatu zerrendako qubit guztiak 1 diren oinarri-egoerak (CZ, CPHASE, MCZ)
    PhaseAll {
        qubits: Vec<u32>,
        theta: f64,
    },
    Swap {
        a: u32,
        b: u32,
        controls: Vec<u32>,
    },

    // ---- entrelazamendu-laburdurak (bakoitzak exekutatzen duen ate-segida dokumentatuta) ----
    Bell {
        a: u32,
        b: u32,
    },
    Ghz {
        qubits: Vec<u32>,
    },
    Cluster {
        qubits: Vec<u32>,
    },
    EntangleSwap {
        a: u32,
        b: u32,
        c: u32,
        d: u32,
    },

    // ---- qubit-neurketa ----
    Measure {
        q: u32,
        basis: Basis,
        dest: Dest,
    },
    MeasureAll,
    Reset {
        q: u32,
    },
    ResetAll,
    ToBasis {
        q: u32,
        basis: Basis,
    },

    // ---- modu optikoak ----
    // qubit baten kitzikapena modu huts batera eraman: |1>|0> <-> |0>|1>
    Emit {
        q: u32,
        o: u32,
    },
    PhotonCount {
        o: u32,
        dest: Dest,
    },
    PhotonDetect {
        o: u32,
        dest: Dest,
    },
    Parity {
        o: u32,
        dest: Dest,
    },
    Coincidence {
        modes: Vec<u32>,
        dest: Dest,
    },
    Displace {
        o: u32,
        re: f64,
        im: f64,
    },
    DisplaceReg {
        o: u32,
        r: u32,
    },
    Squeeze {
        o: u32,
        r: f64,
        phi: f64,
    },
    SqueezeReg {
        o: u32,
        reg: u32,
    },
    ModePhase {
        o: u32,
        theta: f64,
    },
    Kerr {
        o: u32,
        kappa: f64,
    },
    CrossKerr {
        a: u32,
        b: u32,
        kappa: f64,
    },
    BeamSplitter {
        a: u32,
        b: u32,
        theta: f64,
        phi: f64,
    },
    ModeSwap {
        a: u32,
        b: u32,
    },
    AddPhoton {
        o: u32,
    },
    SubPhoton {
        o: u32,
    },
    Loss {
        o: u32,
        gamma: f64,
    },
    Fock {
        o: u32,
        n: u32,
    },
    Dispersive {
        q: u32,
        o: u32,
        theta: f64,
    },

    // ---- simulagailuan bakarrik egin daitekeen ikuskapena (benetako gailu batek ezin du) ----
    DumpState,
    DumpRegs,
    Expect {
        q: u32,
        basis: Basis,
        dest: Option<u32>,
    },
    BellCheck {
        a: u32,
        b: u32,
        dest: Option<u32>,
    },
    MeanPhoton {
        o: u32,
        dest: Option<u32>,
    },

    // ---- klasikoa ----
    Set {
        r: u32,
        value: f64,
    },
    Copy {
        dst: u32,
        src: u32,
    },
    Bin {
        op: BinOp,
        dst: u32,
        a: u32,
        b: u32,
    },
    Un {
        op: UnOp,
        dst: u32,
        src: u32,
    },
    Rand {
        r: u32,
    },
    Seed {
        seed: u64,
    },
    Input {
        r: u32,
    },
    Time {
        r: u32,
    },
    Push {
        r: u32,
    },
    Pop {
        r: u32,
    },
    // bit klasikoak zenbaki oso gisa irakurri (bits[0] bit baxuena; idatzi gabeak 0)
    PackBits {
        dst: u32,
        bits: Vec<u32>,
    },

    // ---- kontrol-fluxua ----
    Jump {
        to: usize,
    },
    JumpIf {
        src: Source,
        value: f64,
        to: usize,
    },
    Branch {
        cmp: Cmp,
        a: u32,
        b: u32,
        to: usize,
    },
    Call {
        to: usize,
    },
    Return,
    LoopStart {
        count: Count,
        end: usize,
    },
    LoopEnd {
        start: usize,
    },
    Halt,
    Barrier,

    // ---- irteera ----
    Print {
        text: String,
        newline: bool,
    },
    Printf {
        format: String,
        regs: Vec<u32>,
    },
    CharOut {
        r: u32,
    },
    Log {
        q: u32,
        text: String,
    },

    // ---- 0.4.3: memoria eta byte-sarrera/irteera (formatu bitarraren 2. bertsioa) ----
    // memoria: 0, 1, 2, ... gelaxkak, bakoitzean zenbaki bat; idatzi gabeak 0 dira
    Load {
        dst: u32,
        addr: u32,
    },
    Store {
        src: u32,
        addr: u32,
    },
    // testuaren byteak (utf-8) addr-en helbidetik aurrera, gelaxka bakoitzean bat, eta gero 0; addr 0aren ondora mugitzen da
    StoreText {
        addr: u32,
        text: String,
    },
    // sarrera estandarraren hurrengo bytea (0..255), edo -1 amaieran
    GetByte {
        r: u32,
    },
    // byte bat irteerara, bere horretan
    PutByte {
        r: u32,
    },
    // hemendik aurrerako irteera: irteera estandarra (false) edo errore-irteera (true)
    Stream {
        stderr: bool,
    },
    // HALT bezala, irteera-egoera batekin
    Exit {
        status: u8,
    },
    // 1 QOA euskaraz ari bada, 0 ingelesez
    GetLang {
        r: u32,
    },
    // programari `--` ondoren emandako argumentuak
    ArgCount {
        r: u32,
    },
    ArgByte {
        dst: u32,
        arg: u32,
        index: u32,
    },

    // ---- 0.4.4: eragigai bat zenbaki bat denean (Val), goiko eragiketa berak ----
    BinV {
        op: BinOp,
        dst: u32,
        a: Val,
        b: Val,
    },
    UnV {
        op: UnOp,
        dst: u32,
        src: Val,
    },
    BranchV {
        cmp: Cmp,
        a: Val,
        b: Val,
        to: usize,
    },
    LoadV {
        dst: u32,
        addr: Val,
    },
    StoreV {
        src: Val,
        addr: Val,
    },
    PutByteV {
        v: Val,
    },
    CharOutV {
        v: Val,
    },
    PushV {
        v: Val,
    },
    ArgByteV {
        dst: u32,
        arg: Val,
        index: Val,
    },
}

impl Op {
    // ate-moduko eragiketa batek ukitzen dituen qubitak (zaratarako). hutsa ateak ez badira.
    pub fn gate_qubits(&self) -> Vec<u32> {
        match self {
            Op::Gate { q, .. } => vec![*q],
            Op::Controlled { controls, q, .. } => {
                let mut v = controls.clone();
                v.push(*q);
                v
            }
            Op::PhaseAll { qubits, .. } => qubits.clone(),
            Op::Swap { a, b, controls } => {
                let mut v = controls.clone();
                v.push(*a);
                v.push(*b);
                v
            }
            Op::Bell { a, b } => vec![*a, *b],
            Op::Ghz { qubits } | Op::Cluster { qubits } => qubits.clone(),
            Op::ToBasis { q, .. } => vec![*q],
            Op::Emit { q, .. } | Op::Dispersive { q, .. } => vec![*q],
            _ => vec![],
        }
    }

    // eragiketa optiko batek ukitzen dituen moduak (--loss zarata-eredurako). hutsa optikoa ez bada.
    pub fn optical_modes(&self) -> Vec<u32> {
        match self {
            Op::Emit { o, .. }
            | Op::Displace { o, .. }
            | Op::DisplaceReg { o, .. }
            | Op::Squeeze { o, .. }
            | Op::SqueezeReg { o, .. }
            | Op::ModePhase { o, .. }
            | Op::Kerr { o, .. }
            | Op::AddPhoton { o }
            | Op::SubPhoton { o }
            | Op::Fock { o, .. }
            | Op::Dispersive { o, .. } => vec![*o],
            Op::CrossKerr { a, b, .. } | Op::BeamSplitter { a, b, .. } | Op::ModeSwap { a, b } => {
                vec![*a, *b]
            }
            _ => vec![],
        }
    }

    // qubit edo modu bat bere oinarri konputazionalean / fotoi-kopuruan irakurtzen duten eragiketak,
    // aurretik biraketarik gabe. hauek bakarrik ordezka daitezke laginketaz.
    pub fn is_diagonal_readout(&self) -> bool {
        matches!(
            self,
            Op::Measure {
                basis: Basis::Z,
                ..
            } | Op::MeasureAll
                | Op::PhotonCount { .. }
                | Op::PhotonDetect { .. }
                | Op::Parity { .. }
                | Op::Coincidence { .. }
        )
    }

    // egoera kuantikoa aldatzen duten eragiketak
    pub fn touches_quantum_state(&self) -> bool {
        !matches!(
            self,
            Op::DumpState
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
                | Op::ArgByteV { .. }
        )
    }

    // eragiketa hau eduki dezakeen formatu-bertsio txikiena: 1 (QOA 0.4.2-k irakurtzen du), 2 (QOA 0.4.3-ko
    // aginduak: memoria, byteak, SIN/COS/TAN/POW...) edo 3 (QOA 0.4.4-ko (zenbaki) eragigaiak)
    pub fn format_needed(&self) -> u16 {
        match self {
            Op::BinV { .. }
            | Op::UnV { .. }
            | Op::BranchV { .. }
            | Op::LoadV { .. }
            | Op::StoreV { .. }
            | Op::PutByteV { .. }
            | Op::CharOutV { .. }
            | Op::PushV { .. }
            | Op::ArgByteV { .. } => 3,
            Op::Bin { op, .. } if *op == BinOp::Pow => 2,
            Op::Un {
                op: UnOp::Sin | UnOp::Cos | UnOp::Tan,
                ..
            } => 2,
            Op::Load { .. }
            | Op::Store { .. }
            | Op::StoreText { .. }
            | Op::GetByte { .. }
            | Op::PutByte { .. }
            | Op::Stream { .. }
            | Op::Exit { .. }
            | Op::GetLang { .. }
            | Op::ArgCount { .. }
            | Op::ArgByte { .. } => 2,
            _ => 1,
        }
    }

    // jauzi-helburua duen eragiketa baten helburua (Jump, JumpIf, Branch, BranchV, Call)
    pub fn jump_target(&self) -> Option<usize> {
        match self {
            Op::Jump { to }
            | Op::JumpIf { to, .. }
            | Op::Branch { to, .. }
            | Op::BranchV { to, .. }
            | Op::Call { to } => Some(*to),
            _ => None,
        }
    }

    pub fn is_control_flow(&self) -> bool {
        matches!(
            self,
            Op::Jump { .. }
                | Op::JumpIf { .. }
                | Op::Branch { .. }
                | Op::BranchV { .. }
                | Op::Call { .. }
                | Op::Return
                | Op::LoopStart { .. }
                | Op::LoopEnd { .. }
        )
    }
}

// programa mihiztatu bat
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Program {
    // programa nondik datorren (fitxategi-izena), mezuetarako
    pub name: String,
    pub n_qubits: u32,
    pub n_modes: u32,
    // modu bakoitzeko fotoi-muga: modu bakoitzak |0> .. |cutoff-1> gordetzen ditu
    pub cutoff: u32,
    // egia muga QMODE-tik badator, ez lehenetsitik
    pub cutoff_declared: bool,
    pub n_bits: u32,
    pub ops: Vec<Op>,
    // eragiketa bakoitzaren iturburu-lerroa (1etik hasita; 0 sortua bada)
    pub lines: Vec<u32>,
}

pub const DEFAULT_CUTOFF: u32 = 8;

impl Program {
    // egoera-bektoreak behar dituen anplitude konplexuak: 2^qubitak * muga^moduak
    pub fn state_dim(&self) -> Option<u128> {
        let mut d: u128 = 1u128.checked_shl(self.n_qubits)?;
        for _ in 0..self.n_modes {
            d = d.checked_mul(self.cutoff as u128)?;
        }
        Some(d)
    }

    pub fn line_of(&self, pc: usize) -> u32 {
        self.lines.get(pc).copied().unwrap_or(0)
    }

    // neurketa guztiak egoera aldatzen duen azken eragiketaren ondoren badaude, eta aurretik
    // ezer ausazkorik edo adarkatzerik ez badago, simulazio bakar batetik lagin daitezke shot guztiak.
    // kasu horretan lehen irakurketaren indizea itzultzen du.
    pub fn sampling_split(&self) -> Option<usize> {
        let first = self.ops.iter().position(|op| op.is_diagonal_readout())?;
        // kopuru finkoko begiztak deterministak dira: onartzen dira, osorik lehen irakurketaren aurretik badaude.
        // beste kontrol-fluxu guztiak (jauziak, deiak, erregistroz gidatutako begiztak) ez.
        for op in &self.ops {
            match op {
                Op::LoopStart {
                    count: Count::Lit(_),
                    end,
                } if *end <= first => {}
                Op::LoopEnd { start } if *start < first => {}
                other if other.is_control_flow() => return None,
                _ => {}
            }
        }
        for op in &self.ops[..first] {
            // irakurketen aurretik ausazkoa edo erregistroek gidatua den edozerk egoera aldatzen du shot-etik shot-era
            if matches!(
                op,
                Op::Measure { .. }
                    | Op::Reset { .. }
                    | Op::ResetAll
                    | Op::Loss { .. }
                    | Op::EntangleSwap { .. }
                    | Op::Fock { .. }
                    | Op::Rand { .. }
                    | Op::Seed { .. }
                    | Op::Input { .. }
                    | Op::GetByte { .. }
                    | Op::Time { .. }
                    | Op::DisplaceReg { .. }
                    | Op::SqueezeReg { .. }
            ) {
                return None;
            }
        }
        for op in &self.ops[first..] {
            if op.touches_quantum_state() && !op.is_diagonal_readout() {
                return None;
            }
            if matches!(
                op,
                Op::Rand { .. } | Op::Seed { .. } | Op::Input { .. } | Op::GetByte { .. }
            ) {
                return None;
            }
        }
        Some(first)
    }

    pub fn has_readout(&self) -> bool {
        self.ops.iter().any(|op| {
            op.is_diagonal_readout() || matches!(op, Op::Measure { .. } | Op::EntangleSwap { .. })
        })
    }

    // ---- formatu bitarra ----

    pub const MAGIC: &'static [u8; 4] = b"QOAX";
    // irakur daitekeen bertsio berriena. bertsio bakoitza hurrengoaren azpimultzoa da, eta programa bat
    // bere eragiketek behar duten bertsio txikienean idazten da, QOA zaharragoek ere irakur dezaten: 1
    // (0.4.2), 2 (0.4.3-ko aginduekin) edo 3 (0.4.4-ko (zenbaki) eragigaiekin). QOA zahar batek bertsio
    // berriago bat ikustean argi esaten du ezin duela irakurri, "hondatua" esan beharrean
    pub const FORMAT_VERSION: u16 = 3;

    pub fn format_version(&self) -> u16 {
        self.ops.iter().map(Op::format_needed).max().unwrap_or(1)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(Self::MAGIC);
        out.extend_from_slice(&self.format_version().to_le_bytes());
        out.extend(bincode::serialize(self).expect("programa beti serializatzen da"));
        out
    }

    pub fn from_bytes(data: &[u8]) -> Result<Program, String> {
        if data.len() < 6 {
            return Err(t!(
                "the file is too short to be a QOA binary",
                "fitxategia laburregia da QOA bitar bat izateko"
            ));
        }
        let magic = &data[..4];
        if magic == b"QEXE" || magic == b"OEXE" || magic == b"QX\0\0" {
            return Err(t!(
                "this binary was built by QOA 0.3.x, whose format is not supported; recompile the .qoa source with `qoa compile`",
                "bitar hau QOA 0.3.x-ek sortu zuen, eta haren formatua ez da onartzen; konpilatu berriz .qoa iturburua `qoa compile`-rekin"
            ));
        }
        if magic != Self::MAGIC {
            return Err(t!(
                "not a QOA binary (the QOAX header is missing)",
                "ez da QOA bitar bat (QOAX goiburua falta da)"
            ));
        }
        let version = u16::from_le_bytes([data[4], data[5]]);
        if version == 0 || version > Self::FORMAT_VERSION {
            return Err(t!(
                "binary format version {} is not supported by this QOA (it reads versions 1 to {})",
                "{}. bertsioko formatu bitarra ez du QOA honek onartzen (1. bertsiotik {}.era irakurtzen ditu)",
                version,
                Self::FORMAT_VERSION
            ));
        }
        let p: Program = bincode::deserialize(&data[6..])
            .map_err(|e| t!("corrupt QOA binary: {}", "QOA bitar hondatua: {}", e))?;
        // edukia ere bai: mihiztatzaileak sor dezakeen programa bat, bestela makinak ez du ikusten
        crate::disasm::check_loaded(&p)
            .map_err(|e| t!("corrupt QOA binary: {}", "QOA bitar hondatua: {}", e))?;
        let needed = p.format_version();
        if version < needed {
            let release = if needed == 2 { "0.4.3" } else { "0.4.4" };
            return Err(t!(
                "corrupt QOA binary: version {} cannot hold QOA {} instructions",
                "QOA bitar hondatua: {}. bertsioak ezin ditu QOA {}-ko aginduak izan",
                version,
                release
            ));
        }
        Ok(p)
    }
}
