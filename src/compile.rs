// fitxategi honek programa bat urratsez urrats konpilatzen du, urrats bakoitzean geldi daitekeela aztertzeko edo
// arazketarako: QOA kanonikoa, .qx bytecode-a, IR-a (oinarrizko blokeak, eragiketa bakoitza natiboa ala
// exekuzio-garaikoa), arkitektura baten mihiztadura (GNU as), objektu-fitxategia (makina-kodea) eta liburutegi
// partekatua. arkitekturak: x86_64, aarch64, riscv64, powerpc64le, s390x eta loongarch64, ELF sistemetan (linux).
//
// kode natiboaren itxura: eragiketa bakoitzak bere etiketa du (.Lop_N). eragiketa klasiko sinpleak (SET, COPY,
// ADD/SUB/MUL erregistroekin edo zenbakiekin, JUMP) makina-aginduak dira zuzenean, erregistroen memorian (32
// f64, makinarenak berak); gainerakoak QOA-ren makinari deitzen diote (step): hark eragiketa exekutatzen du,
// interpretatzaileak bezala, eta hurrengo eragiketa itzultzen du. hurrengoa ondokoa bada, kodea jarraian
// doa; bestela taula batek eragiketa bakoitzaren kodea aurkitzen du (jauziak, deiak, itzulerak, begiztak).
// eragiketa natibo bakoitzaren aurretik `tick`-i deitzen zaio (urrats-kontaketa eta muga, laginketaren
// unea), beraz exekuzioa interpretatzailearena da urratsez urrats: emaitza berberak, hazi berarekin.
//
// kodeak ez du kanpoko ikurrik behar: makinaren funtzioak argumentu gisa jasotzen ditu (erakusleak), eta
// liburutegiak programaren .qx-a darama, `qoa run programa.so`-k exekuta dezan.

use crate::program::{BinOp, Op, Program, Val};
use crate::t;
use std::ffi::c_void;
use std::fmt::Write as _;

// kode natiboaren sarrera: (makina, erregistroak, step, tick) -> 0 amaituta, 1 errorea (makinak gordeta), 2
// kodea eta programa ez datoz bat (QOA-ren akatsa)
pub type NativeEntry = unsafe extern "C" fn(*mut c_void, *mut f64, StepFn, TickFn) -> i64;
// step(makina, pc) -> hurrengo pc, -1 gelditu, -2 errorea
pub type StepFn = extern "C" fn(*mut c_void, u64) -> i64;
// tick(makina, pc) -> 0 jarraitu, -2 errorea
pub type TickFn = extern "C" fn(*mut c_void, u64) -> i64;

pub const ENTRY: &str = "qoa_native_entry";
pub const QX: &str = "qoa_native_qx";
pub const QX_LEN: &str = "qoa_native_qx_len";
pub const ABI: &str = "qoa_native_abi";
// kode natiboaren eta makinaren arteko hitzarmenaren bertsioa
pub const ABI_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Aarch64,
    Riscv64,
    Powerpc64le,
    S390x,
    Loongarch64,
}

impl Arch {
    pub const ALL: [Arch; 6] = [
        Arch::X86_64,
        Arch::Aarch64,
        Arch::Riscv64,
        Arch::Powerpc64le,
        Arch::S390x,
        Arch::Loongarch64,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Arch::X86_64 => "x86_64",
            Arch::Aarch64 => "aarch64",
            Arch::Riscv64 => "riscv64",
            Arch::Powerpc64le => "powerpc64le",
            Arch::S390x => "s390x",
            Arch::Loongarch64 => "loongarch64",
        }
    }

    // clang-en eta gcc gurutzatuen helburua
    pub fn triple(self) -> &'static str {
        match self {
            Arch::X86_64 => "x86_64-linux-gnu",
            Arch::Aarch64 => "aarch64-linux-gnu",
            Arch::Riscv64 => "riscv64-linux-gnu",
            Arch::Powerpc64le => "powerpc64le-linux-gnu",
            Arch::S390x => "s390x-linux-gnu",
            Arch::Loongarch64 => "loongarch64-linux-gnu",
        }
    }

    pub fn parse(s: &str) -> Option<Arch> {
        match s.to_ascii_lowercase().as_str() {
            "x86_64" | "x86-64" | "amd64" | "x64" => Some(Arch::X86_64),
            "aarch64" | "arm64" => Some(Arch::Aarch64),
            "riscv64" | "riscv64gc" | "rv64" => Some(Arch::Riscv64),
            "powerpc64le" | "ppc64le" | "power" => Some(Arch::Powerpc64le),
            "s390x" | "ibmz" | "z" => Some(Arch::S390x),
            "loongarch64" | "loong64" => Some(Arch::Loongarch64),
            _ => None,
        }
    }

    // arkitektura hau ELF sistema batean (linux): kode natiboa sor daitekeen lekua. macos eta windows-en
    // objektu-formatuak (mach-o, coff) ez dira egiten oraingoz
    pub fn host() -> Result<Arch, String> {
        let arch = match std::env::consts::ARCH {
            "x86_64" => Arch::X86_64,
            "aarch64" => Arch::Aarch64,
            "riscv64" => Arch::Riscv64,
            "powerpc64" if cfg!(target_endian = "little") => Arch::Powerpc64le,
            "s390x" => Arch::S390x,
            "loongarch64" => Arch::Loongarch64,
            other => {
                return Err(t!(
                    "native code for {} is not written; --arch takes {}",
                    "{}-rentzako kode natiboa ez dago idatzita; --arch-ek hauek hartzen ditu: {}",
                    other,
                    names()
                ));
            }
        };
        if !cfg!(target_os = "linux") {
            return Err(t!(
                "native code is made for ELF systems such as Linux so far, not {}; --emit asm --arch ARCH still writes the assembly for Linux",
                "kode natiboa ELF sistemetarako egiten da oraingoz, linux bezalakoetarako, ez {}-rako; --emit asm --arch ARKITEKTURA-k linux-erako mihiztadura idazten du hala ere",
                std::env::consts::OS
            ));
        }
        Ok(arch)
    }
}

pub fn names() -> String {
    Arch::ALL.map(|a| a.name()).join(", ")
}

// ---------------------------------------------------------------- IR-a

// eragiketa batek kode natiboan duen forma
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lower {
    // makina-aginduak: R[dst] = balioa
    Set {
        r: u32,
        bits: u64,
    },
    Copy {
        dst: u32,
        src: u32,
    },
    // R[dst] = a (+ - *) b; eragigaiak erregistroak edo zenbakiak
    Arith {
        op: Arith,
        dst: u32,
        a: Operand,
        b: Operand,
    },
    Jump {
        to: usize,
    },
    // makinari deitu (step) eta hurrengo eragiketara joan
    Runtime,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Arith {
    Add,
    Sub,
    Mul,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Operand {
    Reg(u32),
    Num(u64),
}

fn operand(v: Val) -> Operand {
    match v {
        Val::Reg(r) => Operand::Reg(r),
        Val::Num(x) => Operand::Num(x.to_bits()),
    }
}

// batuketa, kenketa eta biderketa IEEE 754 eragiketa hutsak dira makinan (ez dute errorerik ematen), beraz
// arkitektura guztietan bit berberak ematen dituzte makina-agindu gisa ere
fn lower(op: &Op) -> Lower {
    let arith = |o: BinOp| match o {
        BinOp::Add => Some(Arith::Add),
        BinOp::Sub => Some(Arith::Sub),
        BinOp::Mul => Some(Arith::Mul),
        _ => None,
    };
    match op {
        Op::Set { r, value } => Lower::Set {
            r: *r,
            bits: value.to_bits(),
        },
        Op::Copy { dst, src } => Lower::Copy {
            dst: *dst,
            src: *src,
        },
        Op::Bin { op, dst, a, b } => match arith(*op) {
            Some(op) => Lower::Arith {
                op,
                dst: *dst,
                a: Operand::Reg(*a),
                b: Operand::Reg(*b),
            },
            None => Lower::Runtime,
        },
        Op::BinV { op, dst, a, b } => match arith(*op) {
            Some(op) => Lower::Arith {
                op,
                dst: *dst,
                a: operand(*a),
                b: operand(*b),
            },
            None => Lower::Runtime,
        },
        Op::Jump { to } => Lower::Jump { to: *to },
        _ => Lower::Runtime,
    }
}

// eragiketa baten ondorengo posibleak (pc), estatikoki ezagunak; None: dinamikoa (RETURN)
fn successors(p: &Program, pc: usize) -> Option<Vec<usize>> {
    let next = pc + 1;
    Some(match &p.ops[pc] {
        Op::Jump { to } | Op::Call { to } => vec![*to],
        Op::JumpIf { to, .. } | Op::Branch { to, .. } | Op::BranchV { to, .. } => {
            vec![next, *to]
        }
        Op::LoopStart { end, .. } => vec![next, *end],
        Op::LoopEnd { start } => vec![next, start + 1],
        Op::Return => return None,
        Op::Halt => vec![],
        _ => vec![next],
    })
}

// oinarrizko blokeen hasierak: 0, jauzi-helburu bakoitza, eta kontrol-fluxuko eragiketa baten ondorengoa
fn leaders(p: &Program) -> Vec<usize> {
    let n = p.ops.len();
    let mut is = vec![false; n + 1];
    is[0] = true;
    for (pc, op) in p.ops.iter().enumerate() {
        if let Some(t) = op.jump_target() {
            is[t.min(n)] = true;
        }
        if let Op::LoopStart { end, .. } = op {
            is[(*end).min(n)] = true;
        }
        if let Op::LoopEnd { start } = op {
            is[(start + 1).min(n)] = true;
        }
        if op.is_control_flow() || matches!(op, Op::Halt) {
            is[pc + 1] = true;
        }
        if let Op::Call { .. } = op {
            is[pc + 1] = true;
        }
    }
    (0..n).filter(|&i| is[i]).collect()
}

fn fmt_operand(o: Operand) -> String {
    match o {
        Operand::Reg(r) => format!("R{}", r),
        Operand::Num(b) => format!("{:?}", f64::from_bits(b)),
    }
}

// IR-a testu gisa: bloke bakoitza, haren eragiketak (natiboak edo makinari deia) eta ondorengoak
pub fn ir(p: &Program) -> Result<String, String> {
    let texts = crate::disasm::op_texts(p)?;
    let n = p.ops.len();
    let lowered: Vec<Lower> = p.ops.iter().map(lower).collect();
    let native = lowered.iter().filter(|l| **l != Lower::Runtime).count();
    let starts = leaders(p);
    let mut out = String::new();
    let _ = writeln!(out, "; {}", t!("QOA IR for {}", "{}-ren QOA IR-a", p.name));
    let _ = writeln!(
        out,
        "; {}",
        t!(
            "{} operations in {} blocks: {} native (machine instructions), {} through QOA's machine (step)",
            "{} eragiketa {} blokeetan: {} natibo (makina-aginduak), {} QOA-ren makinaren bidez (step)",
            n,
            starts.len(),
            native,
            n - native
        )
    );
    let _ = writeln!(
        out,
        "; {}",
        t!(
            "native operations first call tick (step count and limit), so every run matches the interpreter",
            "eragiketa natiboek lehenik tick-i deitzen diote (urrats-kontaketa eta muga), beraz exekuzio orok interpretatzailearena du"
        )
    );
    let _ = writeln!(out, "func {}(machine, regs, step, tick)", ENTRY);
    for (b, &start) in starts.iter().enumerate() {
        let end = starts.get(b + 1).copied().unwrap_or(n);
        let last = end - 1;
        let succ = match successors(p, last) {
            Some(s) if s.is_empty() => t!("stop", "gelditu"),
            Some(s) => s
                .iter()
                .map(|&t| match starts.iter().position(|&x| x == t) {
                    Some(i) => format!("b{}", i),
                    None if t >= n => t!("end", "amaiera"),
                    None => format!("op {}", t),
                })
                .collect::<Vec<_>>()
                .join(", "),
            None => t!("return (any call site)", "itzulera (edozein dei-gune)"),
        };
        let _ = writeln!(
            out,
            "\nb{}:  ; {} {}..{} -> {}",
            b,
            t!("ops", "eragiketak"),
            start,
            last,
            succ
        );
        for pc in start..end {
            let body = match lowered[pc] {
                Lower::Set { r, bits } => format!("native   R{} = {:?}", r, f64::from_bits(bits)),
                Lower::Copy { dst, src } => format!("native   R{} = R{}", dst, src),
                Lower::Arith { op, dst, a, b } => format!(
                    "native   R{} = {} {} {}",
                    dst,
                    fmt_operand(a),
                    match op {
                        Arith::Add => "+",
                        Arith::Sub => "-",
                        Arith::Mul => "*",
                    },
                    fmt_operand(b)
                ),
                Lower::Jump { to } => format!("native   goto {}", to),
                Lower::Runtime => format!("step     {}", pc),
            };
            let line = p.lines.get(pc).copied().unwrap_or(0);
            let _ = writeln!(
                out,
                "  {:04}  {:<34} ; {}{}",
                pc,
                body,
                texts[pc],
                if line > 0 {
                    format!("  ({})", t!("line {}", "{}. lerroa", line))
                } else {
                    String::new()
                }
            );
        }
    }
    let _ = writeln!(out, "\nend:  ; {}", t!("return 0", "itzuli 0"));
    Ok(out)
}

// ---------------------------------------------------------------- mihiztadura

// arkitektura bakoitzaren agindu-sortzailea: kode natiboaren zatiak
trait Target {
    fn header(&self, o: &mut String);
    fn prologue(&self, o: &mut String);
    // R[r] = bits
    fn set(&self, o: &mut String, r: u32, bits: u64);
    fn copy(&self, o: &mut String, dst: u32, src: u32);
    fn arith(&self, o: &mut String, op: Arith, dst: u32, a: Operand, b: Operand);
    // tick(pc): errorea bada, .Lerr-era
    fn tick(&self, o: &mut String, pc: usize);
    // step(pc): emaitza `next` bada, hurrengo etiketara (ondoan); bestela .Lmiss-era
    fn step(&self, o: &mut String, pc: usize, next: usize);
    fn goto(&self, o: &mut String, label: &str);
    // amaiera (.Lop_len): itzuli 0; .Lerr: 1; .Lbad: 2; .Lmiss: banatzailea taularekin
    fn tail(&self, o: &mut String, len: usize);
    fn note_section(&self) -> &'static str {
        "\t.section .note.GNU-stack,\"\",@progbits\n"
    }
}

fn label(pc: usize) -> String {
    format!(".Lop_{}", pc)
}

// 64 biteko balio baten 16 biteko zatiak (behetik gora)
fn halves(v: u64) -> [u64; 4] {
    [
        v & 0xffff,
        (v >> 16) & 0xffff,
        (v >> 32) & 0xffff,
        (v >> 48) & 0xffff,
    ]
}

fn sext(v: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((v << shift) as i64) >> shift
}

struct X86;
impl Target for X86 {
    fn header(&self, _o: &mut String) {}
    fn prologue(&self, o: &mut String) {
        // rbx = makina, r12 = erregistroak, r13 = step, r14 = tick; 5 push-ek pila 16ra lerrokatzen dute
        o.push_str("\tpushq %rbx\n\tpushq %r12\n\tpushq %r13\n\tpushq %r14\n\tpushq %r15\n");
        o.push_str("\tmovq %rdi, %rbx\n\tmovq %rsi, %r12\n\tmovq %rdx, %r13\n\tmovq %rcx, %r14\n");
    }
    fn set(&self, o: &mut String, r: u32, bits: u64) {
        let _ = writeln!(
            o,
            "\tmovabsq ${:#x}, %rax\n\tmovq %rax, {}(%r12)",
            bits,
            r * 8
        );
    }
    fn copy(&self, o: &mut String, dst: u32, src: u32) {
        let _ = writeln!(
            o,
            "\tmovq {}(%r12), %rax\n\tmovq %rax, {}(%r12)",
            src * 8,
            dst * 8
        );
    }
    fn arith(&self, o: &mut String, op: Arith, dst: u32, a: Operand, b: Operand) {
        for (x, xmm) in [(a, "%xmm0"), (b, "%xmm1")] {
            match x {
                Operand::Reg(r) => {
                    let _ = writeln!(o, "\tmovsd {}(%r12), {}", r * 8, xmm);
                }
                Operand::Num(bits) => {
                    let _ = writeln!(o, "\tmovabsq ${:#x}, %rax\n\tmovq %rax, {}", bits, xmm);
                }
            }
        }
        let ins = match op {
            Arith::Add => "addsd",
            Arith::Sub => "subsd",
            Arith::Mul => "mulsd",
        };
        let _ = writeln!(
            o,
            "\t{} %xmm1, %xmm0\n\tmovsd %xmm0, {}(%r12)",
            ins,
            dst * 8
        );
    }
    fn tick(&self, o: &mut String, pc: usize) {
        let _ = writeln!(
            o,
            "\tmovq %rbx, %rdi\n\tmovl ${}, %esi\n\tcallq *%r14\n\ttestq %rax, %rax\n\tjne .Lerr",
            pc
        );
    }
    fn step(&self, o: &mut String, pc: usize, next: usize) {
        let _ = writeln!(
            o,
            "\tmovq %rbx, %rdi\n\tmovl ${}, %esi\n\tcallq *%r13\n\tcmpq ${}, %rax\n\tjne .Lmiss",
            pc, next
        );
    }
    fn goto(&self, o: &mut String, l: &str) {
        let _ = writeln!(o, "\tjmp {}", l);
    }
    fn tail(&self, o: &mut String, len: usize) {
        let _ = write!(
            o,
            "{end}:\n\txorl %eax, %eax\n.Lret:\n\tpopq %r15\n\tpopq %r14\n\tpopq %r13\n\tpopq %r12\n\tpopq %rbx\n\tretq\n\
.Lerr:\n\tmovl $1, %eax\n\tjmp .Lret\n\
.Lbad:\n\tmovl $2, %eax\n\tjmp .Lret\n\
.Lmiss:\n\tcmpq $-1, %rax\n\tje {end}\n\ttestq %rax, %rax\n\tjs .Lerr\n\tcmpq ${len}, %rax\n\tje {end}\n\tja .Lbad\n\
\tleaq .Ltab(%rip), %rdx\n\tmovslq (%rdx,%rax,4), %rcx\n\taddq %rdx, %rcx\n\tjmpq *%rcx\n",
            end = label(len),
            len = len
        );
    }
}

struct A64;
impl A64 {
    fn imm(o: &mut String, reg: &str, v: u64) {
        let h = halves(v);
        let _ = writeln!(o, "\tmovz {}, #{:#x}", reg, h[0]);
        for (i, part) in h.iter().enumerate().skip(1) {
            if *part != 0 {
                let _ = writeln!(o, "\tmovk {}, #{:#x}, lsl #{}", reg, part, i * 16);
            }
        }
    }
}
impl Target for A64 {
    fn header(&self, _o: &mut String) {}
    fn prologue(&self, o: &mut String) {
        // x19 = makina, x20 = erregistroak, x21 = step, x22 = tick
        o.push_str("\tstp x29, x30, [sp, #-48]!\n\tmov x29, sp\n\tstp x19, x20, [sp, #16]\n\tstp x21, x22, [sp, #32]\n");
        o.push_str("\tmov x19, x0\n\tmov x20, x1\n\tmov x21, x2\n\tmov x22, x3\n");
    }
    fn set(&self, o: &mut String, r: u32, bits: u64) {
        A64::imm(o, "x9", bits);
        let _ = writeln!(o, "\tstr x9, [x20, #{}]", r * 8);
    }
    fn copy(&self, o: &mut String, dst: u32, src: u32) {
        let _ = writeln!(
            o,
            "\tldr x9, [x20, #{}]\n\tstr x9, [x20, #{}]",
            src * 8,
            dst * 8
        );
    }
    fn arith(&self, o: &mut String, op: Arith, dst: u32, a: Operand, b: Operand) {
        for (x, d) in [(a, "d0"), (b, "d1")] {
            match x {
                Operand::Reg(r) => {
                    let _ = writeln!(o, "\tldr {}, [x20, #{}]", d, r * 8);
                }
                Operand::Num(bits) => {
                    A64::imm(o, "x9", bits);
                    let _ = writeln!(o, "\tfmov {}, x9", d);
                }
            }
        }
        let ins = match op {
            Arith::Add => "fadd",
            Arith::Sub => "fsub",
            Arith::Mul => "fmul",
        };
        let _ = writeln!(o, "\t{} d0, d0, d1\n\tstr d0, [x20, #{}]", ins, dst * 8);
    }
    fn tick(&self, o: &mut String, pc: usize) {
        o.push_str("\tmov x0, x19\n");
        A64::imm(o, "x1", pc as u64);
        o.push_str("\tblr x22\n\tcbz x0, 1f\n\tb .Lerr\n1:\n");
    }
    fn step(&self, o: &mut String, pc: usize, next: usize) {
        o.push_str("\tmov x0, x19\n");
        A64::imm(o, "x1", pc as u64);
        o.push_str("\tblr x21\n");
        A64::imm(o, "x9", next as u64);
        o.push_str("\tcmp x0, x9\n\tb.eq 1f\n\tb .Lmiss\n1:\n");
    }
    fn goto(&self, o: &mut String, l: &str) {
        let _ = writeln!(o, "\tb {}", l);
    }
    fn tail(&self, o: &mut String, len: usize) {
        let mut lim = String::new();
        A64::imm(&mut lim, "x9", len as u64);
        let _ = write!(
            o,
            "{end}:\n\tmov x0, #0\n.Lret:\n\tldp x21, x22, [sp, #32]\n\tldp x19, x20, [sp, #16]\n\tldp x29, x30, [sp], #48\n\tret\n\
.Lerr:\n\tmov x0, #1\n\tb .Lret\n\
.Lbad:\n\tmov x0, #2\n\tb .Lret\n\
.Lmiss:\n\tcmn x0, #1\n\tb.eq {end}\n\ttbnz x0, #63, .Lerr\n{lim}\tcmp x0, x9\n\tb.eq {end}\n\tb.hi .Lbad\n\
\tadrp x10, .Ltab\n\tadd x10, x10, :lo12:.Ltab\n\tldrsw x11, [x10, x0, lsl #2]\n\tadd x10, x10, x11\n\tbr x10\n",
            end = label(len),
            lim = lim
        );
    }
    fn note_section(&self) -> &'static str {
        "\t.section .note.GNU-stack,\"\",%progbits\n"
    }
}

struct Rv64;
impl Target for Rv64 {
    fn header(&self, o: &mut String) {
        o.push_str("\t.option pic\n");
    }
    fn prologue(&self, o: &mut String) {
        // s1 = makina, s2 = erregistroak, s3 = step, s4 = tick
        o.push_str("\taddi sp, sp, -48\n\tsd ra, 40(sp)\n\tsd s1, 32(sp)\n\tsd s2, 24(sp)\n\tsd s3, 16(sp)\n\tsd s4, 8(sp)\n");
        o.push_str("\tmv s1, a0\n\tmv s2, a1\n\tmv s3, a2\n\tmv s4, a3\n");
    }
    fn set(&self, o: &mut String, r: u32, bits: u64) {
        let _ = writeln!(o, "\tli t0, {}\n\tsd t0, {}(s2)", bits as i64, r * 8);
    }
    fn copy(&self, o: &mut String, dst: u32, src: u32) {
        let _ = writeln!(o, "\tld t0, {}(s2)\n\tsd t0, {}(s2)", src * 8, dst * 8);
    }
    fn arith(&self, o: &mut String, op: Arith, dst: u32, a: Operand, b: Operand) {
        for (x, f) in [(a, "fa0"), (b, "fa1")] {
            match x {
                Operand::Reg(r) => {
                    let _ = writeln!(o, "\tfld {}, {}(s2)", f, r * 8);
                }
                Operand::Num(bits) => {
                    let _ = writeln!(o, "\tli t0, {}\n\tfmv.d.x {}, t0", bits as i64, f);
                }
            }
        }
        let ins = match op {
            Arith::Add => "fadd.d",
            Arith::Sub => "fsub.d",
            Arith::Mul => "fmul.d",
        };
        let _ = writeln!(o, "\t{} fa0, fa0, fa1\n\tfsd fa0, {}(s2)", ins, dst * 8);
    }
    fn tick(&self, o: &mut String, pc: usize) {
        let _ = writeln!(
            o,
            "\tmv a0, s1\n\tli a1, {}\n\tjalr s4\n\tbeqz a0, 1f\n\tj .Lerr\n1:",
            pc
        );
    }
    fn step(&self, o: &mut String, pc: usize, next: usize) {
        let _ = writeln!(
            o,
            "\tmv a0, s1\n\tli a1, {}\n\tjalr s3\n\tli t0, {}\n\tbeq a0, t0, 1f\n\tj .Lmiss\n1:",
            pc, next
        );
    }
    fn goto(&self, o: &mut String, l: &str) {
        let _ = writeln!(o, "\tj {}", l);
    }
    fn tail(&self, o: &mut String, len: usize) {
        let _ = write!(
            o,
            "{end}:\n\tli a0, 0\n.Lret:\n\tld ra, 40(sp)\n\tld s1, 32(sp)\n\tld s2, 24(sp)\n\tld s3, 16(sp)\n\tld s4, 8(sp)\n\taddi sp, sp, 48\n\tret\n\
.Lerr:\n\tli a0, 1\n\tj .Lret\n\
.Lbad:\n\tli a0, 2\n\tj .Lret\n\
.Lmiss:\n\tli t0, -1\n\tbeq a0, t0, {end}\n\tbltz a0, .Lerr\n\tli t0, {len}\n\tbeq a0, t0, {end}\n\tbgtu a0, t0, .Lbad\n\
\tlla t0, .Ltab\n\tslli t1, a0, 2\n\tadd t1, t0, t1\n\tlw t1, 0(t1)\n\tadd t0, t0, t1\n\tjr t0\n",
            end = label(len),
            len = len
        );
    }
}

struct Ppc64le;
impl Ppc64le {
    // 64 biteko balioa erregistro batean (5 agindu gehienez)
    fn imm(o: &mut String, reg: &str, v: u64) {
        if v < 0x8000 {
            let _ = writeln!(o, "\tli {}, {}", reg, v);
            return;
        }
        let h = halves(v);
        let _ = writeln!(
            o,
            "\tlis {r}, {}\n\tori {r}, {r}, {}\n\tsldi {r}, {r}, 32\n\toris {r}, {r}, {}\n\tori {r}, {r}, {}",
            sext(h[3], 16),
            h[2],
            h[1],
            h[0],
            r = reg
        );
    }
}
// markoa: 32 byteko goiburua (atzeko katea, CR, LR, TOC 24-n), r27-r31 32..72, aldi baterako lekua 72-n
const PPC_FRAME: u32 = 80;
impl Target for Ppc64le {
    fn header(&self, o: &mut String) {
        o.push_str("\t.abiversion 2\n");
    }
    fn prologue(&self, o: &mut String) {
        // r31 = makina, r30 = erregistroak, r29 = step, r28 = tick; r2 (TOC) gordeta, deien ondoren berreskuratzeko
        let _ = writeln!(
            o,
            "\tmflr 0\n\tstd 0, 16(1)\n\tstdu 1, -{f}(1)\n\tstd 27, 32(1)\n\tstd 28, 40(1)\n\tstd 29, 48(1)\n\tstd 30, 56(1)\n\tstd 31, 64(1)\n\tstd 2, 24(1)\n\
\tmr 31, 3\n\tmr 30, 4\n\tmr 29, 5\n\tmr 28, 6",
            f = PPC_FRAME
        );
    }
    fn set(&self, o: &mut String, r: u32, bits: u64) {
        Ppc64le::imm(o, "9", bits);
        let _ = writeln!(o, "\tstd 9, {}(30)", r * 8);
    }
    fn copy(&self, o: &mut String, dst: u32, src: u32) {
        let _ = writeln!(o, "\tld 9, {}(30)\n\tstd 9, {}(30)", src * 8, dst * 8);
    }
    fn arith(&self, o: &mut String, op: Arith, dst: u32, a: Operand, b: Operand) {
        for (x, f) in [(a, "0"), (b, "1")] {
            match x {
                Operand::Reg(r) => {
                    let _ = writeln!(o, "\tlfd {}, {}(30)", f, r * 8);
                }
                Operand::Num(bits) => {
                    Ppc64le::imm(o, "9", bits);
                    let _ = writeln!(o, "\tstd 9, 72(1)\n\tlfd {}, 72(1)", f);
                }
            }
        }
        let ins = match op {
            Arith::Add => "fadd",
            Arith::Sub => "fsub",
            Arith::Mul => "fmul",
        };
        let _ = writeln!(o, "\t{} 0, 0, 1\n\tstfd 0, {}(30)", ins, dst * 8);
    }
    fn tick(&self, o: &mut String, pc: usize) {
        o.push_str("\tmr 3, 31\n");
        Ppc64le::imm(o, "4", pc as u64);
        o.push_str("\tmr 12, 28\n\tmtctr 12\n\tbctrl\n\tld 2, 24(1)\n\tcmpdi 3, 0\n\tbeq 0, 1f\n\tb .Lerr\n1:\n");
    }
    fn step(&self, o: &mut String, pc: usize, next: usize) {
        o.push_str("\tmr 3, 31\n");
        Ppc64le::imm(o, "4", pc as u64);
        o.push_str("\tmr 12, 29\n\tmtctr 12\n\tbctrl\n\tld 2, 24(1)\n");
        Ppc64le::imm(o, "9", next as u64);
        o.push_str("\tcmpd 3, 9\n\tbeq 0, 1f\n\tb .Lmiss\n1:\n");
    }
    fn goto(&self, o: &mut String, l: &str) {
        let _ = writeln!(o, "\tb {}", l);
    }
    fn tail(&self, o: &mut String, len: usize) {
        let mut lim = String::new();
        Ppc64le::imm(&mut lim, "9", len as u64);
        let _ = write!(
            o,
            "{end}:\n\tli 3, 0\n.Lret:\n\tld 27, 32(1)\n\tld 28, 40(1)\n\tld 29, 48(1)\n\tld 30, 56(1)\n\tld 31, 64(1)\n\taddi 1, 1, {f}\n\tld 0, 16(1)\n\tmtlr 0\n\tblr\n\
.Lerr:\n\tli 3, 1\n\tb .Lret\n\
.Lbad:\n\tli 3, 2\n\tb .Lret\n\
.Lmiss:\n\tcmpdi 3, -1\n\tbeq 0, {end}\n\tcmpdi 3, 0\n\tblt 0, .Lerr\n{lim}\tcmpd 3, 9\n\tbeq 0, {end}\n\tbgt 0, .Lbad\n\
\tbcl 20, 31, .Lpc\n.Lpc:\n\tmflr 11\n\taddis 11, 11, (.Ltab-.Lpc)@ha\n\taddi 11, 11, (.Ltab-.Lpc)@l\n\
\tsldi 12, 3, 2\n\tlwax 12, 11, 12\n\tadd 11, 11, 12\n\tmtctr 11\n\tbctr\n",
            end = label(len),
            f = PPC_FRAME,
            lim = lim
        );
    }
}

struct S390x;
// markoa: 160 byte (ABI-aren gordelekua) + 8 aldi baterako lekua
const S390_FRAME: u32 = 168;
impl Target for S390x {
    fn header(&self, _o: &mut String) {}
    fn prologue(&self, o: &mut String) {
        // r13 = makina, r12 = erregistroak, r11 = step, r10 = tick
        let _ = writeln!(
            o,
            "\tstmg %r6, %r15, 48(%r15)\n\taghi %r15, -{f}\n\tlgr %r13, %r2\n\tlgr %r12, %r3\n\tlgr %r11, %r4\n\tlgr %r10, %r5",
            f = S390_FRAME
        );
    }
    fn set(&self, o: &mut String, r: u32, bits: u64) {
        let _ = writeln!(
            o,
            "\tllihf %r1, {:#x}\n\tiilf %r1, {:#x}\n\tstg %r1, {}(%r12)",
            bits >> 32,
            bits & 0xffff_ffff,
            r * 8
        );
    }
    fn copy(&self, o: &mut String, dst: u32, src: u32) {
        let _ = writeln!(
            o,
            "\tlg %r1, {}(%r12)\n\tstg %r1, {}(%r12)",
            src * 8,
            dst * 8
        );
    }
    fn arith(&self, o: &mut String, op: Arith, dst: u32, a: Operand, b: Operand) {
        for (x, f) in [(a, "%f0"), (b, "%f2")] {
            match x {
                Operand::Reg(r) => {
                    let _ = writeln!(o, "\tld {}, {}(%r12)", f, r * 8);
                }
                Operand::Num(bits) => {
                    let _ = writeln!(
                        o,
                        "\tllihf %r1, {:#x}\n\tiilf %r1, {:#x}\n\tldgr {}, %r1",
                        bits >> 32,
                        bits & 0xffff_ffff,
                        f
                    );
                }
            }
        }
        let ins = match op {
            Arith::Add => "adbr",
            Arith::Sub => "sdbr",
            Arith::Mul => "mdbr",
        };
        let _ = writeln!(o, "\t{} %f0, %f2\n\tstd %f0, {}(%r12)", ins, dst * 8);
    }
    fn tick(&self, o: &mut String, pc: usize) {
        let _ = writeln!(
            o,
            "\tlgr %r2, %r13\n\tllilf %r3, {}\n\tbasr %r14, %r10\n\tltgr %r2, %r2\n\tjgne .Lerr",
            pc
        );
    }
    fn step(&self, o: &mut String, pc: usize, next: usize) {
        let _ = writeln!(
            o,
            "\tlgr %r2, %r13\n\tllilf %r3, {}\n\tbasr %r14, %r11\n\tclgfi %r2, {}\n\tjgne .Lmiss",
            pc, next
        );
    }
    fn goto(&self, o: &mut String, l: &str) {
        let _ = writeln!(o, "\tjg {}", l);
    }
    fn tail(&self, o: &mut String, len: usize) {
        let _ = write!(
            o,
            "{end}:\n\tlghi %r2, 0\n.Lret:\n\tlmg %r6, %r15, {save}(%r15)\n\tbr %r14\n\
.Lerr:\n\tlghi %r2, 1\n\tj .Lret\n\
.Lbad:\n\tlghi %r2, 2\n\tj .Lret\n\
.Lmiss:\n\tcghi %r2, -1\n\tjge {end}\n\tltgr %r2, %r2\n\tjl .Lerr\n\tclgfi %r2, {len}\n\tjge {end}\n\tjh .Lbad\n\
\tlarl %r1, .Ltab\n\tsllg %r2, %r2, 2\n\tlgf %r3, 0(%r2,%r1)\n\tagr %r1, %r3\n\tbr %r1\n",
            end = label(len),
            save = S390_FRAME + 48,
            len = len
        );
    }
}

struct La64;
impl La64 {
    // 64 biteko balioa erregistro batean: lu12i.w, ori, lu32i.d, lu52i.d
    fn imm(o: &mut String, reg: &str, v: u64) {
        let _ = writeln!(
            o,
            "\tlu12i.w {r}, {}\n\tori {r}, {r}, {}\n\tlu32i.d {r}, {}\n\tlu52i.d {r}, {r}, {}",
            sext((v >> 12) & 0xfffff, 20),
            v & 0xfff,
            sext((v >> 32) & 0xfffff, 20),
            sext(v >> 52, 12),
            r = reg
        );
    }
}
impl Target for La64 {
    fn header(&self, _o: &mut String) {}
    fn prologue(&self, o: &mut String) {
        // s0 = makina, s1 = erregistroak, s2 = step, s3 = tick
        o.push_str("\taddi.d $sp, $sp, -48\n\tst.d $ra, $sp, 40\n\tst.d $s0, $sp, 32\n\tst.d $s1, $sp, 24\n\tst.d $s2, $sp, 16\n\tst.d $s3, $sp, 8\n");
        o.push_str("\tmove $s0, $a0\n\tmove $s1, $a1\n\tmove $s2, $a2\n\tmove $s3, $a3\n");
    }
    fn set(&self, o: &mut String, r: u32, bits: u64) {
        La64::imm(o, "$t0", bits);
        let _ = writeln!(o, "\tst.d $t0, $s1, {}", r * 8);
    }
    fn copy(&self, o: &mut String, dst: u32, src: u32) {
        let _ = writeln!(
            o,
            "\tld.d $t0, $s1, {}\n\tst.d $t0, $s1, {}",
            src * 8,
            dst * 8
        );
    }
    fn arith(&self, o: &mut String, op: Arith, dst: u32, a: Operand, b: Operand) {
        for (x, f) in [(a, "$fa0"), (b, "$fa1")] {
            match x {
                Operand::Reg(r) => {
                    let _ = writeln!(o, "\tfld.d {}, $s1, {}", f, r * 8);
                }
                Operand::Num(bits) => {
                    La64::imm(o, "$t0", bits);
                    let _ = writeln!(o, "\tmovgr2fr.d {}, $t0", f);
                }
            }
        }
        let ins = match op {
            Arith::Add => "fadd.d",
            Arith::Sub => "fsub.d",
            Arith::Mul => "fmul.d",
        };
        let _ = writeln!(
            o,
            "\t{} $fa0, $fa0, $fa1\n\tfst.d $fa0, $s1, {}",
            ins,
            dst * 8
        );
    }
    fn tick(&self, o: &mut String, pc: usize) {
        o.push_str("\tmove $a0, $s0\n");
        La64::imm(o, "$a1", pc as u64);
        o.push_str("\tjirl $ra, $s3, 0\n\tbeqz $a0, 1f\n\tb .Lerr\n1:\n");
    }
    fn step(&self, o: &mut String, pc: usize, next: usize) {
        o.push_str("\tmove $a0, $s0\n");
        La64::imm(o, "$a1", pc as u64);
        o.push_str("\tjirl $ra, $s2, 0\n");
        La64::imm(o, "$t0", next as u64);
        o.push_str("\tbeq $a0, $t0, 1f\n\tb .Lmiss\n1:\n");
    }
    fn goto(&self, o: &mut String, l: &str) {
        let _ = writeln!(o, "\tb {}", l);
    }
    fn tail(&self, o: &mut String, len: usize) {
        let mut lim = String::new();
        La64::imm(&mut lim, "$t0", len as u64);
        let _ = write!(
            o,
            "{end}:\n\tmove $a0, $zero\n.Lret:\n\tld.d $ra, $sp, 40\n\tld.d $s0, $sp, 32\n\tld.d $s1, $sp, 24\n\tld.d $s2, $sp, 16\n\tld.d $s3, $sp, 8\n\taddi.d $sp, $sp, 48\n\tjirl $zero, $ra, 0\n\
.Lerr:\n\taddi.d $a0, $zero, 1\n\tb .Lret\n\
.Lbad:\n\taddi.d $a0, $zero, 2\n\tb .Lret\n\
.Lmiss:\n\taddi.d $t0, $zero, -1\n\tbeq $a0, $t0, {end}\n\tblt $a0, $zero, .Lerr\n{lim}\tbeq $a0, $t0, {end}\n\tbltu $t0, $a0, .Lbad\n\
\tpcalau12i $t0, %pc_hi20(.Ltab)\n\taddi.d $t0, $t0, %pc_lo12(.Ltab)\n\tslli.d $t1, $a0, 2\n\tldx.w $t1, $t0, $t1\n\tadd.d $t0, $t0, $t1\n\tjirl $zero, $t0, 0\n",
            end = label(len),
            lim = lim
        );
    }
}

fn target(arch: Arch) -> Box<dyn Target> {
    match arch {
        Arch::X86_64 => Box::new(X86),
        Arch::Aarch64 => Box::new(A64),
        Arch::Riscv64 => Box::new(Rv64),
        Arch::Powerpc64le => Box::new(Ppc64le),
        Arch::S390x => Box::new(S390x),
        Arch::Loongarch64 => Box::new(La64),
    }
}

// programaren mihiztadura arkitektura baterako (GNU as, ELF): kodea, taula, eta .qx-a datu gisa
pub fn asm(p: &Program, arch: Arch) -> Result<String, String> {
    let t = target(arch);
    let n = p.ops.len();
    let qx = p.to_bytes();
    let mut o = String::new();
    let _ = writeln!(
        o,
        "# {}\n# {}",
        t!(
            "QOA native code for {} ({}), made by qoa {}",
            "{}-ren QOA kode natiboa ({}), qoa {}-k egina",
            p.name.replace('\n', " "),
            arch.name(),
            env!("CARGO_PKG_VERSION")
        ),
        t!(
            "entry: qoa_native_entry(machine, regs, step, tick); qoa run program.so runs it",
            "sarrera: qoa_native_entry(makina, erregistroak, step, tick); qoa run programa.so-k exekutatzen du"
        )
    );
    // `#` iruzkina da arkitektura guztietan, aarch64-n izan ezik (han `//`)
    if arch == Arch::Aarch64 {
        o = o.replace("# ", "// ");
    }
    t.header(&mut o);
    let _ = writeln!(
        o,
        "\t.text\n\t.globl {e}\n\t.type {e}, {kind}\n\t.p2align 4\n{e}:",
        e = ENTRY,
        kind = if arch == Arch::Aarch64 {
            "%function"
        } else {
            "@function"
        }
    );
    t.prologue(&mut o);
    for (pc, op) in p.ops.iter().enumerate() {
        let _ = writeln!(o, "{}:", label(pc));
        match lower(op) {
            Lower::Set { r, bits } => {
                t.tick(&mut o, pc);
                t.set(&mut o, r, bits);
            }
            Lower::Copy { dst, src } => {
                t.tick(&mut o, pc);
                t.copy(&mut o, dst, src);
            }
            Lower::Arith { op, dst, a, b } => {
                t.tick(&mut o, pc);
                t.arith(&mut o, op, dst, a, b);
            }
            Lower::Jump { to } => {
                t.tick(&mut o, pc);
                t.goto(&mut o, &label(to.min(n)));
            }
            Lower::Runtime => t.step(&mut o, pc, pc + 1),
        }
    }
    t.tail(&mut o, n);
    // taula: eragiketa bakoitzaren kodea, taularen hasieratik neurtuta (kokapenaren menpe ez dagoena)
    o.push_str("\t.p2align 2\n.Ltab:\n");
    for pc in 0..n {
        let _ = writeln!(o, "\t.long {} - .Ltab", label(pc));
    }
    let _ = writeln!(o, "\t.size {e}, .-{e}", e = ENTRY);
    // programa: .qx-a, haren luzera eta hitzarmenaren bertsioa
    let _ = writeln!(
        o,
        "\t.section .rodata\n\t.globl {q}\n\t.p2align 3\n{q}:",
        q = QX
    );
    for chunk in qx.chunks(16) {
        let bytes: Vec<String> = chunk.iter().map(|b| b.to_string()).collect();
        let _ = writeln!(o, "\t.byte {}", bytes.join(","));
    }
    let _ = writeln!(
        o,
        "\t.globl {l}\n\t.p2align 3\n{l}:\n\t.quad {}\n\t.globl {a}\n\t.p2align 2\n{a}:\n\t.long {}",
        qx.len(),
        ABI_VERSION,
        l = QX_LEN,
        a = ABI
    );
    o.push_str(t.note_section());
    Ok(o)
}
