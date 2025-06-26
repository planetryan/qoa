// this file defines the instruction set architecture (ISA) for the quantum optical assembly (QOA)
// it includes the instruction enum, parsing logic, and byte serialization.

// defines the different types of instructions available in qoa.
// each variant represents a specific operation the qoa vm can perform.
#[derive(Debug, PartialEq, Clone)]
pub enum Instruction {
    // core
    QInit(u8),
    QMeas(u8),
    ApplyHadamard(u8),
    ControlledNot(u8, u8),
    ApplyPhaseFlip(u8),
    ApplyBitFlip(u8),
    ApplyTGate(u8),
    ApplySGate(u8),
    PhaseShift(u8, f64),
    Wait(u64),
    Reset(u8),
    Swap(u8, u8),
    ControlledSwap(u8, u8, u8),
    Entangle(u8, u8),
    EntangleBell(u8, u8),
    EntangleMulti(Vec<u8>),
    EntangleCluster(Vec<u8>),
    EntangleSwap(u8, u8, u8, u8),
    EntangleSwapMeasure(u8, u8, u8, u8, String),
    EntangleWithClassicalFeedback(u8, u8, String),
    EntangleDistributed(u8, String),
    MeasureInBasis(u8, String),
    Sync,
    ResetAll,
    VerboseLog(u8, String),
    SetPhase(u8, f64),
    ApplyGate(String, u8),
    Measure(u8),   // Duplicate of QMeas, but keeping for now
    InitQubit(u8), // Duplicate of QInit, but keeping for now

    // charprinting
    CharLoad(u8, u8),
    CharOut(u8),

    // ionq isa
    RX(u8, f64),
    RY(u8, f64),
    RZ(u8, f64),
    Phase(u8, f64),
    CNOT(u8, u8),
    CZ(u8, u8),
    QReset(u8),
    ThermalAvg(u8, u8),
    WkbFactor(u8, u8, u8),

    // regset
    RegSet(u8, f64),

    // loop
    LoopStart(u8),
    LoopEnd,

    // rotations
    ApplyRotation(u8, char, f64),
    ApplyMultiQubitRotation(Vec<u8>, char, Vec<f64>),
    ControlledPhaseRotation(u8, u8, f64),
    ApplyCPhase(u8, u8, f64),
    ApplyKerrNonlinearity(u8, f64, u64),
    ApplyFeedforwardGate(u8, String),
    DecoherenceProtect(u8, u64),
    ApplyMeasurementBasisChange(u8, String),

    // memory/classical ops
    Load(u8, String),
    Store(u8, String),
    LoadMem(String, String),
    StoreMem(String, String),
    LoadClassical(String, String),
    StoreClassical(String, String),
    Add(String, String, String),
    Sub(String, String, String),
    And(String, String, String),
    Or(String, String, String),
    Xor(String, String, String),
    Not(String),
    Push(String),
    Pop(String),
    // classical
    Jump(String),
    JumpIfZero(String, String),
    JumpIfOne(String, String),
    Call(String),
    Barrier,
    Return,
    TimeDelay(u8, u64),
    Rand(u8),
    Sqrt(u8, u8),
    Exp(u8, u8),
    Log(u8, u8),
    // arithmetic operations
    RegAdd(u8, u8, u8), // rd, ra, rb
    RegSub(u8, u8, u8),
    RegMul(u8, u8, u8),
    RegDiv(u8, u8, u8),
    RegCopy(u8, u8), // rd, ra

    // optics
    PhotonEmit(u8),
    PhotonDetect(u8),
    PhotonCount(u8, String),
    PhotonAddition(u8),
    ApplyPhotonSubtraction(u8),
    PhotonEmissionPattern(u8, String, u64),
    PhotonDetectWithThreshold(u8, u64, String),
    PhotonDetectCoincidence(Vec<u8>, String),
    SinglePhotonSourceOn(u8),
    SinglePhotonSourceOff(u8),
    PhotonBunchingControl(u8, bool),
    PhotonRoute(u8, String, String),
    OpticalRouting(u8, u8),
    SetOpticalAttenuation(u8, f64),
    DynamicPhaseCompensation(u8, f64),
    OpticalDelayLineControl(u8, u64),
    CrossPhaseModulation(u8, u8, f64),
    ApplyDisplacement(u8, f64),
    ApplyDisplacementFeedback(u8, String),
    ApplyDisplacementOperator(u8, f64, u64),
    ApplySqueezing(u8, f64),
    ApplySqueezingFeedback(u8, String),
    MeasureParity(u8),
    MeasureWithDelay(u8, u64, String),
    OpticalSwitchControl(u8, bool),
    PhotonLossSimulate(u8, f64, u64),
    PhotonLossCorrection(u8, String),

    // qubit measurement
    ApplyQndMeasurement(u8, String),
    ErrorCorrect(u8, String),
    ErrorSyndrome(u8, String, String),
    QuantumStateTomography(u8, String),
    BellStateVerification(u8, u8, String),
    QuantumZenoEffect(u8, u64, u64),
    ApplyNonlinearPhaseShift(u8, f64),
    ApplyNonlinearSigma(u8, f64),
    ApplyLinearOpticalTransform(String, Vec<u8>, Vec<u8>, usize),
    PhotonNumberResolvingDetection(u8, String),
    FeedbackControl(u8, String),

    // misc
    SetPos(u8, f64, f64),
    SetWl(u8, f64),
    WlShift(u8, f64),
    Move(u8, f64, f64),
    Comment(String),
    MarkObserved(u8),
    Release(u8),
    Halt,

    // new instructions for v0.3.0 development todo

    // control flow & program structure
    Jmp(i64),          // jump relative by offset (i64 for signed offset)
    JmpAbs(u64),       // jump absolute to instruction index
    IfGt(u8, u8, i64), // if reg1 > reg2, jump relative by offset
    IfLt(u8, u8, i64), // if reg1 < reg2, jump relative by offset
    IfEq(u8, u8, i64), // if reg1 == reg2, jump relative by offset
    IfNe(u8, u8, i64), // if reg1 != reg2, jump relative by offset
    // LABEL is a pseudo-instruction, handled during assembly/compilation, not runtime
    CallAddr(u64), // call subroutine at absolute address, push return address to stack (renamed to avoid conflict with existing Call(String))
    RetSub, // return from subroutine, pop return address from stack (renamed to avoid conflict with existing Return)

    // input/output & debugging
    Printf(String, Vec<u8>), // c-style formatted output (format string, register indices)
    Print(String),           // print a string literal
    Println(String),         // print a string literal with newline
    Input(u8),               // read floating-point value from stdin into register
    DumpState,               // output quantum amplitudes and phases
    DumpRegs,                // output all register values

    // memory & stack
    LoadRegMem(u8, u64), // load value from memory address into register (reg, mem_addr) (renamed to avoid conflict with existing Load)
    StoreMemReg(u64, u8), // store value from register into memory address (mem_addr, reg) (renamed to avoid conflict with existing Store)
    PushReg(u8), // push register value onto stack (renamed to avoid conflict with existing Push)
    PopReg(u8),  // pop value from stack into register (renamed to avoid conflict with existing Pop)
    Alloc(u8, u64), // allocate memory, store start address in register (reg_addr, size)
    Free(u64),   // free memory at address

    // comparison & bitwise logic
    Cmp(u8, u8), // compare reg1 and reg2, set internal flags (flags handled by runtime)
    AndBits(u8, u8, u8), // bitwise and (dest, op1, op2) (renamed to avoid conflict with existing And)
    OrBits(u8, u8, u8),  // bitwise or (dest, op1, op2) (renamed to avoid conflict with existing Or)
    XorBits(u8, u8, u8), // bitwise xor (dest, op1, op2) (renamed to avoid conflict with existing Xor)
    NotBits(u8, u8),     // bitwise not (dest, op) (renamed to avoid conflict with existing Not)
    Shl(u8, u8, u8),     // shift left (dest, op, amount_reg)
    Shr(u8, u8, u8),     // shift right (dest, op, amount_reg)

    // system & debug utilities
    BreakPoint,  // breakpoint instruction (renamed to avoid conflict with 'break' keyword)
    GetTime(u8), // get system timestamp into register (renamed to avoid conflict with existing TimeDelay)
    SeedRng(u64), // seed rng for reproducible results (renamed to avoid conflict with existing Rand)
    ExitCode(i32), // terminate program with exit code (renamed to avoid conflict with existing Halt)
}

// opcodes for each instruction. these are used for serialization/deserialization
// and are crucial for the vm to understand the bytecode.
impl Instruction {
    pub fn encode(&self) -> Vec<u8> {
        match self {
            // Core
            Instruction::QInit(q) => vec![0x04, *q],
            Instruction::QMeas(q) => vec![0x32, *q],
            Instruction::ApplyHadamard(q) => vec![0x05, *q],
            Instruction::ControlledNot(c, t) => vec![0x17, *c, *t],
            Instruction::ApplyPhaseFlip(q) => vec![0x06, *q],
            Instruction::ApplyBitFlip(q) => vec![0x07, *q],
            Instruction::ApplyTGate(q) => vec![0x0D, *q],
            Instruction::ApplySGate(q) => vec![0x0E, *q],
            Instruction::PhaseShift(q, angle) => {
                let mut v = vec![0x08, *q];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::Wait(cycles) => {
                let mut v = vec![0x09];
                v.extend(&cycles.to_le_bytes());
                v
            }
            Instruction::Reset(q) => vec![0x0A, *q],
            Instruction::Swap(q1, q2) => vec![0x0B, *q1, *q2],
            Instruction::ControlledSwap(c, t1, t2) => vec![0x0C, *c, *t1, *t2],
            Instruction::Entangle(q1, q2) => vec![0x11, *q1, *q2],
            Instruction::EntangleBell(q1, q2) => vec![0x12, *q1, *q2],
            Instruction::EntangleMulti(qs) => {
                let mut v = vec![0x13, qs.len() as u8];
                v.extend(qs.iter());
                v
            }
            Instruction::EntangleCluster(qs) => {
                let mut v = vec![0x14, qs.len() as u8];
                v.extend(qs.iter());
                v
            }
            Instruction::EntangleSwap(a, b, c, d) => vec![0x15, *a, *b, *c, *d],
            Instruction::EntangleSwapMeasure(a, b, c, d, label) => {
                let mut v = vec![0x16, *a, *b, *c, *d];
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::EntangleWithClassicalFeedback(q1, q2, signal) => {
                let mut v = vec![0x19, *q1, *q2];
                v.extend(signal.as_bytes());
                v.push(0);
                v
            }
            Instruction::EntangleDistributed(q, node) => {
                let mut v = vec![0x1A, *q];
                v.extend(node.as_bytes());
                v.push(0);
                v
            }
            Instruction::MeasureInBasis(q, basis) => {
                let mut v = vec![0x1B, *q];
                v.extend(basis.as_bytes());
                v.push(0);
                v
            }
            Instruction::Sync => vec![0x48],
            Instruction::ResetAll => vec![0x1C],
            Instruction::VerboseLog(q, msg) => {
                let mut v = vec![0x87, *q];
                v.extend(msg.as_bytes());
                v.push(0);
                v
            }
            Instruction::SetPhase(q, phase) => {
                let mut v = vec![0x1D, *q];
                v.extend(&phase.to_le_bytes());
                v
            }
            Instruction::ApplyGate(name, q) => {
                let mut v = vec![0x02, *q];
                v.extend(name.as_bytes());
                for _ in name.len()..8 {
                    v.push(0);
                }
                v
            }
            Instruction::Measure(q) => vec![0x32, *q],
            Instruction::InitQubit(q) => vec![0x04, *q],

            // Char printing
            Instruction::CharLoad(reg, val) => vec![0x31, *reg, *val],
            Instruction::CharOut(reg) => vec![0x18, *reg],

            // IonQ ISA gates
            Instruction::RX(q, angle) => {
                let mut v = vec![0x22, *q];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::RY(q, angle) => {
                let mut v = vec![0x23, *q];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::RZ(q, angle) => {
                let mut v = vec![0x0F, *q];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::Phase(q, angle) => {
                let mut v = vec![0x24, *q];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::CNOT(c, t) => vec![0x17, *c, *t],
            Instruction::CZ(c, t) => vec![0x1E, *c, *t],
            Instruction::QReset(q) => vec![0x0A, *q],
            Instruction::ThermalAvg(q, param) => vec![0x1F, *q, *param],
            Instruction::WkbFactor(q1, q2, param) => vec![0x20, *q1, *q2, *param],

            // Regset
            Instruction::RegSet(reg, val) => {
                let mut v = vec![0x21, *reg];
                v.extend(&val.to_le_bytes());
                v
            }

            // Loop
            Instruction::LoopStart(reg) => vec![0x01, *reg],
            Instruction::LoopEnd => vec![0x10],

            // Rotations
            Instruction::ApplyRotation(q, axis, angle) => {
                let mut v = vec![0x33, *q, *axis as u8];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::ApplyMultiQubitRotation(qs, axis, angles) => {
                let mut v = vec![0x34, *axis as u8, qs.len() as u8];
                v.extend(qs.iter());
                for a in angles {
                    v.extend(&a.to_le_bytes());
                }
                v
            }
            Instruction::ControlledPhaseRotation(c, t, angle) => {
                let mut v = vec![0x35, *c, *t];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::ApplyCPhase(q1, q2, angle) => {
                let mut v = vec![0x36, *q1, *q2];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::ApplyKerrNonlinearity(q, strength, duration) => {
                let mut v = vec![0x37, *q];
                v.extend(&strength.to_le_bytes());
                v.extend(&duration.to_le_bytes());
                v
            }
            Instruction::ApplyFeedforwardGate(q, reg) => {
                let mut v = vec![0x38, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::DecoherenceProtect(q, duration) => {
                let mut v = vec![0x39, *q];
                v.extend(&duration.to_le_bytes());
                v
            }
            Instruction::ApplyMeasurementBasisChange(q, basis) => {
                let mut v = vec![0x3A, *q];
                v.extend(basis.as_bytes());
                v.push(0);
                v
            }

            // Memory/Classical Ops (Originals)
            Instruction::Load(q, reg) => {
                let mut v = vec![0x3B, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::Store(q, reg) => {
                let mut v = vec![0x3C, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::LoadMem(reg, addr) => {
                let mut v = vec![0x3D];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(addr.as_bytes());
                v.push(0);
                v
            }
            Instruction::StoreMem(reg, addr) => {
                let mut v = vec![0x3E];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(addr.as_bytes());
                v.push(0);
                v
            }
            Instruction::LoadClassical(reg, var) => {
                let mut v = vec![0x3F];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(var.as_bytes());
                v.push(0);
                v
            }
            Instruction::StoreClassical(reg, var) => {
                let mut v = vec![0x40];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(var.as_bytes());
                v.push(0);
                v
            }
            Instruction::Add(dst, src1, src2) => {
                let mut v = vec![0x41];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::Sub(dst, src1, src2) => {
                let mut v = vec![0x42];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::And(dst, src1, src2) => {
                let mut v = vec![0x43];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::Or(dst, src1, src2) => {
                let mut v = vec![0x44];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::Xor(dst, src1, src2) => {
                let mut v = vec![0x45];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::Not(reg) => {
                let mut v = vec![0x46];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::Push(reg) => {
                let mut v = vec![0x47];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::Pop(reg) => {
                let mut v = vec![0x4F];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }

            // Classical flow control (Originals)
            Instruction::Jump(label) => {
                let mut v = vec![0x49];
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::JumpIfZero(cond, label) => {
                let mut v = vec![0x4A];
                v.extend(cond.as_bytes());
                v.push(0);
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::JumpIfOne(cond, label) => {
                let mut v = vec![0x4B];
                v.extend(cond.as_bytes());
                v.push(0);
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::Call(label) => {
                let mut v = vec![0x4C];
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::Return => vec![0x4D],
            Instruction::TimeDelay(q, cycles) => {
                let mut v = vec![0x4E, *q];
                v.extend(&cycles.to_le_bytes());
                v
            }
            Instruction::Rand(reg) => vec![0x50, *reg],
            Instruction::Sqrt(rd, rs) => vec![0x51, *rd, *rs],
            Instruction::Exp(rd, rs) => vec![0x52, *rd, *rs],
            Instruction::Log(rd, rs) => vec![0x53, *rd, *rs],
            Instruction::RegAdd(rd, ra, rb) => vec![0x54, *rd, *ra, *rb],
            Instruction::RegSub(rd, ra, rb) => vec![0x55, *rd, *ra, *rb],
            Instruction::RegMul(rd, ra, rb) => vec![0x56, *rd, *ra, *rb],
            Instruction::RegDiv(rd, ra, rb) => vec![0x57, *rd, *ra, *rb],
            Instruction::RegCopy(rd, ra) => vec![0x58, *rd, *ra],

            // Optics / photonics
            Instruction::PhotonEmit(q) => vec![0x59, *q],
            Instruction::PhotonDetect(q) => vec![0x5A, *q],
            Instruction::PhotonCount(q, reg) => {
                let mut v = vec![0x5B, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::PhotonAddition(q) => vec![0x5C, *q],
            Instruction::ApplyPhotonSubtraction(q) => vec![0x5D, *q],
            Instruction::PhotonEmissionPattern(q, reg, cycles) => {
                let mut v = vec![0x5E, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(&cycles.to_le_bytes());
                v
            }
            Instruction::PhotonDetectWithThreshold(q, thresh, reg) => {
                let mut v = vec![0x5F, *q];
                v.extend(&thresh.to_le_bytes());
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::PhotonDetectCoincidence(qs, reg) => {
                let mut v = vec![0x60, qs.len() as u8];
                v.extend(qs.iter());
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::SinglePhotonSourceOn(q) => vec![0x61, *q],
            Instruction::SinglePhotonSourceOff(q) => vec![0x62, *q],
            Instruction::PhotonBunchingControl(q, b) => vec![0x63, *q, *b as u8],
            Instruction::PhotonRoute(q, from, to) => {
                let mut v = vec![0x64, *q];
                v.extend(from.as_bytes());
                v.push(0);
                v.extend(to.as_bytes());
                v.push(0);
                v
            }
            Instruction::OpticalRouting(q1, q2) => vec![0x65, *q1, *q2],
            Instruction::SetOpticalAttenuation(q, att) => {
                let mut v = vec![0x66, *q];
                v.extend(&att.to_le_bytes());
                v
            }
            Instruction::DynamicPhaseCompensation(q, phase) => {
                let mut v = vec![0x67, *q];
                v.extend(&phase.to_le_bytes());
                v
            }
            Instruction::OpticalDelayLineControl(q, delay) => {
                let mut v = vec![0x68, *q];
                v.extend(&delay.to_le_bytes());
                v
            }
            Instruction::CrossPhaseModulation(c, t, stren) => {
                let mut v = vec![0x69, *c, *t];
                v.extend(&stren.to_le_bytes());
                v
            }
            Instruction::ApplyDisplacement(q, a) => {
                let mut v = vec![0x6A, *q];
                v.extend(&a.to_le_bytes());
                v
            }
            Instruction::ApplyDisplacementFeedback(q, reg) => {
                let mut v = vec![0x6B, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::ApplyDisplacementOperator(q, alpha, dur) => {
                let mut v = vec![0x6C, *q];
                v.extend(&alpha.to_le_bytes());
                v.extend(&dur.to_le_bytes());
                v
            }
            Instruction::ApplySqueezing(q, s) => {
                let mut v = vec![0x6D, *q];
                v.extend(&s.to_le_bytes());
                v
            }
            Instruction::ApplySqueezingFeedback(q, reg) => {
                let mut v = vec![0x6E, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::MeasureParity(q) => vec![0x6F, *q],
            Instruction::MeasureWithDelay(q, delay, reg) => {
                let mut v = vec![0x70, *q];
                v.extend(&delay.to_le_bytes());
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::OpticalSwitchControl(q, b) => vec![0x71, *q, *b as u8],
            Instruction::PhotonLossSimulate(q, prob, seed) => {
                let mut v = vec![0x72, *q];
                v.extend(&prob.to_le_bytes());
                v.extend(&seed.to_le_bytes());
                v
            }
            Instruction::PhotonLossCorrection(q, reg) => {
                let mut v = vec![0x73, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }

            // Misc (Originals)
            Instruction::SetPos(q, x, y) => {
                let mut v = vec![0x74, *q];
                v.extend(&x.to_le_bytes());
                v.extend(&y.to_le_bytes());
                v
            }
            Instruction::SetWl(q, wl) => {
                let mut v = vec![0x75, *q];
                v.extend(&wl.to_le_bytes());
                v
            }
            Instruction::WlShift(q, wl_delta) => {
                let mut v = vec![0x76, *q];
                v.extend(&wl_delta.to_le_bytes());
                v
            }
            Instruction::Move(q, dx, dy) => {
                let mut v = vec![0x77, *q];
                v.extend(&dx.to_le_bytes());
                v.extend(&dy.to_le_bytes());
                v
            }
            Instruction::Comment(text) => {
                let mut v = vec![0x88];
                v.extend(text.as_bytes());
                v.push(0);
                v
            }
            Instruction::MarkObserved(q) => vec![0x79, *q],
            Instruction::Release(q) => vec![0x7A, *q],
            Instruction::Halt => vec![0xFF],

            // QGATE H will now map to ApplyHadamard(u8) which uses 0x05.
            Instruction::Barrier => vec![0x89],
            Instruction::ApplyQndMeasurement(q, reg) => {
                let mut v = vec![0x7C, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::ErrorCorrect(q, syndrome_type) => {
                let mut v = vec![0x7D, *q];
                v.extend(syndrome_type.as_bytes());
                v.push(0);
                v
            }
            Instruction::ErrorSyndrome(q, syndrome_type, result_reg) => {
                let mut v = vec![0x7E, *q];
                v.extend(syndrome_type.as_bytes());
                v.push(0);
                v.extend(result_reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::QuantumStateTomography(q, basis) => {
                let mut v = vec![0x7F, *q];
                v.extend(basis.as_bytes());
                v.push(0);
                v
            }
            Instruction::BellStateVerification(q1, q2, result_reg) => {
                let mut v = vec![0x80, *q1, *q2];
                v.extend(result_reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::QuantumZenoEffect(q, num_measurements, interval_cycles) => {
                let mut v = vec![0x81, *q];
                v.extend(&num_measurements.to_le_bytes());
                v.extend(&interval_cycles.to_le_bytes());
                v
            }
            Instruction::ApplyNonlinearPhaseShift(q, strength) => {
                let mut v = vec![0x82, *q];
                v.extend(&strength.to_le_bytes());
                v
            }
            Instruction::ApplyNonlinearSigma(q, strength) => {
                let mut v = vec![0x83, *q];
                v.extend(&strength.to_le_bytes());
                v
            }
            Instruction::ApplyLinearOpticalTransform(name, input_qs, output_qs, num_modes) => {
                let mut v = vec![
                    0x84,
                    input_qs.len() as u8,
                    output_qs.len() as u8,
                    *num_modes as u8,
                ];
                v.extend(name.as_bytes());
                v.push(0);
                v.extend(input_qs.iter());
                v.extend(output_qs.iter());
                v
            }
            Instruction::PhotonNumberResolvingDetection(q, reg) => {
                let mut v = vec![0x85, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::FeedbackControl(q, reg) => {
                let mut v = vec![0x86, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }

            // new instruction serialization for v0.3.0
            Instruction::Jmp(offset) => {
                let mut v = vec![0x90];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::JmpAbs(addr) => {
                let mut v = vec![0x91]; // Explicit opcode for JmpAbs
                v.extend(&addr.to_le_bytes());
                v
            }
            Instruction::IfGt(r1, r2, offset) => {
                let mut v = vec![0x92, *r1, *r2];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::IfLt(r1, r2, offset) => {
                let mut v = vec![0x93, *r1, *r2];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::IfEq(r1, r2, offset) => {
                let mut v = vec![0x94, *r1, *r2];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::IfNe(r1, r2, offset) => {
                let mut v = vec![0x95, *r1, *r2];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::CallAddr(addr) => {
                let mut v = vec![0x96];
                v.extend(&addr.to_le_bytes());
                v
            }
            Instruction::RetSub => {
                vec![0x97]
            }
            Instruction::Printf(format_str, regs) => {
                let mut v = vec![0x98];
                v.extend(&(format_str.len() as u64).to_le_bytes()); // Length of string
                v.extend(format_str.as_bytes()); // Format string
                v.push(regs.len() as u8); // Number of registers
                v.extend(regs); // Register indices
                v
            }
            Instruction::Print(s) => {
                let mut v = vec![0x99];
                v.extend(&(s.len() as u64).to_le_bytes()); // Length of string
                v.extend(s.as_bytes()); // The string itself
                v
            }
            Instruction::Println(s) => {
                let mut v = vec![0x9A];
                v.extend(&(s.len() as u64).to_le_bytes()); // Length of string
                v.extend(s.as_bytes()); // The string itself
                v
            }
            Instruction::Input(q) => {
                vec![0x9B, *q]
            }
            Instruction::DumpState => {
                vec![0x9C]
            }
            Instruction::DumpRegs => {
                vec![0x9D]
            }
            Instruction::LoadRegMem(reg, addr) => {
                let mut v = vec![0x9E, *reg];
                v.extend(&addr.to_le_bytes());
                v
            }
            Instruction::StoreMemReg(addr, reg) => {
                let mut v = vec![0x9F];
                v.extend(&addr.to_le_bytes());
                v.push(*reg);
                v
            }
            Instruction::PushReg(q) => {
                vec![0xA0, *q]
            }
            Instruction::PopReg(q) => {
                vec![0xA1, *q]
            }
            Instruction::Alloc(reg_addr, size) => {
                let mut v = vec![0xA2, *reg_addr];
                v.extend(&size.to_le_bytes());
                v
            }
            Instruction::Free(addr) => {
                let mut v = vec![0xA3];
                v.extend(&addr.to_le_bytes());
                v
            }
            Instruction::Cmp(r1, r2) => {
                vec![0xA4, *r1, *r2]
            }
            Instruction::AndBits(d, o1, o2) => {
                vec![0xA5, *d, *o1, *o2]
            }
            Instruction::OrBits(d, o1, o2) => {
                vec![0xA6, *d, *o1, *o2]
            }
            Instruction::XorBits(d, o1, o2) => {
                vec![0xA7, *d, *o1, *o2]
            }
            Instruction::NotBits(d, o) => {
                vec![0xA8, *d, *o]
            }
            Instruction::Shl(d, o1, o2) => {
                vec![0xA9, *d, *o1, *o2]
            }
            Instruction::Shr(d, o1, o2) => {
                vec![0xAA, *d, *o1, *o2]
            }
            Instruction::BreakPoint => {
                vec![0xAB]
            }
            Instruction::GetTime(q) => {
                vec![0xAC, *q]
            }
            Instruction::SeedRng(s) => {
                let mut v = vec![0xAD];
                v.extend(&s.to_le_bytes());
                v
            }
            Instruction::ExitCode(code) => {
                let mut v = vec![0xAE];
                v.extend(&code.to_le_bytes());
                v
            }
        }
    }
}

pub fn parse_instruction(line: &str) -> Result<Instruction, String> {
    use Instruction::*;
    let trimmed_line = line.trim();

    if trimmed_line.is_empty() {
        return Err("empty instruction line".into());
    }

    // Handle full-line comments starting with ';'
    if trimmed_line.starts_with(';') {
        // Return a Comment instruction directly
        return Ok(Comment(trimmed_line[1..].trim().to_string()));
    }

    // Strip inline comments before tokenizing
    let instruction_part = trimmed_line.split(';').next().unwrap_or("").trim();

    if instruction_part.is_empty() {
        return Err("empty instruction after stripping comment".into());
    }

    let tokens: Vec<&str> = instruction_part.split_whitespace().collect();
    let op = tokens[0].to_uppercase(); // Get the opcode, convert to uppercase

    // helper closures
    let parse_u8 = |s: &str| s.parse::<u8>().map_err(|_| format!("invalid u8 '{}'", s));
    let _parse_u16 = |s: &str| s.parse::<u16>().map_err(|_| format!("invalid u16 '{}'", s));
    let _parse_i16 = |s: &str| s.parse::<i16>().map_err(|_| format!("invalid i16 '{}'", s));
    let parse_u64 = |s: &str| s.parse::<u64>().map_err(|_| format!("invalid u64 '{}'", s));
    let parse_f64 = |s: &str| s.parse::<f64>().map_err(|_| format!("invalid f64 '{}'", s));
    let parse_i64 = |s: &str| s.parse::<i64>().map_err(|_| format!("invalid i64 '{}'", s));
    let _parse_bool = |s: &str| match s.to_uppercase().as_str() {
        "TRUE" | "ON" => Ok(true),
        "FALSE" | "OFF" => Ok(false),
        _ => Err(format!("invalid bool '{}'", s)),
    };
    let parse_axis = |s: &str| {
        let c = s.to_uppercase();
        if c == "X" || c == "Y" || c == "Z" {
            Ok(c.chars().next().unwrap())
        } else {
            Err(format!("invalid axis '{}'", s))
        }
    };
    // new helper closure to parse a string literal (quoted)
    let parse_string_literal = |arg_str: &str| -> Result<String, String> {
        if arg_str.starts_with('"') && arg_str.ends_with('"') && arg_str.len() >= 2 {
            Ok(arg_str[1..arg_str.len() - 1].to_string())
        } else {
            Err(format!(
                "invalid string literal '{}': must be double-quoted",
                arg_str
            ))
        }
    };

    // new helper closure to parse multiple register arguments
    let parse_reg_list = |args_slice: &[&str]| -> Result<Vec<u8>, String> {
        let mut regs = vec![];
        for (i, arg) in args_slice.iter().enumerate() {
            regs.push(
                arg.parse::<u8>()
                    .map_err(|e| format!("invalid register argument at index {}: {}", i, e))?,
            );
        }
        Ok(regs)
    };

    match op.as_str() {
        // Core
        "QINIT" => {
            if tokens.len() == 2 {
                Ok(QInit(parse_u8(tokens[1])?))
            } else {
                Err("qinit <qubit>".into())
            }
        }
        "QMEAS" => {
            if tokens.len() == 2 {
                Ok(QMeas(parse_u8(tokens[1])?))
            } else {
                Err("qmeas <qubit>".into())
            }
        }
        "APPLYHADAMARD" | "H" | "AH" => {
            if tokens.len() == 2 {
                Ok(ApplyHadamard(parse_u8(tokens[1])?))
            } else {
                Err("usage: h <qubit> / ah <qubit> / applyhadamard <qubit>".into())
            }
        }
        "CONTROLLEDNOT" => {
            if tokens.len() == 3 {
                Ok(ControlledNot(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("controllednot <c> <t>".into())
            }
        }
        "APPLYPHASEFLIP" => {
            if tokens.len() == 2 {
                Ok(ApplyPhaseFlip(parse_u8(tokens[1])?))
            } else {
                Err("applyphaseflip <qubit>".into())
            }
        }
        "APPLYBITFLIP" => {
            if tokens.len() == 2 {
                Ok(ApplyBitFlip(parse_u8(tokens[1])?))
            } else {
                Err("applybitflip <qubit>".into())
            }
        }
        "APPLYTGATE" => {
            if tokens.len() == 2 {
                Ok(ApplyTGate(parse_u8(tokens[1])?))
            } else {
                Err("applytgate <qubit>".into())
            }
        }
        "APPLYSGATE" => {
            if tokens.len() == 2 {
                Ok(ApplySGate(parse_u8(tokens[1])?))
            } else {
                Err("applysgate <qubit>".into())
            }
        }
        "PHASESHIFT" => {
            if tokens.len() == 3 {
                Ok(PhaseShift(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("phaseshift <qubit> <angle>".into())
            }
        }
        "WAIT" => {
            if tokens.len() == 2 {
                Ok(Wait(parse_u64(tokens[1])?))
            } else {
                Err("wait <cycles>".into())
            }
        }
        "RESET" => {
            if tokens.len() == 2 {
                Ok(Reset(parse_u8(tokens[1])?))
            } else {
                Err("reset <qubit>".into())
            }
        }
        "SWAP" => {
            if tokens.len() == 3 {
                Ok(Swap(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("swap <q1> <q2>".into())
            }
        }
        "CONTROLLEDSWAP" => {
            if tokens.len() == 4 {
                Ok(ControlledSwap(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_u8(tokens[3])?,
                ))
            } else {
                Err("controlledswap <c> <t1> <t2>".into())
            }
        }
        "ENTANGLE" => {
            if tokens.len() == 3 {
                Ok(Entangle(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("entangle <q1> <q2>".into())
            }
        }
        "ENTANGLEBELL" => {
            if tokens.len() == 3 {
                Ok(EntangleBell(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("entanglebell <q1> <q2>".into())
            }
        }
        "ENTANGLEMULTI" => {
            if tokens.len() >= 2 {
                Ok(EntangleMulti(
                    tokens[1..]
                        .iter()
                        .map(|q| parse_u8(q))
                        .collect::<Result<_, _>>()?,
                ))
            } else {
                Err("entanglemulti <q1> <q2> ...".into())
            }
        }
        "ENTANGLECLUSTER" => {
            if tokens.len() >= 2 {
                Ok(EntangleCluster(
                    tokens[1..]
                        .iter()
                        .map(|q| parse_u8(q))
                        .collect::<Result<_, _>>()?,
                ))
            } else {
                Err("entanglecluster <q1> <q2> ...".into())
            }
        }
        "ENTANGLESWAP" => {
            if tokens.len() == 5 {
                Ok(EntangleSwap(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_u8(tokens[3])?,
                    parse_u8(tokens[4])?,
                ))
            } else {
                Err("entangleswap <a> <b> <c> <d>".into())
            }
        }
        "ENTANGLESWAPMEASURE" => {
            if tokens.len() == 6 {
                Ok(EntangleSwapMeasure(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_u8(tokens[3])?,
                    parse_u8(tokens[4])?,
                    parse_string_literal(tokens[5])?,
                ))
            } else {
                Err("entangleswapmeasure <a> <b> <c> <d> <label>".into())
            }
        }
        "ENTANGLEWITHCLASSICALFEEDBACK" => {
            if tokens.len() == 4 {
                Ok(EntangleWithClassicalFeedback(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_string_literal(tokens[3])?,
                ))
            } else {
                Err("entanglewithclassicalfeedback <q1> <q2> <signal>".into())
            }
        }
        "ENTANGLEDISTRIBUTED" => {
            if tokens.len() == 3 {
                Ok(EntangleDistributed(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("entangledistributed <qubit> <node>".into())
            }
        }
        "MEASUREINBASIS" => {
            if tokens.len() == 3 {
                Ok(MeasureInBasis(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("measureinbasis <qubit> <basis>".into())
            }
        }
        "SYNC" => {
            if tokens.len() == 1 {
                Ok(Sync)
            } else {
                Err("sync".into())
            }
        }
        "RESETALL" => {
            if tokens.len() == 1 {
                Ok(ResetAll)
            } else {
                Err("resetall".into())
            }
        }
        "VERBOSELOG" => {
            if tokens.len() >= 3 {
                let q = parse_u8(tokens[1])?;
                let msg = tokens[2..].join(" ");
                Ok(VerboseLog(q, parse_string_literal(&msg)?))
            } else {
                Err("verboselog <qubit> <message>".into())
            }
        }
        "SETPHASE" => {
            if tokens.len() == 3 {
                Ok(SetPhase(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("setphase <qubit> <phase>".into())
            }
        }
        "APPLYGATE" => {
            if tokens.len() == 3 {
                Ok(ApplyGate(
                    parse_string_literal(tokens[1])?,
                    parse_u8(tokens[2])?,
                ))
            } else {
                Err("applygate <gate_name> <qubit>".into())
            }
        }
        "MEASURE" => {
            if tokens.len() == 2 {
                Ok(Measure(parse_u8(tokens[1])?))
            } else {
                Err("measure <qubit>".into())
            }
        }
        "INITQUBIT" => {
            if tokens.len() == 2 {
                Ok(InitQubit(parse_u8(tokens[1])?))
            } else {
                Err("initqubit <qubit>".into())
            }
        }

        // charprinting
        "CHARLOAD" => {
            if tokens.len() == 3 {
                let reg = parse_u8(tokens[1])?;
                let val_str = tokens[2];
                if val_str.len() == 3 && val_str.starts_with('\'') && val_str.ends_with('\'') {
                    let ch = val_str.chars().nth(1).unwrap();
                    Ok(CharLoad(reg, ch as u8))
                } else {
                    Err("usage: charload <reg> '<char>'".into())
                }
            } else {
                Err("usage: charload <reg> '<char>'".into())
            }
        }
        "CHAROUT" => {
            if tokens.len() == 2 {
                Ok(CharOut(parse_u8(tokens[1])?))
            } else {
                Err("usage: charout <reg>".into())
            }
        }
        // ionq isa
        "RX" => {
            if tokens.len() == 3 {
                Ok(RX(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("rx <qubit> <angle>".into())
            }
        }
        "RY" => {
            if tokens.len() == 3 {
                Ok(RY(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("ry <qubit> <angle>".into())
            }
        }
        "RZ" => {
            if tokens.len() == 3 {
                Ok(RZ(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("rz <qubit> <angle>".into())
            }
        }
        "PHASE" => {
            if tokens.len() == 3 {
                Ok(Phase(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("phase <qubit> <angle>".into())
            }
        }
        "CNOT" => {
            if tokens.len() == 3 {
                Ok(CNOT(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("cnot <control_qubit> <target_qubit>".into())
            }
        }
        "CZ" => {
            if tokens.len() == 3 {
                Ok(CZ(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("cz <control_qubit> <target_qubit>".into())
            }
        }
        "QRESET" => {
            if tokens.len() == 2 {
                Ok(QReset(parse_u8(tokens[1])?))
            } else {
                Err("qreset <qubit>".into())
            }
        }
        "THERMALAVG" => {
            if tokens.len() == 3 {
                Ok(ThermalAvg(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("thermalavg <qubit> <param>".into())
            }
        }
        "WKBFACTOR" => {
            if tokens.len() == 4 {
                Ok(WkbFactor(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_u8(tokens[3])?,
                ))
            } else {
                Err("wkbfactor <q1> <q2> <param>".into())
            }
        }

        // regset
        "REGSET" => {
            if tokens.len() == 3 {
                Ok(RegSet(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("usage: regset <reg> <float_value>".into())
            }
        }

        // loop
        "LOOPSTART" => {
            if tokens.len() == 2 {
                Ok(LoopStart(parse_u8(tokens[1])?))
            } else {
                Err("loopstart <reg>".into())
            }
        }
        "ENDLOOP" => {
            if tokens.len() == 1 {
                Ok(LoopEnd)
            } else {
                Err("endloop".into())
            }
        }

        // rotations
        "APPLYROTATION" => {
            if tokens.len() == 4 {
                Ok(ApplyRotation(
                    parse_u8(tokens[1])?,
                    parse_axis(tokens[2])?,
                    parse_f64(tokens[3])?,
                ))
            } else {
                Err("applyrotation <q> <x|y|z> <angle>".into())
            }
        }
        "APPLYMULTIQUBITROTATION" => {
            if tokens.len() >= 4 {
                let axis = parse_axis(tokens[2])?;
                let qs = tokens[1]
                    .split(',')
                    .map(parse_u8)
                    .collect::<Result<Vec<_>, _>>()?;
                let angles = tokens[3..]
                    .iter()
                    .map(|s| parse_f64(s))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(ApplyMultiQubitRotation(qs, axis, angles))
            } else {
                Err("applymultiqubitrotation <q1,q2,...> <x|y|z> <a1> <a2> ...".into())
            }
        }
        "CONTROLLEDPHASEROTATION" => {
            if tokens.len() == 4 {
                Ok(ControlledPhaseRotation(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_f64(tokens[3])?,
                ))
            } else {
                Err("controlledphaserotation <c> <t> <angle>".into())
            }
        }
        "APPLYC_PHASE" => {
            if tokens.len() == 4 {
                Ok(ApplyCPhase(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_f64(tokens[3])?,
                ))
            } else {
                Err("applyc_phase <q1> <q2> <angle>".into())
            }
        }
        "APPLYKERRNONLINEARITY" => {
            if tokens.len() == 4 {
                Ok(ApplyKerrNonlinearity(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                    parse_u64(tokens[3])?,
                ))
            } else {
                Err("applykerrnonlinearity <q> <strength> <duration>".into())
            }
        }
        "APPLYFEEDFORWARDGATE" => {
            if tokens.len() == 3 {
                Ok(ApplyFeedforwardGate(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("applyfeedforwardgate <q> <reg>".into())
            }
        }
        "DECOHERENCEPROTECT" => {
            if tokens.len() == 3 {
                Ok(DecoherenceProtect(
                    parse_u8(tokens[1])?,
                    parse_u64(tokens[2])?,
                ))
            } else {
                Err("decoherenceprotect <q> <duration>".into())
            }
        }
        "APPLYMEASUREMENTBASISCHANGE" => {
            if tokens.len() == 3 {
                Ok(ApplyMeasurementBasisChange(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("applymeasurementbasischange <q> <basis>".into())
            }
        }

        // memory/classical ops
        "LOAD" => {
            if tokens.len() == 3 {
                Ok(Load(parse_u8(tokens[1])?, parse_string_literal(tokens[2])?))
            } else {
                Err("load <qubit> <var>".into())
            }
        }
        "STORE" => {
            if tokens.len() == 3 {
                Ok(Store(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("store <qubit> <var>".into())
            }
        }
        "LOADMEM" => {
            if tokens.len() == 3 {
                Ok(LoadMem(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("loadmem <reg> <mem>".into())
            }
        }
        "STOREMEM" => {
            if tokens.len() == 3 {
                Ok(StoreMem(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("storemem <reg> <mem>".into())
            }
        }
        "LOADCLASSICAL" => {
            if tokens.len() == 3 {
                Ok(LoadClassical(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("loadclassical <reg> <var>".into())
            }
        }
        "STORECLASSICAL" => {
            if tokens.len() == 3 {
                Ok(StoreClassical(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("storeclassical <reg> <var>".into())
            }
        }
        "ADD" => {
            if tokens.len() == 4 {
                Ok(Add(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                    parse_string_literal(tokens[3])?,
                ))
            } else {
                Err("add <dst> <src1> <src2>".into())
            }
        }
        "SUB" => {
            if tokens.len() == 4 {
                Ok(Sub(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                    parse_string_literal(tokens[3])?,
                ))
            } else {
                Err("sub <dst> <src1> <src2>".into())
            }
        }
        "AND" => {
            if tokens.len() == 4 {
                Ok(And(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                    parse_string_literal(tokens[3])?,
                ))
            } else {
                Err("and <dst> <src1> <src2>".into())
            }
        }
        "OR" => {
            if tokens.len() == 4 {
                Ok(Or(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                    parse_string_literal(tokens[3])?,
                ))
            } else {
                Err("or <dst> <src1> <src2>".into())
            }
        }
        "XOR" => {
            if tokens.len() == 4 {
                Ok(Xor(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                    parse_string_literal(tokens[3])?,
                ))
            } else {
                Err("xor <dst> <src1> <src2>".into())
            }
        }
        "NOT" => {
            if tokens.len() == 2 {
                Ok(Not(parse_string_literal(tokens[1])?))
            } else {
                Err("not <reg>".into())
            }
        }
        "PUSH" => {
            if tokens.len() == 2 {
                Ok(Push(parse_string_literal(tokens[1])?))
            } else {
                Err("push <reg>".into())
            }
        }
        "POP" => {
            if tokens.len() == 2 {
                Ok(Pop(parse_string_literal(tokens[1])?))
            } else {
                Err("pop <reg>".into())
            }
        }

        // Classical flow control
        "JUMP" => {
            if tokens.len() == 2 {
                Ok(Jump(parse_string_literal(tokens[1])?))
            } else {
                Err("jump <label>".into())
            }
        }
        "JUMPIFZERO" => {
            if tokens.len() == 3 {
                Ok(JumpIfZero(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("jumpifzero <cond_reg> <label>".into())
            }
        }
        "JUMPIFONE" => {
            if tokens.len() == 3 {
                Ok(JumpIfOne(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("jumpifone <cond_reg> <label>".into())
            }
        }
        "CALL" => {
            if tokens.len() == 2 {
                Ok(Call(parse_string_literal(tokens[1])?))
            } else {
                Err("call <label>".into())
            }
        }
        "RETURN" => {
            if tokens.len() == 1 {
                Ok(Return)
            } else {
                Err("return".into())
            }
        }
        "BARRIER" => {
            if tokens.len() == 1 {
                Ok(Barrier)
            } else {
                Err("barrier".into())
            }
        }
        "TIMEDELAY" => {
            if tokens.len() == 3 {
                Ok(TimeDelay(parse_u8(tokens[1])?, parse_u64(tokens[2])?))
            } else {
                Err("timedelay <qubit> <cycles>".into())
            }
        }
        "RAND" => {
            if tokens.len() == 2 {
                Ok(Rand(parse_u8(tokens[1])?))
            } else {
                Err("rand <reg>".into())
            }
        }
        "SQRT" => {
            if tokens.len() == 3 {
                Ok(Sqrt(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("sqrt <rd> <rs>".into())
            }
        }
        "EXP" => {
            if tokens.len() == 3 {
                Ok(Exp(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("exp <rd> <rs>".into())
            }
        }
        "LOG" => {
            if tokens.len() == 3 {
                Ok(Log(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("log <rd> <rs>".into())
            }
        }
        // arithmetic operations
        "REGADD" => {
            if tokens.len() == 4 {
                Ok(RegAdd(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_u8(tokens[3])?,
                ))
            } else {
                Err("regadd <rd> <ra> <rb>".into())
            }
        }
        "REGSUB" => {
            if tokens.len() == 4 {
                Ok(RegSub(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_u8(tokens[3])?,
                ))
            } else {
                Err("regsub <rd> <ra> <rb>".into())
            }
        }
        "REGMUL" => {
            if tokens.len() == 4 {
                Ok(RegMul(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_u8(tokens[3])?,
                ))
            } else {
                Err("regmul <rd> <ra> <rb>".into())
            }
        }
        "REGDIV" => {
            if tokens.len() == 4 {
                Ok(RegDiv(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_u8(tokens[3])?,
                ))
            } else {
                Err("regdiv <rd> <ra> <rb>".into())
            }
        }
        "REGCOPY" => {
            if tokens.len() == 3 {
                Ok(RegCopy(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("regcopy <rd> <ra>".into())
            }
        }

        // optics
        "PHOTONEMIT" => {
            if tokens.len() == 2 {
                Ok(PhotonEmit(parse_u8(tokens[1])?))
            } else {
                Err("photonemit <qubit>".into())
            }
        }
        "PHOTONDETECT" => {
            if tokens.len() == 2 {
                Ok(PhotonDetect(parse_u8(tokens[1])?))
            } else {
                Err("photondetect <qubit>".into())
            }
        }
        "PHOTONCOUNT" => {
            if tokens.len() == 3 {
                Ok(PhotonCount(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("photoncount <qubit> <result_reg>".into())
            }
        }
        "PHOTONADDITION" => {
            if tokens.len() == 2 {
                Ok(PhotonAddition(parse_u8(tokens[1])?))
            } else {
                Err("photonaddition <qubit>".into())
            }
        }
        "APPLYPHOTONSUBTRACTION" => {
            if tokens.len() == 2 {
                Ok(ApplyPhotonSubtraction(parse_u8(tokens[1])?))
            } else {
                Err("applyphotonsubtraction <qubit>".into())
            }
        }
        "PHOTONEMISSIONPATTERN" => {
            if tokens.len() == 4 {
                Ok(PhotonEmissionPattern(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                    parse_u64(tokens[3])?,
                ))
            } else {
                Err("photonemissionpattern <qubit> <pattern_reg> <cycles>".into())
            }
        }
        "PHOTONDETECTWITHTHRESHOLD" => {
            if tokens.len() == 4 {
                Ok(PhotonDetectWithThreshold(
                    parse_u8(tokens[1])?,
                    parse_u64(tokens[2])?,
                    parse_string_literal(tokens[3])?,
                ))
            } else {
                Err("photondetectwiththreshold <qubit> <threshold> <result_reg>".into())
            }
        }
        "PHOTONDETECTCOINCIDENCE" => {
            if tokens.len() >= 3 {
                let qs = tokens[1]
                    .split(',')
                    .map(parse_u8)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(PhotonDetectCoincidence(
                    qs,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("photondetectcoincidence <q1,q2,...> <result_reg>".into())
            }
        }
        "SINGLEPHOTONSOURCEON" => {
            if tokens.len() == 2 {
                Ok(SinglePhotonSourceOn(parse_u8(tokens[1])?))
            } else {
                Err("singlephotonsourceon <qubit>".into())
            }
        }
        "SINGLEPHOTONSOURCEOFF" => {
            if tokens.len() == 2 {
                Ok(SinglePhotonSourceOff(parse_u8(tokens[1])?))
            } else {
                Err("singlephotonsourceoff <qubit>".into())
            }
        }
        "PHOTONBUNCHINGCONTROL" => {
            if tokens.len() == 3 {
                Ok(PhotonBunchingControl(
                    parse_u8(tokens[1])?,
                    _parse_bool(tokens[2])?,
                ))
            } else {
                Err("photonbunchingcontrol <qubit> <true|false>".into())
            }
        }
        "PHOTONROUTE" => {
            if tokens.len() == 4 {
                Ok(PhotonRoute(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                    parse_string_literal(tokens[3])?,
                ))
            } else {
                Err("photonroute <qubit> <from_port> <to_port>".into())
            }
        }
        "OPTICALROUTING" => {
            if tokens.len() == 3 {
                Ok(OpticalRouting(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("opticalrouting <q1> <q2>".into())
            }
        }
        "SETOPTICALATTENUATION" => {
            if tokens.len() == 3 {
                Ok(SetOpticalAttenuation(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("setopticalattenuation <qubit> <attenuation>".into())
            }
        }
        "DYNAMICPHASECOMPENSATION" => {
            if tokens.len() == 3 {
                Ok(DynamicPhaseCompensation(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("dynamicphasecompensation <qubit> <phase>".into())
            }
        }
        "OPTICALDELAYLINECONTROL" => {
            if tokens.len() == 3 {
                Ok(OpticalDelayLineControl(
                    parse_u8(tokens[1])?,
                    parse_u64(tokens[2])?,
                ))
            } else {
                Err("opticaldelaylinecontrol <qubit> <delay_cycles>".into())
            }
        }
        "CROSSPHASEMODULATION" => {
            if tokens.len() == 4 {
                Ok(CrossPhaseModulation(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_f64(tokens[3])?,
                ))
            } else {
                Err("crossphasemodulation <q1> <q2> <strength>".into())
            }
        }
        "APPLYDISPLACEMENT" => {
            if tokens.len() == 3 {
                Ok(ApplyDisplacement(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("applydisplacement <qubit> <alpha>".into())
            }
        }
        "APPLYDISPLACEMENTFEEDBACK" => {
            if tokens.len() == 3 {
                Ok(ApplyDisplacementFeedback(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("applydisplacementfeedback <qubit> <feedback_reg>".into())
            }
        }
        "APPLYDISPLACEMENTOPERATOR" => {
            if tokens.len() == 4 {
                Ok(ApplyDisplacementOperator(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                    parse_u64(tokens[3])?,
                ))
            } else {
                Err("applydisplacementoperator <qubit> <alpha> <duration>".into())
            }
        }
        "APPLYSQUEEZING" => {
            if tokens.len() == 3 {
                Ok(ApplySqueezing(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("applysqueezing <qubit> <squeezing_factor>".into())
            }
        }
        "APPLYSQUEEZINGFEEDBACK" => {
            if tokens.len() == 3 {
                Ok(ApplySqueezingFeedback(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("applysqueezingfeedback <qubit> <feedback_reg>".into())
            }
        }
        "MEASUREPARITY" => {
            if tokens.len() == 2 {
                Ok(MeasureParity(parse_u8(tokens[1])?))
            } else {
                Err("measureparity <qubit>".into())
            }
        }
        "MEASUREWITHDELAY" => {
            if tokens.len() == 4 {
                Ok(MeasureWithDelay(
                    parse_u8(tokens[1])?,
                    parse_u64(tokens[2])?,
                    parse_string_literal(tokens[3])?,
                ))
            } else {
                Err("measurewithdelay <qubit> <delay_cycles> <result_reg>".into())
            }
        }
        "OPTICALSWITCHCONTROL" => {
            if tokens.len() == 3 {
                Ok(OpticalSwitchControl(
                    parse_u8(tokens[1])?,
                    _parse_bool(tokens[2])?,
                ))
            } else {
                Err("opticalswitchcontrol <qubit> <on|off>".into())
            }
        }
        "PHOTONLOSSSIMULATE" => {
            if tokens.len() == 4 {
                Ok(PhotonLossSimulate(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                    parse_u64(tokens[3])?,
                ))
            } else {
                Err("photonlosssimulate <qubit> <loss_probability> <seed>".into())
            }
        }
        "PHOTONLOSSCORRECTION" => {
            if tokens.len() == 3 {
                Ok(PhotonLossCorrection(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("photonlosscorrection <qubit> <correction_reg>".into())
            }
        }

        // qubit measurement
        "APPLYQNDMEASUREMENT" => {
            if tokens.len() == 3 {
                Ok(ApplyQndMeasurement(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("applyqndmeasurement <qubit> <result_reg>".into())
            }
        }
        "ERRORCORRECT" => {
            if tokens.len() == 3 {
                Ok(ErrorCorrect(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("errorcorrect <qubit> <syndrome_type>".into())
            }
        }
        "ERRORSYNDROME" => {
            if tokens.len() == 4 {
                Ok(ErrorSyndrome(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                    parse_string_literal(tokens[3])?,
                ))
            } else {
                Err("errorsyndrome <qubit> <syndrome_type> <result_reg>".into())
            }
        }
        "QUANTUMSTATETOMOGRAPHY" => {
            if tokens.len() == 3 {
                Ok(QuantumStateTomography(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("quantumstatetomography <qubit> <basis>".into())
            }
        }
        "BELLSTATEVERIFICATION" => {
            if tokens.len() == 4 {
                Ok(BellStateVerification(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_string_literal(tokens[3])?,
                ))
            } else {
                Err("bellstateverification <q1> <q2> <result_reg>".into())
            }
        }
        "QUANTUMZENOEFFECT" => {
            if tokens.len() == 4 {
                Ok(QuantumZenoEffect(
                    parse_u8(tokens[1])?,
                    parse_u64(tokens[2])?,
                    parse_u64(tokens[3])?,
                ))
            } else {
                Err("quantumzenoeffect <qubit> <num_measurements> <interval_cycles>".into())
            }
        }
        "APPLYNONLINEARPHASESHIFT" => {
            if tokens.len() == 3 {
                Ok(ApplyNonlinearPhaseShift(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("applynonlinearphaseshift <qubit> <strength>".into())
            }
        }
        "APPLYNONLINEARSiGMA" => {
            if tokens.len() == 3 {
                Ok(ApplyNonlinearSigma(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("applynonlinearsigma <qubit> <strength>".into())
            }
        }
        "APPLYLINEAROPTICALTRANSFORM" => {
            if tokens.len() == 5 {
                let transform_name = parse_string_literal(tokens[1])?;
                let input_qubits_str = tokens[2];
                let output_qubits_str = tokens[3];
                let mode_count = parse_u8(tokens[4])? as usize;

                let input_qubits = input_qubits_str
                    .split(',')
                    .map(parse_u8)
                    .collect::<Result<Vec<_>, _>>()?;
                let output_qubits = output_qubits_str
                    .split(',')
                    .map(parse_u8)
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(ApplyLinearOpticalTransform(
                    transform_name,
                    input_qubits,
                    output_qubits,
                    mode_count,
                ))
            } else {
                Err("usage: applylinearopticaltransform <name> <in_q1,in_q2...> <out_q1,out_q2...> <num_modes>".into())
            }
        }
        "PHOTONNUMBERRESOLVINGDETECTION" => {
            if tokens.len() == 3 {
                Ok(PhotonNumberResolvingDetection(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("photonnumberresolvingdetection <qubit> <result_reg>".into())
            }
        }
        "FEEDBACKCONTROL" => {
            if tokens.len() == 3 {
                Ok(FeedbackControl(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("feedbackcontrol <qubit> <classical_control_reg>".into())
            }
        }
        // misc
        "SETPOS" => {
            if tokens.len() == 4 {
                Ok(SetPos(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                    parse_f64(tokens[3])?,
                ))
            } else {
                Err("setpos <reg> <x> <y>".into())
            }
        }
        "SETWL" => {
            if tokens.len() == 3 {
                Ok(SetWl(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("setwl <reg> <wavelength>".into())
            }
        }
        "WLSHIFT" => {
            if tokens.len() == 3 {
                Ok(WlShift(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("wlshift <reg> <delta_wavelength>".into())
            }
        }
        "MOVE" => {
            if tokens.len() == 4 {
                Ok(Move(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                    parse_f64(tokens[3])?,
                ))
            } else {
                Err("move <reg> <dx> <dy>".into())
            }
        }
        "COMMENT" => {
            if tokens.len() >= 2 {
                Ok(Comment(tokens[1..].join(" ")))
            } else {
                Err("comment <text>".into())
            }
        }
        "MARKOBSERVED" => {
            if tokens.len() == 2 {
                Ok(MarkObserved(parse_u8(tokens[1])?))
            } else {
                Err("markobserved <reg>".into())
            }
        }
        "RELEASE" => {
            if tokens.len() == 2 {
                Ok(Release(parse_u8(tokens[1])?))
            } else {
                Err("release <reg>".into())
            }
        }
        "HALT" => {
            if tokens.len() == 1 {
                Ok(Halt)
            } else {
                Err("halt".into())
            }
        }
        // new instruction parsing for v0.3.0
        "JMP" => Ok(Jmp(parse_i64(tokens[1])?)),
        "JMPABS" => Ok(JmpAbs(parse_u64(tokens[1])?)),
        "IFGT" => Ok(IfGt(
            parse_u8(tokens[1])?,
            parse_u8(tokens[2])?,
            parse_i64(tokens[3])?,
        )),
        "IFLT" => Ok(IfLt(
            parse_u8(tokens[1])?,
            parse_u8(tokens[2])?,
            parse_i64(tokens[3])?,
        )),
        "IFEQ" => Ok(IfEq(
            parse_u8(tokens[1])?,
            parse_u8(tokens[2])?,
            parse_i64(tokens[3])?,
        )),
        "IFNE" => Ok(IfNe(
            parse_u8(tokens[1])?,
            parse_u8(tokens[2])?,
            parse_i64(tokens[3])?,
        )),
        "LABEL" => Err(
            "LABEL is a pseudo-instruction for jump targets, not for direct execution.".to_string(),
        ),
        "CALL_ADDR" => Ok(CallAddr(parse_u64(tokens[1])?)),
        "RET_SUB" => Ok(RetSub),
        "PRINTF" => {
            let format_str = parse_string_literal(tokens[1])?;
            let regs = parse_reg_list(&tokens[2..])?;
            Ok(Printf(format_str, regs))
        }
        "PRINT" => Ok(Print(parse_string_literal(tokens[1])?)),
        "PRINTLN" => Ok(Println(parse_string_literal(tokens[1])?)),
        "INPUT" => Ok(Input(parse_u8(tokens[1])?)),
        "DUMP_STATE" => Ok(DumpState),
        "DUMP_REGS" => Ok(DumpRegs),
        "LOAD_REG_MEM" => Ok(LoadRegMem(parse_u8(tokens[1])?, parse_u64(tokens[2])?)),
        "STORE_MEM_REG" => Ok(StoreMemReg(parse_u64(tokens[1])?, parse_u8(tokens[2])?)),
        "PUSH_REG" => Ok(PushReg(parse_u8(tokens[1])?)),
        "POP_REG" => Ok(PopReg(parse_u8(tokens[1])?)),
        "ALLOC" => Ok(Alloc(parse_u8(tokens[1])?, parse_u64(tokens[2])?)),
        "FREE" => Ok(Free(parse_u64(tokens[1])?)),
        "CMP" => Ok(Cmp(parse_u8(tokens[1])?, parse_u8(tokens[2])?)),
        "AND_BITS" => Ok(AndBits(
            parse_u8(tokens[1])?,
            parse_u8(tokens[2])?,
            parse_u8(tokens[3])?,
        )),
        "OR_BITS" => Ok(OrBits(
            parse_u8(tokens[1])?,
            parse_u8(tokens[2])?,
            parse_u8(tokens[3])?,
        )),
        "XOR_BITS" => Ok(XorBits(
            parse_u8(tokens[1])?,
            parse_u8(tokens[2])?,
            parse_u8(tokens[3])?,
        )),
        "NOT_BITS" => Ok(NotBits(parse_u8(tokens[1])?, parse_u8(tokens[2])?)),
        "SHL" => Ok(Shl(
            parse_u8(tokens[1])?,
            parse_u8(tokens[2])?,
            parse_u8(tokens[3])?,
        )),
        "SHR" => Ok(Shr(
            parse_u8(tokens[1])?,
            parse_u8(tokens[2])?,
            parse_u8(tokens[3])?,
        )),
        "BREAK_POINT" => Ok(BreakPoint),
        "GET_TIME" => Ok(GetTime(parse_u8(tokens[1])?)),
        "SEED_RNG" => Ok(SeedRng(parse_u64(tokens[1])?)),
        "EXIT_CODE" => Ok(ExitCode(parse_i64(tokens[1])? as i32)),
        _ => Err(format!("unknown instruction '{}'", op)),
    }
}
