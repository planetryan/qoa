// this file defines the instruction set architecture (ISA) for the quantum optical assembly (QOA)
// it includes the instruction enum, parsing logic, and byte serialization.

// defines the different types of instructions available in qoa.
// each variant represents a specific operation the qoa vm can perform.
#[derive(Debug, PartialEq, Clone)]
pub enum Instruction {
    // core
    QINIT(u8),
    QMEAS(u8),
    H(u8), // shortened from ApplyHadamard
    APPLYHADAMARD(u8), // explicit ApplyHadamard instruction
    CONTROLLEDNOT(u8, u8),
    APPLYPHASEFLIP(u8),
    APPLYBITFLIP(u8),
    APPLYTGATE(u8),
    APPLYSGATE(u8),
    PHASESHIFT(u8, f64),
    WAIT(u64),
    RESET(u8),
    SWAP(u8, u8),
    CONTROLLEDSWAP(u8, u8, u8),
    ENTANGLE(u8, u8),
    ENTANGLEBELL(u8, u8),
    ENTANGLEMULTI(Vec<u8>),
    ENTANGLECLUSTER(Vec<u8>),
    ENTANGLESWAP(u8, u8, u8, u8),
    ENTANGLESWAPMEASURE(u8, u8, u8, u8, String),
    ENTANGLEWITHCLASSICALFEEDBACK(u8, u8, String),
    ENTANGLEDISTRIBUTED(u8, String),
    MEASUREINBASIS(u8, String),
    SYNC,
    RESETALL,
    VERBOSELOG(u8, String),
    SETPHASE(u8, f64),
    APPLYGATE(String, u8),
    MEASURE(u8),   // duplicate of QMEAS, but keeping for now
    INITQUBIT(u8), // duplicate of QINIT, but keeping for now
    LABEL(String), // represents a named label for jump targets (nop-instruction)

    // charprinting
    CHARLOAD(u8, u8),
    CHAROUT(u8),

    // ionq isa
    RX(u8, f64),
    RY(u8, f64),
    RZ(u8, f64),
    PHASE(u8, f64),
    CNOT(u8, u8),
    CZ(u8, u8),
    QRESET(u8),
    THERMALAVG(u8, u8),
    WKBFACTOR(u8, u8, u8),

    // regset
    REGSET(u8, f64),

    // loop
    LOOPSTART(u8),
    LOOPEND,

    // rotations
    APPLYROTATION(u8, char, f64),
    APPLYMULTIQUBITROTATION(Vec<u8>, char, Vec<f64>),
    CONTROLLEDPHASEROTATION(u8, u8, f64),
    APPLYCPHASE(u8, u8, f64),
    APPLYKERRNONLINEARITY(u8, f64, u64),
    APPLYFEEDFORWARDGATE(u8, String),
    DECOHERENCEPROTECT(u8, u64),
    APPLYMEASUREMENTBASISCHANGE(u8, String),

    // memory/classical ops
    LOAD(u8, String),
    STORE(u8, String),
    LOADMEM(String, String),
    STOREMEM(String, String),
    LOADCLASSICAL(String, String),
    STORECLASSICAL(String, String),
    ADD(String, String, String),
    SUB(String, String, String),
    AND(String, String, String),
    OR(String, String, String),
    XOR(String, String, String),
    NOT(String),
    PUSH(String),
    POP(String),
    // classical
    JUMP(String), // original, string-based jump (will be resolved to JMPABS/JMP)
    JUMPIFZERO(String, String), // original, string-based conditional jump
    JUMPIFONE(String, String),  // original, string-based conditional jump
    CALL(String), // original, string-based call
    BARRIER,
    RETURN,
    TIMEDELAY(u8, u64),
    RAND(u8),
    SQRT(u8, u8),
    EXP(u8, u8),
    LOG(u8, u8),
    // arithmetic operations
    REGADD(u8, u8, u8), // rd, ra, rb
    REGSUB(u8, u8, u8),
    REGMUL(u8, u8, u8),
    REGDIV(u8, u8, u8),
    REGCOPY(u8, u8), // rd, ra

    // optics
    PHOTONEMIT(u8),
    PHOTONDETECT(u8),
    PHOTONCOUNT(u8, String),
    PHOTONADDITION(u8),
    APPLYPHOTONSUBTRACTION(u8),
    PHOTONEMISSIONPATTERN(u8, String, u64),
    PHOTONDETECTWITHTHRESHOLD(u8, u64, String),
    PHOTONDETECTCOINCIDENCE(Vec<u8>, String),
    SINGLEPHOTONSOURCEON(u8),
    SINGLEPHOTONSOURCEOFF(u8),
    PHOTONBUNCHINGCONTROL(u8, bool),
    PHOTONROUTE(u8, String, String),
    OPTICALROUTING(u8, u8),
    SETOPTICALATTENUATION(u8, f64),
    DYNAMICPHASECOMPENSATION(u8, f64),
    OPTICALDELAYLINECONTROL(u8, u64),
    CROSSPHASEMODULATION(u8, u8, f64),
    APPLYDISPLACEMENT(u8, f64),
    APPLYDISPLACEMENTFEEDBACK(u8, String),
    APPLYDISPLACEMENTOPERATOR(u8, f64, u64),
    APPLYSQUEEZING(u8, f64),
    APPLYSQUEEZINGFEEDBACK(u8, String),
    MEASUREPARITY(u8),
    MEASUREWITHDELAY(u8, u64, String),
    OPTICALSWITCHCONTROL(u8, bool),
    PHOTONLOSSSIMULATE(u8, f64, u64),
    PHOTONLOSSCORRECTION(u8, String),

    // qubit measurement
    APPLYQNDMEASUREMENT(u8, String),
    ERRORCORRECT(u8, String),
    ERRORSYNDROME(u8, String, String),
    QUANTUMSTATETOMOGRAPHY(u8, String),
    BELLSTATEVERIFICATION(u8, u8, String),
    QUANTUMZENOEFFECT(u8, u64, u64),
    APPLYNONLINEARPHASESHIFT(u8, f64),
    APPLYNONLINEARSiGMA(u8, f64),
    APPLYLINEAROPTICALTRANSFORM(String, Vec<u8>, Vec<u8>, usize),
    PHOTONNUMBERRESOLVINGDETECTION(u8, String),
    FEEDBACKCONTROL(u8, String),

    // misc
    SETPOS(u8, f64, f64),
    SETWL(u8, f64),
    WLSHIFT(u8, f64),
    MOVE(u8, f64, f64),
    COMMENT(String),
    MARKOBSERVED(u8),
    RELEASE(u8),
    HALT,

    // new instructions for v0.3.0 development todo

    // control flow & program structure
    JMP(i64),          // jump relative by offset (i64 for signed offset)
    JMPABS(u64),       // jump absolute to instruction index
    IFGT(u8, u8, i64), // if reg1 > reg2, jump relative by offset
    IFLT(u8, u8, i64), // if reg1 < reg2, jump relative by offset
    IFEQ(u8, u8, i64), // if reg1 == reg2, jump relative by offset
    IFNE(u8, u8, i64), // if reg1 != reg2, jump relative by offset
    CALLADDR(u64), // call subroutine at absolute address, push return address to stack
    RETSUB, // return from subroutine, pop return address from stack

    // input/output & debugging
    PRINTF(String, Vec<u8>), // c-style formatted output (format string, register indices)
    PRINT(String),           // print a string literal
    PRINTLN(String),         // print a string literal with newline
    INPUT(u8),               // read floating-point value from stdin into register
    DUMPSTATE,               // output quantum amplitudes and phases
    DUMPREGS,                // output all register values

    // memory & stack
    LOADREGMEM(u8, u64), // load value from memory address into register (reg, mem_addr)
    STOREMEMREG(u64, u8), // store value from register into memory address (mem_addr, reg)
    PUSHREG(u8), // push register value onto stack
    POPREG(u8),  // pop value from stack into register
    ALLOC(u8, u64), // allocate memory, store start address in register (reg_addr, size)
    FREE(u64),   // free memory at address

    // comparison & bitwise logic
    CMP(u8, u8), // compare reg1 and reg2, set internal flags (flags handled by runtime)
    ANDBITS(u8, u8, u8), // bitwise and (dest, op1, op2)
    ORBITS(u8, u8, u8),  // bitwise or (dest, op1, op2)
    XORBITS(u8, u8, u8), // bitwise xor (dest, op1, op2)
    NOTBITS(u8, u8),     // bitwise not (dest, op)
    SHL(u8, u8, u8),     // shift left (dest, op, amount_reg)
    SHR(u8, u8, u8),     // shift right (dest, op, amount_reg)

    // system & debug utilities
    BREAKPOINT,  // breakpoint instruction
    GETTIME(u8), // get system timestamp into register
    SEEDRNG(u64), // seed rng for reproducible results
    EXITCODE(i32), // terminate program with exit code
}

// opcodes for each instruction. these are used for serialization/deserialization
// and are crucial for the vm to understand the bytecode.
impl Instruction {
    pub fn encode(&self) -> Vec<u8> {
        match self {
            // core
            Instruction::QINIT(q) => vec![0x04, *q],
            Instruction::QMEAS(q) => vec![0x32, *q],
            Instruction::H(q) => vec![0x05, *q],
            Instruction::APPLYHADAMARD(q) => vec![0x05, *q], // same opcode as H
            Instruction::CONTROLLEDNOT(c, t) => vec![0x17, *c, *t],
            Instruction::APPLYPHASEFLIP(q) => vec![0x06, *q],
            Instruction::APPLYBITFLIP(q) => vec![0x07, *q],
            Instruction::APPLYTGATE(q) => vec![0x0D, *q],
            Instruction::APPLYSGATE(q) => vec![0x0E, *q],
            Instruction::PHASESHIFT(q, angle) => {
                let mut v = vec![0x08, *q];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::WAIT(cycles) => {
                let mut v = vec![0x09];
                v.extend(&cycles.to_le_bytes());
                v
            }
            Instruction::RESET(q) => vec![0x0A, *q],
            Instruction::SWAP(q1, q2) => vec![0x0B, *q1, *q2],
            Instruction::CONTROLLEDSWAP(c, t1, t2) => vec![0x0C, *c, *t1, *t2],
            Instruction::ENTANGLE(q1, q2) => vec![0x11, *q1, *q2],
            Instruction::ENTANGLEBELL(q1, q2) => vec![0x12, *q1, *q2],
            Instruction::ENTANGLEMULTI(qs) => {
                let mut v = vec![0x13, qs.len() as u8];
                v.extend(qs.iter());
                v
            }
            Instruction::ENTANGLECLUSTER(qs) => {
                let mut v = vec![0x14, qs.len() as u8];
                v.extend(qs.iter());
                v
            }
            Instruction::ENTANGLESWAP(a, b, c, d) => vec![0x15, *a, *b, *c, *d],
            Instruction::ENTANGLESWAPMEASURE(a, b, c, d, label) => {
                let mut v = vec![0x16, *a, *b, *c, *d];
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::ENTANGLEWITHCLASSICALFEEDBACK(q1, q2, signal) => {
                let mut v = vec![0x19, *q1, *q2];
                v.extend(signal.as_bytes());
                v.push(0);
                v
            }
            Instruction::ENTANGLEDISTRIBUTED(q, node) => {
                let mut v = vec![0x1A, *q];
                v.extend(node.as_bytes());
                v.push(0);
                v
            }
            Instruction::MEASUREINBASIS(q, basis) => {
                let mut v = vec![0x1B, *q];
                v.extend(basis.as_bytes());
                v.push(0);
                v
            }
            Instruction::SYNC => vec![0x48],
            Instruction::RESETALL => vec![0x1C],
            Instruction::VERBOSELOG(q, msg) => {
                let mut v = vec![0x87, *q];
                v.extend(msg.as_bytes());
                v.push(0);
                v
            }
            Instruction::SETPHASE(q, phase) => {
                let mut v = vec![0x1D, *q];
                v.extend(&phase.to_le_bytes());
                v
            }
            Instruction::APPLYGATE(name, q) => {
                let mut v = vec![0x02, *q];
                v.extend(name.as_bytes());
                for _ in name.len()..8 {
                    v.push(0);
                }
                v
            }
            Instruction::MEASURE(q) => vec![0x32, *q],
            Instruction::INITQUBIT(q) => vec![0x04, *q],
            Instruction::LABEL(_) => vec![], // LABEL is a nop-instruction, no bytecode

            // char printing
            Instruction::CHARLOAD(reg, val) => vec![0x31, *reg, *val],
            Instruction::CHAROUT(reg) => vec![0x18, *reg],

            // ionq isa gates
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
            Instruction::PHASE(q, angle) => {
                let mut v = vec![0x24, *q];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::CNOT(c, t) => vec![0x17, *c, *t],
            Instruction::CZ(c, t) => vec![0x1E, *c, *t],
            Instruction::QRESET(q) => vec![0x0A, *q],
            Instruction::THERMALAVG(q, param) => vec![0x1F, *q, *param],
            Instruction::WKBFACTOR(q1, q2, param) => vec![0x20, *q1, *q2, *param],

            // regset
            Instruction::REGSET(reg, val) => {
                let mut v = vec![0x21, *reg];
                v.extend(&val.to_le_bytes());
                v
            }

            // loop
            Instruction::LOOPSTART(reg) => vec![0x01, *reg],
            Instruction::LOOPEND => vec![0x10],

            // rotations
            Instruction::APPLYROTATION(q, axis, angle) => {
                let mut v = vec![0x33, *q, *axis as u8];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::APPLYMULTIQUBITROTATION(qs, axis, angles) => {
                let mut v = vec![0x34, *axis as u8, qs.len() as u8];
                v.extend(qs.iter());
                for a in angles {
                    v.extend(&a.to_le_bytes());
                }
                v
            }
            Instruction::CONTROLLEDPHASEROTATION(c, t, angle) => {
                let mut v = vec![0x35, *c, *t];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::APPLYCPHASE(q1, q2, angle) => {
                let mut v = vec![0x36, *q1, *q2];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::APPLYKERRNONLINEARITY(q, strength, duration) => {
                let mut v = vec![0x37, *q];
                v.extend(&strength.to_le_bytes());
                v.extend(&duration.to_le_bytes());
                v
            }
            Instruction::APPLYFEEDFORWARDGATE(q, reg) => {
                let mut v = vec![0x38, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::DECOHERENCEPROTECT(q, duration) => {
                let mut v = vec![0x39, *q];
                v.extend(&duration.to_le_bytes());
                v
            }
            Instruction::APPLYMEASUREMENTBASISCHANGE(q, basis) => {
                let mut v = vec![0x3A, *q];
                v.extend(basis.as_bytes());
                v.push(0);
                v
            }

            // memory/classical ops (originals)
            Instruction::LOAD(q, reg) => {
                let mut v = vec![0x3B, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::STORE(q, reg) => {
                let mut v = vec![0x3C, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::LOADMEM(reg, addr) => {
                let mut v = vec![0x3D];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(addr.as_bytes());
                v.push(0);
                v
            }
            Instruction::STOREMEM(reg, addr) => {
                let mut v = vec![0x3E];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(addr.as_bytes());
                v.push(0);
                v
            }
            Instruction::LOADCLASSICAL(reg, var) => {
                let mut v = vec![0x3F];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(var.as_bytes());
                v.push(0);
                v
            }
            Instruction::STORECLASSICAL(reg, var) => {
                let mut v = vec![0x40];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(var.as_bytes());
                v.push(0);
                v
            }
            Instruction::ADD(dst, src1, src2) => {
                let mut v = vec![0x41];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::SUB(dst, src1, src2) => {
                let mut v = vec![0x42];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::AND(dst, src1, src2) => {
                let mut v = vec![0x43];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::OR(dst, src1, src2) => {
                let mut v = vec![0x44];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::XOR(dst, src1, src2) => {
                let mut v = vec![0x45];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::NOT(reg) => {
                let mut v = vec![0x46];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::PUSH(reg) => {
                let mut v = vec![0x47];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::POP(reg) => {
                let mut v = vec![0x4F];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }

            // classical flow control (originals)
            Instruction::JUMP(label) => {
                let mut v = vec![0x49];
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::JUMPIFZERO(cond, label) => {
                let mut v = vec![0x4A];
                v.extend(cond.as_bytes());
                v.push(0);
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::JUMPIFONE(cond, label) => {
                let mut v = vec![0x4B];
                v.extend(cond.as_bytes());
                v.push(0);
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::CALL(label) => {
                let mut v = vec![0x4C];
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::RETURN => vec![0x4D],
            Instruction::TIMEDELAY(q, cycles) => {
                let mut v = vec![0x4E, *q];
                v.extend(&cycles.to_le_bytes());
                v
            }
            Instruction::RAND(reg) => vec![0x50, *reg],
            Instruction::SQRT(rd, rs) => vec![0x51, *rd, *rs],
            Instruction::EXP(rd, rs) => vec![0x52, *rd, *rs],
            Instruction::LOG(rd, rs) => vec![0x53, *rd, *rs],
            Instruction::REGADD(rd, ra, rb) => vec![0x54, *rd, *ra, *rb],
            Instruction::REGSUB(rd, ra, rb) => vec![0x55, *rd, *ra, *rb],
            Instruction::REGMUL(rd, ra, rb) => vec![0x56, *rd, *ra, *rb],
            Instruction::REGDIV(rd, ra, rb) => vec![0x57, *rd, *ra, *rb],
            Instruction::REGCOPY(rd, ra) => vec![0x58, *rd, *ra],

            // optics / photonics
            Instruction::PHOTONEMIT(q) => vec![0x59, *q],
            Instruction::PHOTONDETECT(q) => vec![0x5A, *q],
            Instruction::PHOTONCOUNT(q, reg) => {
                let mut v = vec![0x5B, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::PHOTONADDITION(q) => vec![0x5C, *q],
            Instruction::APPLYPHOTONSUBTRACTION(q) => vec![0x5D, *q],
            Instruction::PHOTONEMISSIONPATTERN(q, reg, cycles) => {
                let mut v = vec![0x5E, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(&cycles.to_le_bytes());
                v
            }
            Instruction::PHOTONDETECTWITHTHRESHOLD(q, thresh, reg) => {
                let mut v = vec![0x5F, *q];
                v.extend(&thresh.to_le_bytes());
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::PHOTONDETECTCOINCIDENCE(qs, reg) => {
                let mut v = vec![0x60, qs.len() as u8];
                v.extend(qs.iter());
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::SINGLEPHOTONSOURCEON(q) => vec![0x61, *q],
            Instruction::SINGLEPHOTONSOURCEOFF(q) => vec![0x62, *q],
            Instruction::PHOTONBUNCHINGCONTROL(q, b) => vec![0x63, *q, *b as u8],
            Instruction::PHOTONROUTE(q, from, to) => {
                let mut v = vec![0x64, *q];
                v.extend(from.as_bytes());
                v.push(0);
                v.extend(to.as_bytes());
                v.push(0);
                v
            }
            Instruction::OPTICALROUTING(q1, q2) => vec![0x65, *q1, *q2],
            Instruction::SETOPTICALATTENUATION(q, att) => {
                let mut v = vec![0x66, *q];
                v.extend(&att.to_le_bytes());
                v
            }
            Instruction::DYNAMICPHASECOMPENSATION(q, phase) => {
                let mut v = vec![0x67, *q];
                v.extend(&phase.to_le_bytes());
                v
            }
            Instruction::OPTICALDELAYLINECONTROL(q, delay) => {
                let mut v = vec![0x68, *q];
                v.extend(&delay.to_le_bytes());
                v
            }
            Instruction::CROSSPHASEMODULATION(c, t, stren) => {
                let mut v = vec![0x69, *c, *t];
                v.extend(&stren.to_le_bytes());
                v
            }
            Instruction::APPLYDISPLACEMENT(q, a) => {
                let mut v = vec![0x6A, *q];
                v.extend(&a.to_le_bytes());
                v
            }
            Instruction::APPLYDISPLACEMENTFEEDBACK(q, reg) => {
                let mut v = vec![0x6B, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::APPLYDISPLACEMENTOPERATOR(q, alpha, dur) => {
                let mut v = vec![0x6C, *q];
                v.extend(&alpha.to_le_bytes());
                v.extend(&dur.to_le_bytes());
                v
            }
            Instruction::APPLYSQUEEZING(q, s) => {
                let mut v = vec![0x6D, *q];
                v.extend(&s.to_le_bytes());
                v
            }
            Instruction::APPLYSQUEEZINGFEEDBACK(q, reg) => {
                let mut v = vec![0x6E, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::MEASUREPARITY(q) => vec![0x6F, *q],
            Instruction::MEASUREWITHDELAY(q, delay, reg) => {
                let mut v = vec![0x70, *q];
                v.extend(&delay.to_le_bytes());
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::OPTICALSWITCHCONTROL(q, b) => vec![0x71, *q, *b as u8],
            Instruction::PHOTONLOSSSIMULATE(q, prob, seed) => {
                let mut v = vec![0x72, *q];
                v.extend(&prob.to_le_bytes());
                v.extend(&seed.to_le_bytes());
                v
            }
            Instruction::PHOTONLOSSCORRECTION(q, reg) => {
                let mut v = vec![0x73, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }

            // misc (originals)
            Instruction::SETPOS(q, x, y) => {
                let mut v = vec![0x74, *q];
                v.extend(&x.to_le_bytes());
                v.extend(&y.to_le_bytes());
                v
            }
            Instruction::SETWL(q, wl) => {
                let mut v = vec![0x75, *q];
                v.extend(&wl.to_le_bytes());
                v
            }
            Instruction::WLSHIFT(q, wl_delta) => {
                let mut v = vec![0x76, *q];
                v.extend(&wl_delta.to_le_bytes());
                v
            }
            Instruction::MOVE(q, dx, dy) => {
                let mut v = vec![0x77, *q];
                v.extend(&dx.to_le_bytes());
                v.extend(&dy.to_le_bytes());
                v
            }
            Instruction::COMMENT(text) => {
                let mut v = vec![0x88];
                v.extend(text.as_bytes());
                v.push(0);
                v
            }
            Instruction::MARKOBSERVED(q) => vec![0x79, *q],
            Instruction::RELEASE(q) => vec![0x7A, *q],
            Instruction::HALT => vec![0xFF],

            Instruction::BARRIER => vec![0x89],
            Instruction::APPLYQNDMEASUREMENT(q, reg) => {
                let mut v = vec![0x7C, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::ERRORCORRECT(q, syndrome_type) => {
                let mut v = vec![0x7D, *q];
                v.extend(syndrome_type.as_bytes());
                v.push(0);
                v
            }
            Instruction::ERRORSYNDROME(q, syndrome_type, result_reg) => {
                let mut v = vec![0x7E, *q];
                v.extend(syndrome_type.as_bytes());
                v.push(0);
                v.extend(result_reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::QUANTUMSTATETOMOGRAPHY(q, basis) => {
                let mut v = vec![0x7F, *q];
                v.extend(basis.as_bytes());
                v.push(0);
                v
            }
            Instruction::BELLSTATEVERIFICATION(q1, q2, result_reg) => {
                let mut v = vec![0x80, *q1, *q2];
                v.extend(result_reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::QUANTUMZENOEFFECT(q, num_measurements, interval_cycles) => {
                let mut v = vec![0x81, *q];
                v.extend(&num_measurements.to_le_bytes());
                v.extend(&interval_cycles.to_le_bytes());
                v
            }
            Instruction::APPLYNONLINEARPHASESHIFT(q, strength) => {
                let mut v = vec![0x82, *q];
                v.extend(&strength.to_le_bytes());
                v
            }
            Instruction::APPLYNONLINEARSiGMA(q, strength) => {
                let mut v = vec![0x83, *q];
                v.extend(&strength.to_le_bytes());
                v
            }
            Instruction::APPLYLINEAROPTICALTRANSFORM(_name, input_qs, output_qs, _num_modes) => {
                let mut v = vec![
                    0x84,
                    input_qs.len() as u8,
                    output_qs.len() as u8,
                    *_num_modes as u8,
                ];
                v.extend(input_qs.iter());
                v.extend(output_qs.iter());
                v
            }
            Instruction::PHOTONNUMBERRESOLVINGDETECTION(q, reg) => {
                let mut v = vec![0x85, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::FEEDBACKCONTROL(q, reg) => {
                let mut v = vec![0x86, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }

            // new instruction serialization for v0.3.0
            Instruction::JMP(offset) => {
                let mut v = vec![0x90];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::JMPABS(addr) => {
                let mut v = vec![0x91]; // explicit opcode for JMPABS
                v.extend(&addr.to_le_bytes());
                v
            }
            Instruction::IFGT(r1, r2, offset) => {
                let mut v = vec![0x92, *r1, *r2];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::IFLT(r1, r2, offset) => {
                let mut v = vec![0x93, *r1, *r2];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::IFEQ(r1, r2, offset) => {
                let mut v = vec![0x94, *r1, *r2];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::IFNE(r1, r2, offset) => {
                let mut v = vec![0x95, *r1, *r2];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::CALLADDR(addr) => {
                let mut v = vec![0x96];
                v.extend(&addr.to_le_bytes());
                v
            }
            Instruction::RETSUB => {
                vec![0x97]
            }
            Instruction::PRINTF(format_str, regs) => {
                let mut v = vec![0x98];
                v.extend(&(format_str.len() as u64).to_le_bytes()); // length of string
                v.extend(format_str.as_bytes()); // format string
                v.push(regs.len() as u8); // number of registers
                v.extend(regs); // register indices
                v
            }
            Instruction::PRINT(s) => {
                let mut v = vec![0x99];
                v.extend(&(s.len() as u64).to_le_bytes()); // length of string
                v.extend(s.as_bytes()); // the string itself
                v
            }
            Instruction::PRINTLN(s) => {
                let mut v = vec![0x9A];
                v.extend(&(s.len() as u64).to_le_bytes()); // length of string
                v.extend(s.as_bytes()); // the string itself
                v
            }
            Instruction::INPUT(q) => {
                vec![0x9B, *q]
            }
            Instruction::DUMPSTATE => {
                vec![0x9C]
            }
            Instruction::DUMPREGS => {
                vec![0x9D]
            }
            Instruction::LOADREGMEM(reg, addr) => {
                let mut v = vec![0x9E, *reg];
                v.extend(&addr.to_le_bytes());
                v
            }
            Instruction::STOREMEMREG(addr, reg) => {
                let mut v = vec![0x9F];
                v.extend(&addr.to_le_bytes());
                v.push(*reg);
                v
            }
            Instruction::PUSHREG(q) => {
                vec![0xA0, *q]
            }
            Instruction::POPREG(q) => {
                vec![0xA1, *q]
            }
            Instruction::ALLOC(reg_addr, size) => {
                let mut v = vec![0xA2, *reg_addr];
                v.extend(&size.to_le_bytes());
                v
            }
            Instruction::FREE(addr) => {
                let mut v = vec![0xA3];
                v.extend(&addr.to_le_bytes());
                v
            }
            Instruction::CMP(r1, r2) => {
                vec![0xA4, *r1, *r2]
            }
            Instruction::ANDBITS(d, o1, o2) => {
                vec![0xA5, *d, *o1, *o2]
            }
            Instruction::ORBITS(d, o1, o2) => {
                vec![0xA6, *d, *o1, *o2]
            }
            Instruction::XORBITS(d, o1, o2) => {
                vec![0xA7, *d, *o1, *o2]
            }
            Instruction::NOTBITS(d, o) => {
                vec![0xA8, *d, *o]
            }
            Instruction::SHL(d, o1, o2) => {
                vec![0xA9, *d, *o1, *o2]
            }
            Instruction::SHR(d, o1, o2) => {
                vec![0xAA, *d, *o1, *o2]
            }
            Instruction::BREAKPOINT => {
                vec![0xAB]
            }
            Instruction::GETTIME(q) => {
                vec![0xAC, *q]
            }
            Instruction::SEEDRNG(s) => {
                let mut v = vec![0xAD];
                v.extend(&s.to_le_bytes());
                v
            }
            Instruction::EXITCODE(code) => {
                let mut v = vec![0xAE];
                v.extend(&code.to_le_bytes());
                v
            }
        }
    }

    // helper function to get the byte size of an instruction.
    // this method is currently not used in the provided code, but is kept for potential future use.
    #[allow(dead_code)] // added to suppress the unused method warning
    pub fn get_byte_size(&self) -> usize {
        match self {
            // nop-instructions, no bytecode
            Instruction::LABEL(_) => 0,
            Instruction::COMMENT(text) => 1 + text.len() + 1, // opcode + string length + null terminator

            // 1-byte opcodes
            Instruction::SYNC => 1,
            Instruction::LOOPEND => 1,
            Instruction::RESETALL => 1,
            Instruction::HALT => 1,
            Instruction::RETSUB => 1,
            Instruction::BARRIER => 1,
            Instruction::DUMPSTATE => 1,
            Instruction::DUMPREGS => 1,
            Instruction::BREAKPOINT => 1,
            Instruction::RETURN => 1, // original return

            // 2-byte opcodes (opcode + u8)
            Instruction::QINIT(_) => 2,
            Instruction::QMEAS(_) => 2,
            Instruction::H(_) => 2,
            Instruction::APPLYHADAMARD(_) => 2, // same size as H
            Instruction::APPLYPHASEFLIP(_) => 2,
            Instruction::APPLYBITFLIP(_) => 2,
            Instruction::APPLYTGATE(_) => 2,
            Instruction::APPLYSGATE(_) => 2,
            Instruction::RESET(_) => 2,
            Instruction::CHAROUT(_) => 2,
            Instruction::QRESET(_) => 2,
            Instruction::LOOPSTART(_) => 2,
            Instruction::RAND(_) => 2,
            Instruction::PHOTONEMIT(_) => 2,
            Instruction::PHOTONDETECT(_) => 2,
            Instruction::PHOTONADDITION(_) => 2,
            Instruction::APPLYPHOTONSUBTRACTION(_) => 2,
            Instruction::SINGLEPHOTONSOURCEON(_) => 2,
            Instruction::SINGLEPHOTONSOURCEOFF(_) => 2,
            Instruction::MEASUREPARITY(_) => 2,
            Instruction::OPTICALSWITCHCONTROL(_, _) => 3, // opcode + u8 + bool (1 byte)
            Instruction::INPUT(_) => 2,
            Instruction::PUSHREG(_) => 2,
            Instruction::POPREG(_) => 2,
            Instruction::GETTIME(_) => 2,
            Instruction::MARKOBSERVED(_) => 2,
            Instruction::RELEASE(_) => 2,
            Instruction::INITQUBIT(_) => 2,
            Instruction::MEASURE(_) => 2,
            Instruction::SETPHASE(_, _) => 1 + 1 + 8,

            // 3-byte opcodes (two-qubit or reg/reg)
            Instruction::CONTROLLEDNOT(_, _) => 3,
            Instruction::SWAP(_, _) => 3,
            Instruction::ENTANGLE(_, _) => 3,
            Instruction::ENTANGLEBELL(_, _) => 3,
            Instruction::CNOT(_, _) => 3,
            Instruction::CZ(_, _) => 3,
            Instruction::THERMALAVG(_, _) => 3,
            Instruction::SQRT(_, _) => 3,
            Instruction::EXP(_, _) => 3,
            Instruction::LOG(_, _) => 3,
            Instruction::OPTICALROUTING(_, _) => 3,
            Instruction::CMP(_, _) => 3,
            Instruction::REGCOPY(_, _) => 3, // opcode + u8 + u8
            Instruction::NOTBITS(_, _) => 3, // opcode + u8 + u8
            Instruction::PHOTONBUNCHINGCONTROL(_, _) => 3, // opcode + u8 + bool (1 byte)
            Instruction::CHARLOAD(_, _) => 3, // opcode + u8 + u8

            // 4-byte opcodes (three regs)
            Instruction::CONTROLLEDSWAP(_, _, _) => 4,
            Instruction::WKBFACTOR(_, _, _) => 4,
            Instruction::REGADD(_, _, _) => 4,
            Instruction::REGSUB(_, _, _) => 4,
            Instruction::REGMUL(_, _, _) => 4,
            Instruction::REGDIV(_, _, _) => 4,
            Instruction::ANDBITS(_, _, _) => 4,
            Instruction::ORBITS(_, _, _) => 4,
            Instruction::XORBITS(_, _, _) => 4,
            Instruction::SHL(_, _, _) => 4,
            Instruction::SHR(_, _, _) => 4,

            // 9-byte opcodes (opcode + u64)
            Instruction::WAIT(_) => 1 + 8,
            Instruction::JMP(_) => 1 + 8,
            Instruction::JMPABS(_) => 1 + 8,
            Instruction::CALLADDR(_) => 1 + 8,
            Instruction::FREE(_) => 1 + 8,
            Instruction::SEEDRNG(_) => 1 + 8,

            // 10-byte opcodes (opcode + u8 + f64)
            Instruction::PHASESHIFT(_, _) => 1 + 1 + 8,
            Instruction::RX(_, _) => 1 + 1 + 8,
            Instruction::RY(_, _) => 1 + 1 + 8,
            Instruction::RZ(_, _) => 1 + 1 + 8,
            Instruction::PHASE(_, _) => 1 + 1 + 8,
            Instruction::REGSET(_, _) => 1 + 1 + 8,
            Instruction::TIMEDELAY(_, _) => 1 + 1 + 8,
            Instruction::SETOPTICALATTENUATION(_, _) => 1 + 1 + 8,
            Instruction::DYNAMICPHASECOMPENSATION(_, _) => 1 + 1 + 8,
            Instruction::OPTICALDELAYLINECONTROL(_, _) => 1 + 1 + 8,
            Instruction::APPLYDISPLACEMENT(_, _) => 1 + 1 + 8,
            Instruction::APPLYSQUEEZING(_, _) => 1 + 1 + 8,
            Instruction::APPLYNONLINEARPHASESHIFT(_, _) => 1 + 1 + 8,
            Instruction::APPLYNONLINEARSiGMA(_, _) => 1 + 1 + 8,
            Instruction::DECOHERENCEPROTECT(_, _) => 1 + 1 + 8,
            Instruction::LOADREGMEM(_, _) => 1 + 1 + 8, // opcode + reg + addr (u64)
            Instruction::ALLOC(_, _) => 1 + 1 + 8, // opcode + reg_addr + size (u64)
            Instruction::SETPOS(_, _, _) => 1 + 1 + 8 + 8, // opcode + q + x + y
            Instruction::MOVE(_, _, _) => 1 + 1 + 8 + 8, // opcode + q + dx + dy
            Instruction::SETWL(_, _) => 1 + 1 + 8,
            Instruction::WLSHIFT(_, _) => 1 + 1 + 8,


            // 11-byte opcodes (opcode + 2 u8 + f64)
            Instruction::CONTROLLEDPHASEROTATION(_, _, _) => 1 + 1 + 1 + 8,
            Instruction::APPLYCPHASE(_, _, _) => 1 + 1 + 1 + 8,
            Instruction::CROSSPHASEMODULATION(_, _, _) => 1 + 1 + 1 + 8,
            Instruction::APPLYROTATION(_, _, _) => 1 + 1 + 1 + 8, // opcode + q + axis + angle

            // 12-byte opcodes (opcode + 2 u8 + i64 offset)
            Instruction::IFGT(_, _, _) => 1 + 1 + 1 + 8,
            Instruction::IFLT(_, _, _) => 1 + 1 + 1 + 8,
            Instruction::IFEQ(_, _, _) => 1 + 1 + 1 + 8,
            Instruction::IFNE(_, _, _) => 1 + 1 + 1 + 8,

            // 18-byte opcodes (opcode + u8 + 2 f64 or u64 + f64)
            Instruction::APPLYKERRNONLINEARITY(_, _, _) => 1 + 1 + 8 + 8,
            Instruction::APPLYDISPLACEMENTOPERATOR(_, _, _) => 1 + 1 + 8 + 8,
            Instruction::PHOTONLOSSSIMULATE(_, _, _) => 1 + 1 + 8 + 8,
            Instruction::QUANTUMZENOEFFECT(_, _, _) => 1 + 1 + 8 + 8,


            // variable length based on string/vec lengths
            Instruction::APPLYGATE(_name, _) => 1 + 1 + 8, // opcode + q + fixed 8-byte name (name is not used here for size calculation)
            Instruction::ENTANGLEMULTI(qs) => 1 + 1 + qs.len(),
            Instruction::ENTANGLECLUSTER(qs) => 1 + 1 + qs.len(),
            Instruction::ENTANGLESWAP(_, _, _, _) => 5,
            Instruction::ENTANGLESWAPMEASURE(_, _, _, _, label) => 1 + 4 + label.len() + 1,
            Instruction::ENTANGLEWITHCLASSICALFEEDBACK(_, _, signal) => 1 + 2 + signal.len() + 1,
            Instruction::ENTANGLEDISTRIBUTED(_, node) => 1 + 1 + node.len() + 1,
            Instruction::MEASUREINBASIS(_, basis) => 1 + 1 + basis.len() + 1,
            Instruction::VERBOSELOG(_, msg) => 1 + 1 + msg.len() + 1,
            Instruction::APPLYFEEDFORWARDGATE(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::APPLYMEASUREMENTBASISCHANGE(_, basis) => 1 + 1 + basis.len() + 1,
            Instruction::LOAD(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::STORE(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::LOADMEM(reg, addr) => 1 + reg.len() + 1 + addr.len() + 1,
            Instruction::STOREMEM(reg, addr) => 1 + reg.len() + 1 + addr.len() + 1,
            Instruction::LOADCLASSICAL(reg, var) => 1 + reg.len() + 1 + var.len() + 1,
            Instruction::STORECLASSICAL(reg, var) => 1 + reg.len() + 1 + var.len() + 1,
            Instruction::ADD(dst, src1, src2) => 1 + dst.len() + 1 + src1.len() + 1 + src2.len() + 1,
            Instruction::SUB(dst, src1, src2) => 1 + dst.len() + 1 + src1.len() + 1 + src2.len() + 1,
            Instruction::AND(dst, src1, src2) => 1 + dst.len() + 1 + src1.len() + 1 + src2.len() + 1,
            Instruction::OR(dst, src1, src2) => 1 + dst.len() + 1 + src1.len() + 1 + src2.len() + 1,
            Instruction::XOR(dst, src1, src2) => 1 + dst.len() + 1 + src1.len() + 1 + src2.len() + 1,
            Instruction::NOT(reg) => 1 + reg.len() + 1,
            Instruction::PUSH(reg) => 1 + reg.len() + 1,
            Instruction::POP(reg) => 1 + reg.len() + 1,
            Instruction::JUMP(label) => 1 + label.len() + 1,
            Instruction::JUMPIFZERO(cond, label) => 1 + cond.len() + 1 + label.len() + 1,
            Instruction::JUMPIFONE(cond, label) => 1 + cond.len() + 1 + label.len() + 1,
            Instruction::CALL(label) => 1 + label.len() + 1,
            Instruction::PHOTONCOUNT(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::PHOTONEMISSIONPATTERN(_, reg, _) => 1 + 1 + reg.len() + 1 + 8, // u64 cycles
            Instruction::PHOTONDETECTWITHTHRESHOLD(_, _, reg) => 1 + 1 + 8 + reg.len() + 1, // u64 threshold
            Instruction::PHOTONDETECTCOINCIDENCE(qs, reg) => 1 + 1 + qs.len() + reg.len() + 1,
            Instruction::PHOTONROUTE(_, from, to) => 1 + 1 + from.len() + 1 + to.len() + 1,
            Instruction::APPLYDISPLACEMENTFEEDBACK(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::APPLYSQUEEZINGFEEDBACK(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::MEASUREWITHDELAY(_, _, reg) => 1 + 1 + 8 + reg.len() + 1, // u64 delay
            Instruction::PHOTONLOSSCORRECTION(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::APPLYQNDMEASUREMENT(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::ERRORCORRECT(_, syndrome_type) => 1 + 1 + syndrome_type.len() + 1,
            Instruction::ERRORSYNDROME(_, syndrome_type, result_reg) => 1 + 1 + syndrome_type.len() + 1 + result_reg.len() + 1,
            Instruction::QUANTUMSTATETOMOGRAPHY(_, basis) => 1 + 1 + basis.len() + 1,
            Instruction::BELLSTATEVERIFICATION(_, _, result_reg) => 1 + 2 + result_reg.len() + 1,
            Instruction::APPLYLINEAROPTICALTRANSFORM(_name, input_qs, output_qs, _num_modes) => {
                1 + 1 + 1 + 1 + input_qs.len() + output_qs.len() // name and num_modes are not used for size calculation here
            },
            Instruction::PHOTONNUMBERRESOLVINGDETECTION(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::FEEDBACKCONTROL(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::STOREMEMREG(_, _) => 1 + 8 + 1, // opcode + addr (u64) + reg
            Instruction::EXITCODE(_) => 1 + 4, // opcode + i32

            // special case for ApplyMultiQubitRotation
            Instruction::APPLYMULTIQUBITROTATION(qs, _, angles) => {
                1 + 1 + 1 + qs.len() + (angles.len() * 8) // opcode + axis + num_qs + qs bytes + angles bytes
            },

            // printf and print/println
            Instruction::PRINTF(format_str, regs) => {
                1 + 8 + format_str.len() + 1 + regs.len() // opcode + str_len (u64) + format_str + num_regs (u8) + regs
            },
            Instruction::PRINT(s) => 1 + 8 + s.len(), // opcode + str_len (u64) + string
            Instruction::PRINTLN(s) => 1 + 8 + s.len(), // opcode + str_len (u64) + string
        }
    }
}

pub fn parse_instruction(line: &str) -> Result<Instruction, String> {
    use Instruction::*;
    let trimmed_line = line.trim();

    if trimmed_line.is_empty() {
        return Err("empty instruction line".into());
    }

    // handle full-line comments starting with ';'
    if trimmed_line.starts_with(';') {
        // return a COMMENT instruction directly
        return Ok(COMMENT(trimmed_line[1..].trim().to_string()));
    }

    // check for label definition (e.g., "MY_LABEL:")
    if trimmed_line.ends_with(':') {
        let label_name = trimmed_line.trim_end_matches(':').to_string();
        if label_name.is_empty() {
            return Err("empty label name".into());
        }
        // label is a nop-instruction, returned here for the first pass of compilation
        return Ok(LABEL(label_name));
    }

    // strip inline comments before tokenizing
    let instruction_part = trimmed_line.split(';').next().unwrap_or("").trim();

    if instruction_part.is_empty() {
        return Err("empty instruction after stripping comment".into());
    }

    let tokens: Vec<&str> = instruction_part.split_whitespace().collect();
    let op = tokens[0].to_uppercase(); // get the opcode, convert to uppercase

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
        // core
        "QINIT" => {
            if tokens.len() == 2 {
                Ok(QINIT(parse_u8(tokens[1])?))
            } else {
                Err("qinit <qubit>".into())
            }
        }
        "QMEAS" => {
            if tokens.len() == 2 {
                Ok(QMEAS(parse_u8(tokens[1])?))
            } else {
                Err("qmeas <qubit>".into())
            }
        }
        "H" => { // h is now the primary alias
            if tokens.len() == 2 {
                Ok(H(parse_u8(tokens[1])?))
            } else {
                Err("usage: h <qubit>".into())
            }
        }
        "APPLYHADAMARD" => { // explicit applyhadamard instruction
            if tokens.len() == 2 {
                Ok(APPLYHADAMARD(parse_u8(tokens[1])?))
            } else {
                Err("usage: applyhadamard <qubit>".into())
            }
        }
        "CONTROLLEDNOT" => {
            if tokens.len() == 3 {
                Ok(CONTROLLEDNOT(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("controllednot <c> <t>".into())
            }
        }
        "APPLYPHASEFLIP" => {
            if tokens.len() == 2 {
                Ok(APPLYPHASEFLIP(parse_u8(tokens[1])?))
            } else {
                Err("applyphaseflip <qubit>".into())
            }
        }
        "APPLYBITFLIP" => {
            if tokens.len() == 2 {
                Ok(APPLYBITFLIP(parse_u8(tokens[1])?))
            } else {
                Err("applybitflip <qubit>".into())
            }
        }
        "APPLYTGATE" => {
            if tokens.len() == 2 {
                Ok(APPLYTGATE(parse_u8(tokens[1])?))
            } else {
                Err("applytgate <qubit>".into())
            }
        }
        "APPLYSGATE" => {
            if tokens.len() == 2 {
                Ok(APPLYSGATE(parse_u8(tokens[1])?))
            } else {
                Err("applysgate <qubit>".into())
            }
        }
        "PHASESHIFT" => {
            if tokens.len() == 3 {
                Ok(PHASESHIFT(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("phaseshift <qubit> <angle>".into())
            }
        }
        "WAIT" => {
            if tokens.len() == 2 {
                Ok(WAIT(parse_u64(tokens[1])?))
            } else {
                Err("wait <cycles>".into())
            }
        }
        "RESET" => {
            if tokens.len() == 2 {
                Ok(RESET(parse_u8(tokens[1])?))
            } else {
                Err("reset <qubit>".into())
            }
        }
        "SWAP" => {
            if tokens.len() == 3 {
                Ok(SWAP(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("swap <q1> <q2>".into())
            }
        }
        "CONTROLLEDSWAP" => {
            if tokens.len() == 4 {
                Ok(CONTROLLEDSWAP(
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
                Ok(ENTANGLE(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("entangle <q1> <q2>".into())
            }
        }
        "ENTANGLEBELL" => {
            if tokens.len() == 3 {
                Ok(ENTANGLEBELL(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("entanglebell <q1> <q2>".into())
            }
        }
        "ENTANGLEMULTI" => {
            if tokens.len() >= 2 {
                Ok(ENTANGLEMULTI(
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
                Ok(ENTANGLECLUSTER(
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
                Ok(ENTANGLESWAP(
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
                Ok(ENTANGLESWAPMEASURE(
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
                Ok(ENTANGLEWITHCLASSICALFEEDBACK(
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
                Ok(ENTANGLEDISTRIBUTED(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("entangledistributed <qubit> <node>".into())
            }
        }
        "MEASUREINBASIS" => {
            if tokens.len() == 3 {
                Ok(MEASUREINBASIS(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("measureinbasis <qubit> <basis>".into())
            }
        }
        "SYNC" => {
            if tokens.len() == 1 {
                Ok(SYNC)
            } else {
                Err("sync".into())
            }
        }
        "RESETALL" => {
            if tokens.len() == 1 {
                Ok(RESETALL)
            } else {
                Err("resetall".into())
            }
        }
        "VERBOSELOG" => {
            if tokens.len() >= 3 {
                let q = parse_u8(tokens[1])?;
                let msg = tokens[2..].join(" ");
                Ok(VERBOSELOG(q, parse_string_literal(&msg)?))
            } else {
                Err("verboselog <qubit> <message>".into())
            }
        }
        "SETPHASE" => {
            if tokens.len() == 3 {
                Ok(SETPHASE(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("setphase <qubit> <phase>".into())
            }
        }
        "APPLYGATE" => {
            if tokens.len() == 3 {
                Ok(APPLYGATE(
                    parse_string_literal(tokens[1])?,
                    parse_u8(tokens[2])?,
                ))
            } else {
                Err("applygate <gate_name> <qubit>".into())
            }
        }
        "MEASURE" => {
            if tokens.len() == 2 {
                Ok(MEASURE(parse_u8(tokens[1])?))
            } else {
                Err("measure <qubit>".into())
            }
        }
        "INITQUBIT" => {
            if tokens.len() == 2 {
                Ok(INITQUBIT(parse_u8(tokens[1])?))
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
                    Ok(CHARLOAD(reg, ch as u8))
                } else {
                    Err("usage: charload <reg> '<char>'".into())
                }
            } else {
                Err("usage: charload <reg> '<char>'".into())
            }
        }
        "CHAROUT" => {
            if tokens.len() == 2 {
                Ok(CHAROUT(parse_u8(tokens[1])?))
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
                Ok(PHASE(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
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
                Ok(QRESET(parse_u8(tokens[1])?))
            } else {
                Err("qreset <qubit>".into())
            }
        }
        "THERMALAVG" => {
            if tokens.len() == 3 {
                Ok(THERMALAVG(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("thermalavg <qubit> <param>".into())
            }
        }
        "WKBFACTOR" => {
            if tokens.len() == 4 {
                Ok(WKBFACTOR(
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
                Ok(REGSET(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("usage: regset <reg> <float_value>".into())
            }
        }

        // loop
        "LOOPSTART" => {
            if tokens.len() == 2 {
                Ok(LOOPSTART(parse_u8(tokens[1])?))
            } else {
                Err("loopstart <reg>".into())
            }
        }
        "LOOPEND" => {
            if tokens.len() == 1 {
                Ok(LOOPEND)
            } else {
                Err("loopend".into())
            }
        }

        // rotations
        "APPLYROTATION" => {
            if tokens.len() == 4 {
                Ok(APPLYROTATION(
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
                Ok(APPLYMULTIQUBITROTATION(qs, axis, angles))
            } else {
                Err("applymultiqubitrotation <q1,q2,...> <x|y|z> <a1> <a2> ...".into())
            }
        }
        "CONTROLLEDPHASEROTATION" => {
            if tokens.len() == 4 {
                Ok(CONTROLLEDPHASEROTATION(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_f64(tokens[3])?,
                ))
            } else {
                Err("controlledphaserotation <c> <t> <angle>".into())
            }
        }
        "APPLYCPHASE" => {
            if tokens.len() == 4 {
                Ok(APPLYCPHASE(
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
                Ok(APPLYKERRNONLINEARITY(
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
                Ok(APPLYFEEDFORWARDGATE(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("applyfeedforwardgate <q> <reg>".into())
            }
        }
        "DECOHERENCEPROTECT" => {
            if tokens.len() == 3 {
                Ok(DECOHERENCEPROTECT(
                    parse_u8(tokens[1])?,
                    parse_u64(tokens[2])?,
                ))
            } else {
                Err("decoherenceprotect <q> <duration>".into())
            }
        }
        "APPLYMEASUREMENTBASISCHANGE" => {
            if tokens.len() == 3 {
                Ok(APPLYMEASUREMENTBASISCHANGE(
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
                Ok(LOAD(parse_u8(tokens[1])?, parse_string_literal(tokens[2])?))
            } else {
                Err("load <qubit> <var>".into())
            }
        }
        "STORE" => {
            if tokens.len() == 3 {
                Ok(STORE(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("store <qubit> <var>".into())
            }
        }
        "LOADMEM" => {
            if tokens.len() == 3 {
                Ok(LOADMEM(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("loadmem <reg> <mem>".into())
            }
        }
        "STOREMEM" => {
            if tokens.len() == 3 {
                Ok(STOREMEM(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("storemem <reg> <mem>".into())
            }
        }
        "LOADCLASSICAL" => {
            if tokens.len() == 3 {
                Ok(LOADCLASSICAL(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("loadclassical <reg> <var>".into())
            }
        }
        "STORECLASSICAL" => {
            if tokens.len() == 3 {
                Ok(STORECLASSICAL(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("storeclassical <reg> <var>".into())
            }
        }
        "ADD" => {
            if tokens.len() == 4 {
                Ok(ADD(
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
                Ok(SUB(
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
                Ok(AND(
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
                Ok(OR(
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
                Ok(XOR(
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
                Ok(NOT(parse_string_literal(tokens[1])?))
            } else {
                Err("not <reg>".into())
            }
        }
        "PUSH" => {
            if tokens.len() == 2 {
                Ok(PUSH(parse_string_literal(tokens[1])?))
            } else {
                Err("push <reg>".into())
            }
        }
        "POP" => {
            if tokens.len() == 2 {
                Ok(POP(parse_string_literal(tokens[1])?))
            } else {
                Err("pop <reg>".into())
            }
        }

        // classical flow control
        "JUMP" => {
            if tokens.len() == 2 {
                Ok(JUMP(parse_string_literal(tokens[1])?))
            } else {
                Err("jump <label>".into())
            }
        }
        "JUMPIFZERO" => {
            if tokens.len() == 3 {
                Ok(JUMPIFZERO(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("jumpifzero <cond_reg> <label>".into())
            }
        }
        "JUMPIFONE" => {
            if tokens.len() == 3 {
                Ok(JUMPIFONE(
                    parse_string_literal(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("jumpifone <cond_reg> <label>".into())
            }
        }
        "CALL" => {
            if tokens.len() == 2 {
                Ok(CALL(parse_string_literal(tokens[1])?))
            } else {
                Err("call <label>".into())
            }
        }
        "RETURN" => {
            if tokens.len() == 1 {
                Ok(RETURN)
            } else {
                Err("return".into())
            }
        }
        "BARRIER" => {
            if tokens.len() == 1 {
                Ok(BARRIER)
            } else {
                Err("barrier".into())
            }
        }
        "TIMEDELAY" => {
            if tokens.len() == 3 {
                Ok(TIMEDELAY(parse_u8(tokens[1])?, parse_u64(tokens[2])?))
            } else {
                Err("timedelay <qubit> <cycles>".into())
            }
        }
        "RAND" => {
            if tokens.len() == 2 {
                Ok(RAND(parse_u8(tokens[1])?))
            } else {
                Err("rand <reg>".into())
            }
        }
        "SQRT" => {
            if tokens.len() == 3 {
                Ok(SQRT(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("sqrt <rd> <rs>".into())
            }
        }
        "EXP" => {
            if tokens.len() == 3 {
                Ok(EXP(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("exp <rd> <rs>".into())
            }
        }
        "LOG" => {
            if tokens.len() == 3 {
                Ok(LOG(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("log <rd> <rs>".into())
            }
        }
        // arithmetic operations
        "REGADD" => {
            if tokens.len() == 4 {
                Ok(REGADD(
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
                Ok(REGSUB(
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
                Ok(REGMUL(
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
                Ok(REGDIV(
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
                Ok(REGCOPY(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("regcopy <rd> <ra>".into())
            }
        }

        // optics
        "PHOTONEMIT" => {
            if tokens.len() == 2 {
                Ok(PHOTONEMIT(parse_u8(tokens[1])?))
            } else {
                Err("photonemit <qubit>".into())
            }
        }
        "PHOTONDETECT" => {
            if tokens.len() == 2 {
                Ok(PHOTONDETECT(parse_u8(tokens[1])?))
            } else {
                Err("photondetect <qubit>".into())
            }
        }
        "PHOTONCOUNT" => {
            if tokens.len() == 3 {
                Ok(PHOTONCOUNT(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("photoncount <qubit> <result_reg>".into())
            }
        }
        "PHOTONADDITION" => {
            if tokens.len() == 2 {
                Ok(PHOTONADDITION(parse_u8(tokens[1])?))
            } else {
                Err("photonaddition <qubit>".into())
            }
        }
        "APPLYPHOTONSUBTRACTION" => {
            if tokens.len() == 2 {
                Ok(APPLYPHOTONSUBTRACTION(parse_u8(tokens[1])?))
            } else {
                Err("applyphotonsubtraction <qubit>".into())
            }
        }
        "PHOTONEMISSIONPATTERN" => {
            if tokens.len() == 4 {
                Ok(PHOTONEMISSIONPATTERN(
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
                Ok(PHOTONDETECTWITHTHRESHOLD(
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
                Ok(PHOTONDETECTCOINCIDENCE(
                    qs,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("photondetectcoincidence <q1,q2,...> <result_reg>".into())
            }
        }
        "SINGLEPHOTONSOURCEON" => {
            if tokens.len() == 2 {
                Ok(SINGLEPHOTONSOURCEON(parse_u8(tokens[1])?))
            } else {
                Err("singlephotonsourceon <qubit>".into())
            }
        }
        "SINGLEPHOTONSOURCEOFF" => {
            if tokens.len() == 2 {
                Ok(SINGLEPHOTONSOURCEOFF(parse_u8(tokens[1])?))
            } else {
                Err("singlephotonsourceoff <qubit>".into())
            }
        }
        "PHOTONBUNCHINGCONTROL" => {
            if tokens.len() == 3 {
                Ok(PHOTONBUNCHINGCONTROL(
                    parse_u8(tokens[1])?,
                    _parse_bool(tokens[2])?,
                ))
            } else {
                Err("photonbunchingcontrol <qubit> <true|false>".into())
            }
        }
        "PHOTONROUTE" => {
            if tokens.len() == 4 {
                Ok(PHOTONROUTE(
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
                Ok(OPTICALROUTING(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("opticalrouting <q1> <q2>".into())
            }
        }
        "SETOPTICALATTENUATION" => {
            if tokens.len() == 3 {
                Ok(SETOPTICALATTENUATION(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("setopticalattenuation <qubit> <attenuation>".into())
            }
        }
        "DYNAMICPHASECOMPENSATION" => {
            if tokens.len() == 3 {
                Ok(DYNAMICPHASECOMPENSATION(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("dynamicphasecompensation <qubit> <phase>".into())
            }
        }
        "OPTICALDELAYLINECONTROL" => {
            if tokens.len() == 3 {
                Ok(OPTICALDELAYLINECONTROL(
                    parse_u8(tokens[1])?,
                    parse_u64(tokens[2])?,
                ))
            } else {
                Err("opticaldelaylinecontrol <qubit> <delay_cycles>".into())
            }
        }
        "CROSSPHASEMODULATION" => {
            if tokens.len() == 4 {
                Ok(CROSSPHASEMODULATION(
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
                Ok(APPLYDISPLACEMENT(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("applydisplacement <qubit> <alpha>".into())
            }
        }
        "APPLYDISPLACEMENTFEEDBACK" => {
            if tokens.len() == 3 {
                Ok(APPLYDISPLACEMENTFEEDBACK(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("applydisplacementfeedback <qubit> <feedback_reg>".into())
            }
        }
        "APPLYDISPLACEMENTOPERATOR" => {
            if tokens.len() == 4 {
                Ok(APPLYDISPLACEMENTOPERATOR(
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
                Ok(APPLYSQUEEZING(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("applysqueezing <qubit> <squeezing_factor>".into())
            }
        }
        "APPLYSQUEEZINGFEEDBACK" => {
            if tokens.len() == 3 {
                Ok(APPLYSQUEEZINGFEEDBACK(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("applysqueezingfeedback <qubit> <feedback_reg>".into())
            }
        }
        "MEASUREPARITY" => {
            if tokens.len() == 2 {
                Ok(MEASUREPARITY(parse_u8(tokens[1])?))
            } else {
                Err("measureparity <qubit>".into())
            }
        }
        "MEASUREWITHDELAY" => {
            if tokens.len() == 4 {
                Ok(MEASUREWITHDELAY(
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
                Ok(OPTICALSWITCHCONTROL(
                    parse_u8(tokens[1])?,
                    _parse_bool(tokens[2])?,
                ))
            } else {
                Err("opticalswitchcontrol <qubit> <on|off>".into())
            }
        }
        "PHOTONLOSSSIMULATE" => {
            if tokens.len() == 4 {
                Ok(PHOTONLOSSSIMULATE(
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
                Ok(PHOTONLOSSCORRECTION(
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
                Ok(APPLYQNDMEASUREMENT(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("applyqndmeasurement <qubit> <result_reg>".into())
            }
        }
        "ERRORCORRECT" => {
            if tokens.len() == 3 {
                Ok(ERRORCORRECT(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("errorcorrect <qubit> <syndrome_type>".into())
            }
        }
        "ERRORSYNDROME" => {
            if tokens.len() == 4 {
                Ok(ERRORSYNDROME(
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
                Ok(QUANTUMSTATETOMOGRAPHY(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("quantumstatetomography <qubit> <basis>".into())
            }
        }
        "BELLSTATEVERIFICATION" => {
            if tokens.len() == 4 {
                Ok(BELLSTATEVERIFICATION(
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
                Ok(QUANTUMZENOEFFECT(
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
                Ok(APPLYNONLINEARPHASESHIFT(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("applynonlinearphaseshift <qubit> <strength>".into())
            }
        }
        "APPLYNONLINEARSiGMA" => {
            if tokens.len() == 3 {
                Ok(APPLYNONLINEARSiGMA(
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

                Ok(APPLYLINEAROPTICALTRANSFORM(
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
                Ok(PHOTONNUMBERRESOLVINGDETECTION(
                    parse_u8(tokens[1])?,
                    parse_string_literal(tokens[2])?,
                ))
            } else {
                Err("photonnumberresolvingdetection <qubit> <result_reg>".into())
            }
        }
        "FEEDBACKCONTROL" => {
            if tokens.len() == 3 {
                Ok(FEEDBACKCONTROL(
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
                Ok(SETPOS(
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
                Ok(SETWL(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("setwl <reg> <wavelength>".into())
            }
        }
        "WLSHIFT" => {
            if tokens.len() == 3 {
                Ok(WLSHIFT(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("wlshift <reg> <delta_wavelength>".into())
            }
        }
        "MOVE" => {
            if tokens.len() == 4 {
                Ok(MOVE(
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
                Ok(COMMENT(tokens[1..].join(" ")))
            } else {
                Err("comment <text>".into())
            }
        }
        "MARKOBSERVED" => {
            if tokens.len() == 2 {
                Ok(MARKOBSERVED(parse_u8(tokens[1])?))
            } else {
                Err("markobserved <reg>".into())
            }
        }
        "RELEASE" => {
            if tokens.len() == 2 {
                Ok(RELEASE(parse_u8(tokens[1])?))
            } else {
                Err("release <reg>".into())
            }
        }
        "HALT" => {
            if tokens.len() == 1 {
                Ok(HALT)
            } else {
                Err("halt".into())
            }
        }
        // new instruction parsing for v0.3.0
        "JMP" => {
            if tokens.len() == 2 {
                Ok(JMP(parse_i64(tokens[1])?))
            } else {
                Err("jmp <offset>".into())
            }
        },
        "JMPABS" => {
            if tokens.len() == 2 {
                Ok(JMPABS(parse_u64(tokens[1])?))
            } else {
                Err("jmpabs <address>".into())
            }
        },
        "IFGT" => {
            if tokens.len() == 4 {
                Ok(IFGT(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_i64(tokens[3])?,
                ))
            } else {
                Err("ifgt <reg1> <reg2> <offset>".into())
            }
        },
        "IFLT" => {
            if tokens.len() == 4 {
                Ok(IFLT(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_i64(tokens[3])?,
                ))
            } else {
                Err("iflt <reg1> <reg2> <offset>".into())
            }
        },
        "IFEQ" => {
            if tokens.len() == 4 {
                Ok(IFEQ(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_i64(tokens[3])?,
                ))
            } else {
                Err("ifeq <reg1> <reg2> <offset>".into())
            }
        },
        "IFNE" => {
            if tokens.len() == 4 {
                Ok(IFNE(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_i64(tokens[3])?,
                ))
            } else {
                Err("ifne <reg1> <reg2> <offset>".into())
            }
        },
        "CALLADDR" => {
            if tokens.len() == 2 {
                Ok(CALLADDR(parse_u64(tokens[1])?))
            } else {
                Err("calladdr <address>".into())
            }
        },
        "RETSUB" => {
            if tokens.len() == 1 {
                Ok(RETSUB)
            } else {
                Err("retsub".into())
            }
        },
        "PRINTF" => {
            if tokens.len() >= 2 {
                let format_str = parse_string_literal(tokens[1])?;
                let regs = parse_reg_list(&tokens[2..])?;
                Ok(PRINTF(format_str, regs))
            } else {
                Err("printf <format_string> [reg1] [reg2] ...".into())
            }
        },
        "PRINT" => {
            if tokens.len() == 2 {
                Ok(PRINT(parse_string_literal(tokens[1])?))
            } else {
                Err("print <string>".into())
            }
        },
        "PRINTLN" => {
            if tokens.len() == 2 {
                Ok(PRINTLN(parse_string_literal(tokens[1])?))
            } else {
                Err("println <string>".into())
            }
        },
        "INPUT" => {
            if tokens.len() == 2 {
                Ok(INPUT(parse_u8(tokens[1])?))
            } else {
                Err("input <reg>".into())
            }
        },
        "DUMPSTATE" => {
            if tokens.len() == 1 {
                Ok(DUMPSTATE)
            } else {
                Err("dumpstate".into())
            }
        },
        "DUMPREGS" => {
            if tokens.len() == 1 {
                Ok(DUMPREGS)
            } else {
                Err("dumpregs".into())
            }
        },
        "LOADREGMEM" => {
            if tokens.len() == 3 {
                Ok(LOADREGMEM(parse_u8(tokens[1])?, parse_u64(tokens[2])?))
            } else {
                Err("loadregmem <reg> <address>".into())
            }
        },
        "STOREMEMREG" => {
            if tokens.len() == 3 {
                Ok(STOREMEMREG(parse_u64(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("storememreg <address> <reg>".into())
            }
        },
        "PUSHREG" => {
            if tokens.len() == 2 {
                Ok(PUSHREG(parse_u8(tokens[1])?))
            } else {
                Err("pushreg <reg>".into())
            }
        },
        "POPREG" => {
            if tokens.len() == 2 {
                Ok(POPREG(parse_u8(tokens[1])?))
            } else {
                Err("popreg <reg>".into())
            }
        },
        "ALLOC" => {
            if tokens.len() == 3 {
                Ok(ALLOC(parse_u8(tokens[1])?, parse_u64(tokens[2])?))
            } else {
                Err("alloc <reg_addr> <size>".into())
            }
        },
        "FREE" => {
            if tokens.len() == 2 {
                Ok(FREE(parse_u64(tokens[1])?))
            } else {
                Err("free <address>".into())
            }
        },
        "CMP" => {
            if tokens.len() == 3 {
                Ok(CMP(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("cmp <reg1> <reg2>".into())
            }
        },
        "ANDBITS" => {
            if tokens.len() == 4 {
                Ok(ANDBITS(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_u8(tokens[3])?,
                ))
            } else {
                Err("andbits <dest> <op1> <op2>".into())
            }
        },
        "ORBITS" => {
            if tokens.len() == 4 {
                Ok(ORBITS(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_u8(tokens[3])?,
                ))
            } else {
                Err("orbits <dest> <op1> <op2>".into())
            }
        },
        "XORBITS" => {
            if tokens.len() == 4 {
                Ok(XORBITS(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_u8(tokens[3])?,
                ))
            } else {
                Err("xorbits <dest> <op1> <op2>".into())
            }
        },
        "NOTBITS" => {
            if tokens.len() == 3 {
                Ok(NOTBITS(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("notbits <dest> <op>".into())
            }
        },
        "SHL" => {
            if tokens.len() == 4 {
                Ok(SHL(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_u8(tokens[3])?,
                ))
            } else {
                Err("shl <dest> <op> <amount>".into())
            }
        },
        "SHR" => {
            if tokens.len() == 4 {
                Ok(SHR(
                    parse_u8(tokens[1])?,
                    parse_u8(tokens[2])?,
                    parse_u8(tokens[3])?,
                ))
            } else {
                Err("shr <dest> <op> <amount>".into())
            }
        },
        "BREAKPOINT" => {
            if tokens.len() == 1 {
                Ok(BREAKPOINT)
            } else {
                Err("breakpoint".into())
            }
        },
        "GETTIME" => {
            if tokens.len() == 2 {
                Ok(GETTIME(parse_u8(tokens[1])?))
            } else {
                Err("gettime <reg>".into())
            }
        },
        "SEEDRNG" => {
            if tokens.len() == 2 {
                Ok(SEEDRNG(parse_u64(tokens[1])?))
            } else {
                Err("seedrng <seed>".into())
            }
        },
        "EXITCODE" => {
            if tokens.len() == 2 {
                Ok(EXITCODE(parse_i64(tokens[1])? as i32))
            } else {
                Err("exitcode <code>".into())
            }
        },
        _ => Err(format!("unknown instruction '{}'", op)),
    }
}
