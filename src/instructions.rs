// not done yet, but seems pretty verbose for now

// all supported instructions in QOA

#![allow(dead_code)] // this makes the compiler shut up

#[derive(Debug)]
pub enum Instruction {
    // Core quantum instructions
    InitQubit(u8),
    ApplyGate(String, u8),
    Entangle(u8, u8),
    Measure(u8),
    PhaseShift(u8, f64),
    Wait(u64),
    Reset(u8),
    ApplyKerrNonlinearity(u8, f64, u64),
    
    // Memory and classical data ops
    Load(u8, String),
    Store(u8, String),

    // Two qubit or multi qubit ops
    Swap(u8, u8),
    EntangleBell(u8, u8),
    PhotonEmit(u8),
    PhotonDetect(u8),

    // Single qubit gates
    ApplyHadamard(u8),
    ControlledNot(u8, u8),
    ApplyPhaseFlip(u8),
    ApplyBitFlip(u8),
    ApplyTGate(u8),
    ApplySGate(u8),

    // Logic ops (classical)
    Add(String, String, String),
    Sub(String, String, String),
    And(String, String, String),
    Or(String, String, String),
    Xor(String, String, String),
    Not(String, String),

    // Flow control
    Jump(String),
    JumpIfZero(String, String),
    JumpIfOne(String, String),
    Call(String),
    Return,
    Push(String),
    Pop(String),

    // Memory ops with classical registers
    LoadMem(String, String),
    StoreMem(String, String),
    LoadClassical(String, String),
    StoreClassical(String, String),

    // Multi qubit operations
    EntangleMulti(Vec<u8>),
    EntangleCluster(Vec<u8>),
    EntangleSwap(u8, u8, u8, u8),
    EntangleSwapMeasure(u8, u8, u8, u8, String),
    EntangleWithClassicalFeedback(u8, u8, String),
    EntangleDistributed(u8, String),

    // Rotations
    ApplyRotation(u8, char, f64), // axis: X, Y, Z
    ApplyMultiQubitRotation(Vec<u8>, char, Vec<f64>),
    ApplyFeedforwardGate(u8, String),
    ControlledPhaseRotation(u8, u8, f64),
    ApplyCPhase(u8, u8, f64),

    // Advanced/Optics
    ResetAll,
    PhotonRoute(u8, String, String),
    Sync,
    ErrorCorrect(u8, String),
    PhotonCount(u8, String),
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
    PhotonAddition(u8),
    ApplyPhotonSubtraction(u8),
    PhotonEmissionPattern(u8, String, u64),
    PhotonDetectWithThreshold(u8, u64, String),
    PhotonDetectCoincidence(Vec<u8>, String),
    SinglePhotonSourceOn(u8),
    SinglePhotonSourceOff(u8),
    PhotonBunchingControl(u8, bool),
    OpticalRouting(u8, u8),
    SetOpticalAttenuation(u8, f64),
    DynamicPhaseCompensation(u8, f64),
    OpticalDelayLineControl(u8, u64),
    CrossPhaseModulation(u8, u8, f64),
    QuantumStateTomography(u8, String),
    BellStateVerification(u8, u8, String),
    QuantumZenoEffect(u8, u64, u64),
    ApplyNonlinearPhaseShift(u8, f64),
    ApplyNonlinearSigma(u8, f64),
    ApplyLinearOpticalTransform(String, Vec<u8>, Vec<u8>, usize),

    // QND, error correction, advanced
    ApplyQndMeasurement(u8, String),
    ErrorSyndrome(u8, String, String),

    // Other
    TimeDelay(u8, u64),
    QInit(u8),
    CharLoad(u8, u8),
    QMeas(u8),
    ControlledSwap(u8, u8, u8),
    MeasureInBasis(u8, String),
    DecoherenceProtect(u8, u64),
    FeedbackControl(u8, String),
    PhotonNumberResolvingDetection(u8, String),
    ApplyMeasurementBasisChange(u8, String),
}

pub fn parse_instruction(line: &str) -> Result<Instruction, String> {
    let tokens: Vec<&str> = line.trim().split_whitespace().collect();
    if tokens.is_empty() {
        return Err("Empty instruction line".into());
    }

    let opcode = tokens[0].to_uppercase();

    let parse_u8 = |s: &str| s.parse::<u8>().map_err(|_| format!("Invalid u8 '{}'", s));
    let parse_u64 = |s: &str| s.parse::<u64>().map_err(|_| format!("Invalid u64 '{}'", s));
    let parse_f64 = |s: &str| s.parse::<f64>().map_err(|_| format!("Invalid f64 '{}'", s));
    let parse_bool = |s: &str| {
        match s.to_uppercase().as_str() {
            "TRUE" | "ON" => Ok(true),
            "FALSE" | "OFF" => Ok(false),
            _ => Err(format!("Invalid boolean '{}'", s)),
        }
    };
    let parse_axis = |s: &str| {
        let c = s.to_uppercase();
        if c == "X" || c == "Y" || c == "Z" {
            Ok(c.chars().next().unwrap())
        } else {
            Err(format!("Invalid axis '{}', must be X, Y, or Z", s))
        }
    };

    match opcode.as_str() {
        "INIT" => {
            if tokens.len() == 3 && tokens[1].to_uppercase() == "QUBIT" {
                let n = parse_u8(tokens[2])?;
                Ok(Instruction::InitQubit(n))
            } else {
                Err("Malformed INIT instruction. Usage: INIT QUBIT <n>".into())
            }
        }
        "QINIT" => {
            if tokens.len() == 2 {
                let n = parse_u8(tokens[1])?;
                Ok(Instruction::QInit(n))
            } else {
                Err("Malformed QINIT instruction. Usage: QINIT <qubit>".into())
            }
        }
        "CHARLOAD" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let val = parse_u8(tokens[2])?;
                Ok(Instruction::CharLoad(n, val))
            } else {
                Err("Malformed CHARLOAD instruction. Usage: CHARLOAD <qubit> <ascii_val>".into())
            }
        }
        "QMEAS" => {
            if tokens.len() == 2 {
                let n = parse_u8(tokens[1])?;
                Ok(Instruction::QMeas(n))
            } else {
                Err("Malformed QMEAS instruction. Usage: QMEAS <qubit>".into())
            }
        }
        "APPLY_GATE" => {
            if tokens.len() == 3 {
                let gate = tokens[1].to_uppercase();
                let n = parse_u8(tokens[2])?;
                Ok(Instruction::ApplyGate(gate, n))
            } else {
                Err("Malformed APPLY_GATE instruction. Usage: APPLY_GATE <gate> <qubit>".into())
            }
        }
        "ENTANGLE" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let m = parse_u8(tokens[2])?;
                Ok(Instruction::Entangle(n, m))
            } else {
                Err("Malformed ENTANGLE instruction. Usage: ENTANGLE <qubit1> <qubit2>".into())
            }
        }
        "MEASURE" => {
            if tokens.len() == 2 {
                let n = parse_u8(tokens[1])?;
                Ok(Instruction::Measure(n))
            } else {
                Err("Malformed MEASURE instruction. Usage: MEASURE <qubit>".into())
            }
        }
        "PHASE_SHIFT" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let val = parse_f64(tokens[2])?;
                Ok(Instruction::PhaseShift(n, val))
            } else {
                Err("Malformed PHASE_SHIFT instruction. Usage: PHASE_SHIFT <qubit> <angle>".into())
            }
        }
        "WAIT" => {
            if tokens.len() == 2 {
                let cycles = parse_u64(tokens[1])?;
                Ok(Instruction::Wait(cycles))
            } else {
                Err("Malformed WAIT instruction. Usage: WAIT <cycles>".into())
            }
        }
        "RESET" => {
            if tokens.len() == 2 {
                let n = parse_u8(tokens[1])?;
                Ok(Instruction::Reset(n))
            } else {
                Err("Malformed RESET instruction. Usage: RESET <qubit>".into())
            }
        }
        "LOAD" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let var = tokens[2].to_string();
                Ok(Instruction::Load(n, var))
            } else {
                Err("Malformed LOAD instruction. Usage: LOAD <qubit> <var>".into())
            }
        }
        "STORE" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let var = tokens[2].to_string();
                Ok(Instruction::Store(n, var))
            } else {
                Err("Malformed STORE instruction. Usage: STORE <qubit> <var>".into())
            }
        }
        "SWAP" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let m = parse_u8(tokens[2])?;
                Ok(Instruction::Swap(n, m))
            } else {
                Err("Malformed SWAP instruction. Usage: SWAP <qubit1> <qubit2>".into())
            }
        }
        "ENTANGLE_BELL" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let m = parse_u8(tokens[2])?;
                Ok(Instruction::EntangleBell(n, m))
            } else {
                Err("Malformed ENTANGLE_BELL instruction. Usage: ENTANGLE_BELL <q1> <q2>".into())
            }
        }
        "PHOTON_EMIT" => {
            if tokens.len() == 2 {
                let n = parse_u8(tokens[1])?;
                Ok(Instruction::PhotonEmit(n))
            } else {
                Err("Malformed PHOTON_EMIT instruction. Usage: PHOTON_EMIT <qubit>".into())
            }
        }
        "PHOTON_DETECT" => {
            if tokens.len() == 2 {
                let n = parse_u8(tokens[1])?;
                Ok(Instruction::PhotonDetect(n))
            } else {
                Err("Malformed PHOTON_DETECT instruction. Usage: PHOTON_DETECT <qubit>".into())
            }
        }
        "APPLY_HADAMARD" => {
            if tokens.len() == 2 {
                let n = parse_u8(tokens[1])?;
                Ok(Instruction::ApplyHadamard(n))
            } else {
                Err("Malformed APPLY_HADAMARD instruction. Usage: APPLY_HADAMARD <qubit>".into())
            }
        }
        "CONTROLLED_NOT" => {
            if tokens.len() == 3 {
                let c = parse_u8(tokens[1])?;
                let t = parse_u8(tokens[2])?;
                Ok(Instruction::ControlledNot(c, t))
            } else {
                Err("Malformed CONTROLLED_NOT instruction. Usage: CONTROLLED_NOT <control> <target>".into())
            }
        }
        "APPLY_PHASE_FLIP" => {
            if tokens.len() == 2 {
                let n = parse_u8(tokens[1])?;
                Ok(Instruction::ApplyPhaseFlip(n))
            } else {
                Err("Malformed APPLY_PHASE_FLIP instruction. Usage: APPLY_PHASE_FLIP <qubit>".into())
            }
        }
        "APPLY_BIT_FLIP" => {
            if tokens.len() == 2 {
                let n = parse_u8(tokens[1])?;
                Ok(Instruction::ApplyBitFlip(n))
            } else {
                Err("Malformed APPLY_BIT_FLIP instruction. Usage: APPLY_BIT_FLIP <qubit>".into())
            }
        }
        "APPLY_TGATE" => {
            if tokens.len() == 2 {
                let n = parse_u8(tokens[1])?;
                Ok(Instruction::ApplyTGate(n))
            } else {
                Err("Malformed APPLY_TGATE instruction. Usage: APPLY_TGATE <qubit>".into())
            }
        }
        "APPLY_SGATE" => {
            if tokens.len() == 2 {
                let n = parse_u8(tokens[1])?;
                Ok(Instruction::ApplySGate(n))
            } else {
                Err("Malformed APPLY_SGATE instruction. Usage: APPLY_SGATE <qubit>".into())
            }
        }
        "ADD" | "SUB" | "AND" | "OR" | "XOR" => {
            if tokens.len() == 4 {
                let dest = tokens[1].to_string();
                let src1 = tokens[2].to_string();
                let src2 = tokens[3].to_string();
                match opcode.as_str() {
                    "ADD" => Ok(Instruction::Add(dest, src1, src2)),
                    "SUB" => Ok(Instruction::Sub(dest, src1, src2)),
                    "AND" => Ok(Instruction::And(dest, src1, src2)),
                    "OR" => Ok(Instruction::Or(dest, src1, src2)),
                    "XOR" => Ok(Instruction::Xor(dest, src1, src2)),
                    _ => unreachable!(),
                }
            } else {
                Err(format!("Malformed {} instruction. Usage: {} <dest> <src1> <src2>", opcode, opcode))
            }
        }
        "NOT" => {
            if tokens.len() == 3 {
                let dest = tokens[1].to_string();
                let src = tokens[2].to_string();
                Ok(Instruction::Not(dest, src))
            } else {
                Err("Malformed NOT instruction. Usage: NOT <dest> <src>".into())
            }
        }
        "JUMP" => {
            if tokens.len() == 2 {
                Ok(Instruction::Jump(tokens[1].to_string()))
            } else {
                Err("Malformed JUMP instruction. Usage: JUMP <label>".into())
            }
        }
        "JUMPIFZERO" => {
            if tokens.len() == 3 {
                Ok(Instruction::JumpIfZero(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("Malformed JUMPIFZERO instruction. Usage: JUMPIFZERO <cond> <label>".into())
            }
        }
        "JUMPIFONE" => {
            if tokens.len() == 3 {
                Ok(Instruction::JumpIfOne(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("Malformed JUMPIFONE instruction. Usage: JUMPIFONE <cond> <label>".into())
            }
        }
        "CALL" => {
            if tokens.len() == 2 {
                Ok(Instruction::Call(tokens[1].to_string()))
            } else {
                Err("Malformed CALL instruction. Usage: CALL <label>".into())
            }
        }
        "RETURN" => {
            if tokens.len() == 1 {
                Ok(Instruction::Return)
            } else {
                Err("Malformed RETURN instruction. Usage: RETURN".into())
            }
        }
        "PUSH" => {
            if tokens.len() == 2 {
                Ok(Instruction::Push(tokens[1].to_string()))
            } else {
                Err("Malformed PUSH instruction. Usage: PUSH <reg>".into())
            }
        }
        "POP" => {
            if tokens.len() == 2 {
                Ok(Instruction::Pop(tokens[1].to_string()))
            } else {
                Err("Malformed POP instruction. Usage: POP <reg>".into())
            }
        }
        "LOADMEM" => {
            if tokens.len() == 3 {
                Ok(Instruction::LoadMem(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("Malformed LOADMEM instruction. Usage: LOADMEM <reg> <mem_addr>".into())
            }
        }
        "STOREMEM" => {
            if tokens.len() == 3 {
                Ok(Instruction::StoreMem(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("Malformed STOREMEM instruction. Usage: STOREMEM <reg> <mem_addr>".into())
            }
        }
        "LOADCLASSICAL" => {
            if tokens.len() == 3 {
                Ok(Instruction::LoadClassical(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("Malformed LOADCLASSICAL instruction. Usage: LOADCLASSICAL <reg> <var>".into())
            }
        }
        "STORECLASSICAL" => {
            if tokens.len() == 3 {
                Ok(Instruction::StoreClassical(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("Malformed STORECLASSICAL instruction. Usage: STORECLASSICAL <reg> <var>".into())
            }
        }
        "ENTANGLEMULTI" => {
            if tokens.len() >= 2 {
                let mut qubits = Vec::new();
                for &q in &tokens[1..] {
                    qubits.push(parse_u8(q)?);
                }
                Ok(Instruction::EntangleMulti(qubits))
            } else {
                Err("Malformed ENTANGLEMULTI instruction. Usage: ENTANGLEMULTI <qubit1> <qubit2> ...".into())
            }
        }
        "ENTANGLE_CLUSTER" => {
            if tokens.len() >= 2 {
                let mut qubits = Vec::new();
                for &q in &tokens[1..] {
                    qubits.push(parse_u8(q)?);
                }
                Ok(Instruction::EntangleCluster(qubits))
            } else {
                Err("Malformed ENTANGLE_CLUSTER instruction. Usage: ENTANGLE_CLUSTER <qubit1> <qubit2> ...".into())
            }
        }
        "APPLY_ROTATION" => {
            if tokens.len() == 4 {
                let n = parse_u8(tokens[1])?;
                let axis = parse_axis(tokens[2])?;
                let angle = parse_f64(tokens[3])?;
                Ok(Instruction::ApplyRotation(n, axis, angle))
            } else {
                Err("Malformed APPLY_ROTATION instruction. Usage: APPLY_ROTATION <qubit> <X|Y|Z> <angle>".into())
            }
        }
        "APPLY_MULTI_QUBIT_ROTATION" => {
            if tokens.len() >= 4 {
                let axis = parse_axis(tokens[2])?;
                let qubits = tokens[1].split(',').map(parse_u8).collect::<Result<Vec<u8>, _>>()?;
                let angles = tokens[3..].iter().map(|s| parse_f64(s)).collect::<Result<Vec<f64>, _>>()?;
                Ok(Instruction::ApplyMultiQubitRotation(qubits, axis, angles))
            } else {
                Err("Malformed APPLY_MULTI_QUBIT_ROTATION instruction. Usage: APPLY_MULTI_QUBIT_ROTATION <q1,q2,...> <X|Y|Z> <angle1> <angle2> ...".into())
            }
        }
        "RESETALL" => {
            if tokens.len() == 1 {
                Ok(Instruction::ResetAll)
            } else {
                Err("Malformed RESETALL instruction. Usage: RESETALL".into())
            }
        }
        "PHOTONROUTE" => {
            if tokens.len() == 4 {
                let n = parse_u8(tokens[1])?;
                let source = tokens[2].to_string();
                let dest = tokens[3].to_string();
                Ok(Instruction::PhotonRoute(n, source, dest))
            } else {
                Err("Malformed PHOTONROUTE instruction. Usage: PHOTONROUTE <qubit> <source> <dest>".into())
            }
        }
        "SYNC" => {
            if tokens.len() == 1 {
                Ok(Instruction::Sync)
            } else {
                Err("Malformed SYNC instruction. Usage: SYNC".into())
            }
        }
        "ENTANGLESWAP" => {
            if tokens.len() == 5 {
                let a = parse_u8(tokens[1])?;
                let b = parse_u8(tokens[2])?;
                let c = parse_u8(tokens[3])?;
                let d = parse_u8(tokens[4])?;
                Ok(Instruction::EntangleSwap(a, b, c, d))
            } else {
                Err("Malformed ENTANGLESWAP instruction. Usage: ENTANGLESWAP <q1> <q2> <q3> <q4>".into())
            }
        }
        "ERRORCORRECT" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let code = tokens[2].to_string();
                Ok(Instruction::ErrorCorrect(n, code))
            } else {
                Err("Malformed ERRORCORRECT instruction. Usage: ERRORCORRECT <qubit> <code>".into())
            }
        }
        "PHOTONCOUNT" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let reg = tokens[2].to_string();
                Ok(Instruction::PhotonCount(n, reg))
            } else {
                Err("Malformed PHOTONCOUNT instruction. Usage: PHOTONCOUNT <qubit> <reg>".into())
            }
        }
        "APPLYDISPLACEMENT" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let val = parse_f64(tokens[2])?;
                Ok(Instruction::ApplyDisplacement(n, val))
            } else {
                Err("Malformed APPLYDISPLACEMENT instruction. Usage: APPLYDISPLACEMENT <qubit> <value>".into())
            }
        }
        "APPLYSQUEEZING" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let val = parse_f64(tokens[2])?;
                Ok(Instruction::ApplySqueezing(n, val))
            } else {
                Err("Malformed APPLYSQUEEZING instruction. Usage: APPLYSQUEEZING <qubit> <value>".into())
            }
        }
        "MEASUREPARITY" => {
            if tokens.len() == 2 {
                let n = parse_u8(tokens[1])?;
                Ok(Instruction::MeasureParity(n))
            } else {
                Err("Malformed MEASUREPARITY instruction. Usage: MEASUREPARITY <qubit>".into())
            }
        }
        "MEASUREWITHDELAY" => {
            if tokens.len() == 4 {
                let n = parse_u8(tokens[1])?;
                let delay = parse_u64(tokens[2])?;
                let reg = tokens[3].to_string();
                Ok(Instruction::MeasureWithDelay(n, delay, reg))
            } else {
                Err("Malformed MEASUREWITHDELAY instruction. Usage: MEASUREWITHDELAY <qubit> <delay> <reg>".into())
            }
        }
        "OPTICALSWITCHCONTROL" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let flag = parse_bool(tokens[2])?;
                Ok(Instruction::OpticalSwitchControl(n, flag))
            } else {
                Err("Malformed OPTICALSWITCHCONTROL instruction. Usage: OPTICALSWITCHCONTROL <qubit> <ON|OFF>".into())
            }
        }
        "PHOTONLOSSSIMULATE" => {
            if tokens.len() == 4 {
                let q = parse_u8(tokens[1])?;
                let prob = parse_f64(tokens[2])?;
                let seed = parse_u64(tokens[3])?;
                Ok(Instruction::PhotonLossSimulate(q, prob, seed))
            } else {
                Err("Malformed PHOTONLOSSSIMULATE instruction. Usage: PHOTONLOSSSIMULATE <qubit> <probability> <seed>".into())
            }
        }
        "PHOTONLOSSCORRECTION" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let reg = tokens[2].to_string();
                Ok(Instruction::PhotonLossCorrection(q, reg))
            } else {
                Err("Malformed PHOTONLOSSCORRECTION instruction. Usage: PHOTONLOSSCORRECTION <qubit> <reg>".into())
            }
        }
        "PHOTONADDITION" => {
            if tokens.len() == 2 {
                let q = parse_u8(tokens[1])?;
                Ok(Instruction::PhotonAddition(q))
            } else {
                Err("Malformed PHOTONADDITION instruction. Usage: PHOTONADDITION <qubit>".into())
            }
        }
        "APPLYPHOTONSUBTRACTION" => {
            if tokens.len() == 2 {
                let q = parse_u8(tokens[1])?;
                Ok(Instruction::ApplyPhotonSubtraction(q))
            } else {
                Err("Malformed APPLYPHOTONSUBTRACTION instruction. Usage: APPLYPHOTONSUBTRACTION <qubit>".into())
            }
        }
        "PHOTONEMISSIONPATTERN" => {
            if tokens.len() == 4 {
                let q = parse_u8(tokens[1])?;
                let reg = tokens[2].to_string();
                let cycles = parse_u64(tokens[3])?;
                Ok(Instruction::PhotonEmissionPattern(q, reg, cycles))
            } else {
                Err("Malformed PHOTONEMISSIONPATTERN instruction. Usage: PHOTONEMISSIONPATTERN <qubit> <reg> <cycles>".into())
            }
        }
        "PHOTONDETECTWITHTHRESHOLD" => {
            if tokens.len() == 4 {
                let q = parse_u8(tokens[1])?;
                let threshold = parse_u64(tokens[2])?;
                let reg = tokens[3].to_string();
                Ok(Instruction::PhotonDetectWithThreshold(q, threshold, reg))
            } else {
                Err("Malformed PHOTONDETECTWITHTHRESHOLD instruction. Usage: PHOTONDETECTWITHTHRESHOLD <qubit> <threshold> <reg>".into())
            }
        }
        "PHOTONDETECTCOINCIDENCE" => {
            if tokens.len() >= 3 {
                let qubits = tokens[1].split(',').map(parse_u8).collect::<Result<Vec<u8>, _>>()?;
                let reg = tokens[2].to_string();
                Ok(Instruction::PhotonDetectCoincidence(qubits, reg))
            } else {
                Err("Malformed PHOTONDETECTCOINCIDENCE instruction. Usage: PHOTONDETECTCOINCIDENCE <q1,q2,...> <reg>".into())
            }
        }
        "SINGLEPHOTONSOURCEON" => {
            if tokens.len() == 2 {
                let q = parse_u8(tokens[1])?;
                Ok(Instruction::SinglePhotonSourceOn(q))
            } else {
                Err("Malformed SINGLEPHOTONSOURCEON instruction. Usage: SINGLEPHOTONSOURCEON <qubit>".into())
            }
        }
        "SINGLEPHOTONSOURCEOFF" => {
            if tokens.len() == 2 {
                let q = parse_u8(tokens[1])?;
                Ok(Instruction::SinglePhotonSourceOff(q))
            } else {
                Err("Malformed SINGLEPHOTONSOURCEOFF instruction. Usage: SINGLEPHOTONSOURCEOFF <qubit>".into())
            }
        }
        "PHOTONBUNCHINGCONTROL" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let flag = parse_bool(tokens[2])?;
                Ok(Instruction::PhotonBunchingControl(q, flag))
            } else {
                Err("Malformed PHOTONBUNCHINGCONTROL instruction. Usage: PHOTONBUNCHINGCONTROL <qubit> <true|false>".into())
            }
        }
        "APPLYLINEAROPTICALTRANSFORM" => {
            if tokens.len() >= 5 {
                let name = tokens[1].to_string();
                let inputs = tokens[2].split(',').map(parse_u8).collect::<Result<Vec<u8>, _>>()?;
                let outputs = tokens[3].split(',').map(parse_u8).collect::<Result<Vec<u8>, _>>()?;
                let size = tokens[4].parse::<usize>().map_err(|_| format!("Invalid usize '{}'", tokens[4]))?;
                Ok(Instruction::ApplyLinearOpticalTransform(name, inputs, outputs, size))
            } else {
                Err("Malformed APPLYLINEAROPTICALTRANSFORM instruction. Usage: APPLYLINEAROPTICALTRANSFORM <name> <in1,in2,...> <out1,out2,...> <size>".into())
            }
        }
        "CONTROLLEDSWAP" => {
            if tokens.len() == 4 {
                let c = parse_u8(tokens[1])?;
                let t1 = parse_u8(tokens[2])?;
                let t2 = parse_u8(tokens[3])?;
                Ok(Instruction::ControlledSwap(c, t1, t2))
            } else {
                Err("Malformed CONTROLLEDSWAP instruction. Usage: CONTROLLEDSWAP <control> <target1> <target2>".into())
            }
        }
        "APPLYDISPLACEMENTFEEDBACK" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let reg = tokens[2].to_string();
                Ok(Instruction::ApplyDisplacementFeedback(q, reg))
            } else {
                Err("Malformed APPLYDISPLACEMENTFEEDBACK instruction. Usage: APPLYDISPLACEMENTFEEDBACK <qubit> <reg>".into())
            }
        }
        "APPLYNONLINEARSIGMA" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let val = parse_f64(tokens[2])?;
                Ok(Instruction::ApplyNonlinearSigma(q, val))
            } else {
                Err("Malformed APPLYNONLINEARSIGMA instruction. Usage: APPLYNONLINEARSIGMA <qubit> <value>".into())
            }
        }
        "APPLYSQUEEZINGFEEDBACK" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let reg = tokens[2].to_string();
                Ok(Instruction::ApplySqueezingFeedback(q, reg))
            } else {
                Err("Malformed APPLYSQUEEZINGFEEDBACK instruction. Usage: APPLYSQUEEZINGFEEDBACK <qubit> <reg>".into())
            }
        }
        "APPLYMEASUREMENTBASISCHANGE" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let basis = tokens[2].to_string();
                Ok(Instruction::ApplyMeasurementBasisChange(q, basis))
            } else {
                Err("Malformed APPLYMEASUREMENTBASISCHANGE instruction. Usage: APPLYMEASUREMENTBASISCHANGE <qubit> <basis>".into())
            }
        }
        "CONTROLLEDPHASEROTATION" => {
            if tokens.len() == 4 {
                let c = parse_u8(tokens[1])?;
                let t = parse_u8(tokens[2])?;
                let angle = parse_f64(tokens[3])?;
                Ok(Instruction::ControlledPhaseRotation(c, t, angle))
            } else {
                Err("Malformed CONTROLLEDPHASEROTATION instruction. Usage: CONTROLLEDPHASEROTATION <control> <target> <angle>".into())
            }
        }
        "APPLYFEEDFORWARDGATE" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let reg = tokens[2].to_string();
                Ok(Instruction::ApplyFeedforwardGate(q, reg))
            } else {
                Err("Malformed APPLYFEEDFORWARDGATE instruction. Usage: APPLYFEEDFORWARDGATE <qubit> <reg>".into())
            }
        }
        "MEASUREINBASIS" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let basis = tokens[2].to_string();
                Ok(Instruction::MeasureInBasis(n, basis))
            } else {
                Err("Malformed MEASUREINBASIS instruction. Usage: MEASUREINBASIS <qubit> <basis>".into())
            }
        }
        "DECOHERENCEPROTECT" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                let duration = parse_u64(tokens[2])?;
                Ok(Instruction::DecoherenceProtect(n, duration))
            } else {
                Err("Malformed DECOHERENCEPROTECT instruction. Usage: DECOHERENCEPROTECT <qubit> <duration>".into())
            }
        }
        "FEEDBACKCONTROL" => {
            if tokens.len() == 3 {
                let n = parse_u8(tokens[1])?;
                Ok(Instruction::FeedbackControl(n, tokens[2].to_string()))
            } else {
                Err("Malformed FEEDBACKCONTROL instruction. Usage: FEEDBACKCONTROL <qubit> <signal>".into())
            }
        }
        "APPLYC_PHASE" => {
            if tokens.len() == 4 {
                let a = parse_u8(tokens[1])?;
                let b = parse_u8(tokens[2])?;
                let angle = parse_f64(tokens[3])?;
                Ok(Instruction::ApplyCPhase(a, b, angle))
            } else {
                Err("Malformed APPLYC_PHASE instruction. Usage: APPLYC_PHASE <q1> <q2> <angle>".into())
            }
        }
        "APPLYKERRNONLINEARITY" => {
            if tokens.len() == 4 {
                let q = parse_u8(tokens[1])?;
                let strength = parse_f64(tokens[2])?;
                let duration = parse_u64(tokens[3])?;
                Ok(Instruction::ApplyKerrNonlinearity(q, strength, duration))
            } else {
                Err("Malformed APPLYKERRNONLINEARITY instruction. Usage: APPLYKERRNONLINEARITY <qubit> <strength> <duration>".into())
            }
        }
        "ENTANGLESWAPMEASURE" => {
            if tokens.len() == 6 {
                let a = parse_u8(tokens[1])?;
                let b = parse_u8(tokens[2])?;
                let c = parse_u8(tokens[3])?;
                let d = parse_u8(tokens[4])?;
                let label = tokens[5].to_string();
                Ok(Instruction::EntangleSwapMeasure(a, b, c, d, label))
            } else {
                Err("Malformed ENTANGLESWAPMEASURE instruction. Usage: ENTANGLESWAPMEASURE <q1> <q2> <q3> <q4> <label>".into())
            }
        }
        "ENTANGLEWITHCLASSICALFEEDBACK" => {
            if tokens.len() == 4 {
                let q1 = parse_u8(tokens[1])?;
                let q2 = parse_u8(tokens[2])?;
                let signal = tokens[3].to_string();
                Ok(Instruction::EntangleWithClassicalFeedback(q1, q2, signal))
            } else {
                Err("Malformed ENTANGLEWITHCLASSICALFEEDBACK instruction. Usage: ENTANGLEWITHCLASSICALFEEDBACK <q1> <q2> <signal>".into())
            }
        }
        "OPTICALROUTING" => {
            if tokens.len() == 3 {
                let from = parse_u8(tokens[1])?;
                let to = parse_u8(tokens[2])?;
                Ok(Instruction::OpticalRouting(from, to))
            } else {
                Err("Malformed OPTICALROUTING instruction. Usage: OPTICALROUTING <from_qubit> <to_qubit>".into())
            }
        }
        "ENTANGLEDISTRIBUTED" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let node = tokens[2].to_string();
                Ok(Instruction::EntangleDistributed(q, node))
            } else {
                Err("Malformed ENTANGLEDISTRIBUTED instruction. Usage: ENTANGLEDISTRIBUTED <qubit> <remote_node>".into())
            }
        }
        "SETOPTICALATTENUATION" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let attenuation = parse_f64(tokens[2])?;
                Ok(Instruction::SetOpticalAttenuation(q, attenuation))
            } else {
                Err("Malformed SETOPTICALATTENUATION instruction. Usage: SETOPTICALATTENUATION <qubit> <attenuation_value>".into())
            }
        }
        "DYNAMICPHASECOMPENSATION" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let phase = parse_f64(tokens[2])?;
                Ok(Instruction::DynamicPhaseCompensation(q, phase))
            } else {
                Err("Malformed DYNAMICPHASECOMPENSATION instruction. Usage: DYNAMICPHASECOMPENSATION <qubit> <phase_shift>".into())
            }
        }
        "OPTICALDELAYLINECONTROL" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let delay = parse_u64(tokens[2])?;
                Ok(Instruction::OpticalDelayLineControl(q, delay))
            } else {
                Err("Malformed OPTICALDELAYLINECONTROL instruction. Usage: OPTICALDELAYLINECONTROL <qubit> <delay_time>".into())
            }
        }
        "CROSSPHASEMODULATION" => {
            if tokens.len() == 4 {
                let control = parse_u8(tokens[1])?;
                let target = parse_u8(tokens[2])?;
                let strength = parse_f64(tokens[3])?;
                Ok(Instruction::CrossPhaseModulation(control, target, strength))
            } else {
                Err("Malformed CROSSPHASEMODULATION instruction. Usage: CROSSPHASEMODULATION <control_qubit> <target_qubit> <strength>".into())
            }
        }
        "APPLYDISPLACEMENTOPERATOR" => {
            if tokens.len() == 4 {
                let q = parse_u8(tokens[1])?;
                let alpha = parse_f64(tokens[2])?;
                let duration = parse_u64(tokens[3])?;
                Ok(Instruction::ApplyDisplacementOperator(q, alpha, duration))
            } else {
                Err("Malformed APPLYDISPLACEMENTOPERATOR instruction. Usage: APPLYDISPLACEMENTOPERATOR <qubit> <alpha> <duration>".into())
            }
        }
        "QUANTUMSTATETOMOGRAPHY" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let basis = tokens[2].to_string();
                Ok(Instruction::QuantumStateTomography(q, basis))
            } else {
                Err("Malformed QUANTUMSTATETOMOGRAPHY instruction. Usage: QUANTUMSTATETOMOGRAPHY <qubit> <basis>".into())
            }
        }
        "BELLSTATEVERIFICATION" => {
            if tokens.len() == 4 {
                let q1 = parse_u8(tokens[1])?;
                let q2 = parse_u8(tokens[2])?;
                let mode = tokens[3].to_string();
                Ok(Instruction::BellStateVerification(q1, q2, mode))
            } else {
                Err("Malformed BELLSTATEVERIFICATION instruction. Usage: BELLSTATEVERIFICATION <q1> <q2> <mode>".into())
            }
        }
        "QUANTUMZENOEFFECT" => {
            if tokens.len() == 4 {
                let q = parse_u8(tokens[1])?;
                let freq = parse_u64(tokens[2])?;
                let duration = parse_u64(tokens[3])?;
                Ok(Instruction::QuantumZenoEffect(q, freq, duration))
            } else {
                Err("Malformed QUANTUMZENOEFFECT instruction. Usage: QUANTUMZENOEFFECT <qubit> <frequency> <duration>".into())
            }
        }
        "APPLYNONLINEARPHASESHIFT" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let phase = parse_f64(tokens[2])?;
                Ok(Instruction::ApplyNonlinearPhaseShift(q, phase))
            } else {
                Err("Malformed APPLYNONLINEARPHASESHIFT instruction. Usage: APPLYNONLINEARPHASESHIFT <qubit> <phase_shift>".into())
            }
        }
        "APPLYQNDMEASUREMENT" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let mode = tokens[2].to_string();
                Ok(Instruction::ApplyQndMeasurement(q, mode))
            } else {
                Err("Malformed APPLYQNDMEASUREMENT instruction. Usage: APPLYQNDMEASUREMENT <qubit> <mode>".into())
            }
        }
        "ERRORSYNDROME" => {
            if tokens.len() == 4 {
                let q = parse_u8(tokens[1])?;
                Ok(Instruction::ErrorSyndrome(q, tokens[2].to_string(), tokens[3].to_string()))
            } else {
                Err("Malformed ERRORSYNDROME instruction. Usage: ERRORSYNDROME <qubit> <type> <result_reg>".into())
            }
        }
        "PHOTONNUMBERRESOLVINGDETECTION" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let result_reg = tokens[2].to_string();
                Ok(Instruction::PhotonNumberResolvingDetection(q, result_reg))
            } else {
                Err("Malformed PHOTONNUMBERRESOLVINGDETECTION instruction. \
                     Usage: PHOTONNUMBERRESOLVINGDETECTION <qubit> <result_reg>"
                    .into())
            }
        }
        "TIMEDELAY" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let delay = parse_u64(tokens[2])?;
                Ok(Instruction::TimeDelay(q, delay))
            } else {
                Err("Malformed TIMEDELAY instruction. Usage: TIMEDELAY <qubit> <cycles>".into())
            }
        }
        _ => Err(format!("Unknown instruction: {}", tokens[0]).into()),
    }
}

impl Instruction {
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Instruction::QInit(q) => vec![0x01, *q],
            Instruction::InitQubit(q) => vec![0x01, *q],
            Instruction::ApplyGate(gate, q) => {
                let mut v = vec![0x02, *q];
                let mut gate_bytes = gate.as_bytes().to_vec();
                gate_bytes.resize(8, 0); // pad/truncate to 8 bytes
                v.extend(gate_bytes);
                v
            }
            Instruction::QMeas(q) => vec![0x03, *q],
            Instruction::Measure(q) => vec![0x03, *q],
            Instruction::CharLoad(q, val) => vec![0x04, *q, *val],
            // will add more if needed
            _ => vec![],
        }
    }
}