#![allow(dead_code)] // use for debug / testing
//#![deny(dead_code)] // use for release
#![allow(unused_variables)]

#[derive(Debug)]
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

    // CHARPRINTING!!!
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
    let parse_u16 = |s: &str| s.parse::<u16>().map_err(|_| format!("invalid u16 '{}'", s));
    let parse_i16 = |s: &str| s.parse::<i16>().map_err(|_| format!("invalid i16 '{}'", s));
    let parse_u64 = |s: &str| s.parse::<u64>().map_err(|_| format!("invalid u64 '{}'", s));
    let parse_f64 = |s: &str| s.parse::<f64>().map_err(|_| format!("invalid f64 '{}'", s));
    let parse_i64 = |s: &str| s.parse::<i64>().map_err(|_| format!("invalid i64 '{}'", s));
    let parse_bool = |s: &str| match s.to_uppercase().as_str() {
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

    match op.as_str() {
        // char printing
        "CHARLOAD" => {
            if tokens.len() == 3 {
                let reg = parse_u8(tokens[1])?;
                let val_str = tokens[2];
                if val_str.len() == 3 && val_str.starts_with('\'') && val_str.ends_with('\'') {
                    let ch = val_str.chars().nth(1).unwrap();
                    Ok(Instruction::CharLoad(reg, ch as u8))
                } else {
                    Err("usage: charload <reg> '<char>'".into())
                }
            } else {
                Err("usage: charload <reg> '<char>'".into())
            }
        }
        "CHAROUT" => {
            if tokens.len() == 2 {
                let reg = parse_u8(tokens[1])?;
                Ok(Instruction::CharOut(reg))
            } else {
                Err("usage: charout <reg>".into())
            }
        }
        // regset
        "REGSET" => {
            if tokens.len() == 3 {
                let reg = parse_u8(tokens[1])?;
                let val = parse_f64(tokens[2])?;
                Ok(Instruction::RegSet(reg, val))
            } else {
                Err("usage: regset <reg> <float_value>".into())
            }
        }
        // loop stuff
        "LOOPSTART" => {
            // Changed from "LOOP" to "LOOPSTART" for exact match
            if tokens.len() == 2 {
                let reg = parse_u8(tokens[1])?;
                Ok(Instruction::LoopStart(reg))
            } else {
                Err("loopstart <reg>".into()) // Updated usage message
            }
        }
        "ENDLOOP" => {
            if tokens.len() == 1 {
                Ok(Instruction::LoopEnd)
            } else {
                Err("endloop".into())
            }
        }
        // normal qoa base stuff
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
        "QGATE" => {
            if tokens.len() >= 3 {
                let q1 = parse_u8(tokens[1])?;
                let gate_name = tokens[2].to_uppercase();
                match gate_name.as_str() {
                    "CZ" => {
                        if tokens.len() == 4 {
                            let q2 = parse_u8(tokens[3])?;
                            Ok(Instruction::CZ(q1, q2))
                        } else {
                            Err("usage: qgate <q1> cz <q2>".into())
                        }
                    }
                    _ => {
                        // Removed specific "H" handling here
                        // default for single qubit gates
                        if tokens.len() == 3 {
                            Ok(Instruction::ApplyGate(gate_name, q1))
                        } else {
                            Err("usage: qgate <qubit> <gate>".into())
                        }
                    }
                }
            } else {
                Err("usage: qgate <qubit> <gate> or qgate <q1> cz <q2>".into())
            }
        }
        "H" | "AH" | "APPLYHADAMARD" => {
            // Modified: added "H" and "AH"
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
        "RESETALL" => {
            if tokens.len() == 1 {
                Ok(Instruction::ResetAll)
            } else {
                Err("resetall".into())
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
                    tokens[5].to_string(),
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
                    tokens[3].to_string(),
                ))
            } else {
                Err("entanglewithclassicalfeedback <q1> <q2> <signal>".into())
            }
        }
        "ENTANGLEDISTRIBUTED" => {
            if tokens.len() == 3 {
                Ok(EntangleDistributed(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                ))
            } else {
                Err("entangledistributed <qubit> <node>".into())
            }
        }
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
                    tokens[2].to_string(),
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
                    tokens[2].to_string(),
                ))
            } else {
                Err("applymeasurementbasischange <q> <basis>".into())
            }
        }
        "LOAD" => {
            if tokens.len() == 3 {
                Ok(Load(parse_u8(tokens[1])?, tokens[2].to_string()))
            } else {
                Err("load <qubit> <var>".into())
            }
        }
        "STORE" => {
            if tokens.len() == 3 {
                Ok(Store(parse_u8(tokens[1])?, tokens[2].to_string()))
            } else {
                Err("store <qubit> <var>".into())
            }
        }
        "LOADMEM" => {
            if tokens.len() == 3 {
                Ok(LoadMem(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("loadmem <reg> <mem>".into())
            }
        }
        "STOREMEM" => {
            if tokens.len() == 3 {
                Ok(StoreMem(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("storemem <reg> <mem>".into())
            }
        }
        "LOADCLASSICAL" => {
            if tokens.len() == 3 {
                Ok(LoadClassical(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("loadclassical <reg> <var>".into())
            }
        }
        "STORECLASSICAL" => {
            if tokens.len() == 3 {
                Ok(StoreClassical(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("storeclassical <reg> <var>".into())
            }
        }
        "ADD" => {
            if tokens.len() == 4 {
                Ok(Add(
                    tokens[1].to_string(),
                    tokens[2].to_string(),
                    tokens[3].to_string(),
                ))
            } else {
                Err("add <dst> <src1> <src2>".into())
            }
        }
        "SUB" => {
            if tokens.len() == 4 {
                Ok(Sub(
                    tokens[1].to_string(),
                    tokens[2].to_string(),
                    tokens[3].to_string(),
                ))
            } else {
                Err("sub <dst> <src1> <src2>".into())
            }
        }
        "AND" => {
            if tokens.len() == 4 {
                Ok(And(
                    tokens[1].to_string(),
                    tokens[2].to_string(),
                    tokens[3].to_string(),
                ))
            } else {
                Err("and <dst> <src1> <src2>".into())
            }
        }
        "OR" => {
            if tokens.len() == 4 {
                Ok(Or(
                    tokens[1].to_string(),
                    tokens[2].to_string(),
                    tokens[3].to_string(),
                ))
            } else {
                Err("or <dst> <src1> <src2>".into())
            }
        }
        "XOR" => {
            if tokens.len() == 4 {
                Ok(Xor(
                    tokens[1].to_string(),
                    tokens[2].to_string(),
                    tokens[3].to_string(),
                ))
            } else {
                Err("xor <dst> <src1> <src2>".into())
            }
        }
        "NOT" => {
            if tokens.len() == 2 {
                Ok(Not(tokens[1].to_string()))
            } else {
                Err("not <reg>".into())
            }
        }
        "PUSH" => {
            if tokens.len() == 2 {
                Ok(Push(tokens[1].to_string()))
            } else {
                Err("push <reg>".into())
            }
        }
        "POP" => {
            if tokens.len() == 2 {
                Ok(Pop(tokens[1].to_string()))
            } else {
                Err("pop <reg>".into())
            }
        }
        "JUMP" => {
            if tokens.len() == 2 {
                Ok(Jump(tokens[1].to_string()))
            } else {
                Err("jump <label>".into())
            }
        }
        "JUMPIFZERO" => {
            if tokens.len() == 3 {
                Ok(JumpIfZero(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("jumpifzero <cond_reg> <label>".into())
            }
        }
        "JUMPIFONE" => {
            if tokens.len() == 3 {
                Ok(JumpIfOne(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("jumpifone <cond_reg> <label>".into())
            }
        }
        "CALL" => {
            if tokens.len() == 2 {
                Ok(Call(tokens[1].to_string()))
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
        "MEASUREINBASIS" => {
            if tokens.len() == 3 {
                Ok(MeasureInBasis(parse_u8(tokens[1])?, tokens[2].to_string()))
            } else {
                Err("measureinbasis <qubit> <basis>".into())
            }
        }
        "SET_PHASE" => {
            if tokens.len() == 3 {
                Ok(SetPhase(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("set_phase <qubit> <angle>".into())
            }
        }
        // ionq isa gates
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
        "APPLYQNDMEASUREMENT" => {
            if tokens.len() == 3 {
                Ok(ApplyQndMeasurement(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                ))
            } else {
                Err("applyqndmeasurement <qubit> <result_reg>".into())
            }
        }
        "ERRORCORRECT" => {
            if tokens.len() == 3 {
                Ok(ErrorCorrect(parse_u8(tokens[1])?, tokens[2].to_string()))
            } else {
                Err("errorcorrect <qubit> <syndrome_type>".into())
            }
        }
        "ERRORSYNDROME" => {
            if tokens.len() == 4 {
                Ok(ErrorSyndrome(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                    tokens[3].to_string(),
                ))
            } else {
                Err("errorsyndrome <qubit> <syndrome_type> <result_reg>".into())
            }
        }
        "QUANTUMSTATETOMOGRAPHY" => {
            if tokens.len() == 3 {
                Ok(QuantumStateTomography(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
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
                    tokens[3].to_string(),
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
            // Fixed case of 'APPLYNONLINEARSIgMA' to be all uppercase
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
                // Expecting opcode + 4 arguments
                let transform_name = tokens[1].to_string();
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
                    tokens[2].to_string(),
                ))
            } else {
                Err("photonnumberresolvingdetection <qubit> <result_reg>".into())
            }
        }
        "FEEDBACKCONTROL" => {
            if tokens.len() == 3 {
                Ok(FeedbackControl(parse_u8(tokens[1])?, tokens[2].to_string()))
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
        // Removed the problematic "HADAMARD" match arm due to redundancy with ApplyHadamard
        "SYNC" => {
            if tokens.len() == 1 {
                Ok(Sync)
            } else {
                Err("sync".into())
            }
        }
        _ => Err(format!("unknown instruction '{}'", op)),
    }
}

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
                let mut v = vec![0x87, *q]; // Resolved opcode conflict with JumpIfZero
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
                let mut v = vec![0x02, *q]; // Generic gate opcode
                v.extend(name.as_bytes());
                for _ in name.len()..8 {
                    v.push(0);
                }
                v
            }
            Instruction::Measure(q) => vec![0x32, *q], // Duplicate of QMeas, keeping for now
            Instruction::InitQubit(q) => vec![0x04, *q], // Duplicate of QInit, keeping for now

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

            // Memory/Classical Ops
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

            // Classical flow control
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

            // Misc
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
        }
    }
}
