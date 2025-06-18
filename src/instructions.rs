// not done yet, but seems pretty verbose for now

// all supported instructions in QOA

#![allow(dead_code)] // this makes the compiler shut up
#![allow(unused_variables)]
#[derive(Debug)]

pub enum Instruction {
    // core
    QInit(u8),
    QMeas(u8),
    CharLoad(u8, u8),
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
    InitQubit(u8),
    ApplyGate(String, u8),
    Measure(u8),

    // regset
    RegSet(u8, i64),

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

    // some advanced stuff
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

    // fix hadamard gate error conflicting with char print
    Hadamard(usize),
}

pub fn parse_instruction(line: &str) -> Result<Instruction, String> {
    use Instruction::*;
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err("Empty instruction line".into());
    }
    // skip comment lines starting with ';'
    if trimmed.starts_with(';') {
        return Err("Comment".into());
    }

    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    let op = tokens[0].to_uppercase();

    // Helper closures
    let parse_u8 = |s: &str| s.parse::<u8>().map_err(|_| format!("Invalid u8 '{}'", s));
    let parse_u16 = |s: &str| s.parse::<u16>().map_err(|_| format!("Invalid u16 '{}'", s));
    let parse_i16 = |s: &str| s.parse::<i16>().map_err(|_| format!("Invalid i16 '{}'", s));
    let parse_u64 = |s: &str| s.parse::<u64>().map_err(|_| format!("Invalid u64 '{}'", s));
    let parse_f64 = |s: &str| s.parse::<f64>().map_err(|_| format!("Invalid f64 '{}'", s));
    let parse_i64 = |s: &str| s.parse::<i64>().map_err(|_| format!("Invalid i64 '{}'", s));
    let parse_bool = |s: &str| match s.to_uppercase().as_str() {
        "TRUE" | "ON" => Ok(true),
        "FALSE" | "OFF" => Ok(false),
        _ => Err(format!("Invalid bool '{}'", s)),
    };
    let parse_axis = |s: &str| {
        let c = s.to_uppercase();
        if c == "X" || c == "Y" || c == "Z" {
            Ok(c.chars().next().unwrap())
        } else {
            Err(format!("Invalid axis '{}'", s))
        }
    };

    match op.as_str() {
        // regset
        "REGSET" => {
            if tokens.len() == 3 {
                let reg = parse_u8(tokens[1])?;
                let val = parse_i64(tokens[2])?;
                Ok(Instruction::RegSet(reg, val))
            } else {
                Err("Usage: REGSET <reg> <int_value>".into())
            }
        }

        // loop stuff
        "LOOP" => {
            if tokens.len() == 2 {
                let reg = parse_u8(tokens[1])?;
                Ok(Instruction::LoopStart(reg))
            } else {
                Err("LOOP <reg>".into())
            }
        }
        "ENDLOOP" => {
            if tokens.len() == 1 {
                Ok(Instruction::LoopEnd)
            } else {
                Err("ENDLOOP".into())
            }
        }

        // normal qoa base stuff
        "QINIT" => {
            if tokens.len() == 2 {
                Ok(QInit(parse_u8(tokens[1])?))
            } else {
                Err("QINIT <qubit>".into())
            }
        }
        // "INITQUBIT" => { if tokens.len() == 2 { Ok(InitQubit(parse_u8(tokens[1])?)) } else { Err("INITQUBIT <qubit>".into()) } }
        "QMEAS" => {
            if tokens.len() == 2 {
                Ok(QMeas(parse_u8(tokens[1])?))
            } else {
                Err("QMEAS <qubit>".into())
            }
        }
        "CHARLOAD" => {
            if tokens.len() == 3 {
                Ok(CharLoad(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("CHARLOAD <qubit> <ascii_val>".into())
            }
        }
        // "MEASURE" => { if tokens.len() == 2 { Ok(Measure(parse_u8(tokens[1])?)) } else { Err("MEASURE <qubit>".into()) } }
        "QGATE" => {
            if tokens.len() == 3 {
                let q = parse_u8(tokens[1])?;
                let gate = tokens[2].to_string();
                Ok(Instruction::ApplyGate(gate, q))
            } else {
                Err("QGATE <qubit> <gate>".into())
            }
        }
        "APPLYHADAMARD" => {
            if tokens.len() == 2 {
                Ok(ApplyHadamard(parse_u8(tokens[1])?))
            } else {
                Err("APPLYHADAMARD <qubit>".into())
            }
        }
        "CONTROLLEDNOT" => {
            if tokens.len() == 3 {
                Ok(ControlledNot(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("CONTROLLEDNOT <c> <t>".into())
            }
        }
        "APPLYPHASEFLIP" => {
            if tokens.len() == 2 {
                Ok(ApplyPhaseFlip(parse_u8(tokens[1])?))
            } else {
                Err("APPLYPHASEFLIP <qubit>".into())
            }
        }
        "APPLYBITFLIP" => {
            if tokens.len() == 2 {
                Ok(ApplyBitFlip(parse_u8(tokens[1])?))
            } else {
                Err("APPLYBITFLIP <qubit>".into())
            }
        }
        "APPLYTGATE" => {
            if tokens.len() == 2 {
                Ok(ApplyTGate(parse_u8(tokens[1])?))
            } else {
                Err("APPLYTGATE <qubit>".into())
            }
        }
        "APPLYSGATE" => {
            if tokens.len() == 2 {
                Ok(ApplySGate(parse_u8(tokens[1])?))
            } else {
                Err("APPLYSGATE <qubit>".into())
            }
        }
        "PHASESHIFT" => {
            if tokens.len() == 3 {
                Ok(PhaseShift(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("PHASESHIFT <qubit> <angle>".into())
            }
        }
        "WAIT" => {
            if tokens.len() == 2 {
                Ok(Wait(parse_u64(tokens[1])?))
            } else {
                Err("WAIT <cycles>".into())
            }
        }
        "RESET" => {
            if tokens.len() == 2 {
                Ok(Reset(parse_u8(tokens[1])?))
            } else {
                Err("RESET <qubit>".into())
            }
        }
        "RESETALL" => {
            if tokens.len() == 1 {
                Ok(Instruction::ResetAll)
            } else {
                Err("RESETALL".into())
            }
        }
        "SWAP" => {
            if tokens.len() == 3 {
                Ok(Swap(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("SWAP <q1> <q2>".into())
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
                Err("CONTROLLEDSWAP <c> <t1> <t2>".into())
            }
        }
        "ENTANGLE" => {
            if tokens.len() == 3 {
                Ok(Entangle(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("ENTANGLE <q1> <q2>".into())
            }
        }
        "ENTANGLEBELL" => {
            if tokens.len() == 3 {
                Ok(EntangleBell(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("ENTANGLEBELL <q1> <q2>".into())
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
                Err("ENTANGLEMULTI <q1> <q2> ...".into())
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
                Err("ENTANGLECLUSTER <q1> <q2> ...".into())
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
                Err("ENTANGLESWAP <a> <b> <c> <d>".into())
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
                Err("ENTANGLESWAPMEASURE <a> <b> <c> <d> <label>".into())
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
                Err("ENTANGLEWITHCLASSICALFEEDBACK <q1> <q2> <signal>".into())
            }
        }
        "ENTANGLEDISTRIBUTED" => {
            if tokens.len() == 3 {
                Ok(EntangleDistributed(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                ))
            } else {
                Err("ENTANGLEDISTRIBUTED <qubit> <node>".into())
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
                Err("APPLYROTATION <q> <X|Y|Z> <angle>".into())
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
                Err("APPLYMULTIQUBITROTATION <q1,q2,...> <X|Y|Z> <a1> <a2> ...".into())
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
                Err("CONTROLLEDPHASEROTATION <c> <t> <angle>".into())
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
                Err("APPLYC_PHASE <q1> <q2> <angle>".into())
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
                Err("APPLYKERRNONLINEARITY <q> <strength> <duration>".into())
            }
        }
        "APPLYFEEDFORWARDGATE" => {
            if tokens.len() == 3 {
                Ok(ApplyFeedforwardGate(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                ))
            } else {
                Err("APPLYFEEDFORWARDGATE <q> <reg>".into())
            }
        }
        "DECOHERENCEPROTECT" => {
            if tokens.len() == 3 {
                Ok(DecoherenceProtect(
                    parse_u8(tokens[1])?,
                    parse_u64(tokens[2])?,
                ))
            } else {
                Err("DECOHERENCEPROTECT <q> <duration>".into())
            }
        }
        "APPLYMEASUREMENTBASISCHANGE" => {
            if tokens.len() == 3 {
                Ok(ApplyMeasurementBasisChange(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                ))
            } else {
                Err("APPLYMEASUREMENTBASISCHANGE <q> <basis>".into())
            }
        }
        "LOAD" => {
            if tokens.len() == 3 {
                Ok(Load(parse_u8(tokens[1])?, tokens[2].to_string()))
            } else {
                Err("LOAD <qubit> <var>".into())
            }
        }
        "STORE" => {
            if tokens.len() == 3 {
                Ok(Store(parse_u8(tokens[1])?, tokens[2].to_string()))
            } else {
                Err("STORE <qubit> <var>".into())
            }
        }
        "LOADMEM" => {
            if tokens.len() == 3 {
                Ok(LoadMem(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("LOADMEM <reg> <mem>".into())
            }
        }
        "STOREMEM" => {
            if tokens.len() == 3 {
                Ok(StoreMem(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("STOREMEM <reg> <mem>".into())
            }
        }
        "LOADCLASSICAL" => {
            if tokens.len() == 3 {
                Ok(LoadClassical(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("LOADCLASSICAL <reg> <var>".into())
            }
        }
        "STORECLASSICAL" => {
            if tokens.len() == 3 {
                Ok(StoreClassical(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("STORECLASSICAL <reg> <var>".into())
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
                Err("ADD <dst> <src1> <src2>".into())
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
                Err("SUB <dst> <src1> <src2>".into())
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
                Err("AND <dst> <src1> <src2>".into())
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
                Err("OR <dst> <src1> <src2>".into())
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
                Err("XOR <dst> <src1> <src2>".into())
            }
        }
        "NOT" => {
            if tokens.len() == 2 {
                Ok(Not(tokens[1].to_string()))
            } else {
                Err("NOT <reg>".into())
            }
        }
        "PUSH" => {
            if tokens.len() == 2 {
                Ok(Push(tokens[1].to_string()))
            } else {
                Err("PUSH <reg>".into())
            }
        }
        "POP" => {
            if tokens.len() == 2 {
                Ok(Pop(tokens[1].to_string()))
            } else {
                Err("POP <reg>".into())
            }
        }
        "JUMP" => {
            if tokens.len() == 2 {
                Ok(Jump(tokens[1].to_string()))
            } else {
                Err("JUMP <label>".into())
            }
        }
        "JUMPIFZERO" => {
            if tokens.len() == 3 {
                Ok(JumpIfZero(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("JUMPIFZERO <cond> <label>".into())
            }
        }
        "JUMPIFONE" => {
            if tokens.len() == 3 {
                Ok(JumpIfOne(tokens[1].to_string(), tokens[2].to_string()))
            } else {
                Err("JUMPIFONE <cond> <label>".into())
            }
        }
        "CALL" => {
            if tokens.len() == 2 {
                Ok(Call(tokens[1].to_string()))
            } else {
                Err("CALL <label>".into())
            }
        }
        "RETURN" => {
            if tokens.len() == 1 {
                Ok(Return)
            } else {
                Err("RETURN".into())
            }
        }
        "SYNC" => {
            if tokens.len() == 1 {
                Ok(Instruction::Sync)
            } else {
                Err("SYNC".into())
            }
        }
        "TIMEDELAY" => {
            if tokens.len() == 3 {
                Ok(TimeDelay(parse_u8(tokens[1])?, parse_u64(tokens[2])?))
            } else {
                Err("TIMEDELAY <q> <cycles>".into())
            }
        }
        "VERBOSELOG" => {
            if tokens.len() >= 2 {
                Ok(VerboseLog(parse_u8(tokens[1])?, tokens[2..].join(" ")))
            } else {
                Err("VERBOSELOG <qubit> <msg>".into())
            }
        }
        // optical / photonic stuff
        "PHOTONEMIT" => {
            if tokens.len() == 2 {
                Ok(PhotonEmit(parse_u8(tokens[1])?))
            } else {
                Err("PHOTONEMIT <reg>".into())
            }
        }
        "PHOTONDETECT" => {
            if tokens.len() == 2 {
                Ok(PhotonDetect(parse_u8(tokens[1])?))
            } else {
                Err("PHOTONDETECT <reg>".into())
            }
        }
        "PHOTONCOUNT" => {
            if tokens.len() == 3 {
                Ok(PhotonCount(parse_u8(tokens[1])?, tokens[2].to_string()))
            } else {
                Err("PHOTONCOUNT <reg> <dest>".into())
            }
        }
        "PHOTONADDITION" => {
            if tokens.len() == 2 {
                Ok(PhotonAddition(parse_u8(tokens[1])?))
            } else {
                Err("PHOTONADDITION <reg>".into())
            }
        }
        "APPLYPHOTONSUBTRACTION" => {
            if tokens.len() == 2 {
                Ok(ApplyPhotonSubtraction(parse_u8(tokens[1])?))
            } else {
                Err("APPLYPHOTONSUBTRACTION <reg>".into())
            }
        }
        "PHOTONEMISSIONPATTERN" => {
            if tokens.len() == 4 {
                Ok(PhotonEmissionPattern(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                    parse_u64(tokens[3])?,
                ))
            } else {
                Err("PHOTONEMISSIONPATTERN <reg> <pattern> <cycles>".into())
            }
        }
        "PHOTONDETECTWITHTHRESHOLD" => {
            if tokens.len() == 4 {
                Ok(PhotonDetectWithThreshold(
                    parse_u8(tokens[1])?,
                    parse_u64(tokens[2])?,
                    tokens[3].to_string(),
                ))
            } else {
                Err("PHOTONDETECTWITHTHRESHOLD <reg> <thresh> <dest>".into())
            }
        }
        "PHOTONDETECTCOINCIDENCE" => {
            if tokens.len() >= 3 {
                Ok(PhotonDetectCoincidence(
                    tokens[1]
                        .split(',')
                        .map(parse_u8)
                        .collect::<Result<Vec<_>, _>>()?,
                    tokens[2].to_string(),
                ))
            } else {
                Err("PHOTONDETECTCOINCIDENCE <q1,...> <dest>".into())
            }
        }
        "SINGLEPHOTONSOURCEON" => {
            if tokens.len() == 2 {
                Ok(SinglePhotonSourceOn(parse_u8(tokens[1])?))
            } else {
                Err("SINGLEPHOTONSOURCEON <reg>".into())
            }
        }
        "SINGLEPHOTONSOURCEOFF" => {
            if tokens.len() == 2 {
                Ok(SinglePhotonSourceOff(parse_u8(tokens[1])?))
            } else {
                Err("SINGLEPHOTONSOURCEOFF <reg>".into())
            }
        }
        "PHOTONBUNCHINGCONTROL" => {
            if tokens.len() == 3 {
                Ok(PhotonBunchingControl(
                    parse_u8(tokens[1])?,
                    parse_bool(tokens[2])?,
                ))
            } else {
                Err("PHOTONBUNCHINGCONTROL <reg> <on|off>".into())
            }
        }
        "PHOTONROUTE" => {
            if tokens.len() == 4 {
                Ok(PhotonRoute(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                    tokens[3].to_string(),
                ))
            } else {
                Err("PHOTONROUTE <reg> <src> <dst>".into())
            }
        }
        "OPTICALROUTING" => {
            if tokens.len() == 3 {
                Ok(OpticalRouting(parse_u8(tokens[1])?, parse_u8(tokens[2])?))
            } else {
                Err("OPTICALROUTING <src> <dst>".into())
            }
        }
        "SETOPTICALATTENUATION" => {
            if tokens.len() == 3 {
                Ok(SetOpticalAttenuation(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("SETOPTICALATTENUATION <reg> <atten>".into())
            }
        }
        "DYNAMICPHASECOMPENSATION" => {
            if tokens.len() == 3 {
                Ok(DynamicPhaseCompensation(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("DYNAMICPHASECOMPENSATION <reg> <phase>".into())
            }
        }
        "OPTICALDELAYLINECONTROL" => {
            if tokens.len() == 3 {
                Ok(OpticalDelayLineControl(
                    parse_u8(tokens[1])?,
                    parse_u64(tokens[2])?,
                ))
            } else {
                Err("OPTICALDELAYLINECONTROL <reg> <delay>".into())
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
                Err("CROSSPHASEMODULATION <c> <t> <str>".into())
            }
        }
        "APPLYDISPLACEMENT" => {
            if tokens.len() == 3 {
                Ok(ApplyDisplacement(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("APPLYDISPLACEMENT <reg> <val>".into())
            }
        }
        "APPLYDISPLACEMENTFEEDBACK" => {
            if tokens.len() == 3 {
                Ok(ApplyDisplacementFeedback(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                ))
            } else {
                Err("APPLYDISPLACEMENTFEEDBACK <reg> <feed>".into())
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
                Err("APPLYDISPLACEMENTOPERATOR <reg> <alpha> <dur>".into())
            }
        }
        "APPLYSQUEEZING" => {
            if tokens.len() == 3 {
                Ok(ApplySqueezing(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("APPLYSQUEEZING <reg> <val>".into())
            }
        }
        "APPLYSQUEEZINGFEEDBACK" => {
            if tokens.len() == 3 {
                Ok(ApplySqueezingFeedback(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                ))
            } else {
                Err("APPLYSQUEEZINGFEEDBACK <reg> <feed>".into())
            }
        }
        "MEASUREPARITY" => {
            if tokens.len() == 2 {
                Ok(MeasureParity(parse_u8(tokens[1])?))
            } else {
                Err("MEASUREPARITY <reg>".into())
            }
        }
        "MEASUREWITHDELAY" => {
            if tokens.len() == 4 {
                Ok(MeasureWithDelay(
                    parse_u8(tokens[1])?,
                    parse_u64(tokens[2])?,
                    tokens[3].to_string(),
                ))
            } else {
                Err("MEASUREWITHDELAY <reg> <delay> <dest>".into())
            }
        }
        "OPTICALSWITCHCONTROL" => {
            if tokens.len() == 3 {
                Ok(OpticalSwitchControl(
                    parse_u8(tokens[1])?,
                    parse_bool(tokens[2])?,
                ))
            } else {
                Err("OPTICALSWITCHCONTROL <reg> <on|off>".into())
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
                Err("PHOTONLOSSSIMULATE <reg> <prob> <seed>".into())
            }
        }
        "PHOTONLOSSCORRECTION" => {
            if tokens.len() == 3 {
                Ok(PhotonLossCorrection(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                ))
            } else {
                Err("PHOTONLOSSCORRECTION <reg> <feed>".into())
            }
        }
        // advanced stuff
        "APPLYQNDMEASUREMENT" => {
            if tokens.len() == 3 {
                Ok(ApplyQndMeasurement(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                ))
            } else {
                Err("APPLYQNDMEASUREMENT <reg> <mode>".into())
            }
        }
        "ERRORCORRECT" => {
            if tokens.len() == 3 {
                Ok(ErrorCorrect(parse_u8(tokens[1])?, tokens[2].to_string()))
            } else {
                Err("ERRORCORRECT <reg> <code>".into())
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
                Err("ERRORSYNDROME <reg> <type> <res>".into())
            }
        }
        "QUANTUMSTATETOMOGRAPHY" => {
            if tokens.len() == 3 {
                Ok(QuantumStateTomography(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                ))
            } else {
                Err("QUANTUMSTATETOMOGRAPHY <reg> <basis>".into())
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
                Err("BELLSTATEVERIFICATION <q1> <q2> <mode>".into())
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
                Err("QUANTUMZENOEFFECT <reg> <freq> <dur>".into())
            }
        }
        "APPLYNONLINEARPHASESHIFT" => {
            if tokens.len() == 3 {
                Ok(ApplyNonlinearPhaseShift(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("APPLYNONLINEARPHASESHIFT <reg> <phase>".into())
            }
        }
        "APPLYNONLINEARSIGMA" => {
            if tokens.len() == 3 {
                Ok(ApplyNonlinearSigma(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("APPLYNONLINEARSIGMA <reg> <sigma>".into())
            }
        }
        "APPLYLINEAROPTICALTRANSFORM" => {
            if tokens.len() >= 5 {
                let name = tokens[1].to_string();
                let ins = tokens[2]
                    .split(',')
                    .map(parse_u8)
                    .collect::<Result<Vec<_>, _>>()?;
                let outs = tokens[3]
                    .split(',')
                    .map(parse_u8)
                    .collect::<Result<Vec<_>, _>>()?;
                let sz = tokens[4]
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid usize '{}'", tokens[4]))?;
                Ok(ApplyLinearOpticalTransform(name, ins, outs, sz))
            } else {
                Err("APPLYLINEAROPTICALTRANSFORM <name> <ins> <outs> <sz>".into())
            }
        }
        "PHOTONNUMBERRESOLVINGDETECTION" => {
            if tokens.len() == 3 {
                Ok(PhotonNumberResolvingDetection(
                    parse_u8(tokens[1])?,
                    tokens[2].to_string(),
                ))
            } else {
                Err("PHOTONNUMBERRESOLVINGDETECTION <reg> <dest>".into())
            }
        }
        "FEEDBACKCONTROL" => {
            if tokens.len() == 3 {
                Ok(FeedbackControl(parse_u8(tokens[1])?, tokens[2].to_string()))
            } else {
                Err("FEEDBACKCONTROL <reg> <signal>".into())
            }
        }
        // qoa specific
        "SETPOS" => {
            if tokens.len() == 4 {
                Ok(SetPos(
                    parse_u8(tokens[1])?,
                    parse_u16(tokens[2])?.into(),
                    parse_u16(tokens[3])?.into(),
                ))
            } else {
                Err("SETPOS <reg> <x> <y>".into())
            }
        }
        "SETWL" => {
            if tokens.len() == 3 {
                Ok(Instruction::SetWl(
                    parse_u8(tokens[1])?,
                    parse_u16(tokens[2])?.into(),
                ))
            } else {
                Err("SETWL <reg> <wl>".into())
            }
        }
        "SETPHASE" => {
            if tokens.len() == 3 {
                Ok(SetPhase(parse_u8(tokens[1])?, parse_f64(tokens[2])?))
            } else {
                Err("SETPHASE <reg> <phase>".into())
            }
        }
        "WLSHIFT" => {
            if tokens.len() == 3 {
                Ok(WlShift(parse_u8(tokens[1])?, parse_i16(tokens[2])?.into()))
            } else {
                Err("WLSHIFT <reg> <shift>".into())
            }
        }
        "MOVE" => {
            if tokens.len() == 4 {
                Ok(Move(
                    parse_u8(tokens[1])?,
                    parse_i16(tokens[2])?.into(),
                    parse_i16(tokens[3])?.into(),
                ))
            } else {
                Err("MOVE <reg> <dx> <dy>".into())
            }
        }
        "COMMENT" => {
            if tokens.len() >= 2 {
                Ok(Comment(tokens[1..].join(" ")))
            } else {
                Err("COMMENT <txt>".into())
            }
        }
        "MARKOBSERVED" => {
            if tokens.len() == 2 {
                Ok(MarkObserved(parse_u8(tokens[1])?))
            } else {
                Err("MARKOBSERVED <reg>".into())
            }
        }
        "RELEASE" => {
            if tokens.len() == 2 {
                Ok(Release(parse_u8(tokens[1])?))
            } else {
                Err("RELEASE <reg>".into())
            }
        }
        "HALT" => {
            if tokens.len() == 1 {
                Ok(Halt)
            } else {
                Err("HALT".into())
            }
        }
        _ => Err(format!("Unknown opcode '{}'", op)),
    }
}

pub fn parse_simple_opcode(tokens: &[&str]) -> Result<Instruction, String> {
    if tokens.is_empty() {
        return Err("Empty instruction line".into());
    }

    let opcode = tokens[0].to_uppercase();

    // for parsing
    let parse_u8 = |s: &str| s.parse::<u8>().map_err(|_| format!("Invalid u8 '{}'", s));
    let parse_u16 = |s: &str| s.parse::<u16>().map_err(|_| format!("Invalid u16 '{}'", s));
    let parse_i16 = |s: &str| s.parse::<i16>().map_err(|_| format!("Invalid i16 '{}'", s));
    let parse_u64 = |s: &str| s.parse::<u64>().map_err(|_| format!("Invalid u64 '{}'", s));
    let parse_f64 = |s: &str| s.parse::<f64>().map_err(|_| format!("Invalid f64 '{}'", s));
    let parse_bool = |s: &str| match s.to_uppercase().as_str() {
        "TRUE" | "ON" => Ok(true),
        "FALSE" | "OFF" => Ok(false),
        _ => Err(format!("Invalid bool '{}'", s)),
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
                Err(
                    "Malformed APPLY_PHASE_FLIP instruction. Usage: APPLY_PHASE_FLIP <qubit>"
                        .into(),
                )
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
                Err(format!(
                    "Malformed {} instruction. Usage: {} <dest> <src1> <src2>",
                    opcode, opcode
                ))
            }
        }
        "NOT" => {
            if tokens.len() == 2 {
                let reg = tokens[1].to_string();
                Ok(Instruction::Not(reg))
            } else {
                Err("Malformed NOT instruction. Usage: NOT <reg>".into())
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
                Ok(Instruction::JumpIfZero(
                    tokens[1].to_string(),
                    tokens[2].to_string(),
                ))
            } else {
                Err("Malformed JUMPIFZERO instruction. Usage: JUMPIFZERO <cond> <label>".into())
            }
        }
        "JUMPIFONE" => {
            if tokens.len() == 3 {
                Ok(Instruction::JumpIfOne(
                    tokens[1].to_string(),
                    tokens[2].to_string(),
                ))
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
                Ok(Instruction::LoadMem(
                    tokens[1].to_string(),
                    tokens[2].to_string(),
                ))
            } else {
                Err("Malformed LOADMEM instruction. Usage: LOADMEM <reg> <mem_addr>".into())
            }
        }
        "STOREMEM" => {
            if tokens.len() == 3 {
                Ok(Instruction::StoreMem(
                    tokens[1].to_string(),
                    tokens[2].to_string(),
                ))
            } else {
                Err("Malformed STOREMEM instruction. Usage: STOREMEM <reg> <mem_addr>".into())
            }
        }
        "LOADCLASSICAL" => {
            if tokens.len() == 3 {
                Ok(Instruction::LoadClassical(
                    tokens[1].to_string(),
                    tokens[2].to_string(),
                ))
            } else {
                Err("Malformed LOADCLASSICAL instruction. Usage: LOADCLASSICAL <reg> <var>".into())
            }
        }
        "STORECLASSICAL" => {
            if tokens.len() == 3 {
                Ok(Instruction::StoreClassical(
                    tokens[1].to_string(),
                    tokens[2].to_string(),
                ))
            } else {
                Err(
                    "Malformed STORECLASSICAL instruction. Usage: STORECLASSICAL <reg> <var>"
                        .into(),
                )
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
                let qubits = tokens[1]
                    .split(',')
                    .map(parse_u8)
                    .collect::<Result<Vec<u8>, _>>()?;
                let angles = tokens[3..]
                    .iter()
                    .map(|s| parse_f64(s))
                    .collect::<Result<Vec<f64>, _>>()?;
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
                Err(
                    "Malformed PHOTONROUTE instruction. Usage: PHOTONROUTE <qubit> <source> <dest>"
                        .into(),
                )
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
                Err(
                    "Malformed ENTANGLESWAP instruction. Usage: ENTANGLESWAP <q1> <q2> <q3> <q4>"
                        .into(),
                )
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
                Err(
                    "Malformed APPLYSQUEEZING instruction. Usage: APPLYSQUEEZING <qubit> <value>"
                        .into(),
                )
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
                let qubits = tokens[1]
                    .split(',')
                    .map(parse_u8)
                    .collect::<Result<Vec<u8>, _>>()?;
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
                let inputs = tokens[2]
                    .split(',')
                    .map(parse_u8)
                    .collect::<Result<Vec<u8>, _>>()?;
                let outputs = tokens[3]
                    .split(',')
                    .map(parse_u8)
                    .collect::<Result<Vec<u8>, _>>()?;
                let size = tokens[4]
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid usize '{}'", tokens[4]))?;
                Ok(Instruction::ApplyLinearOpticalTransform(
                    name, inputs, outputs, size,
                ))
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
                Err(
                    "Malformed MEASUREINBASIS instruction. Usage: MEASUREINBASIS <qubit> <basis>"
                        .into(),
                )
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
                Err(
                    "Malformed APPLYC_PHASE instruction. Usage: APPLYC_PHASE <q1> <q2> <angle>"
                        .into(),
                )
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
                Ok(Instruction::ErrorSyndrome(
                    q,
                    tokens[2].to_string(),
                    tokens[3].to_string(),
                ))
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
        "VERBOSELOG" => {
            if tokens.len() >= 2 {
                let r = parse_u8(tokens[1])?;
                let msg = if tokens.len() > 2 {
                    tokens[2..].join(" ")
                } else {
                    "".into()
                };
                Ok(Instruction::VerboseLog(r, msg))
            } else {
                Err("Usage: VerboseLog reg [message]".into())
            }
        }
        "SETPOS" => {
            if tokens.len() == 4 {
                Ok(Instruction::SetPos(
                    parse_u8(tokens[1])?,
                    tokens[2]
                        .parse::<u16>()
                        .map_err(|_| format!("Invalid u16 '{}'", tokens[2]))?
                        .into(),
                    tokens[3]
                        .parse::<u16>()
                        .map_err(|_| format!("Invalid u16 '{}'", tokens[3]))?
                        .into(),
                ))
            } else {
                Err("Malformed SETPOS instruction. Usage: SETPOS <reg> <x> <y>".into())
            }
        }

        "SETWL" => {
            if tokens.len() == 3 {
                Ok(Instruction::SetWl(
                    tokens[1]
                        .parse::<u8>()
                        .map_err(|_| format!("Invalid u8 '{}'", tokens[1]))?,
                    tokens[2]
                        .parse::<u16>()
                        .map_err(|_| format!("Invalid u16 '{}'", tokens[2]))?
                        .into(),
                ))
            } else {
                Err("Malformed SETWL instruction. Usage: SETWL <reg> <wavelength>".into())
            }
        }
        "SETPHASE" => {
            if tokens.len() == 3 {
                Ok(Instruction::SetPhase(
                    parse_u8(tokens[1])?,
                    parse_f64(tokens[2])?,
                ))
            } else {
                Err("Malformed SETPHASE instruction. Usage: SETPHASE <reg> <phase>".into())
            }
        }
        "WLSHIFT" => {
            if tokens.len() == 3 {
                Ok(Instruction::WlShift(
                    parse_u8(tokens[1])?,
                    tokens[2]
                        .parse::<i16>()
                        .map_err(|_| format!("Invalid i16 '{}'", tokens[2]))?
                        .into(),
                ))
            } else {
                Err("Malformed WLSHIFT instruction. Usage: WLSHIFT <reg> <shift>".into())
            }
        }
        "MOVE" => {
            if tokens.len() == 4 {
                Ok(Instruction::Move(
                    parse_u8(tokens[1])?,
                    tokens[2]
                        .parse::<i16>()
                        .map_err(|_| format!("Invalid i16 '{}'", tokens[2]))?
                        .into(),
                    tokens[3]
                        .parse::<i16>()
                        .map_err(|_| format!("Invalid i16 '{}'", tokens[3]))?
                        .into(),
                ))
            } else {
                Err("Malformed MOVE instruction. Usage: MOVE <reg> <dx> <dy>".into())
            }
        }
        "COMMENT" => {
            if tokens.len() >= 2 {
                Ok(Instruction::Comment(tokens[1..].join(" ")))
            } else {
                Err("Malformed COMMENT instruction. Usage: COMMENT <txt>".into())
            }
        }
        "MARKOBSERVED" => {
            if tokens.len() == 2 {
                Ok(Instruction::MarkObserved(parse_u8(tokens[1])?))
            } else {
                Err("Malformed MARKOBSERVED instruction. Usage: MARKOBSERVED <reg>".into())
            }
        }
        "RELEASE" => {
            if tokens.len() == 2 {
                Ok(Instruction::Release(parse_u8(tokens[1])?))
            } else {
                Err("Malformed RELEASE instruction. Usage: RELEASE <reg>".into())
            }
        }
        "HALT" => {
            if tokens.len() == 1 {
                Ok(Instruction::Halt)
            } else {
                Err("Malformed HALT instruction. Usage: HALT".into())
            }
        }
        _ => Err(format!("Unknown opcode '{}'", opcode)),
    }
}

impl Instruction {
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Instruction::Halt => vec![0xFF],

            // Loop
            Instruction::ApplyGate(gate, q) => {
                let mut v = vec![0x02, *q];
                let mut gate_bytes = gate.as_bytes().to_vec();
                gate_bytes.resize(8, 0); // Pad or truncate gate name to 8 bytes
                v.extend(gate_bytes);
                v
            }

            Instruction::RegSet(reg, val) => {
                // opcode 0x21 to match emulator run code
                let mut v = vec![0x21, *reg];
                v.extend_from_slice(&val.to_le_bytes());
                v
            }

            // Looping stuff
            Instruction::LoopStart(times) => vec![0x20, *times],
            Instruction::LoopEnd => vec![0x21],
            Instruction::Measure(q) => vec![0x03, *q],

            // Hadamard gate
            Instruction::Hadamard(reg) => vec![0x10, *reg as u8],

            // Handle variants with data by pattern matching on their content
            Instruction::InitQubit(reg) => vec![0x01, *reg],
            Instruction::Barrier => vec![0x03],

            Instruction::CharLoad(reg, val) => vec![0x31, *reg, *val],
            Instruction::QMeas(reg) => vec![0x32, *reg],

            // Instruction::Not(_) => vec![0x00],
            Instruction::QInit(n) => vec![0x04, *n],

            // Core
            Instruction::ControlledPhaseRotation(q1, q2, angle) => {
                let mut v = vec![0x90, *q1, *q2];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::ApplyCPhase(q1, q2, angle) => {
                let mut v = vec![0x91, *q1, *q2];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::ApplyKerrNonlinearity(q, strength, duration) => {
                let mut v = vec![0x92, *q];
                v.extend(&strength.to_le_bytes());
                v.extend(&duration.to_le_bytes());
                v
            }
            Instruction::ApplyFeedforwardGate(q, signal) => {
                let mut v = vec![0x93, *q];
                v.extend(signal.as_bytes());
                v.push(0);
                v
            }
            Instruction::DecoherenceProtect(q, duration) => {
                let mut v = vec![0x94, *q];
                v.extend(&duration.to_le_bytes());
                v
            }
            Instruction::ApplyMeasurementBasisChange(q, basis) => {
                let mut v = vec![0x95, *q];
                v.extend(basis.as_bytes());
                v.push(0);
                v
            }

            // Measurement in specific basis
            Instruction::MeasureInBasis(n, basis) => {
                let basis_code = match basis.as_str() {
                    "X" => 0x00,
                    "Y" => 0x01,
                    "Z" => 0x02,
                    _ => panic!("Unknown basis: {}", basis),
                };
                vec![0xA1, *n, basis_code]
            }

            Instruction::ApplyHadamard(q) => vec![0x05, *q],
            Instruction::ControlledNot(c, t) => vec![0x06, *c, *t],
            Instruction::ApplyPhaseFlip(q) => vec![0x07, *q],
            Instruction::ApplyBitFlip(q) => vec![0x08, *q],
            Instruction::ApplyTGate(q) => vec![0x09, *q],
            Instruction::ApplySGate(q) => vec![0x0A, *q],

            // Phase shift with floating-point value
            Instruction::PhaseShift(q, val) => {
                let mut v = vec![0x0B, *q];
                v.extend(&val.to_le_bytes());
                v
            }

            // Wait (delay) cycles
            Instruction::Wait(cycles) => {
                let mut v = vec![0x0C];
                v.extend(&cycles.to_le_bytes());
                v
            }

            // Reset operations
            Instruction::Reset(q) => vec![0x0D, *q],
            Instruction::ResetAll => vec![0x0E],

            // Swapping qubits
            Instruction::Swap(q1, q2) => vec![0x0F, *q1, *q2],
            Instruction::ControlledSwap(c, t1, t2) => vec![0x10, *c, *t1, *t2],

            // Entanglement operations
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

            Instruction::EntangleWithClassicalFeedback(q1, q2, reg) => {
                let mut v = vec![0x17, *q1, *q2];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }

            Instruction::EntangleDistributed(q, node) => {
                let mut v = vec![0x18, *q];
                v.extend(node.as_bytes());
                v.push(0);
                v
            }

            // Rotations
            Instruction::ApplyRotation(q, axis, angle) => {
                let mut v = vec![0x20, *q, *axis as u8];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::ApplyMultiQubitRotation(qs, axis, angles) => {
                let mut v = vec![0x21, *axis as u8, qs.len() as u8];
                v.extend(qs.iter());
                for a in angles {
                    v.extend(&a.to_le_bytes());
                }
                v
            }

            // Memory/classical ops
            Instruction::Load(q, reg) => {
                let mut v = vec![0x30, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::Store(q, reg) => {
                let mut v = vec![0x31, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::LoadMem(reg, addr) => {
                let mut v = vec![0x32];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(addr.as_bytes());
                v.push(0);
                v
            }
            Instruction::StoreMem(reg, addr) => {
                let mut v = vec![0x33];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(addr.as_bytes());
                v.push(0);
                v
            }
            Instruction::LoadClassical(reg, var) => {
                let mut v = vec![0x34];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(var.as_bytes());
                v.push(0);
                v
            }
            Instruction::StoreClassical(reg, var) => {
                let mut v = vec![0x35];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(var.as_bytes());
                v.push(0);
                v
            }
            Instruction::Add(dst, src1, src2) => {
                let mut v = vec![0x36];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::Sub(dst, src1, src2) => {
                let mut v = vec![0x37];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::And(dst, src1, src2) => {
                let mut v = vec![0x38];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::Or(dst, src1, src2) => {
                let mut v = vec![0x39];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::Xor(dst, src1, src2) => {
                let mut v = vec![0x3A];
                v.extend(dst.as_bytes());
                v.push(0);
                v.extend(src1.as_bytes());
                v.push(0);
                v.extend(src2.as_bytes());
                v.push(0);
                v
            }
            Instruction::Not(reg) => {
                let mut v = vec![0x3B];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::Push(reg) => {
                let mut v = vec![0x3C];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::Pop(reg) => {
                let mut v = vec![0x3D];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }

            // classical flow control
            Instruction::Jump(label) => {
                let mut v = vec![0x40];
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::JumpIfZero(cond, label) => {
                let mut v = vec![0x41];
                v.extend(cond.as_bytes());
                v.push(0);
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::JumpIfOne(cond, label) => {
                let mut v = vec![0x42];
                v.extend(cond.as_bytes());
                v.push(0);
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::Call(label) => {
                let mut v = vec![0x43];
                v.extend(label.as_bytes());
                v.push(0);
                v
            }
            Instruction::Return => vec![0x44],
            Instruction::Sync => vec![0x45],
            Instruction::TimeDelay(q, cycles) => {
                let mut v = vec![0x46, *q];
                v.extend(&cycles.to_le_bytes());
                v
            }
            Instruction::VerboseLog(q, msg) => {
                let mut v = vec![0x47, *q];
                v.extend(msg.as_bytes());
                v.push(0);
                v
            }

            // optics / photonics
            Instruction::PhotonEmit(q) => vec![0x50, *q],
            Instruction::PhotonDetect(q) => vec![0x51, *q],
            Instruction::PhotonCount(q, reg) => {
                let mut v = vec![0x52, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::PhotonAddition(q) => vec![0x53, *q],
            Instruction::ApplyPhotonSubtraction(q) => vec![0x54, *q],
            Instruction::PhotonEmissionPattern(q, reg, cycles) => {
                let mut v = vec![0x55, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v.extend(&cycles.to_le_bytes());
                v
            }
            Instruction::PhotonDetectWithThreshold(q, thresh, reg) => {
                let mut v = vec![0x56, *q];
                v.extend(&thresh.to_le_bytes());
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::PhotonDetectCoincidence(qs, reg) => {
                let mut v = vec![0x57, qs.len() as u8];
                v.extend(qs.iter());
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::SinglePhotonSourceOn(q) => vec![0x58, *q],
            Instruction::SinglePhotonSourceOff(q) => vec![0x59, *q],
            Instruction::PhotonBunchingControl(q, b) => vec![0x5A, *q, *b as u8],
            Instruction::PhotonRoute(q, from, to) => {
                let mut v = vec![0x5B, *q];
                v.extend(from.as_bytes());
                v.push(0);
                v.extend(to.as_bytes());
                v.push(0);
                v
            }
            Instruction::OpticalRouting(q1, q2) => vec![0x5C, *q1, *q2],
            Instruction::SetOpticalAttenuation(q, att) => {
                let mut v = vec![0x5D, *q];
                v.extend(&att.to_le_bytes());
                v
            }
            Instruction::DynamicPhaseCompensation(q, phase) => {
                let mut v = vec![0x5E, *q];
                v.extend(&phase.to_le_bytes());
                v
            }
            Instruction::OpticalDelayLineControl(q, delay) => {
                let mut v = vec![0x5F, *q];
                v.extend(&delay.to_le_bytes());
                v
            }
            Instruction::CrossPhaseModulation(c, t, stren) => {
                let mut v = vec![0x60, *c, *t];
                v.extend(&stren.to_le_bytes());
                v
            }
            Instruction::ApplyDisplacement(q, a) => {
                let mut v = vec![0x61, *q];
                v.extend(&a.to_le_bytes());
                v
            }
            Instruction::ApplyDisplacementFeedback(q, reg) => {
                let mut v = vec![0x62, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::ApplyDisplacementOperator(q, alpha, dur) => {
                let mut v = vec![0x63, *q];
                v.extend(&alpha.to_le_bytes());
                v.extend(&dur.to_le_bytes());
                v
            }
            Instruction::ApplySqueezing(q, s) => {
                let mut v = vec![0x64, *q];
                v.extend(&s.to_le_bytes());
                v
            }
            Instruction::ApplySqueezingFeedback(q, reg) => {
                let mut v = vec![0x65, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::MeasureParity(q) => vec![0x66, *q],
            Instruction::MeasureWithDelay(q, delay, reg) => {
                let mut v = vec![0x67, *q];
                v.extend(&delay.to_le_bytes());
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::OpticalSwitchControl(q, b) => vec![0x68, *q, *b as u8],
            Instruction::PhotonLossSimulate(q, prob, seed) => {
                let mut v = vec![0x69, *q];
                v.extend(&prob.to_le_bytes());
                v.extend(&seed.to_le_bytes());
                v
            }
            Instruction::PhotonLossCorrection(q, reg) => {
                let mut v = vec![0x6A, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            // Advanced / error correction
            Instruction::ApplyQndMeasurement(q, mode) => {
                let mut v = vec![0x70, *q];
                v.extend(mode.as_bytes());
                v.push(0);
                v
            }
            Instruction::ErrorCorrect(q, code) => {
                let mut v = vec![0x71, *q];
                v.extend(code.as_bytes());
                v.push(0);
                v
            }
            Instruction::ErrorSyndrome(q, typ, reg) => {
                let mut v = vec![0x72, *q];
                v.extend(typ.as_bytes());
                v.push(0);
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::QuantumStateTomography(q, basis) => {
                let mut v = vec![0x73, *q];
                v.extend(basis.as_bytes());
                v.push(0);
                v
            }
            Instruction::BellStateVerification(q1, q2, mode) => {
                let mut v = vec![0x74, *q1, *q2];
                v.extend(mode.as_bytes());
                v.push(0);
                v
            }
            Instruction::QuantumZenoEffect(q, freq, dur) => {
                let mut v = vec![0x75, *q];
                v.extend(&freq.to_le_bytes());
                v.extend(&dur.to_le_bytes());
                v
            }
            Instruction::ApplyNonlinearPhaseShift(q, phase) => {
                let mut v = vec![0x76, *q];
                v.extend(&phase.to_le_bytes());
                v
            }
            Instruction::ApplyNonlinearSigma(q, sigma) => {
                let mut v = vec![0x77, *q];
                v.extend(&sigma.to_le_bytes());
                v
            }
            Instruction::ApplyLinearOpticalTransform(name, ins, outs, sz) => {
                let mut v = vec![0x78];
                v.extend(name.as_bytes());
                v.push(0);
                v.push(ins.len() as u8);
                v.extend(ins.iter());
                v.push(outs.len() as u8);
                v.extend(outs.iter());
                v.extend(&sz.to_le_bytes());
                v
            }
            Instruction::PhotonNumberResolvingDetection(q, reg) => {
                let mut v = vec![0x79, *q];
                v.extend(reg.as_bytes());
                v.push(0);
                v
            }
            Instruction::FeedbackControl(q, sig) => {
                let mut v = vec![0x7A, *q];
                v.extend(sig.as_bytes());
                v.push(0);
                v
            }

            // Misc / QOA specific
            Instruction::SetPos(reg, x, y) => {
                let mut v = vec![0x80, *reg];
                v.extend(&x.to_le_bytes());
                v.extend(&y.to_le_bytes());
                v
            }
            Instruction::SetWl(reg, wl) => {
                let mut v = vec![0x81, *reg];
                v.extend(&wl.to_le_bytes());
                v
            }
            Instruction::SetPhase(reg, phase) => {
                let mut v = vec![0x82, *reg];
                v.extend(&phase.to_le_bytes());
                v
            }
            Instruction::WlShift(reg, shift) => {
                let mut v = vec![0x83, *reg];
                v.extend(&shift.to_le_bytes());
                v
            }
            Instruction::Move(reg, dx, dy) => {
                let mut v = vec![0x84, *reg];
                v.extend(&dx.to_le_bytes());
                v.extend(&dy.to_le_bytes());
                v
            }
            Instruction::Comment(msg) => {
                let mut v = vec![0xFE];
                v.extend(msg.as_bytes());
                v.push(0);
                v
            }
            Instruction::MarkObserved(reg) => vec![0xF0, *reg],
            Instruction::Release(reg) => vec![0xF1, *reg],
            // catchall arm
            // => todo!(),
        }
    }
}
