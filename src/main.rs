use clap::CommandFactory;
use clap::Parser;
use qoa::runtime::quantum_state::NoiseConfig;
use qoa::runtime::quantum_state::QuantumState;
use serde::Serialize;
use serde_json::to_writer_pretty;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng}; // keep Rng for rng.gen()

mod instructions;
#[cfg(test)] // for testing
mod test;

const QEX: &[u8; 4] = b"QEX ";
const QEXE: &[u8; 4] = b"QEXE";
const OEXE: &[u8; 4] = b"OEXE";
const QOX: &[u8; 4] = b"QOX ";
const XEXE: &[u8; 4] = b"XEXE";
const QX: &[u8; 4] = b"QX\0\0";

const QOA_VERSION: &str = "0.2.4";
const QOA_AUTHOR: &str = "Rayan (@planetryan on GitHub)";

#[derive(Parser, Debug)]
#[command(name = "QOA", author = QOA_AUTHOR, version = QOA_VERSION,
    about = format!("QOA (Quantum Optical Assembly Language) - A Free, Open Source, Quantum QPU simulator and assembly language.\n\
             Author: {QOA_AUTHOR}\n\
             Version: {QOA_VERSION}\n\n\
             Use 'qoa help <command>' for more information on a specific command, e.g., 'qoa help run'."),
    long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Parser, Debug)]
enum Commands {
    /// Compiles a .qoa source file into a .qexe binary executable.
    Compile {
        /// Source .qoa file path
        source: String,
        /// Output .qexe file path
        output: String,
        /// Enable debug mode for compilation.
        #[arg(long)]
        debug: bool,
    },
    /// Runs a .qexe binary executable.
    Run {
        /// Program .qexe file path
        program: String,
        /// Enable debug mode for runtime.
        #[arg(long)]
        debug: bool,
        /// Set simulation to ideal (no noise) conditions. Disables --noise and --final-noise.
        #[arg(long, conflicts_with = "noise")]
        ideal: bool,
        /// Apply noise simulation for gates. Can be `--noise` for random probability (0.1-1.0) or `--noise <PROBABILITY>` for a fixed probability (0.0-1.0).
        #[arg(long, num_args = 0..=1, default_missing_value = "random", value_name = "PROBABILITY")]
        noise: Option<String>,
        /// Apply an additional noise step to the final amplitudes before displaying them (default: true). Use --final-noise false to disable this specific noise.
        #[arg(long, default_value_t = true)] // This makes --final-noise true by default
        final_noise: bool,
    },
    /// Compiles a .qoa source file into a .json circuit description (IonQ format).
    CompileJson {
        /// Source .qoa file path
        source: String,
        /// Output .json file path
        output: String,
    },
    /// Prints the qoa version.
    Version,
    /// Prints all available global flags (options) for the 'run' command.
    Flags,
}

fn compile_qoa_to_bin(src_path: &str, debug_mode: bool) -> io::Result<Vec<u8>> {
    let file = File::open(src_path)?;
    let reader = BufReader::new(file);
    let mut payload = Vec::new();
    for line_result in reader.lines() {
        let line = line_result?;
        if line.trim().starts_with("//") || line.trim().starts_with(";") || line.trim().is_empty() {
            continue;
        }
        match instructions::parse_instruction(&line) {
            Ok(inst) => {
                let encoded = inst.encode();
                if debug_mode {
                    println!("parsing line: '{}', encoded bytes: {:?}", line, encoded);
                }
                payload.extend(encoded);
            }
            Err(e) => {
                eprintln!("warning: failed to parse instruction '{}': {}", line, e);
            }
        }
    }
    Ok(payload)
}

fn write_exe(payload: &[u8], out_path: &str, magic: &[u8; 4]) -> io::Result<()> {
    let mut f = File::create(out_path)?;
    f.write_all(magic)?;
    f.write_all(&[1])?;
    f.write_all(&(payload.len() as u32).to_le_bytes())?;
    f.write_all(payload)?;
    Ok(())
}

fn parse_exe_file(filedata: &[u8]) -> Option<(&'static str, u8, &[u8])> {
    if filedata.len() < 9 {
        eprintln!("error: file too short to contain a valid header and payload length.");
        return None;
    }
    let actual_header_bytes = &filedata[0..4];
    let name = match actual_header_bytes {
        m if *m == *QEXE => "QEXE",
        m if *m == *OEXE => "OEXE",
        m if *m == *QOX => "QOX",
        m if *m == *XEXE => "XEXE",
        m if *m == *QEX => "QEX",
        m if *m == *QX => "QX",
        _ => {
            eprintln!(
                "error: unknown or unsupported header: {:?} (as string: {:?})",
                actual_header_bytes,
                String::from_utf8_lossy(actual_header_bytes)
            );
            return None;
        }
    };
    let version = filedata[4];
    let payload_len =
        u32::from_le_bytes([filedata[5], filedata[6], filedata[7], filedata[8]]) as usize;
    if filedata.len() < 9 + payload_len {
        eprintln!(
            "error: file too short. expected {} bytes, got {}",
            9 + payload_len,
            filedata.len()
        );
        return None;
    }
    Some((name, version, &filedata[9..9 + payload_len]))
}

fn run_exe(
    filedata: &[u8],
    debug_mode: bool,
    noise_config: Option<NoiseConfig>,
    apply_final_noise_flag: bool,
) {
    let (header, version, payload) = match parse_exe_file(filedata) {
        Some(x) => x,
        None => {
            eprintln!("invalid or unsupported exe file, please check its header.");
            return;
        }
    };

    if debug_mode {
        eprintln!("payload length: {}", payload.len());
        eprintln!("payload snippet (first 32 bytes):");
        for j in 0..payload.len().min(32) {
            eprintln!(
                "byte[{:#04}] = 0x{:02X} '{}'",
                j,
                payload[j],
                if payload[j].is_ascii_graphic() || payload[j] == b' ' {
                    payload[j] as char
                } else {
                    '.'
                }
            );
        }
    }

    if let Some(config) = &noise_config {
        match config {
            NoiseConfig::Random => {
                eprintln!("[info] noise mode: random depolarizing");
            }
            NoiseConfig::Fixed(value) => {
                eprintln!("[info] noise mode: fixed depolarizing ({})", value);
            }
            NoiseConfig::Ideal => {
                eprintln!("[info] noise mode: ideal state (no noise)");
            }
        }
    }

    let mut max_q = 0usize;
    let mut i = 0usize; // for first pass

    // first pass: determine max_q without executing quantum operations
    // this pass also correctly advances the instruction pointer for variable-length instructions.
    // it also tracks the maximum qubit index accessed.
    while i < payload.len() {
        if debug_mode {
            eprintln!("scanning opcode 0x{:02X} at byte {}", payload[i], i);
        }
        let opcode = payload[i];
        match opcode {
            0x04 /* QInit / InitQubit */ | 0x32 /* QMeas / Measure */ | 0x05 /* ApplyHadamard */ |
            0x06 /* ApplyPhaseFlip */ | 0x07 /* ApplyBitFlip */ | 0x0D /* ApplyTGate */ |
            0x0E /* ApplySGate */ | 0x0A /* Reset / QReset */ | 0x59 /* PhotonEmit */ |
            0x5A /* PhotonDetect */ | 0x5C /* PhotonAddition */ | 0x5D /* ApplyPhotonSubtraction */ |
            0x61 /* SinglePhotonSourceOn */ | 0x62 /* SinglePhotonSourceOff */ |
            0x6F /* MeasureParity */ | 0x71 /* OpticalSwitchControl */ | 0x79 /* MarkObserved */ |
            0x7A /* Release */ | 0x9B /* Input */ | 0xA0 /* PushReg */ | 0xA1 /* PopReg */ |
            0xAC /* GetTime */ | 0x50 /* Rand */ | 0x18 /* CharOut */ => { // Added Rand and CharOut
                if i + 1 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }
            0x08 /* PhaseShift */ | 0x22 /* RX */ | 0x23 /* RY */ | 0x0F /* RZ */ |
            0x24 /* Phase */ | 0x66 /* SetOpticalAttenuation */ | 0x67 /* DynamicPhaseCompensation */ |
            0x6A /* ApplyDisplacement */ | 0x6D /* ApplySqueezing */ |
            0x82 /* ApplyNonlinearPhaseShift */ | 0x83 /* ApplyNonlinearSigma */ |
            0x21 /* RegSet */ => { // Added RegSet
                if i + 9 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x17 /* ControlledNot / CNOT */ | 0x1E /* CZ */ | 0x0B /* Swap */ |
            0x1F /* ThermalAvg */ | 0x65 /* OpticalRouting */ | 0x69 /* CrossPhaseModulation */ |
            0xA4 /* Cmp */ | 0x51 /* Sqrt */ | 0x52 /* Exp */ | 0x53 /* Log */ => { // Added Sqrt, Exp, Log
                if i + 2 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                i += 3;
            }
            0x0C /* ControlledSwap */ | 0x20 /* WkbFactor */ | 0x54 /* RegAdd */ |
            0x55 /* RegSub */ | 0x56 /* RegMul */ | 0x57 /* RegDiv */ | 0x58 /* RegCopy */ |
            0x63 /* PhotonBunchingControl */ | 0xA8 /* NotBits */ | 0x31 /* CharLoad */ => { // Added CharLoad
                if i + 3 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize).max(payload[i+3] as usize); // This line might be incorrect for all these opcodes, review
                i += 4;
            }
            0x11 /* Entangle */ | 0x12 /* EntangleBell */ => {
                if i + 2 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                i += 3;
            }
            0x13 /* EntangleMulti */ | 0x14 /* EntangleCluster */ => {
                if i + 1 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                let num_qubits_in_list = payload[i+1] as usize;
                if i + 1 + num_qubits_in_list >= payload.len() { eprintln!("incomplete instruction (entangle multi/cluster) at byte {}", i); break; }
                for q_idx in 0..num_qubits_in_list {
                    max_q = max_q.max(payload[i + 2 + q_idx] as usize);
                }
                i += 2 + num_qubits_in_list;
            }
            0x15 /* EntangleSwap */ => {
                if i + 4 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize).max(payload[i+3] as usize).max(payload[i+4] as usize);
                i += 5;
            }
            0x16 /* EntangleSwapMeasure */ => {
                if i + 4 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize).max(payload[i+3] as usize).max(payload[i+4] as usize);
                let label_start = i + 5;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = label_end + 1;
            }
            0x19 /* EntangleWithClassicalFeedback */ | 0x1A /* EntangleDistributed */ |
            0x1B /* MeasureInBasis */ | 0x87 /* VerboseLog */ | 0x38 /* ApplyFeedforwardGate */ |
            0x3A /* ApplyMeasurementBasisChange */ | 0x3B /* Load */ | 0x3C /* Store */ |
            0x5B /* PhotonCount */ | 0x6B /* ApplyDisplacementFeedback */ |
            0x6E /* ApplySqueezingFeedback */ | 0x73 /* PhotonLossCorrection */ |
            0x7C /* ApplyQndMeasurement */ | 0x7D /* ErrorCorrect */ |
            0x7F /* QuantumStateTomography */ | 0x85 /* PhotonNumberResolvingDetection */ |
            0x86 /* FeedbackControl */ => {
                if i + 1 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = label_end + 1;
            }
            0x1C /* ResetAll */ | 0x48 /* Sync */ | 0x10 /* LoopEnd */ | 0xFF /* Halt */ |
            0x4D /* Return */ | 0x89 /* Barrier */ | 0x97 /* RetSub */ | 0x9C /* DumpState */ |
            0x9D /* DumpRegs */ | 0xAB /* BreakPoint */ => {
                i += 1;
            }
            0x01 /* LoopStart */ => {
                if i + 1 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                i += 2;
            }
            0x02 /* ApplyGate (QGATE) */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete qgate instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let name_bytes = &payload[i + 2..i + 10];
                let name = String::from_utf8_lossy(name_bytes).trim_end_matches('\0').to_string();
                if name.as_str() == "cz" {
                    if i + 10 >= payload.len() { eprintln!("incomplete cz qgate instruction at byte {}", i); break; }
                    max_q = max_q.max(payload[i+10] as usize);
                    i += 11;
                } else {
                    i += 10;
                }
            }
            0x33 /* ApplyRotation */ => {
                if i + 10 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 11; // q (1 byte) + axis (1 byte) + angle (8 bytes) + opcode (1 byte)
            }
            0x34 /* ApplyMultiQubitRotation */ => {
                if i + 2 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                let num_qubits_in_list = payload[i+2] as usize;
                if i + 3 + num_qubits_in_list * 9 > payload.len() { eprintln!("incomplete instruction (multi qubit rotation) at byte {}", i); break; }
                for q_idx in 0..num_qubits_in_list {
                    max_q = max_q.max(payload[i + 3 + q_idx] as usize);
                }
                i += 3 + num_qubits_in_list + num_qubits_in_list * 8; // opcode + axis + num_qs + qs_list + angles
            }
            0x35 /* ControlledPhaseRotation */ | 0x36 /* ApplyCPhase */ => {
                if i + 10 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                i += 11; // c (1 byte) + t (1 byte) + angle (8 bytes) + opcode (1 byte)
            }
            0x37 /* ApplyKerrNonlinearity */ => {
                if i + 17 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18; // q (1 byte) + strength (8 bytes) + duration (8 bytes) + opcode (1 byte)
            }
            0x39 /* DecoherenceProtect */ | 0x68 /* OpticalDelayLineControl */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10; // q (1 byte) + duration (8 bytes) + opcode (1 byte)
            }
            0x3D /* LoadMem */ | 0x3E /* StoreMem */ => {
                let reg_start = i + 1;
                let reg_end = reg_start + payload[reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let addr_start = reg_end + 1;
                let addr_end = addr_start + payload[addr_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = addr_end + 1;
            }
            0x3F /* LoadClassical */ | 0x40 /* StoreClassical */ => {
                let reg_start = i + 1;
                let reg_end = reg_start + payload[reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let var_start = reg_end + 1;
                let var_end = var_start + payload[var_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = var_end + 1;
            }
            0x41 /* Add */ | 0x42 /* Sub */ | 0x43 /* And */ | 0x44 /* Or */ | 0x45 /* Xor */ => {
                let dst_start = i + 1;
                let dst_end = dst_start + payload[dst_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src1_start = dst_end + 1;
                let src1_end = src1_start + payload[src1_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src2_start = src1_end + 1;
                let src2_end = src2_start + payload[src2_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = src2_end + 1;
            }
            0x46 /* Not */ | 0x47 /* Push */ | 0x4F /* Pop */ => {
                let reg_start = i + 1;
                let reg_end = reg_start + payload[reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = reg_end + 1;
            }
            0x49 /* Jump */ | 0x4C /* Call */ => {
                let label_start = i + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = label_end + 1;
            }
            0x4A /* JumpIfZero */ | 0x4B /* JumpIfOne */ => {
                let cond_start = i + 1;
                let cond_end = cond_start + payload[cond_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label_start = cond_end + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = label_end + 1;
            }
            0x4E /* TimeDelay */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x5E /* PhotonEmissionPattern */ => {
                if i + 1 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let reg_start = i + 2;
                let reg_end = reg_start + payload[reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                if reg_end + 8 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                i = reg_end + 9; // opcode + q + reg_str + null_term + cycles (8 bytes)
            }
            0x5F /* PhotonDetectWithThreshold */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let reg_start = i + 10;
                let reg_end = reg_start + payload[reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = reg_end + 1; // opcode + q + thresh (8 bytes) + reg_str + null_term
            }
            0x60 /* PhotonDetectCoincidence */ => {
                if i + 1 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                let num_qubits_in_list = payload[i+1] as usize;
                if i + 2 + num_qubits_in_list >= payload.len() { eprintln!("incomplete instruction (photon detect coincidence) at byte {}", i); break; }
                for q_idx in 0..num_qubits_in_list {
                    max_q = max_q.max(payload[i + 2 + q_idx] as usize);
                }
                let reg_start = i + 2 + num_qubits_in_list;
                let reg_end = reg_start + payload[reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = reg_end + 1;
            }
            0x64 /* PhotonRoute */ => {
                if i + 1 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let from_start = i + 2;
                let from_end = from_start + payload[from_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let to_start = from_end + 1;
                let to_end = to_start + payload[to_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = to_end + 1;
            }
            0x6C /* ApplyDisplacementOperator */ => {
                if i + 17 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18; // q (1 byte) + alpha (8 bytes) + dur (8 bytes) + opcode (1 byte)
            }
            0x70 /* MeasureWithDelay */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let reg_start = i + 10;
                let reg_end = reg_start + payload[reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = reg_end + 1; // opcode + q + delay (8 bytes) + reg_str + null_term
            }
            0x72 /* PhotonLossSimulate */ => {
                if i + 17 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18; // q (1 byte) + prob (8 bytes) + seed (8 bytes) + opcode (1 byte)
            }
            0x74 /* SetPos */ | 0x77 /* Move */ => {
                if i + 17 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18; // q (1 byte) + x (8 bytes) + y (8 bytes) + opcode (1 byte)
            }
            0x75 /* SetWl */ | 0x76 /* WlShift */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10; // q (1 byte) + wl (8 bytes) + opcode (1 byte)
            }
            0x7E /* ErrorSyndrome */ => {
                if i + 1 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let syndrome_start = i + 2;
                let syndrome_end = syndrome_start + payload[syndrome_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let result_reg_start = syndrome_end + 1;
                let result_reg_end = result_reg_start + payload[result_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = result_reg_end + 1;
            }
            0x80 /* BellStateVerification */ => {
                if i + 2 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                let reg_start = i + 3;
                let reg_end = reg_start + payload[reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = reg_end + 1;
            }
            0x81 /* QuantumZenoEffect */ => {
                if i + 17 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18; // q (1 byte) + num_measurements (8 bytes) + interval_cycles (8 bytes) + opcode (1 byte)
            }
            0x84 /* ApplyLinearOpticalTransform */ => {
                if i + 4 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                let name_start = i + 4;
                let name_end = name_start + payload[name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let input_qs_len = payload[i+1] as usize;
                let output_qs_len = payload[i+2] as usize;
                let _num_modes = payload[i+3] as usize; // Renamed to _num_modes to suppress warning

                let input_qs_start = name_end + 1;
                let input_qs_end = input_qs_start + input_qs_len;
                if input_qs_end > payload.len() { eprintln!("incomplete instruction (linear optical transform input qs) at byte {}", i); break; }
                for q_idx in 0..input_qs_len {
                    max_q = max_q.max(payload[input_qs_start + q_idx] as usize);
                }

                let output_qs_start = input_qs_end;
                let output_qs_end = output_qs_start + output_qs_len;
                if output_qs_end > payload.len() { eprintln!("incomplete instruction (linear optical transform output qs) at byte {}", i); break; }
                for q_idx in 0..output_qs_len {
                    max_q = max_q.max(payload[output_qs_start + q_idx] as usize);
                }
                i = output_qs_end;
            }
            0x88 /* Comment */ => {
                let text_start = i + 1;
                let text_end = text_start + payload[text_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = text_end + 1;
            }
            0x90 /* Jmp */ | 0x91 /* JmpAbs */ | 0xA3 /* Free */ | 0xAD /* SeedRng */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                i += 9;
            }
            0x92 /* IfGt */ | 0x93 /* IfLt */ | 0x94 /* IfEq */ | 0x95 /* IfNe */ => {
                if i + 11 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                i += 11;
            }
            0x96 /* CallAddr */ | 0x9E /* LoadRegMem */ | 0xA2 /* Alloc */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                i += 10;
            }
            0x9F /* StoreMemReg */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                i += 10;
            }
            0x98 /* Printf */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete printf string length at byte {}", i); break; }
                let str_len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                let num_regs_idx = i + 9 + str_len;
                if num_regs_idx + 1 > payload.len() { eprintln!("incomplete printf num_regs at byte {}", i); break; }
                let num_regs = payload[num_regs_idx] as usize;
                i += 1 + 8 + str_len + 1 + num_regs;
            }
            0x99 /* Print */ | 0x9A /* Println */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete print/println string length at byte {}", i); break; }
                let str_len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                i += 1 + 8 + str_len;
            }
            0xA5 /* AndBits */ | 0xA6 /* OrBits */ | 0xA7 /* XorBits */ | 0xA9 /* Shl */ | 0xAA /* Shr */ => {
                if i + 3 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                i += 4;
            }
            0xAE /* ExitCode */ => {
                if i + 5 >= payload.len() { eprintln!("incomplete instruction at byte {}", i); break; }
                i += 5;
            }
            _ => {
                eprintln!("warning: unknown opcode 0x{:02X} in scan at byte {}, skipping.", opcode, i);
                if debug_mode {
                    eprintln!("payload near unknown opcode:");
                    for j in i..(i + 10).min(payload.len()) {
                        eprintln!(
                            "byte[{:#04}] = 0x{:02X} '{}'",
                            j,
                            payload[j],
                            if payload[j].is_ascii_graphic() || payload[j] == b' ' {
                                payload[j] as char
                            } else {
                                '.'
                            }
                        );
                    }
                }
                i += 1;
            }
        }
    }

    let num_qubits = if max_q == 0 && payload.is_empty() {
        0
    } else {
        max_q + 1
    };

    if num_qubits > 16 {
        eprintln!("warning: simulating more than 16 qubits can be very memory intensive.");
    }

    println!(
        "initializing quantum state with {} qubits (type {}, ver {})",
        num_qubits, header, version
    );
    let mut qs = QuantumState::new(num_qubits, noise_config.clone());
    let mut last_stats = Instant::now();
    let mut char_count: u64 = 0;
    let mut char_sum: u64 = 0;

    // declare registers, loop_stack, call_stack, memory, and rng for the second pass
    let mut registers: Vec<f64> = vec![0.0; 24]; // assuming 24 registers, adjust as needed
    let mut loop_stack: Vec<(usize, u64)> = Vec::new();
    let mut call_stack: Vec<usize> = Vec::new(); // for call/ret instructions
    let mut memory: Vec<u8> = vec![0; 1024 * 1024]; // 1mb linear byte-addressable memory
    let mut rng = StdRng::from_entropy(); // default seeded rng
                                          // removed `last_cmp_result` as it's not used by current conditional jumps
                                          // let mut last_cmp_result: Option<Ordering> = None; // for cmp instruction

    let mut i = 0; // Reset 'i' for the second pass
                   // second pass: execute instructions and interact with quantumstate
    while i < payload.len() {
        if debug_mode {
            eprintln!("executing opcode 0x{:02X} at byte {}", payload[i], i);
        }
        let opcode = payload[i];
        match opcode {
            0x04 /* QInit */ => {
                i += 2;
            }
            0x02 /* ApplyGate (QGATE) */ => {
                if i + 9 >= payload.len() {
                    eprintln!("incomplete qgate instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                let name_bytes = &payload[i + 2..i + 10];
                let name = String::from_utf8_lossy(name_bytes)
                    .trim_end_matches('\0')
                    .to_string();

                match name.as_str() {
                    "h" => {
                        qs.apply_h(q);
                        println!("applied h gate on qubit {} (via qgate)", q);
                        i += 10;
                    }
                    "x" => {
                        qs.apply_x(q);
                        println!("applied x gate on qubit {}", q);
                        i += 10;
                    }
                    "cz" => {
                        if i + 10 >= payload.len() {
                            eprintln!(
                                "incomplete cz qgate instruction: missing target qubit at byte {}",
                                i
                            );
                            break;
                        }
                        let tgt = payload[i + 10] as usize;
                        qs.apply_cz(q, tgt);
                        println!(
                            "applied cz gate between qubits {} (control) and {} (target)",
                            q, tgt
                        );
                        i += 11;
                    }
                    other => {
                        eprintln!("unknown gate: {}", other);
                        i += 10;
                    }
                }
            }
            0x05 /* ApplyHadamard */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete applyhadamard instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                qs.apply_h(q);
                println!("applied h gate on qubit {}", q);
                i += 2;
            }
            0x0d /* ApplyTGate */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete applytgate instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                qs.apply_t_gate(q);
                println!("applied t gate on qubit {}", q);
                i += 2;
            }
            0x0e /* ApplySGate */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete applysgate instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                qs.apply_s_gate(q);
                println!("applied s gate on qubit {}", q);
                i += 2;
            }
            0x17 /* ControlledNot */ => {
                if i + 2 >= payload.len() {
                    eprintln!("incomplete controllednot instruction at byte {}", i);
                    break;
                }
                let c = payload[i + 1] as usize;
                let t = payload[i + 2] as usize;

                qs.apply_cnot(c, t);
                println!("applied cnot gate from control {} to target {}", c, t);
                i += 3;
            }
            0x1E /* CZ */ => {
                if i + 2 >= payload.len() {
                    eprintln!("incomplete cz instruction at byte {}", i);
                    break;
                }
                let c = payload[i + 1] as usize;
                let t = payload[i + 2] as usize;
                qs.apply_cz(c, t);
                println!("applied cz gate between qubits {} and {}", c, t);
                i += 3;
            }
            0x31 /* CharLoad */ => {
                if i + 2 >= payload.len() {
                    eprintln!("incomplete charload instruction at byte {}", i);
                    break;
                }
                let reg = payload[i + 1] as usize;
                if reg >= registers.len() {
                    eprintln!("charload: register index {} out of range", reg);
                    break;
                }
                let val = payload[i + 2];
                registers[reg] = val as f64; // Store char as its ASCII value in the register
                print!("{}", val as char);
                std::io::stdout().flush().unwrap();
                i += 3;

                char_count += 1;
                char_sum += val as u64;
                if last_stats.elapsed() >= std::time::Duration::from_secs(1) {
                    let avg = char_sum as f64 / char_count as f64;
                    eprintln!(
                        "\n[stats] {} chars → avg numeric value: {:.2}",
                        char_count, avg
                    );
                    char_count = 0;
                    char_sum = 0;
                    last_stats = std::time::Instant::now();
                }
            }
            0x18 /* CharOut */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete charout instruction at byte {}", i);
                    break;
                }
                let reg = payload[i + 1] as usize;
                if reg >= registers.len() {
                    eprintln!("charout: register index {} out of range", reg);
                    break;
                }
                let char_val = registers[reg] as u8;
                print!("{}", char_val as char);
                std::io::stdout().flush().unwrap();
                i += 2;

                char_count += 1;
                char_sum += char_val as u64;
                if last_stats.elapsed() >= std::time::Duration::from_secs(1) {
                    let avg = char_sum as f64 / char_count as f64;
                    eprintln!(
                        "\n[stats] {} chars → avg numeric value: {:.2}",
                        char_count, avg
                    );
                    char_count = 0;
                    char_sum = 0;
                    last_stats = std::time::Instant::now();
                }
            }
            0x32 /* QMeas */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete qmeas instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                let meas_result = qs.measure(q);
                println!("\nmeasurement of qubit {}: {}", q, meas_result);
                i += 2;
            }
            0x21 /* RegSet */ => {
                if i + 9 >= payload.len() {
                    eprintln!("incomplete regset instruction at byte {}", i);
                    break;
                }
                let reg = payload[i + 1] as usize;
                if reg >= registers.len() {
                    eprintln!("regset: register index {} out of range", reg);
                    break;
                }
                let val_bytes = &payload[i + 2..i + 10];
                let val = f64::from_le_bytes(val_bytes.try_into().unwrap());
                registers[reg] = val;
                println!("set register {} to value {}", reg, val);
                i += 10;
            }
            0x48 /* Sync */ => {
                println!("synchronized state (sync instruction encountered)");
                i += 1;
            }
            0x00 /* NoOp */ => {
                i += 1;
            }
            0xff /* Halt */ => {
                std::io::stdout().flush().unwrap();
                break;
            }
            0x01 /* LoopStart */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete loopstart instruction at byte {}", i);
                    break;
                }
                let reg_idx = payload[i + 1] as usize;
                if reg_idx >= registers.len() {
                    eprintln!("loopstart: register index {} out of range", reg_idx);
                    break;
                }
                let count = registers[reg_idx] as u64;
                loop_stack.push((i + 2, count));
                println!("loopstart: reg {} (count {})", reg_idx, count);
                i += 2;
            }
            0x10 /* LoopEnd */ => {
                if let Some((loop_start_ptr, count)) = loop_stack.pop() {
                    if count > 1 {
                        loop_stack.push((loop_start_ptr, count - 1));
                        i = loop_start_ptr; // jump back to loop start
                        println!("looping back to {} ({} iterations left)", loop_start_ptr, count - 1);
                    } else {
                        println!("loop finished.");
                        i += 1; // proceed to next instruction after loop
                    }
                } else {
                    eprintln!("error: endloop without matching loopstart at byte {}", i);
                    i += 1; // advance to avoid infinite loop on error
                }
            }
            0x0f /* RZ */ => {
                if i + 9 >= payload.len() {
                    eprintln!("incomplete rz instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                let angle_bytes = &payload[i + 2..i + 10];
                let angle = f64::from_le_bytes(angle_bytes.try_into().unwrap());

                qs.apply_rz(q, angle);
                println!("applied rz gate on qubit {} with angle {}", q, angle);
                i += 10;
            }
            // new instructions for v0.3.0
            0x90 /* Jmp */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete jmp instruction at byte {}", i); break; }
                let offset = i64::from_le_bytes(payload[i+1..i+9].try_into().unwrap());
                i = (i as i64 + offset) as usize; // perform relative jump
                println!("jmp to relative offset {}", offset);
            }
            0x91 /* JmpAbs */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete jmpabs instruction at byte {}", i); break; }
                let addr = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                i = addr; // perform absolute jump
                println!("jmpabs to absolute address {}", addr);
            }
            0x92 /* IfGt */ => {
                if i + 11 >= payload.len() { eprintln!("incomplete ifgt instruction at byte {}", i); break; }
                let r1_idx = payload[i+1] as usize;
                let r2_idx = payload[i+2] as usize;
                let offset = i64::from_le_bytes(payload[i+3..i+11].try_into().unwrap());

                if r1_idx >= registers.len() || r2_idx >= registers.len() {
                    eprintln!("ifgt: register index out of range at byte {}", i); break;
                }
                // epsilon-based comparison for floats
                if (registers[r1_idx] - registers[r2_idx]) > f64::EPSILON {
                    i = (i as i64 + offset) as usize;
                    println!("ifgt true: jump to relative offset {}", offset);
                } else {
                    i += 11; // move past instruction
                    println!("ifgt false: no jump");
                }
            }
            0x93 /* IfLt */ => {
                if i + 11 >= payload.len() { eprintln!("incomplete iflt instruction at byte {}", i); break; }
                let r1_idx = payload[i+1] as usize;
                let r2_idx = payload[i+2] as usize;
                let offset = i64::from_le_bytes(payload[i+3..i+11].try_into().unwrap());

                if r1_idx >= registers.len() || r2_idx >= registers.len() {
                    eprintln!("iflt: register index out of range at byte {}", i); break;
                }
                if (registers[r2_idx] - registers[r1_idx]) > f64::EPSILON {
                    i = (i as i64 + offset) as usize;
                    println!("iflt true: jump to relative offset {}", offset);
                } else {
                    i += 11;
                    println!("iflt false: no jump");
                }
            }
            0x94 /* IfEq */ => {
                if i + 11 >= payload.len() { eprintln!("incomplete ifeq instruction at byte {}", i); break; }
                let r1_idx = payload[i+1] as usize;
                let r2_idx = payload[i+2] as usize;
                let offset = i64::from_le_bytes(payload[i+3..i+11].try_into().unwrap());

                if r1_idx >= registers.len() || r2_idx >= registers.len() {
                    eprintln!("ifeq: register index out of range at byte {}", i); break;
                }
                if (registers[r1_idx] - registers[r2_idx]).abs() < f64::EPSILON {
                    i = (i as i64 + offset) as usize;
                    println!("ifeq true: jump to relative offset {}", offset);
                } else {
                    i += 11;
                    println!("ifeq false: no jump");
                }
            }
            0x95 /* IfNe */ => {
                if i + 11 >= payload.len() { eprintln!("incomplete ifne instruction at byte {}", i); break; }
                let r1_idx = payload[i+1] as usize;
                let r2_idx = payload[i+2] as usize;
                let offset = i64::from_le_bytes(payload[i+3..i+11].try_into().unwrap());

                if r1_idx >= registers.len() || r2_idx >= registers.len() {
                    eprintln!("ifne: register index out of range at byte {}", i); break;
                }
                if (registers[r1_idx] - registers[r2_idx]).abs() >= f64::EPSILON {
                    i = (i as i64 + offset) as usize;
                    println!("ifne true: jump to relative offset {}", offset);
                } else {
                    i += 11;
                    println!("ifne false: no jump");
                }
            }
            0x96 /* CallAddr */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete call_addr instruction at byte {}", i); break; }
                let addr = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                call_stack.push(i + 9); // push return address (next instruction after call)
                i = addr; // jump to subroutine
                println!("call_addr to {} (return address {})", addr, call_stack.last().unwrap());
            }
            0x97 /* RetSub */ => {
                if let Some(return_addr) = call_stack.pop() {
                    i = return_addr;
                    println!("ret_sub to return address {}", return_addr);
                } else {
                    eprintln!("error: ret_sub without matching call_addr at byte {}", i);
                    break; // halt on error
                }
            }
            0x98 /* Printf */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete printf string length at byte {}", i); break; }
                let str_len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                let format_str_start = i + 9;
                let format_str_end = format_str_start + str_len;
                if format_str_end > payload.len() { eprintln!("incomplete printf format string at byte {}", i); break; }
                let format_str = String::from_utf8_lossy(&payload[format_str_start..format_str_end]);

                let num_regs_idx = format_str_end;
                if num_regs_idx + 1 > payload.len() { eprintln!("incomplete printf num_regs at byte {}", i); break; }
                let num_regs = payload[num_regs_idx] as usize;

                let regs_start = num_regs_idx + 1;
                let regs_end = regs_start + num_regs;
                if regs_end > payload.len() { eprintln!("incomplete printf register list at byte {}", i); break; }
                let reg_indices = &payload[regs_start..regs_end];

                let mut output = format_str.to_string();
                for &reg_idx_u8 in reg_indices {
                    let reg_idx = reg_idx_u8 as usize;
                    if reg_idx >= registers.len() {
                        eprintln!("printf: register index {} out of range", reg_idx);
                        output = format!("error: printf: register index {} out of range", reg_idx);
                        break;
                    }
                    // simple replacement for %f, more robust parsing would be needed for full C-style printf
                    output = output.replacen("%f", &registers[reg_idx].to_string(), 1);
                }
                print!("{}", output);
                std::io::stdout().flush().unwrap();
                i = regs_end;
                println!("printf: {}", output);
            }
            0x99 /* Print */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete print string length at byte {}", i); break; }
                let str_len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                let str_start = i + 9;
                let str_end = str_start + str_len;
                if str_end > payload.len() { eprintln!("incomplete print string at byte {}", i); break; }
                let s = String::from_utf8_lossy(&payload[str_start..str_end]);
                print!("{}", s);
                std::io::stdout().flush().unwrap();
                i = str_end;
                println!("print: {}", s);
            }
            0x9A /* Println */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete println string length at byte {}", i); break; }
                let str_len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                let str_start = i + 9;
                let str_end = str_start + str_len;
                if str_end > payload.len() { eprintln!("incomplete println string at byte {}", i); break; }
                let s = String::from_utf8_lossy(&payload[str_start..str_end]);
                println!("{}", s);
                std::io::stdout().flush().unwrap();
                i = str_end;
                println!("println: {}", s);
            }
            0x9B /* Input */ => {
                if i + 1 >= payload.len() { eprintln!("incomplete input instruction at byte {}", i); break; }
                let reg_idx = payload[i+1] as usize;
                if reg_idx >= registers.len() {
                    eprintln!("input: register index {} out of range", reg_idx); break;
                }
                print!("input (float for reg {}): ", reg_idx);
                std::io::stdout().flush().unwrap();
                let mut input_line = String::new();
                io::stdin().read_line(&mut input_line).expect("failed to read line");
                match input_line.trim().parse::<f64>() {
                    Ok(val) => {
                        registers[reg_idx] = val;
                        println!("read {} into register {}", val, reg_idx);
                    },
                    Err(e) => {
                        eprintln!("invalid input: {}", e);
                        registers[reg_idx] = 0.0; // default to 0.0 on error
                    }
                }
                i += 2;
            }
            0x9C /* DumpState */ => {
                println!("\n--- quantum state dump ---");
                if num_qubits > 0 {
                    for (idx, amp) in qs.amps.iter().enumerate() {
                        println!("|{}⟩: {:.10} + {:.10}i", idx, amp.re, amp.im);
                    }
                } else {
                    println!("no qubits initialized.");
                }
                println!("--------------------------");
                i += 1;
            }
            0x9D /* DumpRegs */ => {
                println!("\n--- register dump ---");
                for (idx, val) in registers.iter().enumerate() {
                    println!("reg[{}]: {:.10}", idx, val);
                }
                println!("---------------------");
                i += 1;
            }
            0x9E /* LoadRegMem */ => {
                if i + 10 >= payload.len() { eprintln!("incomplete load_reg_mem instruction at byte {}", i); break; }
                let reg_idx = payload[i+1] as usize;
                let addr = u64::from_le_bytes(payload[i+2..i+10].try_into().unwrap()) as usize;

                if reg_idx >= registers.len() { eprintln!("load_reg_mem: register index {} out of range", reg_idx); break; }
                if addr + 8 > memory.len() { eprintln!("load_reg_mem: memory address {} out of bounds", addr); break; }

                let val_bytes: [u8; 8] = memory[addr..addr+8].try_into().unwrap();
                registers[reg_idx] = f64::from_le_bytes(val_bytes);
                println!("loaded {:.10} from memory address {} into register {}", registers[reg_idx], addr, reg_idx);
                i += 10;
            }
            0x9F /* StoreMemReg */ => {
                if i + 10 >= payload.len() { eprintln!("incomplete store_mem_reg instruction at byte {}", i); break; }
                let addr = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                let reg_idx = payload[i+9] as usize;

                if reg_idx >= registers.len() { eprintln!("store_mem_reg: register index {} out of range", reg_idx); break; }
                if addr + 8 > memory.len() { eprintln!("store_mem_reg: memory address {} out of bounds", addr); break; }

                let val_bytes = registers[reg_idx].to_le_bytes();
                memory[addr..addr+8].copy_from_slice(&val_bytes);
                println!("stored {:.10} from register {} into memory address {}", registers[reg_idx], reg_idx, addr);
                i += 10;
            }
            0xA0 /* PushReg */ => {
                if i + 1 >= payload.len() { eprintln!("incomplete push_reg instruction at byte {}", i); break; }
                let reg_idx = payload[i+1] as usize;
                if reg_idx >= registers.len() { eprintln!("push_reg: register index {} out of range", reg_idx); break; }
                // For now, push to a simple value stack (not the call_stack)
                // Assuming a `value_stack: Vec<f64>` is defined at the top of run_exe
                // For simplicity, let's just print a message for now.
                eprintln!("error: push_reg not fully implemented without a dedicated value stack. skipping.");
                i += 2;
            }
            0xA1 /* PopReg */ => {
                if i + 1 >= payload.len() { eprintln!("incomplete pop_reg instruction at byte {}", i); break; }
                let reg_idx = payload[i+1] as usize;
                if reg_idx >= registers.len() { eprintln!("pop_reg: register index {} out of range", reg_idx); break; }
                // For simplicity, let's just print a message for now.
                eprintln!("error: pop_reg not fully implemented without a dedicated value stack. skipping.");
                i += 2;
            }
            0xA2 /* Alloc */ => {
                if i + 10 >= payload.len() { eprintln!("incomplete alloc instruction at byte {}", i); break; }
                let reg_addr_idx = payload[i+1] as usize;
                let size = u64::from_le_bytes(payload[i+2..i+10].try_into().unwrap()) as usize;

                if reg_addr_idx >= registers.len() { eprintln!("alloc: register index {} out of range", reg_addr_idx); break; }

                // simple allocation: find first fit. in a real vm, this would be more complex.
                // for now, just simulate by giving an address and ensuring memory is large enough.
                let allocated_addr = 0; // simplified: always allocate from start for now
                if allocated_addr + size > memory.len() {
                    eprintln!("alloc: not enough memory for {} bytes", size);
                    registers[reg_addr_idx] = -1.0; // indicate failure
                } else {
                    registers[reg_addr_idx] = allocated_addr as f64;
                    println!("allocated {} bytes at address {} (stored in reg {})", size, allocated_addr, reg_addr_idx);
                }
                i += 10;
            }
            0xA3 /* Free */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete free instruction at byte {}", i); break; }
                let addr = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                // in a simple simulator, free might just be a no-op or a print
                println!("freed memory at address {}", addr);
                i += 9;
            }
            0xA4 /* Cmp */ => {
                if i + 3 >= payload.len() { eprintln!("incomplete cmp instruction at byte {}", i); break; }
                let r1_idx = payload[i+1] as usize;
                let r2_idx = payload[i+2] as usize;

                if r1_idx >= registers.len() || r2_idx >= registers.len() {
                    eprintln!("cmp: register index out of range at byte {}", i); break;
                }

                let val1 = registers[r1_idx];
                let val2 = registers[r2_idx];

                // last_cmp_result is no longer used by conditional jumps, removing assignment
                // last_cmp_result = if (val1 - val2).abs() < f64::EPSILON {
                //     Some(Ordering::Equal)
                // } else if val1 > val2 {
                //     Some(Ordering::Greater)
                // } else {
                //     Some(Ordering::Less)
                // };
                println!("compared reg {} ({:.10}) and reg {} ({:.10})", r1_idx, val1, r2_idx, val2);
                i += 3;
            }
            0xA5 /* AndBits */ => {
                if i + 4 >= payload.len() { eprintln!("incomplete and_bits instruction at byte {}", i); break; }
                let dest_idx = payload[i+1] as usize;
                let op1_idx = payload[i+2] as usize;
                let op2_idx = payload[i+3] as usize;

                if dest_idx >= registers.len() || op1_idx >= registers.len() || op2_idx >= registers.len() {
                    eprintln!("and_bits: register index out of range at byte {}", i); break;
                }
                registers[dest_idx] = ((registers[op1_idx] as u64) & (registers[op2_idx] as u64)) as f64;
                println!("and_bits: reg[{}] = reg[{}] & reg[{}] = {:.0}", dest_idx, op1_idx, op2_idx, registers[dest_idx]);
                i += 4;
            }
            0xA6 /* OrBits */ => {
                if i + 4 >= payload.len() { eprintln!("incomplete or_bits instruction at byte {}", i); break; }
                let dest_idx = payload[i+1] as usize;
                let op1_idx = payload[i+2] as usize;
                let op2_idx = payload[i+3] as usize;

                if dest_idx >= registers.len() || op1_idx >= registers.len() || op2_idx >= registers.len() {
                    eprintln!("or_bits: register index out of range at byte {}", i); break;
                }
                registers[dest_idx] = ((registers[op1_idx] as u64) | (registers[op2_idx] as u64)) as f64;
                println!("or_bits: reg[{}] = reg[{}] | reg[{}] = {:.0}", dest_idx, op1_idx, op2_idx, registers[dest_idx]);
                i += 4;
            }
            0xA7 /* XorBits */ => {
                if i + 4 >= payload.len() { eprintln!("incomplete xor_bits instruction at byte {}", i); break; }
                let dest_idx = payload[i+1] as usize;
                let op1_idx = payload[i+2] as usize;
                let op2_idx = payload[i+3] as usize;

                if dest_idx >= registers.len() || op1_idx >= registers.len() || op2_idx >= registers.len() {
                    eprintln!("xor_bits: register index out of range at byte {}", i); break;
                }
                registers[dest_idx] = ((registers[op1_idx] as u64) ^ (registers[op2_idx] as u64)) as f64;
                println!("xor_bits: reg[{}] = reg[{}] ^ reg[{}] = {:.0}", dest_idx, op1_idx, op2_idx, registers[dest_idx]);
                i += 4;
            }
            0xA8 /* NotBits */ => {
                if i + 3 >= payload.len() { eprintln!("incomplete not_bits instruction at byte {}", i); break; }
                let dest_idx = payload[i+1] as usize;
                let op_idx = payload[i+2] as usize;

                if dest_idx >= registers.len() || op_idx >= registers.len() {
                    eprintln!("not_bits: register index out of range at byte {}", i); break;
                }
                registers[dest_idx] = (!(registers[op_idx] as u64)) as f64;
                println!("not_bits: reg[{}] = ~reg[{}] = {:.0}", dest_idx, op_idx, registers[dest_idx]);
                i += 3;
            }
            0xA9 /* Shl */ => {
                if i + 4 >= payload.len() { eprintln!("incomplete shl instruction at byte {}", i); break; }
                let dest_idx = payload[i+1] as usize;
                let op_idx = payload[i+2] as usize;
                let amount_idx = payload[i+3] as usize;

                if dest_idx >= registers.len() || op_idx >= registers.len() || amount_idx >= registers.len() {
                    eprintln!("shl: register index out of range at byte {}", i); break;
                }
                registers[dest_idx] = ((registers[op_idx] as u64) << (registers[amount_idx] as u64)) as f64;
                println!("shl: reg[{}] = reg[{}] << reg[{}] = {:.0}", dest_idx, op_idx, amount_idx, registers[dest_idx]);
                i += 4;
            }
            0xAA /* Shr */ => {
                if i + 4 >= payload.len() { eprintln!("incomplete shr instruction at byte {}", i); break; }
                let dest_idx = payload[i+1] as usize;
                let op_idx = payload[i+2] as usize;
                let amount_idx = payload[i+3] as usize;

                if dest_idx >= registers.len() || op_idx >= registers.len() || amount_idx >= registers.len() {
                    eprintln!("shr: register index out of range at byte {}", i); break;
                }
                registers[dest_idx] = ((registers[op_idx] as u64) >> (registers[amount_idx] as u64)) as f64;
                println!("shr: reg[{}] = reg[{}] >> reg[{}] = {:.0}", dest_idx, op_idx, amount_idx, registers[dest_idx]);
                i += 4;
            }
            0xAB /* BreakPoint */ => {
                eprintln!("\n--- breakpoint hit at byte {} ---", i);
                // in a real debugger, this would pause execution
                i += 1;
            }
            0xAC /* GetTime */ => {
                if i + 1 >= payload.len() { eprintln!("incomplete get_time instruction at byte {}", i); break; }
                let reg_idx = payload[i+1] as usize;
                if reg_idx >= registers.len() { eprintln!("get_time: register index {} out of range", reg_idx); break; }
                let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
                registers[reg_idx] = current_time;
                println!("current time {:.6} stored in reg {}", current_time, reg_idx);
                i += 2;
            }
            0xAD /* SeedRng */ => {
                if i + 9 >= payload.len() { eprintln!("incomplete seed_rng instruction at byte {}", i); break; }
                let seed_val = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap());
                rng = StdRng::seed_from_u64(seed_val);
                println!("rng seeded with value {}", seed_val);
                i += 9;
            }
            0xAE /* ExitCode */ => {
                if i + 5 >= payload.len() { eprintln!("incomplete exit_code instruction at byte {}", i); break; }
                let code = i32::from_le_bytes(payload[i+1..i+5].try_into().unwrap());
                println!("program exiting with code {}", code);
                std::process::exit(code);
            }
            0x50 /* Rand */ => {
                if i + 1 >= payload.len() { eprintln!("incomplete rand instruction at byte {}", i); break; }
                let reg_idx = payload[i+1] as usize;
                if reg_idx >= registers.len() { eprintln!("rand: register index {} out of range", reg_idx); break; }
                registers[reg_idx] = rng.gen::<f64>(); // Generate a random f64 between 0.0 and 1.0
                println!("generated random number {:.10} into reg {}", registers[reg_idx], reg_idx);
                i += 2;
            }
            0x51 /* Sqrt */ => {
                if i + 2 >= payload.len() { eprintln!("incomplete sqrt instruction at byte {}", i); break; }
                let rd = payload[i+1] as usize;
                let rs = payload[i+2] as usize;
                if rd >= registers.len() || rs >= registers.len() { eprintln!("sqrt: register index out of range at byte {}", i); break; }
                registers[rd] = registers[rs].sqrt();
                println!("sqrt: reg[{}] = sqrt(reg[{}]) = {:.10}", rd, rs, registers[rd]);
                i += 3;
            }
            0x52 /* Exp */ => {
                if i + 2 >= payload.len() { eprintln!("incomplete exp instruction at byte {}", i); break; }
                let rd = payload[i+1] as usize;
                let rs = payload[i+2] as usize;
                if rd >= registers.len() || rs >= registers.len() { eprintln!("exp: register index out of range at byte {}", i); break; }
                registers[rd] = registers[rs].exp();
                println!("exp: reg[{}] = exp(reg[{}]) = {:.10}", rd, rs, registers[rd]);
                i += 3;
            }
            0x53 /* Log */ => {
                if i + 2 >= payload.len() { eprintln!("incomplete log instruction at byte {}", i); break; }
                let rd = payload[i+1] as usize;
                let rs = payload[i+2] as usize;
                if rd >= registers.len() || rs >= registers.len() { eprintln!("log: register index out of range at byte {}", i); break; }
                registers[rd] = registers[rs].log(std::f64::consts::E); // Natural logarithm
                println!("log: reg[{}] = log(reg[{}]) = {:.10}", rd, rs, registers[rd]);
                i += 3;
            }
            0x54 /* RegAdd */ => {
                if i + 3 >= payload.len() { eprintln!("incomplete regadd instruction at byte {}", i); break; }
                let rd = payload[i+1] as usize;
                let ra = payload[i+2] as usize;
                let rb = payload[i+3] as usize;
                if rd >= registers.len() || ra >= registers.len() || rb >= registers.len() { eprintln!("regadd: register index out of range at byte {}", i); break; }
                registers[rd] = registers[ra] + registers[rb];
                println!("regadd: reg[{}] = reg[{}] + reg[{}] = {:.10}", rd, ra, rb, registers[rd]);
                i += 4;
            }
            0x55 /* RegSub */ => {
                if i + 3 >= payload.len() { eprintln!("incomplete regsub instruction at byte {}", i); break; }
                let rd = payload[i+1] as usize;
                let ra = payload[i+2] as usize;
                let rb = payload[i+3] as usize;
                if rd >= registers.len() || ra >= registers.len() || rb >= registers.len() { eprintln!("regsub: register index out of range at byte {}", i); break; }
                registers[rd] = registers[ra] - registers[rb];
                println!("regsub: reg[{}] = reg[{}] - reg[{}] = {:.10}", rd, ra, rb, registers[rd]);
                i += 4;
            }
            0x56 /* RegMul */ => {
                if i + 3 >= payload.len() { eprintln!("incomplete regmul instruction at byte {}", i); break; }
                let rd = payload[i+1] as usize;
                let ra = payload[i+2] as usize;
                let rb = payload[i+3] as usize;
                if rd >= registers.len() || ra >= registers.len() || rb >= registers.len() { eprintln!("regmul: register index out of range at byte {}", i); break; }
                registers[rd] = registers[ra] * registers[rb];
                println!("regmul: reg[{}] = reg[{}] * reg[{}] = {:.10}", rd, ra, rb, registers[rd]);
                i += 4;
            }
            0x57 /* RegDiv */ => {
                if i + 3 >= payload.len() { eprintln!("incomplete regdiv instruction at byte {}", i); break; }
                let rd = payload[i+1] as usize;
                let ra = payload[i+2] as usize;
                let rb = payload[i+3] as usize;
                if rd >= registers.len() || ra >= registers.len() || rb >= registers.len() { eprintln!("regdiv: register index out of range at byte {}", i); break; }
                if registers[rb] == 0.0 {
                    eprintln!("error: division by zero in regdiv at byte {}", i);
                    break;
                }
                registers[rd] = registers[ra] / registers[rb];
                println!("regdiv: reg[{}] = reg[{}] / reg[{}] = {:.10}", rd, ra, rb, registers[rd]);
                i += 4;
            }
            0x58 /* RegCopy */ => {
                if i + 2 >= payload.len() { eprintln!("incomplete regcopy instruction at byte {}", i); break; }
                let rd = payload[i+1] as usize;
                let ra = payload[i+2] as usize;
                if rd >= registers.len() || ra >= registers.len() { eprintln!("regcopy: register index out of range at byte {}", i); break; }
                registers[rd] = registers[ra];
                println!("regcopy: reg[{}] = reg[{}] = {:.10}", rd, ra, registers[rd]);
                i += 3;
            }
            _ => {
                eprintln!("warning: unknown opcode 0x{:02X} at byte {}, skipping.", opcode, i);

                if debug_mode {
                    eprintln!("payload near unknown opcode:");
                    for j in i..(i + 10).min(payload.len()) {
                        eprintln!(
                            "byte[{:#04}] = 0x{:02X} '{}'",
                            j,
                            payload[j],
                            if payload[j].is_ascii_graphic() || payload[j] == b' ' {
                                payload[j] as char
                            } else {
                                '.'
                            }
                        );
                    }
                }

                i += 1;
            }
        }
    }

    if apply_final_noise_flag {
        if let Some(NoiseConfig::Ideal) = &noise_config {
            eprintln!("[info] final state noise is skipped due to ideal mode.");
        } else {
            qs.apply_final_state_noise();
        }
    }

    println!("\nfinal amplitudes:");
    if num_qubits > 0 {
        for (idx, amp) in qs.amps.iter().enumerate() {
            println!("|{}⟩: {:.10} + {:.10}i", idx, amp.re, amp.im);
        }
    } else {
        println!("no qubits initialized, no final state to display.");
    }
}

#[derive(Serialize)]
struct JsonGate {
    gate: String,
    target: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    control: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    angle: Option<f64>,
}

fn parse_line_to_json(line: &str) -> Option<(JsonGate, usize)> {
    let parts: Vec<_> = line.trim().split_whitespace().collect();
    if parts.is_empty() || parts[0].starts_with("//") {
        return None;
    }

    match parts[0].to_uppercase().as_str() {
        "QGATE" => {
            if parts.len() == 3 {
                let target = parts[1].parse().ok()?;
                let gate = parts[2].to_lowercase();
                Some((
                    JsonGate {
                        gate,
                        target,
                        control: None,
                        angle: None,
                    },
                    target,
                ))
            } else if parts.len() == 4 {
                let control = parts[1].parse().ok()?;
                let gate = parts[2].to_lowercase();
                let target = parts[3].parse().ok()?;
                Some((
                    JsonGate {
                        gate,
                        target,
                        control: Some(control),
                        angle: None,
                    },
                    control.max(target),
                ))
            } else {
                None
            }
        }
        "RZ" => {
            if parts.len() == 3 {
                let target = parts[1].parse().ok()?;
                let angle = parts[2].parse().ok()?;
                Some((
                    JsonGate {
                        gate: "rz".to_string(),
                        target,
                        control: None,
                        angle: Some(angle),
                    },
                    target,
                ))
            } else {
                None
            }
        }
        "QINIT" => {
            if parts.len() == 2 {
                let qubit_idx = parts[1].parse().ok()?;
                Some((
                    JsonGate {
                        gate: "no-op".to_string(),
                        target: qubit_idx,
                        control: None,
                        angle: None,
                    },
                    qubit_idx,
                ))
            } else {
                None
            }
        }
        "H" | "AH" | "APPLYHADAMARD" => {
            if parts.len() == 2 {
                let target = parts[1].parse().ok()?;
                Some((
                    JsonGate {
                        gate: "h".to_string(),
                        target,
                        control: None,
                        angle: None,
                    },
                    target,
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn compile_qoa_to_json(src_path: &str, out_path: &str) -> io::Result<()> {
    let file = File::open(src_path)?;
    let reader = BufReader::new(file);
    let mut circuit = Vec::new();
    let mut max_qubit = 0usize;

    for line_result in reader.lines() {
        let line = line_result?;
        if let Some((gate, maxq)) = parse_line_to_json(&line) {
            max_qubit = max_qubit.max(maxq);
            circuit.push(gate);
        }
    }

    let json = serde_json::json!({
        "format": "ionq.circuit",
        "version": 1,
        "qubits": max_qubit + 1,
        "circuit": circuit,
    });

    let outfile = File::create(out_path)?;
    to_writer_pretty(outfile, &json)?;

    Ok(())
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile {
            source,
            output,
            debug,
        } => match compile_qoa_to_bin(&source, debug) {
            Ok(bin) => {
                if let Err(e) = write_exe(&bin, &output, QEXE) {
                    eprintln!("error writing executable: {}", e);
                } else {
                    println!("compiled '{}' to '{}'", source, output);
                }
            }
            Err(e) => {
                eprintln!("error compiling {}: {}", source, e);
            }
        },
        Commands::Run {
            program,
            debug,
            ideal,
            noise,
            final_noise,
        } => {
            let noise_config_for_gates;
            let effective_final_noise;

            if ideal {
                eprintln!("[info] noise mode: ideal state (explicitly set, all noise disabled)");
                noise_config_for_gates = Some(NoiseConfig::Ideal);
                effective_final_noise = false;
            } else {
                noise_config_for_gates = match noise {
                    Some(s) if s == "random" => {
                        eprintln!("[info] noise mode: random depolarizing");
                        Some(NoiseConfig::Random)
                    }
                    Some(s) => {
                        let prob = s.parse::<f64>()
                            .map_err(|_| "invalid probability for --noise. must be a number between 0.0 and 1.0.".to_string())?;
                        if prob < 0.0 || prob > 1.0 {
                            return Err(
                                "noise probability must be between 0.0 and 1.0.".to_string()
                            );
                        }
                        eprintln!("[info] noise mode: fixed depolarizing ({})", prob);
                        Some(NoiseConfig::Fixed(prob))
                    }
                    None => {
                        eprintln!("[info] noise mode: random depolarizing (default)");
                        Some(NoiseConfig::Random)
                    }
                };
                effective_final_noise = final_noise;
            }

            match fs::read(&program) {
                Ok(filedata) => {
                    run_exe(
                        &filedata,
                        debug,
                        noise_config_for_gates,
                        effective_final_noise,
                    );
                }
                Err(e) => {
                    eprintln!("error reading program file {}: {}", program, e);
                }
            }
        }
        Commands::CompileJson { source, output } => match compile_qoa_to_json(&source, &output) {
            Ok(_) => {
                println!("compiled '{}' to json '{}'", source, output);
            }
            Err(e) => {
                eprintln!("error compiling {}: {}", source, e);
            }
        },
        Commands::Version => {
            println!("qoa version {}", QOA_VERSION);
        }
        Commands::Flags => {
            println!("available flags for the 'run' command:\n");
            let mut cli_cmd = Cli::command();
            if let Some(run_cmd) = cli_cmd.find_subcommand_mut("run") {
                let run_command_help = run_cmd.render_help().to_string();
                if let Some(options_start_index) = run_command_help.find("Options:") {
                    let options_section = &run_command_help[options_start_index..];
                    if let Some(end_index) = options_section.find("\n\n") {
                        println!("{}", &options_section[..end_index].trim_end());
                    } else {
                        println!("{}", options_section.trim_end());
                    }
                } else {
                    println!("could not find 'options:' section for 'run' command help.");
                    println!("{}", run_command_help);
                }
            } else {
                eprintln!("error: 'run' subcommand not found. this shouldn't happen.");
            }
        }
    }
    Ok(())
}
