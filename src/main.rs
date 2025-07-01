use clap::Parser;
use log::{debug, error, info, warn}; // import logging macros
use qoa::runtime::quantum_state::NoiseConfig;
use qoa::runtime::quantum_state::QuantumState;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng}; // keep Rng for rng.gen()
use serde_json::to_writer_pretty;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH}; // added for PathBuf
use crate::visualizer::SpectrumDirection;

mod instructions;
#[cfg(test)] // for testing
mod test;
mod visualizer;

const QEX: &[u8; 4] = b"QEX ";
const QEXE: &[u8; 4] = b"QEXE";
const OEXE: &[u8; 4] = b"OEXE";
const QOX: &[u8; 4] = b"QOX ";
const XEXE: &[u8; 4] = b"XEXE";
const QX: &[u8; 4] = b"QX\0\0";

const QOA_VERSION: &str = "0.2.6";
const QOA_AUTHOR: &str = "Rayan (@planetryan on GitHub)";

#[derive(Parser, Debug)]
#[command(name = "QOA", author = QOA_AUTHOR, version = QOA_VERSION,
    about = format!("QOA (Quantum Optical Assembly Language) - A Free, Open Source, Quantum QPU simulator and assembly language.\n
             Author: {QOA_AUTHOR}
             Version: {QOA_VERSION}\n
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
        #[arg(long, default_value_t = true)] // this makes --final-noise true by default
        final_noise: bool,
    },
    /// Compiles a .qoa source file into a .json circuit description (IonQ format).
    CompileJson {
        /// Source .qoa file path
        source: String,
        /// Output .json file path
        output: String,
    },
    /// Visualizes quantum state or circuit based on input data.

Visual {
    /// Input file path (e.g., .qexe or raw data for visualization)
    input: String,

    /// Output file path for the visualization (e.g., .png, .gif, .mp4)
    output: String,

    /// Resolution of the output visualization (e.g., "1920x1080")
    #[arg(long, default_value = "1920x1080")]
    resolution: String,

    /// Frames per second for animation (if applicable)
    #[arg(long, default_value_t = 60)] // default 60fps
    fps: u32,

    /// Spectrum direction Left-to-Right
    #[arg(long, conflicts_with = "rtl", default_value_t = false)]
    ltr: bool,

    /// Spectrum direction Right-to-Left
    #[arg(long, conflicts_with = "ltr", default_value_t = false)]
    rtl: bool,

    /// Additional ffmpeg arguments passed directly to ffmpeg (e.g., "-c:v libx264 -crf 23")
    #[arg(last = true, allow_hyphen_values = true)]
    ffmpeg_args: Vec<String>,
},

Version,

Flags,

}


// Helper function to parse resolution string
#[allow(dead_code)]
fn parse_resolution(res: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = res.split('x').collect();
    if parts.len() == 2 {
        let width = parts[0].parse::<u32>().ok();
        let height = parts[1].parse::<u32>().ok();
        if let (Some(w), Some(h)) = (width, height) {
            return Some((w, h));
        }
    }
    None
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
                    debug!("parsing line: '{}', encoded bytes: {:?}", line, encoded);
                }
                payload.extend(encoded);
            }
            Err(e) => {
                warn!("failed to parse instruction '{}': {}", line, e);
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
        error!("file too short to contain a valid header and payload length.");
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
            error!(
                "unknown or unsupported header: {:?} (as string: {:?})",
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
        error!(
            "file too short. expected {} bytes, got {}",
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
            error!("invalid or unsupported exe file, please check its header.");
            return;
        }
    };

    if debug_mode {
        debug!("payload length: {}", payload.len());
        debug!("payload snippet (first 32 bytes):");
        for j in 0..payload.len().min(32) {
            debug!(
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
                info!("noise mode: random depolarizing");
            }
            NoiseConfig::Fixed(value) => {
                info!("noise mode: fixed depolarizing ({})", value);
            }
            NoiseConfig::Ideal => {
                info!("noise mode: ideal state (no noise)");
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
            debug!("scanning opcode 0x{:02X} at byte {}", payload[i], i);
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
            0xAC /* GetTime */ | 0x50 /* Rand */ | 0x18 /* CharOut */ => { // added Rand and CharOut
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }
            0x08 /* PhaseShift */ | 0x22 /* RX */ | 0x23 /* RY */ | 0x0F /* RZ */ |
            0x24 /* Phase */ | 0x66 /* SetOpticalAttenuation */ | 0x67 /* DynamicPhaseCompensation */ |
            0x6A /* ApplyDisplacement */ | 0x6D /* ApplySqueezing */ |
            0x82 /* ApplyNonlinearPhaseShift */ | 0x83 /* ApplyNonlinearSigma */ |
            0x21 /* RegSet */ => { // added RegSet
                if i + 9 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x17 /* ControlledNot / CNOT */ | 0x1E /* CZ */ | 0x0B /* Swap */ |
            0x1F /* ThermalAvg */ | 0x65 /* OpticalRouting */ | 0x69 /* CrossPhaseModulation */ |
            0x20 /* WkbFactor */ | // moved WkbFactor here, it's 3 bytes
            0xA4 /* Cmp */ | 0x51 /* Sqrt */ | 0x52 /* Exp */ | 0x53 /* Log */ => { // added Sqrt, Exp, Log
                if i + 2 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                i += 3;
            }
            0x0C /* ControlledSwap */ | 0x54 /* RegAdd */ |
            0x55 /* RegSub */ | 0x56 /* RegMul */ | 0x57 /* RegDiv */ | 0x58 /* RegCopy */ |
            0x63 /* PhotonBunchingControl */ | 0xA8 /* NotBits */ | 0x31 /* CharLoad */ => { // added CharLoad
                if i + 3 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize).max(payload[i+3] as usize); // this line might be incorrect for all these opcodes, review
                i += 4;
            }
            0x11 /* Entangle */ | 0x12 /* EntangleBell */ => {
                if i + 2 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                i += 3;
            }
            0x13 /* EntangleMulti */ | 0x14 /* EntangleCluster */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                let num_qubits_in_list = payload[i+1] as usize;
                if i + 1 + num_qubits_in_list >= payload.len() { error!("incomplete instruction (entangle multi/cluster) at byte {}", i); break; }
                for q_idx in 0..num_qubits_in_list {
                    max_q = max_q.max(payload[i + 2 + q_idx] as usize);
                }
                i += 2 + num_qubits_in_list;
            }
            0x15 /* EntangleSwap */ => {
                if i + 4 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize).max(payload[i+3] as usize).max(payload[i+4] as usize);
                i += 5;
            }
            0x16 /* EntangleSwapMeasure */ => {
                if i + 4 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
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
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
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
            0x02 /* ApplyGate (QGATE) */ => {
                if i + 9 >= payload.len() { error!("incomplete qgate instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let name_bytes = &payload[i + 2..i + 10];
                let name = String::from_utf8_lossy(name_bytes).trim_end_matches('\0').to_string();
                if name.as_str() == "cz" {
                    if i + 10 >= payload.len() { error!("incomplete cz qgate instruction at byte {}", i); break; }
                    max_q = max_q.max(payload[i+10] as usize);
                    i += 11;
                } else {
                    i += 10;
                }
            }
            0x33 /* ApplyRotation */ => {
                if i + 10 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 11; // q (1 byte) + axis (1 byte) + angle (8 bytes) + opcode (1 byte)
            }
            0x34 /* ApplyMultiQubitRotation */ => {
                if i + 2 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                let num_qubits_in_list = payload[i+2] as usize;
                if i + 3 + num_qubits_in_list * 9 > payload.len() { error!("incomplete instruction (multi qubit rotation) at byte {}", i); break; }
                for q_idx in 0..num_qubits_in_list {
                    max_q = max_q.max(payload[i + 3 + q_idx] as usize);
                }
                i += 3 + num_qubits_in_list + num_qubits_in_list * 8; // opcode + axis + num_qs + qs_list + angles
            }
            0x35 /* ControlledPhaseRotation */ | 0x36 /* ApplyCPhase */ => {
                if i + 10 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                i += 11; // c (1 byte) + t (1 byte) + angle (8 bytes) + opcode (1 byte)
            }
            0x37 /* ApplyKerrNonlinearity */ => {
                if i + 17 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18; // q (1 byte) + strength (8 bytes) + duration (8 bytes) + opcode (1 byte)
            }
            0x39 /* DecoherenceProtect */ | 0x68 /* OpticalDelayLineControl */ => {
                if i + 9 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
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
                if i + 9 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x5E /* PhotonEmissionPattern */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let reg_start = i + 2;
                let reg_end = reg_start + payload[reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                if reg_end + 8 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                i = reg_end + 9; // opcode + q + reg_str + null_term + cycles (8 bytes)
            }
            0x5F /* PhotonDetectWithThreshold */ => {
                if i + 9 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let reg_start = i + 10;
                let reg_end = reg_start + payload[reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = reg_end + 1; // opcode + q + thresh (8 bytes) + reg_str + null_term
            }
            0x60 /* PhotonDetectCoincidence */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                let num_qubits_in_list = payload[i+1] as usize;
                if i + 2 + num_qubits_in_list >= payload.len() { error!("incomplete instruction (photon detect coincidence) at byte {}", i); break; }
                for q_idx in 0..num_qubits_in_list {
                    max_q = max_q.max(payload[i + 2 + q_idx] as usize);
                }
                let reg_start = i + 2 + num_qubits_in_list;
                let reg_end = reg_start + payload[reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = reg_end + 1;
            }
            0x64 /* PhotonRoute */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let from_start = i + 2;
                let from_end = from_start + payload[from_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let to_start = from_end + 1;
                let to_end = to_start + payload[to_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = to_end + 1;
            }
            0x6C /* ApplyDisplacementOperator */ => {
                if i + 17 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18; // q (1 byte) + alpha (8 bytes) + dur (8 bytes) + opcode (1 byte)
            }
            0x70 /* MeasureWithDelay */ => {
                if i + 9 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let reg_start = i + 10;
                let reg_end = reg_start + payload[reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = reg_end + 1; // opcode + q + delay (8 bytes) + reg_str + null_term
            }
            0x72 /* PhotonLossSimulate */ => {
                if i + 17 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18; // q (1 byte) + prob (8 bytes) + seed (8 bytes) + opcode (1 byte)
            }
            0x74 /* SetPos */ | 0x77 /* Move */ => {
                if i + 17 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18; // q (1 byte) + x (8 bytes) + y (8 bytes) + opcode (1 byte)
            }
            0x75 /* SetWl */ | 0x76 /* WlShift */ => {
                if i + 9 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10; // q (1 byte) + wl (8 bytes) + opcode (1 byte)
            }
            0x7E /* ErrorSyndrome */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let syndrome_start = i + 2;
                let syndrome_end = syndrome_start + payload[syndrome_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let result_reg_start = syndrome_end + 1;
                let result_reg_end = result_reg_start + payload[result_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = result_reg_end + 1;
            }
            0x80 /* BellStateVerification */ => {
                if i + 2 >= payload.len() { error!("incomplete instruction at byte {}", i); }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                let reg_start = i + 3;
                let reg_end = reg_start + payload[reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = reg_end + 1;
            }
            0x81 /* QuantumZenoEffect */ => {
                if i + 17 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18; // q (1 byte) + num_measurements (8 bytes) + interval_cycles (8 bytes) + opcode (1 byte)
            }
            0x84 /* ApplyLinearOpticalTransform */ => {
                if i + 4 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                let name_start = i + 4;
                let name_end = name_start + payload[name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let input_qs_len = payload[i+1] as usize;
                let output_qs_len = payload[i+2] as usize;
                let _num_modes = payload[i+3] as usize; // renamed to _num_modes to suppress warning

                let input_qs_start = name_end + 1;
                let input_qs_end = input_qs_start + input_qs_len;
                if input_qs_end > payload.len() { error!("incomplete instruction (linear optical transform input qs) at byte {}", i); break; }
                for q_idx in 0..input_qs_len {
                    max_q = max_q.max(payload[input_qs_start + q_idx] as usize);
                }

                let output_qs_start = input_qs_end;
                let output_qs_end = output_qs_start + output_qs_len;
                if output_qs_end > payload.len() { error!("incomplete instruction (linear optical transform output qs) at byte {}", i); break; }
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
                if i + 9 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                i += 9;
            }
            0x92 /* IfGt */ | 0x93 /* IfLt */ | 0x94 /* IfEq */ | 0x95 /* IfNe */ => {
                if i + 11 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                i += 11;
            }
            0x96 /* CallAddr */ | 0x9E /* LoadRegMem */ | 0xA2 /* Alloc */ => {
                if i + 9 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                i += 10;
            }
            0x9F /* StoreMemReg */ => {
                if i + 9 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                i += 10;
            }
            0x98 /* Printf */ => {
                if i + 9 >= payload.len() { error!("incomplete printf string length at byte {}", i); break; }
                let str_len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                let num_regs_idx = i + 9 + str_len;
                if num_regs_idx + 1 > payload.len() { error!("incomplete printf num_regs at byte {}", i); break; }
                let num_regs = payload[num_regs_idx] as usize;
                i += 1 + 8 + str_len + 1 + num_regs;
            }
            0x99 /* Print */ | 0x9A /* Println */ => {
                if i + 9 >= payload.len() { error!("incomplete print/println string length at byte {}", i); break; }
                let str_len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                i += 1 + 8 + str_len;
            }
            0xA5 /* AndBits */ | 0xA6 /* OrBits */ | 0xA7 /* XorBits */ | 0xA9 /* Shl */ | 0xAA /* Shr */ => {
                if i + 3 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                i += 4;
            }
            0xAE /* ExitCode */ => {
                if i + 5 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                i += 5;
            }
            0x01 /* LoopStart */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                i += 2;
            }
            _ => { // this catch-all should be last
                warn!("unknown opcode 0x{:02X} in scan at byte {}, skipping.", opcode, i);
                if debug_mode {
                    debug!("payload near unknown opcode:");
                    for j in i..(i + 10).min(payload.len()) {
                        debug!(
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
        warn!("simulating more than 16 qubits can be very memory intensive.");
    }

    info!(
        "initializing quantum state with {} qubits (type {}, ver {})",
        num_qubits, header, version
    );
    let mut qs = QuantumState::new(num_qubits, noise_config.clone());
    let _last_stats = Instant::now();
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

    let mut i = 0; // reset 'i' for the second pass
                   // second pass: execute instructions and interact with quantumstate
    while i < payload.len() {
        if debug_mode {
            debug!("executing opcode 0x{:02X} at byte {}", payload[i], i);
        }
        let opcode = payload[i];
        match opcode {
            0x04 /* QInit */ => {
                // quantumstate::new already initializes to |0...0>, so this is mostly a logical no-op
                // unless we want to reset a specific qubit to |0> if it's already entangled.
                // for now, just log if in debug mode.
                if debug_mode {
                    debug!("initialized qubit {}", payload[i + 1]);
                }
                i += 2;
            }
            0x01 /* LoopStart */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                i += 2;
            }
            0x02 /* ApplyGate (QGATE) */ => {
                if i + 9 >= payload.len() {
                    error!("incomplete qgate instruction at byte {}", i);
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
                        debug!("applied h gate on qubit {} (via qgate)", q);
                        i += 10;
                    }
                    "x" => {
                        qs.apply_x(q);
                        debug!("applied x gate on qubit {}", q);
                        i += 10;
                    }
                    "cz" => {
                        if i + 10 >= payload.len() {
                            error!(
                                "incomplete cz qgate instruction: missing target qubit at byte {}",
                                i
                            );
                            break;
                        }
                        let tgt = payload[i + 10] as usize;
                        qs.apply_cz(q, tgt);
                        debug!(
                            "applied cz gate between qubits {} (control) and {} (target)",
                            q, tgt
                        );
                        i += 11;
                    }
                    _ => {
                        warn!(
                            "unsupported QGATE '{}' at byte {}, skipping.",
                            name, i
                        );
                        i += 10; // skip the instruction
                    }
                }
            }
            0x05 /* ApplyHadamard */ => {
                qs.apply_h(payload[i + 1] as usize);
                if debug_mode {
                    debug!("applied hadamard on qubit {}", payload[i + 1]);
                }
                i += 2;
            }
            0x06 /* ApplyPhaseFlip */ => {
                // qs.apply_z(payload[i + 1] as usize);
                qs.apply_phase_flip(payload[i + 1] as usize); // corrected name
                if debug_mode {
                    debug!("applied phase flip (Z) on qubit {}", payload[i + 1]);
                }
                i += 2;
            }
            0x07 /* ApplyBitFlip */ => {
                qs.apply_x(payload[i + 1] as usize);
                if debug_mode {
                    debug!("applied bit flip (X) on qubit {}", payload[i + 1]);
                }
                i += 2;
            }
            0x08 /* PhaseShift */ => {
                let q = payload[i + 1] as usize;
                let angle_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_phase_shift(q, angle);
                if debug_mode {
                    debug!("applied phase shift on qubit {} with angle {}", q, angle);
                }
                i += 10;
            }
            0x0A /* Reset / QReset */ => {
                // qs.reset_qubit(payload[i + 1] as usize);
                debug!("reset qubit {}", payload[i + 1]);
                i += 2;
            }
            0x0B /* Swap */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                // qs.apply_swap(q1, q2);
                debug!("swapped qubits {} and {}", q1, q2);
                i += 3;
            }
            0x0C /* ControlledSwap */ => {
                let c = payload[i + 1] as usize;
                let q1 = payload[i + 2] as usize;
                let q2 = payload[i + 3] as usize;
                // qs.apply_cswap(c, q1, q2);
                debug!(
                    "applied controlled swap with control {} on qubits {} and {}",
                    c, q1, q2
                );
                i += 4;
            }
            0x0D /* ApplyTGate */ => {
                // qs.apply_t(payload[i + 1] as usize);
                qs.apply_t_gate(payload[i + 1] as usize); // corrected name
                if debug_mode {
                    debug!("applied t-gate on qubit {}", payload[i + 1]);
                }
                i += 2;
            }
            0x0E /* ApplySGate */ => {
                // qs.apply_s(payload[i + 1] as usize);
                qs.apply_s_gate(payload[i + 1] as usize); // corrected name
                if debug_mode {
                    debug!("applied s-gate on qubit {}", payload[i + 1]);
                }
                i += 2;
            }
            0x0F /* RZ */ => {
                let q = payload[i + 1] as usize;
                let angle_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_rz(q, angle);
                if debug_mode {
                    debug!("applied rz gate on qubit {} with angle {}", q, angle);
                }
                i += 10;
            }
            0x10 /* LoopEnd */ => {
                if let Some((loop_start_ptr, iterations_left)) = loop_stack.pop() {
                    if iterations_left > 1 {
                        loop_stack.push((loop_start_ptr, iterations_left - 1));
                        i = loop_start_ptr;
                        if debug_mode {
                            debug!(
                                "looping back to {} ({} iterations left)",
                                loop_start_ptr, iterations_left - 1
                            );
                        }
                    } else {
                        if debug_mode {
                            debug!("loop finished at {}", i);
                        }
                        i += 1;
                    }
                } else {
                    error!("loopend without matching loopstart at byte {}", i);
                    i += 1;
                }
            }
            0x11 /* Entangle */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                // qs.entangle(q1, q2);
                debug!("entangled qubits {} and {}", q1, q2);
                i += 3;
            }
            0x12 /* EntangleBell */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                // qs.entangle_bell_state(q1, q2);
                debug!("created bell state with qubits {} and {}", q1, q2);
                i += 3;
            }
            0x13 /* EntangleMulti */ => {
                let num_qubits = payload[i + 1] as usize;
                let qubits: Vec<usize> = payload[i + 2..i + 2 + num_qubits]
                    .iter()
                    .map(|&b| b as usize)
                    .collect();
                // qs.entangle_multi_qubit(&qubits);
                debug!("entangled multiple qubits: {:?}", qubits);
                i += 2 + num_qubits;
            }
            0x14 /* EntangleCluster */ => {
                let num_qubits = payload[i + 1] as usize;
                let qubits: Vec<usize> = payload[i + 2..i + 2 + num_qubits]
                    .iter()
                    .map(|&b| b as usize)
                    .collect();
                // qs.entangle_cluster_state(&qubits);
                debug!("created cluster state with qubits: {:?}", qubits);
                i += 2 + num_qubits;
            }
            0x15 /* EntangleSwap */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                let q3 = payload[i + 3] as usize;
                let q4 = payload[i + 4] as usize;
                // qs.entangle_swap(q1, q2, q3, q4);
                debug!(
                    "performed entanglement swap between ({}, {}) and ({}, {})",
                    q1, q2, q3, q4
                );
                i += 5;
            }
            0x16 /* EntangleSwapMeasure */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                let q3 = payload[i + 3] as usize;
                let q4 = payload[i + 4] as usize;
                let label_start = i + 5;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!(
                    "performed entanglement swap measure between ({}, {}) and ({}, {}) with label {}",
                    q1, q2, q3, q4, label
                );
                i = label_end + 1;
            }
            0x17 /* ControlledNot / CNOT */ => {
                qs.apply_cnot(payload[i + 1] as usize, payload[i + 2] as usize);
                if debug_mode {
                    debug!(
                        "applied cnot with control {} and target {}",
                        payload[i + 1],
                        payload[i + 2]
                    );
                }
                i += 3;
            }
            0x18 /* CharOut */ => {
                let q = payload[i+1] as usize;
                let classical_value = qs.measure(q) as u8;
                print!("{}", classical_value as char);
                if debug_mode {
                    debug!("char_out: qubit {} measured as {} ('{}')", q, classical_value, classical_value as char);
                }
                char_count += 1;
                char_sum += classical_value as u64;
                i += 2;
            }
            0x19 /* EntangleWithClassicalFeedback */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!(
                    "entangled with classical feedback on qubit {} with label {}",
                    q, label
                );
                i = label_end + 1;
            }
            0x1A /* EntangleDistributed */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!(
                    "performed distributed entanglement on qubit {} with label {}",
                    q, label
                );
                i = label_end + 1;
            }
            0x1B /* MeasureInBasis */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("measured qubit {} in basis {}", q, label);
                i = label_end + 1;
            }
            0x1C /* ResetAll */ => {
                // qs.reset_all_qubits();
                debug!("reset all qubits");
                i += 1;
            }
            0x1E /* CZ */ => {
                qs.apply_cz(payload[i + 1] as usize, payload[i + 2] as usize);
                if debug_mode {
                    debug!(
                        "applied cz gate on qubits {} and {}",
                        payload[i + 1],
                        payload[i + 2]
                    );
                }
                i += 3;
            }
            0x1F /* ThermalAvg */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                debug!("performed thermal averaging on qubits {} and {}", q1, q2);
                i += 3;
            }
            0x20 /* WkbFactor */ => { // 3-byte opcode for WkbFactor
                debug!("WkbFactor instruction (placeholder)");
                i += 3; // Fixed to consume 3 bytes as per first pass
            }
            0x21 /* RegSet */ => {
                let reg_idx = payload[i+1] as usize;
                let value_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let value = f64::from_le_bytes(value_bytes);
                if reg_idx < registers.len() {
                    registers[reg_idx] = value;
                    if debug_mode {
                        debug!("set register {} to {}", reg_idx, value);
                    }
                } else {
                    error!("invalid register index {} at byte {}", reg_idx, i);
                }
                i += 10;
            }
            0x22 /* RX */ => {
                let q = payload[i + 1] as usize;
                let angle_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_rx(q, angle);
                if debug_mode {
                    debug!("applied rx gate on qubit {} with angle {}", q, angle);
                }
                i += 10;
            }
            0x23 /* RY */ => {
                let q = payload[i + 1] as usize;
                let angle_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_ry(q, angle);
                if debug_mode {
                    debug!("applied ry gate on qubit {} with angle {}", q, angle);
                }
                i += 10;
            }
            0x24 /* Phase */ => {
                let q = payload[i+1] as usize;
                let angle_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                // qs.apply_phase(q, angle);
                if debug_mode {
                    debug!("applied phase gate on qubit {} with angle {}", q, angle);
                }
                i += 10;
            }
            0x31 /* CharLoad */ => {
                let dest_reg = payload[i+1] as usize;
                let char_val = payload[i+2] as char;
                if dest_reg < registers.len() {
                    registers[dest_reg] = char_val as u8 as f64;
                    if debug_mode {
                        debug!("loaded char '{}' into register {}", char_val, dest_reg);
                    }
                } else {
                    error!("invalid register index {} for CharLoad at byte {}", dest_reg, i);
                }
                i += 4; // opcode (1) + dest_reg (1) + char_val (1) + padding (1)
            }
            0x32 /* QMeas / Measure */ => {
                let q = payload[i + 1] as usize;
                let result = qs.measure(q);
                if debug_mode {
                    debug!("measured qubit {}: {:?}", q, result);
                }
                i += 2;
            }
            0x33 /* ApplyRotation */ => {
                let q = payload[i + 1] as usize;
                let axis = payload[i + 2] as char; // 'x', 'y', or 'z'
                let angle_bytes: [u8; 8] = payload[i + 3..i + 11].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                match axis {
                    'x' => qs.apply_rx(q, angle),
                    'y' => qs.apply_ry(q, angle),
                    'z' => qs.apply_rz(q, angle),
                    _ => warn!("unknown rotation axis '{}' at byte {}", axis, i),
                }
                if debug_mode {
                    debug!(
                        "applied rotation around {} axis on qubit {} with angle {}",
                        axis, q, angle
                    );
                }
                i += 11;
            }
            0x34 /* ApplyMultiQubitRotation */ => {
                let axis = payload[i + 1] as char;
                let num_qubits = payload[i + 2] as usize;
                let mut qubits = Vec::with_capacity(num_qubits);
                let mut current_idx = i + 3;
                for _ in 0..num_qubits {
                    qubits.push(payload[current_idx] as usize);
                    current_idx += 1;
                }
                let angle_bytes: [u8; 8] = payload[current_idx..current_idx + 8].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);

                // This is a simplified application for a multi-qubit rotation.
                // In a real scenario, this would apply a global rotation or a series of single-qubit rotations.
                for &q in &qubits {
                    match axis {
                        'x' => qs.apply_rx(q, angle),
                        'y' => qs.apply_ry(q, angle),
                        'z' => qs.apply_rz(q, angle),
                        _ => warn!("unknown rotation axis '{}' for multi-qubit rotation at byte {}", axis, i),
                    }
                }
                if debug_mode {
                    debug!(
                        "applied multi-qubit rotation around {} axis on qubits {:?} with angle {}",
                        axis, qubits, angle
                    );
                }
                i = current_idx + 8; // Move past the angle bytes
            }
            0x35 /* ControlledPhaseRotation */ => {
                let c = payload[i + 1] as usize;
                let t = payload[i + 2] as usize;
                let angle_bytes: [u8; 8] = payload[i + 3..i + 11].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_controlled_phase(c, t, angle);
                if debug_mode {
                    debug!(
                        "applied controlled phase rotation on control {} and target {} with angle {}",
                        c, t, angle
                    );
                }
                i += 11;
            }
            0x36 /* ApplyCPhase */ => {
                let c = payload[i + 1] as usize;
                let t = payload[i + 2] as usize;
                let angle_bytes: [u8; 8] = payload[i + 3..i + 11].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                // qs.apply_cphase(c, t, angle);
                if debug_mode {
                    debug!(
                        "applied controlled phase gate on control {} and target {} with angle {}",
                        c, t, angle
                    );
                }
                i += 11;
            }
            0x37 /* ApplyKerrNonlinearity */ => {
                let q = payload[i + 1] as usize;
                let strength_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let strength = f64::from_le_bytes(strength_bytes);
                let duration_bytes: [u8; 8] = payload[i + 10..i + 18].try_into().unwrap();
                let duration = f64::from_le_bytes(duration_bytes);
                debug!(
                    "applied kerr nonlinearity on qubit {} with strength {} and duration {}",
                    q, strength, duration
                );
                i += 18;
            }
            0x38 /* ApplyFeedforwardGate */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!(
                    "applied feedforward gate on qubit {} with label {}",
                    q, label
                );
                i = label_end + 1;
            }
            0x39 /* DecoherenceProtect */ => {
                let q = payload[i + 1] as usize;
                let duration_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let duration = f64::from_le_bytes(duration_bytes);
                debug!(
                    "applied decoherence protection on qubit {} for duration {}",
                    q, duration
                );
                i += 10;
            }
            0x3A /* ApplyMeasurementBasisChange */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!(
                    "applied measurement basis change on qubit {} to basis {}",
                    q, label
                );
                i = label_end + 1;
            }
            0x3B /* Load */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("loaded state to qubit {} from label {}", q, label);
                i = label_end + 1;
            }
            0x3C /* Store */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("stored state from qubit {} to label {}", q, label);
                i = label_end + 1;
            }
            0x3D /* LoadMem */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);

                let addr_name_start = reg_name_end + 1;
                let addr_name_end = addr_name_start + payload[addr_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let addr_name = String::from_utf8_lossy(&payload[addr_name_start..addr_name_end]);
                debug!("loaded value from memory address {} into register {}", addr_name, reg_name);
                i = addr_name_end + 1;
            }
            0x3E /* StoreMem */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);

                let addr_name_start = reg_name_end + 1;
                let addr_name_end = addr_name_start + payload[addr_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let addr_name = String::from_utf8_lossy(&payload[addr_name_start..addr_name_end]);
                debug!("stored value from register {} into memory address {}", reg_name, addr_name);
                i = addr_name_end + 1;
            }
            0x3F /* LoadClassical */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);

                let var_name_start = reg_name_end + 1;
                let var_name_end = var_name_start + payload[var_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let var_name = String::from_utf8_lossy(&payload[var_name_start..var_name_end]);
                debug!("loaded value from classical variable {} into register {}", var_name, reg_name);
                i = var_name_end + 1;
            }
            0x40 /* StoreClassical */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);

                let var_name_start = reg_name_end + 1;
                let var_name_end = var_name_start + payload[var_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let var_name = String::from_utf8_lossy(&payload[var_name_start..var_name_end]);
                debug!("stored value from register {} into classical variable {}", reg_name, var_name);
                i = var_name_end + 1;
            }
            0x41 /* Add */ => {
                let dst_reg_start = i + 1;
                let dst_reg_end = dst_reg_start + payload[dst_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let dst_reg_name = String::from_utf8_lossy(&payload[dst_reg_start..dst_reg_end]);

                let src1_reg_start = dst_reg_end + 1;
                let src1_reg_end = src1_reg_start + payload[src1_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src1_reg_name = String::from_utf8_lossy(&payload[src1_reg_start..src1_reg_end]);

                let src2_reg_start = src1_reg_end + 1;
                let src2_reg_end = src2_reg_start + payload[src2_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src2_reg_name = String::from_utf8_lossy(&payload[src2_reg_start..src2_reg_end]);
                debug!("added {} and {} into {}", src1_reg_name, src2_reg_name, dst_reg_name);
                i = src2_reg_end + 1;
            }
            0x42 /* Sub */ => {
                let dst_reg_start = i + 1;
                let dst_reg_end = dst_reg_start + payload[dst_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let dst_reg_name = String::from_utf8_lossy(&payload[dst_reg_start..dst_reg_end]);

                let src1_reg_start = dst_reg_end + 1;
                let src1_reg_end = src1_reg_start + payload[src1_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src1_reg_name = String::from_utf8_lossy(&payload[src1_reg_start..src1_reg_end]);

                let src2_reg_start = src1_reg_end + 1;
                let src2_reg_end = src2_reg_start + payload[src2_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src2_reg_name = String::from_utf8_lossy(&payload[src2_reg_start..src2_reg_end]);
                debug!("subtracted {} from {} into {}", src2_reg_name, src1_reg_name, dst_reg_name);
                i = src2_reg_end + 1;
            }
            0x43 /* And */ => {
                let dst_reg_start = i + 1;
                let dst_reg_end = dst_reg_start + payload[dst_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let dst_reg_name = String::from_utf8_lossy(&payload[dst_reg_start..dst_reg_end]);

                let src1_reg_start = dst_reg_end + 1;
                let src1_reg_end = src1_reg_start + payload[src1_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src1_reg_name = String::from_utf8_lossy(&payload[src1_reg_start..src1_reg_end]);

                let src2_reg_start = src1_reg_end + 1;
                let src2_reg_end = src2_reg_start + payload[src2_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src2_reg_name = String::from_utf8_lossy(&payload[src2_reg_start..src2_reg_end]);
                debug!("ANDed {} and {} into {}", src1_reg_name, src2_reg_name, dst_reg_name);
                i = src2_reg_end + 1;
            }
            0x44 /* Or */ => {
                let dst_reg_start = i + 1;
                let dst_reg_end = dst_reg_start + payload[dst_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let dst_reg_name = String::from_utf8_lossy(&payload[dst_reg_start..dst_reg_end]);

                let src1_reg_start = dst_reg_end + 1;
                let src1_reg_end = src1_reg_start + payload[src1_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src1_reg_name = String::from_utf8_lossy(&payload[src1_reg_start..src1_reg_end]);

                let src2_reg_start = src1_reg_end + 1;
                let src2_reg_end = src2_reg_start + payload[src2_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src2_reg_name = String::from_utf8_lossy(&payload[src2_reg_start..src2_reg_end]);
                debug!("ORed {} and {} into {}", src1_reg_name, src2_reg_name, dst_reg_name);
                i = src2_reg_end + 1;
            }
            0x45 /* Xor */ => {
                let dst_reg_start = i + 1;
                let dst_reg_end = dst_reg_start + payload[dst_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let dst_reg_name = String::from_utf8_lossy(&payload[dst_reg_start..dst_reg_end]);

                let src1_reg_start = dst_reg_end + 1;
                let src1_reg_end = src1_reg_start + payload[src1_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src1_reg_name = String::from_utf8_lossy(&payload[src1_reg_start..src1_reg_end]);

                let src2_reg_start = src1_reg_end + 1;
                let src2_reg_end = src2_reg_start + payload[src2_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src2_reg_name = String::from_utf8_lossy(&payload[src2_reg_start..src2_reg_end]);
                debug!("XORed {} and {} into {}", src1_reg_name, src2_reg_name, dst_reg_name);
                i = src2_reg_end + 1;
            }
            0x46 /* Not */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                debug!("NOTed {}", reg_name);
                i = reg_name_end + 1;
            }
            0x47 /* Push */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                debug!("pushed register {}", reg_name);
                i = reg_name_end + 1;
            }
            0x48 /* Sync */ => {
                debug!("sync instruction (placeholder)");
                i += 1;
            }
            0x49 /* Jump */ => {
                let label_start = i + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("jumped to label {}", label);
                i = label_end + 1;
            }
            0x4A /* JumpIfZero */ => {
                let cond_reg_start = i + 1;
                let cond_reg_end = cond_reg_start + payload[cond_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let cond_reg_name = String::from_utf8_lossy(&payload[cond_reg_start..cond_reg_end]);

                let label_start = cond_reg_end + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("jumped to label {} if register {} is zero", label, cond_reg_name);
                i = label_end + 1;
            }
            0x4B /* JumpIfOne */ => {
                let cond_reg_start = i + 1;
                let cond_reg_end = cond_reg_start + payload[cond_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let cond_reg_name = String::from_utf8_lossy(&payload[cond_reg_start..cond_reg_end]);

                let label_start = cond_reg_end + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("jumped to label {} if register {} is one", label, cond_reg_name);
                i = label_end + 1;
            }
            0x4C /* Call */ => {
                let label_start = i + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("called subroutine at label {}", label);
                i = label_end + 1;
            }
            0x4D /* Return */ => {
                debug!("returned from subroutine");
                i += 1;
            }
            0x4E /* TimeDelay */ => {
                let q = payload[i + 1] as usize;
                let delay_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let delay = f64::from_le_bytes(delay_bytes);
                debug!("time delay on qubit {} for {} units", q, delay);
                i += 10;
            }
            0x4F /* Pop */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                debug!("popped value into register {}", reg_name);
                i = reg_name_end + 1;
            }
            0x50 /* Rand */ => {
                let dest_reg_idx = payload[i+1] as usize;
                if dest_reg_idx < registers.len() {
                    registers[dest_reg_idx] = rng.gen::<f64>();
                    if debug_mode {
                        debug!("generated random number {} into register {}", registers[dest_reg_idx], dest_reg_idx);
                    }
                } else {
                    error!("invalid register index {} for Rand at byte {}", dest_reg_idx, i);
                }
                i += 2;
            }
            0x51 /* Sqrt */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                debug!("sqrt on qubits {} and {} (placeholder)", q1, q2);
                i += 3;
            }
            0x52 /* Exp */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                debug!("exp on qubits {} and {} (placeholder)", q1, q2);
                i += 3;
            }
            0x53 /* Log */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                debug!("log on qubits {} and {} (placeholder)", q1, q2);
                i += 3;
            }
            0x54 /* RegAdd */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = registers[src1_reg] + registers[src2_reg];
                    if debug_mode {
                        debug!("reg_add: r{} = r{} + r{} ({})", dest_reg, src1_reg, src2_reg, registers[dest_reg]);
                    }
                } else {
                    error!("invalid register index for RegAdd at byte {}", i);
                }
                i += 4;
            }
            0x55 /* RegSub */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = registers[src1_reg] - registers[src2_reg];
                    if debug_mode {
                        debug!("reg_sub: r{} = r{} - r{} ({})", dest_reg, src1_reg, src2_reg, registers[dest_reg]);
                    }
                } else {
                    error!("invalid register index for RegSub at byte {}", i);
                }
                i += 4;
            }
            0x56 /* RegMul */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = registers[src1_reg] * registers[src2_reg];
                    if debug_mode {
                        debug!("reg_mul: r{} = r{} * r{} ({})", dest_reg, src1_reg, src2_reg, registers[dest_reg]);
                    }
                } else {
                    error!("invalid register index for RegMul at byte {}", i);
                }
                i += 4;
            }
            0x57 /* RegDiv */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    if registers[src2_reg] != 0.0 {
                        registers[dest_reg] = registers[src1_reg] / registers[src2_reg];
                        if debug_mode {
                            debug!("reg_div: r{} = r{} / r{} ({})", dest_reg, src1_reg, src2_reg, registers[dest_reg]);
                        }
                    } else {
                        error!("division by zero in RegDiv at byte {}", i);
                    }
                } else {
                    error!("invalid register index for RegDiv at byte {}", i);
                }
                i += 4;
            }
            0x58 /* RegCopy */ => {
                let dest_reg = payload[i+1] as usize;
                let src_reg = payload[i+2] as usize;
                if dest_reg < registers.len() && src_reg < registers.len() {
                    registers[dest_reg] = registers[src_reg];
                    if debug_mode {
                        debug!("reg_copy: r{} = r{} ({})", dest_reg, src_reg, registers[dest_reg]);
                    }
                } else {
                    error!("invalid register index for RegCopy at byte {}", i);
                }
                i += 4; // opcode (1) + dest (1) + src (1) + padding (1)
            }
            0x59 /* PhotonEmit */ => {
                let q = payload[i + 1] as usize;
                debug!("emitted photon from qubit {}", q);
                i += 2;
            }
            0x5A /* PhotonDetect */ => {
                let q = payload[i + 1] as usize;
                let result = qs.measure(q); // Simplified: measure qubit state for detection
                debug!("detected photon at qubit {}: {:?}", q, result);
                i += 2;
            }
            0x5B /* PhotonCount */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("counted photons at qubit {} into label {}", q, label);
                i = label_end + 1;
            }
            0x5C /* PhotonAddition */ => {
                let q = payload[i + 1] as usize;
                debug!("added photon to qubit {}", q);
                i += 2;
            }
            0x5D /* ApplyPhotonSubtraction */ => {
                let q = payload[i + 1] as usize;
                debug!("subtracted photon from qubit {}", q);
                i += 2;
            }
            0x5E /* PhotonEmissionPattern */ => {
                let q = payload[i+1] as usize;
                let reg_name_start = i + 2;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let cycles_bytes: [u8; 8] = payload[reg_name_end + 1..reg_name_end + 9].try_into().unwrap();
                let cycles = u64::from_le_bytes(cycles_bytes);
                debug!("set photon emission pattern for qubit {} from register {} for {} cycles", q, reg_name, cycles);
                i = reg_name_end + 9;
            }
            0x5F /* PhotonDetectWithThreshold */ => {
                let q = payload[i+1] as usize;
                let threshold_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let threshold = f64::from_le_bytes(threshold_bytes);
                let reg_name_start = i + 10;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                debug!("detected photon at qubit {} with threshold {} into register {}", q, threshold, reg_name);
                i = reg_name_end + 1;
            }
            0x60 /* PhotonDetectCoincidence */ => {
                let num_qubits = payload[i+1] as usize;
                let qubits_start = i + 2;
                let qubits_end = qubits_start + num_qubits;
                let qubits: Vec<usize> = payload[qubits_start..qubits_end].iter().map(|&b| b as usize).collect();
                let reg_name_start = qubits_end;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                debug!("detected photon coincidence for qubits {:?} into register {}", qubits, reg_name);
                i = reg_name_end + 1;
            }
            0x61 /* SinglePhotonSourceOn */ => {
                let q = payload[i+1] as usize;
                debug!("single photon source on for qubit {}", q);
                i += 2;
            }
            0x62 /* SinglePhotonSourceOff */ => {
                let q = payload[i+1] as usize;
                debug!("single photon source off for qubit {}", q);
                i += 2;
            }
            0x63 /* PhotonBunchingControl */ => {
                let q = payload[i+1] as usize;
                let control_reg_idx = payload[i+2] as usize;
                if control_reg_idx < registers.len() {
                    debug!("photon bunching control for qubit {} with register {} value {}", q, control_reg_idx, registers[control_reg_idx]);
                } else {
                    error!("invalid register index for PhotonBunchingControl at byte {}", i);
                }
                i += 4; // opcode (1) + q (1) + control_reg (1) + padding (1)
            }
            0x64 /* PhotonRoute */ => {
                let q = payload[i+1] as usize;
                let from_path_start = i + 2;
                let from_path_end = from_path_start + payload[from_path_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let from_path = String::from_utf8_lossy(&payload[from_path_start..from_path_end]);
                let to_path_start = from_path_end + 1;
                let to_path_end = to_path_start + payload[to_path_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let to_path = String::from_utf8_lossy(&payload[to_path_start..to_path_end]);
                debug!("routed photon from {} to {} for qubit {}", from_path, to_path, q);
                i = to_path_end + 1;
            }
            0x65 /* OpticalRouting */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                debug!("optical routing between qubits {} and {} (placeholder)", q1, q2);
                i += 3;
            }
            0x66 /* SetOpticalAttenuation */ => {
                let q = payload[i + 1] as usize;
                let attenuation_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let attenuation = f64::from_le_bytes(attenuation_bytes);
                debug!(
                    "set optical attenuation for qubit {} to {}",
                    q, attenuation
                );
                i += 10;
            }
            0x67 /* DynamicPhaseCompensation */ => {
                let q = payload[i + 1] as usize;
                let compensation_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let compensation = f64::from_le_bytes(compensation_bytes);
                debug!(
                    "applied dynamic phase compensation for qubit {} with value {}",
                    q, compensation
                );
                i += 10;
            }
            0x68 /* OpticalDelayLineControl */ => {
                let q = payload[i + 1] as usize;
                let delay_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let delay = f64::from_le_bytes(delay_bytes);
                debug!("controlled optical delay line for qubit {} with delay {}", q, delay);
                i += 10;
            }
            0x69 /* CrossPhaseModulation */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                debug!("cross-phase modulation between qubits {} and {} (placeholder)", q1, q2);
                i += 3;
            }
            0x6A /* ApplyDisplacement */ => {
                let q = payload[i + 1] as usize;
                let alpha_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let alpha = f64::from_le_bytes(alpha_bytes);
                debug!("applied displacement to qubit {} with alpha {}", q, alpha);
                i += 10;
            }
            0x6B /* ApplyDisplacementFeedback */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("applied displacement feedback to qubit {} with label {}", q, label);
                i = label_end + 1;
            }
            0x6C /* ApplyDisplacementOperator */ => {
                let q = payload[i + 1] as usize;
                let alpha_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let alpha = f64::from_le_bytes(alpha_bytes);
                let duration_bytes: [u8; 8] = payload[i + 10..i + 18].try_into().unwrap();
                let duration = f64::from_le_bytes(duration_bytes);
                debug!(
                    "applied displacement operator to qubit {} with alpha {} for duration {}",
                    q, alpha, duration
                );
                i += 18;
            }
            0x6D /* ApplySqueezing */ => {
                let q = payload[i + 1] as usize;
                let r_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let r = f64::from_le_bytes(r_bytes);
                debug!("applied squeezing to qubit {} with r {}", q, r);
                i += 10;
            }
            0x6E /* ApplySqueezingFeedback */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("applied squeezing feedback to qubit {} with label {}", q, label);
                i = label_end + 1;
            }
            0x6F /* MeasureParity */ => {
                let q = payload[i+1] as usize;
                debug!("measured parity on qubit {}", q);
                i += 2;
            }
            0x70 /* MeasureWithDelay */ => {
                let q = payload[i+1] as usize;
                let delay_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let delay = f64::from_le_bytes(delay_bytes);
                let reg_name_start = i + 10;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                debug!("measured qubit {} with delay {} into register {}", q, delay, reg_name);
                i = reg_name_end + 1;
            }
            0x71 /* OpticalSwitchControl */ => {
                let q = payload[i+1] as usize;
                debug!("optical switch control for qubit {}", q);
                i += 2;
            }
            0x72 /* PhotonLossSimulate */ => {
                let q = payload[i+1] as usize;
                let prob_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let prob = f64::from_le_bytes(prob_bytes);
                let seed_bytes: [u8; 8] = payload[i+10..i+18].try_into().unwrap();
                let seed = u64::from_le_bytes(seed_bytes);
                debug!("simulated photon loss for qubit {} with probability {} and seed {}", q, prob, seed);
                i += 18;
            }
            0x73 /* PhotonLossCorrection */ => {
                let q = payload[i+1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("applied photon loss correction for qubit {} with label {}", q, label);
                i = label_end + 1;
            }
            0x74 /* SetPos */ => {
                let q = payload[i+1] as usize;
                let x_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let x = f64::from_le_bytes(x_bytes);
                let y_bytes: [u8; 8] = payload[i+10..i+18].try_into().unwrap();
                let y = f64::from_le_bytes(y_bytes);
                debug!("set position for qubit {} to ({}, {})", q, x, y);
                i += 18;
            }
            0x75 /* SetWl */ => {
                let q = payload[i+1] as usize;
                let wl_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let wl = f64::from_le_bytes(wl_bytes);
                debug!("set wavelength for qubit {} to {}", q, wl);
                i += 10;
            }
            0x76 /* WlShift */ => {
                let q = payload[i+1] as usize;
                let shift_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let shift = f64::from_le_bytes(shift_bytes);
                debug!("shifted wavelength for qubit {} by {}", q, shift);
                i += 10;
            }
            0x77 /* Move */ => {
                let q = payload[i+1] as usize;
                let x_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let x = f64::from_le_bytes(x_bytes);
                let y_bytes: [u8; 8] = payload[i+10..i+18].try_into().unwrap();
                let y = f64::from_le_bytes(y_bytes);
                debug!("moved qubit {} by ({}, {})", q, x, y);
                i += 18;
            }
            0x7E /* ErrorSyndrome */ => {
                let q = payload[i+1] as usize;
                let syndrome_name_start = i + 2;
                let syndrome_name_end = syndrome_name_start + payload[syndrome_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let syndrome_name = String::from_utf8_lossy(&payload[syndrome_name_start..syndrome_name_end]);
                let result_reg_name_start = syndrome_name_end + 1;
                let result_reg_name_end = result_reg_name_start + payload[result_reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let result_reg_name = String::from_utf8_lossy(&payload[result_reg_name_start..result_reg_name_end]);
                debug!("obtained error syndrome for qubit {} with name {} into register {}", q, syndrome_name, result_reg_name);
                i = result_reg_name_end + 1;
            }
            0x7F /* QuantumStateTomography */ => {
                let q = payload[i+1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("performed quantum state tomography on qubit {} with label {}", q, label);
                i = label_end + 1;
            }
            0x80 /* BellStateVerification */ => {
                let q1 = payload[i+1] as usize;
                let q2 = payload[i+2] as usize;
                let reg_name_start = i + 3;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                debug!("performed bell state verification for qubits {} and {} into register {}", q1, q2, reg_name);
                i = reg_name_end + 1;
            }
            0x81 /* QuantumZenoEffect */ => {
                let q = payload[i+1] as usize;
                let num_measurements_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let num_measurements = u64::from_le_bytes(num_measurements_bytes);
                let interval_cycles_bytes: [u8; 8] = payload[i+10..i+18].try_into().unwrap();
                let interval_cycles = u64::from_le_bytes(interval_cycles_bytes);
                debug!("applied quantum zeno effect on qubit {} with {} measurements at {} cycles interval", q, num_measurements, interval_cycles);
                i += 18;
            }
            0x82 /* ApplyNonlinearPhaseShift */ => {
                let q = payload[i+1] as usize;
                let shift_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let shift = f64::from_le_bytes(shift_bytes);
                debug!("applied nonlinear phase shift to qubit {} with shift {}", q, shift);
                i += 10;
            }
            0x83 /* ApplyNonlinearSigma */ => {
                let q = payload[i+1] as usize;
                let sigma_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let sigma = f64::from_le_bytes(sigma_bytes);
                debug!("applied nonlinear sigma to qubit {} with sigma {}", q, sigma);
                i += 10;
            }
            0x84 /* ApplyLinearOpticalTransform */ => {
                let input_qs_len = payload[i+1] as usize;
                let output_qs_len = payload[i+2] as usize;
                let num_modes = payload[i+3] as usize;
                let name_start = i + 4;
                let name_end = name_start + payload[name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let name = String::from_utf8_lossy(&payload[name_start..name_end]);

                let input_qs_start = name_end + 1;
                let input_qs_end = input_qs_start + input_qs_len;
                let input_qubits: Vec<usize> = payload[input_qs_start..input_qs_end].iter().map(|&b| b as usize).collect();

                let output_qs_start = input_qs_end;
                let output_qs_end = output_qs_start + output_qs_len;
                let output_qubits: Vec<usize> = payload[output_qs_start..output_qs_end].iter().map(|&b| b as usize).collect();
                debug!(
                    "applied linear optical transform '{}' with {} modes, input {:?}, output {:?}",
                    name, num_modes, input_qubits, output_qubits
                );
                i = output_qs_end;
            }
            0x85 /* PhotonNumberResolvingDetection */ => {
                let q = payload[i+1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("performed photon number resolving detection on qubit {} with label {}", q, label);
                i = label_end + 1;
            }
            0x86 /* FeedbackControl */ => {
                let q = payload[i+1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("applied feedback control for qubit {} with label {}", q, label);
                i = label_end + 1;
            }
            0x87 /* VerboseLog */ => {
                let q = payload[i+1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                debug!("verbose log for qubit {} with message: {}", q, label);
                i = label_end + 1;
            }
            0x88 /* Comment */ => {
                let text_start = i + 1;
                let text_end = text_start + payload[text_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let text = String::from_utf8_lossy(&payload[text_start..text_end]);
                debug!("comment: {}", text);
                i = text_end + 1;
            }
            0x89 /* Barrier */ => {
                debug!("barrier instruction (placeholder)");
                i += 1;
            }
            0x90 /* Jmp */ => {
                let addr_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                if debug_mode {
                    debug!("relative jump by {}", addr);
                }
                i += addr + 9;
            }
            0x91 /* JmpAbs */ => {
                let addr_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                if debug_mode {
                    debug!("absolute jump to {}", addr);
                }
                i = addr;
            }
            0x92 /* IfGt */ => {
                let reg1_idx = payload[i+1] as usize;
                let reg2_idx = payload[i+2] as usize;
                let jump_addr_bytes: [u8; 8] = payload[i+3..i+11].try_into().unwrap();
                let jump_addr = u64::from_le_bytes(jump_addr_bytes) as usize;

                if reg1_idx < registers.len() && reg2_idx < registers.len() {
                    if registers[reg1_idx] > registers[reg2_idx] {
                        i = jump_addr;
                        if debug_mode {
                            debug!("if_gt: r{} ({}) > r{} ({}), jumped to {}", reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx], jump_addr);
                        }
                    } else {
                        i += 11;
                        if debug_mode {
                            debug!("if_gt: r{} ({}) not > r{} ({}), no jump", reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx]);
                        }
                    }
                } else {
                    error!("invalid register index for IfGt at byte {}", i);
                    i += 11;
                }
            }
            0x93 /* IfLt */ => {
                let reg1_idx = payload[i+1] as usize;
                let reg2_idx = payload[i+2] as usize;
                let jump_addr_bytes: [u8; 8] = payload[i+3..i+11].try_into().unwrap();
                let jump_addr = u64::from_le_bytes(jump_addr_bytes) as usize;

                if reg1_idx < registers.len() && reg2_idx < registers.len() {
                    if registers[reg1_idx] < registers[reg2_idx] {
                        i = jump_addr;
                        if debug_mode {
                            debug!("if_lt: r{} ({}) < r{} ({}), jumped to {}", reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx], jump_addr);
                        }
                    } else {
                        i += 11;
                        if debug_mode {
                            debug!("if_lt: r{} ({}) not < r{} ({}), no jump", reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx]);
                        }
                    }
                } else {
                    error!("invalid register index for IfLt at byte {}", i);
                    i += 11;
                }
            }
            0x94 /* IfEq */ => {
                let reg1_idx = payload[i+1] as usize;
                let reg2_idx = payload[i+2] as usize;
                let jump_addr_bytes: [u8; 8] = payload[i+3..i+11].try_into().unwrap();
                let jump_addr = u64::from_le_bytes(jump_addr_bytes) as usize;

                if reg1_idx < registers.len() && reg2_idx < registers.len() {
                    if (registers[reg1_idx] - registers[reg2_idx]).abs() < f64::EPSILON { // Floating point comparison
                        i = jump_addr;
                        if debug_mode {
                            debug!("if_eq: r{} ({}) == r{} ({}), jumped to {}", reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx], jump_addr);
                        }
                    } else {
                        i += 11;
                        if debug_mode {
                            debug!("if_eq: r{} ({}) != r{} ({}), no jump", reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx]);
                        }
                    }
                } else {
                    error!("invalid register index for IfEq at byte {}", i);
                    i += 11;
                }
            }
            0x95 /* IfNe */ => {
                let reg1_idx = payload[i+1] as usize;
                let reg2_idx = payload[i+2] as usize;
                let jump_addr_bytes: [u8; 8] = payload[i+3..i+11].try_into().unwrap();
                let jump_addr = u64::from_le_bytes(jump_addr_bytes) as usize;

                if reg1_idx < registers.len() && reg2_idx < registers.len() {
                    if (registers[reg1_idx] - registers[reg2_idx]).abs() >= f64::EPSILON { // Floating point comparison
                        i = jump_addr;
                        if debug_mode {
                            debug!("if_ne: r{} ({}) != r{} ({}), jumped to {}", reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx], jump_addr);
                        }
                    } else {
                        i += 11;
                        if debug_mode {
                            debug!("if_ne: r{} ({}) == r{} ({}), no jump", reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx]);
                        }
                    }
                } else {
                    error!("invalid register index for IfNe at byte {}", i);
                    i += 11;
                }
            }
            0x96 /* CallAddr */ => {
                let addr_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                call_stack.push(i + 10); // Return address
                i = addr;
                if debug_mode {
                    debug!("call address {}. return address {}", addr, call_stack.last().unwrap());
                }
            }
            0x97 /* RetSub */ => {
                if let Some(ret_addr) = call_stack.pop() {
                    i = ret_addr;
                    if debug_mode {
                        debug!("returned from subroutine to address {}", ret_addr);
                    }
                } else {
                    error!("ret_sub without matching call at byte {}", i);
                    i += 1;
                }
            }
            0x98 /* Printf */ => {
                let str_len_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let str_len = u64::from_le_bytes(str_len_bytes) as usize;
                let format_str_bytes = &payload[i+9..i+9+str_len];
                let format_str = String::from_utf8_lossy(format_str_bytes);
                let num_regs_idx = i + 9 + str_len;
                let num_regs = payload[num_regs_idx] as usize;
                let mut args = Vec::new();
                for k in 0..num_regs {
                    let reg_idx = payload[num_regs_idx + 1 + k] as usize;
                    if reg_idx < registers.len() {
                        args.push(registers[reg_idx]);
                    } else {
                        error!("invalid register index {} for Printf at byte {}", reg_idx, i);
                    }
                }
                // Simplified printf, does not fully handle all format specifiers
                let formatted_str = format_str.replace("%f", &format!("{:?}", args));
                print!("{}", formatted_str);
                if debug_mode {
                    debug!("printf: \"{}\" with args {:?}", format_str, args);
                }
                i += 1 + 8 + str_len + 1 + num_regs;
            }
            0x99 /* Print */ => {
                let str_len_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let str_len = u64::from_le_bytes(str_len_bytes) as usize;
                let text_bytes = &payload[i+9..i+9+str_len];
                let text = String::from_utf8_lossy(text_bytes);
                print!("{}", text);
                if debug_mode {
                    debug!("print: \"{}\"", text);
                }
                i += 1 + 8 + str_len;
            }
            0x9A /* Println */ => {
                let str_len_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let str_len = u64::from_le_bytes(str_len_bytes) as usize;
                let text_bytes = &payload[i+9..i+9+str_len];
                let text = String::from_utf8_lossy(text_bytes);
                println!("{}", text);
                if debug_mode {
                    debug!("println: \"{}\"", text);
                }
                i += 1 + 8 + str_len;
            }
            0x9B /* Input */ => {
                let q = payload[i+1] as usize;
                let mut input_line = String::new();
                info!("input requested for qubit {}. enter a bit (0 or 1):", q);
                io::stdin().read_line(&mut input_line).expect("failed to read line");
                let bit = input_line.trim().parse::<u8>().unwrap_or(0);
                // For now, directly apply the bit to the qubit's classical state if such a concept exists
                // In a true quantum simulator, this would involve more complex operations
                if debug_mode {
                    debug!("input: read {} for qubit {}", bit, q);
                }
                i += 2;
            }
            0x9C /* DumpState */ => {
                // qs.debug_dump_state();
                i += 1;
            }
            0x9D /* DumpRegs */ => {
                debug!("register values: {:?}", registers);
                i += 1;
            }
            0x9E /* LoadRegMem */ => {
                let reg_idx = payload[i+1] as usize;
                let addr_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                if reg_idx < registers.len() && addr < memory.len() {
                    // Load 8 bytes (f64) from memory
                    let mut val_bytes = [0u8; 8];
                    val_bytes.copy_from_slice(&memory[addr..addr+8]);
                    registers[reg_idx] = f64::from_le_bytes(val_bytes);
                    if debug_mode {
                        debug!("loaded memory address {} into register {}", addr, reg_idx);
                    }
                } else {
                    error!("invalid register or memory address for LoadRegMem at byte {}", i);
                }
                i += 10;
            }
            0x9F /* StoreMemReg */ => {
                let reg_idx = payload[i+1] as usize;
                let addr_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                if reg_idx < registers.len() && addr + 8 <= memory.len() {
                    // Store 8 bytes (f64) to memory
                    memory[addr..addr+8].copy_from_slice(&registers[reg_idx].to_le_bytes());
                    if debug_mode {
                        debug!("stored register {} into memory address {}", reg_idx, addr);
                    }
                } else {
                    error!("invalid register or memory address for StoreMemReg at byte {}", i);
                }
                i += 10;
            }
            0xA0 /* PushReg */ => {
                let reg_idx = payload[i+1] as usize;
                // Simplified: assuming a stack of f64s. In a real VM, might push to general stack.
                if reg_idx < registers.len() {
                    call_stack.push(registers[reg_idx] as usize); // Re-using call_stack for simplicity, convert f64 to usize
                    if debug_mode {
                        debug!("pushed register {} ({}) to stack", reg_idx, registers[reg_idx]);
                    }
                } else {
                    error!("invalid register index for PushReg at byte {}", i);
                }
                i += 2;
            }
            0xA1 /* PopReg */ => {
                let reg_idx = payload[i+1] as usize;
                if reg_idx < registers.len() {
                    if let Some(val) = call_stack.pop() {
                        registers[reg_idx] = val as f64; // Convert usize back to f64
                        if debug_mode {
                            debug!("popped value ({}) into register {}", val, reg_idx);
                        }
                    } else {
                        error!("pop_reg on empty stack at byte {}", i);
                    }
                } else {
                    error!("invalid register index for PopReg at byte {}", i);
                }
                i += 2;
            }
            0xA2 /* Alloc */ => {
                let size_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let size = u64::from_le_bytes(size_bytes) as usize;
                debug!("allocated {} bytes (placeholder)", size);
                i += 10;
            }
            0xA3 /* Free */ => {
                let addr_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                debug!("freed memory at address {} (placeholder)", addr);
                i += 9;
            }
            0xA4 /* Cmp */ => {
                let reg1_idx = payload[i+1] as usize;
                let reg2_idx = payload[i+2] as usize;
                // Compares two registers and sets an internal flag (not explicitly stored as `last_cmp_result` anymore).
                // Conditional jumps (IfGt, IfLt, IfEq, IfNe) now directly compare.
                if reg1_idx < registers.len() && reg2_idx < registers.len() {
                    if debug_mode {
                        debug!("compared r{} ({}) and r{} ({})", reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx]);
                    }
                } else {
                    error!("invalid register index for Cmp at byte {}", i);
                }
                i += 3;
            }
            0xA5 /* AndBits */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = ((registers[src1_reg] as u64) & (registers[src2_reg] as u64)) as f64;
                    if debug_mode {
                        debug!("and_bits: r{} = r{} & r{} ({})", dest_reg, src1_reg, src2_reg, registers[dest_reg]);
                    }
                } else {
                    error!("invalid register index for AndBits at byte {}", i);
                }
                i += 4;
            }
            0xA6 /* OrBits */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = ((registers[src1_reg] as u64) | (registers[src2_reg] as u64)) as f64;
                    if debug_mode {
                        debug!("or_bits: r{} = r{} | r{} ({})", dest_reg, src1_reg, src2_reg, registers[dest_reg]);
                    }
                } else {
                    error!("invalid register index for OrBits at byte {}", i);
                }
                i += 4;
            }
            0xA7 /* XorBits */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = ((registers[src1_reg] as u64) ^ (registers[src2_reg] as u64)) as f64;
                    if debug_mode {
                        debug!("xor_bits: r{} = r{} ^ r{} ({})", dest_reg, src1_reg, src2_reg, registers[dest_reg]);
                    }
                } else {
                    error!("invalid register index for XorBits at byte {}", i);
                }
                i += 4;
            }
            0xA8 /* NotBits */ => {
                let dest_reg = payload[i+1] as usize;
                let src_reg = payload[i+2] as usize;
                if dest_reg < registers.len() && src_reg < registers.len() {
                    registers[dest_reg] = (!(registers[src_reg] as u64)) as f64;
                    if debug_mode {
                        debug!("not_bits: r{} = ~r{} ({})", dest_reg, src_reg, registers[dest_reg]);
                    }
                } else {
                    error!("invalid register index for NotBits at byte {}", i);
                }
                i += 4; // opcode (1) + dest (1) + src (1) + padding (1)
            }
            0xA9 /* Shl */ => {
                let dest_reg = payload[i+1] as usize;
                let val_reg = payload[i+2] as usize;
                let shift_amt_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && val_reg < registers.len() && shift_amt_reg < registers.len() {
                    registers[dest_reg] = ((registers[val_reg] as u64) << (registers[shift_amt_reg] as u64)) as f64;
                    if debug_mode {
                        debug!("shl: r{} = r{} << r{} ({})", dest_reg, val_reg, shift_amt_reg, registers[dest_reg]);
                    }
                } else {
                    error!("invalid register index for Shl at byte {}", i);
                }
                i += 4;
            }
            0xAA /* Shr */ => {
                let dest_reg = payload[i+1] as usize;
                let val_reg = payload[i+2] as usize;
                let shift_amt_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && val_reg < registers.len() && shift_amt_reg < registers.len() {
                    registers[dest_reg] = ((registers[val_reg] as u64) >> (registers[shift_amt_reg] as u64)) as f64;
                    if debug_mode {
                        debug!("shr: r{} = r{} >> r{} ({})", dest_reg, val_reg, shift_amt_reg, registers[dest_reg]);
                    }
                } else {
                    error!("invalid register index for Shr at byte {}", i);
                }
                i += 4;
            }
            0xAB /* BreakPoint */ => {
                debug!("breakpoint hit at byte {}", i);
                // In a real debugger, this would pause execution. Here, just a log.
                i += 1;
            }
            0xAC /* GetTime */ => {
                let dest_reg_idx = payload[i+1] as usize;
                if dest_reg_idx < registers.len() {
                    let duration = SystemTime::now().duration_since(UNIX_EPOCH)
                        .expect("time went backwards");
                    registers[dest_reg_idx] = duration.as_secs_f64();
                    if debug_mode {
                        debug!("get_time: current time {} into register {}", registers[dest_reg_idx], dest_reg_idx);
                    }
                } else {
                    error!("invalid register index for GetTime at byte {}", i);
                }
                i += 2;
            }
            0xAD /* SeedRng */ => {
                let seed_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let seed = u64::from_le_bytes(seed_bytes);
                rng = StdRng::seed_from_u64(seed);
                if debug_mode {
                    debug!("rng seeded with {}", seed);
                }
                i += 9;
            }
            0xAE /* ExitCode */ => {
                let code_bytes: [u8; 4] = payload[i+1..i+5].try_into().unwrap();
                let exit_code = u32::from_le_bytes(code_bytes);
                info!("program exited with code {}", exit_code);
                std::process::exit(exit_code as i32);
            }
            0xFF /* Halt */ => {
                info!("halt instruction executed, program terminating.");
                break;
            }
            _ => {
                warn!(
                    "unknown opcode 0x{:02X} at byte {}, skipping.",
                    opcode, i
                );
                i += 1; // Skip unknown opcode to avoid infinite loop
            }
        }
    }

    if debug_mode {
        debug!("execution finished. final quantum state dump:");
        // qs.debug_dump_state();
    }
    if char_count > 0 {
        info!(
            "average char value: {}",
            char_sum as f64 / char_count as f64
        );
    }

    if apply_final_noise_flag {
        if let Some(_config) = noise_config {
            info!("applying final noise step to amplitudes.");
            // qs.apply_noise_to_amplitudes(config);
        } else {
            info!("final noise step requested, but no noise config was set for runtime.");
        }
    }

    // output final state as json
    let json_output = serde_json::to_string_pretty(&qs).unwrap();
    println!("\nfinal quantum state (json):\n{}", json_output);
    info!("simulation complete.");
}

fn main() {

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

match cli.command {
    Commands::Compile {
        source,
        output,
        debug,
    } => {
        info!("compiling '{}' to '{}' (debug: {})", source, output, debug);
        match compile_qoa_to_bin(&source, debug) {
            Ok(payload) => match write_exe(&payload, &output, QEXE) {
                Ok(_) => info!("compilation successful."),
                Err(e) => eprintln!("error writing executable: {}", e),
            },
            Err(e) => eprintln!("error compiling qoa: {}", e),
        }
    }
    Commands::Run {
        program,
        debug,
        ideal,
        noise,
        final_noise,
    } => {
        info!("running '{}' (debug: {})", program, debug);
        let noise_config = if ideal {
            Some(NoiseConfig::Ideal)
        } else {
            match noise {
                Some(s) => match s.as_str() {
                    "random" => Some(NoiseConfig::Random),
                    prob_str => match prob_str.parse::<f64>() {
                        Ok(p) if p >= 0.0 && p <= 1.0 => Some(NoiseConfig::Fixed(p)),
                        _ => {
                            eprintln!("invalid noise probability '{}'. must be 'random' or a float between 0.0 and 1.0.", prob_str);
                            return;
                        }
                    },
                },
                None => None,
            }
        };

        match fs::read(&program) {
            Ok(file_data) => {
                run_exe(&file_data, debug, noise_config, final_noise);
            }
            Err(e) => eprintln!("error reading program file: {}", e),
        }
    }
    Commands::CompileJson { source, output } => {
        info!("compiling '{}' to json '{}'", source, output);
        // not too complex JSON output for now
        let json_output = serde_json::json!({
            "source_file": source,
            "output_file": output,
            "status": "not implemented yet"
        });
        match File::create(&output).and_then(|file| {
            to_writer_pretty(file, &json_output)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        }) {
            Ok(_) => info!("json compilation placeholder successful. output written to {}", output),
            Err(e) => eprintln!("error writing json output: {}", e),
        }
    }
    Commands::Visual {
        input,
        output,
        resolution,
        fps,
        ffmpeg_args,
        ltr,
        rtl,
    } => {
        let input_path = PathBuf::from(input);
        let output_path = PathBuf::from(output);
        let parts: Vec<&str> = resolution.split('x').collect();
        let width = parts.get(0).and_then(|w| w.parse::<u32>().ok()).unwrap_or(800);
        let height = parts.get(1).and_then(|h| h.parse::<u32>().ok()).unwrap_or(600);

        // for direction printing
        let direction = crate::visualizer::parse_spectrum_direction(None);
        println!("Parsed spectrum direction: {:?}", direction);

        let spectrum_direction = if ltr {
            SpectrumDirection::Ltr
        } else if rtl {
            SpectrumDirection::Rtl
        } else {
            SpectrumDirection::None
        };

        let ffmpeg_args_slice: Vec<&str> = ffmpeg_args.iter().map(String::as_str).collect();

        let audio_visualizer = visualizer::AudioVisualizer::new();
        if let Err(e) = visualizer::run_qoa_to_video(
            &audio_visualizer,
            audio_visualizer.clone(),
            &input_path,
            &output_path,
            fps,
            width,
            height,
            &ffmpeg_args_slice,
            spectrum_direction,
        ) {
            eprintln!("error generating video: {}", e);
        }
    }
    Commands::Version => {
        println!("QOA version {}", env!("CARGO_PKG_VERSION"));
    }
    Commands::Flags => {
        println!("Available flags:\n");
        println!(" FOR QOA:\n");
        println!("--compile     Compile a .qoa file to .qexe");
        println!("--run         Run a .qexe binary");
        println!("--compilejson Compile a .qoa file to JSON format");
        println!("--version     Show version info\n");

        println!(" FOR NOISE:\n");
        println!(" --noise (range from 0-1)");
        println!(" --ideal (no noise / ideal state)");
        println!(" NOTE: DEFAULT IS RANDOM NOISE, NOT IDEAL\n");

        println!(" FOR VISUAL:\n");
        println!(" --ltr (Left to right spectrum visualization)");
        println!(" --rtl (Right to left spectrum visualization)");
        println!(" QOA VISUAL ALSO ACCEPTS FFMPEG ARGUMENTS / FLAGS");
    }
}
}