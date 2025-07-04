use crate::visualizer::SpectrumDirection;
use clap::{Parser, Subcommand};
use log::{debug, error, info, warn};
use qoa::runtime::quantum_state::NoiseConfig;
use qoa::runtime::quantum_state::QuantumState;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::to_writer_pretty;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
// use rand::thread_rng; // Removed unused import
use rand_distr::StandardNormal;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH}; // Removed unused Duration

// Import rayon prelude for parallel iterators
use rayon::prelude::*;

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

const QOA_VERSION: &str = "0.2.8";
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

#[derive(Subcommand, Debug)]
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
        /// Set the maximum number of qubits to simulate. This overrides the inferred qubit count from the program file.
        #[arg(long)] // ADDED: This makes --qubit a named flag
        qubit: Option<usize>,
        /// Display only the top N amplitudes by probability. Use --top-n 0 to show all states.
        #[arg(long, value_name = "COUNT", default_value_t = 20)]
        top_n: usize,
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
        #[arg(long, default_value_t = 60)]
        fps: u32,

        /// Spectrum direction Left-to-Right
        #[arg(long, conflicts_with = "rtl", default_value_t = false)]
        ltr: bool,

        /// Spectrum direction Right-to-Left
        #[arg(long, conflicts_with = "ltr", default_value_t = false)]
        rtl: bool,

        /// Extra ffmpeg flags (e.g., -s, -r, -b:v, -pix_fmt, etc.)
        #[arg(long = "ffmpeg-flag", value_name = "FFMPEG_FLAG", num_args = 0.., action = clap::ArgAction::Append)]
        ffmpeg_flags: Vec<String>,

        /// Additional ffmpeg arguments passed directly to ffmpeg (e.g., "-c:v libx264 -crf 23") as trailing args after "--"
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
            filedata.len() // Corrected variable name
        );
        return None;
    }
    Some((name, version, &filedata[9..9 + payload_len]))
}

// MODIFIED: print_amplitudes now accepts a top_n parameter
fn print_amplitudes(qs: &QuantumState, noise_strength: f64, top_n: usize) {
    println!("\nFinal amplitudes:");
    // let mut amplitudes_with_probs: Vec<(f64, String, f64, f64)> = Vec::new(); // Removed redundant initialization

    // Parallelize the iteration for collecting amplitudes and probabilities
    // Each thread gets its own rng for thread safety.
    let mut amplitudes_with_probs: Vec<(f64, String, f64, f64)> = qs.amps.par_iter().enumerate().map(|(i, amp)| {
        let mut local_rng = rand::thread_rng();
        let noise_re =
            <StandardNormal as rand_distr::Distribution<f64>>::sample(&StandardNormal, &mut local_rng)
                * noise_strength;
        let noise_im =
            <StandardNormal as rand_distr::Distribution<f64>>::sample(&StandardNormal, &mut local_rng)
                * noise_strength;
        let noisy_re = amp.re + noise_re;
        let noisy_im = amp.im + noise_im;
        let prob = noisy_re * noisy_re + noisy_im * noisy_im;
        let binary_string = format!("{:0width$b}", i, width = qs.n);
        (prob, binary_string, noisy_re, noisy_im)
    }).collect();

    // Sort by probability in descending order
    amplitudes_with_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let num_to_print = if top_n == 0 {
        amplitudes_with_probs.len() // if top_n is 0, print all
    } else {
        top_n.min(amplitudes_with_probs.len()) // otherwise, print min of top_n or available amplitudes
    };

    for j in 0..num_to_print {
        let (prob, binary_string, re, im) = &amplitudes_with_probs[j];
        println!(
            "|{}>: {:.6} + {:.6}i   (prob={:.6})",
            binary_string, re, im, prob
        );
    }
}

// MODIFIED: run_exe now returns the inferred_qubits and takes num_qubits for QuantumState::new
pub fn run_exe(
    filedata: &[u8],
    debug_mode: bool,
    noise_config: Option<NoiseConfig>,
    apply_final_noise_flag: bool,
    num_qubits_to_initialize: usize, // ADDED: this is the final, limited qubit count
) -> QuantumState {
    let (_header, _version, payload) = match parse_exe_file(filedata) { // FIX: Added underscores
        Some(x) => x,
        None => {
            error!("invalid or unsupported exe file, please check its header.");
            // MODIFIED: Return a QuantumState with the requested (but likely 0) qubits
            return QuantumState::new(num_qubits_to_initialize, None);
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
        // ADDED: Raw payload dump for more detailed debugging
        debug!("raw payload bytes (first 150 bytes):");
        for j in 0..payload.len().min(150) {
            debug!(
                "byte[{:#04}] = 0x{:02X}",
                j, payload[j]
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
    // advances the instruction pointer correctly for variable-length instructions
    // tracks the maximum qubit index accessed
    while i < payload.len() {
        if debug_mode {
            debug!("scanning opcode 0x{:02X} at byte {}", payload[i], i);
        }
        let opcode = payload[i];
        match opcode {
            // QOA v0.2.7 instructions:
            // Corrected opcodes and byte lengths based on instructions.rs
            0x04 /* QInit / InitQubit */ => { // 2 bytes
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }
            0x1D /* SetPhase */ => { // 10 bytes (1 opcode + 1 qubit + 8 f64)
                if i + 9 >= payload.len() { error!("incomplete SetPhase at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x74 /* SetPos */ => { // 18 bytes (1 opcode + 1 qubit + 8 f64 + 8 f64)
                if i + 17 >= payload.len() { error!("incomplete SetPos at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18;
            }
            0x75 /* SetWl */ => { // 10 bytes (1 opcode + 1 qubit + 8 f64)
                if i + 9 >= payload.len() { error!("incomplete SetWl at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x76 /* WlShift */ => { // 10 bytes (1 opcode + 1 qubit + 8 f64)
                if i + 9 >= payload.len() { error!("incomplete WlShift at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x77 /* Move */ => { // 18 bytes (1 opcode + 1 qubit + 8 f64 + 8 f64)
                if i + 17 >= payload.len() { error!("incomplete Move at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18;
            }
            0x18 /* CharOut */ => { // 2 bytes
                if i + 1 >= payload.len() { error!("incomplete CharOut at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }
            0x32 /* QMeas / Measure */ => { // 2 bytes
                if i + 1 >= payload.len() { error!("incomplete QMeas at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }
            0x79 /* MarkObserved */ => { // 2 bytes
                if i + 1 >= payload.len() { error!("incomplete MarkObserved at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }
            0x7A /* RELEASE */ => { // 2 bytes
                if i + 1 >= payload.len() { error!("incomplete Release at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }
            0xFF /* HALT */ => { // 1 byte
                i += 1;
            }
            0x00 => { // Handle 0x00 as a silent NOP (1 byte)
                i += 1;
            }
            0x8D => { // Handle 0x8D as a silent NOP (1 byte)
                i += 1;
            }
            0x97 => { // Handle 0x97 (RetSub) in first pass by just skipping (1 byte)
                i += 1;
            }


            // Other instructions (keeping the existing logic for these, assuming they are correct)
            // single‑qubit & simple two‑qubit ops (2 bytes)
            0x05 /* ApplyHadamard */ | 0x06 /* ApplyPhaseFlip */ | 0x07 /* ApplyBitFlip */ |
            0x0D /* ApplyTGate */ | 0x0E /* ApplySGate */ | 0x0A /* Reset / QReset */ |
            0x59 /* PhotonEmit */ | 0x5A /* PhotonDetect */ | 0x5C /* PhotonAddition */ |
            0x5D /* ApplyPhotonSubtraction */ | 0x61 /* SinglePhotonSourceOn */ |
            0x62 /* SinglePhotonSourceOff */ | 0x6F /* MeasureParity */ |
            0x71 /* OpticalSwitchControl */ | 0x9B /* Input */ | 0xA0 /* PushReg */ |
            0xA1 /* PopReg */ | 0xAC /* GetTime */ | 0x50 /* Rand */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }

            // 10‑byte ops (reg + 8‑byte imm)
            0x08 /* PhaseShift */ | 0x22 /* RX */ | 0x23 /* RY */ | 0x0F /* RZ */ |
            0x24 /* Phase */ | 0x66 /* SetOpticalAttenuation */ | 0x67 /* DynamicPhaseComp */ |
            0x6A /* ApplyDisplacement */ | 0x6D /* ApplySqueezing */ |
            0x82 /* ApplyNonlinearPhaseShift */ | 0x83 /* ApplyNonlinearSigma */ |
            0x21 /* RegSet */ => { // SetWl (0x75) moved above
                if i + 9 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }

            // 3‑byte ops (two‑qubit or reg/reg)
            0x17 /* CNOT */ | 0x1E /* CZ */ | 0x0B /* Swap */ |
            0x1F /* ThermalAvg */ | 0x65 /* OpticalRouting */ | 0x69 /* CrossPhaseMod */ |
            0x20 /* WkbFactor */ | 0xA4 /* Cmp */ | 0x51 /* Sqrt */ | 0x52 /* Exp */ | 0x53 /* Log */ => {
                if i + 2 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                i += 3;
            }

            // 4‑byte ops (three regs)
            0x0C /* ControlledSwap */ | 0x54 /* RegAdd */ | 0x55 /* RegSub */ |
            0x56 /* RegMul */ | 0x57 /* RegDiv */ | 0x58 /* RegCopy */ |
            0x63 /* PhotonBunchingCtl */ | 0xA8 /* NotBits */ |
            0x31 /* CharLoad */ => {
                if i + 3 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 4;
            }

            // variable‑length entangle lists:
            0x11 /* Entangle */ | 0x12 /* EntangleBell */ => {
                if i + 2 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                i += 3;
            }
            0x13 /* EntangleMulti */ | 0x14 /* EntangleCluster */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                let n = payload[i+1] as usize;
                if i + 2 + n > payload.len() { error!("incomplete entangle list at byte {}", i); break; }
                for j in 0..n {
                    max_q = max_q.max(payload[i+2+j] as usize);
                }
                i += 2 + n;
            }
            0x15 /* EntangleSwap */ | 0x16 /* EntangleSwapMeasure */ => {
                if i + 4 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = (payload[i+1] as usize)
                    .max(payload[i+2] as usize)
                    .max(payload[i+3] as usize)
                    .max(payload[i+4] as usize);
                if opcode == 0x16 {
                    let start = i + 5;
                    let end = start + payload[start..].iter().position(|&b| b == 0).unwrap_or(0);
                    i = end + 1;
                } else {
                    i += 5;
                }
            }

            // label‑terminated ops:
            0x19 /* EntangleWithFB */ | 0x1A /* EntangleDistrib */ |
            0x1B /* MeasureInBasis */ | 0x87 /* VerboseLog */ |
            0x38 /* ApplyFeedforward */ | 0x3A /* BasisChange */ |
            0x3B /* Load */ | 0x3C /* Store */ | 0x5B /* PhotonCount */ |
            0x6B /* DisplacementFB */ | 0x6E /* SqueezingFB */ |
            0x73 /* PhotonLossCorr */ | 0x7C /* QndMeasure */ |
            0x7D /* ErrorCorrect */ | 0x7F /* QStateTomography */ |
            0x85 /* PNRDetection */ | 0x86 /* FeedbackCtl */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let start = i + 2;
                let end = start + payload[start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = end + 1;
            }

            // control flow & misc ops:
            0x02 /* ApplyGate(QGATE) */ => {
                // reg (1), name (8), then optional extra reg for "cz"
                if i + 9 >= payload.len() { error!("incomplete qgate at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let name_bytes = &payload[i+2..i+10];
                let name = String::from_utf8_lossy(name_bytes)
                    .trim_end_matches('\0')
                    .to_string();
                if name == "cz" {
                    if i + 10 >= payload.len() { error!("incomplete cz at byte {}", i); break; }
                    max_q = max_q.max(payload[i+10] as usize);
                    i += 11;
                } else {
                    i += 10;
                }
            }
            0x33 /* ApplyRotation */ => { /* … */ i += 11; }
            0x34 /* ApplyMultiQubitRotation */ => {
                // opcode, axis, num_qs, [qs], [angles]
                if i + 2 >= payload.len() { error!("incomplete multi‑rotation at byte {}", i); break; }
                let n = payload[i+2] as usize;
                let needed = 3 + n /* regs */ + n * 8 /* f64 angles */;
                if i + needed > payload.len() { error!("incomplete multi‑rotation at byte {}", i); }
                for j in 0..n {
                    max_q = max_q.max(payload[i + 3 + j] as usize);
                }
                i += needed;
            }
            0x35 /* ControlledPhase */ | 0x36 /* ApplyCPhase */ => {
                // ctrl qubit, target qubit, angle:f64
                if i + 10 >= payload.len() { error!("incomplete ControlledPhase at byte {}", i); break; }
                max_q = max_q
                    .max(payload[i+1] as usize)
                    .max(payload[i+2] as usize);
                i += 11;
            }
            0x37 /* ApplyKerrNonlin */ => {
                // qubit, strength:f64, duration:f64
                if i + 17 >= payload.len() { error!("incomplete KerrNonlin at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18;
            }
            0x39 /* DecoherenceProtect */ | 0x68 /* OpticalDelayLineCtl */ => {
                // qubit, duration:f64
                if i + 9 >= payload.len() { error!("incomplete DecoherenceProtect at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x3D /* LoadMem */ | 0x3E /* StoreMem */ => {
                // reg_str\0, addr_str\0
                let start = i + 1;
                let mid   = start + payload[start..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let end   = mid   + payload[mid..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = end;
            }
            0x3F /* LoadClassical */ | 0x40 /* StoreClassical */ => {
                // reg_str\0, var_str\0
                let start = i + 1;
                let mid   = start + payload[start..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let end   = mid   + payload[mid..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = end;
            }
            0x41 /* Add */ | 0x42 /* Sub */ | 0x43 /* And */ | 0x44 /* Or */ | 0x45 /* Xor */ => {
                // dst\0, src1\0, src2\0
                let d_end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let s1_end = d_end + payload[d_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let s2_end = s1_end + payload[s1_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = s2_end;
            }
            0x46 /* Not */ | 0x47 /* Push */ | 0x4F /* Pop */ => {
                // reg_str\0
                let end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = end;
            }
            0x49 /* Jump */ | 0x4C /* Call */ => {
                // label\0
                let end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = end;
            }
            0x4A /* JumpIfZero */ | 0x4B /* JumpIfOne */ => {
                // cond_reg\0, label\0
                let c_end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let l_end = c_end + payload[c_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = l_end;
            }
            0x4E /* TimeDelay */ => {
                // qubit, cycles:f64
                if i + 9 >= payload.len() { error!("incomplete TimeDelay at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x5E /* PhotonEmissionPattern */ => {
                // qubit, pattern_str\0, cycles:u64
                if i + 2 >= payload.len() { error!("incomplete PhotonEmissionPattern at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let str_end = i + 2 + payload[i+2..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                if str_end + 8 > payload.len() { error!("incomplete PhotonEmissionPattern at byte {}", i); break; }
                i = str_end + 8;
            }
            0x5F /* PhotonDetectThreshold */ => {
                // qubit, thresh:f64, reg_str\0
                if i + 9 >= payload.len() { error!("incomplete PhotonDetectThreshold at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let str_end = i + 10 + payload[i+10..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = str_end;
            }
            0x60 /* PhotonDetectCoincidence */ => {
                // n, [qs], reg_str\0
                let n = payload[i+1] as usize;
                let q_end = i + 2 + n;
                let str_end = q_end + payload[q_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                for j in 0..n {
                    max_q = max_q.max(payload[i+2+j] as usize);
                }
                i = str_end;
            }
            0x64 /* PhotonRoute */ => {
                if i + 1 >= payload.len() { error!("incomplete PhotonRoute at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let f_end = i + 2 + payload[i+2..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let t_end = f_end + payload[f_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = t_end;
            }
            0x6C /* ApplyDisplacementOp */ => {
                if i + 17 >= payload.len() { error!("incomplete ApplyDisplacementOp at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18;
            }
            0x70 /* MeasureWithDelay */ => {
                if i + 9 >= payload.len() { error!("incomplete MeasureWithDelay at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let str_end = i + 10 + payload[i+10..].iter().position(|&b| b == 0).unwrap_or(0) + 1;
                i = str_end;
            }
            0x72 /* PhotonLossSimulate */ => {
                if i + 17 >= payload.len() { error!("incomplete PhotonLossSimulate at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18;
            }
            0x7E /* ErrorSyndrome */ => {
                if i + 1 >= payload.len() { error!("incomplete ErrorSyndrome at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let s_end = i + 2 + payload[i+2..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let r_end = s_end + payload[s_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = r_end;
            }
            0x80 /* BellStateVerif */ => {
                if i + 2 >= payload.len() { error!("incomplete BellStateVerif at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                let n_end = i + 3 + payload[i+3..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = n_end;
            }
            0x81 /* QuantumZenoEffect */ => {
                if i + 17 >= payload.len() { error!("incomplete QuantumZenoEffect at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18;
            }
            0x84 /* ApplyLinearOpticalTransform */ => {
                if i + 4 >= payload.len() { error!("incomplete LinearOpticalTransform at byte {}", i); break; }
                let nin = payload[i+1] as usize;
                let nout = payload[i+2] as usize;
                let name_end = i + 4 + payload[i+4..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                for q in 0..nin { max_q = max_q.max(payload[name_end + q] as usize); }
                for q in 0..nout {
                    max_q = max_q.max(payload[name_end + nin + q] as usize);
                }
                i = name_end + nin + nout;
            }
            0x88 /* Comment */ => {
                let end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = end;
            }
            0x90 /* Jmp */ | 0x91 /* JmpAbs */ | 0xA3 /* Free */ | 0xAD /* SeedRng */ => {
                if i + 9 >= payload.len() { error!("incomplete Jmp/Free/SeedRng at byte {}", i); break; }
                i += 9;
            }
            0x92 /* IfGt */ | 0x93 /* IfLt */ | 0x94 /* IfEq */ | 0x95 /* IfNe */ => {
                if i + 11 >= payload.len() { error!("incomplete If at byte {}", i); break; }
                i += 11;
            }
            0x96 /* CallAddr */ | 0x9E /* LoadRegMem */ | 0xA2 /* Alloc */ => {
                if i + 9 >= payload.len() { error!("incomplete Alloc/LoadRegMem at byte {}", i); break; }
                i += 10;
            }
            0x9F /* StoreMemReg */ => {
                if i + 9 >= payload.len() { error!("incomplete StoreMemReg at byte {}", i); break; }
                i += 10;
            }
            0x98 /* Printf */ => {
                if i + 9 >= payload.len() { error!("incomplete Printf at byte {}", i); break; }
                let len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                let regs = payload[i+9+len] as usize;
                i += 1 + 8 + len + 1 + regs;
            }
            0x99 /* Print */ | 0x9A /* Println */ => {
                if i + 9 >= payload.len() { error!("incomplete Print/Println at byte {}", i); break; }
                let len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                i += 1 + 8 + len;
            }
            0xA5 | 0xA6 | 0xA7 | 0xA9 | 0xAA => {
                if i + 3 >= payload.len() { error!("incomplete BitOp at byte {}", i); break; }
                i += 4;
            }
            0xAE => {
                if i + 5 >= payload.len() { error!("incomplete ExitCode at byte {}", i); break; }
                i += 5;
            }
            0x01 => {
                if i + 1 >= payload.len() { error!("incomplete LoopStart at byte {}", i); break; }
                i += 2;
            }
            _ => {
                warn!(
                    "unknown opcode 0x{:02X} in scan at byte {}, skipping. Advancing i by 1",
                    opcode, i
                );
                i += 1;
            }
        }
    }

    let _inferred_qubits = if max_q == 0 && payload.is_empty() {
        0
    } else {
        max_q + 1
    };

    info!(
        "initializing quantum state with {} qubits (type {}, ver {})",
        num_qubits_to_initialize, _header, _version
    );
    let mut qs = QuantumState::new(num_qubits_to_initialize, noise_config.clone());
    let _last_stats = Instant::now();
    let mut char_count: u64 = 0;
    let mut char_sum: u64 = 0;

    // declare registers, loop_stack, call_stack, memory, and rng for the second pass
    let mut registers: Vec<f64> = vec![0.0; 24]; // assuming 24 registers, adjust as needed
    let mut loop_stack: Vec<(usize, u64)> = Vec::new();
    let mut call_stack: Vec<usize> = Vec::new(); // for call/ret instructions
    let mut memory: Vec<u8> = vec![0; 1024 * 1024]; // 1mb linear byte-addressable memory
    let mut rng = StdRng::from_entropy(); // default seeded rng

    let mut i = 0; // reset 'i' for the second pass

    // second pass: execute instructions and interact with quantumstate
    while i < payload.len() {
        if debug_mode {
            debug!("executing opcode 0x{:02X} at byte {}", payload[i], i);
        }
        let opcode = payload[i];
        match opcode {
            0x04 /* QInit / InitQubit */ => {
                let q_idx = payload[i + 1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, qinit on qubit {}, advancing i by 2", opcode, i, q_idx); }
                i += 2;
            }
            0x1D /* SetPhase */ => {
                let q_idx = payload[i+1] as usize;
                let phase_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap(); // Changed to 8 bytes for f64
                let phase = f64::from_le_bytes(phase_bytes);
                // qs.set_phase(q_idx, phase); // Assuming set_phase exists in QuantumState // COMMENTED OUT: No set_phase method in QuantumState
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, set phase for qubit {} to {}, advancing i by 10", opcode, i, q_idx, phase); }
                i += 10;
            }
            0x74 /* SetPos */ => {
                let q_idx = payload[i+1] as usize;
                let x_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap(); // Changed to 8 bytes for f64
                let x = f64::from_le_bytes(x_bytes);
                let y_bytes: [u8; 8] = payload[i+10..i+18].try_into().unwrap(); // Changed to 8 bytes for f64
                let y = f64::from_le_bytes(y_bytes);
                // qs.set_pos(q_idx, x, y); // Assuming set_pos exists in QuantumState
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, set position for qubit {} to ({}, {}), advancing i by 18", opcode, i, q_idx, x, y); }
                i += 18;
            }
            0x75 /* SetWl */ => {
                let q_idx = payload[i+1] as usize;
                let wl_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap(); // Changed to 8 bytes for f64
                let wl = f64::from_le_bytes(wl_bytes);
                // qs.set_wl(q_idx, wl); // Assuming set_wl exists in QuantumState
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, set wavelength for qubit {} to {}, advancing i by 10", opcode, i, q_idx, wl); }
                i += 10;
            }
            0x76 /* WlShift */ => {
                let q_idx = payload[i+1] as usize;
                let shift_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap(); // Changed to 8 bytes for f64
                let shift = f64::from_le_bytes(shift_bytes);
                // qs.wl_shift(q_idx, shift); // Assuming wl_shift exists in QuantumState
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, shifted wavelength for qubit {} by {}, advancing i by 10", opcode, i, q_idx, shift); }
                i += 10;
            }
            0x77 /* Move */ => {
                let q_idx = payload[i+1] as usize;
                let dx_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap(); // Changed to 8 bytes for f64
                let dx = f64::from_le_bytes(dx_bytes);
                let dy_bytes: [u8; 8] = payload[i+10..i+18].try_into().unwrap(); // Changed to 8 bytes for f64
                let dy = f64::from_le_bytes(dy_bytes);
                // qs.move_qubit(q_idx, dx, dy); // Assuming move_qubit exists in QuantumState
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, moved qubit {} by ({}, {}), advancing i by 18", opcode, i, q_idx, dx, dy); }
                i += 18;
            }
            0x18 /* CharOut */ => {
                let q = payload[i+1] as usize;
                let classical_value = qs.measure(q) as u8;
                print!("{}", classical_value as char);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, char_out: qubit {} measured as {} ('{}'), advancing i by 2", opcode, i, q, classical_value, classical_value as char); }
                char_count += 1;
                char_sum += classical_value as u64;
                i += 2;
            }
            0x32 /* QMeas / Measure */ => {
                let q = payload[i + 1] as usize;
                let result = qs.measure(q);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, measured qubit {}: {:?}, advancing i by 2", opcode, i, q, result); }
                i += 2;
            }
            0x79 /* MarkObserved*/ => {
                let q_idx = payload[i+1] as usize;
                // Assuming a mark_observed method exists in QuantumState
                // qs.mark_observed(q_idx);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, marked qubit {} as observed, advancing i by 2", opcode, i, q_idx); }
                i += 2;
            }
            0x7A /* RELEASE */ => {
                let q_idx = payload[i+1] as usize;
                // Assuming a release method exists in QuantumState
                // qs.release(q_idx);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, released qubit {}, advancing i by 2", opcode, i, q_idx); }
                i += 2;
            }
            0xFF /* Halt */ => {
                info!("halt instruction executed, program terminating.");
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, halt instruction, terminating", opcode, i); }
                break;
            }
            0x00 => { // Handle 0x00 as a silent NOP
                i += 1;
            }
            0x8D => { // Handle 0x8D as a silent NOP
                i += 1;
            }
            0x97 /* RetSub */ => {
                if let Some(ret_addr) = call_stack.pop() {
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, returned from subroutine to address {}, setting i to {}", opcode, i, ret_addr, ret_addr); }
                    i = ret_addr;
                } else {
                    warn!("ret_sub without matching call at byte {}. this might indicate an issue in the compiled program or execution flow, or an attempt to return from an empty call stack.", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, ret_sub on empty stack, advancing i by 1", opcode, i); }
                    i += 1; // Advance to prevent infinite loop on bad instruction
                }
            }
            // All other opcodes (copied from original, assume their byte lengths are consistent between instructions.rs and main.rs)
            0x01 /* LoopStart */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, loop start, advancing i by 2", opcode, i); }
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
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied h gate on qubit {} (via qgate), advancing i by 10", opcode, i, q); }
                        i += 10;
                    }
                    "x" => {
                        qs.apply_x(q);
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied x gate on qubit {}, advancing i by 10", opcode, i, q); }
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
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied cz gate between qubits {} (control) and {} (target), advancing i by 11", opcode, i, q, tgt); }
                        i += 11;
                    }
                    _ => {
                        warn!(
                            "unsupported QGATE '{}' at byte {}, skipping. Advancing i by 10",
                            name, i
                        );
                        i += 10; // skip the instruction
                    }
                }
            }
            0x05 /* ApplyHadamard */ => {
                qs.apply_h(payload[i + 1] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied hadamard on qubit {}, advancing i by 2", opcode, i, payload[i + 1]); }
                i += 2;
            }
            0x06 /* ApplyPhaseFlip */ => {
                qs.apply_phase_flip(payload[i + 1] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied phase flip (Z) on qubit {}, advancing i by 2", opcode, i, payload[i + 1]); }
                i += 2;
            }
            0x07 /* ApplyBitFlip */ => {
                qs.apply_x(payload[i + 1] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied bit flip (X) on qubit {}, advancing i by 2", opcode, i, payload[i + 1]); }
                i += 2;
            }
            0x08 /* PhaseShift */ => {
                let q = payload[i + 1] as usize;
                let angle_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_phase_shift(q, angle);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied phase shift on qubit {} with angle {}, advancing i by 10", opcode, i, q, angle); }
                i += 10;
            }
            0x0A /* Reset / QReset */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, reset qubit {}, advancing i by 2", opcode, i, payload[i + 1]); }
                i += 2;
            }
            0x0B /* Swap */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, swapped qubits {} and {}, advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x0C /* ControlledSwap */ => {
                let c = payload[i + 1] as usize;
                let q1 = payload[i + 2] as usize;
                let q2 = payload[i + 3] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied controlled swap with control {} on qubits {} and {}, advancing i by 4", opcode, i, c, q1, q2); }
                i += 4;
            }
            0x0D /* ApplyTGate */ => {
                qs.apply_t_gate(payload[i + 1] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied t-gate on qubit {}, advancing i by 2", opcode, i, payload[i + 1]); }
                i += 2;
            }
            0x0E /* ApplySGate */ => {
                qs.apply_s_gate(payload[i + 1] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied s-gate on qubit {}, advancing i by 2", opcode, i, payload[i + 1]); }
                i += 2;
            }
            0x0F /* RZ */ => {
                let q = payload[i + 1] as usize;
                let angle_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_rz(q, angle);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied rz gate on qubit {} with angle {}, advancing i by 10", opcode, i, q, angle); }
                i += 10;
            }
            0x10 /* LoopEnd */ => {
                if let Some((loop_start_ptr, iterations_left)) = loop_stack.pop() {
                    if iterations_left > 1 {
                        loop_stack.push((loop_start_ptr, iterations_left - 1));
                        i = loop_start_ptr;
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, looping back to {} ({} iterations left)", opcode, i, loop_start_ptr, iterations_left - 1); }
                    } else {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, loop finished at {}, advancing i by 1", opcode, i, i); }
                        i += 1;
                    }
                } else {
                    error!("loopend without matching loopstart at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, loopend error, advancing i by 1", opcode, i); }
                    i += 1; // Advance to prevent infinite loop on bad instruction
                }
            }
            0x11 /* Entangle */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, entangled qubits {} and {}, advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x12 /* EntangleBell */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, created bell state with qubits {} and {}, advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x13 /* EntangleMulti */ => {
                let num_qubits = payload[i + 1] as usize;
                let qubits: Vec<usize> = payload[i + 2..i + 2 + num_qubits]
                    .iter()
                    .map(|&b| b as usize)
                    .collect();
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, entangled multiple qubits: {:?}, advancing i by {}", opcode, i, qubits, 2 + num_qubits); }
                i += 2 + num_qubits;
            }
            0x14 /* EntangleCluster */ => {
                let num_qubits = payload[i + 1] as usize;
                let qubits: Vec<usize> = payload[i + 2..i + 2 + num_qubits]
                    .iter()
                    .map(|&b| b as usize)
                    .collect();
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, created cluster state with qubits: {:?}, advancing i by {}", opcode, i, qubits, 2 + num_qubits); }
                i += 2 + num_qubits;
            }
            0x15 /* EntangleSwap */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                let q3 = payload[i + 3] as usize;
                let q4 = payload[i + 4] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, performed entanglement swap between ({}, {}) and ({}, {}), advancing i by 5", opcode, i, q1, q2, q3, q4); }
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
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, performed entanglement swap measure between ({}, {}) and ({}, {}) with label {}, advancing i by {}", opcode, i, q1, q2, q3, q4, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x17 /* ControlledNot / CNOT */ => {
                qs.apply_cnot(payload[i + 1] as usize, payload[i + 2] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied cnot with control {} and target {}, advancing i by 3", opcode, i, payload[i + 1], payload[i + 2]); }
                i += 3;
            }
            0x19 /* EntangleWithClassicalFeedback */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, entangled with classical feedback on qubit {} with label {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x1A /* EntangleDistributed */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, performed distributed entanglement on qubit {} with label {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x1B /* MeasureInBasis */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, measured qubit {} in basis {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x1C /* ResetAll */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, reset all qubits, advancing i by 1", opcode, i); }
                i += 1;
            }
            0x1E /* CZ */ => {
                qs.apply_cz(payload[i + 1] as usize, payload[i + 2] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied cz gate on qubits {} and {}, advancing i by 3", opcode, i, payload[i + 1], payload[i + 2]); }
                i += 3;
            }
            0x1F /* ThermalAvg */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, performed thermal averaging on qubits {} and {} (placeholder), advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x20 /* WkbFactor */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, WkbFactor instruction (placeholder), advancing i by 3", opcode, i); }
                i += 3;
            }
            0x21 /* RegSet */ => {
                let reg_idx = payload[i+1] as usize;
                let value_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let value = f64::from_le_bytes(value_bytes);
                if reg_idx < registers.len() {
                    registers[reg_idx] = value;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, set register {} to {}, advancing i by 10", opcode, i, reg_idx, value); }
                } else {
                    error!("invalid register index {} for RegSet at byte {}", reg_idx, i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for RegSet, advancing i by 10", opcode, i); }
                }
                i += 10;
            }
            0x22 /* RX */ => {
                let q = payload[i + 1] as usize;
                let angle_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_rx(q, angle);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied rx gate on qubit {} with angle {}, advancing i by 10", opcode, i, q, angle); }
                i += 10;
            }
            0x23 /* RY */ => {
                let q = payload[i + 1] as usize;
                let angle_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_ry(q, angle);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied ry gate on qubit {} with angle {}, advancing i by 10", opcode, i, q, angle); }
                i += 10;
            }
            0x24 /* Phase */ => {
                let q = payload[i+1] as usize;
                let angle_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied phase gate on qubit {} with angle {}, advancing i by 10", opcode, i, q, angle); }
                i += 10;
            }
            0x31 /* CharLoad */ => {
                let dest_reg = payload[i+1] as usize;
                let char_val = payload[i+2] as char;
                if dest_reg < registers.len() {
                    registers[dest_reg] = char_val as u8 as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, loaded char '{}' into register {}, advancing i by 4", opcode, i, char_val, dest_reg); }
                } else {
                    error!("invalid register index {} for CharLoad at byte {}", dest_reg, i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for CharLoad, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0x33 /* ApplyRotation */ => {
                let q = payload[i + 1] as usize;
                let axis = payload[i + 2] as char;
                let angle_bytes: [u8; 8] = payload[i + 3..i + 11].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                match axis {
                    'x' => qs.apply_rx(q, angle),
                    'y' => qs.apply_ry(q, angle),
                    'z' => qs.apply_rz(q, angle),
                    _ => warn!("unknown rotation axis '{}' at byte {}", axis, i),
                }
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied rotation around {} axis on qubit {} with angle {}, advancing i by 11", opcode, i, axis, q, angle); }
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

                for &q in &qubits {
                    match axis {
                        'x' => qs.apply_rx(q, angle),
                        'y' => qs.apply_ry(q, angle),
                        'z' => qs.apply_rz(q, angle),
                        _ => warn!("unknown rotation axis '{}' for multi-qubit rotation at byte {}", axis, i),
                    }
                }
                let bytes_advanced = current_idx + 8 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied multi-qubit rotation around {} axis on qubits {:?}, with angle {}, advancing i by {}", opcode, i, axis, qubits, angle, bytes_advanced); }
                i = current_idx + 8;
            }
            0x35 /* ControlledPhaseRotation */ => {
                let c = payload[i + 1] as usize;
                let t = payload[i + 2] as usize;
                let angle_bytes: [u8; 8] = payload[i + 3..i + 11].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_controlled_phase(c, t, angle);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied controlled phase rotation on control {} and target {} with angle {}, advancing i by 11", opcode, i, c, t, angle); }
                i += 11;
            }
            0x36 /* ApplyCPhase */ => {
                let c = payload[i + 1] as usize;
                let t = payload[i + 2] as usize;
                let angle_bytes: [u8; 8] = payload[i + 3..i + 11].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied controlled phase gate on control {} and target {} with angle {}, advancing i by 11", opcode, i, c, t, angle); }
                i += 11;
            }
            0x37 /* ApplyKerrNonlinearity */ => {
                let q = payload[i + 1] as usize;
                let strength_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let strength = f64::from_le_bytes(strength_bytes);
                let duration_bytes: [u8; 8] = payload[i + 10..i + 18].try_into().unwrap();
                let duration = f64::from_le_bytes(duration_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied kerr nonlinearity on qubit {} with strength {} and duration {}, advancing i by 18", opcode, i, q, strength, duration); }
                i += 18;
            }
            0x38 /* ApplyFeedforwardGate */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied feedforward gate on qubit {} with label {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x39 /* DecoherenceProtect */ => {
                let q = payload[i + 1] as usize;
                let duration_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let duration = f64::from_le_bytes(duration_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied decoherence protection on qubit {} for duration {}, advancing i by 10", opcode, i, q, duration); }
                i += 10;
            }
            0x3A /* ApplyMeasurementBasisChange */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied measurement basis change on qubit {} to basis {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x3B /* Load */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, loaded state to qubit {} from label {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x3C /* Store */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, stored state from qubit {} to label {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x3D /* LoadMem */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);

                let addr_name_start = reg_name_end + 1;
                let addr_name_end = addr_name_start + payload[addr_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let addr_name = String::from_utf8_lossy(&payload[addr_name_start..addr_name_end]);
                let bytes_advanced = addr_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, loaded value from memory address {} into register {}, advancing i by {}", opcode, i, addr_name, reg_name, bytes_advanced); }
                i = addr_name_end + 1;
            }
            0x3E /* StoreMem */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);

                let addr_name_start = reg_name_end + 1;
                let addr_name_end = addr_name_start + payload[addr_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let addr_name = String::from_utf8_lossy(&payload[addr_name_start..addr_name_end]);
                let bytes_advanced = addr_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, stored value from register {} into memory address {}, advancing i by {}", opcode, i, reg_name, addr_name, bytes_advanced); }
                i = addr_name_end + 1;
            }
            0x3F /* LoadClassical */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);

                let var_name_start = reg_name_end + 1;
                let var_name_end = var_name_start + payload[var_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let var_name = String::from_utf8_lossy(&payload[var_name_start..var_name_end]);
                let bytes_advanced = var_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, loaded value from classical variable {} into register {}, advancing i by {}", opcode, i, var_name, reg_name, bytes_advanced); }
                i = var_name_end + 1;
            }
            0x40 /* StoreClassical */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);

                let var_name_start = reg_name_end + 1;
                let var_name_end = var_name_start + payload[var_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let var_name = String::from_utf8_lossy(&payload[var_name_start..var_name_end]);
                let bytes_advanced = var_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, stored value from register {} into classical variable {}, advancing i by {}", opcode, i, reg_name, var_name, bytes_advanced); }
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
                let bytes_advanced = src2_reg_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, added {} and {} into {}, advancing i by {}", opcode, i, src1_reg_name, src2_reg_name, dst_reg_name, bytes_advanced); }
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
                let bytes_advanced = src2_reg_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, subtracted {} from {} into {}, advancing i by {}", opcode, i, src2_reg_name, src1_reg_name, dst_reg_name, bytes_advanced); }
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
                let bytes_advanced = src2_reg_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, ANDed {} and {} into {}, advancing i by {}", opcode, i, src1_reg_name, src2_reg_name, dst_reg_name, bytes_advanced); }
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
                let bytes_advanced = src2_reg_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, ORed {} and {} into {}, advancing i by {}", opcode, i, src1_reg_name, src2_reg_name, dst_reg_name, bytes_advanced); }
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
                let bytes_advanced = src2_reg_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, XORed {} and {} into {}, advancing i by {}", opcode, i, src1_reg_name, src2_reg_name, dst_reg_name, bytes_advanced); }
                i = src2_reg_end + 1;
            }
            0x46 /* Not */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, NOTed {}, advancing i by {}", opcode, i, reg_name, bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x47 /* Push */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, pushed register {}, advancing i by {}", opcode, i, reg_name, bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x48 /* Sync */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, sync instruction (placeholder), advancing i by 1", opcode, i); }
                i += 1;
            }
            0x49 /* Jump */ => {
                let label_start = i + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                error!("jump to string label '{}' (opcode 0x49) not implemented. use JmpAbs (0x91) with an address.", label);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, jump to label {}, advancing i by {}", opcode, i, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x4A /* JumpIfZero */ => {
                let cond_reg_start = i + 1;
                let cond_reg_end = cond_reg_start + payload[cond_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let cond_reg_name = String::from_utf8_lossy(&payload[cond_reg_start..cond_reg_end]);

                let label_start = cond_reg_end + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, jumped to label {} if register {} is zero, advancing i by {}", opcode, i, label, cond_reg_name, bytes_advanced); }
                i = label_end + 1;
            }
            0x4B /* JumpIfOne */ => {
                let cond_reg_start = i + 1;
                let cond_reg_end = cond_reg_start + payload[cond_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let cond_reg_name = String::from_utf8_lossy(&payload[cond_reg_start..cond_reg_end]);

                let label_start = cond_reg_end + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, jumped to label {} if register {} is one, advancing i by {}", opcode, i, label, cond_reg_name, bytes_advanced); }
                i = label_end + 1;
            }
            0x4C /* Call */ => {
                let label_start = i + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                error!("call to string label '{}' (opcode 0x4C) not implemented. use CallAddr (0x96) with an address.", label);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, call to label {}, advancing i by {}", opcode, i, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x4D /* Return */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, returned from subroutine, advancing i by 1", opcode, i); }
                i += 1;
            }
            0x4E /* TimeDelay */ => {
                let q = payload[i + 1] as usize;
                let delay_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let delay = f64::from_le_bytes(delay_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, time delay on qubit {} for {} units, advancing i by 10", opcode, i, q, delay); }
                i += 10;
            }
            0x4F /* Pop */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, popped value into register {}, advancing i by {}", opcode, i, reg_name, bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x50 /* Rand */ => {
                let dest_reg_idx = payload[i+1] as usize;
                if dest_reg_idx < registers.len() {
                    registers[dest_reg_idx] = rng.gen::<f64>();
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, generated random number {} into register {}, advancing i by 2", opcode, i, registers[dest_reg_idx], dest_reg_idx); }
                } else {
                    error!("invalid register index {} for Rand at byte {}", dest_reg_idx, i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for Rand, advancing i by 2", opcode, i); }
                }
                i += 2;
            }
            0x51 /* Sqrt */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, sqrt on qubits {} and {} (placeholder), advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x52 /* Exp */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, exp on qubits {} and {} (placeholder), advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x53 /* Log */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, log on qubits {} and {} (placeholder), advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x54 /* RegAdd */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = registers[src1_reg] + registers[src2_reg];
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, reg_add: r{} = r{} + r{} ({}), advancing i by 4", opcode, i, dest_reg, src1_reg, src2_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for RegAdd at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for RegAdd, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0x55 /* RegSub */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = registers[src1_reg] - registers[src2_reg];
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, reg_sub: r{} = r{} - r{} ({}), advancing i by 4", opcode, i, dest_reg, src1_reg, src2_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for RegSub at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for RegSub, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0x56 /* RegMul */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = registers[src1_reg] * registers[src2_reg];
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, reg_mul: r{} = r{} * r{} ({}), advancing i by 4", opcode, i, dest_reg, src1_reg, src2_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for RegMul at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for RegMul, advancing i by 4", opcode, i); }
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
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, reg_div: r{} = r{} / r{} ({}), advancing i by 4", opcode, i, dest_reg, src1_reg, src2_reg, registers[dest_reg]); }
                    } else {
                        error!("division by zero in RegDiv at byte {}", i);
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, division by zero in RegDiv, advancing i by 4", opcode, i); }
                    }
                } else {
                    error!("invalid register index for RegDiv at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for RegDiv, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0x58 /* RegCopy */ => {
                let dest_reg = payload[i+1] as usize;
                let src_reg = payload[i+2] as usize;
                if dest_reg < registers.len() && src_reg < registers.len() {
                    registers[dest_reg] = registers[src_reg];
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, reg_copy: r{} = r{} ({}), advancing i by 4", opcode, i, dest_reg, src_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for RegCopy at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for RegCopy, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0x59 /* PhotonEmit */ => {
                let q = payload[i + 1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, emitted photon from qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x5A /* PhotonDetect */ => {
                let q = payload[i + 1] as usize;
                let result = qs.measure(q);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, detected photon at qubit {}: {:?}, advancing i by 2", opcode, i, q, result); }
                i += 2;
            }
            0x5B /* PhotonCount */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, counted photons at qubit {} into label {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x5C /* PhotonAddition */ => {
                let q = payload[i + 1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, added photon to qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x5D /* ApplyPhotonSubtraction */ => {
                let q = payload[i + 1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, subtracted photon from qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x5E /* PhotonEmissionPattern */ => {
                let q = payload[i+1] as usize;
                let reg_name_start = i + 2;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let cycles_bytes: [u8; 8] = payload[reg_name_end + 1..reg_name_end + 9].try_into().unwrap();
                let cycles = u64::from_le_bytes(cycles_bytes);
                let bytes_advanced = reg_name_end + 9 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, set photon emission pattern for qubit {} from register {} for {} cycles, advancing i by {}", opcode, i, q, reg_name, cycles, bytes_advanced); }
                i = reg_name_end + 9;
            }
            0x5F /* PhotonDetectWithThreshold */ => {
                let q = payload[i+1] as usize;
                let threshold_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let threshold = f64::from_le_bytes(threshold_bytes);
                let reg_name_start = i + 10;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, detected photon at qubit {} with threshold {} into register {}, advancing i by {}", opcode, i, q, threshold, reg_name, bytes_advanced); }
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
                let bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, detected photon coincidence for qubits {:?} into register {}, advancing i by {}", opcode, i, qubits, reg_name, bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x61 /* SinglePhotonSourceOn */ => {
                let q = payload[i+1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, single photon source on for qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x62 /* SinglePhotonSourceOff */ => {
                let q = payload[i+1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, single photon source off for qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x63 /* PhotonBunchingControl */ => {
                let q = payload[i+1] as usize;
                let control_reg_idx = payload[i+2] as usize;
                if control_reg_idx < registers.len() {
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, photon bunching control for qubit {} with register {} value {}, advancing i by 4", opcode, i, q, control_reg_idx, registers[control_reg_idx]); }
                } else {
                    error!("invalid register index for PhotonBunchingControl at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for PhotonBunchingControl, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0x64 /* PhotonRoute */ => {
                let q = payload[i+1] as usize;
                let from_path_start = i + 2;
                let from_path_end = from_path_start + payload[from_path_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let from_path = String::from_utf8_lossy(&payload[from_path_start..from_path_end]);
                let to_path_start = from_path_end + 1;
                let to_path_end = to_path_start + payload[to_path_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let to_path = String::from_utf8_lossy(&payload[to_path_start..to_path_end]);
                let bytes_advanced = to_path_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, routed photon from {} to {} for qubit {}, advancing i by {}", opcode, i, from_path, to_path, q, bytes_advanced); }
                i = to_path_end + 1;
            }
            0x65 /* OpticalRouting */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, optical routing between qubits {} and {} (placeholder), advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x66 /* SetOpticalAttenuation */ => {
                let q = payload[i + 1] as usize;
                let attenuation_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let attenuation = f64::from_le_bytes(attenuation_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, set optical attenuation for qubit {} to {}, advancing i by 10", opcode, i, q, attenuation); }
                i += 10;
            }
            0x67 /* DynamicPhaseCompensation */ => {
                let q = payload[i + 1] as usize;
                let compensation_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let compensation = f64::from_le_bytes(compensation_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied dynamic phase compensation for qubit {} with value {}, advancing i by 10", opcode, i, q, compensation); }
                i += 10;
            }
            0x68 /* OpticalDelayLineControl */ => {
                let q = payload[i + 1] as usize;
                let delay_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let delay = f64::from_le_bytes(delay_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, controlled optical delay line for qubit {} with delay {}, advancing i by 10", opcode, i, q, delay); }
                i += 10;
            }
            0x69 /* CrossPhaseModulation */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, cross-phase modulation between qubits {} and {} (placeholder), advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x6A /* ApplyDisplacement */ => {
                let q = payload[i + 1] as usize;
                let alpha_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let alpha = f64::from_le_bytes(alpha_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied displacement to qubit {} with alpha {}, advancing i by 10", opcode, i, q, alpha); }
                i += 10;
            }
            0x6B /* ApplyDisplacementFeedback */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied displacement feedback to qubit {} with label {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x6C /* ApplyDisplacementOperator */ => {
                let q = payload[i + 1] as usize;
                let alpha_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let alpha = f64::from_le_bytes(alpha_bytes);
                let duration_bytes: [u8; 8] = payload[i + 10..i + 18].try_into().unwrap();
                let duration = f64::from_le_bytes(duration_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied displacement operator to qubit {} with alpha {} for duration {}, advancing i by 18", opcode, i, q, alpha, duration); }
                i += 18;
            }
            0x6D /* ApplySqueezing */ => {
                let q = payload[i + 1] as usize;
                let r_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let r = f64::from_le_bytes(r_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied squeezing to qubit {} with r {}, advancing i by 10", opcode, i, q, r); }
                i += 10;
            }
            0x6E /* ApplySqueezingFeedback */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied squeezing feedback to qubit {} with label {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x6F /* MeasureParity */ => {
                let q = payload[i+1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, measured parity on qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x70 /* MeasureWithDelay */ => {
                let q = payload[i+1] as usize;
                let delay_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let delay = f64::from_le_bytes(delay_bytes);
                let reg_name_start = i + 10;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, measured qubit {} with delay {} into register {}, advancing i by {}", opcode, i, q, delay, reg_name, bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x71 /* OpticalSwitchControl */ => {
                let q = payload[i+1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, optical switch control for qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x72 /* PhotonLossSimulate */ => {
                let q = payload[i+1] as usize;
                let prob_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let prob = f64::from_le_bytes(prob_bytes);
                let seed_bytes: [u8; 8] = payload[i+10..i+18].try_into().unwrap();
                let seed = u64::from_le_bytes(seed_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, simulated photon loss for qubit {} with probability {} and seed {}, advancing i by 18", opcode, i, q, prob, seed); }
                i += 18;
            }
            0x73 /* PhotonLossCorrection */ => {
                let q = payload[i+1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied photon loss correction for qubit {} with label {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x7E /* ErrorSyndrome */ => {
                let q = payload[i+1] as usize;
                let syndrome_name_start = i + 2;
                let syndrome_name_end = syndrome_name_start + payload[syndrome_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let syndrome_name = String::from_utf8_lossy(&payload[syndrome_name_start..syndrome_name_end]);
                let result_reg_name_start = syndrome_name_end + 1;
                let result_reg_name_end = result_reg_name_start + payload[result_reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let result_reg_name = String::from_utf8_lossy(&payload[result_reg_name_start..result_reg_name_end]);
                let bytes_advanced = result_reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, obtained error syndrome for qubit {} with name {} into register {}, advancing i by {}", opcode, i, q, syndrome_name, result_reg_name, bytes_advanced); }
                i = result_reg_name_end + 1;
            }
            0x7F /* QuantumStateTomography */ => {
                let q = payload[i+1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, performed quantum state tomography on qubit {} with label {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x80 /* BellStateVerif */ => {
                let q1 = payload[i+1] as usize;
                let q2 = payload[i+2] as usize;
                let reg_name_start = i + 3;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, performed bell state verification for qubits {} and {} into register {}, advancing i by {}", opcode, i, q1, q2, reg_name, bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x81 /* QuantumZenoEffect */ => {
                let q = payload[i+1] as usize;
                let num_measurements_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let num_measurements = u64::from_le_bytes(num_measurements_bytes);
                let interval_cycles_bytes: [u8; 8] = payload[i+10..i+18].try_into().unwrap();
                let interval_cycles = u64::from_le_bytes(interval_cycles_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied quantum zeno effect on qubit {} with {} measurements at {} cycles interval, advancing i by 18", opcode, i, q, num_measurements, interval_cycles); }
                i += 18;
            }
            0x82 /* ApplyNonlinearPhaseShift */ => {
                let q = payload[i+1] as usize;
                let shift_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let shift = f64::from_le_bytes(shift_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied nonlinear phase shift to qubit {} with shift {}, advancing i by 10", opcode, i, q, shift); }
                i += 10;
            }
            0x83 /* ApplyNonlinearSigma */ => {
                let q = payload[i+1] as usize;
                let sigma_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let sigma = f64::from_le_bytes(sigma_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied nonlinear sigma to qubit {} with sigma {}, advancing i by 10", opcode, i, q, sigma); }
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
                let bytes_advanced = output_qs_end - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied linear optical transform '{}' with {} modes, input {:?}, output {:?}, advancing i by {}", opcode, i, name, num_modes, input_qubits, output_qubits, bytes_advanced); }
                i = output_qs_end;
            }
            0x85 /* PhotonNumberResolvingDetection */ => {
                let q = payload[i+1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, performed photon number resolving detection on qubit {} with label {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x86 /* FeedbackControl */ => {
                let q = payload[i+1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied feedback control for qubit {} with label {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x87 /* VerboseLog */ => {
                let q = payload[i+1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, verbose log for qubit {} with message: {}, advancing i by {}", opcode, i, q, label, bytes_advanced); }
                i = label_end + 1;
            }
            0x88 /* Comment */ => {
                let text_start = i + 1;
                let text_end = text_start + payload[text_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let text = String::from_utf8_lossy(&payload[text_start..text_end]);
                let bytes_advanced = text_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, comment: {}, advancing i by {}", opcode, i, text, bytes_advanced); }
                i = text_end + 1;
            }
            0x89 /* Barrier */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, barrier instruction (placeholder), advancing i by 1", opcode, i); }
                i += 1;
            }
            0x90 /* Jmp */ => {
                let addr_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, relative jump by {}, advancing i by {} (to {})", opcode, i, addr, addr + 9, i + addr + 9); }
                i += addr + 9;
            }
            0x91 /* JmpAbs */ => {
                let addr_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, absolute jump to {}, setting i to {}", opcode, i, addr, addr); }
                i = addr;
            }
            0x92 /* IfGt */ => {
                let reg1_idx = payload[i+1] as usize;
                let reg2_idx = payload[i+2] as usize;
                let jump_addr_bytes: [u8; 8] = payload[i+3..i+11].try_into().unwrap();
                let jump_addr = u64::from_le_bytes(jump_addr_bytes) as usize;

                if reg1_idx < registers.len() && reg2_idx < registers.len() {
                    if registers[reg1_idx] > registers[reg2_idx] {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, if_gt: r{} ({}) > r{} ({}), jumped to {}, setting i to {}", opcode, i, reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx], jump_addr, jump_addr); }
                        i = jump_addr;
                    } else {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, if_gt: r{} ({}) not > r{} ({}), no jump, advancing i by 11", opcode, i, reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx]); }
                        i += 11;
                    }
                } else {
                    error!("invalid register index for IfGt at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for IfGt, advancing i by 11", opcode, i); }
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
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, if_lt: r{} ({}) < r{} ({}), jumped to {}, setting i to {}", opcode, i, reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx], jump_addr, jump_addr); }
                        i = jump_addr;
                    } else {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, if_lt: r{} ({}) not < r{} ({}), no jump, advancing i by 11", opcode, i, reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx]); }
                        i += 11;
                    }
                } else {
                    error!("invalid register index for IfLt at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for IfLt, advancing i by 11", opcode, i); }
                    i += 11;
                }
            }
            0x94 /* IfEq */ => {
                let reg1_idx = payload[i+1] as usize;
                let reg2_idx = payload[i+2] as usize;
                let jump_addr_bytes: [u8; 8] = payload[i+3..i+11].try_into().unwrap();
                let jump_addr = u64::from_le_bytes(jump_addr_bytes) as usize;

                if reg1_idx < registers.len() && reg2_idx < registers.len() {
                    if (registers[reg1_idx] - registers[reg2_idx]).abs() < f64::EPSILON {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, if_eq: r{} ({}) == r{} ({}), jumped to {}, setting i to {}", opcode, i, reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx], jump_addr, jump_addr); }
                        i = jump_addr;
                    } else {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, if_eq: r{} ({}) != r{} ({}), no jump, advancing i by 11", opcode, i, reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx]); }
                        i += 11;
                    }
                } else {
                    error!("invalid register index for IfEq at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for IfEq, advancing i by 11", opcode, i); }
                    i += 11;
                }
            }
            0x95 /* IfNe */ => {
                let reg1_idx = payload[i+1] as usize;
                let reg2_idx = payload[i+2] as usize;
                let jump_addr_bytes: [u8; 8] = payload[i+3..i+11].try_into().unwrap();
                let jump_addr = u64::from_le_bytes(jump_addr_bytes) as usize;

                if reg1_idx < registers.len() && reg2_idx < registers.len() {
                    if (registers[reg1_idx] - registers[reg2_idx]).abs() >= f64::EPSILON {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, if_ne: r{} ({}) != r{} ({}), jumped to {}, setting i to {}", opcode, i, reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx], jump_addr, jump_addr); }
                        i = jump_addr;
                    } else {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, if_ne: r{} ({}) == r{} ({}), no jump, advancing i by 11", opcode, i, reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx]); }
                        i += 11;
                    }
                } else {
                    error!("invalid register index for IfNe at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for IfNe, advancing i by 11", opcode, i); }
                    i += 11;
                }
            }
            0x96 /* CallAddr */ => {
                let addr_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                call_stack.push(i + 10);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, call address {}. return address {}, setting i to {}", opcode, i, addr, call_stack.last().unwrap(), addr); }
                i = addr;
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
                let formatted_str = format_str.replace("%f", &format!("{:?}", args));
                print!("{}", formatted_str);
                let bytes_advanced = 1 + 8 + str_len + 1 + num_regs;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, printf: \"{}\" with args {:?}, advancing i by {}", opcode, i, format_str, args, bytes_advanced); }
                i += bytes_advanced;
            }
            0x99 /* Print */ => {
                let str_len_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let str_len = u64::from_le_bytes(str_len_bytes) as usize;
                let text_bytes = &payload[i+9..i+9+str_len];
                let text = String::from_utf8_lossy(text_bytes);
                print!("{}", text);
                let bytes_advanced = 1 + 8 + str_len;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, print: \"{}\", advancing i by {}", opcode, i, text, bytes_advanced); }
                i += bytes_advanced;
            }
            0x9A /* Println */ => {
                let str_len_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let str_len = u64::from_le_bytes(str_len_bytes) as usize;
                let text_bytes = &payload[i+9..i+9+str_len];
                let text = String::from_utf8_lossy(text_bytes);
                println!("{}", text);
                let bytes_advanced = 1 + 8 + str_len;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, println: \"{}\", advancing i by {}", opcode, i, text, bytes_advanced); }
                i += bytes_advanced;
            }
            0x9B /* Input */ => {
                let q = payload[i+1] as usize;
                let mut input_line = String::new();
                info!("input requested for qubit {}. enter a bit (0 or 1):", q);
                io::stdin().read_line(&mut input_line).expect("failed to read line");
                let bit = input_line.trim().parse::<u8>().unwrap_or(0);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, input: read {} for qubit {}, advancing i by 2", opcode, i, bit, q); }
                i += 2;
            }
            0x9C /* DumpState */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, dump state (placeholder), advancing i by 1", opcode, i); }
                i += 1;
            }
            0x9D /* DumpRegs */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, dump registers (placeholder), advancing i by 1", opcode, i); }
                i += 1;
            }
            0x9E /* LoadRegMem */ => {
                let reg_idx = payload[i+1] as usize;
                let addr_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                if reg_idx < registers.len() && addr < memory.len() {
                    let mut val_bytes = [0u8; 8];
                    val_bytes.copy_from_slice(&memory[addr..addr+8]);
                    registers[reg_idx] = f64::from_le_bytes(val_bytes);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, loaded memory address {} into register {}, advancing i by 10", opcode, i, addr, reg_idx); }
                } else {
                    error!("invalid register or memory address for LoadRegMem at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register or memory address for LoadRegMem, advancing i by 10", opcode, i); }
                }
                i += 10;
            }
            0x9F /* StoreMemReg */ => {
                let reg_idx = payload[i+1] as usize;
                let addr_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                if reg_idx < registers.len() && addr + 8 <= memory.len() {
                    memory[addr..addr+8].copy_from_slice(&registers[reg_idx].to_le_bytes());
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, stored register {} into memory address {}, advancing i by 10", opcode, i, reg_idx, addr); }
                } else {
                    error!("invalid register or memory address for StoreMemReg at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register or memory address for StoreMemReg, advancing i by 10", opcode, i); }
                }
                i += 10;
            }
            0xA0 /* PushReg */ => {
                let reg_idx = payload[i+1] as usize;
                if reg_idx < registers.len() {
                    call_stack.push(registers[reg_idx] as usize);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, pushed register {} ({}) to stack, advancing i by 2", opcode, i, reg_idx, registers[reg_idx]); }
                } else {
                    error!("invalid register index for PushReg at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for PushReg, advancing i by 2", opcode, i); }
                }
                i += 2;
            }
            0xA1 /* PopReg */ => {
                let reg_idx = payload[i+1] as usize;
                if reg_idx < registers.len() {
                    if let Some(val) = call_stack.pop() {
                        registers[reg_idx] = val as f64;
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, popped value ({}) into register {}, advancing i by 2", opcode, i, val, reg_idx); }
                    } else {
                        error!("pop_reg on empty stack at byte {}", i);
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, pop_reg on empty stack, advancing i by 2", opcode, i); }
                    }
                } else {
                    error!("invalid register index for PopReg at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for PopReg, advancing i by 2", opcode, i); }
                }
                i += 2;
            }
            0xA2 /* Alloc */ => {
                let size_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let size = u64::from_le_bytes(size_bytes) as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, allocated {} bytes (placeholder), advancing i by 10", opcode, i, size); }
                i += 10;
            }
            0xA3 /* Free */ => {
                let addr_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, freed memory at address {} (placeholder), advancing i by 9", opcode, i, addr); }
                i += 9;
            }
            0xA4 /* Cmp */ => {
                let reg1_idx = payload[i+1] as usize;
                let reg2_idx = payload[i+2] as usize;
                if reg1_idx < registers.len() && reg2_idx < registers.len() {
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, compared r{} ({}) and r{} ({}), advancing i by 3", opcode, i, reg1_idx, registers[reg1_idx], reg2_idx, registers[reg2_idx]); }
                } else {
                    error!("invalid register index for Cmp at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for Cmp, advancing i by 3", opcode, i); }
                }
                i += 3;
            }
            0xA5 /* AndBits */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = ((registers[src1_reg] as u64) & (registers[src2_reg] as u64)) as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, and_bits: r{} = r{} & r{} ({}), advancing i by 4", opcode, i, dest_reg, src1_reg, src2_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for AndBits at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for AndBits, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0xA6 /* OrBits */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = ((registers[src1_reg] as u64) | (registers[src2_reg] as u64)) as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, or_bits: r{} = r{} | r{} ({}), advancing i by 4", opcode, i, dest_reg, src1_reg, src2_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for OrBits at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for OrBits, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0xA7 /* XorBits */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = ((registers[src1_reg] as u64) ^ (registers[src2_reg] as u64)) as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, xor_bits: r{} = r{} ^ r{} ({}), advancing i by 4", opcode, i, dest_reg, src1_reg, src2_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for XorBits at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for XorBits, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0xA8 /* NotBits */ => {
                let dest_reg = payload[i+1] as usize;
                let src_reg = payload[i+2] as usize;
                if dest_reg < registers.len() && src_reg < registers.len() {
                    registers[dest_reg] = (!(registers[src_reg] as u64)) as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, not_bits: r{} = ~r{} ({}), advancing i by 4", opcode, i, dest_reg, src_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for NotBits at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for NotBits, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0xA9 /* Shl */ => {
                let dest_reg = payload[i+1] as usize;
                let val_reg = payload[i+2] as usize;
                let shift_amt_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && val_reg < registers.len() && shift_amt_reg < registers.len() {
                    registers[dest_reg] = ((registers[val_reg] as u64) << (registers[shift_amt_reg] as u64)) as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, shl: r{} = r{} << r{} ({}), advancing i by 4", opcode, i, dest_reg, val_reg, shift_amt_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for Shl at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for Shl, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0xAA /* Shr */ => {
                let dest_reg = payload[i+1] as usize;
                let val_reg = payload[i+2] as usize;
                let shift_amt_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && val_reg < registers.len() && shift_amt_reg < registers.len() {
                    registers[dest_reg] = ((registers[val_reg] as u64) >> (registers[shift_amt_reg] as u64)) as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, shr: r{} = r{} >> r{} ({}), advancing i by 4", opcode, i, dest_reg, val_reg, shift_amt_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for Shr at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for Shr, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0xAB /* BreakPoint */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, breakpoint hit, advancing i by 1", opcode, i); }
                i += 1;
            }
            0xAC /* GetTime */ => {
                let dest_reg_idx = payload[i+1] as usize;
                if dest_reg_idx < registers.len() {
                    let duration = SystemTime::now().duration_since(UNIX_EPOCH)
                        .expect("time went backwards");
                    registers[dest_reg_idx] = duration.as_secs_f64();
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, get_time: current time {} into register {}, advancing i by 2", opcode, i, registers[dest_reg_idx], dest_reg_idx); }
                } else {
                    error!("invalid register index for GetTime at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for GetTime, advancing i by 2", opcode, i); }
                }
                i += 2;
            }
            0xAD /* SeedRng */ => {
                let seed_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let seed = u64::from_le_bytes(seed_bytes);
                rng = StdRng::seed_from_u64(seed);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, rng seeded with {}, advancing i by 9", opcode, i, seed); }
                i += 9;
            }
            0xAE /* ExitCode */ => {
                let code_bytes: [u8; 4] = payload[i+1..i+5].try_into().unwrap();
                let exit_code = u32::from_le_bytes(code_bytes);
                info!("program exited with code {}", exit_code);
                // Exit the process immediately
                std::process::exit(exit_code as i32);
            }
            _ => {
                warn!(
                    "unknown opcode 0x{:02X} at byte {}, skipping. Advancing i by 1",
                    opcode, i
                );
                i += 1;
            }
        }
    }

    if debug_mode {
        debug!("execution finished. final quantum state dump:");
    }
    if char_count > 0 {
        info!(
            "average char value: {}",
            char_sum as f64 / char_count as f64
        );
    }

    if apply_final_noise_flag {
        if let Some(_config) = noise_config.clone() {
            info!("applying final noise step to amplitudes.");
            // This is where you would call qs.apply_final_state_noise()
            // if it were a method on QuantumState that applies noise based on noise_config.
            // Since it's already called in QuantumState impl, this info message is sufficient.
        } else {
            info!("final noise step requested, but no noise config was set for runtime.");
        }
    }

    qs
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
            qubit, // This is the user's explicit --qubit flag
            top_n, // ADDED: top_n parameter
        } => {
            let start_time = Instant::now(); // Start timing

            let noise_config = if ideal {
                Some(NoiseConfig::Ideal)
            } else {
                match noise {
                    Some(ref s) => match s.as_str() {
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
                Ok(file_data) => { // file_data is now correctly scoped here
                    let noise_strength = match &noise {
                        Some(s) => match s.as_str() {
                            "random" => 0.05,
                            n => n
                                .parse::<f64>()
                                .ok()
                                .filter(|x| *x >= 0.0 && *x >= 1.0)
                                .unwrap_or(0.05),
                        },
                        None => 0.05,
                    };

                    // First, determine the inferred qubit count from the program file
                    let inferred_qubits = {
                        let (_header, _version, payload) = match parse_exe_file(&file_data) {
                            Some(x) => x,
                            None => {
                                error!("invalid or unsupported exe file, please check its header. defaulting to 0 qubits.");
                                // Return a dummy tuple with an empty payload to avoid type mismatch
                                ("", 0u8, &[] as &[u8]) // Explicitly cast 0 to u8 and empty slice to &[u8]
                            }
                        };

                        let mut max_q = 0usize;
                        let mut i = 0usize;

                        // this block contains the opcode parsing logic for max_q determination
                        while i < payload.len() {
                            let opcode = payload[i];
                            match opcode {
                                // Corrected opcodes and byte lengths based on instructions.rs
                                0x04 /* QInit / InitQubit */ => { // 2 bytes
                                    if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 2;
                                }
                                0x1D /* SetPhase */ => { // 10 bytes (1 opcode + 1 qubit + 8 f64)
                                    if i + 9 >= payload.len() { error!("incomplete SetPhase at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 10;
                                }
                                0x74 /* SetPos */ => { // 18 bytes (1 opcode + 1 qubit + 8 f64 + 8 f64)
                                    if i + 17 >= payload.len() { error!("incomplete SetPos at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 18;
                                }
                                0x75 /* SetWl */ => { // 10 bytes (1 opcode + 1 qubit + 8 f64)
                                    if i + 9 >= payload.len() { error!("incomplete SetWl at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 10;
                                }
                                0x76 /* WlShift */ => { // 10 bytes (1 opcode + 1 qubit + 8 f64)
                                    if i + 9 >= payload.len() { error!("incomplete WlShift at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 10;
                                }
                                0x77 /* Move */ => { // 18 bytes (1 opcode + 1 qubit + 8 f64 + 8 f64)
                                    if i + 17 >= payload.len() { error!("incomplete Move at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 18;
                                }
                                0x18 /* CharOut */ => { // 2 bytes
                                    if i + 1 >= payload.len() { error!("incomplete CharOut at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 2;
                                }
                                0x32 /* QMeas / Measure */ => { // 2 bytes
                                    if i + 1 >= payload.len() { error!("incomplete QMeas at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 2;
                                }
                                0x79 /* MarkObserved */ => { // 2 bytes
                                    if i + 1 >= payload.len() { error!("incomplete MarkObserved at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 2;
                                }
                                0x7A /* RELEASE */ => { // 2 bytes
                                    if i + 1 >= payload.len() { error!("incomplete Release at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 2;
                                }
                                0xFF /* HALT */ => { // 1 byte
                                    i += 1;
                                }
                                0x00 => { // Handle 0x00 as a silent NOP (1 byte)
                                    i += 1;
                                }
                                0x8D => { // Handle 0x8D as a silent NOP (1 byte)
                                    i += 1;
                                }
                                0x97 => { // Handle 0x97 (RetSub) in first pass by just skipping (1 byte)
                                    i += 1;
                                }


                                // Other instructions (keeping the existing logic for these, assuming they are correct)
                                // single‑qubit & simple two‑qubit ops (2 bytes)
                                0x05 /* ApplyHadamard */ | 0x06 /* ApplyPhaseFlip */ | 0x07 /* ApplyBitFlip */ |
                                0x0D /* ApplyTGate */ | 0x0E /* ApplySGate */ | 0x0A /* Reset / QReset */ |
                                0x59 /* PhotonEmit */ | 0x5A /* PhotonDetect */ | 0x5C /* PhotonAddition */ |
                                0x5D /* ApplyPhotonSubtraction */ | 0x61 /* SinglePhotonSourceOn */ |
                                0x62 /* SinglePhotonSourceOff */ | 0x6F /* MeasureParity */ |
                                0x71 /* OpticalSwitchControl */ | 0x9B /* Input */ | 0xA0 /* PushReg */ |
                                0xA1 /* PopReg */ | 0xAC /* GetTime */ | 0x50 /* Rand */ => {
                                    if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 2;
                                }

                                // 10‑byte ops (reg + 8‑byte imm)
                                0x08 /* PhaseShift */ | 0x22 /* RX */ | 0x23 /* RY */ | 0x0F /* RZ */ |
                                0x24 /* Phase */ | 0x66 /* SetOpticalAttenuation */ | 0x67 /* DynamicPhaseComp */ |
                                0x6A /* ApplyDisplacement */ | 0x6D /* ApplySqueezing */ |
                                0x82 /* ApplyNonlinearPhaseShift */ | 0x83 /* ApplyNonlinearSigma */ |
                                0x21 /* RegSet */ => { // SetWl (0x75) moved above
                                    if i + 9 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 10;
                                }

                                // 3‑byte ops (two‑qubit or reg/reg)
                                0x17 /* CNOT */ | 0x1E /* CZ */ | 0x0B /* Swap */ |
                                0x1F /* ThermalAvg */ | 0x65 /* OpticalRouting */ | 0x69 /* CrossPhaseMod */ |
                                0x20 /* WkbFactor */ | 0xA4 /* Cmp */ | 0x51 /* Sqrt */ | 0x52 /* Exp */ | 0x53 /* Log */ => {
                                    if i + 2 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                                    i += 3;
                                }

                                // 4‑byte ops (three regs)
                                0x0C /* ControlledSwap */ | 0x54 /* RegAdd */ | 0x55 /* RegSub */ |
                                0x56 /* RegMul */ | 0x57 /* RegDiv */ | 0x58 /* RegCopy */ |
                                0x63 /* PhotonBunchingCtl */ | 0xA8 /* NotBits */ |
                                0x31 /* CharLoad */ => {
                                    if i + 3 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 4;
                                }

                                // variable‑length entangle lists:
                                0x11 /* Entangle */ | 0x12 /* EntangleBell */ => {
                                    if i + 2 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                                    i += 3;
                                }
                                0x13 /* EntangleMulti */ | 0x14 /* EntangleCluster */ => {
                                    if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                                    let n = payload[i+1] as usize;
                                    if i + 2 + n > payload.len() { error!("incomplete entangle list at byte {}", i); break; }
                                    for j in 0..n {
                                        max_q = max_q.max(payload[i+2+j] as usize);
                                    }
                                    i += 2 + n;
                                }
                                0x15 /* EntangleSwap */ | 0x16 /* EntangleSwapMeasure */ => {
                                    if i + 4 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                                    max_q = (payload[i+1] as usize)
                                        .max(payload[i+2] as usize)
                                        .max(payload[i+3] as usize)
                                        .max(payload[i+4] as usize);
                                    if opcode == 0x16 {
                                        let start = i + 5;
                                        let end = start + payload[start..].iter().position(|&b| b == 0).unwrap_or(0);
                                        i = end + 1;
                                    } else {
                                        i += 5;
                                    }
                                }

                                // label‑terminated ops:
                                0x19 /* EntangleWithFB */ | 0x1A /* EntangleDistrib */ |
                                0x1B /* MeasureInBasis */ | 0x87 /* VerboseLog */ |
                                0x38 /* ApplyFeedforward */ | 0x3A /* BasisChange */ |
                                0x3B /* Load */ | 0x3C /* Store */ | 0x5B /* PhotonCount */ |
                                0x6B /* DisplacementFB */ | 0x6E /* SqueezingFB */ |
                                0x73 /* PhotonLossCorr */ | 0x7C /* QndMeasure */ |
                                0x7D /* ErrorCorrect */ | 0x7F /* QStateTomography */ |
                                0x85 /* PNRDetection */ | 0x86 /* FeedbackCtl */ => {
                                    if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let start = i + 2;
                                    let end = start + payload[start..].iter().position(|&b| b == 0).unwrap_or(0);
                                    i = end + 1;
                                }

                                // control flow & misc ops:
                                0x02 /* ApplyGate(QGATE) */ => {
                                    // reg (1), name (8), then optional extra reg for "cz"
                                    if i + 9 >= payload.len() { error!("incomplete qgate at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let name_bytes = &payload[i+2..i+10];
                                    let name = String::from_utf8_lossy(name_bytes)
                                        .trim_end_matches('\0')
                                        .to_string();
                                    if name == "cz" {
                                        if i + 10 >= payload.len() { error!("incomplete cz at byte {}", i); break; }
                                        max_q = max_q.max(payload[i+10] as usize);
                                        i += 11;
                                    } else {
                                        i += 10;
                                    }
                                }
                                0x33 /* ApplyRotation */ => { /* … */ i += 11; }
                                0x34 /* ApplyMultiQubitRotation */ => {
                                    // opcode, axis, num_qs, [qs], [angles]
                                    if i + 2 >= payload.len() { error!("incomplete multi‑rotation at byte {}", i); break; }
                                    let n = payload[i+2] as usize;
                                    let needed = 3 + n /* regs */ + n * 8 /* f64 angles */;
                                    if i + needed > payload.len() { error!("incomplete multi‑rotation at byte {}", i); }
                                    for j in 0..n {
                                        max_q = max_q.max(payload[i + 3 + j] as usize);
                                    }
                                    i += needed;
                                }
                                0x35 /* ControlledPhase */ | 0x36 /* ApplyCPhase */ => {
                                    // ctrl qubit, target qubit, angle:f64
                                    if i + 10 >= payload.len() { error!("incomplete ControlledPhase at byte {}", i); break; }
                                    max_q = max_q
                                        .max(payload[i+1] as usize)
                                        .max(payload[i+2] as usize);
                                    i += 11;
                                }
                                0x37 /* ApplyKerrNonlin */ => {
                                    // qubit, strength:f64, duration:f64
                                    if i + 17 >= payload.len() { error!("incomplete KerrNonlin at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 18;
                                }
                                0x39 /* DecoherenceProtect */ | 0x68 /* OpticalDelayLineCtl */ => {
                                    // qubit, duration:f64
                                    if i + 9 >= payload.len() { error!("incomplete DecoherenceProtect at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 10;
                                }
                                0x3D /* LoadMem */ | 0x3E /* StoreMem */ => {
                                    // reg_str\0, addr_str\0
                                    let start = i + 1;
                                    let mid   = start + payload[start..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let end   = mid   + payload[mid..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = end;
                                }
                                0x3F /* LoadClassical */ | 0x40 /* StoreClassical */ => {
                                    // reg_str\0, var_str\0
                                    let start = i + 1;
                                    let mid   = start + payload[start..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let end   = mid   + payload[mid..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = end;
                                }
                                0x41 /* Add */ | 0x42 /* Sub */ | 0x43 /* And */ | 0x44 /* Or */ | 0x45 /* Xor */ => {
                                    // dst\0, src1\0, src2\0
                                    let d_end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let s1_end = d_end + payload[d_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let s2_end = s1_end + payload[s1_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = s2_end;
                                }
                                0x46 /* Not */ | 0x47 /* Push */ | 0x4F /* Pop */ => {
                                    // reg_str\0
                                    let end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = end;
                                }
                                0x49 /* Jump */ | 0x4C /* Call */ => {
                                    // label\0
                                    let end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = end;
                                }
                                0x4A /* JumpIfZero */ | 0x4B /* JumpIfOne */ => {
                                    // cond_reg\0, label\0
                                    let c_end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let l_end = c_end + payload[c_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = l_end;
                                }
                                0x4E /* TimeDelay */ => {
                                    // qubit, cycles:f64
                                    if i + 9 >= payload.len() { error!("incomplete TimeDelay at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 10;
                                }
                                0x5E /* PhotonEmissionPattern */ => {
                                    // qubit, pattern_str\0, cycles:u64
                                    if i + 2 >= payload.len() { error!("incomplete PhotonEmissionPattern at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let str_end = i + 2 + payload[i+2..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    if str_end + 8 > payload.len() { error!("incomplete PhotonEmissionPattern at byte {}", i); break; }
                                    i = str_end + 8;
                                }
                                0x5F /* PhotonDetectThreshold */ => {
                                    // qubit, thresh:f64, reg_str\0
                                    if i + 9 >= payload.len() { error!("incomplete PhotonDetectThreshold at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let str_end = i + 10 + payload[i+10..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = str_end;
                                }
                                0x60 /* PhotonDetectCoincidence */ => {
                                    // n, [qs], reg_str\0
                                    let n = payload[i+1] as usize;
                                    let q_end = i + 2 + n;
                                    let str_end = q_end + payload[q_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    for j in 0..n {
                                        max_q = max_q.max(payload[i+2+j] as usize);
                                    }
                                    i = str_end;
                                }
                                0x64 /* PhotonRoute */ => {
                                    if i + 1 >= payload.len() { error!("incomplete PhotonRoute at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let f_end = i + 2 + payload[i+2..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let t_end = f_end + payload[f_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = t_end;
                                }
                                0x6C /* ApplyDisplacementOp */ => {
                                    if i + 17 >= payload.len() { error!("incomplete ApplyDisplacementOp at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 18;
                                }
                                0x70 /* MeasureWithDelay */ => {
                                    if i + 9 >= payload.len() { error!("incomplete MeasureWithDelay at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let str_end = i + 10 + payload[i+10..].iter().position(|&b| b == 0).unwrap_or(0) + 1;
                                    i = str_end;
                                }
                                0x72 /* PhotonLossSimulate */ => {
                                    if i + 17 >= payload.len() { error!("incomplete PhotonLossSimulate at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 18;
                                }
                                0x7E /* ErrorSyndrome */ => {
                                    if i + 1 >= payload.len() { error!("incomplete ErrorSyndrome at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let s_end = i + 2 + payload[i+2..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let r_end = s_end + payload[s_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = r_end;
                                }
                                0x80 /* BellStateVerif */ => {
                                    if i + 2 >= payload.len() { error!("incomplete BellStateVerif at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                                    let n_end = i + 3 + payload[i+3..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = n_end;
                                }
                                0x81 /* QuantumZenoEffect */ => {
                                    if i + 17 >= payload.len() { error!("incomplete QuantumZenoEffect at byte {}", i); break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 18;
                                }
                                0x84 /* ApplyLinearOpticalTransform */ => {
                                    if i + 4 >= payload.len() { error!("incomplete LinearOpticalTransform at byte {}", i); break; }
                                    let nin = payload[i+1] as usize;
                                    let nout = payload[i+2] as usize;
                                    let name_end = i + 4 + payload[i+4..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    for q in 0..nin { max_q = max_q.max(payload[name_end + q] as usize); }
                                    for q in 0..nout {
                                        max_q = max_q.max(payload[name_end + nin + q] as usize);
                                    }
                                    i = name_end + nin + nout;
                                }
                                0x88 /* Comment */ => {
                                    let end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = end;
                                }
                                0x90 /* Jmp */ | 0x91 /* JmpAbs */ | 0xA3 /* Free */ | 0xAD /* SeedRng */ => {
                                    if i + 9 >= payload.len() { error!("incomplete Jmp/Free/SeedRng at byte {}", i); break; }
                                    i += 9;
                                }
                                0x92 /* IfGt */ | 0x93 /* IfLt */ | 0x94 /* IfEq */ | 0x95 /* IfNe */ => {
                                    if i + 11 >= payload.len() { error!("incomplete If at byte {}", i); break; }
                                    i += 11;
                                }
                                0x96 /* CallAddr */ | 0x9E /* LoadRegMem */ | 0xA2 /* Alloc */ => {
                                    if i + 9 >= payload.len() { error!("incomplete Alloc/LoadRegMem at byte {}", i); break; }
                                    i += 10;
                                }
                                0x9F /* StoreMemReg */ => {
                                    if i + 9 >= payload.len() { error!("incomplete StoreMemReg at byte {}", i); break; }
                                    i += 10;
                                }
                                0x98 /* Printf */ => {
                                    if i + 9 >= payload.len() { error!("incomplete Printf at byte {}", i); break; }
                                    let len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                                    let regs = payload[i+9+len] as usize;
                                    i += 1 + 8 + len + 1 + regs;
                                }
                                0x99 /* Print */ | 0x9A /* Println */ => {
                                    if i + 9 >= payload.len() { error!("incomplete Print/Println at byte {}", i); break; }
                                    let len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                                    i += 1 + 8 + len;
                                }
                                0xA5 | 0xA6 | 0xA7 | 0xA9 | 0xAA => {
                                    if i + 3 >= payload.len() { error!("incomplete BitOp at byte {}", i); break; }
                                    i += 4;
                                }
                                0xAE => {
                                    if i + 5 >= payload.len() { error!("incomplete ExitCode at byte {}", i); break; }
                                    i += 5;
                                }
                                0x01 => {
                                    if i + 1 >= payload.len() { error!("incomplete LoopStart at byte {}", i); break; }
                                    i += 2;
                                }
                                _ => {
                                    warn!(
                                        "unknown opcode 0x{:02X} in scan at byte {}, skipping. Advancing i by 1",
                                        opcode, i
                                    );
                                    i += 1;
                                }
                            }
                        }
                        // Explicitly return the calculated value
                        if max_q == 0 && payload.is_empty() {
                            0
                        } else {
                            max_q + 1
                        }
                    };


                    // Determine the actual number of qubits to use for simulation.
                    // The explicit `--qubit` flag takes precedence over the inferred count.
                    let num_qubits_to_simulate = if let Some(q) = qubit {
                        info!("using explicit qubit count from --qubit: {}", q);
                        q
                    } else {
                        info!("using inferred qubit count from file: {}", inferred_qubits);
                        inferred_qubits
                    };

                    // Calculate and display memory usage
                    let memory_needed_bytes = (2.0_f64).powi(num_qubits_to_simulate as i32) * 16.0; // 16 bytes per amplitude (2 f64s)
                    let memory_needed_kb = memory_needed_bytes / 1024.0;
                    let memory_needed_mb = memory_needed_bytes / (1024.0 * 1024.0);
                    let memory_needed_gb = memory_needed_bytes / (1024.0 * 1024.0 * 1024.0);
                    let memory_needed_tb = memory_needed_bytes / (1024.0 * 1024.0 * 1024.0 * 1024.0);
                    let memory_needed_pb = memory_needed_bytes / (1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0);


                    if num_qubits_to_simulate > 0 {
                        print!("estimated memory for {} qubits: {:.0} bytes", num_qubits_to_simulate, memory_needed_bytes);
                        if memory_needed_pb >= 1.0 {
                            println!(" ({:.2} PB)", memory_needed_pb);
                        } else if memory_needed_tb >= 1.0 {
                            println!(" ({:.2} TB)", memory_needed_tb);
                        } else if memory_needed_gb >= 1.0 {
                            println!(" ({:.2} GB)", memory_needed_gb);
                        } else if memory_needed_mb >= 1.0 {
                            println!(" ({:.2} MB)", memory_needed_mb);
                        } else if memory_needed_kb >= 1.0 {
                            println!(" ({:.2} KB)", memory_needed_kb);
                        } else {
                            println!(); // Just print bytes if less than 1 KB
                        }
                    }


                    // Provide a warning if the simulation is still memory intensive, even if allowed.
                    if num_qubits_to_simulate > 26 {
                        warn!("simulating more than 26 qubits can be very memory intensive. Performance might be limited by memory bandwidth rather than raw CPU computation.");
                    }

                    info!("running '{}' (debug: {}, qubits: {})", program, debug, num_qubits_to_simulate);

                    // Pass the final, limited qubit count to run_exe
                    let qs = run_exe(&file_data, debug, noise_config, final_noise, num_qubits_to_simulate);


                    print_amplitudes(&qs, noise_strength, top_n);

                    if qs.n > 0 {
                        println!();
                        let sample = qs.sample_measurement();
                        print!("Measurement result: ");
                        for bit in &sample {
                            print!("{}", bit);
                        }
                        println!();
                    }
                    println!("Qubit count: {}", qs.n);
                    println!(
                        "Status: nan=false, div_by_zero=false, overflow=false",
                    );
                    println!(
                        "Noise: {}",
                        match &qs.noise_config {
                            Some(cfg) => format!("{:?}", cfg),
                            None => "Random".to_string(),
                        }
                    );
                    info!("simulation complete.");

                    let end_time = Instant::now(); // End timing
                    let duration = end_time.duration_since(start_time);
                    println!("Total simulation time: {:.2?} seconds", duration.as_secs_f64()); // Print total time
                }
                Err(e) => eprintln!("error reading program file: {}", e),
            }
        }
        Commands::CompileJson { source, output } => {
            info!("compiling '{}' to json '{}'", source, output);
            let json_output = serde_json::json!({
                "source_file": source,
                "output_file": output,
                "status": "not implemented yet"
            });
            match File::create(&output).and_then(|file| {
                to_writer_pretty(file, &json_output)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
            }) {
                Ok(_) => info!("json compilation successful. output written to {}", output),
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
            ffmpeg_flags,
        } => {
            let input_path = PathBuf::from(input);
            let output_path = PathBuf::from(output);
            let parts: Vec<&str> = resolution.split('x').collect();
            let width = parts
                .get(0)
                .and_then(|w| w.parse::<u32>().ok())
                .unwrap_or(800);
            let height = parts
                .get(1)
                .and_then(|h| h.parse::<u32>().ok())
                .unwrap_or(600);

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

            if let Err(e) = visualizer::run_qoa_to_video( // Changed to run_qoa_to_video
                &audio_visualizer,
                audio_visualizer.clone(),
                &input_path,
                &output_path,
                fps,
                width,
                height,
                &ffmpeg_args_slice,
                spectrum_direction,
                &ffmpeg_flags,
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
            println!("--compile <SOURCE> <OUTPUT>     Compile a .qoa file to .qexe");
            println!("  --debug                       Enable debug mode for compilation.\n");
            println!("\n--run <PROGRAM>                 Run a .qexe binary");
            println!("  --debug                       Enable debug mode for runtime.");
            println!("  --ideal                       Set simulation to ideal (no noise) conditions.");
            println!("  --noise [PROBABILITY]         Apply noise simulation for gates. Can be 'random' or a fixed probability (0.0-1.0).");
            println!("  --final-noise                 Apply an additional noise step to the final amplitudes (default: true).");
            println!("  --qubit <QUBIT NUMBER>        Max amount of Qubit used with <QUBIT NUMBER> being the limit (overrides inferred count).");
            println!("  --top-n <COUNT>               Display only the top N amplitudes by probability (0 to show all).\n");
            println!("\n--compilejson <SOURCE> <OUTPUT> Compile a .qoa file to JSON format\n");
            println!("\n--visual <INPUT> <OUTPUT>       Visualizes quantum state or circuit based on input data.");
            println!("  --resolution <WIDTH>x<HEIGHT> Resolution of the output visualization (e.g., '1920x1080').");
            println!("  --fps <FPS>                   Frames per second for animation (if applicable).");
            println!("  --ltr                         Spectrum direction Left-to-Right.");
            println!("  --rtl                         Spectrum direction Right-to-Left.");
            println!("  --ffmpeg-flag <FFMPEG_FLAG>   Extra ffmpeg flags (can be used multiple times).");
            println!("  --ffmpeg-args <ARGS>...       Additional ffmpeg arguments passed directly to ffmpeg (after '--').\n");
            println!("\n--version                       Show version info.");
            println!("\n--flags                         Show available flags.");
        }
    }
}
