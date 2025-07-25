#![feature(portable_simd)]

use crate::visualizer::SpectrumDirection;
use clap::{Parser, Subcommand};
use log::{debug, error, info, warn};
use qoa::runtime::quantum_state::NoiseConfig;
use qoa::runtime::quantum_state::QuantumState;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use serde_json::to_writer_pretty;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use rand::rngs::ThreadRng;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

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

const QOA_VERSION: &str = "0.3.1";
const QOA_AUTHOR: &str = "Rayan (@planetryan on github)";

#[derive(Parser, Debug)]
#[command(name = "qoa", author = QOA_AUTHOR, version = QOA_VERSION,
    about = format!("QOA (Quantum Optical Assembly) - a free, open source, quantum QPU simulator and assembly language.\n
             author: {QOA_AUTHOR}
             version: {QOA_VERSION}\n
             use 'qoa help <command>' for more information on a specific command, e.g., 'qoa help run'."),
    long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    // compiles a .qoa source file into a .qexe binary executable.
    Compile {
        // source .qoa file path
        source: String,
        // output .qexe file path
        output: String,
        // enable debug mode for compilation.
        #[arg(long)]
        debug: bool,
    },
    // runs a .qexe binary executable.
    Run {
        // program .qexe file path
        program: String,
        // enable debug mode for runtime.
        #[arg(long)]
        debug: bool,
        // set simulation to ideal (no noise) conditions. disables --noise and --final-noise.
        #[arg(long, conflicts_with = "noise")]
        ideal: bool,
        // apply noise simulation for gates. can be `--noise` for random probability (0.1-1.0) or `--noise <probability>` for a fixed probability (0.0-1.0).
        #[arg(long, num_args = 0..=1, default_missing_value = "random", value_name = "probability")]
        noise: Option<String>,
        // apply an additional noise step to the final amplitudes before displaying them (default: true). use --final-noise false to disable this specific noise.
        #[arg(long, default_value_t = true)] // this makes --final-noise true by default
        final_noise: bool,
        // set the maximum number of qubits to simulate. this overrides the inferred qubit count from the program file.
        #[arg(long)] // added: this makes --qubit a named flag
        qubit: Option<usize>,
        // display only the top n amplitudes by probability. use --top-n 0 to show all states.
        #[arg(long, value_name = "count", default_value_t = 20)]
        top_n: usize,
        // path to save the final quantum state and registers.
        #[arg(long)]
        save_state: Option<String>,
        // path to load a quantum state and registers from.
        #[arg(long)]
        load_state: Option<String>,
    },
    // compiles a .qoa source file into a .json circuit description (ionq format).
    CompileJson {
        // source .qoa file path
        source: String,
        // output .json file path
        output: String,
    },
    // visualizes quantum state or circuit based on input data.
    Visual {
        // input file path (e.g., .qexe or raw data for visualization)
        input: String,

        // output file path for the visualization (e.g., .png, .gif, .mp4)
        output: String,

        // resolution of the output visualization (e.g., "1920x1080")
        #[arg(long, default_value = "1920x1080")]
        resolution: String,

        // frames per second for animation (if applicable)
        #[arg(long, default_value_t = 60)]
        fps: u32,

        // spectrum direction left-to-right
        #[arg(long, conflicts_with = "rtl", default_value_t = false)]
        ltr: bool,

        // spectrum direction right-to-left
        #[arg(long, conflicts_with = "ltr", default_value_t = false)]
        rtl: bool,

        // extra ffmpeg flags (e.g., -s, -r, -b:v, -pix_fmt, etc.)
        #[arg(long = "ffmpeg-flag", value_name = "ffmpeg_flag", num_args = 0.., action = clap::ArgAction::Append)]
        ffmpeg_flags: Vec<String>,

        // additional ffmpeg arguments passed directly to ffmpeg (e.g., "-c:v libx264 -crf 23") as trailing args after "--"
        #[arg(last = true, allow_hyphen_values = true)]
        ffmpeg_args: Vec<String>,
    },

    Version,

    Flags,
}

// helper function to parse resolution string
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
        match instructions::Instruction::parse_instruction(&line) {
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
        m if *m == *QEXE => "qexe",
        m if *m == *OEXE => "oexe",
        m if *m == *QOX => "qox",
        m if *m == *XEXE => "xexe",
        m if *m == *QEX => "qex",
        m if *m == *QX => "qx",
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
            filedata.len() // corrected variable name
        );
        return None;
    }
    Some((name, version, &filedata[9..9 + payload_len]))
}

// print_amplitudes now accepts a top_n parameter and classical registers
fn print_amplitudes(qs: &QuantumState, registers: &Vec<f64>, noise_strength: f64, top_n: usize) {
    println!("\nfinal amplitudes:");

    // parallelize the iteration for collecting amplitudes and probabilities
    // each thread gets its own rng for thread safety.
    let mut amplitudes_with_probs: Vec<(f64, String, f64, f64)> = qs
        .amps
        .par_iter()
        .enumerate()
        .map(|(i, amp)| {
            let mut local_rng = ThreadRng::default();
            let noise_re = <StandardNormal as rand_distr::Distribution<f64>>::sample(
                &StandardNormal,
                &mut local_rng,
            ) * noise_strength;
            let noise_im = <StandardNormal as rand_distr::Distribution<f64>>::sample(
                &StandardNormal,
                &mut local_rng,
            ) * noise_strength;
            let noisy_re = amp.re + noise_re;
            let noisy_im = amp.im + noise_im;
            let prob = noisy_re * noisy_re + noisy_im * noisy_im;
            let binary_string = format!("{:0width$b}", i, width = qs.n);
            (prob, binary_string, noisy_re, noisy_im)
        })
        .collect();

    // sort by probability in descending order
    amplitudes_with_probs
        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

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

    // print the quantum register result in binary on one line
    if let Some((_prob, binary_string, _re, _im)) = amplitudes_with_probs.first() {
        println!("\nfinal results (quantum registers): {}", binary_string);
    }

    // print classical register results in binary on one line, only if any are non-zero
    let non_zero_classical_registers_exist = registers.iter().any(|&val| val != 0.0);
    if non_zero_classical_registers_exist {
        let classical_results_binary: Vec<String> = registers.iter().map(|&val| {
            if val.is_finite() {
                format!("{:b}", val as u64)
            } else {
                // handle non-finite values, though unlikely for classical registers
                "X".to_string() // Use 'X' for NaN/Inf to keep it compact
            }
        }).collect();
        println!("final results (classical registers): {}", classical_results_binary.join(""));
    }
}

pub fn run_exe(
    filedata: &[u8],
    debug_mode: bool,
    noise_config: Option<NoiseConfig>,
    apply_final_noise_flag: bool,
    num_qubits_to_initialize: usize,
    initial_state_data: Option<(QuantumState, Vec<f64>)>,
) -> (QuantumState, Vec<f64>) {
    // returns both quantumstate and registers
    let (_header, _version, payload) = match parse_exe_file(filedata) {
        // added underscores to mark as unused
        Some(x) => x,
        None => {
            error!(
                "invalid or unsupported exe file, please check its header, defaulting to 0 qubits."
            );
            // return a quantumstate with the requested (but likely 0) qubits
            return (
                QuantumState::new(num_qubits_to_initialize, None),
                vec![0.0; 24],
            );
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
        // raw payload dump for more detailed debugging
        debug!("raw payload bytes (first 150 bytes):");
        for j in 0..payload.len().min(150) {
            debug!("byte[{:#04}] = 0x{:02X}", j, payload[j]);
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
            // qoa v0.2.7 instructions:
            // corrected opcodes and byte lengths based on instructions.rs
            0x04 /* qinit / initqubit */ => { // 2 bytes
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }
            0x1d /* setphase */ => { // 10 bytes (1 opcode + 1 qubit + 8 f64)
                if i + 9 >= payload.len() { error!("incomplete setphase at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x74 /* setpos */ => { // 18 bytes (1 opcode + 1 qubit + 8 f64 + 8 f64)
                if i + 17 >= payload.len() { error!("incomplete setpos at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18;
            }
            0x75 /* setwl */ => { // 10 bytes (1 opcode + 1 qubit + 8 f64)
                if i + 9 >= payload.len() { error!("incomplete setwl at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x76 /* wlshift */ => { // 10 bytes (1 opcode + 1 qubit + 8 f64)
                if i + 9 >= payload.len() { error!("incomplete wlshift at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x77 /* move */ => { // 18 bytes (1 opcode + 1 qubit + 8 f64 + 8 f64)
                if i + 17 >= payload.len() { error!("incomplete move at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18;
            }
            0x18 /* charout */ => { // 2 bytes
                if i + 1 >= payload.len() { error!("incomplete charout at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }
            0x32 /* qmeas / measure */ => { // 2 bytes
                if i + 1 >= payload.len() { error!("incomplete qmeas at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }
            0x79 /* markobserved */ => { // 2 bytes
                if i + 1 >= payload.len() { error!("incomplete markobserved at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }
            0x7a /* release */ => { // 2 bytes
                if i + 1 >= payload.len() { error!("incomplete release at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }
            0xff /* halt */ => { // 1 byte
                i += 1;
            }
            0x00 => { // handle 0x00 as a silent nop (1 byte)
                i += 1;
            }
            0x8d => { // handle 0x8d as a silent nop (1 byte)
                i += 1;
            }
            0x97 => { // handle 0x97 (retsub) in first pass by just skipping (1 byte)
                i += 1;
            }


            // other instructions (keeping the existing logic for these, assuming they are correct)
            // single‑qubit & simple two‑qubit ops (2 bytes)
            0x05 /* applyhadamard */ | 0x06 /* applyphaseflip */ | 0x07 /* applybitflip */ |
            0x0d /* applytgate */ | 0x0e /* applysgate */ | 0x0a /* reset / qreset */ |
            0x59 /* photonemit */ | 0x5a /* photondetect */ | 0x5c /* photonaddition */ |
            0x5d /* applyphotonsubtraction */ | 0x61 /* singlephotonsourceon */ |
            0x62 /* singlephotonsourceoff */ | 0x6f /* measureparity */ |
            0x71 /* opticalswitchcontrol */ | 0x9b /* input */ | 0xa0 /* pushreg */ |
            0xa1 /* popreg */ | 0xac /* gettime */ | 0x50 /* rand */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 2;
            }

            // 10‑byte ops (reg + 8‑byte imm)
            0x08 /* phaseshift */ | 0x22 /* rx */ | 0x23 /* ry */ | 0x0f /* rz */ |
            0x24 /* phase */ | 0x66 /* setopticalattenuation */ | 0x67 /* dynamicphasecomp */ |
            0x6a /* applydisplacement */ | 0x6d /* applysqueezing */ |
            0x82 /* applynonlinearphaseshift */ | 0x83 /* applynonlinearsigma */ |
            0x21 /* regset */ => {
                if i + 9 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }

            // 3‑byte ops (two‑qubit or reg/reg)
            0x17 /* cnot */ | 0x1e /* cz */ | 0x0b /* swap */ |
            0x1f /* thermalavg */ | 0x65 /* opticalrouting */ | 0x69 /* crossphasemod */ |
            0x20 /* wkbfactor */ | 0xa4 /* cmp */ | 0x51 /* sqrt */ | 0x52 /* exp */ | 0x53 /* log */ => {
                if i + 2 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                i += 3;
            }

            // 4‑byte ops (three regs)
            0x0c /* controlledswap */ | 0x54 /* regadd */ | 0x55 /* regsub */ |
            0x56 /* regmul */ | 0x57 /* regdiv */ | 0x58 /* regcopy */ |
            0x63 /* photonbunchingctl */ | 0xa8 /* notbits */ |
            0x31 /* charload */ => {
                if i + 3 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 4;
            }

            // variable‑length entangle lists:
            0x11 /* entangle */ | 0x12 /* entanglebell */ => {
                if i + 2 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                i += 3;
            }
            0x13 /* entanglemulti */ | 0x14 /* entanglecluster */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                let n = payload[i+1] as usize;
                if i + 2 + n > payload.len() { error!("incomplete entangle list at byte {}", i); break; }
                for j in 0..n {
                    max_q = max_q.max(payload[i+2+j] as usize);
                }
                i += 2 + n;
            }
            0x15 /* entangleswap */ | 0x16 /* entangleswapmeasure */ => {
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
            0x19 /* entanglewithfb */ | 0x1a /* entangledistrib */ |
            0x1b /* measureinbasis */ | 0x87 /* verboselog */ |
            0x38 /* applyfeedforward */ | 0x3a /* basischange */ |
            0x3b /* load */ | 0x3c /* store */ | 0x5b /* photoncount */ |
            0x6b /* displacementfb */ | 0x6e /* squeezingfb */ |
            0x73 /* photonlosscorr */ | 0x7c /* qndmeasure */ |
            0x7d /* errorcorrect */ | 0x7f /* qstatetomography */ |
            0x85 /* pnrdetection */ | 0x86 /* feedbackctl */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let start = i + 2;
                let end = start + payload[start..].iter().position(|&b| b == 0).unwrap_or(0);
                i = end + 1;
            }

            // control flow & misc ops:
            0x02 /* applygate(qgate) */ => {
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
            0x33 /* applyrotation */ => { i += 11; }
            0x34 /* applymultiqubitrotation */ => {
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
            0x35 /* controlledphase */ | 0x36 /* applycphase */ => {
                // ctrl qubit, target qubit, angle:f64
                if i + 10 >= payload.len() { error!("incomplete controlledphase at byte {}", i); break; }
                max_q = max_q
                    .max(payload[i+1] as usize)
                    .max(payload[i+2] as usize);
                i += 11;
            }
            0x37 /* applykerrnonlin */ => {
                // qubit, strength:f64, duration:f64
                if i + 17 >= payload.len() { error!("incomplete kerrnonlin at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18;
            }
            0x39 /* decoherenceprotect */ | 0x68 /* opticaldelaylinectl */ => {
                // qubit, duration:f64
                if i + 9 >= payload.len() { error!("incomplete decoherenceprotect at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x3d /* loadmem */ | 0x3e /* storemem */ => {
                // reg_str\0, addr_str\0
                let start = i + 1;
                let mid   = start + payload[start..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let end   = mid   + payload[mid..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = end;
            }
            0x3f /* loadclassical */ | 0x40 /* storeclassical */ => {
                // reg_str\0, var_str\0
                let start = i + 1;
                let mid   = start + payload[start..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let end   = mid   + payload[mid..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = end;
            }
            0x41 /* add */ | 0x42 /* sub */ | 0x43 /* and */ | 0x44 /* or */ | 0x45 /* xor */ => {
                // dst\0, src1\0, src2\0
                let d_end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let s1_end = d_end + payload[d_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let s2_end = s1_end + payload[s1_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = s2_end;
            }
            0x46 /* not */ | 0x47 /* push */ | 0x4f /* pop */ => {
                // reg_str\0
                let end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = end;
            }
            0x49 /* jump */ | 0x4c /* call */ => {
                // label\0
                let end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = end;
            }
            0x4a /* jumpifzero */ | 0x4b /* jumpifone */ => {
                // cond_reg\0, label\0
                let c_end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let l_end = c_end + payload[c_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = l_end;
            }
            0x4e /* timedelay */ => {
                // qubit, cycles:f64
                if i + 9 >= payload.len() { error!("incomplete timedelay at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 10;
            }
            0x5e /* photonemissionpattern */ => {
                // qubit, pattern_str\0, cycles:u64
                if i + 2 >= payload.len() { error!("incomplete photonemissionpattern at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let str_end = i + 2 + payload[i+2..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                if str_end + 8 > payload.len() { error!("incomplete photonemissionpattern at byte {}", i); break; }
                i = str_end + 8;
            }
            0x5f /* photondetectthreshold */ => {
                // qubit, thresh:f64, reg_str\0
                if i + 9 >= payload.len() { error!("incomplete photondetectthreshold at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let str_end = i + 10 + payload[i+10..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = str_end;
            }
            0x60 /* photondetectcoincidence */ => {
                // n, [qs], reg_str\0
                let n = payload[i+1] as usize;
                let q_end = i + 2 + n;
                let str_end = q_end + payload[q_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                for j in 0..n {
                    max_q = max_q.max(payload[i+2+j] as usize);
                }
                i = str_end;
            }
            0x64 /* photonroute */ => {
                if i + 1 >= payload.len() { error!("incomplete photonroute at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let f_end = i + 2 + payload[i+2..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let t_end = f_end + payload[f_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = t_end;
            }
            0x6c /* applydisplacementop */ => {
                if i + 17 >= payload.len() { error!("incomplete applydisplacementop at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18;
            }
            0x70 /* measurewithdelay */ => {
                if i + 9 >= payload.len() { error!("incomplete measurewithdelay at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let str_end = i + 10 + payload[i+10..].iter().position(|&b| b == 0).unwrap_or(0) + 1;
                i = str_end;
            }
            0x72 /* photonlosssimulate */ => {
                if i + 17 >= payload.len() { error!("incomplete photonlosssimulate at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18;
            }
            0x7e /* errorsyndrome */ => {
                if i + 1 >= payload.len() { error!("incomplete errorsyndrome at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                let s_end = i + 2 + payload[i+2..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                let r_end = s_end + payload[s_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = r_end;
            }
            0x80 /* bellstateverif */ => {
                if i + 2 >= payload.len() { error!("incomplete bellstateverif at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                let n_end = i + 3 + payload[i+3..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = n_end;
            }
            0x81 /* quantumzenoeffect */ => {
                if i + 17 >= payload.len() { error!("incomplete quantumzenoeffect at byte {}", i); break; }
                max_q = max_q.max(payload[i+1] as usize);
                i += 18;
            }
            0x84 /* applylinearopticaltransform */ => {
                if i + 4 >= payload.len() { error!("incomplete linearopticaltransform at byte {}", i); break; }
                let _nin = payload[i+1] as usize;
                let _nout = payload[i+2] as usize;
                let _num_modes = payload[i+3] as usize;
                let name_start = i + 4;
                let name_end = name_start + payload[name_start..].iter().position(|&b| b==0).unwrap_or(0);
                let _name = String::from_utf8_lossy(&payload[name_start..name_end]);
                let mut input_qs = Vec::with_capacity(_nin);
                let mut current_idx = name_end + 1;
                for _ in 0.._nin {
                    input_qs.push(payload[current_idx] as usize);
                    current_idx += 1;
                }
                let mut output_qs = Vec::with_capacity(_nout);
                for _ in 0.._nout {
                    output_qs.push(payload[current_idx] as usize);
                    current_idx += 1;
                }
                let _bytes_advanced = current_idx - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied linear optical transform '{}' from input qubits {:?} to output qubits {:?} with {} modes, advancing i by {}", opcode, i, _name, input_qs, output_qs, _num_modes, _bytes_advanced); }
                i = current_idx;
            }
            0x88 /* comment */ => {
                let end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                i = end;
            }
            0x90 /* jmp */ | 0x91 /* jmpabs */ | 0xa3 /* free */ | 0xad /* seedrng */ => {
                if i + 9 >= payload.len() { error!("incomplete jmp/free/seedrng at byte {}", i); break; }
                i += 9;
            }
            0x92 /* ifgt */ | 0x93 /* iflt */ | 0x94 /* ifeq */ | 0x95 /* ifne */ => {
                if i + 11 >= payload.len() { error!("incomplete if at byte {}", i); break; }
                i += 11;
            }
            0x96 /* calladdr */ | 0x9e /* loadregmem */ | 0xa2 /* alloc */ => {
                if i + 9 >= payload.len() { error!("incomplete alloc/loadregmem at byte {}", i); break; }
                i += 10;
            }
            0x9f /* storememreg */ => {
                if i + 9 >= payload.len() { error!("incomplete storememreg at byte {}", i); break; }
                i += 10;
            }
            0x98 /* printf */ => {
                if i + 9 >= payload.len() { error!("incomplete printf at byte {}", i); break; }
                let len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                let regs = payload[i+9+len] as usize;
                i += 1 + 8 + len + 1 + regs;
            }
            0x99 /* print */ | 0x9a /* println */ => {
                if i + 9 >= payload.len() { error!("incomplete print/println at byte {}", i); break; }
                let len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                i += 1 + 8 + len;
            }
            0xa5 | 0xa6 | 0xa7 | 0xa9 | 0xaa => {
                if i + 3 >= payload.len() { error!("incomplete bitop at byte {}", i); break; }
                i += 4;
            }
            0xae => {
                if i + 5 >= payload.len() { error!("incomplete exitcode at byte {}", i); break; }
                i += 5;
            }
            0x01 => {
                if i + 1 >= payload.len() { error!("incomplete loopstart at byte {}", i); break; }
                i += 2;
            }
            _ => {
                warn!(
                    "unknown opcode 0x{:02X} in scan at byte {}, skipping. advancing i by 1",
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



    // Initialize qs and registers based on initial_state_data or create new ones
    let (mut qs, mut registers) = if let Some((loaded_qs, loaded_regs)) = initial_state_data {
        info!("loading quantum state and registers from file.");
        (loaded_qs, loaded_regs)
    } else {
        info!(
            "initializing quantum state with {} qubits (type {}, ver {})",
            num_qubits_to_initialize, _header, _version
        );
        (
            QuantumState::new(num_qubits_to_initialize, noise_config.clone()),
            vec![0.0; 24],
        )
    };

    let _last_stats = Instant::now();
    let mut char_count: u64 = 0;
    let mut char_sum: u64 = 0;

    // declare registers, loop_stack, call_stack, memory, and rng for the second pass
    let mut loop_stack: Vec<(usize, u64)> = Vec::new();
    let mut call_stack: Vec<usize> = Vec::new(); // for call/ret instructions
    let mut memory: Vec<u8> = vec![0; 1024 * 1024]; // 1mb linear byte-addressable memory
    let mut rng = StdRng::from_rng(&mut ThreadRng::default());

    let mut i = 0; // reset 'i' for the second pass

    // second pass: execute instructions and interact with quantumstate
    while i < payload.len() {
        if debug_mode {
            debug!("executing opcode 0x{:02X} at byte {}", payload[i], i);
        }
        let opcode = payload[i];
        match opcode {
            0x04 /* qinit / initqubit */ => {
                let q_idx = payload[i + 1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, qinit on qubit {}, advancing i by 2", opcode, i, q_idx); }
                i += 2;
            }
            0x1d /* setphase */ => {
                let q_idx = payload[i+1] as usize;
                let phase_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let phase = f64::from_le_bytes(phase_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, set phase for qubit {} to {}, advancing i by 10", opcode, i, q_idx, phase); }
                i += 10;
            }
            0x74 /* setpos */ => {
                let q_idx = payload[i+1] as usize;
                let x_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let x = f64::from_le_bytes(x_bytes);
                let y_bytes: [u8; 8] = payload[i+10..i+18].try_into().unwrap();
                let y = f64::from_le_bytes(y_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, set position for qubit {} to ({}, {}), advancing i by 18", opcode, i, q_idx, x, y); }
                i += 18;
            }
            0x75 /* setwl */ => {
                let q_idx = payload[i+1] as usize;
                let wl_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let wl = f64::from_le_bytes(wl_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, set wavelength for qubit {} to {}, advancing i by 10", opcode, i, q_idx, wl); }
                i += 10;
            }
            0x76 /* wlshift */ => {
                let q_idx = payload[i+1] as usize;
                let shift_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let shift = f64::from_le_bytes(shift_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, shifted wavelength for qubit {} by {}, advancing i by 10", opcode, i, q_idx, shift); }
                i += 10;
            }
            0x77 /* move */ => {
                let q_idx = payload[i+1] as usize;
                let dx_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let dx = f64::from_le_bytes(dx_bytes);
                let dy_bytes: [u8; 8] = payload[i+10..i+18].try_into().unwrap();
                let dy = f64::from_le_bytes(dy_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, moved qubit {} by ({}, {}), advancing i by 18", opcode, i, q_idx, dx, dy); }
                i += 18;
            }
            0x18 /* charout */ => {
                let q = payload[i+1] as usize;
                // handle result of measure
                let classical_value = match qs.measure(q) {
                    Ok(val) => val as u8,
                    Err(e) => {
                        error!("error measuring qubit {}: {}", q, e);
                        0 // default to 0 on error
                    }
                };
                print!("{}", classical_value as char);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, char_out: qubit {} measured as {} ('{}'), advancing i by 2", opcode, i, q, classical_value, classical_value as char); }
                char_count += 1;
                char_sum += classical_value as u64;
                i += 2;
            }
            0x32 /* qmeas / measure */ => {
                let q = payload[i + 1] as usize;
                let result = qs.measure(q);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, measured qubit {}: {:?}, advancing i by 2", opcode, i, q, result); }
                i += 2;
            }
            0x79 /* markobserved*/ => {
                let q_idx = payload[i+1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, marked qubit {} as observed, advancing i by 2", opcode, i, q_idx); }
                i += 2;
            }
            0x7a /* release */ => {
                let q_idx = payload[i+1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, released qubit {}, advancing i by 2", opcode, i, q_idx); }
                i += 2;
            }
            0xff /* halt */ => {
                info!("halt instruction executed, program terminating.");
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, halt instruction, terminating", opcode, i); }
                break;
            }
            0x00 => { // handle 0x00 as a silent nop
                i += 1;
            }
            0x8d => { // handle 0x8d as a silent nop
                i += 1;
            }
            0x97 /* retsub */ => {
                if let Some(ret_addr) = call_stack.pop() {
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, returned from subroutine to address {}, setting i to {}", opcode, i, ret_addr, ret_addr); }
                    i = ret_addr;
                } else {
                    warn!("retsub without matching call at byte {}. this might indicate an issue in the compiled program or execution flow, or an attempt to return from an empty call stack.", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, retsub on empty stack, advancing i by 1", opcode, i); }
                    i += 1;
                }
            }
            // all other opcodes
            0x01 /* loopstart */ => {
                if i + 1 >= payload.len() { error!("incomplete instruction at byte {}", i); break; }
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, loop start, advancing i by 2", opcode, i); }
                i += 2;
            }
            0x02 /* applygate (qgate) */ => {
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
                            "unsupported qgate '{}' at byte {}, skipping. advancing i by 10",
                            name, i
                        );
                        i += 10;
                    }
                }
            }
            0x05 /* applyhadamard */ => {
                qs.apply_h(payload[i + 1] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied hadamard on qubit {}, advancing i by 2", opcode, i, payload[i + 1]); }
                i += 2;
            }
            0x06 /* applyphaseflip */ => {
                qs.apply_phase_flip(payload[i + 1] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied phase flip (z) on qubit {}, advancing i by 2", opcode, i, payload[i + 1]); }
                i += 2;
            }
            0x07 /* applybitflip */ => {
                qs.apply_x(payload[i + 1] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied bit flip (x) on qubit {}, advancing i by 2", opcode, i, payload[i + 1]); }
                i += 2;
            }
            0x08 /* phaseshift */ => {
                let q = payload[i + 1] as usize;
                let angle_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_phase_shift(q, angle);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied phase shift on qubit {} with angle {}, advancing i by 10", opcode, i, q, angle); }
                i += 10;
            }
            0x0a /* reset / qreset */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, reset qubit {}, advancing i by 2", opcode, i, payload[i + 1]); }
                i += 2;
            }
            0x0b /* swap */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                qs.apply_swap(q1, q2);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, swapped qubits {} and {}, advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x0c /* controlledswap */ => {
                let c = payload[i + 1] as usize;
                let q1 = payload[i + 2] as usize;
                let q2 = payload[i + 3] as usize;
                qs.apply_controlled_swap(c, q1, q2);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied controlled swap with control {} on qubits {} and {}, advancing i by 4", opcode, i, c, q1, q2); }
                i += 4;
            }
            0x0d /* applytgate */ => {
                qs.apply_t_gate(payload[i + 1] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied t-gate on qubit {}, advancing i by 2", opcode, i, payload[i + 1]); }
                i += 2;
            }
            0x0e /* applysgate */ => {
                qs.apply_s_gate(payload[i + 1] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied s-gate on qubit {}, advancing i by 2", opcode, i, payload[i + 1]); }
                i += 2;
            }
            0x0f /* rz */ => {
                let q = payload[i + 1] as usize;
                let angle_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_rz(q, angle);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied rz gate on qubit {} with angle {}, advancing i by 10", opcode, i, q, angle); }
                i += 10;
            }
            0x10 /* loopend */ => {
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
                    i += 1;
                }
            }
            0x11 /* entangle */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, entangled qubits {} and {}, advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x12 /* entanglebell */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, created bell state with qubits {} and {}, advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x13 /* entanglemulti */ => {
                let num_qubits = payload[i + 1] as usize;
                let qubits: Vec<usize> = payload[i + 2..i + 2 + num_qubits]
                    .iter()
                    .map(|&b| b as usize)
                    .collect();
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, entangled multiple qubits: {:?}, advancing i by {}", opcode, i, qubits, 2 + num_qubits); }
                i += 2 + num_qubits;
            }
            0x14 /* entanglecluster */ => {
                let num_qubits = payload[i + 1] as usize;
                let qubits: Vec<usize> = payload[i + 2..i + 2 + num_qubits]
                    .iter()
                    .map(|&b| b as usize)
                    .collect();
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, created cluster state with qubits: {:?}, advancing i by {}", opcode, i, qubits, 2 + num_qubits); }
                i += 2 + num_qubits;
            }
            0x15 /* entangleswap */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                let q3 = payload[i + 3] as usize;
                let q4 = payload[i + 4] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, performed entanglement swap between ({}, {}) and ({}, {}), advancing i by 5", opcode, i, q1, q2, q3, q4); }
                i += 5;
            }
            0x16 /* entangleswapmeasure */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                let q3 = payload[i + 3] as usize;
                let q4 = payload[i + 4] as usize;
                let label_start = i + 5;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let _bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, performed entanglement swap measure between ({}, {}) and ({}, {}) with label {}, advancing i by {}", opcode, i, q1, q2, q3, q4, label, _bytes_advanced); }
                i = label_end + 1;
            }
            0x17 /* controllednot / cnot */ => {
                qs.apply_cnot(payload[i + 1] as usize, payload[i + 2] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied cnot with control {} and target {}, advancing i by 3", opcode, i, payload[i + 1], payload[i + 2]); }
                i += 3;
            }
            0x19 /* entanglewithclassicalfeedback */ => {
                let q = payload[i + 1] as usize;
                let q_target = payload[i + 2] as usize;
                let label_start = i + 3;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let _bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, entangled with classical feedback on qubits {} and {} with label {}, advancing i by {}", opcode, i, q, q_target, label, _bytes_advanced); }
                i = label_end + 1;
            }
            0x1a /* entangledistributed */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let _bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, performed distributed entanglement on qubit {} with label {}, advancing i by {}", opcode, i, q, label, _bytes_advanced); }
                i = label_end + 1;
            }
            0x1b /* measureinbasis */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let basis = String::from_utf8_lossy(&payload[label_start..label_end]);
                let _bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, measured qubit {} in basis {}, advancing i by {}", opcode, i, q, basis, _bytes_advanced); }
                i = label_end + 1;
            }
            0x1c /* resetall */ => {
                qs.apply_reset_all();
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, reset all qubits, advancing i by 1", opcode, i); }
                i += 1;
            }
            0x1e /* cz */ => {
                qs.apply_cz(payload[i + 1] as usize, payload[i + 2] as usize);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied cz gate on qubits {} and {}, advancing i by 3", opcode, i, payload[i + 1], payload[i + 2]); }
                i += 3;
            }
            0x1f /* thermalavg */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, performed thermal averaging on qubits {} and {} (placeholder), advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x20 /* wkbfactor */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, wkbfactor instruction (placeholder), advancing i by 3", opcode, i); }
                i += 3;
            }
            0x21 /* regset */ => {
                let reg_idx = payload[i+1] as usize;
                let value_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let value = f64::from_le_bytes(value_bytes);
                if reg_idx < registers.len() {
                    registers[reg_idx] = value;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, set register {} to {}, advancing i by 10", opcode, i, reg_idx, value); }
                } else {
                    error!("invalid register index {} for regset at byte {}", reg_idx, i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for regset, advancing i by 10", opcode, i); }
                }
                i += 10;
            }
            0x22 /* rx */ => {
                let q = payload[i + 1] as usize;
                let angle_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_rx(q, angle);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied rx gate on qubit {} with angle {}, advancing i by 10", opcode, i, q, angle); }
                i += 10;
            }
            0x23 /* ry */ => {
                let q = payload[i + 1] as usize;
                let angle_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_ry(q, angle);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied ry gate on qubit {} with angle {}, advancing i by 10", opcode, i, q, angle); }
                i += 10;
            }
            0x24 /* phase */ => {
                let q = payload[i+1] as usize;
                let angle_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied phase gate on qubit {} with angle {}, advancing i by 10", opcode, i, q, angle); }
                i += 10;
            }
            0x31 /* charload */ => {
                let dest_reg = payload[i+1] as usize;
                let char_val = payload[i+2] as char;
                if dest_reg < registers.len() {
                    registers[dest_reg] = char_val as u8 as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, loaded char '{}' into register {}, advancing i by 4", opcode, i, char_val, dest_reg); }
                } else {
                    error!("invalid register index {} for charload at byte {}", dest_reg, i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for charload, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0x33 /* applyrotation */ => {
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
            0x34 /* applymultiqubitrotation */ => {
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
                let _bytes_advanced = current_idx + 8 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied multi-qubit rotation around {} axis on qubits {:?}, with angle {}, advancing i by {}", opcode, i, axis, qubits, angle, _bytes_advanced); }
                i = current_idx + 8;
            }
            0x35 /* controlledphaserotation */ => {
                let c = payload[i + 1] as usize;
                let t = payload[i + 2] as usize;
                let angle_bytes: [u8; 8] = payload[i + 3..i + 11].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                qs.apply_controlled_phase(c, t, angle);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied controlled phase rotation on control {} and target {} with angle {}, advancing i by 11", opcode, i, c, t, angle); }
                i += 11;
            }
            0x36 /* applycphase */ => {
                let c = payload[i + 1] as usize;
                let t = payload[i + 2] as usize;
                let angle_bytes: [u8; 8] = payload[i + 3..i + 11].try_into().unwrap();
                let angle = f64::from_le_bytes(angle_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied controlled phase gate on control {} and target {} with angle {}, advancing i by 11", opcode, i, c, t, angle); }
                i += 11;
            }
            0x37 /* applykerrnonlinearity */ => {
                let q = payload[i + 1] as usize;
                let strength_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let strength = f64::from_le_bytes(strength_bytes);
                let duration_bytes: [u8; 8] = payload[i + 10..i + 18].try_into().unwrap();
                let duration = f64::from_le_bytes(duration_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied kerr nonlinearity on qubit {} with strength {} and duration {}, advancing i by 18", opcode, i, q, strength, duration); }
                i += 18;
            }
            0x38 /* applyfeedforwardgate */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let _bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied feedforward gate on qubit {} with label {}, advancing i by {}", opcode, i, q, label, _bytes_advanced); }
                i = label_end + 1;
            }
            0x39 /* decoherenceprotect */ => {
                let q = payload[i + 1] as usize;
                let duration_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let duration = f64::from_le_bytes(duration_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied decoherence protection on qubit {} for duration {}, advancing i by 10", opcode, i, q, duration); }
                i += 10;
            }
            0x3a /* applymeasurementbasischange */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let _bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied measurement basis change on qubit {} to basis {}, advancing i by {}", opcode, i, q, label, _bytes_advanced); }
                i = label_end + 1;
            }
            0x3b /* load */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let _bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, loaded state to qubit {} from label {}, advancing i by {}", opcode, i, q, label, _bytes_advanced); }
                i = label_end + 1;
            }
            0x3c /* store */ => {
                let q = payload[i + 1] as usize;
                let label_start = i + 2;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                let _bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, stored state from qubit {} to label {}, advancing i by {}", opcode, i, q, label, _bytes_advanced); }
                i = label_end + 1;
            }
            0x3d /* loadmem */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);

                let addr_name_start = reg_name_end + 1;
                let addr_name_end = addr_name_start + payload[addr_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let addr_name = String::from_utf8_lossy(&payload[addr_name_start..addr_name_end]);
                let _bytes_advanced = addr_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, loaded value from memory address {} into register {}, advancing i by {}", opcode, i, addr_name, reg_name, _bytes_advanced); }
                i = addr_name_end + 1;
            }
            0x3e /* storemem */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let addr_name_start = reg_name_end + 1;
                let addr_name_end = addr_name_start + payload[addr_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let addr_name = String::from_utf8_lossy(&payload[addr_name_start..addr_name_end]);
                let _bytes_advanced = addr_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, stored value from register {} into memory address {}, advancing i by {}", opcode, i, reg_name, addr_name, _bytes_advanced); }
                i = addr_name_end + 1;
            }
            0x3f /* loadclassical */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let var_name_start = reg_name_end + 1;
                let var_name_end = var_name_start + payload[var_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let var_name = String::from_utf8_lossy(&payload[var_name_start..var_name_end]);
                let _bytes_advanced = var_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, loaded classical variable {} into register {}, advancing i by {}", opcode, i, var_name, reg_name, _bytes_advanced); }
                i = var_name_end + 1;
            }
            0x40 /* storeclassical */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let var_name_start = reg_name_end + 1;
                let var_name_end = var_name_start + payload[var_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let var_name = String::from_utf8_lossy(&payload[var_name_start..var_name_end]);
                let _bytes_advanced = var_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, stored register {} into classical variable {}, advancing i by {}", opcode, i, reg_name, var_name, _bytes_advanced); }
                i = var_name_end + 1;
            }
            0x41 /* add */ => {
                let dst_start = i + 1;
                let dst_end = dst_start + payload[dst_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let dst_name = String::from_utf8_lossy(&payload[dst_start..dst_end]);

                let src1_start = dst_end + 1;
                let src1_end = src1_start + payload[src1_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src1_name = String::from_utf8_lossy(&payload[src1_start..src1_end]);

                let src2_start = src1_end + 1;
                let src2_end = src2_start + payload[src2_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src2_name = String::from_utf8_lossy(&payload[src2_start..src2_end]);
                let _bytes_advanced = src2_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, add: {} = {} + {}, advancing i by {}", opcode, i, dst_name, src1_name, src2_name, _bytes_advanced); }
                i = src2_end + 1;
            }
            0x42 /* sub */ => {
                let dst_start = i + 1;
                let dst_end = dst_start + payload[dst_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let dst_name = String::from_utf8_lossy(&payload[dst_start..dst_end]);

                let src1_start = dst_end + 1;
                let src1_end = src1_start + payload[src1_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src1_name = String::from_utf8_lossy(&payload[src1_start..src1_end]);

                let src2_start = src1_end + 1;
                let src2_end = src2_start + payload[src2_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src2_name = String::from_utf8_lossy(&payload[src2_start..src2_end]);
                let _bytes_advanced = src2_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, sub: {} = {} - {}, advancing i by {}", opcode, i, dst_name, src1_name, src2_name, _bytes_advanced); }
                i = src2_end + 1;
            }
            0x43 /* and */ => {
                let dst_start = i + 1;
                let dst_end = dst_start + payload[dst_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let dst_name = String::from_utf8_lossy(&payload[dst_start..dst_end]);

                let src1_start = dst_end + 1;
                let src1_end = src1_start + payload[src1_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src1_name = String::from_utf8_lossy(&payload[src1_start..src1_end]);

                let src2_start = src1_end + 1;
                let src2_end = src2_start + payload[src2_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src2_name = String::from_utf8_lossy(&payload[src2_start..src2_end]);
                let _bytes_advanced = src2_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, and: {} = {} & {}, advancing i by {}", opcode, i, dst_name, src1_name, src2_name, _bytes_advanced); }
                i = src2_end + 1;
            }
            0x44 /* or */ => {
                let dst_start = i + 1;
                let dst_end = dst_start + payload[dst_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let dst_name = String::from_utf8_lossy(&payload[dst_start..dst_end]);

                let src1_start = dst_end + 1;
                let src1_end = src1_start + payload[src1_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src1_name = String::from_utf8_lossy(&payload[src1_start..src1_end]);

                let src2_start = src1_end + 1;
                let src2_end = src2_start + payload[src2_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src2_name = String::from_utf8_lossy(&payload[src2_start..src2_end]);
                let _bytes_advanced = src2_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, or: {} = {} | {}, advancing i by {}", opcode, i, dst_name, src1_name, src2_name, _bytes_advanced); }
                i = src2_end + 1;
            }
            0x45 /* xor */ => {
                let dst_start = i + 1;
                let dst_end = dst_start + payload[dst_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let dst_name = String::from_utf8_lossy(&payload[dst_start..dst_end]);

                let src1_start = dst_end + 1;
                let src1_end = src1_start + payload[src1_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src1_name = String::from_utf8_lossy(&payload[src1_start..src1_end]);

                let src2_start = src1_end + 1;
                let src2_end = src2_start + payload[src2_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let src2_name = String::from_utf8_lossy(&payload[src2_start..src2_end]);
                let _bytes_advanced = src2_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, xor: {} = {} ^ {}, advancing i by {}", opcode, i, dst_name, src1_name, src2_name, _bytes_advanced); }
                i = src2_end + 1;
            }
            0x46 /* not */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let _bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, not: ~{}, advancing i by {}", opcode, i, reg_name, _bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x47 /* push */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let _bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, push {}, advancing i by {}", opcode, i, reg_name, _bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x49 /* jump */ => {
                let label_start = i + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                warn!("jump to label '{}' (opcode 0x49) not implemented. use jmpabs (0x91) or jmp (0x90) with an address.", label);
                let _bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, jump to label {}, advancing i by {}", opcode, i, label, _bytes_advanced); }
                i = label_end + 1;
            }
            0x4a /* jumpifzero */ => {
                let cond_start = i + 1;
                let cond_end = cond_start + payload[cond_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let cond = String::from_utf8_lossy(&payload[cond_start..cond_end]);

                let label_start = cond_end + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                warn!("jumpifzero on cond '{}' to label '{}' (opcode 0x4a) not implemented. use ifeq (0x94) with a register and offset.", cond, label);
                let _bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, jumpifzero on {} to label {}, advancing i by {}", opcode, i, cond, label, _bytes_advanced); }
                i = label_end + 1;
            }
            0x4b /* jumpifone */ => {
                let cond_start = i + 1;
                let cond_end = cond_start + payload[cond_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let cond = String::from_utf8_lossy(&payload[cond_start..cond_end]);

                let label_start = cond_end + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                warn!("jumpifone on cond '{}' to label '{}' (opcode 0x4b) not implemented. use ifne (0x95) with a register and offset.", cond, label);
                let _bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, jumpifone on {} to label {}, advancing i by {}", opcode, i, cond, label, _bytes_advanced); }
                i = label_end + 1;
            }
            0x4c /* call */ => {
                let label_start = i + 1;
                let label_end = label_start + payload[label_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let label = String::from_utf8_lossy(&payload[label_start..label_end]);
                warn!("call to label '{}' (opcode 0x4c) not implemented. use calladdr (0x96) with an address.", label);
                let _bytes_advanced = label_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, call to label {}, advancing i by {}", opcode, i, label, _bytes_advanced); }
                i = label_end + 1;
            }
            0x4d /* return */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, returned from subroutine, advancing i by 1", opcode, i); }
                i += 1;
            }
            0x4e /* timedelay */ => {
                let q = payload[i + 1] as usize;
                let delay_bytes: [u8; 8] = payload[i + 2..i + 10].try_into().unwrap();
                let delay = f64::from_le_bytes(delay_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, time delay on qubit {} for {} units, advancing i by 10", opcode, i, q, delay); }
                i += 10;
            }
            0x4f /* pop */ => {
                let reg_name_start = i + 1;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let _bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, popped into register {}, advancing i by {}", opcode, i, reg_name, _bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x50 /* rand */ => {
                let dest_reg_idx = payload[i+1] as usize;
                if dest_reg_idx < registers.len() {
                    registers[dest_reg_idx] = rng.random::<f64>();
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, generated random number {} into register {}, advancing i by 2", opcode, i, registers[dest_reg_idx], dest_reg_idx); }
                } else {
                    error!("invalid register index {} for rand at byte {}", dest_reg_idx, i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for rand, advancing i by 2", opcode, i); }
                }
                i += 2;
            }
            0x51 /* sqrt */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, sqrt on qubits {} and {} (placeholder), advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x52 /* exp */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, exp on qubits {} and {} (placeholder), advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x53 /* log */ => {
                let q1 = payload[i + 1] as usize;
                let q2 = payload[i + 2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, log on qubits {} and {} (placeholder), advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x54 /* regadd */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = registers[src1_reg] + registers[src2_reg];
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, reg_add: r{} = r{} + r{} ({}), advancing i by 4", opcode, i, dest_reg, src1_reg, src2_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for regadd at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for regadd, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0x55 /* regsub */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = registers[src1_reg] - registers[src2_reg];
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, reg_sub: r{} = r{} - r{} ({}), advancing i by 4", opcode, i, dest_reg, src1_reg, src2_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for regsub at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for regsub, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0x56 /* regmul */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    registers[dest_reg] = registers[src1_reg] * registers[src2_reg];
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, reg_mul: r{} = r{} * r{} ({}), advancing i by 4", opcode, i, dest_reg, src1_reg, src2_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for regmul at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for regmul, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0x57 /* regdiv */ => {
                let dest_reg = payload[i+1] as usize;
                let src1_reg = payload[i+2] as usize;
                let src2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && src1_reg < registers.len() && src2_reg < registers.len() {
                    if registers[src2_reg] != 0.0 {
                        registers[dest_reg] = registers[src1_reg] / registers[src2_reg];
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, reg_div: r{} = r{} / r{} ({}), advancing i by 4", opcode, i, dest_reg, src1_reg, src2_reg, registers[dest_reg]); }
                    } else {
                        error!("division by zero in regdiv at byte {}", i);
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, division by zero for regdiv, advancing i by 4", opcode, i); }
                    }
                } else {
                    error!("invalid register index for regdiv at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for regdiv, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0x58 /* regcopy */ => {
                let dest_reg = payload[i+1] as usize;
                let src_reg = payload[i+2] as usize;
                if dest_reg < registers.len() && src_reg < registers.len() {
                    registers[dest_reg] = registers[src_reg];
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, reg_copy: r{} = r{} ({}), advancing i by 3", opcode, i, dest_reg, src_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for regcopy at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for regcopy, advancing i by 3", opcode, i); }
                }
                i += 3;
            }
            0x59 /* photonemit */ => {
                let q = payload[i+1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, photon emitted from qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x5a /* photondetect */ => {
                let q = payload[i+1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, photon detected at qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x5b /* photoncount */ => {
                let q = payload[i+1] as usize;
                let reg_name_start = i + 2;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let _bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, photon count on qubit {} into register {}, advancing i by {}", opcode, i, q, reg_name, _bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x5c /* photonaddition */ => {
                let q = payload[i+1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, photon added to qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x5d /* applyphotonsubtraction */ => {
                let q = payload[i+1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, photon subtracted from qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x5e /* photonemissionpattern */ => {
                let q = payload[i+1] as usize;
                let pattern_start = i + 2;
                let pattern_end = pattern_start + payload[pattern_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let pattern = String::from_utf8_lossy(&payload[pattern_start..pattern_end]);
                let cycles_bytes: [u8; 8] = payload[pattern_end + 1..pattern_end + 9].try_into().unwrap();
                let cycles = u64::from_le_bytes(cycles_bytes);
                let _bytes_advanced = pattern_end + 9 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, photon emission pattern on qubit {} with pattern '{}' for {} cycles, advancing i by {}", opcode, i, q, pattern, cycles, _bytes_advanced); }
                i = pattern_end + 9;
            }
            0x5f /* photondetectthreshold */ => {
                let q = payload[i+1] as usize;
                let thresh_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let thresh = u64::from_le_bytes(thresh_bytes);
                let reg_name_start = i + 10;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let _bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, photon detection with threshold on qubit {} with threshold {} into register {}, advancing i by {}", opcode, i, q, thresh, reg_name, _bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x60 /* photondetectcoincidence */ => {
                let num_qubits = payload[i+1] as usize;
                let mut qubits = Vec::with_capacity(num_qubits);
                let mut current_idx = i + 2;
                for _ in 0..num_qubits {
                    qubits.push(payload[current_idx] as usize);
                    current_idx += 1;
                }
                let reg_name_start = current_idx;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let _bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, photon detection coincidence on qubits {:?} into register {}, advancing i by {}", opcode, i, qubits, reg_name, _bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x61 /* singlephotonsourceon */ => {
                let q = payload[i+1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, single photon source on for qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x62 /* singlephotonsourceoff */ => {
                let q = payload[i+1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, single photon source off for qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x63 /* photonbunchingcontrol */ => {
                let q = payload[i+1] as usize;
                let b = payload[i+2] != 0;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, photon bunching control on qubit {} to {}, advancing i by 3", opcode, i, q, b); }
                i += 3;
            }
            0x64 /* photonroute */ => {
                let q = payload[i+1] as usize;
                let from_start = i + 2;
                let from_end = from_start + payload[from_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let from = String::from_utf8_lossy(&payload[from_start..from_end]);
                let to_start = from_end + 1;
                let to_end = to_start + payload[to_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let to = String::from_utf8_lossy(&payload[to_start..to_end]);
                let _bytes_advanced = to_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, photon route for qubit {} from {} to {}, advancing i by {}", opcode, i, q, from, to, _bytes_advanced); }
                i = to_end + 1;
            }
            0x65 /* opticalrouting */ => {
                let q1 = payload[i+1] as usize;
                let q2 = payload[i+2] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, optical routing between qubits {} and {}, advancing i by 3", opcode, i, q1, q2); }
                i += 3;
            }
            0x66 /* setopticalattenuation */ => {
                let q = payload[i+1] as usize;
                let att_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let att = f64::from_le_bytes(att_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, set optical attenuation for qubit {} to {}, advancing i by 10", opcode, i, q, att); }
                i += 10;
            }
            0x67 /* dynamicphasecompensation */ => {
                let q = payload[i+1] as usize;
                let phase_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let phase = f64::from_le_bytes(phase_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, dynamic phase compensation for qubit {} with phase {}, advancing i by 10", opcode, i, q, phase); }
                i += 10;
            }
            0x68 /* opticaldelaylinecontrol */ => {
                let q = payload[i+1] as usize;
                let delay_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let delay = u64::from_le_bytes(delay_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, optical delay line control for qubit {} with delay {}, advancing i by 10", opcode, i, q, delay); }
                i += 10;
            }
            0x69 /* crossphasemodulation */ => {
                let c = payload[i+1] as usize;
                let t = payload[i+2] as usize;
                let stren_bytes: [u8; 8] = payload[i+3..i+11].try_into().unwrap();
                let stren = f64::from_le_bytes(stren_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, cross phase modulation between qubits {} and {} with strength {}, advancing i by 11", opcode, i, c, t, stren); }
                i += 11;
            }
            0x6a /* applydisplacement */ => {
                let q = payload[i+1] as usize;
                let alpha_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let alpha = f64::from_le_bytes(alpha_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied displacement to qubit {} with alpha {}, advancing i by 10", opcode, i, q, alpha); }
                i += 10;
            }
            0x6b /* applydisplacementfeedback */ => {
                let q = payload[i+1] as usize;
                let reg_name_start = i + 2;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let _bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied displacement feedback on qubit {} with register {}, advancing i by {}", opcode, i, q, reg_name, _bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x6c /* applydisplacementoperator */ => {
                let q = payload[i+1] as usize;
                let alpha_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let alpha = f64::from_le_bytes(alpha_bytes);
                let duration_bytes: [u8; 8] = payload[i+10..i+18].try_into().unwrap();
                let duration = u64::from_le_bytes(duration_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied displacement operator to qubit {} with alpha {} and duration {}, advancing i by 18", opcode, i, q, alpha, duration); }
                i += 18;
            }
            0x6d /* applysqueezing */ => {
                let q = payload[i+1] as usize;
                let squeeze_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let squeeze = f64::from_le_bytes(squeeze_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied squeezing to qubit {} with squeeze {}, advancing i by 10", opcode, i, q, squeeze); }
                i += 10;
            }
            0x6e /* applysqueezingfeedback */ => {
                let q = payload[i+1] as usize;
                let reg_name_start = i + 2;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let _bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, applied squeezing feedback on qubit {} with register {}, advancing i by {}", opcode, i, q, reg_name, _bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x6f /* measureparity */ => {
                let q = payload[i+1] as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, measured parity on qubit {}, advancing i by 2", opcode, i, q); }
                i += 2;
            }
            0x70 /* measurewithdelay */ => {
                let q = payload[i+1] as usize;
                let delay_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let delay = u64::from_le_bytes(delay_bytes);
                let reg_name_start = i + 10;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let _bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, measured qubit {} with delay {} into register {}, advancing i by {}", opcode, i, q, delay, reg_name, _bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x71 /* opticalswitchcontrol */ => {
                let q = payload[i+1] as usize;
                let state = payload[i+2] != 0;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, optical switch control on qubit {} to state {}, advancing i by 3", opcode, i, q, state); }
                i += 3;
            }
            0x72 /* photonlosssimulate */ => {
                let q = payload[i+1] as usize;
                let prob_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let prob = f64::from_le_bytes(prob_bytes);
                let seed_bytes: [u8; 8] = payload[i+10..i+18].try_into().unwrap();
                let seed = u64::from_le_bytes(seed_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, photon loss simulation on qubit {} with probability {} and seed {}, advancing i by 18", opcode, i, q, prob, seed); }
                i += 18;
            }
            0x73 /* photonlosscorrection */ => {
                let q = payload[i+1] as usize;
                let reg_name_start = i + 2;
                let reg_name_end = reg_name_start + payload[reg_name_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let reg_name = String::from_utf8_lossy(&payload[reg_name_start..reg_name_end]);
                let _bytes_advanced = reg_name_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, photon loss correction on qubit {} with register {}, advancing i by {}", opcode, i, q, reg_name, _bytes_advanced); }
                i = reg_name_end + 1;
            }
            0x7e /* errorsyndrome */ => {
                let q = payload[i+1] as usize;
                let syndrome_type_start = i + 2;
                let syndrome_type_end = syndrome_type_start + payload[syndrome_type_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let syndrome_type = String::from_utf8_lossy(&payload[syndrome_type_start..syndrome_type_end]);
                let result_reg_start = syndrome_type_end + 1;
                let result_reg_end = result_reg_start + payload[result_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let result_reg = String::from_utf8_lossy(&payload[result_reg_start..result_reg_end]);
                let _bytes_advanced = result_reg_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, error syndrome on qubit {} with type '{}' into register {}, advancing i by {}", opcode, i, q, syndrome_type, result_reg, _bytes_advanced); }
                i = result_reg_end + 1;
            }
            0x80 /* bellstateverif */ => {
                let q1 = payload[i+1] as usize;
                let q2 = payload[i+2] as usize;
                let result_reg_start = i + 3;
                let result_reg_end = result_reg_start + payload[result_reg_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let result_reg = String::from_utf8_lossy(&payload[result_reg_start..result_reg_end]);
                let _bytes_advanced = result_reg_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, bell state verification on qubits {} and {} into register {}, advancing i by {}", opcode, i, q1, q2, result_reg, _bytes_advanced); }
                i = result_reg_end + 1;
            }
            0x81 /* quantumzenoeffect */ => {
                let q = payload[i+1] as usize;
                let num_measurements_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let num_measurements = u64::from_le_bytes(num_measurements_bytes);
                let interval_cycles_bytes: [u8; 8] = payload[i+10..i+18].try_into().unwrap();
                let interval_cycles = u64::from_le_bytes(interval_cycles_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, quantum zeno effect on qubit {} with {} measurements at {} cycles interval, advancing i by 18", opcode, i, q, num_measurements, interval_cycles); }
                i += 18;
            }
            0x88 /* comment */ => {
                let comment_start = i + 1;
                let comment_end = comment_start + payload[comment_start..].iter().position(|&b| b == 0).unwrap_or(0);
                let comment = String::from_utf8_lossy(&payload[comment_start..comment_end]);
                let _bytes_advanced = comment_end + 1 - i;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, comment: \"{}\", advancing i by {}", opcode, i, comment, _bytes_advanced); }
                i = comment_end + 1;
            }
            0x90 /* jmp */ => {
                let offset_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let offset = i64::from_le_bytes(offset_bytes);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, jmp by offset {}, current i {}, new i {}", opcode, i, offset, i, (i as i64 + offset) as usize); }
                i = (i as i64 + offset) as usize;
            }
            0x91 /* jmpabs */ => {
                let addr_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, jmpabs to address {}, setting i to {}", opcode, i, addr, addr); }
                i = addr;
            }
            0x92 /* ifgt */ => {
                let r1_idx = payload[i+1] as usize;
                let r2_idx = payload[i+2] as usize;
                let offset_bytes: [u8; 8] = payload[i+3..i+11].try_into().unwrap();
                let offset = i64::from_le_bytes(offset_bytes);
                if r1_idx < registers.len() && r2_idx < registers.len() {
                    if registers[r1_idx] > registers[r2_idx] {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, ifgt: r{} ({}) > r{} ({}), jumping by offset {}", opcode, i, r1_idx, registers[r1_idx], r2_idx, registers[r2_idx], offset); }
                        i = (i as i64 + offset) as usize;
                    } else {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, ifgt: r{} ({}) not > r{} ({}), not jumping, advancing i by 11", opcode, i, r1_idx, registers[r1_idx], r2_idx, registers[r2_idx]); }
                        i += 11;
                    }
                } else {
                    error!("invalid register index for ifgt at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for ifgt, advancing i by 11", opcode, i); }
                    i += 11;
                }
            }
            0x93 /* iflt */ => {
                let r1_idx = payload[i+1] as usize;
                let r2_idx = payload[i+2] as usize;
                let offset_bytes: [u8; 8] = payload[i+3..i+11].try_into().unwrap();
                let offset = i64::from_le_bytes(offset_bytes);
                if r1_idx < registers.len() && r2_idx < registers.len() {
                    if registers[r1_idx] < registers[r2_idx] {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, iflt: r{} ({}) < r{} ({}), jumping by offset {}", opcode, i, r1_idx, registers[r1_idx], r2_idx, registers[r2_idx], offset); }
                        i = (i as i64 + offset) as usize;
                    } else {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, iflt: r{} ({}) not < r{} ({}), not jumping, advancing i by 11", opcode, i, r1_idx, registers[r1_idx], r2_idx, registers[r2_idx]); }
                        i += 11;
                    }
                } else {
                    error!("invalid register index for iflt at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for iflt, advancing i by 11", opcode, i); }
                    i += 11;
                }
            }
            0x94 /* ifeq */ => {
                let r1_idx = payload[i+1] as usize;
                let r2_idx = payload[i+2] as usize;
                let offset_bytes: [u8; 8] = payload[i+3..i+11].try_into().unwrap();
                let offset = i64::from_le_bytes(offset_bytes);
                if r1_idx < registers.len() && r2_idx < registers.len() {
                    if (registers[r1_idx] - registers[r2_idx]).abs() < f64::EPSILON {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, ifeq: r{} ({}) == r{} ({}), jumping by offset {}", opcode, i, r1_idx, registers[r1_idx], r2_idx, registers[r2_idx], offset); }
                        i = (i as i64 + offset) as usize;
                    } else {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, ifeq: r{} ({}) not == r{} ({}), not jumping, advancing i by 11", opcode, i, r1_idx, registers[r1_idx], r2_idx, registers[r2_idx]); }
                        i += 11;
                    }
                } else {
                    error!("invalid register index for ifeq at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for ifeq, advancing i by 11", opcode, i); }
                    i += 11;
                }
            }
            0x95 /* ifne */ => {
                let r1_idx = payload[i+1] as usize;
                let r2_idx = payload[i+2] as usize;
                let offset_bytes: [u8; 8] = payload[i+3..i+11].try_into().unwrap();
                let offset = i64::from_le_bytes(offset_bytes);
                if r1_idx < registers.len() && r2_idx < registers.len() {
                    if (registers[r1_idx] - registers[r2_idx]).abs() >= f64::EPSILON {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, ifne: r{} ({}) != r{} ({}), jumping by offset {}", opcode, i, r1_idx, registers[r1_idx], r2_idx, registers[r2_idx], offset); }
                        i = (i as i64 + offset) as usize;
                    } else {
                        if debug_mode { debug!("opcode 0x{:02X} at byte {}, ifne: r{} ({}) not != r{} ({}), not jumping, advancing i by 11", opcode, i, r1_idx, registers[r1_idx], r2_idx, registers[r2_idx]); }
                        i += 11;
                    }
                } else {
                    error!("invalid register index for ifne at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for ifne, advancing i by 11", opcode, i); }
                    i += 11;
                }
            }
            0x96 /* calladdr */ => {
                let addr_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                call_stack.push(i + 9);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, calladdr to address {}, pushing return address {} to stack, setting i to {}", opcode, i, addr, i + 9, addr); }
                i = addr;
            }
            0x98 /* printf */ => {
                let str_len_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let str_len = u64::from_le_bytes(str_len_bytes) as usize;
                let format_str_bytes = &payload[i+9..i+9+str_len];
                let format_str = String::from_utf8_lossy(format_str_bytes);
                let num_regs = payload[i+9+str_len] as usize;
                let mut reg_values = Vec::with_capacity(num_regs);
                for k in 0..num_regs {
                    let reg_idx = payload[i+9+str_len+1+k] as usize;
                    if reg_idx < registers.len() {
                        reg_values.push(registers[reg_idx]);
                    } else {
                        error!("invalid register index {} for printf at byte {}", reg_idx, i);
                        reg_values.push(f64::NAN);
                    }
                }
                // basic printf implementation (can be expanded)
                let mut output = format_str.to_string();
                for (idx, &val) in reg_values.iter().enumerate() {
                    output = output.replace(&format!("%{}", idx), &format!("{}", val));
                }
                print!("{}", output);
                let _bytes_advanced = 1 + 8 + str_len + 1 + num_regs;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, printf: \"{}\", advancing i by {}", opcode, i, output, _bytes_advanced); }
                i += _bytes_advanced;
            }
            0x99 /* print */ => {
                let str_len_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let str_len = u64::from_le_bytes(str_len_bytes) as usize;
                let text_bytes = &payload[i+9..i+9+str_len];
                let text = String::from_utf8_lossy(text_bytes);
                print!("{}", text);
                let _bytes_advanced = 1 + 8 + str_len;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, print: \"{}\", advancing i by {}", opcode, i, text, _bytes_advanced); }
                i += _bytes_advanced;
            }
            0x9a /* println */ => {
                let str_len_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let str_len = u64::from_le_bytes(str_len_bytes) as usize;
                let text_bytes = &payload[i+9..i+9+str_len];
                let text = String::from_utf8_lossy(text_bytes);
                println!("{}", text);
                let _bytes_advanced = 1 + 8 + str_len;
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, println: \"{}\", advancing i by {}", opcode, i, text, _bytes_advanced); }
                i += _bytes_advanced;
            }
            0x9b /* input */ => {
                let dest_reg_idx = payload[i+1] as usize;
                let mut input_line = String::new();
                info!("input requested for register {}. enter a floating-point value:", dest_reg_idx);
                io::stdin().read_line(&mut input_line).expect("failed to read line");
                let value = input_line.trim().parse::<f64>().unwrap_or(0.0);
                if dest_reg_idx < registers.len() {
                    registers[dest_reg_idx] = value;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, input: read {} into register {}, advancing i by 2", opcode, i, value, dest_reg_idx); }
                } else {
                    error!("invalid register index {} for input at byte {}", dest_reg_idx, i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for input, advancing i by 2", opcode, i); }
                }
                i += 2;
            }
            0x9c /* dumpstate */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, dump state (full), advancing i by 1", opcode, i); }
                print_amplitudes(&qs, &registers, 0.0, 0);
                i += 1;
            }
            0x9d /* dumpregs */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, dump registers (full), advancing i by 1", opcode, i); }
                println!("\nclassical registers:");
                for (idx, val) in registers.iter().enumerate() {
                    println!("r{}: {:.6}", idx, val);
                }
                i += 1;
            }
            0x9e /* loadregmem */ => {
                let reg_idx = payload[i+1] as usize;
                let addr_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                if reg_idx < registers.len() && addr + 8 <= memory.len() {
                    let mut val_bytes = [0u8; 8];
                    val_bytes.copy_from_slice(&memory[addr..addr+8]);
                    registers[reg_idx] = f64::from_le_bytes(val_bytes);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, loaded 0x{:X} from memory address {} into register {} ({}), advancing i by 10", opcode, i, u64::from_le_bytes(val_bytes), addr, reg_idx, registers[reg_idx]); }
                } else {
                    error!("invalid register index or memory address for loadregmem at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register or memory address for loadregmem, advancing i by 10", opcode, i); }
                }
                i += 10;
            }
            0x9f /* storememreg */ => {
                let addr_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                let reg_idx = payload[i+9] as usize;
                if reg_idx < registers.len() && addr + 8 <= memory.len() {
                    memory[addr..addr+8].copy_from_slice(&registers[reg_idx].to_le_bytes());
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, stored {} (r{}) into memory address {}, advancing i by 10", opcode, i, registers[reg_idx], reg_idx, addr); }
                } else {
                    error!("invalid register index or memory address for storememreg at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register or memory address for storememreg, advancing i by 10", opcode, i); }
                }
                i += 10;
            }
            0xa0 /* pushreg */ => {
                let reg_idx = payload[i+1] as usize;
                if reg_idx < registers.len() {
                    // in a real vm, this would push to a separate stack.
                    // for now, just log the operation.
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, pushed register {} ({}), advancing i by 2", opcode, i, reg_idx, registers[reg_idx]); }
                } else {
                    error!("invalid register index {} for pushreg at byte {}", reg_idx, i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for pushreg, advancing i by 2", opcode, i); }
                }
                i += 2;
            }
            0xa1 /* popreg */ => {
                let reg_idx = payload[i+1] as usize;
                if reg_idx < registers.len() {
                    // in a real vm, this would pop from a separate stack.
                    // for now, just log the operation.
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, popped into register {}, advancing i by 2", opcode, i, reg_idx); }
                } else {
                    error!("invalid register index {} for popreg at byte {}", reg_idx, i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for popreg, advancing i by 2", opcode, i); }
                }
                i += 2;
            }
            0xa2 /* alloc */ => {
                let reg_idx = payload[i+1] as usize;
                let size_bytes: [u8; 8] = payload[i+2..i+10].try_into().unwrap();
                let size = u64::from_le_bytes(size_bytes) as usize;
                // in a real vm, this would allocate memory and store the address in reg_idx
                if reg_idx < registers.len() {
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, allocated {} bytes, address stored in r{}, advancing i by 10", opcode, i, size, reg_idx); }
                } else {
                    error!("invalid register index {} for alloc at byte {}", reg_idx, i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for alloc, advancing i by 10", opcode, i); }
                }
                i += 10;
            }
            0xa3 /* free */ => {
                let addr_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let addr = u64::from_le_bytes(addr_bytes) as usize;
                // in a real vm, this would free memory at the given address
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, freed memory at address {}, advancing i by 9", opcode, i, addr); }
                i += 9;
            }
            0xa4 /* cmp */ => {
                let r1_idx = payload[i+1] as usize;
                let r2_idx = payload[i+2] as usize;
                if r1_idx < registers.len() && r2_idx < registers.len() {
                    // this would set internal flags in a real cpu.
                    // for now, we'll just log the comparison.
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, compared r{} ({}) and r{} ({}), advancing i by 3", opcode, i, r1_idx, registers[r1_idx], r2_idx, registers[r2_idx]); }
                } else {
                    error!("invalid register index for cmp at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for cmp, advancing i by 3", opcode, i); }
                }
                i += 3;
            }
            0xa5 /* andbits */ => {
                let dest_reg = payload[i+1] as usize;
                let op1_reg = payload[i+2] as usize;
                let op2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && op1_reg < registers.len() && op2_reg < registers.len() {
                    registers[dest_reg] = ((registers[op1_reg] as u64) & (registers[op2_reg] as u64)) as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, andbits: r{} = r{} & r{} ({}), advancing i by 4", opcode, i, dest_reg, op1_reg, op2_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for andbits at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for andbits, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0xa6 /* orbitz */ => {
                let dest_reg = payload[i+1] as usize;
                let op1_reg = payload[i+2] as usize;
                let op2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && op1_reg < registers.len() && op2_reg < registers.len() {
                    registers[dest_reg] = ((registers[op1_reg] as u64) | (registers[op2_reg] as u64)) as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, orbitz: r{} = r{} | r{} ({}), advancing i by 4", opcode, i, dest_reg, op1_reg, op2_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for orbitz at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for orbitz, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0xa7 /* xorbits */ => {
                let dest_reg = payload[i+1] as usize;
                let op1_reg = payload[i+2] as usize;
                let op2_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && op1_reg < registers.len() && op2_reg < registers.len() {
                    registers[dest_reg] = ((registers[op1_reg] as u64) ^ (registers[op2_reg] as u64)) as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, xorbits: r{} = r{} ^ r{} ({}), advancing i by 4", opcode, i, dest_reg, op1_reg, op2_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for xorbits at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for xorbits, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0xa8 /* notbits */ => {
                let dest_reg = payload[i+1] as usize;
                let op_reg = payload[i+2] as usize;
                if dest_reg < registers.len() && op_reg < registers.len() {
                    registers[dest_reg] = (!(registers[op_reg] as u64)) as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, notbits: r{} = ~r{} ({}), advancing i by 3", opcode, i, dest_reg, op_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for notbits at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for notbits, advancing i by 3", opcode, i); }
                }
                i += 3;
            }
            0xa9 /* shl */ => {
                let dest_reg = payload[i+1] as usize;
                let op_reg = payload[i+2] as usize;
                let amount_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && op_reg < registers.len() && amount_reg < registers.len() {
                    registers[dest_reg] = ((registers[op_reg] as u64) << (registers[amount_reg] as u64)) as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, shl: r{} = r{} << r{} ({}), advancing i by 4", opcode, i, dest_reg, op_reg, amount_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for shl at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for shl, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0xaa /* shr */ => {
                let dest_reg = payload[i+1] as usize;
                let op_reg = payload[i+2] as usize;
                let amount_reg = payload[i+3] as usize;
                if dest_reg < registers.len() && op_reg < registers.len() && amount_reg < registers.len() {
                    registers[dest_reg] = ((registers[op_reg] as u64) >> (registers[amount_reg] as u64)) as f64;
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, shr: r{} = r{} >> r{} ({}), advancing i by 4", opcode, i, dest_reg, op_reg, amount_reg, registers[dest_reg]); }
                } else {
                    error!("invalid register index for shr at byte {}", i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for shr, advancing i by 4", opcode, i); }
                }
                i += 4;
            }
            0xab /* breakpoint */ => {
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, breakpoint hit, advancing i by 1", opcode, i); }
                i += 1;
            }
            0xac /* gettime */ => {
                let dest_reg_idx = payload[i+1] as usize;
                if dest_reg_idx < registers.len() {
                    let duration = SystemTime::now().duration_since(UNIX_EPOCH)
                        .expect("time went backwards");
                    registers[dest_reg_idx] = duration.as_secs_f64();
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, get_time: current time {} into register {}, advancing i by 2", opcode, i, registers[dest_reg_idx], dest_reg_idx); }
                } else {
                    error!("invalid register index {} for gettime at byte {}", dest_reg_idx, i);
                    if debug_mode { debug!("opcode 0x{:02X} at byte {}, invalid register index for gettime, advancing i by 2", opcode, i); }
                }
                i += 2;
            }
            0xad /* seedrng */ => {
                let seed_bytes: [u8; 8] = payload[i+1..i+9].try_into().unwrap();
                let seed = u64::from_le_bytes(seed_bytes);
                rng = StdRng::seed_from_u64(seed);
                if debug_mode { debug!("opcode 0x{:02X} at byte {}, rng seeded with {}, advancing i by 9", opcode, i, seed); }
                i += 9;
            }
            0xae /* exitcode */ => {
                let code_bytes: [u8; 4] = payload[i+1..i+5].try_into().unwrap();
                let exit_code = u32::from_le_bytes(code_bytes);
                info!("program exited with code {}", exit_code);
                // exit the process immediately
                std::process::exit(exit_code as i32);
            }
            _ => {
                warn!(
                    "unknown opcode 0x{:02X} at byte {}, skipping. advancing i by 1",
                    opcode, i
                );
                i += 1;
            }
        }
    }

    if debug_mode { debug!("execution finished. final quantum state dump:"); }
    if char_count > 0 {
        info!(
            "average char value: {}",
            char_sum as f64 / char_count as f64
        );
    }

    if apply_final_noise_flag {
        if let Some(_config) = noise_config.clone() {
            info!("applying final noise step to amplitudes.");
        } else {
            info!("final noise step requested, but no noise config was set for runtime.");
        }
    }

    (qs, registers)
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile { source, output, debug, } => {
            info!("compiling '{}' to '{}' (debug: {})", source, output, debug);
            match compile_qoa_to_bin(&source, debug) {
                Ok(payload) => match write_exe(&payload, &output, QEXE) {
                    Ok(_) => info!("compilation successful."),
                    Err(e) => eprintln!("error writing executable: {}", e),
                },
                Err(e) => eprintln!("error compiling qoa: {}", e),
            }
        }
        Commands::Run { program, debug, ideal, noise, final_noise, qubit,
            top_n, save_state, load_state,
        } => {
            let start_time = Instant::now();

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

            let initial_state_data = if let Some(load_path) = load_state {
                match fs::read_to_string(&load_path) {
                    Ok(json_str) => {
                        match serde_json::from_str::<(QuantumState, Vec<f64>)>(&json_str) {
                            Ok(data) => {
                                info!(
                                    "successfully loaded quantum state and registers from '{}'.",
                                    load_path
                                );
                                Some(data)
                            }
                            Err(e) => {
                                eprintln!(
                                    "error deserializing quantum state from '{}': {}",
                                    load_path, e
                                );
                                None
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("error reading quantum state file '{}': {}", load_path, e);
                        None
                    }
                }
            } else {
                None
            };

            match fs::read(&program) {
                Ok(file_data) => {
                    let noise_strength = match &noise {
                        Some(s) => match s.as_str() {
                            "random" => 0.05,
                            n => n
                                .parse::<f64>()
                                .ok()
                                .filter(|x| *x >= 0.0 && *x <= 1.0)
                                .unwrap_or(0.05),
                        },
                        None => 0.05,
                    };

                    // first, determine the inferred qubit count from the program file
                    let inferred_qubits = {
                        let (_header, _version, payload) = match parse_exe_file(&file_data) {
                            Some(x) => x,
                            None => {
                                error!("invalid or unsupported exe file, please check its header. defaulting to 0 qubits.");
                                // return a dummy tuple with an empty payload to avoid type mismatch
                                ("", 0u8, &[] as &[u8])
                            }
                        };
                        let mut max_q = 0usize;
                        let mut i = 0usize;
                        while i < payload.len() {
                            let opcode = payload[i];
                            match opcode {
                                0x04 /* qinit / initqubit */ => { // 2 bytes
                                    if i + 1 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 2;
                                }
                                0x1d /* setphase */ => { // 10 bytes (1 opcode + 1 qubit + 8 f64)
                                    if i + 9 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 10;
                                }
                                0x74 /* setpos */ => { // 18 bytes (1 opcode + 1 qubit + 8 f64 + 8 f64)
                                    if i + 17 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 18;
                                }
                                0x75 /* setwl */ => { // 10 bytes (1 opcode + 1 qubit + 8 f64)
                                    if i + 9 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 10;
                                }
                                0x76 /* wlshift */ => { // 10 bytes (1 opcode + 1 qubit + 8 f64)
                                    if i + 9 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 10;
                                }
                                0x77 /* move */ => { // 18 bytes (1 opcode + 1 qubit + 8 f64 + 8 f64)
                                    if i + 17 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 18;
                                }
                                0x18 /* charout */ => { // 2 bytes
                                    if i + 1 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 2;
                                }
                                0x32 /* qmeas / measure */ => { // 2 bytes
                                    if i + 1 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 2;
                                }
                                0x79 /* markobserved */ => { // 2 bytes
                                    if i + 1 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 2;
                                }
                                0x7a /* release */ => { // 2 bytes
                                    if i + 1 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 2;
                                }
                                0xff /* halt */ => { // 1 byte
                                    i += 1;
                                }
                                0x00 => { // handle 0x00 as a silent nop (1 byte)
                                    i += 1;
                                }
                                0x8d => { // handle 0x8d as a silent nop (1 byte)
                                    i += 1;
                                }
                                0x97 => { // handle 0x97 (retsub) in first pass by just skipping (1 byte)
                                    i += 1;
                                }
                                // other instructions
                                // single‑qubit & simple two‑qubit ops (2 bytes)
                                0x05 /* applyhadamard */ | 0x06 /* applyphaseflip */ | 0x07 /* applybitflip */ |
                                0x0d /* applytgate */ | 0x0e /* applysgate */ | 0x0a /* reset / qreset */ |
                                0x59 /* photonemit */ | 0x5a /* photondetect */ | 0x5c /* photonaddition */ |
                                0x5d /* applyphotonsubtraction */ | 0x61 /* singlephotonsourceon */ |
                                0x62 /* singlephotonsourceoff */ | 0x6f /* measureparity */ |
                                0x71 /* opticalswitchcontrol */ | 0x9b /* input */ | 0xa0 /* pushreg */ |
                                0xa1 /* popreg */ | 0xac /* gettime */ | 0x50 /* rand */ => {
                                    if i + 1 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 2;
                                }
                                // 10‑byte ops (reg + 8‑byte imm)
                                0x08 /* phaseshift */ | 0x22 /* rx */ | 0x23 /* ry */ | 0x0f /* rz */ |
                                0x24 /* phase */ | 0x66 /* setopticalattenuation */ | 0x67 /* dynamicphasecomp */ |
                                0x6a /* applydisplacement */ | 0x6d /* applysqueezing */ |
                                0x82 /* applynonlinearphaseshift */ | 0x83 /* applynonlinearsigma */ |
                                0x21 /* regset */ => {
                                    if i + 9 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 10;
                                }
                                // 3‑byte ops (two‑qubit or reg/reg)
                                0x17 /* cnot */ | 0x1e /* cz */ | 0x0b /* swap */ |
                                0x1f /* thermalavg */ | 0x65 /* opticalrouting */ | 0x69 /* crossphasemod */ |
                                0x20 /* wkbfactor */ | 0xa4 /* cmp */ | 0x51 /* sqrt */ | 0x52 /* exp */ | 0x53 /* log */ => {
                                    if i + 2 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                                    i += 3;
                                }
                                // 4‑byte ops (three regs)
                                0x0c /* controlledswap */ | 0x54 /* regadd */ | 0x55 /* regsub */ |
                                0x56 /* regmul */ | 0x57 /* regdiv */ | 0x58 /* regcopy */ |
                                0x63 /* photonbunchingctl */ | 0xa8 /* notbits */ |
                                0x31 /* charload */ => {
                                    if i + 3 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 4;
                                }
                                // variable‑length entangle lists:
                                0x11 /* entangle */ | 0x12 /* entanglebell */ => {
                                    if i + 2 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                                    i += 3;
                                }
                                0x13 /* entanglemulti */ | 0x14 /* entanglecluster */ => {
                                    if i + 1 >= payload.len() { break; }
                                    let n = payload[i+1] as usize;
                                    if i + 2 + n > payload.len() { break; }
                                    for j in 0..n {
                                        max_q = max_q.max(payload[i+2+j] as usize);
                                    }
                                    i += 2 + n;
                                }
                                0x15 /* entangleswap */ | 0x16 /* entangleswapmeasure */ => {
                                    if i + 4 >= payload.len() { break; }
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
                                0x19 /* entanglewithfb */ | 0x1a /* entangledistrib */ |
                                0x1b /* measureinbasis */ | 0x87 /* verboselog */ |
                                0x38 /* applyfeedforward */ | 0x3a /* basischange */ |
                                0x3b /* load */ | 0x3c /* store */ | 0x5b /* photoncount */ |
                                0x6b /* displacementfb */ | 0x6e /* squeezingfb */ |
                                0x73 /* photonlosscorr */ | 0x7c /* qndmeasure */ |
                                0x7d /* errorcorrect */ | 0x7f /* qstatetomography */ |
                                0x85 /* pnrdetection */ | 0x86 /* feedbackctl */ => {
                                    if i + 1 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let start = i + 2;
                                    let end = start + payload[start..].iter().position(|&b| b == 0).unwrap_or(0);
                                    i = end + 1;
                                }
                                // control flow & misc ops:
                                0x02 /* applygate(qgate) */ => {
                                    if i + 9 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let name_bytes = &payload[i+2..i+10];
                                    let name = String::from_utf8_lossy(name_bytes)
                                        .trim_end_matches('\0')
                                        .to_string();
                                    if name == "cz" {
                                        if i + 10 >= payload.len() { break; }
                                        max_q = max_q.max(payload[i+10] as usize);
                                        i += 11;
                                    } else {
                                        i += 10;
                                    }
                                }
                                0x33 /* applyrotation */ => { i += 11; }
                                0x34 /* applymultiqubitrotation */ => {
                                    if i + 2 >= payload.len() { break; }
                                    let n = payload[i+2] as usize;
                                    let needed = 3 + n + n * 8;
                                    if i + needed > payload.len() { break; }
                                    for j in 0..n {
                                        max_q = max_q.max(payload[i + 3 + j] as usize);
                                    }
                                    i += needed;
                                }
                                0x35 /* controlledphase */ | 0x36 /* applycphase */ => {
                                    if i + 10 >= payload.len() { break; }
                                    max_q = max_q
                                        .max(payload[i+1] as usize)
                                        .max(payload[i+2] as usize);
                                    i += 11;
                                }
                                0x37 /* applykerrnonlin */ => {
                                    if i + 17 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 18;
                                }
                                0x39 /* decoherenceprotect */ | 0x68 /* opticaldelaylinectl */ => {
                                    if i + 9 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 10;
                                }
                                0x3d /* loadmem */ | 0x3e /* storemem */ => {
                                    let start = i + 1;
                                    let mid   = start + payload[start..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let end   = mid   + payload[mid..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = end;
                                }
                                0x3f /* loadclassical */ | 0x40 /* storeclassical */ => {
                                    let start = i + 1;
                                    let mid   = start + payload[start..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let end   = mid   + payload[mid..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = end;
                                }
                                0x41 /* add */ | 0x42 /* sub */ | 0x43 /* and */ | 0x44 /* or */ | 0x45 /* xor */ => {
                                    let d_end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let s1_end = d_end + payload[d_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let s2_end = s1_end + payload[s1_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = s2_end;
                                }
                                0x46 /* not */ | 0x47 /* push */ | 0x4f /* pop */ => {
                                    let end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = end;
                                }
                                0x49 /* jump */ | 0x4c /* call */ => {
                                    let end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = end;
                                }
                                0x4a /* jumpifzero */ | 0x4b /* jumpifone */ => {
                                    let c_end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let l_end = c_end + payload[c_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = l_end;
                                }
                                0x4e /* timedelay */ => {
                                    if i + 9 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 10;
                                }
                                0x5e /* photonemissionpattern */ => {
                                    if i + 2 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let str_end = i + 2 + payload[i+2..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    if str_end + 8 > payload.len() { break; }
                                    i = str_end + 8;
                                }
                                0x5f /* photondetectthreshold */ => {
                                    if i + 9 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let str_end = i + 10 + payload[i+10..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = str_end;
                                }
                                0x60 /* photondetectcoincidence */ => {
                                    let n = payload[i+1] as usize;
                                    let q_end = i + 2 + n;
                                    let str_end = q_end + payload[q_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    for j in 0..n {
                                        max_q = max_q.max(payload[i+2+j] as usize);
                                    }
                                    i = str_end;
                                }
                                0x64 /* photonroute */ => {
                                    if i + 1 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let f_end = i + 2 + payload[i+2..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let t_end = f_end + payload[f_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = t_end;
                                }
                                0x6c /* applydisplacementop */ => {
                                    if i + 17 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 18;
                                }
                                0x70 /* measurewithdelay */ => {
                                    if i + 9 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let str_end = i + 10 + payload[i+10..].iter().position(|&b| b == 0).unwrap_or(0) + 1;
                                    i = str_end;
                                }
                                0x72 /* photonlosssimulate */ => {
                                    if i + 17 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 18;
                                }
                                0x7e /* errorsyndrome */ => {
                                    if i + 1 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    let s_end = i + 2 + payload[i+2..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    let r_end = s_end + payload[s_end..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = r_end;
                                }
                                0x80 /* bellstateverif */ => {
                                    if i + 2 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize).max(payload[i+2] as usize);
                                    let n_end = i + 3 + payload[i+3..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = n_end;
                                }
                                0x81 /* quantumzenoeffect */ => {
                                    if i + 17 >= payload.len() { break; }
                                    max_q = max_q.max(payload[i+1] as usize);
                                    i += 18;
                                }
                                0x84 /* applylinearopticaltransform */ => {
                                    if i + 4 >= payload.len() { break; }
                                    let _nin = payload[i+1] as usize;
                                    let _nout = payload[i+2] as usize;
                                    let _num_modes = payload[i+3] as usize;
                                    let name_start = i + 4;
                                    let name_end = name_start + payload[name_start..].iter().position(|&b| b==0).unwrap_or(0);
                                    let _name = String::from_utf8_lossy(&payload[name_start..name_end]);
                                    let mut input_qs = Vec::with_capacity(_nin);
                                    let mut current_idx = name_end + 1;
                                    for _ in 0.._nin {
                                        input_qs.push(payload[current_idx] as usize);
                                        current_idx += 1;
                                    }
                                    let mut output_qs = Vec::with_capacity(_nout);
                                    for _ in 0.._nout {
                                        output_qs.push(payload[current_idx] as usize);
                                        current_idx += 1;
                                    }
                                    let _bytes_advanced = current_idx - i;
                                    i = current_idx;
                                }
                                0x88 /* comment */ => {
                                    let end = i + 1 + payload[i+1..].iter().position(|&b| b==0).unwrap_or(0) + 1;
                                    i = end;
                                }
                                0x90 /* jmp */ | 0x91 /* jmpabs */ | 0xa3 /* free */ | 0xad /* seedrng */ => {
                                    if i + 9 >= payload.len() { break; }
                                    i += 9;
                                }
                                0x92 /* ifgt */ | 0x93 /* iflt */ | 0x94 /* ifeq */ | 0x95 /* ifne */ => {
                                    if i + 11 >= payload.len() { break; }
                                    i += 11;
                                }
                                0x96 /* calladdr */ | 0x9e /* loadregmem */ | 0xa2 /* alloc */ => {
                                    if i + 9 >= payload.len() { break; }
                                    i += 10;
                                }
                                0x9f /* storememreg */ => {
                                    if i + 9 >= payload.len() { break; }
                                    i += 10;
                                }
                                0x98 /* printf */ => {
                                    if i + 9 >= payload.len() { break; }
                                    let len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                                    let regs = payload[i+9+len] as usize;
                                    i += 1 + 8 + len + 1 + regs;
                                }
                                0x99 /* print */ | 0x9a /* println */ => {
                                    if i + 9 >= payload.len() { break; }
                                    let len = u64::from_le_bytes(payload[i+1..i+9].try_into().unwrap()) as usize;
                                    i += 1 + 8 + len;
                                }
                                0xa5 | 0xa6 | 0xa7 | 0xa9 | 0xaa => {
                                    if i + 3 >= payload.len() { break; }
                                    i += 4;
                                }
                                0xae => {
                                    if i + 5 >= payload.len() { break; }
                                    i += 5;
                                }
                                0x01 => {
                                    if i + 1 >= payload.len() { break; }
                                    i += 2;
                                }
                                _ => {
                                    // unknown opcode, advance by 1 to avoid infinite loop
                                    i += 1;
                                }
                            }
                        }
                        if max_q == 0 && payload.is_empty() { 0 } else { max_q + 1 }
                    };

                    // determine the actual number of qubits to use for simulation.
                    // the explicit `--qubit` flag takes precedence over the inferred count.
                    let final_num_qubits = if let Some(q) = qubit {
                        info!("using explicit qubit count from --qubit: {}", q);
                        q
                    } else {
                        info!("using inferred qubit count from file: {}", inferred_qubits);
                        inferred_qubits
                    };

                    // calculate and display memory usage
                    let memory_needed_bytes = (2.0_f64).powi(final_num_qubits as i32) * 16.0; // 16 bytes per amplitude (2 f64s)
                    let memory_needed_kb = memory_needed_bytes / 1024.0;
                    let memory_needed_mb = memory_needed_bytes / (1024.0 * 1024.0);
                    let memory_needed_gb = memory_needed_bytes / (1024.0 * 1024.0 * 1024.0);
                    let memory_needed_tb = memory_needed_bytes / (1024.0 * 1024.0 * 1024.0 * 1024.0);
                    let memory_needed_pb = memory_needed_bytes / (1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0);

                    if final_num_qubits > 0 {
                        print!("estimated memory for {} qubits: {:.0} bytes", final_num_qubits, memory_needed_bytes);
                        if memory_needed_pb >= 1.0 {
                            println!(" ({:.2} pb)", memory_needed_pb);
                        } else if memory_needed_tb >= 1.0 {
                            println!(" ({:.2} tb)", memory_needed_tb);
                        } else if memory_needed_gb >= 1.0 {
                            println!(" ({:.2} gb)", memory_needed_gb);
                        } else if memory_needed_mb >= 1.0 {
                            println!(" ({:.2} mb)", memory_needed_mb);
                        } else if memory_needed_kb >= 1.0 {
                            println!(" ({:.2} kb)", memory_needed_kb);
                        } else {
                            println!(); // just print bytes if less than 1 kb
                        }
                    }

                    // provide a warning if the simulation is still memory intensive, even if allowed.
                    if final_num_qubits > 26 {
                        warn!("simulating more than 26 qubits can be very memory intensive. performance might be limited by memory bandwidth rather than raw cpu computation.");
                    }

                    info!("running '{}' (debug: {}, qubits: {})", program, debug, final_num_qubits);
                    let (qs, registers) = run_exe(
                        &file_data,
                        debug,
                        noise_config.clone(),
                        final_noise,
                        final_num_qubits,
                        initial_state_data,
                    );

                    print_amplitudes(&qs, &registers, noise_strength, top_n);

                    // save state if requested
                    if let Some(save_path) = save_state {
                        let serialized_state = serde_json::to_string_pretty(&(qs, registers))
                            .expect("failed to serialize state");
                        fs::write(&save_path, serialized_state)
                            .expect("failed to write state to file");
                        info!("quantum state and registers saved to '{}'.", save_path);
                    }

                    info!("simulation complete.");
                    let end_time = Instant::now();
                    let duration = end_time.duration_since(start_time);
                    println!(
                        "total simulation time: {:.2?} seconds",
                        duration.as_secs_f64()
                    );
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
            println!("parsed spectrum direction: {:?}", direction);
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
                &ffmpeg_flags,
            ) {
                eprintln!("error generating video: {}", e);
            }
        }
        Commands::Version => {
            println!("qoa version {}", QOA_VERSION);
        }
        Commands::Flags => {
            println!("available flags:\n");
            println!(" for qoa:\n");
            println!("--compile <source> <output> compile a .qoa file to .qexe");
            println!(" --debug enable debug mode for compilation.");
            println!("\n for run:");
            println!(" --program <program file> path to the .qexe program file.");
            println!(" --debug enable debug mode for runtime.");
            println!(" --ideal set simulation to ideal (no noise) conditions (disables --noise and --final-noise).");
            println!(" --noise [probability] apply noise simulation for gates. can be `--noise` for random probability (0.1-1.0) or `--noise <probability>` for a fixed probability (0.0-1.0).");
            println!(" --final-noise apply an additional noise step to the final amplitudes (default: true).");
            println!(" --qubit <qubit number> max amount of qubit used with <qubit number> being the limit (overrides inferred count).");
            println!(" --top-n <count> display only the top n amplitudes by probability (0 to show all).\n");
            println!(
                " --save-state <path> save the final quantum state and registers to a file.\n"
            );
            println!(" --load-state <path> load a quantum state and registers from a file.\n");
            println!("\n--compilejson <source> <output> compile a .qoa file to json format\n");
            println!("\n--visual <input> <output> visualizes quantum state or circuit based on input data.");
            println!(" --resolution <width>x<height> resolution of the output visualization (e.g., '1920x1080').");
            println!(" --fps <fps> frames per second for animation (if applicable).");
            println!(" --ltr spectrum direction left-to-right.");
            println!(" --rtl spectrum direction right-to-left.");
            println!(
                " --ffmpeg-flag <ffmpeg_flag> extra ffmpeg flags (can be used multiple times)."
            );
            println!(" --ffmpeg-args <args>... additional ffmpeg arguments passed directly to ffmpeg (after '--').\n");
            println!("\n--version show version info.");
            println!("\n--flags show available flags.");
        }
    }
}