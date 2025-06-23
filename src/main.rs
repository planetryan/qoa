use clap::Parser;
use clap::CommandFactory;
use qoa::runtime::quantum_state::NoiseConfig;
use qoa::runtime::quantum_state::QuantumState;
use serde::Serialize;
use serde_json::to_writer_pretty;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::time::Instant;

mod instructions;
#[cfg(test)]
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

fn run_exe(filedata: &[u8], debug_mode: bool, noise_config: Option<NoiseConfig>, apply_final_noise_flag: bool) {
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
    let mut i = 0usize;

    // Declare registers and loop_stack here so they are in scope for both passes
    let mut registers: Vec<f64> = vec![0.0; 24]; // Assuming 24 registers, adjust as needed
    let mut loop_stack: Vec<(usize, u64)> = Vec::new();

    // First pass: Determine max_q without executing quantum operations
    while i < payload.len() {
        if debug_mode {
            eprintln!("scanning opcode 0x{:02X} at byte {}", payload[i], i);
        }
        match payload[i] {
            0x04 /* qinit */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete qinit instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                max_q = max_q.max(q);
                i += 2;
            }
            0x02 /* qgate */ => {
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
                    "cz" => {
                        if i + 10 >= payload.len() {
                            eprintln!("incomplete cz qgate instruction: missing target qubit at byte {}", i);
                            break;
                        }
                        let target_q = payload[i + 10] as usize;
                        max_q = max_q.max(q).max(target_q);
                        i += 11;
                    }
                    _ => {
                        max_q = max_q.max(q);
                        i += 10;
                    }
                }
            }
            0x05 /* applyhadamard */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete applyhadamard instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                max_q = max_q.max(q);
                i += 2;
            }
            0x0d /* applytgate */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete applytgate instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                max_q = max_q.max(q);
                i += 2;
            }
            0x0e /* applysgate */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete applysgate instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                max_q = max_q.max(q);
                i += 2;
            }
            0x17 /* controllednot (cnot/cz via cnot opcode) */ => {
                if i + 2 >= payload.len() {
                    eprintln!("incomplete controllednot instruction at byte {}", i);
                    break;
                }
                let c = payload[i + 1] as usize;
                let t = payload[i + 2] as usize;
                max_q = max_q.max(c).max(t);
                i += 3;
            }
            0x1E /* cz */ => {
                if i + 2 >= payload.len() {
                    eprintln!("incomplete cz instruction at byte {}", i);
                    break;
                }
                let c = payload[i + 1] as usize;
                let t = payload[i + 2] as usize;
                max_q = max_q.max(c).max(t);
                i += 3;
            }
            0x31 /* charload */ => {
                if i + 2 >= payload.len() {
                    eprintln!("incomplete charload instruction at byte {}", i);
                    break;
                }
                // In scan mode, we just advance the pointer, no actual char processing
                i += 3;
            }
            0x18 /* charout */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete charout instruction at byte {}", i);
                    break;
                }
                // In scan mode, we just advance the pointer
                i += 2;
            }
            0x32 /* qmeas */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete qmeas instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                max_q = max_q.max(q);
                i += 2;
            }
            0x21 /* regset */ => {
                if i + 9 >= payload.len() {
                    eprintln!("incomplete regset instruction at byte {}", i);
                    break;
                }
                // In scan mode, we just advance the pointer
                i += 10;
            }
            0x48 /* sync */ => {
                i += 1;
            }
            0x00 /* no_op */ => {
                i += 1;
            }
            0xff /* halt */ => {
                break;
            }
            0x01 /* loopstart */ => {
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
            0x10 /* endloop */ => {
                // In scan mode, we just advance the pointer for loop instructions.
                i += 1;
            }
            0x0f /* rz */ => {
                if i + 9 >= payload.len() {
                    eprintln!("incomplete rz instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                max_q = max_q.max(q);
                i += 10;
            }
            op => {
                eprintln!("warning: unknown opcode 0x{:02X} in scan at byte {}, skipping.", op, i);

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
    // Fix: Clone noise_config so it's not moved here
    let mut qs = QuantumState::new(num_qubits, noise_config.clone()); 
    let mut last_stats = Instant::now();
    let mut char_count: u64 = 0;
    let mut char_sum: u64 = 0;

    let mut i = 0;
    // Second pass: Execute instructions and interact with QuantumState
    while i < payload.len() {
        if debug_mode {
            eprintln!("executing opcode 0x{:02X} at byte {}", payload[i], i);
        }
        match payload[i] {
            0x04 /* qinit */ => {
                i += 2;
            }
            0x02 /* qgate */ => {
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
            0x05 /* applyhadamard */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete applyhadamard instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                qs.apply_h(q);
                println!("applied h gate on qubit {}", q);
                i += 2;
            }
            0x0d /* applytgate */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete applytgate instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                qs.apply_t_gate(q);
                println!("applied t gate on qubit {}", q);
                i += 2;
            }
            0x0e /* applysgate */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete applysgate instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                qs.apply_s_gate(q);
                println!("applied s gate on qubit {}", q);
                i += 2;
            }
            0x17 /* controllednot */ => {
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
            0x1E /* cz */ => {
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
            0x31 /* charload */ => {
                if i + 2 >= payload.len() {
                    eprintln!("incomplete charload instruction at byte {}", i);
                    break;
                }
                let _reg = payload[i + 1] as usize;
                let val = payload[i + 2];
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
            0x18 /* charout */ => {
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
            0x32 /* qmeas */ => {
                if i + 1 >= payload.len() {
                    eprintln!("incomplete qmeas instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                let meas_result = qs.measure(q);
                println!("\nmeasurement of qubit {}: {}", q, meas_result);
                i += 2;
            }
            0x21 /* regset */ => {
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
            0x48 /* sync */ => {
                println!("synchronized state (sync instruction encountered)");
                i += 1;
            }
            0x00 /* no_op */ => {
                i += 1;
            }
            0xff /* halt */ => {
                std::io::stdout().flush().unwrap();
                break;
            }
            0x01 /* loopstart */ => {
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
            0x10 /* endloop */ => {
                // In scan mode, we just advance the pointer for loop instructions.
                i += 1;
            }
            0x0f /* rz */ => {
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
            op => {
                eprintln!("warning: unknown opcode 0x{:02X} at byte {}, skipping.", op, i);

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

    if apply_final_noise_flag { // Check the boolean flag passed from CLI logic
        // Use the cloned noise_config to determine if ideal mode is active
        if let Some(NoiseConfig::Ideal) = &noise_config { 
            eprintln!("[info] final state noise is skipped due to ideal mode.");
        } else {
            qs.apply_final_state_noise();
        }
    }

    println!("\nfinal amplitudes:");
    if num_qubits > 0 {
        for (idx, amp) in qs.amps.iter().enumerate() {
            // Changed precision from .6 to .10 to reveal noise
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
        Commands::Compile { source, output, debug } => {
            match compile_qoa_to_bin(&source, debug) {
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
            }
        }
        Commands::Run { program, debug, ideal, noise, final_noise } => {
            let noise_config_for_gates; // no need for 'mut' here
            let effective_final_noise;

            if ideal {
                eprintln!("[info] noise mode: ideal state (explicitly set, all noise disabled)");
                noise_config_for_gates = Some(NoiseConfig::Ideal);
                effective_final_noise = false; // ideal mode explicitly disables final noise
            } else {
                // determine gate noise config
                noise_config_for_gates = match noise {
                    Some(s) if s == "random" => {
                        eprintln!("[info] noise mode: random depolarizing");
                        Some(NoiseConfig::Random)
                    },
                    Some(s) => {
                        let prob = s.parse::<f64>()
                            .map_err(|_| "invalid probability for --noise. must be a number between 0.0 and 1.0.".to_string())?;
                        if prob < 0.0 || prob > 1.0 {
                            return Err("noise probability must be between 0.0 and 1.0.".to_string());
                        }
                        eprintln!("[info] noise mode: fixed depolarizing ({})", prob);
                        Some(NoiseConfig::Fixed(prob))
                    },
                    None => {
                        // default to random depolarizing noise if no --noise flag is present
                        eprintln!("[info] noise mode: random depolarizing (default)");
                        Some(NoiseConfig::Random)
                    },
                };
                // effective_final_noise retains the value of the 'final_noise' flag (true by default, or false if specified)
                effective_final_noise = final_noise; 
            }

            match fs::read(&program) {
                Ok(filedata) => {
                    run_exe(&filedata, debug, noise_config_for_gates, effective_final_noise);
                }
                Err(e) => {
                    eprintln!("error reading program file {}: {}", program, e);
                }
            }
        }
        Commands::CompileJson { source, output } => {
            match compile_qoa_to_json(&source, &output) {
                Ok(_) => {
                    println!("compiled '{}' to json '{}'", source, output);
                }
                Err(e) => {
                    eprintln!("error compiling {}: {}", source, e);
                }
            }
        }
        Commands::Version => {
            println!("qoa version {}", QOA_VERSION);
        }
        // updated flags command to show run-specific options
        Commands::Flags => {
            println!("available flags for the 'run' command:\n");
            // get a mutable command object for the CLI
            let mut cli_cmd = Cli::command(); // Use command_mut() if you need a mutable reference to the root command later
            // find the 'run' subcommand from the mutable cli_cmd
            if let Some(run_cmd) = cli_cmd.find_subcommand_mut("run") {
                let run_command_help = run_cmd.render_help().to_string();
                // extract and print only the options/flags section
                if let Some(options_start_index) = run_command_help.find("Options:") {
                    let options_section = &run_command_help[options_start_index..];
                    // a heuristic to stop at the end of the options section, before arguments or other command help
                    if let Some(end_index) = options_section.find("\n\n") {
                        println!("{}", &options_section[..end_index].trim_end());
                    } else {
                        println!("{}", options_section.trim_end());
                    }
                } else {
                    println!("could not find 'options:' section for 'run' command help.");
                    println!("{}", run_command_help); // fallback: print full run help
                }
            } else {
                eprintln!("error: 'run' subcommand not found. this shouldn't happen.");
            }
        }
    }
    Ok(())
}