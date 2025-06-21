use num_complex::Complex64;
use rand::Rng;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};

// compile test.rs for testing
#[cfg(test)]
mod test;

mod instructions;
use serde::Serialize;
use serde_json::to_writer_pretty;

/*
    Written by Rayan
    9/6/2025

    This is the main compiler + emulator + quantum simulator for QOA.
    This compiles qoa assembly (using src/instructions.rs) into .qexe binaries.
    Runs .qexe, decoding instructions and simulating up to n-qubit quantum programs
    with Hadamard, X, CZ, and measurement.

    You can also compile into .oexe, .qoexe or .xexe, but .qexe is enough for now.

    This is the main interpreter, this effectively "simulates" random chance and measurement in quantum mechanics.
    This does NOT make your CPU a QPU, in the same way simulating another CPU on your CPU doesn't make the simulated CPU inherently.
    This is effectively emulating behavior outputted by a QPU; to actually test QOA effectively, I recommend a QPU of at least 160 logical Qubits or more.
*/

// ----------- Supported Executable Headers -----------

const QEXE: &[u8; 4] = b"QEXE";
const OEXE: &[u8; 4] = b"OEXE";
const QOEXE: &[u8; 4] = b"QOEX";
const XEXE: &[u8; 4] = b"XEXE";

// Compile a .qoa file into a binary payload (Vec<u8>)
fn compile_qoa_to_bin(src_path: &str) -> io::Result<Vec<u8>> {
    let file = File::open(src_path)?;
    let reader = BufReader::new(file);
    let mut payload = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if let Ok(inst) = instructions::parse_instruction(&line) {
            payload.extend(inst.encode());
        }
    }
    Ok(payload)
}

// Write a payload buffer into an executable file with the specified header (magic), version, and length.
fn write_exe(payload: &[u8], out_path: &str, magic: &[u8; 4]) -> io::Result<()> {
    let mut f = File::create(out_path)?;
    f.write_all(magic)?; // magic header bytes
    f.write_all(&[1])?; // version byte = 1
    f.write_all(&(payload.len() as u32).to_le_bytes())?; // payload length u32 LE
    f.write_all(payload)?; // binary payload
    Ok(())
}

// ---------- Quantum Simulator Core ----------

/// Represents an n-qubit state vector of length 2^n.
struct QuantumState {
    n: usize,
    amps: Vec<Complex64>,
}

impl QuantumState {
    /// Initialize to |0…0⟩.
    fn new(n: usize) -> Self {
        let mut amps = vec![Complex64::new(0.0, 0.0); 1 << n];
        amps[0] = Complex64::new(1.0, 0.0);
        QuantumState { n, amps }
    }

    /// Hadamard on qubit q.
    fn apply_h(&mut self, q: usize) {
        let mask = 1 << q;
        let norm = Complex64::new(1.0 / 2f64.sqrt(), 0.0);
        let size = self.amps.len();
        let mut new_amps = vec![Complex64::new(0.0, 0.0); size];

        for i in 0..size {
            if i & mask == 0 {
                let flipped = i ^ mask;
                let a = self.amps[i];
                let b = self.amps[flipped];
                new_amps[i] += norm * (a + b);
                new_amps[flipped] += norm * (a - b);
            }
        }

        self.amps = new_amps;
    }

    /// Pauli-X (NOT) on qubit q.
    fn apply_x(&mut self, q: usize) {
        let mask = 1 << q;
        let len = self.amps.len();
        for i in 0..len {
            if (i & mask) == 0 && i < (i | mask) {
                self.amps.swap(i, i | mask);
            }
        }
    }

    /// Controlled-Z between control c and target t.
    fn apply_cz(&mut self, c: usize, t: usize) {
        let cm = 1 << c;
        let tm = 1 << t;
        for i in 0..self.amps.len() {
            if (i & cm != 0) && (i & tm != 0) {
                self.amps[i] = -self.amps[i];
            }
        }
    }

    /// Measure qubit q, collapse state and return outcome 0 or 1.
    fn measure(&mut self, q: usize) -> usize {
        let mask = 1 << q;
        let prob1: f64 = self
            .amps
            .iter()
            .enumerate()
            .filter(|(i, _)| *i & mask != 0)
            .map(|(_, a)| a.norm_sqr())
            .sum();

        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let outcome = if r < prob1 { 1 } else { 0 };

        let norm = if outcome == 1 {
            prob1.sqrt()
        } else {
            (1.0 - prob1).sqrt()
        };
        for (i, amp) in self.amps.iter_mut().enumerate() {
            if ((i & mask != 0) as usize) != outcome {
                *amp = Complex64::new(0.0, 0.0);
            } else {
                *amp /= norm;
            }
        }
        outcome
    }
}

// ---------- Executor / Runner ----------

// Returns (header name, version, payload slice)
fn parse_exe_file(filedata: &[u8]) -> Option<(&'static str, u8, &[u8])> {
    if filedata.len() < 9 {
        return None;
    }
    let name = match &filedata[0..4] {
        m if m == QEXE => "QEXE",
        m if m == OEXE => "OEXE",
        m if m == QOEXE => "QOEXE",
        m if m == XEXE => "XEXE",
        _ => return None,
    };
    let version = filedata[4];
    let payload_len =
        u32::from_le_bytes([filedata[5], filedata[6], filedata[7], filedata[8]]) as usize;
    if filedata.len() < 9 + payload_len {
        return None;
    }
    Some((name, version, &filedata[9..9 + payload_len]))
}

fn run_exe(filedata: &[u8]) {
    // parse header + get payload slice
    let (header, version, payload) = match parse_exe_file(filedata) {
        Some(x) => x,
        None => {
            eprintln!("Invalid or unsupported EXE file");
            return;
        }
    };

    // FIRST PASS: scan payload to find highest qubit index only
    let mut max_q = 0usize;
    let mut i = 0usize;

    // 16 registers, initialize to 0.0
    let mut registers: Vec<f64> = vec![0.0; 16];

    while i < payload.len() {
        match payload[i] {
            0x04 /* QInit */ => {
                if i + 2 > payload.len() {
                    eprintln!("Incomplete QInit instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                max_q = max_q.max(q);
                i += 2;
            }
            0x02 /* QGate */ => {
                if i + 10 > payload.len() {
                    eprintln!("Incomplete QGate instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                max_q = max_q.max(q);
                i += 10;
            }
            0x05 /* ApplyHadamard */ => {
                if i + 2 > payload.len() {
                    eprintln!("Incomplete ApplyHadamard instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                max_q = max_q.max(q);
                i += 2;
            }
            0x31 /* CHARLOAD */ => {
                if i + 3 > payload.len() {
                    eprintln!("Incomplete CHARLOAD instruction at byte {}", i);
                    break;
                }
                i += 3;
            }
            0x32 /* QMEAS */ => {
                if i + 2 > payload.len() {
                    eprintln!("Incomplete QMEAS instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                max_q = max_q.max(q);
                i += 2;
            }
            0x21 /* REGSET */ => {
                if i + 10 > payload.len() {
                    eprintln!("Incomplete REGSET instruction at byte {}", i);
                    break;
                }
                let reg = payload[i + 1] as usize;
                if reg >= registers.len() {
                    eprintln!("Register index {} out of range", reg);
                    break;
                }
                let val_bytes = &payload[i + 2..i + 10];
                let val = f64::from_le_bytes(val_bytes.try_into().unwrap());
                registers[reg] = val;
                i += 10;
            }
            0xFF /* HALT */ => {
                i += 1;
            }
            op => {
                eprintln!("Unknown opcode 0x{:02X} in scan at byte {}", op, i);
                break;
            }
        }
    }

    println!(
        "Initializing quantum state with {} qubits (type {}, ver {})",
        max_q + 1,
        header,
        version
    );
    let mut qs = QuantumState::new(max_q + 1);

    // SECOND PASS: execute instructions + print chars + measure qubits
    let mut i = 0;
    while i < payload.len() {
        match payload[i] {
            0x04 /* QInit */ => {
                i += 2;
            }
            0x02 /* QGate */ => {
                if i + 10 > payload.len() {
                    eprintln!("Incomplete QGate instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                let name_bytes = &payload[i + 2..i + 10];
                let name = String::from_utf8_lossy(name_bytes)
                    .trim_end_matches('\0')
                    .to_string();

                match name.as_str() {
                    "H" => {
                        qs.apply_h(q);
                        println!("Applied H gate on qubit {} (via QGate)", q);
                    }
                    "X" => {
                        qs.apply_x(q);
                        println!("Applied X gate on qubit {}", q);
                    }
                    "CZ" => {
                        let tgt = if q + 1 < qs.n { q + 1 } else { qs.n - 1 };
                        qs.apply_cz(q, tgt);
                        println!(
                            "Applied CZ gate between qubits {} (control) and {} (target)",
                            q, tgt
                        );
                    }
                    other => eprintln!("Unknown gate: {}", other),
                }

                println!("Current amplitudes after gate:");
                for (idx, amp) in qs.amps.iter().enumerate() {
                    println!(
                        "{:0width$b}: {:.4} + {:.4}i",
                        idx, amp.re, amp.im,
                        width = qs.n
                    );
                }

                i += 10;
            }
            0x05 /* ApplyHadamard */ => {
                if i + 2 > payload.len() {
                    eprintln!("Incomplete ApplyHadamard instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                qs.apply_h(q);
                println!("Applied H gate on qubit {}", q);
                i += 2;
            }
            0x31 /* CHARLOAD */ => {
                if i + 3 > payload.len() {
                    eprintln!("Incomplete CHARLOAD instruction at byte {}", i);
                    break;
                }
                let val = payload[i + 2];
                print!("{}", val as char);
                io::stdout().flush().unwrap();
                i += 3;
            }
            0x32 /* QMEAS */ => {
                if i + 2 > payload.len() {
                    eprintln!("Incomplete QMEAS instruction at byte {}", i);
                    break;
                }
                let q = payload[i + 1] as usize;
                let meas_result = qs.measure(q);
                println!("\nMeasurement of qubit {}: {}", q, meas_result);
                i += 2;
            }
            0x21 /* REGSET */ => {
                if i + 10 > payload.len() {
                    eprintln!("Incomplete REGSET instruction at byte {}", i);
                    break;
                }
                let reg = payload[i + 1] as usize;
                if reg >= registers.len() {
                    eprintln!("Register index {} out of range", reg);
                    break;
                }
                let val_bytes = &payload[i + 2..i + 10];
                let val = f64::from_le_bytes(val_bytes.try_into().unwrap());
                registers[reg] = val;
                println!("Set register {} to value {}", reg, val);
                i += 10;
            }
            0xFF /* HALT */ => {
                io::stdout().flush().unwrap();
                i += 1;
            }
            op => {
                eprintln!("Unknown opcode 0x{:02X} at byte {}", op, i);
                break;
            }
        }
    }

    // Final amplitudes dump at end
    println!("\nFinal amplitudes:");
    for (idx, amp) in qs.amps.iter().enumerate() {
        println!(
            "{:0width$b}: {:.4} + {:.4}i",
            idx,
            amp.re,
            amp.im,
            width = qs.n
        );
    }
}

// ------------------ IonQ JSON support structs and function ------------------

#[derive(Serialize)]
struct JsonGate {
    gate: String,
    target: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    control: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    angle: Option<f64>,
}

// Parse minimal subset of QOA source lines into IonQ JSON gates
fn parse_line_to_json(line: &str) -> Option<(JsonGate, usize)> {
    let parts: Vec<_> = line.trim().split_whitespace().collect();
    if parts.is_empty() {
        return None;
    }

    match parts[0].to_uppercase().as_str() {
        "QGATE" => {
            if parts.len() == 3 {
                // QGATE <target> <gate>
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
                // QGATE <control> <gate> <target>
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
                // RZ <target> <angle>
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
        _ => None,
    }
}

// Compile QOA source to IonQ JSON file
fn compile_qoa_to_json(src_path: &str, out_path: &str) -> io::Result<()> {
    let file = File::open(src_path)?;
    let reader = BufReader::new(file);
    let mut circuit = Vec::new();
    let mut max_qubit = 0usize;

    for line in reader.lines() {
        let line = line?;
        if let Some((gate, max_q)) = parse_line_to_json(&line) {
            max_qubit = max_qubit.max(max_q);
            circuit.push(gate);
        }
    }

    let json_obj = serde_json::json!({
        "name": "compiled-qoa-circuit",
        "shots": 1024,
        "target": "simulator",
        "input": {
            "qubits": max_qubit + 1,
            "circuit": circuit,
        }
    });

    let out_file = File::create(out_path)?;
    to_writer_pretty(out_file, &json_obj)?;
    println!("Sucessfully Compiled QOA source to {}", out_path);
    Ok(())
}

// ------------------ CLI ------------------

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!(
            "  {} compile <source.qoa> <out.[qexe|oexe|qoexe|xexe]>",
            args[0]
        );
        eprintln!("  {} run <program.[qexe|oexe|qoexe|xexe]>", args[0]);
        eprintln!("  {} compile-json <source.qoa> <out.json>", args[0]);
        std::process::exit(1);
    }

    match args[1].as_str() {
        "compile" => {
            if args.len() != 4 {
                eprintln!(
                    "Usage: {} compile <source.qoa> <out.[qexe|oexe|qoexe|xexe]>",
                    args[0]
                );
                std::process::exit(1);
            }
            let payload = compile_qoa_to_bin(&args[2]).unwrap_or_else(|e| {
                eprintln!("Compile error: {}", e);
                std::process::exit(1);
            });
            let magic = if args[3].ends_with(".qexe") {
                QEXE
            } else if args[3].ends_with(".oexe") {
                OEXE
            } else if args[3].ends_with(".qoexe") {
                QOEXE
            } else if args[3].ends_with(".xexe") {
                XEXE
            } else {
                eprintln!("Unknown output extension. Use .qexe, .oexe, .qoexe, or .xexe");
                std::process::exit(1);
            };
            write_exe(&payload, &args[3], magic).unwrap_or_else(|e| {
                eprintln!("Write error: {}", e);
                std::process::exit(1);
            });
            println!("Compiled {} -> {}", &args[2], &args[3]);
        }
        "run" => {
            if args.len() != 3 {
                eprintln!("Usage: {} run <program.qexe|.oexe|.qoexe|.xexe>", args[0]);
                std::process::exit(1);
            }
            let filedata = fs::read(&args[2]).unwrap_or_else(|e| {
                eprintln!("Read error: {}", e);
                std::process::exit(1);
            });
            run_exe(&filedata);
        }
        "compile-json" => {
            // CLI command to compile QOA source to IonQ JSON format with automatic qubit count detection
            if args.len() != 4 {
                eprintln!("Usage: {} compile-json <source.qoa> <out.json>", args[0]);
                std::process::exit(1);
            }
            compile_qoa_to_json(&args[2], &args[3]).unwrap_or_else(|e| {
                eprintln!("Error compiling to JSON: {}", e);
                std::process::exit(1);
            });
        }
        other => {
            eprintln!("Unknown command: {}", other);
            std::process::exit(1);
        }
    }
}
