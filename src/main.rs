use num_complex::Complex64;
use rand::prelude::*;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};

mod instructions;

/*
    Written by Rayan
    9/6/2025

    This is the main compiler + emulator + quantum simulator for QOA.
    This Compiles qoa assembly (using src/instructions.rs) into .qexe binaries.
    Runs .qexe, decoding instructions and simulating up to n-qubit quantum programs
    with Hadamard, X, CZ, and measurement.

    You can also compile into .oexe, .qoexe or .xexe, but .qexe is enough for now.

    This is the main interpreter, this effectively "simulates" random chance and measurement in quantum mechanics.
    This does NOT make your CPU a QPU, in the same way simulating another CPU on your CPU doesn't make it the simulated CPU inherently.
    This is effectively emulating behavior outputted by a QPU; to actually test QOA effectively, I recommend a QPU of at least 160 logical Qubits or more.
*/

// ----------- Supported Executable Headers -----------
const QEXE_MAGIC: &[u8; 4] = b"QEXE";
const OEXE_MAGIC: &[u8; 4] = b"OEXE";
const QOEXE_MAGIC: &[u8; 4] = b"QOEX";
const XEXE_MAGIC: &[u8; 4] = b"XEXE";

// Compile a .qoa file into a binary payload (Vec<u8>).
fn compile_qoa_to_bin(src_path: &str) -> io::Result<Vec<u8>> {
    let file = File::open(src_path)?;
    let reader = BufReader::new(file);
    let mut payload = Vec::new();

    // Read each line and parse to instructions, encode into binary payload
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
        for i in 0..self.amps.len() {
            if i & mask == 0 {
                let a = self.amps[i];
                let b = self.amps[i | mask];
                self.amps[i] = norm * (a + b);
                self.amps[i | mask] = norm * (a - b);
            }
        }
    }

    /// Pauli-X (NOT) on qubit q.
    fn apply_x(&mut self, q: usize) {
        let mask = 1 << q;
        for i in 0..self.amps.len() {
            if i & mask == 0 {
                self.amps.swap(i, i | mask);
            }
        }
    }

    /// Controlled-Z between control c and target t.
    fn apply_cz(&mut self, c: usize, t: usize) {
        let cm = 1 << c;
        let tm = 1 << t;
        for i in 0..self.amps.len() {
            if i & cm != 0 && i & tm != 0 {
                self.amps[i] *= -1.0;
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

        let mut rng = thread_rng();
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
    let name = match &filedata[0..4] {
        m if m == QEXE_MAGIC => "QEXE",
        m if m == OEXE_MAGIC => "OEXE",
        m if m == QOEXE_MAGIC => "QOEXE",
        m if m == XEXE_MAGIC => "XEXE",
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

// run executables
fn run_exe(filedata: &[u8]) {
    // parse header + get payload slice
    let (header, version, payload) = match parse_exe_file(filedata) {
        Some(x) => x,
        None => {
            eprintln!("Invalid or unsupported EXE file");
            return;
        }
    };

// FIRST PASS: scan payload to find highest qubit index
let mut max_q = 0usize;
let mut i = 0usize;
while i < payload.len() {
    match payload[i] {
        0x04 /* QInit */ => {
            // [opcode, qubit]
            if i + 1 >= payload.len() { break }
            let q = payload[i + 1] as usize;
            max_q = max_q.max(q);
            i += 2;
        }
        0x02 /* QGate */ => {
            // [opcode, qubit, 8‑byte name]
            if i + 9 >= payload.len() { break }
            let q = payload[i + 1] as usize;
            max_q = max_q.max(q);
            i += 10;
        }
        0x31 /* CHARLOAD */ => {
            // [opcode, qubit, char]
            if i + 2 >= payload.len() { break }
            let q = payload[i + 1] as usize;
            max_q = max_q.max(q);
            i += 3;
        }
        0x32 /* QMEAS */ => {
            // [opcode, qubit]
            if i + 1 >= payload.len() { break }
            let q = payload[i + 1] as usize;
            max_q = max_q.max(q);
            i += 2;
        }
        0xFF /* HALT */ => break,
        op => {
            eprintln!("Unknown opcode 0x{:02X} in scan at byte {}", op, i);
            break;
        }
    }
}

// Print header + init quantum state
let n_qubits = max_q + 1;
println!(
    "Initializing quantum state with {} qubits (type {}, ver {})",
    n_qubits, header, version
);
let mut qs = QuantumState::new(n_qubits);

// SECOND PASS: actually execute + print chars
i = 0;
while i < payload.len() {
    match payload[i] {
        0x04 /* QInit */ => {
            // already sized qstate in first pass, so just skip
            i += 2;
        }
        0x02 /* QGate */ => {
            // gate_name handling unchanged
            let q = payload[i + 1] as usize;
            let name_bytes = &payload[i + 2..i + 10];
            let name = String::from_utf8_lossy(name_bytes)
                .trim_end_matches('\0')
                .to_string();
            match name.as_str() {
                "H" => qs.apply_h(q),
                "X" => qs.apply_x(q),
                "CZ" => {
                    let tgt = (q + 1).min(qs.n - 1);
                    qs.apply_cz(q, tgt);
                }
                other => eprintln!("Unknown gate: {}", other),
            }
            i += 10;
        }
        0x31 /* CHARLOAD */ => {
            // load and print
            let val = payload[i + 2];
            print!("{}", val as char);
            i += 3;
        }
        0x32 /* QMEAS */ => {
            let q = payload[i + 1] as usize;
            let _ = qs.measure(q);
            i += 2;
        }
        0xFF /* HALT */ => break,
        op => {
            eprintln!("Unknown opcode 0x{:02X} at byte {}", op, i);
            break;
        }
    }
}

    // final newline + amplitude dump
    io::stdout().flush().unwrap();
    println!();
    println!("\nFinal amplitudes:");
    for (idx, amp) in qs.amps.iter().enumerate() {
        println!("{:0width$b}: {:.4} + {:.4}i", idx, amp.re, amp.im, width = qs.n);
    }
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
            // Choose header based on extension
            let magic = if args[3].ends_with(".qexe") {
                QEXE_MAGIC
            } else if args[3].ends_with(".oexe") {
                OEXE_MAGIC
            } else if args[3].ends_with(".qoexe") {
                QOEXE_MAGIC
            } else if args[3].ends_with(".xexe") {
                XEXE_MAGIC
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
        other => {
            eprintln!("Unknown command: {}", other);
            std::process::exit(1);
        }
    }
}
