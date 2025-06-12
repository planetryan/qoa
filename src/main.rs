use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use num_complex::Complex64;
use rand::prelude::*;

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
    This does NOT make your CPU a QPU, in the same way simulating another CPU on your CPU doesn’t make it the simulated CPU inherently.
    This is effectively emulating behavior outputted by a QPU; to actually test QOA effectively, I recommend a QPU of at least 160 logical Qubits or more.
*/

// Compile a .qoa file into a binary payload (Vec<u8>).
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

// Write a payload buffer into a `.qexe` file (with header "QEXE", version, length).
fn write_qexe(payload: &[u8], out_path: &str) -> io::Result<()> {
    let mut f = File::create(out_path)?;
    f.write_all(b"QEXE")?;                            // magic
    f.write_all(&[1])?;                               // version
    f.write_all(&(payload.len() as u32).to_le_bytes())?; // length
    f.write_all(payload)?;
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
                self.amps[i]        = norm * (a + b);
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

    /// Controlled-Z between c and t.
    fn apply_cz(&mut self, c: usize, t: usize) {
        let cm = 1 << c;
        let tm = 1 << t;
        for i in 0..self.amps.len() {
            if i & cm != 0 && i & tm != 0 {
                self.amps[i] *= -1.0;
            }
        }
    }

    /// Measure qubit q, collapse and return 0 or 1.
    fn measure(&mut self, q: usize) -> usize {
        let mask = 1 << q;
        let prob1: f64 = self.amps.iter()
            .enumerate()
            .filter(|(i, _)| *i & mask != 0)
            .map(|(_, a)| a.norm_sqr())
            .sum();

        let mut rng = thread_rng();
        let r: f64 = rng.gen();
        let outcome = if r < prob1 { 1 } else { 0 };

        let norm = if outcome == 1 { prob1.sqrt() } else { (1.0 - prob1).sqrt() };
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

fn run_qexe(filedata: &[u8]) {
    // --- 1) Strip header ---
    if &filedata[0..4] != b"QEXE" {
        eprintln!("Invalid magic, expected QEXE");
        return;
    }
    let version = filedata[4];
    let payload_len = u32::from_le_bytes([filedata[5], filedata[6], filedata[7], filedata[8]]) as usize;
    let payload = &filedata[9..9 + payload_len];

    // --- 2) Determine number of qubits via first pass ---
    let mut max_q = 0usize;
    let mut i = 0;
    while i < payload.len() {
        match payload[i] {
            0x01 /* QInit */ => {
                if i + 1 < payload.len() {
                    max_q = max_q.max(payload[i + 1] as usize);
                }
                i += 2;
            }
            0x02 /* QGate */     => { i += 10; } // opcode + reg + 8-byte gate
            0x03 /* QMeas */     => { i += 2;  }
            0x04 /* CharLoad */  => { i += 3;  }
            _ => break,
        }
    }
    let n = max_q + 1;
    println!("Initializing quantum state with {} qubits (ver {})", n, version);
    let mut qs = QuantumState::new(n);

    // --- 3) Execute instructions ---
    i = 0;
    while i < payload.len() {
        match payload[i] {
            0x01 => {
                // QInit q
                let _q = payload[i + 1] as usize;
                i += 2;
            }
            0x02 => {
                // QGate q, gate_name (8 bytes)
                let q = payload[i + 1] as usize;
                let name = String::from_utf8_lossy(&payload[i + 2..i + 10])
                    .trim_end_matches('\0')
                    .to_string();
                match name.as_str() {
                    "H"  => qs.apply_h(q),
                    "X"  => qs.apply_x(q),
                    "CZ" => qs.apply_cz(q, q),
                    _    => {}
                }
                i += 10;
            }
            0x03 => {
                // QMeas q
                let q = payload[i + 1] as usize;
                let _res = qs.measure(q);
                i += 2;
            }
            0x04 => {
                // CharLoad: print ASCII
                let val = payload[i + 2];
                print!("{}", val as char);
                io::stdout().flush().unwrap();
                i += 3;
            }
            op => {
                eprintln!("Unknown opcode 0x{:02x}", op);
                break;
            }
        }
    }
    println!(); // newline after payload

    // --- 4) Dump final state ---
    println!("\nFinal amplitudes:");
    for (idx, amp) in qs.amps.iter().enumerate() {
        println!("{:0width$b}: {:.4} + {:.4}i",
            idx, amp.re, amp.im,
            width = qs.n);
    }
}

// ------------------ CLI ------------------

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  {} compile <source.qoa> <out.qexe>", args[0]);
        eprintln!("  {} run <program.qexe>", args[0]);
        std::process::exit(1);
    }

    match args[1].as_str() {
        "compile" => {
            if args.len() != 4 {
                eprintln!("Usage: {} compile <source.qoa> <out.qexe>", args[0]);
                std::process::exit(1);
            }
            let payload = compile_qoa_to_bin(&args[2]).unwrap_or_else(|e| {
                eprintln!("Compile error: {}", e);
                std::process::exit(1);
            });
            write_qexe(&payload, &args[3]).unwrap_or_else(|e| {
                eprintln!("Write error: {}", e);
                std::process::exit(1);
            });
            println!("Compiled {} -> {}", &args[2], &args[3]);
        }
        "run" => {
            if args.len() != 3 {
                eprintln!("Usage: {} run <program.qexe>", args[0]);
                std::process::exit(1);
            }
            let filedata = fs::read(&args[2]).unwrap_or_else(|e| {
                eprintln!("Read error: {}", e);
                std::process::exit(1);
            });
            run_qexe(&filedata);
        }
        other => {
            eprintln!("Unknown command: {}", other);
            std::process::exit(1);
        }
    }
}
