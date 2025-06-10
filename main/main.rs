use std::env;
use std::fs::File;
use std::io::{self, BufRead, Write};

/*
    Written by Rayan
    9/6/2025

    This is the main interpreter, this effetively "simulates" random chance and measurment in quantum mechanics
    This does NOT make your CPU a QPU, in the same way simulating a NES on your CPU doesnt make it a nes inherently
    This is effectively emulating behavior outputted by a QPU, to actually test qoa effectivly, I reccomend a QPU of at least 16 logical Qubits or more

    Anyways, Thank you for reviewing my code
*/

// Define the instruction set
#[derive(Debug)]
enum Instruction {
    QInit(u8),
    QGate(u8, String),
    QMeas(u8),
    CharLoad(u8, u8),
}

impl Instruction {
    // Encode instructions into a binary format
    fn encode(&self) -> Vec<u8> {
        match self {
            Instruction::QInit(reg) => vec![0x01, *reg],
            Instruction::QGate(reg, gate) => {
                let mut bytes = vec![0x02, *reg];
                let mut gate_bytes = gate.as_bytes().to_vec();
                gate_bytes.resize(8, 0); // Pad or truncate to 8 bytes
                bytes.extend(gate_bytes);
                bytes
            }
            Instruction::QMeas(reg) => vec![0x03, *reg],
            Instruction::CharLoad(reg, val) => vec![0x04, *reg, *val],
        }
    }
}

// Parse a single line of QOA into an Instruction
fn parse_line(line: &str) -> Option<Instruction> {
    let line = line.trim();
    if line.is_empty() || line.starts_with(';') {
        return None;
    }

    let tokens: Vec<&str> = line.split_whitespace().collect();
    match tokens.as_slice() {
        ["QINIT", reg] => Some(Instruction::QInit(reg.parse().ok()?)),
        ["QGATE", reg, gate] => Some(Instruction::QGate(reg.parse().ok()?, gate.to_string())),
        ["QMEAS", reg] => Some(Instruction::QMeas(reg.parse().ok()?)),
        ["CHARLOAD", reg, val] => Some(Instruction::CharLoad(reg.parse().ok()?, val.parse().ok()?)),
        _ => None,
    }
}

// Compile a .qoa file into a vector of instructions
fn compile_qoa(file_path: &str) -> io::Result<Vec<Instruction>> {
    let file = File::open(file_path)?;
    let reader = io::BufReader::new(file);
    let mut instructions = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if let Some(inst) = parse_line(&line) {
            instructions.push(inst);
        }
    }

    Ok(instructions)
}

// Generate instructions for CHARLOAD 0 to 255 loop
fn generate_charload_0_to_255() -> Vec<Instruction> {
    let mut instructions = vec![Instruction::QInit(0)];
    for val in 0u8..=255 {
        instructions.push(Instruction::CharLoad(0, val));
        instructions.push(Instruction::QMeas(0));
    }
    instructions
}

// Generate simplified Grover's algorithm on 10 qubits (single iteration)
fn generate_grovers_10q() -> Vec<Instruction> {
    let mut instructions = Vec::new();

    // QINIT 0..9
    for q in 0..10 {
        instructions.push(Instruction::QInit(q));
    }

    // Apply H to all qubits (create superposition)
    for q in 0..10 {
        instructions.push(Instruction::QGate(q, "H".to_string()));
    }

    // Oracle: simulate multi-controlled Z by applying CZ on all qubits (simplified)
    for q in 0..10 {
        instructions.push(Instruction::QGate(q, "CZ".to_string()));
    }

    // Diffusion operator
    // H on all
    for q in 0..10 {
        instructions.push(Instruction::QGate(q, "H".to_string()));
    }
    // X on all
    for q in 0..10 {
        instructions.push(Instruction::QGate(q, "X".to_string()));
    }
    // Multi-controlled Z (again simplified as CZ on all)
    for q in 0..10 {
        instructions.push(Instruction::QGate(q, "CZ".to_string()));
    }
    // X on all
    for q in 0..10 {
        instructions.push(Instruction::QGate(q, "X".to_string()));
    }
    // H on all
    for q in 0..10 {
        instructions.push(Instruction::QGate(q, "H".to_string()));
    }

    // Measure all qubits
    for q in 0..10 {
        instructions.push(Instruction::QMeas(q));
    }

    instructions
}

// Write compiled binary to .qexe/.oexe/.qoexe file
fn write_binary_file(instructions: &[Instruction], output_path: &str, magic: &[u8]) -> io::Result<()> {
    let mut file = File::create(output_path)?;

    // Write header
    file.write_all(magic)?;                 // Magic number (4 bytes)
    file.write_all(&[1])?;                  // Version number (1 byte)
    let count = instructions.len() as u32;
    file.write_all(&count.to_le_bytes())?;  // Instruction count (4 bytes LE)

    // Write each encoded instruction
    for inst in instructions {
        file.write_all(&inst.encode())?;
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage:");
        eprintln!("  {} <input.qoa> <output.[qexe|oexe|qoexe]>  # compile .qoa file", args[0]);
        eprintln!("  {} --generate <output.[qexe|oexe|qoexe]>   # generate 0-255 CHARLOAD loop", args[0]);
        eprintln!("  {} --grover <output.[qexe|oexe|qoexe]>     # generate 10-qubit Grover's algorithm", args[0]);
        std::process::exit(1);
    }

    let (instructions, output_path, magic) = match args[1].as_str() {
        "--generate" => {
            let output_path = &args[2];
            let magic = if output_path.ends_with(".qexe") {
                b"QEXE"
            } else if output_path.ends_with(".oexe") {
                b"OEXE"
            } else if output_path.ends_with(".qoexe") {
                b"QOEX"
            } else {
                eprintln!("Error: Output file must end with .qexe, .oexe, or .qoexe");
                std::process::exit(1);
            };
            (generate_charload_0_to_255(), output_path, magic)
        }
        "--grover" => {
            let output_path = &args[2];
            let magic = if output_path.ends_with(".qexe") {
                b"QEXE"
            } else if output_path.ends_with(".oexe") {
                b"OEXE"
            } else if output_path.ends_with(".qoexe") {
                b"QOEX"
            } else {
                eprintln!("Error: Output file must end with .qexe, .oexe, or .qoexe");
                std::process::exit(1);
            };
            (generate_grovers_10q(), output_path, magic)
        }
        _ => {
            if args.len() < 3 {
                eprintln!("Error: Not enough arguments");
                std::process::exit(1);
            }
            let input_path = &args[1];
            let output_path = &args[2];
            let magic = if output_path.ends_with(".qexe") {
                b"QEXE"
            } else if output_path.ends_with(".oexe") {
                b"OEXE"
            } else if output_path.ends_with(".qoexe") {
                b"QOEX"
            } else {
                eprintln!("Error: Output file must end with .qexe, .oexe, or .qoexe");
                std::process::exit(1);
            };
            let instructions = match compile_qoa(input_path) {
                Ok(insts) => insts,
                Err(e) => {
                    eprintln!("Failed to compile {}: {}", input_path, e);
                    std::process::exit(1);
                }
            };
            (instructions, output_path, magic)
        }
    };

    if let Err(e) = write_binary_file(&instructions, output_path, magic) {
        eprintln!("Failed to write binary: {}", e);
        std::process::exit(1);
    }

    println!("Compiled {} instructions to {}", instructions.len(), output_path);

    // Print 1 if grover algorithm generated (simple indicator)
    if args[1] == "--grover" {
        println!("1");
    }
}
