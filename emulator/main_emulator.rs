use std::fs::File;
use std::io::{self, Read, Write};
use std::env;
use num_complex::Complex64;
use rand::Rng;

#[derive(Debug)]
enum Instruction {
    QInit(u8),
    QGate(u8, String),
    QMeas(u8),
    CharLoad(u8, u8),
}

// Decode binary instructions from qoa format
fn decode_instruction(bytes: &[u8]) -> Option<(Instruction, usize)> {
    if bytes.is_empty() {
        return None;
    }
    match bytes[0] {
        0x01 => {
            if bytes.len() < 2 { return None; }
            Some((Instruction::QInit(bytes[1]), 2))
        }
        0x02 => {
            if bytes.len() < 10 { return None; }
            let reg = bytes[1];
            let gate_bytes = &bytes[2..10];
            let gate_name = String::from_utf8(gate_bytes.iter().cloned().take_while(|&b| b != 0).collect()).unwrap_or_default();
            Some((Instruction::QGate(reg, gate_name), 10))
        }
        0x03 => {
            if bytes.len() < 2 { return None; }
            Some((Instruction::QMeas(bytes[1]), 2))
        }
        0x04 => {
            if bytes.len() < 3 { return None; }
            Some((Instruction::CharLoad(bytes[1], bytes[2]), 3))
        }
        _ => None,
    }
}

struct QuantumState {
    n_qubits: usize,
    state: Vec<Complex64>,
}

impl QuantumState {
    fn new(n_qubits: usize) -> Self {
        let size = 1 << n_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); size];
        state[0] = Complex64::new(1.0, 0.0);
        QuantumState { n_qubits, state }
    }

    fn apply_gate(&mut self, target: u8, matrix: [[Complex64; 2]; 2]) {
        let target = target as usize;
        let n = self.n_qubits;
        let old_state = self.state.clone();

        for i in 0..(1 << n) {
            if (i & (1 << target)) == 0 {
                let j = i | (1 << target);

                let amp0 = old_state[i];
                let amp1 = old_state[j];

                self.state[i] = matrix[0][0] * amp0 + matrix[0][1] * amp1;
                self.state[j] = matrix[1][0] * amp0 + matrix[1][1] * amp1;
            }
        }
    }

    fn measure(&mut self, qubit: u8) -> u8 {
        let qubit = qubit as usize;
        let n = self.n_qubits;

        let mut prob0 = 0.0;
        for i in 0..(1 << n) {
            if (i & (1 << qubit)) == 0 {
                prob0 += self.state[i].norm_sqr();
            }
        }

        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();

        let result = if r < prob0 { 0 } else { 1 };

        let mut norm_factor = 0.0;
        for i in 0..(1 << n) {
            let bit = (i >> qubit) & 1;
            if bit == result {
                norm_factor += self.state[i].norm_sqr();
            } else {
                self.state[i] = Complex64::new(0.0, 0.0);
            }
        }
        let norm_factor = norm_factor.sqrt();
        if norm_factor > 0.0 {
            for amp in &mut self.state {
                *amp /= norm_factor;
            }
        }
        result as u8
    }

    fn init_qubit(&mut self, _qubit: u8) {
        self.state = vec![Complex64::new(0.0, 0.0); 1 << self.n_qubits];
        self.state[0] = Complex64::new(1.0, 0.0);
    }
}

fn hadamard() -> [[Complex64; 2]; 2] {
    let f = 1.0 / (2.0f64).sqrt();
    [
        [Complex64::new(f, 0.0), Complex64::new(f, 0.0)],
        [Complex64::new(f, 0.0), Complex64::new(-f, 0.0)],
    ]
}

fn pauli_x() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ]
}

fn pauli_z() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
    ]
}

fn identity() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ]
}

struct Emulator {
    state: QuantumState,
    max_qubit: u8,
}

impl Emulator {
    fn new() -> Self {
        Emulator {
            state: QuantumState::new(8),
            max_qubit: 7,
        }
    }

    fn update_qubit_count(&mut self, reg: u8) {
        if reg > self.max_qubit {
            self.max_qubit = reg;
            self.state = QuantumState::new((self.max_qubit + 1) as usize);
        }
    }

    fn run(&mut self, instructions: &[Instruction]) {
        for inst in instructions {
            match inst {
                Instruction::QInit(reg) => {
                    println!("Init qubit {}", reg);
                    self.update_qubit_count(*reg);
                    self.state.init_qubit(*reg);
                }
                Instruction::QGate(reg, gate) => {
                    println!("Apply gate {} to qubit {}", gate, reg);
                    self.update_qubit_count(*reg);
                    let gate_matrix = match gate.to_uppercase().as_str() {
                        "H" | "HADAMARD" => hadamard(),
                        "X" | "PAULI-X" => pauli_x(),
                        "Z" | "PAULI-Z" => pauli_z(),
                        "I" | "IDENTITY" => identity(),
                        "CNOT" => {
                            println!("Warning: CNOT gate requires two qubits; skipping.");
                            continue;
                        }
                        _ => {
                            println!("Unknown gate: {}", gate);
                            continue;
                        }
                    };
                    self.state.apply_gate(*reg, gate_matrix);
                }
                Instruction::QMeas(reg) => {
                    println!("Measure qubit {}", reg);
                    self.update_qubit_count(*reg);
                    let result = self.state.measure(*reg);
                    println!("Measurement result: {}", result);
                }
                Instruction::CharLoad(reg, val) => {
                    println!("Load char {} into register {}", val, reg);
                }
            }
        }
    }
}

fn parse_instructions(data: &[u8]) -> Vec<Instruction> {
    let mut instructions = Vec::new();
    let mut i = 0;
    while i < data.len() {
        if let Some((inst, size)) = decode_instruction(&data[i..]) {
            instructions.push(inst);
            i += size;
        } else {
            break;
        }
    }
    instructions
}

// Simple hardcoded program for now
fn compile_qoa_source(_source: &str) -> Vec<Instruction> {
    vec![
        Instruction::QInit(0),
        Instruction::QGate(0, "H".into()),
        Instruction::QMeas(0),
    ]
}

fn write_qexe_file(path: &str, instructions: &[Instruction]) -> io::Result<()> {
    if !check_output_extension(path) {
        eprintln!("Error: Output file must end with .qexe, .oexe, or .qoexe");
        std::process::exit(1);
    }

    let mut file = File::create(path)?;
    file.write_all(b"QEXE")?;
    file.write_all(&[1u8])?;
    let count = (instructions.len() as u32).to_le_bytes();
    file.write_all(&count)?;

    for inst in instructions {
        match inst {
            Instruction::QInit(reg) => {
                file.write_all(&[0x01, *reg])?;
            }
            Instruction::QGate(reg, name) => {
                let mut bytes = vec![0x02, *reg];
                let mut gate_bytes = name.clone().into_bytes();
                gate_bytes.resize(8, 0);
                bytes.extend(gate_bytes);
                file.write_all(&bytes)?;
            }
            Instruction::QMeas(reg) => {
                file.write_all(&[0x03, *reg])?;
            }
            Instruction::CharLoad(reg, val) => {
                file.write_all(&[0x04, *reg, *val])?;
            }
        }
    }

    Ok(())
}

fn check_output_extension(path: &str) -> bool {
    let lower = path.to_lowercase();
    lower.ends_with(".qexe") || lower.ends_with(".oexe") || lower.ends_with(".qoexe")
}

fn main() -> io::Result<()> {
    let mut args = env::args().skip(1);

    let cmd = args.next().expect("Usage: QOA <command> [args]\nCommands: compile, run");

    match cmd.as_str() {
        "compile" => {
            let input = args.next().expect("Usage: QOA compile <input.qoa> <output.qexe>");
            let output = args.next().expect("Usage: QOA compile <input.qoa> <output.qexe>");

            println!("Compiling {} -> {}", input, output);

            let source = std::fs::read_to_string(&input)?;
            let instructions = compile_qoa_source(&source);

            write_qexe_file(&output, &instructions)?;
            println!("Compiled {} instructions.", instructions.len());
        }

        "run" => {
            let input = args.next().expect("Usage: QOA run <input.qexe>");

            println!("Running emulator on {}", input);

            let mut file = File::open(&input)?;
            let mut header = [0u8; 9];
            file.read_exact(&mut header)?;

            let magic = &header[0..4];
            if magic != b"QEXE" && magic != b"OEXE" && magic != b"QOEX" {
                panic!("Invalid file format");
            }

            let version = header[4];
            let _count = u32::from_le_bytes([header[5], header[6], header[7], header[8]]);

            let mut data = Vec::new();
            file.read_to_end(&mut data)?;

            let instructions = parse_instructions(&data);
            println!("Loaded {} instructions version {}", instructions.len(), version);

            let mut emulator = Emulator::new();
            emulator.run(&instructions);
        }

        _ => {
            eprintln!("Unknown command: {}", cmd);
            eprintln!("Usage:\n  QOA compile <input.qoa> <output.qexe>\n  QOA run <input.qexe>");
            std::process::exit(1);
        }
    }

    Ok(())
}
