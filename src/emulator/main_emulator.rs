use std::fs::File;
use std::io::{self, Read, Write};
use std::env;
use std::collections::HashMap;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// --- Config ---

const NOISE_PROBABILITY: f64 = 0.01; // 1% chance of an error after each gate
const MAX_REGISTERS: u8 = 255; // Using u8, so this is the natural limit

// Opcodes for the binary .qexe format, improving readability over magic numbers
mod opcodes {
    pub const QINIT: u8 = 0x01;
    pub const QGATE: u8 = 0x02;
    pub const QMEAS: u8 = 0x03;
    pub const CHARLOAD: u8 = 0x04;
    pub const REGSET: u8 = 0x11;
    pub const LOOPSTART: u8 = 0x20;
    pub const ENDLOOP: u8 = 0x21;
    pub const RZ: u8 = 0x30;
    pub const CZ: u8 = 0x31;
    pub const HALT: u8 = 0xFF;
}

// --- Custom Handling ---

#[derive(Debug)]
pub struct ParseError {
    pub line_number: usize,
    pub message: String,
}

#[derive(Debug)]
pub struct DecodeError {
    pub position: usize,
    pub message: String,
}


// --- Core Data  ---

#[derive(Debug)]
enum Instruction {
    QInit(u8),
    QGate(u8, String), // Using String for gate name to support various gates
    QMeas(u8),
    CharLoad(u8, u8),
    RegSet(u8, i64),
    LoopStart(u8),
    EndLoop,
    RZ(u8, f64),
    CZ(u8, u8),
    HALT,
}

struct QuantumState {
    n_qubits: usize,
    state: Vec<Complex64>,
}

impl QuantumState {
    fn new(n_qubits: usize) -> Self {
        let size = 1 << n_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); size];
        if size > 0 {
            state[0] = Complex64::new(1.0, 0.0);
        }
        QuantumState { n_qubits, state }
    }
    
    /// This is the gate application logic
    /// It correctly applies a single qubit gate by transforming amplitude pairs
    fn apply_single_qubit_gate(&mut self, qubit_idx: usize, gate_matrix: &[[Complex64; 2]; 2]) {
        if qubit_idx >= self.n_qubits { return; }
        
        let k = 1 << qubit_idx;
        let n_pairs = 1 << (self.n_qubits - 1);

        for i in 0..n_pairs {
            let i0 = (i & !(k - 1)) << 1 | (i & (k - 1));
            let i1 = i0 | k;

            let amp0_old = self.state[i0];
            let amp1_old = self.state[i1];

            self.state[i0] = gate_matrix[0][0] * amp0_old + gate_matrix[0][1] * amp1_old;
            self.state[i1] = gate_matrix[1][0] * amp0_old + gate_matrix[1][1] * amp1_old;
        }
    }

    fn apply_rz(&mut self, qubit_idx: u8, angle: f64) {
        let qubit_idx = qubit_idx as usize;
        if qubit_idx >= self.n_qubits { return; }
        
        let phase_neg = Complex64::from_polar(1.0, -angle / 2.0);
        let phase_pos = Complex64::from_polar(1.0, angle / 2.0);

        for i in 0..(1 << self.n_qubits) {
            if (i >> qubit_idx) & 1 == 0 {
                self.state[i] *= phase_neg;
            } else {
                self.state[i] *= phase_pos;
            }
        }
    }

    fn apply_cz(&mut self, control_qubit: u8, target_qubit: u8) {
        let control_qubit = control_qubit as usize;
        let target_qubit = target_qubit as usize;
        if control_qubit >= self.n_qubits || target_qubit >= self.n_qubits { return; }

        let control_mask = 1 << control_qubit;
        let target_mask = 1 << target_qubit;

        for i in 0..(1 << self.n_qubits) {
            if (i & control_mask) != 0 && (i & target_mask) != 0 {
                self.state[i] *= -1.0;
            }
        }
    }

    fn measure(&mut self, qubit: u8, rng: &mut dyn Rng) -> u8 {
        let qubit = qubit as usize;
        if qubit >= self.n_qubits { return 255; } // Return an error code

        let prob0: f64 = self.state.iter().enumerate()
            .filter(|(i, _)| (i >> qubit) & 1 == 0)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();

        let r: f64 = rng.gen();
        let result = if r < prob0 { 0 } else { 1 };

        let norm_factor_sq: f64 = self.state.iter().enumerate()
            .filter(|(i, _)| ((i >> qubit) & 1) as u8 == result)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();

        if norm_factor_sq > 1e-9 {
            let norm_factor = norm_factor_sq.sqrt();
            for (i, amp) in self.state.iter_mut().enumerate() {
                if ((i >> qubit) & 1) as u8 == result {
                    *amp /= norm_factor;
                } else {
                    *amp = Complex64::new(0.0, 0.0);
                }
            }
        }
        result
    }

    /// Resets the quantum state to the ground state |0...0>.
    fn init_qubit(&mut self, _qubit: u8) {
        self.state.fill(Complex64::new(0.0, 0.0));
        if !self.state.is_empty() {
            self.state[0] = Complex64::new(1.0, 0.0);
        }
    }

    /// Normalizes the state vector to have a magnitude of 1.
    /// Crucial for maintaining stability with floating point errors and noise.
    fn normalize(&mut self) {
        let norm_sq: f64 = self.state.iter().map(|c| c.norm_sqr()).sum();
        if norm_sq.abs() < 1e-9 { return; }
        
        let norm_factor = norm_sq.sqrt();
        for amp in &mut self.state {
            *amp /= norm_factor;
        }
    }
}

// --- Gate Matrix Definitions ---

fn hadamard() -> [[Complex64; 2]; 2] {
    let f = 1.0 / (2.0f64).sqrt();
    [[Complex64::new(f, 0.0), Complex64::new(f, 0.0)], [Complex64::new(f, 0.0), Complex64::new(-f, 0.0)]]
}

fn pauli_x() -> [[Complex64; 2]; 2] {
    [[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)], [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]]
}

fn pauli_y() -> [[Complex64; 2]; 2] {
    [[Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)], [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)]]
}

fn pauli_z() -> [[Complex64; 2]; 2] {
    [[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)], [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]]
}

fn identity() -> [[Complex64; 2]; 2] {
    [[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)], [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]]
}


// --- Emulator Logic ---

struct Emulator {
    registers: HashMap<u8, i64>,
    state: QuantumState,
    max_qubit: u8,
    loop_stack: Vec<usize>,
    rng: Box<dyn Rng>, // Use a trait object for deterministic testing
}

impl Emulator {
    fn new(seed: Option<u64>) -> Self {
        let rng: Box<dyn Rng> = match seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::thread_rng()),
        };
        Emulator {
            registers: HashMap::new(),
            state: QuantumState::new(8), // Default size
            max_qubit: 7,
            loop_stack: Vec::new(),
            rng,
        }
    }
    
    /// Helper function to apply random Pauli noise to a qubit.
    /// This avoids repeating the same logic multiple times.
    fn apply_pauli_noise(&mut self, qubit: u8) {
        if self.rng.gen::<f64>() < NOISE_PROBABILITY {
            let error_type = self.rng.gen_range(0..3);
            let (noise_matrix, noise_name) = match error_type {
                0 => (pauli_x(), "X"),
                1 => (pauli_y(), "Y"),
                _ => (pauli_z(), "Z"),
            };
            println!("! Applying Pauli-{} noise on qubit {}", noise_name, qubit);
            self.state.apply_single_qubit_gate(qubit as usize, &noise_matrix);
            self.state.normalize(); // Normalize after non unitary noise operation
        }
    }

    fn update_qubit_count(&mut self, reg: u8) {
        if reg > self.max_qubit {
            let old_n = self.max_qubit + 1;
            self.max_qubit = reg;
            let new_n = self.max_qubit + 1;
            println!("-> Resizing quantum state from {} to {} qubits.", old_n, new_n);
            self.state = QuantumState::new(new_n as usize);
        }
    }

    fn run(&mut self, instructions: &[Instruction]) {
        let mut ip = 0;
        while ip < instructions.len() {
            let inst = &instructions[ip];
            match inst {
                Instruction::QInit(reg) => {
                    println!("Initializing quantum state with {} qubits", self.max_qubit + 1);
                    self.update_qubit_count(*reg);
                    self.state.init_qubit(*reg);
                }
                Instruction::QGate(reg, gate_name) => {
                    println!("Applied {} gate on qubit {}", gate_name, reg);
                    self.update_qubit_count(*reg);
                    
                    let gate_matrix = match gate_name.to_uppercase().as_str() {
                        "H" | "HADAMARD" => hadamard(),
                        "X" | "PAULI-X"  => pauli_x(),
                        "Y" | "PAULI-Y"  => pauli_y(),
                        "Z" | "PAULI-Z"  => pauli_z(),
                        "I" | "IDENTITY" => identity(),
                        _ => {
                            println!("Warning: Unknown single-qubit gate '{}'. Skipping.", gate_name);
                            ip += 1;
                            continue;
                        }
                    };
                    self.state.apply_single_qubit_gate(*reg as usize, &gate_matrix);
                    self.apply_pauli_noise(*reg);
                }
                Instruction::QMeas(reg) => {
                    self.update_qubit_count(*reg);
                    let result = self.state.measure(*reg, &mut self.rng);
                    println!("Measurement of qubit {}: {}", reg, result);
                }
                Instruction::CharLoad(reg, val) => {
                    println!("Load char {} into register r{}", val, reg);
                    self.registers.insert(*reg, *val as i64);
                }
                Instruction::RegSet(reg, val) => {
                    println!("Set register r{} <- {}", reg, val);
                    self.registers.insert(*reg, *val);
                }
                Instruction::LoopStart(reg_idx) => {
                    let loop_count = self.registers.entry(*reg_idx).or_insert(0);
                    if *loop_count > 0 {
                        self.loop_stack.push(ip);
                        *loop_count -= 1;
                    } else {
                        // Skip to matching ENDLOOP
                        let mut loop_depth = 1;
                        ip += 1;
                        while ip < instructions.len() && loop_depth > 0 {
                            match &instructions[ip] {
                                Instruction::LoopStart(_) => loop_depth += 1,
                                Instruction::EndLoop => loop_depth -= 1,
                                _ => {}
                            }
                            if loop_depth > 0 { ip += 1; }
                        }
                    }
                }
                Instruction::EndLoop => {
                    if let Some(&loop_start_ip) = self.loop_stack.last() {
                         if let Instruction::LoopStart(reg_idx) = instructions[loop_start_ip] {
                            if *self.registers.get(&reg_idx).unwrap_or(&0) > 0 {
                                ip = loop_start_ip; // Jump back
                                *self.registers.entry(reg_idx).or_insert(0) -= 1;
                            } else {
                                self.loop_stack.pop(); // Loop finished
                            }
                         }
                    } else {
                        println!("Warning: ENDLOOP without matching LOOPSTART.");
                    }
                }
                Instruction::RZ(reg, angle) => {
                    println!("Applied RZ gate with angle {} to qubit {}", angle, reg);
                    self.update_qubit_count(*reg);
                    self.state.apply_rz(*reg, *angle);
                    self.apply_pauli_noise(*reg);
                }
                Instruction::CZ(control, target) => {
                    println!("Applied CZ gate on control {} target {}", control, target);
                    self.update_qubit_count(*control);
                    self.update_qubit_count(*target);
                    self.state.apply_cz(*control, *target);
                    self.apply_pauli_noise(*control);
                    self.apply_pauli_noise(*target);
                }
                Instruction::HALT => {
                    println!("HALT instruction encountered. Stopping execution.");
                    return;
                }
            }
            ip += 1;
        }
    }
}


// --- Parsing and Compilation ---

/// Decodes binary instructions from a byte slice. Returns a Result for robust error handling.
fn decode_instruction(bytes: &[u8], position: usize) -> Result<(Instruction, usize), DecodeError> {
    if bytes.is_empty() {
        return Err(DecodeError { position, message: "Unexpected end of file".into() });
    }
    let opcode = bytes[0];
    let min_len = |n| if bytes.len() < n { Err(DecodeError { position, message: format!("Incomplete instruction for opcode {:#04x}", opcode) }) } else { Ok(()) };

    match opcode {
        opcodes::QINIT => {
            min_len(2)?;
            Ok((Instruction::QInit(bytes[1]), 2))
        }
        opcodes::QGATE => {
            min_len(10)?;
            let reg = bytes[1];
            let gate_bytes = &bytes[2..10];
            let gate_name = String::from_utf8(
                gate_bytes.iter().cloned().take_while(|&b| b != 0).collect()
            ).map_err(|e| DecodeError { position, message: format!("Invalid UTF-8 in gate name: {}", e)})?;
            Ok((Instruction::QGate(reg, gate_name), 10))
        }
        opcodes::QMEAS => {
            min_len(2)?;
            Ok((Instruction::QMeas(bytes[1]), 2))
        }
        opcodes::CHARLOAD => {
            min_len(3)?;
            Ok((Instruction::CharLoad(bytes[1], bytes[2]), 3))
        }
        opcodes::REGSET => {
            min_len(10)?;
            let reg = bytes[1];
            let raw = &bytes[2..10];
            let val = i64::from_le_bytes(raw.try_into().map_err(|e| DecodeError {position, message: format!("Invalid byte slice for REGSET value: {}", e)})?);
            Ok((Instruction::RegSet(reg, val), 10))
        }
        opcodes::LOOPSTART => {
            min_len(2)?;
            Ok((Instruction::LoopStart(bytes[1]), 2))
        }
        opcodes::ENDLOOP => Ok((Instruction::EndLoop, 1)),
        opcodes::RZ => {
            min_len(10)?;
            let reg = bytes[1];
            let raw = &bytes[2..10];
            let val = f64::from_le_bytes(raw.try_into().map_err(|e| DecodeError {position, message: format!("Invalid byte slice for RZ value: {}", e)})?);
            Ok((Instruction::RZ(reg, val), 10))
        }
        opcodes::CZ => {
            min_len(3)?;
            Ok((Instruction::CZ(bytes[1], bytes[2]), 3))
        }
        opcodes::HALT => Ok((Instruction::HALT, 1)),
        _ => Err(DecodeError { position, message: format!("Unknown opcode: {:#04x}", opcode)}),
    }
}

/// Parses a slice of bytes into a vector of instructions.
fn parse_instructions(data: &[u8]) -> Result<Vec<Instruction>, DecodeError> {
    let mut instructions = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let (inst, size) = decode_instruction(&data[i..], i)?;
        instructions.push(inst);
        i += size;
    }
    Ok(instructions)
}

/// Compiles .qoa source text into a vector of instructions. Returns a Result for robust error handling.
fn compile_qoa_source(source: &str) -> Result<Vec<Instruction>, ParseError> {
    let mut instructions = Vec::new();

    for (line_num, line) in source.lines().enumerate() {
        let current_line_display = line_num + 1;
        let trimmed_line = line.split(';').next().unwrap_or("").trim(); // Ignore comments

        if trimmed_line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = trimmed_line.split_whitespace().collect();
        let opcode = parts[0].to_uppercase();

        let parse_u8 = |s: &str| s.parse::<u8>().map_err(|e| ParseError { line_number: current_line_display, message: format!("Invalid u8 value '{}': {}", s, e) });
        let parse_i64 = |s: &str| s.parse::<i64>().map_err(|e| ParseError { line_number: current_line_display, message: format!("Invalid i64 value '{}': {}", s, e) });
        let parse_f64 = |s: &str| s.parse::<f64>().map_err(|e| ParseError { line_number: current_line_display, message: format!("Invalid f64 value '{}': {}", s, e) });
        
        let check_args = |n| if parts.len() != n { Err(ParseError { line_number: current_line_display, message: format!("'{}' expects {} argument(s), but got {}", opcode, n - 1, parts.len() - 1)}) } else { Ok(()) };

        let instruction = match opcode.as_str() {
            "QINIT" | "QMEAS" | "LOOPSTART" => {
                check_args(2)?;
                let reg = parse_u8(parts[1])?;
                match opcode.as_str() {
                    "QINIT" => Instruction::QInit(reg),
                    "QMEAS" => Instruction::QMeas(reg),
                    "LOOPSTART" => Instruction::LoopStart(reg),
                    _ => unreachable!(),
                }
            }
            "QGATE" => {
                check_args(3)?;
                let reg = parse_u8(parts[1])?;
                let gate_name = parts[2].to_string();
                Instruction::QGate(reg, gate_name)
            }
            "CHARLOAD" => {
                check_args(3)?;
                let reg = parse_u8(parts[1])?;
                let val = parse_u8(parts[2])?;
                Instruction::CharLoad(reg, val)
            }
            "REGSET" => {
                check_args(3)?;
                let reg = parse_u8(parts[1])?;
                let val = parse_i64(parts[2])?;
                Instruction::RegSet(reg, val)
            }
            "ENDLOOP" | "HALT" => {
                check_args(1)?;
                match opcode.as_str() {
                    "ENDLOOP" => Instruction::EndLoop,
                    "HALT" => Instruction::HALT,
                    _ => unreachable!(),
                }
            }
            "RZ" => {
                check_args(3)?;
                let reg = parse_u8(parts[1])?;
                let angle = parse_f64(parts[2])?;
                Instruction::RZ(reg, angle)
            }
            "CZ" => {
                check_args(3)?;
                let control = parse_u8(parts[1])?;
                let target = parse_u8(parts[2])?;
                Instruction::CZ(control, target)
            }
            _ => return Err(ParseError { line_number: current_line_display, message: format!("Unknown instruction '{}'", opcode)}),
        };
        instructions.push(instruction);
    }
    Ok(instructions)
}

/// Writes a vector of instructions to a .qexe binary file.
fn write_qexe_file(path: &str, instructions: &[Instruction]) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(b"QEXE")?; // Magic bytes
    file.write_all(&[1u8])?; // Version
    file.write_all(&(instructions.len() as u32).to_le_bytes())?;

    for inst in instructions {
        match inst {
            Instruction::QInit(reg) => file.write_all(&[opcodes::QINIT, *reg])?,
            Instruction::QGate(reg, name) => {
                let mut bytes = vec![opcodes::QGATE, *reg];
                let mut gate_bytes = name.clone().into_bytes();
                gate_bytes.resize(8, 0); // Pad/truncate to 8 bytes
                bytes.extend(gate_bytes);
                file.write_all(&bytes)?;
            }
            Instruction::QMeas(reg) => file.write_all(&[opcodes::QMEAS, *reg])?,
            Instruction::CharLoad(reg, val) => file.write_all(&[opcodes::CHARLOAD, *reg, *val])?,
            Instruction::RegSet(reg, val) => {
                file.write_all(&[opcodes::REGSET, *reg])?;
                file.write_all(&val.to_le_bytes())?;
            }
            Instruction::LoopStart(reg) => file.write_all(&[opcodes::LOOPSTART, *reg])?,
            Instruction::EndLoop => file.write_all(&[opcodes::ENDLOOP])?,
            Instruction::RZ(reg, angle) => {
                file.write_all(&[opcodes::RZ, *reg])?;
                file.write_all(&angle.to_le_bytes())?;
            }
            Instruction::CZ(control, target) => file.write_all(&[opcodes::CZ, *control, *target])?,
            Instruction::HALT => file.write_all(&[opcodes::HALT])?,
        }
    }
    Ok(())
}


// --- Main Application Logic ---

fn handle_compile_command(mut args: env::Args) -> io::Result<()> {
    let input = args.next().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Missing input file path for 'compile'"))?;
    let output = args.next().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Missing output file path for 'compile'"))?;
    
    println!("Compiling {} -> {}", input, output);
    let source = std::fs::read_to_string(&input)?;
    
    match compile_qoa_source(&source) {
        Ok(instructions) => {
            if instructions.is_empty() {
                println!("Warning: No instructions compiled from '{}'. Output file will be minimal.", input);
            }
            write_qexe_file(&output, &instructions)?;
            println!("Successfully compiled {} instructions to {}.", instructions.len(), output);
        }
        Err(e) => {
            eprintln!("Error on line {}: {}", e.line_number, e.message);
            std::process::exit(1);
        }
    }
    Ok(())
}

fn handle_run_command(mut args: env::Args) -> io::Result<()> {
    let mut input_path: Option<String> = None;
    let mut seed: Option<u64> = None;

    while let Some(arg) = args.next() {
        if arg == "--seed" {
            let seed_val_str = args.next().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "--seed requires a number"))?;
            seed = Some(seed_val_str.parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid seed value"))?);
        } else {
            input_path = Some(arg);
        }
    }

    let input = input_path.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Missing input file path for 'run'"))?;
    
    println!("Running emulator on {} with seed {:?}", input, seed);
    let mut file = File::open(&input)?;
    let mut header = [0u8; 9];
    file.read_exact(&mut header)?;

    if &header[0..4] != b"QEXE" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid file format: Magic bytes not recognized"));
    }
    
    let version = header[4];
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    match parse_instructions(&data) {
        Ok(instructions) => {
            println!("Loaded {} instructions (version {})", instructions.len(), version);
            if instructions.is_empty() {
                println!("No instructions to run. Exiting.");
                return Ok(());
            }
            let mut emulator = Emulator::new(seed);
            emulator.run(&instructions);
        }
        Err(e) => {
            eprintln!("Error decoding instruction at byte {}: {}", e.position, e.message);
            std::process::exit(1);
        }
    }
    Ok(())
}


fn main() {
    let mut args = env::args();
    if args.len() < 2 {
        eprintln!("Usage: qoa <command> [args]");
        eprintln!("Commands:");
        eprintln!("  compile <input.qoa> <output.qexe>");
        eprintln!("  run <input.qexe> [--seed <number>]");
        std::process::exit(1);
    }
    
    let _program_name = args.next(); // Skip program name
    let command = args.next().unwrap();

    let result = match command.as_str() {
        "compile" => handle_compile_command(args),
        "run" => handle_run_command(args),
        _ => {
            eprintln!("Unknown command: {}", command);
            std::process::exit(1);
        }
    };
    
    if let Err(e) = result {
        eprintln!("Application error: {}", e);
        std::process::exit(1);
    }
}