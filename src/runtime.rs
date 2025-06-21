/*
	QPU MAIN RUNTIME, WRITTEN ON 20/6/2025 BY RAYAN
*/

use crate::instructions::Instruction;

// QPU exception flags (NaN, overflow, divide-by-zero)
#[derive(Default, Debug)]
pub struct QStatus {
    pub nan: bool,
    pub overflow: bool,
    pub div_by_zero: bool,
}

impl QStatus {
    // Clear all exception flags
    pub fn clear(&mut self) {
        *self = QStatus::default();
    }

    // Pack flags into a bitmask for QGETFLAGS
    pub fn to_bits(&self) -> u8 {
        (self.nan as u8)
            | ((self.overflow as u8) << 1)
            | ((self.div_by_zero as u8) << 2)
    }
}

// QuantumState holds floating-point registers and status flags
pub struct QuantumState {
    pub regs: [f64; 16],  // R0..R15
    pub status: QStatus,
}

impl QuantumState {
    // Create a new QuantumState with zeroed registers and flags
    pub fn new() -> Self {
        QuantumState {
            regs: [0.0; 16],
            status: QStatus::default(),
        }
    }

    // Bounds-checked register access
    fn check_reg(idx: usize) -> Result<(), String> {
        if idx < 16 {
            Ok(())
        } else {
            Err(format!("Register index out of bounds: {}", idx))
        }
    }

    // Read a register
    pub fn get(&self, idx: usize) -> Result<f64, String> {
        Self::check_reg(idx)?;
        Ok(self.regs[idx])
    }

    // Write a register
    pub fn set(&mut self, idx: usize, val: f64) -> Result<(), String> {
        Self::check_reg(idx)?;
        self.regs[idx] = val;
        Ok(())
    }
}

impl QuantumState {
    // REGADD: rd = ra + rb
    pub fn reg_add(&mut self, rd: usize, ra: usize, rb: usize) -> Result<(), String> {
        let a = self.get(ra)?;
        let b = self.get(rb)?;
        // NaN propagation
        if a.is_nan() || b.is_nan() {
            self.status.nan = true;
            return self.set(rd, f64::NAN);
        }
        let sum = a + b;
        if !sum.is_finite() {
            self.status.overflow = true;
        }
        self.set(rd, sum)
    }

    // REGSUB: rd = ra - rb
    pub fn reg_sub(&mut self, rd: usize, ra: usize, rb: usize) -> Result<(), String> {
        let a = self.get(ra)?;
        let b = self.get(rb)?;
        if a.is_nan() || b.is_nan() {
            self.status.nan = true;
            return self.set(rd, f64::NAN);
        }
        let diff = a - b;
        if !diff.is_finite() {
            self.status.overflow = true;
        }
        self.set(rd, diff)
    }

    // REGMUL: rd = ra * rb
    pub fn reg_mul(&mut self, rd: usize, ra: usize, rb: usize) -> Result<(), String> {
        let a = self.get(ra)?;
        let b = self.get(rb)?;
        if a.is_nan() || b.is_nan() {
            self.status.nan = true;
            return self.set(rd, f64::NAN);
        }
        let prod = a * b;
        if !prod.is_finite() {
            self.status.overflow = true;
        }
        self.set(rd, prod)
    }

    // REGDIV: rd = ra / rb
    pub fn reg_div(&mut self, rd: usize, ra: usize, rb: usize) -> Result<(), String> {
        let a = self.get(ra)?;
        let b = self.get(rb)?;
        if b == 0.0 {
            self.status.div_by_zero = true;
            return self.set(rd, f64::NAN);
        }
        if a.is_nan() || b.is_nan() {
            self.status.nan = true;
            return self.set(rd, f64::NAN);
        }
        let quot = a / b;
        if !quot.is_finite() {
            self.status.overflow = true;
        }
        self.set(rd, quot)
    }

    // REGCOPY: rd = ra
    pub fn reg_copy(&mut self, rd: usize, ra: usize) -> Result<(), String> {
        let val = self.get(ra)?;
        self.set(rd, val)
    }

    pub fn execute_arithmetic(instr: &Instruction, state: &mut QuantumState) -> Result<(), String> {
    match instr {
        Instruction::RegAdd(rd, ra, rb) => state.reg_add(*rd as usize, *ra as usize, *rb as usize),
        Instruction::RegSub(rd, ra, rb) => state.reg_sub(*rd as usize, *ra as usize, *rb as usize),
        Instruction::RegMul(rd, ra, rb) => state.reg_mul(*rd as usize, *ra as usize, *rb as usize),
        Instruction::RegDiv(rd, ra, rb) => state.reg_div(*rd as usize, *ra as usize, *rb as usize),
        Instruction::RegCopy(rd, ra) => state.reg_copy(*rd as usize, *ra as usize),
        _ => Err("Unsupported instruction".to_string()),
    }
}
}
