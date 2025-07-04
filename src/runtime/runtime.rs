/*
    QPU MAIN RUNTIME, WRITTEN ON 20/6/2025 BY RAYAN
*/

use rand::thread_rng;
use rand::Rng;
mod quantum_state;
pub use quantum_state::QuantumState;

/// QPU exception flags (NaN, overflow, divide by zero)
#[derive(Default, Debug)]
pub struct QStatus {
    pub nan: bool,
    pub overflow: bool,
    pub div_by_zero: bool,
}

impl QStatus {
    pub fn clear(&mut self) {
        *self = QStatus::default();
    }

    pub fn to_bits(&self) -> u8 {
        (self.nan as u8) | ((self.overflow as u8) << 1) | ((self.div_by_zero as u8) << 2)
    }
}

/// Represents the classical register bank and status flags of the QPU.
pub struct RegisterBank {
    pub regs: [f64; 16],
    pub status: QStatus,
    // (program counter of LoopStart, remaining iterations)
    pub loop_stack: Vec<(usize, u8)>,
}

impl RegisterBank {
    pub fn new() -> Self {
        RegisterBank {
            regs: [0.0; 16],
            status: QStatus::default(),
            loop_stack: Vec::new(),
        }
    }

    fn check_reg_idx(idx: usize) -> Result<(), String> {
        if idx < 16 {
            Ok(())
        } else {
            Err(format!("Register index out of bounds: {}", idx))
        }
    }

    pub fn get(&self, idx: usize) -> Result<f64, String> {
        Self::check_reg_idx(idx)?;
        Ok(self.regs[idx])
    }

    pub fn set(&mut self, idx: usize, val: f64) -> Result<(), String> {
        Self::check_reg_idx(idx)?;
        self.regs[idx] = val;
        // When a register is successfully set, clear the status flags as a new, valid operation has occurred.
        self.status.clear();
        Ok(())
    }

    pub fn reg_add(&mut self, rd: usize, ra: usize, rb: usize) -> Result<(), String> {
        self.status.clear(); // Clear flags before operation
        let a = self.get(ra)?;
        let b = self.get(rb)?;

        if a.is_nan() || b.is_nan() {
            self.status.nan = true;
            return self.set(rd, f64::NAN);
        }

        let sum = a + b;
        if sum.is_infinite() {
            self.status.overflow = true;
        } else if sum.is_nan() {
            // Catches cases like Inf - Inf
            self.status.nan = true;
        }
        self.set(rd, sum)
    }

    pub fn reg_sub(&mut self, rd: usize, ra: usize, rb: usize) -> Result<(), String> {
        self.status.clear();
        let a = self.get(ra)?;
        let b = self.get(rb)?;

        if a.is_nan() || b.is_nan() {
            self.status.nan = true;
            return self.set(rd, f64::NAN);
        }

        let diff = a - b;
        if diff.is_infinite() {
            self.status.overflow = true;
        } else if diff.is_nan() {
            // Catches cases like Inf - Inf
            self.status.nan = true;
        }
        self.set(rd, diff)
    }

    pub fn reg_mul(&mut self, rd: usize, ra: usize, rb: usize) -> Result<(), String> {
        self.status.clear();
        let a = self.get(ra)?;
        let b = self.get(rb)?;

        if a.is_nan() || b.is_nan() {
            self.status.nan = true;
            return self.set(rd, f64::NAN);
        }

        let prod = a * b;
        if prod.is_infinite() {
            self.status.overflow = true;
        } else if prod.is_nan() {
            // Catches cases like 0 * Inf
            self.status.nan = true;
        }
        self.set(rd, prod)
    }

    pub fn reg_div(&mut self, rd: usize, ra: usize, rb: usize) -> Result<(), String> {
        self.status.clear();
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
        if quot.is_infinite() {
            self.status.overflow = true;
        } else if quot.is_nan() {
            // Catches 0/0
            self.status.nan = true;
        }
        self.set(rd, quot)
    }

    pub fn reg_copy(&mut self, rd: usize, ra: usize) -> Result<(), String> {
        self.status.clear();
        let val = self.get(ra)?;
        self.set(rd, val)
    }

    pub fn rand(&mut self, rd: usize) -> Result<(), String> {
        self.status.clear();
        self.set(rd, thread_rng().gen::<f64>())
    }

    pub fn sqrt(&mut self, rd: usize, ra: usize) -> Result<(), String> {
        self.status.clear();
        let val = self.get(ra)?;
        if val.is_nan() {
            self.status.nan = true;
            return self.set(rd, f64::NAN);
        }
        if val < 0.0 {
            self.status.nan = true;
            return self.set(rd, f64::NAN);
        }
        self.set(rd, val.sqrt())
    }

    pub fn exp(&mut self, rd: usize, ra: usize) -> Result<(), String> {
        self.status.clear();
        let val = self.get(ra)?;
        if val.is_nan() {
            self.status.nan = true;
            return self.set(rd, f64::NAN);
        }
        let result = val.exp();
        if result.is_infinite() {
            self.status.overflow = true;
        } else if result.is_nan() {
            self.status.nan = true;
        }
        self.set(rd, result)
    }

    pub fn ln(&mut self, rd: usize, ra: usize) -> Result<(), String> {
        self.status.clear();
        let val = self.get(ra)?;
        if val.is_nan() {
            self.status.nan = true;
            return self.set(rd, f64::NAN);
        }
        if val <= 0.0 {
            self.status.nan = true;
            return self.set(rd, f64::NAN);
        }
        self.set(rd, val.ln())
    }

    /// Push a new loop frame onto the stack.
    pub fn push_loop_frame(&mut self, pc: usize, times: u8) {
        self.loop_stack.push((pc, times));
        self.status.clear();
    }

    /// Pop a loop frame from the stack, decrementing count if necessary.
    /// Returns (start_pc, remaining_times) if the loop continues, or None if finished.
    pub fn pop_loop_frame(&mut self) -> Option<(usize, u8)> {
        self.status.clear();
        if let Some((pc, mut times)) = self.loop_stack.pop() {
            if times > 1 {
                times -= 1;
                self.loop_stack.push((pc, times)); // Push back with decremented count
                Some((pc, times))
            } else {
                None // Loop finished
            }
        } else {
            self.status.nan = true; // Indicate an error state
            None // No loop frame to pop
        }
    }

    pub fn execute_arithmetic(
        &mut self,
        instr: &crate::instructions::Instruction,
    ) -> Result<(), String> {
        use crate::instructions::Instruction;
        match instr {
            Instruction::RegAdd(rd, ra, rb) => {
                self.reg_add(*rd as usize, *ra as usize, *rb as usize)
            }
            Instruction::RegSub(rd, ra, rb) => {
                self.reg_sub(*rd as usize, *ra as usize, *rb as usize)
            }
            Instruction::RegMul(rd, ra, rb) => {
                self.reg_mul(*rd as usize, *ra as usize, *rb as usize)
            }
            Instruction::RegDiv(rd, ra, rb) => {
                self.reg_div(*rd as usize, *ra as usize, *rb as usize)
            }
            Instruction::RegCopy(rd, ra) => self.reg_copy(*rd as usize, *ra as usize),
            Instruction::Rand(rd) => self.rand(*rd as usize),
            Instruction::Sqrt(rd, ra) => self.sqrt(*rd as usize, *ra as usize),
            Instruction::Exp(rd, ra) => self.exp(*rd as usize, *ra as usize),
            Instruction::Log(rd, ra) => self.ln(*rd as usize, *ra as usize),
            _ => Err(format!(
                "Unsupported arithmetic/runtime instruction for RegisterBank: {:?}",
                instr
            )),
        }
    }
}
