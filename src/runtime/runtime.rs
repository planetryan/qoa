use rand::thread_rng;
use rand::Rng;
mod quantum_state;
pub use quantum_state::QuantumState;

// qpu exception flags (nan, overflow, divide by zero)
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

// represents the classical register bank and status flags of the qpu.
pub struct RegisterBank {
    pub regs: [f64; 16],
    pub status: QStatus,
    // (program counter of loopstart, remaining iterations)
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
            Err(format!("register index out of bounds: {}", idx))
        }
    }

    pub fn get(&self, idx: usize) -> Result<f64, String> {
        Self::check_reg_idx(idx)?;
        Ok(self.regs[idx])
    }

    pub fn set(&mut self, idx: usize, val: f64) -> Result<(), String> {
        Self::check_reg_idx(idx)?;
        self.regs[idx] = val;
        // when a register is successfully set, clear the status flags as a new, valid operation has occurred.
        self.status.clear();
        Ok(())
    }

    pub fn reg_add(&mut self, rd: usize, ra: usize, rb: usize) -> Result<(), String> {
        self.status.clear(); // clear flags before operation
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
            // catches cases like inf - inf
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
            // catches cases like inf - inf
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
            // catches cases like 0 * inf
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
            // catches 0/0
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

    // push a new loop frame onto the stack.
    pub fn push_loop_frame(&mut self, pc: usize, times: u8) {
        self.loop_stack.push((pc, times));
        self.status.clear();
    }

    // pop a loop frame from the stack, decrementing count if necessary.
    // returns (start_pc, remaining_times) if the loop continues, or none if finished.
    pub fn pop_loop_frame(&mut self) -> Option<(usize, u8)> {
        self.status.clear();
        if let Some((pc, mut times)) = self.loop_stack.pop() {
            if times > 1 {
                times -= 1;
                self.loop_stack.push((pc, times)); // push back with decremented count
                Some((pc, times))
            } else {
                None // loop finished
            }
        } else {
            self.status.nan = true; // indicate an error state
            None // no loop frame to pop
        }
    }

    pub fn execute_arithmetic(
        &mut self,
        instr: &crate::instructions::Instruction,
    ) -> Result<(), String> {
        use crate::instructions::Instruction;
        match instr {
            Instruction::REGADD(rd, ra, rb) | Instruction::RADD(rd, ra, rb) => {
                self.reg_add(*rd as usize, *ra as usize, *rb as usize)
            }
            Instruction::REGSUB(rd, ra, rb) | Instruction::RSUB(rd, ra, rb) => {
                self.reg_sub(*rd as usize, *ra as usize, *rb as usize)
            }
            Instruction::REGMUL(rd, ra, rb) | Instruction::RMUL(rd, ra, rb) => {
                self.reg_mul(*rd as usize, *ra as usize, *rb as usize)
            }
            Instruction::REGDIV(rd, ra, rb) | Instruction::RDIV(rd, ra, rb) => {
                self.reg_div(*rd as usize, *ra as usize, *rb as usize)
            }
            Instruction::REGCOPY(rd, ra) | Instruction::RCOPY(rd, ra) => self.reg_copy(*rd as usize, *ra as usize),
            Instruction::RAND(rd) => self.rand(*rd as usize),
            Instruction::SQRT(rd, ra) => self.sqrt(*rd as usize, *ra as usize),
            Instruction::EXP(rd, ra) => self.exp(*rd as usize, *ra as usize),
            Instruction::LOG(rd, ra) => self.ln(*rd as usize, *ra as usize),
            // bitwise operations
            Instruction::ANDBITS(rd, ra, rb) | Instruction::ANDB(rd, ra, rb) => {
                self.status.clear();
                let a = self.get(ra)? as u64; // convert f64 to u64 for bitwise ops
                let b = self.get(rb)? as u64;
                self.set(*rd as usize, (a & b) as f64)
            }
            Instruction::ORBITS(rd, ra, rb) | Instruction::ORB(rd, ra, rb) => {
                self.status.clear();
                let a = self.get(ra)? as u64;
                let b = self.get(rb)? as u64;
                self.set(*rd as usize, (a | b) as f64)
            }
            Instruction::XORBITS(rd, ra, rb) | Instruction::XORB(rd, ra, rb) => {
                self.status.clear();
                let a = self.get(ra)? as u64;
                let b = self.get(rb)? as u64;
                self.set(*rd as usize, (a ^ b) as f64)
            }
            Instruction::NOTBITS(rd, ra) | Instruction::NOTB(rd, ra) => {
                self.status.clear();
                let a = self.get(ra)? as u64;
                self.set(*rd as usize, (!a) as f64) // bitwise not
            }
            Instruction::SHL(rd, ra, rb) => {
                self.status.clear();
                let a = self.get(ra)? as u64;
                let b = self.get(rb)? as u32; // shift amount is u32
                self.set(*rd as usize, (a << b) as f64)
            }
            Instruction::SHR(rd, ra, rb) => {
                self.status.clear();
                let a = self.get(ra)? as u64;
                let b = self.get(rb)? as u32; // shift amount is u32
                self.set(*rd as usize, (a >> b) as f64)
            }
            // stub implementations for control flow instructions
            Instruction::IFGT(_, _, _) | Instruction::IGT(_, _, _) => {
                Err("control flow instruction ifgt/igt not handled by registerbank".into())
            }
            Instruction::IFLT(_, _, _) | Instruction::ILT(_, _, _) => {
                Err("control flow instruction iflt/ilt not handled by registerbank".into())
            }
            Instruction::IFEQ(_, _, _) | Instruction::IEQ(_, _, _) => {
                Err("control flow instruction ifeq/ieq not handled by registerbank".into())
            }
            Instruction::IFNE(_, _, _) | Instruction::INE(_, _, _) => {
                Err("control flow instruction ifne/ine not handled by registerbank".into())
            }
            _ => Err(format!(
                "unsupported arithmetic/runtime instruction for registerbank: {:?}",
                instr
            )),
        }
    }
}
