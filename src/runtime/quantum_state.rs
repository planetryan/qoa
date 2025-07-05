use crate::instructions::Instruction;
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*; // import rayon for parallel iterators
use serde::{Deserialize, Serialize}; // import serialize and deserialize

#[derive(Debug, Serialize, Deserialize, Clone)] // derive serialize and deserialize
pub enum NoiseConfig {
    Random,
    Fixed(f64),
    Ideal,
}

#[derive(Debug, Serialize, Deserialize, Clone)] // derive serialize and deserialize
pub struct Status {
    pub nan: bool,
    pub div_by_zero: bool,
    pub overflow: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)] // added Clone derive here
pub struct QuantumState {
    pub n: usize,
    pub amps: Vec<Complex64>,
    pub status: Status,
    pub noise_config: Option<NoiseConfig>,
    // ensure #[serde(skip_serializing)] is directly above this field
    #[serde(skip_serializing, skip_deserializing)] // skip both serializing and deserializing
    rng: Option<StdRng>,
}

impl QuantumState {
    pub fn new(n: usize, noise_config: Option<NoiseConfig>) -> Self {
        let mut amps = vec![Complex64::new(0.0, 0.0); 1 << n];
        amps[0] = Complex64::new(1.0, 0.0);

        // initialize rng from entropy. this is done once per new quantumstate instance.
        // for global, non-reproducible randomness, `rand::thread_rng()` could be used
        // if `quantumstate` didn't need to explicitly own its rng.
        let rng = Some(StdRng::from_entropy());

        QuantumState {
            n,
            amps,
            status: Status {
                nan: false,
                div_by_zero: false,
                overflow: false,
            },
            noise_config,
            rng,
        }
    }

    pub fn get(&self, index: usize) -> Option<&Complex64> {
        self.amps.get(index)
    }

    pub fn set(&mut self, index: usize, val: Complex64) -> Result<(), String> {
        if index >= self.amps.len() {
            return Err(format!("index {} out of bounds", index));
        }
        if val.re.is_nan() || val.im.is_nan() {
            self.status.nan = true;
        }
        if val.re.is_infinite() || val.im.is_infinite() {
            self.status.overflow = true;
        }
        self.amps[index] = val;
        Ok(())
    }

    // new method to get all probabilities without measuring
    pub fn get_probabilities(&self) -> Vec<f64> {
        self.amps.par_iter().map(|a| a.norm_sqr()).collect() // parallel map and collect
    }

    // adds a method to validate the quantum state
    pub fn validate_state(&self) -> Result<(), String> {
        if self.amps.is_empty() {
            return Err("quantum state amplitudes vector is empty.".to_string());
        }

        // use parallel iterators with `any` and `sum` for efficient checks
        let has_nan = self.amps.par_iter().any(|amp| amp.re.is_nan() || amp.im.is_nan());
        let has_inf = self.amps.par_iter().any(|amp| amp.re.is_infinite() || amp.im.is_infinite());
        let norm_sqr_sum: f64 = self.amps.par_iter().map(|amp| amp.norm_sqr()).sum();

        if has_nan {
            return Err("quantum state contains NaN values.".to_string());
        }
        if has_inf {
            return Err("quantum state contains infinite values.".to_string());
        }

        // check for normalization within a small epsilon
        if (norm_sqr_sum - 1.0).abs() > 1e-9 {
            return Err(format!(
                "quantum state is not normalized. norm squared: {}",
                norm_sqr_sum
            ));
        }

        Ok(())
    }

    pub fn execute_arithmetic(instr: &Instruction, state: &mut QuantumState) -> Result<(), String> {
        use Instruction::*;
        match instr {
            QINIT(n) => { // Corrected from QInit
                let current_noise_config = state.noise_config.clone();
                *state = QuantumState::new(*n as usize, current_noise_config);
                Ok(())
            }
            RESET(q) => { // Corrected from Reset
                if (*q as usize) >= state.n {
                    return Err(format!("reset qubit {} out of range", q));
                }
                let mask = 1 << q;
                // parallelize reset
                state.amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
                    if i & mask != 0 {
                        *amp = Complex64::new(0.0, 0.0);
                    }
                });
                state.normalize();
                Ok(())
            }
            RESETALL => {
                let current_noise_config = state.noise_config.clone();
                *state = QuantumState::new(state.n, current_noise_config);
                Ok(())
            }

            H(q) => { // Corrected from ApplyHadamard
                state.apply_h(*q as usize);
                Ok(())
            }
            APPLYBITFLIP(q) => { // Corrected from ApplyBitFlip
                state.apply_x(*q as usize);
                Ok(())
            }
            APPLYPHASEFLIP(q) => { // Corrected from ApplyPhaseFlip
                state.apply_phase_flip(*q as usize);
                Ok(())
            }
            APPLYTGATE(q) => { // Corrected from ApplyTGate
                state.apply_t_gate(*q as usize);
                Ok(())
            }
            APPLYSGATE(q) => { // Corrected from ApplySGate
                state.apply_s_gate(*q as usize);
                Ok(())
            }
            PHASESHIFT(q, angle) | SETPHASE(q, angle) => { // Corrected from PhaseShift, SetPhase
                state.apply_phase_shift(*q as usize, *angle);
                Ok(())
            }

            RX(q, angle) => {
                state.apply_rx(*q as usize, *angle as f64);
                Ok(())
            }
            RY(q, angle) => {
                state.apply_ry(*q as usize, *angle as f64);
                Ok(())
            }
            RZ(q, angle) => {
                state.apply_rz(*q as usize, *angle as f64);
                Ok(())
            }

            CONTROLLEDNOT(c, t) | CNOT(c, t) => { // Corrected from ControlledNot
                state.apply_cnot(*c as usize, *t as usize);
                Ok(())
            }
            CZ(c, t) => {
                state.apply_cz(*c as usize, *t as usize);
                Ok(())
            }
            CONTROLLEDPHASEROTATION(c, t, angle) | APPLYCPHASE(c, t, angle) => { // Corrected from ControlledPhaseRotation, ApplyCPhase
                state.apply_controlled_phase(*c as usize, *t as usize, *angle);
                Ok(())
            }

            ENTANGLE(c, t) => { // Corrected from Entangle
                state.apply_cnot(*c as usize, *t as usize);
                Ok(())
            }
            ENTANGLEBELL(q1, q2) => { // Corrected from EntangleBell
                state.apply_h(*q1 as usize);
                state.apply_cnot(*q1 as usize, *q2 as usize);
                Ok(())
            }
            ENTANGLEMULTI(qubits) => { // Corrected from EntangleMulti
                if qubits.is_empty() {
                    return Err("empty qubit list for entanglemulti".to_string());
                }
                let first = qubits[0];
                state.apply_h(first as usize);
                for &q in &qubits[1..] {
                    state.apply_cnot(first as usize, q as usize);
                }
                Ok(())
            }
            ENTANGLECLUSTER(qubits) => { // Corrected from EntangleCluster
                for &q in qubits {
                    state.apply_h(q as usize);
                }
                for pair in qubits.windows(2) {
                    state.apply_cz(pair[0] as usize, pair[1] as usize);
                }
                Ok(())
            }

            REGADD(dst, lhs, rhs) => { // Corrected from RegAdd
                let a = state.get(*lhs as usize).ok_or("invalid lhs register")?;
                let b = state.get(*rhs as usize).ok_or("invalid rhs register")?;
                let sum = *a + *b;
                if sum.re.is_nan() || sum.im.is_nan() {
                    state.status.nan = true;
                }
                state.set(*dst as usize, sum)?;
                Ok(())
            }
            REGSUB(dst, lhs, rhs) => { // Corrected from RegSub
                let a = state.get(*lhs as usize).ok_or("invalid lhs register")?;
                let b = state.get(*rhs as usize).ok_or("invalid rhs register")?;
                let diff = *a - *b;
                if diff.re.is_nan() || diff.im.is_nan() {
                    state.status.nan = true;
                }
                state.set(*dst as usize, diff)?;
                Ok(())
            }
            REGMUL(dst, lhs, rhs) => { // Corrected from RegMul
                let a = state.get(*lhs as usize).ok_or("invalid lhs register")?;
                let b = state.get(*rhs as usize).ok_or("invalid rhs register")?;
                let prod = *a * *b;
                if !prod.re.is_finite() || !prod.im.is_finite() {
                    state.status.overflow = true;
                }
                state.set(*dst as usize, prod)?;
                Ok(())
            }
            REGDIV(dst, lhs, rhs) => { // Corrected from RegDiv
                let a = state.get(*lhs as usize).ok_or("invalid lhs register")?;
                let b = state.get(*rhs as usize).ok_or("invalid rhs register")?;
                if b.re == 0.0 && b.im == 0.0 {
                    state.status.div_by_zero = true;
                    state.set(*dst as usize, Complex64::new(f64::NAN, 0.0))?;
                } else {
                    state.set(*dst as usize, *a / *b)?;
                }
                Ok(())
            }
            REGCOPY(dst, src) => { // Corrected from RegCopy
                let val = state.get(*src as usize).ok_or("invalid src register")?;
                state.set(*dst as usize, *val)?;
                Ok(())
            }
            _ => Err(format!(
                "instruction {:?} not implemented in execute_arithmetic",
                instr
            )),
        }
    }

    fn _apply_x_pure(&mut self, q: usize) {
        if q >= self.n {
            eprintln!(
                "error: qubit index {} out of bounds for {}-qubit state (_apply_x_pure)",
                q, self.n
            );
            return;
        }
        let mask = 1 << q;
        let _size = self.amps.len(); // renamed to _size to suppress warning
        let old_amps = &self.amps; // immutable reference for parallel reads

        // create a new vector to store results, then fill it in parallel
        let new_amps: Vec<Complex64> = (0..self.amps.len()).into_par_iter().map(|i| { // use self.amps.len() directly
            if i & mask == 0 { // process only if the q-th bit is 0
                let b_idx = i | mask; // the corresponding index where q-th bit is 1
                old_amps[b_idx] // new value for current index `i` is old value at `b_idx`
            } else { // process if the q-th bit is 1
                let a_idx = i ^ mask; // the corresponding index where q-th bit is 0
                old_amps[a_idx] // new value for current index `i` is old value at `a_idx`
            }
        }).collect();
        self.amps = new_amps;
    }

    fn _apply_y_pure(&mut self, q: usize) {
        if q >= self.n {
            eprintln!(
                "error: qubit index {} out of bounds for {}-qubit state (_apply_y_pure)",
                q, self.n
            );
            return;
        }
        let mask = 1 << q;
        let i_comp = Complex64::new(0.0, 1.0);
        let neg_i_comp = Complex64::new(0.0, -1.0);

        let _size = self.amps.len(); // renamed to _size to suppress warning
        let old_amps = &self.amps; // immutable reference for parallel reads

        let new_amps: Vec<Complex64> = (0..self.amps.len()).into_par_iter().map(|i| { // use self.amps.len() directly
            if (i & mask) == 0 { // process if the q-th bit is 0
                let flipped = i | mask;
                neg_i_comp * old_amps[flipped] // new value for current index `i`
            } else { // process if the q-th bit is 1
                let original = i ^ mask;
                i_comp * old_amps[original] // new value for current index `i`
            }
        }).collect();
        self.amps = new_amps;
    }

    fn _apply_z_pure(&mut self, q: usize) {
        if q >= self.n {
            eprintln!("error: qubit index {} out of bounds for _apply_z_pure", q);
            return;
        }
        let mask = 1 << q;
        // parallelize the z gate operation
        self.amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            if (i & mask) != 0 {
                *amp = -*amp;
            }
        });
    }

    fn apply_noise(&mut self) {
        if let Some(config) = &self.noise_config {
            // temporarily take ownership of the rng from self.
            // this ensures `self` is not mutably borrowed by `rng` during the loop
            // when `_apply_*_pure` methods (which also borrow `self` mutably) are called.
            let mut temp_rng = self // this needs to be mutable because its methods are called mutably
                .rng
                .take()
                .expect("rng should be initialized for noise application in apply_noise");

            match config {
                NoiseConfig::Random => {
                    let noise_level = temp_rng.gen_range(0.1..1.0);
                    // pass the temporarily held rng to apply_depolarizing_noise
                    self.apply_depolarizing_noise(noise_level, &mut temp_rng);
                }
                NoiseConfig::Fixed(value) => {
                    if *value > 0.0 {
                        // pass the temporarily held rng to apply_depolarizing_noise
                        self.apply_depolarizing_noise(*value, &mut temp_rng);
                    }
                }
                NoiseConfig::Ideal => {
                    // no noise applied in ideal mode
                }
            }
            // put the rng back into self after all operations are done
            self.rng = Some(temp_rng);
        }
        self.normalize();
    }

    // `apply_depolarizing_noise` now accepts a mutable reference to the rng
    fn apply_depolarizing_noise(&mut self, p: f64, rng: &mut StdRng) {
        for q_idx in 0..self.n {
            let r: f64 = rng.gen(); // use the passed-in mutable rng reference
            if r < p {
                let error_type = rng.gen_range(0..3);
                match error_type {
                    0 => self._apply_x_pure(q_idx),
                    1 => self._apply_y_pure(q_idx),
                    2 => self._apply_z_pure(q_idx),
                    _ => unreachable!(),
                }
            }
        }
    }

    fn normalize(&mut self) {
        let norm_sqr: f64 = self.amps.par_iter().map(|a| a.norm_sqr()).sum(); // parallel sum
        if norm_sqr > 1e-12 {
            let norm = norm_sqr.sqrt();
            self.amps.par_iter_mut().for_each(|amp| { // parallel division
                *amp /= norm;
            });
        } else {
            eprintln!("warning: state became zero due to noise or numerical instability, resetting to |0...0>.");
            self.amps
                .par_iter_mut() // parallel reset
                .for_each(|amp| *amp = Complex64::new(0.0, 0.0));
            self.amps[0] = Complex64::new(1.0, 0.0);
        }
    }

    pub fn apply_h(&mut self, q: usize) {
        if q >= self.n {
            eprintln!(
                "error: qubit index {} out of bounds for {}-qubit state",
                q, self.n
            );
            return;
        }
        let mask = 1 << q;
        let norm_factor = Complex64::new(1.0 / 2f64.sqrt(), 0.0);
        let _size = self.amps.len(); // renamed to _size to suppress warning
        let old_amps = &self.amps; // immutable reference for parallel reads
        // use into_par_iter().map().collect() to create the new_amps vector in parallel
        let new_amps: Vec<Complex64> = (0..self.amps.len()).into_par_iter().map(|i| { // use self.amps.len() directly
            if i & mask == 0 { // process if the q-th bit is 0
                let flipped_idx = i | mask; // the corresponding index where q-th bit is 1
                let a_val = old_amps[i];
                let b_val = old_amps[flipped_idx];
                norm_factor * (a_val + b_val) // new value for current index `i`
            } else { // process if the q-th bit is 1
                let original_idx = i ^ mask; // the corresponding index where q-th bit is 0
                let a_val = old_amps[original_idx];
                let b_val = old_amps[i];
                norm_factor * (a_val - b_val) // new value for current index `i`
            }
        }).collect();
        self.amps = new_amps; // replace the original amplitudes with the new ones
        self.apply_noise();
    }

    pub fn apply_x(&mut self, q: usize) {
        self._apply_x_pure(q);
        self.apply_noise();
    }

    pub fn apply_y(&mut self, q: usize) {
        self._apply_y_pure(q);
        self.apply_noise();
    }

    pub fn apply_phase_flip(&mut self, q: usize) {
        self._apply_z_pure(q);
        self.apply_noise();
    }

    pub fn apply_cz(&mut self, c: usize, t: usize) {
        if c >= self.n || t >= self.n {
            eprintln!(
                "error: control ({}) or target ({}) qubit index out of bounds for {}-qubit state",
                c, t, self.n
            );
            return;
        }
        if c == t {
            eprintln!("error: control and target qubits cannot be the same for cz gate.");
            return;
        }

        let c_mask = 1 << c;
        let t_mask = 1 << t;

        // parallelize the cz gate operation
        self.amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            // apply a phase flip if both control and target qubits are in the |1> state
            if (i & c_mask != 0) && (i & t_mask != 0) {
                *amp = -*amp;
            }
        });
        self.apply_noise();
    }

    pub fn apply_cnot(&mut self, c: usize, t: usize) {
        if c >= self.n || t >= self.n {
            eprintln!(
                "error: control ({}) or target ({}) qubit index out of bounds for {}-qubit state",
                c, t, self.n
            );
            return;
        }
        if c == t {
            eprintln!("error: control and target qubits cannot be the same for cnot gate.");
            return;
        }

        let c_mask = 1 << c;
        let t_mask = 1 << t;
        let _size = self.amps.len(); // renamed to _size to suppress warning
        let old_amps = self.amps.clone(); // clone for safe parallel reads

        self.amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            if (i & c_mask) != 0 {
                // if control qubit is 1
                if (i & t_mask) == 0 {
                    // if target qubit is 0, flip it to 1
                    let flipped_idx = i | t_mask;
                    *amp = old_amps[flipped_idx];
                } else {
                    // if target qubit is 1, flip it to 0
                    let flipped_idx = i ^ t_mask;
                    *amp = old_amps[flipped_idx];
                }
            } else {
                // if control qubit is 0, do nothing to target
                *amp = old_amps[i];
            }
        });
        self.apply_noise();
    }

    pub fn apply_phase_shift(&mut self, q: usize, angle: f64) {
        if q >= self.n {
            eprintln!(
                "error: qubit index {} out of bounds for {}-qubit state",
                q, self.n
            );
            return;
        }
        let mask = 1 << q;
        let phase_factor = Complex64::new(0.0, angle).exp(); // e^(i * angle)

        // parallelize the phase shift operation
        self.amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            if (i & mask) != 0 {
                // apply phase factor if the qubit is in the |1> state
                *amp *= phase_factor;
            }
        });
        self.apply_noise();
    }

    pub fn apply_t_gate(&mut self, q: usize) {
        self.apply_phase_shift(q, std::f64::consts::FRAC_PI_4);
    }

    pub fn apply_s_gate(&mut self, q: usize) {
        self.apply_phase_shift(q, std::f64::consts::FRAC_PI_2);
    }

    pub fn apply_rx(&mut self, q: usize, angle: f64) {
        if q >= self.n {
            eprintln!(
                "error: qubit index {} out of bounds for {}-qubit state",
                q, self.n
            );
            return;
        }
        let mask = 1 << q;
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        let i_sin_half = Complex64::new(0.0, -sin_half);

        let _size = self.amps.len(); // renamed to _size to suppress warning
        let old_amps = self.amps.clone(); // clone for safe parallel reads

        self.amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            if (i & mask) == 0 {
                // if q-th bit is 0
                let flipped_idx = i | mask;
                *amp = cos_half * old_amps[i] + i_sin_half * old_amps[flipped_idx];
            } else {
                // if q-th bit is 1
                let original_idx = i ^ mask;
                *amp = i_sin_half * old_amps[original_idx] + cos_half * old_amps[i];
            }
        });
        self.apply_noise();
    }

    pub fn apply_ry(&mut self, q: usize, angle: f64) {
        if q >= self.n {
            eprintln!(
                "error: qubit index {} out of bounds for {}-qubit state",
                q, self.n
            );
            return;
        }
        let mask = 1 << q;
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        let neg_sin_half = -sin_half;

        let _size = self.amps.len(); // renamed to _size to suppress warning
        let old_amps = self.amps.clone(); // clone for safe parallel reads

        self.amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            if (i & mask) == 0 {
                // if q-th bit is 0
                let flipped_idx = i | mask;
                *amp = cos_half * old_amps[i] + neg_sin_half * old_amps[flipped_idx];
            } else {
                // if q-th bit is 1
                let original_idx = i ^ mask;
                *amp = sin_half * old_amps[original_idx] + cos_half * old_amps[i];
            }
        });
        self.apply_noise();
    }

    pub fn apply_rz(&mut self, q: usize, angle: f64) {
        if q >= self.n {
            eprintln!(
                "error: qubit index {} out of bounds for {}-qubit state",
                q, self.n
            );
            return;
        }
        let mask = 1 << q;
        let phase_factor_0 = Complex64::new(0.0, -angle / 2.0).exp();
        let phase_factor_1 = Complex64::new(0.0, angle / 2.0).exp();

        // parallelize the rz gate operation
        self.amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            if (i & mask) == 0 {
                // if q-th bit is 0
                *amp *= phase_factor_0;
            } else {
                // if q-th bit is 1
                *amp *= phase_factor_1;
            }
        });
        self.apply_noise();
    }

    pub fn apply_controlled_phase(&mut self, c: usize, t: usize, angle: f64) {
        if c >= self.n || t >= self.n {
            eprintln!(
                "error: control ({}) or target ({}) qubit index out of bounds for {}-qubit state",
                c, t, self.n
            );
            return;
        }
        if c == t {
            eprintln!("error: control and target qubits cannot be the same for controlled phase gate.");
            return;
        }

        let c_mask = 1 << c;
        let t_mask = 1 << t;
        let phase_factor = Complex64::new(0.0, angle).exp(); // e^(i * angle)

        // parallelize the controlled phase operation
        self.amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            // apply phase factor if both control and target qubits are in the |1> state
            if (i & c_mask != 0) && (i & t_mask != 0) {
                *amp *= phase_factor;
            }
        });
        self.apply_noise();
    }

    pub fn measure(&mut self, q: usize) -> u8 {
        if q >= self.n {
            eprintln!(
                "error: qubit index {} out of bounds for {}-qubit state (measure)",
                q, self.n
            );
            return 0; // return 0 on error
        }

        let mask = 1 << q;
        let prob_zero_sq: f64 = self
            .amps
            .par_iter()
            .enumerate()
            .filter(|(i, _)| (i & mask) == 0) // filter states where q-th bit is 0
            .map(|(_, amp)| amp.norm_sqr())
            .sum();

        let mut temp_rng = self
            .rng
            .take()
            .expect("rng should be initialized for measurement.");
        let random_val: f64 = temp_rng.gen();
        self.rng = Some(temp_rng); // put rng back

        let result = if random_val < prob_zero_sq {
            // collapse to |0> state for the measured qubit
            self.amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
                if (i & mask) != 0 {
                    *amp = Complex64::new(0.0, 0.0);
                }
            });
            0
        } else {
            // collapse to |1> state for the measured qubit
            self.amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
                if (i & mask) == 0 {
                    *amp = Complex64::new(0.0, 0.0);
                }
            });
            1
        };
        self.normalize(); // re-normalize after collapse
        self.apply_noise(); // apply noise after measurement
        result
    }

    pub fn sample_measurement(&self) -> Vec<u8> {
        let mut result_bits = vec![0u8; self.n];
        // self.clone() now correctly creates a new, independent QuantumState instance
        let mut temp_state = self.clone(); 

        for q_idx in 0..self.n {
            result_bits[q_idx] = temp_state.measure(q_idx);
        }
        result_bits
    }

    pub fn apply_final_state_noise(&mut self) {
        // only apply final noise if not in ideal mode
        if let Some(NoiseConfig::Ideal) = &self.noise_config {
            eprintln!("[info] final state noise is skipped in ideal mode.");
            return;
        }

        eprintln!("[info] applying final state amplitude randomization.");

        // temporarily take the rng for use. it doesn't need to be mutable here
        // because its methods are not called directly after `take()`.
        let temp_rng = self // removed 'mut' here, as it's not needed
            .rng
            .take()
            .expect("rng should be initialized for final state noise.");

        let noise_strength = 0.1; // noise strength

        // parallelize final state noise application
        self.amps.par_iter_mut().for_each(|amp| {
            // each thread gets its own local rng for thread-safe random number generation
            let mut local_rng = rand::thread_rng();
            // add small random gaussian noise to real and imaginary parts
            let random_re: f64 =
                <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut local_rng)
                    * noise_strength;
            let random_im: f64 =
                <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut local_rng)
                    * noise_strength;
            *amp += Complex64::new(random_re, random_im);
        });

        // put the rng back
        self.rng = Some(temp_rng);
        self.normalize(); // ensure the state is still normalized after randomization
    }
}
