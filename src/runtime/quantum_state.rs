use crate::instructions::Instruction;
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

#[derive(Debug, Clone)]
pub enum NoiseConfig {
    Random,
    Fixed(f64),
    Ideal,
}

pub struct Status {
    pub nan: bool,
    pub div_by_zero: bool,
    pub overflow: bool,
}

pub struct QuantumState {
    pub n: usize,
    pub amps: Vec<Complex64>,
    pub status: Status,
    pub noise_config: Option<NoiseConfig>,
    rng: Option<StdRng>,
}

impl QuantumState {
    pub fn new(n: usize, noise_config: Option<NoiseConfig>) -> Self {
        if n > 16 {
            eprintln!("warning: simulating more than 16 qubits can be very memory intensive.");
        }
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

    pub fn execute_arithmetic(instr: &Instruction, state: &mut QuantumState) -> Result<(), String> {
        use Instruction::*;
        match instr {
            QInit(n) => {
                let current_noise_config = state.noise_config.clone();
                *state = QuantumState::new(*n as usize, current_noise_config);
                Ok(())
            }
            Reset(q) => {
                if (*q as usize) >= state.n {
                    return Err(format!("reset qubit {} out of range", q));
                }
                let mask = 1 << q;
                for i in 0..state.amps.len() {
                    if i & mask != 0 {
                        state.amps[i] = Complex64::new(0.0, 0.0);
                    }
                }
                state.normalize();
                Ok(())
            }
            ResetAll => {
                let current_noise_config = state.noise_config.clone();
                *state = QuantumState::new(state.n, current_noise_config);
                Ok(())
            }

            ApplyHadamard(q) => {
                state.apply_h(*q as usize);
                Ok(())
            }
            ApplyBitFlip(q) => {
                state.apply_x(*q as usize);
                Ok(())
            }
            ApplyPhaseFlip(q) => {
                state.apply_phase_flip(*q as usize);
                Ok(())
            }
            ApplyTGate(q) => {
                state.apply_t_gate(*q as usize);
                Ok(())
            }
            ApplySGate(q) => {
                state.apply_s_gate(*q as usize);
                Ok(())
            }
            PhaseShift(q, angle) | SetPhase(q, angle) => {
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

            ControlledNot(c, t) | CNOT(c, t) => {
                state.apply_cnot(*c as usize, *t as usize);
                Ok(())
            }
            CZ(c, t) => {
                state.apply_cz(*c as usize, *t as usize);
                Ok(())
            }
            ControlledPhaseRotation(c, t, angle) | ApplyCPhase(c, t, angle) => {
                state.apply_controlled_phase(*c as usize, *t as usize, *angle);
                Ok(())
            }

            Entangle(c, t) => {
                state.apply_cnot(*c as usize, *t as usize);
                Ok(())
            }
            EntangleBell(q1, q2) => {
                state.apply_h(*q1 as usize);
                state.apply_cnot(*q1 as usize, *q2 as usize);
                Ok(())
            }
            EntangleMulti(qubits) => {
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
            EntangleCluster(qubits) => {
                for &q in qubits {
                    state.apply_h(q as usize);
                }
                for pair in qubits.windows(2) {
                    state.apply_cz(pair[0] as usize, pair[1] as usize);
                }
                Ok(())
            }

            RegAdd(dst, lhs, rhs) => {
                let a = state.get(*lhs as usize).ok_or("invalid lhs register")?;
                let b = state.get(*rhs as usize).ok_or("invalid rhs register")?;
                let sum = *a + *b;
                if sum.re.is_nan() || sum.im.is_nan() {
                    state.status.nan = true;
                }
                state.set(*dst as usize, sum)?;
                Ok(())
            }
            RegSub(dst, lhs, rhs) => {
                let a = state.get(*lhs as usize).ok_or("invalid lhs register")?;
                let b = state.get(*rhs as usize).ok_or("invalid rhs register")?;
                let diff = *a - *b;
                if diff.re.is_nan() || diff.im.is_nan() {
                    state.status.nan = true;
                }
                state.set(*dst as usize, diff)?;
                Ok(())
            }
            RegMul(dst, lhs, rhs) => {
                let a = state.get(*lhs as usize).ok_or("invalid lhs register")?;
                let b = state.get(*rhs as usize).ok_or("invalid rhs register")?;
                let prod = *a * *b;
                if !prod.re.is_finite() || !prod.im.is_finite() {
                    state.status.overflow = true;
                }
                state.set(*dst as usize, prod)?;
                Ok(())
            }
            RegDiv(dst, lhs, rhs) => {
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
            RegCopy(dst, src) => {
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
        let len = self.amps.len();
        for i in 0..len {
            if (i & mask) == 0 {
                self.amps.swap(i, i | mask);
            }
        }
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

        let len = self.amps.len();
        let mut new_amps = self.amps.clone();

        for i in 0..len {
            if (i & mask) == 0 {
                let flipped = i | mask;
                let a0 = self.amps[i];
                let a1 = self.amps[flipped];
                new_amps[i] = neg_i_comp * a1;
                new_amps[flipped] = i_comp * a0;
            }
        }
        self.amps = new_amps;
    }

    fn _apply_z_pure(&mut self, q: usize) {
        if q >= self.n {
            eprintln!("error: qubit index {} out of bounds for _apply_z_pure", q);
            return;
        }
        let mask = 1 << q;
        for i in 0..self.amps.len() {
            if (i & mask) != 0 {
                self.amps[i] = -self.amps[i];
            }
        }
    }

    fn apply_noise(&mut self) {
        if let Some(config) = &self.noise_config {
            // temporarily take ownership of the rng from self.
            // this ensures `self` is not mutably borrowed by `rng` during the loop
            // when `_apply_*_pure` methods (which also borrow `self` mutably) are called.
            let mut temp_rng = self
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
        let norm_sqr: f64 = self.amps.iter().map(|a| a.norm_sqr()).sum();
        if norm_sqr > 1e-12 {
            let norm = norm_sqr.sqrt();
            for amp in &mut self.amps {
                *amp /= norm;
            }
        } else {
            eprintln!("warning: state became zero due to noise or numerical instability, resetting to |0...0>.");
            self.amps
                .iter_mut()
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
        let size = self.amps.len();

        let mut new_amps = vec![Complex64::new(0.0, 0.0); size];

        for i in 0..size {
            if i & mask == 0 {
                let flipped = i | mask;
                let a = self.amps[i];
                let b = self.amps[flipped];
                new_amps[i] = norm_factor * (a + b);
                new_amps[flipped] = norm_factor * (a - b);
            }
        }
        self.amps = new_amps;
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
            eprintln!("error: control and target qubits cannot be the same for cz gate");
            return;
        }
        let cm = 1 << c;
        let tm = 1 << t;
        for i in 0..self.amps.len() {
            if (i & cm != 0) && (i & tm != 0) {
                self.amps[i] = -self.amps[i];
            }
        }
        self.apply_noise();
    }

    pub fn apply_cnot(&mut self, c: usize, t: usize) {
        if c >= self.n || t >= self.n {
            eprintln!(
                "error: control ({}) or target ({}) qubit index out of bounds for cnot",
                c, t
            );
            return;
        }
        if c == t {
            eprintln!("error: control and target qubits cannot be the same for cnot gate");
            return;
        }
        let cm = 1 << c;
        let tm = 1 << t;
        let len = self.amps.len();

        let mut new_amps = self.amps.clone();

        for i in 0..len {
            if (i & cm) != 0 {
                let flipped = i ^ tm;
                new_amps[flipped] = self.amps[i];
            } else {
                new_amps[i] = self.amps[i];
            }
        }
        self.amps = new_amps;
        self.apply_noise();
    }

    pub fn apply_t_gate(&mut self, q: usize) {
        if q >= self.n {
            eprintln!("error: qubit index {} out of bounds for t gate", q);
            return;
        }
        let mask = 1 << q;
        let phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
        for i in 0..self.amps.len() {
            if (i & mask) != 0 {
                self.amps[i] *= phase;
            }
        }
        self.apply_noise();
    }

    pub fn apply_s_gate(&mut self, q: usize) {
        if q >= self.n {
            eprintln!("error: qubit index {} out of bounds for s gate", q);
            return;
        }
        let mask = 1 << q;
        let phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_2);
        for i in 0..self.amps.len() {
            if (i & mask) != 0 {
                self.amps[i] *= phase;
            }
        }
        self.apply_noise();
    }

    pub fn apply_phase_shift(&mut self, q: usize, angle: f64) {
        if q >= self.n {
            eprintln!("error: qubit index {} out of bounds for phase shift", q);
            return;
        }
        let mask = 1 << q;
        let phase = Complex64::from_polar(1.0, angle);
        for i in 0..self.amps.len() {
            if (i & mask) != 0 {
                self.amps[i] *= phase;
            }
        }
        self.apply_noise();
    }

    pub fn apply_controlled_phase(&mut self, c: usize, t: usize, angle: f64) {
        if c >= self.n || t >= self.n {
            eprintln!("error: qubit indices out of bounds for controlled phase rotation");
            return;
        }
        if c == t {
            eprintln!("error: control and target qubits cannot be the same");
            return;
        }
        let cm = 1 << c;
        let tm = 1 << t;
        let phase = Complex64::from_polar(1.0, angle);
        for i in 0..self.amps.len() {
            if (i & cm != 0) && (i & tm != 0) {
                self.amps[i] *= phase;
            }
        }
        self.apply_noise();
    }

    pub fn apply_rx(&mut self, q: usize, angle: f64) {
        if q >= self.n {
            eprintln!("error: rx qubit index out of bounds");
            return;
        }
        let cos = (angle / 2.0).cos();
        let isin = Complex64::new(0.0, -(angle / 2.0).sin());

        let mask = 1 << q;
        let size = self.amps.len();
        let mut new_amps = self.amps.clone();

        for i in 0..size {
            if (i & mask) == 0 {
                let j = i | mask;
                let a0 = self.amps[i];
                let a1 = self.amps[j];
                new_amps[i] = cos * a0 + isin * a1;
                new_amps[j] = cos * a1 + isin * a0;
            }
        }
        self.amps = new_amps;
        self.apply_noise();
    }

    pub fn apply_ry(&mut self, q: usize, angle: f64) {
        if q >= self.n {
            eprintln!("error: ry qubit index out of bounds");
            return;
        }
        let cos = (angle / 2.0).cos();
        let sin = (angle / 2.0).sin();

        let mask = 1 << q;
        let size = self.amps.len();
        let mut new_amps = self.amps.clone();

        for i in 0..size {
            if (i & mask) == 0 {
                let j = i | mask;
                let a0 = self.amps[i];
                let a1 = self.amps[j];
                new_amps[i] = cos * a0 - Complex64::new(0.0, 1.0) * sin * a1;
                new_amps[j] = cos * a1 + Complex64::new(0.0, 1.0) * sin * a0;
            }
        }
        self.amps = new_amps;
        self.apply_noise();
    }

    pub fn apply_rz(&mut self, q: usize, angle: f64) {
        if q >= self.n {
            eprintln!("error: rz qubit index out of bounds");
            return;
        }
        let mask = 1 << q;
        let phase0 = Complex64::from_polar(1.0, -angle / 2.0);
        let phase1 = Complex64::from_polar(1.0, angle / 2.0);

        for i in 0..self.amps.len() {
            if (i & mask) == 0 {
                self.amps[i] *= phase0;
            } else {
                self.amps[i] *= phase1;
            }
        }
        self.apply_noise();
    }

    pub fn measure(&mut self, q: usize) -> usize {
        if q >= self.n {
            eprintln!("error: qubit index {} out of bounds for measurement", q);
            return 0;
        }
        let mask = 1 << q;
        let prob1: f64 = self
            .amps
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i & mask) != 0)
            .map(|(_, a)| a.norm_sqr())
            .sum();

        // temporarily take the rng for use in this function
        let mut temp_rng = self
            .rng
            .take()
            .expect("rng should be initialized for measurement in current context.");

        let outcome = {
            let r: f64 = temp_rng.gen(); // use the temporary rng
            if r < prob1 {
                1
            } else {
                0
            }
        };

        // put the rng back before any subsequent `self` mutable operations that might conflict
        self.rng = Some(temp_rng);

        let norm_sqr = if outcome == 1 { prob1 } else { 1.0 - prob1 };

        if norm_sqr < 1e-12 {
            eprintln!("warning: normalizing by very small number during measurement; state may be invalid. outcome: {}", outcome);
            self.amps
                .iter_mut()
                .for_each(|amp| *amp = Complex64::new(0.0, 0.0));
            if outcome == 1 {
                self.amps[mask] = Complex64::new(1.0, 0.0);
            } else {
                self.amps[0] = Complex64::new(1.0, 0.0);
            }
            return outcome;
        }

        let norm = norm_sqr.sqrt();

        for (i, amp) in self.amps.iter_mut().enumerate() {
            if ((i & mask != 0) as usize) != outcome {
                *amp = Complex64::new(0.0, 0.0);
            } else {
                *amp /= norm;
            }
        }
        outcome
    }

    pub fn apply_final_state_noise(&mut self) {
        if let Some(NoiseConfig::Ideal) = &self.noise_config {
            eprintln!("[info] final state noise is skipped in ideal mode.");
            return;
        }

        eprintln!("[info] applying final state amplitude randomization.");

        // temporarily take the rng for use
        let mut temp_rng = self
            .rng
            .take()
            .expect("rng should be initialized for final state noise.");

        let noise_strength = 0.1; // noise strength

        for amp in self.amps.iter_mut() {
            // add small random gaussian noise to real and imaginary parts
            let random_re: f64 =
                <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut temp_rng)
                    * noise_strength;
            let random_im: f64 =
                <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut temp_rng)
                    * noise_strength;
            *amp += Complex64::new(random_re, random_im);
        }

        // put the rng back
        self.rng = Some(temp_rng);
        self.normalize(); // ensure the state is still normalized after randomization
    }
}
