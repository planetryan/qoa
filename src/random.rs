use rand::rngs::OsRng;
use rand::Rng;
use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

/// Quantum state simulator for n qubits.
struct QuantumState {
    n: usize,
    amps: Vec<Complex64>, // state vector amplitudes
}

impl QuantumState {
    /// Initialize n-qubit state to |0...0⟩
    fn new(n: usize) -> Self {
        let mut amps = vec![Complex64::new(0.0, 0.0); 1 << n];
        amps[0] = Complex64::new(1.0, 0.0);
        QuantumState { n, amps }
    }

    /// Apply Hadamard gate on qubit q
    fn apply_h(&mut self, q: usize) {
        let mask = 1 << q;
        let norm = Complex64::new(FRAC_1_SQRT_2, 0.0);
        for i in 0..self.amps.len() {
            if (i & mask) == 0 {
                let a = self.amps[i];
                let b = self.amps[i | mask];
                self.amps[i]        = norm * (a + b);
                self.amps[i | mask] = norm * (a - b);
            }
        }
        println!("Applied H gate on qubit {}", q);
    }

    /// Apply Pauli-X (NOT) gate on qubit q
    fn apply_x(&mut self, q: usize) {
        let mask = 1 << q;
        for i in 0..self.amps.len() {
            if (i & mask) == 0 {
                self.amps.swap(i, i | mask);
            }
        }
        println!("Applied X gate on qubit {}", q);
    }

    /// Apply Controlled-Z gate between control c and target t
    fn apply_cz(&mut self, c: usize, t: usize) {
        let cm = 1 << c;
        let tm = 1 << t;
        for i in 0..self.amps.len() {
            if (i & cm != 0) && (i & tm != 0) {
                self.amps[i] *= -1.0;
            }
        }
        println!("Applied CZ gate c={}, t={}", c, t);
    }

    /// Inject depolarizing noise on qubit q with probability p:
    /// With prob p, replace the qubit by the maximally mixed channel:
    ///   I/2 = (X + Y + Z) each with prob 1/3
    fn apply_depolarizing(&mut self, q: usize, p: f64) {
        let mut rng = OsRng;
        if rng.gen::<f64>() < p {
            // choose one of X,Y,Z uniformly
            match rng.gen_range(0..3) {
                0 => { self.apply_x(q); }
                1 => { self.apply_phase_flip(q); } // Z
                2 => { 
                    // Y = iXZ, flip then phase, add phase i
                    self.apply_x(q);
                    self.apply_phase_flip(q);
                    // multiply affected amps by i
                    let i_phase = Complex64::new(0.0, 1.0);
                    let mask = 1 << q;
                    for idx in 0..self.amps.len() {
                        if (idx & mask) != 0 {
                            self.amps[idx] *= i_phase;
                        }
                    }
                }
                _ => unreachable!(),
            }
            println!("Depolarizing noise applied on q{} with p={}", q, p);
        }
    }

    /// Bit-flip noise: apply X with probability p
    fn apply_bit_flip(&mut self, q: usize, p: f64) {
        let mut rng = OsRng;
        if rng.gen::<f64>() < p {
            self.apply_x(q);
            println!("Bit-flip on qubit {} (p={})", q, p);
        }
    }

    /// Phase-flip noise: apply Z with probability p
    fn apply_phase_flip(&mut self, q: usize) {
        let mask = 1 << q;
        for idx in 0..self.amps.len() {
            if (idx & mask) != 0 {
                self.amps[idx] *= -1.0;
            }
        }
        // Note: we assume caller already checked probability
        println!("Phase-flip on qubit {}", q);
    }

    /// Measure qubit q, collapse state, and return 0 or 1
    fn measure(&mut self, q: usize) -> usize {
        let mask = 1 << q;
        let prob_1: f64 = self.amps.iter()
            .enumerate()
            .filter(|(i, _)| (*i & mask) != 0)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();

        let mut rng = OsRng;
        let r: f64 = rng.gen();
        let outcome = if r < prob_1 { 1 } else { 0 };

        let norm = if outcome == 1 { prob_1.sqrt() } else { (1.0 - prob_1).sqrt() };
        for (i, amp) in self.amps.iter_mut().enumerate() {
            let bit = ((i & mask) != 0) as usize;
            if bit != outcome {
                *amp = Complex64::new(0.0, 0.0);
            } else {
                *amp /= norm;
            }
        }

        println!("Measured q{} → {}", q, outcome);
        outcome
    }

    /// Print current amplitudes with their basis states
    fn print_state(&self) {
        println!("State ({} qubits):", self.n);
        for i in 0..self.amps.len() {
            let prob = self.amps[i].norm_sqr();
            if prob > 1e-8 {
                println!("{:0width$b}: {:.4} + {:.4}i (p={:.4})",
                    i, self.amps[i].re, self.amps[i].im, prob, width=self.n);
            }
        }
        println!();
    }
}

fn main() {
    let mut qs = QuantumState::new(3);
    println!("Initial:");
    qs.print_state();

    // create superposition
    qs.apply_h(0);
    qs.apply_h(1);

    // inject some realistic noise before entangling
    qs.apply_depolarizing(0, 0.05);
    qs.apply_bit_flip(1, 0.02);

    // entangle qubit 1→2
    qs.apply_cz(1, 2);

    println!("After gates + noise:");
    qs.print_state();

    // measure all
    for q in 0..3 {
        let _ = qs.measure(q);
    }
    println!("Final:");
    qs.print_state();
}
