use rand::rngs::OsRng;
use rand::Rng;
use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2; // 1/sqrt(2)

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
                self.amps[i] = norm * (a + b);
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
        println!("Applied CZ gate between qubits {} (control) and {} (target)", c, t);
    }

    /// Measure qubit q, collapse state, and return 0 or 1
    fn measure(&mut self, q: usize) -> usize {
        let mask = 1 << q;

        // Probability of measuring 1
        let prob_1: f64 = self
            .amps
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i & mask) != 0)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();

        // Sample measurement outcome with true randomness
        let mut rng = OsRng;
        let r: f64 = rng.gen();

        let outcome = if r < prob_1 { 1 } else { 0 };

        // Normalize post-measurement state
        let norm_factor = if outcome == 1 {
            prob_1.sqrt()
        } else {
            (1.0 - prob_1).sqrt()
        };

        for (i, amp) in self.amps.iter_mut().enumerate() {
            if (((i & mask) != 0) as usize) != outcome {
                *amp = Complex64::new(0.0, 0.0);
            } else {
                *amp /= norm_factor;
            }
        }

        println!("Measured qubit {}: {}", q, outcome);
        outcome
    }

    /// Print current amplitudes with their basis states
    fn print_state(&self) {
        println!("Current quantum state ({} qubits):", self.n);
        for i in 0..self.amps.len() {
            if self.amps[i].norm_sqr() > 1e-8 {
                println!(
                    "{:0width$b}: {:.4} + {:.4}i (prob {:.4})",
                    i,
                    self.amps[i].re,
                    self.amps[i].im,
                    self.amps[i].norm_sqr(),
                    width = self.n
                );
            }
        }
        println!();
    }
}

fn main() {
    // Example: 4-qubit quantum simulation of simple tunneling / fusion event
    let mut qs = QuantumState::new(4);

    println!("Initial state:");
    qs.print_state();

    // Put protons into superposition (simulate tunneling possibilities)
    qs.apply_h(0);
    qs.apply_h(1);

    // Qubit 2 models tunneling effect; put into superposition
    qs.apply_h(2);

    // Entangle tunneling effect with fusion detector (qubit 3)
    qs.apply_cz(2, 3);

    println!("State after gates:");
    qs.print_state();

    // Measure fusion detector qubit (3)
    let fusion_result = qs.measure(3);

    // Measure tunneling qubit (2)
    let tunnel_result = qs.measure(2);

    println!("Measurement results:");
    println!("Qubit 3 (fusion detector): {}", fusion_result);
    println!("Qubit 2 (tunneling effect): {}", tunnel_result);

    println!("Final state:");
    qs.print_state();
}
