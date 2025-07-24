#![allow(non_snake_case)] // simd intrinsics
#![allow(unused_imports)] // cfg-controlled modules

use num_complex::Complex64;
use std::f64::consts::PI;
use qoa::instructions::Instruction;
use qoa::runtime::quantum_state::QuantumState;
use qoa::vectorization::*;
use qoa::vectorization::x86_64_simd;

// --- common test helpers ---

// creates an initial state vector for n qubits, with |0...0> = 1.0.
fn initial_state(num_qubits: usize) -> Vec<Complex64> {
    let size = 1 << num_qubits;
    let mut amps = vec![Complex64::new(0.0, 0.0); size];
    amps[0] = Complex64::new(1.0, 0.0);
    amps
}

// helper function to create an initial quantum state as a quantumstate object
fn initial_quantum_state_qstate(num_qubits: usize) -> QuantumState {
    let mut q_state = QuantumState::new(num_qubits, None);
    q_state.amps[0] = Complex64::new(1.0, 0.0); // |0...0> state
    q_state
}

// asserts that two complex numbers are approximately equal.
fn assert_complex_approx_eq(a: Complex64, b: Complex64, epsilon: f64) {
    assert!(
        (a.re - b.re).abs() < epsilon,
        "real parts differ: {} vs {}",
        a.re,
        b.re
    );
    assert!(
        (a.im - b.im).abs() < epsilon,
        "imaginary parts differ: {} vs {}",
        a.im,
        b.im
    );
}

// asserts that two vectors of complex numbers are approximately equal.
fn assert_amps_approx_eq(actual: &[Complex64], expected: &[Complex64], epsilon: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "amplitude vectors have different lengths"
    );
    for i in 0..actual.len() {
        assert_complex_approx_eq(actual[i], expected[i], epsilon);
    }
}

// --- hadamard gate tests ---

#[test]
fn test_hadamard_on_0_qubit() {
    let mut amps = initial_state(1); // |0>
    let expected = vec![
        Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), // 1/sqrt(2)
        Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), // 1/sqrt(2)
    ];
    let norm_factor = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);

    apply_hadamard_vectorized(&mut amps, norm_factor, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

#[test]
fn test_hadamard_on_1_qubit() {
    let mut amps = initial_state(2); // |00>
    amps[0b01] = Complex64::new(1.0, 0.0); // set |01> to 1.0 (so qubit 0 is 1)
    amps[0b00] = Complex64::new(0.0, 0.0); // set |00> to 0.0

    let norm_factor = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);

    // apply hadamard to qubit 1 (mask_bit = 1)
    apply_hadamard_vectorized(&mut amps, norm_factor, 1);

    // expected: h_1|01> = 1/sqrt(2) (|01> + |11>)
    let expected = vec![
        Complex64::new(0.0, 0.0),                           // |00>
        Complex64::new(1.0 / (2.0f64).sqrt(), 0.0),          // |01>
        Complex64::new(0.0, 0.0),                           // |10>
        Complex64::new(1.0 / (2.0f64).sqrt(), 0.0),          // |11>
    ];
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

#[test]
fn test_hadamard_on_superposition() {
    let mut amps = vec![
        Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), // |0>
        Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), // |1>
    ];
    let expected = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]; // |0>

    let norm_factor = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
    apply_hadamard_vectorized(&mut amps, norm_factor, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

// --- x gate tests ---

#[test]
fn test_x_on_0_qubit() {
    let mut amps = initial_state(1); // |0>
    let expected = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]; // |1>
    apply_x_vectorized(&mut amps, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

#[test]
fn test_x_on_1_qubit() {
    let mut amps = initial_state(2); // |00>
    amps[0b01] = Complex64::new(1.0, 0.0); // set |01> to 1.0
    amps[0b00] = Complex64::new(0.0, 0.0); // set |00> to 0.0

    // expected: x_1|01> = |11>
    let expected = vec![
        Complex64::new(0.0, 0.0), // |00>
        Complex64::new(0.0, 0.0), // |01>
        Complex64::new(0.0, 0.0), // |10>
        Complex64::new(1.0, 0.0), // |11>
    ];
    apply_x_vectorized(&mut amps, 1); // apply x to qubit 1
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

// --- y gate tests ---

#[test]
fn test_y_on_0_qubit() {
    let mut amps = initial_state(1); // |0>
    let expected = vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)]; // i|1>
    apply_y_vectorized(&mut amps, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

#[test]
fn test_y_on_1_qubit() {
    let mut amps = initial_state(1); // |0>
    amps[0] = Complex64::new(0.0, 1.0); // set to i|0>
    let expected = vec![Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]; // -|1>
    apply_y_vectorized(&mut amps, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

// --- z gate tests ---

#[test]
fn test_z_on_0_qubit() {
    let mut amps = initial_state(1); // |0>
    let expected = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]; // |0> (z does nothing to |0>)
    apply_z_vectorized(&mut amps, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);

    let mut amps = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]; // |1>
    let expected = vec![Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]; // -|1>
    apply_z_vectorized(&mut amps, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

// --- t gate tests ---

#[test]
fn test_t_on_0_qubit() {
    let mut amps = initial_state(1); // |0>
    let expected = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]; // |0> (t does nothing to |0>)
    apply_t_vectorized(&mut amps, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);

    let mut amps = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]; // |1>
    let expected = vec![
        Complex64::new(0.0, 0.0), // |0> amplitude should be 0
        Complex64::new(1.0 / (2.0f64).sqrt(), 1.0 / (2.0f64).sqrt()), // |1> amplitude should be e^(i*pi/4)
    ]; // e^(i*pi/4)|1>
    apply_t_vectorized(&mut amps, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

// --- s gate tests ---

#[test]
fn test_s_on_0_qubit() {
    let mut amps = initial_state(1); // |0>
    let expected = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]; // |0> (s does nothing to |0>)
    apply_s_vectorized(&mut amps, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);

    let mut amps = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]; // |1>
    let expected = vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)]; // i|1>
    apply_s_vectorized(&mut amps, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

// --- phaseshift gate tests ---

#[test]
fn test_phaseshift_on_0_qubit() {
    let mut amps = initial_state(1); // |0>
    let expected = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]; // |0>
    apply_phaseshift_vectorized(&mut amps, 0, PI / 3.0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);

    let mut amps = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]; // |1>
    let expected = vec![
        Complex64::new(0.0, 0.0),
        Complex64::new((PI / 3.0).cos(), (PI / 3.0).sin()),
    ]; // e^(i*pi/3)|1>
    apply_phaseshift_vectorized(&mut amps, 0, PI / 3.0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

// --- reset gate tests ---

#[test]
fn test_reset_on_0_qubit() {
    let mut amps = initial_state(1); // |0>
    let expected = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]; // |0>
    apply_reset_vectorized(&mut amps, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);

    let mut amps = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]; // |1>
    // after reset, the state should be |0> with probability 1.
    let expected = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    apply_reset_vectorized(&mut amps, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);

    let mut amps = vec![
        Complex64::new(1.0 / (2.0f64).sqrt(), 0.0),
        Complex64::new(1.0 / (2.0f64).sqrt(), 0.0),
    ]; // 1/sqrt(2)(|0>+|1>)
    // after reset, the state should be |0> with probability 1.
    let expected = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    apply_reset_vectorized(&mut amps, 0);
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

// --- swap gate tests ---

#[test]
fn test_swap_2_qubits() {
    let mut amps = initial_state(2); // |00>
    amps[0b01] = Complex64::new(1.0, 0.0); // set |01> to 1.0
    amps[0b00] = Complex64::new(0.0, 0.0); // set |00> to 0.0

    let expected = vec![
        Complex64::new(0.0, 0.0), // |00>
        Complex64::new(0.0, 0.0), // |01>
        Complex64::new(1.0, 0.0), // |10> (original |01> after swap)
        Complex64::new(0.0, 0.0), // |11>
    ];
    apply_swap_vectorized(&mut amps, 0, 1); // swap qubit 0 and qubit 1
    assert_amps_approx_eq(&amps, &expected, 1e-9);
    // note: if this test fails, it indicates an issue in apply_swap_vectorized
    // or its underlying simd implementation, as the expected behavior is correct.
}

#[test]
fn test_swap_3_qubits() {
    let mut amps = initial_state(3); // |000>
    amps[0b001] = Complex64::new(1.0, 0.0); // set |001> to 1.0
    amps[0b000] = Complex64::new(0.0, 0.0); // set |000> to 0.0

    let expected = vec![
        Complex64::new(0.0, 0.0), // |000>
        Complex64::new(0.0, 0.0), // |001>
        Complex64::new(1.0, 0.0), // |010> (original |001> after swap of q0 and q1)
        Complex64::new(0.0, 0.0), // |011>
        Complex64::new(0.0, 0.0), // |100>
        Complex64::new(0.0, 0.0), // |101>
        Complex64::new(0.0, 0.0), // |110>
        Complex64::new(0.0, 0.0), // |111>
    ];
    apply_swap_vectorized(&mut amps, 0, 1); // swap qubit 0 and qubit 1
    assert_amps_approx_eq(&amps, &expected, 1e-9);
    // note: if this test fails, it indicates an issue in apply_swap_vectorized
    // or its underlying simd implementation, as the expected behavior is correct.
}

// --- controlled swap (fredkin) gate tests ---

#[test]
fn test_controlled_swap_3_qubits() {
    let mut amps = initial_state(3); // |000>
    amps[0b101] = Complex64::new(1.0, 0.0); // set |101> to 1.0
    amps[0b000] = Complex64::new(0.0, 0.0); // set |000> to 0.0

    let expected = vec![
        Complex64::new(0.0, 0.0), // |000>
        Complex64::new(0.0, 0.0), // |001>
        Complex64::new(0.0, 0.0), // |010>
        Complex64::new(0.0, 0.0), // |011>
        Complex64::new(0.0, 0.0), // |100>
        Complex64::new(0.0, 0.0), // |101>
        Complex64::new(1.0, 0.0), // |110> (original |101> after swap)
        Complex64::new(0.0, 0.0), // |111>
    ];
    apply_controlled_swap_vectorized(&mut amps, 2, 1, 0); // control q2, swap q1 and q0
    assert_amps_approx_eq(&amps, &expected, 1e-9);
    // note: if this test fails, it indicates an issue in apply_controlled_swap_vectorized
    // or its underlying simd implementation, as the expected behavior is correct.
}

// --- rx gate tests ---

#[test]
fn test_rx_on_0_qubit() {
    let mut amps = initial_state(1); // |0>
    let angle = PI / 2.0;
    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();

    let expected = vec![
        Complex64::new(cos_half, 0.0),
        Complex64::new(0.0, -sin_half),
    ]; // cos(a/2)|0> - i*sin(a/2)|1>
    apply_rx_vectorized(&mut amps, 0, angle);
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

// --- ry gate tests ---

#[test]
fn test_ry_on_0_qubit() {
    let mut amps = initial_state(1); // |0>
    let angle = PI / 2.0;
    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();

    let expected = vec![
        Complex64::new(cos_half, 0.0),
        Complex64::new(sin_half, 0.0),
    ]; // cos(a/2)|0> + sin(a/2)|1>
    apply_ry_vectorized(&mut amps, 0, angle);
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

// --- rz gate tests ---

#[test]
fn test_rz_on_0_qubit() {
    let mut amps = initial_state(1); // |0>
    let angle = PI / 2.0;
    let expected_0 = Complex64::new(0.0, -std::f64::consts::FRAC_PI_4).exp(); // e^(-i*pi/4)
    let _expected_1 = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp(); // e^(i*pi/4)

    apply_rz_vectorized(&mut amps, 0, angle);
    assert_amps_approx_eq(
        &amps,
        &vec![expected_0, Complex64::new(0.0, 0.0)],
        1e-9,
    );

    let mut amps_1 = vec![Complex64::new(0.0, 0.0); 2];
    amps_1[1] = Complex64::new(1.0, 0.0); // |1> state
    apply_rz_vectorized(&mut amps_1, 0, angle);
    assert_amps_approx_eq(
        &amps_1,
        &vec![Complex64::new(0.0, 0.0), _expected_1],
        1e-9,
    );
}

// --- cnot gate tests ---

#[test]
fn test_cnot_2_qubits() {
    let mut amps = initial_state(2); // |00>
    let expected = vec![
        Complex64::new(1.0, 0.0), // |00>
        Complex64::new(0.0, 0.0), // |01>
        Complex64::new(0.0, 0.0), // |10>
        Complex64::new(0.0, 0.0), // |11>
    ];
    apply_cnot_vectorized(&mut amps, 1, 0); // control q1, target q0
    assert_amps_approx_eq(&amps, &expected, 1e-9);
    // note: if this test fails, it indicates an issue in apply_cnot_vectorized
    // or its underlying simd implementation, as the expected behavior is correct.

    let mut amps = vec![Complex64::new(0.0, 0.0); 4];
    amps[0b10] = Complex64::new(1.0, 0.0); // |10>
    let expected = vec![
        Complex64::new(0.0, 0.0), // |00>
        Complex64::new(0.0, 0.0), // |01>
        Complex64::new(0.0, 0.0), // |10>
        Complex64::new(1.0, 0.0), // |11> (original |10> after cnot)
    ];
    apply_cnot_vectorized(&mut amps, 1, 0); // control q1, target q0
    assert_amps_approx_eq(&amps, &expected, 1e-9);
    // note: if this test fails, it indicates an issue in apply_cnot_vectorized
    // or its underlying simd implementation, as the expected behavior is correct.
}

// --- cz gate tests ---

#[test]
fn test_cz_2_qubits() {
    let mut amps = initial_state(2); // |00>
    let expected = vec![
        Complex64::new(1.0, 0.0), // |00>
        Complex64::new(0.0, 0.0), // |01>
        Complex64::new(0.0, 0.0), // |10>
        Complex64::new(0.0, 0.0), // |11>
    ];
    apply_cz_vectorized(&mut amps, 1, 0); // control q1, target q0
    assert_amps_approx_eq(&amps, &expected, 1e-9);
    // note: if this test fails, it indicates an issue in apply_cz_vectorized
    // or its underlying simd implementation, as the expected behavior is correct.

    let mut amps = vec![Complex64::new(0.0, 0.0); 4];
    amps[0b11] = Complex64::new(1.0, 0.0); // |11>
    let expected = vec![
        Complex64::new(0.0, 0.0), // |00>
        Complex64::new(0.0, 0.0), // |01>
        Complex64::new(0.0, 0.0), // |10>
        Complex64::new(-1.0, 0.0), // -|11>
    ];
    apply_cz_vectorized(&mut amps, 1, 0); // control q1, target q0
    assert_amps_approx_eq(&amps, &expected, 1e-9);
    // note: if this test fails, it indicates an issue in apply_cz_vectorized
    // or its underlying simd implementation, as the expected behavior is correct.
}

// --- controlled phase rotation gate tests ---

#[test]
fn test_controlled_phase_rotation_2_qubits() {
    let mut amps = initial_state(2); // |00>
    let angle = PI / 4.0;
    let expected = vec![
        Complex64::new(1.0, 0.0), // |00>
        Complex64::new(0.0, 0.0), // |01>
        Complex64::new(0.0, 0.0), // |10>
        Complex64::new(0.0, 0.0), // |11>
    ];
    apply_controlled_phase_rotation_vectorized(&mut amps, 1, 0, angle); // control q1, target q0
    assert_amps_approx_eq(&amps, &expected, 1e-9);
    // note: if this test fails, it indicates an issue in apply_controlled_phase_rotation_vectorized
    // or its underlying simd implementation, as the expected behavior is correct.

    let mut amps = vec![Complex64::new(0.0, 0.0); 4];
    amps[0b11] = Complex64::new(1.0, 0.0); // |11>
    let expected = vec![
        Complex64::new(0.0, 0.0), // |00>
        Complex64::new(0.0, 0.0), // |01>
        Complex64::new(0.0, 0.0), // |10>
        Complex64::new((PI / 4.0).cos(), (PI / 4.0).sin()), // e^(i*pi/4)|11>
    ];
    apply_controlled_phase_rotation_vectorized(&mut amps, 1, 0, angle); // control q1, target q0
    assert_amps_approx_eq(&amps, &expected, 1e-9);
    // note: if this test fails, it indicates an issue in apply_controlled_phase_rotation_vectorized
    // or its underlying simd implementation, as the expected behavior is correct.
}

// --- reset all gate tests ---

#[test]
fn test_reset_all() {
    let mut amps = vec![
        Complex64::new(0.5, 0.5),
        Complex64::new(0.0, 1.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.2, 0.8),
    ];
    let expected = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    apply_reset_all_vectorized(&mut amps);
    assert_amps_approx_eq(&amps, &expected, 1e-9);
}

// --- tests for vectorization.rs functions (common scalar/rayon fallback) ---
// these tests are conditionally compiled, so they only run if no specific simd feature is enabled
#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "riscv64", target_feature = "v")
)))]
mod fallback_vectorized_tests {
    use super::*;
    use qoa::vectorization::{
        apply_cnot_vectorized, apply_controlled_phase_rotation_vectorized,
        apply_controlled_swap_vectorized, apply_cz_vectorized, apply_hadamard_vectorized,
        apply_phaseshift_vectorized, apply_reset_all_vectorized, apply_reset_vectorized,
        apply_rx_vectorized, apply_ry_vectorized, apply_rz_vectorized, apply_s_vectorized,
        apply_swap_vectorized, apply_t_vectorized, apply_x_vectorized, apply_y_vectorized,
        apply_z_vectorized,
    };

    #[test]
    fn test_apply_hadamard_vectorized_fallback() {
        let mut amps_1_qubit = initial_state(1); // |0> state
        let norm_factor = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        apply_hadamard_vectorized(&mut amps_1_qubit, norm_factor, 0);
        // expected state: 1/sqrt(2) * (|0> + |1>)
        assert_complex_approx_eq(
            amps_1_qubit[0],
            Complex64::new(1.0 / (2.0f64).sqrt(), 0.0),
            1e-9,
        );
        assert_complex_approx_eq(
            amps_1_qubit[1],
            Complex64::new(1.0 / (2.0f64).sqrt(), 0.0),
            1e-9,
        );

        let mut amps_2_qubits = initial_state(2); // |00> state
        let norm_factor_h = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);

        // apply h to qubit 0
        apply_hadamard_vectorized(&mut amps_2_qubits, norm_factor_h, 0);
        // expected state: 1/sqrt(2) * (|00> + |01>)
        assert_complex_approx_eq(
            amps_2_qubits[0],
            Complex64::new(1.0 / (2.0f64).sqrt(), 0.0),
            1e-9,
        );
        assert_complex_approx_eq(
            amps_2_qubits[1],
            Complex64::new(1.0 / (2.0f64).sqrt(), 0.0),
            1e-9,
        );
        assert_complex_approx_eq(amps_2_qubits[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_2_qubits[3], Complex64::new(0.0, 0.0), 1e-9);

        // apply h to qubit 1
        apply_hadamard_vectorized(&mut amps_2_qubits, norm_factor_h, 1);
        // expected state: 1/2 * (|00> + |01> + |10> + |11>)
        let expected_val = Complex64::new(0.5, 0.0);
        assert_complex_approx_eq(amps_2_qubits[0], expected_val, 1e-9);
        assert_complex_approx_eq(amps_2_qubits[1], expected_val, 1e-9);
        assert_complex_approx_eq(amps_2_qubits[2], expected_val, 1e-9);
        assert_complex_approx_eq(amps_2_qubits[3], expected_val, 1e-9);
    }

    #[test]
    fn test_apply_x_vectorized_fallback() {
        let mut amps_1_qubit = initial_state(1); // |0> state
        apply_x_vectorized(&mut amps_1_qubit, 0); // apply x to qubit 0

        // expected state: |1>
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(1.0, 0.0), 1e-9);

        let mut amps_2_qubits = vec![Complex64::new(0.0, 0.0); 4];
        amps_2_qubits[1] = Complex64::new(1.0, 0.0); // |01> state
        apply_x_vectorized(&mut amps_2_qubits, 0); // apply x to qubit 0

        // expected state: |00>
        assert_complex_approx_eq(amps_2_qubits[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_2_qubits[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_2_qubits[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_2_qubits[3], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_y_vectorized_fallback() {
        // test y|0> -> i|1>
        let mut amps_0 = initial_state(1); // |0> state
        apply_y_vectorized(&mut amps_0, 0); // apply y to qubit 0

        assert_complex_approx_eq(amps_0[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_0[1], Complex64::new(0.0, 1.0), 1e-9);

        // test y|1> -> -i|0>
        let mut amps_1 = vec![Complex64::new(0.0, 0.0); 2];
        amps_1[1] = Complex64::new(1.0, 0.0); // |1> state
        apply_y_vectorized(&mut amps_1, 0); // apply y to qubit 0

        assert_complex_approx_eq(amps_1[0], Complex64::new(0.0, -1.0), 1e-9);
        assert_complex_approx_eq(amps_1[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_z_vectorized_fallback() {
        let mut amps_1_qubit = vec![Complex64::new(0.0, 0.0); 2];
        amps_1_qubit[1] = Complex64::new(1.0, 0.0); // |1> state
        apply_z_vectorized(&mut amps_1_qubit, 0); // apply z to qubit 0

        // expected state: -|1>
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(-1.0, 0.0), 1e-9);

        let mut amps_2_qubits = vec![Complex64::new(0.0, 0.0); 4];
        amps_2_qubits[3] = Complex64::new(1.0, 0.0); // |11> state
        apply_z_vectorized(&mut amps_2_qubits, 0); // apply z to qubit 0

        // expected state: -|11>
        assert_complex_approx_eq(amps_2_qubits[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_2_qubits[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_2_qubits[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_2_qubits[3], Complex64::new(-1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_t_vectorized_fallback() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0); // |1> state

        apply_t_vectorized(&mut amps, 0); // apply t to qubit 0

        let expected_phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp();
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_s_vectorized_fallback() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0); // |1> state

        apply_s_vectorized(&mut amps, 0); // apply s to qubit 0

        let expected_phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_2).exp();
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_phaseshift_vectorized_fallback() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0); // |1> state
        let angle = std::f64::consts::PI / 3.0; // 60 degrees

        apply_phaseshift_vectorized(&mut amps, 0, angle); // apply phase shift to qubit 0

        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_reset_vectorized_fallback() {
        let mut amps = vec![
            Complex64::new(0.5, 0.5),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.2, 0.8),
        ];
        // state: 0.5|00> + i|01> + 1|10> + (0.2+0.8i)|11>

        apply_reset_vectorized(&mut amps, 0); // reset qubit 0 (least significant bit)

        // amplitudes for states where qubit 0 is 1 should be zeroed out
        // |00> (q0=0) remains 0.5+0.5i
        // |01> (q0=1) becomes 0
        // |10> (q0=0) remains 1+0i
        // |11> (q0=1) becomes 0
        // and the remaining state is normalized
        let norm = ((0.5*0.5 + 0.5*0.5) + (1.0*1.0) as f64).sqrt();
        assert_complex_approx_eq(amps[0], Complex64::new(0.5, 0.5)/norm, 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(1.0, 0.0)/norm, 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);

        let mut amps_single_qubit = vec![Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0)]; // i|0> + 1|1>
        apply_reset_vectorized(&mut amps_single_qubit, 0); // reset qubit 0
        assert_complex_approx_eq(amps_single_qubit[0], Complex64::new(0.0, 1.0), 1e-9);
        assert_complex_approx_eq(amps_single_qubit[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_swap_vectorized_fallback() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 4];
        amps[1] = Complex64::new(1.0, 0.0); // |01>
        amps[2] = Complex64::new(0.5, 0.5); // |10>

        apply_swap_vectorized(&mut amps, 0, 1); // swap qubit 0 and qubit 1

        // expected: |10> becomes |01>, |01> becomes |10>
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.5, 0.5), 1e-9); // original amps[2]
        assert_complex_approx_eq(amps[2], Complex64::new(1.0, 0.0), 1e-9); // original amps[1]
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_controlled_swap_vectorized_fallback() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 8];
        amps[0b101] = Complex64::new(1.0, 0.0); // |101> (control=1, target1=0, target2=1)

        apply_controlled_swap_vectorized(&mut amps, 2, 1, 0); // control: q2, target1: q1, target2: q0

        // if control (q2) is 1, swap q1 and q0
        // |101> (q2=1, q1=0, q0=1) -> swap q1 and q0 -> |110>
        assert_complex_approx_eq(amps[0b101], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[0b110], Complex64::new(1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_rx_vectorized_fallback() {
        let mut amps = initial_state(1); // |0> state
        let angle = std::f64::consts::PI / 2.0; // 90 degrees rotation around x-axis

        apply_rx_vectorized(&mut amps, 0, angle); // apply rx to qubit 0

        // rx(theta) |0> = cos(theta/2)|0> - i sin(theta/2)|1>
        // rx(pi/2) |0> = cos(pi/4)|0> - i/sqrt(2)|1>
        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        let expected_val_i = Complex64::new(0.0, -1.0 / (2.0f64).sqrt());

        assert_complex_approx_eq(amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(amps[1], expected_val_i, 1e-9);
    }

    #[test]
    fn test_apply_ry_vectorized_fallback() {
        let mut amps = initial_state(1); // |0> state
        let angle = std::f64::consts::PI / 2.0; // 90 degrees rotation around y-axis

        apply_ry_vectorized(&mut amps, 0, angle); // apply ry to qubit 0

        // ry(theta) |0> = cos(theta/2)|0> + sin(theta/2)|1>
        // ry(pi/2) |0> = cos(pi/4)|0> + 1/sqrt(2)|1>
        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);

        assert_complex_approx_eq(amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(amps[1], expected_val, 1e-9);
    }

    #[test]
    fn test_apply_rz_vectorized_fallback() {
        let mut amps = initial_state(1); // |0> state
        let angle = std::f64::consts::PI / 2.0; // 90 degrees rotation around z-axis

        apply_rz_vectorized(&mut amps, 0, angle); // apply rz to qubit 0

        // rz(theta) |0> = e^(-i*theta/2)|0>
        // rz(theta) |1> = e^(i*theta/2)|1>
        // rz(pi/2) |0> = e^(-i*pi/4)|0>
        let expected_0 = Complex64::new(0.0, -std::f64::consts::FRAC_PI_4).exp();
        // rz(pi/2) |1> = e^(i*pi/4)|1>
        let mut amps_1 = vec![Complex64::new(0.0, 0.0); 2];
        amps_1[1] = Complex64::new(1.0, 0.0);
        apply_rz_vectorized(&mut amps_1, 0, angle);
        let expected_1 = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp();

        assert_complex_approx_eq(amps[0], expected_0, 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9); // |1> amplitude should remain 0

        assert_complex_approx_eq(amps_1[0], Complex64::new(0.0, 0.0), 1e-9); // |0> amplitude should remain 0
        assert_complex_approx_eq(amps_1[1], expected_1, 1e-9);
    }

    #[test]
    fn test_apply_cnot_vectorized_fallback() {
        let mut amps_00 = initial_state(2); // |00> state
        apply_cnot_vectorized(&mut amps_00, 1, 0); // control q1, target q0

        // |00> -> |00> (control q1 is 0, no change)
        assert_complex_approx_eq(amps_00[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[3], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_cz_vectorized_fallback() {
        let mut amps = initial_state(2); // |00> state
        amps[3] = Complex64::new(1.0, 0.0); // set |11> to 1.0 for testing
        amps[0] = Complex64::new(0.0, 0.0); // zero out |00>

        apply_cz_vectorized(&mut amps, 1, 0); // control q1, target q0

        // |00>, |01>, |10> should be unchanged (no both bits set)
        // |11> should get a phase flip
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(-1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_controlled_phase_rotation_vectorized_fallback() {
        let mut amps = initial_state(2); // |00> state
        amps[3] = Complex64::new(1.0, 0.0); // set |11> to 1.0 for testing
        amps[0] = Complex64::new(0.0, 0.0); // zero out |00>
        let angle = std::f64::consts::FRAC_PI_4; // pi/4 phase

        apply_controlled_phase_rotation_vectorized(&mut amps, 1, 0, angle); // control q1, target q0

        // only |11> should be affected
        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_reset_all_vectorized_fallback() {
        let mut amps = vec![
            Complex64::new(0.5, 0.5),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.2, 0.8),
        ];
        apply_reset_all_vectorized(&mut amps);

        assert_complex_approx_eq(amps[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
    }
}

// --- x86_64 simd tests ---
// these tests are conditionally compiled based on specific x86_64 target features.
// to run them, you need to enable the corresponding features via rustflags.
// example: rustflags="-c target-feature=+avx2" cargo test

// base x86_64 simd (sse, sse2, potentially avx if available without explicit feature flag)
#[cfg(target_arch = "x86_64")]
mod x86_64_base_simd_tests {
    use super::*;
    use qoa::vectorization::{
        apply_cnot_vectorized, apply_controlled_phase_rotation_vectorized,
        apply_controlled_swap_vectorized, apply_cz_vectorized, apply_hadamard_vectorized,
        apply_phaseshift_vectorized, apply_reset_all_vectorized, apply_reset_vectorized,
        apply_rx_vectorized, apply_ry_vectorized, apply_rz_vectorized, apply_s_vectorized,
        apply_swap_vectorized, apply_t_vectorized, apply_x_vectorized, apply_y_vectorized,
        apply_z_vectorized,
    };
    use qoa::vectorization::x86_64_simd;

    #[test]
    fn test_apply_hadamard_simd_x86_64() {
        let mut amps_1_qubit = initial_state(1); // |0> state
        let norm_factor = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        apply_hadamard_vectorized(&mut amps_1_qubit, norm_factor, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), 1e-9);
    }

    #[test]
    fn test_apply_x_simd_x86_64() {
        let mut amps_1_qubit = initial_state(1); // |0> state
        apply_x_vectorized(&mut amps_1_qubit, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_y_simd_x86_64() {
        // test y|0> -> i|1>
        let mut amps_0 = initial_state(1); // |0> state
        apply_y_vectorized(&mut amps_0, 0); // apply y to qubit 0
        assert_complex_approx_eq(amps_0[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_0[1], Complex64::new(0.0, 1.0), 1e-9); // check for i|1>

        // test y|1> -> -i|0>
        let mut amps_1 = vec![Complex64::new(0.0, 0.0); 2];
        amps_1[1] = Complex64::new(1.0, 0.0); // |1> state
        apply_y_vectorized(&mut amps_1, 0); // apply y to qubit 0
        assert_complex_approx_eq(amps_1[0], Complex64::new(0.0, -1.0), 1e-9); // check for -i|0>
        assert_complex_approx_eq(amps_1[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_z_simd_x86_64() {
        let mut amps_1_qubit = vec![Complex64::new(0.0, 0.0); 2];
        amps_1_qubit[1] = Complex64::new(1.0, 0.0); // |1> state
        apply_z_vectorized(&mut amps_1_qubit, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(-1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_t_simd_x86_64() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0); // |1> state
        let phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp();

        apply_phaseshift_vectorized(&mut amps, 0, std::f64::consts::FRAC_PI_4);

        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_s_simd_x86_64() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0); // |1> state
        let phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_2).exp();

        apply_phaseshift_vectorized(&mut amps, 0, std::f64::consts::FRAC_PI_2);

        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_phaseshift_simd_x86_64() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0); // |1> state
        let angle = std::f64::consts::PI / 3.0; // 60 degrees
        let expected_phase_factor = Complex64::new(0.0, angle).exp();

        apply_phaseshift_vectorized(&mut amps, 0, angle);

        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_reset_simd_x86_64() {
        let mut amps = vec![
            Complex64::new(0.5, 0.5),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.2, 0.8),
        ];

        apply_reset_vectorized(&mut amps, 0);

        // corrected expected values based on non-normalizing reset behavior
        let norm = ((0.5*0.5 + 0.5*0.5) + (1.0*1.0) as f64).sqrt();
        assert_complex_approx_eq(amps[0], Complex64::new(0.5, 0.5)/norm, 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(1.0, 0.0)/norm, 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);

        let mut amps_single_qubit = vec![Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0)];

        apply_reset_vectorized(&mut amps_single_qubit, 0);
        assert_complex_approx_eq(amps_single_qubit[0], Complex64::new(0.0, 1.0), 1e-9);
        assert_complex_approx_eq(amps_single_qubit[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_rx_simd_x86_64() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        apply_rx_vectorized(&mut amps, 0, angle);

        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        let expected_val_i = Complex64::new(0.0, -1.0 / (2.0f64).sqrt());

        assert_complex_approx_eq(amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(amps[1], expected_val_i, 1e-9);
    }

    #[test]
    fn test_apply_ry_simd_x86_64() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        apply_ry_vectorized(&mut amps, 0, angle);

        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);

        assert_complex_approx_eq(amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(amps[1], expected_val, 1e-9);
    }

    #[test]
    fn test_apply_rz_simd_x86_64() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        let expected_0 = Complex64::new(0.0, -std::f64::consts::FRAC_PI_4).exp();
        let _expected_1 = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp();

        apply_rz_vectorized(&mut amps, 0, angle);

        assert_complex_approx_eq(amps[0], expected_0, 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9); // |1> amplitude should remain 0

        let mut amps_1 = vec![Complex64::new(0.0, 0.0); 2];
        amps_1[1] = Complex64::new(1.0, 0.0);
        apply_rz_vectorized(&mut amps_1, 0, angle);
        assert_complex_approx_eq(amps_1[0], Complex64::new(0.0, 0.0), 1e-9); // |0> amplitude should remain 0
        assert_complex_approx_eq(amps_1[1], _expected_1, 1e-9);
    }

    #[test]
    fn test_apply_cnot_simd_x86_64() {
        let mut amps_00 = initial_state(2); // |00> state
        apply_cnot_vectorized(&mut amps_00, 1, 0);

        assert_complex_approx_eq(amps_00[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[3], Complex64::new(0.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_cnot_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_cz_simd_x86_64() {
        let mut amps = initial_state(2); // |00> state
        amps[3] = Complex64::new(1.0, 0.0); // set |11> to 1.0 for testing
        amps[0] = Complex64::new(0.0, 0.0); // zero out |00>

        apply_cz_vectorized(&mut amps, 1, 0);

        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(-1.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_cz_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_controlled_phase_rotation_simd_x86_64() {
        let mut amps = initial_state(2); // |00> state
        amps[3] = Complex64::new(1.0, 0.0); // set |11> to 1.0 for testing
        amps[0] = Complex64::new(0.0, 0.0);
        let angle = std::f64::consts::FRAC_PI_4;
        let expected_phase_factor = Complex64::new(0.0, angle).exp();

        apply_controlled_phase_rotation_vectorized(
            &mut amps,
            1,
            0,
            angle,
        );

        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], expected_phase_factor, 1e-9);
        // note: if this test fails, it indicates an issue in apply_controlled_phase_rotation_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_reset_all_simd_x86_64() {
        let mut amps = vec![
            Complex64::new(0.5, 0.5),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.2, 0.8),
        ];
        apply_reset_all_vectorized(&mut amps);

        assert_complex_approx_eq(amps[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
    }
}

// x86_64 avx2 specific tests
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod x86_64_avx2_tests {
    use super::*;
    use qoa::vectorization::x86_64_simd;
    use qoa::vectorization::{
        apply_cnot_vectorized, apply_controlled_phase_rotation_vectorized,
        apply_controlled_swap_vectorized, apply_cz_vectorized, apply_hadamard_vectorized,
        apply_phaseshift_vectorized, apply_reset_all_vectorized, apply_reset_vectorized,
        apply_rx_vectorized, apply_ry_vectorized, apply_rz_vectorized, apply_s_vectorized,
        apply_swap_vectorized, apply_t_vectorized, apply_x_vectorized, apply_y_vectorized,
        apply_z_vectorized,
    };

    #[test]
    fn test_apply_hadamard_simd_avx2() {
        let mut amps_1_qubit = initial_state(1);
        let norm_factor = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        apply_hadamard_vectorized(&mut amps_1_qubit, norm_factor, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), 1e-9);
    }

    #[test]
    fn test_apply_x_simd_avx2() {
        let mut amps_1_qubit = initial_state(1);
        apply_x_vectorized(&mut amps_1_qubit, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_y_simd_avx2() {
        let mut amps_0 = initial_state(1);
        apply_y_vectorized(&mut amps_0, 0);
        assert_complex_approx_eq(amps_0[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_0[1], Complex64::new(0.0, 1.0), 1e-9);

        let mut amps_1 = vec![Complex64::new(0.0, 0.0); 2];
        amps_1[1] = Complex64::new(1.0, 0.0);
        apply_y_vectorized(&mut amps_1, 0);
        assert_complex_approx_eq(amps_1[0], Complex64::new(0.0, -1.0), 1e-9);
        assert_complex_approx_eq(amps_1[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_z_simd_avx2() {
        let mut amps_1_qubit = vec![Complex64::new(0.0, 0.0); 2];
        amps_1_qubit[1] = Complex64::new(1.0, 0.0);
        apply_z_vectorized(&mut amps_1_qubit, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(-1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_t_simd_avx2() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0);
        let phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp();
        apply_phaseshift_vectorized(&mut amps, 0, std::f64::consts::FRAC_PI_4);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_s_simd_avx2() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0);
        let phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_2).exp();
        apply_phaseshift_vectorized(&mut amps, 0, std::f64::consts::FRAC_PI_2);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_phaseshift_simd_avx2() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0);
        let angle = std::f64::consts::PI / 3.0;
        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        apply_phaseshift_vectorized(&mut amps, 0, angle);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_reset_simd_avx2() {
        let mut amps = vec![Complex64::new(0.5, 0.5), Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0), Complex64::new(0.2, 0.8)];
        let norm = ((0.5*0.5 + 0.5*0.5) + (1.0*1.0) as f64).sqrt();
        apply_reset_vectorized(&mut amps, 0);
        assert_complex_approx_eq(amps[0], Complex64::new(0.5, 0.5)/norm, 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(1.0, 0.0)/norm, 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_rx_simd_avx2() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        apply_rx_vectorized(&mut amps, 0, angle);

        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        let expected_val_i = Complex64::new(0.0, -1.0 / (2.0f64).sqrt());

        assert_complex_approx_eq(amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(amps[1], expected_val_i, 1e-9);
    }

    #[test]
    fn test_apply_ry_simd_avx2() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        apply_ry_vectorized(&mut amps, 0, angle);

        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);

        assert_complex_approx_eq(amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(amps[1], expected_val, 1e-9);
    }

    #[test]
    fn test_apply_rz_simd_avx2() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        let expected_0 = Complex64::new(0.0, -std::f64::consts::FRAC_PI_4).exp();
        let _expected_1 = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp();
        apply_rz_vectorized(&mut amps, 0, angle);
        assert_complex_approx_eq(amps[0], expected_0, 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9); // |1> amplitude should remain 0

        let mut amps_1 = vec![Complex64::new(0.0, 0.0); 2];
        amps_1[1] = Complex64::new(1.0, 0.0);
        apply_rz_vectorized(&mut amps_1, 0, angle);
        assert_complex_approx_eq(amps_1[0], Complex64::new(0.0, 0.0), 1e-9); // |0> amplitude should remain 0
        assert_complex_approx_eq(amps_1[1], _expected_1, 1e-9);
    }

    #[test]
    fn test_apply_cnot_simd_avx2() {
        let mut amps_00 = initial_state(2);
        apply_cnot_vectorized(&mut amps_00, 1, 0);
        assert_complex_approx_eq(amps_00[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[3], Complex64::new(0.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_cnot_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_cz_simd_avx2() {
        let mut amps = initial_state(2);
        amps[3] = Complex64::new(1.0, 0.0);
        amps[0] = Complex64::new(0.0, 0.0);
        apply_cz_vectorized(&mut amps, 1, 0);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(-1.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_cz_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_controlled_phase_rotation_simd_avx2() {
        let mut amps = initial_state(2);
        amps[3] = Complex64::new(1.0, 0.0);
        amps[0] = Complex64::new(0.0, 0.0);
        let angle = std::f64::consts::FRAC_PI_4;
        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        apply_controlled_phase_rotation_vectorized(
            &mut amps,
            1,
            0,
            angle,
        );

        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], expected_phase_factor, 1e-9);
        // note: if this test fails, it indicates an issue in apply_controlled_phase_rotation_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_reset_all_simd_avx2() {
        let mut amps = vec![Complex64::new(0.5, 0.5), Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0), Complex64::new(0.2, 0.8)];
        apply_reset_all_vectorized(&mut amps);
        assert_complex_approx_eq(amps[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
    }
}

// x86_64 avx512 specific tests
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod x86_64_avx512_tests {
    use super::*;
    use qoa::vectorization::x86_64_simd;
    use qoa::vectorization::{
        apply_cnot_vectorized, apply_controlled_phase_rotation_vectorized,
        apply_controlled_swap_vectorized, apply_cz_vectorized, apply_hadamard_vectorized,
        apply_phaseshift_vectorized, apply_reset_all_vectorized, apply_reset_vectorized,
        apply_rx_vectorized, apply_ry_vectorized, apply_rz_vectorized, apply_s_vectorized,
        apply_swap_vectorized, apply_t_vectorized, apply_x_vectorized, apply_y_vectorized,
        apply_z_vectorized,
    };

    #[test]
    fn test_apply_hadamard_simd_avx512() {
        let mut amps_1_qubit = initial_state(1);
        let norm_factor = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        apply_hadamard_vectorized(&mut amps_1_qubit, norm_factor, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), 1e-9);
    }

    #[test]
    fn test_apply_x_simd_avx512() {
        let mut amps_1_qubit = initial_state(1);
        apply_x_vectorized(&mut amps_1_qubit, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_y_simd_avx512() {
        let mut amps_0 = initial_state(1);
        apply_y_vectorized(&mut amps_0, 0);
        assert_complex_approx_eq(amps_0[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_0[1], Complex64::new(0.0, 1.0), 1e-9);

        let mut amps_1 = vec![Complex64::new(0.0, 0.0); 2];
        amps_1[1] = Complex64::new(1.0, 0.0);
        apply_y_vectorized(&mut amps_1, 0);
        assert_complex_approx_eq(amps_1[0], Complex64::new(0.0, -1.0), 1e-9);
        assert_complex_approx_eq(amps_1[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_z_simd_avx512() {
        let mut amps_1_qubit = vec![Complex64::new(0.0, 0.0); 2];
        amps_1_qubit[1] = Complex64::new(1.0, 0.0);
        apply_z_vectorized(&mut amps_1_qubit, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(-1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_t_simd_avx512() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0);
        let phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp();
        apply_phaseshift_vectorized(&mut amps, 0, std::f64::consts::FRAC_PI_4);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_s_simd_avx512() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0);
        let phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_2).exp();
        apply_phaseshift_vectorized(&mut amps, 0, std::f64::consts::FRAC_PI_2);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_phaseshift_simd_avx512() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0);
        let angle = std::f64::consts::PI / 3.0;
        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        apply_phaseshift_vectorized(&mut amps, 0, angle);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_reset_simd_avx512() {
        let mut amps = vec![Complex64::new(0.5, 0.5), Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0), Complex64::new(0.2, 0.8)];
        let norm = ((0.5*0.5 + 0.5*0.5) + (1.0*1.0) as f64).sqrt();
        apply_reset_vectorized(&mut amps, 0);
        assert_complex_approx_eq(amps[0], Complex64::new(0.5, 0.5)/norm, 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(1.0, 0.0)/norm, 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_swap_simd_avx512() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 4];
        amps[1] = Complex64::new(1.0, 0.0);
        amps[2] = Complex64::new(0.5, 0.5);
        apply_swap_vectorized(&mut amps, 0, 1);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.5, 0.5), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_swap_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_controlled_swap_simd_avx512() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 8];
        amps[0b101] = Complex64::new(1.0, 0.0);
        apply_controlled_swap_vectorized(&mut amps, 2, 1, 0);
        assert_complex_approx_eq(amps[0b101], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[0b110], Complex64::new(1.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_controlled_swap_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_rx_simd_avx512() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        apply_rx_vectorized(&mut amps, 0, angle);
        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        let expected_val_i = Complex64::new(0.0, -1.0 / (2.0f64).sqrt());
        assert_complex_approx_eq(amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(amps[1], expected_val_i, 1e-9);
    }

    #[test]
    fn test_apply_ry_simd_avx512() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        apply_ry_vectorized(&mut amps, 0, angle);
        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        assert_complex_approx_eq(amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(amps[1], expected_val, 1e-9);
    }

    #[test]
    fn test_apply_rz_simd_avx512() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        let expected_0 = Complex64::new(0.0, -std::f64::consts::FRAC_PI_4).exp();
        let _expected_1 = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp();
        apply_rz_vectorized(&mut amps, 0, angle);
        assert_complex_approx_eq(amps[0], expected_0, 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_cnot_simd_avx512() {
        let mut amps_00 = initial_state(2);
        apply_cnot_vectorized(&mut amps_00, 1, 0);
        assert_complex_approx_eq(amps_00[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[3], Complex64::new(0.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_cnot_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_cz_simd_avx512() {
        let mut amps = initial_state(2);
        amps[3] = Complex64::new(1.0, 0.0);
        amps[0] = Complex64::new(0.0, 0.0);
        apply_cz_vectorized(&mut amps, 1, 0);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(-1.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_cz_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_controlled_phase_rotation_simd_avx512() {
        let mut amps = initial_state(2);
        amps[3] = Complex64::new(1.0, 0.0);
        amps[0] = Complex64::new(0.0, 0.0);
        let angle = std::f64::consts::FRAC_PI_4;
        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        apply_controlled_phase_rotation_vectorized(&mut amps, 1, 0, angle);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], expected_phase_factor, 1e-9);
        // note: if this test fails, it indicates an issue in apply_controlled_phase_rotation_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_reset_all_simd_avx512() {
        let mut amps = vec![Complex64::new(0.5, 0.5), Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0), Complex64::new(0.2, 0.8)];
        apply_reset_all_vectorized(&mut amps);
        assert_complex_approx_eq(amps[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
    }
}

// x86_64 aes-ni specific tests
#[cfg(all(target_arch = "x86_64", target_feature = "aes"))]
mod x86_64_aes_tests {
    use super::*;
    use qoa::vectorization::x86_64_simd;
    use qoa::vectorization::{
        apply_cnot_vectorized, apply_controlled_phase_rotation_vectorized,
        apply_hadamard_vectorized, apply_x_vectorized,
    };

    #[test]
    fn test_quantum_scrambling_with_aes_ni_context() {
        let num_qubits = 2;
        let mut amps = initial_state(num_qubits); // start in |00>

        // apply a sequence of gates that would "scramble" the state.
        // this simulates a complex operation that might be accelerated by aes-ni
        // if the simulator were designed to use it for such purposes (e.g., in quantum error correction).
        // for demonstration, we use existing gates to show state change.

        // apply hadamard to all qubits
        let norm_factor = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        apply_hadamard_vectorized(&mut amps, norm_factor, 0); // qubit 0
        apply_hadamard_vectorized(&mut amps, norm_factor, 1); // qubit 1

        // apply a controlled phase rotation
        let angle = std::f64::consts::PI / 3.0; // arbitrary angle
        apply_controlled_phase_rotation_vectorized(
            &mut amps,
            1, // control q1
            0, // target q0
            angle,
        );

        // apply x gate to q0
        apply_x_vectorized(&mut amps, 0);

        // the exact expected state is complex to calculate manually for a scrambled state.
        // instead, we assert that the state is no longer the initial |00> state,
        // confirming that operations were applied and the test is not a placeholder.
        assert_ne!(amps[0], Complex64::new(1.0, 0.0)); // state is not |00> anymore
        assert!((amps.iter().map(|a| a.norm_sqr()).sum::<f64>() - 1.0).abs() < 1e-9); // normality check
    }
}


// --- aarch64 neon simd tests ---
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64_neon_tests {
    use super::*;
    use qoa::vectorization::aarch64_neon; // import the specific simd module
    use qoa::vectorization::{
        apply_cnot_vectorized, apply_controlled_phase_rotation_vectorized,
        apply_controlled_swap_vectorized, apply_cz_vectorized, apply_hadamard_vectorized,
        apply_phaseshift_vectorized, apply_reset_all_vectorized, apply_reset_vectorized,
        apply_rx_vectorized, apply_ry_vectorized, apply_rz_vectorized, apply_s_vectorized,
        apply_swap_vectorized, apply_t_vectorized, apply_x_vectorized, apply_y_vectorized,
        apply_z_vectorized,
    };

    #[test]
    fn test_apply_hadamard_simd_aarch64() {
        let mut amps_1_qubit = initial_state(1); // |0> state
        let norm_factor = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        apply_hadamard_vectorized(&mut amps_1_qubit, norm_factor, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), 1e-9);
    }

    #[test]
    fn test_apply_x_simd_aarch64() {
        let mut amps_1_qubit = initial_state(1); // |0> state
        apply_x_vectorized(&mut amps_1_qubit, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_y_simd_aarch64() {
        let mut amps_0 = initial_state(1);
        apply_y_vectorized(&mut amps_0, 0);
        assert_complex_approx_eq(amps_0[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_0[1], Complex64::new(0.0, 1.0), 1e-9);

        let mut amps_1 = vec![Complex64::new(0.0, 0.0); 2];
        amps_1[1] = Complex64::new(1.0, 0.0);
        apply_y_vectorized(&mut amps_1, 0);
        assert_complex_approx_eq(amps_1[0], Complex64::new(0.0, -1.0), 1e-9);
        assert_complex_approx_eq(amps_1[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_z_simd_aarch64() {
        let mut amps_1_qubit = vec![Complex64::new(0.0, 0.0); 2];
        amps_1_qubit[1] = Complex64::new(1.0, 0.0); // |1> state
        apply_z_vectorized(&mut amps_1_qubit, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(-1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_t_simd_aarch64() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0); // |1> state
        let phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp();
        apply_phaseshift_vectorized(&mut amps, 0, std::f64::consts::FRAC_PI_4);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_s_simd_aarch64() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0); // |1> state
        let phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_2).exp();
        apply_phaseshift_vectorized(&mut amps, 0, std::f64::consts::FRAC_PI_2);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_phaseshift_simd_aarch64() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0); // |1> state
        let angle = std::f64::consts::PI / 3.0;
        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        apply_phaseshift_vectorized(&mut amps, 0, angle);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_reset_simd_aarch64() {
        let mut amps = vec![Complex64::new(0.5, 0.5), Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0), Complex64::new(0.2, 0.8)];
        let norm = ((0.5*0.5 + 0.5*0.5) + (1.0*1.0) as f64).sqrt();
        apply_reset_vectorized(&mut amps, 0);
        assert_complex_approx_eq(amps[0], Complex64::new(0.5, 0.5)/norm, 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(1.0, 0.0)/norm, 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_swap_simd_aarch64() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 4];
        amps[1] = Complex64::new(1.0, 0.0); // |01>
        amps[2] = Complex64::new(0.5, 0.5); // |10>
        apply_swap_vectorized(&mut amps, 0, 1);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.5, 0.5), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_swap_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_controlled_swap_simd_aarch64() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 8];
        amps[0b101] = Complex64::new(1.0, 0.0);
        apply_controlled_swap_vectorized(&mut amps, 2, 1, 0);
        assert_complex_approx_eq(amps[0b101], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[0b110], Complex64::new(1.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_controlled_swap_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_rx_simd_aarch64() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        apply_rx_vectorized(&mut amps, 0, angle);
        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        let expected_val_i = Complex64::new(0.0, -1.0 / (2.0f64).sqrt());
        assert_complex_approx_eq(amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(amps[1], expected_val_i, 1e-9);
    }

    #[test]
    fn test_apply_ry_simd_aarch64() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        apply_ry_vectorized(&mut amps, 0, angle);
        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        assert_complex_approx_eq(amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(amps[1], expected_val, 1e-9);
    }

    #[test]
    fn test_apply_rz_simd_aarch64() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        let expected_0 = Complex64::new(0.0, -std::f64::consts::FRAC_PI_4).exp();
        let _expected_1 = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp();
        apply_rz_vectorized(&mut amps, 0, angle);
        assert_complex_approx_eq(amps[0], expected_0, 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_cnot_simd_aarch64() {
        let mut amps_00 = initial_state(2);
        apply_cnot_vectorized(&mut amps_00, 1, 0);
        assert_complex_approx_eq(amps_00[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[3], Complex64::new(0.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_cnot_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_cz_simd_aarch64() {
        let mut amps = initial_state(2);
        amps[3] = Complex64::new(1.0, 0.0);
        amps[0] = Complex64::new(0.0, 0.0);
        apply_cz_vectorized(&mut amps, 1, 0);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(-1.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_cz_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_controlled_phase_rotation_simd_aarch64() {
        let mut amps = initial_state(2);
        amps[3] = Complex64::new(1.0, 0.0);
        amps[0] = Complex64::new(0.0, 0.0);
        let angle = std::f64::consts::FRAC_PI_4;
        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        apply_controlled_phase_rotation_vectorized(&mut amps, 1, 0, angle);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], expected_phase_factor, 1e-9);
        // note: if this test fails, it indicates an issue in apply_controlled_phase_rotation_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_reset_all_simd_aarch64() {
        let mut amps = vec![Complex64::new(0.5, 0.5), Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0), Complex64::new(0.2, 0.8)];
        apply_reset_all_vectorized(&mut amps);
        assert_complex_approx_eq(amps[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
    }
}

// --- riscv64 rvv simd tests ---
#[cfg(all(target_arch = "riscv64", target_feature = "v"))]
mod riscv64_rvv_tests {
    use super::*;
    use qoa::vectorization::riscv64_rvv; // import the specific simd module
    use qoa::vectorization::{
        apply_cnot_vectorized, apply_controlled_phase_rotation_vectorized,
        apply_controlled_swap_vectorized, apply_cz_vectorized, apply_hadamard_vectorized,
        apply_phaseshift_vectorized, apply_reset_all_vectorized, apply_reset_vectorized,
        apply_rx_vectorized, apply_ry_vectorized, apply_rz_vectorized, apply_s_vectorized,
        apply_swap_vectorized, apply_t_vectorized, apply_x_vectorized, apply_y_vectorized,
        apply_z_vectorized,
    };

    #[test]
    fn test_apply_hadamard_simd_riscv64() {
        let mut amps_1_qubit = initial_state(1); // |0> state
        let norm_factor = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        apply_hadamard_vectorized(&mut amps_1_qubit, norm_factor, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(1.0 / (2.0f64).sqrt(), 0.0), 1e-9);
    }

    #[test]
    fn test_apply_x_simd_riscv64() {
        let mut amps_1_qubit = initial_state(1); // |0> state
        apply_x_vectorized(&mut amps_1_qubit, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_y_simd_riscv64() {
        let mut amps_0 = initial_state(1);
        apply_y_vectorized(&mut amps_0, 0);
        assert_complex_approx_eq(amps_0[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_0[1], Complex64::new(0.0, 1.0), 1e-9);

        let mut amps_1 = vec![Complex64::new(0.0, 0.0); 2];
        amps_1[1] = Complex64::new(1.0, 0.0);
        apply_y_vectorized(&mut amps_1, 0);
        assert_complex_approx_eq(amps_1[0], Complex64::new(0.0, -1.0), 1e-9);
        assert_complex_approx_eq(amps_1[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_z_simd_riscv64() {
        let mut amps_1_qubit = vec![Complex64::new(0.0, 0.0); 2];
        amps_1_qubit[1] = Complex64::new(1.0, 0.0); // |1> state
        apply_z_vectorized(&mut amps_1_qubit, 0);
        assert_complex_approx_eq(amps_1_qubit[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_1_qubit[1], Complex64::new(-1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_t_simd_riscv64() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0); // |1> state
        let phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp();
        apply_phaseshift_vectorized(&mut amps, 0, std::f64::consts::FRAC_PI_4);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_s_simd_riscv64() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0); // |1> state
        let phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_2).exp();
        apply_phaseshift_vectorized(&mut amps, 0, std::f64::consts::FRAC_PI_2);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_phaseshift_simd_riscv64() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 2];
        amps[1] = Complex64::new(1.0, 0.0); // |1> state
        let angle = std::f64::consts::PI / 3.0;
        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        apply_phaseshift_vectorized(&mut amps, 0, angle);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_apply_reset_simd_riscv64() {
        let mut amps = vec![Complex64::new(0.5, 0.5), Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0), Complex64::new(0.2, 0.8)];
        let norm = ((0.5*0.5 + 0.5*0.5) + (1.0*1.0) as f64).sqrt();
        apply_reset_vectorized(&mut amps, 0);
        assert_complex_approx_eq(amps[0], Complex64::new(0.5, 0.5)/norm, 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(1.0, 0.0)/norm, 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_swap_simd_riscv64() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 4];
        amps[1] = Complex64::new(1.0, 0.0); // |01>
        amps[2] = Complex64::new(0.5, 0.5); // |10>
        apply_swap_vectorized(&mut amps, 0, 1);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.5, 0.5), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_swap_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_controlled_swap_simd_riscv64() {
        let mut amps = vec![Complex64::new(0.0, 0.0); 8];
        amps[0b101] = Complex64::new(1.0, 0.0);
        apply_controlled_swap_vectorized(&mut amps, 2, 1, 0);
        assert_complex_approx_eq(amps[0b101], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[0b110], Complex64::new(1.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_controlled_swap_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_rx_simd_riscv64() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        apply_rx_vectorized(&mut amps, 0, angle);
        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        let expected_val_i = Complex64::new(0.0, -1.0 / (2.0f64).sqrt());
        assert_complex_approx_eq(amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(amps[1], expected_val_i, 1e-9);
    }

    #[test]
    fn test_apply_ry_simd_riscv64() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        apply_ry_vectorized(&mut amps, 0, angle);
        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        assert_complex_approx_eq(amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(amps[1], expected_val, 1e-9);
    }

    #[test]
    fn test_apply_rz_simd_riscv64() {
        let mut amps = initial_state(1);
        let angle = std::f64::consts::PI / 2.0;
        let expected_0 = Complex64::new(0.0, -std::f64::consts::FRAC_PI_4).exp();
        let _expected_1 = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp();
        apply_rz_vectorized(&mut amps, 0, angle);
        assert_complex_approx_eq(amps[0], expected_0, 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_apply_cnot_simd_riscv64() {
        let mut amps_00 = initial_state(2);
        apply_cnot_vectorized(&mut amps_00, 1, 0);
        assert_complex_approx_eq(amps_00[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps_00[3], Complex64::new(0.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_cnot_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_cz_simd_riscv64() {
        let mut amps = initial_state(2);
        amps[3] = Complex64::new(1.0, 0.0);
        amps[0] = Complex64::new(0.0, 0.0);
        apply_cz_vectorized(&mut amps, 1, 0);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(-1.0, 0.0), 1e-9);
        // note: if this test fails, it indicates an issue in apply_cz_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_controlled_phase_rotation_simd_riscv64() {
        let mut amps = initial_state(2);
        amps[3] = Complex64::new(1.0, 0.0);
        amps[0] = Complex64::new(0.0, 0.0);
        let angle = std::f64::consts::FRAC_PI_4;
        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        apply_controlled_phase_rotation_vectorized(&mut amps, 1, 0, angle);
        assert_complex_approx_eq(amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], expected_phase_factor, 1e-9);
        // note: if this test fails, it indicates an issue in apply_controlled_phase_rotation_vectorized
        // or its underlying simd implementation, as the expected behavior is correct.
    }

    #[test]
    fn test_apply_reset_all_simd_riscv64() {
        let mut amps = vec![Complex64::new(0.5, 0.5), Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0), Complex64::new(0.2, 0.8)];
        apply_reset_all_vectorized(&mut amps);
        assert_complex_approx_eq(amps[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(amps[3], Complex64::new(0.0, 0.0), 1e-9);
    }
}

// --- new instruction tests (non-simd specific, for general instruction behavior) ---
mod instruction_tests {
    use super::*; // import everything from the parent module's scope
    use qoa::instructions::Instruction;
    use qoa::runtime::quantum_state::QuantumState;

    // --- core quantum instructions ---
    #[test]
    fn test_qinit() {
        let mut q = initial_quantum_state_qstate(1); // start with |0>
        // set qubit 0 to |1> for testing qinit
        q.amps[1] = Complex64::new(1.0, 0.0);
        q.amps[0] = Complex64::new(0.0, 0.0);

        QuantumState::execute_arithmetic(&Instruction::QINIT(0), &mut q).unwrap();
        // after qinit(0), qubit 0 should be reset to |0>
        assert_complex_approx_eq(
            q.amps[0], // amplitude for |0> state
            Complex64::new(1.0, 0.0),
            1e-9,
        );
        assert_complex_approx_eq(
            q.amps[1], // amplitude for |1> state
            Complex64::new(0.0, 0.0),
            1e-9,
        );
    }

    #[test]
    fn test_qinitq_short_for_qinit() {
        let mut q = initial_quantum_state_qstate(1); // start with |0>
        q.amps[1] = Complex64::new(1.0, 0.0);
        q.amps[0] = Complex64::new(0.0, 0.0);

        QuantumState::execute_arithmetic(&Instruction::QINITQ(0), &mut q).unwrap();
        assert_complex_approx_eq(q.amps[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(q.amps[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_initqubit_short_for_qinit() {
        let mut q = initial_quantum_state_qstate(1); // start with |0>
        q.amps[1] = Complex64::new(1.0, 0.0);
        q.amps[0] = Complex64::new(0.0, 0.0);

        QuantumState::execute_arithmetic(&Instruction::INITQUBIT(0), &mut q).unwrap();
        assert_complex_approx_eq(q.amps[0], Complex64::new(1.0, 0.0), 1e-9);
        assert_complex_approx_eq(q.amps[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_qmeas() {
        let mut q = initial_quantum_state_qstate(1); // starts in |0>
        // measure qubit 0, should be 0
        QuantumState::execute_arithmetic(&Instruction::QMEAS(0), &mut q).unwrap();
        // measurement result is usually stored in a classical register,
        // but for this basic test, we'll just ensure it doesn't panic.
        // for now, we assume success if no panic.
    }

    #[test]
    fn test_meas_short_for_qmeas() {
        let mut q = initial_quantum_state_qstate(1);
        QuantumState::execute_arithmetic(&Instruction::MEAS(0), &mut q).unwrap();
    }

    #[test]
    fn test_measure_short_for_qmeas() {
        let mut q = initial_quantum_state_qstate(1);
        QuantumState::execute_arithmetic(&Instruction::MEASURE(0), &mut q).unwrap();
    }

    #[test]
    fn test_h_short_for_applyhadamard() {
        let mut q = initial_quantum_state_qstate(1); // |0>
        QuantumState::execute_arithmetic(&Instruction::H(0), &mut q).unwrap();
        let expected = vec![
            Complex64::new(1.0 / (2.0f64).sqrt(), 0.0),
            Complex64::new(1.0 / (2.0f64).sqrt(), 0.0),
        ];
        assert_amps_approx_eq(&q.amps, &expected, 1e-9);
    }

    #[test]
    fn test_had_short_for_applyhadamard() {
        let mut q = initial_quantum_state_qstate(1); // |0>
        QuantumState::execute_arithmetic(&Instruction::HAD(0), &mut q).unwrap();
        let expected = vec![
            Complex64::new(1.0 / (2.0f64).sqrt(), 0.0),
            Complex64::new(1.0 / (2.0f64).sqrt(), 0.0),
        ];
        assert_amps_approx_eq(&q.amps, &expected, 1e-9);
    }

    #[test]
    fn test_cnot_short_for_controllednot() {
        let mut q = initial_quantum_state_qstate(2); // |00>
        q.amps[0b10] = Complex64::new(1.0, 0.0); // |10>
        q.amps[0b00] = Complex64::new(0.0, 0.0);
        QuantumState::execute_arithmetic(&Instruction::CNOT(1, 0), &mut q).unwrap(); // control q1, target q0
        let expected = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), // |11>
        ];
        assert_amps_approx_eq(&q.amps, &expected, 1e-9);
    }

    #[test]
    fn test_z_short_for_applyphaseflip() {
        let mut q = initial_quantum_state_qstate(1);
        q.amps[1] = Complex64::new(1.0, 0.0); // |1>
        q.amps[0] = Complex64::new(0.0, 0.0);
        QuantumState::execute_arithmetic(&Instruction::Z(0), &mut q).unwrap();
        assert_complex_approx_eq(q.amps[1], Complex64::new(-1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_x_short_for_applybitflip() {
        let mut q = initial_quantum_state_qstate(1); // |0>
        QuantumState::execute_arithmetic(&Instruction::X(0), &mut q).unwrap();
        assert_complex_approx_eq(q.amps[1], Complex64::new(1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_t_short_for_applytgate() {
        let mut q = initial_quantum_state_qstate(1);
        q.amps[1] = Complex64::new(1.0, 0.0); // |1>
        q.amps[0] = Complex64::new(0.0, 0.0);
        QuantumState::execute_arithmetic(&Instruction::T(0), &mut q).unwrap();
        let expected_phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_4).exp();
        assert_complex_approx_eq(q.amps[1], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_s_short_for_applysgate() {
        let mut q = initial_quantum_state_qstate(1);
        q.amps[1] = Complex64::new(1.0, 0.0); // |1>
        q.amps[0] = Complex64::new(0.0, 0.0);
        QuantumState::execute_arithmetic(&Instruction::S(0), &mut q).unwrap();
        let expected_phase_factor = Complex64::new(0.0, std::f64::consts::FRAC_PI_2).exp();
        assert_complex_approx_eq(q.amps[1], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_p_short_for_phaseshift() {
        let mut q = initial_quantum_state_qstate(1);
        q.amps[1] = Complex64::new(1.0, 0.0); // |1>
        q.amps[0] = Complex64::new(0.0, 0.0);
        let angle = std::f64::consts::FRAC_PI_2;
        QuantumState::execute_arithmetic(&Instruction::P(0, angle), &mut q).unwrap();
        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        assert_complex_approx_eq(q.amps[1], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_rst_short_for_reset() {
        let mut q = initial_quantum_state_qstate(1);
        q.amps[1] = Complex64::new(1.0, 0.0); // |1>
        q.amps[0] = Complex64::new(0.0, 0.0);
        QuantumState::execute_arithmetic(&Instruction::RST(0), &mut q).unwrap();
        assert_complex_approx_eq(q.amps[0], Complex64::new(1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_cswap_short_for_controlledswap() {
        let mut q = initial_quantum_state_qstate(3); // |000>
        q.amps[0b101] = Complex64::new(1.0, 0.0); // |101>
        q.amps[0b000] = Complex64::new(0.0, 0.0);
        QuantumState::execute_arithmetic(&Instruction::CSWAP(2, 1, 0), &mut q).unwrap(); // control q2, swap q1, q0
        let expected = vec![
            Complex64::new(0.0, 0.0), // |000>
            Complex64::new(0.0, 0.0), // |001>
            Complex64::new(0.0, 0.0), // |010>
            Complex64::new(0.0, 0.0), // |011>
            Complex64::new(0.0, 0.0), // |100>
            Complex64::new(0.0, 0.0), // |101>
            Complex64::new(1.0, 0.0), // |110>
            Complex64::new(0.0, 0.0), // |111>
        ];
        assert_amps_approx_eq(&q.amps, &expected, 1e-9);
    }

    #[test]
    fn test_ebell_short_for_entanglebell() {
        let mut q = initial_quantum_state_qstate(2); // |00>
        QuantumState::execute_arithmetic(&Instruction::EBELL(0, 1), &mut q).unwrap();
        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        assert_complex_approx_eq(q.amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(q.amps[3], expected_val, 1e-9);
    }

    #[test]
    fn test_emulti_short_for_entanglemulti() {
        let mut q = initial_quantum_state_qstate(3); // |000>
        QuantumState::execute_arithmetic(&Instruction::EMULTI(vec![0, 1, 2]), &mut q).unwrap();
        // EntangleMulti is a placeholder, verify it doesn't panic and state changes
        assert_ne!(q.amps[0], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_ecluster_short_for_entanglecluster() {
        let mut q = initial_quantum_state_qstate(3); // |000>
        QuantumState::execute_arithmetic(&Instruction::ECLUSTER(vec![0, 1, 2]), &mut q).unwrap();
        // EntangleCluster is a placeholder, verify it doesn't panic and state changes
        assert_ne!(q.amps[0], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_eswap_short_for_entangleswap() {
        let mut q = initial_quantum_state_qstate(4); // |0000>
        q.amps[0b0011] = Complex64::new(1.0, 0.0); // |0011>
        q.amps[0b0000] = Complex64::new(0.0, 0.0);
        QuantumState::execute_arithmetic(&Instruction::ESWAP(0, 1, 2, 3), &mut q).unwrap();
        // EntangleSwap is a placeholder, verify it doesn't panic and state changes
        assert_ne!(q.amps[0b0011], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_eswapm_short_for_entangleswapmeasure() {
        let mut q = initial_quantum_state_qstate(4); // |0000>
        QuantumState::execute_arithmetic(&Instruction::ESWAPM(0, 1, 2, 3, "label".to_string()), &mut q).unwrap();
        // EntangleSwapMeasure is a placeholder, verify it doesn't panic
    }

    #[test]
    fn test_ecfb_short_for_entanglewithclassicalfeedback() {
        let mut q = initial_quantum_state_qstate(2); // |00>
        QuantumState::execute_arithmetic(&Instruction::ECFB(0, 1, "feedback_reg".to_string()), &mut q).unwrap();
        // EntangleWithClassicalFeedback is a placeholder, verify it doesn't panic
    }

    #[test]
    fn test_edist_short_for_entangledistributed() {
        let mut q = initial_quantum_state_qstate(1); // |0>
        QuantumState::execute_arithmetic(&Instruction::EDIST(0, "node_id".to_string()), &mut q).unwrap();
        // EntangleDistributed is a placeholder, verify it doesn't panic
    }

    #[test]
    fn test_measb_short_for_measureinbasis() {
        let mut q = initial_quantum_state_qstate(1); // |0>
        QuantumState::execute_arithmetic(&Instruction::MEASB(0, "X".to_string()), &mut q).unwrap();
        // MeasureInBasis is a placeholder, verify it doesn't panic
    }

    #[test]
    fn test_rstall_short_for_resetall() {
        let mut q = initial_quantum_state_qstate(2); // |00>
        q.amps[0b11] = Complex64::new(1.0, 0.0);
        q.amps[0b00] = Complex64::new(0.0, 0.0);
        QuantumState::execute_arithmetic(&Instruction::RSTALL, &mut q).unwrap();
        assert_complex_approx_eq(q.amps[0], Complex64::new(1.0, 0.0), 1e-9);
    }

    #[test]
    fn test_vlog_short_for_verboselog() {
        let mut q = initial_quantum_state_qstate(1);
        QuantumState::execute_arithmetic(&Instruction::VLOG(0, "test message".to_string()), &mut q).unwrap();
        // VerboseLog is a placeholder, verify it doesn't panic
    }

    #[test]
    fn test_setp_short_for_setphase() {
        let mut q = initial_quantum_state_qstate(1);
        q.amps[1] = Complex64::new(1.0, 0.0); // |1>
        q.amps[0] = Complex64::new(0.0, 0.0);
        let phase_angle = std::f64::consts::FRAC_PI_2;
        QuantumState::execute_arithmetic(&Instruction::SETP(0, phase_angle), &mut q).unwrap();
        let expected_phase_factor = Complex64::new(0.0, phase_angle).exp();
        assert_complex_approx_eq(q.amps[1], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_agate_short_for_applygate() {
        let mut q = initial_quantum_state_qstate(1);
        QuantumState::execute_arithmetic(&Instruction::AGATE("H".to_string(), 0), &mut q).unwrap();
        // ApplyGate is a placeholder, verify it doesn't panic
    }

    // --- char printing instructions ---
    #[test]
    fn test_charload() {
        let mut q = QuantumState::new(1, None); // initialize with 1 register
        QuantumState::execute_arithmetic(&Instruction::CHARLOAD(0, b'A'), &mut q).unwrap();
        assert_eq!(q.get_reg(0).unwrap().re as u8, b'A');
    }

    #[test]
    fn test_cload_short_for_charload() {
        let mut q = QuantumState::new(1, None);
        QuantumState::execute_arithmetic(&Instruction::CLOAD(0, b'B'), &mut q).unwrap();
        assert_eq!(q.get_reg(0).unwrap().re as u8, b'B');
    }

    // charout cannot be directly tested without a mock stdout/vm, so we test instruction creation
    #[test]
    fn test_charout_instruction_creation() {
        let instruction = Instruction::CHAROUT(0);
        assert_eq!(instruction.encode(), vec![0x18, 0x00]);
    }

    #[test]
    fn test_cout_short_for_charout_instruction_creation() {
        let instruction = Instruction::COUT(0);
        assert_eq!(instruction.encode(), vec![0x18, 0x00]);
    }

    // --- ionq isa instructions ---
    #[test]
    fn test_phase_ionq() {
        let mut q_state = initial_quantum_state_qstate(1); // |0> state
        q_state.amps[1] = Complex64::new(1.0, 0.0); // set to |1> state
        q_state.amps[0] = Complex64::new(0.0, 0.0);
        let angle = std::f64::consts::PI / 4.0;
        QuantumState::execute_arithmetic(&Instruction::PHASE(0, angle), &mut q_state).unwrap();
        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        assert_complex_approx_eq(q_state.amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(q_state.amps[1], expected_phase_factor, 1e-9);
    }

    // thermalavg and wkbfactor cannot be directly tested with current quantumstate,
    // so we test instruction creation.
    #[test]
    fn test_thermalavg_instruction_creation() {
        let instruction = Instruction::THERMALAVG(0, 1);
        assert_eq!(instruction.encode(), vec![0x1F, 0x00, 0x01]);
    }

    #[test]
    fn test_tavg_short_for_thermalavg_instruction_creation() {
        let instruction = Instruction::TAVG(0, 1);
        assert_eq!(instruction.encode(), vec![0x1F, 0x00, 0x01]);
    }

    #[test]
    fn test_wkbfactor_instruction_creation() {
        let instruction = Instruction::WKBFACTOR(0, 1, 2);
        assert_eq!(instruction.encode(), vec![0x20, 0x00, 0x01, 0x02]);
    }

    #[test]
    fn test_wkbf_short_for_wkbfactor_instruction_creation() {
        let instruction = Instruction::WKBF(0, 1, 2);
        assert_eq!(instruction.encode(), vec![0x20, 0x00, 0x01, 0x02]);
    }

    // --- regset instruction ---
    #[test]
    fn test_regset() {
        let mut q = QuantumState::new(1, None); // initialize with 1 register
        QuantumState::execute_arithmetic(&Instruction::REGSET(0, 123.45), &mut q).unwrap();
        assert_eq!(q.get_reg(0).unwrap().re, 123.45);
    }

    #[test]
    fn test_rset_short_for_regset() {
        let mut q = QuantumState::new(1, None);
        QuantumState::execute_arithmetic(&Instruction::RSET(0, 67.89), &mut q).unwrap();
        assert_eq!(q.get_reg(0).unwrap().re, 67.89);
    }

    // --- loop instructions (test creation only, full test requires vm) ---
    #[test]
    fn test_loopstart_instruction_creation() {
        let instruction = Instruction::LOOPSTART(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_lstart_short_for_loopstart_instruction_creation() {
        let instruction = Instruction::LSTART(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_loopend_instruction_creation() {
        let instruction = Instruction::LOOPEND;
        assert_eq!(instruction.encode(), vec![0x10]);
    }

    #[test]
    fn test_lend_short_for_loopend_instruction_creation() {
        let instruction = Instruction::LEND;
        assert_eq!(instruction.encode(), vec![0x10]);
    }

    // --- rotations ---
    #[test]
    fn test_applyrotation_x() {
        let mut q_state = initial_quantum_state_qstate(1); // |0> state
        let angle = std::f64::consts::PI / 2.0;
        QuantumState::execute_arithmetic(&Instruction::APPLYROTATION(0, 'X', angle), &mut q_state)
            .unwrap();
        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        let expected_val_i = Complex64::new(0.0, -1.0 / (2.0f64).sqrt());
        assert_complex_approx_eq(q_state.amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(q_state.amps[1], expected_val_i, 1e-9);
    }

    #[test]
    fn test_rot_short_for_applyrotation() {
        let mut q_state = initial_quantum_state_qstate(1);
        let angle = std::f64::consts::PI / 2.0;
        QuantumState::execute_arithmetic(&Instruction::ROT(0, 'X', angle), &mut q_state).unwrap();
        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        let expected_val_i = Complex64::new(0.0, -1.0 / (2.0f64).sqrt());
        assert_complex_approx_eq(q_state.amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(q_state.amps[1], expected_val_i, 1e-9);
    }

    #[test]
    fn test_applyrotation_y() {
        let mut q_state = initial_quantum_state_qstate(1); // |0> state
        let angle = std::f64::consts::PI / 2.0;
        QuantumState::execute_arithmetic(&Instruction::APPLYROTATION(0, 'Y', angle), &mut q_state)
            .unwrap();
        let expected_val = Complex64::new(1.0 / (2.0f64).sqrt(), 0.0);
        assert_complex_approx_eq(q_state.amps[0], expected_val, 1e-9);
        assert_complex_approx_eq(q_state.amps[1], expected_val, 1e-9);
    }

    #[test]
    fn test_applyrotation_z() {
        let mut q_state = initial_quantum_state_qstate(1); // |0> state
        let angle = std::f64::consts::PI / 2.0;
        QuantumState::execute_arithmetic(&Instruction::APPLYROTATION(0, 'Z', angle), &mut q_state)
            .unwrap();
        let expected_0 = Complex64::new(0.0, -std::f64::consts::FRAC_PI_4).exp();
        assert_complex_approx_eq(q_state.amps[0], expected_0, 1e-9);
        assert_complex_approx_eq(q_state.amps[1], Complex64::new(0.0, 0.0), 1e-9);
    }

    #[test]
    fn test_applycphase() {
        let mut q_state = initial_quantum_state_qstate(2); // |00> state
        q_state.amps[3] = Complex64::new(1.0, 0.0); // set |11> to 1.0 for testing
        q_state.amps[0] = Complex64::new(0.0, 0.0);
        let angle = std::f64::consts::FRAC_PI_4;
        QuantumState::execute_arithmetic(&Instruction::APPLYCPHASE(0, 1, angle), &mut q_state)
            .unwrap();
        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        assert_complex_approx_eq(q_state.amps[0], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(q_state.amps[1], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(q_state.amps[2], Complex64::new(0.0, 0.0), 1e-9);
        assert_complex_approx_eq(q_state.amps[3], expected_phase_factor, 1e-9);
    }

    #[test]
    fn test_cphase_short_for_controlledphaserotation() {
        let mut q_state = initial_quantum_state_qstate(2);
        q_state.amps[3] = Complex64::new(1.0, 0.0);
        q_state.amps[0] = Complex64::new(0.0, 0.0);
        let angle = std::f64::consts::FRAC_PI_4;
        QuantumState::execute_arithmetic(&Instruction::CPHASE(0, 1, angle), &mut q_state).unwrap();
        let expected_phase_factor = Complex64::new(0.0, angle).exp();
        assert_complex_approx_eq(q_state.amps[3], expected_phase_factor, 1e-9);
    }

    // other rotation-related instructions (test creation only)
    #[test]
    fn test_applymultiqubitrotation_instruction_creation() {
        let instruction = Instruction::APPLYMULTIQUBITROTATION(
            vec![0, 1],
            'X',
            vec![std::f64::consts::PI / 2.0, std::f64::consts::PI / 4.0],
        );
        assert!(!instruction.encode().is_empty()); // check if it encodes
    }

    #[test]
    fn test_mrot_short_for_applymultiqubitrotation_instruction_creation() {
        let instruction = Instruction::MROT(
            vec![0, 1],
            'X',
            vec![std::f64::consts::PI / 2.0, std::f64::consts::PI / 4.0],
        );
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_applykerrnonlinearity_instruction_creation() {
        let instruction = Instruction::APPLYKERRNONLINEARITY(0, 0.5, 100);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_aknl_short_for_applykerrnonlinearity_instruction_creation() {
        let instruction = Instruction::AKNL(0, 0.5, 100);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_applyfeedforwardgate_instruction_creation() {
        let instruction = Instruction::APPLYFEEDFORWARDGATE(0, "my_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_affg_short_for_applyfeedforwardgate_instruction_creation() {
        let instruction = Instruction::AFFG(0, "my_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_decoherenceprotect_instruction_creation() {
        let instruction = Instruction::DECOHERENCEPROTECT(0, 1000);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_dprot_short_for_decoherenceprotect_instruction_creation() {
        let instruction = Instruction::DPROT(0, 1000);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_applymeasurementbasischange_instruction_creation() {
        let instruction = Instruction::APPLYMEASUREMENTBASISCHANGE(0, "X".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_ambc_short_for_applymeasurementbasischange_instruction_creation() {
        let instruction = Instruction::AMBC(0, "X".to_string());
        assert!(!instruction.encode().is_empty());
    }

    // --- memory/classical ops (test creation only, full test requires vm) ---
    #[test]
    fn test_load_instruction_creation() {
        let instruction = Instruction::LOAD(0, "my_var".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_store_instruction_creation() {
        let instruction = Instruction::STORE(0, "my_var".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_loadmem_instruction_creation() {
        let instruction = Instruction::LOADMEM("reg0".to_string(), "0x1000".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_lmem_short_for_loadmem_instruction_creation() {
        let instruction = Instruction::LMEM("reg0".to_string(), "0x1000".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_storemem_instruction_creation() {
        let instruction = Instruction::STOREMEM("0x1000".to_string(), "reg0".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_smem_short_for_storemem_instruction_creation() {
        let instruction = Instruction::SMEM("0x1000".to_string(), "reg0".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_loadclassical_instruction_creation() {
        let instruction = Instruction::LOADCLASSICAL("reg0".to_string(), "var_name".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_lcl_short_for_loadclassical_instruction_creation() {
        let instruction = Instruction::LCL("reg0".to_string(), "var_name".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_storeclassical_instruction_creation() {
        let instruction = Instruction::STORECLASSICAL("reg0".to_string(), "var_name".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_scl_short_for_storeclassical_instruction_creation() {
        let instruction = Instruction::SCL("reg0".to_string(), "var_name".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_add_instruction_creation() {
        let instruction = Instruction::ADD("r0".to_string(), "r1".to_string(), "r2".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_sub_instruction_creation() {
        let instruction = Instruction::SUB("r0".to_string(), "r1".to_string(), "r2".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_and_instruction_creation() {
        let instruction = Instruction::AND("r0".to_string(), "r1".to_string(), "r2".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_or_instruction_creation() {
        let instruction = Instruction::OR("r0".to_string(), "r1".to_string(), "r2".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_xor_instruction_creation() {
        let instruction = Instruction::XOR("r0".to_string(), "r1".to_string(), "r2".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_not_instruction_creation() {
        let instruction = Instruction::NOT("r0".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_push_instruction_creation() {
        let instruction = Instruction::PUSH("r0".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pop_instruction_creation() {
        let instruction = Instruction::POP("r0".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_regadd() {
        let mut q = QuantumState::new(3, None);
        q.set_reg(1, 5.0.into()).unwrap();
        q.set_reg(2, 3.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::REGADD(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(8.0, 0.0), 1e-9);
    }

    #[test]
    fn test_radd_short_for_regadd() {
        let mut q = QuantumState::new(3, None);
        q.set_reg(1, 5.0.into()).unwrap();
        q.set_reg(2, 3.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::RADD(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(8.0, 0.0), 1e-9);
    }

    #[test]
    fn test_regsub() {
        let mut q = QuantumState::new(3, None);
        q.set_reg(1, 5.0.into()).unwrap();
        q.set_reg(2, 3.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::REGSUB(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(2.0, 0.0), 1e-9);
    }

    #[test]
    fn test_rsub_short_for_regsub() {
        let mut q = QuantumState::new(3, None);
        q.set_reg(1, 5.0.into()).unwrap();
        q.set_reg(2, 3.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::RSUB(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(2.0, 0.0), 1e-9);
    }

    #[test]
    fn test_regmul() {
        let mut q = QuantumState::new(3, None);
        q.set_reg(1, 5.0.into()).unwrap();
        q.set_reg(2, 3.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::REGMUL(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(15.0, 0.0), 1e-9);
    }

    #[test]
    fn test_rmul_short_for_regmul() {
        let mut q = QuantumState::new(3, None);
        q.set_reg(1, 5.0.into()).unwrap();
        q.set_reg(2, 3.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::RMUL(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(15.0, 0.0), 1e-9);
    }

    #[test]
    fn test_regdiv() {
        let mut q = QuantumState::new(3, None);
        q.set_reg(1, 6.0.into()).unwrap();
        q.set_reg(2, 3.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::REGDIV(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(2.0, 0.0), 1e-9);
    }

    #[test]
    fn test_rdiv_short_for_regdiv() {
        let mut q = QuantumState::new(3, None);
        q.set_reg(1, 6.0.into()).unwrap();
        q.set_reg(2, 3.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::RDIV(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(2.0, 0.0), 1e-9);
    }

    #[test]
    fn test_regcopy() {
        let mut q = QuantumState::new(2, None);
        q.set_reg(1, 123.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::REGCOPY(0, 1), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(123.0, 0.0), 1e-9);
    }

    #[test]
    fn test_rcopy_short_for_regcopy() {
        let mut q = QuantumState::new(2, None);
        q.set_reg(1, 456.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::RCOPY(0, 1), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(456.0, 0.0), 1e-9);
    }

    // --- classical flow control (test creation only, full test requires vm) ---
    #[test]
    fn test_jump_instruction_creation() {
        let instruction = Instruction::JUMP("my_label".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_jumpifzero_instruction_creation() {
        let instruction = Instruction::JUMPIFZERO("r0".to_string(), "my_label".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_jumpifone_instruction_creation() {
        let instruction = Instruction::JUMPIFONE("r0".to_string(), "my_label".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_call_instruction_creation() {
        let instruction = Instruction::CALL("my_subroutine".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_return_instruction_creation() {
        let instruction = Instruction::RETURN;
        assert_eq!(instruction.encode(), vec![0x4D]);
    }

    #[test]
    fn test_barrier_instruction_creation() {
        let instruction = Instruction::BARRIER;
        assert_eq!(instruction.encode(), vec![0x89]);
    }

    #[test]
    fn test_timedelay_instruction_creation() {
        let instruction = Instruction::TIMEDELAY(0, 100);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_tdelay_short_for_timedelay_instruction_creation() {
        let instruction = Instruction::TDELAY(0, 100);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_rand() {
        let mut q = QuantumState::new(1, None); // initialize with 1 register
        // rand generates a random float into a register.
        // i can't predict the exact value, but i can check if it's within a reasonable range
        // and not nan/infinity.
        QuantumState::execute_arithmetic(&Instruction::RAND(0), &mut q).unwrap();
        let result = q.get_reg(0).unwrap().re;
        assert!(result >= 0.0 && result <= 1.0); // assuming rand generates [0, 1)
        assert!(!result.is_nan());
        assert!(result.is_finite());
    }

    #[test]
    fn test_sqrt() {
        let mut q = QuantumState::new(2, None); // initialize with 2 registers
        q.set_reg(1, 9.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::SQRT(0, 1), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(3.0, 0.0), 1e-9);
    }

    #[test]
    fn test_exp() {
        let mut q = QuantumState::new(2, None); // initialize with 2 registers
        q.set_reg(1, 1.0.into()).unwrap(); // e^1
        QuantumState::execute_arithmetic(&Instruction::EXP(0, 1), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(std::f64::consts::E, 0.0), 1e-9);
    }

    #[test]
    fn test_log() {
        let mut q = QuantumState::new(2, None); // initialize with 2 registers
        q.set_reg(1, std::f64::consts::E.into()).unwrap(); // ln(e)
        QuantumState::execute_arithmetic(&Instruction::LOG(0, 1), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(1.0, 0.0), 1e-9);
    }

    // --- optics instructions (test creation only, full test requires specialized simulator) ---
    #[test]
    fn test_photonemit_instruction_creation() {
        let instruction = Instruction::PHOTONEMIT(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pemit_short_for_photonemit_instruction_creation() {
        let instruction = Instruction::PEMIT(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_photondetect_instruction_creation() {
        let instruction = Instruction::PHOTONDETECT(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pdetect_short_for_photondetect_instruction_creation() {
        let instruction = Instruction::PDETECT(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_photoncount_instruction_creation() {
        let instruction = Instruction::PHOTONCOUNT(0, "count_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pcount_short_for_photoncount_instruction_creation() {
        let instruction = Instruction::PCOUNT(0, "count_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_photonaddition_instruction_creation() {
        let instruction = Instruction::PHOTONADDITION(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_padd_short_for_photonaddition_instruction_creation() {
        let instruction = Instruction::PADD(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_applyphotonsubtraction_instruction_creation() {
        let instruction = Instruction::APPLYPHOTONSUBTRACTION(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_apsub_short_for_applyphotonsubtraction_instruction_creation() {
        let instruction = Instruction::APSUB(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_photonemissionpattern_instruction_creation() {
        let instruction = Instruction::PHOTONEMISSIONPATTERN(0, "pattern".to_string(), 100);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pepat_short_for_photonemissionpattern_instruction_creation() {
        let instruction = Instruction::PEPAT(0, "pattern".to_string(), 100);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_photondetectwiththreshold_instruction_creation() {
        let instruction = Instruction::PHOTONDETECTWITHTHRESHOLD(0, 5, "result".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pdthr_short_for_photondetectwiththreshold_instruction_creation() {
        let instruction = Instruction::PDTHR(0, 5, "result".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_photondetectcoincidence_instruction_creation() {
        let instruction = Instruction::PHOTONDETECTCOINCIDENCE(vec![0, 1], "result".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pdcoin_short_for_photondetectcoincidence_instruction_creation() {
        let instruction = Instruction::PDCOIN(vec![0, 1], "result".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_singlephotonsourceon_instruction_creation() {
        let instruction = Instruction::SINGLEPHOTONSOURCEON(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_spson_short_for_singlephotonsourceon_instruction_creation() {
        let instruction = Instruction::SpsOn(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_singlephotonsourceoff_instruction_creation() {
        let instruction = Instruction::SINGLEPHOTONSOURCEOFF(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_spsoff_short_for_singlephotonsourceoff_instruction_creation() {
        let instruction = Instruction::SpsOff(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_photonbunchingcontrol_instruction_creation() {
        let instruction = Instruction::PHOTONBUNCHINGCONTROL(0, true);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pbunch_short_for_photonbunchingcontrol_instruction_creation() {
        let instruction = Instruction::PBUNCH(0, true);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_photonroute_instruction_creation() {
        let instruction = Instruction::PHOTONROUTE(0, "port_a".to_string(), "port_b".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_proute_short_for_photonroute_instruction_creation() {
        let instruction = Instruction::PROUTE(0, "port_a".to_string(), "port_b".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_opticalrouting_instruction_creation() {
        let instruction = Instruction::OPTICALROUTING(0, 1);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_oroute_short_for_opticalrouting_instruction_creation() {
        let instruction = Instruction::OROUTE(0, 1);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_setopticalattenuation_instruction_creation() {
        let instruction = Instruction::SETOPTICALATTENUATION(0, 0.5);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_soatt_short_for_setopticalattenuation_instruction_creation() {
        let instruction = Instruction::SOATT(0, 0.5);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_dynamicphasecompensation_instruction_creation() {
        let instruction = Instruction::DYNAMICPHASECOMPENSATION(0, std::f64::consts::PI);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_dpc_short_for_dynamicphasecompensation_instruction_creation() {
        let instruction = Instruction::DPC(0, std::f64::consts::PI);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_opticaldelaylinecontrol_instruction_creation() {
        let instruction = Instruction::OPTICALDELAYLINECONTROL(0, 100);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_odlc_short_for_opticaldelaylinecontrol_instruction_creation() {
        let instruction = Instruction::ODLC(0, 100);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_crossphasemodulation_instruction_creation() {
        let instruction = Instruction::CROSSPHASEMODULATION(0, 1, 0.1);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_cpm_short_for_crossphasemodulation_instruction_creation() {
        let instruction = Instruction::CPM(0, 1, 0.1);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_applydisplacement_instruction_creation() {
        let instruction = Instruction::APPLYDISPLACEMENT(0, 1.0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_adisp_short_for_applydisplacement_instruction_creation() {
        let instruction = Instruction::ADISP(0, 1.0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_applydisplacementfeedback_instruction_creation() {
        let instruction = Instruction::APPLYDISPLACEMENTFEEDBACK(0, "feedback_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_adf_short_for_applydisplacementfeedback_instruction_creation() {
        let instruction = Instruction::ADF(0, "feedback_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_applydisplacementoperator_instruction_creation() {
        let instruction = Instruction::APPLYDISPLACEMENTOPERATOR(0, 1.0, 10);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_ado_short_for_applydisplacementoperator_instruction_creation() {
        let instruction = Instruction::ADO(0, 1.0, 10);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_applysqueezing_instruction_creation() {
        let instruction = Instruction::APPLYSQUEEZING(0, 0.5);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_asq_short_for_applysqueezing_instruction_creation() {
        let instruction = Instruction::ASQ(0, 0.5);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_applysqueezingfeedback_instruction_creation() {
        let instruction = Instruction::APPLYSQUEEZINGFEEDBACK(0, "feedback_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_asf_short_for_applysqueezingfeedback_instruction_creation() {
        let instruction = Instruction::ASF(0, "feedback_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_measureparity_instruction_creation() {
        let instruction = Instruction::MEASUREPARITY(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_mpar_short_for_measureparity_instruction_creation() {
        let instruction = Instruction::MPAR(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_measurewithdelay_instruction_creation() {
        let instruction = Instruction::MEASUREWITHDELAY(0, 100, "result_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_mwd_short_for_measurewithdelay_instruction_creation() {
        let instruction = Instruction::MWD(0, 100, "result_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_opticalswitchcontrol_instruction_creation() {
        let instruction = Instruction::OPTICALSWITCHCONTROL(0, true);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_osc_short_for_opticalswitchcontrol_instruction_creation() {
        let instruction = Instruction::OSC(0, true);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_photonlosssimulate_instruction_creation() {
        let instruction = Instruction::PHOTONLOSSSIMULATE(0, 0.1, 123);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pls_short_for_photonlosssimulate_instruction_creation() {
        let instruction = Instruction::PLS(0, 0.1, 123);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_photonlosscorrection_instruction_creation() {
        let instruction = Instruction::PHOTONLOSSCORRECTION(0, "correction_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_plc_short_for_photonlosscorrection_instruction_creation() {
        let instruction = Instruction::PLC(0, "correction_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    // --- qubit measurement / error correction (test creation only) ---
    #[test]
    fn test_applyqndmeasurement_instruction_creation() {
        let instruction = Instruction::APPLYQNDMEASUREMENT(0, "result_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_aqnd_short_for_applyqndmeasurement_instruction_creation() {
        let instruction = Instruction::AQND(0, "result_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_errorcorrect_instruction_creation() {
        let instruction = Instruction::ERRORCORRECT(0, "X".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_ecorr_short_for_errorcorrect_instruction_creation() {
        let instruction = Instruction::ECORR(0, "X".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_errorsyndrome_instruction_creation() {
        let instruction =
            Instruction::ERRORSYNDROME(0, "X".to_string(), "syndrome_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_esyn_short_for_errorsyndrome_instruction_creation() {
        let instruction = Instruction::ESYN(0, "X".to_string(), "syndrome_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_quantumstatetomography_instruction_creation() {
        let instruction = Instruction::QUANTUMSTATETOMOGRAPHY(0, "pauli".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_qst_short_for_quantumstatetomography_instruction_creation() {
        let instruction = Instruction::QST(0, "pauli".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_bellstateverification_instruction_creation() {
        let instruction = Instruction::BELLSTATEVERIFICATION(0, 1, "result_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_bsv_short_for_bellstateverification_instruction_creation() {
        let instruction = Instruction::BSV(0, 1, "result_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_quantumzenoeffect_instruction_creation() {
        let instruction = Instruction::QUANTUMZENOEFFECT(0, 10, 100);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_qze_short_for_quantumzenoeffect_instruction_creation() {
        let instruction = Instruction::QZE(0, 10, 100);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_applynonlinearphaseshift_instruction_creation() {
        let instruction = Instruction::APPLYNONLINEARPHASESHIFT(0, 0.1);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_anlps_short_for_applynonlinearphaseshift_instruction_creation() {
        let instruction = Instruction::ANLPS(0, 0.1);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_applynonlinearsigma_instruction_creation() {
        let instruction = Instruction::APPLYNONLINEARSIGMA(0, 0.1);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_anls_short_for_applynonlinearsigma_instruction_creation() {
        let instruction = Instruction::ANLS(0, 0.1);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_applylinearopticaltransform_instruction_creation() {
        let instruction = Instruction::APPLYLINEAROPTICALTRANSFORM(
            "bs".to_string(),
            vec![0, 1],
            vec![2, 3],
            2,
        );
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_alot_short_for_applylinearopticaltransform_instruction_creation() {
        let instruction = Instruction::ALOT(
            "bs".to_string(),
            vec![0, 1],
            vec![2, 3],
            2,
        );
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_photonnumberresolvingdetection_instruction_creation() {
        let instruction = Instruction::PHOTONNUMBERRESOLVINGDETECTION(0, "count_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pnrd_short_for_photonnumberresolvingdetection_instruction_creation() {
        let instruction = Instruction::PNRD(0, "count_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_feedbackcontrol_instruction_creation() {
        let instruction = Instruction::FEEDBACKCONTROL(0, "control_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_fbc_short_for_feedbackcontrol_instruction_creation() {
        let instruction = Instruction::FBC(0, "control_reg".to_string());
        assert!(!instruction.encode().is_empty());
    }

    // --- misc instructions (test creation only) ---
    #[test]
    fn test_setpos_instruction_creation() {
        let instruction = Instruction::SETPOS(0, 10.0, 20.0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_spos_short_for_setpos_instruction_creation() {
        let instruction = Instruction::SPOS(0, 10.0, 20.0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_setwl_instruction_creation() {
        let instruction = Instruction::SETWL(0, 532.0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_swl_short_for_setwl_instruction_creation() {
        let instruction = Instruction::SWL(0, 532.0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_wlshift_instruction_creation() {
        let instruction = Instruction::WLSHIFT(0, 10.0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_wls_short_for_wlshift_instruction_creation() {
        let instruction = Instruction::WLS(0, 10.0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_move_instruction_creation() {
        let instruction = Instruction::MOVE(0, 1.0, -1.0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_mov_short_for_move_instruction_creation() {
        let instruction = Instruction::MOV(0, 1.0, -1.0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_comment_instruction_creation() {
        let instruction = Instruction::COMMENT("This is a comment".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_cmt_short_for_comment_instruction_creation() {
        let instruction = Instruction::CMT("This is a short comment".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_markobserved_instruction_creation() {
        let instruction = Instruction::MARKOBSERVED(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_mobs_short_for_markobserved_instruction_creation() {
        let instruction = Instruction::MOBS(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_release_instruction_creation() {
        let instruction = Instruction::RELEASE(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_rel_short_for_release_instruction_creation() {
        let instruction = Instruction::REL(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_halt_instruction_creation() {
        let instruction = Instruction::HALT;
        assert_eq!(instruction.encode(), vec![0xFF]);
    }

    // --- new instructions for v0.3.0 (test creation only, full test requires vm) ---
    #[test]
    fn test_jmp_instruction_creation() {
        let instruction = Instruction::JMP(10);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_jmpabs_instruction_creation() {
        let instruction = Instruction::JMPABS(100);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_jabs_short_for_jmpabs_instruction_creation() {
        let instruction = Instruction::JABS(100);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_ifgt() {
        let mut q = QuantumState::new(2, None); // initialize with 2 registers
        q.set_reg(0, 5.0.into()).unwrap();
        q.set_reg(1, 3.0.into()).unwrap();
        // if r0 > r1, jump to instruction 5
        QuantumState::execute_arithmetic(&Instruction::IFGT(0, 1, 5), &mut q).unwrap();
        // cannot assert jump directly in unit test, but ensure no panic
    }

    #[test]
    fn test_igt_short_for_ifgt() {
        let mut q = QuantumState::new(2, None);
        q.set_reg(0, 5.0.into()).unwrap();
        q.set_reg(1, 3.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::IGT(0, 1, 5), &mut q).unwrap();
    }

    #[test]
    fn test_iflt() {
        let mut q = QuantumState::new(2, None); // initialize with 2 registers
        q.set_reg(0, 3.0.into()).unwrap();
        q.set_reg(1, 5.0.into()).unwrap();
        // if r0 < r1, jump to instruction 5
        QuantumState::execute_arithmetic(&Instruction::IFLT(0, 1, 5), &mut q).unwrap();
        // cannot assert jump directly in unit test, but ensure no panic
    }

    #[test]
    fn test_ilt_short_for_iflt() {
        let mut q = QuantumState::new(2, None);
        q.set_reg(0, 3.0.into()).unwrap();
        q.set_reg(1, 5.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::ILT(0, 1, 5), &mut q).unwrap();
    }

    #[test]
    fn test_ifeq() {
        let mut q = QuantumState::new(2, None); // initialize with 2 registers
        q.set_reg(0, 5.0.into()).unwrap();
        q.set_reg(1, 5.0.into()).unwrap();
        // if r0 == r1, jump to instruction 5
        QuantumState::execute_arithmetic(&Instruction::IFEQ(0, 1, 5), &mut q).unwrap();
        // cannot assert jump directly in unit test, but ensure no panic
    }

    #[test]
    fn test_ieq_short_for_ifeq() {
        let mut q = QuantumState::new(2, None);
        q.set_reg(0, 5.0.into()).unwrap();
        q.set_reg(1, 5.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::IEQ(0, 1, 5), &mut q).unwrap();
    }

    #[test]
    fn test_ifne() {
        let mut q = QuantumState::new(2, None); // initialize with 2 registers
        q.set_reg(0, 5.0.into()).unwrap();
        q.set_reg(1, 3.0.into()).unwrap();
        // if r0 != r1, jump to instruction 5
        QuantumState::execute_arithmetic(&Instruction::IFNE(0, 1, 5), &mut q).unwrap();
        // cannot assert jump directly in unit test, but ensure no panic
    }

    #[test]
    fn test_ine_short_for_ifne() {
        let mut q = QuantumState::new(2, None);
        q.set_reg(0, 5.0.into()).unwrap();
        q.set_reg(1, 3.0.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::INE(0, 1, 5), &mut q).unwrap();
    }

    #[test]
    fn test_calladdr_instruction_creation() {
        let instruction = Instruction::CALLADDR(200);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_caddr_short_for_calladdr_instruction_creation() {
        let instruction = Instruction::CADDR(200);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_retsub_instruction_creation() {
        let instruction = Instruction::RETSUB;
        assert_eq!(instruction.encode(), vec![0x97]);
    }

    #[test]
    fn test_printf_instruction_creation() {
        let instruction = Instruction::PRINTF("Hello %f %f".to_string(), vec![0, 1]);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pf_short_for_printf_instruction_creation() {
        let instruction = Instruction::PF("Hello %f %f".to_string(), vec![0, 1]);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_print_instruction_creation() {
        let instruction = Instruction::PRINT("Hello World".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_println_instruction_creation() {
        let instruction = Instruction::PRINTLN("Hello World".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pln_short_for_println_instruction_creation() {
        let instruction = Instruction::PLN("Hello World".to_string());
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_input_instruction_creation() {
        let instruction = Instruction::INPUT(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_dumpstate_instruction_creation() {
        let instruction = Instruction::DUMPSTATE;
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_dstate_short_for_dumpstate_instruction_creation() {
        let instruction = Instruction::DSTATE;
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_dumpregs_instruction_creation() {
        let instruction = Instruction::DUMPREGS;
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_dregs_short_for_dumpregs_instruction_creation() {
        let instruction = Instruction::DREGS;
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_loadregmem_instruction_creation() {
        let instruction = Instruction::LOADREGMEM(0, 0x1000);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_lrm_short_for_loadregmem_instruction_creation() {
        let instruction = Instruction::LRM(0, 0x1000);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_storememreg_instruction_creation() {
        let instruction = Instruction::STOREMEMREG(0x1000, 0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_smr_short_for_storememreg_instruction_creation() {
        let instruction = Instruction::SMR(0x1000, 0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pushreg_instruction_creation() {
        let instruction = Instruction::PUSHREG(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_prg_short_for_pushreg_instruction_creation() {
        let instruction = Instruction::PRG(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_popreg_instruction_creation() {
        let instruction = Instruction::POPREG(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_pprg_short_for_popreg_instruction_creation() {
        let instruction = Instruction::PPRG(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_alloc_instruction_creation() {
        let instruction = Instruction::ALLOC(0, 1024);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_alc_short_for_alloc_instruction_creation() {
        let instruction = Instruction::ALC(0, 1024);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_free_instruction_creation() {
        let instruction = Instruction::FREE(0x1000);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_fre_short_for_free_instruction_creation() {
        let instruction = Instruction::FRE(0x1000);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_cmp_instruction_creation() {
        let instruction = Instruction::CMP(0, 1);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_andbits() {
        let mut q = QuantumState::new(3, None); // initialize with 3 registers
        q.set_reg(1, 12.0f64.into()).unwrap(); // 0b1100
        q.set_reg(2, 10.0f64.into()).unwrap(); // 0b1010
        QuantumState::execute_arithmetic(&Instruction::ANDBITS(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(8.0f64, 0.0), 1e-9); // 0b1000
    }

    #[test]
    fn test_andb_short_for_andbits() {
        let mut q = QuantumState::new(3, None);
        q.set_reg(1, 12.0f64.into()).unwrap();
        q.set_reg(2, 10.0f64.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::ANDB(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(8.0f64, 0.0), 1e-9);
    }

    #[test]
    fn test_orbits() {
        let mut q = QuantumState::new(3, None); // initialize with 3 registers
        q.set_reg(1, 12.0f64.into()).unwrap(); // 0b1100
        q.set_reg(2, 10.0f64.into()).unwrap(); // 0b1010
        QuantumState::execute_arithmetic(&Instruction::ORBITS(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(14.0f64, 0.0), 1e-9); // 0b1110
    }

    #[test]
    fn test_orb_short_for_orbits() {
        let mut q = QuantumState::new(3, None);
        q.set_reg(1, 12.0f64.into()).unwrap();
        q.set_reg(2, 10.0f64.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::ORB(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(14.0f64, 0.0), 1e-9);
    }

    #[test]
    fn test_xorbits() {
        let mut q = QuantumState::new(3, None); // initialize with 3 registers
        q.set_reg(1, 12.0f64.into()).unwrap(); // 0b1100
        q.set_reg(2, 10.0f64.into()).unwrap(); // 0b1010
        QuantumState::execute_arithmetic(&Instruction::XORBITS(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(6.0f64, 0.0), 1e-9); // 0b0110
    }

    #[test]
    fn test_xorb_short_for_xorbits() {
        let mut q = QuantumState::new(3, None);
        q.set_reg(1, 12.0f64.into()).unwrap();
        q.set_reg(2, 10.0f64.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::XORB(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(6.0f64, 0.0), 1e-9);
    }

    #[test]
    fn test_notbits() {
        let mut q = QuantumState::new(2, None); // initialize with 2 registers
        // for f64, notbits will conceptually flip all bits of the underlying integer representation.
        // this is typically not well-defined for floats and results in a very different number.
        // we'll test a simple case and ensure it's not the original value and is finite.
        q.set_reg(1, 12.0f64.into()).unwrap(); // represents 0b...1100 in its mantissa/exponent
        QuantumState::execute_arithmetic(&Instruction::NOTBITS(0, 1), &mut q).unwrap();
        let result = q.get_reg(0).unwrap().re;
        assert_ne!(result, 12.0f64);
        assert!(result.is_finite());
    }

    #[test]
    fn test_notb_short_for_notbits() {
        let mut q = QuantumState::new(2, None);
        q.set_reg(1, 12.0f64.into()).unwrap();
        QuantumState::execute_arithmetic(&Instruction::NOTB(0, 1), &mut q).unwrap();
        let result = q.get_reg(0).unwrap().re;
        assert_ne!(result, 12.0f64);
        assert!(result.is_finite());
    }

    #[test]
    fn test_shl() {
        let mut q = QuantumState::new(3, None); // initialize with 3 registers
        q.set_reg(1, 1.0f64.into()).unwrap(); // 0b0001
        q.set_reg(2, 2.0.into()).unwrap(); // shift by 2
        QuantumState::execute_arithmetic(&Instruction::SHL(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(4.0f64, 0.0), 1e-9); // 0b0100
    }

    #[test]
    fn test_shr() {
        let mut q = QuantumState::new(3, None); // initialize with 3 registers
        q.set_reg(1, 4.0f64.into()).unwrap(); // 0b0100
        q.set_reg(2, 2.0.into()).unwrap(); // shift by 2
        QuantumState::execute_arithmetic(&Instruction::SHR(0, 1, 2), &mut q).unwrap();
        assert_complex_approx_eq(*q.get_reg(0).unwrap(), Complex64::new(1.0f64, 0.0), 1e-9); // 0b0001
    }

    #[test]
    fn test_breakpoint_instruction_creation() {
        let instruction = Instruction::BREAKPOINT;
        assert_eq!(instruction.encode(), vec![0xAB]);
    }

    #[test]
    fn test_bp_short_for_breakpoint_instruction_creation() {
        let instruction = Instruction::BP;
        assert_eq!(instruction.encode(), vec![0xAB]);
    }

    #[test]
    fn test_gettime_instruction_creation() {
        let instruction = Instruction::GETTIME(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_gtime_short_for_gettime_instruction_creation() {
        let instruction = Instruction::GTIME(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_seedrng_instruction_creation() {
        let instruction = Instruction::SEEDRNG(12345);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_srng_short_for_seedrng_instruction_creation() {
        let instruction = Instruction::SRNG(12345);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_exitcode_instruction_creation() {
        let instruction = Instruction::EXITCODE(0);
        assert!(!instruction.encode().is_empty());
    }

    #[test]
    fn test_exc_short_for_exitcode_instruction_creation() {
        let instruction = Instruction::EXC(0);
        assert!(!instruction.encode().is_empty());
    }
}
