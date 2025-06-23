use qoa::instructions::Instruction;
use qoa::runtime::quantum_state::{NoiseConfig, QuantumState}; // Import NoiseConfig

/// Allocates a new `QuantumState` with room for at least `n` registers.
fn test_state(n: usize) -> QuantumState {
    QuantumState::new(n, None) // Pass None for noise_config as it's a test setup
}

#[test]
fn test_regadd_nan_propagation() {
    let mut q = test_state(3); // for regs 0, 1, 2
    q.set(1, f64::NAN.into()).unwrap();
    q.set(2, 1.0.into()).unwrap();
    QuantumState::execute_arithmetic(&Instruction::RegAdd(0, 1, 2), &mut q).unwrap();
    assert!(q.get(0).unwrap().re.is_nan());
    assert!(q.status.nan);
}

#[test]
fn test_regdiv_by_zero() {
    let mut q = test_state(3);
    q.set(1, 10.0.into()).unwrap();
    q.set(2, 0.0.into()).unwrap();
    QuantumState::execute_arithmetic(&Instruction::RegDiv(0, 1, 2), &mut q).unwrap();
    assert!(q.get(0).unwrap().re.is_nan());
    assert!(q.status.div_by_zero);
}

#[test]
fn test_regmul_overflow() {
    let mut q = test_state(3);
    q.set(1, 1e308.into()).unwrap();
    q.set(2, 1e308.into()).unwrap();
    QuantumState::execute_arithmetic(&Instruction::RegMul(0, 1, 2), &mut q).unwrap();
    assert!(!q.get(0).unwrap().re.is_finite());
    assert!(q.status.overflow);
}

#[test]
fn test_regcopy() {
    let mut q = test_state(5); // using reg 3 and 4
    q.set(3, 42.42.into()).unwrap();
    QuantumState::execute_arithmetic(&Instruction::RegCopy(4, 3), &mut q).unwrap();
    assert_eq!(q.get(4).unwrap().re, 42.42);
}

#[test]
fn test_regsub_nan() {
    let mut q = test_state(3);
    q.set(1, f64::NAN.into()).unwrap();
    q.set(2, 1.0.into()).unwrap();
    QuantumState::execute_arithmetic(&Instruction::RegSub(0, 1, 2), &mut q).unwrap();
    assert!(q.get(0).unwrap().re.is_nan());
    assert!(q.status.nan);
}

#[test]
fn test_bounds_check_invalid_read() {
    let q = test_state(5);
    let result = q.get(99);
    assert!(result.is_none()); // Option, so check with is_none()
}

#[test]
fn test_bounds_check_invalid_write() {
    let mut q = test_state(5);
    let result = q.set(99, 1.23.into());
    assert!(result.is_err());
}
