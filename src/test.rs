 use qoa::runtime::QuantumState; // import quantum state struct
use qoa::instructions::Instruction; // import instructions enum

#[test]
fn test_regadd_nan_propagation() {
    let mut q = QuantumState::new();
    q.set(1, f64::NAN).unwrap();
    q.set(2, 1.0).unwrap();
    QuantumState::execute_arithmetic(&Instruction::RegAdd(0, 1, 2), &mut q).unwrap();
    assert!(q.get(0).unwrap().is_nan());
    assert!(q.status.nan);
}

#[test]
fn test_regdiv_by_zero() {
    let mut q = QuantumState::new();
    q.set(1, 10.0).unwrap();
    q.set(2, 0.0).unwrap();
    QuantumState::execute_arithmetic(&Instruction::RegDiv(0, 1, 2), &mut q).unwrap();
    assert!(q.get(0).unwrap().is_nan());
    assert!(q.status.div_by_zero);
}

#[test]
fn test_regmul_overflow() {
    let mut q = QuantumState::new();
    q.set(1, 1e308).unwrap();
    q.set(2, 1e308).unwrap();
    QuantumState::execute_arithmetic(&Instruction::RegMul(0, 1, 2), &mut q).unwrap();
    assert!(!q.get(0).unwrap().is_finite());
    assert!(q.status.overflow);
}

#[test]
fn test_regcopy() {
    let mut q = QuantumState::new();
    q.set(3, 42.42).unwrap();
    QuantumState::execute_arithmetic(&Instruction::RegCopy(4, 3), &mut q).unwrap();
    assert_eq!(q.get(4).unwrap(), 42.42);
}

#[test]
fn test_regsub_nan() {
    let mut q = QuantumState::new();
    q.set(1, f64::NAN).unwrap();
    q.set(2, 1.0).unwrap();
    QuantumState::execute_arithmetic(&Instruction::RegSub(0, 1, 2), &mut q).unwrap();
    assert!(q.get(0).unwrap().is_nan());
    assert!(q.status.nan);
}

#[test]
fn test_bounds_check_invalid_read() {
    let q = QuantumState::new();
    let result = q.get(99);
    assert!(result.is_err());
}

#[test]
fn test_bounds_check_invalid_write() {
    let mut q = QuantumState::new();
    let result = q.set(99, 1.23);
    assert!(result.is_err());
}
