use criterion::measurement::WallTime;
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use num_complex::Complex64;
use qoa::vectorization::{
    apply_cnot_vectorized, apply_controlled_phase_rotation_vectorized,
    apply_controlled_swap_vectorized, apply_cz_vectorized, apply_hadamard_vectorized,
    apply_phaseshift_vectorized, apply_reset_all_vectorized, apply_reset_vectorized,
    apply_rx_vectorized, apply_ry_vectorized, apply_rz_vectorized, apply_s_vectorized,
    apply_swap_vectorized, apply_t_vectorized,
};
use std::f64::consts::PI;
use std::io::Write;
use tempfile::tempfile;

// custom criterion configuration for all benchmarks
// this allows for setting global parameters like sample size and measurement time
fn custom_criterion_config() -> Criterion<WallTime> {
    Criterion::default()
        .confidence_level(0.99999999999999) // confidence level for result (gets even higher with more samples!)
        // in reality, with the sample size i've set, confidence level is ~85%-99%. this can be improved with more samples and sampling time.
        .sample_size(1000) // sample size for all benchmarks (20-30 recommended for dev, very inaccurate)
        .measurement_time(std::time::Duration::from_secs(10)) // measurement duration (1-2s for dev, very inaccurate)
        .warm_up_time(std::time::Duration::from_secs(5)) // warm-up duration (0.5-1s for dev, very inaccurate)
        .with_plots() // enables generating plot data
}

// initial quantum state vector (|0⟩)
fn initial_state(num_qubits: usize) -> Vec<Complex64> {
    let size = 1 << num_qubits;
    let mut amps = vec![Complex64::new(0.0, 0.0); size];
    amps[0] = Complex64::new(1.0, 0.0);
    amps
}

// common values used across benchmarks
struct BenchmarkParams {
    norm_factor: Complex64,
    angle_pi_2: f64,
    angle_pi_3: f64,
    angle_pi_4: f64,
}

fn get_params() -> BenchmarkParams {
    BenchmarkParams {
        norm_factor: Complex64::new(1.0 / (2.0f64).sqrt(), 0.0),
        angle_pi_2: PI / 2.0,
        angle_pi_3: PI / 3.0,
        angle_pi_4: PI / 4.0,
    }
}

// benchmarks for various quantum gates
fn quantum_gate_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_gates");

    // define qubit counts based on build configuration
    // for faster development runs (debug mode), use fewer qubits
    // for more comprehensive, optimized runs (release mode), use a wider range
    #[cfg(debug_assertions)]
    let qubit_counts = vec![4, 8]; // for faster development runs

    #[cfg(not(debug_assertions))]
    let qubit_counts = vec![4, 8, 12, 16, 20]; // for optimized release mode (full range)

    // common parameters
    let params = get_params();

    // run benchmarks for each qubit count
    for &num_qubits in &qubit_counts {
        let size = 1 << num_qubits;
        group.throughput(Throughput::Elements(size as u64));

        // single-qubit gates
        benchmark_single_qubit_gates(&mut group, num_qubits, &params);

        // two-qubit gates (only if we have at least 2 qubits)
        if num_qubits >= 2 {
            benchmark_two_qubit_gates(&mut group, num_qubits, &params);
        }

        // three-qubit gates (only if we have at least 3 qubits)
        if num_qubits >= 3 {
            benchmark_three_qubit_gates(&mut group, num_qubits, &params);
        }
    }

    group.finish();
}

// single qubit gate benchmarks
fn benchmark_single_qubit_gates(
    group: &mut criterion::BenchmarkGroup<WallTime>,
    num_qubits: usize,
    params: &BenchmarkParams,
) {
    // hadamard gate
    group.bench_function(format!("hadamard_{}_qubits", num_qubits), |b| {
        b.iter(|| {
            let mut amps = initial_state(num_qubits);
            apply_hadamard_vectorized(
                black_box(&mut amps),
                black_box(params.norm_factor),
                black_box(0),
            );
        });
    });

    // rotation gates
    group.bench_function(format!("rx_{}_qubits", num_qubits), |b| {
        b.iter(|| {
            let mut amps = initial_state(num_qubits);
            apply_rx_vectorized(
                black_box(&mut amps),
                black_box(0),
                black_box(params.angle_pi_2),
            );
        });
    });

    group.bench_function(format!("ry_{}_qubits", num_qubits), |b| {
        b.iter(|| {
            let mut amps = initial_state(num_qubits);
            apply_ry_vectorized(
                black_box(&mut amps),
                black_box(0),
                black_box(params.angle_pi_2),
            );
        });
    });

    group.bench_function(format!("rz_{}_qubits", num_qubits), |b| {
        b.iter(|| {
            let mut amps = initial_state(num_qubits);
            apply_rz_vectorized(
                black_box(&mut amps),
                black_box(0),
                black_box(params.angle_pi_2),
            );
        });
    });

    // phase shift gate
    group.bench_function(format!("phaseshift_{}_qubits", num_qubits), |b| {
        b.iter(|| {
            let mut amps = initial_state(num_qubits);
            apply_phaseshift_vectorized(
                black_box(&mut amps),
                black_box(0),
                black_box(params.angle_pi_3),
            );
        });
    });

    // reset gate
    group.bench_function(format!("reset_{}_qubits", num_qubits), |b| {
        b.iter(|| {
            let mut amps = initial_state(num_qubits);
            apply_reset_vectorized(black_box(&mut amps), black_box(0));
        });
    });

    // reset all gates
    group.bench_function(format!("reset_all_{}_qubits", num_qubits), |b| {
        b.iter(|| {
            let mut amps = initial_state(num_qubits);
            apply_reset_all_vectorized(black_box(&mut amps));
        });
    });

    // t-gate
    group.bench_function(format!("t_gate_{}_qubits", num_qubits), |b| {
        b.iter(|| {
            let mut amps = initial_state(num_qubits);
            apply_t_vectorized(black_box(&mut amps), black_box(0));
        });
    });

    // s-gate
    group.bench_function(format!("s_gate_{}_qubits", num_qubits), |b| {
        b.iter(|| {
            let mut amps = initial_state(num_qubits);
            apply_s_vectorized(black_box(&mut amps), black_box(0));
        });
    });
}

// two qubit gate benchmarks
fn benchmark_two_qubit_gates(
    group: &mut criterion::BenchmarkGroup<WallTime>,
    num_qubits: usize,
    params: &BenchmarkParams,
) {
    // cnot gate
    group.bench_function(format!("cnot_{}_qubits", num_qubits), |b| {
        b.iter(|| {
            let mut amps = initial_state(num_qubits);
            apply_cnot_vectorized(black_box(&mut amps), black_box(0), black_box(1));
        });
    });

    // swap gate
    group.bench_function(format!("swap_{}_qubits", num_qubits), |b| {
        b.iter(|| {
            let mut amps = initial_state(num_qubits);
            apply_swap_vectorized(black_box(&mut amps), black_box(0), black_box(1));
        });
    });

    // cz gate
    group.bench_function(format!("cz_gate_{}_qubits", num_qubits), |b| {
        b.iter(|| {
            let mut amps = initial_state(num_qubits);
            apply_cz_vectorized(black_box(&mut amps), black_box(0), black_box(1));
        });
    });

    // controlled phase rotation
    group.bench_function(
        format!("controlled_phase_rotation_{}_qubits", num_qubits),
        |b| {
            b.iter(|| {
                let mut amps = initial_state(num_qubits);
                apply_controlled_phase_rotation_vectorized(
                    black_box(&mut amps),
                    black_box(0),
                    black_box(1),
                    black_box(params.angle_pi_4),
                );
            });
        },
    );
}

// three qubit gate benchmarks
fn benchmark_three_qubit_gates(
    group: &mut criterion::BenchmarkGroup<WallTime>,
    num_qubits: usize,
    _params: &BenchmarkParams,
) {
    // controlled swap (fredkin) gate
    group.bench_function(format!("controlled_swap_{}_qubits", num_qubits), |b| {
        b.iter(|| {
            let mut amps = initial_state(num_qubits);
            apply_controlled_swap_vectorized(
                black_box(&mut amps),
                black_box(2),
                black_box(0),
                black_box(1),
            );
        });
    });
}

// system stress benchmark
fn system_stress_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("system_stress");

    // use smaller data size for faster runs
    const IO_DATA_SIZE: usize = 2048 * 2048; // 2mb
    // pre-allocate data once, outside the benchmark
    let dummy_data: Vec<u8> = (0..IO_DATA_SIZE).map(|i| (i % 256) as u8).collect();

    group.bench_function("cpu_and_io_stress", |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..500_000 {
                sum += (i as f64).sin() * (i as f64).cos();
            }
            black_box(sum);

            // io part
            let mut temp_file = tempfile().expect("failed to create temp file");
            temp_file
                .write_all(black_box(&dummy_data))
                .expect("failed to write to temp file");
        });
    });

    group.finish();
}

// use custom criterion configuration for all benchmark targets
criterion_group! {
    name = benches;
    config = custom_criterion_config(); // apply the custom configuration
    targets = quantum_gate_benchmarks, system_stress_benchmarks
}
criterion_main!(benches);
