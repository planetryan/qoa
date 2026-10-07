// fitxategi honek simulagailuaren kernelak neurtzen ditu criterion-ekin.
// `cargo bench` exekutatu; emaitzak target/criterion-en gordetzen dira.
//
// throughput-a bytetan ematen da: ate batek egoera osoa irakurri eta idazten du (16 B anplitude bakoitzeko,
// bi aldiz), beraz criterion-ek GB/s erakusten ditu eta makinaren memoria-banda zabalerarekin konpara daiteke.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use qoa::fock;
use qoa::program::Gate;
use qoa::state::State;
use qoa::vm::{self, RunConfig};

// ate bakoitzak egoera irakurri eta idazten du: 2 * 16 byte anplitude bakoitzeko
fn bytes_touched(n_qubits: usize) -> u64 {
    2 * 16 * (1u64 << n_qubits)
}

fn qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("qubit_gates");
    group.sample_size(20);
    for n in [16usize, 20, 24] {
        group.throughput(Throughput::Bytes(bytes_touched(n)));
        let mut s = State::new(n, 0, 2).unwrap();
        let h = Gate::H.matrix();
        // qubit baxua (bloke txiki asko) eta altua (bloke handi gutxi)
        group.bench_with_input(BenchmarkId::new("h_low", n), &n, |b, _| {
            b.iter(|| s.apply_1q(0, black_box(h), 0))
        });
        group.bench_with_input(BenchmarkId::new("h_high", n), &n, |b, &n| {
            b.iter(|| s.apply_1q(n - 1, black_box(h), 0))
        });
        group.bench_with_input(BenchmarkId::new("cnot", n), &n, |b, &n| {
            b.iter(|| s.apply_x(n / 2, 1))
        });
        group.bench_with_input(BenchmarkId::new("rz_diagonal", n), &n, |b, &n| {
            b.iter(|| s.apply_1q(n / 2, black_box(Gate::RZ(0.3).matrix()), 0))
        });
        group.bench_with_input(BenchmarkId::new("cz_phase", n), &n, |b, &n| {
            b.iter(|| s.apply_phase_mask(1 | (1 << (n - 1)), std::f64::consts::PI))
        });
        group.bench_with_input(BenchmarkId::new("swap", n), &n, |b, &n| {
            b.iter(|| s.apply_swap(0, n - 1, 0))
        });
        group.bench_with_input(BenchmarkId::new("prob_one", n), &n, |b, &n| {
            b.iter(|| black_box(s.prob_one(n / 2)))
        });
    }
    group.finish();
}

fn mode_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("mode_ops");
    group.sample_size(10);
    // 3 modu, muga 12: 1728 anplitude; 10 qubitekin 1.7 milioi
    let mut s = State::new(10, 3, 12).unwrap();
    group.throughput(Throughput::Bytes(2 * 16 * s.len() as u64));
    let d = fock::displacement(num_complex::Complex64::new(0.5, 0.1), 12);
    let bs = fock::beamsplitter(std::f64::consts::FRAC_PI_4, 0.0, 12);
    group.bench_function("displace", |b| b.iter(|| s.apply_mode_matrix(1, &d)));
    group.bench_function("beamsplitter", |b| {
        b.iter(|| s.apply_two_mode_matrix(0, 2, &bs))
    });
    group.bench_function("emit", |b| b.iter(|| s.emit(3, 1)));
    group.finish();

    let mut group = c.benchmark_group("fock_matrices");
    group.bench_function("displacement_d20", |b| {
        b.iter(|| fock::displacement(black_box(num_complex::Complex64::new(1.2, 0.3)), 20))
    });
    group.bench_function("squeezing_d20", |b| {
        b.iter(|| fock::squeezing(black_box(0.7), 0.0, 20))
    });
    group.bench_function("beamsplitter_d10", |b| {
        b.iter(|| fock::beamsplitter(black_box(0.6), 0.2, 10))
    });
    group.finish();
}

fn programs(c: &mut Criterion) {
    let mut group = c.benchmark_group("programs");
    group.sample_size(10);
    // 20 qubiteko qft-antzeko geruzak: ate-segida errealista bat vm osotik
    let mut src = String::from("QINIT 20\n");
    for layer in 0..5 {
        for q in 0..20 {
            src.push_str(&format!("H q{}\n", q));
        }
        for q in 0..19 {
            src.push_str(&format!(
                "CPHASE q{} q{} {}\n",
                q,
                q + 1,
                0.1 * (layer + 1) as f64
            ));
        }
    }
    let prog = qoa::asm::assemble(&src, "bench.qoa").unwrap();
    let cfg = RunConfig {
        shots: 1,
        seed: 1,
        ..RunConfig::default()
    };
    group.bench_function("layers_20q", |b| {
        b.iter(|| vm::run(&prog, &cfg, &mut std::io::sink()).unwrap())
    });
    // erdiko neurketa duten shot asko (shot bakoitza hasieratik)
    let retry = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/retry.qoa"))
        .unwrap();
    let prog = qoa::asm::assemble(&retry, "retry.qoa").unwrap();
    let cfg = RunConfig {
        shots: 2000,
        seed: 1,
        ..RunConfig::default()
    };
    group.bench_function("retry_2000_shots", |b| {
        b.iter(|| vm::run(&prog, &cfg, &mut std::io::sink()).unwrap())
    });
    group.finish();
}

criterion_group!(benches, qubit_gates, mode_ops, programs);
criterion_main!(benches);
