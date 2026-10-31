// fitxategi honek src/quad.rs egiaztatzen du: double-double aritmetikaren sin, cos eta erro karratuak
// 50 digituko erreferentzien aurka (python-en decimal-ekin kalkulatuak), 128 biteko motorrak araua 1e-27-raino
// gordetzen duela 20000 ateko sortetan (f64-k askoz gehiago galtzen du), f64 motorrarekin bat datorrela,
// identitateak (H·H, T⁸) zehazki ematen dituela, emaitza hari-kopuruaren menpe ez dagoela, eta modu optikoak
// edo GPUa quad moduan erroreak direla.

use qoa::lang::{self, Lang};
use qoa::program::Gate;
use qoa::quad::{self, Q, Qc, QuadGate, QuadState};
use qoa::vm::{self, Precision, RunConfig};

// double-double bat (hi, lo)
type Pair = (f64, f64);

// (x, (sin hi, sin lo), (cos hi, cos lo)), x-ren f64 balio zehatzerako
const TRIG: [(f64, Pair, Pair); 6] = [
    (
        1.0,
        (0.8414709848078965, 1.776845092935536e-18),
        (0.5403023058681398, -4.760954612604417e-17),
    ),
    (
        0.5,
        (0.479425538604203, -5.103969860556013e-18),
        (0.8775825618903728, -4.2623149864279997e-17),
    ),
    (
        -2.5,
        (-0.5984721441039565, 5.521403334082375e-17),
        (-0.8011436155469337, -1.8674742705085553e-17),
    ),
    (
        100.0,
        (-0.5063656411097588, -3.050947053792115e-18),
        (0.8623188722876839, 4.334809858136501e-17),
    ),
    (
        12345.678,
        (-0.7040813137533816, -1.9646969196301474e-17),
        (0.7101193587160628, -2.4815747415365305e-17),
    ),
    (1e-8, (1e-08, -1.6666666666666668e-25), (1.0, -5e-17)),
];

fn close(a: Q, hi: f64, lo: f64, tol: f64) -> bool {
    let d = a.minus(Q { hi, lo });
    d.value().abs() <= tol
}

#[test]
fn trigonometry_and_roots_have_32_digits() {
    for (x, (sh, sl), (ch, cl)) in TRIG {
        let (s, c) = Q::new(x).sin_cos();
        assert!(close(s, sh, sl, 1e-31), "sin {}: {:?}", x, s);
        assert!(close(c, ch, cl, 1e-31), "cos {}: {:?}", x, c);
    }
    assert!(close(
        Q::new(2.0).sqrt(),
        std::f64::consts::SQRT_2,
        -9.667293313452913e-17,
        1e-31
    ));
    assert!(close(
        quad::frac_1_sqrt_2(),
        std::f64::consts::FRAC_1_SQRT_2,
        -4.833646656726457e-17,
        1e-32
    ));
    let third = Q::ONE.over(Q::new(3.0));
    assert!(third.times_f64(3.0).minus(Q::ONE).value().abs() < 1e-32);
}

// ausazko zirkuitu bat (zehaztua), ate guztiekin
fn circuit(n: u32, depth: usize) -> String {
    let mut s = format!("QINIT {}\n", n);
    let mut x: u64 = 12345;
    let mut next = |m: u64| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 33) % m
    };
    for _ in 0..depth {
        let q = next(n as u64);
        let r = (next(n as u64 - 1) + q + 1) % n as u64;
        let a = next(1000) as f64 / 137.0;
        s.push_str(&match next(9) {
            0 => format!("H q{}\n", q),
            1 => format!("RX q{} {}\n", q, a),
            2 => format!("RY q{} {}\n", q, a),
            3 => format!("RZ q{} {}\n", q, a),
            4 => format!("CNOT q{} q{}\n", q, r),
            5 => format!("T q{}\n", q),
            6 => format!("SX q{}\n", q),
            7 => format!("SWAP q{} q{}\n", q, r),
            _ => format!("CZ q{} q{}\n", q, r),
        });
    }
    s
}

fn run(src: &str, precision: Precision) -> vm::RunOutput {
    lang::set(Lang::En);
    let prog = qoa::asm::assemble(src, "quad.qoa").unwrap();
    let cfg = RunConfig {
        shots: 1,
        seed: 1,
        precision,
        ..RunConfig::default()
    };
    vm::run(&prog, &cfg, &mut std::io::sink()).unwrap_or_else(|e| panic!("{}", e))
}

#[test]
fn quad_keeps_the_norm_where_double_drifts() {
    let src = circuit(10, 20_000);
    let d = run(&src, Precision::Double);
    let q = run(&src, Precision::Quad);
    // 20000 ate: f64-k 1e-14 inguru galtzen du, 128 bitek 1e-29 inguru
    println!(
        "norm error: double {:e}, quad {:e}",
        d.norm_error, q.norm_error
    );
    assert!(q.norm_error < 1e-27, "quad {}", q.norm_error);
    assert!(
        d.norm_error > 1e10 * q.norm_error.max(1e-31),
        "double {}",
        d.norm_error
    );
    // bi motorrak bat datoz f64-ren zehaztasunean
    let worst = d
        .final_state
        .amplitudes()
        .iter()
        .zip(q.final_state.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(worst < 1e-11, "{}", worst);
}

#[test]
fn identities_come_out_exact() {
    // H 1000 aldiz eta T 8 aldiz: |0> eta |1> berriro, 1e-30-raino
    let mut src = String::from("QINIT 2\nX q1\n");
    for _ in 0..1000 {
        src.push_str("H q0\n");
    }
    for _ in 0..8 {
        src.push_str("H q1\nT q1\nH q1\n");
    }
    let mut s = QuadState::new(2).unwrap();
    let h = quad::matrix(Gate::H);
    let t = quad::matrix(Gate::T);
    let x = quad::matrix(Gate::X);
    s.apply(&[QuadGate::Mat {
        q: 1,
        m: x,
        controls: 0,
    }]);
    for _ in 0..1000 {
        s.apply(&[QuadGate::Mat {
            q: 0,
            m: h,
            controls: 0,
        }]);
    }
    for _ in 0..8 {
        s.apply(&[QuadGate::Mat {
            q: 1,
            m: t,
            controls: 0,
        }]);
    }
    let a = s.amplitudes();
    let one = a[2].sub_one();
    assert!(one < 1e-30, "{}", one);
    for (i, v) in a.iter().enumerate() {
        if i != 2 {
            assert!(v.norm_sqr().value() < 1e-60, "{} {:?}", i, v);
        }
    }
}

trait Off {
    fn sub_one(&self) -> f64;
}

impl Off for Qc {
    fn sub_one(&self) -> f64 {
        let d = Qc::new(self.re.minus(Q::ONE), self.im);
        d.norm_sqr().value().sqrt()
    }
}

#[test]
fn quad_results_do_not_depend_on_threads() {
    let src = circuit(15, 400);
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| run(&src, Precision::Quad));
    let many = run(&src, Precision::Quad);
    assert_eq!(one.final_state.amplitudes(), many.final_state.amplitudes());
    assert_eq!(one.norm_error, many.norm_error);
}

#[test]
fn quad_measures_and_samples() {
    let out = run("QINIT 3\nGHZ q0..q2\nMEASUREALL\n", Precision::Quad);
    assert_eq!(out.counts.iter().map(|c| c.1).sum::<u64>(), 1);
    lang::set(Lang::En);
    let prog = qoa::asm::assemble("QINIT 2\nH q0\nCNOT q0 q1\nMEASUREALL\n", "bell.qoa").unwrap();
    let cfg = RunConfig {
        shots: 4000,
        seed: 9,
        precision: Precision::Quad,
        ..RunConfig::default()
    };
    let out = vm::run(&prog, &cfg, &mut std::io::sink()).unwrap();
    let total: u64 = out.counts.iter().map(|c| c.1).sum();
    assert_eq!(total, 4000);
    for (rec, n) in &out.counts {
        let bits: Vec<_> = rec.bits.iter().map(|b| b.unwrap()).collect();
        assert_eq!(bits[0], bits[1], "bell pairs agree");
        assert!((*n as f64 - 2000.0).abs() < 200.0, "{}", n);
    }
}

#[test]
fn quad_refuses_what_it_cannot_do_in_128_bits() {
    lang::set(Lang::En);
    let prog = qoa::asm::assemble("QMODE 1 4\nDISPLACE o0 0.5\n", "optics.qoa").unwrap();
    let cfg = RunConfig {
        precision: Precision::Quad,
        ..RunConfig::default()
    };
    let e = vm::run(&prog, &cfg, &mut std::io::sink()).unwrap_err();
    assert!(
        e.message.contains("qubit programs only") && e.message.contains("optical modes"),
        "{}",
        e.message
    );
}

#[test]
fn cache_blocked_gates_match_double() {
    // 16 qubit: 2^16 anplitude, cache-blokeak (2^13) eta sorta luzeak erabiltzen dira
    let src = circuit(16, 3000);
    let d = run(&src, Precision::Double);
    let q = run(&src, Precision::Quad);
    let worst = d
        .final_state
        .amplitudes()
        .iter()
        .zip(q.final_state.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(worst < 1e-12, "{}", worst);
    assert!(q.norm_error < 1e-28, "{}", q.norm_error);
}

// egoeraren bit guztien hash bat (fnv-1a, hi eta lo), arkitektura guztietan berdina izan behar duena
fn fingerprint(s: &QuadState) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for a in s.amplitudes() {
        for v in [a.re.hi, a.re.lo, a.im.hi, a.im.lo] {
            for byte in v.to_bits().to_le_bytes() {
                h = (h ^ byte as u64).wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    h
}

#[test]
fn quad_gives_the_same_bits_on_every_system() {
    // ate guztiak, angelu askorekin (sin/cos), kontrolekin eta trukaketekin, eta neurketa bat: CI-k
    // x86-64, ARM64, RISC-V, POWER10 eta IBM Z-n exekutatzen du, eta hatz-marka bera eman behar dute
    let mut s = QuadState::new(13).unwrap();
    let mut gates = Vec::new();
    for k in 0..600u32 {
        let q = (k * 7 + 3) as usize % 13;
        let c = (k * 5 + 1) as usize % 13;
        let a = (k as f64 + 0.5) * 0.731;
        let g = match k % 8 {
            0 => Gate::H,
            1 => Gate::RX(a),
            2 => Gate::RY(-a),
            3 => Gate::RZ(a * 3.0),
            4 => Gate::U(a, a * 0.5, -a * 0.25),
            5 => Gate::P(a),
            6 => Gate::T,
            _ => Gate::SX,
        };
        let controls = if k % 3 == 0 && c != q { 1 << c } else { 0 };
        gates.push(QuadGate::Mat { q, m: quad::matrix(g), controls });
        if k % 50 == 49 {
            gates.push(QuadGate::Swap { a: q, b: (q + 4) % 13, controls: 0 });
        }
    }
    s.apply(&gates);
    let bit = s.measure_qubit(5, 0.3);
    let fp = fingerprint(&s);
    println!("measured {} fingerprint {:#018x}", bit, fp);
    assert_eq!((bit, fp), (EXPECT_BIT, EXPECT_FP));
}

// x86-64-n kalkulatuak (AVX2 + FMA bidea); beste arkitektura guztiek berdin eman behar dute
const EXPECT_BIT: u8 = 1;
const EXPECT_FP: u64 = 0x420f_9e13_75ea_46bc;
