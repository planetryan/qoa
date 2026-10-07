// fitxategi honek qoa-ren ateak erreferentziazko simulagailu sinple baten aurka egiaztatzen ditu.
// erreferentzia nahita xumea da (matrize-biderketa zuzena, optimizaziorik gabe), beraz bi
// inplementazioek emaitza bera ematen badute, ateak zuzenak dira. 0.3.5-eko cz akatsa horrela aurkitu zen.
//
// bi maila egiaztatzen dira:
//   1. kernelak zuzenean (State), 1-16 qubit, haril-bide guztiak barne (egoera txikia, bloke asko, bloke handi gutxi)
//   2. agindu bakoitza testutik amaierara (mihiztatzailea -> vm -> kernela), mapaketa okerrak harrapatzeko

use num_complex::Complex64 as C;
use qoa::program::Gate;
use qoa::state::State;
use qoa::vm::{self, RunConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

// ---------------------------------------------------------------- erreferentzia

fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

// 2x2 matrizea qubit batean, kontrol guztiak 1 diren lekuetan bakarrik
fn ref_1q(s: &mut [C], controls: &[usize], t: usize, u: [C; 4]) {
    let b = 1 << t;
    let m: usize = controls.iter().map(|c| 1 << c).sum();
    for i in 0..s.len() {
        if i & b == 0 && i & m == m {
            let (a0, a1) = (s[i], s[i | b]);
            s[i] = u[0] * a0 + u[1] * a1;
            s[i | b] = u[2] * a0 + u[3] * a1;
        }
    }
}

fn ref_swap(s: &mut [C], controls: &[usize], a: usize, b: usize) {
    let m: usize = controls.iter().map(|c| 1 << c).sum();
    for i in 0..s.len() {
        if i & m == m && (i >> a) & 1 == 0 && (i >> b) & 1 == 1 {
            s.swap(i, i ^ (1 << a) ^ (1 << b));
        }
    }
}

fn ref_phase(s: &mut [C], qubits: &[usize], theta: f64) {
    let m: usize = qubits.iter().map(|c| 1 << c).sum();
    for (i, a) in s.iter_mut().enumerate() {
        if i & m == m {
            *a *= C::from_polar(1.0, theta);
        }
    }
}

// erreferentziaren ateak, testuliburuko matrizeekin (programaren matrizeak erabili gabe)
fn ref_gate(name: &str, p: &[f64]) -> [C; 4] {
    let z = c(0.0, 0.0);
    let one = c(1.0, 0.0);
    let r = FRAC_1_SQRT_2;
    match name {
        "H" => [c(r, 0.0), c(r, 0.0), c(r, 0.0), c(-r, 0.0)],
        "X" => [z, one, one, z],
        "Y" => [z, c(0.0, -1.0), c(0.0, 1.0), z],
        "Z" => [one, z, z, -one],
        "S" => [one, z, z, c(0.0, 1.0)],
        "SDG" => [one, z, z, c(0.0, -1.0)],
        "T" => [one, z, z, C::from_polar(1.0, PI / 4.0)],
        "TDG" => [one, z, z, C::from_polar(1.0, -PI / 4.0)],
        "SX" => [c(0.5, 0.5), c(0.5, -0.5), c(0.5, -0.5), c(0.5, 0.5)],
        "SXDG" => [c(0.5, -0.5), c(0.5, 0.5), c(0.5, 0.5), c(0.5, -0.5)],
        "PHASE" => [one, z, z, C::from_polar(1.0, p[0])],
        "RX" => {
            let (co, si) = ((p[0] / 2.0).cos(), (p[0] / 2.0).sin());
            [c(co, 0.0), c(0.0, -si), c(0.0, -si), c(co, 0.0)]
        }
        "RY" => {
            let (co, si) = ((p[0] / 2.0).cos(), (p[0] / 2.0).sin());
            [c(co, 0.0), c(-si, 0.0), c(si, 0.0), c(co, 0.0)]
        }
        "RZ" => [
            C::from_polar(1.0, -p[0] / 2.0),
            z,
            z,
            C::from_polar(1.0, p[0] / 2.0),
        ],
        "U" => {
            let (t, ph, l) = (p[0], p[1], p[2]);
            [
                c((t / 2.0).cos(), 0.0),
                -C::from_polar((t / 2.0).sin(), l),
                C::from_polar((t / 2.0).sin(), ph),
                C::from_polar((t / 2.0).cos(), ph + l),
            ]
        }
        _ => panic!("ez dago erreferentziarik {}-rentzat", name),
    }
}

fn rand_state(n: usize, rng: &mut StdRng) -> Vec<C> {
    let mut v: Vec<C> = (0..1usize << n)
        .map(|_| c(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
        .collect();
    let norm: f64 = v.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    v.iter_mut().for_each(|a| *a /= norm);
    v
}

fn max_diff(a: &[C], b: &[C]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

// n qubitetik k desberdin
fn distinct(rng: &mut StdRng, n: usize, k: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.random_range(i..n);
        v.swap(i, j);
    }
    v.truncate(k);
    v
}

const SINGLE: [&str; 15] = [
    "H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "SX", "SXDG", "PHASE", "RX", "RY", "RZ", "U",
];

fn gate_of(name: &str, p: &[f64]) -> Gate {
    match name {
        "H" => Gate::H,
        "X" => Gate::X,
        "Y" => Gate::Y,
        "Z" => Gate::Z,
        "S" => Gate::S,
        "SDG" => Gate::Sdg,
        "T" => Gate::T,
        "TDG" => Gate::Tdg,
        "SX" => Gate::SX,
        "SXDG" => Gate::SXdg,
        "PHASE" => Gate::P(p[0]),
        "RX" => Gate::RX(p[0]),
        "RY" => Gate::RY(p[0]),
        "RZ" => Gate::RZ(p[0]),
        "U" => Gate::U(p[0], p[1], p[2]),
        _ => unreachable!(),
    }
}

fn params(rng: &mut StdRng) -> Vec<f64> {
    (0..3).map(|_| rng.random_range(-PI..PI)).collect()
}

// ---------------------------------------------------------------- 1. kernelak

// qubit-kopuru hauek haril-bide guztiak estaltzen dituzte (PARALLEL_MIN = 2^14)
const SIZES: [usize; 11] = [1, 2, 3, 4, 5, 6, 8, 10, 14, 15, 16];

#[test]
fn single_qubit_kernels_match_reference() {
    let mut rng = StdRng::seed_from_u64(1);
    for &n in &SIZES {
        for name in SINGLE {
            // qubit baxuena, altuena eta ausazko bat: bloke txikiak eta handiak
            for t in [0, n - 1, rng.random_range(0..n)] {
                let p = params(&mut rng);
                let start = rand_state(n, &mut rng);
                let mut expected = start.clone();
                ref_1q(&mut expected, &[], t, ref_gate(name, &p));
                let mut s = State::from_amplitudes(n, 0, 2, start).unwrap();
                let g = gate_of(name, &p);
                if g == Gate::X {
                    s.apply_x(t, 0);
                } else {
                    s.apply_1q(t, g.matrix(), 0);
                }
                let d = max_diff(&expected, s.amplitudes());
                assert!(
                    d < 1e-12,
                    "{} on q{} of {} qubits: max difference {:e}",
                    name,
                    t,
                    n,
                    d
                );
            }
        }
    }
}

#[test]
fn controlled_kernels_match_reference() {
    let mut rng = StdRng::seed_from_u64(2);
    for &n in SIZES.iter().filter(|&&n| n >= 2) {
        for name in SINGLE {
            for n_controls in 1..=(n - 1).min(3) {
                let qs = distinct(&mut rng, n, n_controls + 1);
                let (t, controls) = (qs[0], &qs[1..]);
                let p = params(&mut rng);
                let start = rand_state(n, &mut rng);
                let mut expected = start.clone();
                ref_1q(&mut expected, controls, t, ref_gate(name, &p));
                let mut s = State::from_amplitudes(n, 0, 2, start).unwrap();
                let mask = controls.iter().fold(0, |m, c| m | (1 << c));
                let g = gate_of(name, &p);
                if g == Gate::X {
                    s.apply_x(t, mask);
                } else {
                    s.apply_1q(t, g.matrix(), mask);
                }
                let d = max_diff(&expected, s.amplitudes());
                assert!(
                    d < 1e-12,
                    "{}-controlled {} on {} qubits: max difference {:e}",
                    n_controls,
                    name,
                    n,
                    d
                );
            }
        }
    }
}

#[test]
fn phase_and_swap_kernels_match_reference() {
    let mut rng = StdRng::seed_from_u64(3);
    for &n in SIZES.iter().filter(|&&n| n >= 2) {
        for k in 1..=n.min(4) {
            let qs = distinct(&mut rng, n, k);
            let theta = rng.random_range(-PI..PI);
            let start = rand_state(n, &mut rng);
            let mut expected = start.clone();
            ref_phase(&mut expected, &qs, theta);
            let mut s = State::from_amplitudes(n, 0, 2, start).unwrap();
            s.apply_phase_mask(qs.iter().fold(0, |m, c| m | (1 << c)), theta);
            assert!(
                max_diff(&expected, s.amplitudes()) < 1e-12,
                "phase on {:?} of {} qubits",
                qs,
                n
            );
        }
        for n_controls in 0..=(n - 2).min(2) {
            let qs = distinct(&mut rng, n, n_controls + 2);
            let start = rand_state(n, &mut rng);
            let mut expected = start.clone();
            ref_swap(&mut expected, &qs[2..], qs[0], qs[1]);
            let mut s = State::from_amplitudes(n, 0, 2, start).unwrap();
            s.apply_swap(qs[0], qs[1], qs[2..].iter().fold(0, |m, c| m | (1 << c)));
            assert!(
                max_diff(&expected, s.amplitudes()) < 1e-12,
                "{}-controlled swap {:?} of {} qubits",
                n_controls,
                qs,
                n
            );
        }
    }
}

// ---------------------------------------------------------------- 2. aginduak amaieraraino

// programa exekutatu eta amaierako egoera itzuli (ezer neurtzen ez duenez, egoera zehatza da)
fn final_state(src: &str) -> Vec<C> {
    let prog = qoa::asm::assemble(src, "test.qoa")
        .unwrap_or_else(|d| panic!("{}", qoa::asm::render(&d, src, "test.qoa")));
    let cfg = RunConfig {
        shots: 1,
        seed: 1,
        ..RunConfig::default()
    };
    let out = vm::run(&prog, &cfg, &mut std::io::sink()).unwrap();
    out.final_state.amplitudes().to_vec()
}

// egoera generiko bat sortzeko: U ausazkoa qubit bakoitzean (testua eta erreferentzia)
fn scramble(n: usize, rng: &mut StdRng, src: &mut String, s: &mut [C]) {
    for q in 0..n {
        let p = params(rng);
        src.push_str(&format!("U q{} {} {} {}\n", q, p[0], p[1], p[2]));
        ref_1q(s, &[], q, ref_gate("U", &p));
    }
}

#[test]
fn every_gate_instruction_matches_reference_end_to_end() {
    let mut rng = StdRng::seed_from_u64(4);
    let n = 5;
    // (agindua, kontrol-kopurua, parametroak)
    let cases: &[(&str, usize, usize)] = &[
        ("H", 0, 0),
        ("X", 0, 0),
        ("Y", 0, 0),
        ("Z", 0, 0),
        ("S", 0, 0),
        ("SDG", 0, 0),
        ("T", 0, 0),
        ("TDG", 0, 0),
        ("SX", 0, 0),
        ("SXDG", 0, 0),
        ("PHASE", 0, 1),
        ("RX", 0, 1),
        ("RY", 0, 1),
        ("RZ", 0, 1),
        ("U", 0, 3),
        ("CNOT", 1, 0),
        ("CY", 1, 0),
        ("CZ", 1, 0),
        ("CH", 1, 0),
        ("CSX", 1, 0),
        ("CU", 1, 3),
        ("CPHASE", 1, 1),
        ("CRX", 1, 1),
        ("CRY", 1, 1),
        ("CRZ", 1, 1),
        ("CCX", 2, 0),
        ("MCX", 3, 0),
        ("MCZ", 3, 0),
        ("SWAP", 0, 0),
        ("CSWAP", 1, 0),
    ];
    for &(name, n_controls, n_params) in cases {
        for _ in 0..4 {
            let mut src = format!("QINIT {}\n", n);
            let mut expected = vec![c(0.0, 0.0); 1 << n];
            expected[0] = c(1.0, 0.0);
            scramble(n, &mut rng, &mut src, &mut expected);
            let p: Vec<f64> = params(&mut rng)[..n_params].to_vec();
            let pstr: Vec<String> = p.iter().map(|x| format!("{}", x)).collect();
            let swaps = matches!(name, "SWAP" | "CSWAP");
            let qs = distinct(&mut rng, n, n_controls + 1 + swaps as usize);
            // testuan: kontrolak lehenik, gero helburua(k)
            let operands: Vec<String> = match name {
                "SWAP" => vec![format!("q{}", qs[0]), format!("q{}", qs[1])],
                "CSWAP" => vec![
                    format!("q{}", qs[2]),
                    format!("q{}", qs[0]),
                    format!("q{}", qs[1]),
                ],
                "MCZ" => qs.iter().map(|q| format!("q{}", q)).collect(),
                _ => qs[1..]
                    .iter()
                    .chain(std::iter::once(&qs[0]))
                    .map(|q| format!("q{}", q))
                    .collect(),
            };
            src.push_str(&format!(
                "{} {} {}\n",
                name,
                operands.join(" "),
                pstr.join(" ")
            ));
            match name {
                "SWAP" => ref_swap(&mut expected, &[], qs[0], qs[1]),
                "CSWAP" => ref_swap(&mut expected, &qs[2..], qs[0], qs[1]),
                "CZ" | "MCZ" => ref_phase(&mut expected, &qs, PI),
                "CPHASE" => ref_phase(&mut expected, &qs, p[0]),
                _ => {
                    let base = match name {
                        "CNOT" | "CCX" | "MCX" => "X",
                        "CY" => "Y",
                        "CH" => "H",
                        "CSX" => "SX",
                        "CU" => "U",
                        "CRX" => "RX",
                        "CRY" => "RY",
                        "CRZ" => "RZ",
                        other => other,
                    };
                    ref_1q(&mut expected, &qs[1..], qs[0], ref_gate(base, &p));
                }
            }
            let got = final_state(&src);
            let d = max_diff(&expected, &got);
            assert!(
                d < 1e-12,
                "{} gave a different state (max difference {:e}) for:\n{}",
                name,
                d,
                src
            );
        }
    }
}

#[test]
fn entangling_shorthands_do_what_they_say() {
    let r = FRAC_1_SQRT_2;
    let bell = final_state("QINIT 2\nBELL q0 q1\n");
    assert!(max_diff(&bell, &[c(r, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(r, 0.0)]) < 1e-15);
    let ghz = final_state("QINIT 3\nGHZ q0 q1 q2\n");
    assert!((ghz[0].re - r).abs() < 1e-15 && (ghz[7].re - r).abs() < 1e-15);
    assert!(ghz[1..7].iter().all(|a| a.norm() < 1e-15));
    // lineako cluster-egoera: H guztietan, gero CZ auzokideen artean
    let cl = final_state("QINIT 3\nCLUSTER q0 q1 q2\n");
    let mut expected = vec![c(0.0, 0.0); 8];
    expected[0] = c(1.0, 0.0);
    for q in 0..3 {
        ref_1q(&mut expected, &[], q, ref_gate("H", &[]));
    }
    ref_phase(&mut expected, &[0, 1], PI);
    ref_phase(&mut expected, &[1, 2], PI);
    assert!(max_diff(&cl, &expected) < 1e-15);
}

// ---------------------------------------------------------------- neurketa

fn counts(src: &str, shots: u64, seed: u64) -> Vec<(String, u64)> {
    let prog = qoa::asm::assemble(src, "test.qoa")
        .unwrap_or_else(|d| panic!("{}", qoa::asm::render(&d, src, "test.qoa")));
    let cfg = RunConfig {
        shots,
        seed,
        ..RunConfig::default()
    };
    let out = vm::run(&prog, &cfg, &mut std::io::sink()).unwrap();
    out.counts
        .iter()
        .map(|(rec, n)| (qoa::report::bits_string(rec), *n))
        .collect()
}

#[test]
fn bell_measurements_are_perfectly_correlated() {
    let c = counts("QINIT 2\nBELL q0 q1\nMEASURE q0..q1\n", 4000, 11);
    let total: u64 = c.iter().map(|x| x.1).sum();
    assert_eq!(total, 4000);
    for (k, n) in &c {
        assert!(
            k == "00" || k == "11",
            "impossible outcome {} ({} times)",
            k,
            n
        );
        assert!(
            (*n as f64 / 4000.0 - 0.5).abs() < 0.04,
            "{} came up {} times",
            k,
            n
        );
    }
}

#[test]
fn measurement_probability_follows_born_rule() {
    // ry(θ)|0> neurtzean P(1) = sin²(θ/2)
    let theta: f64 = 1.1;
    let c = counts(&format!("QINIT 1\nRY q0 {}\nMEASURE q0\n", theta), 20000, 5);
    let ones = c.iter().find(|x| x.0 == "1").map_or(0, |x| x.1) as f64 / 20000.0;
    let expected = (theta / 2.0).sin().powi(2);
    assert!(
        (ones - expected).abs() < 0.015,
        "P(1) = {} but expected {}",
        ones,
        expected
    );
}

#[test]
fn sampled_and_simulated_shots_agree() {
    // programa bera bi eratara: amaierako neurketa (lagindua) eta erdiko neurketa (shot bakoitza simulatua)
    let sampled = counts(
        "QINIT 3\nH q0\nCNOT q0 q1\nRY q2 0.7\nMEASURE q0..q2\n",
        20000,
        9,
    );
    let simulated = counts(
        "QINIT 3\nH q0\nCNOT q0 q1\nRY q2 0.7\nMEASURE q0..q2\nBARRIER\nX q0\nX q0\n",
        20000,
        9,
    );
    let get = |v: &Vec<(String, u64)>, k: &str| {
        v.iter().find(|x| x.0 == k).map_or(0, |x| x.1) as f64 / 20000.0
    };
    for k in ["000", "011", "100", "111"] {
        assert!(
            (get(&sampled, k) - get(&simulated, k)).abs() < 0.02,
            "outcome {} differs",
            k
        );
    }
}

#[test]
fn teleportation_with_feed_forward_delivers_the_state() {
    // q0-ren egoera q2-ra teleportatu, neurketak eta zuzenketa klasikoak erabiliz; gero q2-ren
    // egoera desegin eta neurtu: beti 0 eman behar du
    let src = "QINIT 3
U q0 1.0 0.5 -0.3
BELL q1 q2
CNOT q0 q1
H q0
MEASURE q0 -> c0
MEASURE q1 -> c1
JUMPIFZERO c1 skip_x
X q2
skip_x:
JUMPIFZERO c0 skip_z
Z q2
skip_z:
U q2 -1.0 0.3 -0.5
MEASURE q2 -> c2
";
    let c = counts(src, 2000, 3);
    for (k, n) in &c {
        assert_eq!(
            &k[..1],
            "0",
            "teleported qubit read 1 ({} times in outcome {})",
            n,
            k
        );
    }
}
