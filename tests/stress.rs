// fitxategi honek proba handiak sortzen ditu kodez: ausazko zirkuitu sakonak aurrera eta atzera,
// boson-laginketa permanenteen aurka, eta zarata-ereduen estatistikak.
//
// `cargo test --release --test stress` proba arruntak exekutatzen ditu (segundo batzuk).
// `cargo test --release --test stress -- --ignored` mugetako probak ere bai: 28 qubit (4 GiB),
// hamarnaka milioi anplitudeko boson-laginketa eta sistema hibrido erraldoi bat. minutu bat inguru behar du
// eta 15 GiB inguruko ram-a duen makina batean probatu da.

use num_complex::Complex64 as C;
use qoa::lang::{self, Lang};
use qoa::vm::{self, RunConfig, RunOutput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt::Write;

fn run(src: &str, shots: u64, seed: u64) -> RunOutput {
    lang::set(Lang::En);
    let prog = qoa::asm::assemble(src, "stress.qoa")
        .unwrap_or_else(|d| panic!("{}", qoa::asm::render(&d, src, "stress.qoa")));
    let cfg = RunConfig {
        shots,
        seed,
        ..RunConfig::default()
    };
    vm::run(&prog, &cfg, &mut std::io::sink()).unwrap_or_else(|e| panic!("{}", e))
}

fn run_noisy(src: &str, shots: u64, noise: f64, loss: f64) -> RunOutput {
    lang::set(Lang::En);
    let prog = qoa::asm::assemble(src, "stress.qoa").unwrap();
    let cfg = RunConfig {
        shots,
        seed: 3,
        noise,
        loss,
        ..RunConfig::default()
    };
    vm::run(&prog, &cfg, &mut std::io::sink()).unwrap()
}

// ---------------------------------------------------------------- ausazko zirkuituak aurrera eta atzera

// ausazko ate bat eta haren alderantzizkoa, qoa testu gisa
fn random_gate(rng: &mut StdRng, n: usize) -> (String, String) {
    let q = |rng: &mut StdRng| rng.random_range(0..n);
    let a: f64 = rng.random_range(-3.0..3.0);
    let b: f64 = rng.random_range(-3.0..3.0);
    let c: f64 = rng.random_range(-3.0..3.0);
    let distinct = |rng: &mut StdRng, k: usize| -> Vec<usize> {
        let mut v: Vec<usize> = Vec::new();
        while v.len() < k {
            let x = rng.random_range(0..n);
            if !v.contains(&x) {
                v.push(x);
            }
        }
        v
    };
    let kind = rng.random_range(0..16);
    match kind {
        0 => {
            let t = q(rng);
            (format!("H q{}", t), format!("H q{}", t))
        }
        1 => {
            let t = q(rng);
            (format!("X q{}", t), format!("X q{}", t))
        }
        2 => {
            let t = q(rng);
            (format!("Y q{}", t), format!("Y q{}", t))
        }
        3 => {
            let t = q(rng);
            (format!("S q{}", t), format!("SDG q{}", t))
        }
        4 => {
            let t = q(rng);
            (format!("T q{}", t), format!("TDG q{}", t))
        }
        5 => {
            let t = q(rng);
            (format!("SX q{}", t), format!("SXDG q{}", t))
        }
        6 => {
            let t = q(rng);
            (format!("RX q{} {}", t, a), format!("RX q{} {}", t, -a))
        }
        7 => {
            let t = q(rng);
            (format!("RY q{} {}", t, a), format!("RY q{} {}", t, -a))
        }
        8 => {
            let t = q(rng);
            (format!("RZ q{} {}", t, a), format!("RZ q{} {}", t, -a))
        }
        9 => {
            let t = q(rng);
            (
                format!("U q{} {} {} {}", t, a, b, c),
                format!("U q{} {} {} {}", t, -a, -c, -b),
            )
        }
        10 => {
            let v = distinct(rng, 2);
            let s = format!("CNOT q{} q{}", v[0], v[1]);
            (s.clone(), s)
        }
        11 => {
            let v = distinct(rng, 2);
            (
                format!("CPHASE q{} q{} {}", v[0], v[1], a),
                format!("CPHASE q{} q{} {}", v[0], v[1], -a),
            )
        }
        12 => {
            let v = distinct(rng, 2);
            let s = format!("SWAP q{} q{}", v[0], v[1]);
            (s.clone(), s)
        }
        13 => {
            let v = distinct(rng, 3);
            let s = format!("CCX q{} q{} q{}", v[0], v[1], v[2]);
            (s.clone(), s)
        }
        14 => {
            let v = distinct(rng, 3);
            let s = format!("CSWAP q{} q{} q{}", v[0], v[1], v[2]);
            (s.clone(), s)
        }
        _ => {
            let v = distinct(rng, 2);
            (
                format!("CRY q{} q{} {}", v[0], v[1], a),
                format!("CRY q{} q{} {}", v[0], v[1], -a),
            )
        }
    }
}

// zirkuitua eta gero bere alderantzizkoa: |0...0> berreskuratu behar da
fn there_and_back(n: usize, gates: usize, seed: u64) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut fwd = Vec::new();
    let mut back = Vec::new();
    for _ in 0..gates {
        let (g, inv) = random_gate(&mut rng, n);
        fwd.push(g);
        back.push(inv);
    }
    back.reverse();
    let mut src = format!("QINIT {}\n", n);
    for l in fwd.iter().chain(back.iter()) {
        src.push_str(l);
        src.push('\n');
    }
    let out = run(&src, 1, 1);
    out.final_state.amplitudes()[0].norm_sqr()
}

#[test]
fn deep_random_circuit_runs_backwards_to_the_start() {
    // 20 qubit (16 MiB), 3000 ate aurrera eta 3000 atzera
    let p0 = there_and_back(20, 3000, 42);
    assert!(
        (p0 - 1.0).abs() < 1e-9,
        "P(|0..0>) = {} after the circuit and its inverse",
        p0
    );
}

#[test]
fn many_small_random_circuits_run_backwards_to_the_start() {
    for seed in 0..40 {
        // gutxienez 3 qubit: CCX eta CSWAP-ek hiru qubit desberdin behar dituzte
        let n = 3 + (seed as usize % 8);
        let p0 = there_and_back(n, 200, seed);
        assert!(
            (p0 - 1.0).abs() < 1e-12,
            "{} qubits, seed {}: P(|0..0>) = {}",
            n,
            seed,
            p0
        );
    }
}

// ---------------------------------------------------------------- boson-laginketa permanenteen aurka

// matrize baten permanentea, ryser-en formularekin
fn permanent(m: &[Vec<C>]) -> C {
    let n = m.len();
    if n == 0 {
        return C::new(1.0, 0.0);
    }
    let mut total = C::new(0.0, 0.0);
    for subset in 1u32..(1 << n) {
        let mut prod = C::new(1.0, 0.0);
        for row in m {
            let mut s = C::new(0.0, 0.0);
            for (j, x) in row.iter().enumerate() {
                if subset & (1 << j) != 0 {
                    s += x;
                }
            }
            prod *= s;
        }
        let sign = if (n - subset.count_ones() as usize).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        total += prod * sign;
    }
    total
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

// ausazko sare optiko bat (BS eta MODEPHASE)
fn random_network(rng: &mut StdRng, modes: usize, elements: usize) -> Vec<String> {
    let mut v = Vec::new();
    for _ in 0..elements {
        if rng.random_bool(0.75) {
            let a = rng.random_range(0..modes);
            let mut b = rng.random_range(0..modes);
            while b == a {
                b = rng.random_range(0..modes);
            }
            v.push(format!(
                "BS o{} o{} {} {}",
                a,
                b,
                rng.random_range(0.0..1.6),
                rng.random_range(-3.0..3.0)
            ));
        } else {
            v.push(format!(
                "MODEPHASE o{} {}",
                rng.random_range(0..modes),
                rng.random_range(-3.0..3.0)
            ));
        }
    }
    v
}

// fotoi bakarreko transferentzia-matrizea: fotoi bat i moduan sartu eta irteeren anplitudeak irakurri
fn transfer_matrix(modes: usize, network: &[String]) -> Vec<Vec<C>> {
    let mut u = vec![vec![C::new(0.0, 0.0); modes]; modes];
    for i in 0..modes {
        let mut src = format!("QMODE {} 2\nSINGLEPHOTON o{}\n", modes, i);
        for l in network {
            src.push_str(l);
            src.push('\n');
        }
        let st = run(&src, 1, 1).final_state;
        for (j, row) in u.iter_mut().enumerate() {
            // fotoi bakarra j moduan: indizea 2^j
            row[i] = st.amplitudes()[1 << j];
        }
    }
    u
}

// fotoi-banaketa guztiak: `photons` fotoi `modes` modutan
fn patterns(modes: usize, photons: usize) -> Vec<Vec<usize>> {
    if modes == 1 {
        return vec![vec![photons]];
    }
    let mut out = Vec::new();
    for k in 0..=photons {
        for mut rest in patterns(modes - 1, photons - k) {
            let mut p = vec![k];
            p.append(&mut rest);
            out.push(p);
        }
    }
    out
}

// sare bat n fotoirekin simulatu eta irteera-probabilitate bakoitza permanentearekin konparatu
fn boson_sampling_check(modes: usize, input: &[usize], elements: usize, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let network = random_network(&mut rng, modes, elements);
    let u = transfer_matrix(modes, &network);
    let photons: usize = input.iter().sum();
    let cutoff = photons + 1;
    let mut src = format!("QMODE {} {}\n", modes, cutoff);
    for (m, &k) in input.iter().enumerate() {
        if k > 0 {
            src.push_str(&format!("FOCK o{} {}\n", m, k));
        }
    }
    for l in &network {
        src.push_str(l);
        src.push('\n');
    }
    let out = run(&src, 1, 1);
    assert!(
        out.leakage < 1e-12,
        "cutoff {} should hold every pattern, leakage {}",
        cutoff,
        out.leakage
    );
    let st = out.final_state;
    // sarrerako moduak, fotoi bakoitzeko behin
    let in_modes: Vec<usize> = input
        .iter()
        .enumerate()
        .flat_map(|(m, &k)| std::iter::repeat_n(m, k))
        .collect();
    let in_norm: f64 = input.iter().map(|&k| factorial(k)).product();
    let mut total = 0.0;
    let mut worst: f64 = 0.0;
    for pat in patterns(modes, photons) {
        let out_modes: Vec<usize> = pat
            .iter()
            .enumerate()
            .flat_map(|(m, &k)| std::iter::repeat_n(m, k))
            .collect();
        let sub: Vec<Vec<C>> = out_modes
            .iter()
            .map(|&r| in_modes.iter().map(|&c| u[r][c]).collect())
            .collect();
        let out_norm: f64 = pat.iter().map(|&k| factorial(k)).product();
        let p_perm = permanent(&sub).norm_sqr() / (in_norm * out_norm);
        let idx: usize = pat
            .iter()
            .enumerate()
            .map(|(m, &k)| k * cutoff.pow(m as u32))
            .sum();
        let p_sim = st.amplitudes()[idx].norm_sqr();
        worst = worst.max((p_perm - p_sim).abs());
        total += p_sim;
    }
    assert!(
        (total - 1.0).abs() < 1e-10,
        "probabilities sum to {}",
        total
    );
    assert!(
        worst < 1e-10,
        "{} modes, input {:?}: simulator and permanents differ by {:e}",
        modes,
        input,
        worst
    );
}

#[test]
fn boson_sampling_matches_permanents() {
    // hong-ou-mandel-en orokortzea: 2, 3 eta 4 fotoi sare ausazkoetan
    boson_sampling_check(3, &[1, 1, 0], 10, 1);
    boson_sampling_check(5, &[1, 1, 1, 0, 0], 25, 2);
    boson_sampling_check(6, &[1, 0, 1, 0, 1, 1], 40, 3);
    // fotoi anitz modu berean sartuta
    boson_sampling_check(4, &[2, 0, 1, 0], 20, 4);
}

// ---------------------------------------------------------------- zarata-ereduak

#[test]
fn depolarizing_noise_flips_at_two_thirds_of_p() {
    // X eta Y-k |1> iraultzen dute, Z-k ez: P(irauli) = 2p/3
    let p = 0.3;
    let out = run_noisy("QINIT 1\nX q0\nMEASURE q0\n", 40000, p, 0.0);
    let zeros = out
        .counts
        .iter()
        .find(|(r, _)| r.bits[0] == Some(0))
        .map_or(0, |c| c.1) as f64
        / 40000.0;
    assert!(
        (zeros - 2.0 * p / 3.0).abs() < 0.01,
        "P(flip) = {} but expected {}",
        zeros,
        2.0 * p / 3.0
    );
}

#[test]
fn photon_loss_noise_matches_the_loss_channel() {
    // fotoi bakarra eta eragiketa optiko bat: P(galdua) = loss
    let out = run_noisy(
        "QMODE 1 3\nSINGLEPHOTON o0\nMODEPHASE o0 0.1\nCOUNT o0\n",
        40000,
        0.0,
        0.25,
    );
    let lost = out
        .counts
        .iter()
        .find(|(r, _)| r.readouts[0] == Some(0))
        .map_or(0, |c| c.1) as f64
        / 40000.0;
    assert!(
        (lost - 0.25).abs() < 0.01,
        "P(lost) = {} but expected 0.25",
        lost
    );
    // LOSS aginduak: bi fotoi, galera 0.5 bakoitzeko -> binomiala (1/4, 1/2, 1/4)
    let out = run("QMODE 1 3\nFOCK o0 2\nLOSS o0 0.5\nCOUNT o0\n", 40000, 5);
    for (n, want) in [(0u32, 0.25), (1, 0.5), (2, 0.25)] {
        let got = out
            .counts
            .iter()
            .find(|(r, _)| r.readouts[0] == Some(n))
            .map_or(0, |c| c.1) as f64
            / 40000.0;
        assert!(
            (got - want).abs() < 0.01,
            "P({} left) = {} but expected {}",
            n,
            got,
            want
        );
    }
}

// ---------------------------------------------------------------- mugak (--ignored)

#[test]
#[ignore]
fn limit_ghz_28_qubits() {
    // 2^28 anplitude = 4 GiB
    let mut src = String::from("QINIT 28\nGHZ q0..q27\nMEASUREALL\n");
    let _ = writeln!(src);
    let out = run(&src, 10000, 7);
    for (rec, n) in &out.counts {
        let bits = qoa::report::bits_string(rec);
        assert!(
            bits.chars().all(|c| c == '0') || bits.chars().all(|c| c == '1'),
            "impossible outcome {} ({} times)",
            bits,
            n
        );
    }
}

#[test]
#[ignore]
fn limit_random_circuit_26_qubits() {
    // 2^26 anplitude = 1 GiB, 400 ate aurrera eta 400 atzera
    let p0 = there_and_back(26, 400, 26);
    assert!((p0 - 1.0).abs() < 1e-9, "P(|0..0>) = {}", p0);
}

#[test]
#[ignore]
fn limit_boson_sampling_5_photons_in_9_modes() {
    // muga 6, 9 modu: 6^9 = 10 milioi anplitude; 2002 irteera-patroi, bakoitza permanente batekin
    boson_sampling_check(9, &[1, 1, 1, 1, 1, 0, 0, 0, 0], 60, 9);
}
