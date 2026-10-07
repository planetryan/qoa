// fitxategi honek src/kernel.rs egiaztatzen du: ate-sortak (fusioa eta cache-blokeak) eta egoera osoko
// kernelak, ausazko ateekin (kontrolak edozein qubitetan, trukaketak, diagonalak, x), erreferentzia
// inozo baten aurka. bloke-tamaina txikiak behartzen dira egoera txikietan ere etapa mota guztiak
// probatzeko, eta qubit + modu egoerak (luzera ez da 2ren berretura).

use num_complex::Complex64 as C;
use qoa::kernel::{self, QGate};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_state(rng: &mut StdRng, n: usize) -> Vec<C> {
    (0..n)
        .map(|_| C::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
        .collect()
}

// erreferentzia: bikote bakoitza bere aldetik, optimizaziorik gabe
fn reference(amps: &mut [C], g: &QGate) {
    match *g {
        QGate::Mat { q, m, controls } => {
            for i in 0..amps.len() {
                if i >> q & 1 == 0 && i & controls == controls && (i | 1 << q) < amps.len() {
                    let j = i | 1 << q;
                    let (x, y) = (amps[i], amps[j]);
                    amps[i] = m[0] * x + m[1] * y;
                    amps[j] = m[2] * x + m[3] * y;
                }
            }
        }
        QGate::Swap { a, b, controls } => {
            for i in 0..amps.len() {
                if i >> a & 1 == 1 && i >> b & 1 == 0 && i & controls == controls {
                    amps.swap(i, i ^ (1 << a) ^ (1 << b));
                }
            }
        }
    }
}

fn random_gate(rng: &mut StdRng, nq: usize) -> QGate {
    let q = rng.random_range(0..nq);
    // kontrolak: helburuak ez diren qubitak, % 30 probabilitatez bakoitza
    let controls_without = |rng: &mut StdRng, skip: usize| {
        (0..nq)
            .filter(|&c| skip >> c & 1 == 0 && rng.random_bool(0.3))
            .fold(0usize, |m, c| m | 1 << c)
    };
    let z = C::new(0.0, 0.0);
    let o = C::new(1.0, 0.0);
    match rng.random_range(0..6) {
        0 if nq >= 2 => {
            let mut b = rng.random_range(0..nq);
            while b == q {
                b = rng.random_range(0..nq);
            }
            let controls = if rng.random_bool(0.5) {
                controls_without(rng, 1 << q | 1 << b)
            } else {
                0
            };
            QGate::Swap { a: q, b, controls }
        }
        1 => {
            let controls = controls_without(rng, 1 << q);
            QGate::Mat {
                q,
                m: [z, o, o, z],
                controls,
            }
        }
        2 => {
            let d = [
                C::from_polar(1.0, rng.random_range(0.0..6.3)),
                C::from_polar(1.0, rng.random_range(0.0..6.3)),
            ];
            let controls = controls_without(rng, 1 << q);
            QGate::Mat {
                q,
                m: [d[0], z, z, d[1]],
                controls,
            }
        }
        3 => QGate::phase(controls_without(rng, 0).max(1), rng.random_range(0.0..6.3)),
        _ => {
            let m = random_state(rng, 4);
            let controls = if rng.random_bool(0.5) {
                controls_without(rng, 1 << q)
            } else {
                0
            };
            QGate::Mat {
                q,
                m: [m[0], m[1], m[2], m[3]],
                controls,
            }
        }
    }
}

fn max_diff(a: &[C], b: &[C]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

#[test]
fn batches_match_the_reference_for_every_block_size() {
    let mut rng = StdRng::seed_from_u64(11);
    for case in 0..400 {
        let nq = 1 + case % 12;
        // modu bat (3 maila) batzuetan: luzera 3 * 2^nq
        let len = if case % 3 == 0 { 3 << nq } else { 1 << nq };
        let gates: Vec<QGate> = (0..rng.random_range(1..40))
            .map(|_| random_gate(&mut rng, nq))
            .collect();
        let start = random_state(&mut rng, len);
        let mut want = start.clone();
        for g in &gates {
            reference(&mut want, g);
        }
        // bloke-tamaina eta qubit altuen kopuru guztiak: tokian bertako blokeak eta bilketa-blokeak
        for b in 1..=nq {
            for high in 0..=4 {
                let mut got = start.clone();
                kernel::apply_gates_blocked(&mut got, nq, &gates, b, high);
                assert!(
                    max_diff(&got, &want) < 1e-10,
                    "kasua {} b={} high={} {:?}",
                    case,
                    b,
                    high,
                    gates
                );
            }
        }
        let mut got = start.clone();
        kernel::apply_gates(&mut got, nq, &gates);
        assert!(max_diff(&got, &want) < 1e-10, "kasua {} fusioarekin", case);
    }
}

#[test]
fn whole_state_sweeps_match_the_reference_on_large_states() {
    // 15-17 qubit: paraleloko bideak (bloke txiki asko eta bloke handi gutxi)
    let mut rng = StdRng::seed_from_u64(12);
    for nq in [15usize, 16, 17] {
        let start = random_state(&mut rng, 1 << nq);
        for _ in 0..40 {
            let g = random_gate(&mut rng, nq);
            let mut want = start.clone();
            reference(&mut want, &g);
            let mut got = start.clone();
            match g {
                QGate::Mat { q, m, controls } => kernel::sweep_mat(&mut got, q, &m, controls),
                QGate::Swap { a, b, controls } => kernel::sweep_swap(&mut got, a, b, controls),
            }
            assert!(max_diff(&got, &want) < 1e-12, "{} qubit {:?}", nq, g);
        }
        // sorta oso bat ere bai, fusio eta bloke lehenetsiekin
        let gates: Vec<QGate> = (0..60).map(|_| random_gate(&mut rng, nq)).collect();
        let mut want = start.clone();
        for g in &gates {
            reference(&mut want, g);
        }
        let mut got = start.clone();
        kernel::apply_gates(&mut got, nq, &gates);
        assert!(max_diff(&got, &want) < 1e-10, "{} qubit sorta", nq);
    }
}

#[test]
fn fusion_merges_through_gates_on_other_qubits_only() {
    let h = qoa::program::Gate::H.matrix();
    let t = qoa::program::Gate::T.matrix();
    let x = qoa::program::Gate::X.matrix();
    let mat = |q, m| QGate::Mat { q, m, controls: 0 };
    // H q0, T q1, H q0: bi H-ak bateratzen dira (identitatea) eta kentzen dira; T geratzen da
    let fused = kernel::fuse(&[mat(0, h), mat(1, t), mat(0, h)]);
    assert_eq!(fused.len(), 1, "{:?}", fused);
    // CNOT q0 -> q1 tartean: q0-ko ateak ezin dira bateratu
    let cx = QGate::Mat {
        q: 1,
        m: x,
        controls: 1,
    };
    assert_eq!(kernel::fuse(&[mat(0, h), cx, mat(0, h)]).len(), 3);
    // kontrol gisa ere blokeatzen du: q1-eko ateak ezin dira CNOT-aren gainetik bateratu
    assert_eq!(kernel::fuse(&[mat(1, t), cx, mat(1, t)]).len(), 3);
    // trukaketak bi qubitak blokeatzen ditu
    let sw = QGate::Swap {
        a: 0,
        b: 2,
        controls: 0,
    };
    assert_eq!(
        kernel::fuse(&[mat(2, t), sw, mat(2, t), mat(1, t), mat(1, t)]).len(),
        4
    );
}
