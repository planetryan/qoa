// fitxategi honek src/kernel.rs egiaztatzen du: ate-sortak (fusioa eta cache-blokeak) eta egoera osoko
// kernelak, ausazko ateekin (kontrolak edozein qubitetan, trukaketak, diagonalak, x), erreferentzia
// inozo baten aurka. bloke-tamaina txikiak behartzen dira egoera txikietan ere etapa mota guztiak
// probatzeko, eta qubit + modu egoerak (luzera ez da 2ren berretura).

mod common;

use common::{max_diff, random_gate, random_state, reference};
use qoa::kernel::{self, QGate};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
    sweeps_on_large_states();
}

// hari-kopurua ez da beti 2ren berredura (3, 6, 12 nukleo): zatien tamainak horren arabera ez dira aldatu
// behar (macos-eko 3 nukleoko makina batek aurkitu zuen akatsa)
#[test]
fn sweeps_are_right_with_any_thread_count() {
    // 19 qubit eta ate altuak: bloke handi gutxiko bidea, non zatiak (erdia / zereginak) TASK_MIN baino
    // handiagoak diren eta beraz hari-kopuruaren araberakoak
    let nq = 19;
    let mut rng = StdRng::seed_from_u64(13);
    let start = random_state(&mut rng, 1 << nq);
    let m = random_state(&mut rng, 4);
    let gates = [
        QGate::Mat {
            q: 18,
            m: [m[0], m[1], m[2], m[3]],
            controls: 1 << 3,
        },
        QGate::Mat {
            q: 17,
            m: [m[0], m[1], m[2], m[3]],
            controls: 0,
        },
        QGate::Swap {
            a: 2,
            b: 18,
            controls: 1 << 9,
        },
        QGate::Swap {
            a: 17,
            b: 5,
            controls: 0,
        },
    ];
    for threads in [3, 6, 12] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for g in &gates {
            let mut want = start.clone();
            reference(&mut want, g);
            let mut got = start.clone();
            pool.install(|| match *g {
                QGate::Mat { q, m, controls } => kernel::sweep_mat(&mut got, q, &m, controls),
                QGate::Swap { a, b, controls } => kernel::sweep_swap(&mut got, a, b, controls),
            });
            assert!(max_diff(&got, &want) < 1e-12, "{} hari {:?}", threads, g);
        }
        pool.install(sweeps_on_large_states);
    }
}

fn sweeps_on_large_states() {
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
