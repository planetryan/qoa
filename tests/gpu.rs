// fitxategi honek src/gpu.rs eta src/vulkan.rs egiaztatzen ditu.
//
//   1. planifikatzailea eta indize-kalkulua, GPUrik gabe: `emulate`-k shader-aren lauzaz lauzako kalkulu bera
//      egiten du rust-en, eta erreferentzia inozoarekin konparatzen da (beti exekutatzen da).
//   2. gailua: vulkan gailu bat badago, GPUaren emaitza CPUarenarekin. gailurik ez badago, proba hauek
//      zergatik esaten dute eta ez dute ezer egiaztatzen; QOA_REQUIRE_GPU=1-ekin huts egiten dute.
//   3. shader-a: src/shaders/tile.spv src/shaders/tile.comp-etik konpilatua dagoela (bertsio-marka).

mod common;

use common::{max_diff, random_gate, random_state, reference};
use num_complex::Complex64 as C;
use qoa::gpu::{self, Gpu};
use qoa::kernel::{self, QGate};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::OnceLock;

// ausazko atea; batzuetan matrize erreala (shader-eko REAL bidea: h, ry, ...)
fn gate(rng: &mut StdRng, nq: usize) -> QGate {
    match random_gate(rng, nq) {
        QGate::Mat { q, m, controls } if rng.random_bool(0.3) => QGate::Mat {
            q,
            m: m.map(|c| C::new(c.re, 0.0)),
            controls,
        },
        g => g,
    }
}

// luzera: 2^nq, edo modu batekin (3 maila) 3 * 2^nq
fn length(case: usize, nq: usize) -> usize {
    if case.is_multiple_of(3) {
        3 << nq
    } else {
        1 << nq
    }
}

#[test]
fn the_plan_matches_the_reference() {
    let mut rng = StdRng::seed_from_u64(21);
    for case in 0..300 {
        // lauza (10 qubit) baino gutxiago eta gehiago
        let nq = 1 + case % 14;
        let len = length(case, nq);
        let gates: Vec<QGate> = (0..rng.random_range(1..60))
            .map(|_| gate(&mut rng, nq))
            .collect();
        let start = random_state(&mut rng, len);
        let mut want = start.clone();
        for g in &gates {
            reference(&mut want, g);
        }
        let stages = gpu::plan(nq, &gates);
        for st in &stages {
            assert_eq!(gpu::races(st), None, "kasua {case} nq={nq}");
        }
        let mut got = start;
        gpu::emulate(&mut got, &stages);
        assert!(
            max_diff(&got, &want) < 1e-10,
            "kasua {case} nq={nq} {gates:?}"
        );
    }
}

// lasterketa-egiaztatzaileak berak barrera falta bat aurkitzen du
#[test]
fn the_race_check_finds_a_missing_barrier() {
    let x = |q| QGate::Mat {
        q,
        m: [
            C::new(0.0, 0.0),
            C::new(1.0, 0.0),
            C::new(1.0, 0.0),
            C::new(0.0, 0.0),
        ],
        controls: 0,
    };
    let mut st = gpu::plan(12, &[x(0), x(9)]).remove(0);
    assert_eq!(gpu::races(&st), None);
    st.gates[0].sync = 0;
    assert!(gpu::races(&st).is_some());
    // bi diagonalen artean ez da behar
    let z = |q: usize| QGate::phase(1 << q, 0.5);
    let st = gpu::plan(12, &[z(0), z(9), z(3)]).remove(0);
    assert!(st.gates.iter().all(|g| g.sync == 0));
    assert_eq!(gpu::races(&st), None);
}

#[test]
fn the_plan_keeps_low_qubits_and_groups_gates() {
    let h = {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let a = C::new(s, 0.0);
        [a, a, a, -a]
    };
    let on = |q| QGate::Mat {
        q,
        m: h,
        controls: 0,
    };
    // 20 qubit: 0..4 beti lauzan; 5 helburu altu etapa bakarrean sartzen dira, seigarrenak berria irekitzen du
    let gates: Vec<QGate> = [15, 16, 17, 18, 19, 3, 12].into_iter().map(on).collect();
    let stages = gpu::plan(20, &gates);
    assert_eq!(stages.len(), 2);
    assert_eq!(stages[0].pos, [0, 1, 2, 3, 4, 15, 16, 17, 18, 19]);
    assert_eq!(stages[0].low, 5);
    assert_eq!(stages[0].gates.len(), 6);
    // bigarrena: 0..4, 12 eta hurrengo baxuenak
    assert_eq!(stages[1].pos, [0, 1, 2, 3, 4, 5, 6, 7, 8, 12]);
    assert_eq!(stages[1].low, 9);
    // kanpoko kontrolak lauzaren oinarriarekin egiaztatzen dira, ez lauzan
    let cx = QGate::Mat {
        q: 0,
        m: h,
        controls: 1 << 19,
    };
    let s = &gpu::plan(20, &[cx])[0];
    assert_eq!(s.gates[0].outside, 1 << 19);
    assert_eq!(s.gates[0].cmask, 0);
}

// gailua behin ireki proba guztietarako; ez badago, zergatik
fn device() -> Option<&'static Gpu> {
    static GPU: OnceLock<Result<Gpu, String>> = OnceLock::new();
    match GPU.get_or_init(Gpu::open) {
        Ok(g) => Some(g),
        Err(why) => {
            if std::env::var("QOA_REQUIRE_GPU").is_ok_and(|v| v == "1") {
                panic!("QOA_REQUIRE_GPU=1 baina gailurik ez: {why}");
            }
            eprintln!("GPU proba saltatua, ez dago gailurik: {why}");
            None
        }
    }
}

#[test]
fn the_device_matches_the_cpu() {
    let Some(gpu) = device() else { return };
    let mut rng = StdRng::seed_from_u64(31);
    for case in 0..120 {
        let nq = 1 + case % 16;
        let len = length(case, nq);
        let gates: Vec<QGate> = (0..rng.random_range(1..80))
            .map(|_| gate(&mut rng, nq))
            .collect();
        let start = random_state(&mut rng, len);
        let mut want = start.clone();
        kernel::apply_gates(&mut want, nq, &gates);
        let mut got = start;
        gpu.apply(&mut got, nq, &gates).unwrap();
        assert!(max_diff(&got, &want) < 1e-10, "kasua {case} nq={nq}");
    }
}

#[test]
fn the_device_handles_a_large_state_and_reuses_its_buffers() {
    let Some(gpu) = device() else { return };
    let mut rng = StdRng::seed_from_u64(41);
    let nq = 21;
    let start = random_state(&mut rng, 1 << nq);
    for round in 0..2 {
        let gates: Vec<QGate> = (0..200).map(|_| gate(&mut rng, nq)).collect();
        let mut want = start.clone();
        kernel::apply_gates(&mut want, nq, &gates);
        let mut got = start.clone();
        gpu.apply(&mut got, nq, &gates).unwrap();
        assert!(max_diff(&got, &want) < 1e-10, "txanda {round}");
    }
    let s = gpu.stats();
    assert!(s.batches >= 2 && s.bytes_up >= 2 * (16 << nq) as u64);
}

// FNV-1a 64, tools/shaders.sh-ekoa bera
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[test]
fn the_compiled_kernel_is_built_from_the_current_source() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/");
    let source = std::fs::read(format!("{dir}tile.comp")).unwrap();
    let stamp = std::fs::read_to_string(format!("{dir}tile.spv.fnv")).unwrap();
    assert_eq!(
        format!("{:016x}", fnv(&source)),
        stamp.trim(),
        "src/shaders/tile.comp aldatu da tile.spv konpilatu ondoren: exekutatu tools/shaders.sh"
    );
}
