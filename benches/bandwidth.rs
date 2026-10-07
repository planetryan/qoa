// fitxategi honek makinaren memoria-banda zabalera neurtzen du eta qubit-ate bakoitzaren kernelak haren
// ondoan jartzen ditu, kernelak memoriaren mugatik zein urrun dauden ikusteko.
//
//   cargo bench --bench bandwidth               # 20, 24 eta 27 qubit
//   cargo bench --bench bandwidth -- 23 28      # qubit-kopuru jakin batzuk
//   QOA_SIMD=scalar cargo bench --bench bandwidth   # mihiztadurarik gabe, konparatzeko
//
// "irakurri+idatzi" lerroa da muga: bektore bat tokian bertan eskalatzea (ate batek egiten duen memoria-
// trafiko bera, kalkulurik gabe). GB/s = 2 * egoeraren tamaina / denbora (irakurri eta idatzi).
// egoera cache-n sartzen bada (makina honen L2/L3-aren arabera), mugak memoria nagusiarenak baino askoz
// handiagoak dira.

use qoa::kernel::QGate;
use qoa::program::Gate;
use qoa::state::State;
use rayon::prelude::*;
use std::time::Instant;

fn best(mut f: impl FnMut(), reps: usize) -> f64 {
    let mut b = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let sizes = if sizes.is_empty() {
        vec![20, 24, 27]
    } else {
        sizes
    };
    println!(
        "kernelak: {}, hariak: {}",
        qoa::simd::active().name(),
        rayon::current_num_threads()
    );
    for n in sizes {
        let bytes = 2.0 * 16.0 * (1u64 << n) as f64;
        let reps = if n < 24 { 20 } else { 4 };
        let mut s = State::new(n, 0, 2).unwrap();
        let line = |name: &str, t: f64| {
            println!(
                "{:>2} qubit  {:<22} {:9.2} ms  {:7.1} GB/s",
                n,
                name,
                t * 1e3,
                bytes / t / 1e9
            )
        };

        // muga: bektore bera irakurri eta idatzi, hari guztiekin
        let mut v = vec![1.0f64; 2 << n];
        line(
            "irakurri+idatzi (muga)",
            best(
                || {
                    v.par_chunks_mut(1 << 14)
                        .for_each(|c| c.iter_mut().for_each(|x| *x *= 0.999_999))
                },
                reps,
            ),
        );
        drop(v);

        let h = Gate::H.matrix();
        for q in [0, 1, 3, n / 2, n - 1] {
            line(&format!("h q{}", q), best(|| s.apply_1q(q, h, 0), reps));
        }
        line(
            "rx q5",
            best(|| s.apply_1q(5, Gate::RX(0.3).matrix(), 0), reps),
        );
        line(
            "rz (diagonala)",
            best(|| s.apply_1q(n / 2, Gate::RZ(0.3).matrix(), 0), reps),
        );
        line("cnot q0 -> erdikoa", best(|| s.apply_x(n / 2, 1), reps));
        line(
            "cz q0, azkena",
            best(
                || s.apply_phase_mask(1 | 1 << (n - 1), std::f64::consts::PI),
                reps,
            ),
        );
        line("swap q0, azkena", best(|| s.apply_swap(0, n - 1, 0), reps));

        // geruza bat (h qubit guztietan): sorta gisa, fusio eta cache-blokeekin
        let layer: Vec<QGate> = (0..n)
            .map(|q| QGate::Mat {
                q,
                m: h,
                controls: 0,
            })
            .collect();
        let t = best(|| s.apply_gates(&layer), reps);
        println!(
            "{:>2} qubit  {:<22} {:9.2} ms  {:7.1} GB/s ate bakoitzeko, {} ate",
            n,
            "h geruza (sorta)",
            t * 1e3,
            bytes * n as f64 / t / 1e9,
            n
        );
        println!();
    }
}
