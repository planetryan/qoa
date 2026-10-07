// proben laguntzaileak: ausazko egoerak eta ateak, eta erreferentziazko aplikazio inozoa
// (tests/kernel.rs eta tests/gpu.rs)

#![allow(dead_code)]

use num_complex::Complex64 as C;
use qoa::kernel::QGate;
use rand::Rng;
use rand::rngs::StdRng;

pub fn random_state(rng: &mut StdRng, n: usize) -> Vec<C> {
    (0..n)
        .map(|_| C::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
        .collect()
}

// erreferentzia: bikote bakoitza bere aldetik, optimizaziorik gabe
pub fn reference(amps: &mut [C], g: &QGate) {
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

pub fn random_gate(rng: &mut StdRng, nq: usize) -> QGate {
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

pub fn max_diff(a: &[C], b: &[C]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}
