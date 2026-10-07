// fitxategi honek mihiztadurako mikro-kernel bakoitza (src/simd.rs) bertsio eskalarraren aurka egiaztatzen du,
// makina honetan dauden isa guztiekin: lasterketa-sorta askorekin (luzera, pausoa, kopurua), erreika
// bakoitzeko koefiziente desberdinekin eta ausazko datuekin. slice-tik kanpo ezer ez dela idazten ere bai.
// aarch64 eta riscv64 qemu-n exekutatzeko, ikus docs/performance.md.

use num_complex::Complex64 as C;
use qoa::simd::{self, Isa, Spans};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn c(rng: &mut StdRng) -> C {
    C::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
}

fn random(rng: &mut StdRng, n: usize) -> Vec<C> {
    (0..n).map(|_| c(rng)).collect()
}

fn close(a: &[C], b: &[C]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).norm() < 1e-14)
}

fn others() -> Vec<Isa> {
    simd::available()
        .iter()
        .copied()
        .filter(|&i| i != Isa::Scalar)
        .collect()
}

fn random_spans(rng: &mut StdRng) -> Spans {
    let len = 4 * rng.random_range(1..9);
    let gap = 4 * rng.random_range(0..5);
    Spans {
        start: rng.random_range(0..7),
        len,
        stride: len + gap,
        count: rng.random_range(1..5),
    }
}

#[test]
fn every_isa_matches_scalar() {
    println!(
        "isa-k: {:?}, aktiboa: {:?}",
        simd::available(),
        simd::active()
    );
    let mut rng = StdRng::seed_from_u64(1);
    for isa in others() {
        for case in 0..400 {
            let sp = random_spans(&mut rng);
            let size = sp.start + (sp.count - 1) * sp.stride + sp.len + rng.random_range(0..9);
            // erdietan erreika guztiak berdinak (ohikoena), bestean desberdinak
            let same = case % 2 == 0;
            let one = [c(&mut rng), c(&mut rng), c(&mut rng), c(&mut rng)];
            let k: simd::Mat2Lanes = std::array::from_fn(|_| {
                if same {
                    one
                } else {
                    [c(&mut rng), c(&mut rng), c(&mut rng), c(&mut rng)]
                }
            });
            let lo = random(&mut rng, size);
            let hi = random(&mut rng, size);

            let (mut l1, mut h1) = (lo.clone(), hi.clone());
            let (mut l2, mut h2) = (lo.clone(), hi.clone());
            simd::mat2_with(Isa::Scalar, &mut l1, &mut h1, sp, &k);
            simd::mat2_with(isa, &mut l2, &mut h2, sp, &k);
            assert!(
                close(&l1, &l2) && close(&h1, &h2),
                "{:?} mat2 {:?}",
                isa,
                sp
            );

            // lo eta hi slice berean
            let off = size;
            let mut both = lo.clone();
            both.extend_from_slice(&hi);
            let mut b2 = both.clone();
            simd::mat2_in_with(Isa::Scalar, &mut both, off, sp, &k);
            simd::mat2_in_with(isa, &mut b2, off, sp, &k);
            assert!(close(&both, &b2), "{:?} mat2_in {:?}", isa, sp);

            let ks: simd::SwizzleLanes = std::array::from_fn(|j| [k[j][0], k[j][1]]);
            for half in [1, 2] {
                let mut v1 = lo.clone();
                let mut v2 = lo.clone();
                simd::swizzle_with(Isa::Scalar, &mut v1, half, sp, &ks);
                simd::swizzle_with(isa, &mut v2, half, sp, &ks);
                assert!(close(&v1, &v2), "{:?} swizzle half={} {:?}", isa, half, sp);
            }

            let kd: simd::ScaleLanes = std::array::from_fn(|j| k[j][2]);
            let mut v1 = lo.clone();
            let mut v2 = lo.clone();
            simd::scale_with(Isa::Scalar, &mut v1, sp, &kd);
            simd::scale_with(isa, &mut v2, sp, &kd);
            assert!(close(&v1, &v2), "{:?} scale {:?}", isa, sp);
        }
    }
}

#[test]
fn kernels_write_only_inside_their_spans() {
    let mut rng = StdRng::seed_from_u64(2);
    for isa in simd::available().iter().copied() {
        for _ in 0..100 {
            let sp = random_spans(&mut rng);
            let size = sp.start + (sp.count - 1) * sp.stride + sp.len + 3;
            let inside = |i: usize| {
                i >= sp.start
                    && (i - sp.start) / sp.stride < sp.count
                    && (i - sp.start) % sp.stride < sp.len
            };
            let k: simd::Mat2Lanes =
                std::array::from_fn(|_| [c(&mut rng), c(&mut rng), c(&mut rng), c(&mut rng)]);
            let orig = random(&mut rng, size);
            let mut v = orig.clone();
            let mut w = orig.clone();
            simd::mat2_with(isa, &mut v, &mut w, sp, &k);
            for i in (0..size).filter(|&i| !inside(i)) {
                assert_eq!((v[i], w[i]), (orig[i], orig[i]), "{:?} {:?} {}", isa, sp, i);
            }
            let mut v = orig.clone();
            simd::swizzle_with(
                isa,
                &mut v,
                2,
                sp,
                &std::array::from_fn(|j| [k[j][0], k[j][1]]),
            );
            for i in (0..size).filter(|&i| !inside(i)) {
                assert_eq!(v[i], orig[i], "{:?} swizzle {:?} {}", isa, sp, i);
            }
            let mut v = orig.clone();
            simd::scale_with(isa, &mut v, sp, &std::array::from_fn(|j| k[j][0]));
            for i in (0..size).filter(|&i| !inside(i)) {
                assert_eq!(v[i], orig[i], "{:?} scale {:?} {}", isa, sp, i);
            }
        }
    }
}

#[test]
fn identity_lanes_leave_amplitudes_exactly_unchanged() {
    // kontrolak betetzen ez dituzten erreiketan identitatea: emaitzak bit berak izan behar ditu
    let one = C::new(1.0, 0.0);
    let zero = C::new(0.0, 0.0);
    let mut rng = StdRng::seed_from_u64(3);
    for isa in simd::available().iter().copied() {
        let lo = random(&mut rng, 64);
        let hi = random(&mut rng, 64);
        let (mut a, mut b) = (lo.clone(), hi.clone());
        let id = [[one, zero, zero, one]; 4];
        simd::mat2_with(isa, &mut a, &mut b, Spans::one(0, 64), &id);
        assert!(a == lo && b == hi, "{:?}", isa);
        let mut v = lo.clone();
        simd::swizzle_with(isa, &mut v, 1, Spans::one(0, 64), &[[one, zero]; 4]);
        assert!(v == lo, "{:?}", isa);
        simd::scale_with(isa, &mut v, Spans::one(0, 64), &[one; 4]);
        assert!(v == lo, "{:?}", isa);
    }
}
