// fitxategi honek src/dd.rs eta egoeraren baturak egiaztatzen ditu: errorerik gabeko transformazioak
// zehatzak direla (FMA-rekin eta dekker-ekin), double-double baturak f64 arruntak galtzen duena gordetzen
// duela, eta egoeraren baturek (araua, qubit baten probabilitatea) bit berberak ematen dituztela hari-kopurua
// edozein dela ere.

mod common;

use num_complex::Complex64 as C;
use qoa::dd::{self, Dd, two_prod, two_sum};
use qoa::state::State;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn error_free_transformations_are_exact() {
    let x = 1.0 + 2f64.powi(-30);
    // (1 + 2^-30)² = 1 + 2^-29 + 2^-60: azken gaia f64 batean ez da sartzen
    for (p, e) in [two_prod::<true>(x, x), two_prod::<false>(x, x)] {
        assert_eq!(p, 1.0 + 2f64.powi(-29));
        assert_eq!(e, 2f64.powi(-60));
    }
    let (s, e) = two_sum(1.0, 2f64.powi(-60));
    assert_eq!((s, e), (1.0, 2f64.powi(-60)));
    // dekker eta FMA bat datoz ausazko balioetan
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..10_000 {
        let a: f64 = rand::Rng::random_range(&mut rng, -1.0..1.0);
        let b: f64 = rand::Rng::random_range(&mut rng, -1.0..1.0);
        assert_eq!(two_prod::<true>(a, b), two_prod::<false>(a, b));
    }
}

#[test]
fn double_double_sums_keep_what_plain_sums_lose() {
    // 1 eta gero 2^20 aldiz 2^-60: f64 arruntean gai txikiak galtzen dira
    let tiny = 2f64.powi(-60);
    let plain = (0..1 << 20).fold(1.0, |acc, _| acc + tiny);
    assert_eq!(plain, 1.0);
    let exact = dd::sum(std::iter::once(1.0).chain((0..1 << 20).map(|_| tiny)));
    assert_eq!(exact, 1.0 + 2f64.powi(-40));
    // araua: anplitude bat 1 eta 2^20 anplitude 2^-30 (|a|² = 2^-60)
    let mut amps = vec![C::new(2f64.powi(-30), 0.0); 1 << 20];
    amps[0] = C::new(1.0, 0.0);
    let n = dd::norm_sum(&amps);
    assert_eq!(n.value(), 1.0 + 2f64.powi(-40) - 2f64.powi(-60));
    assert_eq!(
        Dd::from(1.0).plus(-n).value(),
        -(2f64.powi(-40) - 2f64.powi(-60))
    );
}

#[test]
fn state_sums_do_not_depend_on_the_thread_count() {
    let mut rng = StdRng::seed_from_u64(9);
    let amps = common::random_state(&mut rng, 1 << 20);
    let state = State::from_amplitudes(20, 0, 2, amps).unwrap();
    let in_pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                (
                    state.norm_sqr().to_bits(),
                    (0..20)
                        .map(|q| state.prob_one(q).to_bits())
                        .collect::<Vec<_>>(),
                )
            })
    };
    let one = in_pool(1);
    for threads in [2, 3, 8] {
        assert_eq!(in_pool(threads), one, "{} hari", threads);
    }
}
