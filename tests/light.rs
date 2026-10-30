// fitxategi honek src/light.rs egiaztatzen du: mdct-ak seinalea zehazki berreraikitzen duela eta energia
// gordetzen duela (modu ortonormalak), efekturik gabe argiak seinalea bera itzultzen duela, homodinoaren
// zarata hutsaren (eta estutze eta galeren) bariantza zehatza dela, kerr-en batez besteko eremua fock
// oinarriko batura zuzenarekin bat datorrela, poisson laginketa zehatza dela, konbinazio ez-zehatzak
// erroreak direla, kerr-en ondoko homodinoaren dentsitatea fock oinarriko batura zuzenarekin bat datorrela
// eta haren laginek batez besteko eta bariantza zehatzak dituztela, eta hazi berak emaitza bera ematen duela
// hari-kopurua edozein dela ere.

use num_complex::Complex64 as Cx;
use qoa::light::{
    self, Detect, KerrSampler, Light, Mdct, Pure, count_pure, kerr_mean, kerr_mean_mixed,
    kerr_mean_pure, normal_pair, number_distribution, poisson, poisson_log_pmf,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn signal(len: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len)
        .map(|i| 0.6 * (i as f64 * 0.031).sin() + 0.3 * rng.random_range(-1.0..1.0))
        .collect()
}

#[test]
fn mdct_reconstructs_the_signal_and_keeps_its_energy() {
    for n in [64, 256, 1024] {
        let mdct = Mdct::new(n);
        let x = signal(10_007, n as u64);
        let frames = light::analyze(&mdct, &x);
        let y = light::synthesize(&mdct, &frames, x.len());
        let worst = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-13, "n {}: {}", n, worst);
        let ex: f64 = x.iter().map(|v| v * v).sum();
        let ec: f64 = frames.iter().flatten().map(|v| v * v).sum();
        assert!(((ex - ec) / ex).abs() < 1e-12, "parseval: {} {}", ex, ec);
    }
}

fn run(light: &Light, x: &[f64], seed: u64) -> Vec<f64> {
    let mdct = Mdct::new(256);
    let mut channels = vec![light::analyze(&mdct, x)];
    light.process(&mut channels, seed);
    light::synthesize(&mdct, &channels[0], x.len())
}

#[test]
fn light_without_effects_returns_the_signal() {
    let x = signal(5000, 1);
    let mean = Light {
        detect: Detect::Mean,
        ..Light::default()
    };
    let y = run(&mean, &x, 7);
    let worst = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-12, "{}", worst);
    // galerarekin ere: argia ahulagoa da, baina detekzioaren ondoko irabazi klasikoak berreskuratzen du
    let lossy = Light {
        loss: 0.5,
        ..mean.clone()
    };
    let y = run(&lossy, &x, 7);
    assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-12));
}

// isiltasunean, homodinoaren zarata koefiziente bakoitzean: V/(2 t s²), s² = 2P, V = t·10^{-dB/10}/2 + (1-t)/2
#[test]
fn homodyne_noise_is_vacuum_noise_squeezed_and_lost_exactly() {
    let mdct = Mdct::new(512);
    for (loss, db) in [(0.0, 0.0), (0.0, 6.0), (0.3, 6.0), (0.5, 0.0)] {
        let light = Light {
            photons: 100.0,
            loss,
            squeeze_db: db,
            ..Light::default()
        };
        let mut channels = vec![light::analyze(&mdct, &vec![0.0; 400_000])];
        light.process(&mut channels, 3);
        let values: Vec<f64> = channels[0].iter().flatten().copied().collect();
        let n = values.len() as f64;
        let var = values.iter().map(|v| v * v).sum::<f64>() / n;
        let t = 1.0 - loss;
        let v = t * 10f64.powf(-db / 10.0) / 2.0 + (1.0 - t) / 2.0;
        let want = v / (2.0 * t * 2.0 * light.photons);
        // bariantza estimatzailearen errore estandarra: want·√(2/n)
        assert!(
            (var - want).abs() < 5.0 * want * (2.0 / n).sqrt(),
            "loss {} db {}: {} ≠ {}",
            loss,
            db,
            var,
            want
        );
    }
}

#[test]
fn the_beamsplitter_mesh_keeps_the_energy() {
    let x = signal(8000, 4);
    let light = Light {
        mix: 0.7,
        detect: Detect::Mean,
        ..Light::default()
    };
    let y = run(&light, &x, 1);
    let (ex, ey): (f64, f64) = (x.iter().map(|v| v * v).sum(), y.iter().map(|v| v * v).sum());
    // amaierako irudi erdiek energia pixka bat galtzen dute bete gabeko gainjartzean; barrualdea gordetzen da
    assert!(((ex - ey) / ex).abs() < 0.02, "{} {}", ex, ey);
    assert!(x.iter().zip(&y).any(|(a, b)| (a - b).abs() > 1e-3));
}

// ⟨a⟩ fock oinarrian: Σ_n c_n c_{n+1} √(n+1) e^{-iκ((n+1)n - n(n-1))}, c_n poisson anplitudeak (α erreala)
#[test]
fn kerr_mean_field_matches_the_fock_sum() {
    for (alpha, kappa) in [(3.0, 0.05), (7.0, 0.01), (-5.0, 0.2), (10.0, 0.003)] {
        let n_max = 400;
        let mut c = vec![0.0f64; n_max + 1];
        c[0] = (-alpha * alpha / 2.0f64).exp();
        for n in 1..=n_max {
            c[n] = c[n - 1] * alpha / (n as f64).sqrt();
        }
        let mut re = 0.0;
        for n in 0..n_max {
            let phase = -kappa * (2 * n) as f64;
            re += c[n] * c[n + 1] * ((n + 1) as f64).sqrt() * phase.cos();
        }
        let closed = kerr_mean(alpha, kappa);
        assert!(
            (re - closed).abs() < 1e-12,
            "{} {}: {} {}",
            alpha,
            kappa,
            re,
            closed
        );
    }
}

#[test]
fn poisson_samples_have_the_right_mean_and_variance() {
    let mut rng = StdRng::seed_from_u64(9);
    for lambda in [0.5, 3.0, 25.0, 1e4, 1e9] {
        let m = 200_000;
        let xs: Vec<f64> = (0..m).map(|_| poisson(&mut rng, lambda) as f64).collect();
        let mean = xs.iter().sum::<f64>() / m as f64;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (m - 1) as f64;
        let se_mean = (lambda / m as f64).sqrt();
        assert!(
            (mean - lambda).abs() < 5.0 * se_mean,
            "λ {}: batez bestekoa {}",
            lambda,
            mean
        );
        let se_var = lambda * (2.0 / m as f64).sqrt() * 1.2;
        assert!(
            (var - lambda).abs() < 5.0 * se_var,
            "λ {}: bariantza {}",
            lambda,
            var
        );
    }
    // log-pmf-a batura zuzenarekin, k handietan ere
    for (k, lambda) in [(30u64, 25.0), (500, 480.0), (10_000, 10_100.0)] {
        let ln_fact: f64 = (2..=k).map(|j| (j as f64).ln()).sum();
        let direct = -lambda + k as f64 * f64::ln(lambda) - ln_fact;
        assert!(
            (poisson_log_pmf(k, lambda) - direct).abs() < 1e-9,
            "{} {}",
            k,
            lambda
        );
    }
}

#[test]
fn normals_have_unit_variance() {
    let mut rng = StdRng::seed_from_u64(2);
    let m = 200_000;
    let mut sum = 0.0;
    let mut sq = 0.0;
    for _ in 0..m / 2 {
        let (a, b) = normal_pair(&mut rng);
        sum += a + b;
        sq += a * a + b * b;
    }
    assert!((sum / m as f64).abs() < 5.0 / (m as f64).sqrt());
    assert!((sq / m as f64 - 1.0).abs() < 5.0 * (2.0 / m as f64).sqrt());
}

#[test]
fn every_combination_has_an_exact_computation() {
    for l in [
        Light {
            loss: 1.0,
            ..Light::default()
        },
        Light {
            photons: 0.0,
            ..Light::default()
        },
        Light {
            squeeze_db: 40.0,
            ..Light::default()
        },
    ] {
        assert!(l.check().is_err(), "{:?}", l);
    }
    for detect in [Detect::Mean, Detect::Count, Detect::Homodyne] {
        for (kerr, squeeze_db) in [(0.0, 0.0), (2.0, 0.0), (0.0, 6.0), (2.0, 6.0)] {
            let l = Light {
                kerr,
                squeeze_db,
                loss: 0.2,
                detect,
                ..Light::default()
            };
            assert!(l.check().is_ok(), "{:?}", l);
        }
    }
}

#[test]
fn the_same_seed_gives_the_same_light_on_any_thread_count() {
    let x = signal(20_000, 8);
    let light = Light {
        photons: 50.0,
        detect: Detect::Count,
        kerr: 1.0,
        ..Light::default()
    };
    let in_pool = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| run(&light, &x, 5))
    };
    let one = in_pool(1);
    assert_eq!(in_pool(4), one);
    assert_ne!(run(&light, &x, 6), one);
}

// kerr-en ondoko X dentsitatea fock oinarrian zuzenean: Σ_n c_n e^{-iκn(n-1)} ψ_n(x), hermite funtzioak
// errepikapen egonkorrarekin (n txikiak)
fn fock_density(alpha: f64, kappa: f64, x: f64) -> f64 {
    let n_max = 300;
    let mut c = vec![0.0f64; n_max];
    c[0] = (-alpha * alpha / 2.0).exp();
    for n in 1..n_max {
        c[n] = c[n - 1] * alpha / (n as f64).sqrt();
    }
    let mut psi_prev = 0.0;
    let mut psi = std::f64::consts::PI.powf(-0.25) * (-x * x / 2.0).exp();
    let mut amp = Cx::new(0.0, 0.0);
    for (n, &cn) in c.iter().enumerate() {
        let ph = -kappa * (n * n.saturating_sub(1)) as f64;
        amp += Cx::from_polar(cn, ph) * psi;
        let next =
            (2.0 / (n + 1) as f64).sqrt() * x * psi - (n as f64 / (n + 1) as f64).sqrt() * psi_prev;
        psi_prev = psi;
        psi = next;
    }
    amp.norm_sqr()
}

#[test]
fn kerr_homodyne_density_matches_the_fock_sum() {
    for (alpha, kappa) in [
        (2.0, 0.1),
        (4.0, 0.05),
        (-3.0, 0.3),
        (6.0, 0.01),
        (8.0, 0.002),
    ] {
        let sampler = KerrSampler::new(kappa);
        let mut worst: f64 = 0.0;
        for i in -400..=400 {
            let x = i as f64 * 0.03;
            worst = worst.max((sampler.density(alpha, x) - fock_density(alpha, kappa, x)).abs());
        }
        assert!(worst < 1e-12, "α {} κ {}: {}", alpha, kappa, worst);
    }
}

// ⟨x⟩ = √2 Re⟨a⟩, ⟨x²⟩ = Re⟨a²⟩ + n̄ + 1/2, ⟨a²⟩ = α² e^{-2iκ} exp(n̄(e^{-4iκ} - 1))
fn kerr_moments(alpha: f64, kappa: f64) -> (f64, f64) {
    let n = alpha * alpha;
    let a1 = alpha * (Cx::new(0.0, -2.0 * kappa).exp() - 1.0).scale(n).exp();
    let a2 =
        Cx::from_polar(n, -2.0 * kappa) * (Cx::new(0.0, -4.0 * kappa).exp() - 1.0).scale(n).exp();
    let mean = 2f64.sqrt() * a1.re;
    (mean, a2.re + n + 0.5 - mean * mean)
}

#[test]
fn kerr_homodyne_samples_have_the_exact_mean_and_variance() {
    // txikia, ertaina eta distiratsua (n̄ = 9e4, kerr fasea 2 rad)
    for (alpha, kappa, m) in [
        (4.0, 0.05, 6_000),
        (30.0, 0.0005, 3_000),
        (300.0, 2.0 / (2.0 * 9e4), 1_500),
    ] {
        let sampler = KerrSampler::new(kappa);
        let mut rng = StdRng::seed_from_u64(17);
        let xs: Vec<f64> = (0..m).map(|_| sampler.sample(alpha, &mut rng)).collect();
        let mean = xs.iter().sum::<f64>() / m as f64;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (m - 1) as f64;
        let (want_mean, want_var) = kerr_moments(alpha, kappa);
        let se = (want_var / m as f64).sqrt();
        assert!(
            (mean - want_mean).abs() < 5.0 * se,
            "α {}: ⟨x⟩ {} ≠ {}",
            alpha,
            mean,
            want_mean
        );
        // bariantzaren errore estandarra: σ²·√(2/m), kurtosia kontuan hartzeko 2 aldiz
        assert!(
            (var - want_var).abs() < 5.0 * 2.0 * want_var * (2.0 / m as f64).sqrt(),
            "α {}: var {} ≠ {}",
            alpha,
            var,
            want_var
        );
    }
}

// banaketa metatua dentsitatearen integralarekin bat dator (trapezioak, h = 0.002) eta 0-tik 1-era doa
#[test]
fn kerr_homodyne_cdf_is_the_integral_of_the_density() {
    for (alpha, rad) in [(3.0, 0.6), (300.0, 2.0)] {
        let kappa = rad / (2.0 * alpha * alpha);
        let sampler = KerrSampler::new(kappa);
        let center = kerr_moments(alpha, kappa).0;
        let (lo, n, h) = (center - 40.0, 40_000, 0.002);
        let xs: Vec<f64> = (0..=n).map(|i| lo + i as f64 * h).collect();
        let v = sampler.cdf_and_density(alpha, &xs);
        assert!(
            v[0].0.abs() < 1e-14 && (v[n].0 - 1.0).abs() < 1e-12,
            "{:?} {:?}",
            v[0],
            v[n]
        );
        let mut acc = 0.0;
        for i in 1..=n {
            acc += (v[i - 1].1 + v[i].1) * h / 2.0;
            if i % 4000 == 0 {
                assert!(
                    (v[i].0 - v[0].0 - acc).abs() < 1e-6,
                    "α {} x {}: {} ≠ {}",
                    alpha,
                    xs[i],
                    v[i].0 - v[0].0,
                    acc
                );
            }
        }
    }
}

// D(α) S(r) |0> fock oinarrian, n = 0-tik: c_0 = e^{-|α|²/2 - ᾱ² tanh r/2}/√cosh r,
// c_{n+1} = ((α + ᾱ tanh r) c_n - tanh r √n c_{n-1})/√(n+1)
fn squeezed_fock(alpha: Cx, r: f64, n_max: usize) -> Vec<Cx> {
    let tr = r.tanh();
    let mut c = vec![Cx::new(0.0, 0.0); n_max];
    c[0] =
        (-alpha.norm_sqr() / 2.0 - alpha.conj() * alpha.conj() * tr / 2.0).exp() / r.cosh().sqrt();
    let big = alpha + alpha.conj() * tr;
    for n in 0..n_max - 1 {
        let prev = if n > 0 { c[n - 1] } else { Cx::new(0.0, 0.0) };
        c[n + 1] = (big * c[n] - prev * tr * (n as f64).sqrt()) / ((n + 1) as f64).sqrt();
    }
    c
}

#[test]
fn photon_numbers_of_squeezed_light_are_exact() {
    // estutzerik gabe: poisson zehazki
    let (lo, p) = number_distribution(Cx::new(30.0, 0.0), 0.0);
    for (j, &pj) in p.iter().enumerate() {
        let n = lo + j as i64;
        let want = if n < 0 {
            0.0
        } else {
            poisson_log_pmf(n as u64, 900.0).exp()
        };
        assert!((pj - want).abs() < 1e-14, "n {}: {} {}", n, pj, want);
    }
    // estutuarekin: fock-eko errepikapen zuzenarekin, eta batez bestekoa eta bariantza formulekin
    for (alpha, r) in [
        (Cx::new(3.0, 0.0), 0.5),
        (Cx::new(2.0, 1.5), 0.8),
        (Cx::new(0.0, 0.0), 1.0),
        (Cx::new(-4.0, 0.5), 0.3),
    ] {
        let (lo, p) = number_distribution(alpha, r);
        let c = squeezed_fock(alpha, r, 400);
        for (j, &pj) in p.iter().enumerate() {
            let n = lo + j as i64;
            if (0..400).contains(&n) {
                let want = c[n as usize].norm_sqr();
                assert!(
                    (pj - want).abs() < 1e-13,
                    "α {} r {} n {}: {} {}",
                    alpha,
                    r,
                    n,
                    pj,
                    want
                );
            }
        }
        let total: f64 = p.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "{}", total);
    }
    // laginak: batez bestekoa |α|² + sinh² r
    let mut rng = StdRng::seed_from_u64(4);
    let (alpha, r) = (Cx::new(20.0, 0.0), 0.6);
    let m = 20_000;
    let mean = (0..m)
        .map(|_| count_pure(alpha, r, &mut rng) as f64)
        .sum::<f64>()
        / m as f64;
    let want = 400.0 + r.sinh().powi(2);
    let var = 400.0 * (-2.0 * r).exp() + 2.0 * (r.sinh() * r.cosh()).powi(2);
    assert!(
        (mean - want).abs() < 5.0 * (var / m as f64).sqrt(),
        "{} {}",
        mean,
        want
    );
}

#[test]
fn kerr_mean_field_of_squeezed_light_is_exact() {
    for (alpha, r, kappa) in [
        (Cx::new(3.0, 0.0), 0.4, 0.05),
        (Cx::new(2.0, -1.0), 0.7, 0.02),
        (Cx::new(0.0, 0.0), 0.9, 0.1),
    ] {
        let c = squeezed_fock(alpha, r, 400);
        let mut want = Cx::new(0.0, 0.0);
        for n in 0..399 {
            let a = c[n] * Cx::from_polar(1.0, -kappa * (n * n.saturating_sub(1)) as f64);
            let b = c[n + 1] * Cx::from_polar(1.0, -kappa * ((n + 1) * n) as f64);
            want += a.conj() * b * ((n + 1) as f64).sqrt();
        }
        let got = kerr_mean_pure(alpha, r, kappa);
        assert!(
            (got - want).norm() < 1e-12,
            "{} {} {}: {} {}",
            alpha,
            r,
            kappa,
            got,
            want
        );
    }
    // galeraren zarataren gaineko batez bestekoa, trapezioekin u-ren gainean
    let (a, r, s, kappa) = (5.0, 0.5, 0.7, 0.01);
    let n = 8000;
    let h = 24.0 * s / n as f64;
    let mut num = Cx::new(0.0, 0.0);
    for i in 0..=n {
        let u = -12.0 * s + i as f64 * h;
        let w = (-u * u / (2.0 * s * s)).exp() / ((2.0 * std::f64::consts::PI).sqrt() * s);
        let f = kerr_mean_pure(Cx::new(a, u), r, kappa) * w * h;
        num += if i == 0 || i == n { f / 2.0 } else { f };
    }
    let closed = kerr_mean_mixed(a, r, s, kappa);
    assert!((closed - num).norm() < 1e-12, "{} {}", closed, num);
}

#[test]
fn kerr_homodyne_of_squeezed_light_matches_the_fock_sum() {
    for (alpha, r, kappa) in [
        (Cx::new(3.0, 0.0), 0.4, 0.05),
        (Cx::new(2.0, 1.0), 0.6, 0.03),
        (Cx::new(0.0, 0.0), 0.7, 0.1),
    ] {
        let c = squeezed_fock(alpha, r, 300);
        let sampler = KerrSampler::new(kappa);
        let state = Pure::lab(alpha, r);
        let mut worst: f64 = 0.0;
        for i in -300..=300 {
            let x = i as f64 * 0.04;
            let mut psi_prev = 0.0;
            let mut psi = std::f64::consts::PI.powf(-0.25) * (-x * x / 2.0).exp();
            let mut amp = Cx::new(0.0, 0.0);
            for (n, cn) in c.iter().enumerate() {
                amp += cn * Cx::from_polar(1.0, -kappa * (n * n.saturating_sub(1)) as f64) * psi;
                let next = (2.0 / (n + 1) as f64).sqrt() * x * psi
                    - (n as f64 / (n + 1) as f64).sqrt() * psi_prev;
                psi_prev = psi;
                psi = next;
            }
            worst = worst.max((sampler.density_pure(&state, x) - amp.norm_sqr()).abs());
        }
        assert!(worst < 1e-12, "{} {} {}: {}", alpha, r, kappa, worst);
    }
    // distiratsua eta kerr ia gabe: X ~ N(√2 a, e^{-2r}/2)
    let (a, r) = (300.0, 0.5);
    let sampler = KerrSampler::new(1e-16);
    let state = Pure::lab(Cx::new(a, 0.0), r);
    let v = (-2.0 * r).exp() / 2.0;
    for dx in [-1.0, -0.3, 0.0, 0.2, 0.8] {
        let x = 2f64.sqrt() * a + dx;
        let want = (-dx * dx / (2.0 * v)).exp() / (2.0 * std::f64::consts::PI * v).sqrt();
        assert!(
            (sampler.density_pure(&state, x) - want).abs() < 1e-9,
            "{}",
            dx
        );
    }
}

// --------------------------------------------------------------------- fase-espazioa

// panelaren integrala eta batez besteko (x, p)
fn panel_moments(p: &light::Panel) -> (f64, f64, f64) {
    let (h, n) = (p.pixel(), p.n);
    let (mut tot, mut mx, mut mp) = (0.0, 0.0, 0.0);
    for row in 0..n {
        for col in 0..n {
            let w = p.w[row * n + col] * h * h;
            tot += w;
            mx += w * (p.center.0 - p.half + (col as f64 + 0.5) * h);
            mp += w * (p.center.1 + p.half - (row as f64 + 0.5) * h);
        }
    }
    (tot, mx / tot, mp / tot)
}

#[test]
fn phase_space_panels_hold_the_exact_state() {
    // (kerr, estutzea, galera, α): gaussiarra, kerr purua, kerr estutua, eta nahastea (galera + estutzea + kerr)
    for (kerr, sq, loss, alpha) in [
        (0.0, 0.0, 0.0, 30.0),
        (0.0, 6.0, 0.2, -12.0),
        (3.0, 0.0, 0.0, 40.0),
        (3.0, 6.0, 0.0, 25.0),
        (3.0, 6.0, 0.2, 30.0),
    ] {
        let light = Light {
            photons: 1e3,
            loss,
            squeeze_db: sq,
            kerr,
            mix: 0.0,
            detect: Detect::Homodyne,
        };
        let kappa = kerr / (2.0 * 1600.0);
        let sampler = KerrSampler::new(kappa);
        let p = light.phase_space(alpha, Some(&sampler), 96);
        let (_, _, r, s) = light.noise_params();
        let field = if kerr == 0.0 {
            Cx::new(alpha, 0.0)
        } else if s > 0.0 {
            kerr_mean_mixed(alpha, r, s, kappa)
        } else {
            kerr_mean_pure(Cx::new(alpha, 0.0), r, kappa)
        };
        let (tot, mx, mp) = panel_moments(&p);
        let want = (2f64.sqrt() * field.re, 2f64.sqrt() * field.im);
        println!(
            "{kerr} {sq} {loss} {alpha}: ∫W {tot:.10} ⟨x⟩ {mx:.8} ({:.8}) ⟨p⟩ {mp:.8} ({:.8}) nodes {}",
            want.0, want.1, p.nodes
        );
        assert!((tot - 1.0).abs() < 1e-4, "∫W = {}", tot);
        assert!((mx - want.0).abs() < 1e-3 && (mp - want.1).abs() < 1e-3);
    }
}

// W(x, p) definiziotik, trapezio fin batekin (erreferentzia)
fn wigner_direct(d: &[Cx], x: f64, p: f64) -> f64 {
    let psi = |z: f64| {
        let (mut prev, mut cur) = (0.0, std::f64::consts::PI.powf(-0.25) * (-z * z / 2.0).exp());
        let mut sum = Cx::new(0.0, 0.0);
        for (j, dj) in d.iter().enumerate() {
            sum += dj * cur;
            let next =
                (2.0 / (j + 1) as f64).sqrt() * z * cur - (j as f64 / (j + 1) as f64).sqrt() * prev;
            prev = cur;
            cur = next;
        }
        sum
    };
    let dy = 0.01;
    let mut acc = 0.0;
    for k in -1500..=1500 {
        let y = k as f64 * dy;
        acc += (psi(x + y).conj() * psi(x - y) * Cx::from_polar(1.0, 2.0 * p * y)).re;
    }
    acc * dy / std::f64::consts::PI
}

#[test]
fn wigner_pixels_are_pixel_averages_without_aliasing() {
    // |1>: W(0,0) = -1/π; eta superposizio bat, pixel handiekin (marra finak pixel bakoitzean batez bestekoa)
    let one = [Cx::new(0.0, 0.0), Cx::new(1.0, 0.0)];
    let mix: Vec<Cx> = (0..12)
        .map(|k| Cx::from_polar(1.0 / 12f64.sqrt(), 0.7 * k as f64))
        .collect();
    for (d, c, half, n) in [
        (&one[..], (0.0, 0.0), 3.5, 35),
        (&mix[..], (0.4, -0.3), 6.0, 16),
    ] {
        let w = light::wigner_expansion(d, c, half, n);
        let h = 2.0 * half / n as f64;
        let top = w.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut worst: f64 = 0.0;
        for (row, col) in [
            (n / 2, n / 2),
            (n / 2 - 1, n / 2),
            (3, 7),
            (n - 4, 2),
            (n / 3, n / 2 + 2),
        ] {
            // pixelaren batez bestekoa: gauss-legendre konposatua, 4 tarte × 4 puntu dimentsio bakoitzean
            let gl = [
                (-0.861_136_311_594_052_6, 0.347_854_845_137_453_8),
                (-0.339_981_043_584_856_3, 0.652_145_154_862_546_1),
                (0.339_981_043_584_856_3, 0.652_145_154_862_546_1),
                (0.861_136_311_594_052_6, 0.347_854_845_137_453_8),
            ];
            let nodes: Vec<(f64, f64)> = (0..4)
                .flat_map(|t| {
                    gl.iter()
                        .map(move |&(u, w)| ((t as f64 + 0.5 + u / 2.0) / 4.0, w / 8.0))
                })
                .collect();
            let mut avg = 0.0;
            for &(fx, wx) in &nodes {
                for &(fp, wp) in &nodes {
                    let x = c.0 - half + (col as f64 + fx) * h;
                    let p = c.1 + half - (row as f64 + fp) * h;
                    avg += wx * wp * wigner_direct(d, x, p);
                }
            }
            worst = worst.max((w[row * n + col] - avg).abs());
        }
        println!(
            "{} modes, pixel {h:.3}: worst {:.2e} of max {:.3}",
            d.len(),
            worst,
            top
        );
        assert!(worst < 1e-5 * top, "{} {}", worst, top);
    }
    let w = light::wigner_expansion(&one, (0.0, 0.0), 3.5, 35);
    assert!((w[17 * 35 + 17] + 1.0 / std::f64::consts::PI).abs() < 0.01);
}

#[test]
fn strong_kerr_states_have_negative_wigner_regions() {
    let light = Light {
        kerr: 8.0,
        ..Light::default()
    };
    let kappa = 8.0 / (2.0 * 2e5);
    let p = light.phase_space(2e5f64.sqrt(), Some(&KerrSampler::new(kappa)), 128);
    let max = p.w.iter().cloned().fold(f64::MIN, f64::max);
    let min = p.w.iter().cloned().fold(f64::MAX, f64::min);
    assert!(min < -0.05 * max, "{} {}", min, max);
}
