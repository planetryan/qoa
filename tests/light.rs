// fitxategi honek src/light.rs egiaztatzen du: mdct-ak seinalea zehazki berreraikitzen duela eta energia
// gordetzen duela (modu ortonormalak), efekturik gabe argiak seinalea bera itzultzen duela, homodinoaren
// zarata hutsaren (eta estutze eta galeren) bariantza zehatza dela, kerr-en batez besteko eremua fock
// oinarriko batura zuzenarekin bat datorrela, poisson laginketa zehatza dela, konbinazio ez-zehatzak
// erroreak direla, eta hazi berak emaitza bera ematen duela hari-kopurua edozein dela ere.

use qoa::light::{self, Detect, Light, Mdct, kerr_mean, normal_pair, poisson, poisson_log_pmf};
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
fn combinations_without_an_exact_computation_are_errors() {
    let bad = [
        Light {
            kerr: 1.0,
            ..Light::default()
        },
        Light {
            squeeze_db: 3.0,
            detect: Detect::Mean,
            ..Light::default()
        },
        Light {
            squeeze_db: 3.0,
            detect: Detect::Count,
            ..Light::default()
        },
        Light {
            loss: 1.0,
            ..Light::default()
        },
        Light {
            photons: 0.0,
            ..Light::default()
        },
    ];
    for l in bad {
        assert!(l.check().is_err(), "{:?}", l);
    }
    for detect in [Detect::Mean, Detect::Count] {
        assert!(
            Light {
                kerr: 2.0,
                detect,
                ..Light::default()
            }
            .check()
            .is_ok()
        );
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
