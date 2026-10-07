// fitxategi honek modu optikoen kernelak (src/state.rs, "moduen kernelak") egiaztatzen ditu: eragiketa
// bakoitza ausazko egoeretan, inplementazio zuzen eta motel baten aurka (anplitude bakoitza bere indizetik,
// lehengo bertsioa). qubit-kopuru, modu-kopuru eta muga desberdinekin, paraleloko bideak barne.

use num_complex::Complex64 as C;
use qoa::fock::{self, Mat};
use qoa::state::State;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

struct Shape {
    nq: usize,
    modes: usize,
    d: usize,
}

impl Shape {
    fn len(&self) -> usize {
        (1 << self.nq) * self.d.pow(self.modes as u32)
    }
    fn stride(&self, o: usize) -> usize {
        (1 << self.nq) * self.d.pow(o as u32)
    }
    fn n(&self, i: usize, o: usize) -> usize {
        (i / self.stride(o)) % self.d
    }
}

fn random_state(rng: &mut StdRng, len: usize) -> Vec<C> {
    let mut v: Vec<C> = (0..len)
        .map(|_| C::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
        .collect();
    let norm: f64 = v.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    v.iter_mut().for_each(|a| *a /= norm);
    v
}

fn random_mat(rng: &mut StdRng, n: usize) -> Mat {
    let mut m = Mat::zeros(n);
    for x in m.a.iter_mut() {
        // matrize sakabanatuak ere bai: % 40 zero
        if rng.random_bool(0.6) {
            *x = C::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
        }
    }
    m
}

fn renormalize(v: &mut [C]) -> f64 {
    let norm: f64 = v.iter().map(|a| a.norm_sqr()).sum();
    let lost = (1.0 - norm).max(0.0);
    if lost > 0.0 && norm > 0.0 {
        let s = 1.0 / norm.sqrt();
        v.iter_mut().for_each(|a| *a *= s);
    }
    lost
}

// ---- erreferentziak: lehengo inplementazioak, indizez indize

fn ref_mode_matrix(sh: &Shape, v: &mut [C], o: usize, m: &Mat) -> f64 {
    let (d, s) = (sh.d, sh.stride(o));
    for chunk in v.chunks_mut(s * d) {
        for j in 0..s {
            let x: Vec<C> = (0..d).map(|k| chunk[j + k * s]).collect();
            for r in 0..d {
                chunk[j + r * s] = (0..d).map(|c| m.a[r * d + c] * x[c]).sum();
            }
        }
    }
    renormalize(v)
}

fn ref_two_mode_matrix(sh: &Shape, v: &mut [C], a: usize, b: usize, m: &Mat) -> f64 {
    let d = sh.d;
    let (sa, sb) = (sh.stride(a), sh.stride(b));
    for base in 0..v.len() {
        if sh.n(base, a) != 0 || sh.n(base, b) != 0 {
            continue;
        }
        let x: Vec<C> = (0..d * d)
            .map(|k| v[base + (k / d) * sa + (k % d) * sb])
            .collect();
        for r in 0..d * d {
            v[base + (r / d) * sa + (r % d) * sb] =
                (0..d * d).map(|c| m.a[r * d * d + c] * x[c]).sum();
        }
    }
    renormalize(v)
}

fn ref_emit(sh: &Shape, v: &mut [C], q: usize, o: usize) {
    let bit = 1 << q;
    for i in 0..v.len() {
        if i & bit != 0 && sh.n(i, o) == 0 {
            v.swap(i, (i & !bit) + sh.stride(o));
        }
    }
}

fn ref_swap_modes(sh: &Shape, v: &mut [C], a: usize, b: usize) {
    let (sa, sb) = (sh.stride(a), sh.stride(b));
    for i in 0..v.len() {
        let (na, nb) = (sh.n(i, a), sh.n(i, b));
        if na < nb {
            v.swap(i, i - na * sa - nb * sb + nb * sa + na * sb);
        }
    }
}

fn ref_ladder(sh: &Shape, v: &mut [C], o: usize, up: bool) {
    let (d, s) = (sh.d, sh.stride(o));
    for chunk in v.chunks_mut(s * d) {
        for j in 0..s {
            let x: Vec<C> = (0..d).map(|n| chunk[j + n * s]).collect();
            for n in 0..d {
                chunk[j + n * s] = if up {
                    if n == 0 {
                        C::new(0.0, 0.0)
                    } else {
                        x[n - 1] * (n as f64).sqrt()
                    }
                } else if n + 1 < d {
                    x[n + 1] * ((n + 1) as f64).sqrt()
                } else {
                    C::new(0.0, 0.0)
                };
            }
        }
    }
}

fn close(a: &[C], b: &[C]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

fn shapes() -> Vec<Shape> {
    let mut v = Vec::new();
    for nq in [0usize, 1, 2, 3] {
        for (modes, d) in [(1, 2), (1, 5), (2, 3), (2, 4), (3, 3)] {
            v.push(Shape { nq, modes, d });
        }
    }
    // paraleloko bideak (2^14 anplitude baino gehiago): xafla luzeak eta talde asko
    v.push(Shape {
        nq: 12,
        modes: 2,
        d: 3,
    });
    v.push(Shape {
        nq: 4,
        modes: 3,
        d: 12,
    });
    v.push(Shape {
        nq: 14,
        modes: 1,
        d: 4,
    });
    v
}

#[test]
fn optical_kernels_match_the_reference() {
    let mut rng = StdRng::seed_from_u64(21);
    for sh in shapes() {
        let len = sh.len();
        for _ in 0..3 {
            let start = random_state(&mut rng, len);
            let state =
                |v: &[C]| State::from_amplitudes(sh.nq, sh.modes, sh.d, v.to_vec()).unwrap();
            let o = rng.random_range(0..sh.modes);
            let tag = format!("nq={} modes={} d={} o={}", sh.nq, sh.modes, sh.d, o);

            // modu bateko matrizea: desplazamendua eta ausazkoa (sakabanatua)
            for m in [
                fock::displacement(C::new(0.4, -0.2), sh.d),
                random_mat(&mut rng, sh.d),
            ] {
                let mut want = start.clone();
                let lw = ref_mode_matrix(&sh, &mut want, o, &m);
                let mut s = state(&start);
                let lg = s.apply_mode_matrix(o, &m);
                assert!(
                    close(s.amplitudes(), &want) < 1e-12 && (lg - lw).abs() < 1e-12,
                    "matrizea {}",
                    tag
                );
            }

            // diagonala, banaketa, kolapsoa
            let f: Vec<C> = (0..sh.d)
                .map(|n| C::from_polar(1.0 + n as f64 * 0.1, n as f64))
                .collect();
            let mut want = start.clone();
            for (i, a) in want.iter_mut().enumerate() {
                *a *= f[sh.n(i, o)];
            }
            let mut s = state(&start);
            s.apply_mode_diagonal(o, &f);
            assert!(close(s.amplitudes(), &want) < 1e-12, "diagonala {}", tag);

            let mut dist = vec![0.0; sh.d];
            for (i, a) in start.iter().enumerate() {
                dist[sh.n(i, o)] += a.norm_sqr();
            }
            let got = state(&start).photon_distribution(o);
            assert!(
                got.iter().zip(&dist).all(|(x, y)| (x - y).abs() < 1e-12),
                "banaketa {}",
                tag
            );

            let keep = |n: usize| n % 2 == 1;
            let p: f64 = dist
                .iter()
                .enumerate()
                .filter(|(n, _)| keep(*n))
                .map(|(_, x)| x)
                .sum();
            let mut want = start.clone();
            for (i, a) in want.iter_mut().enumerate() {
                *a = if keep(sh.n(i, o)) {
                    *a / p.sqrt()
                } else {
                    C::new(0.0, 0.0)
                };
            }
            let mut s = state(&start);
            s.collapse_mode(o, p, keep);
            assert!(close(s.amplitudes(), &want) < 1e-12, "kolapsoa {}", tag);

            // eskailera-eragileak
            for up in [true, false] {
                let mut want = start.clone();
                ref_ladder(&sh, &mut want, o, up);
                let mut s = state(&start);
                s.ladder(o, up);
                assert!(
                    close(s.amplitudes(), &want) < 1e-12,
                    "eskailera {} {}",
                    up,
                    tag
                );
            }

            // fotoiak mugitu (permutazioa)
            if sh.d >= 3 {
                let mut want = start.clone();
                let so = sh.stride(o);
                for i in 0..want.len() {
                    if sh.n(i, o) == 0 {
                        want.swap(i, i + 2 * so);
                    }
                }
                let mut s = state(&start);
                s.move_photons(o, 0, 2);
                assert!(close(s.amplitudes(), &want) < 1e-15, "mugitu {}", tag);
            }

            // qubit-modu eragiketak
            if sh.nq > 0 {
                let q = rng.random_range(0..sh.nq);
                let mut want = start.clone();
                ref_emit(&sh, &mut want, q, o);
                let mut s = state(&start);
                s.emit(q, o);
                assert!(close(s.amplitudes(), &want) < 1e-15, "emit q{} {}", q, tag);

                let mut want = start.clone();
                for (i, a) in want.iter_mut().enumerate() {
                    if i >> q & 1 == 1 {
                        *a *= C::from_polar(1.0, 0.7 * sh.n(i, o) as f64);
                    }
                }
                let mut s = state(&start);
                s.dispersive(q, o, 0.7);
                assert!(
                    close(s.amplitudes(), &want) < 1e-12,
                    "dispertsiboa q{} {}",
                    q,
                    tag
                );
            }

            // bi modu: izpi-zatitzailea, ausazko matrizea, trukaketa, kerr gurutzatua (bi ordenetan)
            if sh.modes >= 2 {
                let (a, b) = if rng.random_bool(0.5) {
                    (0, sh.modes - 1)
                } else {
                    (sh.modes - 1, 0)
                };
                let tag = format!("{} a={} b={}", tag, a, b);
                for m in [
                    fock::beamsplitter(0.6, 0.3, sh.d),
                    random_mat(&mut rng, sh.d * sh.d),
                ] {
                    let mut want = start.clone();
                    let lw = ref_two_mode_matrix(&sh, &mut want, a, b, &m);
                    let mut s = state(&start);
                    let lg = s.apply_two_mode_matrix(a, b, &m);
                    assert!(
                        close(s.amplitudes(), &want) < 1e-12 && (lg - lw).abs() < 1e-12,
                        "bi modu {}",
                        tag
                    );
                }
                let mut want = start.clone();
                ref_swap_modes(&sh, &mut want, a, b);
                let mut s = state(&start);
                s.swap_modes(a, b);
                assert!(close(s.amplitudes(), &want) < 1e-15, "trukaketa {}", tag);

                let f: Vec<C> = (0..sh.d * sh.d)
                    .map(|k| C::from_polar(1.0, k as f64 * 0.3))
                    .collect();
                let mut want = start.clone();
                for (i, x) in want.iter_mut().enumerate() {
                    *x *= f[sh.n(i, a) * sh.d + sh.n(i, b)];
                }
                let mut s = state(&start);
                s.apply_two_mode_diagonal(a, b, &f);
                assert!(
                    close(s.amplitudes(), &want) < 1e-12,
                    "kerr gurutzatua {}",
                    tag
                );
            }
        }
    }
}

#[test]
fn photon_loss_matches_the_kraus_operator() {
    // k fotoi galtzea: |n> -> sqrt(C(n,k) γ^k (1-γ)^(n-k)) |n-k>, normalizatuta
    let mut rng = StdRng::seed_from_u64(22);
    for sh in [
        Shape {
            nq: 1,
            modes: 2,
            d: 5,
        },
        Shape {
            nq: 12,
            modes: 2,
            d: 4,
        },
    ] {
        let start = random_state(&mut rng, sh.len());
        let o = 1;
        let gamma: f64 = 0.3;
        let mut s = State::from_amplitudes(sh.nq, sh.modes, sh.d, start.clone()).unwrap();
        let k = s.photon_loss(o, gamma, 0.5);
        let choose =
            |n: usize, k: usize| (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64);
        let mut want = vec![C::new(0.0, 0.0); start.len()];
        for (i, a) in start.iter().enumerate() {
            let n = sh.n(i, o);
            if n >= k {
                let w = choose(n, k) * gamma.powi(k as i32) * (1.0 - gamma).powi((n - k) as i32);
                want[i - k * sh.stride(o)] = a * w.sqrt();
            }
        }
        renormalize(&mut want);
        let norm: f64 = want.iter().map(|a| a.norm_sqr()).sum();
        want.iter_mut().for_each(|a| *a /= norm.sqrt());
        assert!(
            close(s.amplitudes(), &want) < 1e-12,
            "galera k={} nq={}",
            k,
            sh.nq
        );
    }
}
