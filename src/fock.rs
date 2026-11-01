// fitxategi honek eragile optikoak kalkulatzen ditu fotoi-kopuruaren oinarrian (fock oinarria).
//
// modu bakoitzak 0..d-1 fotoi gordetzen ditu (d = muga). fotoiak sortzen dituzten eragileen (desplazamendua,
// estutzea) d x d elementuak zehatzak dira: eragile infinituaren elementuak dira, ez eragile moztu baten
// esponentziala. horretarako ordena normaleko formak erabiltzen dira (a† ezkerrean, a eskuinean): bitarteko
// batura d-ren barruko egoeretan bakarrik ibiltzen da, beraz mozteak ez du ezer aldatzen. mugaz kanpora
// bultzatzen duten probabilitatea simulagailuak neurtzen du eta trunkatze-ihes gisa jakinarazten da.
// izpi-zatitzaileak fotoi-kopuru osoa mantentzen du, beraz zehazki kalkulatzen da, fotoi-kopuru
// osoaren azpiespazio bakoitzean bereiz.

use num_complex::Complex64 as C;

// matrize konplexu karratu trinkoa, errenkadaka gordeta
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub n: usize,
    pub a: Vec<C>,
}

impl Mat {
    pub fn zeros(n: usize) -> Mat {
        Mat {
            n,
            a: vec![C::new(0.0, 0.0); n * n],
        }
    }

    pub fn identity(n: usize) -> Mat {
        let mut m = Mat::zeros(n);
        for i in 0..n {
            m.a[i * n + i] = C::new(1.0, 0.0);
        }
        m
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> C {
        self.a[r * self.n + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: C) {
        self.a[r * self.n + c] = v;
    }

    pub fn mul(&self, other: &Mat) -> Mat {
        let n = self.n;
        let mut out = Mat::zeros(n);
        for i in 0..n {
            for k in 0..n {
                let x = self.a[i * n + k];
                if x == C::new(0.0, 0.0) {
                    continue;
                }
                for j in 0..n {
                    out.a[i * n + j] += x * other.a[k * n + j];
                }
            }
        }
        out
    }

    pub fn scale(&self, s: f64) -> Mat {
        Mat {
            n: self.n,
            a: self.a.iter().map(|x| x * s).collect(),
        }
    }

    pub fn add(&self, other: &Mat) -> Mat {
        Mat {
            n: self.n,
            a: self.a.iter().zip(&other.a).map(|(x, y)| x + y).collect(),
        }
    }

    // errenkaden batura absolutu handiena (infinitu-araua)
    pub fn norm_inf(&self) -> f64 {
        (0..self.n)
            .map(|i| (0..self.n).map(|j| self.get(i, j).norm()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    // goiko ezkerreko d x d blokea
    pub fn block(&self, d: usize) -> Mat {
        let mut m = Mat::zeros(d);
        for i in 0..d {
            for j in 0..d {
                m.set(i, j, self.get(i, j));
            }
        }
        m
    }

    pub fn max_abs_diff(&self, other: &Mat) -> f64 {
        self.a
            .iter()
            .zip(&other.a)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max)
    }

    // matrize-esponentziala, eskalatu eta karratura jasoz, taylor serie batekin
    pub fn expm(&self) -> Mat {
        let norm = self.norm_inf();
        // araua 1/2 azpitik jarri, han taylor-en 20 gaiak makina-doitasunera iristen dira
        let mut squarings = 0u32;
        let mut s = 1.0;
        while norm * s > 0.5 {
            s *= 0.5;
            squarings += 1;
        }
        let a = self.scale(s);
        let mut result = Mat::identity(self.n);
        let mut term = Mat::identity(self.n);
        for k in 1..=20 {
            term = term.mul(&a).scale(1.0 / k as f64);
            result = result.add(&term);
        }
        for _ in 0..squarings {
            result = result.mul(&result);
        }
        result
    }
}

// a deuseztatze-eragilea 0..n-1 fotoietan: a|k> = sqrt(k)|k-1>
pub fn annihilation(n: usize) -> Mat {
    let mut m = Mat::zeros(n);
    for k in 1..n {
        m.set(k - 1, k, C::new((k as f64).sqrt(), 0.0));
    }
    m
}

// a† sortze-eragilea 0..n-1 fotoietan, moztuta (a†|n-1> galtzen da)
pub fn creation(n: usize) -> Mat {
    let mut m = Mat::zeros(n);
    for k in 0..n - 1 {
        m.set(k + 1, k, C::new(((k + 1) as f64).sqrt(), 0.0));
    }
    m
}

// desplazamendua D(α) = exp(α a† − α* a), d x d bloke gisa. ordena normalean
// D(α) = e^(-|α|²/2) e^(α a†) e^(-α* a), eta elementuak laguerre-ren polinomioak dira:
//   ⟨m|D|n⟩ = e^(-|α|²/2) √(n!/m!) α^(m-n) L_n^(m-n)(|α|²)            m >= n denean
//   ⟨m|D|n⟩ = e^(-|α|²/2) √(m!/n!) (-α*)^(n-m) L_m^(n-m)(|α|²)        m < n denean
pub fn displacement(alpha: C, d: usize) -> Mat {
    let x = alpha.norm_sqr();
    let mut out = Mat::zeros(d);
    if x == 0.0 {
        return Mat::identity(d);
    }
    let ln_abs = alpha.norm().ln();
    let ln_fact = ln_factorials(d);
    // L_j^(k)(x), j = 0..d, k bakoitzeko, aurreranzko errekurtsioarekin
    for k in 0..d {
        let mut lag = vec![0.0; d];
        lag[0] = 1.0;
        if d > 1 {
            lag[1] = 1.0 + k as f64 - x;
        }
        for j in 1..d.saturating_sub(1) {
            lag[j + 1] = ((2 * j + 1 + k) as f64 - x) * lag[j] / (j + 1) as f64
                - (j + k) as f64 * lag[j - 1] / (j + 1) as f64;
        }
        for (j, l) in lag.iter().enumerate() {
            if j + k >= d {
                break;
            }
            let (lo, hi) = (j, j + k);
            // √(lo!/hi!) |α|^k e^(-x/2), log-ean gainezkatzerik ez izateko
            let mag = (0.5 * (ln_fact[lo] - ln_fact[hi]) + k as f64 * ln_abs - x / 2.0).exp();
            // m >= n: α^k (m = hi, n = lo); m < n: (-α*)^k (m = lo, n = hi)
            let phase_down = C::from_polar(1.0, k as f64 * alpha.arg());
            let phase_up = C::from_polar(1.0, k as f64 * (-alpha.conj()).arg());
            out.set(hi, lo, phase_down * (mag * l));
            if k > 0 {
                out.set(lo, hi, phase_up * (mag * l));
            }
        }
    }
    out
}

// estutzea S(z) = exp((z* a² − z a†²) / 2), z = r e^{iφ}, d x d bloke gisa.
// φ = 0 eta r > 0 direnean x koadratura estutzen da. ordena normalean (t = e^{iφ} tanh r):
//   S(z) = exp(-t a†² / 2) (sech r)^(a†a + 1/2) exp(t* a² / 2)
// ⟨m|S|n⟩ = Σ_k ⟨m|e^(-t a†²/2)|k⟩ (sech r)^(k + 1/2) ⟨k|e^(t* a²/2)|n⟩, k <= min(m, n) eta paritate berekoa.
// baturaren gaiek zeinu txandakatuak dituzte; fotoi-kopuru handietan (m, n > 40 inguru, r txikiekin) gai
// handiak elkar ezeztatzen dira eta zehaztasuna galtzen da. gai handiena 1e4 baino handiagoa bada (1e-10
// baino errore handiagoa), sortzailearen esponentziala erabiltzen da espazio handiago batean
pub fn squeezing(r: f64, phi: f64, d: usize) -> Mat {
    if r == 0.0 {
        return Mat::identity(d);
    }
    // r < 0 -> |r| eta φ + π (z bera)
    let (r, phi) = if r < 0.0 {
        (-r, phi + std::f64::consts::PI)
    } else {
        (r, phi)
    };
    let (th, ch) = (r.tanh(), r.cosh());
    let ln_half_t = (th / 2.0).ln();
    let ln_fact = ln_factorials(d);
    // ⟨m|e^(c a†²)|k⟩ = c^j / j! √(m!/k!), m = k + 2j; hemen |c| = tanh r / 2 eta fasea bereiz
    let ln_raise = |m: usize, k: usize| -> f64 {
        let j = (m - k) / 2;
        j as f64 * ln_half_t - ln_fact[j] + 0.5 * (ln_fact[m] - ln_fact[k])
    };
    let mut out = Mat::zeros(d);
    let mut largest: f64 = 0.0;
    for m in 0..d {
        for n in (m % 2..d).step_by(2) {
            let mut acc = C::new(0.0, 0.0);
            for k in (m % 2..=m.min(n)).step_by(2) {
                let (jm, jn) = ((m - k) / 2, (n - k) / 2);
                let term = (ln_raise(m, k) + ln_raise(n, k) - (k as f64 + 0.5) * ch.ln()).exp();
                largest = largest.max(term);
                // faseak: (-e^{iφ})^jm eta (e^{-iφ})^jn
                let sign = if jm % 2 == 0 { 1.0 } else { -1.0 };
                acc += C::from_polar(sign * term, phi * (jm as f64 - jn as f64));
            }
            out.set(m, n, acc);
        }
    }
    if largest <= 1e4 {
        return out;
    }
    let z = C::from_polar(r, phi);
    padded_exp(d, (8.0 * r.sinh().powi(2)) as usize + 16, |n| {
        let (a, ad) = (annihilation(n), creation(n));
        let (a2, ad2) = (a.mul(&a), ad.mul(&ad));
        let mut g = Mat::zeros(n);
        for i in 0..n * n {
            g.a[i] = (z.conj() * a2.a[i] - z * ad2.a[i]) * 0.5;
        }
        g
    })
}

// exp(sortzailea), betegarria handituz gordetako d x d blokea egonkortu arte
fn padded_exp(d: usize, start_pad: usize, generator: impl Fn(usize) -> Mat) -> Mat {
    let mut pad = start_pad.max(16);
    let mut prev = generator(d + pad).expm().block(d);
    loop {
        pad *= 2;
        let next = generator(d + pad).expm().block(d);
        if next.max_abs_diff(&prev) < 1e-12 || pad >= 512 {
            return next;
        }
        prev = next;
    }
}

// izpi-zatitzailea B(θ, φ) = exp(θ (e^{iφ} a b† − e^{−iφ} a† b)) bi modutan, d² x d² matrize gisa,
// (n_a * d + n_b) indizearekin. θ = π/4 izpi-zatitzaile 50:50 bat da.
//
// zehatza: eragileak n_a + n_b mantentzen du, beraz fotoi-kopuru osoko N azpiespazio bakoitzaren
// barruan esponentziatzen da. irteeran modu batek d fotoi edo gehiago beharko lituzkeen sarreren
// anplitudea galtzen da; simulagailuak galera hori trunkatze-ihes gisa neurtzen du.
pub fn beamsplitter(theta: f64, phi: f64, d: usize) -> Mat {
    let mut out = Mat::zeros(d * d);
    let e = C::from_polar(1.0, phi);
    for total in 0..=(2 * (d - 1)) {
        // oinarria |k, total-k>, k = 0..=total
        let m = total + 1;
        let mut g = Mat::zeros(m);
        for k in 0..m {
            let nb = total - k;
            // a b† |k, nb> = sqrt(k) sqrt(nb+1) |k-1, nb+1>
            if k >= 1 {
                let amp = (k as f64).sqrt() * ((nb + 1) as f64).sqrt();
                g.set(k - 1, k, g.get(k - 1, k) + e * amp * theta);
            }
            // a† b |k, nb> = sqrt(k+1) sqrt(nb) |k+1, nb-1>
            if nb >= 1 {
                let amp = ((k + 1) as f64).sqrt() * (nb as f64).sqrt();
                g.set(k + 1, k, g.get(k + 1, k) - e.conj() * amp * theta);
            }
        }
        let u = g.expm();
        for kin in 0..m {
            let nb_in = total - kin;
            if kin >= d || nb_in >= d {
                continue;
            }
            for kout in 0..m {
                let nb_out = total - kout;
                if kout >= d || nb_out >= d {
                    continue;
                }
                out.set(kout * d + nb_out, kin * d + nb_in, u.get(kout, kin));
            }
        }
    }
    out
}

// ln(n k-ka), fotoi-galeraren kanalerako
pub fn ln_choose(n: u32, k: u32) -> f64 {
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

pub fn ln_factorial(n: u32) -> f64 {
    (2..=n).map(|i| (i as f64).ln()).sum()
}

// ln(k!) k = 0..n guztietarako, O(n)-n: ln_factorial-ek ematen dituen bit berberak (batura bera, ordena
// berean), baina bakoitza ez da hutsetik kalkulatzen. muga handiko moduetan eragileak O(d³) izan
// beharrean O(d²) eraikitzen dira
pub fn ln_factorials(n: usize) -> Vec<f64> {
    let mut acc = 0.0f64;
    (0..n)
        .map(|k| {
            if k >= 2 {
                acc += (k as f64).ln();
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // fotoi-kopuru baxuen zutabeak normalizatuta eta ortogonalak izan behar dira
    fn is_unitary_on_low_states(m: &Mat, keep: usize) -> f64 {
        let mut worst: f64 = 0.0;
        for i in 0..keep {
            for j in 0..keep {
                let mut dot = C::new(0.0, 0.0);
                for r in 0..m.n {
                    dot += m.get(r, i).conj() * m.get(r, j);
                }
                let target = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((dot - C::new(target, 0.0)).norm());
            }
        }
        worst
    }

    // erreferentzia: sortzailearen esponentziala espazio handiago batean, d x d-ra moztuta (lehengo metodoa)
    fn padded(d: usize, generator: impl Fn(usize) -> Mat) -> Mat {
        generator(d + 160).expm().block(d)
    }

    fn displacement_by_expm(alpha: C, d: usize) -> Mat {
        padded(d, |n| {
            let (ad, a) = (creation(n), annihilation(n));
            let mut g = Mat::zeros(n);
            for i in 0..n * n {
                g.a[i] = alpha * ad.a[i] - alpha.conj() * a.a[i];
            }
            g
        })
    }

    fn squeezing_by_expm(r: f64, phi: f64, d: usize) -> Mat {
        let z = C::from_polar(r, phi);
        padded(d, |n| {
            let (a, ad) = (annihilation(n), creation(n));
            let (a2, ad2) = (a.mul(&a), ad.mul(&ad));
            let mut g = Mat::zeros(n);
            for i in 0..n * n {
                g.a[i] = (z.conj() * a2.a[i] - z * ad2.a[i]) * 0.5;
            }
            g
        })
    }

    #[test]
    fn closed_forms_match_the_exponential_of_the_generator() {
        for alpha in [
            C::new(0.3, 0.0),
            C::new(1.1, -0.4),
            C::new(-2.0, 1.5),
            C::new(0.0, 3.0),
        ] {
            let d = 30;
            let diff = displacement(alpha, d).max_abs_diff(&displacement_by_expm(alpha, d));
            assert!(diff < 1e-10, "D({}) {}", alpha, diff);
        }
        for (r, phi) in [(0.2, 0.0), (0.6, 1.0), (1.2, -2.0), (-0.5, 0.3)] {
            let d = 30;
            let diff = squeezing(r, phi, d).max_abs_diff(&squeezing_by_expm(r, phi, d));
            assert!(diff < 1e-10, "S({}, {}) {}", r, phi, diff);
        }
    }

    #[test]
    fn large_cutoffs_stay_exact_and_unitary() {
        // muga handiekin ere elementuak zehatzak dira (betegarririk gabe). zutabe baxuen buztana mugaren
        // azpitik geratzen da, beraz zutabe horiek unitarioak dira
        let d = displacement(C::new(2.5, -1.0), 90);
        assert!(is_unitary_on_low_states(&d, 30) < 1e-10);
        assert!(
            displacement(C::new(4.0, 2.0), 70)
                .max_abs_diff(&displacement_by_expm(C::new(4.0, 2.0), 70))
                < 1e-12
        );
        // estutze handia muga txikian (forma itxia) eta txikia muga handian (esponentzialera itzultzen da)
        // r = 2: erreferentzia esponentzialak ez du konbergitzen (fotoiak ehunka dira), beraz zutabe analitikoak:
        //   ⟨2j|S|0⟩ = (-e^{iφ} tanh r)^j √((2j)!) / (2^j j!) / √(cosh r)
        //   ⟨2j+1|S|1⟩ = (-e^{iφ} tanh r)^j √((2j+1)!) / (2^j j!) / (cosh r)^(3/2)
        let (r, phi, d) = (2.0f64, 0.4, 24);
        let s = squeezing(r, phi, d);
        for j in 0..d / 2 {
            let base = C::from_polar(r.tanh(), phi) * -1.0;
            let common =
                (0.5 * ln_factorial(2 * j as u32) - j as f64 * 2f64.ln() - ln_factorial(j as u32))
                    .exp();
            let want0 = base.powu(j as u32) * common / r.cosh().sqrt();
            let want1 =
                base.powu(j as u32) * common * ((2 * j + 1) as f64).sqrt() / r.cosh().powf(1.5);
            assert!((s.get(2 * j, 0) - want0).norm() < 1e-12, "j={}", j);
            assert!((s.get(2 * j + 1, 1) - want1).norm() < 1e-12, "j={}", j);
        }
        let s = squeezing(0.5, 0.0, 70);
        assert!(s.max_abs_diff(&squeezing_by_expm(0.5, 0.0, 70)) < 1e-10);
        assert!((s.get(0, 0).norm_sqr() - 1.0 / 0.5f64.cosh()).abs() < 1e-12);
    }

    #[test]
    fn expm_of_zero_is_identity() {
        let z = Mat::zeros(5);
        assert!(z.expm().max_abs_diff(&Mat::identity(5)) < 1e-15);
    }

    #[test]
    fn displacement_of_vacuum_is_coherent_state() {
        let alpha = C::new(1.1, -0.4);
        let d = 30;
        let m = displacement(alpha, d);
        let n2 = alpha.norm_sqr();
        for n in 0..12 {
            let p = m.get(n, 0).norm_sqr();
            let expected =
                (-n2).exp() * n2.powi(n as i32) / (1..=n).map(|k| k as f64).product::<f64>();
            assert!(
                (p - expected).abs() < 1e-12,
                "n={} p={} expected={}",
                n,
                p,
                expected
            );
        }
        assert!(is_unitary_on_low_states(&m, 8) < 1e-10);
    }

    #[test]
    fn squeezed_vacuum_has_only_even_photons() {
        let r = 0.6;
        let m = squeezing(r, 0.0, 30);
        let p0 = m.get(0, 0).norm_sqr();
        assert!((p0 - 1.0 / r.cosh()).abs() < 1e-12);
        for n in (1..20).step_by(2) {
            assert!(m.get(n, 0).norm() < 1e-14);
        }
        // P(2) = tanh² r / (2 cosh r)
        let p2 = m.get(2, 0).norm_sqr();
        assert!((p2 - r.tanh().powi(2) / (2.0 * r.cosh())).abs() < 1e-12);
    }

    #[test]
    fn beamsplitter_shows_hong_ou_mandel_dip() {
        let d = 4;
        let b = beamsplitter(std::f64::consts::FRAC_PI_4, 0.0, d);
        // |1,1> sarrera = 1*d + 1 indizea
        let col = d + 1;
        let p11 = b.get(d + 1, col).norm_sqr();
        let p20 = b.get(2 * d, col).norm_sqr();
        let p02 = b.get(2, col).norm_sqr();
        assert!(p11 < 1e-15);
        assert!((p20 - 0.5).abs() < 1e-12);
        assert!((p02 - 0.5).abs() < 1e-12);
    }
}
