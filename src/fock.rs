// fitxategi honek eragile optikoak kalkulatzen ditu fotoi-kopuruaren oinarrian (fock oinarria).
//
// modu bakoitzak 0..d-1 fotoi gordetzen ditu (d = muga). fotoiak sortzen dituzten eragileak
// (desplazamendua, estutzea) espazio handiago batean kalkulatzen dira eta gero d x d-ra mozten dira,
// gordetako elementuak zehatzak izan daitezen; mugaz kanpora bultzatzen duten probabilitatea
// simulagailuak neurtzen du eta trunkatze-ihes gisa jakinarazten da, ezkutatu gabe.
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

// desplazamendua D(α) = exp(α a† − α* a), d x d bloke gisa
pub fn displacement(alpha: C, d: usize) -> Mat {
    let start = (4.0 * alpha.norm_sqr()) as usize + 8;
    padded_exp(d, start, |n| {
        let ad = creation(n);
        let a = annihilation(n);
        let mut g = Mat::zeros(n);
        for i in 0..n * n {
            g.a[i] = alpha * ad.a[i] - alpha.conj() * a.a[i];
        }
        g
    })
}

// estutzea S(z) = exp((z* a² − z a†²) / 2), z = r e^{iφ}, d x d bloke gisa.
// φ = 0 eta r > 0 direnean x koadratura estutzen da.
pub fn squeezing(r: f64, phi: f64, d: usize) -> Mat {
    let z = C::from_polar(r, phi);
    let start = (8.0 * r.sinh().powi(2)) as usize + 16;
    padded_exp(d, start, |n| {
        let a = annihilation(n);
        let ad = creation(n);
        let a2 = a.mul(&a);
        let ad2 = ad.mul(&ad);
        let mut g = Mat::zeros(n);
        for i in 0..n * n {
            g.a[i] = (z.conj() * a2.a[i] - z * ad2.a[i]) * 0.5;
        }
        g
    })
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
