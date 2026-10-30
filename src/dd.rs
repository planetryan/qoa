// fitxategi honek batura zehatzak egiten ditu double-double aritmetikarekin: zenbaki bat bi f64-ren batura
// gisa (hi + lo, ~106 bit esanguratsu), errorerik gabeko transformazioekin (two_sum, two_prod). egoeraren
// baturetarako da (araua, neurketa-probabilitateak, laginketa-banaketa, itxarondako balioak): 2^n gai
// batzean f64 arruntaren biribiltze-errorea n-rekin hazten da, eta hemen ez da f64 baten zehaztasunetik
// pasatzen. anplitudeak berak f64 dira (kernelek FMA erabiltzen dute); hardware-ak ez du f128-rik
// (x86 eta arm-en ez dago), eta zehaztasun gehigarria behar den lekuetan bakarrik ordaintzen da.
//
// FMA-rik gabeko prozesadoreetan, two_prod-ek dekker-en zatiketa erabiltzen du (zehatza hura ere).

use num_complex::Complex64 as C;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

// a + b = s + e zehazki (knuth)
#[inline(always)]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

// a + b = s + e zehazki, |a| >= |b| bada
#[inline(always)]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

// a · b = p + e zehazki. FMA-rekin instrukzio bat; bestela dekker-en zatiketa (zehatza hura ere)
#[inline(always)]
pub fn two_prod<const FMA: bool>(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    if FMA {
        (p, a.mul_add(b, -p))
    } else {
        let (ah, al) = split(a);
        let (bh, bl) = split(b);
        (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
    }
}

// dekker: a = h + l, bakoitza 26 bitekin, h·h eta antzekoak zehatzak izan daitezen
#[inline(always)]
fn split(a: f64) -> (f64, f64) {
    let c = 134_217_729.0 * a;
    let h = c - (c - a);
    (h, a - h)
}

// FMA hardware-an: aarch64 eta riscv64gc-n beti; x86_64-n exekuzioan begiratzen da
#[cfg(any(target_arch = "aarch64", target_arch = "riscv64"))]
const FMA_ALWAYS: bool = true;
#[cfg(not(any(target_arch = "aarch64", target_arch = "riscv64")))]
const FMA_ALWAYS: bool = false;

impl Dd {
    pub const ZERO: Dd = Dd { hi: 0.0, lo: 0.0 };

    pub fn from(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }

    // double-double batuketa zehatza (qd liburutegiaren "ieee" batuketa)
    #[inline(always)]
    pub fn plus(self, b: Dd) -> Dd {
        let (s1, s2) = two_sum(self.hi, b.hi);
        let (t1, t2) = two_sum(self.lo, b.lo);
        let (s1, s2) = fast_two_sum(s1, s2 + t1);
        let (hi, lo) = fast_two_sum(s1, s2 + t2);
        Dd { hi, lo }
    }

    #[inline(always)]
    pub fn add(&mut self, x: f64) {
        let (s, e) = two_sum(self.hi, x);
        let (hi, lo) = fast_two_sum(s, e + self.lo);
        *self = Dd { hi, lo };
    }

    // a · b gehitu, biderketaren errorea galdu gabe
    #[inline(always)]
    pub fn add_prod<const FMA: bool>(&mut self, a: f64, b: f64) {
        let (p, e) = two_prod::<FMA>(a, b);
        *self = self.plus(Dd { hi: p, lo: e });
    }

    // |a|² gehitu
    #[inline(always)]
    pub fn add_norm<const FMA: bool>(&mut self, a: C) {
        self.add_prod::<FMA>(a.re, a.re);
        self.add_prod::<FMA>(a.im, a.im);
    }

    pub fn value(self) -> f64 {
        self.hi + self.lo
    }
}

impl std::ops::Neg for Dd {
    type Output = Dd;
    fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

// Σ |x_i|² zehatza, `bit` 0 bada guztiena, bestela (base + i) indizean `bit` bita 1 dutenena. lau
// metagailu txandaka (menpekotasun-kate laburragoak), gero batu
#[inline(always)]
fn norm_sum_with<const FMA: bool>(xs: &[C], base: usize, bit: usize) -> Dd {
    let mut acc = [Dd::ZERO; 4];
    for (i, a) in xs.iter().enumerate() {
        if bit == 0 || (base + i) & bit != 0 {
            acc[i & 3].add_norm::<FMA>(*a);
        }
    }
    acc[0].plus(acc[1]).plus(acc[2].plus(acc[3]))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
fn norm_sum_fma(xs: &[C], base: usize, bit: usize) -> Dd {
    norm_sum_with::<true>(xs, base, bit)
}

fn norm_sum_where(xs: &[C], base: usize, bit: usize) -> Dd {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: prozesadoreak FMA duela egiaztatu da
        return unsafe { norm_sum_fma(xs, base, bit) };
    }
    norm_sum_with::<FMA_ALWAYS>(xs, base, bit)
}

// Σ |x|²
pub fn norm_sum(xs: &[C]) -> Dd {
    norm_sum_where(xs, 0, 0)
}

// Σ |x_i|², (base + i) indizean `bit` bita 1 denean (bit bakarreko maskara)
pub fn norm_sum_bit(xs: &[C], base: usize, bit: usize) -> Dd {
    debug_assert!(bit.is_power_of_two());
    norm_sum_where(xs, base, bit)
}

// f64 zerrenda baten batura zehatza
pub fn sum(xs: impl IntoIterator<Item = f64>) -> f64 {
    let mut acc = Dd::ZERO;
    for x in xs {
        acc.add(x);
    }
    acc.value()
}
