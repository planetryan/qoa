// fitxategi honek doitasun laukoitzeko (128 bit) qubit-motorra du: `qoa run --precision quad`.
//
// zenbaki bakoitza double-double bat da: bi f64-ren batura (hi + lo), ~106 bit esanguratsu (~32 digitu),
// errorerik gabeko transformazioekin (two_sum eta FMA-ren bidezko two_prod). hardware-ak FMA duen
// edozein 64 biteko sistematan dabil (x86-64, ARM64, RISC-V, POWER, IBM Z, LoongArch).
//
// zer dagoen 128 bitetan: anplitudeak, ateen matrizeak (sin, cos eta erro karratuak hemen kalkulatuta,
// programaren angeluetatik), neurketa-probabilitateak, kolapsoaren normalizazioa eta egoeraren gaineko
// baturak. programaren angeluak bere 64 biteko balioak dira (.qx formatuak f64 gordetzen ditu), zehatz gisa
// hartuta: kalkuluak ez du errorerik gehitzen, baina pi/4 bezalako angelu bat f64-ra biribildua dago jada
// (1e-17 inguru). qubit-programak bakarrik: modu optikoak, GPUa eta gailu-ereduak f64-n daude.
//
// baturak zati finkoetan egiten dira eta ordena berean batzen dira, beraz emaitza ez dago hari-kopuruaren
// menpe.
// unsafe hemen: fma eta avx2-rekin konpilatutako kernelei deitzea, prozesadoreak badituela egiaztatu ondoren
#![allow(unsafe_code)]

use crate::kernel::QGate;
use crate::program::Gate;
use crate::state::State;
use crate::t;
use num_complex::Complex64 as C;
use rayon::prelude::*;

// zati paraleloen tamaina (anplitudetan)
const PIECE: usize = 1 << 13;

// cache-blokeak: 2^13 anplitude × 32 byte = 256 KiB, nukleo bateko L2-an sartzen da
const BLOCK_BITS: usize = 13;
const BLOCK: usize = 1 << BLOCK_BITS;

// ------------------------------------------------------------------ double-double

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Q {
    pub hi: f64,
    pub lo: f64,
}

#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

#[inline(always)]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

// a·b = p + e zehazki (mul_add: hardware-ko FMA, edo libm-ren fma zehatza)
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl Q {
    pub const ZERO: Q = Q { hi: 0.0, lo: 0.0 };
    pub const ONE: Q = Q { hi: 1.0, lo: 0.0 };
    // π eta π/2, double-double-n (qd liburutegiaren balioak)
    pub const PI: Q = Q {
        hi: std::f64::consts::PI,
        lo: 1.224_646_799_147_353_2e-16,
    };
    pub const FRAC_PI_2: Q = Q {
        hi: std::f64::consts::FRAC_PI_2,
        lo: 6.123_233_995_736_766e-17,
    };

    pub const fn new(x: f64) -> Q {
        Q { hi: x, lo: 0.0 }
    }

    pub fn value(self) -> f64 {
        self.hi + self.lo
    }

    #[inline(always)]
    pub fn plus(self, b: Q) -> Q {
        let (s1, s2) = two_sum(self.hi, b.hi);
        let (t1, t2) = two_sum(self.lo, b.lo);
        let (s1, s2) = fast_two_sum(s1, s2 + t1);
        let (hi, lo) = fast_two_sum(s1, s2 + t2);
        Q { hi, lo }
    }

    #[inline(always)]
    pub fn minus(self, b: Q) -> Q {
        self.plus(b.negate())
    }

    #[inline(always)]
    pub fn negate(self) -> Q {
        Q {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    #[inline(always)]
    pub fn times(self, b: Q) -> Q {
        let (p, e) = two_prod(self.hi, b.hi);
        let e = e + (self.hi * b.lo + self.lo * b.hi);
        let (hi, lo) = fast_two_sum(p, e);
        Q { hi, lo }
    }

    #[inline(always)]
    pub fn times_f64(self, b: f64) -> Q {
        let (p, e) = two_prod(self.hi, b);
        let e = e + self.lo * b;
        let (hi, lo) = fast_two_sum(p, e);
        Q { hi, lo }
    }

    // zatiketa: hiru zatidura partzial (qd-ren "accurate" bertsioa)
    pub fn over(self, b: Q) -> Q {
        let q1 = self.hi / b.hi;
        let r = self.minus(b.times_f64(q1));
        let q2 = r.hi / b.hi;
        let r = r.minus(b.times_f64(q2));
        let q3 = r.hi / b.hi;
        let (hi, lo) = fast_two_sum(q1, q2);
        Q { hi, lo }.plus(Q::new(q3))
    }

    // √x: f64-ren erroa eta newton-en urrats bat double-double-n (zehaztasuna bikoizten du)
    pub fn sqrt(self) -> Q {
        if self.hi <= 0.0 {
            return Q::ZERO;
        }
        let s = self.hi.sqrt();
        let r = self.minus(Q::new(s).times(Q::new(s)));
        Q::new(s).plus(Q::new(r.hi / (2.0 * s)))
    }

    // sin eta cos batera. x-ren murrizketa k·π/2-z (k = biribildu(x / (π/2))), π/2 hiru f64-tan (k·p_i
    // biderkadura bakoitza zehatza da), gero taylor-en serieak |t| <= π/4 tartean, gaiak 1e-35 azpitik egon
    // arte. |x| < 1e9 bitartean errorea 1e-31 ingurukoa da
    pub fn sin_cos(self) -> (Q, Q) {
        const P3: f64 = -1.497_384_904_859_169_8e-33;
        let k = (self.value() / Q::FRAC_PI_2.hi).round();
        let t = self
            .minus(Q::new(Q::FRAC_PI_2.hi).times_f64(k))
            .minus(Q::new(Q::FRAC_PI_2.lo).times_f64(k))
            .minus(Q::new(P3).times_f64(k));
        let t2 = t.times(t);
        let (mut s, mut term) = (t, t);
        let mut n = 1.0;
        loop {
            term = term.times(t2).negate().over(Q::new((n + 1.0) * (n + 2.0)));
            s = s.plus(term);
            n += 2.0;
            if term.hi.abs() < 1e-35 {
                break;
            }
        }
        let (mut c, mut term) = (Q::ONE, Q::ONE);
        let mut n = 0.0;
        loop {
            term = term.times(t2).negate().over(Q::new((n + 1.0) * (n + 2.0)));
            c = c.plus(term);
            n += 2.0;
            if term.hi.abs() < 1e-35 {
                break;
            }
        }
        match (k as i64).rem_euclid(4) {
            0 => (s, c),
            1 => (c, s.negate()),
            2 => (s.negate(), c.negate()),
            _ => (c.negate(), s),
        }
    }
}

// ------------------------------------------------------------------ zenbaki konplexuak

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Qc {
    pub re: Q,
    pub im: Q,
}

impl Qc {
    pub const ZERO: Qc = Qc {
        re: Q::ZERO,
        im: Q::ZERO,
    };
    pub const ONE: Qc = Qc {
        re: Q::ONE,
        im: Q::ZERO,
    };

    pub const fn new(re: Q, im: Q) -> Qc {
        Qc { re, im }
    }

    // f64 konplexu bat, zehazki
    pub fn from_c(c: C) -> Qc {
        Qc::new(Q::new(c.re), Q::new(c.im))
    }

    pub fn to_c(self) -> C {
        C::new(self.re.value(), self.im.value())
    }

    // r·e^{iθ}
    pub fn polar(r: Q, theta: Q) -> Qc {
        let (s, c) = theta.sin_cos();
        Qc::new(r.times(c), r.times(s))
    }

    #[inline(always)]
    pub fn plus(self, b: Qc) -> Qc {
        Qc::new(self.re.plus(b.re), self.im.plus(b.im))
    }

    #[inline(always)]
    pub fn times(self, b: Qc) -> Qc {
        Qc::new(
            self.re.times(b.re).minus(self.im.times(b.im)),
            self.re.times(b.im).plus(self.im.times(b.re)),
        )
    }

    #[inline(always)]
    pub fn scale(self, s: Q) -> Qc {
        Qc::new(self.re.times(s), self.im.times(s))
    }

    pub fn negate(self) -> Qc {
        Qc::new(self.re.negate(), self.im.negate())
    }

    pub fn conj(self) -> Qc {
        Qc::new(self.re, self.im.negate())
    }

    #[inline(always)]
    pub fn norm_sqr(self) -> Q {
        self.re.times(self.re).plus(self.im.times(self.im))
    }
}

// ------------------------------------------------------------------ ateak

// 1/√2
pub fn frac_1_sqrt_2() -> Q {
    Q::new(0.5).sqrt()
}

// atearen 2x2 matrize unitarioa (errenkadaka) 128 bitetan; angeluak zehatz gisa
pub fn matrix(g: Gate) -> [Qc; 4] {
    let (z, one) = (Qc::ZERO, Qc::ONE);
    let i = Qc::new(Q::ZERO, Q::ONE);
    let r = frac_1_sqrt_2();
    let h = Q::new(0.5);
    let half = |t: f64| Q::new(t).times_f64(0.5);
    match g {
        Gate::H => [
            Qc::new(r, Q::ZERO),
            Qc::new(r, Q::ZERO),
            Qc::new(r, Q::ZERO),
            Qc::new(r.negate(), Q::ZERO),
        ],
        Gate::X => [z, one, one, z],
        Gate::Y => [z, i.negate(), i, z],
        Gate::Z => [one, z, z, one.negate()],
        Gate::S => [one, z, z, i],
        Gate::Sdg => [one, z, z, i.negate()],
        Gate::T => [one, z, z, Qc::new(r, r)],
        Gate::Tdg => [one, z, z, Qc::new(r, r.negate())],
        Gate::SX => [
            Qc::new(h, h),
            Qc::new(h, h.negate()),
            Qc::new(h, h.negate()),
            Qc::new(h, h),
        ],
        Gate::SXdg => [
            Qc::new(h, h.negate()),
            Qc::new(h, h),
            Qc::new(h, h),
            Qc::new(h, h.negate()),
        ],
        Gate::RX(t) => {
            let (s, c) = half(t).sin_cos();
            [
                Qc::new(c, Q::ZERO),
                Qc::new(Q::ZERO, s.negate()),
                Qc::new(Q::ZERO, s.negate()),
                Qc::new(c, Q::ZERO),
            ]
        }
        Gate::RY(t) => {
            let (s, c) = half(t).sin_cos();
            [
                Qc::new(c, Q::ZERO),
                Qc::new(s.negate(), Q::ZERO),
                Qc::new(s, Q::ZERO),
                Qc::new(c, Q::ZERO),
            ]
        }
        Gate::RZ(t) => [
            Qc::polar(Q::ONE, half(t).negate()),
            z,
            z,
            Qc::polar(Q::ONE, half(t)),
        ],
        Gate::P(t) => [one, z, z, Qc::polar(Q::ONE, Q::new(t))],
        Gate::U(theta, phi, lambda) => {
            let (s, c) = half(theta).sin_cos();
            let sum = Q::new(phi).plus(Q::new(lambda));
            [
                Qc::new(c, Q::ZERO),
                Qc::polar(s, Q::new(lambda)).negate(),
                Qc::polar(s, Q::new(phi)),
                Qc::polar(c, sum),
            ]
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuadGate {
    Mat {
        q: usize,
        m: [Qc; 4],
        controls: usize,
    },
    Swap {
        a: usize,
        b: usize,
        controls: usize,
    },
}

impl QuadGate {
    // `mask`-eko qubit guztiak 1 dituzten oinarri-egoerak e^{iθ}-z (QGate::phase-ren bertsioa)
    pub fn phase(mask: usize, theta: f64) -> QuadGate {
        let p = Qc::polar(Q::ONE, Q::new(theta));
        if mask == 0 {
            return QuadGate::Mat {
                q: 0,
                m: [p, Qc::ZERO, Qc::ZERO, p],
                controls: 0,
            };
        }
        let top = usize::BITS as usize - 1 - mask.leading_zeros() as usize;
        QuadGate::Mat {
            q: top,
            m: [Qc::ONE, Qc::ZERO, Qc::ZERO, p],
            controls: mask & !(1 << top),
        }
    }

    // f64 ate bat, bere sarrerak zehatz gisa hartuta
    pub fn from_f64(g: &QGate) -> QuadGate {
        match *g {
            QGate::Mat { q, m, controls } => QuadGate::Mat {
                q,
                m: m.map(Qc::from_c),
                controls,
            },
            QGate::Swap { a, b, controls } => QuadGate::Swap { a, b, controls },
        }
    }
}

// ------------------------------------------------------------------ egoera

#[derive(Clone, Debug)]
pub struct QuadState {
    n_qubits: usize,
    amps: Vec<Qc>,
}

impl QuadState {
    // |0...0>
    pub fn new(n_qubits: usize) -> Result<QuadState, String> {
        let dim = 1u128.checked_shl(n_qubits as u32).unwrap_or(u128::MAX);
        crate::state::check_memory(
            &t!(
                "{} qubits in 128-bit precision: {} amplitudes",
                "{} qubit 128 biteko doitasunean: {} anplitude",
                n_qubits,
                dim
            ),
            dim.saturating_mul(32),
        )?;
        let mut amps = vec![Qc::ZERO; dim as usize];
        amps[0] = Qc::ONE;
        Ok(QuadState { n_qubits, amps })
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn amplitudes(&self) -> &[Qc] {
        &self.amps
    }

    // ate-sorta bat. qubit bereko ate kontrolgabe jarraituak (tartean qubit hori ukitzen ez dutenak egon
    // daitezke) matrize bakar batean biltzen dira, 128 bitetan biderkatuta
    // helburu-qubit guztiak BLOCK_BITS-en azpitik dituen ate-segida bat bloke bakoitzean osorik aplikatzen da,
    // blokea cache-n dagoen bitartean (memoria behin zeharkatzen da segida osorako)
    pub fn apply(&mut self, gates: &[QuadGate]) {
        let gates = fuse(gates);
        let mut i = 0;
        while i < gates.len() {
            let mut j = i;
            while j < gates.len() && low(&gates[j]) {
                j += 1;
            }
            if j - i >= 2 && self.amps.len() > BLOCK {
                let stage = &gates[i..j];
                self.amps
                    .par_chunks_mut(BLOCK)
                    .enumerate()
                    .for_each(|(b, chunk)| {
                        for g in stage {
                            in_slice(chunk, b * BLOCK, g);
                        }
                    });
                i = j;
                continue;
            }
            match gates[i] {
                QuadGate::Mat { q, m, controls } => self.apply_mat(q, &m, controls),
                QuadGate::Swap { a, b, controls } => self.apply_swap(a, b, controls),
            }
            i += 1;
        }
    }

    // (i, i + 2^q) bikoteak, kontrol guztiak 1 direnean
    fn apply_mat(&mut self, q: usize, m: &[Qc; 4], controls: usize) {
        let bit = 1usize << q;
        let min = (PIECE / (2 * bit)).max(1);
        self.amps
            .par_chunks_mut(2 * bit)
            .with_min_len(min)
            .enumerate()
            .for_each(|(blk, chunk)| {
                let (lo, hi) = chunk.split_at_mut(bit);
                lo.par_chunks_mut(PIECE)
                    .zip(hi.par_chunks_mut(PIECE))
                    .enumerate()
                    .for_each(|(p, (l, h))| pairs(l, h, blk * 2 * bit + p * PIECE, m, controls));
            });
    }

    // a eta b qubitak trukatu (kontrolekin): bit altua 1 eta baxua 0 dituen anplitudea, alderantzizkoarekin
    fn apply_swap(&mut self, a: usize, b: usize, controls: usize) {
        if a == b {
            return;
        }
        let (hb, lb) = (1usize << a.max(b), 1usize << a.min(b));
        self.amps
            .par_chunks_mut(2 * hb)
            .with_min_len((PIECE / (2 * hb)).max(1))
            .enumerate()
            .for_each(|(blk, chunk)| {
                let (lower, upper) = chunk.split_at_mut(hb);
                for k in 0..hb {
                    if k & lb == 0 {
                        continue;
                    }
                    let idx = blk * 2 * hb + k;
                    if idx & controls != controls {
                        continue;
                    }
                    std::mem::swap(&mut lower[k], &mut upper[k - lb]);
                }
            });
    }

    // Σ f(i, a_i) zati finkoetan, ordena berean batuta
    fn sum(&self, f: impl Fn(usize, &Qc) -> Q + Sync) -> Q {
        let parts: Vec<Q> = self
            .amps
            .par_chunks(PIECE)
            .enumerate()
            .map(|(k, xs)| chunk_sum(xs, k * PIECE, &f))
            .collect();
        parts.into_iter().fold(Q::ZERO, Q::plus)
    }

    pub fn norm_sqr(&self) -> Q {
        self.sum(|_, a| a.norm_sqr())
    }

    pub fn norm_error(&self) -> f64 {
        Q::ONE.minus(self.norm_sqr()).value().abs()
    }

    pub fn prob_one(&self, q: usize) -> Q {
        let bit = 1usize << q;
        self.sum(|i, a| if i & bit != 0 { a.norm_sqr() } else { Q::ZERO })
    }

    // q qubita neurtu `r`-rekin ([0, 1) uniformea); kolapsoa 1/√p-z 128 bitetan
    pub fn measure_qubit(&mut self, q: usize, r: f64) -> u8 {
        let p1 = self.prob_one(q);
        let outcome = if r < p1.value().clamp(0.0, 1.0) { 1 } else { 0 };
        let p = if outcome == 1 { p1 } else { Q::ONE.minus(p1) };
        let s = Qc::new(Q::ONE.over(p.sqrt()), Q::ZERO);
        let m = if outcome == 1 {
            [Qc::ZERO, Qc::ZERO, Qc::ZERO, s]
        } else {
            [s, Qc::ZERO, Qc::ZERO, Qc::ZERO]
        };
        self.apply_mat(q, &m, 0);
        outcome
    }

    pub fn probabilities(&self) -> Vec<f64> {
        self.amps.par_iter().map(|a| a.norm_sqr().value()).collect()
    }

    // q qubitaren <X>, <Y>, <Z>
    pub fn bloch(&self, q: usize) -> [f64; 3] {
        let bit = 1usize << q;
        let x = self.sum(|i, a| {
            if i & bit == 0 {
                a.conj().times(self.amps[i | bit]).re
            } else {
                Q::ZERO
            }
        });
        let y = self.sum(|i, a| {
            if i & bit == 0 {
                a.conj().times(self.amps[i | bit]).im
            } else {
                Q::ZERO
            }
        });
        let z = self.sum(|i, a| {
            if i & bit == 0 {
                a.norm_sqr()
            } else {
                a.norm_sqr().negate()
            }
        });
        [
            x.times_f64(2.0).value(),
            y.times_f64(2.0).value(),
            z.value(),
        ]
    }

    // (a, b) qubiten fideltasuna (|00> + |11>)/√2 bell egoerarekin
    pub fn bell_fidelity(&self, a: usize, b: usize) -> f64 {
        let (ba, bb) = (1usize << a, 1usize << b);
        self.sum(|i, x| {
            if i & ba == 0 && i & bb == 0 {
                x.plus(self.amps[i | ba | bb]).norm_sqr()
            } else {
                Q::ZERO
            }
        })
        .times_f64(0.5)
        .value()
    }

    // f64 egoera bat (biribilduta), txostenetarako eta laginketarako
    pub fn to_state(&self) -> State {
        let amps = self.amps.par_iter().map(|a| a.to_c()).collect();
        State::from_amplitudes(self.n_qubits, 0, 2, amps).expect("tamaina berekoa")
    }
}

// ------------------------------------------------------------------ kernelak

// FMA hardware-an: x86-64-n exekuzioan begiratzen da (bestela mul_add libm-ren deia da); beste 64 biteko
// arkitektura guztietan (ARM64, RISC-V, POWER, IBM Z, LoongArch) oinarrizkoa da
#[cfg(target_arch = "x86_64")]
fn fma_avx2() -> bool {
    std::arch::is_x86_feature_detected!("fma") && std::arch::is_x86_feature_detected!("avx2")
}

// helburu guztiak bloke baten barruan
fn low(g: &QuadGate) -> bool {
    match *g {
        QuadGate::Mat { q, .. } => q < BLOCK_BITS,
        QuadGate::Swap { a, b, .. } => a.max(b) < BLOCK_BITS,
    }
}

// ate bat zati jarraitu baten barruan (`base` haren lehen indizea; helburuak zatiaren barruan)
fn in_slice(xs: &mut [Qc], base: usize, g: &QuadGate) {
    match *g {
        QuadGate::Mat { q, m, controls } => {
            let bit = 1usize << q;
            for (blk, chunk) in xs.chunks_mut(2 * bit).enumerate() {
                let (l, h) = chunk.split_at_mut(bit);
                pairs(l, h, base + blk * 2 * bit, &m, controls);
            }
        }
        QuadGate::Swap { a, b, controls } => {
            if a == b {
                return;
            }
            let (hb, lb) = (1usize << a.max(b), 1usize << a.min(b));
            for (blk, chunk) in xs.chunks_mut(2 * hb).enumerate() {
                let (lower, upper) = chunk.split_at_mut(hb);
                for k in (0..hb).filter(|k| k & lb != 0) {
                    if (base + blk * 2 * hb + k) & controls == controls {
                        std::mem::swap(&mut lower[k], &mut upper[k - lb]);
                    }
                }
            }
        }
    }
}

// matrizearen forma: trukaketa (X), diagonala ala orokorra
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Flip,
    Diagonal,
    General,
}

fn kind(m: &[Qc; 4]) -> Kind {
    if m[1] == Qc::ZERO && m[2] == Qc::ZERO {
        Kind::Diagonal
    } else if m[0] == Qc::ZERO && m[3] == Qc::ZERO && m[1] == Qc::ONE && m[2] == Qc::ONE {
        Kind::Flip
    } else {
        Kind::General
    }
}

// lau bikote batera, osagai bakoitza [f64; 4] batean (konpiladoreak bektore-erregistroetan jartzen ditu):
// double-double eragiketa berak, erreika bakoitzean
type L = [f64; 4];

#[derive(Clone, Copy)]
struct Lq {
    hi: L,
    lo: L,
}

#[derive(Clone, Copy)]
struct Lc {
    re: Lq,
    im: Lq,
}

impl Lq {
    #[inline(always)]
    fn splat(q: Q) -> Lq {
        Lq {
            hi: [q.hi; 4],
            lo: [q.lo; 4],
        }
    }

    #[inline(always)]
    fn plus(self, b: Lq) -> Lq {
        let mut out = self;
        for i in 0..4 {
            let (s1, s2) = two_sum(self.hi[i], b.hi[i]);
            let (t1, t2) = two_sum(self.lo[i], b.lo[i]);
            let (s1, s2) = fast_two_sum(s1, s2 + t1);
            let (hi, lo) = fast_two_sum(s1, s2 + t2);
            out.hi[i] = hi;
            out.lo[i] = lo;
        }
        out
    }

    #[inline(always)]
    fn minus(self, b: Lq) -> Lq {
        let neg = Lq {
            hi: b.hi.map(|v| -v),
            lo: b.lo.map(|v| -v),
        };
        self.plus(neg)
    }

    #[inline(always)]
    fn times(self, b: Lq) -> Lq {
        let mut out = self;
        for i in 0..4 {
            let (p, e) = two_prod(self.hi[i], b.hi[i]);
            let e = e + (self.hi[i] * b.lo[i] + self.lo[i] * b.hi[i]);
            let (hi, lo) = fast_two_sum(p, e);
            out.hi[i] = hi;
            out.lo[i] = lo;
        }
        out
    }
}

impl Lc {
    #[inline(always)]
    fn splat(c: Qc) -> Lc {
        Lc {
            re: Lq::splat(c.re),
            im: Lq::splat(c.im),
        }
    }

    #[inline(always)]
    fn load(xs: &[Qc]) -> Lc {
        Lc {
            re: Lq {
                hi: std::array::from_fn(|i| xs[i].re.hi),
                lo: std::array::from_fn(|i| xs[i].re.lo),
            },
            im: Lq {
                hi: std::array::from_fn(|i| xs[i].im.hi),
                lo: std::array::from_fn(|i| xs[i].im.lo),
            },
        }
    }

    #[inline(always)]
    fn store(self, xs: &mut [Qc]) {
        for (i, x) in xs.iter_mut().enumerate().take(4) {
            *x = Qc::new(
                Q {
                    hi: self.re.hi[i],
                    lo: self.re.lo[i],
                },
                Q {
                    hi: self.im.hi[i],
                    lo: self.im.lo[i],
                },
            );
        }
    }

    #[inline(always)]
    fn plus(self, b: Lc) -> Lc {
        Lc {
            re: self.re.plus(b.re),
            im: self.im.plus(b.im),
        }
    }

    #[inline(always)]
    fn times(self, b: Lc) -> Lc {
        Lc {
            re: self.re.times(b.re).minus(self.im.times(b.im)),
            im: self.re.times(b.im).plus(self.im.times(b.re)),
        }
    }
}

#[inline(always)]
fn pairs_body(l: &mut [Qc], h: &mut [Qc], base: usize, m: &[Qc; 4], controls: usize) {
    let k = kind(m);
    // kontrolik gabe: lauko taldeak erreiketan, gero hondarra banan-banan
    let mut start = 0;
    if controls == 0 && k != Kind::Flip {
        let ms = m.map(Lc::splat);
        let n = l.len() / 4 * 4;
        let (lcs, hcs) = (l[..n].as_chunks_mut::<4>().0, h[..n].as_chunks_mut::<4>().0);
        for (lc, hc) in lcs.iter_mut().zip(hcs.iter_mut()) {
            let (lc, hc) = (&mut lc[..], &mut hc[..]);
            let (x, y) = (Lc::load(lc), Lc::load(hc));
            if k == Kind::Diagonal {
                if m[0] != Qc::ONE {
                    ms[0].times(x).store(lc);
                }
                ms[3].times(y).store(hc);
            } else {
                ms[0].times(x).plus(ms[1].times(y)).store(lc);
                ms[2].times(x).plus(ms[3].times(y)).store(hc);
            }
        }
        start = n;
    }
    let (l, h) = (&mut l[start..], &mut h[start..]);
    let base = base + start;
    for (i, (a, b)) in l.iter_mut().zip(h.iter_mut()).enumerate() {
        if (base + i) & controls != controls {
            continue;
        }
        match k {
            Kind::Flip => std::mem::swap(a, b),
            Kind::Diagonal => {
                if m[0] != Qc::ONE {
                    *a = m[0].times(*a);
                }
                *b = m[3].times(*b);
            }
            Kind::General => {
                let (x, y) = (*a, *b);
                *a = m[0].times(x).plus(m[1].times(y));
                *b = m[2].times(x).plus(m[3].times(y));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn pairs_fma(l: &mut [Qc], h: &mut [Qc], base: usize, m: &[Qc; 4], controls: usize) {
    pairs_body(l, h, base, m, controls)
}

// (l[i], h[i]) bikoteak matrizearekin, (base + i) indizeko kontrolak 1 direnean
fn pairs(l: &mut [Qc], h: &mut [Qc], base: usize, m: &[Qc; 4], controls: usize) {
    #[cfg(target_arch = "x86_64")]
    if fma_avx2() {
        // SAFETY: prozesadoreak FMA eta AVX2 dituela egiaztatu da
        return unsafe { pairs_fma(l, h, base, m, controls) };
    }
    pairs_body(l, h, base, m, controls)
}

#[inline(always)]
fn chunk_body(xs: &[Qc], base: usize, f: &(impl Fn(usize, &Qc) -> Q + Sync)) -> Q {
    let mut acc = Q::ZERO;
    for (i, a) in xs.iter().enumerate() {
        acc = acc.plus(f(base + i, a));
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn chunk_fma(xs: &[Qc], base: usize, f: &(impl Fn(usize, &Qc) -> Q + Sync)) -> Q {
    chunk_body(xs, base, f)
}

fn chunk_sum(xs: &[Qc], base: usize, f: &(impl Fn(usize, &Qc) -> Q + Sync)) -> Q {
    #[cfg(target_arch = "x86_64")]
    if fma_avx2() {
        // SAFETY: prozesadoreak FMA eta AVX2 dituela egiaztatu da
        return unsafe { chunk_fma(xs, base, f) };
    }
    chunk_body(xs, base, f)
}

// 2x2 matrizeen biderkadura: `then` · `first`
fn product(then: &[Qc; 4], first: &[Qc; 4]) -> [Qc; 4] {
    let e = |r: usize, c: usize| {
        then[2 * r]
            .times(first[c])
            .plus(then[2 * r + 1].times(first[2 + c]))
    };
    [e(0, 0), e(0, 1), e(1, 0), e(1, 1)]
}

// qubit bereko ate kontrolgabe jarraituak bat egin (src/kernel.rs-ko `fuse`-ren bertsioa): ate bakoitza
// zain dagoen bere qubitaren matrizearekin biderkatzen da; qubit hori ukitzen duen beste ate batek (kontrol
// edo helburu gisa) zain dagoena isurtzen du lehenik
fn fuse(gates: &[QuadGate]) -> Vec<QuadGate> {
    let mut out = Vec::with_capacity(gates.len());
    let mut open: Vec<Option<[Qc; 4]>> = Vec::new();
    let flush = |q: usize, open: &mut Vec<Option<[Qc; 4]>>, out: &mut Vec<QuadGate>| {
        if let Some(m) = open.get_mut(q).and_then(|m| m.take()) {
            out.push(QuadGate::Mat { q, m, controls: 0 });
        }
    };
    for g in gates {
        match *g {
            QuadGate::Mat { q, m, controls: 0 } => {
                if open.len() <= q {
                    open.resize(q + 1, None);
                }
                open[q] = Some(match open[q] {
                    Some(prev) => product(&m, &prev),
                    None => m,
                });
            }
            QuadGate::Mat { q, controls, .. } => {
                for c in (0..usize::BITS as usize).filter(|c| controls >> c & 1 == 1) {
                    flush(c, &mut open, &mut out);
                }
                flush(q, &mut open, &mut out);
                out.push(*g);
            }
            QuadGate::Swap { a, b, controls } => {
                for c in (0..usize::BITS as usize).filter(|c| controls >> c & 1 == 1) {
                    flush(c, &mut open, &mut out);
                }
                flush(a, &mut open, &mut out);
                flush(b, &mut open, &mut out);
                out.push(*g);
            }
        }
    }
    for q in 0..open.len() {
        flush(q, &mut open, &mut out);
    }
    out
}
