// fitxategi honek simulagailuaren barne-begizta beroenak ditu, arkitektura bakoitzeko bektore-aginduekin:
// avx-512 eta avx2+fma (x86-64) eta neon (aarch64) rust seguruan, core::arch-en funtzio intrintsekoekin;
// rvv 1.0 (riscv64) mihiztaduran (`asm!`), rust egonkorrak ez baitu oraindik rvv-rik onartzen; eta
// erreferentziazko bertsio eskalarra. makinak duen onena exekuzio-garaian aukeratzen da behin;
// `QOA_SIMD=scalar` (edo avx2, avx512, neon, rvv) ingurune-aldagaiak bat behartzen du.
//
// segurtasuna: x86 eta arm kernelek memoria slice-en bidez bakarrik ukitzen dute (rust-ek mugak egiaztatuta),
// eta bektore-aritmetika funtzio intrintseko seguruekin egiten dute. `unsafe` bi lekutan bakarrik dago:
// prozesadorearen ezaugarriekin konpilatutako funtzio bati deitzea, ezaugarri hori makina honetan dagoela
// egiaztatu ondoren (`Isa::is_available`), eta bektore bat erreferentzia batetik kargatzea edo hara
// gordetzea (erreferentziak bektorearen memoria osoa bermatzen du; rust egonkorrak ez du karga/gordeketa
// segururik). rvv-k erakusle gordinak behar ditu; haien mugak goian egiaztatzen dira.
// bertsio guztiek mihiztadurako bertsio zaharrek egiten zituzten eragiketa berberak egiten dituzte ordena
// berean, beraz emaitzak bitez bit berdinak dira.
//
// hiru kernel, guztiak "lasterketa"-sorten gainean (`Spans`): `count` lasterketa, bakoitza `len`
// anplitudekoa, `stride`-ko tartearekin. horrela dei bakar batek bloke oso bat prozesatzen du, eta deiaren
// kostua ez da elementu gutxiko lasterketetan errepikatzen.
//
//   mat2       lo eta hi lasterketa parekoei 2x2 matrizea: lo' = m00 lo + m01 hi, hi' = m10 lo + m11 hi
//   swizzle    lasterketa jarraitu baten barruan, `half` (1 edo 2) distantziako bikoteei: x' = a x + b y
//   scale      x' = d x
//
// koefizienteak "erreika" bakoitzeko dira: 4 anplitudeko talde bakoitzean, posizio bakoitzak (indizearen
// 0. eta 1. bitak) bere koefizienteak ditu. horrela 0. eta 1. qubitetako kontrolak eta q0/q1-eko ateak
// bektore-erregistroen barruan egiten dira: kontrolak betetzen ez dituzten erreietan identitatea dago
// (1 * x + 0 * y, zehatza). horregatik `len` 4ren multiploa da beti.
//
// anplitudeak `Complex64` dira memorian: [re, im] tartekatuta. x86-n konplexuen biderketa fmaddsub-ekin
// egiten da; neon eta rvv-k re eta im banatzen dituzte kargatzean. probek bertsio guztiak eskalarraren
// aurka egiaztatzen dituzte (tests/simd.rs), aarch64 eta riscv64 qemu-n barne.
// unsafe hemen: prozesadorearen ezaugarriekin konpilatutako kernelei deitzea (egiaztatu ondoren), bektore
// bat erreferentzia batetik kargatzea edo gordetzea, eta rvv-ren mihiztadura
#![allow(unsafe_code)]

use num_complex::Complex64 as C;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    Avx2,
    Avx512,
    Neon,
    Rvv,
}

impl Isa {
    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
            Isa::Neon => "neon",
            Isa::Rvv => "rvv",
        }
    }

    pub fn is_available(self) -> bool {
        available().contains(&self)
    }
}

// makina honetan erabil daitezkeenak, onenetik txarrenera; eskalarra beti azkena
// (push-ak arkitekturaren araberakoak dira, beraz bektorea ezin da literal bakar batean eraiki)
#[allow(clippy::vec_init_then_push)]
pub fn available() -> &'static [Isa] {
    static AVAILABLE: OnceLock<Vec<Isa>> = OnceLock::new();
    AVAILABLE.get_or_init(|| {
        let mut v = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                v.push(Isa::Avx512);
            }
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                v.push(Isa::Avx2);
            }
        }
        #[cfg(target_arch = "aarch64")]
        v.push(Isa::Neon);
        #[cfg(all(target_arch = "riscv64", not(target_feature = "v")))]
        if rv::has_vector() {
            v.push(Isa::Rvv);
        }
        v.push(Isa::Scalar);
        v
    })
}

// kernelek erabiltzen dutena: onena, edo QOA_SIMD-k aukeratutakoa (makina honetan badago)
pub fn active() -> Isa {
    static ACTIVE: OnceLock<Isa> = OnceLock::new();
    *ACTIVE.get_or_init(|| {
        let wanted = std::env::var("QOA_SIMD").unwrap_or_default();
        available()
            .iter()
            .copied()
            .find(|i| i.name() == wanted)
            .unwrap_or(available()[0])
    })
}

// lasterketa-sorta: `count` lasterketa, k-garrena [start + k * stride, + len) tartean
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spans {
    pub start: usize,
    pub len: usize,
    pub stride: usize,
    pub count: usize,
}

impl Spans {
    pub fn one(start: usize, len: usize) -> Spans {
        Spans {
            start,
            len,
            stride: len,
            count: 1,
        }
    }

    // azken elementuaren ondorengo indizea (0 lasterketarik ez badago)
    fn end(&self) -> usize {
        if self.count == 0 || self.len == 0 {
            0
        } else {
            self.start + (self.count - 1) * self.stride + self.len
        }
    }

    fn check(&self, slice_len: usize) {
        assert!(
            self.len.is_multiple_of(4),
            "lasterketen luzerak 4ren multiploa izan behar du"
        );
        assert!(self.end() <= slice_len, "lasterketak slice-tik kanpo");
    }

    fn is_empty(&self) -> bool {
        self.count == 0 || self.len == 0
    }
}

// erreika bakoitzeko 2x2 matrizea: m[erreika] = [m00, m01, m10, m11]
pub type Mat2Lanes = [[C; 4]; 4];
// erreika bakoitzeko [a, b]: x' = a x + b y (y = bikotekidea)
pub type SwizzleLanes = [[C; 2]; 4];
// erreika bakoitzeko faktorea
pub type ScaleLanes = [C; 4];

pub fn uniform_mat2(m: &[C; 4]) -> Mat2Lanes {
    [*m; 4]
}

// lo/hi lasterketa-bikoteak 4 anplitudeko taldetan: bi slice bereizitan (`Two`), edo slice berean `hi_off`
// aurrerago (`One`). taldeak slice-en zatiak dira, beraz ezin dira mugetatik atera
trait Pairs {
    fn each(self, sp: Spans, f: impl FnMut(&mut [C; 4], &mut [C; 4]));
    // rvv-ren mihiztadurarentzat: lehen lo eta hi elementuen helbideak (mugak goian egiaztatuta)
    #[cfg(all(target_arch = "riscv64", not(target_feature = "v")))]
    fn raw(self) -> (*mut C, *mut C);
}

struct Two<'a>(&'a mut [C], &'a mut [C]);
struct One<'a>(&'a mut [C], usize);

// lasterketa bakoitza: `stride`-ko zatiak `start`-etik, bakoitzaren lehen `len` elementuak
// (azken lasterketaren ondoren `stride`-ra iristeko lekurik ez badago ere: azkena bere luzerarekin)
#[inline(always)]
fn runs(v: &mut [C], sp: Spans) -> impl Iterator<Item = &mut [[C; 4]]> {
    let (full, last) = v[sp.start..sp.end()].split_at_mut((sp.count - 1) * sp.stride);
    full.chunks_exact_mut(sp.stride)
        .map(move |r| r.split_at_mut(sp.len).0)
        .chain(std::iter::once(last))
        .map(|r| r.as_chunks_mut::<4>().0)
}

impl Pairs for Two<'_> {
    #[inline(always)]
    fn each(self, sp: Spans, mut f: impl FnMut(&mut [C; 4], &mut [C; 4])) {
        let Two(lo, hi) = self;
        for (x, y) in runs(lo, sp).zip(runs(hi, sp)) {
            for (x, y) in x.iter_mut().zip(y) {
                f(x, y);
            }
        }
    }

    #[cfg(all(target_arch = "riscv64", not(target_feature = "v")))]
    fn raw(self) -> (*mut C, *mut C) {
        (self.0.as_mut_ptr(), self.1.as_mut_ptr())
    }
}

impl Pairs for One<'_> {
    #[inline(always)]
    fn each(self, sp: Spans, mut f: impl FnMut(&mut [C; 4], &mut [C; 4])) {
        let One(v, off) = self;
        if off + sp.len <= sp.stride {
            // ohikoena: hi lasterketa lo-ren zati berean (lo, hi, hutsunea)
            for r in v[sp.start..].chunks_mut(sp.stride).take(sp.count) {
                let (lo, hi) = r.split_at_mut(off);
                let x = lo[..sp.len].as_chunks_mut::<4>().0;
                let y = hi[..sp.len].as_chunks_mut::<4>().0;
                for (x, y) in x.iter_mut().zip(y) {
                    f(x, y);
                }
            }
            return;
        }
        for s in 0..sp.count {
            let b = sp.start + s * sp.stride;
            let (left, right) = v.split_at_mut(b + off);
            let x = left[b..b + sp.len].as_chunks_mut::<4>().0;
            let y = right[..sp.len].as_chunks_mut::<4>().0;
            for (x, y) in x.iter_mut().zip(y) {
                f(x, y);
            }
        }
    }

    #[cfg(all(target_arch = "riscv64", not(target_feature = "v")))]
    fn raw(self) -> (*mut C, *mut C) {
        let p = self.0.as_mut_ptr();
        (p, p.wrapping_add(self.1))
    }
}

// lasterketa bakoitzeko 4 anplitudeko taldeak (swizzle eta scale)
#[inline(always)]
fn groups(v: &mut [C], sp: Spans, mut f: impl FnMut(&mut [C; 4])) {
    for r in runs(v, sp) {
        for x in r {
            f(x);
        }
    }
}

// lo eta hi bi slice bereizitan, lasterketa berak bietan
pub fn mat2(lo: &mut [C], hi: &mut [C], sp: Spans, k: &Mat2Lanes) {
    mat2_with(active(), lo, hi, sp, k)
}

// lo eta hi slice berean: hi lasterketak lo lasterketak + `hi_off`
pub fn mat2_in(v: &mut [C], hi_off: usize, sp: Spans, k: &Mat2Lanes) {
    mat2_in_with(active(), v, hi_off, sp, k)
}

pub fn swizzle(v: &mut [C], half: usize, sp: Spans, k: &SwizzleLanes) {
    swizzle_with(active(), v, half, sp, k)
}

pub fn scale(v: &mut [C], sp: Spans, k: &ScaleLanes) {
    scale_with(active(), v, sp, k)
}

pub fn mat2_with(isa: Isa, lo: &mut [C], hi: &mut [C], sp: Spans, k: &Mat2Lanes) {
    sp.check(lo.len());
    sp.check(hi.len());
    mat2_pairs(isa, Two(lo, hi), sp, k)
}

pub fn mat2_in_with(isa: Isa, v: &mut [C], hi_off: usize, sp: Spans, k: &Mat2Lanes) {
    sp.check(v.len());
    sp.check(v.len().saturating_sub(hi_off));
    assert!(
        sp.is_empty() || hi_off >= sp.len,
        "mat2_in: hi lasterketak lo-ren ondoren hasi behar du"
    );
    mat2_pairs(isa, One(v, hi_off), sp, k)
}

fn mat2_pairs(isa: Isa, p: impl Pairs, sp: Spans, k: &Mat2Lanes) {
    assert!(isa.is_available(), "{} ez dago makina honetan", isa.name());
    if sp.is_empty() {
        return;
    }
    match isa {
        // segurtasuna (hiru hauetan): isa-ren ezaugarriak makina honetan daude (goian egiaztatua)
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe { x86::mat2_avx512(p, sp, &x86::mat2_coefficients(k)) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { x86::mat2_avx2(p, sp, &x86::mat2_coefficients(k)) },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { arm::mat2(p, sp, &arm::mat2_coefficients(k)) },
        #[cfg(all(target_arch = "riscv64", not(target_feature = "v")))]
        Isa::Rvv => {
            let (lo, hi) = p.raw();
            // segurtasuna: lasterketak bi slice-en barruan daude (mat2_with / mat2_in_with), V badago
            unsafe { rv::mat2(lo, hi, sp, k) }
        }
        _ => scalar::mat2(p, sp, k),
    }
}

pub fn swizzle_with(isa: Isa, v: &mut [C], half: usize, sp: Spans, k: &SwizzleLanes) {
    assert!(half == 1 || half == 2, "swizzle: half 1 edo 2");
    sp.check(v.len());
    assert!(isa.is_available(), "{} ez dago makina honetan", isa.name());
    if sp.is_empty() {
        return;
    }
    match isa {
        // segurtasuna (hiru hauetan): isa-ren ezaugarriak makina honetan daude (goian egiaztatua)
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe { x86::swizzle_avx512(v, half, sp, &x86::swizzle_coefficients(k)) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { x86::swizzle_avx2(v, half, sp, &x86::swizzle_coefficients(k)) },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { arm::swizzle(v, half, sp, &arm::swizzle_coefficients(k)) },
        #[cfg(all(target_arch = "riscv64", not(target_feature = "v")))]
        // segurtasuna: lasterketak slice-aren barruan, V badago (goian egiaztatua)
        Isa::Rvv => unsafe { rv::swizzle(v.as_mut_ptr(), half, sp, k) },
        _ => scalar::swizzle(v, half, sp, k),
    }
}

pub fn scale_with(isa: Isa, v: &mut [C], sp: Spans, k: &ScaleLanes) {
    sp.check(v.len());
    assert!(isa.is_available(), "{} ez dago makina honetan", isa.name());
    if sp.is_empty() {
        return;
    }
    match isa {
        // segurtasuna (hiru hauetan): isa-ren ezaugarriak makina honetan daude (goian egiaztatua)
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe { x86::scale_avx512(v, sp, &x86::scale_coefficients(k)) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { x86::scale_avx2(v, sp, &x86::scale_coefficients(k)) },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { arm::scale(v, sp, &arm::scale_coefficients(k)) },
        #[cfg(all(target_arch = "riscv64", not(target_feature = "v")))]
        // segurtasuna: lasterketak slice-aren barruan, V badago (goian egiaztatua)
        Isa::Rvv => unsafe { rv::scale(v.as_mut_ptr(), sp, k) },
        _ => scalar::scale(v, sp, k),
    }
}

// erreferentzia: beste bertsio guztiak honen aurka egiaztatzen dira
mod scalar {
    use super::{C, Mat2Lanes, Pairs, ScaleLanes, Spans, SwizzleLanes, groups};

    pub fn mat2(p: impl Pairs, sp: Spans, k: &Mat2Lanes) {
        p.each(sp, |x, y| {
            for j in 0..4 {
                let m = &k[j];
                let (a, b) = (x[j], y[j]);
                x[j] = m[0] * a + m[1] * b;
                y[j] = m[2] * a + m[3] * b;
            }
        });
    }

    pub fn swizzle(v: &mut [C], half: usize, sp: Spans, k: &SwizzleLanes) {
        groups(v, sp, |x| {
            let old = *x;
            for (j, kj) in k.iter().enumerate() {
                x[j] = kj[0] * old[j] + kj[1] * old[j ^ half];
            }
        });
    }

    pub fn scale(v: &mut [C], sp: Spans, k: &ScaleLanes) {
        groups(v, sp, |x| {
            for j in 0..4 {
                x[j] *= k[j];
            }
        });
    }
}

// ------------------------------------------------------------------ x86-64: avx-512 eta avx2 + fma
//
// konplexuen biderketa c * x, x = [re, im] tartekatuta:
//   swap(x) = [im, re]                  (vpermilpd)
//   t = swap(x) * c.im
//   c * x = fmaddsub(x, c.re, t)        posizio bikoitietan x*c.re - t, bakoitietan x*c.re + t
// 2x2 matrizean bi biderketak batera: lo' = m00*lo + m01*hi
//   t = swap(lo)*m00.im + swap(hi)*m01.im;  lo' = fmaddsub(lo, m00.re, t) + hi*m01.re
//
// koefizienteak bektore osoka (8 f64 = 4 erreika, bakoitzaren balioa bi aldiz: re,re edo im,im). avx2-k
// erdi baxua (0-1 erreikak) eta erdi altua (2-3) bereiz erabiltzen ditu.

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{C, Mat2Lanes, Pairs, ScaleLanes, Spans, SwizzleLanes, groups};
    use std::arch::x86_64::*;

    pub struct Coef<const N: usize>([[f64; 8]; N]);

    // sarrera-zerrenda bakoitzeko bi bektore (re bikoiztua, im bikoiztua)
    fn layout<const N: usize, const E: usize>(entries: impl Fn(usize) -> [C; E]) -> Coef<N> {
        let mut out = [[0.0; 8]; N];
        for lane in 0..4 {
            for (e, c) in entries(lane).iter().enumerate() {
                out[2 * e][2 * lane] = c.re;
                out[2 * e][2 * lane + 1] = c.re;
                out[2 * e + 1][2 * lane] = c.im;
                out[2 * e + 1][2 * lane + 1] = c.im;
            }
        }
        Coef(out)
    }

    pub fn mat2_coefficients(k: &Mat2Lanes) -> Coef<8> {
        layout::<8, 4>(|l| k[l])
    }

    pub fn swizzle_coefficients(k: &SwizzleLanes) -> Coef<4> {
        layout::<4, 2>(|l| k[l])
    }

    pub fn scale_coefficients(k: &ScaleLanes) -> Coef<2> {
        layout::<2, 1>(|l| [k[l]])
    }

    // ---- kargak eta gordeketak: erreferentzia batek bektore oso baten memoria ematen du
    //
    // segurtasuna (lau hauetan): `&[C; N]` erreferentzia batek N konplexu irakurgarri (eta `&mut`-ek
    // idazgarri) bermatzen ditu, hau da zehazki bektorearen 16 * N byteak (`Complex64` #[repr(C)] da, bi
    // f64), eta loadu/storeu-k ez du lerrokatzerik behar. ez dago beste erakuslerik: indize eta muga guztiak
    // rust seguruan doaz

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn ld8(x: &[C; 4]) -> __m512d {
        unsafe { _mm512_loadu_pd(x.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn st8(x: &mut [C; 4], v: __m512d) {
        unsafe { _mm512_storeu_pd(x.as_mut_ptr().cast(), v) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    fn ld4(x: &[C; 2]) -> __m256d {
        unsafe { _mm256_loadu_pd(x.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    fn st4(x: &mut [C; 2], v: __m256d) {
        unsafe { _mm256_storeu_pd(x.as_mut_ptr().cast(), v) }
    }

    // talde bat bi erreika-bikotetan (0-1, 2-3)
    #[inline(always)]
    fn halves(x: &mut [C; 4]) -> &mut [[C; 2]; 2] {
        let (h, _) = x.as_chunks_mut::<2>();
        h.try_into().expect("4 = 2 + 2")
    }

    // ---- avx-512: zmm batek talde oso bat (4 anplitude)

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn coef8(k: &[f64; 8]) -> __m512d {
        _mm512_setr_pd(k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7])
    }

    // lo' = m0 x + m1 y, koefizienteak [m0.re, m0.im, m1.re, m1.im] zmm gisa
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn mul2_512(x: __m512d, y: __m512d, c: &[__m512d]) -> __m512d {
        let xs = _mm512_permute_pd::<0x55>(x);
        let ys = _mm512_permute_pd::<0x55>(y);
        let t = _mm512_fmadd_pd(ys, c[3], _mm512_mul_pd(xs, c[1]));
        let t = _mm512_fmaddsub_pd(x, c[0], t);
        _mm512_fmadd_pd(y, c[2], t)
    }

    #[target_feature(enable = "avx512f")]
    pub fn mat2_avx512(p: impl Pairs, sp: Spans, k: &Coef<8>) {
        let c: [__m512d; 8] = std::array::from_fn(|i| coef8(&k.0[i]));
        p.each(sp, |x4, y4| {
            let (x, y) = (ld8(x4), ld8(y4));
            st8(x4, mul2_512(x, y, &c[..4]));
            st8(y4, mul2_512(x, y, &c[4..]));
        });
    }

    // bikotekidea vshuff64x2-rekin: half 1 -> 128 biteko erdiak [1 0 3 2] (0xb1), half 2 -> [2 3 0 1] (0x4e)
    #[target_feature(enable = "avx512f")]
    pub fn swizzle_avx512(v: &mut [C], half: usize, sp: Spans, k: &Coef<4>) {
        let c: [__m512d; 4] = std::array::from_fn(|i| coef8(&k.0[i]));
        groups(v, sp, |x4| {
            let x = ld8(x4);
            let y = if half == 1 {
                _mm512_shuffle_f64x2::<0xb1>(x, x)
            } else {
                _mm512_shuffle_f64x2::<0x4e>(x, x)
            };
            st8(x4, mul2_512(x, y, &c));
        });
    }

    #[target_feature(enable = "avx512f")]
    pub fn scale_avx512(v: &mut [C], sp: Spans, k: &Coef<2>) {
        let (re, im) = (coef8(&k.0[0]), coef8(&k.0[1]));
        groups(v, sp, |x4| {
            let x = ld8(x4);
            let t = _mm512_mul_pd(_mm512_permute_pd::<0x55>(x), im);
            st8(x4, _mm512_fmaddsub_pd(x, re, t));
        });
    }

    // ---- avx2: ymm batek 2 anplitude (erreika-bikote bat)

    // koefiziente-errenkada baten erdia: o = 0 (0-1 erreikak) edo 4 (2-3)
    #[inline]
    #[target_feature(enable = "avx")]
    fn coef4(k: &[f64; 8], o: usize) -> __m256d {
        _mm256_setr_pd(k[o], k[o + 1], k[o + 2], k[o + 3])
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn mul2_256(x: __m256d, y: __m256d, c: &[__m256d]) -> __m256d {
        let xs = _mm256_permute_pd::<0x5>(x);
        let ys = _mm256_permute_pd::<0x5>(y);
        let t = _mm256_fmadd_pd(ys, c[3], _mm256_mul_pd(xs, c[1]));
        let t = _mm256_fmaddsub_pd(x, c[0], t);
        _mm256_fmadd_pd(y, c[2], t)
    }

    #[target_feature(enable = "avx2,fma")]
    pub fn mat2_avx2(p: impl Pairs, sp: Spans, k: &Coef<8>) {
        let c: [[__m256d; 8]; 2] =
            std::array::from_fn(|h| std::array::from_fn(|i| coef4(&k.0[i], 4 * h)));
        p.each(sp, |x4, y4| {
            let (xh, yh) = (halves(x4), halves(y4));
            for (h, c) in c.iter().enumerate() {
                let (x, y) = (ld4(&xh[h]), ld4(&yh[h]));
                st4(&mut xh[h], mul2_256(x, y, &c[..4]));
                st4(&mut yh[h], mul2_256(x, y, &c[4..]));
            }
        });
    }

    // talde bat bi ymm-tan (0-1 erreikak, 2-3). bikotekideak: half 1 -> ymm bakoitzaren erdiak trukatuta
    // (vperm2f128); half 2 -> beste ymm-a bera
    #[target_feature(enable = "avx2,fma")]
    pub fn swizzle_avx2(v: &mut [C], half: usize, sp: Spans, k: &Coef<4>) {
        let c: [[__m256d; 4]; 2] =
            std::array::from_fn(|h| std::array::from_fn(|i| coef4(&k.0[i], 4 * h)));
        groups(v, sp, |x4| {
            let xh = halves(x4);
            let (a, b) = (ld4(&xh[0]), ld4(&xh[1]));
            let (pa, pb) = if half == 1 {
                (
                    _mm256_permute2f128_pd::<0x01>(a, a),
                    _mm256_permute2f128_pd::<0x01>(b, b),
                )
            } else {
                (b, a)
            };
            st4(&mut xh[0], mul2_256(a, pa, &c[0]));
            st4(&mut xh[1], mul2_256(b, pb, &c[1]));
        });
    }

    #[target_feature(enable = "avx2,fma")]
    pub fn scale_avx2(v: &mut [C], sp: Spans, k: &Coef<2>) {
        let c: [[__m256d; 2]; 2] =
            std::array::from_fn(|h| std::array::from_fn(|i| coef4(&k.0[i], 4 * h)));
        groups(v, sp, |x4| {
            let xh = halves(x4);
            for (h, c) in c.iter().enumerate() {
                let x = ld4(&xh[h]);
                let t = _mm256_mul_pd(_mm256_permute_pd::<0x5>(x), c[1]);
                st4(&mut xh[h], _mm256_fmaddsub_pd(x, c[0], t));
            }
        });
    }
}

// ------------------------------------------------------------------ aarch64: neon
//
// bi konplexu re eta im bereizita kargatzen dira (ld2: re = [re0 re1], im = [im0 im1]). talde bat
// (4 anplitude) bi zatitan: 0-1 erreikak eta 2-3 erreikak, bakoitzak bere koefizienteekin.
//   lo'.re = m00r*lr - m00i*li + m01r*hr - m01i*hi       lo'.im = m00r*li + m00i*lr + m01r*hi + m01i*hr
// biderketa-batuketa bakoitza fusionatua da (fmla, fmls), mihiztadurako bertsio zaharrean bezala.

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{C, Mat2Lanes, Pairs, ScaleLanes, Spans, SwizzleLanes, groups};
    use std::arch::aarch64::*;

    // erreika-bikote bakoitzeko (0-1, 2-3), sarrera bakoitzeko re eta im bektoreak: [erdia][sarrera][erreika]
    pub struct Coef<const N: usize>([[[f64; 2]; N]; 2]);

    fn layout<const N: usize, const E: usize>(entries: impl Fn(usize) -> [C; E]) -> Coef<N> {
        let mut out = [[[0.0; 2]; N]; 2];
        for lane in 0..4 {
            for (e, c) in entries(lane).iter().enumerate() {
                out[lane / 2][2 * e][lane % 2] = c.re;
                out[lane / 2][2 * e + 1][lane % 2] = c.im;
            }
        }
        Coef(out)
    }

    pub fn mat2_coefficients(k: &Mat2Lanes) -> Coef<8> {
        layout::<8, 4>(|l| k[l])
    }

    pub fn swizzle_coefficients(k: &SwizzleLanes) -> Coef<4> {
        layout::<4, 2>(|l| k[l])
    }

    pub fn scale_coefficients(k: &ScaleLanes) -> Coef<2> {
        layout::<2, 1>(|l| [k[l]])
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn pair(x: &[f64; 2]) -> float64x2_t {
        vsetq_lane_f64::<1>(x[1], vdupq_n_f64(x[0]))
    }

    // bi konplexu -> (re, im), ld2-k bezala, eta alderantziz
    //
    // segurtasuna (bi hauetan): `&[C; 2]` erreferentzia batek bi konplexu irakurgarri (eta `&mut`-ek
    // idazgarri) bermatzen ditu, hau da zehazki ld2/st2-k ukitzen dituen 32 byteak (`Complex64` #[repr(C)]
    // da, bi f64). ez dago beste erakuslerik: indize eta muga guztiak rust seguruan doaz
    #[inline]
    #[target_feature(enable = "neon")]
    fn ld2(x: &[C; 2]) -> (float64x2_t, float64x2_t) {
        let v = unsafe { vld2q_f64(x.as_ptr().cast()) };
        (v.0, v.1)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn st2(x: &mut [C; 2], (re, im): (float64x2_t, float64x2_t)) {
        unsafe { vst2q_f64(x.as_mut_ptr().cast(), float64x2x2_t(re, im)) }
    }

    // talde bat bi erreika-bikotetan (0-1, 2-3)
    #[inline(always)]
    fn halves(x: &mut [C; 4]) -> &mut [[C; 2]; 2] {
        let (h, _) = x.as_chunks_mut::<2>();
        h.try_into().expect("4 = 2 + 2")
    }

    // a x + b y, koefizienteak [a.re, a.im, b.re, b.im]: (re, im)
    #[inline]
    #[target_feature(enable = "neon")]
    fn mul2(
        (xr, xi): (float64x2_t, float64x2_t),
        (yr, yi): (float64x2_t, float64x2_t),
        c: &[float64x2_t],
    ) -> (float64x2_t, float64x2_t) {
        let re = vmulq_f64(xr, c[0]);
        let re = vfmsq_f64(re, xi, c[1]);
        let re = vfmaq_f64(re, yr, c[2]);
        let re = vfmsq_f64(re, yi, c[3]);
        let im = vmulq_f64(xi, c[0]);
        let im = vfmaq_f64(im, xr, c[1]);
        let im = vfmaq_f64(im, yi, c[2]);
        let im = vfmaq_f64(im, yr, c[3]);
        (re, im)
    }

    #[target_feature(enable = "neon")]
    pub fn mat2(p: impl Pairs, sp: Spans, k: &Coef<8>) {
        let c: [[float64x2_t; 8]; 2] =
            std::array::from_fn(|h| std::array::from_fn(|i| pair(&k.0[h][i])));
        p.each(sp, |x4, y4| {
            let (xh, yh) = (halves(x4), halves(y4));
            for (h, c) in c.iter().enumerate() {
                let (x, y) = (ld2(&xh[h]), ld2(&yh[h]));
                st2(&mut xh[h], mul2(x, y, &c[..4]));
                st2(&mut yh[h], mul2(x, y, &c[4..]));
            }
        });
    }

    // bikotekidea: half 1 -> erregistro beraren beste erdia (ext #8), half 2 -> beste erreika-bikotea
    #[target_feature(enable = "neon")]
    pub fn swizzle(v: &mut [C], half: usize, sp: Spans, k: &Coef<4>) {
        let c: [[float64x2_t; 4]; 2] =
            std::array::from_fn(|h| std::array::from_fn(|i| pair(&k.0[h][i])));
        groups(v, sp, |x4| {
            let xh = halves(x4);
            let (a, b) = (ld2(&xh[0]), ld2(&xh[1]));
            let swap =
                |(r, i): (float64x2_t, float64x2_t)| (vextq_f64::<1>(r, r), vextq_f64::<1>(i, i));
            let (pa, pb) = if half == 1 {
                (swap(a), swap(b))
            } else {
                (b, a)
            };
            st2(&mut xh[0], mul2(a, pa, &c[0]));
            st2(&mut xh[1], mul2(b, pb, &c[1]));
        });
    }

    #[target_feature(enable = "neon")]
    pub fn scale(v: &mut [C], sp: Spans, k: &Coef<2>) {
        let c: [[float64x2_t; 2]; 2] =
            std::array::from_fn(|h| std::array::from_fn(|i| pair(&k.0[h][i])));
        groups(v, sp, |x4| {
            let xh = halves(x4);
            for (h, c) in c.iter().enumerate() {
                let (xr, xi) = ld2(&xh[h]);
                let re = vfmsq_f64(vmulq_f64(xr, c[0]), xi, c[1]);
                let im = vfmaq_f64(vmulq_f64(xi, c[0]), xr, c[1]);
                st2(&mut xh[h], (re, im));
            }
        });
    }
}

// ------------------------------------------------------------------ riscv64: rvv 1.0
//
// bektore-luzeraren menpekotasunik gabe (vsetvli): edozein VLEN-ekin dabil. erreika-koefizienteak ez datoz
// bat bektore-luzera aldakorrarekin, beraz erreika bakoitza (edo erreika-bikote bakoitza swizzle-n) bere
// pasaldian prozesatzen da, 4 anplitudeko pausoarekin (vlsseg2e64, segmentu-karga pausoduna); koefizienteak
// erreika guztietan berdinak direnean (ohikoena), lasterketa jarraituak pasaldi bakarrean (vlseg2e64).
// vlseg2e64-k konplexuak re eta im taldetan banatzen ditu. LMUL=2: lo.re v0-v1, lo.im v2-v3, hi.re v4-v5,
// hi.im v6-v7; emaitzak v8-v15.
//
// rust egonkorrean `v` ezaugarria ez dago oraindik (#[target_feature] gabe), beraz instrukzioak
// `.option arch, +v`-rekin mihiztatzen dira eta exekuzio-garaian egiaztatzen da prozesadoreak badituela
// (AT_HWCAP-eko V bita). bektore-erregistroak ezin dira clobber gisa adierazi `v` gabe; ez da beharrezkoa,
// konpilatzaileak ez baititu erabiltzen crate-a `v` gabe konpilatzen denean (eta `v`-rekin konpilatzen
// bada, modulu hau ez da erabiltzen: llvm-k berak bektorizatzen du kode eskalarra).

#[cfg(all(target_arch = "riscv64", not(target_feature = "v")))]
mod rv {
    use super::{C, Mat2Lanes, ScaleLanes, Spans, SwizzleLanes};
    use std::arch::asm;

    // linux-ek prozesadorearen ezaugarriak auxv-n ematen ditu: AT_HWCAP (16) gakoaren balioan, letra
    // bakoitzeko bit bat ('V' - 'A' = 21)
    pub fn has_vector() -> bool {
        let Ok(auxv) = std::fs::read("/proc/self/auxv") else {
            return false;
        };
        auxv.as_chunks::<16>().0.iter().any(|e| {
            let key = u64::from_ne_bytes(e[..8].try_into().unwrap());
            let val = u64::from_ne_bytes(e[8..].try_into().unwrap());
            key == 16 && val & (1 << 21) != 0
        })
    }

    // `n` bikote: lo[k * step], hi[k * step] (step bytetan; 16 = jarraian). m matrize bakarra
    // segurtasuna: deitzaileak egiaztatzen du bikote guztiak memorian daudela eta V badagoela
    unsafe fn mat2_strided(lo: *mut C, hi: *mut C, n: usize, step: usize, m: &[C; 4]) {
        unsafe {
            asm!(
                ".option push",
                ".option arch, +v",
                "2:",
                "vsetvli {vl}, {n}, e64, m2, ta, ma",
                "vlsseg2e64.v v0, ({a}), {step}",
                "vlsseg2e64.v v4, ({b}), {step}",
                "vfmul.vf v8, v0, {m00r}",
                "vfnmsac.vf v8, {m00i}, v2",
                "vfmacc.vf v8, {m01r}, v4",
                "vfnmsac.vf v8, {m01i}, v6",
                "vfmul.vf v10, v2, {m00r}",
                "vfmacc.vf v10, {m00i}, v0",
                "vfmacc.vf v10, {m01r}, v6",
                "vfmacc.vf v10, {m01i}, v4",
                "vfmul.vf v12, v0, {m10r}",
                "vfnmsac.vf v12, {m10i}, v2",
                "vfmacc.vf v12, {m11r}, v4",
                "vfnmsac.vf v12, {m11i}, v6",
                "vfmul.vf v14, v2, {m10r}",
                "vfmacc.vf v14, {m10i}, v0",
                "vfmacc.vf v14, {m11r}, v6",
                "vfmacc.vf v14, {m11i}, v4",
                "vssseg2e64.v v8, ({a}), {step}",
                "vssseg2e64.v v12, ({b}), {step}",
                "mul {t}, {vl}, {step}",
                "add {a}, {a}, {t}",
                "add {b}, {b}, {t}",
                "sub {n}, {n}, {vl}",
                "bnez {n}, 2b",
                ".option pop",
                m00r = in(freg) m[0].re, m00i = in(freg) m[0].im,
                m01r = in(freg) m[1].re, m01i = in(freg) m[1].im,
                m10r = in(freg) m[2].re, m10i = in(freg) m[2].im,
                m11r = in(freg) m[3].re, m11i = in(freg) m[3].im,
                a = inout(reg) lo => _,
                b = inout(reg) hi => _,
                n = inout(reg) n => _,
                step = in(reg) step,
                vl = out(reg) _,
                t = out(reg) _,
                options(nostack),
            );
        }
    }

    // `n` elementu: p[k * step] *= d
    unsafe fn scale_strided(p: *mut C, n: usize, step: usize, d: C) {
        unsafe {
            asm!(
                ".option push",
                ".option arch, +v",
                "2:",
                "vsetvli {vl}, {n}, e64, m2, ta, ma",
                "vlsseg2e64.v v0, ({p}), {step}",
                "vfmul.vf v8, v0, {dr}",
                "vfnmsac.vf v8, {di}, v2",
                "vfmul.vf v10, v2, {dr}",
                "vfmacc.vf v10, {di}, v0",
                "vssseg2e64.v v8, ({p}), {step}",
                "mul {t}, {vl}, {step}",
                "add {p}, {p}, {t}",
                "sub {n}, {n}, {vl}",
                "bnez {n}, 2b",
                ".option pop",
                dr = in(freg) d.re,
                di = in(freg) d.im,
                p = inout(reg) p => _,
                n = inout(reg) n => _,
                step = in(reg) step,
                vl = out(reg) _,
                t = out(reg) _,
                options(nostack),
            );
        }
    }

    fn uniform<T: PartialEq>(k: &[T; 4]) -> bool {
        k.iter().all(|x| *x == k[0])
    }

    pub unsafe fn mat2(lo: *mut C, hi: *mut C, sp: Spans, k: &Mat2Lanes) {
        for s in 0..sp.count {
            let off = sp.start + s * sp.stride;
            unsafe {
                if uniform(k) {
                    mat2_strided(lo.add(off), hi.add(off), sp.len, 16, &k[0]);
                } else {
                    for (j, m) in k.iter().enumerate() {
                        mat2_strided(lo.add(off + j), hi.add(off + j), sp.len / 4, 64, m);
                    }
                }
            }
        }
    }

    // erreika-bikoteak (j, j ^ half): j-k lo betetzen du eta j ^ half-ek hi; matrizea [a_j b_j; b_h a_h]
    pub unsafe fn swizzle(p: *mut C, half: usize, sp: Spans, k: &SwizzleLanes) {
        for s in 0..sp.count {
            let off = sp.start + s * sp.stride;
            for j in (0..4).filter(|j| j & half == 0) {
                let h = j | half;
                let m = [k[j][0], k[j][1], k[h][1], k[h][0]];
                unsafe { mat2_strided(p.add(off + j), p.add(off + h), sp.len / 4, 64, &m) };
            }
        }
    }

    pub unsafe fn scale(p: *mut C, sp: Spans, k: &ScaleLanes) {
        for s in 0..sp.count {
            let off = sp.start + s * sp.stride;
            unsafe {
                if uniform(k) {
                    scale_strided(p.add(off), sp.len, 16, k[0]);
                } else {
                    for (j, d) in k.iter().enumerate() {
                        scale_strided(p.add(off + j), sp.len / 4, 64, *d);
                    }
                }
            }
        }
    }
}
