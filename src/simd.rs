// fitxategi honek simulagailuaren barne-begizta beroenak ditu, arkitektura bakoitzeko eskuz idatzitako
// mihiztaduran (`asm!`): avx-512 eta avx2+fma (x86-64), neon (aarch64), rvv 1.0 (riscv64), eta
// erreferentziazko bertsio eskalarra. makinak duen onena exekuzio-garaian aukeratzen da behin;
// `QOA_SIMD=scalar` (edo avx2, avx512, neon, rvv) ingurune-aldagaiak bat behartzen du.
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
// egiten da; neon eta rvv-k re eta im banatzen dituzte kargatzean (ld2, vlseg2). probek bertsio guztiak
// eskalarraren aurka egiaztatzen dituzte (tests/simd.rs), aarch64 eta riscv64 qemu-n barne.

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
    if sp.is_empty() {
        return;
    }
    // segurtasuna: lasterketa guztiak bi slice-en barruan daude (goian egiaztatua)
    unsafe { mat2_raw(isa, lo.as_mut_ptr(), hi.as_mut_ptr(), sp, k) }
}

pub fn mat2_in_with(isa: Isa, v: &mut [C], hi_off: usize, sp: Spans, k: &Mat2Lanes) {
    sp.check(v.len());
    sp.check(v.len().saturating_sub(hi_off));
    if sp.is_empty() {
        return;
    }
    let p = v.as_mut_ptr();
    // segurtasuna: lo eta hi (hi_off aurrerago) lasterketak slice-aren barruan daude (goian egiaztatua)
    unsafe { mat2_raw(isa, p, p.add(hi_off), sp, k) }
}

pub fn swizzle_with(isa: Isa, v: &mut [C], half: usize, sp: Spans, k: &SwizzleLanes) {
    assert!(half == 1 || half == 2, "swizzle: half 1 edo 2");
    sp.check(v.len());
    assert!(isa.is_available(), "{} ez dago makina honetan", isa.name());
    if sp.is_empty() {
        return;
    }
    let p = v.as_mut_ptr();
    // segurtasuna: lasterketak slice-aren barruan, isa makina honetan (goian egiaztatua)
    unsafe {
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => x86::swizzle_avx512(p, half, sp, &x86::swizzle_coefficients(k)),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => x86::swizzle_avx2(p, half, sp, &x86::swizzle_coefficients(k)),
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => arm::swizzle(p, half, sp, &arm::swizzle_coefficients(k)),
            #[cfg(all(target_arch = "riscv64", not(target_feature = "v")))]
            Isa::Rvv => rv::swizzle(p, half, sp, k),
            _ => scalar::swizzle(p, half, sp, k),
        }
    }
}

pub fn scale_with(isa: Isa, v: &mut [C], sp: Spans, k: &ScaleLanes) {
    sp.check(v.len());
    assert!(isa.is_available(), "{} ez dago makina honetan", isa.name());
    if sp.is_empty() {
        return;
    }
    let p = v.as_mut_ptr();
    // segurtasuna: lasterketak slice-aren barruan, isa makina honetan (goian egiaztatua)
    unsafe {
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => x86::scale_avx512(p, sp, &x86::scale_coefficients(k)),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => x86::scale_avx2(p, sp, &x86::scale_coefficients(k)),
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => arm::scale(p, sp, &arm::scale_coefficients(k)),
            #[cfg(all(target_arch = "riscv64", not(target_feature = "v")))]
            Isa::Rvv => rv::scale(p, sp, k),
            _ => scalar::scale(p, sp, k),
        }
    }
}

// segurtasuna: deitzaileak egiaztatzen du lasterketak `lo` eta `hi`-tik hasitako memorian daudela
unsafe fn mat2_raw(isa: Isa, lo: *mut C, hi: *mut C, sp: Spans, k: &Mat2Lanes) {
    assert!(isa.is_available(), "{} ez dago makina honetan", isa.name());
    unsafe {
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => x86::mat2_avx512(lo, hi, sp, &x86::mat2_coefficients(k)),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => x86::mat2_avx2(lo, hi, sp, &x86::mat2_coefficients(k)),
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => arm::mat2(lo, hi, sp, &arm::mat2_coefficients(k)),
            #[cfg(all(target_arch = "riscv64", not(target_feature = "v")))]
            Isa::Rvv => rv::mat2(lo, hi, sp, k),
            _ => scalar::mat2(lo, hi, sp, k),
        }
    }
}

// erreferentzia: beste bertsio guztiak honen aurka egiaztatzen dira. erakusle gordinekin, bertsio
// guztiek interfaze bera izan dezaten; deitzaileak egiaztatzen ditu mugak
mod scalar {
    use super::{C, Mat2Lanes, ScaleLanes, Spans, SwizzleLanes};

    pub unsafe fn mat2(lo: *mut C, hi: *mut C, sp: Spans, k: &Mat2Lanes) {
        for s in 0..sp.count {
            for j in 0..sp.len {
                let i = sp.start + s * sp.stride + j;
                let m = &k[j & 3];
                // segurtasuna: i lasterketen barruan dago (deitzaileak egiaztatua)
                unsafe {
                    let (x, y) = (*lo.add(i), *hi.add(i));
                    *lo.add(i) = m[0] * x + m[1] * y;
                    *hi.add(i) = m[2] * x + m[3] * y;
                }
            }
        }
    }

    pub unsafe fn swizzle(p: *mut C, half: usize, sp: Spans, k: &SwizzleLanes) {
        for s in 0..sp.count {
            let base = sp.start + s * sp.stride;
            for g in (0..sp.len).step_by(4) {
                // segurtasuna: 4ko taldea lasterketaren barruan dago (len 4ren multiploa)
                unsafe {
                    let x: [C; 4] = std::array::from_fn(|j| *p.add(base + g + j));
                    for (j, kj) in k.iter().enumerate() {
                        *p.add(base + g + j) = kj[0] * x[j] + kj[1] * x[j ^ half];
                    }
                }
            }
        }
    }

    pub unsafe fn scale(p: *mut C, sp: Spans, k: &ScaleLanes) {
        for s in 0..sp.count {
            for j in 0..sp.len {
                let i = sp.start + s * sp.stride + j;
                // segurtasuna: i lasterketen barruan dago
                unsafe { *p.add(i) *= k[j & 3] }
            }
        }
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
// koefizienteak memorian bektore osoka (8 f64 = 4 erreika, bakoitzaren balioa bi aldiz: re,re edo im,im).
// avx-512-k zmm erregistroetan gordetzen ditu; avx2-k (16 erregistro bakarrik) memoriatik irakurtzen ditu
// eragigai gisa, erdi baxua (0-1 erreikak) eta erdi altua (2-3) bereiz.

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{C, Mat2Lanes, ScaleLanes, Spans, SwizzleLanes};
    use std::arch::asm;

    #[repr(C, align(64))]
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

    // segurtasuna (guztietan): avx-512f / avx2+fma badaude, eta lasterketak erakusleen memorian daude
    // (deitzaileak egiaztatua). len 4ren multiploa eta > 0, count > 0
    #[target_feature(enable = "avx512f")]
    pub unsafe fn mat2_avx512(lo: *mut C, hi: *mut C, sp: Spans, k: &Coef<8>) {
        unsafe {
            asm!(
                "vmovapd zmm8, zmmword ptr [{k}]",
                "vmovapd zmm9, zmmword ptr [{k} + 64]",
                "vmovapd zmm10, zmmword ptr [{k} + 128]",
                "vmovapd zmm11, zmmword ptr [{k} + 192]",
                "vmovapd zmm12, zmmword ptr [{k} + 256]",
                "vmovapd zmm13, zmmword ptr [{k} + 320]",
                "vmovapd zmm14, zmmword ptr [{k} + 384]",
                "vmovapd zmm15, zmmword ptr [{k} + 448]",
                "3:",
                "mov {a}, {pa}",
                "mov {b}, {pb}",
                "mov {n}, {len}",
                "2:",
                "vmovupd zmm0, zmmword ptr [{a}]",
                "vmovupd zmm1, zmmword ptr [{b}]",
                "vpermilpd zmm2, zmm0, 0x55",
                "vpermilpd zmm3, zmm1, 0x55",
                "vmulpd zmm4, zmm2, zmm9",
                "vfmadd231pd zmm4, zmm3, zmm11",
                "vfmaddsub231pd zmm4, zmm0, zmm8",
                "vfmadd231pd zmm4, zmm1, zmm10",
                "vmulpd zmm5, zmm2, zmm13",
                "vfmadd231pd zmm5, zmm3, zmm15",
                "vfmaddsub231pd zmm5, zmm0, zmm12",
                "vfmadd231pd zmm5, zmm1, zmm14",
                "vmovupd zmmword ptr [{a}], zmm4",
                "vmovupd zmmword ptr [{b}], zmm5",
                "add {a}, 64",
                "add {b}, 64",
                "sub {n}, 4",
                "jnz 2b",
                "add {pa}, {stride}",
                "add {pb}, {stride}",
                "dec {cnt}",
                "jnz 3b",
                "vzeroupper",
                k = in(reg) k.0.as_ptr(),
                pa = inout(reg) lo.add(sp.start) => _,
                pb = inout(reg) hi.add(sp.start) => _,
                len = in(reg) sp.len,
                stride = in(reg) sp.stride * 16,
                cnt = inout(reg) sp.count => _,
                a = out(reg) _,
                b = out(reg) _,
                n = out(reg) _,
                out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _, out("zmm4") _,
                out("zmm5") _, out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
                out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
                options(nostack),
            );
        }
    }

    // 2x2 matrizearen aritmetika ymm-tan: sarrerak ymm0 (lo) eta ymm1 (hi), koefizienteak [k + o + ...]
    // (o = 0 erreika 0-1etarako, 32 erreika 2-3etarako), emaitzak ymm4 eta ymm5
    macro_rules! avx2_mat2_math {
        ($o:literal) => {
            concat!(
                "vpermilpd ymm2, ymm0, 0x5\n",
                "vpermilpd ymm3, ymm1, 0x5\n",
                "vmulpd ymm4, ymm2, ymmword ptr [{k} + 64 + ",
                $o,
                "]\n",
                "vfmadd231pd ymm4, ymm3, ymmword ptr [{k} + 192 + ",
                $o,
                "]\n",
                "vfmaddsub231pd ymm4, ymm0, ymmword ptr [{k} + ",
                $o,
                "]\n",
                "vfmadd231pd ymm4, ymm1, ymmword ptr [{k} + 128 + ",
                $o,
                "]\n",
                "vmulpd ymm5, ymm2, ymmword ptr [{k} + 320 + ",
                $o,
                "]\n",
                "vfmadd231pd ymm5, ymm3, ymmword ptr [{k} + 448 + ",
                $o,
                "]\n",
                "vfmaddsub231pd ymm5, ymm0, ymmword ptr [{k} + 256 + ",
                $o,
                "]\n",
                "vfmadd231pd ymm5, ymm1, ymmword ptr [{k} + 384 + ",
                $o,
                "]\n",
            )
        };
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn mat2_avx2(lo: *mut C, hi: *mut C, sp: Spans, k: &Coef<8>) {
        unsafe {
            asm!(
                "3:",
                "mov {a}, {pa}",
                "mov {b}, {pb}",
                "mov {n}, {len}",
                "2:",
                "vmovupd ymm0, ymmword ptr [{a}]",
                "vmovupd ymm1, ymmword ptr [{b}]",
                avx2_mat2_math!("0"),
                "vmovupd ymmword ptr [{a}], ymm4",
                "vmovupd ymmword ptr [{b}], ymm5",
                "vmovupd ymm0, ymmword ptr [{a} + 32]",
                "vmovupd ymm1, ymmword ptr [{b} + 32]",
                avx2_mat2_math!("32"),
                "vmovupd ymmword ptr [{a} + 32], ymm4",
                "vmovupd ymmword ptr [{b} + 32], ymm5",
                "add {a}, 64",
                "add {b}, 64",
                "sub {n}, 4",
                "jnz 2b",
                "add {pa}, {stride}",
                "add {pb}, {stride}",
                "dec {cnt}",
                "jnz 3b",
                "vzeroupper",
                k = in(reg) k.0.as_ptr(),
                pa = inout(reg) lo.add(sp.start) => _,
                pb = inout(reg) hi.add(sp.start) => _,
                len = in(reg) sp.len,
                stride = in(reg) sp.stride * 16,
                cnt = inout(reg) sp.count => _,
                a = out(reg) _,
                b = out(reg) _,
                n = out(reg) _,
                out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _, out("ymm4") _,
                out("ymm5") _,
                options(nostack),
            );
        }
    }

    // swizzle avx-512: zmm batek 4 anplitude (talde oso bat); bikotekidea vshuff64x2-rekin:
    // half 1 -> 128 biteko erdiak [1 0 3 2] (0xb1), half 2 -> [2 3 0 1] (0x4e)
    macro_rules! avx512_swizzle {
        ($imm:literal, $p:expr, $sp:expr, $k:expr) => {
            asm!(
                "vmovapd zmm8, zmmword ptr [{k}]",
                "vmovapd zmm9, zmmword ptr [{k} + 64]",
                "vmovapd zmm10, zmmword ptr [{k} + 128]",
                "vmovapd zmm11, zmmword ptr [{k} + 192]",
                "3:",
                "mov {a}, {pa}",
                "mov {n}, {len}",
                "2:",
                "vmovupd zmm0, zmmword ptr [{a}]",
                concat!("vshuff64x2 zmm1, zmm0, zmm0, ", $imm),
                "vpermilpd zmm2, zmm0, 0x55",
                "vpermilpd zmm3, zmm1, 0x55",
                "vmulpd zmm4, zmm2, zmm9",
                "vfmadd231pd zmm4, zmm3, zmm11",
                "vfmaddsub231pd zmm4, zmm0, zmm8",
                "vfmadd231pd zmm4, zmm1, zmm10",
                "vmovupd zmmword ptr [{a}], zmm4",
                "add {a}, 64",
                "sub {n}, 4",
                "jnz 2b",
                "add {pa}, {stride}",
                "dec {cnt}",
                "jnz 3b",
                "vzeroupper",
                k = in(reg) $k.0.as_ptr(),
                pa = inout(reg) $p.add($sp.start) => _,
                len = in(reg) $sp.len,
                stride = in(reg) $sp.stride * 16,
                cnt = inout(reg) $sp.count => _,
                a = out(reg) _,
                n = out(reg) _,
                out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _, out("zmm4") _,
                out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
                options(nostack),
            )
        };
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn swizzle_avx512(p: *mut C, half: usize, sp: Spans, k: &Coef<4>) {
        unsafe {
            if half == 1 {
                avx512_swizzle!("0xb1", p, sp, k);
            } else {
                avx512_swizzle!("0x4e", p, sp, k);
            }
        }
    }

    // swizzle avx2: talde bat bi ymm-tan (ymm0 = 0-1 erreikak, ymm6 = 2-3). bikotekideak: half 1 -> ymm
    // bakoitzaren erdiak trukatuta (vperm2f128); half 2 -> beste ymm-a bera
    macro_rules! avx2_swizzle_math {
        ($x:literal, $y:literal, $o:literal) => {
            concat!(
                "vpermilpd ymm2, ",
                $x,
                ", 0x5\n",
                "vpermilpd ymm3, ",
                $y,
                ", 0x5\n",
                "vmulpd ymm4, ymm2, ymmword ptr [{k} + 64 + ",
                $o,
                "]\n",
                "vfmadd231pd ymm4, ymm3, ymmword ptr [{k} + 192 + ",
                $o,
                "]\n",
                "vfmaddsub231pd ymm4, ",
                $x,
                ", ymmword ptr [{k} + ",
                $o,
                "]\n",
                "vfmadd231pd ymm4, ",
                $y,
                ", ymmword ptr [{k} + 128 + ",
                $o,
                "]\n",
            )
        };
    }

    macro_rules! avx2_swizzle {
        ($pre:expr, $y0:literal, $y1:literal, $p:expr, $sp:expr, $k:expr) => {
            asm!(
                "3:",
                "mov {a}, {pa}",
                "mov {n}, {len}",
                "2:",
                "vmovupd ymm0, ymmword ptr [{a}]",
                "vmovupd ymm6, ymmword ptr [{a} + 32]",
                $pre,
                avx2_swizzle_math!("ymm0", $y0, "0"),
                "vmovapd ymm5, ymm4",
                avx2_swizzle_math!("ymm6", $y1, "32"),
                "vmovupd ymmword ptr [{a}], ymm5",
                "vmovupd ymmword ptr [{a} + 32], ymm4",
                "add {a}, 64",
                "sub {n}, 4",
                "jnz 2b",
                "add {pa}, {stride}",
                "dec {cnt}",
                "jnz 3b",
                "vzeroupper",
                k = in(reg) $k.0.as_ptr(),
                pa = inout(reg) $p.add($sp.start) => _,
                len = in(reg) $sp.len,
                stride = in(reg) $sp.stride * 16,
                cnt = inout(reg) $sp.count => _,
                a = out(reg) _,
                n = out(reg) _,
                out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _, out("ymm4") _,
                out("ymm5") _, out("ymm6") _, out("ymm7") _,
                options(nostack),
            )
        };
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn swizzle_avx2(p: *mut C, half: usize, sp: Spans, k: &Coef<4>) {
        unsafe {
            if half == 1 {
                avx2_swizzle!(
                    "vperm2f128 ymm1, ymm0, ymm0, 0x01\nvperm2f128 ymm7, ymm6, ymm6, 0x01",
                    "ymm1",
                    "ymm7",
                    p,
                    sp,
                    k
                );
            } else {
                avx2_swizzle!("", "ymm6", "ymm0", p, sp, k);
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn scale_avx512(p: *mut C, sp: Spans, k: &Coef<2>) {
        unsafe {
            asm!(
                "vmovapd zmm8, zmmword ptr [{k}]",
                "vmovapd zmm9, zmmword ptr [{k} + 64]",
                "3:",
                "mov {a}, {pa}",
                "mov {n}, {len}",
                "2:",
                "vmovupd zmm0, zmmword ptr [{a}]",
                "vpermilpd zmm1, zmm0, 0x55",
                "vmulpd zmm1, zmm1, zmm9",
                "vfmaddsub231pd zmm1, zmm0, zmm8",
                "vmovupd zmmword ptr [{a}], zmm1",
                "add {a}, 64",
                "sub {n}, 4",
                "jnz 2b",
                "add {pa}, {stride}",
                "dec {cnt}",
                "jnz 3b",
                "vzeroupper",
                k = in(reg) k.0.as_ptr(),
                pa = inout(reg) p.add(sp.start) => _,
                len = in(reg) sp.len,
                stride = in(reg) sp.stride * 16,
                cnt = inout(reg) sp.count => _,
                a = out(reg) _,
                n = out(reg) _,
                out("zmm0") _, out("zmm1") _, out("zmm8") _, out("zmm9") _,
                options(nostack),
            );
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn scale_avx2(p: *mut C, sp: Spans, k: &Coef<2>) {
        unsafe {
            asm!(
                "3:",
                "mov {a}, {pa}",
                "mov {n}, {len}",
                "2:",
                "vmovupd ymm0, ymmword ptr [{a}]",
                "vpermilpd ymm1, ymm0, 0x5",
                "vmulpd ymm1, ymm1, ymmword ptr [{k} + 64]",
                "vfmaddsub231pd ymm1, ymm0, ymmword ptr [{k}]",
                "vmovupd ymmword ptr [{a}], ymm1",
                "vmovupd ymm0, ymmword ptr [{a} + 32]",
                "vpermilpd ymm1, ymm0, 0x5",
                "vmulpd ymm1, ymm1, ymmword ptr [{k} + 96]",
                "vfmaddsub231pd ymm1, ymm0, ymmword ptr [{k} + 32]",
                "vmovupd ymmword ptr [{a} + 32], ymm1",
                "add {a}, 64",
                "sub {n}, 4",
                "jnz 2b",
                "add {pa}, {stride}",
                "dec {cnt}",
                "jnz 3b",
                "vzeroupper",
                k = in(reg) k.0.as_ptr(),
                pa = inout(reg) p.add(sp.start) => _,
                len = in(reg) sp.len,
                stride = in(reg) sp.stride * 16,
                cnt = inout(reg) sp.count => _,
                a = out(reg) _,
                n = out(reg) _,
                out("ymm0") _, out("ymm1") _,
                options(nostack),
            );
        }
    }
}

// ------------------------------------------------------------------ aarch64: neon
//
// ld2-k bi konplexu kargatzen ditu re eta im bereizita (v0 = [re0 re1], v1 = [im0 im1]). talde bat (4
// anplitude) bi zatitan: 0-1 erreikak eta 2-3 erreikak, bakoitzak bere koefiziente-erregistroekin.
//   lo'.re = m00r*lr - m00i*li + m01r*hr - m01i*hi       lo'.im = m00r*li + m00i*lr + m01r*hi + m01i*hr

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{C, Mat2Lanes, ScaleLanes, Spans, SwizzleLanes};
    use std::arch::asm;

    // erreika-bikote bakoitzeko (0-1, 2-3), sarrera bakoitzeko re eta im bektoreak: [erdia][sarrera][erreika]
    #[repr(C, align(16))]
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

    // 2x2: sarrera v0-v3 (lo.re lo.im hi.re hi.im), irteera v4-v7, koefizienteak 8 erregistro
    macro_rules! neon_mat2_math {
        ($m00r:literal, $m00i:literal, $m01r:literal, $m01i:literal,
         $m10r:literal, $m10i:literal, $m11r:literal, $m11i:literal) => {
            concat!(
                "fmul v4.2d, v0.2d, ",
                $m00r,
                ".2d\n",
                "fmls v4.2d, v1.2d, ",
                $m00i,
                ".2d\n",
                "fmla v4.2d, v2.2d, ",
                $m01r,
                ".2d\n",
                "fmls v4.2d, v3.2d, ",
                $m01i,
                ".2d\n",
                "fmul v5.2d, v1.2d, ",
                $m00r,
                ".2d\n",
                "fmla v5.2d, v0.2d, ",
                $m00i,
                ".2d\n",
                "fmla v5.2d, v3.2d, ",
                $m01r,
                ".2d\n",
                "fmla v5.2d, v2.2d, ",
                $m01i,
                ".2d\n",
                "fmul v6.2d, v0.2d, ",
                $m10r,
                ".2d\n",
                "fmls v6.2d, v1.2d, ",
                $m10i,
                ".2d\n",
                "fmla v6.2d, v2.2d, ",
                $m11r,
                ".2d\n",
                "fmls v6.2d, v3.2d, ",
                $m11i,
                ".2d\n",
                "fmul v7.2d, v1.2d, ",
                $m10r,
                ".2d\n",
                "fmla v7.2d, v0.2d, ",
                $m10i,
                ".2d\n",
                "fmla v7.2d, v3.2d, ",
                $m11r,
                ".2d\n",
                "fmla v7.2d, v2.2d, ",
                $m11i,
                ".2d\n",
            )
        };
    }

    // segurtasuna (guztietan): neon aarch64-ren oinarria da; lasterketak erakusleen memorian daude
    // (deitzaileak egiaztatua), len 4ren multiploa eta > 0, count > 0
    pub unsafe fn mat2(lo: *mut C, hi: *mut C, sp: Spans, k: &Coef<8>) {
        unsafe {
            asm!(
                "ld1 {{v16.2d, v17.2d, v18.2d, v19.2d}}, [{k}], #64",
                "ld1 {{v20.2d, v21.2d, v22.2d, v23.2d}}, [{k}], #64",
                "ld1 {{v24.2d, v25.2d, v26.2d, v27.2d}}, [{k}], #64",
                "ld1 {{v28.2d, v29.2d, v30.2d, v31.2d}}, [{k}]",
                "3:",
                "mov {a}, {pa}",
                "mov {b}, {pb}",
                "mov {n}, {len}",
                "2:",
                "ld2 {{v0.2d, v1.2d}}, [{a}]",
                "ld2 {{v2.2d, v3.2d}}, [{b}]",
                neon_mat2_math!("v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"),
                "st2 {{v4.2d, v5.2d}}, [{a}], #32",
                "st2 {{v6.2d, v7.2d}}, [{b}], #32",
                "ld2 {{v0.2d, v1.2d}}, [{a}]",
                "ld2 {{v2.2d, v3.2d}}, [{b}]",
                neon_mat2_math!("v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"),
                "st2 {{v4.2d, v5.2d}}, [{a}], #32",
                "st2 {{v6.2d, v7.2d}}, [{b}], #32",
                "subs {n}, {n}, #4",
                "b.ne 2b",
                "add {pa}, {pa}, {stride}",
                "add {pb}, {pb}, {stride}",
                "subs {cnt}, {cnt}, #1",
                "b.ne 3b",
                k = inout(reg) k.0.as_ptr() => _,
                pa = inout(reg) lo.add(sp.start) => _,
                pb = inout(reg) hi.add(sp.start) => _,
                len = in(reg) sp.len,
                stride = in(reg) sp.stride * 16,
                cnt = inout(reg) sp.count => _,
                a = out(reg) _,
                b = out(reg) _,
                n = out(reg) _,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                out("v6") _, out("v7") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _, out("v25") _,
                out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                options(nostack),
            );
        }
    }

    // swizzle: x' = a x + b y. sarrera x (xr, xi) eta y (yr, yi) erregistro-izenak, irteera v4-v5
    macro_rules! neon_swizzle_math {
        ($xr:literal, $xi:literal, $yr:literal, $yi:literal,
         $ar:literal, $ai:literal, $br:literal, $bi:literal) => {
            concat!(
                "fmul v4.2d, ",
                $xr,
                ".2d, ",
                $ar,
                ".2d\n",
                "fmls v4.2d, ",
                $xi,
                ".2d, ",
                $ai,
                ".2d\n",
                "fmla v4.2d, ",
                $yr,
                ".2d, ",
                $br,
                ".2d\n",
                "fmls v4.2d, ",
                $yi,
                ".2d, ",
                $bi,
                ".2d\n",
                "fmul v5.2d, ",
                $xi,
                ".2d, ",
                $ar,
                ".2d\n",
                "fmla v5.2d, ",
                $xr,
                ".2d, ",
                $ai,
                ".2d\n",
                "fmla v5.2d, ",
                $yi,
                ".2d, ",
                $br,
                ".2d\n",
                "fmla v5.2d, ",
                $yr,
                ".2d, ",
                $bi,
                ".2d\n",
            )
        };
    }

    macro_rules! neon_swizzle {
        ($pre:expr, $y0r:literal, $y0i:literal, $y1r:literal, $y1i:literal, $p:expr, $sp:expr, $k:expr) => {
            asm!(
                "ld1 {{v16.2d, v17.2d, v18.2d, v19.2d}}, [{k}], #64",
                "ld1 {{v20.2d, v21.2d, v22.2d, v23.2d}}, [{k}]",
                "3:",
                "mov {a}, {pa}",
                "mov {n}, {len}",
                "2:",
                "ld2 {{v0.2d, v1.2d}}, [{a}]",
                "add {t}, {a}, #32",
                "ld2 {{v2.2d, v3.2d}}, [{t}]",
                $pre,
                neon_swizzle_math!("v0", "v1", $y0r, $y0i, "v16", "v17", "v18", "v19"),
                "mov v6.16b, v4.16b",
                "mov v7.16b, v5.16b",
                neon_swizzle_math!("v2", "v3", $y1r, $y1i, "v20", "v21", "v22", "v23"),
                "st2 {{v6.2d, v7.2d}}, [{a}], #32",
                "st2 {{v4.2d, v5.2d}}, [{a}], #32",
                "subs {n}, {n}, #4",
                "b.ne 2b",
                "add {pa}, {pa}, {stride}",
                "subs {cnt}, {cnt}, #1",
                "b.ne 3b",
                k = inout(reg) $k.0.as_ptr() => _,
                pa = inout(reg) $p.add($sp.start) => _,
                len = in(reg) $sp.len,
                stride = in(reg) $sp.stride * 16,
                cnt = inout(reg) $sp.count => _,
                a = out(reg) _,
                n = out(reg) _,
                t = out(reg) _,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                options(nostack),
            )
        };
    }

    pub unsafe fn swizzle(p: *mut C, half: usize, sp: Spans, k: &Coef<4>) {
        unsafe {
            if half == 1 {
                // bikotekidea erregistro beraren beste erdian: ext #8-k 64 biteko erdiak trukatzen ditu
                neon_swizzle!(
                    "ext v8.16b, v0.16b, v0.16b, #8\next v9.16b, v1.16b, v1.16b, #8\next v10.16b, v2.16b, v2.16b, #8\next v11.16b, v3.16b, v3.16b, #8",
                    "v8",
                    "v9",
                    "v10",
                    "v11",
                    p,
                    sp,
                    k
                );
            } else {
                neon_swizzle!("", "v2", "v3", "v0", "v1", p, sp, k);
            }
        }
    }

    pub unsafe fn scale(p: *mut C, sp: Spans, k: &Coef<2>) {
        unsafe {
            asm!(
                "ld1 {{v16.2d, v17.2d}}, [{k}], #32",
                "ld1 {{v18.2d, v19.2d}}, [{k}]",
                "3:",
                "mov {a}, {pa}",
                "mov {n}, {len}",
                "2:",
                "ld2 {{v0.2d, v1.2d}}, [{a}]",
                "fmul v4.2d, v0.2d, v16.2d",
                "fmls v4.2d, v1.2d, v17.2d",
                "fmul v5.2d, v1.2d, v16.2d",
                "fmla v5.2d, v0.2d, v17.2d",
                "st2 {{v4.2d, v5.2d}}, [{a}], #32",
                "ld2 {{v0.2d, v1.2d}}, [{a}]",
                "fmul v4.2d, v0.2d, v18.2d",
                "fmls v4.2d, v1.2d, v19.2d",
                "fmul v5.2d, v1.2d, v18.2d",
                "fmla v5.2d, v0.2d, v19.2d",
                "st2 {{v4.2d, v5.2d}}, [{a}], #32",
                "subs {n}, {n}, #4",
                "b.ne 2b",
                "add {pa}, {pa}, {stride}",
                "subs {cnt}, {cnt}, #1",
                "b.ne 3b",
                k = inout(reg) k.0.as_ptr() => _,
                pa = inout(reg) p.add(sp.start) => _,
                len = in(reg) sp.len,
                stride = in(reg) sp.stride * 16,
                cnt = inout(reg) sp.count => _,
                a = out(reg) _,
                n = out(reg) _,
                out("v0") _, out("v1") _, out("v4") _, out("v5") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                options(nostack),
            );
        }
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
        auxv.chunks_exact(16).any(|e| {
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
