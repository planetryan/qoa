// fitxategi honek qubit-ateak egoera-bektorearen gainean aplikatzen ditu, ahalik eta memoria-trafiko gutxienarekin.
//
// egoera handietan simulagailua memoriaren banda-zabalerak mugatzen du, ez kalkuluak: ate bakoitzak bektore
// osoa irakurri eta idazten du (28 qubit = 4 GiB). hiru neurri:
//
//   1. lasterketa jarraituak. q qubitaren ateak bikoteak (i, i + 2^q) nahasten ditu; bikoteak 2^q luzerako
//      lasterketa jarraituetan daude, eta lasterketa bakoitza mikro-kernel batek (src/simd.rs) prozesatzen
//      du. kontrolak lasterketa osoari aplikatzen zaizkio, ez elementuz elementu.
//   2. fusioa. qubit bereko ate kontrolgabe jarraituak (tartean qubit hori ukitzen ez duten ateak egon
//      daitezke) 2x2 matrize bakar bihurtzen dira.
//   3. cache-blokeak. helburu-qubit guztiak `b`-ren azpitik dituen ate-segida bat 2^b anplitudeko bloke
//      bakoitzean osorik aplikatzen da, blokea cache-n dagoen bitartean: memoria behin zeharkatzen da
//      segida osorako, ate bakoitzeko behin beharrean. kontrolak eta fasearen maskarak edozein qubitetan
//      egon daitezke (blokearen hasierako indizeak ematen ditu goiko bitak).
//
// egiaztapena: tests/reference.rs (ate bakoitza erreferentziazko simulagailuaren aurka) eta tests/kernel.rs
// (sortak banan-banako aplikazioaren aurka, bloke-tamaina txikiekin etapa guztiak behartuz).

use crate::simd::{self, Spans};
use num_complex::Complex64 as C;
use rayon::prelude::*;

// anplitude gutxiago badira, hari bakarrean (hariek aurrezten dutena baino gehiago kostatzen dute)
pub const PARALLEL_MIN: usize = 1 << 14;

// cache-blokeen tamaina lehenetsia: 2^14 anplitude = 256 KiB, nukleo bateko L2-an sartzen da
pub const BLOCK_BITS: usize = 14;

// zeregin paralelo baten gutxieneko tamaina (anplitudetan)
const TASK_MIN: usize = 1 << 13;

const ZERO: C = C::new(0.0, 0.0);
const ONE: C = C::new(1.0, 0.0);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QGate {
    // 2x2 matrizea [m00, m01, m10, m11] `q` qubitari, `controls` maskarako qubit guztiak 1 direnean
    Mat {
        q: usize,
        m: [C; 4],
        controls: usize,
    },
    // `a` eta `b` qubitak trukatu, `controls` guztiak 1 direnean
    Swap {
        a: usize,
        b: usize,
        controls: usize,
    },
}

impl QGate {
    // `mask`-eko qubit guztiak 1 dituzten oinarri-egoerak e^{iθ}-z biderkatu: ate diagonal bat maskaren
    // qubit altuenean, besteak kontrol gisa
    pub fn phase(mask: usize, theta: f64) -> QGate {
        let p = C::from_polar(1.0, theta);
        if mask == 0 {
            // qubitik gabe: fase globala
            return QGate::Mat {
                q: 0,
                m: [p, ZERO, ZERO, p],
                controls: 0,
            };
        }
        let top = usize::BITS as usize - 1 - mask.leading_zeros() as usize;
        QGate::Mat {
            q: top,
            m: [ONE, ZERO, ZERO, p],
            controls: mask & !(1 << top),
        }
    }

    // nahasten dituen qubit altuena (kontrolak ez dira kontatzen: blokearen hasieratik irakurtzen dira)
    fn top_target(&self) -> usize {
        match *self {
            QGate::Mat { q, .. } => q,
            QGate::Swap { a, b, .. } => a.max(b),
        }
    }

    // ukitzen dituen qubitak (helburuak eta kontrolak), maskara gisa
    fn touches(&self) -> usize {
        match *self {
            QGate::Mat { q, controls, .. } => controls | 1 << q,
            QGate::Swap { a, b, controls } => controls | 1 << a | 1 << b,
        }
    }
}

// matrizearen forma, kernel merkeena aukeratzeko
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    General,
    // m01 = m10 = 0: eskalatu bakarrik
    Diagonal,
    // pauli-x: trukaketa, biderketarik gabe
    Flip,
}

impl Kind {
    fn of(m: &[C; 4]) -> Kind {
        if m[1] == ZERO && m[2] == ZERO {
            Kind::Diagonal
        } else if m[0] == ZERO && m[3] == ZERO && m[1] == ONE && m[2] == ONE {
            Kind::Flip
        } else {
            Kind::General
        }
    }
}

// mikro-kernelek 4 anplitudeko taldeak prozesatzen dituzte (src/simd.rs): 0. eta 1. bitak "erreika" dira
// eta haien kontrolak erreika bakoitzeko koefizienteetan sartzen dira; 2. bitetik gorakoak lasterketen
// egituran (`for_spans`)
const LANE_MASK: usize = 3;
const IDENTITY: [C; 4] = [ONE, ZERO, ZERO, ONE];

// `pattern` bitetan (guztiak >= 2) `want` balioak dituzten indizeak, lasterketa-sortetan: bit baxuenak
// lasterketen luzera eta pausoa ematen ditu, besteek zein talde dagoen. dei bat talde bakoitzeko.
// `len` slice-aren luzera da eta `base` haren lehen indizea egoeran (len-ren berreturarekin lerrokatua)
fn for_spans(len: usize, base: usize, pattern: usize, want: usize, mut f: impl FnMut(Spans)) {
    debug_assert_eq!(pattern & LANE_MASK, 0);
    if pattern == 0 {
        f(Spans::one(0, len));
        return;
    }
    let run = 1usize << pattern.trailing_zeros();
    if run >= len {
        // bit guztiak slice-aren gainetik: slice osoak baldintza betetzen du ala ez
        if base & pattern == want {
            f(Spans::one(0, len));
        }
        return;
    }
    let rest = pattern & !run;
    let group = if rest == 0 {
        len
    } else {
        (1usize << rest.trailing_zeros()).min(len)
    };
    let offset = if want & run != 0 { run } else { 0 };
    for g in (0..len).step_by(group) {
        if (base + g) & rest == want & rest {
            f(Spans {
                start: g + offset,
                len: run,
                stride: 2 * run,
                count: group / (2 * run),
            });
        }
    }
}

// erreika bakoitzak kontrol baxuak (0. eta 1. qubitak) betetzen ditu?
fn lane_ok(j: usize, low: usize) -> bool {
    j & low == low
}

// 4ren multiploa ez den luzerako egoerak (qubit bakarrekoak): zuzenean
fn mat_small(amps: &mut [C], base: usize, q: usize, m: &[C; 4], controls: usize) {
    for i in 0..amps.len() {
        let j = i | 1 << q;
        if i >> q & 1 == 0 && (base + i) & controls == controls && j < amps.len() {
            let (x, y) = (amps[i], amps[j]);
            amps[i] = m[0] * x + m[1] * y;
            amps[j] = m[2] * x + m[3] * y;
        }
    }
}

// ate bat egoeraren zati jarraitu batean, hari bakarrean. `amps`-en luzera 2^(q+1)-ren eta 4ren multiploa
// da eta `base` (amps[0]-ren indizea) haren lerrokadura berekoa
pub fn mat_on(amps: &mut [C], base: usize, q: usize, m: &[C; 4], controls: usize) {
    let len = amps.len();
    if !len.is_multiple_of(4) {
        // qubit bakarra (2 anplitude, edo 2 x moduak): zuzenean
        mat_small(amps, base, q, m, controls);
        return;
    }
    let mut kind = Kind::of(m);
    let low = controls & LANE_MASK;
    let outer = controls & !LANE_MASK;
    if kind == Kind::Flip && (q < 2 || low != 0) {
        // erreika barruko trukaketa: aritmetikarekin (koefizienteak 0 eta 1, zehatza)
        kind = Kind::General;
    }
    if q < 2 {
        // bikoteak 4ko talde beraren barruan: swizzle (edo eskalatzea diagonaletan)
        let half = 1usize << q;
        if kind == Kind::Diagonal {
            let k = std::array::from_fn(|j| match (lane_ok(j, low), j & half == 0) {
                (false, _) => ONE,
                (true, true) => m[0],
                (true, false) => m[3],
            });
            for_spans(len, base, outer, outer, |sp| simd::scale(amps, sp, &k));
        } else {
            let k = std::array::from_fn(|j| match (lane_ok(j, low), j & half == 0) {
                (false, _) => [ONE, ZERO],
                (true, true) => [m[0], m[1]],
                (true, false) => [m[3], m[2]],
            });
            for_spans(len, base, outer, outer, |sp| {
                simd::swizzle(amps, half, sp, &k)
            });
        }
        return;
    }
    // q >= 2: lo lasterketak (q bita 0) eta hi lasterketak 2^q aurrerago
    let half = 1usize << q;
    let pattern = outer | half;
    match kind {
        Kind::Diagonal => {
            let k0 = std::array::from_fn(|j| if lane_ok(j, low) { m[0] } else { ONE });
            let k1 = std::array::from_fn(|j| if lane_ok(j, low) { m[3] } else { ONE });
            if m[0] != ONE {
                for_spans(len, base, pattern, outer, |sp| simd::scale(amps, sp, &k0));
            }
            if m[3] != ONE {
                for_spans(len, base, pattern, outer | half, |sp| {
                    simd::scale(amps, sp, &k1)
                });
            }
        }
        Kind::Flip => for_spans(len, base, pattern, outer, |sp| {
            for s in 0..sp.count {
                let a = sp.start + s * sp.stride;
                let (x, y) = amps.split_at_mut(a + half);
                x[a..a + sp.len].swap_with_slice(&mut y[..sp.len]);
            }
        }),
        Kind::General => {
            let k = std::array::from_fn(|j| if lane_ok(j, low) { *m } else { IDENTITY });
            for_spans(len, base, pattern, outer, |sp| {
                simd::mat2_in(amps, half, sp, &k)
            });
        }
    }
}

// q >= 2: bloke baten bi erdien zati parekoak (lo eta hi, luzera berekoak), egoera osoko zeharkaldiaren
// bide paraleloan. `lo_base` lo[0]-ren indizea da egoeran
fn halves(lo: &mut [C], hi: &mut [C], lo_base: usize, m: &[C; 4], controls: usize) {
    let low = controls & LANE_MASK;
    // q-ren gaineko kontrolak lo_base-tik irakurtzen dira (konstanteak zati osoan)
    let outer = controls & !LANE_MASK;
    let len = lo.len();
    let mut kind = Kind::of(m);
    if kind == Kind::Flip && low != 0 {
        kind = Kind::General;
    }
    match kind {
        Kind::Diagonal => {
            let k0 = std::array::from_fn(|j| if lane_ok(j, low) { m[0] } else { ONE });
            let k1 = std::array::from_fn(|j| if lane_ok(j, low) { m[3] } else { ONE });
            for_spans(len, lo_base, outer, outer, |sp| {
                if m[0] != ONE {
                    simd::scale(lo, sp, &k0);
                }
                if m[3] != ONE {
                    simd::scale(hi, sp, &k1);
                }
            });
        }
        Kind::Flip => for_spans(len, lo_base, outer, outer, |sp| {
            for s in 0..sp.count {
                let a = sp.start + s * sp.stride;
                lo[a..a + sp.len].swap_with_slice(&mut hi[a..a + sp.len]);
            }
        }),
        Kind::General => {
            let k = std::array::from_fn(|j| if lane_ok(j, low) { *m } else { IDENTITY });
            for_spans(len, lo_base, outer, outer, |sp| simd::mat2(lo, hi, sp, &k));
        }
    }
}

// ate bat egoera osoan, paraleloan
// x baino txikiagoa edo berdina den 2ren berredura handiena (gutxienez 1). zatien tamainak 2ren berredurak
// izan behar dira: lasterketak 4ren multiploak dira (simd), eta trukaketen a=0|a=1 egiturak zatien barruan
// geratu behar du. hari-kopurua (eta beraz `tasks`) ez da beti 2ren berredura (3, 6, 12 nukleo...)
fn pow2_floor(x: usize) -> usize {
    if x == 0 { 1 } else { 1 << x.ilog2() }
}

pub fn sweep_mat(amps: &mut [C], q: usize, m: &[C; 4], controls: usize) {
    let n = amps.len();
    let block = 2usize << q;
    if n < PARALLEL_MIN {
        mat_on(amps, 0, q, m, controls);
        return;
    }
    let tasks = rayon::current_num_threads() * 4;
    if n / block >= tasks {
        // bloke txiki asko: zeregin bakoitzak bloke oso batzuk
        let chunk = block * (TASK_MIN / block).max(1);
        amps.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(k, c)| mat_on(c, k * chunk, q, m, controls));
    } else {
        // bloke handi gutxi (qubit altua): bloke bakoitzaren erdiak zatitan banatu harien artean
        let half = block / 2;
        let sub = pow2_floor(half / tasks).max(TASK_MIN).min(half);
        for (k, b) in amps.chunks_mut(block).enumerate() {
            let (lo, hi) = b.split_at_mut(half);
            lo.par_chunks_mut(sub)
                .zip(hi.par_chunks_mut(sub))
                .enumerate()
                .for_each(|(j, (l, h))| halves(l, h, k * block + j * sub, m, controls));
        }
    }
}

// a < b qubitak trukatu egoeraren zati jarraitu batean. b-ren bloke bakoitza bi erditan (b=0 | b=1);
// trukatzen direnak: erdi baxuko a=1 elementuak erdi altuko a=0 elementuekin
pub fn swap_on(amps: &mut [C], base: usize, a: usize, b: usize, controls: usize) {
    let (lo, hi) = (a.min(b), a.max(b));
    let half = 1usize << hi;
    for (k, block) in amps.chunks_mut(2 * half).enumerate() {
        let (l, h) = block.split_at_mut(half);
        swap_halves(l, h, base + 2 * half * k, lo, controls);
    }
}

// `l`: blokearen erdi baxua (edo zati bat), `l_base` haren indizea; `h`: erdi altuko zati parekoa
fn swap_halves(l: &mut [C], h: &mut [C], l_base: usize, lo: usize, controls: usize) {
    let lo_bit = 1usize << lo;
    let sub = lo_bit << 1;
    // a=1 lasterketak erdi baxuan (lo_bit..sub), a=0 lasterketak erdi altuan (0..lo_bit), 2^lo luzerakoak
    let low = controls & (lo_bit - 1);
    let run = if low == 0 {
        lo_bit
    } else {
        1usize << low.trailing_zeros()
    };
    let mid = controls & !(lo_bit - 1);
    for (s, (lp, hp)) in l.chunks_mut(sub).zip(h.chunks_mut(sub)).enumerate() {
        let start = l_base + s * sub + lo_bit;
        // kontrolak a=1 elementuaren indizearekin (erdi baxua): hi bita 0 da hor, beraz kontrolek ez dute
        // hi barne (swap-ek ez du bere qubitak kontrol gisa onartzen)
        if start & mid != mid {
            continue;
        }
        let (_, l1) = lp.split_at_mut(lo_bit);
        let (h0, _) = hp.split_at_mut(lo_bit);
        if low == 0 {
            l1.swap_with_slice(h0);
        } else {
            for (k, (x, y)) in l1.chunks_mut(run).zip(h0.chunks_mut(run)).enumerate() {
                if (start + k * run) & low == low {
                    x.swap_with_slice(y);
                }
            }
        }
    }
}

// trukaketa egoera osoan, paraleloan
pub fn sweep_swap(amps: &mut [C], a: usize, b: usize, controls: usize) {
    let (lo, hi) = (a.min(b), a.max(b));
    let n = amps.len();
    let block = 2usize << hi;
    if n < PARALLEL_MIN {
        swap_on(amps, 0, a, b, controls);
        return;
    }
    let tasks = rayon::current_num_threads() * 4;
    if n / block >= tasks {
        let chunk = block * (TASK_MIN / block).max(1);
        amps.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(k, c)| swap_on(c, k * chunk, a, b, controls));
    } else {
        // zatiak 2^(lo+1)-en multiploak, a=0|a=1 egitura zatien barruan gera dadin
        let half = block / 2;
        let sub = pow2_floor(half / tasks)
            .max(TASK_MIN)
            .max(2 << lo)
            .min(half);
        for (k, blk) in amps.chunks_mut(block).enumerate() {
            let (l, h) = blk.split_at_mut(half);
            l.par_chunks_mut(sub)
                .zip(h.par_chunks_mut(sub))
                .enumerate()
                .for_each(|(j, (lp, hp))| swap_halves(lp, hp, k * block + j * sub, lo, controls));
        }
    }
}

fn apply_on(amps: &mut [C], base: usize, g: &QGate) {
    match g {
        QGate::Mat { q, m, controls } => mat_on(amps, base, *q, m, *controls),
        QGate::Swap { a, b, controls } => swap_on(amps, base, *a, *b, *controls),
    }
}

fn sweep(amps: &mut [C], g: &QGate) {
    match g {
        QGate::Mat { q, m, controls } => sweep_mat(amps, *q, m, *controls),
        QGate::Swap { a, b, controls } => sweep_swap(amps, *a, *b, *controls),
    }
}

// 2x2 biderketa: lehen `first`, gero `then`
fn product(then: &[C; 4], first: &[C; 4]) -> [C; 4] {
    [
        then[0] * first[0] + then[1] * first[2],
        then[0] * first[1] + then[1] * first[3],
        then[2] * first[0] + then[3] * first[2],
        then[2] * first[1] + then[3] * first[3],
    ]
}

fn is_identity(m: &[C; 4]) -> bool {
    (m[0] - ONE).norm() < 1e-15
        && m[1].norm() < 1e-15
        && m[2].norm() < 1e-15
        && (m[3] - ONE).norm() < 1e-15
}

// qubit bereko ate kontrolgabeak bateratu. ate bat aurreko batekin bateratzen da tartean qubit hori ukitzen
// duen aterik ez badago (beste qubitetako ateek trukatzen dute harekin). identitate bihurtzen direnak kendu
pub fn fuse(gates: &[QGate]) -> Vec<QGate> {
    let mut out: Vec<QGate> = Vec::with_capacity(gates.len());
    // qubit bakoitzeko azken ate kontrolgabe "irekia" (out-en indizea)
    let mut open: Vec<Option<usize>> = Vec::new();
    for g in gates {
        if let QGate::Mat { q, m, controls: 0 } = *g
            && let Some(Some(i)) = open.get(q)
            && let QGate::Mat { m: prev, .. } = &mut out[*i]
        {
            *prev = product(&m, prev);
            continue;
        }
        let touches = g.touches();
        let top = usize::BITS as usize - touches.leading_zeros() as usize;
        if open.len() < top {
            open.resize(top, None);
        }
        for (q, slot) in open.iter_mut().enumerate() {
            if touches >> q & 1 == 1 {
                *slot = None;
            }
        }
        out.push(*g);
        if let QGate::Mat { q, controls: 0, .. } = *g {
            open[q] = Some(out.len() - 1);
        }
    }
    out.retain(|g| !matches!(g, QGate::Mat { m, .. } if is_identity(m)));
    out
}

// cache-blokeen tamaina egoera honetarako: gehienez BLOCK_BITS eta qubit-kopurua, eta hari bakoitzak bloke
// batzuk izan ditzan
fn block_bits(n: usize, n_qubits: usize) -> usize {
    let tasks = rayon::current_num_threads() * 4;
    let fit = (n / tasks).max(1).ilog2() as usize;
    BLOCK_BITS.min(n_qubits).min(fit)
}

// bilketa-bloke batean sar daitezkeen qubit altuen gehienezko kopurua (ikus `run_gathered`)
pub const HIGH_MAX: usize = 4;

// ate-sorta bat: fusioa, gero etapak (cache-blokeak edo egoera osoaren zeharkaldiak)
pub fn apply_gates(amps: &mut [C], n_qubits: usize, gates: &[QGate]) {
    let fused = fuse(gates);
    if amps.len() < PARALLEL_MIN {
        for g in &fused {
            apply_on(amps, 0, g);
        }
        return;
    }
    let b = block_bits(amps.len(), n_qubits);
    run_stages(amps, n_qubits, &fused, b, HIGH_MAX);
}

// probetarako: fusiorik gabe, bloke-tamaina eta qubit altuen kopuru jakin batekin (egoera txikietan ere etapa
// mota guztiak probatzeko)
pub fn apply_gates_blocked(
    amps: &mut [C],
    n_qubits: usize,
    gates: &[QGate],
    block_bits: usize,
    high_max: usize,
) {
    run_stages(amps, n_qubits, gates, block_bits, high_max);
}

impl QGate {
    // nahasten dituen qubitak, maskara gisa
    fn targets(&self) -> usize {
        match *self {
            QGate::Mat { q, .. } => 1 << q,
            QGate::Swap { a, b, .. } => 1 << a | 1 << b,
        }
    }
}

// etapak, ordenan:
//   A) helburu guztiak `b`-ren azpitik: 2^b anplitudeko bloke jarraituak, tokian bertan
//   B) helburuak `b - high_max`-en azpitik edo gehienez `high_max` qubit altutan: bilketa-blokeak
//   bestela (edo ate bakarra) egoera osoaren zeharkaldia
fn run_stages(amps: &mut [C], n_qubits: usize, gates: &[QGate], b: usize, high_max: usize) {
    let low = b.saturating_sub(high_max);
    let gather_ok = n_qubits > b && low >= 2;
    let mut i = 0;
    while i < gates.len() {
        let rest = &gates[i..];
        let in_place = rest.iter().take_while(|g| g.top_target() < b).count();
        let (mut gathered, mut high) = (0, 0usize);
        if gather_ok {
            for g in rest {
                let h = high | (g.targets() >> low << low);
                if h.count_ones() as usize > high_max {
                    break;
                }
                high = h;
                gathered += 1;
            }
        }
        if in_place > 0 && in_place >= gathered {
            run_in_place(amps, &rest[..in_place], b);
            i += in_place;
        } else if gathered > 1 {
            run_gathered(amps, n_qubits, &rest[..gathered], low, high);
            i += gathered;
        } else {
            sweep(amps, &rest[0]);
            i += 1;
        }
    }
}

fn run_in_place(amps: &mut [C], stage: &[QGate], b: usize) {
    let size = 1usize << b;
    let blocks = |(k, blk): (usize, &mut [C])| {
        for g in stage {
            apply_on(blk, k * size, g);
        }
    };
    if amps.len() >= PARALLEL_MIN {
        amps.par_chunks_mut(size).enumerate().for_each(blocks);
    } else {
        amps.chunks_mut(size).enumerate().for_each(blocks);
    }
}

// x-ren bitak `mask`-eko posizioetan banatu, baxuenetik hasita (x86-ko pdep bezala)
fn deposit(mut x: usize, mut mask: usize) -> usize {
    let mut out = 0;
    while mask != 0 && x != 0 {
        let bit = mask & mask.wrapping_neg();
        if x & 1 == 1 {
            out |= bit;
        }
        x >>= 1;
        mask &= mask - 1;
    }
    out
}

// egoeraren erakusle partekatua harien artean. bloke bakoitzak bere indize-multzoa ukitzen du, eta multzo
// horiek ez dute elkar gainjartzen (ikus `run_gathered`)
struct Shared(*mut C);
// segurtasuna: erakuslea bloke disjuntuetarako bakarrik erabiltzen da, hari bakoitzak berea
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

// bilketa-blokeak. bloke bat = `low` qubit baxuak (2^low anplitudeko lasterketa jarraituak) gehi `high`
// maskarako qubit altuak; beste qubit eta moduen balioak finko blokearen barruan. blokea buffer batera
// biltzen da (2^|high| lasterketa), sortako ate guztiak han aplikatzen zaizkio, qubitak tokiko
// posizioetara berrizendatuta, eta itzuli egiten da. memoria behin irakurri eta idazten da etapa osorako.
// blokeek ez dute indizerik partekatzen: bi bloke desberdinek balio desberdina dute bloke-barneko ez diren
// bitetan gutxienez batean, beraz paraleloan idatz daitezke
fn run_gathered(amps: &mut [C], n_qubits: usize, stage: &[QGate], low: usize, high: usize) {
    let h = high.count_ones() as usize;
    let run = 1usize << low;
    let size = run << h;
    let qubit_mask = (1usize << n_qubits) - 1;
    let outer = qubit_mask & !((1usize << low) - 1) & !high;
    let outer_bits = outer.count_ones();
    let blocks = amps.len() / size;
    // qubit globala -> tokikoa
    let local = |q: usize| -> usize {
        if q < low {
            q
        } else {
            low + (high & ((1 << q) - 1)).count_ones() as usize
        }
    };
    let remap = |mask: usize| -> usize {
        let mut out = mask & ((1 << low) - 1);
        let mut m = mask & high;
        while m != 0 {
            let q = m.trailing_zeros() as usize;
            out |= 1 << local(q);
            m &= m - 1;
        }
        out
    };
    // ate bakoitza tokiko qubitekin, eta blokearen kanpoko kontrolak (blokearen oinarriarekin egiaztatzeko)
    let plan: Vec<(QGate, usize)> = stage
        .iter()
        .map(|g| match *g {
            QGate::Mat { q, m, controls } => (
                QGate::Mat {
                    q: local(q),
                    m,
                    controls: remap(controls),
                },
                controls & !((1 << low) - 1) & !high,
            ),
            QGate::Swap { a, b, controls } => (
                QGate::Swap {
                    a: local(a),
                    b: local(b),
                    controls: remap(controls),
                },
                controls & !((1 << low) - 1) & !high,
            ),
        })
        .collect();
    let offsets: Vec<usize> = (0..1usize << h).map(|r| deposit(r, high)).collect();
    let shared = Shared(amps.as_mut_ptr());
    let len = amps.len();
    let work = |buf: &mut Vec<C>, k: usize| {
        let base = deposit(k & ((1 << outer_bits) - 1), outer) + ((k >> outer_bits) << n_qubits);
        let ptr = &shared;
        for (r, off) in offsets.iter().enumerate() {
            debug_assert!(base + off + run <= len);
            // segurtasuna: [base + off, + run) bloke honena da bakarrik (goiko iruzkina) eta egoeraren barruan
            let src = unsafe { std::slice::from_raw_parts(ptr.0.add(base + off), run) };
            buf[r * run..(r + 1) * run].copy_from_slice(src);
        }
        for (g, outside) in &plan {
            if base & outside == *outside {
                apply_on(buf, 0, g);
            }
        }
        for (r, off) in offsets.iter().enumerate() {
            // segurtasuna: goian bezala; beste hariek ez dute tarte hau ukitzen
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr.0.add(base + off), run) };
            dst.copy_from_slice(&buf[r * run..(r + 1) * run]);
        }
    };
    if len >= PARALLEL_MIN {
        (0..blocks)
            .into_par_iter()
            .for_each_init(|| vec![ZERO; size], |buf, k| work(buf, k));
    } else {
        let mut buf = vec![ZERO; size];
        (0..blocks).for_each(|k| work(&mut buf, k));
    }
}
