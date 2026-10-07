// fitxategi honek sistema kuantiko simulatua gordetzen du: qubitak eta modu optikoak egoera-bektore bakarrean.
//
// anplitude-indizearen antolaketa: behe-mailako `n_qubits` bitak qubitak dira (q0 0. bita da); haien
// gainetik, modu bakoitza `cutoff` oinarriko digitu bat da (o0 digitu baxuena). beraz bektoreak
// 2^n_qubits * cutoff^n_modes anplitude konplexu ditu.
//
// ateak tokian bertan aplikatzen dira, bektorea behin zeharkatzen duten kernelekin (egoera handietan
// paraleloan). ate unitarioek araua mantentzen dute, beraz ez da ezer normalizatzen haien ondoren;
// neurketek eta eragiketa optiko ez-unitarioek bakarrik normalizatzen dute, eta fotoi-mugaz kanpo
// galdutako probabilitatea `leakage`-ri gehitzen zaio, jakinarazi ahal izateko.

use crate::fock::Mat;
use crate::kernel::{self, PARALLEL_MIN, QGate};
use crate::simd::{self, Spans};
use crate::t;
use num_complex::Complex64 as C;
use rayon::prelude::*;

const ZERO: C = C::new(0.0, 0.0);
const ONE: C = C::new(1.0, 0.0);

// zati paraleloen tamaina handiena (anplitudetan) xaflak zatitzean
const PIECE: usize = 1 << 14;

// simulagailuak erreserbatuko duen egoera-bektore handiena (2^32 anplitude = 64 GiB)
pub const MAX_AMPLITUDES: u128 = 1 << 32;

#[derive(Clone, Debug)]
pub struct State {
    n_qubits: usize,
    n_modes: usize,
    cutoff: usize,
    amps: Vec<C>,
    // fotoi-mugaz kanpo erori den probabilitate osoa (eta normalizazioz kendua)
    pub leakage: f64,
}

impl State {
    // |0...0> qubitetan, hutsa modu guztietan
    pub fn new(n_qubits: usize, n_modes: usize, cutoff: usize) -> Result<State, String> {
        let dim = Self::dim_for(n_qubits, n_modes, cutoff).ok_or_else(|| {
            t!(
                "the state is too large to index",
                "egoera handiegia da indexatzeko"
            )
        })?;
        if dim > MAX_AMPLITUDES {
            return Err(t!(
                "{} qubits and {} modes (cutoff {}) need {} amplitudes ({}), more than the {} limit",
                "{} qubit eta {} modu ({} muga) {} anplitude behar dituzte ({}), {} muga baino gehiago",
                n_qubits,
                n_modes,
                cutoff,
                dim,
                format_bytes(dim * 16),
                format_bytes(MAX_AMPLITUDES * 16)
            ));
        }
        if n_modes > 0 && cutoff < 2 {
            return Err(t!(
                "a mode needs a cutoff of at least 2 (photon numbers 0 and 1)",
                "modu batek gutxienez 2ko muga behar du (0 eta 1 fotoi)"
            ));
        }
        let mut amps = vec![C::new(0.0, 0.0); dim as usize];
        amps[0] = C::new(1.0, 0.0);
        Ok(State {
            n_qubits,
            n_modes,
            cutoff,
            amps,
            leakage: 0.0,
        })
    }

    pub fn dim_for(n_qubits: usize, n_modes: usize, cutoff: usize) -> Option<u128> {
        let mut d = 1u128.checked_shl(n_qubits as u32)?;
        for _ in 0..n_modes {
            d = d.checked_mul(cutoff as u128)?;
        }
        Some(d)
    }

    // egoera bat anplitude zehatzetatik (probetarako eta tresnetarako); luzera zuzena behar du
    pub fn from_amplitudes(
        n_qubits: usize,
        n_modes: usize,
        cutoff: usize,
        amps: Vec<C>,
    ) -> Result<State, String> {
        let dim = Self::dim_for(n_qubits, n_modes, cutoff).ok_or_else(|| {
            t!(
                "the state is too large to index",
                "egoera handiegia da indexatzeko"
            )
        })?;
        if amps.len() as u128 != dim {
            return Err(t!(
                "expected {} amplitudes, got {}",
                "{} anplitude espero ziren, {} jaso dira",
                dim,
                amps.len()
            ));
        }
        Ok(State {
            n_qubits,
            n_modes,
            cutoff,
            amps,
            leakage: 0.0,
        })
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }
    pub fn n_modes(&self) -> usize {
        self.n_modes
    }
    pub fn cutoff(&self) -> usize {
        self.cutoff
    }
    pub fn amplitudes(&self) -> &[C] {
        &self.amps
    }
    pub fn len(&self) -> usize {
        self.amps.len()
    }
    pub fn is_empty(&self) -> bool {
        self.amps.is_empty()
    }

    pub fn norm_sqr(&self) -> f64 {
        if self.amps.len() >= PARALLEL_MIN {
            self.amps.par_iter().map(|a| a.norm_sqr()).sum()
        } else {
            self.amps.iter().map(|a| a.norm_sqr()).sum()
        }
    }

    // o moduaren ondoz ondoko fotoi-kopuruen arteko distantzia indize-espazioan
    #[inline]
    pub fn mode_stride(&self, o: usize) -> usize {
        (1usize << self.n_qubits) * self.cutoff.pow(o as u32)
    }

    // o moduaren fotoi-kopurua idx oinarri-egoeran
    #[inline]
    pub fn photons(&self, idx: usize, o: usize) -> usize {
        (idx / self.mode_stride(o)) % self.cutoff
    }

    // q qubitaren balioa (0/1) idx oinarri-egoeran
    #[inline]
    pub fn bit(idx: usize, q: usize) -> u8 {
        ((idx >> q) & 1) as u8
    }

    // ------------------------------------------------------------------ qubit-ateak

    // 2x2 matrizea [m00, m01, m10, m11] q qubitari aplikatu, `controls` maskarako qubit guztiak 1
    // diren oinarri-egoeretan (`controls` bit-maskara bat da; 0 kontrolik gabe). ikus src/kernel.rs
    pub fn apply_1q(&mut self, q: usize, m: [C; 4], controls: usize) {
        kernel::sweep_mat(&mut self.amps, q, &m, controls);
    }

    // pauli-x q-n (kontrolekin edo gabe): trukaketa, biderketarik gabe
    pub fn apply_x(&mut self, q: usize, controls: usize) {
        let x = [
            C::new(0.0, 0.0),
            C::new(1.0, 0.0),
            C::new(1.0, 0.0),
            C::new(0.0, 0.0),
        ];
        kernel::sweep_mat(&mut self.amps, q, &x, controls);
    }

    // e^{iθ}-z biderkatu `mask`-eko qubit guztiak 1 diren oinarri-egoera guztiak
    pub fn apply_phase_mask(&mut self, mask: usize, theta: f64) {
        self.apply_gates(&[QGate::phase(mask, theta)]);
    }

    // a eta b qubitak trukatu, `controls` guztiak 1 diren oinarri-egoeretan
    pub fn apply_swap(&mut self, a: usize, b: usize, controls: usize) {
        kernel::sweep_swap(&mut self.amps, a, b, controls);
    }

    // ate-sorta bat: fusioarekin eta cache-blokeekin, banan-banan baino memoria-trafiko gutxiagorekin
    pub fn apply_gates(&mut self, gates: &[QGate]) {
        kernel::apply_gates(&mut self.amps, self.n_qubits, gates);
    }

    // ------------------------------------------------------------------ neurketa

    // q qubitak 1 irakurtzeko probabilitatea
    pub fn prob_one(&self, q: usize) -> f64 {
        let bit = 1usize << q;
        let f = |(i, a): (usize, &C)| if i & bit != 0 { a.norm_sqr() } else { 0.0 };
        if self.amps.len() >= PARALLEL_MIN {
            self.amps.par_iter().enumerate().map(f).sum()
        } else {
            self.amps.iter().enumerate().map(f).sum()
        }
    }

    // q qubita `outcome`-ra proiektatu eta normalizatu. `p` emaitza horren probabilitatea da.
    pub fn collapse_qubit(&mut self, q: usize, outcome: u8, p: f64) {
        let scale = C::new(1.0 / p.sqrt(), 0.0);
        let m = if outcome == 1 {
            [ZERO, ZERO, ZERO, scale]
        } else {
            [scale, ZERO, ZERO, ZERO]
        };
        kernel::sweep_mat(&mut self.amps, q, &m, 0);
    }

    // q qubita oinarri konputazionalean neurtu, `r` erabiliz ([0, 1) tarteko uniformea)
    pub fn measure_qubit(&mut self, q: usize, r: f64) -> u8 {
        let p1 = self.prob_one(q).clamp(0.0, 1.0);
        let outcome = if r < p1 { 1 } else { 0 };
        let p = if outcome == 1 { p1 } else { 1.0 - p1 };
        self.collapse_qubit(q, outcome, p);
        outcome
    }

    // o moduaren fotoi-kopuruaren banaketa
    pub fn photon_distribution(&self, o: usize) -> Vec<f64> {
        let (stride, d) = (self.mode_stride(o), self.cutoff);
        let slab = |(k, x): (usize, &[C])| {
            let mut dist = vec![0.0; d];
            dist[k % d] = x.iter().map(|a| a.norm_sqr()).sum();
            dist
        };
        let add = |mut a: Vec<f64>, b: Vec<f64>| {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
            a
        };
        // xafla bakoitzak (o moduaren n jakin bat) bere batura; xafla luzeak zatitan hari askotarako
        let piece = stride.min(PIECE);
        let per_slab = stride / piece;
        if self.amps.len() >= PARALLEL_MIN {
            self.amps
                .par_chunks(piece)
                .enumerate()
                .map(|(k, x)| slab((k / per_slab, x)))
                .reduce(|| vec![0.0; d], add)
        } else {
            self.amps
                .chunks(stride)
                .enumerate()
                .map(slab)
                .fold(vec![0.0; d], add)
        }
    }

    // o moduko fotoi-kopuruak `keep` betetzen duen oinarri-egoerak bakarrik gorde, gordetako `p`
    // probabilitatearekin normalizatuta
    pub fn collapse_mode(&mut self, o: usize, p: f64, keep: impl Fn(usize) -> bool + Sync + Send) {
        let (stride, d) = (self.mode_stride(o), self.cutoff);
        let scale = C::new(1.0 / p.sqrt(), 0.0);
        let factors: Vec<C> = (0..d).map(|n| if keep(n) { scale } else { ZERO }).collect();
        par_slabs(&mut self.amps, stride, |k, x| scale_run(x, factors[k % d]));
    }

    // o moduaren fotoi-kopurua neurtu (proiektiboa, kopurua bereizten du)
    pub fn measure_photons(&mut self, o: usize, r: f64) -> usize {
        let dist = self.photon_distribution(o);
        let n = sample_index(&dist, r);
        let p = dist[n];
        self.collapse_mode(o, p, |k| k == n);
        n
    }

    // atalase-detekzioa ("klik"): 0 hutsarentzat, 1 fotoi bat edo gehiagorentzat
    pub fn measure_click(&mut self, o: usize, r: f64) -> u8 {
        let dist = self.photon_distribution(o);
        let p0 = dist[0];
        if r < p0 {
            self.collapse_mode(o, p0, |k| k == 0);
            0
        } else {
            self.collapse_mode(o, 1.0 - p0, |k| k > 0);
            1
        }
    }

    // fotoi-kopuruaren paritatearen neurketa: 0 bikoitia, 1 bakoitia
    pub fn measure_parity(&mut self, o: usize, r: f64) -> u8 {
        let dist = self.photon_distribution(o);
        let p_even: f64 = dist.iter().step_by(2).sum();
        if r < p_even {
            self.collapse_mode(o, p_even, |k| k % 2 == 0);
            0
        } else {
            self.collapse_mode(o, 1.0 - p_even, |k| k % 2 == 1);
            1
        }
    }

    // oinarri-egoera bakoitzaren probabilitatea (shot guztiak simulazio bakarretik lagintzeko)
    pub fn probabilities(&self) -> Vec<f64> {
        if self.amps.len() >= PARALLEL_MIN {
            self.amps.par_iter().map(|a| a.norm_sqr()).collect()
        } else {
            self.amps.iter().map(|a| a.norm_sqr()).collect()
        }
    }

    // ------------------------------------------------------------------ ikuskapena

    // q qubitaren <X>, <Y>, <Z> zehatzak (bloch bektorea). simulagailuan bakarrik.
    pub fn bloch(&self, q: usize) -> [f64; 3] {
        let bit = 1usize << q;
        let mut cross = C::new(0.0, 0.0); // conj(a0) * a1 batura
        let mut z = 0.0;
        for (i, a) in self.amps.iter().enumerate() {
            if i & bit == 0 {
                let b = self.amps[i | bit];
                cross += a.conj() * b;
                z += a.norm_sqr() - b.norm_sqr();
            }
        }
        [2.0 * cross.re, 2.0 * cross.im, z]
    }

    // (a, b) qubiten fideltasuna (|00> + |11>)/√2 bell egoerarekin. simulagailuan bakarrik.
    pub fn bell_fidelity(&self, a: usize, b: usize) -> f64 {
        let (ba, bb) = (1usize << a, 1usize << b);
        let mut f = 0.0;
        for i in 0..self.amps.len() {
            if i & ba == 0 && i & bb == 0 {
                let s = (self.amps[i] + self.amps[i | ba | bb]) * std::f64::consts::FRAC_1_SQRT_2;
                f += s.norm_sqr();
            }
        }
        f
    }

    // o moduaren batez besteko fotoi-kopurua <n>. simulagailuan bakarrik.
    pub fn mean_photons(&self, o: usize) -> f64 {
        self.photon_distribution(o)
            .iter()
            .enumerate()
            .map(|(n, p)| n as f64 * p)
            .sum()
    }

    // ------------------------------------------------------------------ modu optikoak

    // d x d matrize bat o moduari aplikatu. gordetako fotoi-kopuruetatik kanpo erori den
    // probabilitatea itzultzen du (egoera normalizatzen da eta galera `leakage`-ri gehitzen zaio)
    pub fn apply_mode_matrix(&mut self, o: usize, m: &Mat) -> f64 {
        let d = self.cutoff;
        debug_assert_eq!(m.n, d);
        let stride = self.mode_stride(o);
        let sparse = Sparse::of(m);
        par_groups(&mut self.amps, stride, stride, d, 1, |buf, out, t| {
            sparse.apply(buf, out, t)
        });
        self.renormalize_after_truncation()
    }

    // anplitude bakoitza `f[o moduaren fotoi-kopurua]`-z biderkatu (fasea, kerr)
    pub fn apply_mode_diagonal(&mut self, o: usize, f: &[C]) {
        let (stride, d) = (self.mode_stride(o), self.cutoff);
        par_slabs(&mut self.amps, stride, |k, x| scale_run(x, f[k % d]));
    }

    // anplitude bakoitza `f[na * d + nb]`-z biderkatu (a, b) moduetarako (kerr gurutzatua)
    pub fn apply_two_mode_diagonal(&mut self, a: usize, b: usize, f: &[C]) {
        let (sa, sb) = (self.mode_stride(a), self.mode_stride(b));
        let d = self.cutoff;
        // pauso txikienaren xaflak: bi fotoi-kopuruak konstanteak dira xafla bakoitzean
        let s = sa.min(sb);
        par_slabs(&mut self.amps, s, |k, x| {
            let i = k * s;
            scale_run(x, f[((i / sa) % d) * d + (i / sb) % d])
        });
    }

    // d² x d² matrizea (na * d + nb indizearekin) (a, b) moduei aplikatu. trunkatze-ihesa itzultzen du.
    pub fn apply_two_mode_matrix(&mut self, a: usize, b: usize, m: &Mat) -> f64 {
        let d = self.cutoff;
        let (sa, sb) = (self.mode_stride(a), self.mode_stride(b));
        // taldeen bufferra (n_baxua, n_altua) ordenan dago; matrizea (na, nb) ordenan
        let sparse = if sa < sb {
            Sparse::of(m)
        } else {
            Sparse::of(&transpose_modes(m, d))
        };
        par_groups(
            &mut self.amps,
            sa.min(sb),
            sa.max(sb),
            d,
            2,
            |buf, out, t| sparse.apply(buf, out, t),
        );
        self.renormalize_after_truncation()
    }

    // a eta b moduen edukiak trukatu
    pub fn swap_modes(&mut self, a: usize, b: usize) {
        let d = self.cutoff;
        let (sa, sb) = (self.mode_stride(a), self.mode_stride(b));
        par_groups(&mut self.amps, sa.min(sb), sa.max(sb), d, 2, |buf, _, t| {
            for x in 0..d {
                for y in x + 1..d {
                    let (i, j) = ((x * d + y) * t, (y * d + x) * t);
                    let (lo, hi) = buf.split_at_mut(j);
                    lo[i..i + t].swap_with_slice(&mut hi[..t]);
                }
            }
        });
    }

    // moduaren fotoi-kopurua `from` denean, `to` bihurtzen du (eta alderantziz): permutazio bat.
    // FOCK-ek erabiltzen du modua neurtu ondoren, egoera n fotoira eramateko.
    pub fn move_photons(&mut self, o: usize, from: usize, to: usize) {
        if from == to {
            return;
        }
        let (stride, d) = (self.mode_stride(o), self.cutoff);
        let (x, y) = (from.min(to), from.max(to));
        par_slab_pairs(&mut self.amps, stride, d, x, y, 1, |a, b| {
            a.swap_with_slice(b)
        });
    }

    // kitzikapen-transferentzia q qubitaren eta o moduaren artean: |1>_q|0>_o <-> |0>_q|1>_o; beste
    // oinarri-egoerak ez dira aldatzen. permutazio bat da, beraz unitarioa; bi aldiz aplikatuta desegiten da.
    pub fn emit(&mut self, q: usize, o: usize) {
        let bit = 1usize << q;
        let (stride, d) = (self.mode_stride(o), self.cutoff);
        // n = 0 xaflan q = 1 diren lasterketak <-> n = 1 xaflan q = 0 direnak
        par_slab_pairs(&mut self.amps, stride, d, 0, 1, 2 * bit, |s0, s1| {
            for (a, b) in s0.chunks_mut(2 * bit).zip(s1.chunks_mut(2 * bit)) {
                a[bit..].swap_with_slice(&mut b[..bit]);
            }
        });
    }

    // qubit-modu akoplamendu dispertsiboa exp(iθ |1><1|_q n_o)
    pub fn dispersive(&mut self, q: usize, o: usize, theta: f64) {
        let (stride, d) = (self.mode_stride(o), self.cutoff);
        // n fotoiko xaflan, fase-ate bat q-n: diag(1, e^{iθn})
        let phases: Vec<[C; 4]> = (0..d)
            .map(|n| [ONE, ZERO, ZERO, C::from_polar(1.0, theta * n as f64)])
            .collect();
        par_slabs(&mut self.amps, stride, |k, x| {
            if k % d != 0 {
                kernel::mat_on(x, 0, q, &phases[k % d], 0);
            }
        });
    }

    // eragiketa ez-unitario edo moztu baten ondoren normalizatu; galdutako probabilitatea itzuli eta gorde
    pub fn renormalize_after_truncation(&mut self) -> f64 {
        let norm = self.norm_sqr();
        let lost = (1.0 - norm).max(0.0);
        if lost > 0.0 && norm > 0.0 {
            self.scale_all(1.0 / norm.sqrt());
        }
        // galera txikiak koma mugikorreko zarata dira, ez trunkatzea
        if lost > 1e-13 {
            self.leakage += lost;
        }
        lost
    }

    // anplitude guztiak eskalatu
    fn scale_all(&mut self, s: f64) {
        let len = self.amps.len();
        par_slabs(&mut self.amps, len, |_, x| scale_run(x, C::new(s, 0.0)));
    }

    // anplitude guztiak eskalatu araua 1 izan dadin. eskalatu aurreko araua² itzultzen du.
    pub fn normalize(&mut self) -> f64 {
        let norm = self.norm_sqr();
        if norm > 0.0 {
            self.scale_all(1.0 / norm.sqrt());
        }
        norm
    }

    // eskailera-eragileak o moduan: `up` bada a† (n -> n+1, sqrt(n+1)), bestela a (n -> n-1, sqrt n).
    // a†-rekin n = d-1 osagaia espaziotik irtengo litzateke, eta galdu egiten da.
    pub fn ladder(&mut self, o: usize, up: bool) {
        let d = self.cutoff;
        let stride = self.mode_stride(o);
        // a†: n xafla = sqrt(n) * (n-1 xafla zaharra); a: n xafla = sqrt(n+1) * (n+1 xafla zaharra)
        let shift: Vec<(usize, C)> = (0..d)
            .map(|n| {
                if up {
                    (n.wrapping_sub(1), C::new((n as f64).sqrt(), 0.0))
                } else {
                    (n + 1, C::new(((n + 1) as f64).sqrt(), 0.0))
                }
            })
            .collect();
        par_groups(&mut self.amps, stride, stride, d, 1, |buf, out, t| {
            shift_slabs(buf, out, t, &shift)
        });
    }

    // fotoi-galera kanal hutsaren jauzi kuantiko bat o moduan: fotoi bakoitza `gamma`
    // probabilitatez galtzen da. zenbat fotoi galdu diren lagintzen du (`r` erabiliz) eta dagokion
    // kraus eragilea aplikatzen du; exekuzioen batez bestekoak galera-kanala zehazki ematen du.
    // galdutako fotoi-kopurua itzultzen du.
    pub fn photon_loss(&mut self, o: usize, gamma: f64, r: f64) -> usize {
        let d = self.cutoff;
        if gamma <= 0.0 {
            return 0;
        }
        let dist = self.photon_distribution(o);
        // n fotoietatik k galtzeko probabilitate binomiala
        let weight = |n: usize, k: usize| -> f64 {
            if gamma >= 1.0 {
                return if k == n { 1.0 } else { 0.0 };
            }
            (crate::fock::ln_choose(n as u32, k as u32)
                + k as f64 * gamma.ln()
                + (n - k) as f64 * (1.0 - gamma).ln())
            .exp()
        };
        let mut pk = vec![0.0; d];
        for (n, p) in dist.iter().enumerate() {
            for (k, slot) in pk.iter_mut().enumerate().take(n + 1) {
                *slot += p * weight(n, k);
            }
        }
        let k = sample_index(&pk, r);
        // K_k kraus eragilea: |n> -> sqrt(weight(n, k)) |n - k>, hau da n xafla berria = (n + k xafla zaharra)
        let stride = self.mode_stride(o);
        let shift: Vec<(usize, C)> = (0..d)
            .map(|n| {
                (
                    n + k,
                    C::new(
                        if n + k < d {
                            weight(n + k, k).sqrt()
                        } else {
                            0.0
                        },
                        0.0,
                    ),
                )
            })
            .collect();
        par_groups(&mut self.amps, stride, stride, d, 1, |buf, out, t| {
            shift_slabs(buf, out, t, &shift)
        });
        self.normalize();
        k
    }
}

// indize bat aukeratu `weights[i]`-ren proportzioan, `r` erabiliz ([0, 1))
pub fn sample_index(weights: &[f64], r: f64) -> usize {
    let total: f64 = weights.iter().sum();
    let target = r * total;
    let mut acc = 0.0;
    let mut last_nonzero = 0;
    for (i, w) in weights.iter().enumerate() {
        if *w > 0.0 {
            last_nonzero = i;
        }
        acc += w;
        if target < acc {
            return i;
        }
    }
    last_nonzero
}

pub fn format_bytes(b: u128) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{} B", b)
    } else {
        format!("{:.1} {}", v, UNITS[u])
    }
}

// ------------------------------------------------------------------ moduen kernelak
//
// o moduaren egitura: fotoi-kopurua n duten anplitudeak `stride` luzerako lasterketa jarraituak dira
// ("xaflak"); d xafla jarraian bloke bat dira (stride * d). xafla bereko posizio berak talde bat osatzen
// du (n = 0..d), eta eragile optiko batek talde bakoitza bere aldetik nahasten du. bi modutan talde
// bakoitzak d * d anplitude ditu. funtzio hauek egitura horretan lan egiten dute, indize bakoitzeko
// zatiketarik gabe eta paraleloan.

// x-ren 2ren berretura zatitzaile handiena
fn low_bit(x: usize) -> usize {
    x & x.wrapping_neg()
}

// lasterketa bat f-z biderkatu (simd, luzera 4ren multiploa denean)
fn scale_run(x: &mut [C], f: C) {
    if f == ONE {
        return;
    }
    if f == ZERO {
        x.fill(ZERO);
    } else if x.len().is_multiple_of(4) {
        simd::scale(x, Spans::one(0, x.len()), &[f; 4]);
    } else {
        x.iter_mut().for_each(|a| *a *= f);
    }
}

// f(k, zatia) xafla bakoitzeko (k = xaflaren indizea egoeran); xafla luzeak zatitan, hari askotarako
fn par_slabs(amps: &mut [C], s: usize, f: impl Fn(usize, &mut [C]) + Sync) {
    let piece = if s <= PIECE || low_bit(s) < 1024 {
        s
    } else {
        low_bit(s).min(PIECE)
    };
    let per = s / piece;
    if amps.len() >= PARALLEL_MIN {
        amps.par_chunks_mut(piece)
            .enumerate()
            .for_each(|(k, x)| f(k / per, x));
    } else {
        amps.chunks_mut(piece)
            .enumerate()
            .for_each(|(k, x)| f(k / per, x));
    }
}

// bloke bakoitzean x eta y xaflak (x < y) f-ri, zati parekoetan (`align`-en multiploak) paraleloan
fn par_slab_pairs(
    amps: &mut [C],
    s: usize,
    d: usize,
    x: usize,
    y: usize,
    align: usize,
    f: impl Fn(&mut [C], &mut [C]) + Sync,
) {
    let block = s * d;
    fn pair(b: &mut [C], s: usize, x: usize, y: usize) -> (&mut [C], &mut [C]) {
        let (lo, hi) = b.split_at_mut(y * s);
        (&mut lo[x * s..(x + 1) * s], &mut hi[..s])
    }
    let tasks = rayon::current_num_threads() * 4;
    if amps.len() < PARALLEL_MIN {
        amps.chunks_mut(block).for_each(|b| {
            let (a, c) = pair(b, s, x, y);
            f(a, c)
        });
    } else if amps.len() / block >= tasks || low_bit(s) < align.max(1024) {
        amps.par_chunks_mut(block).for_each(|b| {
            let (a, c) = pair(b, s, x, y);
            f(a, c)
        });
    } else {
        let piece = low_bit(s).min(PIECE).max(align);
        for b in amps.chunks_mut(block) {
            let (a, c) = pair(b, s, x, y);
            a.par_chunks_mut(piece)
                .zip(c.par_chunks_mut(piece))
                .for_each(|(p, q)| f(p, q));
        }
    }
}

// egoeraren erakusle partekatua harien artean, talde disjuntuetarako (ikus `par_groups`)
struct Shared(*mut C);
// segurtasuna: talde bakoitzaren indizeak hari bakar batek ukitzen ditu
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

// taldeak (modu bat: d anplitude, s_lo = s_hi; bi modu: d * d, s_lo < s_hi) `t` posizio jarraituko
// zatitan: buf[errenkada * t + j], errenkada = n (modu bat) edo n_baxua * d + n_altua (bi modu).
// f(buf, aldi baterako, t)-k emaitza buf-en uzten du, eta egoerara itzultzen da. zati desberdinek talde
// desberdinak dituzte (oinarri desberdinak), beraz ez dute indizerik partekatzen eta paraleloan doaz
fn par_groups(
    amps: &mut [C],
    s_lo: usize,
    s_hi: usize,
    d: usize,
    modes: usize,
    f: impl Fn(&mut [C], &mut [C], usize) + Sync,
) {
    let n = amps.len();
    let rows = if modes == 1 { d } else { d * d };
    let tile = low_bit(s_lo).min(64);
    let r_tiles = s_lo / tile;
    // bi modutan: s_hi blokearen barruan, n_baxua = 0 duten azpibloke kopurua
    let t_count = if modes == 1 { 1 } else { s_hi / (s_lo * d) };
    let items = n / (rows * tile);
    let shared = Shared(amps.as_mut_ptr());
    let offsets: Vec<usize> = (0..rows)
        .map(|row| {
            if modes == 1 {
                row * s_lo
            } else {
                (row / d) * s_lo + (row % d) * s_hi
            }
        })
        .collect();
    let work = |(buf, tmp): &mut (Vec<C>, Vec<C>), k: usize| {
        let r0 = (k % r_tiles) * tile;
        let rest = k / r_tiles;
        let (t, u) = (rest % t_count, rest / t_count);
        let base = if modes == 1 {
            r0 + u * s_lo * d
        } else {
            r0 + t * s_lo * d + u * s_hi * d
        };
        let ptr = &shared;
        for (row, off) in offsets.iter().enumerate() {
            debug_assert!(base + off + tile <= n);
            // segurtasuna: talde hau zati honena da bakarrik, eta egoeraren barruan dago
            let src = unsafe { std::slice::from_raw_parts(ptr.0.add(base + off), tile) };
            buf[row * tile..(row + 1) * tile].copy_from_slice(src);
        }
        f(buf, tmp, tile);
        for (row, off) in offsets.iter().enumerate() {
            // segurtasuna: goian bezala
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr.0.add(base + off), tile) };
            dst.copy_from_slice(&buf[row * tile..(row + 1) * tile]);
        }
    };
    let init = || (vec![ZERO; rows * tile], vec![ZERO; rows * tile]);
    if n >= PARALLEL_MIN {
        (0..items).into_par_iter().for_each_init(init, work);
    } else {
        let mut bufs = init();
        (0..items).for_each(|k| work(&mut bufs, k));
    }
}

// matrize sakabanatua errenkadaka: (zutabea, balioa) zero ez direnak bakarrik. izpi-zatitzaile batek
// fotoi-kopuru osoa gordetzen du, beraz bere d² x d² matrizeak d³ inguru elementu ditu, ez d⁴
struct Sparse {
    rows: Vec<Vec<(usize, C)>>,
}

impl Sparse {
    fn of(m: &Mat) -> Sparse {
        let rows = (0..m.n)
            .map(|r| {
                (0..m.n)
                    .filter(|&c| m.a[r * m.n + c] != ZERO)
                    .map(|c| (c, m.a[r * m.n + c]))
                    .collect()
            })
            .collect();
        Sparse { rows }
    }

    // buf <- m * buf, errenkada bakoitza `t` luzerako bektore bat
    fn apply(&self, buf: &mut [C], out: &mut [C], t: usize) {
        for (r, row) in self.rows.iter().enumerate() {
            let o = &mut out[r * t..(r + 1) * t];
            o.fill(ZERO);
            for &(c, v) in row {
                for (y, x) in o.iter_mut().zip(&buf[c * t..(c + 1) * t]) {
                    *y += v * x;
                }
            }
        }
        buf.copy_from_slice(&out[..buf.len()]);
    }
}

// (na * d + nb) indizeko matrizea (nb * d + na) indizera
fn transpose_modes(m: &Mat, d: usize) -> Mat {
    let swap = |i: usize| (i % d) * d + i / d;
    let mut t = m.clone();
    for r in 0..d * d {
        for c in 0..d * d {
            t.a[swap(r) * d * d + swap(c)] = m.a[r * d * d + c];
        }
    }
    t
}

// xafla-desplazamendua: n errenkada berria = faktorea * (iturburu errenkada zaharra), iturburua d edo
// gehiago bada zero
fn shift_slabs(buf: &mut [C], out: &mut [C], t: usize, shift: &[(usize, C)]) {
    let d = shift.len();
    for (n, &(src, f)) in shift.iter().enumerate() {
        let o = &mut out[n * t..(n + 1) * t];
        if src < d {
            for (y, x) in o.iter_mut().zip(&buf[src * t..(src + 1) * t]) {
                *y = f * x;
            }
        } else {
            o.fill(ZERO);
        }
    }
    buf.copy_from_slice(&out[..buf.len()]);
}
