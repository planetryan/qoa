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
use crate::t;
use num_complex::Complex64 as C;
use rayon::prelude::*;

// anplitude gutxiago badira, kernelak hari bakarrean (hariek aurrezten dutena baino gehiago kostatzen dute)
const PARALLEL_MIN: usize = 1 << 14;

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
    // diren oinarri-egoeretan (`controls` bit-maskara bat da; 0 kontrolik gabe)
    pub fn apply_1q(&mut self, q: usize, m: [C; 4], controls: usize) {
        let [m00, m01, m10, m11] = m;
        // ate diagonalek eskalatu bakarrik egiten dute: merkeagoa, eta bikoteak ukitu gabe
        if m01 == C::new(0.0, 0.0) && m10 == C::new(0.0, 0.0) {
            let bit = 1usize << q;
            self.for_each_indexed(|i, a| {
                if i & controls == controls {
                    *a *= if i & bit == 0 { m00 } else { m11 };
                }
            });
            return;
        }
        pair_kernel(&mut self.amps, q, controls, |a, b| {
            let x = *a;
            let y = *b;
            *a = m00 * x + m01 * y;
            *b = m10 * x + m11 * y;
        });
    }

    // pauli-x q-n (kontrolekin edo gabe) trukaketa gisa, biderketarik gabe
    pub fn apply_x(&mut self, q: usize, controls: usize) {
        pair_kernel(&mut self.amps, q, controls, std::mem::swap);
    }

    // e^{iθ}-z biderkatu `mask`-eko qubit guztiak 1 diren oinarri-egoera guztiak
    pub fn apply_phase_mask(&mut self, mask: usize, theta: f64) {
        let phase = C::from_polar(1.0, theta);
        self.for_each_indexed(|i, a| {
            if i & mask == mask {
                *a *= phase;
            }
        });
    }

    // a eta b qubitak trukatu, `controls` guztiak 1 diren oinarri-egoeretan.
    // hi qubitaren bloke bakoitza bi erditan banatzen da (hi=0 | hi=1); erdi baxuko (lo=1) elementua
    // erdi altuko (lo=0) elementuarekin trukatzen da. erdiak bereiziak direnez, ez da unsafe behar.
    pub fn apply_swap(&mut self, a: usize, b: usize, controls: usize) {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let lo_bit = 1usize << lo;
        let half = 1usize << hi;
        let block = half << 1;
        let sub = lo_bit << 1;
        // azpibloke bat: lo erdian lo=1 dutenak (lo_bit..sub), hi erdian lo=0 dutenak (0..lo_bit)
        let swap_sub = |base: usize, lo_part: &mut [C], hi_part: &mut [C]| {
            for m in 0..lo_bit {
                if (base + lo_bit + m) & controls == controls {
                    std::mem::swap(&mut lo_part[lo_bit + m], &mut hi_part[m]);
                }
            }
        };
        let n = self.amps.len();
        let run_block = |k: usize, chunk: &mut [C]| {
            let (lo_half, hi_half) = chunk.split_at_mut(half);
            for (s, (lp, hp)) in lo_half
                .chunks_mut(sub)
                .zip(hi_half.chunks_mut(sub))
                .enumerate()
            {
                swap_sub(k * block + s * sub, lp, hp);
            }
        };
        if n < PARALLEL_MIN {
            for (k, chunk) in self.amps.chunks_mut(block).enumerate() {
                run_block(k, chunk);
            }
        } else if n / block >= rayon::current_num_threads() * 4 {
            self.amps
                .par_chunks_mut(block)
                .enumerate()
                .for_each(|(k, chunk)| run_block(k, chunk));
        } else {
            for (k, chunk) in self.amps.chunks_mut(block).enumerate() {
                let (lo_half, hi_half) = chunk.split_at_mut(half);
                lo_half
                    .par_chunks_mut(sub)
                    .zip(hi_half.par_chunks_mut(sub))
                    .enumerate()
                    .for_each(|(s, (lp, hp))| swap_sub(k * block + s * sub, lp, hp));
            }
        }
    }

    fn for_each_indexed(&mut self, f: impl Fn(usize, &mut C) + Sync + Send) {
        if self.amps.len() >= PARALLEL_MIN {
            self.amps
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, a)| f(i, a));
        } else {
            self.amps.iter_mut().enumerate().for_each(|(i, a)| f(i, a));
        }
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
        let bit = 1usize << q;
        let keep = if outcome == 1 { bit } else { 0 };
        let scale = 1.0 / p.sqrt();
        self.for_each_indexed(|i, a| {
            if i & bit == keep {
                *a *= scale;
            } else {
                *a = C::new(0.0, 0.0);
            }
        });
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
        let mut dist = vec![0.0; self.cutoff];
        for (i, a) in self.amps.iter().enumerate() {
            dist[self.photons(i, o)] += a.norm_sqr();
        }
        dist
    }

    // o moduko fotoi-kopuruak `keep` betetzen duen oinarri-egoerak bakarrik gorde, gordetako `p`
    // probabilitatearekin normalizatuta
    pub fn collapse_mode(&mut self, o: usize, p: f64, keep: impl Fn(usize) -> bool + Sync + Send) {
        let stride = self.mode_stride(o);
        let cutoff = self.cutoff;
        let scale = 1.0 / p.sqrt();
        self.for_each_indexed(|i, a| {
            if keep((i / stride) % cutoff) {
                *a *= scale;
            } else {
                *a = C::new(0.0, 0.0);
            }
        });
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
        let block = stride * d;
        let work = |chunk: &mut [C]| {
            let mut v = vec![C::new(0.0, 0.0); d];
            for j in 0..stride {
                for (k, x) in v.iter_mut().enumerate() {
                    *x = chunk[j + k * stride];
                }
                for r in 0..d {
                    let mut acc = C::new(0.0, 0.0);
                    for (c, x) in v.iter().enumerate() {
                        acc += m.a[r * d + c] * x;
                    }
                    chunk[j + r * stride] = acc;
                }
            }
        };
        if self.amps.len() >= PARALLEL_MIN && self.amps.len() / block >= 2 {
            self.amps.par_chunks_mut(block).for_each(work);
        } else {
            self.amps.chunks_mut(block).for_each(work);
        }
        self.renormalize_after_truncation()
    }

    // anplitude bakoitza `f[o moduaren fotoi-kopurua]`-z biderkatu (fasea, kerr)
    pub fn apply_mode_diagonal(&mut self, o: usize, f: &[C]) {
        let stride = self.mode_stride(o);
        let d = self.cutoff;
        self.for_each_indexed(|i, a| *a *= f[(i / stride) % d]);
    }

    // anplitude bakoitza `f[na * d + nb]`-z biderkatu (a, b) moduetarako (kerr gurutzatua)
    pub fn apply_two_mode_diagonal(&mut self, a: usize, b: usize, f: &[C]) {
        let (sa, sb) = (self.mode_stride(a), self.mode_stride(b));
        let d = self.cutoff;
        self.for_each_indexed(|i, x| *x *= f[((i / sa) % d) * d + (i / sb) % d]);
    }

    // d² x d² matrizea (na * d + nb indizearekin) (a, b) moduei aplikatu. trunkatze-ihesa itzultzen du.
    pub fn apply_two_mode_matrix(&mut self, a: usize, b: usize, m: &Mat) -> f64 {
        let d = self.cutoff;
        let (sa, sb) = (self.mode_stride(a), self.mode_stride(b));
        let n = self.amps.len();
        let mut v = vec![C::new(0.0, 0.0); d * d];
        let mut out = vec![C::new(0.0, 0.0); d * d];
        for base in 0..n {
            // talde bakoitza behin bisitatu, bi fotoi-kopuruak 0 dituen kidetik
            if (base / sa) % d != 0 || (base / sb) % d != 0 {
                continue;
            }
            for na in 0..d {
                for nb in 0..d {
                    v[na * d + nb] = self.amps[base + na * sa + nb * sb];
                }
            }
            for (r, o) in out.iter_mut().enumerate() {
                let row = &m.a[r * d * d..(r + 1) * d * d];
                *o = row.iter().zip(&v).map(|(x, y)| x * y).sum();
            }
            for na in 0..d {
                for nb in 0..d {
                    self.amps[base + na * sa + nb * sb] = out[na * d + nb];
                }
            }
        }
        self.renormalize_after_truncation()
    }

    // a eta b moduen edukiak trukatu
    pub fn swap_modes(&mut self, a: usize, b: usize) {
        let d = self.cutoff;
        let (sa, sb) = (self.mode_stride(a), self.mode_stride(b));
        for i in 0..self.amps.len() {
            let (na, nb) = ((i / sa) % d, (i / sb) % d);
            if na < nb {
                let j = i - na * sa - nb * sb + nb * sa + na * sb;
                self.amps.swap(i, j);
            }
        }
    }

    // moduaren fotoi-kopurua `from` denean, `to` bihurtzen du (eta alderantziz): permutazio bat.
    // FOCK-ek erabiltzen du modua neurtu ondoren, egoera n fotoira eramateko.
    pub fn move_photons(&mut self, o: usize, from: usize, to: usize) {
        if from == to {
            return;
        }
        let stride = self.mode_stride(o);
        let d = self.cutoff;
        for i in 0..self.amps.len() {
            if (i / stride) % d == from {
                let j = i - from * stride + to * stride;
                self.amps.swap(i, j);
            }
        }
    }

    // kitzikapen-transferentzia q qubitaren eta o moduaren artean: |1>_q|0>_o <-> |0>_q|1>_o; beste
    // oinarri-egoerak ez dira aldatzen. permutazio bat da, beraz unitarioa; bi aldiz aplikatuta desegiten da.
    pub fn emit(&mut self, q: usize, o: usize) {
        let bit = 1usize << q;
        let stride = self.mode_stride(o);
        let d = self.cutoff;
        for i in 0..self.amps.len() {
            if i & bit != 0 && (i / stride).is_multiple_of(d) {
                let j = (i & !bit) + stride;
                self.amps.swap(i, j);
            }
        }
    }

    // qubit-modu akoplamendu dispertsiboa exp(iθ |1><1|_q n_o)
    pub fn dispersive(&mut self, q: usize, o: usize, theta: f64) {
        let bit = 1usize << q;
        let stride = self.mode_stride(o);
        let d = self.cutoff;
        self.for_each_indexed(|i, a| {
            if i & bit != 0 {
                *a *= C::from_polar(1.0, theta * ((i / stride) % d) as f64);
            }
        });
    }

    // eragiketa ez-unitario edo moztu baten ondoren normalizatu; galdutako probabilitatea itzuli eta gorde
    pub fn renormalize_after_truncation(&mut self) -> f64 {
        let norm = self.norm_sqr();
        let lost = (1.0 - norm).max(0.0);
        if lost > 0.0 && norm > 0.0 {
            let s = 1.0 / norm.sqrt();
            self.for_each_indexed(|_, a| *a *= s);
        }
        // galera txikiak koma mugikorreko zarata dira, ez trunkatzea
        if lost > 1e-13 {
            self.leakage += lost;
        }
        lost
    }

    // anplitude guztiak eskalatu araua 1 izan dadin. eskalatu aurreko araua² itzultzen du.
    pub fn normalize(&mut self) -> f64 {
        let norm = self.norm_sqr();
        if norm > 0.0 {
            let s = 1.0 / norm.sqrt();
            self.for_each_indexed(|_, a| *a *= s);
        }
        norm
    }

    // eskailera-eragileak o moduan: `up` bada a† (n -> n+1, sqrt(n+1)), bestela a (n -> n-1, sqrt n).
    // a†-rekin n = d-1 osagaia espaziotik irtengo litzateke, eta galdu egiten da.
    pub fn ladder(&mut self, o: usize, up: bool) {
        let d = self.cutoff;
        let stride = self.mode_stride(o);
        let block = stride * d;
        let work = |chunk: &mut [C]| {
            for j in 0..stride {
                if up {
                    for n in (0..d).rev() {
                        let v = if n == 0 {
                            C::new(0.0, 0.0)
                        } else {
                            chunk[j + (n - 1) * stride] * (n as f64).sqrt()
                        };
                        chunk[j + n * stride] = v;
                    }
                } else {
                    for n in 0..d {
                        let v = if n + 1 < d {
                            chunk[j + (n + 1) * stride] * ((n + 1) as f64).sqrt()
                        } else {
                            C::new(0.0, 0.0)
                        };
                        chunk[j + n * stride] = v;
                    }
                }
            }
        };
        if self.amps.len() >= PARALLEL_MIN && self.amps.len() / block >= 2 {
            self.amps.par_chunks_mut(block).for_each(work);
        } else {
            self.amps.chunks_mut(block).for_each(work);
        }
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
        // K_k kraus eragilea: |n> -> sqrt(weight(n, k)) |n - k>
        let stride = self.mode_stride(o);
        let block = stride * d;
        for chunk in self.amps.chunks_mut(block) {
            for j in 0..stride {
                for n in 0..d {
                    let v = if n + k < d {
                        chunk[j + (n + k) * stride] * weight(n + k, k).sqrt()
                    } else {
                        C::new(0.0, 0.0)
                    };
                    chunk[j + n * stride] = v;
                }
            }
        }
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

// q qubitaren bikote bakoitza bisitatu (q bita 0 duen indizea, q bita 1 duen indize bera), kontrol-
// maskara betetzen den lekuetan, `f(&mut baxua, &mut altua)` deituz. bikote bakoitza behin bakarrik.
fn pair_kernel(
    amps: &mut [C],
    q: usize,
    controls: usize,
    f: impl Fn(&mut C, &mut C) + Sync + Send,
) {
    let half = 1usize << q;
    let block = half << 1;
    let run_block = |base: usize, chunk: &mut [C]| {
        let (lo, hi) = chunk.split_at_mut(half);
        if controls == 0 {
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                f(a, b);
            }
        } else {
            for (j, (a, b)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                if (base + j) & controls == controls {
                    f(a, b);
                }
            }
        }
    };
    let n = amps.len();
    if n < PARALLEL_MIN {
        for (k, chunk) in amps.chunks_mut(block).enumerate() {
            run_block(k * block, chunk);
        }
    } else if n / block >= rayon::current_num_threads() * 4 {
        // bloke txiki asko: zeregin bat bloke bakoitzeko
        amps.par_chunks_mut(block)
            .enumerate()
            .for_each(|(k, chunk)| run_block(k * block, chunk));
    } else {
        // bloke handi gutxi (qubit altua): bloke bakoitzaren erdiak harien artean banatu
        for (k, chunk) in amps.chunks_mut(block).enumerate() {
            let base = k * block;
            let (lo, hi) = chunk.split_at_mut(half);
            lo.par_iter_mut()
                .zip(hi.par_iter_mut())
                .enumerate()
                .for_each(|(j, (a, b))| {
                    if (base + j) & controls == controls {
                        f(a, b);
                    }
                });
        }
    }
}
