// fitxategi honek egoera-bektore bat prozesu askotan banatzen du (`qoa run --mpi --distribute`): nodo baten
// memoria gainditzen duten simulazioetarako.
//
// 2^g prozesurekin, goiko g qubitak "globalak" dira: prozesuaren rank-a dira. prozesu bakoitzak 2^L anplitude
// ditu (L = n - g), indize globala rank·2^L + tokikoa izanik.
//
//   - tokiko qubitetako ateak (helburua < L) sortan aplikatzen dira kernel arruntekin (fusioa, cache-blokeak,
//     SIMD); kontrol globalak rank-aren bitekin ebazten dira (betetzen ez badira, atea ez da ezer prozesu honetan)
//   - qubit global bateko ate diagonalek (RZ, faseak, CZ) ez dute trukerik behar: prozesu bakoitzak bere
//     faktorea aplikatzen du
//   - qubit global bateko beste ateek egoera bikotearekin trukatzen dute, 64 MiB-eko zatietan (memoria gehigarria
//     zati bat da, ez egoera osoa), eta bakoitzak bere erdia kalkulatzen du
//   - trukaketak qubit global batekin hiru CNOT bihurtzen dira
//   - neurketa: batura partzialak rank-en ordenan batzen dira (double-double), eta prozesu guztiek ausazko
//     zenbaki bera dute (programa bera, hazi bera), beraz emaitza bera ateratzen dute
//
// prozesu-kopurua 2-ren berretura izan behar da.

use crate::dd::{self, Dd};
use crate::kernel::{self, QGate};
use crate::state::State;
use crate::t;
use crate::team::Team;
use num_complex::Complex64 as C;
use rayon::prelude::*;
use std::sync::Arc;

// trukearen zatiak: 2^22 anplitude = 64 MiB
const CHUNK: usize = 1 << 22;

// txostenerako egoera osoa 0. prozesuan biltzen da hau baino txikiagoa bada (2^24 anplitude = 256 MiB)
pub const GATHER_MAX: usize = 1 << 24;

const ZERO: C = C::new(0.0, 0.0);

pub struct DistState {
    team: Arc<dyn Team>,
    n_qubits: usize,
    local_bits: usize,
    amps: Vec<C>,
    // `--gpu`: prozesu bakoitzaren tokiko ate-sortak bere GPUan; huts egiten badu, errorea hemen
    gpu: Option<Arc<crate::gpu::Gpu>>,
    error: Option<String>,
}

fn as_bytes(xs: &[C]) -> &[u8] {
    // SAFETY: Complex64 #[repr(C)] da, bi f64 (16 byte, betegarririk gabe)
    unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs)) }
}

fn as_bytes_mut(xs: &mut [C]) -> &mut [u8] {
    // SAFETY: ikus as_bytes; edozein byte-eredu f64 baliozkoa da
    unsafe { std::slice::from_raw_parts_mut(xs.as_mut_ptr() as *mut u8, std::mem::size_of_val(xs)) }
}

impl DistState {
    // egoera banatu bat posible den: prozesu-kopurua 2-ren berretura, qubit bat gutxienez prozesu bakoitzeko, eta
    // memorian sartzen dela. prozesu guztiek gauza bera kalkulatzen dute, beraz denek huts egiten dute batera,
    // inolako komunikaziorik gabe
    pub fn check(n_qubits: usize, ranks: usize) -> Result<usize, String> {
        if !ranks.is_power_of_two() {
            return Err(t!(
                "--distribute needs a power-of-two number of processes (1, 2, 4, 8, ...), not {}",
                "--distribute-k bi-ren berretura bat den prozesu-kopurua behar du (1, 2, 4, 8, ...), ez {}",
                ranks
            ));
        }
        let global = ranks.trailing_zeros() as usize;
        if n_qubits <= global {
            return Err(t!(
                "{} qubits cannot be split over {} processes: each process needs at least one qubit of its own",
                "{} qubit ezin dira {} prozesutan banatu: prozesu bakoitzak gutxienez bere qubit bat behar du",
                n_qubits,
                ranks
            ));
        }
        let local_bits = n_qubits - global;
        let len = 1u128 << local_bits;
        crate::state::check_memory(
            &t!(
                "{} qubits over {} processes: {} amplitudes in each",
                "{} qubit {} prozesutan: {} anplitude bakoitzean",
                n_qubits,
                ranks,
                len
            ),
            len * 16 + 2 * (CHUNK as u128) * 16,
        )?;
        Ok(local_bits)
    }

    pub fn new(
        n_qubits: usize,
        team: Arc<dyn Team>,
        gpu: Option<Arc<crate::gpu::Gpu>>,
    ) -> Result<DistState, String> {
        let local_bits = Self::check(n_qubits, team.size())?;
        let len = 1usize << local_bits;
        let mut amps = vec![ZERO; len];
        if team.rank() == 0 {
            amps[0] = C::new(1.0, 0.0);
        }
        Ok(DistState {
            team,
            n_qubits,
            local_bits,
            amps,
            gpu,
            error: None,
        })
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn ranks(&self) -> usize {
        self.team.size()
    }

    pub fn reset(&mut self) {
        self.amps.par_iter_mut().for_each(|a| *a = ZERO);
        if self.team.rank() == 0 {
            self.amps[0] = C::new(1.0, 0.0);
        }
    }

    // rank honen bit globalak, indize globalaren goiko bitetan
    fn high(&self) -> usize {
        self.team.rank() << self.local_bits
    }

    fn local_mask(&self) -> usize {
        (1usize << self.local_bits) - 1
    }

    // kontrol globalak betetzen dira prozesu honetan?
    fn global_ok(&self, controls: usize) -> bool {
        let g = controls & !self.local_mask();
        self.high() & g == g
    }

    pub fn apply(&mut self, gates: &[QGate]) {
        let mut local: Vec<QGate> = Vec::new();
        let lm = self.local_mask();
        for g in gates {
            match *g {
                QGate::Mat { q, m, controls } if q < self.local_bits => {
                    if self.global_ok(controls) {
                        local.push(QGate::Mat {
                            q,
                            m,
                            controls: controls & lm,
                        });
                    }
                }
                QGate::Swap { a, b, controls } if a.max(b) < self.local_bits => {
                    if self.global_ok(controls) {
                        local.push(QGate::Swap {
                            a,
                            b,
                            controls: controls & lm,
                        });
                    }
                }
                QGate::Mat { q, m, controls } => {
                    self.flush_local(&mut local);
                    self.global_mat(q, m, controls);
                }
                QGate::Swap { a, b, controls } => {
                    self.flush_local(&mut local);
                    // SWAP = CNOT(a→b) CNOT(b→a) CNOT(a→b), kontrol berekin
                    let x = [ZERO, C::new(1.0, 0.0), C::new(1.0, 0.0), ZERO];
                    let three = [
                        QGate::Mat {
                            q: b,
                            m: x,
                            controls: controls | 1 << a,
                        },
                        QGate::Mat {
                            q: a,
                            m: x,
                            controls: controls | 1 << b,
                        },
                        QGate::Mat {
                            q: b,
                            m: x,
                            controls: controls | 1 << a,
                        },
                    ];
                    self.apply(&three);
                }
            }
        }
        self.flush_local(&mut local);
    }

    fn flush_local(&mut self, local: &mut Vec<QGate>) {
        if local.is_empty() {
            return;
        }
        match &self.gpu {
            Some(g) => {
                if let Err(e) = g.apply(&mut self.amps, self.local_bits, local) {
                    self.error.get_or_insert(e);
                }
            }
            None => kernel::apply_gates(&mut self.amps, self.local_bits, local),
        }
        local.clear();
    }

    // GPUaren errorea, izan bada (VM-k exekuzioa gelditzen du)
    pub fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }

    // helburu globaleko 2x2 matrizea: diagonala bada, faktore bat; bestela bikotearekin trukatuta
    fn global_mat(&mut self, t: usize, m: [C; 4], controls: usize) {
        if !self.global_ok(controls) {
            return;
        }
        let lc = controls & self.local_mask();
        let gbit = t - self.local_bits;
        let mine = (self.team.rank() >> gbit) & 1;
        if m[1] == ZERO && m[2] == ZERO {
            let f = if mine == 0 { m[0] } else { m[3] };
            if f != C::new(1.0, 0.0) {
                self.amps
                    .par_iter_mut()
                    .enumerate()
                    .filter(|(i, _)| i & lc == lc)
                    .for_each(|(_, a)| *a *= f);
            }
            return;
        }
        let partner = self.team.rank() ^ (1 << gbit);
        let mut theirs = vec![ZERO; CHUNK.min(self.amps.len())];
        for (k, part) in self.amps.chunks_mut(CHUNK).enumerate() {
            let other = &mut theirs[..part.len()];
            self.team
                .exchange(partner, as_bytes(part), as_bytes_mut(other));
            let base = k * CHUNK;
            part.par_iter_mut()
                .zip(other.par_iter())
                .enumerate()
                .for_each(|(i, (a, b))| {
                    if (base + i) & lc == lc {
                        *a = if mine == 0 {
                            m[0] * *a + m[1] * *b
                        } else {
                            m[2] * *b + m[3] * *a
                        };
                    }
                });
        }
    }

    // rank guztien double-double batura partzialak, rank-en ordenan batuta (emaitza bera prozesu guztietan)
    fn total(&self, parts: &[Dd]) -> Vec<Dd> {
        let flat: Vec<f64> = parts.iter().flat_map(|d| [d.hi, d.lo]).collect();
        let all = self.team.allgather_f64(&flat);
        (0..parts.len())
            .map(|k| {
                all.iter().fold(Dd::ZERO, |acc, r| {
                    acc.plus(Dd {
                        hi: r[2 * k],
                        lo: r[2 * k + 1],
                    })
                })
            })
            .collect()
    }

    fn local_norm(&self) -> Dd {
        self.amps
            .par_chunks(1 << 14)
            .map(dd::norm_sum)
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Dd::ZERO, Dd::plus)
    }

    pub fn norm_error(&self) -> f64 {
        let n = self.total(&[self.local_norm()])[0];
        (1.0 - n.value()).abs()
    }

    pub fn prob_one(&self, q: usize) -> f64 {
        let part = if q < self.local_bits {
            let bit = 1usize << q;
            self.amps
                .par_chunks(1 << 14)
                .enumerate()
                .map(|(k, x)| dd::norm_sum_bit(x, k << 14, bit))
                .collect::<Vec<_>>()
                .into_iter()
                .fold(Dd::ZERO, Dd::plus)
        } else if (self.team.rank() >> (q - self.local_bits)) & 1 == 1 {
            self.local_norm()
        } else {
            Dd::ZERO
        };
        self.total(&[part])[0].value()
    }

    pub fn measure_qubit(&mut self, q: usize, r: f64) -> u8 {
        let p1 = self.prob_one(q).clamp(0.0, 1.0);
        let outcome = if r < p1 { 1 } else { 0 };
        let p = if outcome == 1 { p1 } else { 1.0 - p1 };
        let s = 1.0 / p.sqrt();
        if q < self.local_bits {
            let sc = C::new(s, 0.0);
            let m = if outcome == 1 {
                [ZERO, ZERO, ZERO, sc]
            } else {
                [sc, ZERO, ZERO, ZERO]
            };
            kernel::sweep_mat(&mut self.amps, q, &m, 0);
        } else {
            let keep = (self.team.rank() >> (q - self.local_bits)) & 1 == outcome as usize;
            self.amps
                .par_iter_mut()
                .for_each(|a| *a = if keep { *a * s } else { ZERO });
        }
        outcome
    }

    // prozesu honen probabilitateak eta prozesu guztien batura, rank-en ordenan
    pub fn local_probabilities(&self) -> (Vec<f64>, Vec<f64>) {
        let probs: Vec<f64> = self.amps.par_iter().map(|a| a.norm_sqr()).collect();
        let mine = dd::sum(probs.iter().copied());
        let totals = self.team.allgather_f64(&[mine]);
        (probs, totals.into_iter().map(|v| v[0]).collect())
    }

    pub fn local_bits(&self) -> usize {
        self.local_bits
    }

    // q qubitaren <X>, <Y>, <Z>
    pub fn bloch(&self, q: usize) -> [f64; 3] {
        let (mut re, mut im, mut z) = (Dd::ZERO, Dd::ZERO, Dd::ZERO);
        let pair = |a: C, b: C, re: &mut Dd, im: &mut Dd| {
            re.add_prod::<false>(a.re, b.re);
            re.add_prod::<false>(a.im, b.im);
            im.add_prod::<false>(a.re, b.im);
            im.add_prod::<false>(-a.im, b.re);
        };
        if q < self.local_bits {
            let bit = 1usize << q;
            for (i, a) in self.amps.iter().enumerate() {
                if i & bit == 0 {
                    let b = self.amps[i | bit];
                    pair(*a, b, &mut re, &mut im);
                    z.add_norm::<false>(*a);
                    z.add_prod::<false>(-b.re, b.re);
                    z.add_prod::<false>(-b.im, b.im);
                }
            }
        } else {
            let gbit = q - self.local_bits;
            let mine = (self.team.rank() >> gbit) & 1;
            let theirs = self.partner_copy(self.team.rank() ^ (1 << gbit));
            let n = self.local_norm();
            z = if mine == 0 { n } else { -n };
            if mine == 0 {
                for (a, b) in self.amps.iter().zip(&theirs) {
                    pair(*a, *b, &mut re, &mut im);
                }
            }
        }
        let t = self.total(&[re, im, z]);
        [2.0 * t[0].value(), 2.0 * t[1].value(), t[2].value()]
    }

    // (a, b) qubiten fideltasuna (|00> + |11>)/√2 bell egoerarekin
    pub fn bell_fidelity(&self, a: usize, b: usize) -> f64 {
        let both = (1usize << a) | (1usize << b);
        let (lm, gm) = (both & self.local_mask(), both >> self.local_bits);
        let rank = self.team.rank();
        let mut f = Dd::ZERO;
        if gm == 0 {
            for i in 0..self.amps.len() {
                if i & lm == 0 {
                    f.add_norm::<false>(self.amps[i] + self.amps[i | lm]);
                }
            }
        } else {
            let theirs = self.partner_copy(rank ^ gm);
            if rank & gm == 0 {
                for i in 0..self.amps.len() {
                    if i & lm == 0 {
                        f.add_norm::<false>(self.amps[i] + theirs[i | lm]);
                    }
                }
            }
        }
        self.total(&[f])[0].value() / 2.0
    }

    // bikotearen egoera osoa (zatika trukatuta)
    fn partner_copy(&self, partner: usize) -> Vec<C> {
        let mut theirs = vec![ZERO; self.amps.len()];
        for (mine, other) in self.amps.chunks(CHUNK).zip(theirs.chunks_mut(CHUNK)) {
            self.team
                .exchange(partner, as_bytes(mine), as_bytes_mut(other));
        }
        theirs
    }

    // egoera osoa 0. prozesuan, GATHER_MAX anplitude baino gutxiago badira (besteek, eta handiegia bada, None)
    pub fn gather(&self) -> Option<State> {
        let total = 1usize << self.n_qubits;
        if total > GATHER_MAX {
            return None;
        }
        let parts = self.team.gather(as_bytes(&self.amps).to_vec())?;
        let mut amps = Vec::with_capacity(total);
        for p in parts {
            amps.extend(p.as_chunks::<16>().0.iter().map(|c| {
                C::new(
                    f64::from_ne_bytes(c[..8].try_into().unwrap()),
                    f64::from_ne_bytes(c[8..].try_into().unwrap()),
                )
            }));
        }
        State::from_amplitudes(self.n_qubits, 0, 2, amps).ok()
    }
}
