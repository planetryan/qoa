// fitxategi honek programa bat hardware-profil batekin egiaztatzen du (`qoa check --target`): mahai gaineko
// optika-banku batek (docs/bench.md) exekuta dezakeena. bankuak egin ezin duen eragiketa bakoitzarentzat
// lerroa eta arrazoia ematen ditu, eta ez du ezer isilean aldatzen.
//
//   bench-a  argi klasikoa: laserra, izpi-zatitzaileak, faseak, fotodiodoak. fotoi bat gehienez (uhin
//            klasikoek fotoi bateko estatistikak ematen dituzte) eta qubit bat gehienez (izpiaren polarizazioa)
//   bench-b  fotoi bakar errealak: bikote-iturri bat (bi fotoi gehienez), klik-detektagailuak eta
//            koinzidentziak; qubitak bikoteko bi fotoien polarizazioak dira, `BELL` iturria da, eta qubit
//            bateko ate guztiak xaflak dira

use crate::program::{Dest, Op, Program};
use crate::t;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    BenchA,
    BenchB,
}

impl Target {
    pub fn name(self) -> &'static str {
        match self {
            Target::BenchA => "bench-a",
            Target::BenchB => "bench-b",
        }
    }

    pub fn parse(s: &str) -> Option<Target> {
        match s {
            "bench-a" => Some(Target::BenchA),
            "bench-b" => Some(Target::BenchB),
            _ => None,
        }
    }

    pub const ALL: [Target; 2] = [Target::BenchA, Target::BenchB];

    // bankuaren detektagailu-kopurua: neurtzen diren moduak eta qubitak (polarizazio bakoitzak bat behar du)
    fn detectors(self) -> usize {
        match self {
            Target::BenchA => 2,
            Target::BenchB => 4,
        }
    }

    fn photons(self) -> usize {
        match self {
            Target::BenchA => 1,
            Target::BenchB => 2,
        }
    }

    fn qubits(self) -> u32 {
        match self {
            Target::BenchA => 1,
            Target::BenchB => 2,
        }
    }
}

// bankuak exekutatu ezin duen eragiketa bat: lerroa (0 programa osoarentzat) eta arrazoia
#[derive(Clone, Debug, PartialEq)]
pub struct Problem {
    pub line: u32,
    pub message: String,
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line > 0 {
            write!(
                f,
                "{}",
                t!("line {}: {}", "{}. lerroa: {}", self.line, self.message)
            )
        } else {
            write!(f, "{}", self.message)
        }
    }
}

// eragiketaren QOA izena mezuetarako
fn mnemonic(op: &Op) -> &'static str {
    crate::disasm::op_name(op).unwrap_or("?")
}

pub fn check(p: &Program, target: Target) -> Vec<Problem> {
    // programa osoari dagozkionak lehenik, gero eragiketenak ordenan
    let mut whole = Vec::new();
    let mut out = Vec::new();
    let mut add = |pc: usize, message: String| {
        out.push(Problem {
            line: p.line_of(pc),
            message,
        })
    };
    let bench = target.name();
    let uses_qubits = p
        .ops
        .iter()
        .any(|o| !o.gate_qubits().is_empty() || matches!(o, Op::Measure { .. } | Op::MeasureAll));
    let uses_modes = p.n_modes > 0;
    if target == Target::BenchB && uses_qubits && uses_modes {
        whole.push(Problem {
            line: 0,
            message: t!(
                "on {} the qubits are photon polarisations and the modes are photon paths; a program uses one or the other",
                "{}-n qubitak fotoien polarizazioak dira eta moduak fotoien bideak; programa batek bata edo bestea erabiltzen du",
                bench
            ),
        });
    }
    if p.n_qubits > target.qubits() {
        whole.push(Problem {
            line: 0,
            message: t!(
                "the program has {} qubits; {} has {} at most",
                "programak {} qubit ditu; {}-k {} gehienez",
                p.n_qubits,
                bench,
                target.qubits()
            ),
        });
    }
    // qubit eta modu bakoitzaren egoera: erabilia (ateak), neurtua (fotoia joan da)
    let mut touched_q: HashSet<u32> = HashSet::new();
    let mut measured_q: HashSet<u32> = HashSet::new();
    let mut used_o: HashSet<u32> = HashSet::new();
    let mut detected_o: HashSet<u32> = HashSet::new();
    let mut photons = 0usize;
    for (pc, op) in p.ops.iter().enumerate() {
        let name = mnemonic(op);
        // qubit neurtu bat: fotoia detektatu da, eta ez dago gehiago
        let after_measure = |qs: &[u32]| qs.iter().find(|q| measured_q.contains(q)).copied();
        match op {
            Op::DumpState | Op::Expect { .. } | Op::BellCheck { .. } | Op::MeanPhoton { .. } => {
                add(
                    pc,
                    t!(
                        "{} reads the exact quantum state without measuring it; only a simulator can do that",
                        "{}-k egoera kuantiko zehatza irakurtzen du neurtu gabe; simulagailu batek bakarrik egin dezake hori",
                        name
                    ),
                )
            }
            Op::Jump { .. }
            | Op::JumpIf { .. }
            | Op::Branch { .. }
            | Op::BranchV { .. }
            | Op::Call { .. }
            | Op::Return
            | Op::LoopStart { .. }
            | Op::LoopEnd { .. } => add(
                pc,
                t!(
                    "{}: the bench runs the same fixed sequence in every shot, without jumps, loops or calls (branching on a measurement needs fast switches: tier C)",
                    "{}: bankuak sekuentzia finko bera exekutatzen du shot bakoitzean, jauzirik, begiztarik eta deirik gabe (neurketa baten araberako adarkatzeak etengailu azkarrak behar ditu: C maila)",
                    name
                ),
            ),
            Op::PackBits { .. } => add(
                pc,
                t!(
                    "PACKBITS reads measurement results during the run; the bench gives them only at the end, as counts",
                    "PACKBITS-ek neurketen emaitzak exekuzioaren erdian irakurtzen ditu; bankuak amaieran bakarrik ematen ditu, kontaketa gisa"
                ),
            ),
            Op::DisplaceReg { .. } | Op::SqueezeReg { .. } => add(
                pc,
                t!(
                    "{} sets the optics from a register during the run (feed-forward needs fast switches: tier C)",
                    "{}-k optika erregistro batetik ezartzen du exekuzioaren erdian (aurrera-elikadurak etengailu azkarrak behar ditu: C maila)",
                    name
                ),
            ),
            Op::Kerr { .. } | Op::CrossKerr { .. } => add(
                pc,
                t!(
                    "{}: a nonlinearity at the single-photon level needs cavity QED or Rydberg atoms, research-lab equipment",
                    "{}: fotoi bakarreko mailako ez-linealtasun batek kabitate-QED edo Rydberg atomoak behar ditu, ikerketa-laborategiko ekipoa",
                    name
                ),
            ),
            Op::Emit { .. } | Op::Dispersive { .. } => add(
                pc,
                t!(
                    "{} couples a qubit to a mode, which needs a quantum emitter in a cavity (research-lab equipment)",
                    "{}-k qubit bat modu batekin akoplatzen du, eta horrek igorle kuantiko bat behar du kabitate batean (ikerketa-laborategiko ekipoa)",
                    name
                ),
            ),
            Op::Squeeze { .. }
            | Op::PhotonCount { .. }
            | Op::Parity { .. }
            | Op::AddPhoton { .. }
            | Op::SubPhoton { .. } => {
                let what = match op {
                    Op::Squeeze { .. } => t!(
                        "squeezed light needs an optical parametric oscillator and homodyne detection",
                        "argi estutuak osziladore parametriko optiko bat eta detekzio homodinoa behar ditu"
                    ),
                    Op::PhotonCount { .. } => t!(
                        "counting photons needs number-resolving (cryogenic) detectors; DETECT (a click) works on this bench",
                        "fotoiak zenbatzeak kopurua bereizten duten detektagailuak (kriogenikoak) behar ditu; DETECT-ek (klik bat) banku honetan balio du"
                    ),
                    Op::Parity { .. } => t!(
                        "photon-number parity needs number-resolving detectors",
                        "fotoi-kopuruaren paritateak kopurua bereizten duten detektagailuak behar ditu"
                    ),
                    _ => t!(
                        "adding or removing a photon is heralded and needs extra sources and detectors",
                        "fotoi bat gehitzea edo kentzea iragarria da eta iturri eta detektagailu gehiago behar ditu"
                    ),
                };
                add(pc, t!("{}: {} (tier C)", "{}: {} (C maila)", name, what));
            }
            Op::Displace { o, .. } => {
                if target == Target::BenchB {
                    add(
                        pc,
                        t!(
                            "DISPLACE needs a laser at the photons' wavelength coupled into the paths, which is not part of {}",
                            "DISPLACE-k fotoien uhin-luzerako laser bat behar du bideetan akoplatuta, eta hori ez dago {}-n",
                            bench
                        ),
                    );
                }
                used_o.insert(*o);
            }
            Op::Fock { o, n } => {
                if *n != 1 {
                    add(
                        pc,
                        t!(
                            "FOCK {}: the source makes single photons; {} photons in one mode need number states the bench cannot make",
                            "FOCK {}: iturriak fotoi bakarrak egiten ditu; {} fotoi modu batean bankuak egin ezin dituen kopuru-egoerak dira",
                            n,
                            n
                        ),
                    );
                } else if used_o.contains(o) {
                    add(
                        pc,
                        t!(
                            "SINGLEPHOTON o{}: photons come from the source at the start of a path; this path is already in use",
                            "SINGLEPHOTON o{}: fotoiak iturritik datoz bide baten hasieran; bide hau jada erabiltzen ari da",
                            o
                        ),
                    );
                } else {
                    photons += 1;
                    if photons > target.photons() {
                        add(
                            pc,
                            t!(
                                "{} makes {} photon(s) per shot at most",
                                "{}-k {} fotoi egiten ditu gehienez shot bakoitzeko",
                                bench,
                                target.photons()
                            ),
                        );
                    }
                }
                used_o.insert(*o);
            }
            Op::BeamSplitter { a, b, .. } | Op::ModeSwap { a, b } => {
                for o in [a, b] {
                    if detected_o.contains(o) {
                        add(
                            pc,
                            t!(
                                "o{} was detected already; a detected photon is gone",
                                "o{} jada detektatu da; detektatutako fotoi bat joan egin da",
                                o
                            ),
                        );
                    }
                    used_o.insert(*o);
                }
            }
            Op::ModePhase { o, .. } | Op::Loss { o, .. } => {
                if detected_o.contains(o) {
                    add(
                        pc,
                        t!(
                            "o{} was detected already; a detected photon is gone",
                            "o{} jada detektatu da; detektatutako fotoi bat joan egin da",
                            o
                        ),
                    );
                }
                used_o.insert(*o);
            }
            Op::PhotonDetect { o, dest } => {
                if matches!(dest, Dest::Reg(_)) {
                    add(
                        pc,
                        t!(
                            "DETECT into a register: the bench gives clicks as bits",
                            "DETECT erregistro batera: bankuak klikak bit gisa ematen ditu"
                        ),
                    );
                }
                detected_o.insert(*o);
            }
            Op::Coincidence { modes, dest } => {
                if target == Target::BenchA {
                    add(
                        pc,
                        t!(
                            "COINCIDENCE needs two photons; classical light gives one-photon statistics only",
                            "COINCIDENCE-k bi fotoi behar ditu; argi klasikoak fotoi bateko estatistikak bakarrik ematen ditu"
                        ),
                    );
                }
                if matches!(dest, Dest::Reg(_)) {
                    add(
                        pc,
                        t!(
                            "COINCIDENCE into a register: the bench gives clicks as bits",
                            "COINCIDENCE erregistro batera: bankuak klikak bit gisa ematen ditu"
                        ),
                    );
                }
                detected_o.extend(modes.iter().copied());
            }
            Op::Gate { q, .. } | Op::ToBasis { q, .. } => {
                if let Some(m) = after_measure(&[*q]) {
                    add(
                        pc,
                        t!(
                            "q{} was measured already; a measured photon is gone",
                            "q{} jada neurtu da; neurtutako fotoi bat joan egin da",
                            m
                        ),
                    );
                }
                touched_q.insert(*q);
            }
            Op::Bell { a, b } => {
                if target == Target::BenchA {
                    add(
                        pc,
                        t!(
                            "BELL needs an entangled photon pair (tier B)",
                            "BELL-ek fotoi-bikote entrelazatu bat behar du (B maila)"
                        ),
                    );
                } else if touched_q.contains(a)
                    || touched_q.contains(b)
                    || measured_q.contains(a)
                    || measured_q.contains(b)
                {
                    add(
                        pc,
                        t!(
                            "BELL is the photon-pair source on {}: it must come before anything else on q{} and q{}",
                            "BELL fotoi-bikoteen iturria da {}-n: q{} eta q{}-ren gaineko beste ezer baino lehen joan behar du",
                            bench,
                            a,
                            b
                        ),
                    );
                }
                touched_q.insert(*a);
                touched_q.insert(*b);
            }
            Op::Measure { q, dest, .. } => {
                if let Some(m) = after_measure(&[*q]) {
                    add(
                        pc,
                        t!(
                            "q{} was measured already; a measured photon is gone",
                            "q{} jada neurtu da; neurtutako fotoi bat joan egin da",
                            m
                        ),
                    );
                }
                if matches!(dest, Dest::Reg(_)) {
                    add(
                        pc,
                        t!(
                            "MEASURE into a register: the bench gives results as bits",
                            "MEASURE erregistro batera: bankuak emaitzak bit gisa ematen ditu"
                        ),
                    );
                }
                measured_q.insert(*q);
            }
            Op::MeasureAll => {
                for q in 0..p.n_qubits {
                    if measured_q.contains(&q) {
                        add(
                            pc,
                            t!(
                                "q{} was measured already; a measured photon is gone",
                                "q{} jada neurtu da; neurtutako fotoi bat joan egin da",
                                q
                            ),
                        );
                    }
                    measured_q.insert(q);
                }
            }
            Op::Controlled { .. }
            | Op::PhaseAll { .. }
            | Op::Swap { .. }
            | Op::Ghz { .. }
            | Op::Cluster { .. }
            | Op::EntangleSwap { .. } => add(
                pc,
                t!(
                    "{}: two-qubit gates on photons work only part of the time (post-selected), so the bench cannot run them as a gate",
                    "{}: bi qubiteko ateak fotoiekin batzuetan bakarrik dabiltza (post-hautatuta), beraz bankuak ezin ditu ate gisa exekutatu",
                    name
                ),
            ),
            Op::Reset { .. } | Op::ResetAll => add(
                pc,
                t!(
                    "{}: a photon cannot be put back; every shot starts with fresh photons from the source",
                    "{}: fotoi bat ezin da itzuli; shot bakoitza iturriko fotoi berriekin hasten da",
                    name
                ),
            ),
            _ => {}
        }
    }
    let detectors = detected_o.len() + measured_q.len();
    if detectors > target.detectors() {
        whole.push(Problem {
            line: 0,
            message: t!(
                "the program reads {} detectors; {} has {}",
                "programak {} detektagailu irakurtzen ditu; {}-k {} ditu",
                detectors,
                bench,
                target.detectors()
            ),
        });
    }
    whole.extend(out);
    whole
}
