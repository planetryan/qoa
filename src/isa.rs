// fitxategi honek qoa-ren agindu-multzoa (isa) definitzen du.
//
// taula hau da egia-iturri bakarra: mihiztatzaileak (asm.rs) hemendik ezagutzen ditu izenak,
// ezizenak eta eragigaiak, eta `qoa isa` komandoak eta docs/ISA.md fitxategiak hemendik sortzen
// dute dokumentazioa. horrela dokumentazioa eta kodea ezin dira bata bestetik aldendu.
// agindu bakoitzaren azalpena bi hizkuntzatan dago (ingelesa, euskara).
//
// eragigaien idazkera (`operands` eremua): hutsunez bereizitako elementuak, `mota:izena`.
// motak:
//   q qubita (`q3` edo `3`)            o modu optikoa (`o1` edo `1`)
//   r erregistroa (`R2` edo `2`)       c bit klasikoa (`c0`, edo tartea `c0..c3`)
//   d emaitzaren helmuga (`c0` bita edo `R2` erregistroa)
//   v balio-iturria (`c0` edo `R2`)    f zenbakizko adierazpena (`pi/2`)
//   i zenbaki osoa                     n kopurua (zenbaki osoa edo `R2`)
//   l etiketa                          s testu-katea ("...")
//   b oinarria/ardatza (X, Y, Z)       g ate-izena (h, x, ...)
//   k karakterea ("A" edo 65)
// atzizkiak: `?` aukerakoa, `+` bat edo gehiago, `*` zero edo gehiago.

use crate::lang::is_eu;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Group {
    Setup,
    Gate,
    Entangle,
    Measure,
    Optical,
    Hybrid,
    Inspect,
    Classical,
    Flow,
    Output,
}

impl Group {
    pub const ALL: [Group; 10] = [
        Group::Setup,
        Group::Gate,
        Group::Entangle,
        Group::Measure,
        Group::Optical,
        Group::Hybrid,
        Group::Inspect,
        Group::Classical,
        Group::Flow,
        Group::Output,
    ];

    pub fn title(&self) -> &'static str {
        let (en, eu) = match self {
            Group::Setup => ("Setup", "Prestaketa"),
            Group::Gate => ("Qubit gates", "Qubit-ateak"),
            Group::Entangle => ("Entangling shorthands", "Entrelazamendu-laburdurak"),
            Group::Measure => (
                "Qubit measurement and reset",
                "Qubit-neurketa eta berrezarpena",
            ),
            Group::Optical => ("Optical modes", "Modu optikoak"),
            Group::Hybrid => ("Qubit-mode coupling", "Qubit-modu akoplamendua"),
            Group::Inspect => (
                "Simulator-only inspection",
                "Ikuskapena (simulagailuan bakarrik)",
            ),
            Group::Classical => ("Classical registers", "Erregistro klasikoak"),
            Group::Flow => ("Control flow", "Kontrol-fluxua"),
            Group::Output => ("Output", "Irteera"),
        };
        if is_eu() { eu } else { en }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    // erabat inplementatuta
    Ready,
    // simulagailuan bakarrik egin daitekeena (benetako gailu batek ezin du)
    SimOnly,
    // kendua: zergatik (ingelesez, euskaraz)
    Removed(&'static str, &'static str),
}

#[derive(Debug)]
pub struct Spec {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub operands: &'static str,
    pub group: Group,
    pub status: Status,
    // lehen eragigaia tarte bat izan daiteke (`H q0..q3`), eta agindua qubit bakoitzeko errepikatzen da
    pub broadcast: bool,
    pub summary_en: &'static str,
    pub summary_eu: &'static str,
}

const fn spec(
    name: &'static str,
    aliases: &'static [&'static str],
    operands: &'static str,
    group: Group,
    summary_en: &'static str,
    summary_eu: &'static str,
) -> Spec {
    Spec {
        name,
        aliases,
        operands,
        group,
        status: Status::Ready,
        broadcast: false,
        summary_en,
        summary_eu,
    }
}

const fn each(s: Spec) -> Spec {
    Spec {
        broadcast: true,
        ..s
    }
}

const fn sim_only(s: Spec) -> Spec {
    Spec {
        status: Status::SimOnly,
        ..s
    }
}

const fn removed(
    name: &'static str,
    aliases: &'static [&'static str],
    why_en: &'static str,
    why_eu: &'static str,
) -> Spec {
    Spec {
        name,
        aliases,
        operands: "",
        group: Group::Setup,
        status: Status::Removed(why_en, why_eu),
        broadcast: false,
        summary_en: "",
        summary_eu: "",
    }
}

use Group::*;

pub static SPECS: &[Spec] = &[
    // ---- prestaketa ----
    spec(
        "QINIT",
        &["QI", "INITQUBIT", "IQ", "QINITQ"],
        "i:qubits",
        Setup,
        "Declare how many qubits the program uses (once). Without it, the count is the highest qubit index used plus one.",
        "Programak zenbat qubit erabiltzen dituen adierazi (behin). Gabe, erabilitako qubit-indize handiena gehi bat da.",
    ),
    spec(
        "QMODE",
        &["MODES"],
        "i:modes i:cutoff?",
        Setup,
        "Declare optical modes, each holding 0..cutoff-1 photons (default cutoff 8). Probability pushed past the cutoff is reported after the run.",
        "Modu optikoak adierazi, bakoitzak 0..muga-1 fotoi gordetzen dituela (muga lehenetsia 8). Mugaz kanpora doan probabilitatea exekuzioaren ondoren jakinarazten da.",
    ),
    // ---- qubit-ateak ----
    each(spec(
        "H",
        &["HAD", "APPLYHADAMARD"],
        "q:qubit",
        Gate,
        "Hadamard.",
        "Hadamard.",
    )),
    each(spec(
        "X",
        &["APPLYBITFLIP"],
        "q:qubit",
        Gate,
        "Pauli-X (bit flip).",
        "Pauli-X (bit-iraulketa).",
    )),
    each(spec("Y", &[], "q:qubit", Gate, "Pauli-Y.", "Pauli-Y.")),
    each(spec(
        "Z",
        &["APPLYPHASEFLIP"],
        "q:qubit",
        Gate,
        "Pauli-Z (phase flip).",
        "Pauli-Z (fase-iraulketa).",
    )),
    each(spec(
        "S",
        &["APPLYSGATE"],
        "q:qubit",
        Gate,
        "S = diag(1, i).",
        "S = diag(1, i).",
    )),
    each(spec(
        "SDG",
        &["SDAG"],
        "q:qubit",
        Gate,
        "S† = diag(1, -i).",
        "S† = diag(1, -i).",
    )),
    each(spec(
        "T",
        &["APPLYTGATE"],
        "q:qubit",
        Gate,
        "T = diag(1, e^(i·pi/4)).",
        "T = diag(1, e^(i·pi/4)).",
    )),
    each(spec(
        "TDG",
        &["TDAG"],
        "q:qubit",
        Gate,
        "T† = diag(1, e^(-i·pi/4)).",
        "T† = diag(1, e^(-i·pi/4)).",
    )),
    each(spec(
        "SX",
        &["SQRTX"],
        "q:qubit",
        Gate,
        "Square root of X.",
        "X-ren erro karratua.",
    )),
    each(spec(
        "SXDG",
        &["SXDAG"],
        "q:qubit",
        Gate,
        "Inverse of SX.",
        "SX-ren alderantzizkoa.",
    )),
    each(spec(
        "PHASE",
        &["P", "PHASESHIFT", "SETPHASE", "SETP", "SPH", "PSE", "U1"],
        "q:qubit f:angle",
        Gate,
        "Phase gate diag(1, e^(i·angle)).",
        "Fase-atea diag(1, e^(i·angelua)).",
    )),
    each(spec(
        "RX",
        &[],
        "q:qubit f:angle",
        Gate,
        "Rotation about X: exp(-i·angle·X/2).",
        "Biraketa X ardatzean: exp(-i·angelua·X/2).",
    )),
    each(spec(
        "RY",
        &[],
        "q:qubit f:angle",
        Gate,
        "Rotation about Y: exp(-i·angle·Y/2).",
        "Biraketa Y ardatzean: exp(-i·angelua·Y/2).",
    )),
    each(spec(
        "RZ",
        &[],
        "q:qubit f:angle",
        Gate,
        "Rotation about Z: exp(-i·angle·Z/2).",
        "Biraketa Z ardatzean: exp(-i·angelua·Z/2).",
    )),
    each(spec(
        "U",
        &["U3"],
        "q:qubit f:theta f:phi f:lambda",
        Gate,
        "General single-qubit gate U(theta, phi, lambda), OpenQASM convention.",
        "Qubit bakarreko ate orokorra U(theta, phi, lambda), OpenQASM-en konbentzioa.",
    )),
    each(spec(
        "ROT",
        &["APPLYROTATION", "AR"],
        "q:qubit b:axis f:angle",
        Gate,
        "Rotation about axis X, Y or Z (the same as RX, RY or RZ).",
        "Biraketa X, Y edo Z ardatzean (RX, RY edo RZ bera).",
    )),
    spec(
        "MROT",
        &["APPLYMULTIQUBITROTATION", "AMQR"],
        "b:axis f:angle q:qubits+",
        Gate,
        "The same rotation on every listed qubit.",
        "Biraketa bera zerrendako qubit guztietan.",
    ),
    spec(
        "GATE",
        &["APPLYGATE", "AGATE", "AG"],
        "g:name q:qubit",
        Gate,
        "Apply a named single-qubit gate: h, x, y, z, s, sdg, t, tdg, sx, sxdg.",
        "Izendun qubit bakarreko atea aplikatu: h, x, y, z, s, sdg, t, tdg, sx, sxdg.",
    ),
    spec(
        "CNOT",
        &["CX", "CONTROLLEDNOT", "CN", "ENTANGLE"],
        "q:control q:target",
        Gate,
        "Controlled-X: flip the target when the control is 1.",
        "X kontrolatua: helburua irauli kontrola 1 denean.",
    ),
    spec(
        "CY",
        &[],
        "q:control q:target",
        Gate,
        "Controlled-Y.",
        "Y kontrolatua.",
    ),
    spec(
        "CZ",
        &[],
        "q:control q:target",
        Gate,
        "Controlled-Z: phase -1 on |11>.",
        "Z kontrolatua: -1 fasea |11>-n.",
    ),
    spec(
        "CH",
        &[],
        "q:control q:target",
        Gate,
        "Controlled-Hadamard.",
        "Hadamard kontrolatua.",
    ),
    spec(
        "CSX",
        &[],
        "q:control q:target",
        Gate,
        "Controlled square root of X.",
        "X-ren erro karratu kontrolatua.",
    ),
    spec(
        "CU",
        &["CU3"],
        "q:control q:target f:theta f:phi f:lambda",
        Gate,
        "Controlled U(theta, phi, lambda).",
        "U(theta, phi, lambda) kontrolatua.",
    ),
    spec(
        "CPHASE",
        &["CP", "CONTROLLEDPHASEROTATION", "APPLYCPHASE", "CPR", "CU1"],
        "q:control q:target f:angle",
        Gate,
        "Controlled phase: e^(i·angle) on |11>.",
        "Fase kontrolatua: e^(i·angelua) |11>-n.",
    ),
    spec(
        "CRX",
        &[],
        "q:control q:target f:angle",
        Gate,
        "Controlled RX.",
        "RX kontrolatua.",
    ),
    spec(
        "CRY",
        &[],
        "q:control q:target f:angle",
        Gate,
        "Controlled RY.",
        "RY kontrolatua.",
    ),
    spec(
        "CRZ",
        &[],
        "q:control q:target f:angle",
        Gate,
        "Controlled RZ.",
        "RZ kontrolatua.",
    ),
    spec(
        "SWAP",
        &[],
        "q:a q:b",
        Gate,
        "Swap two qubits.",
        "Bi qubit trukatu.",
    ),
    spec(
        "CSWAP",
        &["CONTROLLEDSWAP", "CS", "FREDKIN"],
        "q:control q:a q:b",
        Gate,
        "Swap a and b when the control is 1 (Fredkin).",
        "a eta b trukatu kontrola 1 denean (Fredkin).",
    ),
    spec(
        "CCX",
        &["TOFFOLI", "CCNOT"],
        "q:control1 q:control2 q:target",
        Gate,
        "Toffoli: flip the target when both controls are 1.",
        "Toffoli: helburua irauli bi kontrolak 1 direnean.",
    ),
    spec(
        "MCX",
        &[],
        "q:controls+ q:target",
        Gate,
        "Flip the target (the last qubit) when every other listed qubit is 1.",
        "Helburua (azken qubita) irauli zerrendako beste qubit guztiak 1 direnean.",
    ),
    spec(
        "MCZ",
        &[],
        "q:qubits+",
        Gate,
        "Phase -1 on the basis states where every listed qubit is 1.",
        "-1 fasea zerrendako qubit guztiak 1 diren oinarri-egoeretan.",
    ),
    // ---- entrelazamendu-laburdurak ----
    spec(
        "BELL",
        &["ENTANGLEBELL", "EBELL", "EB"],
        "q:a q:b",
        Entangle,
        "Runs H a, then CNOT a b. From |00> this gives (|00> + |11>)/sqrt(2).",
        "H a exekutatzen du, gero CNOT a b. |00>-tik (|00> + |11>)/sqrt(2) ematen du.",
    ),
    spec(
        "GHZ",
        &["ENTANGLEMULTI", "EMULTI", "EM"],
        "q:qubits+",
        Entangle,
        "Runs H on the first qubit, then CNOT from it to each of the others. From |0..0> this gives a GHZ state.",
        "H lehen qubitean, gero CNOT hartatik beste guztietara. |0..0>-tik GHZ egoera bat ematen du.",
    ),
    spec(
        "CLUSTER",
        &["ENTANGLECLUSTER", "ECLUSTER", "ECR"],
        "q:qubits+",
        Entangle,
        "Linear cluster state: H on every qubit, then CZ between each neighbouring pair.",
        "Cluster egoera lineala: H qubit guztietan, gero CZ ondoko bikote bakoitzean.",
    ),
    spec(
        "ESWAP",
        &["ENTANGLESWAP", "ESP"],
        "q:a q:b q:c q:d",
        Entangle,
        "Entanglement swapping. With Bell pairs (a,b) and (c,d): Bell-measures b and c (into c<b> and c<c>), then corrects d, leaving (a,d) as a Bell pair.",
        "Entrelazamendu-trukaketa. (a,b) eta (c,d) bell bikoteekin: b eta c bell-neurtzen ditu (c<b> eta c<c>-ra), gero d zuzentzen du, eta (a,d) bell bikote bihurtzen da.",
    ),
    // ---- neurketa eta berrezarpena ----
    each(spec(
        "MEASURE",
        &[
            "MEAS",
            "M",
            "QMEAS",
            "QM",
            "APPLYQNDMEASUREMENT",
            "AQND",
            "AQAD",
            "AQNM",
        ],
        "q:qubit d:to?",
        Measure,
        "Measure in the Z basis. The result goes to bit c<qubit> unless routed with -> cK or -> RK.",
        "Z oinarrian neurtu. Emaitza c<qubit> bitera doa, -> cK edo -> RK bidez beste nonbait bideratzen ez bada.",
    )),
    spec(
        "MEASUREIN",
        &["MEASUREINBASIS", "MEASB", "MIB"],
        "q:qubit b:basis d:to?",
        Measure,
        "Measure in the X, Y or Z basis (rotates into Z, measures, rotates back).",
        "X, Y edo Z oinarrian neurtu (Z-ra biratu, neurtu, atzera biratu).",
    ),
    spec(
        "MEASUREALL",
        &["MALL"],
        "",
        Measure,
        "Measure every qubit, each into its own bit c<qubit>.",
        "Qubit guztiak neurtu, bakoitza bere c<qubit> bitera.",
    ),
    each(spec(
        "RESET",
        &["RST", "RSTQ", "QRESET", "QR"],
        "q:qubit",
        Measure,
        "Put a qubit in |0>: measure it, then flip it if it read 1.",
        "Qubit bat |0>-n jarri: neurtu, eta 1 irakurri badu irauli.",
    )),
    spec(
        "RESETALL",
        &["RSTALL", "RSA"],
        "",
        Measure,
        "Every qubit to |0> and every mode to vacuum.",
        "Qubit guztiak |0>-ra eta modu guztiak hutsera.",
    ),
    each(spec(
        "TOBASIS",
        &["APPLYMEASUREMENTBASISCHANGE", "AMBC"],
        "q:qubit b:basis",
        Measure,
        "Rotate so that a Z measurement reads the given basis (X: H; Y: SDG then H; Z: nothing).",
        "Biratu Z neurketa batek oinarri hori irakur dezan (X: H; Y: SDG eta gero H; Z: ezer ez).",
    )),
    // ---- modu optikoak ----
    spec(
        "DISPLACE",
        &["APPLYDISPLACEMENT", "ADISP", "AD"],
        "o:mode f:re f:im?",
        Optical,
        "Displacement D(alpha), alpha = re + i·im. Turns vacuum into a coherent state with mean photon number |alpha|^2.",
        "Desplazamendua D(alpha), alpha = re + i·im. Hutsa egoera koherente bihurtzen du, |alpha|^2 batez besteko fotoiekin.",
    ),
    spec(
        "DISPLACEREG",
        &["APPLYDISPLACEMENTFEEDBACK"],
        "o:mode r:alpha",
        Optical,
        "Displacement by the (real) value in a register: classical feed-forward.",
        "Erregistro bateko balioaren (erreala) desplazamendua: aurrera-elikadura klasikoa.",
    ),
    spec(
        "SQUEEZE",
        &["APPLYSQUEEZING", "ASQ", "AS"],
        "o:mode f:r f:phi?",
        Optical,
        "Squeezing S(r·e^(i·phi)). With phi = 0 and r > 0 the x quadrature is squeezed.",
        "Estutzea S(r·e^(i·phi)). phi = 0 eta r > 0 direnean x koadratura estutzen da.",
    ),
    spec(
        "SQUEEZEREG",
        &["APPLYSQUEEZINGFEEDBACK", "ASWF", "ASF", "ASFB"],
        "o:mode r:r",
        Optical,
        "Squeezing by the value in a register: classical feed-forward.",
        "Erregistro bateko balioaren estutzea: aurrera-elikadura klasikoa.",
    ),
    spec(
        "MODEPHASE",
        &["DYNAMICPHASECOMPENSATION", "DPC", "DPCMP", "PHASEROT"],
        "o:mode f:angle",
        Optical,
        "Phase rotation e^(i·angle·n) of a mode.",
        "Modu baten fase-biraketa e^(i·angelua·n).",
    ),
    spec(
        "KERR",
        &["APPLYKERRNONLINEARITY", "AKNL"],
        "o:mode f:kappa f:time?",
        Optical,
        "Self-Kerr nonlinearity e^(i·kappa·time·n^2) (time defaults to 1).",
        "Kerr ez-lineartasuna e^(i·kappa·denbora·n^2) (denbora lehenetsia 1).",
    ),
    spec(
        "CROSSKERR",
        &["CROSSPHASEMODULATION", "CPM", "CPMOD"],
        "o:a o:b f:kappa",
        Optical,
        "Cross-Kerr (cross-phase modulation) e^(i·kappa·n_a·n_b).",
        "Kerr gurutzatua (fase-modulazio gurutzatua) e^(i·kappa·n_a·n_b).",
    ),
    spec(
        "BS",
        &["BEAMSPLITTER"],
        "o:a o:b f:theta f:phi?",
        Optical,
        "Beamsplitter exp(theta·(e^(i·phi)·a·b† - e^(-i·phi)·a†·b)). theta = pi/4 is 50:50.",
        "Izpi-zatitzailea exp(theta·(e^(i·phi)·a·b† - e^(-i·phi)·a†·b)). theta = pi/4 50:50 da.",
    ),
    spec(
        "MODESWAP",
        &["OPTICALROUTING", "OROUTE", "OPTR"],
        "o:a o:b",
        Optical,
        "Exchange the contents of two modes (an ideal router).",
        "Bi moduren edukiak trukatu (bideratzaile ideal bat).",
    ),
    spec(
        "ADDPHOTON",
        &["PHOTONADDITION", "PADD", "PA"],
        "o:mode",
        Optical,
        "Apply a† and renormalize. Post-selected: real photon addition only succeeds some of the time.",
        "a† aplikatu eta normalizatu. Post-hautatua: benetako fotoi-gehiketak batzuetan bakarrik lortzen du.",
    ),
    spec(
        "SUBPHOTON",
        &["APPLYPHOTONSUBTRACTION", "APSUB", "APS"],
        "o:mode",
        Optical,
        "Apply a and renormalize (post-selected). Fails if the mode is certainly empty.",
        "a aplikatu eta normalizatu (post-hautatua). Huts egiten du modua ziur hutsik badago.",
    ),
    spec(
        "LOSS",
        &[
            "PHOTONLOSS",
            "PHOTONLOSSSIMULATE",
            "PLS",
            "PLSIM",
            "SETOPTICALATTENUATION",
            "SOATT",
            "SOA",
        ],
        "o:mode f:probability",
        Optical,
        "Each photon in the mode is lost with this probability. One random outcome per run; averaged over shots it is the exact loss channel.",
        "Moduko fotoi bakoitza probabilitate honekin galtzen da. Ausazko emaitza bat exekuzio bakoitzeko; shot-en batez bestekoa galera-kanal zehatza da.",
    ),
    spec(
        "FOCK",
        &[],
        "o:mode i:photons",
        Optical,
        "Discard what the mode held and prepare exactly that many photons.",
        "Moduak zuena baztertu eta zehazki fotoi-kopuru hori prestatu.",
    ),
    spec(
        "SINGLEPHOTON",
        &["SINGLEPHOTONSOURCEON", "SPSON"],
        "o:mode",
        Optical,
        "An ideal single-photon source: the same as FOCK mode 1.",
        "Fotoi bakarreko iturri ideala: FOCK modua 1 bera.",
    ),
    spec(
        "COUNT",
        &[
            "PHOTONCOUNT",
            "PCOUNT",
            "PC",
            "PHOTONNUMBERRESOLVINGDETECTION",
            "PNRD",
            "PNR",
        ],
        "o:mode d:to?",
        Optical,
        "Number-resolving photon detection. The count goes to the mode's readout unless routed with -> RK.",
        "Fotoi-kopurua bereizten duen detekzioa. Kopurua moduaren irakurketara doa, -> RK bidez bideratzen ez bada.",
    ),
    spec(
        "DETECT",
        &["PHOTONDETECT", "PDETECT", "PD"],
        "o:mode d:to?",
        Optical,
        "Threshold (click) detection: 1 if any photon is present, else 0.",
        "Atalase-detekzioa (klik): 1 fotoiren bat badago, bestela 0.",
    ),
    spec(
        "PARITY",
        &["MEASUREPARITY", "MPAR", "MP"],
        "o:mode d:to?",
        Optical,
        "Photon-number parity measurement: 0 for even, 1 for odd.",
        "Fotoi-kopuruaren paritatearen neurketa: 0 bikoitia, 1 bakoitia.",
    ),
    spec(
        "COINCIDENCE",
        &["PHOTONDETECTCOINCIDENCE", "PDCOIN", "PDC"],
        "o:modes+ d:to",
        Optical,
        "Click-detect every listed mode; writes 1 if all of them clicked.",
        "Zerrendako modu guztiak klik-detektatu; 1 idazten du guztiek klik egin badute.",
    ),
    // ---- qubit-modu akoplamendua ----
    spec(
        "EMIT",
        &["PHOTONEMIT", "PEMIT", "PE", "ABSORB"],
        "q:qubit o:mode",
        Hybrid,
        "Exchange |1>|0> and |0>|1> between a qubit and a mode: an excited emitter puts its excitation into the mode as a photon. Running it again absorbs the photon back.",
        "|1>|0> eta |0>|1> trukatu qubit baten eta modu baten artean: igorle kitzikatu batek bere kitzikapena fotoi gisa jartzen du moduan. Berriz exekutatuta fotoia xurgatzen du.",
    ),
    spec(
        "DISPERSIVE",
        &[],
        "q:qubit o:mode f:angle",
        Hybrid,
        "Dispersive coupling e^(i·angle·n) applied only where the qubit is 1.",
        "Akoplamendu dispertsiboa e^(i·angelua·n), qubita 1 den lekuetan bakarrik.",
    ),
    // ---- simulagailuko ikuskapena ----
    sim_only(spec(
        "DUMPSTATE",
        &["DSTATE", "DS"],
        "",
        Inspect,
        "Print the exact state at this point.",
        "Puntu honetako egoera zehatza inprimatu.",
    )),
    sim_only(spec(
        "EXPECT",
        &["QUANTUMSTATETOMOGRAPHY", "QST", "QSTAT"],
        "q:qubit b:basis r:to?",
        Inspect,
        "Exact <X>, <Y> or <Z> of a qubit, without disturbing it. Printed, or stored in a register.",
        "Qubit baten <X>, <Y> edo <Z> zehatza, hura aldatu gabe. Inprimatuta, edo erregistro batean gordeta.",
    )),
    sim_only(spec(
        "BELLCHECK",
        &["BELLSTATEVERIFICATION", "BSV", "BSTATE"],
        "q:a q:b r:to?",
        Inspect,
        "Exact fidelity of qubits (a, b) with (|00> + |11>)/sqrt(2).",
        "(a, b) qubiten fideltasun zehatza (|00> + |11>)/sqrt(2)-rekin.",
    )),
    sim_only(spec(
        "MEANPHOTON",
        &["NMEAN"],
        "o:mode r:to?",
        Inspect,
        "Exact mean photon number <n> of a mode.",
        "Modu baten batez besteko fotoi-kopuru zehatza <n>.",
    )),
    // ---- erregistro klasikoak ----
    spec(
        "REGSET",
        &["SET", "RSET", "RGST"],
        "r:reg f:value",
        Classical,
        "Set a register to a value.",
        "Erregistro bati balio bat eman.",
    ),
    spec(
        "REGCOPY",
        &["MOV", "RCOPY", "RC"],
        "r:dst r:src",
        Classical,
        "Copy one register into another.",
        "Erregistro bat beste batera kopiatu.",
    ),
    spec(
        "ADD",
        &["REGADD", "RADD", "RGA"],
        "r:dst r:a r:b",
        Classical,
        "dst = a + b",
        "dst = a + b",
    ),
    spec(
        "SUB",
        &["REGSUB", "RSUB", "RGS"],
        "r:dst r:a r:b",
        Classical,
        "dst = a - b",
        "dst = a - b",
    ),
    spec(
        "MUL",
        &["REGMUL"],
        "r:dst r:a r:b",
        Classical,
        "dst = a * b",
        "dst = a * b",
    ),
    spec(
        "DIV",
        &["REGDIV"],
        "r:dst r:a r:b",
        Classical,
        "dst = a / b (division by zero stops the program)",
        "dst = a / b (zeroz zatitzeak programa gelditzen du)",
    ),
    spec(
        "ANDBITS",
        &["AND", "ANDB", "AB"],
        "r:dst r:a r:b",
        Classical,
        "Bitwise AND of the values as 64-bit integers.",
        "Balioen bitez biteko AND, 64 biteko oso gisa.",
    ),
    spec(
        "ORBITS",
        &["OR", "ORB", "OB"],
        "r:dst r:a r:b",
        Classical,
        "Bitwise OR of the values as 64-bit integers.",
        "Balioen bitez biteko OR, 64 biteko oso gisa.",
    ),
    spec(
        "XORBITS",
        &["XOR", "XORB", "XB"],
        "r:dst r:a r:b",
        Classical,
        "Bitwise XOR of the values as 64-bit integers.",
        "Balioen bitez biteko XOR, 64 biteko oso gisa.",
    ),
    spec(
        "NOTBITS",
        &["NOTB", "NB"],
        "r:dst r:src",
        Classical,
        "Bitwise NOT of the value as a 64-bit integer.",
        "Balioaren bitez biteko NOT, 64 biteko oso gisa.",
    ),
    spec(
        "SHL",
        &[],
        "r:dst r:src r:amount",
        Classical,
        "Shift left by the amount in a register.",
        "Ezkerrera desplazatu erregistro bateko kopuruan.",
    ),
    spec(
        "SHR",
        &[],
        "r:dst r:src r:amount",
        Classical,
        "Shift right by the amount in a register.",
        "Eskuinera desplazatu erregistro bateko kopuruan.",
    ),
    spec(
        "SQRT",
        &["SR"],
        "r:dst r:src",
        Classical,
        "Square root.",
        "Erro karratua.",
    ),
    spec(
        "EXP",
        &[],
        "r:dst r:src",
        Classical,
        "e to the power of the value.",
        "e balioaren berretzailera.",
    ),
    spec(
        "LOG",
        &["LN"],
        "r:dst r:src",
        Classical,
        "Natural logarithm.",
        "Logaritmo nepertarra.",
    ),
    spec(
        "RAND",
        &["RN"],
        "r:reg",
        Classical,
        "A uniform random number in [0, 1).",
        "Ausazko zenbaki uniforme bat [0, 1) tartean.",
    ),
    spec(
        "SEEDRNG",
        &["SRNG"],
        "i:seed",
        Classical,
        "Reseed the random number generator. Measurements use it too, so every shot then repeats the same outcomes.",
        "Ausazko zenbaki-sorgailuari hazi berria eman. Neurketek ere hura erabiltzen dute, beraz shot guztiek emaitza berak errepikatuko dituzte.",
    ),
    spec(
        "INPUT",
        &["INP"],
        "r:reg",
        Classical,
        "Read a number from standard input.",
        "Zenbaki bat irakurri sarrera estandarretik.",
    ),
    spec(
        "GETTIME",
        &["GTIME", "GT"],
        "r:reg",
        Classical,
        "Seconds since the Unix epoch.",
        "Unix garaitik igarotako segundoak.",
    ),
    spec(
        "PUSH",
        &["PUSHREG"],
        "r:reg",
        Classical,
        "Push a register onto the stack.",
        "Erregistro bat pilan sartu.",
    ),
    spec(
        "POP",
        &["POPREG"],
        "r:reg",
        Classical,
        "Pop the stack into a register.",
        "Pilatik atera erregistro batera.",
    ),
    spec(
        "CHARLOAD",
        &["CLOAD", "CLD"],
        "r:reg k:char",
        Classical,
        "Load a character code into a register (\"A\" or 65).",
        "Karaktere-kode bat erregistro batean kargatu (\"A\" edo 65).",
    ),
    spec(
        "PACKBITS",
        &[],
        "r:dst c:bits+",
        Classical,
        "Read classical bits as a whole number into a register. The first bit listed is the lowest; bits never written count as 0 (as in OpenQASM).",
        "Bit klasikoak zenbaki oso gisa irakurri erregistro batera. Zerrendako lehen bita baxuena da; inoiz idatzi gabeko bitak 0 dira (OpenQASM-en bezala).",
    ),
    // ---- kontrol-fluxua ----
    spec(
        "JUMP",
        &["JMP"],
        "l:label",
        Flow,
        "Continue at a label.",
        "Etiketa batean jarraitu.",
    ),
    spec(
        "JUMPIFZERO",
        &["JIZ", "JZ"],
        "v:value l:label",
        Flow,
        "Jump if the bit or register is 0.",
        "Jauzi bita edo erregistroa 0 bada.",
    ),
    spec(
        "JUMPIFONE",
        &["JIO", "JO"],
        "v:value l:label",
        Flow,
        "Jump if the bit or register is 1.",
        "Jauzi bita edo erregistroa 1 bada.",
    ),
    spec(
        "IFEQ",
        &["IEQ"],
        "r:a r:b l:label",
        Flow,
        "Jump if a == b.",
        "Jauzi a == b bada.",
    ),
    spec(
        "IFNE",
        &["INE"],
        "r:a r:b l:label",
        Flow,
        "Jump if a != b.",
        "Jauzi a != b bada.",
    ),
    spec(
        "IFGT",
        &["IGT"],
        "r:a r:b l:label",
        Flow,
        "Jump if a > b.",
        "Jauzi a > b bada.",
    ),
    spec(
        "IFLT",
        &["ILT"],
        "r:a r:b l:label",
        Flow,
        "Jump if a < b.",
        "Jauzi a < b bada.",
    ),
    spec(
        "CALL",
        &["CALLLABEL"],
        "l:label",
        Flow,
        "Call a subroutine at a label.",
        "Etiketa bateko azpierrutina deitu.",
    ),
    spec(
        "RETSUB",
        &["RET", "RS"],
        "",
        Flow,
        "Return from a subroutine.",
        "Azpierrutinatik itzuli.",
    ),
    spec(
        "LOOPSTART",
        &["LOOP", "LSTART", "LS"],
        "n:count",
        Flow,
        "Run the block up to the matching LOOPEND this many times (a number or a register).",
        "Dagokion LOOPEND-erainoko blokea hainbat aldiz exekutatu (zenbaki bat edo erregistro bat).",
    ),
    spec(
        "LOOPEND",
        &["ENDLOOP", "LEND", "LE"],
        "",
        Flow,
        "End of a LOOPSTART block.",
        "LOOPSTART bloke baten amaiera.",
    ),
    spec(
        "HALT",
        &["HLT", "END"],
        "",
        Flow,
        "Stop the program.",
        "Programa gelditu.",
    ),
    spec(
        "BARRIER",
        &["BR"],
        "q:qubits*",
        Flow,
        "Does nothing in simulation. Marks a layer boundary and is kept for OpenQASM round trips.",
        "Simulazioan ez du ezer egiten. Geruza-muga bat markatzen du eta OpenQASM-erako joan-etorrietan gordetzen da.",
    ),
    // ---- irteera ----
    spec(
        "PRINT",
        &[],
        "s:text",
        Output,
        "Print text.",
        "Testua inprimatu.",
    ),
    spec(
        "PRINTLN",
        &["PLN"],
        "s:text?",
        Output,
        "Print text and a newline.",
        "Testua eta lerro-jauzi bat inprimatu.",
    ),
    spec(
        "PRINTF",
        &["PF"],
        "s:format r:regs*",
        Output,
        "Print text with each {} replaced by the next register's value.",
        "Testua inprimatu, {} bakoitza hurrengo erregistroaren balioarekin ordezkatuta.",
    ),
    spec(
        "CHAROUT",
        &["COUT", "CO"],
        "r:reg",
        Output,
        "Print a register's value as a character.",
        "Erregistro baten balioa karaktere gisa inprimatu.",
    ),
    spec(
        "VERBOSELOG",
        &["VLOG", "VL"],
        "q:qubit s:text",
        Output,
        "Print a message tagged with a qubit.",
        "Qubit batekin etiketatutako mezu bat inprimatu.",
    ),
    spec(
        "DUMPREGS",
        &["DREGS", "DR"],
        "",
        Output,
        "Print every register.",
        "Erregistro guztiak inprimatu.",
    ),
    spec(
        "COMMENT",
        &["CMT", "CM"],
        "",
        Output,
        "The rest of the line is a comment (like ;).",
        "Lerroaren gainerakoa iruzkina da (; bezala).",
    ),
    // ---- 0.4-n kenduak (bakoitzak zergatik eta zer erabili esaten du) ----
    removed(
        "SETPOS",
        &["SPOS", "STP"],
        "position bookkeeping that never affected the quantum state",
        "egoera kuantikoan inoiz eraginik izan ez zuen posizio-erregistroa",
    ),
    removed(
        "SETWL",
        &["SWL", "SW"],
        "wavelength bookkeeping that never affected the quantum state",
        "egoera kuantikoan inoiz eraginik izan ez zuen uhin-luzera erregistroa",
    ),
    removed(
        "WLSHIFT",
        &["WLS", "WLSH"],
        "wavelength bookkeeping that never affected the quantum state",
        "egoera kuantikoan inoiz eraginik izan ez zuen uhin-luzera erregistroa",
    ),
    removed(
        "MOVE",
        &["MV"],
        "position bookkeeping that never affected the quantum state",
        "egoera kuantikoan inoiz eraginik izan ez zuen posizio-erregistroa",
    ),
    removed(
        "MARKOBSERVED",
        &["MOBS", "MO"],
        "it had no effect; a measured qubit is simply measured",
        "ez zuen eraginik; neurtutako qubit bat neurtuta dago, besterik ez",
    ),
    removed(
        "RELEASE",
        &["REL", "RL"],
        "it had no effect; a simulator has nothing to release",
        "ez zuen eraginik; simulagailu batek ez du ezer askatzeko",
    ),
    removed(
        "DECOHERENCEPROTECT",
        &["DPROT", "DP"],
        "there is no physical model behind it; study decoherence with `qoa run --noise`",
        "ez dago eredu fisikorik atzean; dekoherentzia aztertzeko erabili `qoa run --noise`",
    ),
    removed(
        "ENTANGLESWAPMEASURE",
        &["ESWAPM", "ESM"],
        "use ESWAP, then branch on the measured bits with JUMPIFONE",
        "erabili ESWAP, eta gero adarkatu neurtutako bitekin JUMPIFONE bidez",
    ),
    removed(
        "ENTANGLEWITHCLASSICALFEEDBACK",
        &["EWCFB", "ECFB"],
        "use gates plus JUMPIFONE/IFEQ for classical feedback",
        "erabili ateak eta JUMPIFONE/IFEQ atzeraelikadura klasikorako",
    ),
    removed(
        "ENTANGLEDISTRIBUTED",
        &["EDIST", "ED"],
        "networked entanglement between machines is not modelled",
        "makinen arteko sareko entrelazamendua ez dago modelatuta",
    ),
    removed(
        "APPLYNONLINEARPHASESHIFT",
        &["ANLPS", "ANLP", "ANLPH"],
        "it was never defined; use KERR or CROSSKERR",
        "ez zen inoiz definitu; erabili KERR edo CROSSKERR",
    ),
    removed(
        "APPLYNONLINEARSIGMA",
        &["ANLS", "ANLSI"],
        "it was never defined; use KERR or CROSSKERR",
        "ez zen inoiz definitu; erabili KERR edo CROSSKERR",
    ),
    removed(
        "QUANTUMZENOEFFECT",
        &["QZE", "QZEN"],
        "it needs continuous-time evolution, which QOA does not model; repeated MEASURE in a LOOPSTART shows the freezing",
        "denbora jarraituko bilakaera behar du, eta QOA-k ez du modelatzen; LOOPSTART baten barruko MEASURE errepikatuek izozte bera erakusten dute",
    ),
    removed(
        "APPLYLINEAROPTICALTRANSFORM",
        &["ALOT", "ALOPT"],
        "build the network from BS and MODEPHASE; every linear optical unitary decomposes into them",
        "eraiki sarea BS eta MODEPHASE erabiliz; unitario optiko lineal oro haietan deskonposatzen da",
    ),
    removed(
        "ERRORCORRECT",
        &["ECORR", "EC"],
        "no error-correcting code was ever specified; write the correction circuit with gates",
        "ez zen inoiz errore-zuzentzeko koderik zehaztu; idatzi zuzenketa-zirkuitua ateekin",
    ),
    removed(
        "ERRORSYNDROME",
        &["ESYN", "ES"],
        "no code was ever specified; write the syndrome circuit with gates and MEASURE",
        "ez zen inoiz koderik zehaztu; idatzi sindrome-zirkuitua ateekin eta MEASURE-rekin",
    ),
    removed(
        "PHOTONROUTE",
        &["PROUTE", "PR"],
        "ports were never defined; use MODESWAP or BS",
        "atakak ez ziren inoiz definitu; erabili MODESWAP edo BS",
    ),
    removed(
        "TIMEDELAY",
        &["TDELAY", "TD"],
        "an ideal delay changes nothing; model a real one with LOSS and MODEPHASE",
        "atzerapen ideal batek ez du ezer aldatzen; benetakoa modelatzeko erabili LOSS eta MODEPHASE",
    ),
    removed(
        "OPTICALDELAYLINECONTROL",
        &["ODLC", "ODL"],
        "an ideal delay changes nothing; model a real one with LOSS and MODEPHASE",
        "atzerapen ideal batek ez du ezer aldatzen; benetakoa modelatzeko erabili LOSS eta MODEPHASE",
    ),
    removed(
        "WAIT",
        &[],
        "an ideal wait changes nothing; use `qoa run --noise` to study idle errors",
        "itxaronaldi ideal batek ez du ezer aldatzen; geldirik dauden erroreak aztertzeko erabili `qoa run --noise`",
    ),
    removed(
        "PHOTONBUNCHINGCONTROL",
        &["PBUNCH", "PBC"],
        "bunching comes from interference (see BS), it is not a switch",
        "multzokatzea interferentziatik dator (ikus BS), ez da etengailu bat",
    ),
    removed(
        "SINGLEPHOTONSOURCEOFF",
        &["SPSOFF"],
        "sources are instantaneous (SINGLEPHOTON); there is nothing to switch off",
        "iturriak berehalakoak dira (SINGLEPHOTON); ez dago ezer itzaltzeko",
    ),
    removed(
        "APPLYDISPLACEMENTOPERATOR",
        &["ADO", "ADOP"],
        "use DISPLACE",
        "erabili DISPLACE",
    ),
    removed(
        "OPTICALSWITCHCONTROL",
        &["OSC", "OSW"],
        "use MODESWAP to switch",
        "erabili MODESWAP aldatzeko",
    ),
    removed(
        "MEASUREWITHDELAY",
        &["MWD", "MWDEL"],
        "an ideal delay changes nothing; use MEASURE",
        "atzerapen ideal batek ez du ezer aldatzen; erabili MEASURE",
    ),
    removed(
        "PHOTONLOSSCORRECTION",
        &["PLC", "PLCOR"],
        "no loss-correcting code was ever specified",
        "ez zen inoiz galera zuzentzeko koderik zehaztu",
    ),
    removed(
        "PHOTONEMISSIONPATTERN",
        &["PEPAT", "PEP"],
        "it was never defined; use EMIT inside LOOPSTART",
        "ez zen inoiz definitu; erabili EMIT LOOPSTART baten barruan",
    ),
    removed(
        "APPLYFEEDFORWARDGATE",
        &["AFFG"],
        "use JUMPIFONE or IFEQ to apply gates conditionally",
        "erabili JUMPIFONE edo IFEQ ateak baldintzapean aplikatzeko",
    ),
    removed(
        "THERMALAVG",
        &[],
        "not a quantum-circuit operation",
        "ez da zirkuitu kuantikoko eragiketa bat",
    ),
    removed(
        "WKBFACTOR",
        &[],
        "not a quantum-circuit operation",
        "ez da zirkuitu kuantikoko eragiketa bat",
    ),
    removed(
        "CMP",
        &[],
        "IFEQ, IFNE, IFGT and IFLT compare registers directly",
        "IFEQ, IFNE, IFGT eta IFLT aginduek erregistroak zuzenean konparatzen dituzte",
    ),
    removed(
        "JMPABS",
        &["JMPA", "JABS"],
        "byte addresses no longer exist; jump to a label",
        "byte-helbideak jada ez daude; jauzi etiketa batera",
    ),
    removed(
        "CALLADDR",
        &["CADDR", "CA"],
        "byte addresses no longer exist; CALL a label",
        "byte-helbideak jada ez daude; erabili CALL etiketa batekin",
    ),
    removed(
        "LOAD",
        &[],
        "memory was never implemented; use the 32 registers and PUSH/POP",
        "memoria ez zen inoiz inplementatu; erabili 32 erregistroak eta PUSH/POP",
    ),
    removed(
        "STORE",
        &[],
        "memory was never implemented; use the 32 registers and PUSH/POP",
        "memoria ez zen inoiz inplementatu; erabili 32 erregistroak eta PUSH/POP",
    ),
    removed(
        "LOADMEM",
        &["LMEM", "LM"],
        "memory was never implemented; use the 32 registers and PUSH/POP",
        "memoria ez zen inoiz inplementatu; erabili 32 erregistroak eta PUSH/POP",
    ),
    removed(
        "STOREMEM",
        &["SMEM", "SM"],
        "memory was never implemented; use the 32 registers and PUSH/POP",
        "memoria ez zen inoiz inplementatu; erabili 32 erregistroak eta PUSH/POP",
    ),
    removed(
        "LOADCLASSICAL",
        &["LCL", "LC"],
        "memory was never implemented; use the 32 registers and PUSH/POP",
        "memoria ez zen inoiz inplementatu; erabili 32 erregistroak eta PUSH/POP",
    ),
    removed(
        "STORECLASSICAL",
        &["SCL", "SC"],
        "memory was never implemented; use the 32 registers and PUSH/POP",
        "memoria ez zen inoiz inplementatu; erabili 32 erregistroak eta PUSH/POP",
    ),
    removed(
        "ALLOC",
        &["ALC"],
        "memory was never implemented; use the 32 registers and PUSH/POP",
        "memoria ez zen inoiz inplementatu; erabili 32 erregistroak eta PUSH/POP",
    ),
    removed(
        "FREE",
        &["FRE"],
        "memory was never implemented; use the 32 registers and PUSH/POP",
        "memoria ez zen inoiz inplementatu; erabili 32 erregistroak eta PUSH/POP",
    ),
    removed(
        "BREAKPOINT",
        &["BP"],
        "use DUMPSTATE and DUMPREGS to look at a point in the program",
        "erabili DUMPSTATE eta DUMPREGS programaren puntu bat ikusteko",
    ),
    removed("EXITCODE", &["EXC", "EX"], "use HALT", "erabili HALT"),
    removed(
        "QUBITS",
        &["QUBIT"],
        "use QINIT with the qubit count, e.g. QINIT 25",
        "erabili QINIT qubit-kopuruarekin, adib. QINIT 25",
    ),
    removed(
        "QGATE",
        &[],
        "use GATE name qubit, or the gate's own instruction",
        "erabili GATE izena qubita, edo atearen agindu propioa",
    ),
];

// izen edo ezizen batetik (maiuskula/minuskula berdin) agindu-zehaztapenera
pub fn lookup(mnemonic: &str) -> Option<&'static Spec> {
    static INDEX: OnceLock<HashMap<String, usize>> = OnceLock::new();
    let index = INDEX.get_or_init(|| {
        let mut m = HashMap::new();
        for (i, s) in SPECS.iter().enumerate() {
            m.insert(s.name.to_string(), i);
            for a in s.aliases {
                m.insert(a.to_string(), i);
            }
        }
        m
    });
    index
        .get(&mnemonic.to_ascii_uppercase())
        .map(|&i| &SPECS[i])
}

// eragigai-zehaztapen bakarra, `operands` katetik aterata
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperandSpec {
    pub kind: char,
    pub name: &'static str,
    pub optional: bool,
    pub repeat: Repeat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repeat {
    One,
    OneOrMore,
    ZeroOrMore,
}

// eragigaien izenak euskaraz (erabilera-lerroetarako)
fn operand_name(name: &'static str) -> &'static str {
    if !is_eu() {
        return name;
    }
    match name {
        "qubit" => "qubita",
        "qubits" => "qubitak",
        "control" => "kontrola",
        "control1" => "kontrola1",
        "control2" => "kontrola2",
        "controls" => "kontrolak",
        "target" => "helburua",
        "angle" => "angelua",
        "axis" => "ardatza",
        "basis" => "oinarria",
        "name" => "izena",
        "mode" => "modua",
        "modes" => "moduak",
        "cutoff" => "muga",
        "photons" => "fotoiak",
        "probability" => "probabilitatea",
        "time" => "denbora",
        "to" => "helmuga",
        "reg" => "erregistroa",
        "regs" => "erregistroak",
        "value" => "balioa",
        "amount" => "kopurua",
        "seed" => "hazia",
        "char" => "karakterea",
        "bits" => "bitak",
        "label" => "etiketa",
        "count" => "kopurua",
        "text" => "testua",
        "format" => "formatua",
        other => other,
    }
}

impl Spec {
    pub fn summary(&self) -> &'static str {
        if is_eu() {
            self.summary_eu
        } else {
            self.summary_en
        }
    }

    // kendu izanaren arrazoia, hizkuntza egokian
    pub fn removed_reason(&self) -> Option<&'static str> {
        match self.status {
            Status::Removed(en, eu) => Some(if is_eu() { eu } else { en }),
            _ => None,
        }
    }

    pub fn operand_specs(&self) -> Vec<OperandSpec> {
        self.operands
            .split_whitespace()
            .map(|item| {
                let (kind, rest) = item
                    .split_once(':')
                    .expect("eragigai-zehaztapenak mota:izena forma du");
                let (name, optional, repeat) = if let Some(n) = rest.strip_suffix('?') {
                    (n, true, Repeat::One)
                } else if let Some(n) = rest.strip_suffix('+') {
                    (n, false, Repeat::OneOrMore)
                } else if let Some(n) = rest.strip_suffix('*') {
                    (n, true, Repeat::ZeroOrMore)
                } else {
                    (rest, false, Repeat::One)
                };
                OperandSpec {
                    kind: kind.chars().next().unwrap(),
                    name,
                    optional,
                    repeat,
                }
            })
            .collect()
    }

    // erabiltzaileari erakusteko sinadura, adib. `CNOT control target` edo `MEASURE qubit [-> to]`
    pub fn usage(&self) -> String {
        let mut s = self.name.to_string();
        for op in self.operand_specs() {
            let name = operand_name(op.name);
            let base = match op.kind {
                'd' => format!("-> {}", name),
                _ => name.to_string(),
            };
            let item = match op.repeat {
                Repeat::One => base,
                Repeat::OneOrMore | Repeat::ZeroOrMore => format!("{}...", base),
            };
            s.push(' ');
            if op.optional {
                s.push_str(&format!("[{}]", item));
            } else {
                s.push_str(&item);
            }
        }
        s
    }

    pub fn is_removed(&self) -> bool {
        matches!(self.status, Status::Removed(..))
    }
}

// eragigai-mota bakoitzaren azalpena (dokumentaziorako)
pub fn kind_help(kind: char) -> &'static str {
    let (en, eu) = match kind {
        'q' => (
            "qubit: q3 (or 3); ranges like q0..q3 where allowed",
            "qubita: q3 (edo 3); q0..q3 bezalako tarteak onartzen diren lekuetan",
        ),
        'o' => ("optical mode: o1 (or 1)", "modu optikoa: o1 (edo 1)"),
        'r' => (
            "register: R2 (or 2), R0..R31",
            "erregistroa: R2 (edo 2), R0..R31",
        ),
        'c' => (
            "classical bit: c0, or a range like c0..c3",
            "bit klasikoa: c0, edo c0..c3 bezalako tarte bat",
        ),
        'd' => (
            "result destination: c0 (classical bit) or R2 (register), optionally after ->",
            "emaitzaren helmuga: c0 (bit klasikoa) edo R2 (erregistroa), nahi bada -> ondoren",
        ),
        'v' => (
            "value: c0 (classical bit) or R2 (register)",
            "balioa: c0 (bit klasikoa) edo R2 (erregistroa)",
        ),
        'f' => (
            "number or expression: 0.5, pi/2, -3*pi/4, sqrt(2)/2",
            "zenbakia edo adierazpena: 0.5, pi/2, -3*pi/4, sqrt(2)/2",
        ),
        'i' => ("whole number", "zenbaki osoa"),
        'n' => (
            "count: whole number or register",
            "kopurua: zenbaki osoa edo erregistroa",
        ),
        'l' => ("label", "etiketa"),
        's' => ("text in double quotes", "testua komatxo bikoitzen artean"),
        'b' => (
            "basis or axis: X, Y or Z",
            "oinarria edo ardatza: X, Y edo Z",
        ),
        'g' => ("gate name", "ate-izena"),
        'k' => (
            "character: \"A\" or a character code like 65",
            "karakterea: \"A\" edo 65 bezalako karaktere-kode bat",
        ),
        _ => ("", ""),
    };
    if is_eu() { eu } else { en }
}

// docs/ISA.md (edo docs/ISA.eu.md) fitxategiaren edukia, taula honetatik sortua
pub fn markdown_reference() -> String {
    let eu = is_eu();
    let pick = |en: &'static str, e: &'static str| if eu { e } else { en };
    let mut out = String::new();
    out.push_str(pick(
        "# QOA instruction reference\n\n",
        "# QOA aginduen erreferentzia\n\n",
    ));
    out.push_str(pick(
        "Generated from `src/isa.rs` by `qoa isa --markdown`. Do not edit by hand.\n\n",
        "`src/isa.rs`-tik sortua `qoa isa --markdown` bidez. Ez editatu eskuz.\n\n",
    ));
    out.push_str(pick(
        "Mnemonics are case-insensitive. Every alias does exactly what its main name does.\n\n",
        "Mnemonikoetan berdin dio maiuskula edo minuskula. Ezizen bakoitzak bere izen nagusiak egiten duen gauza bera egiten du.\n\n",
    ));
    out.push_str(pick("## Operands\n\n", "## Eragigaiak\n\n"));
    out.push_str(pick(
        "| Written as | Meaning |\n|---|---|\n",
        "| Idazkera | Esanahia |\n|---|---|\n",
    ));
    for k in ['q', 'o', 'r', 'c', 'd', 'v', 'f', 'n', 'l', 's', 'b', 'k'] {
        let example = match k {
            'q' => "`q3`",
            'o' => "`o1`",
            'r' => "`R2`",
            'c' => "`c0` / `c0..c3`",
            'd' => "`-> c0` / `-> R2`",
            'v' => "`c0` / `R2`",
            'f' => "`pi/2`",
            'n' => "`8` / `R1`",
            'l' => "`done`",
            's' => "`\"text\"`",
            'b' => "`X`",
            'k' => "`\"A\"`",
            _ => "",
        };
        out.push_str(&format!("| {} | {} |\n", example, kind_help(k)));
    }
    out.push('\n');
    out.push_str(pick(
        "Instructions marked *simulator only* read information a real device cannot see without disturbing the state.\n\n",
        "*Simulagailuan bakarrik* markatutako aginduek benetako gailu batek egoera aldatu gabe ikusi ezin duen informazioa irakurtzen dute.\n\n",
    ));
    for group in Group::ALL {
        let specs: Vec<&Spec> = SPECS
            .iter()
            .filter(|s| s.group == group && !s.is_removed())
            .collect();
        if specs.is_empty() {
            continue;
        }
        out.push_str(&format!("## {}\n\n", group.title()));
        out.push_str(pick(
            "| Instruction | Does | Aliases |\n|---|---|---|\n",
            "| Agindua | Zer egiten du | Ezizenak |\n|---|---|---|\n",
        ));
        for s in specs {
            let mut does = s.summary().to_string();
            if s.status == Status::SimOnly {
                does = format!(
                    "{} {}",
                    pick("*Simulator only.*", "*Simulagailuan bakarrik.*"),
                    does
                );
            }
            if s.broadcast {
                does.push_str(pick(
                    " Accepts a qubit range (q0..q3).",
                    " Qubit-tarte bat onartzen du (q0..q3).",
                ));
            }
            out.push_str(&format!(
                "| `{}` | {} | {} |\n",
                s.usage(),
                does.replace('|', "\\|"),
                s.aliases.join(", ")
            ));
        }
        out.push('\n');
    }
    out.push_str(pick("## Removed in 0.4\n\n", "## 0.4-n kenduak\n\n"));
    out.push_str(pick(
        "These were accepted by QOA 0.3 but did nothing, or had no defined meaning. Using them is now an error that explains what to use instead.\n\n",
        "QOA 0.3-k onartzen zituen, baina ez zuten ezer egiten edo ez zuten esanahi zehatzik. Orain erabiltzea errore bat da, eta zer erabili azaltzen du.\n\n",
    ));
    out.push_str(pick(
        "| Instruction | Why |\n|---|---|\n",
        "| Agindua | Zergatik |\n|---|---|\n",
    ));
    for s in SPECS.iter() {
        if let Some(why) = s.removed_reason() {
            out.push_str(&format!("| `{}` | {} |\n", s.name, why));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_aliases_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for s in SPECS {
            assert!(seen.insert(s.name), "izen bikoiztua: {}", s.name);
            for a in s.aliases {
                assert!(seen.insert(a), "ezizen bikoiztua: {} ({}-n)", a, s.name);
            }
        }
    }

    #[test]
    fn operand_specs_parse() {
        for s in SPECS {
            for op in s.operand_specs() {
                assert!(
                    "qorcdvfinlsbgk".contains(op.kind),
                    "{}: bad kind {}",
                    s.name,
                    op.kind
                );
            }
        }
    }

    #[test]
    fn every_instruction_is_documented_in_both_languages() {
        for s in SPECS.iter().filter(|s| !s.is_removed()) {
            assert!(
                !s.summary_en.is_empty() && !s.summary_eu.is_empty(),
                "{} lacks a summary",
                s.name
            );
        }
    }

    #[test]
    fn lookup_is_case_insensitive() {
        assert_eq!(lookup("cnot").unwrap().name, "CNOT");
        assert_eq!(lookup("ToFfOlI").unwrap().name, "CCX");
        assert!(lookup("NOPE").is_none());
    }
}
