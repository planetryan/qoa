// fitxategi honek qubit-ate sortak GPUan aplikatzen ditu, vulkan bidez (gailua src/vulkan.rs-en, kernela
// src/shaders/tile.comp-en) edo OpenCL bidez (src/opencl.rs, kernel bera OpenCL C-n).
//
// CPUan bezala (src/kernel.rs), egoera handietan muga memoriaren banda-zabalera da, eta neurria bera: ate-
// segida bat "lauza" batean osorik aplikatu, egoera behin zeharkatuz segida osorako, ate bakoitzeko behin
// beharrean. GPUan lauza 2^10 anplitude da (16 KiB memoria partekatuan, vulkan-ek edozein gailutan bermatzen
// duena), eta edozein 10 qubitez osa daiteke: beti 5 baxuenak (irakurketa jarraituak, 512 byte) gehi etapako
// ateen helburuak, eta gainerakoa hurrengo baxuenekin. kontrolek ez dute lauzan egon behar: lauzatik
// kanpokoak lauzaren oinarriarekin egiaztatzen dira.
//
// planifikatzailea (`plan`) eta bere emulazioa (`emulate`, shader-aren indize-kalkulu bera rust-en) vulkan
// gabe ere konpilatzen eta probatzen dira; gailua `vulkan` ezaugarriarekin bakarrik.
//
// GPUan qubit-ateak bakarrik: neurketak, optika eta zarata CPUan geratzen dira. sorta bakoitzak egoera
// gailura igotzen du, ateak aplikatzen ditu eta itzultzen du; txostenak zenbat byte mugitu diren esaten du.

use crate::kernel::{self, QGate};
use crate::t;
use num_complex::Complex64 as C;
use std::sync::Mutex;

// src/shaders/tile.comp-eko berdinak
pub const TILE_BITS: usize = 10;
const TILE: usize = 1 << TILE_BITS;
const GENERAL: u32 = 0;
const DIAGONAL: u32 = 1;
const SWAP: u32 = 2;
const FLIP: u32 = 3;
const REAL: u32 = 4;

// lauza guztietan dauden qubit baxuak: 2^5 anplitude jarraitu = 512 byte irakurketa bakoitzean
const LOW_MIN: usize = 5;

// shader-eko indizeak u32 dira
pub const MAX_AMPS: usize = 1 << 31;

const ZERO: C = C::new(0.0, 0.0);

// ate bat GPUrako, tokiko qubitekin (shader-eko `Gate`-ren diseinu bera, std430: 96 byte)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuGate {
    pub m: [[f64; 2]; 4],
    pub kind: u32,
    // helburua (edo trukaketaren lehen qubita) eta trukaketaren bigarrena, lauzako posizioetan
    pub t: u32,
    pub u: u32,
    // lauzaren barruko kontrolak (tokiko posizioak) eta kanpokoak (qubit globalak)
    pub cmask: u32,
    pub outside: u32,
    // 1: ate honen ondoren barrera
    pub sync: u32,
    pad: [u32; 2],
}

// etapa bat: lauza osatzen duten qubitak (goranzko ordenan; lehen `low`-ak 0..low-1 dira) eta haren ateak
#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub pos: [u32; TILE_BITS],
    pub low: u32,
    pub gates: Vec<GpuGate>,
}

fn targets(g: &QGate) -> usize {
    match *g {
        QGate::Mat { q, .. } => 1 << q,
        QGate::Swap { a, b, .. } => 1 << a | 1 << b,
    }
}

// ate-sorta etapetan banatu. ate bat etapa irekira sartzen da bere helburuekin lauza 10 qubitetan sartzen
// bada; bestela etapa itxi eta berri bat hasten da. qubit gutxi badira (≤ 10), lauza 2^10 anplitude
// jarraitu da, qubit guztiak barruan, eta sorta osoa etapa bakarra
pub fn plan(n_qubits: usize, gates: &[QGate]) -> Vec<Stage> {
    let mut stages = Vec::new();
    if gates.is_empty() {
        return stages;
    }
    if n_qubits <= TILE_BITS {
        stages.push(stage(TILE - 1, gates));
        return stages;
    }
    let base = (1usize << LOW_MIN) - 1;
    let mut tile = base;
    let mut start = 0;
    for (i, g) in gates.iter().enumerate() {
        let next = tile | targets(g);
        if next.count_ones() as usize > TILE_BITS {
            stages.push(stage(fill(tile, n_qubits), &gates[start..i]));
            start = i;
            tile = base | targets(g);
        } else {
            tile = next;
        }
    }
    stages.push(stage(fill(tile, n_qubits), &gates[start..]));
    stages
}

// lauza osatu hurrengo qubit baxuenekin: lasterketa jarraitu luzeagoak
fn fill(mut tile: usize, n_qubits: usize) -> usize {
    for q in 0..n_qubits {
        if tile.count_ones() as usize == TILE_BITS {
            break;
        }
        tile |= 1 << q;
    }
    tile
}

// matrizearen forma, shader-eko bide merkeena aukeratzeko. kontsumo-GPUetan f64 kalkulua f32arena baino
// 32-64 aldiz motelagoa da, eta etapa batek ate asko aplikatzen dituenez lauza bakoitzean, kalkulua izan
// daiteke muga (ez memoria): biderketa bakoitza kontatzen da
fn kind(m: &[C; 4]) -> u32 {
    let one = C::new(1.0, 0.0);
    if m[1] == ZERO && m[2] == ZERO {
        DIAGONAL
    } else if m[0] == ZERO && m[3] == ZERO && m[1] == one && m[2] == one {
        FLIP
    } else if m.iter().all(|c| c.im == 0.0) {
        REAL
    } else {
        GENERAL
    }
}

fn stage(tile: usize, gates: &[QGate]) -> Stage {
    let mut pos = [0u32; TILE_BITS];
    let mut m = tile;
    let mut j = 0;
    while m != 0 {
        pos[j] = m.trailing_zeros();
        j += 1;
        m &= m - 1;
    }
    let local = |q: usize| (tile & ((1 << q) - 1)).count_ones();
    let remap = |mask: usize| {
        let mut out = 0u32;
        let mut m = mask & tile;
        while m != 0 {
            out |= 1 << local(m.trailing_zeros() as usize);
            m &= m - 1;
        }
        out
    };
    let mut out: Vec<GpuGate> = gates
        .iter()
        .map(|g| match *g {
            QGate::Mat { q, m, controls } => GpuGate {
                m: m.map(|c| [c.re, c.im]),
                kind: kind(&m),
                t: local(q),
                u: 0,
                cmask: remap(controls),
                outside: (controls & !tile) as u32,
                sync: 1,
                pad: [0; 2],
            },
            QGate::Swap { a, b, controls } => GpuGate {
                m: [[0.0; 2]; 4],
                kind: SWAP,
                t: local(a),
                u: local(b),
                cmask: remap(controls),
                outside: (controls & !tile) as u32,
                sync: 1,
                pad: [0; 2],
            },
        })
        .collect();
    // barrerak: ate diagonal batek hari bakoitzaren elementuak bakarrik ukitzen ditu, eta itzulketak elementu
    // berak idazten ditu; beraz bi diagonalen artean eta azken diagonalaren ondoren ez da barrerarik behar
    for i in 0..out.len() {
        let next_diagonal = out.get(i + 1).is_none_or(|n| n.kind == DIAGONAL);
        out[i].sync = !(out[i].kind == DIAGONAL && next_diagonal) as u32;
    }
    Stage {
        pos,
        low: tile.trailing_ones(),
        gates: out,
    }
}

// x-ri 0 bit bat sartu p posizioan (shader-eko berdina)
fn insert_zero(x: usize, p: u32) -> usize {
    ((x >> p) << (p + 1)) | (x & ((1 << p) - 1))
}

// lauzaren oinarria: g-ren bitak lauzakoak ez diren posizioetan
fn tile_base(s: &Stage, g: usize) -> usize {
    let mut base = g << s.low;
    for &p in &s.pos[s.low as usize..] {
        base = insert_zero(base, p);
    }
    base
}

// lauzako l elementuaren desplazamendua oinarritik
fn tile_offset(s: &Stage, l: usize) -> usize {
    let mut off = l & ((1 << s.low) - 1);
    let mut hi = l >> s.low;
    for &p in &s.pos[s.low as usize..] {
        off |= (hi & 1) << p;
        hi >>= 1;
    }
    off
}

fn cmul(a: [f64; 2], b: C) -> C {
    C::new(a[0], a[1]) * b
}

// ate bat lauzan (shader-eko begizta berak, hari bakarrean)
fn apply_local(tile: &mut [C], g: &GpuGate) {
    match g.kind {
        DIAGONAL => {
            for (l, a) in tile.iter_mut().enumerate() {
                if l as u32 & g.cmask == g.cmask {
                    *a = cmul(if l >> g.t & 1 == 1 { g.m[3] } else { g.m[0] }, *a);
                }
            }
        }
        SWAP => {
            let (lo, hi) = (g.t.min(g.u), g.t.max(g.u));
            for r in 0..TILE / 4 {
                let i = insert_zero(insert_zero(r, lo), hi);
                if i as u32 & g.cmask == g.cmask {
                    tile.swap(i | 1 << g.t, i | 1 << g.u);
                }
            }
        }
        _ => {
            for r in 0..TILE / 2 {
                let i0 = insert_zero(r, g.t);
                if i0 as u32 & g.cmask == g.cmask {
                    let i1 = i0 | 1 << g.t;
                    let (a, b) = (tile[i0], tile[i1]);
                    (tile[i0], tile[i1]) = match g.kind {
                        FLIP => (b, a),
                        REAL => (g.m[0][0] * a + g.m[1][0] * b, g.m[2][0] * a + g.m[3][0] * b),
                        _ => (
                            cmul(g.m[0], a) + cmul(g.m[1], b),
                            cmul(g.m[2], a) + cmul(g.m[3], b),
                        ),
                    };
                }
            }
        }
    }
}

// etapak CPUan, shader-ak egiten duen bezala (lauzaz lauza): planifikatzailearen eta indize-kalkuluaren
// proba, GPUrik gabe
pub fn emulate(amps: &mut [C], stages: &[Stage]) {
    let len = amps.len();
    let mut tile = vec![ZERO; TILE];
    let mut idx = vec![0usize; TILE];
    for s in stages {
        for g in 0..len.div_ceil(TILE) {
            let base = tile_base(s, g);
            for l in 0..TILE {
                idx[l] = base + tile_offset(s, l);
                tile[l] = if idx[l] < len { amps[idx[l]] } else { ZERO };
            }
            for gate in &s.gates {
                if base as u32 & gate.outside == gate.outside {
                    apply_local(&mut tile, gate);
                }
            }
            for l in 0..TILE {
                if idx[l] < len {
                    amps[idx[l]] = tile[l];
                }
            }
        }
    }
}

// shader-eko lan-talde baten hariak
const THREADS: usize = 256;

// etapa baten barrerak nahikoak diren: shader-eko hari bakoitzak zein lauza-elementu ukitzen dituen jarraitzen
// du, barrera batetik hurrengora, eta bi hari desberdinek elementu bera ukitzen badute (bat idazten ari dela),
// hori lasterketa bat da. kontrolak eta kanpoko kontrolak sarbideak kentzen dituzte bakarrik, beraz denak
// betetzat hartzen dira (kasurik txarrena). lavapipe-k lan-talde baten hariak urratsez urrats batera
// exekutatzen ditu eta ez luke barrera falta bat erakutsiko; hau da egiaztapena
pub fn races(s: &Stage) -> Option<String> {
    let owned = |tid: usize| (0..TILE / THREADS).map(move |e| tid + e * THREADS);
    // fase bat: hari bakoitzak ukitutako elementuak; `from`: fasearen lehen eragiketa
    let check = |phase: &[Vec<usize>], from: &str| {
        let mut who = vec![usize::MAX; TILE];
        for (tid, elems) in phase.iter().enumerate() {
            for &l in elems {
                if who[l] != usize::MAX && who[l] != tid {
                    return Some(format!(
                        "{from}: lauzako {l}. elementua {} eta {tid} hariek ukitzen dute barrerarik gabe",
                        who[l]
                    ));
                }
                who[l] = tid;
            }
        }
        None
    };
    // kargaren ondoren barrera beti dago
    let load: Vec<Vec<usize>> = (0..THREADS).map(|t| owned(t).collect()).collect();
    if let Some(r) = check(&load, "kargatzea") {
        return Some(r);
    }
    let mut phase: Vec<Vec<usize>> = vec![Vec::new(); THREADS];
    let mut from = String::from("0. atea");
    for (k, g) in s.gates.iter().enumerate() {
        for (tid, elems) in phase.iter_mut().enumerate() {
            match g.kind {
                DIAGONAL => elems.extend(owned(tid)),
                SWAP => {
                    let (lo, hi) = (g.t.min(g.u), g.t.max(g.u));
                    for r in (tid..TILE / 4).step_by(THREADS) {
                        let i = insert_zero(insert_zero(r, lo), hi);
                        elems.extend([i | 1 << g.t, i | 1 << g.u]);
                    }
                }
                _ => {
                    for r in (tid..TILE / 2).step_by(THREADS) {
                        let i0 = insert_zero(r, g.t);
                        elems.extend([i0, i0 | 1 << g.t]);
                    }
                }
            }
        }
        if g.sync != 0 {
            if let Some(r) = check(&phase, &from) {
                return Some(r);
            }
            phase.iter_mut().for_each(Vec::clear);
            from = format!("{}. atea", k + 1);
        }
    }
    // itzulketa: hari bakoitzak bere elementuak
    for (tid, elems) in phase.iter_mut().enumerate() {
        elems.extend(owned(tid));
    }
    check(&phase, &from)
}

// GPUaren erabilera, txostenerako
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Stats {
    pub device: String,
    pub device_type: String,
    // sortak (bakoitzak egoera igo eta jaitsi), egoeraren gaineko pasaldiak (etapak) eta ateak (fusioaren ondoren)
    pub batches: u64,
    pub passes: u64,
    pub gates: u64,
    pub bytes_up: u64,
    pub bytes_down: u64,
    // denbora: egoera mugitzen eta kalkulatzen
    pub transfer_seconds: f64,
    pub compute_seconds: f64,
}

// zein API erabili: vulkan, OpenCL, edo lehen dabilena (vulkan lehenik)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    Auto,
    Vulkan,
    OpenCl,
}

// gailu bat, eraikuntza honetan dagoen API-rekin
enum Device {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::vulkan::Device>),
    #[cfg(feature = "opencl")]
    OpenCl(Box<crate::opencl::Device>),
    // API gabeko eraikuntzetan mota hutsik ez geratzeko; ezin da sortu
    #[allow(dead_code)]
    Never(std::convert::Infallible),
}

impl Device {
    fn open(api: Api, index: usize) -> Result<Device, String> {
        let _ = index;
        let vulkan = || -> Result<Device, String> {
            #[cfg(feature = "vulkan")]
            return crate::vulkan::Device::open(index).map(|d| Device::Vulkan(Box::new(d)));
            #[cfg(not(feature = "vulkan"))]
            Err(t!(
                "this QOA was built without Vulkan (cargo feature `vulkan`)",
                "QOA hau vulkan gabe konpilatu da (cargo-ren `vulkan` ezaugarria)"
            ))
        };
        let opencl = || -> Result<Device, String> {
            #[cfg(feature = "opencl")]
            return crate::opencl::Device::open(index).map(|d| Device::OpenCl(Box::new(d)));
            #[cfg(not(feature = "opencl"))]
            Err(t!(
                "this QOA was built without OpenCL (cargo feature `opencl`)",
                "QOA hau OpenCL gabe konpilatu da (cargo-ren `opencl` ezaugarria)"
            ))
        };
        match api {
            Api::Vulkan => vulkan(),
            Api::OpenCl => opencl(),
            Api::Auto => {
                vulkan().or_else(|v| opencl().map_err(|o| format!("Vulkan: {}. OpenCL: {}", v, o)))
            }
        }
    }

    fn name(&self) -> String {
        match self {
            #[cfg(feature = "vulkan")]
            Device::Vulkan(d) => d.name(),
            #[cfg(feature = "opencl")]
            Device::OpenCl(d) => d.name(),
            Device::Never(n) => match *n {},
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            #[cfg(feature = "vulkan")]
            Device::Vulkan(d) => d.kind(),
            #[cfg(feature = "opencl")]
            Device::OpenCl(d) => d.kind(),
            Device::Never(n) => match *n {},
        }
    }

    // (APIrik gabe argumentuak ez dira erabiltzen)
    #[allow(unused_variables)]
    fn apply(&mut self, amps: &mut [C], stages: &[Stage]) -> Result<(f64, f64), String> {
        match self {
            #[cfg(feature = "vulkan")]
            Device::Vulkan(d) => d.apply(amps, stages),
            #[cfg(feature = "opencl")]
            Device::OpenCl(d) => d.apply(amps, stages),
            Device::Never(n) => match *n {},
        }
    }
}

// prozesu honen GPU-zenbakia nodoan: QOA_GPU_DEVICE, bestela MPI abiarazlearen tokiko rank-a (openmpi, mpich,
// slurm, mvapich, cray), bestela 0. nodo bateko prozesu bakoitzak bere GPUa hartzen du horrela
pub fn local_index() -> usize {
    [
        "QOA_GPU_DEVICE",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "PMI_LOCAL_RANK",
        "PALS_LOCAL_RANKID",
    ]
    .iter()
    .find_map(|v| std::env::var(v).ok()?.trim().parse().ok())
    .unwrap_or(0)
}

// GPU bat, hari guztien artean partekatua (shot paraleloek txandaka erabiltzen dute)
pub struct Gpu {
    device: Mutex<Device>,
    stats: Mutex<Stats>,
}

impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gpu({})", self.stats.lock().unwrap().device)
    }
}

impl Gpu {
    // gailu egokiena ireki (diskretua > integratua > birtuala > cpu), edo zergatik ez dagoen esan
    pub fn open() -> Result<Gpu, String> {
        Self::open_with(Api::Auto, local_index())
    }

    // API jakin bat eta mota onenekoen artean zenbatgarren gailua
    pub fn open_with(api: Api, index: usize) -> Result<Gpu, String> {
        let device = Device::open(api, index)?;
        let stats = Stats {
            device: device.name(),
            device_type: device.kind().to_string(),
            ..Stats::default()
        };
        Ok(Gpu {
            device: Mutex::new(device),
            stats: Mutex::new(stats),
        })
    }

    // ate-sorta bat: fusioa, etapak, eta gailuan: igo, aplikatu, jaitsi
    pub fn apply(&self, amps: &mut [C], n_qubits: usize, gates: &[QGate]) -> Result<(), String> {
        if amps.len() > MAX_AMPS {
            return Err(t!(
                "the state has {} amplitudes; the GPU kernel indexes at most 2^31",
                "egoerak {} anplitude ditu; GPUaren kernelak gehienez 2^31 indexatzen ditu",
                amps.len()
            ));
        }
        let fused = kernel::fuse(gates);
        if fused.is_empty() {
            return Ok(());
        }
        let stages = plan(n_qubits, &fused);
        let (transfer, compute) = self.device.lock().unwrap().apply(amps, &stages)?;
        let bytes = (amps.len() * 16) as u64;
        let mut s = self.stats.lock().unwrap();
        s.batches += 1;
        s.passes += stages.len() as u64;
        s.gates += fused.len() as u64;
        s.bytes_up += bytes;
        s.bytes_down += bytes;
        s.transfer_seconds += transfer;
        s.compute_seconds += compute;
        Ok(())
    }

    pub fn stats(&self) -> Stats {
        self.stats.lock().unwrap().clone()
    }
}
