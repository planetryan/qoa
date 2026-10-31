// fitxategi honek OpenCL gailua kudeatzen du src/gpu.rs-entzat, vulkan-en ordez edo haren falta denean: AMD
// Instinct azeleragailuak (Frontier bezalako superkonputagailuetan, vulkan gabe), NVIDIA eta Intel GPUak, eta
// CPU inplementazioak (POCL). kernel bera da (src/shaders/tile.comp-en itzulpena OpenCL C-ra, f64 osoan:
// cl_khr_fp64 behar du), etapa bakoitzeko abiarazte bat.
//
// libOpenCL exekuzioan kargatzen da (libloading), beraz konpilatzeko ez da ezer behar, eta gailurik gabeko
// makinetan errore argi bat ematen du. egoera gehienez 16 zatitan doa (OpenCL-k buffer bakoitzaren tamaina
// mugatzen du: CL_DEVICE_MAX_MEM_ALLOC_SIZE).

use crate::gpu::{GpuGate, Stage, TILE_BITS};
use crate::{t, ts};
use libloading::Library;
use num_complex::Complex64 as C;
use std::ffi::{CString, c_char, c_void};
use std::time::Instant;

type Ptr = *mut c_void;

const TILE: usize = 1 << TILE_BITS;
const CHUNKS: usize = 16;
const AMP: usize = std::mem::size_of::<C>();

const CL_SUCCESS: i32 = 0;
const CL_DEVICE_TYPE_CPU: u64 = 1 << 1;
const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_DEVICE_TYPE_ACCELERATOR: u64 = 1 << 3;
const CL_DEVICE_TYPE_ALL: u64 = 0xFFFF_FFFF;
const CL_DEVICE_TYPE: u32 = 0x1000;
const CL_DEVICE_MAX_WORK_GROUP_SIZE: u32 = 0x1004;
const CL_DEVICE_MAX_MEM_ALLOC_SIZE: u32 = 0x1010;
const CL_DEVICE_GLOBAL_MEM_SIZE: u32 = 0x101F;
const CL_DEVICE_LOCAL_MEM_SIZE: u32 = 0x1023;
const CL_DEVICE_NAME: u32 = 0x102B;
const CL_DEVICE_DOUBLE_FP_CONFIG: u32 = 0x1032;
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
const CL_MEM_READ_WRITE: u64 = 1;
const CL_MEM_READ_ONLY: u64 = 1 << 2;

// src/shaders/tile.comp OpenCL C-n. THREADS konpilatzean ematen da (-D), gailuaren lan-talde handienaren arabera
const SOURCE: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define TILE_BITS 10
#define TILE (1u << TILE_BITS)
#define PER (TILE / THREADS)
#define GENERAL 0u
#define DIAGONAL 1u
#define SWAP 2u
#define FLIP 3u
#define REAL 4u

typedef struct {
    double2 m[4];
    uint kind, t, u, cmask, outside, sync, pad0, pad1;
} Gate;

typedef struct {
    uint pos[TILE_BITS];
    uint low, first, count, len, groups, chunk_bits;
} Params;

#define ARGS global double2 *c0, global double2 *c1, global double2 *c2, global double2 *c3, \
    global double2 *c4, global double2 *c5, global double2 *c6, global double2 *c7, \
    global double2 *c8, global double2 *c9, global double2 *c10, global double2 *c11, \
    global double2 *c12, global double2 *c13, global double2 *c14, global double2 *c15
#define PASS c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15

global double2 *pick(uint k, ARGS) {
    switch (k) {
    case 0: return c0; case 1: return c1; case 2: return c2; case 3: return c3;
    case 4: return c4; case 5: return c5; case 6: return c6; case 7: return c7;
    case 8: return c8; case 9: return c9; case 10: return c10; case 11: return c11;
    case 12: return c12; case 13: return c13; case 14: return c14; default: return c15;
    }
}

double2 cmul(double2 a, double2 b) {
    return (double2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

uint insert_zero(uint x, uint p) {
    return ((x >> p) << (p + 1)) | (x & ((1u << p) - 1u));
}

kernel void tile(global const Gate *gates, Params P, ARGS) {
    local double2 tl[TILE];
    uint g = get_group_id(0);
    if (g >= P.groups) {
        return;
    }
    uint base = g << P.low;
    for (uint j = P.low; j < TILE_BITS; j++) {
        base = insert_zero(base, P.pos[j]);
    }
    uint tid = get_local_id(0);
    uint low_mask = (1u << P.low) - 1u;
    uint in_chunk = (1u << P.chunk_bits) - 1u;
    uint idx[PER];
    for (uint e = 0; e < PER; e++) {
        uint l = tid + e * THREADS;
        uint off = l & low_mask;
        uint hi = l >> P.low;
        for (uint j = P.low; j < TILE_BITS; j++) {
            off |= (hi & 1u) << P.pos[j];
            hi >>= 1;
        }
        uint i = base + off;
        idx[e] = i;
        tl[l] = i < P.len ? pick(i >> P.chunk_bits, PASS)[i & in_chunk] : (double2)(0.0, 0.0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < P.count; k++) {
        Gate G = gates[P.first + k];
        if ((base & G.outside) == G.outside) {
            if (G.kind == DIAGONAL) {
                for (uint e = 0; e < PER; e++) {
                    uint l = tid + e * THREADS;
                    if ((l & G.cmask) == G.cmask) {
                        tl[l] = cmul(((l >> G.t) & 1u) != 0u ? G.m[3] : G.m[0], tl[l]);
                    }
                }
            } else if (G.kind == SWAP) {
                uint lo = min(G.t, G.u);
                uint hi = max(G.t, G.u);
                for (uint r = tid; r < TILE / 4; r += THREADS) {
                    uint i = insert_zero(insert_zero(r, lo), hi);
                    if ((i & G.cmask) == G.cmask) {
                        uint x = i | (1u << G.t);
                        uint y = i | (1u << G.u);
                        double2 v = tl[x];
                        tl[x] = tl[y];
                        tl[y] = v;
                    }
                }
            } else {
                for (uint r = tid; r < TILE / 2; r += THREADS) {
                    uint i0 = insert_zero(r, G.t);
                    if ((i0 & G.cmask) == G.cmask) {
                        uint i1 = i0 | (1u << G.t);
                        double2 a = tl[i0];
                        double2 b = tl[i1];
                        if (G.kind == FLIP) {
                            tl[i0] = b;
                            tl[i1] = a;
                        } else if (G.kind == REAL) {
                            tl[i0] = G.m[0].x * a + G.m[1].x * b;
                            tl[i1] = G.m[2].x * a + G.m[3].x * b;
                        } else {
                            tl[i0] = cmul(G.m[0], a) + cmul(G.m[1], b);
                            tl[i1] = cmul(G.m[2], a) + cmul(G.m[3], b);
                        }
                    }
                }
            }
        }
        if (G.sync != 0u) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for (uint e = 0; e < PER; e++) {
        uint i = idx[e];
        if (i < P.len) {
            pick(i >> P.chunk_bits, PASS)[i & in_chunk] = tl[tid + e * THREADS];
        }
    }
}
"#;

// kernel-eko `Params` (16 uint)
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    pos: [u32; TILE_BITS],
    low: u32,
    first: u32,
    count: u32,
    len: u32,
    groups: u32,
    chunk_bits: u32,
}

// erabiltzen diren OpenCL 1.2 funtzioak, liburutegitik kargatuak
#[allow(clippy::type_complexity)]
struct Api {
    _lib: Library,
    get_platform_ids: unsafe extern "C" fn(u32, *mut Ptr, *mut u32) -> i32,
    get_device_ids: unsafe extern "C" fn(Ptr, u64, u32, *mut Ptr, *mut u32) -> i32,
    get_device_info: unsafe extern "C" fn(Ptr, u32, usize, *mut c_void, *mut usize) -> i32,
    create_context: unsafe extern "C" fn(*const isize, u32, *const Ptr, Ptr, Ptr, *mut i32) -> Ptr,
    create_queue: unsafe extern "C" fn(Ptr, Ptr, u64, *mut i32) -> Ptr,
    create_program:
        unsafe extern "C" fn(Ptr, u32, *const *const c_char, *const usize, *mut i32) -> Ptr,
    build_program: unsafe extern "C" fn(Ptr, u32, *const Ptr, *const c_char, Ptr, Ptr) -> i32,
    build_info: unsafe extern "C" fn(Ptr, Ptr, u32, usize, *mut c_void, *mut usize) -> i32,
    create_kernel: unsafe extern "C" fn(Ptr, *const c_char, *mut i32) -> Ptr,
    create_buffer: unsafe extern "C" fn(Ptr, u64, usize, Ptr, *mut i32) -> Ptr,
    write_buffer: unsafe extern "C" fn(
        Ptr,
        Ptr,
        u32,
        usize,
        usize,
        *const c_void,
        u32,
        *const Ptr,
        *mut Ptr,
    ) -> i32,
    read_buffer: unsafe extern "C" fn(
        Ptr,
        Ptr,
        u32,
        usize,
        usize,
        *mut c_void,
        u32,
        *const Ptr,
        *mut Ptr,
    ) -> i32,
    set_arg: unsafe extern "C" fn(Ptr, u32, usize, *const c_void) -> i32,
    enqueue: unsafe extern "C" fn(
        Ptr,
        Ptr,
        u32,
        *const usize,
        *const usize,
        *const usize,
        u32,
        *const Ptr,
        *mut Ptr,
    ) -> i32,
    finish: unsafe extern "C" fn(Ptr) -> i32,
    release_mem: unsafe extern "C" fn(Ptr) -> i32,
    release_kernel: unsafe extern "C" fn(Ptr) -> i32,
    release_program: unsafe extern "C" fn(Ptr) -> i32,
    release_queue: unsafe extern "C" fn(Ptr) -> i32,
    release_context: unsafe extern "C" fn(Ptr) -> i32,
}

impl Api {
    fn load() -> Result<Api, String> {
        let names = [
            "libOpenCL.so.1",
            "libOpenCL.so",
            "OpenCL.dll",
            "/System/Library/Frameworks/OpenCL.framework/OpenCL",
        ];
        // segurtasuna: sistemaren OpenCL liburutegia kargatzea (bere hasieratzea besterik ez du egiten)
        let lib = names
            .iter()
            .find_map(|n| unsafe { Library::new(n) }.ok())
            .ok_or_else(|| {
                ts!(
                    "no OpenCL library found; install your GPU's OpenCL driver (AMD ROCm, NVIDIA, Intel), or POCL for the CPU",
                    "ez da OpenCL liburutegirik aurkitu; instalatu zure GPUaren OpenCL driverra (AMD ROCm, NVIDIA, Intel), edo POCL CPUrako"
                )
                .to_string()
            })?;
        macro_rules! sym {
            ($name:literal) => {
                // segurtasuna: funtzioaren mota OpenCL 1.2-ren goiburukoarena da
                *unsafe { lib.get(concat!($name, "\0").as_bytes()) }
                    .map_err(|e| format!("OpenCL: {}: {}", $name, e))?
            };
        }
        Ok(Api {
            get_platform_ids: sym!("clGetPlatformIDs"),
            get_device_ids: sym!("clGetDeviceIDs"),
            get_device_info: sym!("clGetDeviceInfo"),
            create_context: sym!("clCreateContext"),
            create_queue: sym!("clCreateCommandQueue"),
            create_program: sym!("clCreateProgramWithSource"),
            build_program: sym!("clBuildProgram"),
            build_info: sym!("clGetProgramBuildInfo"),
            create_kernel: sym!("clCreateKernel"),
            create_buffer: sym!("clCreateBuffer"),
            write_buffer: sym!("clEnqueueWriteBuffer"),
            read_buffer: sym!("clEnqueueReadBuffer"),
            set_arg: sym!("clSetKernelArg"),
            enqueue: sym!("clEnqueueNDRangeKernel"),
            finish: sym!("clFinish"),
            release_mem: sym!("clReleaseMemObject"),
            release_kernel: sym!("clReleaseKernel"),
            release_program: sym!("clReleaseProgram"),
            release_queue: sym!("clReleaseCommandQueue"),
            release_context: sym!("clReleaseContext"),
            _lib: lib,
        })
    }

    fn info_u64(&self, dev: Ptr, param: u32) -> u64 {
        let mut v = 0u64;
        // segurtasuna: gehienez 8 byte idazten ditu `v`-n
        unsafe {
            (self.get_device_info)(
                dev,
                param,
                8,
                &mut v as *mut u64 as *mut c_void,
                std::ptr::null_mut(),
            )
        };
        v
    }

    fn info_usize(&self, dev: Ptr, param: u32) -> usize {
        let mut v = 0usize;
        // segurtasuna: gehienez size_of::<usize>() byte idazten ditu `v`-n
        unsafe {
            (self.get_device_info)(
                dev,
                param,
                std::mem::size_of::<usize>(),
                &mut v as *mut usize as *mut c_void,
                std::ptr::null_mut(),
            )
        };
        v
    }

    fn info_string(&self, dev: Ptr, param: u32) -> String {
        let mut buf = vec![0u8; 512];
        let mut n = 0usize;
        // segurtasuna: gehienez buf.len() byte idazten ditu
        unsafe {
            (self.get_device_info)(
                dev,
                param,
                buf.len(),
                buf.as_mut_ptr() as *mut c_void,
                &mut n,
            )
        };
        buf.truncate(n.saturating_sub(1).min(buf.len()));
        String::from_utf8_lossy(&buf).trim().to_string()
    }
}

fn check(what: &str, code: i32) -> Result<(), String> {
    if code == CL_SUCCESS {
        Ok(())
    } else {
        Err(format!("OpenCL: {} ({})", what, code))
    }
}

pub struct Device {
    api: Api,
    context: Ptr,
    queue: Ptr,
    program: Ptr,
    kernel: Ptr,
    name: String,
    kind: &'static str,
    threads: usize,
    max_alloc: u64,
    global: u64,
    // egoeraren zatiak eta haien tamaina (2^chunk_bits anplitude), eta ateen bufferra
    chunks: Vec<Ptr>,
    chunk_bits: u32,
    len: usize,
    gates: Ptr,
    gates_cap: usize,
}

// segurtasuna: OpenCL objektuak hari batetik erabiltzen dira aldi bakoitzean (Gpu-ren Mutex-a)
unsafe impl Send for Device {}

// gailu-hautagai bat: puntuazioa (GPU > azeleragailua > CPU) eta f64 duen
struct Found {
    device: Ptr,
    score: u32,
    kind: &'static str,
}

impl Device {
    // gailu guztiak f64-rekin, onenetik (GPU, azeleragailua, CPU) eta ordena egonkorrean
    fn candidates(api: &Api) -> Vec<Found> {
        let mut n = 0u32;
        // segurtasuna: lehenik kopurua, gero zerrenda
        unsafe { (api.get_platform_ids)(0, std::ptr::null_mut(), &mut n) };
        let mut platforms = vec![std::ptr::null_mut(); n as usize];
        unsafe { (api.get_platform_ids)(n, platforms.as_mut_ptr(), std::ptr::null_mut()) };
        let mut out = Vec::new();
        for p in platforms {
            let mut m = 0u32;
            // segurtasuna: plataforma baliozkoa
            unsafe { (api.get_device_ids)(p, CL_DEVICE_TYPE_ALL, 0, std::ptr::null_mut(), &mut m) };
            let mut devices = vec![std::ptr::null_mut(); m as usize];
            unsafe {
                (api.get_device_ids)(
                    p,
                    CL_DEVICE_TYPE_ALL,
                    m,
                    devices.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            for d in devices {
                if api.info_u64(d, CL_DEVICE_DOUBLE_FP_CONFIG) == 0
                    || (api.info_u64(d, CL_DEVICE_LOCAL_MEM_SIZE) as usize) < TILE * AMP
                {
                    continue;
                }
                let ty = api.info_u64(d, CL_DEVICE_TYPE);
                let (score, kind) = if ty & CL_DEVICE_TYPE_GPU != 0 {
                    (3, ts!("GPU", "GPUa"))
                } else if ty & CL_DEVICE_TYPE_ACCELERATOR != 0 {
                    (2, ts!("accelerator", "azeleragailua"))
                } else if ty & CL_DEVICE_TYPE_CPU != 0 {
                    (1, ts!("cpu (OpenCL)", "cpu (OpenCL)"))
                } else {
                    (0, ts!("other", "beste bat"))
                };
                out.push(Found {
                    device: d,
                    score,
                    kind,
                });
            }
        }
        out.sort_by_key(|f| std::cmp::Reverse(f.score));
        out
    }

    // `index`: zenbatgarren gailua (onenetik), MPI prozesu bakoitzak berea erabil dezan
    pub fn open(index: usize) -> Result<Device, String> {
        let api = Api::load()?;
        let found = Self::candidates(&api);
        if found.is_empty() {
            return Err(ts!(
                "no OpenCL device with 64-bit floating point (cl_khr_fp64) was found",
                "ez da 64 biteko koma higikorra (cl_khr_fp64) duen OpenCL gailurik aurkitu"
            )
            .into());
        }
        // gailu-mota onenekoen artean txandaka (adib. nodo bateko GPU guztiak)
        let best: Vec<&Found> = found.iter().filter(|f| f.score == found[0].score).collect();
        let f = best[index % best.len()];
        let dev = f.device;
        let mut err = 0i32;
        // segurtasuna: gailu baliozkoa, beste parametroak nuluak (lehenetsiak)
        let context = unsafe {
            (api.create_context)(
                std::ptr::null(),
                1,
                &dev,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut err,
            )
        };
        check("clCreateContext", err)?;
        let queue = unsafe { (api.create_queue)(context, dev, 0, &mut err) };
        check("clCreateCommandQueue", err)?;
        let threads = api
            .info_usize(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE)
            .clamp(1, 256)
            .next_power_of_two()
            .min(256);
        let threads = if threads > api.info_usize(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE).max(1) {
            threads / 2
        } else {
            threads
        };
        let src = CString::new(SOURCE).unwrap();
        let ptr = src.as_ptr();
        let len = SOURCE.len();
        let program = unsafe { (api.create_program)(context, 1, &ptr, &len, &mut err) };
        check("clCreateProgramWithSource", err)?;
        let opts = CString::new(format!("-D THREADS={}u", threads)).unwrap();
        let built = unsafe {
            (api.build_program)(
                program,
                1,
                &dev,
                opts.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if built != CL_SUCCESS {
            let mut log = vec![0u8; 1 << 16];
            let mut n = 0usize;
            unsafe {
                (api.build_info)(
                    program,
                    dev,
                    CL_PROGRAM_BUILD_LOG,
                    log.len(),
                    log.as_mut_ptr() as *mut c_void,
                    &mut n,
                )
            };
            log.truncate(n.min(log.len()));
            return Err(t!(
                "OpenCL: the kernel did not build ({}): {}",
                "OpenCL: kernela ez da konpilatu ({}): {}",
                built,
                String::from_utf8_lossy(&log).trim()
            ));
        }
        let name = CString::new("tile").unwrap();
        let kernel = unsafe { (api.create_kernel)(program, name.as_ptr(), &mut err) };
        check("clCreateKernel", err)?;
        Ok(Device {
            name: api.info_string(dev, CL_DEVICE_NAME),
            kind: f.kind,
            threads,
            max_alloc: api.info_u64(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE),
            global: api.info_u64(dev, CL_DEVICE_GLOBAL_MEM_SIZE),
            api,
            context,
            queue,
            program,
            kernel,
            chunks: Vec::new(),
            chunk_bits: 0,
            len: 0,
            gates: std::ptr::null_mut(),
            gates_cap: 0,
        })
    }

    pub fn name(&self) -> String {
        format!("{} (OpenCL)", self.name)
    }

    pub fn kind(&self) -> &'static str {
        self.kind
    }

    fn buffer(&self, flags: u64, bytes: usize) -> Result<Ptr, String> {
        let mut err = 0i32;
        // segurtasuna: testuinguru baliozkoa, ostalariaren erakuslerik gabe
        let b = unsafe {
            (self.api.create_buffer)(self.context, flags, bytes, std::ptr::null_mut(), &mut err)
        };
        check("clCreateBuffer", err).map_err(|e| {
            t!(
                "{} while reserving {} on {}",
                "{} {} erreserbatzean {} gailuan",
                e,
                crate::state::format_bytes(bytes as u128),
                self.name
            )
        })?;
        Ok(b)
    }

    // egoeraren zatiak: 2^k anplitude bakoitza, gailuaren buffer handiena gainditu gabe, gehienez CHUNKS
    fn ensure_state(&mut self, len: usize) -> Result<(), String> {
        if self.len == len {
            return Ok(());
        }
        for c in self.chunks.drain(..) {
            // segurtasuna: geuk sortutako bufferra
            unsafe { (self.api.release_mem)(c) };
        }
        let total = (len * AMP) as u64;
        if total > self.global {
            return Err(t!(
                "the state ({}) does not fit in {}'s memory ({})",
                "egoera ({}) ez da sartzen {}-ren memorian ({})",
                crate::state::format_bytes(total as u128),
                self.name,
                crate::state::format_bytes(self.global as u128)
            ));
        }
        let mut bits = len.next_power_of_two().trailing_zeros();
        while bits > 0 && ((1u64 << bits) * AMP as u64) > self.max_alloc {
            bits -= 1;
        }
        let count = len.div_ceil(1 << bits);
        if count > CHUNKS {
            return Err(t!(
                "the state ({}) needs more than {} of {}'s largest buffers ({} each)",
                "egoerak ({}) {}-ren buffer handienetako {} baino gehiago behar ditu ({} bakoitza)",
                crate::state::format_bytes(total as u128),
                CHUNKS,
                self.name,
                crate::state::format_bytes(self.max_alloc as u128)
            ));
        }
        for k in 0..count {
            let amps = (len - k * (1 << bits)).min(1 << bits);
            let b = self.buffer(CL_MEM_READ_WRITE, amps * AMP)?;
            self.chunks.push(b);
        }
        self.chunk_bits = bits;
        self.len = len;
        Ok(())
    }

    fn ensure_gates(&mut self, count: usize) -> Result<(), String> {
        if count <= self.gates_cap {
            return Ok(());
        }
        if !self.gates.is_null() {
            // segurtasuna: geuk sortutako bufferra
            unsafe { (self.api.release_mem)(self.gates) };
        }
        let cap = count.next_power_of_two();
        self.gates = self.buffer(CL_MEM_READ_ONLY, cap * std::mem::size_of::<GpuGate>())?;
        self.gates_cap = cap;
        Ok(())
    }

    pub fn apply(&mut self, amps: &mut [C], stages: &[Stage]) -> Result<(f64, f64), String> {
        let len = amps.len();
        self.ensure_state(len)?;
        let all: Vec<GpuGate> = stages
            .iter()
            .flat_map(|s| s.gates.iter().copied())
            .collect();
        self.ensure_gates(all.len().max(1))?;
        let api = &self.api;
        let step = 1usize << self.chunk_bits;

        let t0 = Instant::now();
        // segurtasuna: buffer bakoitza bere zatiaren tamainakoa da (ensure_state), idazketak blokeatzaileak
        unsafe {
            for (k, c) in self.chunks.iter().enumerate() {
                let part = &amps[k * step..((k + 1) * step).min(len)];
                check(
                    "clEnqueueWriteBuffer",
                    (api.write_buffer)(
                        self.queue,
                        *c,
                        1,
                        0,
                        part.len() * AMP,
                        part.as_ptr() as *const c_void,
                        0,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                    ),
                )?;
            }
            check(
                "clEnqueueWriteBuffer",
                (api.write_buffer)(
                    self.queue,
                    self.gates,
                    1,
                    0,
                    all.len() * std::mem::size_of::<GpuGate>(),
                    all.as_ptr() as *const c_void,
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                ),
            )?;
        }
        let up = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let groups = len.div_ceil(TILE);
        // segurtasuna: argumentuak kernelaren sinadurarekin bat datoz; erabiltzen ez diren zatiek lehena errepikatzen dute
        unsafe {
            check(
                "clSetKernelArg",
                (api.set_arg)(
                    self.kernel,
                    0,
                    std::mem::size_of::<Ptr>(),
                    &self.gates as *const Ptr as *const c_void,
                ),
            )?;
            for k in 0..CHUNKS {
                let c = self.chunks.get(k).unwrap_or(&self.chunks[0]);
                check(
                    "clSetKernelArg",
                    (api.set_arg)(
                        self.kernel,
                        2 + k as u32,
                        std::mem::size_of::<Ptr>(),
                        c as *const Ptr as *const c_void,
                    ),
                )?;
            }
            let mut first = 0u32;
            for s in stages {
                let p = Params {
                    pos: s.pos,
                    low: s.low,
                    first,
                    count: s.gates.len() as u32,
                    len: len as u32,
                    groups: groups as u32,
                    chunk_bits: self.chunk_bits,
                };
                first += s.gates.len() as u32;
                check(
                    "clSetKernelArg",
                    (api.set_arg)(
                        self.kernel,
                        1,
                        std::mem::size_of::<Params>(),
                        &p as *const Params as *const c_void,
                    ),
                )?;
                let global = groups * self.threads;
                let local = self.threads;
                check(
                    "clEnqueueNDRangeKernel",
                    (api.enqueue)(
                        self.queue,
                        self.kernel,
                        1,
                        std::ptr::null(),
                        &global,
                        &local,
                        0,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                    ),
                )?;
            }
            check("clFinish", (api.finish)(self.queue))?;
        }
        let compute = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        // segurtasuna: irakurketa blokeatzaileak, zati bakoitza bere tamainan
        unsafe {
            for (k, c) in self.chunks.iter().enumerate() {
                let part = &mut amps[k * step..((k + 1) * step).min(len)];
                check(
                    "clEnqueueReadBuffer",
                    (api.read_buffer)(
                        self.queue,
                        *c,
                        1,
                        0,
                        part.len() * AMP,
                        part.as_mut_ptr() as *mut c_void,
                        0,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                    ),
                )?;
            }
        }
        Ok((up + t2.elapsed().as_secs_f64(), compute))
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // segurtasuna: geuk sortutako objektuak, behin askatuak
        unsafe {
            for c in self.chunks.drain(..) {
                (self.api.release_mem)(c);
            }
            if !self.gates.is_null() {
                (self.api.release_mem)(self.gates);
            }
            (self.api.release_kernel)(self.kernel);
            (self.api.release_program)(self.program);
            (self.api.release_queue)(self.queue);
            (self.api.release_context)(self.context);
        }
    }
}
