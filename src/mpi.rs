// fitxategi honek MPI taldea ematen du (`qoa run --mpi`). MPI liburutegia exekuzioan kargatzen da (libloading),
// beraz QOA konpilatzeko ez da MPI-rik, C konpiladorerik edo mpicc-rik behar, eta argitalpen-bitar guztiek
// dakarte; MPI makinan instalatuta egotea nahikoa da.
//
// MPI-k ez du ABI bakarra: heldulekuen (komunikatzailea, datu-mota) eta konstanteen forma inplementazioaren
// araberakoa da. bi familia ezagutzen dira:
//   - open mpi: heldulekuak erakusleak dira, liburutegiko aldagai globalen helbideak (ompi_mpi_comm_world,
//     ompi_mpi_byte), eta MPI_STATUS_IGNORE NULL da
//   - mpich-en ABI-a (mpich, intel mpi, mvapich, cray mpich, parastation, microsoft mpi): heldulekuak int
//     konstanteak dira, mpi.h-n finkoak (MPI_COMM_WORLD 0x44000000, MPI_BYTE 0x4c00010d), eta
//     MPI_STATUS_IGNORE 1 da
// liburutegia: QOA_MPI_LIB bada, hori; bestela abiarazleak erakusten duena (open mpi-ren mpirun-ek OMPI_*
// ingurune-aldagaiak jartzen ditu) eta ohiko izenak. familia liburutegiak berak esaten du: open mpi-ren
// aldagai globalak ditu, eta besteek MPI_Get_library_version-en izena ematen dute. ezagutzen ez den
// liburutegi batekin errore bat ematen da, asmatu beharrean.
//
// mezu handiak 1 GiB-eko zatietan doaz (MPI-ren kontagailuak int dira). luzera 8 byte gisa doa (prozesu
// guztiak QOA bitar bera dira, makina-mota berean).
// unsafe hemen: MPI liburutegiari deiak (kanpoko C funtzioak, exekuzioan kargatuak)
#![allow(unsafe_code)]

use crate::team::Team;
use crate::{t, ts};
use libloading::Library;
use std::ffi::{c_char, c_int, c_void};

const CHUNK: usize = 1 << 30;
const THREAD_FUNNELED: c_int = 1;

type Ptr = *mut c_void;

// MPI funtzioak, heldulekuaren motarekin (H: open mpi-n erakuslea, mpich-en int)
struct Fns<H> {
    initialized: unsafe extern "C" fn(*mut c_int) -> c_int,
    finalized: unsafe extern "C" fn(*mut c_int) -> c_int,
    init_thread:
        unsafe extern "C" fn(*mut c_int, *mut *mut *mut c_char, c_int, *mut c_int) -> c_int,
    finalize: unsafe extern "C" fn() -> c_int,
    rank: unsafe extern "C" fn(H, *mut c_int) -> c_int,
    size: unsafe extern "C" fn(H, *mut c_int) -> c_int,
    barrier: unsafe extern "C" fn(H) -> c_int,
    send: unsafe extern "C" fn(*const c_void, c_int, H, c_int, c_int, H) -> c_int,
    recv: unsafe extern "C" fn(Ptr, c_int, H, c_int, c_int, H, Ptr) -> c_int,
    #[allow(clippy::type_complexity)]
    sendrecv: unsafe extern "C" fn(
        *const c_void,
        c_int,
        H,
        c_int,
        c_int,
        Ptr,
        c_int,
        H,
        c_int,
        c_int,
        H,
        Ptr,
    ) -> c_int,
    world: H,
    byte: H,
    status_ignore: Ptr,
}

enum Api {
    OpenMpi(Fns<Ptr>),
    Mpich(Fns<c_int>),
}

pub struct World {
    api: Api,
    rank: usize,
    size: usize,
    // zein liburutegi, `qoa run`-en txostenerako eta erroreetarako
    pub library: String,
}

// segurtasuna: heldulekuak prozesu osoan baliozkoak diren identifikatzaile aldaezinak dira (MPI_COMM_WORLD,
// MPI_BYTE), eta funtzioak liburutegi kargatu batenak (ez da inoiz deskargatzen). taldea hari nagusitik
// erabiltzen da, MPI_THREAD_FUNNELED-ekin hasieratuta
unsafe impl Send for World {}
unsafe impl Sync for World {}

// liburutegiaren izenak, probatzeko ordenan
fn candidates() -> Vec<String> {
    if let Ok(path) = std::env::var("QOA_MPI_LIB") {
        return vec![path];
    }
    let openmpi = std::env::vars_os().any(|(k, _)| k.to_string_lossy().starts_with("OMPI_"));
    let (ompi, mpich): (&[&str], &[&str]) = if cfg!(target_os = "macos") {
        (
            &["libmpi.40.dylib"],
            &["libmpi.12.dylib", "libmpich.12.dylib"],
        )
    } else if cfg!(windows) {
        (&[], &["msmpi.dll"])
    } else {
        (
            &["libmpi.so.40"],
            &["libmpi.so.12", "libmpich.so.12", "libmpi_cray.so.12"],
        )
    };
    let mut out: Vec<String> = if openmpi {
        ompi.iter().chain(mpich).map(|s| s.to_string()).collect()
    } else {
        mpich.iter().chain(ompi).map(|s| s.to_string()).collect()
    };
    if !cfg!(windows) {
        out.push(
            if cfg!(target_os = "macos") {
                "libmpi.dylib"
            } else {
                "libmpi.so"
            }
            .into(),
        );
    }
    out
}

// open mpi-ren osagaiek libmpi-ren ikurrak behar dituzte: liburutegia RTLD_GLOBAL-ekin kargatu
fn open(name: &str) -> Result<Library, libloading::Error> {
    #[cfg(unix)]
    {
        use libloading::os::unix::{Library as Unix, RTLD_GLOBAL, RTLD_NOW};
        // segurtasuna: MPI liburutegi baten hasieratzaileak ez du baldintzarik QOA-ren aldetik
        unsafe { Unix::open(Some(name), RTLD_NOW | RTLD_GLOBAL) }.map(Library::from)
    }
    #[cfg(not(unix))]
    {
        // segurtasuna: goian bezala
        unsafe { Library::new(name) }
    }
}

fn symbol<T: Copy>(lib: &Library, name: &str) -> Result<T, String> {
    let mut z = name.as_bytes().to_vec();
    z.push(0);
    // segurtasuna: T deitzaileak MPI estandarraren sinadurarekin ematen du (Fns-eko motak)
    unsafe { lib.get::<T>(&z) }.map(|s| *s).map_err(|e| {
        t!(
            "the MPI library has no {}: {}",
            "MPI liburutegiak ez du {}: {}",
            name,
            e
        )
    })
}

fn functions<H: Copy>(
    lib: &Library,
    world: H,
    byte: H,
    status_ignore: Ptr,
) -> Result<Fns<H>, String> {
    Ok(Fns {
        initialized: symbol(lib, "MPI_Initialized")?,
        finalized: symbol(lib, "MPI_Finalized")?,
        init_thread: symbol(lib, "MPI_Init_thread")?,
        finalize: symbol(lib, "MPI_Finalize")?,
        rank: symbol(lib, "MPI_Comm_rank")?,
        size: symbol(lib, "MPI_Comm_size")?,
        barrier: symbol(lib, "MPI_Barrier")?,
        send: symbol(lib, "MPI_Send")?,
        recv: symbol(lib, "MPI_Recv")?,
        sendrecv: symbol(lib, "MPI_Sendrecv")?,
        world,
        byte,
        status_ignore,
    })
}

// MPI_Get_library_version-en testua (MPI 3: hasieratu aurretik dei daiteke), edo hutsa ez badago
fn library_version(lib: &Library) -> String {
    type Version = unsafe extern "C" fn(*mut c_char, *mut c_int) -> c_int;
    let Ok(f) = symbol::<Version>(lib, "MPI_Get_library_version") else {
        return String::new();
    };
    // MPI_MAX_LIBRARY_VERSION_STRING handiena (mpich: 8192)
    let mut buf = vec![0u8; 8192];
    let mut len: c_int = 0;
    // segurtasuna: bufferrak MPI_MAX_LIBRARY_VERSION_STRING byte ditu, estandarrak eskatzen duen bezala
    unsafe { f(buf.as_mut_ptr().cast(), &mut len) };
    let n = (len.max(0) as usize).min(buf.len());
    String::from_utf8_lossy(&buf[..n])
        .trim_end_matches('\0')
        .trim()
        .to_string()
}

// mpich-en ABI-a erabiltzen duten inplementazioen izenak beren bertsio-testuan
const MPICH_FAMILY: &[&str] = &[
    "MPICH",
    "Intel(R) MPI",
    "MVAPICH",
    "CRAY MPICH",
    "ParaStation",
    "Microsoft MPI",
];

fn load() -> Result<(Api, String), String> {
    let names = candidates();
    let mut tried = Vec::new();
    for name in &names {
        let lib = match open(name) {
            Ok(l) => l,
            Err(e) => {
                tried.push(format!("{} ({})", name, e));
                continue;
            }
        };
        let api = if let Ok(world) = symbol::<Ptr>(&lib, "ompi_mpi_comm_world") {
            // open mpi: heldulekuak aldagai globalen helbideak dira (dlsym-ek itzultzen dituenak)
            let (w, b) = (world, symbol::<Ptr>(&lib, "ompi_mpi_byte")?);
            Api::OpenMpi(functions(&lib, w, b, std::ptr::null_mut())?)
        } else {
            let version = library_version(&lib);
            let mpich = MPICH_FAMILY.iter().any(|k| version.contains(k))
                || (cfg!(windows) && name.eq_ignore_ascii_case("msmpi.dll"));
            if !mpich {
                return Err(t!(
                    "{} is an MPI library QOA does not know ({}). QOA works with Open MPI and with the MPICH family (MPICH, Intel MPI, MVAPICH, Cray MPICH, ParaStation, Microsoft MPI); QOA_MPI_LIB chooses the library",
                    "{} QOA-k ezagutzen ez duen MPI liburutegi bat da ({}). QOA-k open mpi-rekin eta mpich familiarekin (MPICH, Intel MPI, MVAPICH, Cray MPICH, ParaStation, Microsoft MPI) lan egiten du; QOA_MPI_LIB-ek liburutegia aukeratzen du",
                    name,
                    if version.is_empty() {
                        ts!("no version", "bertsiorik gabe").to_string()
                    } else {
                        version
                    }
                ));
            }
            // mpich-en ABI-a: mpi.h-ko konstante finkoak
            Api::Mpich(functions(
                &lib,
                0x4400_0000,
                0x4c00_010d,
                std::ptr::without_provenance_mut(1),
            )?)
        };
        // liburutegia ez da inoiz deskargatzen: open mpi-k ez du onartzen, eta funtzioak prozesu osoan behar dira
        std::mem::forget(lib);
        return Ok((api, name.clone()));
    }
    Err(t!(
        "--mpi: no MPI library found (tried {}). Install Open MPI or MPICH, or set QOA_MPI_LIB to the library's path, then start QOA with mpirun or srun",
        "--mpi: ez da MPI liburutegirik aurkitu ({} probatuta). instalatu open mpi edo mpich, edo jarri QOA_MPI_LIB liburutegiaren bidean, eta abiarazi QOA mpirun edo srun-ekin",
        tried.join(", ")
    ))
}

impl<H: Copy> Fns<H> {
    // segurtasuna (guztietan): funtzioak MPI liburutegi kargatu batenak dira, beren sinadura estandarrarekin,
    // eta buffer guztiak baliozkoak dira adierazitako luzerarekin

    fn init(&self) -> Result<(usize, usize), String> {
        let mut done: c_int = 0;
        unsafe { (self.initialized)(&mut done) };
        if done == 0 {
            let mut provided: c_int = 0;
            let rc = unsafe {
                (self.init_thread)(
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    THREAD_FUNNELED,
                    &mut provided,
                )
            };
            if rc != 0 {
                return Err(ts!("MPI did not start", "MPI ez da abiarazi").into());
            }
        }
        let (mut r, mut n): (c_int, c_int) = (0, 1);
        unsafe {
            (self.rank)(self.world, &mut r);
            (self.size)(self.world, &mut n);
        }
        Ok((r.max(0) as usize, n.max(1) as usize))
    }

    fn finalize(&self) {
        let mut done: c_int = 0;
        unsafe { (self.finalized)(&mut done) };
        if done == 0 {
            unsafe { (self.finalize)() };
        }
    }

    fn send(&self, to: usize, data: &[u8]) {
        let n = (data.len() as u64).to_ne_bytes();
        unsafe { (self.send)(n.as_ptr().cast(), 8, self.byte, to as c_int, 1, self.world) };
        for part in data.chunks(CHUNK) {
            unsafe {
                (self.send)(
                    part.as_ptr().cast(),
                    part.len() as c_int,
                    self.byte,
                    to as c_int,
                    2,
                    self.world,
                )
            };
        }
    }

    fn recv(&self, from: usize) -> Vec<u8> {
        let mut n = [0u8; 8];
        unsafe {
            (self.recv)(
                n.as_mut_ptr().cast(),
                8,
                self.byte,
                from as c_int,
                1,
                self.world,
                self.status_ignore,
            )
        };
        let mut out = vec![0u8; u64::from_ne_bytes(n) as usize];
        for part in out.chunks_mut(CHUNK) {
            unsafe {
                (self.recv)(
                    part.as_mut_ptr().cast(),
                    part.len() as c_int,
                    self.byte,
                    from as c_int,
                    2,
                    self.world,
                    self.status_ignore,
                )
            };
        }
        out
    }

    fn exchange(&self, partner: usize, send: &[u8], recv: &mut [u8]) {
        for (s, r) in send.chunks(CHUNK).zip(recv.chunks_mut(CHUNK)) {
            unsafe {
                (self.sendrecv)(
                    s.as_ptr().cast(),
                    s.len() as c_int,
                    self.byte,
                    partner as c_int,
                    3,
                    r.as_mut_ptr().cast(),
                    r.len() as c_int,
                    self.byte,
                    partner as c_int,
                    3,
                    self.world,
                    self.status_ignore,
                )
            };
        }
    }

    fn barrier(&self) {
        unsafe { (self.barrier)(self.world) };
    }
}

macro_rules! each {
    ($api:expr, $f:ident => $body:expr) => {
        match $api {
            Api::OpenMpi($f) => $body,
            Api::Mpich($f) => $body,
        }
    };
}

impl World {
    pub fn init() -> Result<World, String> {
        let (api, library) = load()?;
        let (rank, size) = each!(&api, f => f.init())?;
        Ok(World {
            api,
            rank,
            size,
            library,
        })
    }
}

impl Drop for World {
    fn drop(&mut self) {
        each!(&self.api, f => f.finalize())
    }
}

impl Team for World {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn send(&self, to: usize, data: &[u8]) {
        each!(&self.api, f => f.send(to, data))
    }

    fn recv(&self, from: usize) -> Vec<u8> {
        each!(&self.api, f => f.recv(from))
    }

    fn exchange(&self, partner: usize, send: &[u8], recv: &mut [u8]) {
        assert_eq!(send.len(), recv.len());
        each!(&self.api, f => f.exchange(partner, send, recv))
    }

    fn barrier(&self) {
        each!(&self.api, f => f.barrier())
    }
}
