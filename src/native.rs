// fitxategi honek `qoa compile --emit so`-k egindako kode natiboa exekutatzen du QOA-ren makinaren barruan: kode
// natiboak eragiketa klasiko sinpleak egiten ditu makina-aginduekin, eta gainerakoetarako makinari deitzen dio
// (step, tick). liburutegia kargatzea ere hemen dago (libloading), eta liburutegiak daraman .qx-a irakurtzea.
//
// unsafe hemen: kode natiboari deitzea (kanpoko funtzio bat erakusle batetik), haren deiak (step, tick) gure
// testuingurura itzultzea erakusle gordinen bidez, eta liburutegi bat kargatzea eta haren ikurrak irakurtzea.
// kode natiboak ez du inoiz makinaren erregistroak zuzenean ukitzen: bere kopia du (`regs` hemen), eta step
// bakoitzaren aurretik eta ondoren makinarekin sinkronizatzen da, beraz rust-en erreferentziek ez dute
// memoria hori partekatzen kode natiboa exekutatzen ari den bitartean
#![allow(unsafe_code)]

use crate::compile::NativeEntry;
use crate::program::NUM_REGS;
use std::ffi::c_void;

// kode natiboak exekutatzen duen makina
pub trait Host {
    // `pc` eragiketa exekutatu, interpretatzaileak bezala: hurrengo pc, -1 gelditu, -2 errorea (makinak gordeta).
    // `regs` kode natiboaren erregistro-kopia da: makinara eraman aurretik, eta itzuli ondoren
    fn step(&mut self, pc: usize, regs: &mut [f64; NUM_REGS]) -> i64;
    // eragiketa natibo baten aurretik: urrats-kontaketa eta muga, laginketaren unea. 0 jarraitu, -2 errorea
    fn tick(&mut self, pc: usize) -> i64;
}

// deien testuingurua: makina eta erregistro-kopia, erakusle gordinak (kode natiboa exekutatzen ari den
// bitartean ez dago haien rust erreferentziarik)
struct Ctx<'a> {
    host: *mut (dyn Host + 'a),
    regs: *mut [f64; NUM_REGS],
}

extern "C" fn step_cb(ctx: *mut c_void, pc: u64) -> i64 {
    // SAFETY: ctx `enter`-eko Ctx da, dei osoan bizirik; kode natiboa geldirik dago dei honek irauten duen
    // bitartean, beraz hemengo erreferentziak dira memoria horren erabiltzaile bakarrak
    unsafe {
        let ctx = &mut *(ctx as *mut Ctx);
        (*ctx.host).step(pc as usize, &mut *ctx.regs)
    }
}

extern "C" fn tick_cb(ctx: *mut c_void, pc: u64) -> i64 {
    // SAFETY: step_cb-n bezala
    unsafe {
        let ctx = &mut *(ctx as *mut Ctx);
        (*ctx.host).tick(pc as usize)
    }
}

// kode natiboa exekutatu: (emaitza-kodea, amaierako erregistroak). 0 amaituta, 1 errorea, 2 kodea eta programa
// ez datoz bat
pub fn enter(
    entry: NativeEntry,
    host: &mut dyn Host,
    regs: [f64; NUM_REGS],
) -> (i64, [f64; NUM_REGS]) {
    let mut regs = regs;
    let mut ctx = Ctx {
        host: host as *mut dyn Host,
        regs: &mut regs as *mut [f64; NUM_REGS],
    };
    let regs_ptr = ctx.regs as *mut f64;
    // SAFETY: `entry` `qoa compile`-k sortutako sarrera da (load-ek egiaztatzen du hitzarmenaren bertsioa):
    // (testuingurua, 32 f64-ko erregistro-kopia, step, tick) hartzen ditu C hitzarmenarekin, eta itzuli
    // aurretik ez du erakuslerik gordetzen
    let code = unsafe {
        entry(
            &mut ctx as *mut Ctx as *mut c_void,
            regs_ptr,
            step_cb,
            tick_cb,
        )
    };
    (code, regs)
}

// kargatutako programa natiboa: liburutegia (bizirik egon behar du exekuzioak irauten duen bitartean),
// sarrera eta programa bera (liburutegiak daraman .qx-tik, laburpena eta guzti egiaztatuta)
#[cfg(feature = "native")]
pub struct Loaded {
    _lib: libloading::Library,
    pub entry: NativeEntry,
    pub program: crate::program::Program,
}

// `qoa compile --emit so`-k egindako liburutegi bat kargatu. liburutegi bat kargatzeak haren kodea exekutatzen
// du: zuk eraikitako liburutegiak bakarrik exekutatu, programa exekutagarriak bezala
#[cfg(feature = "native")]
pub fn load(path: &std::path::Path) -> Result<Loaded, String> {
    use crate::compile::{ABI, ABI_VERSION, ENTRY, QX, QX_LEN};
    use crate::t;
    // bide osoa: izen huts bat (`programa.so`) sistemaren bilaketa-bideetan bilatuko litzateke bestela
    let full = std::fs::canonicalize(path).map_err(|e| {
        t!(
            "cannot read {}: {}",
            "ezin da {} irakurri: {}",
            path.display(),
            e
        )
    })?;
    // SAFETY: erabiltzaileak exekutatzeko emandako liburutegia da (programa exekutagarri bat bezala)
    let lib = unsafe { libloading::Library::new(&full) }.map_err(|e| {
        t!(
            "cannot load {} as native code for this machine: {}",
            "ezin da {} kargatu makina honetarako kode natibo gisa: {}",
            path.display(),
            e
        )
    })?;
    let not_qoa = |what: &str| {
        t!(
            "{} is not QOA native code ({} is missing); build it with qoa compile --emit so",
            "{} ez da QOA kode natiboa ({} falta da); eraiki qoa compile --emit so-rekin",
            path.display(),
            what
        )
    };
    // SAFETY: ikurrak `qoa compile`-k idatzitako datuak dira, beren motekin: u32, u64, byteak eta sarrera
    unsafe {
        let abi_sym = lib
            .get::<*const u32>(ABI.as_bytes())
            .map_err(|_| not_qoa(ABI))?;
        let abi = **abi_sym;
        if abi != ABI_VERSION {
            return Err(t!(
                "{} was built for native-code version {}, and this qoa runs version {}; build it again with this qoa",
                "{} kode natiboaren {}. bertsiorako eraiki zen, eta qoa honek {}. bertsioa exekutatzen du; eraiki berriz qoa honekin",
                path.display(),
                abi,
                ABI_VERSION
            ));
        }
        let len_sym = lib
            .get::<*const u64>(QX_LEN.as_bytes())
            .map_err(|_| not_qoa(QX_LEN))?;
        let len = **len_sym;
        let qx_sym = lib
            .get::<*const u8>(QX.as_bytes())
            .map_err(|_| not_qoa(QX))?;
        // luzera zentzuzkoa: liburutegia bera baino handiagoa ezin da izan
        let file_len = std::fs::metadata(&full).map(|m| m.len()).unwrap_or(0);
        if len > file_len {
            return Err(not_qoa(QX_LEN));
        }
        let bytes = std::slice::from_raw_parts(*qx_sym, len as usize).to_vec();
        let program = crate::program::Program::from_bytes(&bytes)
            .map_err(|e| format!("{}: {}", path.display(), e))?;
        let entry = *lib
            .get::<NativeEntry>(ENTRY.as_bytes())
            .map_err(|_| not_qoa(ENTRY))?;
        Ok(Loaded {
            _lib: lib,
            entry,
            program,
        })
    }
}
