// fitxategi honek sistemaren liburutegiak kargatzen ditu (OpenCL, MPI) uneko karpetatik inoiz kargatu gabe.
// izen huts batekin ("OpenCL.dll", "libmpi.40.dylib"), Windows-ek eta macOS-ek uneko karpetan ere bilatzen
// dute: deskarga-karpeta batean jarritako liburutegi gaizto bat exekutatuko litzateke `--gpu` edo `--mpi`
// erabiltzean. hemen izen hutsak sistemaren lekuetan bakarrik bilatzen dira:
//   - linux eta beste unix-ak: kargatzailearen bide arruntak (ld.so.conf, LD_LIBRARY_PATH); uneko karpeta ez
//   - windows: aplikazioaren karpeta eta System32 (LOAD_LIBRARY_SEARCH_DEFAULT_DIRS), ez uneko karpeta ez PATH
//   - macos: /opt/homebrew/lib, /usr/local/lib eta /opt/local/lib (homebrew, macports)
// bide oso bat (QOA_MPI_LIB=/opt/mpi/lib/libmpi.so) bere horretan kargatzen da: erabiltzaileak aukeratu du.
// unsafe hemen: kanpoko liburutegi bat kargatzeak haren hasieratzaileak exekutatzen ditu
#![allow(unsafe_code)]

use libloading::Library;

#[cfg(target_os = "macos")]
const MAC_DIRS: &[&str] = &["/opt/homebrew/lib", "/usr/local/lib", "/opt/local/lib"];

#[cfg(any(windows, target_os = "macos"))]
fn is_path(name: &str) -> bool {
    name.contains('/') || (cfg!(windows) && name.contains('\\'))
}

// `global`: ikurrak beste liburutegientzat ikusgai (open mpi-ren osagaiek behar dute)
pub fn open(name: &str, global: bool) -> Result<Library, String> {
    #[cfg(target_os = "macos")]
    if !is_path(name) {
        let mut last = String::new();
        for dir in MAC_DIRS {
            match open_exact(&format!("{}/{}", dir, name), global) {
                Ok(l) => return Ok(l),
                Err(e) => last = e,
            }
        }
        return Err(last);
    }
    open_exact(name, global)
}

fn open_exact(name: &str, global: bool) -> Result<Library, String> {
    #[cfg(unix)]
    {
        use libloading::os::unix::{Library as Unix, RTLD_GLOBAL, RTLD_LOCAL, RTLD_NOW};
        let flags = RTLD_NOW | if global { RTLD_GLOBAL } else { RTLD_LOCAL };
        // segurtasuna: liburutegi baten hasieratzaileek ez dute baldintzarik QOA-ren aldetik
        unsafe { Unix::open(Some(name), flags) }
            .map(Library::from)
            .map_err(|e| e.to_string())
    }
    #[cfg(windows)]
    {
        let _ = global;
        use libloading::os::windows::{LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, Library as Win};
        // segurtasuna: goian bezala. izen hutsa: aplikazioaren karpeta eta System32 bakarrik
        if is_path(name) {
            unsafe { Win::new(name) }
        } else {
            unsafe { Win::load_with_flags(name, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }
        }
        .map(Library::from)
        .map_err(|e| e.to_string())
    }
}
