// fitxategi honek MPI-ren geruza (src/mpi_shim.c) konpilatzen du `mpi` ezaugarriarekin bakarrik; bestela ez du
// ezer egiten. konpiladorea: QOA_MPICC, MPICC edo `mpicc` (cray-n `cc`, QOA_MPICC=cc). estekatzeko liburutegiak
// QOA_MPI_LIBS-etik (adib. "-L/opt/mpi/lib -lmpi"), edo konpiladore-bilgarriari galdetuta: openmpi-ren
// `-showme:link`, mpich-en `-link_info`.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/mpi_shim.c");
    println!("cargo:rerun-if-env-changed=QOA_MPICC");
    println!("cargo:rerun-if-env-changed=MPICC");
    println!("cargo:rerun-if-env-changed=QOA_MPI_LIBS");
    if std::env::var_os("CARGO_FEATURE_MPI").is_none() {
        return;
    }
    let mpicc = std::env::var("QOA_MPICC")
        .or_else(|_| std::env::var("MPICC"))
        .unwrap_or_else(|_| "mpicc".into());
    #[cfg(feature = "mpi")]
    cc::Build::new()
        .compiler(&mpicc)
        .file("src/mpi_shim.c")
        .warnings(true)
        .compile("qoa_mpi");
    let flags = std::env::var("QOA_MPI_LIBS").ok().or_else(|| {
        ["-showme:link", "-link_info"].iter().find_map(|arg| {
            let out = Command::new(&mpicc).arg(arg).output().ok()?;
            out.status
                .success()
                .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
        })
    });
    let Some(flags) = flags else {
        panic!(
            "--features mpi: cannot ask {} for its link flags; set QOA_MPI_LIBS (e.g. \"-L/opt/mpi/lib -lmpi\")",
            mpicc
        );
    };
    for token in flags.split_whitespace() {
        if let Some(dir) = token.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={}", dir);
        } else if let Some(lib) = token.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={}", lib);
        } else if let Some(rpath) = token.strip_prefix("-Wl,-rpath,") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath);
        }
    }
}
