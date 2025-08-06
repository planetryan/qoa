fn main() {
    use std::env;
    use std::path::Path;
    use std::fs;

    // get the target triple from cargo
    let target = env::var("TARGET").unwrap_or_else(|_| String::from("x86_64-unknown-linux-gnu"));

    // --- compile and link ---
    let mut build = cc::Build::new();
    build.flag("-fPIC"); // add -fPIC to generate position-independent code for linking
    build.opt_level(3);  // set optimization level

    // determine architecture based on target triple
    let (arch_dir, specific_flags) = if target.contains("x86_64") {
        // use GNU math instead of Intel SVML
        ("x86-64", vec!["-O3", "-ffast-math", "-fno-math-errno", "-funsafe-math-optimizations"])
    } else if target.contains("aarch64") {
        // for arm64, use armv8-a instead of native
        ("aarch64", vec!["-march=armv8-a"])
    } else if target.contains("riscv64") {
        // for risc-v, use rv64gc instead of native
        ("riscv64", vec!["-march=rv64gc", "-mabi=lp64d"])
    } else if target.contains("powerpc64") {
        // for power, no specific march flag needed
        ("power64", vec!["-O3"])
    } else {
        ("generic", vec!["-O3"])
    };

    // set appropriate openblas target
    if target.contains("aarch64") {
        println!("cargo:rustc-env=OPENBLAS_TARGET=ARMV8");
    } else if target.contains("riscv64") {
        println!("cargo:rustc-env=OPENBLAS_TARGET=RISCV64_GENERIC");
    } else if target.contains("powerpc64") {
        println!("cargo:rustc-env=OPENBLAS_TARGET=POWER8");
    }

    // conditional compilation of asm_math and linking
    // only compile and link asm_math for x86_64 and aarch64
    if target.contains("x86_64") || target.contains("aarch64") {
        // check if the architecture-specific directory exists
        let asm_dir = format!("src/asm/{}", arch_dir);
        let asm_path = Path::new(&asm_dir);
        
        if !asm_path.exists() {
            // create directory if it doesn't exist
            fs::create_dir_all(&asm_path).expect("failed to create asm directory");
            
            // create a stub assembly file if it doesn't exist
            let asm_file = asm_path.join("main.S");
            if !asm_file.exists() {
                fs::write(
                    &asm_file,
                    format!("// stub assembly file for {}\n// generated for cross-compilation\n", arch_dir)
                ).expect("failed to create stub assembly file");
            }
        }

        // use the appropriate assembly file for the target architecture
        let asm_file = format!("{}/main.S", asm_dir);
        
        // add architecture-specific flags
        for flag in specific_flags {
            build.flag(flag);
        }
        
        // compile the appropriate assembly file
        build.file(&asm_file);
        build.compile("asm_math"); // outputs libasm_math.a
        println!("cargo:rustc-link-lib=static=asm_math");
    }

    // --- rust compiler optimization flags ---
    println!("cargo:rustc-flag=-C opt-level=3");
    println!("cargo:rustc-flag=-C lto=fat");
    println!("cargo:rustc-flag=-C codegen-units=1");
    println!("cargo:rustc-flag=-C panic=abort");

    // explicitly link to libm for math functions
    println!("cargo:rustc-link-lib=m");
    
    // Architecture specific rust flags
    if target.contains("x86_64") {
        // use native cpu but explicitly disable svml and use gnu-based optimizations
        println!("cargo:rustc-flag=-C target-cpu=native"); 
        println!("cargo:rustc-flag=-C target-feature=-svml");
        
        // create a compatibility shim only for __svml_pow2, which is the one that was missing
        let out_dir = env::var("OUT_DIR").unwrap();
        let wrapper_path = Path::new(&out_dir).join("svml_compat.c");
        fs::write(
            &wrapper_path,
            r#"
            #include <math.h>
            
            // only provide implementations for functions that aren't already in assembly
            double __svml_pow2(double x) {
                return exp2(x);
            }
            "#
        ).expect("could not write svml compatibility wrapper");
        
        // compile the svml compatibility wrapper
        cc::Build::new()
            .file(wrapper_path)
            .flag("-O3")
            .flag("-ffast-math")
            .flag("-fno-math-errno")
            .compile("svml_compat");
    } else if target.contains("aarch64") {
        // arm-specific optimizations and native cpu
        println!("cargo:rustc-flag=-C target-cpu=native");
        println!("cargo:rustc-flag=-C target-feature=+neon,+fp-armv8,+crypto");
    }
    // removed target-cpu=native and target-feature flags for riscv64 and powerpc64
    // as we are trying to avoid any architecture-specific assembly/intrinsics for these targets
    // and rely on pure-rust implementations.

    // handle conditional tls compilation
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    match target_arch.as_str() {
        "riscv64" | "powerpc64" => {
            println!("cargo:rustc-cfg=feature=\"openssl-tls\"");
            // force openssl vendored for these architectures
            println!("cargo:rustc-env=OPENSSL_STATIC=1");
            println!("cargo:rustc-env=OPENSSL_VENDORED=1");
        }
        _ => {
            println!("cargo:rustc-cfg=feature=\"rustls-tls\"");
        }
    }
}
