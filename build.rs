fn main() {

    // enable native cpu optimizations, allowing the compiler to generate code
    // specifically for the cpu it's being compiled on.

    println!("cargo:rustc-flag=-C target-cpu=native");

    // set optimization level to 3, which is a good balance between compilation time and performance.

    println!("cargo:rustc-flag=-C opt-level=3");

    // enable link-time optimizations (lto) across all crates, allowing for
    // more aggressive inter-procedural optimizations.

    println!("cargo:rustc-flag=-C lto=fat");

    // set codegen units to 1, which means the entire crate is compiled as a single unit,

    println!("cargo:rustc-flag=-C codegen-units=1");

    // set panic strategy to abort, which can result in smaller binaries and
    // potentially better performance by removing panic unwinding code.

    println!("cargo:rustc-flag=-C panic=abort");

    // these flags instruct llvm to be more aggressive in its optimization passes.

    println!("cargo:rustc-flag=-C llvm-args=-vectorize-slp-aggressive"); // aggressive superword-level parallelism
    println!("cargo:rustc-flag=-C llvm-args=-enable-cond-stores-vec");   // enable vectorization of conditional stores
    println!("cargo:rustc-flag=-C llvm-args=-slp-vectorize-hor-store"); // horizontal store vectorization
    println!("cargo:rustc-flag=-C llvm-args=-enable-masked-vector-loads"); // enable masked vector loads
    println!("cargo:rustc-flag=-C llvm-args=-enable-gvn-hoist");       // enable global value numbering hoisting
    println!("cargo:rustc-flag=-C llvm-args=-enable-coroutines");      // enable coroutines (might not be directly relevant but good for general perf)
    println!("cargo:rustc-flag=-C llvm-args=-force-vector-width=32");  // force a specific vector width for some operations
}
