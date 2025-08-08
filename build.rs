fn main() {
    use std::env;
    use std::fs;
    use std::path::Path;

    // get the target triple from cargo
    let target = env::var("TARGET").unwrap_or_else(|_| String::from("x86_64-unknown-linux-gnu"));

    // compilation and linking setup for C code
    let mut build = cc::Build::new();
    // add -fPIC to generate position-independent code for linking
    build.flag("-fPIC");
    // set optimization level
    build.opt_level(3);

    // determine architecture based on target triple
    let (arch_dir, specific_flags) = if target.contains("x86_64") {
        // use gnu math
        (
            "x86-64",
            vec![
                "-O3",
                "-ffast-math",
                "-fno-math-errno",
                "-funsafe-math-optimizations",
            ],
        )
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
            fs::create_dir_all(&asm_path).expect("Failed to create asm directory");

            // create a stub assembly file if it doesn't exist
            let asm_file = asm_path.join("main.S");
            if !asm_file.exists() {
                fs::write(
                    &asm_file,
                    format!(
                        "// stub assembly file for {}\n// generated for cross-compilation\n",
                        arch_dir
                    ),
                )
                .expect("Failed to create stub assembly file");
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
        // outputs libasm_math.a
        build.compile("asm_math");
        println!("cargo:rustc-link-lib=static=asm_math");
    }

    // rust compiler flags
    println!("cargo:rustc-flag=-C opt-level=3");
    println!("cargo:rustc-flag=-C lto=fat");
    println!("cargo:rustc-flag=-C codegen-units=1");
    println!("cargo:rustc-flag=-C panic=abort");

    // explicitly link to libm for math functions
    println!("cargo:rustc-link-lib=m");

    // architecture specific rust flags
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
"#,
        )
        .expect("could not write svml compatibility wrapper");

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

    // shader compilation
    // only compile shaders if vulkan is enabled
    // this check is done at compile time by the rust compiler
    #[cfg(feature = "vulkan")]
    compile_shaders();
}

#[cfg(feature = "vulkan")]
fn compile_shaders() {
    // moved these imports here because they are only used in this function
    use std::env;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    // tell cargo to re-run this if shaders change
    println!("cargo:rerun-if-changed=src/kernel/shaders/");

    // define our paths
    let shader_dir = Path::new("src/kernel/shaders");
    let out_dir = env::var("OUT_DIR").unwrap();
    let shader_out_dir = Path::new(&out_dir).join("shaders");

    // create output directory if it doesn't exist
    fs::create_dir_all(&shader_out_dir).expect("Failed to create shader output directory");

    // define the shader types to compile
    let shader_types = vec![
        ("hadamard", "hadamard.comp"),
        ("paulix", "paulix.comp"),
        ("pauliz", "pauliz.comp"),
        ("cnot", "cnot.comp"),
        ("rx", "rx.comp"),
        ("ry", "ry.comp"),
        ("rz", "rz.comp"),
        ("s", "s.comp"),
        ("t", "t.comp"),
        ("swap", "swap.comp"),
    ];

    // generate a rust module with the compiled spir-v data
    let shader_rs_path = PathBuf::from(&out_dir).join("shaders.rs");
    let mut shader_rs = File::create(&shader_rs_path).expect("Failed to create shaders.rs");

    // write the header for the shader module
    writeln!(shader_rs, "//! Auto-generated shader module").unwrap();

    // check if glslangValidator is available
    let glslang_available = Command::new("glslangValidator")
        .arg("--version")
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    // helper to create shader paths
    let make_shader_path = |name: &str| shader_dir.join(name);

    if glslang_available {
        for (name, filename) in shader_types {
            let source_path = make_shader_path(filename);
            let spirv_path = shader_out_dir.join(format!("{}.spv", name));

            // check if source file exists, if not create a stub
            if !source_path.exists() {
                let parent_dir = source_path.parent().unwrap();
                fs::create_dir_all(parent_dir).expect("Failed to create shader directory");

                // create stub shader file
                let shader_stub = get_shader_stub(name);
                fs::write(&source_path, shader_stub).expect(&format!(
                    "Failed to write stub shader: {}",
                    source_path.display()
                ));
            }

            // compile glsl to spir-v using glslangValidator
            let status = Command::new("glslangValidator")
                .args(&[
                    "-V",
                    "-o",
                    &spirv_path.to_string_lossy(),
                    &source_path.to_string_lossy(),
                ])
                .status()
                .expect(&format!(
                    "Failed to execute glslangValidator for shader: {}",
                    source_path.display()
                ));

            if !status.success() {
                panic!("Failed to compile shader: {}", source_path.display());
            }

            // include the compiled binary data in the rust file
            writeln!(
                shader_rs,
                "#[cfg(feature = \"vulkan\")]
pub static SPIRV_{}: &[u8] = include_bytes!(r\"{}\");",
                name.to_uppercase(),
                spirv_path.to_string_lossy()
            )
            .unwrap();
        }
    } else {
        println!("cargo:warning=glslangValidator not found. Using pre-compiled SPIR-V stubs.");

        // generate empty stubs for each shader type
        for (name, _) in shader_types {
            writeln!(
                shader_rs,
                "#[cfg(feature = \"vulkan\")]
pub static SPIRV_{}: &[u8] = &[]; // Stub - glslangValidator not available",
                name.to_uppercase()
            )
            .unwrap();
        }
    }
}

// this function is called by compile_shaders to generate a placeholder
// shader file if one doesn't exist
#[cfg(feature = "vulkan")]
fn get_shader_stub(gate_type: &str) -> String {
    match gate_type {
        "hadamard" => r#"#version 450
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// state vector (complex numbers stored as pairs of floats)
layout(std430, binding = 0) buffer StateVector {
    vec2 state[];
};

// gate parameters
layout(std140, binding = 1) uniform GateParams {
    uint gate_type;
    uint target_qubit;
    uint num_qubits;
    uint _pad;
} params;

// constants
const float INV_SQRT_2 = 0.7071067811865475; // 1/sqrt(2)

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint state_size = 1u << params.num_qubits;

    // return if index is out of bounds
    if (gid >= state_size) {
        return;
    }

    // compute paired index (flipping the target qubit)
    uint paired_idx = gid ^ (1u << params.target_qubit);

    // only process the first index of each pair to avoid race conditions
    if (gid < paired_idx) {
        // get current amplitudes
        vec2 amp_0 = state[gid];
        vec2 amp_1 = state[paired_idx];

        // apply hadamard transformation
        vec2 new_amp_0 = vec2(
            INV_SQRT_2 * (amp_0.x + amp_1.x),
            INV_SQRT_2 * (amp_0.y + amp_1.y)
        );

        vec2 new_amp_1 = vec2(
            INV_SQRT_2 * (amp_0.x - amp_1.x),
            INV_SQRT_2 * (amp_0.y - amp_1.y)
        );

        // write back results
        state[gid] = new_amp_0;
        state[paired_idx] = new_amp_1;
    }
}"#.to_string(),
        "paulix" => r#"#version 450
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// state vector (complex numbers stored as pairs of floats)
layout(std430, binding = 0) buffer StateVector {
    vec2 state[];
};

// gate parameters
layout(std140, binding = 1) uniform GateParams {
    uint gate_type;
    uint target_qubit;
    uint num_qubits;
    uint _pad;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint state_size = 1u << params.num_qubits;

    // return if index is out of bounds
    if (gid >= state_size) {
        return;
    }

    // compute paired index (flipping the target qubit)
    uint paired_idx = gid ^ (1u << params.target_qubit);

    // only process the first index of each pair to avoid race conditions
    if (gid < paired_idx) {
        // get current amplitudes
        vec2 amp_0 = state[gid];
        vec2 amp_1 = state[paired_idx];

        // apply x gate (bit flip): swap amplitudes
        state[gid] = amp_1;
        state[paired_idx] = amp_0;
    }
}"#.to_string(),
        "pauliz" => r#"#version 450
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// state vector (complex numbers stored as pairs of floats)
layout(std430, binding = 0) buffer StateVector {
    vec2 state[];
};

// gate parameters
layout(std140, binding = 1) uniform GateParams {
    uint gate_type;
    uint target_qubit;
    uint num_qubits;
    uint _pad;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint state_size = 1u << params.num_qubits;

    // return if index is out of bounds
    if (gid >= state_size) {
        return;
    }

    // check if the target qubit is 1 in this index
    uint mask = 1u << params.target_qubit;
    if ((gid & mask) != 0) {
        // apply z gate: negate the amplitude
        state[gid] = vec2(-state[gid].x, -state[gid].y);
    }
}"#.to_string(),
        "cnot" => r#"#version 450
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// state vector (complex numbers stored as pairs of floats)
layout(std430, binding = 0) buffer StateVector {
    vec2 state[];
};

// gate parameters
layout(std140, binding = 1) uniform GateParams {
    // using target_qubit field for control
    uint control_qubit;
    // using num_qubits field for target
    uint target_qubit;
    // total qubits in separate field
    uint num_qubits;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint state_size = 1u << params.num_qubits;

    // return if index is out of bounds
    if (gid >= state_size) {
        return;
    }

    // check if control qubit is 1
    uint control_mask = 1u << params.control_qubit;
    if ((gid & control_mask) != 0) {
        // compute paired index (flipping the target qubit)
        uint target_mask = 1u << params.target_qubit;
        uint paired_idx = gid ^ target_mask;

        // only process the first index of each pair to avoid race conditions
        if (gid < paired_idx) {
            // get current amplitudes
            vec2 amp_0 = state[gid];
            vec2 amp_1 = state[paired_idx];

            // swap the amplitudes (x gate on target when control is 1)
            state[gid] = amp_1;
            state[paired_idx] = amp_0;
        }
    }
}"#.to_string(),
        "rx" => r#"#version 450
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// state vector (complex numbers stored as pairs of floats)
layout(std430, binding = 0) buffer StateVector {
    vec2 state[];
};

// gate parameters
layout(std140, binding = 1) uniform GateParams {
    uint gate_type;
    uint target_qubit;
    uint num_qubits;
    // using _pad field for the rotation angle
    float angle;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint state_size = 1u << params.num_qubits;

    // return if index is out of bounds
    if (gid >= state_size) {
        return;
    }

    // compute paired index (flipping the target qubit)
    uint paired_idx = gid ^ (1u << params.target_qubit);

    // only process the first index of each pair to avoid race conditions
    if (gid < paired_idx) {
        // get current amplitudes
        vec2 amp_0 = state[gid];
        vec2 amp_1 = state[paired_idx];

        // calculate rotation components
        float cos_half = cos(params.angle / 2.0);
        float sin_half = sin(params.angle / 2.0);

        // apply rx rotation: [cos(θ/2), -i*sin(θ/2); -i*sin(θ/2), cos(θ/2)]
        vec2 new_amp_0 = vec2(
            cos_half * amp_0.x - sin_half * amp_1.y,
            cos_half * amp_0.y + sin_half * amp_1.x
        );

        vec2 new_amp_1 = vec2(
            cos_half * amp_1.x - sin_half * amp_0.y,
            cos_half * amp_1.y + sin_half * amp_0.x
        );

        // write back results
        state[gid] = new_amp_0;
        state[paired_idx] = new_amp_1;
    }
}"#.to_string(),
        "ry" => r#"#version 450
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// state vector (complex numbers stored as pairs of floats)
layout(std430, binding = 0) buffer StateVector {
    vec2 state[];
};

// gate parameters
layout(std140, binding = 1) uniform GateParams {
    uint gate_type;
    uint target_qubit;
    uint num_qubits;
    // using _pad field for the rotation angle
    float angle;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint state_size = 1u << params.num_qubits;

    // return if index is out of bounds
    if (gid >= state_size) {
        return;
    }

    // compute paired index (flipping the target qubit)
    uint paired_idx = gid ^ (1u << params.target_qubit);

    // only process the first index of each pair to avoid race conditions
    if (gid < paired_idx) {
        // get current amplitudes
        vec2 amp_0 = state[gid];
        vec2 amp_1 = state[paired_idx];

        // calculate rotation components
        float cos_half = cos(params.angle / 2.0);
        float sin_half = sin(params.angle / 2.0);

        // apply ry rotation: [cos(θ/2), -sin(θ/2); sin(θ/2), cos(θ/2)]
        vec2 new_amp_0 = vec2(
            cos_half * amp_0.x - sin_half * amp_1.x,
            cos_half * amp_0.y - sin_half * amp_1.y
        );

        vec2 new_amp_1 = vec2(
            cos_half * amp_1.x + sin_half * amp_0.x,
            cos_half * amp_1.y + sin_half * amp_0.y
        );

        // write back results
        state[gid] = new_amp_0;
        state[paired_idx] = new_amp_1;
    }
}"#.to_string(),
        "rz" => r#"#version 450
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// state vector (complex numbers stored as pairs of floats)
layout(std430, binding = 0) buffer StateVector {
    vec2 state[];
};

// gate parameters
layout(std140, binding = 1) uniform GateParams {
    uint gate_type;
    uint target_qubit;
    uint num_qubits;
    // using _pad field for the rotation angle
    float angle;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint state_size = 1u << params.num_qubits;

    // return if index is out of bounds
    if (gid >= state_size) {
        return;
    }

    // check if the target qubit is 1 in this index
    uint mask = 1u << params.target_qubit;
    if ((gid & mask) != 0) {
        // apply rz rotation: phase shift the amplitude by e^(-iθ/2)
        // complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        // here, (c+di) = (cos(-θ/2) + i*sin(-θ/2)) = (cos(θ/2) - i*sin(θ/2))
        float cos_half = cos(-params.angle / 2.0);
        float sin_half = sin(-params.angle / 2.0);
        vec2 current_amp = state[gid];

        vec2 new_amp = vec2(
            current_amp.x * cos_half - current_amp.y * sin_half,
            current_amp.x * sin_half + current_amp.y * cos_half
        );

        // write back result
        state[gid] = new_amp;
    }
}"#.to_string(),
        "s" => r#"#version 450
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// state vector (complex numbers stored as pairs of floats)
layout(std430, binding = 0) buffer StateVector {
    vec2 state[];
};

// gate parameters
layout(std140, binding = 1) uniform GateParams {
    uint gate_type;
    uint target_qubit;
    uint num_qubits;
    uint _pad;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint state_size = 1u << params.num_qubits;

    // return if index is out of bounds
    if (gid >= state_size) {
        return;
    }

    // check if the target qubit is 1 in this index
    uint mask = 1u << params.target_qubit;
    if ((gid & mask) != 0) {
        // apply s gate: phase shift by i
        // complex multiplication: (a+bi)(i) = -b + ai
        vec2 current_amp = state[gid];
        state[gid] = vec2(-current_amp.y, current_amp.x);
    }
}"#.to_string(),
        "t" => r#"#version 450
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// state vector (complex numbers stored as pairs of floats)
layout(std430, binding = 0) buffer StateVector {
    vec2 state[];
};

// gate parameters
layout(std140, binding = 1) uniform GateParams {
    uint gate_type;
    uint target_qubit;
    uint num_qubits;
    uint _pad;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint state_size = 1u << params.num_qubits;

    // return if index is out of bounds
    if (gid >= state_size) {
        return;
    }

    // check if the target qubit is 1 in this index
    uint mask = 1u << params.target_qubit;
    if ((gid & mask) != 0) {
        // apply t gate: phase shift by e^(iπ/4)
        // complex multiplication: (a+bi)(cos(π/4) + i*sin(π/4)) = (a+bi)(1/sqrt(2) + i/sqrt(2))
        const float inv_sqrt_2 = 0.7071067811865475;
        vec2 current_amp = state[gid];

        vec2 new_amp = vec2(
            inv_sqrt_2 * (current_amp.x - current_amp.y),
            inv_sqrt_2 * (current_amp.x + current_amp.y)
        );

        state[gid] = new_amp;
    }
}"#.to_string(),
        "swap" => r#"#version 450
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// state vector (complex numbers stored as pairs of floats)
layout(std430, binding = 0) buffer StateVector {
    vec2 state[];
};

// gate parameters
layout(std140, binding = 1) uniform GateParams {
    // using target_qubit for the first qubit
    uint target_qubit;
    // using num_qubits for the second qubit
    uint second_qubit;
    // total qubits in separate field
    uint num_qubits;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint state_size = 1u << params.num_qubits;

    // return if index is out of bounds
    if (gid >= state_size) {
        return;
    }

    // calculate bitmasks for the qubits
    uint q0_mask = 1u << params.target_qubit;
    uint q1_mask = 1u << params.second_qubit;

    // check the state of the two qubits
    bool q0_state = (gid & q0_mask) != 0;
    bool q1_state = (gid & q1_mask) != 0;

    // we only need to swap if one qubit is 0 and the other is 1
    if (q0_state != q1_state) {
        // compute the swapped index
        uint swapped_idx = (gid & ~q0_mask & ~q1_mask) | (q1_state ? q0_mask : 0u) | (q0_state ? q1_mask : 0u);

        // only process the smaller index to prevent race conditions
        if (gid < swapped_idx) {
            vec2 amp_0 = state[gid];
            vec2 amp_1 = state[swapped_idx];

            state[gid] = amp_1;
            state[swapped_idx] = amp_0;
        }
    }
}"#.to_string(),
        _ => format!(r#"#version 450
// default stub shader for gate type: {}
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer StateVector {{
    vec2 state[];
}};

layout(std140, binding = 1) uniform GateParams {{
    uint gate_type;
    uint target_qubit;
    uint num_qubits;
    uint _pad;
}} params;

void main() {{
    // this is a stub shader that does nothing
    // it needs to be properly implemented for this gate type
    uint gid = gl_GlobalInvocationID.x;
    uint state_size = 1u << params.num_qubits;

    if (gid >= state_size) {{
        return;
    }}
}}
"#, gate_type)
    }
}
