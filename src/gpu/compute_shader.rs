// include the auto-generated shader module
#[cfg(feature = "vulkan")]
include!(concat!(env!("OUT_DIR"), "/shaders.rs"));

// for backwards compatibility with the old implementation
#[cfg(feature = "vulkan")]
pub const COMPUTE_SHADER_SPIRV_HADAMARD: &[u32] = &[];
#[cfg(feature = "vulkan")]
pub const COMPUTE_SHADER_SPIRV_X: &[u32] = &[];
#[cfg(feature = "vulkan")]
pub const COMPUTE_SHADER_SPIRV_Z: &[u32] = &[];
#[cfg(feature = "vulkan")]
pub const COMPUTE_SHADER_SPIRV_CNOT: &[u32] = &[];

// get spir-v binary for a specific quantum gate
#[cfg(feature = "vulkan")]
pub fn get_vulkan_shader_for_gate(gate_type: &str) -> Result<Vec<u32>, String> {
    match gate_type {
        "h" | "hadamard" => Ok(bytes_to_words(SPIRV_HADAMARD)),
        "x" | "not" | "paulix" => Ok(bytes_to_words(SPIRV_PAULIX)),
        "z" | "pauliz" => Ok(bytes_to_words(SPIRV_PAULIZ)),
        "cnot" => Ok(bytes_to_words(SPIRV_CNOT)),
        "rx" => Ok(bytes_to_words(SPIRV_RX)),
        "ry" => Ok(bytes_to_words(SPIRV_RY)),
        "rz" => Ok(bytes_to_words(SPIRV_RZ)),
        "s" => Ok(bytes_to_words(SPIRV_S)),
        "t" => Ok(bytes_to_words(SPIRV_T)),
        "swap" => Ok(bytes_to_words(SPIRV_SWAP)),
        _ => Err(format!("no pre-compiled vulkan shader available for gate type: {}", gate_type))
    }
}

// generate opencl kernel source for a specific quantum gate
#[cfg(feature = "opencl")]
pub fn get_opencl_kernel_source(gate_type: &str) -> Result<String, String> {
    // this function would provide opencl kernel source
    // for now just returning a placeholder
    Err(format!("opencl kernel source not implemented for gate type: {}", gate_type))
}
