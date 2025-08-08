use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::collections::{HashMap, HashSet};
use log::{debug, error, info, warn};
use num_complex::Complex64;
use parking_lot::RwLock as PLRwLock;
use rayon::prelude::*;
use std::cell::RefCell;
use once_cell::sync::Lazy;
use std::convert::TryInto;

// Core shared GPU constants
pub const MAX_BATCH_SIZE: usize = 1024;
pub const MIN_WORK_SIZE_FOR_GPU: usize = 1024;

// Global GPU accelerator instance with lazy initialization
static GLOBAL_GPU_ACCELERATOR: Lazy<Arc<GpuAccelerator>> = Lazy::new(|| {
    Arc::new(GpuAccelerator::new())
});

// ===== GPU Acceleration Capability Detection =====

// Information about a detected GPU device
#[derive(Clone, Debug)]
pub struct GpuDevice {
    pub id: usize,
    pub name: String,
    pub memory_mb: usize,
    pub compute_units: usize,
    pub backend: GpuBackend,
    pub score: f32,  // Performance score (higher is better)
    pub is_available: bool,
    pub features: HashSet<String>,
    pub last_benchmark: Option<f64>, // Time in seconds
    pub max_workgroup_size: usize,
}

// Available GPU backends
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    Vulkan,
    Cuda,
    OpenCL,
    Metal,
    None
}

// Main GPU acceleration interface
pub struct GpuAccelerator {
    devices: PLRwLock<Vec<GpuDevice>>,
    preferred_backends: RwLock<Vec<GpuBackend>>,
    performance_cache: RwLock<HashMap<String, HashMap<GpuBackend, f64>>>,
    initialized: Mutex<bool>,
    
    // Contexts for each backend
    #[cfg(feature = "vulkan")]
    vulkan_contexts: RwLock<HashMap<usize, Arc<VulkanContext>>>,
    
    #[cfg(feature = "cuda")]
    cuda_contexts: RwLock<HashMap<usize, Arc<CudaContext>>>,
    
    #[cfg(feature = "opencl")]
    opencl_contexts: RwLock<HashMap<usize, Arc<OpenCLContext>>>,
}

impl GpuAccelerator {
    pub fn new() -> Self {
        let backends = Self::detect_preferred_backend_order();
        
        Self {
            devices: PLRwLock::new(Vec::new()),
            preferred_backends: RwLock::new(backends),
            performance_cache: RwLock::new(HashMap::new()),
            initialized: Mutex::new(false),
            
            #[cfg(feature = "vulkan")]
            vulkan_contexts: RwLock::new(HashMap::new()),
            
            #[cfg(feature = "cuda")]
            cuda_contexts: RwLock::new(HashMap::new()),
            
            #[cfg(feature = "opencl")]
            opencl_contexts: RwLock::new(HashMap::new()),
        }
    }
    
    // Get the global GPU accelerator instance
    pub fn global() -> Arc<Self> {
        GLOBAL_GPU_ACCELERATOR.clone()
    }
    
    // Initialize GPU backends and devices
    pub fn initialize(&self) -> Result<(), String> {
        let mut initialized = self.initialized.lock().unwrap();
        if *initialized {
            return Ok(());
        }
        
        info!("Initializing GPU acceleration backends");
        let mut devices = Vec::new();
        let mut device_id = 0;
        
        // Try initializing Vulkan first
        #[cfg(feature = "vulkan")]
        {
            match self.initialize_vulkan_devices() {
                Ok(vulkan_devices) => {
                    info!("Detected {} Vulkan-compatible devices", vulkan_devices.len());
                    for mut device in vulkan_devices {
                        device.id = device_id;
                        devices.push(device);
                        device_id += 1;
                    }
                }
                Err(e) => {
                    warn!("Failed to initialize Vulkan: {}", e);
                }
            }
        }
        
        // Then try CUDA
        #[cfg(feature = "cuda")]
        {
            match self.initialize_cuda_devices() {
                Ok(cuda_devices) => {
                    info!("Detected {} CUDA-compatible devices", cuda_devices.len());
                    for mut device in cuda_devices {
                        device.id = device_id;
                        devices.push(device);
                        device_id += 1;
                    }
                }
                Err(e) => {
                    warn!("Failed to initialize CUDA: {}", e);
                }
            }
        }
        
        // Then OpenCL
        #[cfg(feature = "opencl")]
        {
            match self.initialize_opencl_devices() {
                Ok(opencl_devices) => {
                    info!("Detected {} OpenCL-compatible devices", opencl_devices.len());
                    for mut device in opencl_devices {
                        device.id = device_id;
                        devices.push(device);
                        device_id += 1;
                    }
                }
                Err(e) => {
                    warn!("Failed to initialize OpenCL: {}", e);
                }
            }
        }
        
        // Sort devices by score (highest first)
        devices.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        if devices.is_empty() {
            warn!("No GPU acceleration devices detected, CPU fallback will be used");
        } else {
            for device in &devices {
                info!("GPU device: {} ({}), Memory: {}MB, Score: {}, Backend: {:?}", 
                     device.id, device.name, device.memory_mb, device.score, device.backend);
            }
        }
        
        *self.devices.write() = devices;
        *initialized = true;
        
        Ok(())
    }
    
    // Detect optimal backend order based on system capabilities
    fn detect_preferred_backend_order() -> Vec<GpuBackend> {
        let mut backends = Vec::new();
        
        // Check for CUDA first as it generally has the best performance
        #[cfg(feature = "cuda")]
        backends.push(GpuBackend::Cuda);
        
        // Vulkan next, as it's widely supported
        #[cfg(feature = "vulkan")]
        backends.push(GpuBackend::Vulkan);
        
        // OpenCL as a fallback
        #[cfg(feature = "opencl")]
        backends.push(GpuBackend::OpenCL);
        
        // Metal for macOS systems
        #[cfg(feature = "metal")]
        backends.push(GpuBackend::Metal);
        
        // Add CPU fallback
        backends.push(GpuBackend::None);
        
        backends
    }
    
    // Initialize Vulkan devices
    #[cfg(feature = "vulkan")]
    fn initialize_vulkan_devices(&self) -> Result<Vec<GpuDevice>, String> {
        use ash::{Entry, Instance, vk};
        use std::ffi::CStr;
        
        let mut devices = Vec::new();
        let mut contexts = self.vulkan_contexts.write().unwrap();
        
        // Create Vulkan instance
        let entry = match Entry::load() {
            Ok(e) => e,
            Err(e) => return Err(format!("Failed to load Vulkan entry: {}", e))
        };
        
        // Application info
        let app_info = vk::ApplicationInfo::builder()
            .application_name(unsafe { CStr::from_bytes_with_nul_unchecked(b"QOA\0") })
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(unsafe { CStr::from_bytes_with_nul_unchecked(b"QOA Engine\0") })
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0));
            
        // Instance create info
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info);
            
        // Create instance
        let instance = match unsafe { entry.create_instance(&create_info, None) } {
            Ok(i) => i,
            Err(e) => return Err(format!("Failed to create Vulkan instance: {}", e))
        };
        
        // Enumerate physical devices
        let physical_devices = match unsafe { instance.enumerate_physical_devices() } {
            Ok(devices) => devices,
            Err(e) => return Err(format!("Failed to enumerate Vulkan physical devices: {}", e))
        };
        
        for (idx, &physical_device) in physical_devices.iter().enumerate() {
            // Get device properties
            let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
            let device_name = unsafe {
                CStr::from_ptr(device_properties.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };
            
            // Get memory properties
            let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
            let mut total_memory = 0;
            for heap in &memory_properties.memory_heaps[..memory_properties.memory_heap_count as usize] {
                if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                    total_memory += heap.size;
                }
            }
            
            // Check if device is suitable for compute
            let queue_family_props = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let compute_queue_idx = queue_family_props.iter().enumerate()
                .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(idx, _)| idx as u32);
                
            if let Some(queue_idx) = compute_queue_idx {
                // Create logical device for this physical device
                let device_create_info = vk::DeviceCreateInfo::builder()
                    .queue_create_infos(&[
                        vk::DeviceQueueCreateInfo::builder()
                            .queue_family_index(queue_idx)
                            .queue_priorities(&[1.0])
                            .build()
                    ]);
                    
                let logical_device = match unsafe { instance.create_device(physical_device, &device_create_info, None) } {
                    Ok(device) => device,
                    Err(e) => {
                        warn!("Failed to create Vulkan logical device for {}: {}", device_name, e);
                        continue;
                    }
                };
                
                // Create command pool
                let command_pool = match unsafe {
                    logical_device.create_command_pool(
                        &vk::CommandPoolCreateInfo::builder()
                            .queue_family_index(queue_idx)
                            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None
                    )
                } {
                    Ok(pool) => pool,
                    Err(e) => {
                        warn!("Failed to create Vulkan command pool for {}: {}", device_name, e);
                        unsafe { logical_device.destroy_device(None) };
                        continue;
                    }
                };
                
                // Create a VulkanContext for this device
                let context = Arc::new(VulkanContext {
                    entry: entry.clone(),
                    instance: instance.clone(),
                    physical_device,
                    device: logical_device,
                    queue_family_index: queue_idx,
                    compute_queue: unsafe { logical_device.get_device_queue(queue_idx, 0) },
                    command_pool,
                    descriptor_pool: vk::DescriptorPool::null(),
                    descriptor_set_layout: vk::DescriptorSetLayout::null(),
                    pipeline_layout: vk::PipelineLayout::null(),
                    compute_pipeline: vk::Pipeline::null(),
                    #[cfg(debug_assertions)]
                    debug_utils_loader: None,
                    #[cfg(debug_assertions)]
                    debug_messenger: None,
                });
                
                // Calculate a performance score based on device properties
                // This is a simplistic heuristic and could be improved
                let score = match device_properties.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 1000.0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 500.0,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 300.0,
                    vk::PhysicalDeviceType::CPU => 100.0,
                    _ => 50.0,
                } + (total_memory / (1024 * 1024)) as f32 * 0.01;
                
                // Store the context
                contexts.insert(idx, context);
                
                // Create device info
                let device_info = GpuDevice {
                    id: idx,
                    name: device_name,
                    memory_mb: (total_memory / (1024 * 1024)) as usize,
                    compute_units: device_properties.limits.max_compute_work_group_count[0] as usize,
                    backend: GpuBackend::Vulkan,
                    score,
                    is_available: true,
                    features: HashSet::new(),
                    last_benchmark: None,
                    max_workgroup_size: device_properties.limits.max_compute_work_group_invocations as usize,
                };
                
                devices.push(device_info);
            }
        }
        
        Ok(devices)
    }
    
    // Initialize CUDA devices
    #[cfg(feature = "cuda")]
    fn initialize_cuda_devices(&self) -> Result<Vec<GpuDevice>, String> {
        let mut devices = Vec::new();
        let mut contexts = self.cuda_contexts.write().unwrap();
        
        match rust_cuda::device_count() {
            Ok(count) => {
                for device_id in 0..count {
                    if let Ok(device) = rust_cuda::Device::new(device_id) {
                        if let Ok(props) = device.get_properties() {
                            // Calculate score based on device capabilities
                            let score = 1000.0 + (props.total_memory() / (1024 * 1024)) as f32 * 0.01 
                                      + props.multi_processor_count() as f32 * 5.0;
                            
                            // Create and store a CUDA context
                            if let Ok(context) = CudaContext::new(device) {
                                let context = Arc::new(context);
                                contexts.insert(device_id as usize, context);
                                
                                let mut features = HashSet::new();
                                if props.compute_capability().0 >= 6 {
                                    features.insert("tensor_cores".to_string());
                                }
                                if props.compute_capability().0 >= 7 {
                                    features.insert("fp16".to_string());
                                }
                                
                                devices.push(GpuDevice {
                                    id: device_id as usize,
                                    name: props.name().to_string(),
                                    memory_mb: props.total_memory() / (1024 * 1024) as usize,
                                    compute_units: props.multi_processor_count() as usize,
                                    backend: GpuBackend::Cuda,
                                    score,
                                    is_available: true,
                                    features,
                                    last_benchmark: None,
                                    max_workgroup_size: props.max_threads_per_block() as usize,
                                });
                            }
                        }
                    }
                }
                Ok(devices)
            },
            Err(e) => Err(format!("Failed to get CUDA device count: {}", e))
        }
    }
    
    // Initialize OpenCL devices
    #[cfg(feature = "opencl")]
    fn initialize_opencl_devices(&self) -> Result<Vec<GpuDevice>, String> {
        use opencl3::platform::{get_platforms, Platform};
        use opencl3::device::{Device, CL_DEVICE_TYPE_ALL, CL_DEVICE_NAME, CL_DEVICE_GLOBAL_MEM_SIZE, CL_DEVICE_MAX_COMPUTE_UNITS, CL_DEVICE_MAX_WORK_GROUP_SIZE};
        use opencl3::context::Context;
        use opencl3::command_queue::CommandQueue;
        
        let mut devices = Vec::new();
        let mut contexts = self.opencl_contexts.write().unwrap();
        
        match get_platforms() {
            Ok(platforms) => {
                for (platform_idx, platform) in platforms.iter().enumerate() {
                    if let Ok(platform_name) = platform.name() {
                        if let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_ALL) {
                            for (device_idx, &device_id) in device_ids.iter().enumerate() {
                                let device = Device::new(device_id);
                                
                                // Get device info
                                let name = match device.info(CL_DEVICE_NAME) {
                                    Ok(name) => String::from_utf8_lossy(&name).to_string(),
                                    Err(_) => format!("OpenCL Device {}.{}", platform_idx, device_idx)
                                };
                                
                                let memory = match device.info(CL_DEVICE_GLOBAL_MEM_SIZE) {
                                    Ok(mem) => {
                                        if mem.len() >= 8 {
                                            let size = unsafe { *(mem.as_ptr() as *const u64) };
                                            size / (1024 * 1024)
                                        } else {
                                            0
                                        }
                                    },
                                    Err(_) => 0
                                };
                                
                                let compute_units = match device.info(CL_DEVICE_MAX_COMPUTE_UNITS) {
                                    Ok(cu) => {
                                        if cu.len() >= 4 {
                                            unsafe { *(cu.as_ptr() as *const u32) as usize }
                                        } else {
                                            0
                                        }
                                    },
                                    Err(_) => 0
                                };
                                
                                let max_workgroup_size = match device.info(CL_DEVICE_MAX_WORK_GROUP_SIZE) {
                                    Ok(size) => {
                                        if size.len() >= std::mem::size_of::<usize>() {
                                            unsafe { *(size.as_ptr() as *const usize) }
                                        } else {
                                            256 // Default conservative value
                                        }
                                    },
                                    Err(_) => 256
                                };
                                
                                // Create OpenCL context and command queue
                                if let Ok(context) = Context::from_device(&device) {
                                    if let Ok(queue) = CommandQueue::create_with_properties(&context, device_id, 0, 0) {
                                        // Create and store an OpenCL context
                                        let cl_context = Arc::new(OpenCLContext {
                                            platform: platform.clone(),
                                            device,
                                            context,
                                            queue,
                                            compiled_kernels: RwLock::new(HashMap::new()),
                                        });
                                        
                                        let global_id = platform_idx * 100 + device_idx;
                                        contexts.insert(global_id, cl_context);
                                        
                                        // Calculate a performance score
                                        // This is a simplistic heuristic and could be improved
                                        let is_gpu = name.to_lowercase().contains("gpu");
                                        let base_score = if is_gpu { 700.0 } else { 200.0 };
                                        let score = base_score + memory as f32 * 0.01 + compute_units as f32 * 2.0;
                                        
                                        devices.push(GpuDevice {
                                            id: global_id,
                                            name,
                                            memory_mb: memory as usize,
                                            compute_units,
                                            backend: GpuBackend::OpenCL,
                                            score,
                                            is_available: true,
                                            features: HashSet::new(),
                                            last_benchmark: None,
                                            max_workgroup_size,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(devices)
            },
            Err(e) => Err(format!("Failed to get OpenCL platforms: {}", e))
        }
    }
    
    // Get a list of available devices
    pub fn get_devices(&self) -> Vec<GpuDevice> {
        self.devices.read().clone()
    }
    
    // Find the best device for a specific operation
    pub fn find_best_device(&self, operation: &str, data_size: usize) -> Option<GpuDevice> {
        // Ensure we're initialized
        if !(*self.initialized.lock().unwrap()) {
            let _ = self.initialize();
        }
        
        let devices = self.devices.read();
        if devices.is_empty() {
            return None;
        }
        
        // Check performance cache first
        let performance_cache = self.performance_cache.read().unwrap();
        if let Some(op_cache) = performance_cache.get(operation) {
            // Find the best backend based on cached performance
            let mut best_score = f64::MAX;
            let mut best_backend = None;
            
            for (backend, &time) in op_cache {
                if time < best_score {
                    best_score = time;
                    best_backend = Some(backend);
                }
            }
            
            // Find the best device with that backend
            if let Some(best_backend) = best_backend {
                return devices.iter()
                    .filter(|d| &d.backend == best_backend && d.is_available)
                    .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
                    .cloned();
            }
        }
        
        // If we don't have performance data, use the preferred order
        let preferred_backends = self.preferred_backends.read().unwrap();
        for backend in preferred_backends.iter() {
            if let Some(device) = devices.iter()
                .filter(|d| &d.backend == backend && d.is_available)
                .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)) {
                
                return Some(device.clone());
            }
        }
        
        None
    }
    
    // Update performance metrics after an operation
    pub fn update_performance(&self, operation: &str, backend: GpuBackend, execution_time: f64) {
        let mut cache = self.performance_cache.write().unwrap();
        let op_cache = cache.entry(operation.to_string()).or_insert_with(HashMap::new);
        
        // Use exponential moving average to update performance metric
        if let Some(current_time) = op_cache.get(&backend) {
            let alpha = 0.2; // Weight for new measurement
            let updated_time = (1.0 - alpha) * current_time + alpha * execution_time;
            op_cache.insert(backend, updated_time);
        } else {
            op_cache.insert(backend, execution_time);
        }
    }
    
    // Execute a quantum gate operation on the best available device
    pub fn execute_gate_operation(
        &self,
        gate_type: &str,
        amplitudes: &mut [Complex64],
        qubit_indices: &[usize],
        params: &[f64]
    ) -> Result<(), String> {
        // Ensure we're initialized
        if !(*self.initialized.lock().unwrap()) {
            let _ = self.initialize();
        }
        
        let operation_key = format!("gate_{}", gate_type);
        let data_size = amplitudes.len();
        
        // Skip GPU if the data is too small
        if data_size < MIN_WORK_SIZE_FOR_GPU {
            return self.execute_gate_operation_cpu(gate_type, amplitudes, qubit_indices, params);
        }
        
        // Find the best device for this operation
        if let Some(device) = self.find_best_device(&operation_key, data_size) {
            let start_time = Instant::now();
            let result = match device.backend {
                GpuBackend::Vulkan => {
                    #[cfg(feature = "vulkan")]
                    {
                        let contexts = self.vulkan_contexts.read().unwrap();
                        if let Some(context) = contexts.get(&device.id) {
                            self.execute_gate_operation_vulkan(context, gate_type, amplitudes, qubit_indices, params)
                        } else {
                            Err("Vulkan context not available".to_string())
                        }
                    }
                    #[cfg(not(feature = "vulkan"))]
                    {
                        Err("Vulkan support not compiled in".to_string())
                    }
                },
                GpuBackend::Cuda => {
                    #[cfg(feature = "cuda")]
                    {
                        let contexts = self.cuda_contexts.read().unwrap();
                        if let Some(context) = contexts.get(&device.id) {
                            self.execute_gate_operation_cuda(context, gate_type, amplitudes, qubit_indices, params)
                        } else {
                            Err("CUDA context not available".to_string())
                        }
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        Err("CUDA support not compiled in".to_string())
                    }
                },
                GpuBackend::OpenCL => {
                    #[cfg(feature = "opencl")]
                    {
                        let contexts = self.opencl_contexts.read().unwrap();
                        if let Some(context) = contexts.get(&device.id) {
                            self.execute_gate_operation_opencl(context, gate_type, amplitudes, qubit_indices, params)
                        } else {
                            Err("OpenCL context not available".to_string())
                        }
                    }
                    #[cfg(not(feature = "opencl"))]
                    {
                        Err("OpenCL support not compiled in".to_string())
                    }
                },
                _ => Err("Unsupported backend".to_string())
            };
            
            // Update performance metrics on success
            if result.is_ok() {
                let execution_time = start_time.elapsed().as_secs_f64();
                self.update_performance(&operation_key, device.backend, execution_time);
                return Ok(());
            }
            
            // If the preferred backend failed, try CPU fallback
            warn!("GPU execution failed for {}: {}", operation_key, result.unwrap_err());
        }
        
        // Fallback to CPU implementation
        debug!("Using CPU fallback for gate operation {}", gate_type);
        self.execute_gate_operation_cpu(gate_type, amplitudes, qubit_indices, params)
    }
    
    // Execute a gate operation on CPU
    fn execute_gate_operation_cpu(
        &self,
        gate_type: &str,
        amplitudes: &mut [Complex64],
        qubit_indices: &[usize],
        params: &[f64]
    ) -> Result<(), String> {
        // CPU implementation of quantum gates
        match gate_type {
            "h" | "hadamard" => apply_hadamard_cpu(amplitudes, qubit_indices[0]),
            "x" | "not" => apply_x_gate_cpu(amplitudes, qubit_indices[0]),
            "y" => apply_y_gate_cpu(amplitudes, qubit_indices[0]),
            "z" => apply_z_gate_cpu(amplitudes, qubit_indices[0]),
            "rx" => {
                if params.is_empty() {
                    return Err("rx gate requires an angle parameter".to_string());
                }
                apply_rx_gate_cpu(amplitudes, qubit_indices[0], params[0])
            },
            "ry" => {
                if params.is_empty() {
                    return Err("ry gate requires an angle parameter".to_string());
                }
                apply_ry_gate_cpu(amplitudes, qubit_indices[0], params[0])
            },
            "rz" => {
                if params.is_empty() {
                    return Err("rz gate requires an angle parameter".to_string());
                }
                apply_rz_gate_cpu(amplitudes, qubit_indices[0], params[0])
            },
            "cnot" => {
                if qubit_indices.len() < 2 {
                    return Err("cnot gate requires control and target qubits".to_string());
                }
                apply_cnot_gate_cpu(amplitudes, qubit_indices[0], qubit_indices[1])
            },
            "cz" => {
                if qubit_indices.len() < 2 {
                    return Err("cz gate requires control and target qubits".to_string());
                }
                apply_cz_gate_cpu(amplitudes, qubit_indices[0], qubit_indices[1])
            },
            // Add more gates as needed
            _ => Err(format!("Unsupported gate type: {}", gate_type))
        }
    }
    
    // Execute a gate operation using Vulkan
    #[cfg(feature = "vulkan")]
    fn execute_gate_operation_vulkan(
        &self,
        context: &Arc<VulkanContext>,
        gate_type: &str,
        amplitudes: &mut [Complex64],
        qubit_indices: &[usize],
        params: &[f64]
    ) -> Result<(), String> {
        use std::mem::size_of;
        use ash::vk;
        
        let device = &context.device;
        
        // Prepare gate matrix based on gate type
        let gate_matrix = get_gate_matrix(gate_type, params)?;
        let gate_data_size = gate_matrix.len() * size_of::<Complex64>();
        
        // Create buffers
        let buffer_info = vk::BufferCreateInfo::builder()
            .size((amplitudes.len() * size_of::<Complex64>()) as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        
        let state_buffer = unsafe { device.create_buffer(&buffer_info, None) }
            .map_err(|e| format!("Failed to create state buffer: {}", e))?;
        
        // Create gate matrix buffer
        let gate_buffer_info = vk::BufferCreateInfo::builder()
            .size(gate_data_size as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        
        let gate_buffer = unsafe { device.create_buffer(&gate_buffer_info, None) }
            .map_err(|e| format!("Failed to create gate buffer: {}", e))?;
        
        // Create qubit indices buffer
        let indices_buffer_info = vk::BufferCreateInfo::builder()
            .size((qubit_indices.len() * size_of::<u32>()) as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        
        let indices_buffer = unsafe { device.create_buffer(&indices_buffer_info, None) }
            .map_err(|e| format!("Failed to create indices buffer: {}", e))?;
        
        // Get memory requirements and allocate memory
        let state_mem_req = unsafe { device.get_buffer_memory_requirements(state_buffer) };
        let gate_mem_req = unsafe { device.get_buffer_memory_requirements(gate_buffer) };
        let indices_mem_req = unsafe { device.get_buffer_memory_requirements(indices_buffer) };
        
        // Allocate memory for state buffer
        let mem_properties = unsafe { context.instance.get_physical_device_memory_properties(context.physical_device) };
        
        let state_mem_type_idx = find_memory_type_index(
            &mem_properties, 
            state_mem_req.memory_type_bits, 
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        ).ok_or("Failed to find suitable memory type for state buffer")?;
        
        let state_alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(state_mem_req.size)
            .memory_type_index(state_mem_type_idx);
        
        let state_memory = unsafe { device.allocate_memory(&state_alloc_info, None) }
            .map_err(|e| format!("Failed to allocate state memory: {}", e))?;
        
        // Bind state memory
        unsafe { device.bind_buffer_memory(state_buffer, state_memory, 0) }
            .map_err(|e| format!("Failed to bind state memory: {}", e))?;
        
        // Allocate memory for gate buffer
        let gate_mem_type_idx = find_memory_type_index(
            &mem_properties, 
            gate_mem_req.memory_type_bits, 
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        ).ok_or("Failed to find suitable memory type for gate buffer")?;
        
        let gate_alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(gate_mem_req.size)
            .memory_type_index(gate_mem_type_idx);
        
        let gate_memory = unsafe { device.allocate_memory(&gate_alloc_info, None) }
            .map_err(|e| format!("Failed to allocate gate memory: {}", e))?;
        
        // Bind gate memory
        unsafe { device.bind_buffer_memory(gate_buffer, gate_memory, 0) }
            .map_err(|e| format!("Failed to bind gate memory: {}", e))?;
        
        // Allocate memory for indices buffer
        let indices_mem_type_idx = find_memory_type_index(
            &mem_properties, 
            indices_mem_req.memory_type_bits, 
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        ).ok_or("Failed to find suitable memory type for indices buffer")?;
        
        let indices_alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(indices_mem_req.size)
            .memory_type_index(indices_mem_type_idx);
        
        let indices_memory = unsafe { device.allocate_memory(&indices_alloc_info, None) }
            .map_err(|e| format!("Failed to allocate indices memory: {}", e))?;
        
        // Bind indices memory
        unsafe { device.bind_buffer_memory(indices_buffer, indices_memory, 0) }
            .map_err(|e| format!("Failed to bind indices memory: {}", e))?;
        
        // Map and copy state data
        unsafe {
            let state_ptr = device.map_memory(
                    state_memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty()
                )
                .map_err(|e| format!("Failed to map state memory: {}", e))?;
            
            std::ptr::copy_nonoverlapping(
                amplitudes.as_ptr() as *const u8,
                state_ptr as *mut u8,
                amplitudes.len() * size_of::<Complex64>()
            );
            
            device.unmap_memory(state_memory);
            
            // Map and copy gate data
            let gate_ptr = device.map_memory(
                    gate_memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty()
                )
                .map_err(|e| format!("Failed to map gate memory: {}", e))?;
            
            std::ptr::copy_nonoverlapping(
                gate_matrix.as_ptr() as *const u8,
                gate_ptr as *mut u8,
                gate_data_size
            );
            
            device.unmap_memory(gate_memory);
            
            // Map and copy indices data
            let indices_ptr = device.map_memory(
                    indices_memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty()
                )
                .map_err(|e| format!("Failed to map indices memory: {}", e))?;
            
            // Convert usize indices to u32
            let qubit_indices_u32: Vec<u32> = qubit_indices.iter().map(|&idx| idx as u32).collect();
            
            std::ptr::copy_nonoverlapping(
                qubit_indices_u32.as_ptr() as *const u8,
                indices_ptr as *mut u8,
                qubit_indices.len() * size_of::<u32>()
            );
            
            device.unmap_memory(indices_memory);
        }
        
        // Create descriptor set layout
        let descriptor_set_layout_bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
        ];
        
        let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&descriptor_set_layout_bindings);
        
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_info, None) }
            .map_err(|e| format!("Failed to create descriptor set layout: {}", e))?;
        
        // Create descriptor pool
        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3)
                .build()
        ];
        
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(1);
        
        let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }
            .map_err(|e| format!("Failed to create descriptor pool: {}", e))?;
        
        // Allocate descriptor set
        let descriptor_set_alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&[descriptor_set_layout]);
        
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&descriptor_set_alloc_info) }
            .map_err(|e| format!("Failed to allocate descriptor sets: {}", e))?;
        
        let descriptor_set = descriptor_sets[0];
        
        // Update descriptor set
        let state_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(state_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        
        let gate_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(gate_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        
        let indices_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(indices_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        
        let descriptor_writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[state_buffer_info.build()])
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[gate_buffer_info.build()])
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[indices_buffer_info.build()])
                .build(),
        ];
        
        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
        
        // Create pipeline layout
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&[descriptor_set_layout]);
        
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
            .map_err(|e| format!("Failed to create pipeline layout: {}", e))?;
        
        // Create compute shader module
        // We select the appropriate shader based on the gate type
        let shader_code = get_vulkan_shader_for_gate(gate_type)?;
        
        let shader_module_info = vk::ShaderModuleCreateInfo::builder()
            .code(&shader_code);
        
        let shader_module = unsafe { device.create_shader_module(&shader_module_info, None) }
            .map_err(|e| format!("Failed to create shader module: {}", e))?;
        
        // Create compute pipeline
        let shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0") });
        
        let compute_pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(shader_stage_info.build())
            .layout(pipeline_layout);
        
        let compute_pipeline = unsafe { 
            device.create_compute_pipelines(vk::PipelineCache::null(), &[compute_pipeline_info.build()], None)
        }
        .map_err(|e| format!("Failed to create compute pipeline: {}", e))?[0];
        
        // Create command buffer
        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(context.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        
        let command_buffers = unsafe { device.allocate_command_buffers(&command_buffer_alloc_info) }
            .map_err(|e| format!("Failed to allocate command buffers: {}", e))?;
        
        let command_buffer = command_buffers[0];
        
        // Begin command buffer
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        
        unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }
            .map_err(|e| format!("Failed to begin command buffer: {}", e))?;
        
        // Bind pipeline and descriptor set
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, compute_pipeline);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[]
            );
            
            // Calculate workgroup size
            let workgroup_size = 256;
            let num_workgroups = (amplitudes.len() + workgroup_size - 1) / workgroup_size;
            
            // Dispatch compute shader
            device.cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);
            
            // End command buffer
            device.end_command_buffer(command_buffer)
                .map_err(|e| format!("Failed to end command buffer: {}", e))?;
            
            // Submit command buffer
            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(&[command_buffer]);
            
            let fence_info = vk::FenceCreateInfo::builder();
            let fence = device.create_fence(&fence_info, None)
                .map_err(|e| format!("Failed to create fence: {}", e))?;
            
            device.queue_submit(context.compute_queue, &[submit_info.build()], fence)
                .map_err(|e| format!("Failed to submit queue: {}", e))?;
            
            // Wait for computation to complete
            device.wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| format!("Failed to wait for fence: {}", e))?;
            
            // Map and read back the results
            let state_ptr = device.map_memory(
                    state_memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty()
                )
                .map_err(|e| format!("Failed to map state memory for reading: {}", e))?;
            
            std::ptr::copy_nonoverlapping(
                state_ptr as *const u8,
                amplitudes.as_mut_ptr() as *mut u8,
                amplitudes.len() * size_of::<Complex64>()
            );
            
            device.unmap_memory(state_memory);
            
            // Cleanup
            device.destroy_fence(fence, None);
            device.destroy_shader_module(shader_module, None);
            device.destroy_pipeline(compute_pipeline, None);
            device.destroy_pipeline_layout(pipeline_layout, None);
            device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            device.destroy_descriptor_pool(descriptor_pool, None);
            device.free_memory(indices_memory, None);
            device.free_memory(gate_memory, None);
            device.free_memory(state_memory, None);
            device.destroy_buffer(indices_buffer, None);
            device.destroy_buffer(gate_buffer, None);
            device.destroy_buffer(state_buffer, None);
        }
        
        Ok(())
    }
    
    // Execute a gate operation using CUDA
    #[cfg(feature = "cuda")]
    fn execute_gate_operation_cuda(
        &self,
        context: &Arc<CudaContext>,
        gate_type: &str,
        amplitudes: &mut [Complex64],
        qubit_indices: &[usize],
        params: &[f64]
    ) -> Result<(), String> {
        use std::mem::size_of;
        
        // Prepare gate matrix based on gate type
        let gate_matrix = get_gate_matrix(gate_type, params)?;
        
        // Allocate device memory for state vector
        let state_size = amplitudes.len() * size_of::<Complex64>();
        let device_state = match rust_cuda::DeviceBuffer::new(state_size) {
            Ok(buffer) => buffer,
            Err(e) => return Err(format!("Failed to allocate CUDA device memory: {}", e))
        };
        
        // Copy state vector to device
        unsafe {
            match rust_cuda::memcpy_htod(
                device_state.as_device_ptr(),
                amplitudes.as_ptr() as *const std::ffi::c_void,
                state_size
            ) {
                Ok(_) => {},
                Err(e) => return Err(format!("Failed to copy state to device: {}", e))
            }
        }
        
        // Allocate device memory for gate matrix
        let gate_size = gate_matrix.len() * size_of::<Complex64>();
        let device_gate = match rust_cuda::DeviceBuffer::new(gate_size) {
            Ok(buffer) => buffer,
            Err(e) => return Err(format!("Failed to allocate CUDA device memory for gate: {}", e))
        };
        
        // Copy gate matrix to device
        unsafe {
            match rust_cuda::memcpy_htod(
                device_gate.as_device_ptr(),
                gate_matrix.as_ptr() as *const std::ffi::c_void,
                gate_size
            ) {
                Ok(_) => {},
                Err(e) => return Err(format!("Failed to copy gate to device: {}", e))
            }
        }
        
        // Allocate device memory for qubit indices
        let indices_size = qubit_indices.len() * size_of::<u32>();
        let device_indices = match rust_cuda::DeviceBuffer::new(indices_size) {
            Ok(buffer) => buffer,
            Err(e) => return Err(format!("Failed to allocate CUDA device memory for indices: {}", e))
        };
        
        // Convert indices to u32 and copy to device
        let qubit_indices_u32: Vec<u32> = qubit_indices.iter().map(|&idx| idx as u32).collect();
        unsafe {
            match rust_cuda::memcpy_htod(
                device_indices.as_device_ptr(),
                qubit_indices_u32.as_ptr() as *const std::ffi::c_void,
                indices_size
            ) {
                Ok(_) => {},
                Err(e) => return Err(format!("Failed to copy indices to device: {}", e))
            }
        }
        
        // Launch appropriate kernel based on gate type
        let kernel_name = match gate_type {
            "h" | "hadamard" => "apply_hadamard_kernel",
            "x" | "not" => "apply_x_gate_kernel",
            "y" => "apply_y_gate_kernel",
            "z" => "apply_z_gate_kernel",
            "rx" => "apply_rx_gate_kernel",
            "ry" => "apply_ry_gate_kernel",
            "rz" => "apply_rz_gate_kernel",
            "cnot" => "apply_cnot_gate_kernel",
            "cz" => "apply_cz_gate_kernel",
            _ => return Err(format!("Unsupported gate type for CUDA: {}", gate_type))
        };
        
        // Prepare kernel parameters
        let n_qubits = (amplitudes.len() as f64).log2() as u32;
        let threads_per_block = 256;
        let blocks_per_grid = (amplitudes.len() + threads_per_block as usize - 1) / threads_per_block as usize;
        
        // Launch kernel
        match context.launch_kernel(
            kernel_name,
            blocks_per_grid as u32,
            threads_per_block,
            &[
                &device_state,
                &device_gate,
                &device_indices,
                &n_qubits,
                &(amplitudes.len() as u32)
            ]
        ) {
            Ok(_) => {},
            Err(e) => return Err(format!("Failed to launch CUDA kernel: {}", e))
        }
        
        // Copy results back to host
        unsafe {
            match rust_cuda::memcpy_dtoh(
                amplitudes.as_mut_ptr() as *mut std::ffi::c_void,
                device_state.as_device_ptr(),
                state_size
            ) {
                Ok(_) => {},
                Err(e) => return Err(format!("Failed to copy results from device: {}", e))
            }
        }
        
        Ok(())
    }
    
    // Execute a gate operation using OpenCL
    #[cfg(feature = "opencl")]
    fn execute_gate_operation_opencl(
        &self,
        context: &Arc<OpenCLContext>,
        gate_type: &str,
        amplitudes: &mut [Complex64],
        qubit_indices: &[usize],
        params: &[f64]
    ) -> Result<(), String> {
        use opencl3::memory::{Buffer, CL_MEM_READ_WRITE, CL_MEM_READ_ONLY};
        use opencl3::kernel::{ExecuteKernel, Kernel};
        use opencl3::types::cl_float2;
        use std::mem::size_of;
        
        // Prepare gate matrix based on gate type
        let gate_matrix = get_gate_matrix(gate_type, params)?;
        
        // Convert Complex64 to cl_float2
        let mut cl_amplitudes = Vec::with_capacity(amplitudes.len());
        for &amp in amplitudes.iter() {
            cl_amplitudes.push(cl_float2 { s: [amp.re as f32, amp.im as f32] });
        }
        
        let mut cl_gate_matrix = Vec::with_capacity(gate_matrix.len());
        for &g in gate_matrix.iter() {
            cl_gate_matrix.push(cl_float2 { s: [g.re as f32, g.im as f32] });
        }
        
        // Convert qubit indices to u32
        let qubit_indices_u32: Vec<u32> = qubit_indices.iter().map(|&idx| idx as u32).collect();
        
        // Create OpenCL buffers
        let state_buffer = Buffer::create(
            &context.context, 
            CL_MEM_READ_WRITE, 
            cl_amplitudes.len() * size_of::<cl_float2>(), 
            std::ptr::null_mut()
        ).map_err(|e| format!("Failed to create OpenCL state buffer: {}", e))?;
        
        let gate_buffer = Buffer::create(
            &context.context, 
            CL_MEM_READ_ONLY, 
            cl_gate_matrix.len() * size_of::<cl_float2>(), 
            std::ptr::null_mut()
        ).map_err(|e| format!("Failed to create OpenCL gate buffer: {}", e))?;
        
        let indices_buffer = Buffer::create(
            &context.context, 
            CL_MEM_READ_ONLY, 
            qubit_indices_u32.len() * size_of::<u32>(), 
            std::ptr::null_mut()
        ).map_err(|e| format!("Failed to create OpenCL indices buffer: {}", e))?;
        
        // Write data to buffers
        context.queue.enqueue_write_buffer(
            &state_buffer,
            true,
            0,
            &cl_amplitudes,
            &[]
        ).map_err(|e| format!("Failed to write to OpenCL state buffer: {}", e))?;
        
        context.queue.enqueue_write_buffer(
            &gate_buffer,
            true,
            0,
            &cl_gate_matrix,
            &[]
        ).map_err(|e| format!("Failed to write to OpenCL gate buffer: {}", e))?;
        
        context.queue.enqueue_write_buffer(
            &indices_buffer,
            true,
            0,
            &qubit_indices_u32,
            &[]
        ).map_err(|e| format!("Failed to write to OpenCL indices buffer: {}", e))?;
        
        // Get appropriate kernel
        let kernel_name = match gate_type {
            "h" | "hadamard" => "apply_hadamard_kernel",
            "x" | "not" => "apply_x_gate_kernel",
            "y" => "apply_y_gate_kernel",
            "z" => "apply_z_gate_kernel",
            "rx" => "apply_rx_gate_kernel",
            "ry" => "apply_ry_gate_kernel",
            "rz" => "apply_rz_gate_kernel",
            "cnot" => "apply_cnot_gate_kernel",
            "cz" => "apply_cz_gate_kernel",
            _ => return Err(format!("Unsupported gate type for OpenCL: {}", gate_type))
        };
        
        // Get or build the kernel
        let kernel = {
            let mut kernels = context.compiled_kernels.write().unwrap();
            if !kernels.contains_key(kernel_name) {
                // Get OpenCL source code for the gate
                let kernel_source = get_opencl_kernel_source(gate_type)?;
                
                // Build the program
                let program = opencl3::program::Program::create_and_build_from_source(
                    &context.context,
                    &kernel_source,
                    ""
                ).map_err(|e| format!("Failed to build OpenCL program: {}", e))?;
                
                // Create kernel
                let kernel = Kernel::create(&program, kernel_name)
                    .map_err(|e| format!("Failed to create OpenCL kernel: {}", e))?;
                
                kernels.insert(kernel_name.to_string(), kernel);
            }
            
            kernels.get(kernel_name).unwrap().clone()
        };
        
        // Set kernel arguments
        let n_qubits = (amplitudes.len() as f64).log2() as u32;
        let state_size = amplitudes.len() as u32;
        
        ExecuteKernel::new(&kernel)
            .set_arg(&state_buffer)
            .set_arg(&gate_buffer)
            .set_arg(&indices_buffer)
            .set_arg(&n_qubits)
            .set_arg(&state_size)
            .set_global_work_size(amplitudes.len())
            .enqueue_nd_range(&context.queue)
            .map_err(|e| format!("Failed to execute OpenCL kernel: {}", e))?;
        
        // Wait for completion
        context.queue.finish().map_err(|e| format!("Failed to finish OpenCL queue: {}", e))?;
        
        // Read results back
        let mut cl_results = vec![cl_float2 { s: [0.0, 0.0] }; cl_amplitudes.len()];
        context.queue.enqueue_read_buffer(
            &state_buffer,
            true,
            0,
            &mut cl_results,
            &[]
        ).map_err(|e| format!("Failed to read from OpenCL state buffer: {}", e))?;
        
        // Convert results back to Complex64
        for (i, &cl_amp) in cl_results.iter().enumerate() {
            amplitudes[i] = Complex64::new(cl_amp.s[0] as f64, cl_amp.s[1] as f64);
        }
        
        Ok(())
    }
}

// Generates a SPIR-V shader for a specific quantum gate
#[cfg(feature = "vulkan")]
fn get_vulkan_shader_for_gate(gate_type: &str) -> Result<Vec<u32>, String> {
    use crate::kernel::compute_shader::get_vulkan_shader_for_gate;
    get_vulkan_shader_for_gate(gate_type)
}

// Generates OpenCL kernel source for a specific quantum gate
#[cfg(feature = "opencl")]
fn get_opencl_kernel_source(gate_type: &str) -> Result<String, String> {
    use crate::kernel::compute_shader::get_opencl_kernel_source;
    get_opencl_kernel_source(gate_type)
}

// Helper function to find suitable memory type index
#[cfg(feature = "vulkan")]
fn find_memory_type_index(
    memory_properties: &ash::vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: ash::vk::MemoryPropertyFlags
) -> Option<u32> {
    for i in 0..memory_properties.memory_type_count {
        if (type_filter & (1 << i)) != 0 && 
           (memory_properties.memory_types[i as usize].property_flags & properties) == properties {
            return Some(i);
        }
    }
    None
}

// Gets the gate matrix for a specific gate type and parameters
fn get_gate_matrix(gate_type: &str, params: &[f64]) -> Result<Vec<Complex64>, String> {
    match gate_type {
        "h" | "hadamard" => {
            let inv_sqrt2 = 1.0 / f64::sqrt(2.0);
            Ok(vec![
                Complex64::new(inv_sqrt2, 0.0), Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0), Complex64::new(-inv_sqrt2, 0.0)
            ])
        },
        "x" | "not" => {
            Ok(vec![
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)
            ])
        },
        "y" => {
            Ok(vec![
                Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)
            ])
        },
        "z" => {
            Ok(vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)
            ])
        },
        "rx" => {
            if params.is_empty() {
                return Err("rx gate requires an angle parameter".to_string());
            }
            let angle = params[0] / 2.0; // Half angle for SU(2) representation
            let cos = angle.cos();
            let sin = angle.sin();
            Ok(vec![
                Complex64::new(cos, 0.0), Complex64::new(0.0, -sin),
                Complex64::new(0.0, -sin), Complex64::new(cos, 0.0)
            ])
        },
        "ry" => {
            if params.is_empty() {
                return Err("ry gate requires an angle parameter".to_string());
            }
            let angle = params[0] / 2.0; // Half angle for SU(2) representation
            let cos = angle.cos();
            let sin = angle.sin();
            Ok(vec![
                Complex64::new(cos, 0.0), Complex64::new(-sin, 0.0),
                Complex64::new(sin, 0.0), Complex64::new(cos, 0.0)
            ])
        },
        "rz" => {
            if params.is_empty() {
                return Err("rz gate requires an angle parameter".to_string());
            }
            let angle = params[0] / 2.0; // Half angle for SU(2) representation
            let cos = angle.cos();
            let sin = angle.sin();
            Ok(vec![
                Complex64::new(cos, -sin), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(cos, sin)
            ])
        },
        "s" => {
            Ok(vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)
            ])
        },
        "t" => {
            let sqrt2_inv = 1.0 / f64::sqrt(2.0);
            Ok(vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(sqrt2_inv, sqrt2_inv)
            ])
        },
        "cnot" => {
            // CNOT is a 4x4 matrix: I 0 0 0, 0 I 0 0, 0 0 0 X, 0 0 X 0
            // where I is the 2x2 identity and X is the 2x2 Pauli-X
            // However, we only need to pass the effective 2x2 target operation
            Ok(vec![
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)
            ])
        },
        "cz" => {
            // CZ is a 4x4 matrix: I 0 0 0, 0 I 0 0, 0 0 I 0, 0 0 0 Z
            // where I is the 2x2 identity and Z is the 2x2 Pauli-Z
            // However, we only need to pass the effective 2x2 target operation
            Ok(vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)
            ])
        },
        "swap" => {
            // SWAP gate matrix for a pair of qubits
            Ok(vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)
            ])
        },
        _ => Err(format!("Unsupported gate type: {}", gate_type))
    }
}

// CPU implementations of quantum gates
fn apply_hadamard_cpu(amplitudes: &mut [Complex64], target: usize) -> Result<(), String> {
    let n = amplitudes.len();
    let mask = 1 << target;
    
    // Process pairs of amplitudes where the target qubit differs
    for i in 0..n {
        if (i & mask) != 0 {
            continue; // Skip the second element of each pair
        }
        
        let j = i | mask;
        let amp_0 = amplitudes[i];
        let amp_1 = amplitudes[j];
        
        // Apply Hadamard transform: 1/sqrt(2) * [1 1; 1 -1]
        let inv_sqrt2 = 1.0 / f64::sqrt(2.0);
        amplitudes[i] = inv_sqrt2 * (amp_0 + amp_1);
        amplitudes[j] = inv_sqrt2 * (amp_0 - amp_1);
    }
    
    Ok(())
}

fn apply_x_gate_cpu(amplitudes: &mut [Complex64], target: usize) -> Result<(), String> {
    let n = amplitudes.len();
    let mask = 1 << target;
    
    // Process pairs of amplitudes where the target qubit differs
    for i in 0..n {
        if (i & mask) != 0 {
            continue; // Skip the second element of each pair
        }
        
        let j = i | mask;
        
        // Apply X gate: swap amplitudes
        let temp = amplitudes[i];
        amplitudes[i] = amplitudes[j];
        amplitudes[j] = temp;
    }
    
    Ok(())
}

fn apply_y_gate_cpu(amplitudes: &mut [Complex64], target: usize) -> Result<(), String> {
    let n = amplitudes.len();
    let mask = 1 << target;
    
    // Process pairs of amplitudes where the target qubit differs
    for i in 0..n {
        if (i & mask) != 0 {
            continue; // Skip the second element of each pair
        }
        
        let j = i | mask;
        
        // Apply Y gate: [0 -i; i 0]
        let amp_0 = amplitudes[i];
        let amp_1 = amplitudes[j];
        
        amplitudes[i] = Complex64::new(0.0, -1.0) * amp_1;
        amplitudes[j] = Complex64::new(0.0, 1.0) * amp_0;
    }
    
    Ok(())
}

fn apply_z_gate_cpu(amplitudes: &mut [Complex64], target: usize) -> Result<(), String> {
    let n = amplitudes.len();
    let mask = 1 << target;
    
    // Process elements where the target qubit is 1
    for i in 0..n {
        if (i & mask) != 0 {
            // Apply Z gate: negate amplitude when target qubit is 1
            amplitudes[i] = -amplitudes[i];
        }
    }
    
    Ok(())
}

fn apply_rx_gate_cpu(amplitudes: &mut [Complex64], target: usize, angle: f64) -> Result<(), String> {
    let n = amplitudes.len();
    let mask = 1 << target;
    
    // Precompute sine and cosine of half angle
    let half_angle = angle / 2.0;
    let cos = half_angle.cos();
    let sin = half_angle.sin();
    
    // Process pairs of amplitudes where the target qubit differs
    for i in 0..n {
        if (i & mask) != 0 {
            continue; // Skip the second element of each pair
        }
        
        let j = i | mask;
        let amp_0 = amplitudes[i];
        let amp_1 = amplitudes[j];
        
        // Apply Rx rotation: [cos(θ/2) -i*sin(θ/2); -i*sin(θ/2) cos(θ/2)]
        amplitudes[i] = cos * amp_0 + Complex64::new(0.0, -sin) * amp_1;
        amplitudes[j] = Complex64::new(0.0, -sin) * amp_0 + cos * amp_1;
    }
    
    Ok(())
}

fn apply_ry_gate_cpu(amplitudes: &mut [Complex64], target: usize, angle: f64) -> Result<(), String> {
    let n = amplitudes.len();
    let mask = 1 << target;
    
    // Precompute sine and cosine of half angle
    let half_angle = angle / 2.0;
    let cos = half_angle.cos();
    let sin = half_angle.sin();
    
    // Process pairs of amplitudes where the target qubit differs
    for i in 0..n {
        if (i & mask) != 0 {
            continue; // Skip the second element of each pair
        }
        
        let j = i | mask;
        let amp_0 = amplitudes[i];
        let amp_1 = amplitudes[j];
        
        // Apply Ry rotation: [cos(θ/2) -sin(θ/2); sin(θ/2) cos(θ/2)]
        amplitudes[i] = cos * amp_0 - sin * amp_1;
        amplitudes[j] = sin * amp_0 + cos * amp_1;
    }
    
    Ok(())
}

fn apply_rz_gate_cpu(amplitudes: &mut [Complex64], target: usize, angle: f64) -> Result<(), String> {
    let n = amplitudes.len();
    let mask = 1 << target;
    
    // Precompute exponentials
    let half_angle = angle / 2.0;
    let exp_plus = Complex64::new(half_angle.cos(), half_angle.sin());
    let exp_minus = Complex64::new(half_angle.cos(), -half_angle.sin());
    
    // Apply Rz rotation: [e^(-iθ/2) 0; 0 e^(iθ/2)]
    for i in 0..n {
        if (i & mask) == 0 {
            // Target qubit is 0
            amplitudes[i] *= exp_minus;
        } else {
            // Target qubit is 1
            amplitudes[i] *= exp_plus;
        }
    }
    
    Ok(())
}

fn apply_cnot_gate_cpu(amplitudes: &mut [Complex64], control: usize, target: usize) -> Result<(), String> {
    let n = amplitudes.len();
    let control_mask = 1 << control;
    let target_mask = 1 << target;
    
    // Only apply X gate to target qubit when control qubit is 1
    for i in 0..n {
        if (i & control_mask) != 0 {
            // Control qubit is 1, apply X to target
            let j = i ^ target_mask; // Flip the target bit
            
            // Swap amplitudes if the control qubit is set
            if i < j {
                let temp = amplitudes[i];
                amplitudes[i] = amplitudes[j];
                amplitudes[j] = temp;
            }
        }
    }
    
    Ok(())
}

fn apply_cz_gate_cpu(amplitudes: &mut [Complex64], control: usize, target: usize) -> Result<(), String> {
    let n = amplitudes.len();
    let control_mask = 1 << control;
    let target_mask = 1 << target;
    
    // Apply Z gate to target qubit when control qubit is 1
    for i in 0..n {
        if ((i & control_mask) != 0) && ((i & target_mask) != 0) {
            // Both control and target qubits are 1
            amplitudes[i] = -amplitudes[i];
        }
    }
    
    Ok(())
}

// Context structures for GPU backends

#[cfg(feature = "vulkan")]
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: ash::vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family_index: u32,
    pub compute_queue: ash::vk::Queue,
    pub command_pool: ash::vk::CommandPool,
    pub descriptor_pool: ash::vk::DescriptorPool,
    pub descriptor_set_layout: ash::vk::DescriptorSetLayout,
    pub pipeline_layout: ash::vk::PipelineLayout,
    pub compute_pipeline: ash::vk::Pipeline,
    
    #[cfg(debug_assertions)]
    pub debug_utils_loader: Option<ash::extensions::ext::DebugUtils>,
    #[cfg(debug_assertions)]
    pub debug_messenger: Option<ash::vk::DebugUtilsMessengerEXT>,
}

#[cfg(feature = "vulkan")]
impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            
            if self.descriptor_pool != ash::vk::DescriptorPool::null() {
                self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            }
            
            if self.descriptor_set_layout != ash::vk::DescriptorSetLayout::null() {
                self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            }
            
            if self.pipeline_layout != ash::vk::PipelineLayout::null() {
                self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            }
            
            if self.compute_pipeline != ash::vk::Pipeline::null() {
                self.device.destroy_pipeline(self.compute_pipeline, None);
            }
            
            #[cfg(debug_assertions)]
            if let (Some(loader), Some(messenger)) = (
                &self.debug_utils_loader,
                self.debug_messenger
            ) {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(feature = "cuda")]
pub struct CudaContext {
    device: rust_cuda::Device,
    context: rust_cuda::Context,
    modules: HashMap<String, rust_cuda::Module>,
}

#[cfg(feature = "cuda")]
impl CudaContext {
    pub fn new(device: rust_cuda::Device) -> Result<Self, String> {
        let context = device
            .create_context()
            .map_err(|e| format!("Failed to create CUDA context: {}", e))?;
        
        Ok(Self {
            device,
            context,
            modules: HashMap::new(),
        })
    }
    
    pub fn launch_kernel(
        &self,
        kernel_name: &str,
        grid_dim: u32,
        block_dim: u32,
        args: &[&dyn rust_cuda::KernelArg]
    ) -> Result<(), String> {
        // Get or load the module for this kernel
        let module = self.get_or_load_module(kernel_name)?;
        
        // Get function from module
        let function = module
            .get_function(kernel_name)
            .map_err(|e| format!("Failed to get CUDA function '{}': {}", kernel_name, e))?;
        
        // Launch kernel
        unsafe {
            function
                .launch(
                    grid_dim,
                    1,
                    1,
                    block_dim,
                    1,
                    1,
                    0,
                    self.context.stream(),
                    args,
                )
                .map_err(|e| format!("Failed to launch CUDA kernel '{}': {}", kernel_name, e))?;
        }
        
        // Synchronize
        self.context
            .synchronize()
            .map_err(|e| format!("Failed to synchronize CUDA context: {}", e))?;
        
        Ok(())
    }
    
    fn get_or_load_module(&self, kernel_name: &str) -> Result<&rust_cuda::Module, String> {
        if !self.modules.contains_key(kernel_name) {
            // In a real implementation, you would load module from a file or string
            // This is just a placeholder
            return Err(format!("CUDA module for '{}' not found", kernel_name));
        }
        
        Ok(&self.modules[kernel_name])
    }
}

#[cfg(feature = "opencl")]
pub struct OpenCLContext {
    pub platform: opencl3::platform::Platform,
    pub device: opencl3::device::Device,
    pub context: opencl3::context::Context,
    pub queue: opencl3::command_queue::CommandQueue,
    pub compiled_kernels: RwLock<HashMap<String, opencl3::kernel::Kernel>>,
}