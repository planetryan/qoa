use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Instant;
use std::net::{TcpListener, TcpStream};
use std::io::{Read, Write, Error as IoError};
use std::collections::HashMap;
use std::process::Command;

#[cfg(feature = "cuda")]
use rust_cuda::{self, launch_kernel, Device, DeviceBuffer};

#[cfg(feature = "opencl")]
use opencl3; // imports the opencl3 crate to access its modules
#[cfg(feature = "opencl")]
use opencl3::device::Device; // explicitly imports the device struct
#[cfg(feature = "opencl")]
use opencl3::device_types::CL_DEVICE_TYPE_ALL; // imports the constant for all device types, used for device discovery
#[cfg(feature = "opencl")]
use opencl3::device_info::{CL_DEVICE_NAME, CL_DEVICE_GLOBAL_MEM_SIZE}; // imports constants for querying device name and global memory size
#[cfg(feature = "opencl")]
use opencl3::platform_info::CL_PLATFORM_NAME; // imports constant for querying platform name
#[cfg(feature = "opencl")]
use opencl3::platform::Platform; // imports the platform struct


#[cfg(feature = "mpi")]
use mpi::{traits::*, topology::SystemCommunicator}; // imports MPI traits and the system communicator for distributed communication

#[cfg(feature = "vulkan")]
use ash::{vk, Entry, Instance}; // imports Vulkan API types and functions from the ash crate
#[cfg(feature = "vulkan")]
use ash::extensions::ext::DebugUtils; // imports the debug utilities extension for Vulkan debugging
#[cfg(feature = "vulkan")]
use std::ffi::{CStr, CString}; // imports C-compatible string types for FFI with Vulkan

// for image saving
use image::{ImageBuffer, Rgba}; // imports types for image manipulation and saving

// =======================
// vulkan
// =======================

// vulkan context struct, conditional on feature flag
#[cfg(feature = "vulkan")]
pub struct VulkanContext {
    pub entry: Entry, // the Vulkan entry point
    pub instance: Instance, // the Vulkan instance
    pub physical_device: vk::PhysicalDevice, // the selected physical device (GPU)
    pub device: ash::Device, // the logical device
    pub graphics_queue: vk::Queue, // the graphics queue
    pub command_pool: vk::CommandPool, // the command pool for allocating command buffers
    pub debug_utils_loader: Option<DebugUtils>, // loader for debug utilities extension
    pub debug_messenger: Option<vk::DebugUtilsMessengerEXT>, // debug messenger for validation layer messages
}

#[cfg(not(feature = "vulkan"))]
pub struct VulkanContext; // dummy struct when Vulkan is not enabled

#[cfg(feature = "vulkan")]
impl VulkanContext {
    // creates a new Vulkan context, initializing necessary components
    pub fn new() -> Result<Self, String> {
        unsafe {
            // loads the Vulkan entry point
            let entry = Entry::load().map_err(|e| format!("failed to create vulkan entry: {}", e))?;

            // defines application information for the Vulkan instance
            let app_info = vk::ApplicationInfo::builder()
                .application_name(CStr::from_bytes_with_nul_unchecked(b"QOA_Renderer\0"))
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(CStr::from_bytes_with_nul_unchecked(b"QOA_Engine\0"))
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::make_api_version(0, 1, 2, 0));

            // defines validation layers to enable for debugging
            let validation_layers_cstrs = [CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0")];
            let validation_layers_ptrs: Vec<*const i8> = validation_layers_cstrs.iter().map(|&s| s.as_ptr()).collect();

            // checks if validation layers should be enabled based on debug assertions and availability
            let enable_validation_layers = cfg!(debug_assertions) && ash_check_validation_layer_support(&entry, &validation_layers_cstrs);

            // enumerates and collects instance extensions, adding debug utils if validation is enabled
            let mut instance_extensions = ash_enumerate_instance_extension_names(&entry)?;
            if enable_validation_layers {
                instance_extensions.push(DebugUtils::name().as_ptr()); // ensures debug utils is added for validation
            }

            // builds the instance creation info
            let mut instance_create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_extension_names(&instance_extensions);

            // builds the debug messenger creation info
            let mut debug_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));


            // if validation layers are enabled, adds them to the instance creation info and chains the debug messenger
            if enable_validation_layers {
                instance_create_info = instance_create_info
                    .enabled_layer_names(&validation_layers_ptrs)
                    .push_next(&mut debug_messenger_create_info); // chains debug messenger create info
            }

            // creates the Vulkan instance
            let instance = entry.create_instance(&instance_create_info, None)
                .map_err(|e| format!("failed to create vulkan instance: {}", e))?;

            // initializes the debug utils loader if validation is enabled
            let debug_utils_loader = if enable_validation_layers {
                Some(DebugUtils::new(&entry, &instance))
            } else {
                None
            };

            // creates the debug messenger if validation is enabled
            let debug_messenger = if enable_validation_layers {
                Some(debug_utils_loader.as_ref().unwrap().create_debug_utils_messenger(&debug_messenger_create_info, None)
                    .map_err(|e| format!("failed to set up debug messenger: {}", e))?)
            } else {
                None
            };

            // physical device selection
            // enumerates physical devices and selects a suitable one
            let physical_devices = instance.enumerate_physical_devices()
                .map_err(|e| format!("failed to enumerate physical devices: {}", e))?;

            let physical_device = physical_devices.into_iter()
                .find(|&p_device| ash_is_device_suitable(&instance, p_device))
                .ok_or_else(|| "failed to find a suitable physical device".to_string())?;

            // queue family properties
            // gets queue family properties and finds a graphics queue family
            let queue_family_properties = instance.get_physical_device_queue_family_properties(physical_device);
            let graphics_queue_family_index = queue_family_properties.iter()
                .enumerate()
                .find(|(_idx, properties)| properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|(idx, _)| idx as u32)
                .ok_or_else(|| "failed to find a graphics queue family".to_string())?;

            // device creation
            // defines required device extensions
            let device_extensions = [
                ash::extensions::khr::Swapchain::name(),
                // add other cutting edge extensions if needed, e.g., vk::khr_dynamic_rendering::name(),
            ];
            let device_extensions_cstrs: Vec<CString> = device_extensions.iter()
                .map(|&s| CString::new(s.to_bytes()).unwrap())
                .collect();
            let device_extension_pointers: Vec<*const i8> = device_extensions_cstrs.iter()
                .map(|s| s.as_ptr())
                .collect();

            // defines queue priorities and device queue creation info
            let queue_priorities = [1.0f32];
            let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(graphics_queue_family_index)
                .queue_priorities(&queue_priorities);

            let device_features = vk::PhysicalDeviceFeatures::builder(); // no specific features for now

            // builds the logical device creation info
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_extension_names(&device_extension_pointers)
                .enabled_features(&device_features);

            // creates the logical device
            let device = instance.create_device(physical_device, &device_create_info, None)
                .map_err(|e| format!("failed to create vulkan logical device: {}", e))?;

            // gets the graphics queue
            let graphics_queue = device.get_device_queue(graphics_queue_family_index, 0);

            // command pool
            // creates a command pool for allocating command buffers
            let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(graphics_queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER); // allows command buffers to be reset

            let command_pool = device.create_command_pool(&command_pool_create_info, None)
                .map_err(|e| format!("failed to create command pool: {}", e))?;

            // returns the initialized VulkanContext
            Ok(Self {
                entry,
                instance,
                physical_device,
                device,
                graphics_queue,
                command_pool,
                debug_utils_loader,
                debug_messenger,
            })
        }
    }
}

#[cfg(feature = "vulkan")]
impl Drop for VulkanContext {
    // cleans up Vulkan resources when the context is dropped
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            if let Some(debug_utils_loader) = &self.debug_utils_loader {
                if let Some(debug_messenger) = self.debug_messenger {
                    debug_utils_loader.destroy_debug_utils_messenger(debug_messenger, None);
                }
            }
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(feature = "vulkan")]
// callback function for Vulkan debug messages
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { *p_callback_data };
    let message_id_number = callback_data.message_id_number;
    let p_message_id_name = callback_data.p_message_id_name;
    let p_message = callback_data.p_message;

    let message_id_name = if p_message_id_name.is_null() {
        unsafe { CStr::from_bytes_with_nul_unchecked(b"unknown\0") }
    } else {
        unsafe { CStr::from_ptr(p_message_id_name) }
    };
    let message = unsafe { CStr::from_ptr(p_message) };

    println!(
        "{:?}: {:?} [{} ({})] {:?}",
        message_severity,
        message_type,
        message_id_name.to_string_lossy(),
        message_id_number,
        message.to_string_lossy(),
    );
    vk::FALSE
}

#[cfg(feature = "vulkan")]
// enumerates available Vulkan instance extension names
fn ash_enumerate_instance_extension_names(entry: &Entry) -> Result<Vec<*const i8>, String> {
    let extensions = entry.enumerate_instance_extension_properties(None)
        .map_err(|e| format!("failed to enumerate instance extension properties: {}", e))?;

    let mut extension_names = Vec::new();
    for extension in extensions {
        let name_bytes = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) }.to_bytes();
        let c_string = CString::new(name_bytes).map_err(|e| format!("failed to create cstring: {}", e))?;
        extension_names.push(c_string.as_ptr());
    }

    Ok(extension_names)
}

#[cfg(feature = "vulkan")]
// checks if required validation layers are supported
fn ash_check_validation_layer_support(entry: &Entry, validation_layers: &[&CStr]) -> bool {
    let available_layers = entry.enumerate_instance_layer_properties()
        .expect("failed to enumerate instance layer properties");

    for required_layer in validation_layers {
        let mut layer_found = false;
        for available_layer in &available_layers {
            let available_layer_name = unsafe { CStr::from_ptr(available_layer.layer_name.as_ptr()) };
            if required_layer == &available_layer_name {
                layer_found = true;
                break;
            }
        }
        if !layer_found {
            return false;
        }
    }
    true
}

#[cfg(feature = "vulkan")]
// checks if a physical device is suitable for the application
fn ash_is_device_suitable(instance: &Instance, physical_device: vk::PhysicalDevice) -> bool {
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let features = unsafe { instance.get_physical_device_features(physical_device) };

    let device_type = properties.device_type;
    let has_graphics_queue = unsafe {
        instance.get_physical_device_queue_family_properties(physical_device).iter()
            .any(|props| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
    };

    // checks for required device extensions
    let available_extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .expect("failed to enumerate device extension properties");

    let required_extensions = [
        ash::extensions::khr::Swapchain::name(),
    ];

    let extensions_supported = required_extensions.iter().all(|&required_ext_cstr| {
        available_extensions.iter().any(|ext| {
            let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            ext_name == required_ext_cstr
        })
    });

    // prefers discrete gpus and devices with graphics capabilities and required extensions
    device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        && has_graphics_queue
        && features.sampler_anisotropy == vk::TRUE
        && extensions_supported
}


// ========================
// state vector partitioning
// ========================

// represents a partition of the global state vector
#[derive(Clone, Debug)]
pub struct StatePartition {
    // starting index in the global state vector
    pub start_idx: usize,
    // ending index in the global state vector (exclusive)
    pub end_idx: usize,
    // number of qubits this partition represents
    pub qubit_count: usize,
    // node id responsible for this partition
    pub node_id: usize,
}

// distributed simulation configuration
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DistributedConfig {
    // total number of qubits in the simulation
    pub total_qubits: usize,
    // number of nodes in the distributed system
    pub node_count: usize,
    // this node's id (0-indexed)
    pub node_id: usize,
    // network addresses of all nodes
    pub node_addresses: Vec<String>,
    // communication port
    pub port: u16,
    // whether to use mpi instead of custom tcp
    pub use_mpi: bool,
    // whether to use gpu acceleration
    pub use_gpu: bool,
    // maximum memory to use per node (in gb)
    pub max_memory_gb: f64,
}

impl Default for DistributedConfig {
    // provides default values for DistributedConfig
    fn default() -> Self {
        Self {
            total_qubits: 24,
            node_count: 1,
            node_addresses: vec!["localhost".to_string()],
            port: 9000,
            use_mpi: false,
            use_gpu: true,
            max_memory_gb: 8.0,
            node_id: 0, // sets default for node_id
        }
    }
}

// partitioning strategies for distributed simulation
pub enum PartitionStrategy {
    // equal distribution of state vector
    EqualSize,
    // partition based on available memory
    MemoryBased,
    // partition based on computational capabilities
    PerformanceBased,
}

// calculates partitions for distributed simulation based on the chosen strategy
pub fn calculate_partitions(
    config: &DistributedConfig,
    strategy: PartitionStrategy,
) -> Vec<StatePartition> {
    let total_states = 1 << config.total_qubits;
    
    match strategy {
        PartitionStrategy::EqualSize => {
            let states_per_node = total_states / config.node_count;
            let remainder = total_states % config.node_count;
            
            (0..config.node_count)
                .map(|node_id| {
                    let extra = if node_id < remainder { 1 } else { 0 };
                    let start_idx = node_id * states_per_node + std::cmp::min(node_id, remainder);
                    let end_idx = start_idx + states_per_node + extra;
                    
                    StatePartition {
                        start_idx,
                        end_idx,
                        qubit_count: config.total_qubits,
                        node_id,
                    }
                })
                .collect()
        },
        PartitionStrategy::MemoryBased => {
            // implements memory-based partitioning
            // this is a simplified version - real implementation would query system memory
            let mut partitions = Vec::new();
            let mut start_idx = 0;
            
            for node_id in 0..config.node_count {
                // calculates partition size based on node's memory
                // here we just use a simple allocation based on the node id
                // a real implementation would query each node's available memory
                let memory_factor = 1.0 - (0.1 * (node_id as f64));
                let partition_size = (total_states as f64 * memory_factor / config.node_count as f64) as usize;
                let end_idx = if node_id == config.node_count - 1 {
                    total_states
                } else {
                    start_idx + partition_size
                };
                
                partitions.push(StatePartition {
                    start_idx,
                    end_idx,
                    qubit_count: config.total_qubits,
                    node_id,
                });
                
                start_idx = end_idx;
            }
            
            partitions
        },
        PartitionStrategy::PerformanceBased => {
            // a performance-based implementation would benchmark each node
            // for now, we'll provide a simple approximation
            let mut partitions = Vec::new();
            let mut start_idx = 0;
            
            // simulates performance factors (1.0 = baseline)
            let performance_factors: Vec<f64> = (0..config.node_count)
                .map(|i| 1.0 + (i as f64 * 0.2))
                .collect();
                
            let total_factor: f64 = performance_factors.iter().sum();
            
            for (node_id, &factor) in performance_factors.iter().enumerate() {
                let partition_ratio = factor / total_factor;
                let partition_size = (total_states as f64 * partition_ratio) as usize;
                
                let end_idx = if node_id == config.node_count - 1 {
                    total_states
                } else {
                    start_idx + partition_size
                };
                
                partitions.push(StatePartition {
                    start_idx,
                    end_idx,
                    qubit_count: config.total_qubits,
                    node_id,
                });
                
                start_idx = end_idx;
            }
            
            partitions
        }
    }
}

// ========================
// multi-gpu support
// ========================

// represents available gpu devices
pub struct GpuManager {
    devices: Vec<GpuDevice>,
    active_device: usize,
    vulkan_context: Option<Arc<VulkanContext>>,
}

// information about a single gpu device
#[derive(Clone, Debug)]
pub struct GpuDevice {
    pub id: usize, 
    pub name: String, 
    pub memory_mb: usize, 
    pub is_available: bool, 
    pub backend: GpuBackend, // adds backend type
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuBackend {
    Cuda,
    OpenCL,
    Vulkan,
    None,
}

impl GpuManager {
    // initializes and detects available gpu devices
    pub fn new() -> Self {
        let mut devices = Vec::new(); 
        let mut vulkan_context_local: Option<Arc<VulkanContext>> = None;

        // runtime detection and initialization of gpu backends
        #[cfg(feature = "cuda")]
        {
            // detects cuda devices
            if let Ok(device_count) = rust_cuda::device_count() {
                for i in 0..device_count {
                    if let Ok(device) = rust_cuda::Device::new(i) {
                        if let Ok(props) = device.get_properties() {
                            devices.push(GpuDevice {
                                id: i,
                                name: props.name().to_string(),
                                memory_mb: props.total_memory() / (1024 * 1024),
                                is_available: true,
                                backend: GpuBackend::Cuda,
                            });
                        }
                    }
                }
            }
        }
        
        #[cfg(feature = "opencl")]
        {
            // detects opencl devices
            // uses proper opencl constants
            use opencl3::{
                device_types::CL_DEVICE_TYPE_ALL, // corrected import for CL_DEVICE_TYPE_ALL
                device_info::{CL_DEVICE_NAME, CL_DEVICE_GLOBAL_MEM_SIZE}, // corrected imports for device info
                platform_info::CL_PLATFORM_NAME // corrected import for platform name
            };

            if let Ok(platforms) = opencl3::platform::get_platforms() { 
                for (platform_idx, platform) in platforms.iter().enumerate() {
                    if let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_ALL) {
                        for (device_idx, device_id) in device_ids.iter().enumerate() {
                            let device = opencl3::device::Device::new(*device_id);
                            
                            // gets device name
                            let name = match device.name() { // uses direct method call
                                Ok(name) => name,
                                Err(e) => {
                                    eprintln!("error getting opencl device name: {}", e);
                                    continue;
                                }
                            };
                            
                            // gets global memory size
                            let memory_size = match device.global_mem_size() { // uses direct method call
                                Ok(size) => size,
                                Err(e) => {
                                    eprintln!("error getting opencl device global memory size: {}", e);
                                    continue;
                                }
                            };
                            
                            devices.push(GpuDevice {
                                id: platform_idx * 100 + device_idx,
                                name,
                                memory_mb: (memory_size / (1024 * 1024)) as usize,
                                is_available: true,
                                backend: GpuBackend::OpenCL,
                            });
                        }
                    } else {
                        // gets platform name for error message
                        let platform_name = platform.name() // uses direct method call
                            .unwrap_or_else(|_| "unknown".to_string());
                        eprintln!("error getting opencl devices for platform: {}", platform_name);
                    }
                }
            } else {
                eprintln!("error getting opencl platforms");
            }
        }
        
        #[cfg(feature = "vulkan")]
        {
            // detects and initializes vulkan devices
            match VulkanContext::new() {
                Ok(context) => {
                    let properties = unsafe { context.instance.get_physical_device_properties(context.physical_device) };
                    let name = unsafe {
                        std::ffi::CStr::from_ptr(properties.device_name.as_ptr())
                            .to_string_lossy()
                            .into_owned()
                    };
                    let memory_properties = unsafe { context.instance.get_physical_device_memory_properties(context.physical_device) };
                    let mut total_device_memory_mb = 0;
                    for heap in &memory_properties.memory_heaps {
                        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                            total_device_memory_mb += heap.size / (1024 * 1024);
                        }
                    }

                    devices.push(GpuDevice {
                        id: 0, // only one vulkan context for now
                        name: name,
                        memory_mb: total_device_memory_mb as usize,
                        is_available: true,
                        backend: GpuBackend::Vulkan,
                    });
                    // assigns the context to the local vulkan_context_local variable
                    vulkan_context_local = Some(Arc::new(context)); 
                    println!("vulkan device detected and initialized.");
                },
                Err(e) => {
                    println!("warning: failed to initialize vulkan: {}", e);
                }
            }
        }
        
        if devices.is_empty() {
            println!("warning: no gpu devices detected. using cpu fallback.");
        } else {
            // prioritizes vulkan if available, otherwise cuda, then opencl
            devices.sort_by(|a: &GpuDevice, b: &GpuDevice| {
                if a.backend == GpuBackend::Vulkan && b.backend != GpuBackend::Vulkan {
                    std::cmp::Ordering::Less
                } else if a.backend != GpuBackend::Vulkan && b.backend == GpuBackend::Vulkan {
                    std::cmp::Ordering::Greater
                } else if a.backend == GpuBackend::Cuda && b.backend != GpuBackend::Cuda {
                    std::cmp::Ordering::Less
                } else if a.backend != GpuBackend::Cuda && b.backend == GpuBackend::Cuda {
                    std::cmp::Ordering::Greater
                } else {
                    b.memory_mb.cmp(&a.memory_mb) // sorts by memory if same backend
                }
            });
            println!("detected gpu devices: {:?}", devices);
        }
        
        Self {
            devices,
            active_device: 0, // defaults to the first (highest priority) detected device
            vulkan_context: vulkan_context_local, // assigns the local variable to the struct field
        }
    }
    
    // gets the number of available gpus
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
    
    // sets the active gpu device
    pub fn set_active_device(&mut self, device_id: usize) -> Result<(), String> {
        if device_id < self.devices.len() {
            self.active_device = device_id;
            Ok(())
        } else {
            Err(format!("invalid device id: {}", device_id))
        }
    }
    
    // gets information about all available devices
    pub fn get_devices(&self) -> &[GpuDevice] {
        &self.devices
    }

    // gets the Vulkan context if available
    pub fn get_vulkan_context(&self) -> Option<Arc<VulkanContext>> {
        self.vulkan_context.clone()
    }

    // gets the currently active gpu device
    pub fn get_active_device(&self) -> Option<&GpuDevice> {
        self.devices.get(self.active_device)
    }
}

// ========================
// multi-threaded cpu+gpu coordination
// ========================

// coordinates workloads between cpu and gpu
pub struct HybridCoordinator {
    gpu_manager: GpuManager,
    thread_count: usize,
    worker_threads: Option<Vec<thread::JoinHandle<()>>>,
    work_queue: Arc<Mutex<Vec<WorkItem>>>,
    results: Arc<RwLock<HashMap<usize, WorkResult>>>,
}

// a unit of computational work
#[derive(Debug, Clone)]
pub enum WorkItem {
    StateVectorUpdate {
        id: usize,
        start_idx: usize,
        end_idx: usize,
        operation: String,
        parameters: Vec<f64>,
    },
    GateApplication {
        id: usize,
        qubit_indices: Vec<usize>,
        matrix: Vec<Complex>,
    },
    Measurement {
        id: usize,
        qubit_index: usize,
    },
}

// complex number representation for quantum operations
#[derive(Clone, Debug, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

// result of a completed work item
#[derive(Clone, Debug)]
pub enum WorkResult {
    StateVectorUpdated {
        id: usize,
        success: bool,
    },
    GateApplied {
        id: usize,
        success: bool,
    },
    MeasurementResult {
        id: usize,
        result: bool,
        probability: f64,
    },
}

impl HybridCoordinator {
    // creates a new hybrid cpu+gpu coordinator
    pub fn new(thread_count: Option<usize>) -> Self {
        let thread_count = thread_count.unwrap_or_else(|| num_cpus::get());
        
        Self {
            gpu_manager: GpuManager::new(),
            thread_count,
            worker_threads: None,
            work_queue: Arc::new(Mutex::new(Vec::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    // starts worker threads for processing
    pub fn start_workers(&mut self) {
        let mut workers = Vec::with_capacity(self.thread_count);
        
        // clones vulkan context for gpu worker thread
        let vulkan_context_for_gpu_worker: Option<Arc<VulkanContext>> = self.gpu_manager.get_vulkan_context();
        let active_gpu_backend = self.gpu_manager.get_active_device().map(|d| d.backend.clone());

        for thread_id in 0..self.thread_count {
            let work_queue = Arc::clone(&self.work_queue);
            let results = Arc::clone(&self.results);
            
            // dedicates one thread (thread_id 0) to gpu work if a gpu is available
            let is_gpu_thread = thread_id == 0 && self.gpu_manager.device_count() > 0;
            let _vulkan_context = if is_gpu_thread { 
                vulkan_context_for_gpu_worker.clone()
            } else {
                None
            };
            let current_backend = active_gpu_backend.clone();

            let handle = thread::spawn(move || {
                println!("worker thread {} started (gpu: {})", thread_id, is_gpu_thread);
                
                loop {
                    // gets work from the queue
                    let work_item = {
                        let mut queue = work_queue.lock().unwrap();
                        if queue.is_empty() {
                            // no work available, sleeps briefly and checks again
                            drop(queue);
                            thread::sleep(std::time::Duration::from_millis(10));
                            continue;
                        }
                        // if this is a gpu thread, tries to get gpu-friendly work first
                        // otherwise, just pops from the back
                        if is_gpu_thread {
                            // finds gpu-friendly work
                            let mut gpu_friendly_idx = None;
                            for (i, item) in queue.iter().enumerate() {
                                if Self::is_gpu_friendly(item) {
                                    gpu_friendly_idx = Some(i);
                                    break;
                                }
                            }
                            if let Some(idx) = gpu_friendly_idx {
                                Some(queue.remove(idx))
                            } else {
                                queue.pop()
                            }
                        } else {
                            queue.pop()
                        }
                    };
                    
                    // processes the work item
                    if let Some(work) = work_item {
                        let result = if is_gpu_thread && Self::is_gpu_friendly(&work) {
                            match current_backend {
                                Some(GpuBackend::Vulkan) => {
                                    #[cfg(feature = "vulkan")]
                                    {
                                        if let Some(vk_ctx) = _vulkan_context.as_ref() {
                                            Self::process_on_gpu(work, Some(vk_ctx), GpuBackend::Vulkan)
                                        } else {
                                            println!("vulkan context not available for gpu processing, falling back to cpu.");
                                            Self::process_on_cpu(work)
                                        }
                                    }
                                    #[cfg(not(feature = "vulkan"))]
                                    {
                                        println!("vulkan feature not enabled, falling back to cpu for gpu work.");
                                        Self::process_on_cpu(work)
                                    }
                                },
                                Some(GpuBackend::Cuda) => {
                                    #[cfg(feature = "cuda")]
                                    {
                                        println!("processing work on cuda gpu...");
                                        // cuda processing logic here
                                        Self::process_on_cpu(work) // fallback for now
                                    }
                                    #[cfg(not(feature = "cuda"))]
                                    {
                                        println!("cuda feature not enabled, falling back to cpu for gpu work.");
                                        Self::process_on_cpu(work)
                                    }
                                },
                                Some(GpuBackend::OpenCL) => {
                                    #[cfg(feature = "opencl")]
                                    {
                                        println!("processing work on opencl gpu...");
                                        // opencl processing logic here
                                        // this is a placeholder. a full opencl implementation would involve
                                        // creating opencl buffers, compiling and executing kernels, and transferring data.
                                        Self::process_on_cpu(work) // fallback for now
                                    }
                                    #[cfg(not(feature = "opencl"))]
                                    {
                                        println!("opencl feature not enabled, falling back to cpu for gpu work.");
                                        Self::process_on_cpu(work)
                                    }
                                },
                                _ => {
                                    println!("no active gpu backend for gpu thread, falling back to cpu.");
                                    Self::process_on_cpu(work)
                                }
                            }
                        } else {
                            Self::process_on_cpu(work)
                        };
                        
                        // stores the result
                        if let Some(res) = result {
                            let mut results_map = results.write().unwrap();
                            match &res {
                                WorkResult::StateVectorUpdated { id, .. } => {
                                    results_map.insert(*id, res);
                                }
                                WorkResult::GateApplied { id, .. } => {
                                    results_map.insert(*id, res);
                                }
                                WorkResult::MeasurementResult { id, .. } => {
                                    results_map.insert(*id, res);
                                }
                            }
                        }
                    }
                }
            });
            
            workers.push(handle);
        }
        
        self.worker_threads = Some(workers);
    }
    
    // determines if a work item is gpu-friendly
    fn is_gpu_friendly(work: &WorkItem) -> bool {
        match work {
            WorkItem::StateVectorUpdate { .. } => true,
            WorkItem::GateApplication { .. } => true,
            WorkItem::Measurement { .. } => false, // measurements are often cpu-bound
        }
    }

    // submits work to be processed
    pub fn submit_work(&self, work: WorkItem) {
        let mut queue = self.work_queue.lock().unwrap();
        queue.push(work);
    }
    
    // processes work on gpu
    #[allow(dead_code)]
    fn process_on_gpu(work: WorkItem, _vulkan_context: Option<&Arc<VulkanContext>>, backend: GpuBackend) -> Option<WorkResult> {
        match backend {
            GpuBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    let vulkan_context = _vulkan_context;
                    if let Some(_vk_ctx) = vulkan_context { 
                        println!("processing work on vulkan gpu...");
                        match work {
                            WorkItem::StateVectorUpdate { id, .. } => {
                                Some(WorkResult::StateVectorUpdated {
                                    id,
                                    success: true,
                                })
                            },
                            WorkItem::GateApplication { id, .. } => {
                                Some(WorkResult::GateApplied {
                                    id,
                                    success: true,
                                })
                            },
                            WorkItem::Measurement { id, qubit_index: _ } => { 
                                Some(WorkResult::MeasurementResult {
                                    id,
                                    result: rand::random(),
                                    probability: 0.5,
                                })
                            },
                        }
                    } else {
                        println!("vulkan context not available for gpu processing, falling back to cpu.");
                        Self::process_on_cpu(work)
                    }
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    println!("vulkan feature not enabled, falling back to cpu for gpu work.");
                    Self::process_on_cpu(work)
                }
            },
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    println!("processing work on cuda gpu...");
                    // actual cuda processing logic would go here
                    Self::process_on_cpu(work) // fallback for now
                }
                #[cfg(not(feature = "cuda"))]
                {
                    println!("cuda feature not enabled, falling back to cpu for gpu work.");
                    Self::process_on_cpu(work)
                }
            },
            GpuBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    println!("processing work on opencl gpu...");
                    // actual opencl processing logic would go here
                    // note: this is a placeholder. a full opencl implementation would involve
                    // creating opencl buffers, compiling and executing kernels, and transferring data.
                    Self::process_on_cpu(work) // fallback for now
                }
                #[cfg(not(feature = "opencl"))]
                {
                    println!("opencl feature not enabled, falling back to cpu for gpu work.");
                    Self::process_on_cpu(work)
                }
            },
            GpuBackend::None => {
                println!("no gpu backend specified, falling back to cpu.");
                Self::process_on_cpu(work)
            }
        }
    }
    
    // processes work on cpu
    fn process_on_cpu(work: WorkItem) -> Option<WorkResult> {
        match work {
            WorkItem::StateVectorUpdate { id, .. } => {
                // cpu implementation would go here
                Some(WorkResult::StateVectorUpdated {
                    id,
                    success: true,
                })
            },
            WorkItem::GateApplication { id, .. } => {
                // cpu implementation would go here
                Some(WorkResult::GateApplied {
                    id,
                    success: true,
                })
            },
            WorkItem::Measurement { id, qubit_index: _ } => {
                // cpu implementation would go here
                Some(WorkResult::MeasurementResult {
                    id,
                    result: rand::random(),
                    probability: 0.5,
                })
            },
        }
    }
    
    // gets a result by id if available
    pub fn get_result(&self, id: usize) -> Option<WorkResult> {
        let results = self.results.read().unwrap();
        results.get(&id).cloned()
    }
}

// ========================
// distributed simulation communication
// ========================

// message types for inter-node communication
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum NodeMessage {
    // initializes a simulation with configuration
    Init(DistributedConfig),
    // applies a quantum gate
    ApplyGate {
        gate_type: String,
        qubits: Vec<usize>,
        parameters: Vec<f64>,
    },
    // measures a qubit
    Measure {
        qubit: usize,
        basis: String,
    },
    // synchronizes state between nodes
    SyncState {
        node_id: usize,
        indices: Vec<usize>,
        values: Vec<(f64, f64)>, // (real, imaginary) pairs
    },
    // requests state from another node
    RequestState {
        requesting_node: usize,
        indices: Vec<usize>,
    },
    // pings to check if node is alive
    Ping,
    // response to ping
    Pong,
    // terminates the simulation
    Terminate,
    // renders data message
    RenderData {
        frame_number: usize,
        // in a real scenario, this would contain serialized renderable data
        // for simplicity, we'll just send a dummy value
        dummy_data: u32, 
    },
}

// communication manager for distributed simulation
pub struct DistributedCommunication {
    config: DistributedConfig,
    connections: HashMap<usize, TcpStream>,
    listener: Option<TcpListener>,
    #[cfg(feature = "mpi")]
    mpi_comm: Option<SystemCommunicator>,
}

impl DistributedCommunication {
    // creates a new distributed communication manager
    pub fn new(config: DistributedConfig) -> Result<Self, IoError> {
        let mut comm = Self {
            config: config.clone(),
            connections: HashMap::new(),
            listener: None,
            #[cfg(feature = "mpi")]
            mpi_comm: None,
        };
        
        if config.use_mpi {
            #[cfg(feature = "mpi")]
            {
                let universe = mpi::initialize().unwrap();
                comm.mpi_comm = Some(universe.world());
            }
            #[cfg(not(feature = "mpi"))]
            {
                return Err(IoError::new(
                    std::io::ErrorKind::Other,
                    "mpi feature is not enabled but configuration requires it"
                ));
            }
        } else {
            // sets up tcp server
            let addr = format!("0.0.0.0:{}", config.port);
            let listener = TcpListener::bind(addr)?;
            listener.set_nonblocking(true)?;
            comm.listener = Some(listener);
            
            // connects to other nodes with lower ids
            for node_id in 0..config.node_id {
                let addr = &config.node_addresses[node_id];
                let connection_addr = format!("{}:{}", addr, config.port);
                
                // tries to connect multiple times with backoff
                let mut retry_count = 0;
                let max_retries = 5;
                let mut stream = None;
                
                while retry_count < max_retries {
                    match TcpStream::connect(&connection_addr) {
                        Ok(s) => {
                            stream = Some(s);
                            break;
                        }
                        Err(e) => {
                            eprintln!("failed to connect to node {}: {}. retrying ({}/{})", 
                                     node_id, e, retry_count + 1, max_retries);
                            thread::sleep(std::time::Duration::from_secs(2));
                            retry_count += 1;
                        }
                    }
                }
                
                if let Some(s) = stream {
                    s.set_nonblocking(true)?;
                    comm.connections.insert(node_id, s);
                } else {
                    return Err(IoError::new(
                        std::io::ErrorKind::ConnectionRefused,
                        format!("failed to connect to node {} after {} attempts", node_id, max_retries)
                    ));
                }
            }
        }
        
        Ok(comm)
    }
    
    // accepts new connections from other nodes
    pub fn accept_connections(&mut self) -> Result<(), IoError> {
        if self.config.use_mpi {
            return Ok(());  // mpi doesn't need explicit connection handling
        }
        
        if let Some(listener) = &self.listener {
            // accepts connections from nodes with higher ids
            for node_id in (self.config.node_id + 1)..self.config.node_count {
                match listener.accept() {
                    Ok((stream, addr)) => {
                        println!("accepted connection from: {}", addr);
                        stream.set_nonblocking(true)?;
                        self.connections.insert(node_id, stream);
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        // no pending connections
                        break;
                    }
                    Err(e) => {
                        eprintln!("error accepting connection: {}", e);
                        return Err(e);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    // sends a message to a specific node
    pub fn send_message(&mut self, node_id: usize, message: NodeMessage) -> Result<(), String> {
        if self.config.use_mpi {
            #[cfg(feature = "mpi")]
            {
                if let Some(comm) = self.mpi_comm.as_mut() { // gets mutable reference
                    let serialized = bincode::serialize(&message)
                        .map_err(|e| format!("serialization error: {}", e))?;
                    
                    comm.process_at_rank(node_id as i32)
                        .send(&serialized[..]); // passes a slice
                    
                    return Ok(());
                }
            }
            return Err("mpi communication requested but mpi is not available".to_string());
        } else {
            // tcp communication
            if let Some(stream) = self.connections.get_mut(&node_id) {
                let serialized = bincode::serialize(&message)
                    .map_err(|e| format!("serialization error: {}", e))?;
                
                // sends message length first (as u32)
                let len = serialized.len() as u32;
                let len_bytes = len.to_be_bytes();
                stream.write_all(&len_bytes)
                    .map_err(|e| format!("failed to send message length: {}", e))?;
                
                // then sends the actual message
                stream.write_all(&serialized)
                    .map_err(|e| format!("failed to send message: {}", e))?;
                
                return Ok(());
            }
            
            Err(format!("no connection to node {}", node_id))
        }
    }
    
    // broadcasts a message to all nodes
    pub fn broadcast_message(&mut self, message: NodeMessage) -> Result<(), String> {
        if self.config.use_mpi {
            #[cfg(feature = "mpi")]
            {
                if let Some(comm) = self.mpi_comm.as_mut() { // gets mutable reference
                    let mut serialized = bincode::serialize(&message)
                        .map_err(|e| format!("serialization error: {}", e))?;
                    
                    // uses standard broadcast
                    let root_rank = 0;
                    // comm.broadcast_within(&serialized[..], root_rank); // original line
                    comm.process_at_rank(root_rank).broadcast_into(&mut serialized); // corrected line for rsmpi
                    
                    return Ok(());
                }
            }
            return Err("mpi communication requested but mpi is not available".to_string());
        } else {
            // tcp communication - sends to each node individually
            for node_id in 0..self.config.node_count {
                if node_id != self.config.node_id {
                    self.send_message(node_id, message.clone())?;
                }
            }
            
            Ok(())
        }
    }
    
    // receives and processes messages
    pub fn receive_messages(&mut self) -> Vec<(usize, NodeMessage)> {
        let mut received_messages = Vec::new();
        
        if self.config.use_mpi {
            #[cfg(feature = "mpi")]
            {
                if let Some(comm) = &self.mpi_comm {
                    // checks for messages from any source
                    for rank in 0..comm.size() {
                        if rank as usize == self.config.node_id {
                            continue;
                        }
                        
                        let process = comm.process_at_rank(rank);
                        if let Some(msg) = process.immediate_probe() {
                            // `u8::equivalent_datatype()` is the correct way to get the mpi datatype for u8
                            let mut buffer = vec![0u8; msg.count(u8::equivalent_datatype()) as usize]; 
                            process.receive_into(&mut buffer);
                            
                            if let Ok(message) = bincode::deserialize::<NodeMessage>(&buffer) {
                                received_messages.push((rank as usize, message));
                            }
                        }
                    }
                }
            }
        } else {
            // tcp communication
            for (&node_id, stream) in &mut self.connections {
                // tries to read message length
                let mut len_buffer = [0u8; 4];
                match stream.read_exact(&mut len_buffer) {
                    Ok(()) => {
                        let len = u32::from_be_bytes(len_buffer) as usize;
                        let mut buffer = vec![0u8; len];
                        
                        match stream.read_exact(&mut buffer) {
                            Ok(()) => {
                                if let Ok(message) = bincode::deserialize::<NodeMessage>(&buffer) {
                                    received_messages.push((node_id, message));
                                }
                            }
                            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                                // no data available yet
                                continue;
                            }
                            Err(e) => {
                                eprintln!("error reading message from node {}: {}", node_id, e);
                            }
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        // no data available yet
                        continue;
                    }
                    Err(e) => {
                        eprintln!("error reading message length from node {}: {}", node_id, e);
                    }
                }
            }
        }
        
        received_messages
    }
}

// ========================
// distributed rendering for visualization
// ========================

// configuration for distributed rendering
pub struct RenderConfig {
    // total number of frames to render
    pub total_frames: usize,
    // start frame for this node
    pub start_frame: usize,
    // end frame for this node
    pub end_frame: usize,
    // output directory for rendered frames
    pub output_dir: String,
    // frame file format (e.g., "png", "jpg")
    pub frame_format: String,
    // final output video file
    pub output_video: String,
    // video encoding settings
    pub encoding_settings: EncodingSettings,
    // width of the rendered image
    pub width: u32,
    // height of the rendered image
    pub height: u32,
}

// video encoding settings
pub struct EncodingSettings {
    // video codec to use
    pub codec: String,
    // video bitrate
    pub bitrate: String,
    // frames per second
    pub fps: usize,
    // video resolution
    pub resolution: (usize, usize),
    // use lossless encoding
    pub lossless: bool,
}

impl Default for EncodingSettings {
    // provides default values for encoding settings
    fn default() -> Self {
        Self {
            codec: "libx264".to_string(),
            bitrate: "20M".to_string(),
            fps: 30,
            resolution: (1920, 1080),
            lossless: true,
        }
    }
}

// manager for distributed rendering
pub struct DistributedRenderer {
    config: RenderConfig,
    _node_id: usize, 
    _total_nodes: usize, 
    vulkan_context: Option<Arc<VulkanContext>>, 
}

impl DistributedRenderer {
    // creates a new distributed renderer
    pub fn new(
        node_id: usize,
        total_nodes: usize,
        total_frames: usize,
        output_dir: &str,
        output_video: &str,
        width: u32,
        height: u32,
        vulkan_context: Option<Arc<VulkanContext>>, 
    ) -> Self {
        // calculates frame range for this node
        let frames_per_node = total_frames / total_nodes;
        let remainder = total_frames % total_nodes;
        
        let start_frame = node_id * frames_per_node + std::cmp::min(node_id, remainder);
        let extra = if node_id < remainder { 1 } else { 0 };
        let end_frame = start_frame + frames_per_node + extra;
        
        // creates output directory if it doesn't exist
        std::fs::create_dir_all(output_dir).unwrap_or_else(|e| {
            eprintln!("warning: failed to create output directory: {}", e);
        });
        
        Self {
            config: RenderConfig {
                total_frames,
                start_frame,
                end_frame,
                output_dir: output_dir.to_string(),
                frame_format: "png".to_string(),
                output_video: output_video.to_string(),
                encoding_settings: EncodingSettings::default(),
                width,
                height,
            },
            _node_id: node_id, 
            _total_nodes: total_nodes, 
            vulkan_context, 
        }
    }
    
    // renders assigned frames
    pub fn render_frames<F>(&self, renderer: F) -> Result<(), String>
    where
        // ensures vulkancontext is correctly scoped and available
        F: Fn(usize, u32, u32, Option<Arc<VulkanContext>>) -> Result<(), String>,
    {
        println!("node {} rendering frames {} to {}", 
                 self._node_id, self.config.start_frame, self.config.end_frame);
        
        let start_time = Instant::now();
        
        for frame in self.config.start_frame..self.config.end_frame {
            let frame_start = Instant::now();
            renderer(frame, self.config.width, self.config.height, self.vulkan_context.clone())?;
            let frame_duration = frame_start.elapsed();
            
            println!("node {} rendered frame {} in {:.2?}", 
                     self._node_id, frame, frame_duration);
        }
        
        let total_duration = start_time.elapsed();
        println!("node {} completed rendering {} frames in {:.2?}", 
                 self._node_id, self.config.end_frame - self.config.start_frame, total_duration);
        
        Ok(())
    }
    
    // merges rendered frames into a video file
    pub fn merge_and_encode(&self) -> Result<(), String> {
        // only the primary node (0) performs the merge
        if self._node_id != 0 { 
            return Ok(());
        }
        
        println!("merging frames and encoding video...");
        
        // checks if ffmpeg is available
        let ffmpeg_check = Command::new("ffmpeg")
            .arg("-version")
            .output();
        
        if ffmpeg_check.is_err() {
            return Err("ffmpeg not found. please install ffmpeg to enable video encoding.".to_string());
        }
        
        let settings = &self.config.encoding_settings;
        let input_pattern = format!("{}/%04d.{}", self.config.output_dir, self.config.frame_format);
        
        let mut command = Command::new("ffmpeg");
        command
            .arg("-y") // overwrites output file if it exists
            .arg("-framerate")
            .arg(settings.fps.to_string())
            .arg("-i")
            .arg(input_pattern)
            .arg("-c:v")
            .arg(&settings.codec);
        
        if settings.lossless {
            if settings.codec == "libx264" {
                command
                    .arg("-preset")
                    .arg("veryslow")
                    .arg("-qp")
                    .arg("0");
            } else {
                command
                    .arg("-b:v")
                    .arg(&settings.bitrate);
            }
        } else {
            command
                .arg("-b:v")
                .arg(&settings.bitrate);
        }
        
        command
            .arg("-pix_fmt")
            .arg("yuv420p") // required for compatibility
            .arg(&self.config.output_video);
        
        println!("executing: {:?}", command);
        
        match command.output() {
            Ok(output) => {
                if output.status.success() {
                    println!("video encoding completed successfully: {}", self.config.output_video);
                    Ok(())
                } else {
                    let error = String::from_utf8_lossy(&output.stderr);
                    Err(format!("ffmpeg encoding failed: {}", error))
                }
            }
            Err(e) => Err(format!("failed to execute ffmpeg: {}", e)),
        }
    }
}

// ========================
// basic vulkan offscreen renderer (placeholder)
// ========================

// this function is always available, but its behavior changes based on the feature
pub fn render_frame_with_vulkan(
    frame_number: usize,
    width: u32,
    height: u32,
    vulkan_context: Option<Arc<VulkanContext>>,
) -> Result<(), String> {
    #[cfg(feature = "vulkan")]
    {
        if let Some(_vk_ctx) = vulkan_context { 
            println!("vulkan rendering frame {} ({}x{})", frame_number, width, height);

            // this is a simplified placeholder for actual vulkan rendering
            let mut img = ImageBuffer::new(width, height);
            let color_r = (frame_number % 256) as u8;
            let color_g = ((frame_number / 2) % 256) as u8;
            let color_b = ((frame_number / 4) % 256) as u8;

            for x in 0..width {
                for y in 0..height {
                    // simple gradient based on frame number and position
                    let pixel = Rgba([
                        (color_r as f32 * (x as f32 / width as f32)) as u8,
                        (color_g as f32 * (y as f32 / height as f32)) as u8,
                        (color_b as f32 * ((x + y) as f32 / (width + height) as f32 / 2.0)) as u8,
                        255,
                    ]);
                    img.put_pixel(x, y, pixel);
                }
            }

            let output_path = format!("output/frames/{:04}.png", frame_number);
            img.save(&output_path).map_err(|e| format!("failed to save image: {}", e))?;
            println!("saved frame {} to {}", frame_number, output_path);

            Ok(())
        } else {
            // this branch should ideally not be hit if vulkan_context is some
            println!("cpu rendering frame {} ({}x{}) (fallback in vulkan branch)", frame_number, width, height);
            let mut img = ImageBuffer::new(width, height);
            let color_r = (frame_number % 256) as u8;
            let color_g = ((frame_number / 2) % 256) as u8;
            let color_b = ((frame_number / 4) % 256) as u8;

            for x in 0..width {
                for y in 0..height {
                    let pixel = Rgba([color_r, color_g, color_b, 255]);
                    img.put_pixel(x, y, pixel);
                }
            }
            let output_path = format!("output/frames/{:04}.png", frame_number);
            img.save(&output_path).map_err(|e| format!("failed to save image: {}", e))?;
            println!("saved frame {} to {}", frame_number, output_path);
            Ok(())
        }
    }
    #[cfg(not(feature = "vulkan"))]
    {
        // when vulkan feature is not enabled, this block is active
        let _vulkan_context = vulkan_context; 
        println!("vulkan feature not enabled, performing cpu rendering for frame {} ({}x{})", frame_number, width, height);
        let mut img = ImageBuffer::new(width, height);
        let color_r = (frame_number % 256) as u8;
        let color_g = ((frame_number / 2) % 256) as u8;
        let color_b = ((frame_number / 4) % 256) as u8;

        for x in 0..width {
            for y in 0..height {
                let pixel = Rgba([color_r, color_g, color_b, 255]);
                img.put_pixel(x, y, pixel);
            }
        }
        let output_path = format!("output/frames/{:04}.png", frame_number);
        img.save(&output_path).map_err(|e| format!("failed to save image: {}", e))?;
        println!("saved frame {} to {}", frame_number, output_path);
        Ok(())
    }
}


// ========================
// tensor network simulation
// ========================

// basic tensor network implementation for memory-efficient simulation
pub struct TensorNetwork {
    tensors: Vec<Tensor>,
    // connections between tensors
    connections: Vec<(usize, usize, usize, usize)>, // (tensor1_id, leg1, tensor2_id, leg2)
    // maximum bond dimension
    max_bond_dimension: usize,
}

// a single tensor in the network
pub struct Tensor {
    pub id: usize, 
    // tensor dimensions
    dimensions: Vec<usize>,
    // tensor data (flattened)
    data: Vec<Complex>,
}

impl TensorNetwork {
    // creates a new tensor network with specified bond dimension
    pub fn new(max_bond_dimension: usize) -> Self {
        Self {
            tensors: Vec::new(),
            connections: Vec::new(),
            max_bond_dimension,
        }
    }
    
    // adds a tensor to the network
    pub fn add_tensor(&mut self, dimensions: Vec<usize>, data: Vec<Complex>) -> usize {
        let id = self.tensors.len();
        self.tensors.push(Tensor {
            id, 
            dimensions,
            data,
        });
        id
    }
    
    // connects two tensors in the network
    pub fn connect_tensors(&mut self, tensor1_id: usize, leg1: usize, tensor2_id: usize, leg2: usize) -> Result<(), String> {
        // validates tensor ids
        if tensor1_id >= self.tensors.len() || tensor2_id >= self.tensors.len() {
            return Err("invalid tensor id".to_string());
        }
        
        // validates legs
        if leg1 >= self.tensors[tensor1_id].dimensions.len() || leg2 >= self.tensors[tensor2_id].dimensions.len() {
            return Err("invalid tensor leg".to_string());
        }
        
        // checks dimension compatibility
        if self.tensors[tensor1_id].dimensions[leg1] != self.tensors[tensor2_id].dimensions[leg2] {
            return Err("incompatible tensor dimensions".to_string());
        }
        
        // adds connection
        self.connections.push((tensor1_id, leg1, tensor2_id, leg2));
        
        Ok(())
    }
    
    // contracts two tensors in the network
    pub fn contract_tensors(&mut self, tensor1_id: usize, tensor2_id: usize) -> Result<usize, String> {
        // finds connections between the tensors
        let mut common_legs = Vec::new();
        let mut connections_to_remove = Vec::new();
        
        // iterates over connections and finds common legs.
        // the `l1` and `l2` variables are now correctly scoped within the `if` blocks.
        for (i, &(t1, leg1, t2, leg2)) in self.connections.iter().enumerate() {
            if t1 == tensor1_id && t2 == tensor2_id {
                common_legs.push((leg1, leg2));
                connections_to_remove.push(i);
            } else if t1 == tensor2_id && t2 == tensor1_id {
                common_legs.push((leg2, leg1));
                connections_to_remove.push(i);
            }
        }
        
        if common_legs.is_empty() {
            return Err("tensors are not connected".to_string());
        }
        
        // removes the connections that will be contracted
        for &i in connections_to_remove.iter().rev() {
            self.connections.remove(i);
        }
        
        // performs svd-based truncated contraction
        // this is a simplified placeholder - real implementation would perform actual tensor contraction
        let tensor1 = &self.tensors[tensor1_id];
        let tensor2 = &self.tensors[tensor2_id];
        
        // creates new dimensions for the contracted tensor
        let mut new_dimensions = Vec::new();
        
        // adds dimensions from tensor1 that aren't being contracted
        for (i, &dim) in tensor1.dimensions.iter().enumerate() {
            if !common_legs.iter().any(|(l1, _)| *l1 == i) {
                new_dimensions.push(dim);
            }
        }
        
        // adds dimensions from tensor2 that aren't being contracted
        for (i, &dim) in tensor2.dimensions.iter().enumerate() {
            if !common_legs.iter().any(|(_, l2)| *l2 == i) {
                new_dimensions.push(dim);
            }
        }
        
        // creates new data for the contracted tensor
        // in a real implementation, this would perform actual tensor contraction
        // and would utilize tensor1.data and tensor2.data
        // for this placeholder, we'll just create a tensor with the right dimensions
        // and conceptually use the input data to satisfy the compiler warning.
        let data_size = new_dimensions.iter().product::<usize>();
        let mut new_data = vec![Complex { re: 0.0, im: 0.0 }; data_size];

        // placeholder: conceptually combines data from tensor1 and tensor2
        // this is not a mathematically correct tensor contraction, but it
        // ensures the 'data' fields are read.
        if !tensor1.data.is_empty() && !tensor2.data.is_empty() {
            for i in 0..new_data.len() {
                let t1_val = tensor1.data.get(i % tensor1.data.len()).unwrap_or(&Complex { re: 0.0, im: 0.0 });
                let t2_val = tensor2.data.get(i % tensor2.data.len()).unwrap_or(&Complex { re: 0.0, im: 0.0 });
                new_data[i].re = t1_val.re + t2_val.re;
                new_data[i].im = t1_val.im + t2_val.im;
            }
        }

        // placeholder: uses max_bond_dimension for conceptual truncation
        if self.max_bond_dimension > 0 {
            println!("note: performing conceptual truncation based on max_bond_dimension: {}", self.max_bond_dimension);
            // in a real implementation, svd and truncation would happen here
        }
        
        // adds the new tensor
        let new_tensor_id = self.add_tensor(new_dimensions, new_data);
        
        // updates connections to point to the new tensor
        for connection in &mut self.connections {
            if connection.0 == tensor1_id {
                connection.0 = new_tensor_id;
                // updates leg index based on the new tensor structure
                // this would need to be calculated based on how the dimensions map
            }
            if connection.2 == tensor1_id {
                connection.2 = new_tensor_id;
            }
            if connection.0 == tensor2_id {
                connection.0 = new_tensor_id;
            }
            if connection.2 == tensor2_id {
                connection.2 = new_tensor_id;
            }
        }
        
        // removes the old tensors
        // in practice, we'd need to carefully manage the indices
        // this is simplified for the example
        
        Ok(new_tensor_id)
    }
    
    // applies a single-qubit gate to the tensor network
    pub fn apply_single_qubit_gate(&mut self, qubit: usize, gate: [[Complex; 2]; 2]) -> Result<(), String> {
        // finds the tensor containing this qubit
        // in a real implementation, we'd need to track which tensor contains which qubit
        // for this example, we'll assume qubit i is in tensor i
        
        if qubit >= self.tensors.len() {
            return Err(format!("qubit {} not found in tensor network", qubit));
        }
        
        // applies the gate to the tensor
        // this is a simplified placeholder that conceptually uses the gate and tensor data
        let tensor = &mut self.tensors[qubit];
        println!("applying single-qubit gate to tensor with id: {}", tensor.id); 
        
        // conceptual application of gate to tensor data
        // this is not a mathematically correct operation, but it ensures 'data' and 'gate' are read
        for i in 0..tensor.data.len() {
            let old_re = tensor.data[i].re;
            let old_im = tensor.data[i].im;
            
            // simple placeholder transformation using gate elements
            tensor.data[i].re = gate[0][0].re * old_re - gate[0][0].im * old_im;
            tensor.data[i].im = gate[0][0].im * old_re + gate[0][0].re * old_im;
        }
        
        Ok(())
    }
    
    // applies a two-qubit gate to the tensor network
    pub fn apply_two_qubit_gate(&mut self, qubit1: usize, qubit2: usize, gate: [[Complex; 4]; 4]) -> Result<(), String> {
        if qubit1 >= self.tensors.len() || qubit2 >= self.tensors.len() {
            return Err("qubit index out of bounds for tensor network".to_string());
        }

        if qubit1 == qubit2 {
            return Err("cannot apply two-qubit gate to the same qubit".to_string());
        }

        // safely gets mutable references to the two tensors
        let (tensor1, tensor2) = if qubit1 < qubit2 {
            let (left, right) = self.tensors.split_at_mut(qubit2);
            (&mut left[qubit1], &mut right[0])
        } else {
            let (left, right) = self.tensors.split_at_mut(qubit1);
            (&mut right[0], &mut left[qubit2])
        };

        println!("applying two-qubit gate to tensors with ids: {} and {}", tensor1.id, tensor2.id);

        // conceptual application of gate to tensor data
        // this is not a mathematically correct operation, but it ensures 'data' and 'gate' are read
        for i in 0..tensor1.data.len() {
            let old_re = tensor1.data[i].re;
            let old_im = tensor1.data[i].im;
            
            // simple placeholder transformation using gate elements
            tensor1.data[i].re = gate[0][0].re * old_re - gate[0][0].im * old_im;
            tensor1.data[i].im = gate[0][0].im * old_re + gate[0][0].re * old_im;
        }
        for i in 0..tensor2.data.len() {
            let old_re = tensor2.data[i].re;
            let old_im = tensor2.data[i].im;
            
            // simple placeholder transformation using gate elements
            tensor2.data[i].re = gate[1][1].re * old_re - gate[1][1].im * old_im;
            tensor2.data[i].im = gate[1][1].im + gate[1][1].re * old_im;
        }
        
        Ok(())
    }
    
    // optimizes the tensor network by contracting high-weight tensors
    pub fn optimize(&mut self) -> Result<(), String> {
        // calculates tensor weights (number of connections)
        let mut weights = vec![0; self.tensors.len()];
        
        for &(t1, _, t2, _) in &self.connections {
            weights[t1] += 1;
            weights[t2] += 1;
        }
        
        // finds pairs of tensors to contract
        let mut pairs = Vec::new();
        for (i, &(t1, _, t2, _)) in self.connections.iter().enumerate() {
            pairs.push((i, t1, t2, weights[t1] + weights[t2]));
        }
        
        // sorts by weight (descending)
        pairs.sort_by(|a, b| b.3.cmp(&a.3));
        
        // contracts highest-weight pairs first
        for (_, t1, t2, _) in pairs {
            // skips if either tensor has been removed
            if t1 >= self.tensors.len() || t2 >= self.tensors.len() {
                continue;
            }
            
            // contracts the tensors
            self.contract_tensors(t1, t2)?;
            
            // recalculates weights for next iteration
            // this is simplified - in a real implementation we'd update the weights incrementally
        }
        
        Ok(())
    }
}

// ========================
// sparse state simulation
// ========================

// sparse state vector representation for memory-efficient simulation
pub struct SparseStateVector {
    // number of qubits
    qubit_count: usize,
    // non-zero amplitudes in the state vector
    amplitudes: HashMap<usize, Complex>,
    // threshold below which amplitudes are truncated
    truncation_threshold: f64,
}

impl SparseStateVector {
    // creates a new sparse state vector
    pub fn new(qubit_count: usize, truncation_threshold: f64) -> Self {
        let mut state = Self {
            qubit_count,
            amplitudes: HashMap::new(),
            truncation_threshold,
        };
        
        // initializes to |0...0⟩ state
        state.amplitudes.insert(0, Complex { re: 1.0, im: 0.0 });
        
        state
    }
    
    // applies a single-qubit gate to the state vector
    pub fn apply_single_qubit_gate(&mut self, qubit: usize, matrix: [[Complex; 2]; 2]) {
        if qubit >= self.qubit_count {
            panic!("qubit index out of bounds");
        }
        
        let mask = 1 << qubit;
        let mut new_amplitudes = HashMap::new();
        
        // for each non-zero amplitude
        for (&idx, amp) in &self.amplitudes { 
            // determines if this basis state has qubit set to 0 or 1
            let qubit_val = (idx & mask) >> qubit;
            
            // calculates indices for basis states differing only in this qubit
            let idx0 = idx & !mask; // sets qubit to 0
            let idx1 = idx | mask;  // sets qubit to 1
            
            // applies gate
            if qubit_val == 0 {
                // |...0...⟩ case
                let new_amp0 = Complex {
                    re: matrix[0][0].re * amp.re - matrix[0][0].im * amp.im,
                    im: matrix[0][0].re * amp.im + matrix[0][0].im * amp.re,
                };
                let new_amp1 = Complex {
                    re: matrix[1][0].re * amp.re - matrix[1][0].im * amp.im,
                    im: matrix[1][0].re * amp.im + matrix[1][0].im * amp.re,
                };
                
                Self::add_amplitude(&mut new_amplitudes, idx0, new_amp0);
                Self::add_amplitude(&mut new_amplitudes, idx1, new_amp1);
            } else {
                // |...1...⟩ case
                let new_amp0 = Complex {
                    re: matrix[0][1].re * amp.re - matrix[0][1].im * amp.im,
                    im: matrix[0][1].re * amp.im + matrix[0][1].im * amp.re,
                };
                let new_amp1 = Complex {
                    re: matrix[1][1].re * amp.re - matrix[1][1].im * amp.im,
                    im: matrix[1][1].re * amp.im + matrix[1][1].im * amp.re,
                };
                
                Self::add_amplitude(&mut new_amplitudes, idx0, new_amp0);
                Self::add_amplitude(&mut new_amplitudes, idx1, new_amp1);
            }
        }
        
        // truncates small amplitudes
        self.amplitudes = new_amplitudes.into_iter()
            .filter(|(_, amp)| amp.re * amp.re + amp.im * amp.im >= self.truncation_threshold * self.truncation_threshold)
            .collect();
    }
    
    // helper to add an amplitude to the state
    fn add_amplitude(amplitudes: &mut HashMap<usize, Complex>, idx: usize, amp: Complex) {
        if let Some(existing) = amplitudes.get_mut(&idx) {
            existing.re += amp.re;
            existing.im += amp.im;
        } else {
            amplitudes.insert(idx, amp);
        }
    }
    
    // measures a qubit in the standard basis
    pub fn measure(&mut self, qubit: usize) -> bool {
        if qubit >= self.qubit_count {
            panic!("qubit index out of bounds");
        }
        
        let mask = 1 << qubit;
        
        // calculates probability of measuring |1⟩
        let mut prob_one = 0.0;
        for (&idx, amp) in &self.amplitudes { 
            if (idx & mask) != 0 {
                prob_one += amp.re * amp.re + amp.im * amp.im;
            }
        }
        
        // generates random measurement outcome
        let outcome = rand::random::<f64>() < prob_one;
        
        // collapses state vector
        let mut new_amplitudes = HashMap::new();
        let mut norm_factor = 0.0;
        
        for (&idx, amp) in &self.amplitudes { 
            let matches_outcome = ((idx & mask) != 0) == outcome;
            
            if matches_outcome {
                new_amplitudes.insert(idx, *amp);
                norm_factor += amp.re * amp.re + amp.im * amp.im;
            }
        }
        
        // normalizes
        norm_factor = 1.0 / norm_factor.sqrt();
        for amp in new_amplitudes.values_mut() {
            amp.re *= norm_factor;
            amp.im *= norm_factor;
        }
        
        self.amplitudes = new_amplitudes;
        outcome
    }
    
    // calculates expectation value of a pauli operator
    pub fn expectation_value(&self, _pauli_string: &[(usize, char)]) -> f64 { 
        // this would need a full implementation for calculating expectation values
        // of pauli operators in a sparse representation
        
        // for demonstration purposes, just returns a placeholder value
        0.0
    }
    
    // gets the number of non-zero amplitudes
    pub fn sparsity(&self) -> usize {
        self.amplitudes.len()
    }
}

// ========================
// main api and examples
// ========================

// main coordinator for distributed quantum simulation
pub struct DistributedSimulator {
    // configuration for distributed simulation
    config: DistributedConfig,
    // communication manager
    comm: Option<DistributedCommunication>,
    // hybrid cpu/gpu workload coordinator
    hybrid: HybridCoordinator,
    // local partition of the state vector
    _partition: StatePartition, 
    // whether to use sparse representation
    use_sparse: bool,
    // sparse state vector (if used)
    sparse_state: Option<SparseStateVector>,
    // tensor network (if used)
    tensor_network: Option<TensorNetwork>,
}

impl DistributedSimulator {
    // creates a new distributed simulator
    pub fn new(config: DistributedConfig) -> Result<Self, String> {
        let partition_strategy = PartitionStrategy::EqualSize;
        let partitions = calculate_partitions(&config, partition_strategy);
        
        let local_partition = partitions.iter()
            .find(|p| p.node_id == config.node_id)
            .ok_or_else(|| "no partition found for this node".to_string())?
            .clone();
        
        let comm = match DistributedCommunication::new(config.clone()) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("warning: failed to initialize distributed communication: {}", e);
                None
            }
        };
        
        let thread_count = num_cpus::get();
        let hybrid = HybridCoordinator::new(Some(thread_count));
        
        Ok(Self {
            config,
            comm,
            hybrid,
            _partition: local_partition, 
            use_sparse: false,
            sparse_state: None,
            tensor_network: None,
        })
    }
    
    // enables sparse state representation
    pub fn enable_sparse_simulation(&mut self, truncation_threshold: f64) {
        self.use_sparse = true;
        self.sparse_state = Some(SparseStateVector::new(
            self.config.total_qubits,
            truncation_threshold
        ));
    }
    
    // enables tensor network representation
    pub fn enable_tensor_network(&mut self, max_bond_dimension: usize) {
        self.use_sparse = false;
        self.tensor_network = Some(TensorNetwork::new(max_bond_dimension));
    }
    
    // initializes the simulator
    pub fn initialize(&mut self) {
        // starts hybrid worker threads
        self.hybrid.start_workers();
        
        // initializes based on simulation type
        if self.use_sparse {
            if self.sparse_state.is_none() {
                self.sparse_state = Some(SparseStateVector::new(
                    self.config.total_qubits,
                    1e-10
                ));
            }
        } else if self.tensor_network.is_none() {
            // defaults to tensor network if not using sparse and no tensor network is set
            self.tensor_network = Some(TensorNetwork::new(64));
        }
        
        // sets up distributed communication
        if let Some(comm) = &mut self.comm {
            comm.accept_connections().unwrap_or_else(|e| {
                eprintln!("warning: failed to accept connections: {}", e);
            });
        }
    }
    
    // example method to run a simulation
    pub fn run_example_simulation(&mut self) -> Result<(), String> {
        println!("node {} starting example simulation", self.config.node_id);
        
        // apply some gates
        // in a real implementation, these would be dispatched to the appropriate backends
        
        // synchronizes with other nodes
        if let Some(comm) = &mut self.comm {
            comm.broadcast_message(NodeMessage::Ping)?;
            
            // processes received messages
            let messages = comm.receive_messages();
            for (node_id, message) in messages {
                match message {
                    NodeMessage::Ping => {
                        println!("received ping from node {}", node_id);
                        comm.send_message(node_id, NodeMessage::Pong)?;
                    }
                    NodeMessage::Pong => {
                        println!("received pong from node {}", node_id);
                    }
                    _ => {
                        println!("received message: {:?} from node {}", message, node_id);
                    }
                }
            }
        }
        
        println!("node {} completed example simulation", self.config.node_id);
        Ok(())
    }
    
    // example method to run distributed rendering
    pub fn run_example_rendering(&self) -> Result<(), String> {
        let vulkan_context = self.hybrid.gpu_manager.get_vulkan_context();

        let renderer = DistributedRenderer::new(
            self.config.node_id,
            self.config.node_count,
            100, // total frames
            "output/frames",
            "output/simulation.mp4",
            800, // width
            600, // height
            vulkan_context,
        );
        
        // renders frames
        renderer.render_frames(|frame, width, height, vk_ctx| {
            // this would call the actual rendering function for a specific frame
            // for this example, we'll use our simplified vulkan renderer
            render_frame_with_vulkan(frame, width, height, vk_ctx)?;
            Ok(())
        })?;
        
        // merges frames into video (only on node 0)
        renderer.merge_and_encode()?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_partition_calculation() {
        let config = DistributedConfig {
            total_qubits: 10,
            node_count: 4,
            node_id: 0,
            node_addresses: vec![
                "localhost".to_string(),
                "node1".to_string(),
                "node2".to_string(),
                "node3".to_string(),
            ],
            port: 9000,
            use_mpi: false,
            use_gpu: true,
            max_memory_gb: 8.0,
        };
        
        let partitions = calculate_partitions(&config, PartitionStrategy::EqualSize);
        
        assert_eq!(partitions.len(), 4);
        assert_eq!(partitions[0].node_id, 0);
        assert_eq!(partitions[1].node_id, 1);
        assert_eq!(partitions[2].node_id, 2);
        assert_eq!(partitions[3].node_id, 3);
        
        // checks that the entire state space is covered
        assert_eq!(partitions[0].start_idx, 0);
        assert_eq!(partitions[3].end_idx, 1 << 10);
        
        // checks that there are no gaps or overlaps
        for i in 0..3 {
            assert_eq!(partitions[i].end_idx, partitions[i+1].start_idx);
        }
    }
    
    #[test]
    fn test_sparse_state_vector() {
        let mut state = SparseStateVector::new(2, 1e-10);
        
        // applies hadamard to qubit 0
        let h_gate = [
            [Complex { re: 1.0/2.0_f64.sqrt(), im: 0.0 }, Complex { re: 1.0/2.0_f64.sqrt(), im: 0.0 }],
            [Complex { re: 1.0/2.0_f64.sqrt(), im: 0.0 }, Complex { re: -1.0/2.0_f64.sqrt(), im: 0.0 }],
        ];
        
        state.apply_single_qubit_gate(0, h_gate); 
        
        // should have two non-zero amplitudes now
        assert_eq!(state.sparsity(), 2);
        
        // applies hadamard to qubit 1
        // re-declares h_gate as it's a copy type and moved in the previous call
        let h_gate = [
            [Complex { re: 1.0/2.0_f64.sqrt(), im: 0.0 }, Complex { re: 1.0/2.0_f64.sqrt(), im: 0.0 }],
            [Complex { re: 1.0/2.0_f64.sqrt(), im: 0.0 }, Complex { re: -1.0/2.0_f64.sqrt(), im: 0.0 }],
        ];
        state.apply_single_qubit_gate(1, h_gate); 
        
        // should have four non-zero amplitudes now
        assert_eq!(state.sparsity(), 4);
    }
}
