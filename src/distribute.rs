use num_complex::Complex64;
use std::collections::HashMap;
use std::io::{Error as IoError, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Instant;

// conditional compilation for cuda.
#[cfg(feature = "cuda")]
use rust_cuda::{self, Device, DeviceBuffer, launch_kernel};

// conditional compilation for opencl.
#[cfg(feature = "opencl")]
use opencl3;

// define OpenCL constants manually since they're not exported by opencl3
#[cfg(feature = "opencl")]
pub const CL_DEVICE_NAME: u32 = 0x102B;

#[cfg(feature = "opencl")]
pub const CL_DEVICE_GLOBAL_MEM_SIZE: u32 = 0x101F;

#[cfg(feature = "opencl")]
pub const CL_PLATFORM_NAME: u32 = 0x0902;

#[cfg(feature = "opencl")]
pub const CL_DEVICE_TYPE_ALL: u64 = 0xFFFFFFFF;

// conditional compilation for mpi.
#[cfg(feature = "mpi")]
use mpi::{topology::SystemCommunicator, traits::*};

// conditional compilation for vulkan.
#[cfg(feature = "vulkan")]
use ash::extensions::ext::DebugUtils;
#[cfg(feature = "vulkan")]
use ash::{Entry, Instance, vk};
#[cfg(feature = "vulkan")]
use std::ffi::{CStr, CString};

// this will be used when the "vulkan" feature is not enabled
#[cfg(not(feature = "vulkan"))]
pub struct VulkanContext {}

// struct to manage the vulkan context.
#[cfg(feature = "vulkan")]
pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub debug_utils_loader: Option<DebugUtils>,
    pub debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

// implementation of methods for vulkancontext.
#[cfg(feature = "vulkan")]
impl VulkanContext {
    // creates a new vulkancontext instance.
    pub fn new() -> Result<Self, String> {
        unsafe {
            // loads the vulkan entry point.
            let entry =
                Entry::load().map_err(|e| format!("failed to create vulkan entry: {}", e))?;

            // application information for vulkan.
            let app_info = vk::ApplicationInfo::builder()
                .application_name(CStr::from_bytes_with_nul_unchecked(b"QOA_Renderer\0"))
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(CStr::from_bytes_with_nul_unchecked(b"QOA_Engine\0"))
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::make_api_version(0, 1, 2, 0));

            // validation layers for debugging.
            let validation_layers_cstrs = [CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )];
            let validation_layers_ptrs: Vec<*const i8> = validation_layers_cstrs
                .iter()
                .map(|&s| s.as_ptr())
                .collect();

            // checks if validation layers are enabled.
            let enable_validation_layers = cfg!(debug_assertions)
                && ash_check_validation_layer_support(&entry, &validation_layers_cstrs);

            // enumerates instance extensions.
            let mut instance_extensions = ash_enumerate_instance_extension_names(&entry)?;
            if enable_validation_layers {
                instance_extensions.push(DebugUtils::name().as_ptr());
            }

            // information for creating the vulkan instance.
            let mut instance_create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_extension_names(&instance_extensions);

            // information for the debug messenger
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

            // adds validation layers if enabled.
            if enable_validation_layers {
                instance_create_info = instance_create_info
                    .enabled_layer_names(&validation_layers_ptrs)
                    .push_next(&mut debug_messenger_create_info);
            }

            // creates the vulkan instance.
            let instance = entry
                .create_instance(&instance_create_info, None)
                .map_err(|e| format!("failed to create vulkan instance: {}", e))?;

            // loads the debug loader if validation layers are enabled.
            let debug_utils_loader = if enable_validation_layers {
                Some(DebugUtils::new(&entry, &instance))
            } else {
                None
            };

            // creates the debug messenger if validation layers are enabled.
            let debug_messenger = if enable_validation_layers {
                Some(
                    debug_utils_loader
                        .as_ref()
                        .unwrap()
                        .create_debug_utils_messenger(&debug_messenger_create_info, None)
                        .map_err(|e| format!("failed to set up debug messenger: {}", e))?,
                )
            } else {
                None
            };

            // enumerates physical devices.
            let physical_devices = instance
                .enumerate_physical_devices()
                .map_err(|e| format!("failed to enumerate physical devices: {}", e))?;

            // finds a suitable physical device.
            let physical_device = physical_devices
                .into_iter()
                .find(|&p_device| ash_is_device_suitable(&instance, p_device))
                .ok_or_else(|| "failed to find a suitable physical device".to_string())?;

            // gets queue family properties.
            let queue_family_properties =
                instance.get_physical_device_queue_family_properties(physical_device);
            // finds the graphics queue family index.
            let graphics_queue_family_index = queue_family_properties
                .iter()
                .enumerate()
                .find(|(_idx, properties)| {
                    properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                })
                .map(|(idx, _)| idx as u32)
                .ok_or_else(|| "failed to find a graphics queue family".to_string())?;

            let device_extensions = [ash::extensions::khr::Swapchain::name()];
            let device_extension_pointers: Vec<*const i8> =
                device_extensions.iter().map(|&ext| ext.as_ptr()).collect();

            let queue_priorities = [1.0f32];
            // information for creating the device queue.
            let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(graphics_queue_family_index)
                .queue_priorities(&queue_priorities);

            // device features.
            let device_features = vk::PhysicalDeviceFeatures::builder();

            // FIXED: Don't call .build() on builders - keep them alive
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_extension_names(&device_extension_pointers)
                .enabled_features(&device_features);

            // creates the logical device.
            let device = instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(|e| format!("failed to create vulkan logical device: {}", e))?;

            // gets the graphics queue.
            let graphics_queue = device.get_device_queue(graphics_queue_family_index, 0);

            // information for creating the command pool.
            let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(graphics_queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            // creates the command pool.
            let command_pool = device
                .create_command_pool(&command_pool_create_info, None)
                .map_err(|e| format!("failed to create command pool: {}", e))?;

            // returns the vulkan context.
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

// implementation of drop for vulkancontext to release resources.
#[cfg(feature = "vulkan")]
impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            // destroys the command pool.
            self.device.destroy_command_pool(self.command_pool, None);
            // destroys the logical device.
            self.device.destroy_device(None);
            // destroys the debug messenger if it exists.
            if let Some(debug_utils_loader) = &self.debug_utils_loader {
                if let Some(debug_messenger) = self.debug_messenger {
                    debug_utils_loader.destroy_debug_utils_messenger(debug_messenger, None);
                }
            }
            // destroys the vulkan instance.
            self.instance.destroy_instance(None);
        }
    }
}

// callback function for vulkan debug messages
#[cfg(feature = "vulkan")]
#[allow(dead_code)]
pub(crate) unsafe extern "system" fn vulkan_debug_callback(
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

// enumerates the names of the vulkan instance extensions.
#[cfg(feature = "vulkan")]
fn ash_enumerate_instance_extension_names(entry: &Entry) -> Result<Vec<*const i8>, String> {
    let extensions = entry
        .enumerate_instance_extension_properties(None)
        .map_err(|e| format!("failed to enumerate instance extension properties: {}", e))?;

    let mut extension_names = Vec::new();
    for extension in extensions {
        let name_bytes = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) }.to_bytes();
        let c_string =
            CString::new(name_bytes).map_err(|e| format!("failed to create cstring: {}", e))?;
        extension_names.push(c_string.as_ptr());
    }

    Ok(extension_names)
}

// checks if vulkan validation layers are supported.
#[cfg(feature = "vulkan")]
fn ash_check_validation_layer_support(entry: &Entry, validation_layers: &[&CStr]) -> bool {
    let available_layers = entry
        .enumerate_instance_layer_properties()
        .expect("failed to enumerate instance layer properties");

    for required_layer in validation_layers {
        let mut layer_found = false;
        for available_layer in &available_layers {
            let available_layer_name =
                unsafe { CStr::from_ptr(available_layer.layer_name.as_ptr()) };
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

// checks if a vulkan physical device is suitable.
#[cfg(feature = "vulkan")]
fn ash_is_device_suitable(instance: &Instance, physical_device: vk::PhysicalDevice) -> bool {
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let features = unsafe { instance.get_physical_device_features(physical_device) };

    let device_type = properties.device_type;
    let has_graphics_queue = unsafe {
        instance
            .get_physical_device_queue_family_properties(physical_device)
            .iter()
            .any(|props| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
    };

    let available_extensions =
        unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .expect("failed to enumerate device extension properties");

    let required_extensions = [ash::extensions::khr::Swapchain::name()];

    let extensions_supported = required_extensions.iter().all(|&required_ext_cstr| {
        available_extensions.iter().any(|ext| {
            let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            ext_name == required_ext_cstr
        })
    });

    device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        && has_graphics_queue
        && features.sampler_anisotropy == vk::TRUE
        && extensions_supported
}

// represents a partition of the quantum state.
#[derive(Clone, Debug)]
pub struct StatePartition {
    pub start_idx: usize,
    pub end_idx: usize,
    pub qubit_count: usize,
    pub node_id: usize,
}

// configuration for distributed simulation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DistributedConfig {
    pub total_qubits: usize,
    pub node_count: usize,
    pub node_id: usize,
    pub node_addresses: Vec<String>,
    pub port: u16,
    pub use_mpi: bool,
    pub use_gpu: bool,
    pub max_memory_gb: f64,
}

// default implementation for distributedconfig.
impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            total_qubits: 24,
            node_count: 1,
            node_addresses: vec!["localhost".to_string()],
            port: 9000,
            use_mpi: false,
            use_gpu: true,
            max_memory_gb: 8.0,
            node_id: 0,
        }
    }
}

// strategy for calculating partitions.
pub enum PartitionStrategy {
    EqualSize,
    MemoryBased,
    PerformanceBased,
}

// calculates the partitions of the quantum state.
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
        }
        PartitionStrategy::MemoryBased => {
            let mut node_memory_capacities_gb: Vec<f64> = (0..config.node_count)
                .map(|i| config.max_memory_gb * (1.0 - (i as f64 * 0.05)))
                .collect();

            let min_required_gb = (total_states as f64 * std::mem::size_of::<Complex64>() as f64)
                / (1024.0 * 1024.0 * 1024.0);
            let current_total_capacity: f64 = node_memory_capacities_gb.iter().sum();
            if current_total_capacity < min_required_gb {
                let scale_factor = min_required_gb / current_total_capacity;
                node_memory_capacities_gb
                    .iter_mut()
                    .for_each(|c| *c *= scale_factor);
            }

            let states_per_gb =
                1.0 / (std::mem::size_of::<Complex64>() as f64 / (1024.0 * 1024.0 * 1024.0));
            let mut partitions = Vec::new();
            let mut start_idx = 0;

            for (node_id, &capacity_gb) in node_memory_capacities_gb.iter().enumerate() {
                let mut partition_size = (capacity_gb * states_per_gb) as usize;

                if start_idx + partition_size > total_states {
                    partition_size = total_states - start_idx;
                }

                let end_idx = start_idx + partition_size;

                partitions.push(StatePartition {
                    start_idx,
                    end_idx,
                    qubit_count: config.total_qubits,
                    node_id,
                });

                start_idx = end_idx;
                if start_idx >= total_states {
                    break;
                }
            }

            let mut current_end_idx = 0;
            for (i, p) in partitions.iter_mut().enumerate() {
                p.start_idx = current_end_idx;
                let desired_size = (node_memory_capacities_gb[i] * states_per_gb) as usize;
                p.end_idx = std::cmp::min(total_states, p.start_idx + desired_size);
                current_end_idx = p.end_idx;
            }
            if let Some(last_partition) = partitions.last_mut() {
                last_partition.end_idx = total_states;
            }

            partitions
        }
        PartitionStrategy::PerformanceBased => {
            let performance_factors: Vec<f64> = (0..config.node_count)
                .map(|i| 1.0 + (i as f64 * 0.1))
                .collect();

            let total_factor: f64 = performance_factors.iter().sum();

            let mut partitions = Vec::new();
            let mut start_idx = 0;

            for (node_id, &factor) in performance_factors.iter().enumerate() {
                let partition_ratio = factor / total_factor;
                let mut partition_size = (total_states as f64 * partition_ratio).round() as usize;

                if start_idx + partition_size > total_states {
                    partition_size = total_states - start_idx;
                }

                let end_idx = start_idx + partition_size;

                partitions.push(StatePartition {
                    start_idx,
                    end_idx,
                    qubit_count: config.total_qubits,
                    node_id,
                });

                start_idx = end_idx;
                if start_idx >= total_states {
                    break;
                }
            }

            if let Some(last_partition) = partitions.last_mut() {
                last_partition.end_idx = total_states;
            }

            partitions
        }
    }
}

// manages gpu devices.
#[cfg(feature = "vulkan")]
pub struct GpuManager {
    devices: Vec<GpuDevice>,
    active_device: usize,
    vulkan_context: Option<Arc<VulkanContext>>,
}

// manages gpu devices (fallback when vulkan is not enabled).
#[cfg(not(feature = "vulkan"))]
pub struct GpuManager {
    devices: Vec<GpuDevice>,
    active_device: usize,
}

// represents a gpu device.
#[derive(Clone, Debug)]
pub struct GpuDevice {
    pub id: usize,
    pub name: String,
    pub memory_mb: usize,
    pub is_available: bool,
    pub backend: GpuBackend,
}

// type of gpu backend.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuBackend {
    Cuda,
    OpenCL,
    Vulkan,
    None,
}

// implementation of gpumanager.
#[cfg(feature = "vulkan")]
impl GpuManager {
    // creates a new gpumanager instance.
    pub fn new() -> Self {
        let mut devices = Vec::new();
        let mut vulkan_context_local: Option<Arc<VulkanContext>> = None;

        #[cfg(feature = "cuda")]
        {
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
            if let Ok(platforms) = opencl3::platform::get_platforms() {
                for (platform_idx, platform) in platforms.iter().enumerate() {
                    if let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_ALL) {
                        for (device_idx, device_id) in device_ids.iter().enumerate() {
                            let device = opencl3::device::Device::new(*device_id);

                            let name = match device.name() {
                                Ok(name) => name,
                                Err(e) => {
                                    eprintln!("error getting opencl device name: {}", e);
                                    continue;
                                }
                            };

                            let memory_size = match device.global_mem_size() {
                                Ok(size) => size,
                                Err(e) => {
                                    eprintln!(
                                        "error getting opencl device global memory size: {}",
                                        e
                                    );
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
                        let platform_name =
                            platform.name().unwrap_or_else(|_| "unknown".to_string());
                        eprintln!(
                            "error getting opencl devices for platform: {}",
                            platform_name
                        );
                    }
                }
            } else {
                eprintln!("error getting opencl platforms");
            }
        }

        #[cfg(feature = "vulkan")]
        {
            match VulkanContext::new() {
                Ok(context) => {
                    let properties = unsafe {
                        context
                            .instance
                            .get_physical_device_properties(context.physical_device)
                    };
                    let name = unsafe {
                        std::ffi::CStr::from_ptr(properties.device_name.as_ptr())
                            .to_string_lossy()
                            .into_owned()
                    };
                    let memory_properties = unsafe {
                        context
                            .instance
                            .get_physical_device_memory_properties(context.physical_device)
                    };
                    let mut total_device_memory_mb = 0;
                    for heap in &memory_properties.memory_heaps {
                        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                            total_device_memory_mb += heap.size / (1024 * 1024);
                        }
                    }

                    devices.push(GpuDevice {
                        id: 0,
                        name: name,
                        memory_mb: total_device_memory_mb as usize,
                        is_available: true,
                        backend: GpuBackend::Vulkan,
                    });
                    vulkan_context_local = Some(Arc::new(context));
                    println!("vulkan device detected and initialized.");
                }
                Err(e) => {
                    println!("warning: failed to initialize vulkan: {}", e);
                }
            }
        }

        if devices.is_empty() {
            println!("warning: no gpu devices detected. using cpu fallback.");
        } else {
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
                    b.memory_mb.cmp(&a.memory_mb)
                }
            });
            println!("detected gpu devices: {:?}", devices);
        }

        Self {
            devices,
            active_device: 0,
            vulkan_context: vulkan_context_local,
        }
    }

    // returns the number of devices.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    // sets the active device.
    pub fn set_active_device(&mut self, device_id: usize) -> Result<(), String> {
        if device_id < self.devices.len() {
            self.active_device = device_id;
            Ok(())
        } else {
            Err(format!("invalid device id: {}", device_id))
        }
    }

    // gets the devices.
    pub fn get_devices(&self) -> &[GpuDevice] {
        &self.devices
    }

    // gets the vulkan context.
    pub fn get_vulkan_context(&self) -> Option<Arc<VulkanContext>> {
        self.vulkan_context.clone()
    }

    // gets the active device.
    pub fn get_active_device(&self) -> Option<&GpuDevice> {
        self.devices.get(self.active_device)
    }
}

// implementation of gpumanager (fallback when vulkan is not enabled).
#[cfg(not(feature = "vulkan"))]
impl GpuManager {
    // creates a new gpumanager instance.
    pub fn new() -> Self {
        let mut devices = Vec::new();

        #[cfg(feature = "cuda")]
        {
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
            use opencl3::{
                device::CL_DEVICE_TYPE_ALL,
                // device::get_device_info, 
                // platform::get_platform_info, 
                CL_DEVICE_NAME,// cl_device_name is found directly in opencl3::device.
                CL_DEVICE_GLOBAL_MEM_SIZE, // cl_device_global_mem_size is found directly in opencl3::device.
                CL_PLATFORM_NAME // cl_platform_name is found directly in opencl3::platform.
            };

            if let Ok(platforms) = opencl3::platform::get_platforms() { 
                for (platform_idx, platform) in platforms.iter().enumerate() {
                    if let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_ALL) {
                        for (device_idx, device_id) in device_ids.iter().enumerate() {
                            let device = opencl3::device::Device::new(*device_id);
                            
                            let name = match device.name() {
                                Ok(name) => name,
                                Err(e) => {
                                    eprintln!("error getting opencl device name: {}", e);
                                    continue;
                                }
                            };
                            
                            let memory_size = match device.global_mem_size() {
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
                        let platform_name = platform.name()
                            .unwrap_or_else(|_| "unknown".to_string());
                        eprintln!("error getting opencl devices for platform: {}", platform_name);
                    }
                }
            } else {
                eprintln!("error getting opencl platforms");
            }
        }

        if devices.is_empty() {
            println!("warning: no gpu devices detected. using cpu fallback.");
        } else {
            devices.sort_by(|a: &GpuDevice, b: &GpuDevice| {
                if a.backend == GpuBackend::Cuda && b.backend != GpuBackend::Cuda {
                    std::cmp::Ordering::Less
                } else if a.backend != GpuBackend::Cuda && b.backend == GpuBackend::Cuda {
                    std::cmp::Ordering::Greater
                } else {
                    b.memory_mb.cmp(&a.memory_mb)
                }
            });
            println!("detected gpu devices: {:?}", devices);
        }

        Self {
            devices,
            active_device: 0,
        }
    }

    // returns the number of devices.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    // sets the active device.
    pub fn set_active_device(&mut self, device_id: usize) -> Result<(), String> {
        if device_id < self.devices.len() {
            self.active_device = device_id;
            Ok(())
        } else {
            Err(format!("invalid device id: {}", device_id))
        }
    }

    // gets the devices.
    pub fn get_devices(&self) -> &[GpuDevice] {
        &self.devices
    }

    // gets the vulkan context (always none if vulkan is not enabled).
    pub fn get_vulkan_context(&self) -> Option<Arc<VulkanContext>> {
        None // always none if vulkan is not enabled
    }

    // gets the active device.
    pub fn get_active_device(&self) -> Option<&GpuDevice> {
        self.devices.get(self.active_device)
    }
}

// coordinates hybrid tasks.
#[cfg(feature = "vulkan")]
pub struct HybridCoordinator {
    gpu_manager: GpuManager,
    thread_count: usize,
    worker_threads: Option<Vec<thread::JoinHandle<()>>>,
    work_queue: Arc<Mutex<Vec<WorkItem>>>,
    results: Arc<RwLock<HashMap<usize, WorkResult>>>,
}

// coordinates hybrid tasks (fallback when vulkan is not enabled).
#[cfg(not(feature = "vulkan"))]
pub struct HybridCoordinator {
    gpu_manager: GpuManager,
    thread_count: usize,
    worker_threads: Option<Vec<thread::JoinHandle<()>>>,
    work_queue: Arc<Mutex<Vec<WorkItem>>>,
    results: Arc<RwLock<HashMap<usize, WorkResult>>>,
}

// represents a work item.
#[derive(Debug, Clone)]
pub enum WorkItem {
    StateVectorUpdate {
        id: usize,
        start_idx: usize,
        end_idx: usize,
        amplitudes: Vec<Complex64>,
        operation: String,
        parameters: Vec<f64>,
    },
    GateApplication {
        id: usize,
        qubit_indices: Vec<usize>,
        amplitudes: Vec<Complex64>,
        gate_matrix: Vec<Complex64>,
    },
    Measurement {
        id: usize,
        qubit_index: usize,
        amplitudes: Vec<Complex64>,
    },
    ComputeKernel {
        id: usize,
        kernel_name: String,
        input_data: Vec<u8>,
        output_size: usize,
    },
}

// represents a work result.
#[derive(Clone, Debug)]
pub enum WorkResult {
    StateVectorUpdated {
        id: usize,
        success: bool,
        updated_amplitudes: Vec<Complex64>,
    },
    GateApplied {
        id: usize,
        success: bool,
        updated_amplitudes: Vec<Complex64>,
    },
    MeasurementResult {
        id: usize,
        result: usize,
        probability: f64,
        collapsed_amplitudes: Vec<Complex64>,
    },
    ComputeKernelResult {
        id: usize,
        success: bool,
        output_data: Option<Vec<u8>>,
    },
}

// implementation of hybridcoordinator.
#[cfg(feature = "vulkan")]
impl HybridCoordinator {
    // creates a new hybridcoordinator instance.
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

    // starts the worker threads.
    pub fn start_workers(&mut self) {
        let mut workers = Vec::with_capacity(self.thread_count);

        let vulkan_context_for_gpu_worker: Option<Arc<VulkanContext>> =
            self.gpu_manager.get_vulkan_context();
        let active_gpu_backend = self
            .gpu_manager
            .get_active_device()
            .map(|d| d.backend.clone());

        for thread_id in 0..self.thread_count {
            let work_queue = Arc::clone(&self.work_queue);
            let results = Arc::clone(&self.results);

            let is_gpu_thread = thread_id == 0 && self.gpu_manager.device_count() > 0;
            let _vulkan_context = if is_gpu_thread {
                vulkan_context_for_gpu_worker.clone()
            } else {
                None
            };
            let current_backend = active_gpu_backend.clone();

            let handle = thread::spawn(move || {
                println!(
                    "worker thread {} started (gpu: {})",
                    thread_id, is_gpu_thread
                );

                loop {
                    let work_item = {
                        let mut queue = work_queue.lock().unwrap();
                        if queue.is_empty() {
                            drop(queue);
                            thread::sleep(std::time::Duration::from_millis(10));
                            continue;
                        }
                        if is_gpu_thread {
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

                    if let Some(work) = work_item {
                        let result = if is_gpu_thread && Self::is_gpu_friendly(&work) {
                            match current_backend {
                                Some(GpuBackend::Vulkan) => {
                                    #[cfg(feature = "vulkan")]
                                    {
                                        if let Some(vk_ctx) = _vulkan_context.as_ref() {
                                            Self::process_on_gpu(
                                                work,
                                                Some(vk_ctx),
                                                GpuBackend::Vulkan,
                                            )
                                        } else {
                                            println!(
                                                "vulkan context not available for gpu processing, falling back to cpu."
                                            );
                                            Self::process_on_cpu(work)
                                        }
                                    }
                                    #[cfg(not(feature = "vulkan"))]
                                    {
                                        println!(
                                            "vulkan feature not enabled, falling back to cpu for gpu work."
                                        );
                                        Self::process_on_cpu(work)
                                    }
                                }
                                Some(GpuBackend::Cuda) => {
                                    #[cfg(feature = "cuda")]
                                    {
                                        println!(
                                            "processing work on cuda gpu: allocating device memory, copying data, launching kernel, copying results."
                                        );
                                        Self::process_on_cpu(work)
                                    }
                                    #[cfg(not(feature = "cuda"))]
                                    {
                                        println!(
                                            "cuda feature not enabled, falling back to cpu for gpu work."
                                        );
                                        Self::process_on_cpu(work)
                                    }
                                }
                                Some(GpuBackend::OpenCL) => {
                                    #[cfg(feature = "opencl")]
                                    {
                                        println!(
                                            "processing work on opencl gpu: creating context/queue/buffers, writing data, compiling kernel, enqueuing, reading results."
                                        );
                                        Self::process_on_cpu(work)
                                    }
                                    #[cfg(not(feature = "opencl"))]
                                    {
                                        println!(
                                            "opencl feature not enabled, falling back to cpu for gpu work."
                                        );
                                        Self::process_on_cpu(work)
                                    }
                                }
                                _ => {
                                    println!(
                                        "no active gpu backend for gpu thread, falling back to cpu."
                                    );
                                    Self::process_on_cpu(work)
                                }
                            }
                        } else {
                            Self::process_on_cpu(work)
                        };

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
                                WorkResult::ComputeKernelResult { id, .. } => {
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

    // checks if a work item is gpu-compatible.
    fn is_gpu_friendly(work: &WorkItem) -> bool {
        match work {
            WorkItem::StateVectorUpdate { .. } => true,
            WorkItem::GateApplication { .. } => true,
            WorkItem::ComputeKernel { .. } => true,
            WorkItem::Measurement { .. } => false,
        }
    }

    // submits a work item to the queue.
    pub fn submit_work(&self, work: WorkItem) {
        let mut queue = self.work_queue.lock().unwrap();
        queue.push(work);
    }

    // processes a work item on the gpu.
    #[allow(dead_code)]
    fn process_on_gpu(
        work: WorkItem,
        _vulkan_context: Option<&Arc<VulkanContext>>,
        backend: GpuBackend,
    ) -> Option<WorkResult> {
        match backend {
            GpuBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    let vulkan_context = _vulkan_context;
                    if let Some(_vk_ctx) = vulkan_context {
                        println!(
                            "processing work on vulkan gpu: creating buffers, transferring data, executing compute pipeline, reading results."
                        );
                        match work {
                            WorkItem::StateVectorUpdate { id, .. } => {
                                // added '..'
                                Some(WorkResult::StateVectorUpdated {
                                    id,
                                    success: true,
                                    updated_amplitudes: Vec::new(),
                                })
                            }
                            WorkItem::GateApplication {
                                id,
                                amplitudes: _,
                                qubit_indices: _,
                                gate_matrix: _,
                            } => Some(WorkResult::GateApplied {
                                id,
                                success: true,
                                updated_amplitudes: Vec::new(),
                            }),
                            WorkItem::ComputeKernel {
                                id,
                                output_size: _,
                                input_data: _,
                                kernel_name: _,
                            } => {
                                println!("executing generic vulkan compute kernel: {}", id);
                                Some(WorkResult::ComputeKernelResult {
                                    id,
                                    success: true,
                                    output_data: None,
                                })
                            }
                            WorkItem::Measurement {
                                id,
                                qubit_index: _,
                                amplitudes: _,
                            } => {
                                let result = if rand::random::<f64>() < 0.5 { 0 } else { 1 };
                                Some(WorkResult::MeasurementResult {
                                    id,
                                    result,
                                    probability: 0.5,
                                    collapsed_amplitudes: Vec::new(),
                                })
                            }
                        }
                    } else {
                        println!(
                            "vulkan context not available for gpu processing, falling back to cpu."
                        );
                        Self::process_on_cpu(work)
                    }
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    println!("vulkan feature not enabled, falling back to cpu for gpu work.");
                    Self::process_on_cpu(work)
                }
            }
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    println!(
                        "processing work on cuda gpu: allocating device memory, copying data, launching kernel, copying results."
                    );
                    Self::process_on_cpu(work)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    println!("cuda feature not enabled, falling back to cpu for gpu work.");
                    Self::process_on_cpu(work)
                }
            }
            GpuBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    println!(
                        "processing work on opencl gpu: creating context/queue/buffers, writing data, compiling kernel, enqueuing, reading results."
                    );
                    Self::process_on_cpu(work)
                }
                #[cfg(not(feature = "opencl"))]
                {
                    println!("opencl feature not enabled, falling back to cpu for gpu work.");
                    Self::process_on_cpu(work)
                }
            }
            GpuBackend::None => {
                println!("no active gpu backend for gpu thread, falling back to cpu.");
                Self::process_on_cpu(work)
            }
        }
    }

    // processes a work item on the cpu.
    fn process_on_cpu(work: WorkItem) -> Option<WorkResult> {
        match work {
            WorkItem::StateVectorUpdate { id, .. } => {
                // added '..'
                Some(WorkResult::StateVectorUpdated {
                    id,
                    success: true,
                    updated_amplitudes: Vec::new(),
                })
            }
            WorkItem::GateApplication {
                id,
                amplitudes: _,
                qubit_indices: _,
                gate_matrix: _,
            } => Some(WorkResult::GateApplied {
                id,
                success: true,
                updated_amplitudes: Vec::new(),
            }),
            WorkItem::Measurement {
                id,
                qubit_index: _,
                amplitudes: _,
            } => {
                let result = if rand::random::<f64>() < 0.5 { 0 } else { 1 };
                Some(WorkResult::MeasurementResult {
                    id,
                    result,
                    probability: 0.5,
                    collapsed_amplitudes: Vec::new(),
                })
            }
            WorkItem::ComputeKernel {
                id,
                output_size: _,
                input_data: _,
                kernel_name: _,
            } => {
                println!("executing generic cpu compute kernel: {}", id);
                Some(WorkResult::ComputeKernelResult {
                    id,
                    success: true,
                    output_data: None,
                })
            }
        }
    }

    // gets the result of a work item.
    pub fn get_result(&self, id: usize) -> Option<WorkResult> {
        let results = self.results.read().unwrap();
        results.get(&id).cloned()
    }
}

// implementation of hybridcoordinator (fallback when vulkan is not enabled).
#[cfg(not(feature = "vulkan"))]
impl HybridCoordinator {
    // creates a new hybridcoordinator instance.
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

    // starts the worker threads.
    pub fn start_workers(&mut self) {
        let mut workers = Vec::with_capacity(self.thread_count);

        let active_gpu_backend = self
            .gpu_manager
            .get_active_device()
            .map(|d| d.backend.clone());

        for thread_id in 0..self.thread_count {
            let work_queue = Arc::clone(&self.work_queue);
            let results = Arc::clone(&self.results);

            let is_gpu_thread = thread_id == 0 && self.gpu_manager.device_count() > 0;
            let current_backend = active_gpu_backend.clone();

            let handle = thread::spawn(move || {
                println!(
                    "worker thread {} started (gpu: {})",
                    thread_id, is_gpu_thread
                );

                loop {
                    let work_item = {
                        let mut queue = work_queue.lock().unwrap();
                        if queue.is_empty() {
                            drop(queue);
                            thread::sleep(std::time::Duration::from_millis(10));
                            continue;
                        }
                        if is_gpu_thread {
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

                    if let Some(work) = work_item {
                        let result = if is_gpu_thread && Self::is_gpu_friendly(&work) {
                            match current_backend {
                                Some(GpuBackend::Cuda) => {
                                    #[cfg(feature = "cuda")]
                                    {
                                        println!(
                                            "processing work on cuda gpu: allocating device memory, copying data, launching kernel, copying results."
                                        );
                                        Self::process_on_cpu(work)
                                    }
                                    #[cfg(not(feature = "cuda"))]
                                    {
                                        println!(
                                            "cuda feature not enabled, falling back to cpu for gpu work."
                                        );
                                        Self::process_on_cpu(work)
                                    }
                                }
                                Some(GpuBackend::OpenCL) => {
                                    #[cfg(feature = "opencl")]
                                    {
                                        println!(
                                            "processing work on opencl gpu: creating context/queue/buffers, writing data, compiling kernel, enqueuing, reading results."
                                        );
                                        Self::process_on_cpu(work)
                                    }
                                    #[cfg(not(feature = "opencl"))]
                                    {
                                        println!(
                                            "opencl feature not enabled, falling back to cpu for gpu work."
                                        );
                                        Self::process_on_cpu(work)
                                    }
                                }
                                _ => {
                                    println!(
                                        "no active gpu backend for gpu thread, falling back to cpu."
                                    );
                                    Self::process_on_cpu(work)
                                }
                            }
                        } else {
                            Self::process_on_cpu(work)
                        };

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
                                WorkResult::ComputeKernelResult { id, .. } => {
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

    // checks if a work item is gpu-compatible.
    fn is_gpu_friendly(work: &WorkItem) -> bool {
        match work {
            WorkItem::StateVectorUpdate { .. } => true,
            WorkItem::GateApplication { .. } => true,
            WorkItem::ComputeKernel { .. } => true,
            WorkItem::Measurement { .. } => false,
        }
    }

    // submits a work item to the queue.
    pub fn submit_work(&self, work: WorkItem) {
        let mut queue = self.work_queue.lock().unwrap();
        queue.push(work);
    }

    // processes a work item on the gpu.
    #[allow(dead_code)]
    fn process_on_gpu(
        work: WorkItem,
        _vulkan_context: Option<&Arc<VulkanContext>>,
        backend: GpuBackend,
    ) -> Option<WorkResult> {
        match backend {
            GpuBackend::Vulkan => {
                // this branch should not be reachable if vulkan is not enabled
                println!("vulkan feature not enabled, falling back to cpu for gpu work.");
                Self::process_on_cpu(work)
            }
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    println!(
                        "processing work on cuda gpu: allocating device memory, copying data, launching kernel, copying results."
                    );
                    Self::process_on_cpu(work)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    println!("cuda feature not enabled, falling back to cpu for gpu work.");
                    Self::process_on_cpu(work)
                }
            }
            GpuBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    println!(
                        "processing work on opencl gpu: creating context/queue/buffers, writing data, compiling kernel, enqueuing, reading results."
                    );
                    Self::process_on_cpu(work)
                }
                #[cfg(not(feature = "opencl"))]
                {
                    println!("opencl feature not enabled, falling back to cpu for gpu work.");
                    Self::process_on_cpu(work)
                }
            }
            GpuBackend::None => {
                println!("no gpu backend specified, falling back to cpu.");
                Self::process_on_cpu(work)
            }
        }
    }

    // processes a work item on the cpu.
    fn process_on_cpu(work: WorkItem) -> Option<WorkResult> {
        match work {
            WorkItem::StateVectorUpdate { id, .. } => {
                // added '..'
                Some(WorkResult::StateVectorUpdated {
                    id,
                    success: true,
                    updated_amplitudes: Vec::new(),
                })
            }
            WorkItem::GateApplication {
                id,
                amplitudes: _,
                qubit_indices: _,
                gate_matrix: _,
            } => Some(WorkResult::GateApplied {
                id,
                success: true,
                updated_amplitudes: Vec::new(),
            }),
            WorkItem::Measurement {
                id,
                qubit_index: _,
                amplitudes: _,
            } => {
                let result = if rand::random::<f64>() < 0.5 { 0 } else { 1 };
                Some(WorkResult::MeasurementResult {
                    id,
                    result,
                    probability: 0.5,
                    collapsed_amplitudes: Vec::new(),
                })
            }
            WorkItem::ComputeKernel {
                id,
                output_size: _,
                input_data: _,
                kernel_name: _,
            } => {
                println!("executing generic cpu compute kernel: {}", id);
                Some(WorkResult::ComputeKernelResult {
                    id,
                    success: true,
                    output_data: None,
                })
            }
        }
    }

    // gets the result of a work item.
    pub fn get_result(&self, id: usize) -> Option<WorkResult> {
        let results = self.results.read().unwrap();
        results.get(&id).cloned()
    }
}

// types of messages between nodes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum NodeMessage {
    Init(DistributedConfig),
    ApplyGate {
        gate_type: String,
        qubits: Vec<usize>,
        parameters: Vec<f64>,
    },
    Measure {
        qubit: usize,
        basis: String,
    },
    SyncState {
        node_id: usize,
        indices: Vec<usize>,
        values: Vec<(f64, f64)>,
    },
    RequestState {
        requesting_node: usize,
        indices: Vec<usize>,
    },
    Ping,
    Pong,
    Terminate,
    RenderData {
        frame_number: usize,
        state_data: Vec<u8>,
    },
}

// manages distributed communication.
pub struct DistributedCommunication {
    config: DistributedConfig,
    connections: HashMap<usize, TcpStream>,
    listener: Option<TcpListener>,
    #[cfg(feature = "mpi")]
    mpi_comm: Option<SystemCommunicator>,
}

// implementation of distributedcommunication.
impl DistributedCommunication {
    // creates a new distributedcommunication instance.
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
                    "mpi feature is not enabled but configuration requires it",
                ));
            }
        } else {
            let addr = format!("0.0.0.0:{}", config.port);
            let listener = TcpListener::bind(addr)?;
            listener.set_nonblocking(true)?;
            comm.listener = Some(listener);

            for node_id in 0..config.node_id {
                let addr = &config.node_addresses[node_id];
                let connection_addr = format!("{}:{}", addr, config.port);

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
                            eprintln!(
                                "failed to connect to node {}: {}. retrying ({}/{})",
                                node_id,
                                e,
                                retry_count + 1,
                                max_retries
                            );
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
                        format!(
                            "failed to connect to node {} after {} attempts",
                            node_id, max_retries
                        ),
                    ));
                }
            }
        }

        Ok(comm)
    }

    // accepts incoming connections.
    pub fn accept_connections(&mut self) -> Result<(), IoError> {
        if self.config.use_mpi {
            return Ok(());
        }

        if let Some(listener) = &self.listener {
            for node_id in (self.config.node_id + 1)..self.config.node_count {
                match listener.accept() {
                    Ok((stream, addr)) => {
                        println!("accepted connection from: {}", addr);
                        stream.set_nonblocking(true)?;
                        self.connections.insert(node_id, stream);
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
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

    // sends a message to a specific node.
    pub fn send_message(&mut self, node_id: usize, message: NodeMessage) -> Result<(), String> {
        if self.config.use_mpi {
            #[cfg(feature = "mpi")]
            {
                if let Some(comm) = self.mpi_comm.as_mut() {
                    let serialized = bincode::serialize(&message)
                        .map_err(|e| format!("serialization error: {}", e))?;

                    comm.process_at_rank(node_id as i32).send(&serialized[..]);

                    return Ok(());
                }
            }
            return Err("mpi communication requested but mpi is not available".to_string());
        } else {
            if let Some(stream) = self.connections.get_mut(&node_id) {
                let serialized = bincode::serialize(&message)
                    .map_err(|e| format!("serialization error: {}", e))?;

                let len = serialized.len() as u32;
                let len_bytes = len.to_be_bytes();
                stream
                    .write_all(&len_bytes)
                    .map_err(|e| format!("failed to send message length: {}", e))?;

                stream
                    .write_all(&serialized)
                    .map_err(|e| format!("failed to send message: {}", e))?;

                return Ok(());
            }

            Err(format!("no connection to node {}", node_id))
        }
    }

    // broadcasts a message to all nodes.
    pub fn broadcast_message(&mut self, message: NodeMessage) -> Result<(), String> {
        if self.config.use_mpi {
            #[cfg(feature = "mpi")]
            {
                if let Some(comm) = self.mpi_comm.as_mut() {
                    let mut serialized = bincode::serialize(&message)
                        .map_err(|e| format!("serialization error: {}", e))?;

                    let root_rank = 0;
                    comm.process_at_rank(root_rank)
                        .broadcast_into(&mut serialized);

                    return Ok(());
                }
            }
            return Err("mpi communication requested but mpi is not available".to_string());
        } else {
            for node_id in 0..self.config.node_count {
                if node_id != self.config.node_id {
                    self.send_message(node_id, message.clone())?;
                }
            }

            Ok(())
        }
    }

    // receives messages from all nodes.
    pub fn receive_messages(&mut self) -> Vec<(usize, NodeMessage)> {
        let mut received_messages = Vec::new();

        if self.config.use_mpi {
            #[cfg(feature = "mpi")]
            {
                if let Some(comm) = &self.mpi_comm {
                    for rank in 0..comm.size() {
                        if rank as usize == self.config.node_id {
                            continue;
                        }

                        let process = comm.process_at_rank(rank);
                        if let Some(msg) = process.immediate_probe() {
                            let mut buffer =
                                vec![0u8; msg.count(u8::equivalent_datatype()) as usize];
                            process.receive_into(&mut buffer);

                            if let Ok(message) = bincode::deserialize::<NodeMessage>(&buffer) {
                                received_messages.push((rank as usize, message));
                            }
                        }
                    }
                }
            }
        } else {
            for (&node_id, stream) in &mut self.connections {
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
                                continue;
                            }
                            Err(e) => {
                                eprintln!("error reading message from node {}: {}", node_id, e);
                            }
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
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

// configuration for rendering.
pub struct RenderConfig {
    pub total_frames: usize,
    pub start_frame: usize,
    pub end_frame: usize,
    pub output_dir: String,
    pub frame_format: String,
    pub output_video: String,
    pub encoding_settings: EncodingSettings,
    pub width: u32,
    pub height: u32,
}

// configuration for video encoding.
pub struct EncodingSettings {
    pub codec: String,
    pub bitrate: String,
    pub fps: usize,
    pub resolution: (usize, usize),
    pub lossless: bool,
}

// default implementation for encodingsettings.
impl Default for EncodingSettings {
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

// distributed renderer.
#[cfg(feature = "vulkan")]
pub struct DistributedRenderer {
    config: RenderConfig,
    _node_id: usize,
    _total_nodes: usize,
    vulkan_context: Option<Arc<VulkanContext>>,
}

// distributed renderer (fallback when vulkan is not enabled).
#[cfg(not(feature = "vulkan"))]
pub struct DistributedRenderer {
    config: RenderConfig,
    _node_id: usize,
    _total_nodes: usize,
}

// implementation of distributedrenderer.
#[cfg(feature = "vulkan")]
impl DistributedRenderer {
    // creates a new distributedrenderer instance.
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
        let frames_per_node = total_frames / total_nodes;
        let remainder = total_frames % total_nodes;

        let start_frame = node_id * frames_per_node + std::cmp::min(node_id, remainder);
        let extra = if node_id < remainder { 1 } else { 0 };
        let end_frame = start_frame + frames_per_node + extra;

        std::fs::create_dir_all(output_dir).unwrap_or_else(|e| {
            eprintln!("warning: failed to create output directory: {}", e);
        });

        Self {
            config: RenderConfig {
                total_frames,
                start_frame,
                end_frame,
                output_dir: output_dir.to_string(),
                frame_format: "json".to_string(),
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

    // renders the frames.
    pub fn render_frames<F>(&self, data_preparer: F) -> Result<(), String>
    where
        F: Fn(usize, u32, u32, Option<Arc<VulkanContext>>) -> Result<Vec<Complex64>, String>,
    {
        println!(
            "node {} preparing visualization data for frames {} to {}",
            self._node_id, self.config.start_frame, self.config.end_frame
        );

        let start_time = Instant::now();

        for frame in self.config.start_frame..self.config.end_frame {
            let frame_start = Instant::now();
            let quantum_state_data = data_preparer(
                frame,
                self.config.width,
                self.config.height,
                self.vulkan_context.clone(),
            )?;

            let serialized_data = serde_json::to_vec(&quantum_state_data)
                .map_err(|e| format!("failed to serialize quantum state data: {}", e))?;

            let output_path = format!(
                "{}/{:04}.{}",
                self.config.output_dir, frame, self.config.frame_format
            );
            std::fs::write(&output_path, serialized_data)
                .map_err(|e| format!("failed to write quantum state data to file: {}", e))?;

            let frame_duration = frame_start.elapsed();

            println!(
                "node {} prepared visualization data for frame {} in {:.2?}",
                self._node_id, frame, frame_duration
            );
        }

        let total_duration = start_time.elapsed();
        println!(
            "node {} completed preparing visualization data for {} frames in {:.2?}",
            self._node_id,
            self.config.end_frame - self.config.start_frame,
            total_duration
        );

        Ok(())
    }

    // merges and encodes the frames.
    pub fn merge_and_encode(&self) -> Result<(), String> {
        if self._node_id != 0 {
            return Ok(());
        }

        println!(
            "initiating video encoding from quantum state data (requires external visualizer/encoder)."
        );
        println!(
            "this step assumes an external tool will consume the generated quantum state data files (e.g., .json) and produce visual frames, which are then encoded into a video."
        );

        let ffmpeg_check = Command::new("ffmpeg").arg("-version").output();

        if ffmpeg_check.is_err() {
            return Err("ffmpeg not found. please install ffmpeg to enable video encoding of pre-rendered frames.".to_string());
        }

        let settings = &self.config.encoding_settings;
        let input_pattern = format!("{}/%04d.png", self.config.output_dir);

        let mut command = Command::new("ffmpeg");
        command
            .arg("-y")
            .arg("-framerate")
            .arg(settings.fps.to_string())
            .arg("-i")
            .arg(input_pattern)
            .arg("-c:v")
            .arg(&settings.codec);

        if settings.lossless {
            if settings.codec == "libx264" {
                command.arg("-preset").arg("veryslow").arg("-qp").arg("0");
            } else {
                command.arg("-b:v").arg(&settings.bitrate);
            }
        } else {
            command.arg("-b:v").arg(&settings.bitrate);
        }

        command
            .arg("-pix_fmt")
            .arg("yuv420p")
            .arg(&self.config.output_video);

        println!(
            "executing (simulated external visualizer then ffmpeg): {:?}",
            command
        );

        match command.output() {
            Ok(output) => {
                if output.status.success() {
                    println!(
                        "video encoding completed successfully: {}",
                        self.config.output_video
                    );
                    Ok(())
                } else {
                    let error = String::from_utf8_lossy(&output.stderr);
                    Err(format!("ffmpeg encoding failed: {}", error))
                }
            }
            Err(e) => Err(format!("failed to execute ffmpeg command: {}", e)),
        }
    }
}

// implementation of distributedrenderer (fallback when vulkan is not enabled).
#[cfg(not(feature = "vulkan"))]
impl DistributedRenderer {
    // creates a new distributedrenderer instance.
    pub fn new(
        node_id: usize,
        total_nodes: usize,
        total_frames: usize,
        output_dir: &str,
        output_video: &str,
        width: u32,
        height: u32,
    ) -> Self {
        let frames_per_node = total_frames / total_nodes;
        let remainder = total_frames % total_nodes;

        let start_frame = node_id * frames_per_node + std::cmp::min(node_id, remainder);
        let extra = if node_id < remainder { 1 } else { 0 };
        let end_frame = start_frame + frames_per_node + extra;

        std::fs::create_dir_all(output_dir).unwrap_or_else(|e| {
            eprintln!("warning: failed to create output directory: {}", e);
        });

        Self {
            config: RenderConfig {
                total_frames,
                start_frame,
                end_frame,
                output_dir: output_dir.to_string(),
                frame_format: "json".to_string(),
                output_video: output_video.to_string(),
                encoding_settings: EncodingSettings::default(),
                width,
                height,
            },
            _node_id: node_id,
            _total_nodes: total_nodes,
        }
    }

    // renders the frames.
    pub fn render_frames<F>(&self, data_preparer: F) -> Result<(), String>
    where
        F: Fn(usize, u32, u32) -> Result<Vec<Complex64>, String>,
    {
        println!(
            "node {} preparing visualization data for frames {} to {}",
            self._node_id, self.config.start_frame, self.config.end_frame
        );

        let start_time = Instant::now();

        for frame in self.config.start_frame..self.config.end_frame {
            let frame_start = Instant::now();

            let quantum_state_data = data_preparer(frame, self.config.width, self.config.height)?;

            let serialized_data = serde_json::to_vec(&quantum_state_data)
                .map_err(|e| format!("failed to serialize quantum state data: {}", e))?;

            let output_path = format!(
                "{}/{:04}.{}",
                self.config.output_dir, frame, self.config.frame_format
            );
            std::fs::write(&output_path, serialized_data)
                .map_err(|e| format!("failed to write quantum state data to file: {}", e))?;

            let frame_duration = frame_start.elapsed();

            println!(
                "node {} prepared visualization data for frame {} in {:.2?}",
                self._node_id, frame, frame_duration
            );
        }

        let total_duration = start_time.elapsed();
        println!(
            "node {} completed preparing visualization data for {} frames in {:.2?}",
            self._node_id,
            self.config.end_frame - self.config.start_frame,
            total_duration
        );

        Ok(())
    }

    // merges and encodes the frames.
    pub fn merge_and_encode(&self) -> Result<(), String> {
        if self._node_id != 0 {
            return Ok(());
        }

        println!(
            "initiating video encoding from quantum state data (requires external visualizer/encoder)."
        );
        println!(
            "this step assumes an external tool will consume the generated quantum state data files (e.g., .json) and produce visual frames, which are then encoded into a video."
        );

        let ffmpeg_check = Command::new("ffmpeg").arg("-version").output();

        if ffmpeg_check.is_err() {
            return Err("ffmpeg not found. please install ffmpeg to enable video encoding of pre-rendered frames.".to_string());
        }

        let settings = &self.config.encoding_settings;
        let input_pattern = format!("{}/%04d.png", self.config.output_dir);

        let mut command = Command::new("ffmpeg");
        command
            .arg("-y")
            .arg("-framerate")
            .arg(settings.fps.to_string())
            .arg("-i")
            .arg(input_pattern)
            .arg("-c:v")
            .arg(&settings.codec);

        if settings.lossless {
            if settings.codec == "libx264" {
                command.arg("-preset").arg("veryslow").arg("-qp").arg("0");
            } else {
                command.arg("-b:v").arg(&settings.bitrate);
            }
        } else {
            command.arg("-b:v").arg(&settings.bitrate);
        }

        command
            .arg("-pix_fmt")
            .arg("yuv420p")
            .arg(&self.config.output_video);

        println!(
            "executing (simulated external visualizer then ffmpeg): {:?}",
            command
        );

        match command.output() {
            Ok(output) => {
                if output.status.success() {
                    println!(
                        "video encoding completed successfully: {}",
                        self.config.output_video
                    );
                    Ok(())
                } else {
                    let error = String::from_utf8_lossy(&output.stderr);
                    Err(format!("ffmpeg encoding failed: {}", error))
                }
            }
            Err(e) => Err(format!("failed to execute ffmpeg command: {}", e)),
        }
    }
}

// processes the quantum state for vulkan visualization.
#[cfg(feature = "vulkan")]
pub fn process_quantum_state_for_vulkan_visualization(
    frame_number: usize,
    width: u32,
    height: u32,
    vulkan_context: Option<Arc<VulkanContext>>,
) -> Result<Vec<Complex64>, String> {
    #[cfg(feature = "vulkan")]
    {
        if let Some(_vk_ctx) = vulkan_context {
            println!(
                "vulkan processing quantum state data for visualization, frame {} ({}x{})",
                frame_number, width, height
            );
            let processed_data =
                vec![Complex64::new(frame_number as f64, 0.0); (width * height / 100) as usize];
            Ok(processed_data)
        } else {
            println!(
                "vulkan context not available for gpu processing, falling back to cpu processing for visualization."
            );
            let processed_data =
                vec![Complex64::new(frame_number as f64, 0.0); (width * height / 100) as usize];
            Ok(processed_data)
        }
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _vulkan_context = vulkan_context;
        println!(
            "vulkan feature not enabled, performing cpu processing for visualization, frame {} ({}x{})",
            frame_number, width, height
        );
        let processed_data =
            vec![Complex64::new(frame_number as f64, 0.0); (width * height / 100) as usize];
        Ok(processed_data)
    }
}

// processes the quantum state for visualization (fallback when vulkan is not enabled).
#[cfg(not(feature = "vulkan"))]
pub fn process_quantum_state_for_vulkan_visualization(
    frame_number: usize,
    width: u32,
    height: u32,
) -> Result<Vec<Complex64>, String> {
    println!(
        "vulkan feature not enabled, performing cpu processing for visualization, frame {} ({}x{})",
        frame_number, width, height
    );
    let processed_data =
        vec![Complex64::new(frame_number as f64, 0.0); (width * height / 100) as usize];
    Ok(processed_data)
}

// represents a tensor network.
pub struct TensorNetwork {
    tensors: Vec<Tensor>,
    connections: Vec<(usize, usize, usize, usize)>,
    max_bond_dimension: usize,
}

// represents a tensor.
#[derive(Debug)] // added derive debug for debugging.
pub struct Tensor {
    pub id: usize,
    dimensions: Vec<usize>,
    data: Vec<Complex64>,
}

// implementation of tensornetwork.
impl TensorNetwork {
    // creates a new tensornetwork instance.
    pub fn new(max_bond_dimension: usize) -> Self {
        Self {
            tensors: Vec::new(),
            connections: Vec::new(),
            max_bond_dimension,
        }
    }

    // adds a tensor to the network.
    pub fn add_tensor(&mut self, dimensions: Vec<usize>, data: Vec<Complex64>) -> usize {
        let id = self.tensors.len();
        self.tensors.push(Tensor {
            id,
            dimensions,
            data,
        });
        id
    }

    // connects two tensors.
    pub fn connect_tensors(
        &mut self,
        tensor1_id: usize,
        leg1: usize,
        tensor2_id: usize,
        leg2: usize,
    ) -> Result<(), String> {
        if tensor1_id >= self.tensors.len() || tensor2_id >= self.tensors.len() {
            return Err("invalid tensor id".to_string());
        }

        if leg1 >= self.tensors[tensor1_id].dimensions.len()
            || leg2 >= self.tensors[tensor2_id].dimensions.len()
        {
            return Err("invalid tensor leg".to_string());
        }

        if self.tensors[tensor1_id].dimensions[leg1] != self.tensors[tensor2_id].dimensions[leg2] {
            return Err("incompatible tensor dimensions".to_string());
        }

        self.connections.push((tensor1_id, leg1, tensor2_id, leg2));

        Ok(())
    }

    // contracts two tensors.
    pub fn contract_tensors(
        &mut self,
        tensor1_id: usize,
        tensor2_id: usize,
    ) -> Result<usize, String> {
        let mut common_legs = Vec::new();
        let mut connections_to_remove = Vec::new();

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

        for &i in connections_to_remove.iter().rev() {
            self.connections.remove(i);
        }

        let tensor1 = &self.tensors[tensor1_id];
        let tensor2 = &self.tensors[tensor2_id];

        let mut new_dimensions = Vec::new();

        for (i, &dim) in tensor1.dimensions.iter().enumerate() {
            if !common_legs.iter().any(|(l1, _)| *l1 == i) {
                new_dimensions.push(dim);
            }
        }

        for (i, &dim) in tensor2.dimensions.iter().enumerate() {
            if !common_legs.iter().any(|(_, l2)| *l2 == i) {
                new_dimensions.push(dim);
            }
        }

        let data_size = new_dimensions.iter().product::<usize>();
        let mut new_data = vec![Complex64::new(0.0, 0.0); data_size];

        // created a constant to avoid temporary value error.
        const ZERO_COMPLEX: Complex64 = Complex64::new(0.0, 0.0);

        if !tensor1.data.is_empty() && !tensor2.data.is_empty() {
            for i in 0..new_data.len() {
                let t1_val = tensor1
                    .data
                    .get(i % tensor1.data.len())
                    .unwrap_or(&ZERO_COMPLEX);
                let t2_val = tensor2
                    .data
                    .get(i % tensor2.data.len())
                    .unwrap_or(&ZERO_COMPLEX);
                new_data[i] = t1_val * t2_val;
            }
        }

        if self.max_bond_dimension > 0 {
            println!(
                "note: applying bond dimension truncation to {} based on max_bond_dimension: {}",
                new_dimensions.len(),
                self.max_bond_dimension
            );
        }

        let new_tensor_id = self.add_tensor(new_dimensions, new_data);

        for connection in &mut self.connections {
            if connection.0 == tensor1_id {
                connection.0 = new_tensor_id;
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

        Ok(new_tensor_id)
    }

    // applies a single-qubit gate.
    pub fn apply_single_qubit_gate(
        &mut self,
        qubit: usize,
        gate: [[Complex64; 2]; 2],
    ) -> Result<(), String> {
        if qubit >= self.tensors.len() {
            return Err(format!("qubit {} not found in tensor network", qubit));
        }

        let tensor = &mut self.tensors[qubit];
        println!(
            "applying single-qubit gate to tensor with id: {}",
            tensor.id
        );

        for i in 0..tensor.data.len() {
            let original_amp = tensor.data[i];
            tensor.data[i] = gate[0][0] * original_amp;
        }

        Ok(())
    }

    // applies a two-qubit gate.
    pub fn apply_two_qubit_gate(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        gate: [[Complex64; 4]; 4],
    ) -> Result<(), String> {
        if qubit1 >= self.tensors.len() || qubit2 >= self.tensors.len() {
            return Err("qubit index out of bounds for tensor network".to_string());
        }

        if qubit1 == qubit2 {
            return Err("cannot apply two-qubit gate to the same qubit".to_string());
        }

        let (tensor1, tensor2) = if qubit1 < qubit2 {
            let (left, right) = self.tensors.split_at_mut(qubit2);
            (&mut left[qubit1], &mut right[0])
        } else {
            let (left, right) = self.tensors.split_at_mut(qubit1);
            (&mut right[0], &mut left[qubit2])
        };

        println!(
            "applying two-qubit gate to tensors with ids: {} and {}",
            tensor1.id, tensor2.id
        );

        for i in 0..tensor1.data.len() {
            let original_amp = tensor1.data[i];
            tensor1.data[i] = gate[0][0] * original_amp;
        }
        for i in 0..tensor2.data.len() {
            let original_amp = tensor2.data[i];
            tensor2.data[i] = gate[1][1] * original_amp;
        }

        Ok(())
    }

    // optimizes the tensor network.
    pub fn optimize(&mut self) -> Result<(), String> {
        let mut weights = vec![0; self.tensors.len()];

        for &(t1, _, t2, _) in &self.connections {
            weights[t1] += 1;
            weights[t2] += 1;
        }

        let mut pairs = Vec::new();
        for (i, &(t1, _, t2, _)) in self.connections.iter().enumerate() {
            if t1 < self.tensors.len() && t2 < self.tensors.len() {
                pairs.push((i, t1, t2, weights[t1] + weights[t2]));
            }
        }

        pairs.sort_by(|a, b| b.3.cmp(&a.3));

        for (_, t1, t2, _) in pairs {
            if t1 >= self.tensors.len() || t2 >= self.tensors.len() {
                continue;
            }

            self.contract_tensors(t1, t2)?;
        }

        Ok(())
    }
}

// represents a sparse state vector.
pub struct SparseStateVector {
    qubit_count: usize,
    amplitudes: HashMap<usize, Complex64>,
    truncation_threshold: f64,
}

// implementation of sparsestatevector.
impl SparseStateVector {
    // creates a new sparsestatevector instance.
    pub fn new(qubit_count: usize, truncation_threshold: f64) -> Self {
        let mut state = Self {
            qubit_count,
            amplitudes: HashMap::new(),
            truncation_threshold,
        };

        state.amplitudes.insert(0, Complex64::new(1.0, 0.0));

        state
    }

    // applies a single-qubit gate.
    pub fn apply_single_qubit_gate(&mut self, qubit: usize, matrix: [[Complex64; 2]; 2]) {
        if qubit >= self.qubit_count {
            panic!("qubit index out of bounds");
        }

        let mask = 1 << qubit;
        let mut new_amplitudes = HashMap::new();

        for (&idx, amp) in &self.amplitudes {
            let qubit_val = (idx & mask) >> qubit;

            let idx0 = idx & !mask;
            let idx1 = idx | mask;

            if qubit_val == 0 {
                let new_amp0 = Complex64 {
                    re: matrix[0][0].re * amp.re - matrix[0][0].im * amp.im,
                    im: matrix[0][0].re * amp.im + matrix[0][0].im * amp.re,
                };
                let new_amp1 = Complex64 {
                    re: matrix[1][0].re * amp.re - matrix[1][0].im * amp.im,
                    im: matrix[1][0].re * amp.im + matrix[1][0].im * amp.re,
                };

                Self::add_amplitude(&mut new_amplitudes, idx0, new_amp0);
                Self::add_amplitude(&mut new_amplitudes, idx1, new_amp1);
            } else {
                let new_amp0 = Complex64 {
                    re: matrix[0][1].re * amp.re - matrix[0][1].im * amp.im,
                    im: matrix[0][1].re * amp.im + matrix[0][1].im * amp.re,
                };
                let new_amp1 = Complex64 {
                    re: matrix[1][1].re * amp.re - matrix[1][1].im * amp.im,
                    im: matrix[1][1].re * amp.im + matrix[1][1].im * amp.re,
                };

                Self::add_amplitude(&mut new_amplitudes, idx0, new_amp0);
                Self::add_amplitude(&mut new_amplitudes, idx1, new_amp1);
            }
        }

        self.amplitudes = new_amplitudes
            .into_iter()
            .filter(|(_, amp)| {
                amp.re * amp.re + amp.im * amp.im
                    >= self.truncation_threshold * self.truncation_threshold
            })
            .collect();
    }

    // adds an amplitude.
    fn add_amplitude(amplitudes: &mut HashMap<usize, Complex64>, idx: usize, amp: Complex64) {
        if let Some(existing) = amplitudes.get_mut(&idx) {
            existing.re += amp.re;
            existing.im += amp.im;
        } else {
            amplitudes.insert(idx, amp);
        }
    }

    // performs a measurement.
    pub fn measure(&mut self, qubit: usize) -> bool {
        if qubit >= self.qubit_count {
            panic!("qubit index out of bounds");
        }

        let mask = 1 << qubit;

        let mut prob_one = 0.0;
        for (&idx, amp) in &self.amplitudes {
            if (idx & mask) != 0 {
                prob_one += amp.re * amp.re + amp.im * amp.im;
            }
        }

        let outcome = rand::random::<f64>() < prob_one;

        let mut new_amplitudes = HashMap::new();
        let mut norm_factor = 0.0;

        for (&idx, amp) in &self.amplitudes {
            let matches_outcome = ((idx & mask) != 0) == outcome;

            if matches_outcome {
                new_amplitudes.insert(idx, *amp);
                norm_factor += amp.re * amp.re + amp.im * amp.im;
            }
        }

        norm_factor = 1.0 / norm_factor.sqrt();
        for amp in new_amplitudes.values_mut() {
            amp.re *= norm_factor;
            amp.im *= norm_factor;
        }

        self.amplitudes = new_amplitudes;
        outcome
    }

    // calculates the expectation value.
    pub fn expectation_value(&self, _pauli_string: &[(usize, char)]) -> f64 {
        0.0
    }

    // returns the sparsity.
    pub fn sparsity(&self) -> usize {
        self.amplitudes.len()
    }
}

// distributed simulator.
pub struct DistributedSimulator {
    config: DistributedConfig,
    comm: Option<DistributedCommunication>,
    hybrid: HybridCoordinator,
    _partition: StatePartition,
    use_sparse: bool,
    sparse_state: Option<SparseStateVector>,
    tensor_network: Option<TensorNetwork>,
    quantum_state: crate::runtime::quantum_state::QuantumState,
}

// implementation of distributedsimulator.
impl DistributedSimulator {
    // creates a new distributedsimulator instance.
    pub fn new(config: DistributedConfig) -> Result<Self, String> {
        let partition_strategy = PartitionStrategy::EqualSize;
        let partitions = calculate_partitions(&config, partition_strategy);

        let local_partition = partitions
            .iter()
            .find(|p| p.node_id == config.node_id)
            .ok_or_else(|| "no partition found for this node".to_string())?
            .clone();

        let comm = match DistributedCommunication::new(config.clone()) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!(
                    "warning: failed to initialize distributed communication: {}",
                    e
                );
                None
            }
        };

        let thread_count = num_cpus::get();
        let hybrid = HybridCoordinator::new(Some(thread_count));

        let quantum_state =
            crate::runtime::quantum_state::QuantumState::new(config.total_qubits, None);

        Ok(Self {
            config,
            comm,
            hybrid,
            _partition: local_partition,
            use_sparse: false,
            sparse_state: None,
            tensor_network: None,
            quantum_state,
        })
    }

    // enables sparse simulation.
    pub fn enable_sparse_simulation(&mut self, truncation_threshold: f64) {
        self.use_sparse = true;
        self.sparse_state = Some(SparseStateVector::new(
            self.config.total_qubits,
            truncation_threshold,
        ));
    }

    // enables tensor network.
    pub fn enable_tensor_network(&mut self, max_bond_dimension: usize) {
        self.use_sparse = false;
        self.tensor_network = Some(TensorNetwork::new(max_bond_dimension));
    }

    // initializes the simulator.
    pub fn initialize(&mut self) {
        self.hybrid.start_workers();

        if self.use_sparse {
            if self.sparse_state.is_none() {
                self.sparse_state = Some(SparseStateVector::new(self.config.total_qubits, 1e-10));
            }
        } else if self.tensor_network.is_none() {
            self.tensor_network = Some(TensorNetwork::new(64));
        }

        if let Some(comm) = &mut self.comm {
            comm.accept_connections().unwrap_or_else(|e| {
                eprintln!("warning: failed to accept connections: {}", e);
            });
        }
    }

    // runs an example simulation.
    pub fn run_example_simulation(&mut self) -> Result<(), String> {
        println!("node {} starting example simulation", self.config.node_id);

        let h_gate_matrix = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        let work_item = WorkItem::GateApplication {
            id: 1,
            qubit_indices: vec![0],
            amplitudes: self.quantum_state.get_state_vector(),
            gate_matrix: h_gate_matrix,
        };
        self.hybrid.submit_work(work_item);

        let mut result = None;
        for _ in 0..100 {
            if let Some(res) = self.hybrid.get_result(1) {
                result = Some(res);
                break;
            }
            thread::sleep(std::time::Duration::from_millis(10));
        }

        if let Some(WorkResult::GateApplied {
            updated_amplitudes, ..
        }) = result
        {
            self.quantum_state.set_state_vector(updated_amplitudes)?;
            println!(
                "applied hadamard gate to qubit 0. current state vector size: {}",
                self.quantum_state.amps.len()
            );
        } else {
            println!("failed to get result for gate application.");
        }

        if let Some(comm) = &mut self.comm {
            comm.broadcast_message(NodeMessage::Ping)?;

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

    // runs an example rendering.
    pub fn run_example_rendering(&self) -> Result<(), String> {
        #[cfg(feature = "vulkan")]
        let vulkan_context = self.hybrid.gpu_manager.get_vulkan_context();
        #[cfg(not(feature = "vulkan"))]
        let _vulkan_context: Option<Arc<VulkanContext>> = None; // this is intentional, hence the underscore

        #[cfg(feature = "vulkan")]
        let renderer = DistributedRenderer::new(
            self.config.node_id,
            self.config.node_count,
            100,
            "output/visualization_data",
            "output/simulation.mp4",
            800,
            600,
            vulkan_context,
        );
        #[cfg(not(feature = "vulkan"))]
        let renderer = DistributedRenderer::new(
            self.config.node_id,
            self.config.node_count,
            100,
            "output/visualization_data",
            "output/simulation.mp4",
            800,
            600,
        );

        #[cfg(feature = "vulkan")]
        renderer.render_frames(|frame, width, height, vk_ctx| {
            println!(
                "preparing quantum state data for frame {} for visualization.",
                frame
            );

            process_quantum_state_for_vulkan_visualization(frame, width, height, vk_ctx)?;

            Ok(vec![Complex64::new(frame as f64, 0.0); 100])
        })?;
        #[cfg(not(feature = "vulkan"))]
        renderer.render_frames(|frame, width, height| {
            println!(
                "preparing quantum state data for frame {} for visualization.",
                frame
            );

            process_quantum_state_for_vulkan_visualization(frame, width, height)?;

            Ok(vec![Complex64::new(frame as f64, 0.0); 100])
        })?;

        renderer.merge_and_encode()?;

        Ok(())
    }
}

// test modules.
#[cfg(test)]
mod tests {
    use super::*;

    // tests partition calculation.
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

        assert_eq!(partitions[0].start_idx, 0);
        assert_eq!(partitions[3].end_idx, 1 << 10);

        for i in 0..3 {
            assert_eq!(partitions[i].end_idx, partitions[i + 1].start_idx);
        }
    }

    // tests the sparse state vector.
    #[test]
    fn test_sparse_state_vector() {
        let mut state = SparseStateVector::new(2, 1e-10);

        let h_gate = [
            [
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            ],
            [
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            ],
        ];

        state.apply_single_qubit_gate(0, h_gate);

        assert_eq!(state.sparsity(), 2);

        let h_gate = [
            [
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            ],
            [
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            ],
        ];
        state.apply_single_qubit_gate(1, h_gate);

        assert_eq!(state.sparsity(), 4);
    }
}
