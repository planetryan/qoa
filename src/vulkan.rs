#[cfg(not(feature = "vulkan"))]
#[derive(Clone)]
pub struct VulkanContext {}

#[cfg(feature = "vulkan")]
use ash::{self, Device, Entry, Instance, vk};
#[cfg(feature = "vulkan")]
use std::ffi::CStr;
#[cfg(feature = "vulkan")]
use std::sync::Arc;
#[cfg(feature = "vulkan")]
use std::os::raw::c_char;

#[cfg(feature = "vulkan")]
pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_index: u32,
    pub graphics_queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    #[cfg(debug_assertions)]
    pub debug_utils: Option<ash::extensions::ext::DebugUtils>,
    #[cfg(debug_assertions)]
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
}

#[cfg(feature = "vulkan")]
impl VulkanContext {
    pub fn new() -> Result<Arc<Self>, String> {
        unsafe {
            println!("debug: starting vulkancontext::new()");
            
            // step 1: load vulkan entry
            println!("debug: loading vulkan entry...");
            let entry = Entry::load().map_err(|e| {
                eprintln!("error: failed to load vulkan entry: {}", e);
                format!("failed to load entry: {}", e)
            })?;
            println!("debug: entry loaded successfully");

            // step 2: check available instance extensions and layers
            println!("debug: checking available instance extensions...");
            let available_extensions = entry.enumerate_instance_extension_properties(None)
                .map_err(|e| format!("failed to enumerate instance extensions: {}", e))?;
            
            println!("debug: available instance extensions:");
            for ext in &available_extensions {
                let name = CStr::from_ptr(ext.extension_name.as_ptr()).to_str().unwrap_or("invalid");
                println!("  - {}", name);
            }

            println!("debug: checking available instance layers...");
            let available_layers = entry.enumerate_instance_layer_properties()
                .map_err(|e| format!("failed to enumerate instance layers: {}", e))?;
            
            println!("debug: available instance layers:");
            for layer in &available_layers {
                let name = CStr::from_ptr(layer.layer_name.as_ptr()).to_str().unwrap_or("invalid");
                println!("  - {}", name);
            }

            // step 3: setup validation layers if in debug build
            let validation_layers = if cfg!(debug_assertions) {
                let validation_layer_name = CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0")
                    .map_err(|e| format!("invalid validation layer name: {}", e))?;
                
                let has_validation = available_layers.iter().any(|layer| {
                    let layer_name = CStr::from_ptr(layer.layer_name.as_ptr());
                    layer_name == validation_layer_name
                });
                
                if has_validation {
                    println!("debug: validation layers available, enabling them");
                    vec![validation_layer_name]
                } else {
                    println!("debug: validation layers not available, proceeding without them");
                    vec![]
                }
            } else {
                println!("debug: release build, no validation layers");
                vec![]
            };

            let validation_layer_pointers: Vec<*const c_char> = validation_layers
                .iter()
                .map(|layer| layer.as_ptr())
                .collect();

            // step 4: setup instance extensions including debug utils if validation enabled
            println!("debug: setting up instance extensions...");
            let mut required_extensions = Vec::new();
            
            #[cfg(debug_assertions)]
            if !validation_layers.is_empty() {
                let debug_utils_name = ash::extensions::ext::DebugUtils::name();
                let has_debug_utils = available_extensions.iter().any(|ext| {
                    let ext_name = CStr::from_ptr(ext.extension_name.as_ptr());
                    ext_name == debug_utils_name
                });
                
                if has_debug_utils {
                    println!("debug: debug utils extension available");
                    required_extensions.push(debug_utils_name);
                } else {
                    println!("debug: debug utils extension not available");
                }
            }

            let extension_pointers: Vec<*const c_char> = required_extensions
                .iter()
                .map(|ext| ext.as_ptr())
                .collect();

            // step 5: create application info
            println!("debug: creating application info...");
            let app_name = CStr::from_bytes_with_nul(b"qoa quantum\0")
                .map_err(|e| format!("invalid app name: {}", e))?;
            let engine_name = CStr::from_bytes_with_nul(b"qoa engine\0")
                .map_err(|e| format!("invalid engine name: {}", e))?;

            let app_info = vk::ApplicationInfo::builder()
                .application_name(app_name)
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(engine_name)
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::make_api_version(0, 1, 0, 0)); // vulkan 1.0 for max compatibility

            // step 6: create vulkan instance
            println!("debug: creating vulkan instance...");
            println!("debug: using {} validation layers", validation_layer_pointers.len());
            println!("debug: using {} extensions", extension_pointers.len());

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_layer_names(&validation_layer_pointers)
                .enabled_extension_names(&extension_pointers);

            let instance = entry
                .create_instance(&create_info, None)
                .map_err(|e| {
                    eprintln!("error: failed to create vulkan instance: {}", e);
                    format!("failed to create instance: {}", e)
                })?;
            println!("debug: vulkan instance created successfully");

            // step 7: setup debug messenger if validation layers and debug utils available
            #[cfg(debug_assertions)]
            let (debug_utils, debug_messenger) = if !validation_layers.is_empty() && !required_extensions.is_empty() {
                println!("debug: setting up debug messenger...");
                let debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);
                let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(vulkan_debug_callback));
                let debug_messenger = debug_utils
                    .create_debug_utils_messenger(&create_info, None)
                    .map_err(|e| format!("failed to set up debug messenger: {}", e))?;
                println!("debug: debug messenger created successfully");
                (Some(debug_utils), debug_messenger)
            } else {
                println!("debug: skipping debug messenger setup");
                (None, vk::DebugUtilsMessengerEXT::null())
            };

            // step 8: enumerate and pick physical device
            println!("debug: enumerating physical devices...");
            let physical_devices = instance.enumerate_physical_devices()
                .map_err(|e| {
                    eprintln!("error: failed to enumerate physical devices: {}", e);
                    format!("failed to enumerate physical devices: {}", e)
                })?;
            
            if physical_devices.is_empty() {
                return Err("no vulkan-compatible devices found".to_string());
            }

            println!("debug: found {} physical device(s)", physical_devices.len());
            
            for (i, &device) in physical_devices.iter().enumerate() {
                let props = instance.get_physical_device_properties(device);
                let device_name = CStr::from_ptr(props.device_name.as_ptr())
                    .to_str()
                    .unwrap_or("unknown");
                println!("debug: device {}: {} (type: {:?}, driver: {}.{}.{} )", 
                    i, device_name, props.device_type,
                    vk::api_version_major(props.driver_version),
                    vk::api_version_minor(props.driver_version),
                    vk::api_version_patch(props.driver_version)
                );

                let _device_extensions = instance.enumerate_device_extension_properties(device)
                    .map_err(|e| format!("failed to enumerate device extensions: {}", e))?;
                println!("debug: device {} has {} extensions", i, _device_extensions.len());
            }

            let physical_device = physical_devices
                .into_iter()
                .find(|&p_device| is_device_suitable(&instance, p_device))
                .ok_or_else(|| "no suitable physical device found".to_string())?;

            let device_props = instance.get_physical_device_properties(physical_device);
            let device_name = CStr::from_ptr(device_props.device_name.as_ptr())
                .to_str()
                .unwrap_or("unknown");
            println!("debug: selected device: {}", device_name);

            // step 9: find queue families
            println!("debug: finding queue families...");
            let queue_family_properties = instance.get_physical_device_queue_family_properties(physical_device);
            
            if queue_family_properties.is_empty() {
                return Err("no queue families found".to_string());
            }

            println!("debug: found {} queue families", queue_family_properties.len());
            for (i, props) in queue_family_properties.iter().enumerate() {
                println!("debug: queue family {}: flags={:?}, count={}", i, props.queue_flags, props.queue_count);
            }

            let queue_family_index = queue_family_properties
                .iter()
                .position(|properties| {
                    properties.queue_flags.contains(vk::QueueFlags::GRAPHICS) ||
                    properties.queue_flags.contains(vk::QueueFlags::COMPUTE)
                })
                .map(|i| i as u32)
                .ok_or_else(|| "no graphics or compute queue family found".to_string())?;

            println!("debug: selected queue family index: {}", queue_family_index);

            // step 10: check device extensions before device creation
            println!("debug: checking device extensions...");
            let _device_extensions = instance.enumerate_device_extension_properties(physical_device)
                .map_err(|e| format!("failed to enumerate device extensions: {}", e))?;
            
            // for now, use no extensions to isolate issues
            let device_extensions_ptrs: Vec<*const c_char> = vec![];

            println!("debug: using {} device extensions", device_extensions_ptrs.len());

            // step 11: create device queue info
            println!("debug: creating device queue info...");
            let queue_priorities = [1.0f32];
            let device_queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities);

            // step 12: create device create info
            println!("debug: creating device create info...");
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&device_queue_create_info))
                .enabled_extension_names(&device_extensions_ptrs);

            // step 13: create logical device
            println!("debug: about to create logical device...");
            println!("debug: physical device handle: {:?}", physical_device);
            println!("debug: queue family index: {}", queue_family_index);
            println!("debug: queue priorities: {:?}", queue_priorities);
            
            std::thread::sleep(std::time::Duration::from_millis(100));
            
            let device = instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(|e| {
                    eprintln!("error: failed to create logical device: {}", e);
                    format!("failed to create device: {}", e)
                })?;
            
            println!("debug: success! logical device created");

            // step 14: get graphics queue
            println!("debug: getting device queue...");
            let graphics_queue = device.get_device_queue(queue_family_index, 0);
            println!("debug: queue obtained successfully");

            // step 15: create command pool
            println!("debug: creating command pool...");
            let command_pool_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = device
                .create_command_pool(&command_pool_info, None)
                .map_err(|e| format!("failed to create command pool: {}", e))?;
            println!("debug: command pool created successfully");

            // step 16: create descriptor pool
            println!("debug: creating descriptor pool...");
            let descriptor_pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                },
            ];
            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&descriptor_pool_sizes);
            let descriptor_pool = device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .map_err(|e| format!("failed to create descriptor pool: {}", e))?;
            println!("debug: descriptor pool created successfully");

            println!("debug: vulkancontext initialization complete!");

            Ok(Arc::new(Self {
                entry,
                instance,
                device,
                physical_device,
                queue_family_index,
                graphics_queue,
                command_pool,
                descriptor_pool,
                descriptor_sets: vec![],
                #[cfg(debug_assertions)]
                debug_utils,
                #[cfg(debug_assertions)]
                debug_messenger,
            }))
        }
    }
}

#[allow(dead_code)]
#[cfg(feature = "vulkan")]
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    // dereference the raw pointer safely inside unsafe block
    let callback_data = unsafe { *p_callback_data };
    // construct cstr safely inside unsafe block
    let message = unsafe { CStr::from_ptr(callback_data.p_message) }
        .to_str()
        .unwrap_or("invalid message");

    eprintln!("vulkan [{:?}] {:?}: {}", message_severity, message_type, message);
    vk::FALSE
}

#[cfg(feature = "vulkan")]
fn is_device_suitable(instance: &Instance, physical_device: vk::PhysicalDevice) -> bool {
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let _features = unsafe { instance.get_physical_device_features(physical_device) };
    
    // accept any device type for now
    let suitable_type = matches!(
        properties.device_type,
        vk::PhysicalDeviceType::DISCRETE_GPU 
        | vk::PhysicalDeviceType::INTEGRATED_GPU 
        | vk::PhysicalDeviceType::VIRTUAL_GPU
        | vk::PhysicalDeviceType::CPU
    );

    println!("debug: device suitability check - type: {:?}, suitable: {}", 
             properties.device_type, suitable_type);

    suitable_type
}

#[cfg(feature = "vulkan")]
impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            println!("debug: dropping vulkancontext...");
            
            // drop resources in reverse order of creation
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            
            #[cfg(debug_assertions)]
            {
                if let Some(ref debug_utils) = self.debug_utils {
                    if self.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
                        debug_utils.destroy_debug_utils_messenger(self.debug_messenger, None);
                    }
                }
            }
            
            self.instance.destroy_instance(None);
            println!("debug: vulkancontext dropped successfully");
        }
    }
}
