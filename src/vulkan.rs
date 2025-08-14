#![allow(clippy::many_single_char_names)]

#[cfg(feature = "vulkan")]
use ash::extensions::ext::DebugUtils;
#[cfg(feature = "vulkan")]
use ash::{self, Device, Entry, Instance, vk};
#[cfg(feature = "vulkan")]
use std::ffi::{CStr, c_void};
#[cfg(feature = "vulkan")]
use std::os::raw::c_char;
#[cfg(feature = "vulkan")]
use std::sync::Arc;

// debug callback only in debug builds
#[cfg(all(feature = "vulkan", debug_assertions))]
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    if p_callback_data.is_null() {
        return vk::FALSE;
    }

    // copy the callback data (it's POD)
    let callback_data = unsafe { &*p_callback_data };

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        std::borrow::Cow::from("")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() }
    };
    let message = if callback_data.p_message.is_null() {
        std::borrow::Cow::from("")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message).to_string_lossy() }
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            eprintln!(
                "vulkan error: {:?} ({:?}) [{}]: {}",
                message_type, message_id_name, callback_data.message_id_number, message
            );
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            eprintln!(
                "vulkan warning: {:?} ({:?}) [{}]: {}",
                message_type, message_id_name, callback_data.message_id_number, message
            );
        }
        _ => {
            println!(
                "vulkan debug: {:?} ({:?}) [{}]: {}",
                message_type, message_id_name, callback_data.message_id_number, message
            );
        }
    }

    vk::FALSE
}

// quick suitability check
#[cfg(feature = "vulkan")]
fn is_device_suitable(instance: &Instance, physical_device: vk::PhysicalDevice) -> bool {
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    // let _features = unsafe { instance.get_physical_device_features(physical_device) };

    let suitable_type = matches!(
        properties.device_type,
        vk::PhysicalDeviceType::DISCRETE_GPU
            | vk::PhysicalDeviceType::INTEGRATED_GPU
            | vk::PhysicalDeviceType::VIRTUAL_GPU
            | vk::PhysicalDeviceType::CPU
    );

    println!(
        "debug: device suitability check - type: {:?}, suitable: {}",
        properties.device_type, suitable_type
    );

    suitable_type
}

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
    pub debug_utils: Option<DebugUtils>,
    #[cfg(debug_assertions)]
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
}

#[cfg(feature = "vulkan")]
impl VulkanContext {
    pub fn new() -> Result<Arc<Self>, String> {
        unsafe {
            println!("debug: starting vulkancontext::new()");

            // Entry
            println!("debug: loading vulkan entry...");
            let entry = Entry::load().map_err(|e| {
                eprintln!("error: failed to load vulkan entry: {}", e);
                format!("failed to load entry: {}", e)
            })?;
            println!("debug: entry loaded successfully");

            // enumerate instance extensions / layers
            println!("debug: checking available instance extensions...");
            let available_extensions = entry
                .enumerate_instance_extension_properties(None)
                .map_err(|e| format!("failed to enumerate instance extensions: {}", e))?;

            println!("debug: available instance extensions:");
            for ext in &available_extensions {
                let name = CStr::from_ptr(ext.extension_name.as_ptr())
                    .to_str()
                    .unwrap_or("invalid");
                println!("  - {}", name);
            }

            println!("debug: checking available instance layers...");
            let available_layers = entry
                .enumerate_instance_layer_properties()
                .map_err(|e| format!("failed to enumerate instance layers: {}", e))?;
            println!("debug: available instance layers:");
            for layer in &available_layers {
                let name = CStr::from_ptr(layer.layer_name.as_ptr())
                    .to_str()
                    .unwrap_or("invalid");
                println!("  - {}", name);
            }

            // validation layers (debug builds only)
            let validation_layers: Vec<&CStr> = if cfg!(debug_assertions) {
                let requested = CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0")
                    .map_err(|e| {
                        format!(
                            "invalid validation layer name literal (shouldn't happen): {}",
                            e
                        )
                    })?;
                let has = available_layers.iter().any(|layer| {
                    let layer_name = CStr::from_ptr(layer.layer_name.as_ptr());
                    layer_name == requested
                });
                if has {
                    println!("debug: enabling VK_LAYER_KHRONOS_validation");
                    vec![requested]
                } else {
                    println!("debug: validation layer not available");
                    vec![]
                }
            } else {
                vec![]
            };

            let validation_layer_pointers: Vec<*const c_char> =
                validation_layers.iter().map(|l| l.as_ptr()).collect();

            // required instance extensions
            println!("debug: setting up instance extensions...");
            let mut required_extensions: Vec<&CStr> = Vec::new();

            #[cfg(debug_assertions)]
            if !validation_layers.is_empty() {
                // debug utils name via extension wrapper or literal - literal is stable across ash versions
                let debug_utils_name = DebugUtils::name();
                let has_debug = available_extensions.iter().any(|ext| {
                    let ext_name = CStr::from_ptr(ext.extension_name.as_ptr());
                    ext_name == debug_utils_name
                });
                if has_debug {
                    println!("debug: debug utils extension available");
                    required_extensions.push(debug_utils_name);
                } else {
                    println!("debug: debug utils extension not available");
                }
            }

            // use literals for instance extension names to avoid version mismatches
            let get_physical_device_properties2_name =
                CStr::from_bytes_with_nul(b"VK_KHR_get_physical_device_properties2\0")
                    .map_err(|e| format!("invalid extension name literal: {}", e))?;
            let has_get_physical_device_properties2 = available_extensions.iter().any(|ext| {
                let ext_name = CStr::from_ptr(ext.extension_name.as_ptr());
                ext_name == get_physical_device_properties2_name
            });
            if has_get_physical_device_properties2 {
                println!("debug: enabling VK_KHR_get_physical_device_properties2");
                required_extensions.push(get_physical_device_properties2_name);
            } else {
                println!("debug: VK_KHR_get_physical_device_properties2 not available");
            }

            let extension_pointers: Vec<*const c_char> =
                required_extensions.iter().map(|ext| ext.as_ptr()).collect();

            // app info
            println!("debug: creating application info...");
            let app_name =
                CStr::from_bytes_with_nul(b"qoa quantum\0").map_err(|e| format!("{}", e))?;
            let engine_name =
                CStr::from_bytes_with_nul(b"qoa engine\0").map_err(|e| format!("{}", e))?;
            let app_info = vk::ApplicationInfo::builder()
                .application_name(app_name)
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(engine_name)
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::make_api_version(0, 1, 0, 0)); // Vulkan 1.0 for compatibility

            // create instance
            println!("debug: creating vulkan instance...");
            println!(
                "debug: using {} validation layers",
                validation_layer_pointers.len()
            );
            println!("debug: using {} extensions", extension_pointers.len());

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_layer_names(&validation_layer_pointers)
                .enabled_extension_names(&extension_pointers);

            let instance = entry.create_instance(&create_info, None).map_err(|e| {
                eprintln!("error: failed to create vulkan instance: {}", e);
                format!("failed to create instance: {}", e)
            })?;
            println!("debug: vulkan instance created successfully");

            // debug messenger (debug builds)
            #[cfg(debug_assertions)]
            let (debug_utils, debug_messenger) =
                if !validation_layers.is_empty() && extension_pointers.len() > 0 {
                    println!("debug: setting up debug messenger...");
                    let debug_utils = DebugUtils::new(&entry, &instance);
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

            // enumerate devices
            println!("debug: enumerating physical devices...");
            let physical_devices = instance.enumerate_physical_devices().map_err(|e| {
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
                println!(
                    "debug: device {}: {} (type: {:?}, driver: {}.{}.{} )",
                    i,
                    device_name,
                    props.device_type,
                    vk::api_version_major(props.driver_version),
                    vk::api_version_minor(props.driver_version),
                    vk::api_version_patch(props.driver_version)
                );

                let _device_extensions = instance
                    .enumerate_device_extension_properties(device)
                    .map_err(|e| format!("failed to enumerate device extensions: {}", e))?;
                println!(
                    "debug: device {} has {} extensions",
                    i,
                    _device_extensions.len()
                );
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

            // find queue family
            println!("debug: finding queue families...");
            let queue_family_properties =
                instance.get_physical_device_queue_family_properties(physical_device);

            if queue_family_properties.is_empty() {
                return Err("no queue families found".to_string());
            }

            println!(
                "debug: found {} queue families",
                queue_family_properties.len()
            );
            for (i, props) in queue_family_properties.iter().enumerate() {
                println!(
                    "debug: queue family {}: flags={:?}, count={}",
                    i, props.queue_flags, props.queue_count
                );
            }

            let queue_family_index = queue_family_properties
                .iter()
                .position(|properties| {
                    properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        || properties.queue_flags.contains(vk::QueueFlags::COMPUTE)
                })
                .map(|i| i as u32)
                .ok_or_else(|| "no graphics or compute queue family found".to_string())?;

            println!("debug: selected queue family index: {}", queue_family_index);

            // device extensions (check availability)
            println!("debug: checking device extensions...");
            let available_device_extensions = instance
                .enumerate_device_extension_properties(physical_device)
                .map_err(|e| format!("failed to enumerate device extensions: {}", e))?;

            let mut required_device_extensions: Vec<*const c_char> = Vec::new();
            let mut dynamic_indexing_supported = false;

            // use stable literals for device extension names
            let maintenance3_name = CStr::from_bytes_with_nul(b"VK_KHR_maintenance3\0")
                .map_err(|e| format!("invalid extension name literal (maintenance3): {}", e))?;
            let descriptor_indexing_name =
                CStr::from_bytes_with_nul(b"VK_EXT_descriptor_indexing\0").map_err(|e| {
                    format!(
                        "invalid extension name literal (descriptor_indexing): {}",
                        e
                    )
                })?;

            let has_maintenance3 = available_device_extensions.iter().any(|ext| {
                let ext_name = CStr::from_ptr(ext.extension_name.as_ptr());
                ext_name == maintenance3_name
            });
            let has_descriptor_indexing = available_device_extensions.iter().any(|ext| {
                let ext_name = CStr::from_ptr(ext.extension_name.as_ptr());
                ext_name == descriptor_indexing_name
            });

            if has_maintenance3 {
                println!("debug: enabling VK_KHR_maintenance3 device extension");
                required_device_extensions.push(maintenance3_name.as_ptr());
            }

            if has_descriptor_indexing {
                println!("debug: enabling VK_EXT_descriptor_indexing device extension");
                required_device_extensions.push(descriptor_indexing_name.as_ptr());
                dynamic_indexing_supported = true;
            }

            if !dynamic_indexing_supported {
                return Err(
                    "required extension VK_EXT_descriptor_indexing not available".to_string(),
                );
            }

            println!(
                "debug: using {} device extensions",
                required_device_extensions.len()
            );

            // queue create info
            println!("debug: creating device queue info...");
            let queue_priorities = [1.0f32];
            let device_queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities);

            // device create info + features
            println!("debug: creating device create info...");
            println!("debug: requesting device features...");

            // Feature structure for descriptor indexing
            let mut dynamic_indexing_features =
                vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::builder()
                    .shader_storage_buffer_array_non_uniform_indexing(dynamic_indexing_supported)
                    .shader_uniform_buffer_array_non_uniform_indexing(dynamic_indexing_supported)
                    .shader_storage_texel_buffer_array_non_uniform_indexing(
                        dynamic_indexing_supported,
                    )
                    .shader_uniform_texel_buffer_array_non_uniform_indexing(
                        dynamic_indexing_supported,
                    )
                    .descriptor_binding_uniform_buffer_update_after_bind(true)
                    .descriptor_binding_storage_buffer_update_after_bind(true)
                    .descriptor_binding_partially_bound(true)
                    .descriptor_binding_variable_descriptor_count(true)
                    .runtime_descriptor_array(true);

            let mut features2 = vk::PhysicalDeviceFeatures2::builder();
            features2 = features2.push_next(&mut dynamic_indexing_features);

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&device_queue_create_info))
                .enabled_extension_names(&required_device_extensions)
                .push_next(&mut features2);

            // create logical device
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

            // get queue
            println!("debug: getting device queue...");
            let graphics_queue = device.get_device_queue(queue_family_index, 0);
            println!("debug: queue obtained successfully");

            // create command pool
            println!("debug: creating command pool...");
            let command_pool_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = device
                .create_command_pool(&command_pool_info, None)
                .map_err(|e| format!("failed to create command pool: {}", e))?;
            println!("debug: command pool created successfully");

            // create descriptor pool (use valid VkDescriptorType values)
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
                .pool_sizes(&descriptor_pool_sizes)
                .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND_EXT);

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

    // query device limits + descriptor indexing props
    pub fn query_device_limits(
        &self,
    ) -> Result<
        (
            vk::PhysicalDeviceLimits,
            vk::PhysicalDeviceDescriptorIndexingPropertiesEXT,
        ),
        String,
    > {
        unsafe {
            let mut descriptor_indexing_properties =
                vk::PhysicalDeviceDescriptorIndexingPropertiesEXT::builder();
            let mut props2 = vk::PhysicalDeviceProperties2::builder()
                .push_next(&mut descriptor_indexing_properties);
            self.instance
                .get_physical_device_properties2(self.physical_device, &mut props2);

            println!(
                "debug: device limits: max_storage_buffer_range = {}",
                props2.properties.limits.max_storage_buffer_range
            );
            println!(
                "debug: device limits: max_push_constants_size = {}",
                props2.properties.limits.max_push_constants_size
            );

            Ok((
                props2.properties.limits,
                descriptor_indexing_properties.build(),
            ))
        }
    }
}

#[cfg(feature = "vulkan")]
impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            println!("debug: dropping vulkancontext...");

            // teardown in reverse order
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
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
