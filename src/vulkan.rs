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
    pub debug_utils: ash::extensions::ext::DebugUtils,
    #[cfg(debug_assertions)]
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
}

#[cfg(feature = "vulkan")]
impl VulkanContext {
    pub fn new() -> Result<Arc<Self>, String> {
        unsafe {
            // entry point for the vulkan api.
            let entry = Entry::load().map_err(|e| format!("failed to load entry: {}", e))?;

            // set up validation layers and extensions for debugging in a debug build
            let validation_layers = if cfg!(debug_assertions) {
                vec![CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap()]
            } else {
                vec![]
            };

            let validation_layer_pointers: Vec<*const c_char> = validation_layers
                .iter()
                .map(|layer| layer.as_ptr())
                .collect();

            let instance_extensions_cstr = ash_get_instance_extensions()?;
            let instance_extensions_pointers: Vec<*const c_char> = instance_extensions_cstr
                .iter()
                .map(|ext| ext.as_ptr())
                .collect();

            let app_info = vk::ApplicationInfo::builder()
                .application_name(CStr::from_bytes_with_nul_unchecked(b"Hello Triangle\0"))
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(CStr::from_bytes_with_nul_unchecked(b"No Engine\0"))
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::make_api_version(0, 1, 2, 0));

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_layer_names(&validation_layer_pointers)
                .enabled_extension_names(&instance_extensions_pointers);

            let instance = entry
                .create_instance(&create_info, None)
                .map_err(|e| format!("failed to create instance: {}", e))?;

            #[cfg(debug_assertions)]
            let debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);
            #[cfg(debug_assertions)]
            let debug_messenger = {
                let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                    )
                    .pfn_user_callback(Some(vulkan_debug_callback));
                debug_utils
                    .create_debug_utils_messenger(&create_info, None)
                    .map_err(|e| format!("failed to set up debug messenger: {}", e))?
            };

            // find a physical device that supports the extensions.
            let physical_devices = instance.enumerate_physical_devices().unwrap();
            let physical_device = physical_devices
                .into_iter()
                .find(|&p_device| is_device_suitable(&instance, p_device))
                .ok_or_else(|| "no suitable physical device found".to_string())?;

            // get the queue family properties.
            let queue_family_properties =
                instance.get_physical_device_queue_family_properties(physical_device);
            let queue_family_index = queue_family_properties
                .iter()
                .position(|properties| properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|i| i as u32)
                .ok_or_else(|| "no graphics queue family found".to_string())?;

            let queue_priorities = [1.0];
            let device_queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities)
                .build();

            let device_extensions_ptrs = [ash::extensions::khr::Swapchain::name().as_ptr()];
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&[device_queue_create_info])
                .enabled_extension_names(&device_extensions_ptrs)
                .build();

            let device = instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(|e| format!("failed to create device: {}", e))?;

            let graphics_queue = device.get_device_queue(queue_family_index, 0);

            let command_pool_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = device
                .create_command_pool(&command_pool_info, None)
                .map_err(|e| format!("failed to create command pool: {}", e))?;

            let descriptor_pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                },
            ];
            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&descriptor_pool_sizes);
            let descriptor_pool = device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .map_err(|e| format!("failed to create descriptor pool: {}", e))?;

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

// debug callback function for validation layers.
#[allow(dead_code)] // added this here because used conditionally
#[cfg(feature = "vulkan")]
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { *p_callback_data };
    let message = unsafe { CStr::from_ptr(callback_data.p_message) }
        .to_str()
        .unwrap();

    eprintln!("[{:?}]:{:?} - {}", message_severity, message_type, message);
    vk::FALSE
}

// a helper function to get required instance extensions.
#[cfg(feature = "vulkan")]
fn ash_get_instance_extensions() -> Result<Vec<&'static CStr>, String> {
    #[cfg(debug_assertions)]
    let mut extensions = vec![];
    #[cfg(not(debug_assertions))]
    let extensions = vec![];
    #[cfg(debug_assertions)]
    extensions.push(ash::extensions::ext::DebugUtils::name());
    Ok(extensions)
}

// a helper function to check if a device is suitable.
#[cfg(feature = "vulkan")]
fn is_device_suitable(instance: &Instance, physical_device: vk::PhysicalDevice) -> bool {
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
}

#[cfg(feature = "vulkan")]
impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            // drop resources in reverse order of creation.
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            #[cfg(debug_assertions)]
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}
