// this file provides vulkancontext type regardless of feature flags
// define a placeholder vulkancontext struct that's always available
// this will be used when the "vulkan" feature is not enabled

#[cfg(not(feature = "vulkan"))]
#[derive(Clone)]
pub struct VulkanContext {}

