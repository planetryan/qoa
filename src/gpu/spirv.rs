#[cfg(feature = "vulkan")]
use ash::{vk, Device};
#[cfg(feature = "vulkan")]
use std::sync::Arc;
#[cfg(feature = "vulkan")]
use std::io::Read;
#[cfg(feature = "vulkan")]
use std::ffi::CString;
#[cfg(feature = "vulkan")]
use std::mem::size_of;
#[cfg(feature = "vulkan")]
use num_complex::Complex64;

// re-export VulkanContext from the parent module
#[cfg(feature = "vulkan")]
use super::VulkanContext;

// quantum gate enum, matching GLSL u32
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum QuantumGate {
    Hadamard = 0,
    PauliX = 1,
    PauliZ = 2,
}

// GPU kernel parameters struct (must be #[repr(C)])
#[derive(Debug, Clone)]
#[repr(C)]
struct QuantumKernelData {
    gate_type: u32,
    target_qubit: u32,
    num_qubits: u32,
    _pad: u32,
}

#[cfg(feature = "vulkan")]
pub struct QuantumGpuManager {
    vulkan_context: Arc<VulkanContext>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    state_buffer: vk::Buffer,
    state_buffer_memory: vk::DeviceMemory,
    gate_params_buffer: vk::Buffer,
    gate_params_buffer_memory: vk::DeviceMemory,
    command_buffer: vk::CommandBuffer,
}

#[cfg(feature = "vulkan")]
impl QuantumGpuManager {
    pub fn new(
        vulkan_context: Arc<VulkanContext>,
        state_vector: &[Complex64],
        gate: QuantumGate,
    ) -> Result<Self, String> {
        unsafe {
            let device = &vulkan_context.device;

            // Load SPIR-V shader code for gate
            let shader_code = Self::load_shader_code_for_gate(gate)?;
            let shader_module_info = vk::ShaderModuleCreateInfo::builder()
                .code(&shader_code);
            let shader_module = device
                .create_shader_module(&shader_module_info, None)
                .map_err(|e| format!("failed to create shader module: {}", e))?;

            // Descriptor set layout bindings
            let descriptor_set_layout_bindings = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ];
            let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&descriptor_set_layout_bindings);
            let descriptor_set_layout = device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
                .map_err(|e| format!("failed to create descriptor set layout: {}", e))?;

            // Pipeline layout
            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));
            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| format!("failed to create pipeline layout: {}", e))?;

            // Compute pipeline
            let entry_point_name = CString::new("main").unwrap();
            let shader_stage = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&entry_point_name);
            let compute_pipeline_info = vk::ComputePipelineCreateInfo::builder()
                .stage(shader_stage.build())
                .layout(pipeline_layout);

            let pipeline = device
                .create_compute_pipelines(vk::PipelineCache::null(), &[compute_pipeline_info.build()], None)
                .map_err(|e| format!("failed to create compute pipeline: {}", e))?[0];

            // Create buffers
            let (state_buffer, state_buffer_memory) = Self::create_buffer(
                &vulkan_context,
                (state_vector.len() * size_of::<Complex64>()) as vk::DeviceSize,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let (gate_params_buffer, gate_params_buffer_memory) = Self::create_buffer(
                &vulkan_context,
                size_of::<QuantumKernelData>() as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            // Copy initial state vector data to GPU buffer
            let data_ptr = device
                .map_memory(state_buffer_memory, 0, (state_vector.len() * size_of::<Complex64>()) as vk::DeviceSize, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("failed to map memory: {}", e))?;
            let slice = std::slice::from_raw_parts_mut(data_ptr as *mut Complex64, state_vector.len());
            slice.copy_from_slice(state_vector);
            device.unmap_memory(state_buffer_memory);

            // Descriptor pool
            let pool_sizes = [
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .build(),
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .build(),
            ];
            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pool_sizes)
                .max_sets(1);
            let descriptor_pool = device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .map_err(|e| format!("failed to create descriptor pool: {}", e))?;

            // Allocate descriptor set
            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));
            let descriptor_sets = device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .map_err(|e| format!("failed to allocate descriptor sets: {}", e))?;
            let descriptor_set = descriptor_sets[0];

            // Update descriptor sets with buffers
            let buffer_info_state = vk::DescriptorBufferInfo::builder()
                .buffer(state_buffer)
                .offset(0)
                .range((state_vector.len() * size_of::<Complex64>()) as vk::DeviceSize);
            let buffer_info_params = vk::DescriptorBufferInfo::builder()
                .buffer(gate_params_buffer)
                .offset(0)
                .range(size_of::<QuantumKernelData>() as vk::DeviceSize);

            let write_descriptor_sets = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_info_state))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_info_params))
                    .build(),
            ];
            device.update_descriptor_sets(&write_descriptor_sets, &[]);

            // Allocate command buffer
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(vulkan_context.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffers = device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .map_err(|e| format!("failed to allocate command buffers: {}", e))?;
            let command_buffer = command_buffers[0];

            // Destroy shader module (no longer needed after pipeline creation)
            device.destroy_shader_module(shader_module, None);

            Ok(Self {
                vulkan_context,
                pipeline,
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pool,
                descriptor_set,
                state_buffer,
                state_buffer_memory,
                gate_params_buffer,
                gate_params_buffer_memory,
                command_buffer,
            })
        }
    }

    pub fn apply_gate(&self, gate: QuantumGate, target_qubit: u32, num_qubits: u32) -> Result<(), String> {
        unsafe {
            let device = &self.vulkan_context.device;

            // Update gate parameters uniform buffer
            let gate_data = QuantumKernelData {
                gate_type: gate as u32,
                target_qubit,
                num_qubits,
                _pad: 0,
            };
            let data_ptr = device
                .map_memory(self.gate_params_buffer_memory, 0, size_of::<QuantumKernelData>() as vk::DeviceSize, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("failed to map memory for gate params: {}", e))?;
            std::ptr::copy_nonoverlapping(&gate_data, data_ptr as *mut QuantumKernelData, 1);
            device.unmap_memory(self.gate_params_buffer_memory);

            // Begin command buffer recording
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(|e| format!("failed to begin command buffer: {}", e))?;

            // Bind pipeline and descriptor set
            device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptor_set),
                &[],
            );

            // Dispatch compute shader (ceil div for group count)
            let state_len = 1 << num_qubits;
            let group_count_x = ((state_len + 31) / 32) as u32; // local size 32
            device.cmd_dispatch(self.command_buffer, group_count_x, 1, 1);

            // End command buffer recording
            device.end_command_buffer(self.command_buffer)
                .map_err(|e| format!("failed to end command buffer: {}", e))?;

            // Submit to queue and wait
            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            device.queue_submit(self.vulkan_context.graphics_queue, &[submit_info.build()], vk::Fence::null())
                .map_err(|e| format!("failed to submit queue: {}", e))?;
            device.device_wait_idle()
                .map_err(|e| format!("failed to wait for device idle: {}", e))?;

            Ok(())
        }
    }

    pub fn get_state_vector(&self, num_qubits: u32) -> Result<Vec<Complex64>, String> {
        unsafe {
            let device = &self.vulkan_context.device;
            let state_len = 1 << num_qubits;
            let data_ptr = device
                .map_memory(self.state_buffer_memory, 0, (state_len * size_of::<Complex64>()) as vk::DeviceSize, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("failed to map memory for state vector: {}", e))?;
            let slice = std::slice::from_raw_parts(data_ptr as *const Complex64, state_len);
            let result = slice.to_vec();
            device.unmap_memory(self.state_buffer_memory);
            Ok(result)
        }
    }

    // Load SPIR-V binary from file
    fn load_shader_code(path: &str) -> Result<Vec<u32>, String> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("failed to open shader file {}: {}", path, e))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| format!("failed to read shader file {}: {}", path, e))?;

        if buffer.len() % 4 != 0 {
            return Err("shader file size is not a multiple of 4".to_string());
        }

        let mut words = Vec::with_capacity(buffer.len() / 4);
        let mut i = 0;
        while i < buffer.len() {
            words.push(u32::from_le_bytes([
                buffer[i],
                buffer[i + 1],
                buffer[i + 2],
                buffer[i + 3],
            ]));
            i += 4;
        }

        Ok(words)
    }

    // Load shader code filename from QuantumGate enum
    fn load_shader_code_for_gate(gate: QuantumGate) -> Result<Vec<u32>, String> {
        let filename = match gate {
            QuantumGate::Hadamard => "src/gpu/shaders/hadamard.spv",
            QuantumGate::PauliX => "src/gpu/shaders/paulix.spv",
            QuantumGate::PauliZ => "src/gpu/shaders/pauliz.spv",
        };
        Self::load_shader_code(filename)
    }

    // Create Vulkan buffer with memory allocation
    fn create_buffer(
        vulkan_context: &VulkanContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        let device = &vulkan_context.device;

        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device
            .create_buffer(&buffer_info, None)
            .map_err(|e| format!("failed to create buffer: {}", e))?;

        let mem_requirements = device.get_buffer_memory_requirements(buffer);

        let mem_properties = &vulkan_context.memory_properties;

        let mut mem_type_index = u32::MAX;
        for i in 0..mem_properties.memory_type_count {
            if (mem_requirements.memory_type_bits & (1 << i)) != 0
                && (mem_properties.memory_types[i as usize].property_flags & properties) == properties
            {
                mem_type_index = i;
                break;
            }
        }
        if mem_type_index == u32::MAX {
            return Err("failed to find suitable memory type".to_string());
        }

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);

        let buffer_memory = device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| format!("failed to allocate buffer memory: {}", e))?;

        device.bind_buffer_memory(buffer, buffer_memory, 0)
            .map_err(|e| format!("failed to bind buffer memory: {}", e))?;

        Ok((buffer, buffer_memory))
    }
}

#[cfg(feature = "vulkan")]
impl Drop for QuantumGpuManager {
    fn drop(&mut self) {
        unsafe {
            let device = &self.vulkan_context.device;
            device.destroy_buffer(self.state_buffer, None);
            device.free_memory(self.state_buffer_memory, None);
            device.destroy_buffer(self.gate_params_buffer, None);
            device.free_memory(self.gate_params_buffer_memory, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
