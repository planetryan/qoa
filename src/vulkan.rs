// fitxategi honek vulkan gailua kudeatzen du src/gpu.rs-entzat: gailua aukeratu, egoera gailuaren memorian
// gorde, igo eta jaitsi, eta etapak exekutatu (src/shaders/tile.comp, etapa bakoitzeko dispatch bat).
//
// gailuak behar dituenak: vulkan 1.2, f64 shader-etan (shaderFloat64: QOA ez da 32 bitetan kalkulatzen) eta
// buffer device address. egoera 2^k anplitudeko zatietan banatzen da (zati bakoitza bere buffer eta
// memoriarekin), gailu batzuek esleipen bakoitzaren tamaina mugatzen dutelako; shader-ak zatien helbideen
// taula bat irakurtzen du.
//
// igoera eta jaitsiera bitarteko buffer batetik (host-visible) pasatzen dira, 64 MiB-eko zatietan.
// unsafe hemen: vulkan kontrolatzaileari deiak (kanpoko C funtzioak)
#![allow(unsafe_code)]

use crate::gpu::{GpuGate, Stage, TILE_BITS};
use crate::{t, ts};
use ash::vk;
use num_complex::Complex64 as C;
use std::time::Instant;

// src/shaders/tile.comp konpilatua (tools/shaders.sh)
static SPIRV: &[u8] = include_bytes!("shaders/tile.spv");

// shader-eko lan-talde baten hariak eta lauzaren tamaina
const THREADS: u32 = 256;
const TILE: usize = 1 << TILE_BITS;
// x ardatzeko lan-taldeak dispatch bakoitzean (vulkan-ek gutxienez 65535 bermatzen du)
const GROUPS_X: usize = 65535;
// bitarteko bufferraren gehienezko tamaina
const STAGING_MAX: usize = 64 << 20;
// zati baten gehienezko tamaina (1 GiB) eta zati-kopuru handiena (shader-eko `chunk` taula)
const CHUNK_BYTES_MAX: u64 = 1 << 30;
const CHUNKS_MAX: usize = 64;
const AMP: usize = std::mem::size_of::<C>();

// shader-eko `Push` (std430, 64 byte)
#[repr(C)]
struct Push {
    pos: [u32; TILE_BITS],
    low: u32,
    first: u32,
    count: u32,
    len: u32,
    groups: u32,
    chunk_bits: u32,
}

struct Buf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    // mapatutako memoria (host-visible bufferrak), bestela nulua
    ptr: *mut u8,
}

pub struct Device {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    shader: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptors: vk::DescriptorPool,
    set: vk::DescriptorSet,
    memory: vk::PhysicalDeviceMemoryProperties,
    // zati baten anplitudeak (2ren berredura) eta gailuaren memoria lokala
    chunk_bits: u32,
    device_bytes: u64,
    // egoera: zatiak eta haien helbideen taula; ateak; bitarteko bufferra
    len: usize,
    chunks: Vec<Buf>,
    table: Option<Buf>,
    gates: Option<Buf>,
    staging: Option<Buf>,
    name: String,
    kind: &'static str,
}

// segurtasuna: vulkan objektuak edozein haritatik erabil daitezke kanpotik sinkronizatuta; Device beti Mutex
// baten barruan dago (src/gpu.rs), eta mapatutako erakusleak bere memoriarenak dira
unsafe impl Send for Device {}

fn fail(what: &'static str) -> impl Fn(vk::Result) -> String {
    move |e| {
        t!(
            "GPU: {} failed: {:?}",
            "GPUa: {}: huts egin du: {:?}",
            what,
            e
        )
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64)
}

// gailu bat egokia den, eta ez bada zergatik
struct Candidate {
    physical: vk::PhysicalDevice,
    family: u32,
    score: u32,
    name: String,
    kind: &'static str,
}

fn examine(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Result<Candidate, String> {
    // segurtasuna: pd instantzia honek emandakoa da
    let props = unsafe { instance.get_physical_device_properties(pd) };
    let name = props
        .device_name_as_c_str()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let api = props.api_version;
    if api < vk::API_VERSION_1_2 {
        return Err(t!(
            "{}: Vulkan {}.{}, QOA needs 1.2",
            "{}: vulkan {}.{}, QOA-k 1.2 behar du",
            name,
            vk::api_version_major(api),
            vk::api_version_minor(api)
        ));
    }
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let float64 = {
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12);
        // segurtasuna: goian bezala
        unsafe { instance.get_physical_device_features2(pd, &mut f2) };
        f2.features.shader_float64 == vk::TRUE
    };
    if !float64 {
        return Err(t!(
            "{}: no 64-bit floats in shaders (shaderFloat64); QOA does not compute in 32 bits",
            "{}: ez du 64 biteko koma higikorrik shader-etan (shaderFloat64); QOA ez da 32 bitetan kalkulatzen",
            name
        ));
    }
    if f12.buffer_device_address != vk::TRUE {
        return Err(t!(
            "{}: no buffer device addresses (bufferDeviceAddress)",
            "{}: ez du buffer-helbiderik (bufferDeviceAddress)",
            name
        ));
    }
    let limits = &props.limits;
    if limits.max_compute_work_group_invocations < THREADS
        || limits.max_compute_work_group_size[0] < THREADS
    {
        return Err(t!(
            "{}: work groups of {} threads at most; the kernel uses {}",
            "{}: gehienez {} hariko lan-taldeak; kernelak {} erabiltzen ditu",
            name,
            limits.max_compute_work_group_invocations,
            THREADS
        ));
    }
    // segurtasuna: goian bezala
    let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    let Some(family) = families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
    else {
        return Err(t!(
            "{}: no compute queue",
            "{}: ez du kalkulu-ilararik",
            name
        ));
    };
    let (score, kind) = match props.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => (4, ts!("discrete GPU", "GPU diskretua")),
        vk::PhysicalDeviceType::INTEGRATED_GPU => (3, ts!("integrated GPU", "GPU integratua")),
        vk::PhysicalDeviceType::VIRTUAL_GPU => (2, ts!("virtual GPU", "GPU birtuala")),
        vk::PhysicalDeviceType::CPU => (
            1,
            ts!(
                "software Vulkan, runs on the CPU",
                "software bidezko vulkan, CPUan exekutatzen da"
            ),
        ),
        _ => (0, ts!("other device", "bestelako gailua")),
    };
    Ok(Candidate {
        physical: pd,
        family: family as u32,
        score,
        name,
        kind,
    })
}

impl Device {
    // `index`: zenbatgarren gailua mota onenekoen artean (MPI prozesu bakoitzak berea)
    pub fn open(index: usize) -> Result<Device, String> {
        // segurtasuna: sistemaren libvulkan kargatu; huts eginez gero errore bat
        let entry = unsafe { ash::Entry::load() }.map_err(|e| {
            t!(
                "no Vulkan loader found ({}); install your GPU's Vulkan driver (on Linux, Mesa or the vendor's package; mesa-vulkan-drivers includes lavapipe, a software one)",
                "ez da vulkan kargatzailerik aurkitu ({}); instalatu zure GPUaren vulkan driverra (linux-en, mesa edo fabrikatzailearen paketea; mesa-vulkan-drivers-ek lavapipe dakar, software bidezkoa)",
                e
            )
        })?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"qoa")
            .api_version(vk::API_VERSION_1_2);
        // segurtasuna: info baliozkoa da eta deiak irauten duen bitartean bizi da
        let instance = unsafe {
            entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
        }
        .map_err(|e| match e {
            vk::Result::ERROR_INCOMPATIBLE_DRIVER => t!(
                "the Vulkan loader found no driver for Vulkan 1.2 ({:?}); install your GPU's Vulkan driver",
                "vulkan kargatzaileak ez du vulkan 1.2rako driverrik aurkitu ({:?}); instalatu zure GPUaren vulkan driverra",
                e
            ),
            e => fail(ts!("creating the Vulkan instance", "vulkan instantzia sortzea"))(e),
        })?;

        let chosen = Self::choose(&instance, index);
        let c = match chosen {
            Ok(c) => c,
            Err(e) => {
                // segurtasuna: instantzia ez du beste ezerk erabiltzen
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(c.family)
            .queue_priorities(&priorities)];
        let features = vk::PhysicalDeviceFeatures::default().shader_float64(true);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_features(&features)
            .push_next(&mut f12);
        // segurtasuna: gaitutako ezaugarriak `examine`-k egiaztatu ditu
        let device = match unsafe { instance.create_device(c.physical, &info, None) } {
            Ok(d) => d,
            Err(e) => {
                // segurtasuna: goian bezala
                unsafe { instance.destroy_instance(None) };
                return Err(fail(ts!("creating the device", "gailua sortzea"))(e));
            }
        };
        // segurtasuna: c.physical instantzia honena da
        let memory = unsafe { instance.get_physical_device_memory_properties(c.physical) };
        let mut m3 = vk::PhysicalDeviceMaintenance3Properties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut m3);
        // segurtasuna: goian bezala; maintenance3 vulkan 1.1-etik oinarrizkoa da
        unsafe { instance.get_physical_device_properties2(c.physical, &mut p2) };
        let alloc_max = m3.max_memory_allocation_size.min(CHUNK_BYTES_MAX);
        let chunk_bits = (alloc_max / AMP as u64).max(1).ilog2();
        let device_bytes = memory.memory_heaps[..memory.memory_heap_count as usize]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()
            .unwrap_or(0);
        // segurtasuna: familia eta indizea gailua sortzean eskatutakoak dira
        let queue = unsafe { device.get_device_queue(c.family, 0) };
        let mut dev = Device {
            _entry: entry,
            instance,
            device,
            queue,
            pool: vk::CommandPool::null(),
            cmd: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            shader: vk::ShaderModule::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            descriptors: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            memory,
            chunk_bits,
            device_bytes,
            len: 0,
            chunks: Vec::new(),
            table: None,
            gates: None,
            staging: None,
            name: c.name,
            kind: c.kind,
        };
        // huts eginez gero `dev` askatzean sortutakoa suntsitzen da (Drop)
        dev.init(c.family)?;
        Ok(dev)
    }

    // gailurik onena, edo zergatik ez dagoen bat ere ez
    fn choose(instance: &ash::Instance, index: usize) -> Result<Candidate, String> {
        // segurtasuna: instantzia baliozkoa da
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(fail(ts!("listing devices", "gailuak zerrendatzea")))?;
        let mut usable: Vec<Candidate> = Vec::new();
        let mut rejected = Vec::new();
        for pd in devices {
            match examine(instance, pd) {
                Ok(c) => usable.push(c),
                Err(why) => rejected.push(why),
            }
        }
        // mota onenekoen artean, `index`-en arabera txandaka (adib. nodo bateko GPU guztiak)
        let top = usable.iter().map(|c| c.score).max();
        let mut best: Vec<Candidate> = usable
            .into_iter()
            .filter(|c| Some(c.score) == top)
            .collect();
        let pick = (!best.is_empty()).then(|| best.swap_remove(index % best.len()));
        pick.ok_or_else(|| {
            if rejected.is_empty() {
                t!(
                    "the Vulkan loader found no devices; is a Vulkan driver installed?",
                    "vulkan kargatzaileak ez du gailurik aurkitu; vulkan driverrik instalatuta al dago?"
                )
            } else {
                t!(
                    "no usable Vulkan device: {}",
                    "ez dago vulkan gailu erabilgarririk: {}",
                    rejected.join("; ")
                )
            }
        })
    }

    fn init(&mut self, family: u32) -> Result<(), String> {
        let d = &self.device;
        // segurtasuna: objektu guztiak gailu honekin sortzen dira, eta Drop-ek suntsitzen ditu
        unsafe {
            self.pool = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(fail(ts!(
                    "creating the command pool",
                    "komando-biltegia sortzea"
                )))?;
            self.cmd = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(fail(ts!(
                    "allocating a command buffer",
                    "komando-bufferra esleitzea"
                )))?[0];
            self.fence = d
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(fail(ts!("creating a fence", "hesia sortzea")))?;
            let words = ash::util::read_spv(&mut std::io::Cursor::new(SPIRV))
                .map_err(|e| format!("tile.spv: {e}"))?;
            self.shader = d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(fail(ts!("loading the kernel", "kernela kargatzea")))?;
            let bindings = [0, 1].map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            });
            self.set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(fail(ts!(
                    "creating the descriptor layout",
                    "deskriptoreen diseinua sortzea"
                )))?;
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(std::mem::size_of::<Push>() as u32)];
            let layouts = [self.set_layout];
            self.layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(&ranges),
                    None,
                )
                .map_err(fail(ts!(
                    "creating the pipeline layout",
                    "kanalizazioaren diseinua sortzea"
                )))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(self.shader)
                .name(c"main");
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(self.layout);
            self.pipeline = d
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| fail(ts!("compiling the kernel", "kernela konpilatzea"))(e))?[0];
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2)];
            self.descriptors = d
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(fail(ts!(
                    "creating the descriptor pool",
                    "deskriptore-biltegia sortzea"
                )))?;
            self.set = d
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptors)
                        .set_layouts(&layouts),
                )
                .map_err(fail(ts!(
                    "allocating descriptors",
                    "deskriptoreak esleitzea"
                )))?[0];
        }
        let table = self.buffer(
            (CHUNKS_MAX * 8) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &[vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT],
            false,
            true,
        )?;
        self.bind(1, &table);
        self.table = Some(table);
        Ok(())
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn kind(&self) -> &'static str {
        self.kind
    }

    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.memory.memory_type_count).find(|&i| {
            bits >> i & 1 == 1
                && self.memory.memory_types[i as usize]
                    .property_flags
                    .contains(want)
        })
    }

    // buffer bat bere memoriarekin; `want` lehentasun-ordenan; `address`: shader-ak helbidez irakurtzeko
    fn buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        want: &[vk::MemoryPropertyFlags],
        address: bool,
        map: bool,
    ) -> Result<Buf, String> {
        let d = &self.device;
        let what = ts!("allocating GPU memory", "GPUaren memoria esleitzea");
        // segurtasuna: buffer eta memoria berriak, gailu honenak; huts eginez gero askatzen dira
        unsafe {
            let buffer = d
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .map_err(fail(what))?;
            let req = d.get_buffer_memory_requirements(buffer);
            let Some(index) = want
                .iter()
                .find_map(|w| self.memory_type(req.memory_type_bits, *w))
            else {
                d.destroy_buffer(buffer, None);
                return Err(t!(
                    "GPU: no suitable memory type for a {} buffer",
                    "GPUa: ez dago memoria-mota egokirik {} bufferrerako",
                    gib(size)
                ));
            };
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut info = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(index);
            if address {
                info = info.push_next(&mut flags);
            }
            let memory = match d.allocate_memory(&info, None) {
                Ok(m) => m,
                Err(e) => {
                    d.destroy_buffer(buffer, None);
                    return Err(t!(
                        "GPU: allocating {} failed: {:?}",
                        "GPUa: {} esleitzeak huts egin du: {:?}",
                        gib(size),
                        e
                    ));
                }
            };
            let mut ptr = std::ptr::null_mut();
            let bound = d.bind_buffer_memory(buffer, memory, 0).and_then(|_| {
                if map {
                    ptr = d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                        as *mut u8;
                }
                Ok(())
            });
            if let Err(e) = bound {
                d.destroy_buffer(buffer, None);
                d.free_memory(memory, None);
                return Err(fail(what)(e));
            }
            Ok(Buf {
                buffer,
                memory,
                size,
                ptr,
            })
        }
    }

    fn free(&self, b: Buf) {
        // segurtasuna: gailua geldi dago (submit bakoitzak bere hesia itxaroten du); memoria askatzeak mapaketa
        // ere kentzen du
        unsafe {
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }

    // deskriptore-multzoaren `binding`-a buffer honetara
    fn bind(&self, binding: u32, b: &Buf) {
        let infos = [vk::DescriptorBufferInfo::default()
            .buffer(b.buffer)
            .range(vk::WHOLE_SIZE)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&infos);
        // segurtasuna: multzoa ez dago erabiltzen (gailua geldi)
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
    }

    // `len` anplitudeko egoerarako zatiak (tamaina aldatu bada, berriak)
    fn ensure_state(&mut self, len: usize) -> Result<(), String> {
        if self.len == len && !self.chunks.is_empty() {
            return Ok(());
        }
        for b in std::mem::take(&mut self.chunks) {
            self.free(b);
        }
        self.len = 0;
        let bytes = (len * AMP) as u64;
        if bytes > self.device_bytes {
            return Err(t!(
                "the state needs {} and {} has {} of memory",
                "egoerak {} behar ditu eta {}-k {} memoria ditu",
                gib(bytes),
                self.name,
                gib(self.device_bytes)
            ));
        }
        let per = 1usize << self.chunk_bits;
        let count = len.div_ceil(per);
        if count > CHUNKS_MAX {
            return Err(t!(
                "the state needs {} pieces of {} on this device; the kernel takes at most {}",
                "egoerak {}-ko {} zati behar ditu gailu honetan; kernelak gehienez {} onartzen ditu",
                count,
                gib((per * AMP) as u64),
                CHUNKS_MAX
            ));
        }
        let table = self.table.as_ref().expect("taula init-en sortua").ptr as *mut u64;
        for c in 0..count {
            let amps = per.min(len - c * per);
            let b = self.buffer(
                (amps * AMP) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                &[vk::MemoryPropertyFlags::DEVICE_LOCAL],
                true,
                false,
            )?;
            // segurtasuna: buffer-a SHADER_DEVICE_ADDRESS erabilerarekin eta helbide-memoriarekin sortua;
            // taulak CHUNKS_MAX sarrera ditu eta c < CHUNKS_MAX
            unsafe {
                let addr = self.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(b.buffer),
                );
                table.add(c).write(addr);
            }
            self.chunks.push(b);
        }
        self.len = len;
        Ok(())
    }

    // bitarteko bufferra, gutxienez `bytes` (gehienez STAGING_MAX)
    fn ensure_staging(&mut self, bytes: usize) -> Result<usize, String> {
        let want = bytes.min(STAGING_MAX) as u64;
        if self.staging.as_ref().is_none_or(|s| s.size < want) {
            if let Some(old) = self.staging.take() {
                self.free(old);
            }
            let host =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            // cache-dun memoria lehenik: gailutik irakurtzea askoz azkarragoa da
            let b = self.buffer(
                want,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                &[host | vk::MemoryPropertyFlags::HOST_CACHED, host],
                false,
                true,
            )?;
            self.staging = Some(b);
        }
        Ok(self.staging.as_ref().unwrap().size as usize)
    }

    fn ensure_gates(&mut self, count: usize) -> Result<(), String> {
        let want = (count.max(64) * std::mem::size_of::<GpuGate>()) as u64;
        if self.gates.as_ref().is_none_or(|g| g.size < want) {
            if let Some(old) = self.gates.take() {
                self.free(old);
            }
            let b = self.buffer(
                want.next_power_of_two(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                &[vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT],
                false,
                true,
            )?;
            self.bind(0, &b);
            self.gates = Some(b);
        }
        Ok(())
    }

    // komando-buffer bat grabatu, bidali eta amaitu arte itxaron
    fn submit(&self, record: impl FnOnce(&ash::Device, vk::CommandBuffer)) -> Result<(), String> {
        let d = &self.device;
        let what = ts!("running commands", "komandoak exekutatzea");
        // segurtasuna: komando-buffer bakarra, aurreko bidalketa amaituta (hesia itxaron da)
        unsafe {
            d.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .map_err(fail(what))?;
            d.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(fail(what))?;
            record(d, self.cmd);
            d.end_command_buffer(self.cmd).map_err(fail(what))?;
            d.reset_fences(&[self.fence]).map_err(fail(what))?;
            let cmds = [self.cmd];
            d.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                self.fence,
            )
            .map_err(fail(what))?;
            d.wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(fail(what))?;
        }
        Ok(())
    }

    // egoera zatika: (zatia, lehen anplitudea zatian, anplitude-kopurua), bitarteko bufferrean sartzen direnak
    fn pieces(&self, staging_amps: usize) -> Vec<(usize, usize, usize)> {
        let per = 1usize << self.chunk_bits;
        let mut out = Vec::new();
        for c in 0..self.chunks.len() {
            let n = per.min(self.len - c * per);
            let mut at = 0;
            while at < n {
                let k = staging_amps.min(n - at);
                out.push((c, at, k));
                at += k;
            }
        }
        out
    }

    fn upload(&mut self, amps: &[C]) -> Result<(), String> {
        let staging_amps = self.ensure_staging(amps.len() * AMP)? / AMP;
        let staging = self.staging.as_ref().unwrap();
        for (c, at, k) in self.pieces(staging_amps) {
            let start = (c << self.chunk_bits) + at;
            // segurtasuna: k anplitude bitarteko bufferrean sartzen dira (k <= staging_amps), eta hura ez dago
            // erabiltzen (aurreko bidalketa amaituta)
            unsafe {
                std::ptr::copy_nonoverlapping(
                    amps[start..start + k].as_ptr() as *const u8,
                    staging.ptr,
                    k * AMP,
                );
            }
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: (at * AMP) as u64,
                size: (k * AMP) as u64,
            };
            let dst = self.chunks[c].buffer;
            self.submit(|d, cmd| {
                // segurtasuna: bufferrak baliozkoak eta tamaina egokikoak
                unsafe { d.cmd_copy_buffer(cmd, staging.buffer, dst, &[region]) };
            })?;
        }
        Ok(())
    }

    fn download(&mut self, amps: &mut [C]) -> Result<(), String> {
        let staging_amps = self.ensure_staging(amps.len() * AMP)? / AMP;
        let staging = self.staging.as_ref().unwrap();
        for (c, at, k) in self.pieces(staging_amps) {
            let start = (c << self.chunk_bits) + at;
            let region = vk::BufferCopy {
                src_offset: (at * AMP) as u64,
                dst_offset: 0,
                size: (k * AMP) as u64,
            };
            let src = self.chunks[c].buffer;
            self.submit(|d, cmd| {
                // segurtasuna: bufferrak baliozkoak; barrerek kalkuluaren idazketak kopiarentzat eta kopia
                // ostalariarentzat ikusgai egiten dituzte
                unsafe {
                    barrier(
                        d,
                        cmd,
                        (
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::AccessFlags::SHADER_WRITE,
                        ),
                        (
                            vk::PipelineStageFlags::TRANSFER,
                            vk::AccessFlags::TRANSFER_READ,
                        ),
                    );
                    d.cmd_copy_buffer(cmd, src, staging.buffer, &[region]);
                    barrier(
                        d,
                        cmd,
                        (
                            vk::PipelineStageFlags::TRANSFER,
                            vk::AccessFlags::TRANSFER_WRITE,
                        ),
                        (vk::PipelineStageFlags::HOST, vk::AccessFlags::HOST_READ),
                    );
                }
            })?;
            // segurtasuna: kopia amaitu da (hesia), eta k anplitude bufferrean daude
            unsafe {
                std::ptr::copy_nonoverlapping(
                    staging.ptr as *const u8,
                    amps[start..start + k].as_mut_ptr() as *mut u8,
                    k * AMP,
                );
            }
        }
        Ok(())
    }

    // etapak egoeran: igo, kalkulatu, jaitsi. itzultzen du (mugitzen emandako denbora, kalkulatzen emandakoa)
    pub fn apply(&mut self, amps: &mut [C], stages: &[Stage]) -> Result<(f64, f64), String> {
        let len = amps.len();
        self.ensure_state(len)?;
        let total: usize = stages.iter().map(|s| s.gates.len()).sum();
        self.ensure_gates(total)?;
        let gates = self.gates.as_ref().unwrap().ptr as *mut GpuGate;
        let mut first = 0;
        let mut pushes = Vec::with_capacity(stages.len());
        let groups = len.div_ceil(TILE);
        for s in stages {
            // segurtasuna: bufferrak `total` ate hartzen ditu (ensure_gates) eta ez dago erabiltzen
            unsafe {
                std::ptr::copy_nonoverlapping(s.gates.as_ptr(), gates.add(first), s.gates.len());
            }
            pushes.push(Push {
                pos: s.pos,
                low: s.low,
                first: first as u32,
                count: s.gates.len() as u32,
                len: len as u32,
                groups: groups as u32,
                chunk_bits: self.chunk_bits,
            });
            first += s.gates.len();
        }

        let t0 = Instant::now();
        self.upload(amps)?;
        let up = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let x = groups.min(GROUPS_X) as u32;
        let y = groups.div_ceil(GROUPS_X) as u32;
        let (pipeline, layout, set) = (self.pipeline, self.layout, self.set);
        self.submit(|d, cmd| {
            let compute = (
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            );
            // segurtasuna: kanalizazioa, diseinua eta multzoa baliozkoak; push-a diseinuaren tartearen tamainakoa
            unsafe {
                barrier(
                    d,
                    cmd,
                    (
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::TRANSFER_WRITE,
                    ),
                    compute,
                );
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[set],
                    &[],
                );
                for p in &pushes {
                    let bytes = std::slice::from_raw_parts(
                        p as *const Push as *const u8,
                        std::mem::size_of::<Push>(),
                    );
                    d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
                    d.cmd_dispatch(cmd, x, y, 1);
                    barrier(
                        d,
                        cmd,
                        (
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::AccessFlags::SHADER_WRITE,
                        ),
                        compute,
                    );
                }
            }
        })?;
        let compute = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        self.download(amps)?;
        Ok((up + t2.elapsed().as_secs_f64(), compute))
    }
}

// memoria-barrera bat: (etapa, sarbidea) -> (etapa, sarbidea)
//
// segurtasuna: `cmd` grabatzen ari den komando-buffer bat da
unsafe fn barrier(
    d: &ash::Device,
    cmd: vk::CommandBuffer,
    src: (vk::PipelineStageFlags, vk::AccessFlags),
    dst: (vk::PipelineStageFlags, vk::AccessFlags),
) {
    let b = vk::MemoryBarrier::default()
        .src_access_mask(src.1)
        .dst_access_mask(dst.1);
    // segurtasuna: deitzaileak bermatzen du
    unsafe {
        d.cmd_pipeline_barrier(
            cmd,
            src.0,
            dst.0,
            vk::DependencyFlags::empty(),
            &[b],
            &[],
            &[],
        );
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // segurtasuna: gailua geldi ondoren, sortutako guztia suntsitu (handle nuluak ez dira ezer)
        unsafe {
            let _ = self.device.device_wait_idle();
            let bufs: Vec<Buf> = std::mem::take(&mut self.chunks)
                .into_iter()
                .chain(self.table.take())
                .chain(self.gates.take())
                .chain(self.staging.take())
                .collect();
            for b in bufs {
                self.free(b);
            }
            let d = &self.device;
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_pool(self.descriptors, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
            d.destroy_shader_module(self.shader, None);
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.pool, None);
            d.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
