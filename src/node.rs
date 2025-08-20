use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, RwLock, atomic::{AtomicUsize, Ordering}};
use std::io::{Read, Write, self, ErrorKind};
use std::thread;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use std::sync::mpsc::{channel, sync_channel, Sender, Receiver, SyncSender};
use rayon::prelude::*;
use crossbeam_channel as cb;
use crossbeam_utils::sync::{ShardedLock, WaitGroup};
use parking_lot::{RwLock as PLRwLock, Mutex as PLMutex};
use bytes::{Buf, BufMut, BytesMut};
use lz4_flex::block::{compress_prepend_size, decompress_size_prepended};
use slab::Slab;
use socket2::{Socket, Domain, Type, Protocol};
use mio::{Events, Interest, Poll, Token};

#[cfg(feature = "profile")]
use pprof::ProfilerGuard;

#[cfg(feature = "mpi")]
use mpi::prelude::*;

// For Vulkan
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, physical::{PhysicalDevice, PhysicalDeviceType}},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator, StandardDescriptorSetAllocator},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo},
    sync::{self, GpuFuture},
    VulkanLibrary,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;

// configuration for the distributed simulation with network optimizations
pub struct DistributedConfig {
    // number of qubits in the simulation
    pub total_qubits: usize,
    // communication backend type
    pub backend: CommunicationBackend,
    // node addresses if using tcp
    pub nodes: Vec<String>,
    // this node's id (0-indexed)
    pub node_id: usize,
    // port to use for tcp communication
    pub port: u16,
    // chunk size for rendering work distribution
    pub render_chunk_size: usize,
    // maximum memory usage per node (bytes)
    pub max_memory_per_node: usize,
    // compression level for state transfers (0-9)
    pub compression_level: u32,
    // network buffer size for optimized transmission
    pub network_buffer_size: usize,
    // use zero-copy operations where available
    pub use_zero_copy: bool,
    // socket send buffer size
    pub socket_send_buffer: usize,
    // socket receive buffer size
    pub socket_recv_buffer: usize,
    // maximum number of concurrent transfers
    pub max_concurrent_transfers: usize,
    // network priority (0-7, higher is more priority)
    pub network_priority: u8,
    // use direct memory access where available
    pub use_rdma: bool,
    // io thread count (0 for auto)
    pub io_threads: usize,
    // maximum batched messages before flush
    pub max_batch_size: usize,
    // synchronization mode
    pub sync_mode: SyncMode,
    // heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
    // acknowledgment timeout in milliseconds
    pub ack_timeout_ms: u64,
    // use multiple streams per node for parallelism
    pub streams_per_node: usize,
    // enable io_uring for linux
    #[cfg(target_os = "linux")]
    pub use_io_uring: bool,
    // ip type of service (tos) for priority
    pub ip_tos: u8,
    // enable tcp fast open
    pub tcp_fast_open: bool,
    // enable hybrid computing (GPU/CPU/DPU/TPU)
    pub use_hybrid_computing: bool,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            total_qubits: 0,
            backend: CommunicationBackend::TCP,
            nodes: vec![],
            node_id: 0,
            port: 9000,
            render_chunk_size: 1024,
            max_memory_per_node: 8 * 1024 * 1024 * 1024, // 8 gb default
            compression_level: 3,
            network_buffer_size: 8 * 1024 * 1024, // 8 mb buffer
            use_zero_copy: true,
            socket_send_buffer: 16 * 1024 * 1024, // 16 mb send buffer
            socket_recv_buffer: 16 * 1024 * 1024, // 16 mb receive buffer
            max_concurrent_transfers: 4,
            network_priority: 7, // maximum priority
            use_rdma: false, // rdma off by default
            io_threads: 0, // auto (uses number of physical cores)
            max_batch_size: 64,
            sync_mode: SyncMode::Optimistic,
            heartbeat_interval_ms: 100,
            ack_timeout_ms: 1000,
            streams_per_node: 4, // 4 streams per node for parallelism
            #[cfg(target_os = "linux")]
            use_io_uring: true,
            ip_tos: 0x10, // low delay tos
            tcp_fast_open: true,
            use_hybrid_computing: false,
        }
    }
}

// available communication backends
pub enum CommunicationBackend {
    MPI,
    TCP,
    RDMA,
    SharedMemory,
}

// synchronization modes
pub enum SyncMode {
    // traditional barrier synchronization
    Barrier,
    // lock-free synchronization
    LockFree,
    // optimistic concurrency control
    Optimistic,
    // eventual consistency (async updates)
    Eventually,
}

// message types for inter-node communication with batching
#[derive(Debug, Clone, Copy, PartialEq)]
enum MessageType {
    GateRequest = 0,
    MeasurementRequest = 1,
    StateTransfer = 2,
    RenderRequest = 3,
    RenderResult = 4,
    Synchronization = 5,
    Terminate = 6,
    Heartbeat = 7,
    Acknowledgment = 8,
    WorkSteal = 9,
    LoadReport = 10,
}

// network message with optimized binary format and compression
struct NetworkMessage {
    msg_type: MessageType,
    sequence: u64,
    source_node: u16,
    target_node: u16,
    payload_size: u32,
    payload: BytesMut,
    priority: u8,
    requires_ack: bool,
    compressed: bool,
}

impl NetworkMessage {
    fn new(msg_type: MessageType, source: u16, target: u16, mut payload: BytesMut, compression_level: u32) -> Self {
        static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
        let seq = SEQUENCE.fetch_add(1, Ordering::SeqCst) as u64;
        
        let compressed = compression_level > 0;
        if compressed {
            let compressed_payload = compress_prepend_size(&payload, compression_level as i32);
            payload = BytesMut::from(&compressed_payload[..]);
        }
        
        Self {
            msg_type,
            sequence: seq,
            source_node: source,
            target_node: target,
            payload_size: payload.len() as u32,
            payload,
            priority: 0,
            requires_ack: false,
            compressed,
        }
    }
    
    fn serialize(&self) -> BytesMut {
        let mut buf = BytesMut::with_capacity(16 + self.payload.len());
        buf.put_u8(self.msg_type as u8);
        let flags = (self.priority << 2) | ((self.requires_ack as u8) << 1) | (self.compressed as u8);
        buf.put_u8(flags);
        buf.put_u16(self.source_node);
        buf.put_u16(self.target_node);
        buf.put_u64(self.sequence);
        buf.put_u32(self.payload_size);
        buf.put_slice(&self.payload);
        buf
    }
    
    fn deserialize(mut buf: BytesMut) -> Result<Self, String> {
        if buf.len() < 16 {
            return Err("buffer too small for message header".to_string());
        }
        
        let msg_type = match buf.get_u8() {
            0 => MessageType::GateRequest,
            1 => MessageType::MeasurementRequest,
            2 => MessageType::StateTransfer,
            3 => MessageType::RenderRequest,
            4 => MessageType::RenderResult,
            5 => MessageType::Synchronization,
            6 => MessageType::Terminate,
            7 => MessageType::Heartbeat,
            8 => MessageType::Acknowledgment,
            9 => MessageType::WorkSteal,
            10 => MessageType::LoadReport,
            _ => return Err("unknown message type".to_string()),
        };
        
        let flags = buf.get_u8();
        let priority = flags >> 2;
        let requires_ack = ((flags >> 1) & 1) != 0;
        let compressed = (flags & 1) != 0;
        
        let source_node = buf.get_u16();
        let target_node = buf.get_u16();
        let sequence = buf.get_u64();
        let payload_size = buf.get_u32() as usize;
        
        if buf.len() < payload_size {
            return Err("buffer too small for payload".to_string());
        }
        
        let mut payload = buf.split_to(payload_size);
        
        if compressed {
            let decompressed = decompress_size_prepended(&payload).map_err(|e| format!("decompression failed: {}", e))?;
            payload = BytesMut::from(&decompressed[..]);
        }
        
        Ok(Self {
            msg_type,
            sequence,
            source_node,
            target_node,
            payload_size: payload.len() as u32,
            payload,
            priority,
            requires_ack,
            compressed: false, // decompressed now
        })
    }
}

// batched message container for efficient network transmission
struct MessageBatch {
    messages: Vec<NetworkMessage>,
    total_size: usize,
    timestamp: Instant,
}

impl MessageBatch {
    fn new() -> Self {
        Self {
            messages: Vec::with_capacity(64),
            total_size: 0,
            timestamp: Instant::now(),
        }
    }
    
    fn add(&mut self, message: NetworkMessage) -> bool {
        let msg_size = 16 + message.payload.len();
        if self.total_size + msg_size > MAX_BATCH_SIZE {
            return false;
        }
        
        self.total_size += msg_size;
        self.messages.push(message);
        true
    }
    
    fn serialize(&self) -> BytesMut {
        let mut buf = BytesMut::with_capacity(self.total_size + 4);
        buf.put_u32(self.messages.len() as u32);
        
        for message in &self.messages {
            let msg_data = message.serialize();
            buf.put_slice(&msg_data);
        }
        
        buf
    }
}

// constants for network optimization
const MAX_BATCH_SIZE: usize = 1024 * 1024; // 1mb max batch size
const BATCH_FLUSH_INTERVAL_MS: u64 = 5; // flush batches every 5ms
const MAX_SOCKET_BACKLOG: usize = 1024;
const HEARTBEAT_INTERVAL_MS: u64 = 100;

// optimized network connection with batching, zero-copy, and multi-stream support
struct OptimizedConnection {
    streams: Vec<TcpStream>,
    read_buffers: Vec<BytesMut>,
    write_buffers: Vec<BytesMut>,
    write_batches: Vec<MessageBatch>,
    last_flushes: Vec<Instant>,
    node_id: usize,
    metrics: ConnectionMetrics,
    zero_copy: bool,
    current_stream: AtomicUsize,
}

// connection performance metrics
struct ConnectionMetrics {
    bytes_sent: AtomicUsize,
    bytes_received: AtomicUsize,
    messages_sent: AtomicUsize,
    messages_received: AtomicUsize,
    transfer_latency: PLMutex<VecDeque<Duration>>,
}

impl ConnectionMetrics {
    fn new() -> Self {
        Self {
            bytes_sent: AtomicUsize::new(0),
            bytes_received: AtomicUsize::new(0),
            messages_sent: AtomicUsize::new(0),
            messages_received: AtomicUsize::new(0),
            transfer_latency: PLMutex::new(VecDeque::with_capacity(100)),
        }
    }
    
    fn record_send(&self, bytes: usize) {
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
        self.messages_sent.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_receive(&self, bytes: usize) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
        self.messages_received.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_latency(&self, latency: Duration) {
        let mut latencies = self.transfer_latency.lock();
        if latencies.len() >= 100 {
            latencies.pop_front();
        }
        latencies.push_back(latency);
    }
    
    fn average_latency(&self) -> Duration {
        let latencies = self.transfer_latency.lock();
        if latencies.is_empty() {
            return Duration::from_micros(0);
        }
        
        let total: u64 = latencies.iter()
            .map(|d| d.as_micros() as u64)
            .sum();
        Duration::from_micros(total / latencies.len() as u64)
    }
}

impl OptimizedConnection {
    fn new(streams: Vec<TcpStream>, node_id: usize, buffer_size: usize, zero_copy: bool) -> io::Result<Self> {
        let num_streams = streams.len();
        let mut read_buffers = Vec::with_capacity(num_streams);
        let mut write_buffers = Vec::with_capacity(num_streams);
        let mut write_batches = Vec::with_capacity(num_streams);
        let mut last_flushes = Vec::with_capacity(num_streams);
        
        for stream in &streams {
            stream.set_nodelay(true)?;
            stream.set_nonblocking(true)?;
        }
        
        for _ in 0..num_streams {
            read_buffers.push(BytesMut::with_capacity(buffer_size));
            write_buffers.push(BytesMut::with_capacity(buffer_size));
            write_batches.push(MessageBatch::new());
            last_flushes.push(Instant::now());
        }
        
        Ok(Self {
            streams,
            read_buffers,
            write_buffers,
            write_batches,
            last_flushes,
            node_id,
            metrics: ConnectionMetrics::new(),
            zero_copy,
            current_stream: AtomicUsize::new(0),
        })
    }
    
    fn get_stream_index(&self) -> usize {
        let idx = self.current_stream.fetch_add(1, Ordering::Relaxed);
        idx % self.streams.len()
    }
    
    fn send_message(&mut self, message: NetworkMessage) -> io::Result<()> {
        let stream_idx = self.get_stream_index();
        if !self.write_batches[stream_idx].add(message.clone()) {
            self.flush_stream(stream_idx)?;
            self.write_batches[stream_idx] = MessageBatch::new();
            self.write_batches[stream_idx].add(message);
        }
        
        if self.last_flushes[stream_idx].elapsed().as_millis() > BATCH_FLUSH_INTERVAL_MS as u128 {
            self.flush_stream(stream_idx)?;
        }
        
        Ok(())
    }
    
    fn flush_stream(&mut self, stream_idx: usize) -> io::Result<()> {
        if self.write_batches[stream_idx].messages.is_empty() {
            return Ok(());
        }
        
        let batch_data = self.write_batches[stream_idx].serialize();
        self.metrics.record_send(batch_data.len());
        
        if self.zero_copy {
            self.write_zero_copy(stream_idx, &batch_data)?;
        } else {
            self.streams[stream_idx].write_all(&batch_data)?;
        }
        
        self.last_flushes[stream_idx] = Instant::now();
        self.write_batches[stream_idx] = MessageBatch::new();
        
        Ok(())
    }
    
    fn write_zero_copy(&mut self, stream_idx: usize, data: &[u8]) -> io::Result<()> {
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::io::AsRawFd;
            use nix::sys::socket::sendfile;
            
            let file = tempfile::tempfile()?;
            std::io::Write::write_all(&mut &file, data)?;
            file.sync_all()?;
            
            let fd = file.as_raw_fd();
            let socket_fd = self.streams[stream_idx].as_raw_fd();
            
            sendfile(socket_fd, fd, None, data.len())
                .map(|_| ())
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            self.streams[stream_idx].write_all(data)
        }
    }
    
    fn try_receive_messages(&mut self) -> io::Result<Vec<NetworkMessage>> {
        let mut messages = Vec::new();
        let mut total_bytes_read = 0;
        
        for stream_idx in 0..self.streams.len() {
            let mut bytes_read = 0;
            loop {
                if self.read_buffers[stream_idx].capacity() - self.read_buffers[stream_idx].len() < 4096 {
                    self.read_buffers[stream_idx].reserve(4096);
                }
                
                let buf_mut = unsafe { 
                    &mut *(self.read_buffers[stream_idx].chunk_mut() as *mut _ as *mut [u8])
                };
                
                match self.streams[stream_idx].read(buf_mut) {
                    Ok(0) => break,
                    Ok(n) => {
                        bytes_read += n;
                        unsafe { self.read_buffers[stream_idx].advance_mut(n); }
                    },
                    Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(e) => return Err(e),
                }
            }
            
            if bytes_read > 0 {
                total_bytes_read += bytes_read;
                self.metrics.record_receive(bytes_read);
            }
            
            while self.read_buffers[stream_idx].len() >= 4 {
                let batch_size = self.read_buffers[stream_idx].get_u32() as usize;
                
                for _ in 0..batch_size {
                    if let Ok(message) = NetworkMessage::deserialize(self.read_buffers[stream_idx].clone()) {
                        let msg_size = 16 + message.payload.len();
                        if self.read_buffers[stream_idx].len() >= msg_size {
                            self.read_buffers[stream_idx].advance(msg_size);
                            messages.push(message);
                        } else {
                            break;
                        }
                    } else {
                        self.read_buffers[stream_idx].clear();
                        break;
                    }
                }
            }
        }
        
        Ok(messages)
    }
}

// high-performance distributed state vector simulation with network optimizations
pub struct DistributedSimulation {
    config: DistributedConfig,
    // local partition of the state vector
    local_state: Arc<PLRwLock<StateVector>>,
    // maps global qubit indices to node ids
    qubit_mapping: HashMap<usize, usize>,
    // optimized connections for tcp mode
    connections: Option<HashMap<usize, PLMutex<OptimizedConnection>>>,
    // lockless synchronization barrier
    barrier: Arc<ShardedLock<BarrierState>>,
    // rendering work queue
    render_queue: Arc<PLMutex<VecDeque<RenderWorkItem>>>,
    // rendering results
    render_results: Arc<PLMutex<HashMap<usize, RenderChunk>>>,
    // worker threads
    worker_threads: Vec<thread::JoinHandle<()>>,
    // io threads for network processing
    io_threads: Vec<thread::JoinHandle<()>>,
    // work distribution channels
    work_tx: Option<SyncSender<RenderWorkItem>>,
    work_rx: Option<Arc<PLMutex<Receiver<RenderWorkItem>>>>,
    // result collection channels
    result_tx: Option<Sender<RenderChunk>>,
    result_rx: Option<Arc<PLMutex<Receiver<RenderChunk>>>>,
    // command channel for io threads
    io_command_tx: Option<cb::Sender<IoCommand>>,
    io_command_rx: Option<Arc<PLMutex<cb::Receiver<IoCommand>>>>,
    // memory manager
    memory_manager: MemoryManager,
    // load balancer
    load_balancer: LoadBalancer,
    // network metrics
    network_metrics: Arc<NetworkMetrics>,
    // node status tracker
    node_status: Arc<NodeStatusTracker>,
    // heartbeat manager
    heartbeat_manager: Arc<HeartbeatManager>,
    // Vulkan context
    vulkan_device: Option<Arc<Device>>,
    vulkan_queue: Option<Arc<Queue>>,
    vulkan_memory_allocator: Option<Arc<StandardMemoryAllocator>>,
    vulkan_descriptor_set_allocator: Option<Arc<StandardDescriptorSetAllocator>>,
    vulkan_command_buffer_allocator: Option<Arc<StandardCommandBufferAllocator>>,
    #[cfg(feature = "mpi")]
    mpi_world: Option<SystemCommunicator>,
}

// Shader module
mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer ReData {
    float data[];
} re_buf;

layout(set = 0, binding = 1) buffer ImData {
    float data[];
} im_buf;

layout(set = 0, binding = 2) buffer OutputData {
    uvec4 data[];
} out_buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float re = re_buf.data[idx];
    float im = im_buf.data[idx];
    float prob = re * re + im * im;
    uint intensity = uint(min(prob, 1.0) * 255.0);
    out_buf.data[idx] = uvec4(intensity, intensity, intensity, 255u);
}
        "
    }
}

// network performance metrics
struct NetworkMetrics {
    // total bytes sent
    bytes_sent: AtomicUsize,
    // total bytes received
    bytes_received: AtomicUsize,
    // transfer bandwidth (bytes/sec)
    bandwidth: PLMutex<VecDeque<f64>>,
    // message latency (microseconds)
    latency: PLMutex<VecDeque<u64>>,
    // compression ratio
    compression_ratio: PLMutex<VecDeque<f64>>,
}

// commands for io threads
enum IoCommand {
    SendMessage { node_id: usize, message: NetworkMessage },
    FlushConnections,
    Shutdown,
}

// node status tracking
struct NodeStatusTracker {
    status: PLRwLock<HashMap<usize, NodeStatus>>,
    last_updated: PLRwLock<HashMap<usize, Instant>>,
}

// individual node status
#[derive(Clone, Copy, PartialEq)]
enum NodeStatus {
    Connected,
    Busy,
    Disconnected,
    Failed,
}

// heartbeat manager
struct HeartbeatManager {
    last_heartbeat: PLRwLock<HashMap<usize, Instant>>,
    missed_beats: PLRwLock<HashMap<usize, usize>>,
    interval: Duration,
}

// memory manager with numa awareness and zero-copy operations
struct MemoryManager {
    max_memory: usize,
    current_usage: Arc<AtomicUsize>,
    cached_states: PLRwLock<HashMap<usize, StateVector>>,
    numa_regions: Vec<NumaRegion>,
    allocator: Option<Arc<CustomAllocator>>,
}

// numa memory region
struct NumaRegion {
    node_id: usize,
    base_address: *mut u8,
    size: usize,
    used: AtomicUsize,
}

// custom memory allocator for quantum states
struct CustomAllocator {
    // implementation details omitted
}

// load balancer with adaptive work distribution
struct LoadBalancer {
    node_capacities: PLRwLock<HashMap<usize, f64>>,
    performance_history: PLRwLock<HashMap<usize, VecDeque<Duration>>>,
    last_distribution: PLRwLock<HashMap<usize, usize>>,
    pending_requests: Arc<AtomicUsize>,
    completed_requests: Arc<AtomicUsize>,
    work_stealing_enabled: bool,
}

// work item for distributed rendering
#[derive(Debug)]
struct RenderWorkItem {
    start_idx: usize,
    end_idx: usize,
    parameters: RenderParams,
    priority: u8,
}

// parameters for rendering configuration
#[derive(Clone, Debug)]
pub struct RenderParams {
    pub resolution: (usize, usize),
    pub depth: usize,
    pub use_gpu: bool,
    pub visualization_type: VisualizationType,
}

// visualization types for quantum state
#[derive(Clone, Copy, Debug)]
pub enum VisualizationType {
    Probability,
    Bloch,
    Wigner,
    Custom(u8),
}

// represents a chunk of rendered data
#[derive(Clone)]
struct RenderChunk {
    start_idx: usize,
    end_idx: usize,
    data: Vec<u8>,
}

// complex number representation with simd optimization
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    #[inline(always)]
    fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }
    
    #[inline(always)]
    fn zero() -> Self {
        Self { real: 0.0, imag: 0.0 }
    }
    
    #[inline(always)]
    fn one() -> Self {
        Self { real: 1.0, imag: 0.0 }
    }
    
    #[inline(always)]
    fn abs_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }
}

// optimization: use a struct-of-arrays (soa) layout for the state vector.
// this improves cache performance and is more amenable to simd operations,
// as real and imaginary components are stored in contiguous memory blocks.
#[derive(Clone)]
pub struct StateVector {
    re: Vec<f64>,
    im: Vec<f64>,
}

impl StateVector {
    fn new(size: usize) -> Self {
        Self {
            re: vec![0.0; size],
            im: vec![0.0; size],
        }
    }

    fn len(&self) -> usize {
        self.re.len()
    }

    fn is_empty(&self) -> bool {
        self.re.is_empty()
    }

    fn set_initial_state(&mut self) {
        if !self.is_empty() {
            self.re[0] = 1.0;
            // all other elements are already 0.0
        }
    }
}

// internal barrier state for synchronization
struct BarrierState {
    count: AtomicUsize,
    total: usize,
    generation: AtomicUsize,
}

impl DistributedSimulation {
    // create a new distributed simulation with network optimizations
    pub fn new(config: DistributedConfig) -> Result<Self, String> {
        let num_nodes = config.nodes.len();
        if num_nodes == 0 {
            return Err("at least one node is required".to_string());
        }
        
        // calculate local state vector size based on partitioning
        let qubits_per_node = config.total_qubits / num_nodes;
        let remainder = config.total_qubits % num_nodes;
        
        let local_qubits = qubits_per_node + if config.node_id < remainder { 1 } else { 0 };
        let local_state_size = 1 << local_qubits;
        
        // create work channels - using bounded channels for backpressure
        let queue_size = num_cpus::get() * 4;
        let (work_tx, work_rx) = sync_channel(queue_size);
        let (result_tx, result_rx) = channel();
        
        // create io command channel
        let (io_tx, io_rx) = cb::bounded(1024);
        
        // create node status tracker
        let node_status = Arc::new(NodeStatusTracker {
            status: PLRwLock::new(HashMap::new()),
            last_updated: PLRwLock::new(HashMap::new()),
        });
        
        // initialize all nodes as connected
        {
            let mut status = node_status.status.write();
            let mut updated = node_status.last_updated.write();
            for node_id in 0..num_nodes {
                status.insert(node_id, NodeStatus::Connected);
                updated.insert(node_id, Instant::now());
            }
        }
        
        // create heartbeat manager
        let heartbeat_manager = Arc::new(HeartbeatManager {
            last_heartbeat: PLRwLock::new(HashMap::new()),
            missed_beats: PLRwLock::new(HashMap::new()),
            interval: Duration::from_millis(config.heartbeat_interval_ms),
        });
        
        // initialize memory manager with numa awareness if available
        let mut numa_regions = Vec::new();
        
        #[cfg(target_os = "linux")]
        {
            // on linux, attempt to set up numa memory regions
            if let Ok(numa_nodes) = numa::numa_num_configured_nodes() {
                for node in 0..numa_nodes {
                    // allocate a large chunk of memory on this numa node
                    let size = config.max_memory_per_node / numa_nodes as usize;
                    if let Ok(ptr) = numa::numa_alloc_onnode(size, node) {
                        numa_regions.push(NumaRegion {
                            node_id: node as usize,
                            base_address: ptr as *mut u8,
                            size,
                            used: AtomicUsize::new(0),
                        });
                    }
                }
            }
        }
        
        let (vulkan_device, vulkan_queue, vulkan_memory_allocator, vulkan_descriptor_set_allocator, vulkan_command_buffer_allocator) = if config.use_hybrid_computing {
            let library = VulkanLibrary::new().map_err(|e| format!("failed to load Vulkan library: {}", e))?;
            let instance = Instance::new(library, InstanceCreateInfo::default()).map_err(|e| format!("failed to create Vulkan instance: {}", e))?;
            let physical_device = instance
                .enumerate_physical_devices()
                .map_err(|e| format!("failed to enumerate physical devices: {}", e))?
                .min_by_key(|p| match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::Cpu => 2,
                    _ => 3,
                }).ok_or("no Vulkan physical device found".to_string())?;
            let queue_family_index = physical_device
                .queue_family_properties()
                .iter()
                .enumerate()
                .position(|(_, q)| q.queue_flags.compute)
                .ok_or("no compute queue family found".to_string())? as u32;
            let (device, mut queues) = Device::new(
                physical_device,
                DeviceCreateInfo {
                    queue_create_infos: vec![QueueCreateInfo {
                        queue_family_index,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ).map_err(|e| format!("failed to create Vulkan device: {}", e))?;
            let queue = queues.next().ok_or("no Vulkan queue created".to_string())?;
            let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
            let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default()));
            let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));
            (Some(Arc::new(device)), Some(Arc::new(queue)), Some(memory_allocator), Some(descriptor_set_allocator), Some(command_buffer_allocator))
        } else {
            (None, None, None, None, None)
        };
        
        #[cfg(feature = "mpi")]
        let mpi_world = if matches!(config.backend, CommunicationBackend::MPI) {
            let universe = mpi::initialize().map_err(|e| format!("MPI init failed: {}", e))?;
            Some(universe.world())
        } else {
            None
        };

        let mut sim = DistributedSimulation {
            config,
            local_state: Arc::new(PLRwLock::new(StateVector::new(local_state_size))),
            qubit_mapping: HashMap::new(),
            connections: None,
            barrier: Arc::new(ShardedLock::new(BarrierState {
                count: AtomicUsize::new(0),
                total: num_nodes,
                generation: AtomicUsize::new(0),
            })),
            render_queue: Arc::new(PLMutex::new(VecDeque::new())),
            render_results: Arc::new(PLMutex::new(HashMap::new())),
            worker_threads: Vec::new(),
            io_threads: Vec::new(),
            work_tx: Some(work_tx),
            work_rx: Some(Arc::new(PLMutex::new(work_rx))),
            result_tx: Some(result_tx),
            result_rx: Some(Arc::new(PLMutex::new(result_rx))),
            io_command_tx: Some(io_tx),
            io_command_rx: Some(Arc::new(PLMutex::new(io_rx))),
            memory_manager: MemoryManager {
                max_memory: config.max_memory_per_node,
                current_usage: Arc::new(AtomicUsize::new(0)),
                cached_states: PLRwLock::new(HashMap::new()),
                numa_regions,
                allocator: None,
            },
            load_balancer: LoadBalancer {
                node_capacities: PLRwLock::new(HashMap::new()),
                performance_history: PLRwLock::new(HashMap::new()),
                last_distribution: PLRwLock::new(HashMap::new()),
                pending_requests: Arc::new(AtomicUsize::new(0)),
                completed_requests: Arc::new(AtomicUsize::new(0)),
                work_stealing_enabled: true,
            },
            network_metrics: Arc::new(NetworkMetrics {
                bytes_sent: AtomicUsize::new(0),
                bytes_received: AtomicUsize::new(0),
                bandwidth: PLMutex::new(VecDeque::with_capacity(100)),
                latency: PLMutex::new(VecDeque::with_capacity(100)),
                compression_ratio: PLMutex::new(VecDeque::with_capacity(100)),
            }),
            node_status,
            heartbeat_manager,
            vulkan_device,
            vulkan_queue,
            vulkan_memory_allocator,
            vulkan_descriptor_set_allocator,
            vulkan_command_buffer_allocator,
            #[cfg(feature = "mpi")]
            mpi_world,
        };
        
        // initialize the qubit mapping (which qubit is on which node)
        sim.initialize_qubit_mapping();
        
        // initialize communication backend
        match sim.config.backend {
            CommunicationBackend::TCP => sim.initialize_tcp_backend()?,
            CommunicationBackend::MPI => sim.initialize_mpi_backend()?,
            CommunicationBackend::RDMA => sim.initialize_rdma_backend()?,
            CommunicationBackend::SharedMemory => sim.initialize_shared_memory_backend()?,
        }
        
        // set initial state |0...0⟩
        if local_state_size > 0 {
            if let Some(mut state) = sim.local_state.try_write() {
                state.set_initial_state();
            }
        }
        
        // initialize io threads for network communication
        sim.initialize_io_threads()?;
        
        // initialize worker threads for rendering
        sim.initialize_worker_threads()?;
        
        // initialize load balancer with equal capacities initially
        {
            let mut capacities = sim.load_balancer.node_capacities.write();
            let mut history = sim.load_balancer.performance_history.write();
            let mut distribution = sim.load_balancer.last_distribution.write();
            
            for i in 0..num_nodes {
                capacities.insert(i, 1.0 / num_nodes as f64);
                history.insert(i, VecDeque::with_capacity(10));
                distribution.insert(i, 0);
            }
        }
        
        // start the heartbeat thread
        sim.start_heartbeat_thread()?;
        
        // if profiling is enabled, start the profiler
        #[cfg(feature = "profile")]
        {
            let _guard = pprof::ProfilerGuard::new(100)
                .map_err(|e| format!("failed to start profiler: {}", e))?;
                
            // the guard will be dropped when the program exits, writing the profile data
        }
        
        Ok(sim)
    }
    
    // initialize io threads for network communication
    fn initialize_io_threads(&mut self) -> Result<(), String> {
        let io_threads_count = if self.config.io_threads == 0 {
            num_cpus::get_physical().max(2)
        } else {
            self.config.io_threads
        };
        
        log::info!("initializing {} io threads for network communication", io_threads_count);
        
        for i in 0..io_threads_count {
            let io_rx = Arc::clone(self.io_command_rx.as_ref().unwrap());
            let connections = self.connections.as_ref().map(|c| Arc::new(c.clone()));
            let node_id = self.config.node_id;
            
            let handle = thread::Builder::new()
                .name(format!("io-thread-{}", i))
                .spawn(move || {
                    Self::io_thread_loop(i, io_rx, connections, node_id);
                })
                .map_err(|e| format!("failed to spawn io thread: {}", e))?;
                
            self.io_threads.push(handle);
        }
        
        // start the polling thread for incoming messages
        let connections = self.connections.as_ref().map(|c| Arc::new(c.clone()));
        let io_tx = self.io_command_tx.as_ref().unwrap().clone();
        let node_id = self.config.node_id;
        
        let handle = thread::Builder::new()
            .name("io-poll-thread".to_string())
            .spawn(move || {
                Self::io_poll_thread(connections, io_tx, node_id);
            })
            .map_err(|e| format!("failed to spawn io polling thread: {}", e))?;
            
        self.io_threads.push(handle);
        
        Ok(())
    }
    
    // io thread function for processing network commands
    fn io_thread_loop(
        thread_id: usize,
        io_rx: Arc<PLMutex<cb::Receiver<IoCommand>>>,
        connections: Option<Arc<HashMap<usize, PLMutex<OptimizedConnection>>>>,
        node_id: usize,
    ) {
        log::info!("io thread {} started", thread_id);
        
        let connections = match connections {
            Some(c) => c,
            None => {
                log::error!("io thread {} has no connections, exiting", thread_id);
                return;
            }
        };
        
        loop {
            // get a command from the queue
            let cmd = {
                let rx_guard = io_rx.lock();
                match rx_guard.recv() {
                    Ok(cmd) => cmd,
                    Err(_) => {
                        log::info!("io thread {} channel closed, exiting", thread_id);
                        break;
                    }
                }
            };
            
            match cmd {
                IoCommand::SendMessage { node_id: target, message } => {
                    if let Some(conn) = connections.get(&target) {
                        let start = Instant::now();
                        
                        let mut conn_guard = conn.lock();
                        
                        if let Err(e) = conn_guard.send_message(message) {
                            log::error!("io thread {} failed to send message to node {}: {}", 
                                       thread_id, target, e);
                        }
                        
                        let elapsed = start.elapsed();
                        log::trace!("io thread {} sent message to node {} in {:?}", 
                                  thread_id, target, elapsed);
                    } else {
                        log::warn!("io thread {} has no connection to node {}", thread_id, target);
                    }
                },
                
                IoCommand::FlushConnections => {
                    for (target, conn) in connections.iter() {
                        if *target != node_id {
                            let mut conn_guard = conn.lock();
                            for i in 0..conn_guard.streams.len() {
                                if let Err(e) = conn_guard.flush_stream(i) {
                                    log::error!("io thread {} failed to flush stream {} to node {}: {}", 
                                               thread_id, i, target, e);
                                }
                            }
                        }
                    }
                },
                
                IoCommand::Shutdown => {
                    log::info!("io thread {} received shutdown command", thread_id);
                    break;
                }
            }
        }
    }
    
    // io polling thread for receiving incoming messages
    fn io_poll_thread(
        connections: Option<Arc<HashMap<usize, PLMutex<OptimizedConnection>>>>,
        io_tx: cb::Sender<IoCommand>,
        node_id: usize,
    ) {
        log::info!("io polling thread started");
        
        let connections = match connections {
            Some(c) => c,
            None => {
                log::error!("io polling thread has no connections, exiting");
                return;
            }
        };
        
        // set up polling
        let mut poll = match Poll::new() {
            Ok(p) => p,
            Err(e) => {
                log::error!("failed to create polling instance: {}", e);
                return;
            }
        };
        
        let mut events = Events::with_capacity(1024);
        let mut connection_tokens = HashMap::new();
        
        // register all streams for all connections
        for (target, conn) in connections.iter() {
            if *target != node_id {
                let conn_guard = conn.lock();
                for (stream_idx, stream) in conn_guard.streams.iter().enumerate() {
                    let token = Token(*target * conn_guard.streams.len() + stream_idx);
                    if let Err(e) = poll.registry().register(
                        stream,
                        token,
                        Interest::READABLE,
                    ) {
                        log::error!("failed to register stream {} to node {}: {}", stream_idx, target, e);
                        continue;
                    }
                    
                    connection_tokens.insert(token, (*target, stream_idx));
                }
            }
        }
        
        // polling loop
        loop {
            match poll.poll(&mut events, Some(Duration::from_millis(100))) {
                Ok(_) => {
                    for event in events.iter() {
                        let token = event.token();
                        
                        if let Some(&(target, stream_idx)) = connection_tokens.get(&token) {
                            if let Some(conn) = connections.get(&target) {
                                let mut conn_guard = conn.lock();
                                
                                match conn_guard.try_receive_messages() {
                                    Ok(messages) => {
                                        for message in messages {
                                            log::trace!("received message from node {}: {:?}", 
                                                      target, message.msg_type);
                                            // handle message
                                        }
                                    },
                                    Err(e) if e.kind() == ErrorKind::WouldBlock => {},
                                    Err(e) => {
                                        log::error!("error receiving from node {} stream {}: {}", target, stream_idx, e);
                                    }
                                }
                            }
                        }
                    }
                },
                Err(e) => {
                    log::error!("polling error: {}", e);
                    break;
                }
            }
            
            if let Err(e) = io_tx.send(IoCommand::FlushConnections) {
                log::error!("failed to send flush command: {}", e);
                break;
            }
        }
    }
    
    // start the heartbeat thread
    fn start_heartbeat_thread(&self) -> Result<(), String> {
        let heartbeat_manager = Arc::clone(&self.heartbeat_manager);
        let node_status = Arc::clone(&self.node_status);
        let io_tx = self.io_command_tx.as_ref().unwrap().clone();
        let node_id = self.config.node_id;
        let num_nodes = self.config.nodes.len();
        let interval = self.heartbeat_manager.interval;
        let compression_level = self.config.compression_level;
        
        thread::Builder::new()
            .name("heartbeat-thread".to_string())
            .spawn(move || {
                loop {
                    for target in 0..num_nodes {
                        if target != node_id {
                            let payload = BytesMut::new();
                            let message = NetworkMessage::new(
                                MessageType::Heartbeat,
                                node_id as u16,
                                target as u16,
                                payload,
                                compression_level
                            );
                            
                            if let Err(e) = io_tx.send(IoCommand::SendMessage { 
                                node_id: target, 
                                message 
                            }) {
                                log::error!("failed to send heartbeat to node {}: {}", target, e);
                                break;
                            }
                        }
                    }
                    
                    {
                        let last_beats = heartbeat_manager.last_heartbeat.read();
                        let mut missed = heartbeat_manager.missed_beats.write();
                        let mut status = node_status.status.write();
                        
                        for target in 0..num_nodes {
                            if target != node_id {
                                if let Some(last_time) = last_beats.get(&target) {
                                    if last_time.elapsed() > interval * 3 {
                                        let count = missed.entry(target).or_insert(0);
                                        *count += 1;
                                        
                                        if *count > 5 {
                                            status.insert(target, NodeStatus::Disconnected);
                                            log::warn!("node {} marked as disconnected after {} missed heartbeats", 
                                                     target, *count);
                                        }
                                    } else {
                                        missed.insert(target, 0);
                                    }
                                }
                            }
                        }
                    }
                    
                    thread::sleep(interval);
                }
            })
            .map_err(|e| format!("failed to spawn heartbeat thread: {}", e))?;
            
        Ok(())
    }
    
    // initialize worker threads for rendering tasks with work stealing
    fn initialize_worker_threads(&mut self) -> Result<(), String> {
        let num_workers = num_cpus::get().max(2) - 1; // leave one core for main thread
        
        // create a work stealing deque for each worker
        let steal_queues = Arc::new(PLRwLock::new(
            (0..num_workers).map(|_| cb::bounded(128)).collect::<Vec<_>>()
        ));
        
        for worker_id in 0..num_workers {
            let work_rx = Arc::clone(self.work_rx.as_ref().unwrap());
            let result_tx = self.result_tx.as_ref().unwrap().clone();
            let local_state = Arc::clone(&self.local_state);
            let steal_queues_clone = Arc::clone(&steal_queues);
            let vulkan_device = self.vulkan_device.clone();
            let vulkan_queue = self.vulkan_queue.clone();
            let vulkan_memory_allocator = self.vulkan_memory_allocator.clone();
            let vulkan_descriptor_set_allocator = self.vulkan_descriptor_set_allocator.clone();
            let vulkan_command_buffer_allocator = self.vulkan_command_buffer_allocator.clone();
            let use_hybrid = self.config.use_hybrid_computing;
            
            let handle = thread::Builder::new()
                .name(format!("render-worker-{}", worker_id))
                .spawn(move || {
                    #[cfg(target_os = "linux")]
                    {
                        use core_affinity::CoreId;
                        if let Some(core_ids) = core_affinity::get_core_ids() {
                            let core_id = core_ids[worker_id % core_ids.len()];
                            core_affinity::set_for_current(core_id);
                            log::info!("worker {} assigned to cpu core {}", worker_id, core_id.id);
                        }
                    }
                    
                    Self::render_worker_loop_with_stealing(
                        worker_id,
                        work_rx,
                        result_tx,
                        local_state,
                        steal_queues_clone,
                        vulkan_device,
                        vulkan_queue,
                        vulkan_memory_allocator,
                        vulkan_descriptor_set_allocator,
                        vulkan_command_buffer_allocator,
                        use_hybrid,
                    );
                })
                .map_err(|e| format!("failed to spawn worker thread: {}", e))?;
                
            self.worker_threads.push(handle);
        }
        
        log::info!("initialized {} worker threads for rendering with work stealing", num_workers);
        Ok(())
    }
    
    // worker thread function for rendering with work stealing and hybrid computing
    fn render_worker_loop_with_stealing(
        worker_id: usize,
        work_rx: Arc<PLMutex<Receiver<RenderWorkItem>>>,
        result_tx: Sender<RenderChunk>,
        local_state: Arc<PLRwLock<StateVector>>,
        steal_queues: Arc<PLRwLock<Vec<(cb::Sender<RenderWorkItem>, cb::Receiver<RenderWorkItem>)>>>,
        vulkan_device: Option<Arc<Device>>,
        vulkan_queue: Option<Arc<Queue>>,
        vulkan_memory_allocator: Option<Arc<StandardMemoryAllocator>>,
        vulkan_descriptor_set_allocator: Option<Arc<StandardDescriptorSetAllocator>>,
        vulkan_command_buffer_allocator: Option<Arc<StandardCommandBufferAllocator>>,
        use_hybrid: bool,
    ) {
        log::info!("render worker {} started with work stealing and hybrid computing: {}", worker_id, use_hybrid);
        
        let (our_send, our_recv) = {
            let queues = steal_queues.read();
            queues[worker_id].clone()
        };
        
        loop {
            // first, try to get work from our own queue
            let work_item = our_recv.try_recv().ok()
                // if that fails, try the main work queue
                .or_else(|| work_rx.lock().try_recv().ok())
                // if that also fails, try to steal from another worker
                .or_else(|| {
                    let queues = steal_queues.read();
                    let other_workers = queues.len();
                    // try stealing from a random worker to reduce contention
                    let start_victim = (worker_id + 1) % other_workers;
                    (0..other_workers)
                        .map(|i| (start_victim + i) % other_workers)
                        .filter(|&i| i != worker_id)
                        .find_map(|victim_id| queues[victim_id].1.try_recv().ok())
                });

            if let Some(work_item) = work_item {
                if work_item.start_idx == usize::MAX {
                    log::info!("render worker {} received termination signal", worker_id);
                    break;
                }
                
                let mut result_buffer = Vec::new(); // buffer is local to the task
                Self::process_work_item(worker_id, work_item, &mut result_buffer, &local_state, &result_tx, vulkan_device.as_ref(), vulkan_queue.as_ref(), vulkan_memory_allocator.as_ref(), vulkan_descriptor_set_allocator.as_ref(), vulkan_command_buffer_allocator.as_ref(), use_hybrid);
            } else {
                // if no work is found, yield to avoid busy-waiting
                thread::yield_now();
            }
        }
    }
    
    // process a single work item with hybrid computing support
    fn process_work_item(
        worker_id: usize,
        work_item: RenderWorkItem,
        result_buffer: &mut Vec<u8>,
        local_state: &Arc<PLRwLock<StateVector>>,
        result_tx: &Sender<RenderChunk>,
        vulkan_device: Option<&Arc<Device>>,
        vulkan_queue: Option<&Arc<Queue>>,
        vulkan_memory_allocator: Option<&Arc<StandardMemoryAllocator>>,
        vulkan_descriptor_set_allocator: Option<&Arc<StandardDescriptorSetAllocator>>,
        vulkan_command_buffer_allocator: Option<&Arc<StandardCommandBufferAllocator>>,
        use_hybrid: bool,
    ) {
        let start_time = Instant::now();
        log::debug!("worker {} processing range {}..{}", worker_id, work_item.start_idx, work_item.end_idx);
        
        result_buffer.clear();
        {
            let state = local_state.read();
            if use_hybrid && work_item.parameters.use_gpu {
                if let (Some(device), Some(queue), Some(memory_allocator), Some(descriptor_set_allocator), Some(command_buffer_allocator)) = (vulkan_device, vulkan_queue, vulkan_memory_allocator, vulkan_descriptor_set_allocator, vulkan_command_buffer_allocator) {
                    Self::render_state_segment_vulkan(&state, work_item.start_idx, work_item.end_idx, &work_item.parameters, result_buffer, device, queue, memory_allocator, descriptor_set_allocator, command_buffer_allocator);
                } else {
                    Self::render_state_segment_optimized(&state, work_item.start_idx, work_item.end_idx, &work_item.parameters, result_buffer);
                }
            } else {
                Self::render_state_segment_optimized(&state, work_item.start_idx, work_item.end_idx, &work_item.parameters, result_buffer);
            }
        }
        
        let chunk = RenderChunk {
            start_idx: work_item.start_idx,
            end_idx: work_item.end_idx,
            data: result_buffer.clone(),
        };
        
        if let Err(e) = result_tx.send(chunk) {
            log::error!("worker {} failed to send result: {}", worker_id, e);
        }
        
        log::debug!("worker {} completed range {}..{} in {:?}", 
                  worker_id, work_item.start_idx, work_item.end_idx, start_time.elapsed());
    }
    
    fn render_state_segment_vulkan(
        state: &StateVector,
        start_idx: usize,
        end_idx: usize,
        params: &RenderParams,
        result: &mut Vec<u8>,
        device: &Arc<Device>,
        queue: &Arc<Queue>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    ) {
        let len = (end_idx - start_idx) as u64;
        if len == 0 {
            return;
        }

        // Create buffers
        let re_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            state.re[start_idx..end_idx].iter().cloned(),
        ).unwrap();

        let im_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            state.im[start_idx..end_idx].iter().cloned(),
        ).unwrap();

        let out_buffer = Buffer::new_slice::<u32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            len * 4,
        ).unwrap();

        // Load shader
        let shader = cs::load(device.clone()).unwrap();

        let cs = shader.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        ).unwrap();
        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        ).unwrap();

        // Descriptor set
        let descriptor_set = PersistentDescriptorSet::new(
            descriptor_set_allocator,
            compute_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, re_buffer.clone()),
                WriteDescriptorSet::buffer(1, im_buffer.clone()),
                WriteDescriptorSet::buffer(2, out_buffer.clone()),
            ],
            [],
        ).unwrap();

        // Command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .dispatch([(len as u32 + 63) / 64, 1, 1])
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        // Read back
        let out_content = out_buffer.read().unwrap();
        result.resize(len as usize * 4, 0);
        let mut i = 0;
        for &val in out_content.iter() {
            result[i] = (val & 0xFF) as u8;
            i += 1;
        }
    }
    
    // optimized rendering of a state vector segment.
    // this version uses rayon for parallelism and delegates to simd-optimized functions.
    fn render_state_segment_optimized(
        state: &StateVector,
        start_idx: usize,
        end_idx: usize,
        params: &RenderParams,
        result: &mut Vec<u8>
    ) {
        let segment_len = (end_idx - start_idx).min(state.len() - start_idx);
        let num_pixels = segment_len;
        let result_size = num_pixels * 4; // 4 bytes per pixel (rgba)
        result.resize(result_size, 0);

        let re_segment = &state.re[start_idx..start_idx + segment_len];
        let im_segment = &state.im[start_idx..start_idx + segment_len];
        let pixel_chunks = result.par_chunks_mut(4);

        match params.visualization_type {
            VisualizationType::Probability => {
                // use rayon to parallelize the rendering calculation across available cores.
                // zip the real and imaginary parts with the output pixel buffer for efficient processing.
                re_segment.par_iter().zip(im_segment.par_iter()).zip(pixel_chunks)
                    .for_each(|((&re, &im), pixel)| {
                        let prob = re * re + im * im;
                        let intensity = (prob.min(1.0) * 255.0) as u8;
                        pixel[0] = intensity; // r
                        pixel[1] = intensity; // g
                        pixel[2] = intensity; // b
                        pixel[3] = 255;       // a
                    });
            },
            _ => {
                re_segment.par_iter().zip(im_segment.par_iter()).zip(pixel_chunks)
                    .for_each(|((&re, &im), pixel)| {
                        let r = (re.abs().min(1.0) * 255.0) as u8;
                        let g = (im.abs().min(1.0) * 255.0) as u8;
                        let b = ((re * re + im * im).min(1.0) * 255.0) as u8;
                        pixel[0] = r;
                        pixel[1] = g;
                        pixel[2] = b;
                        pixel[3] = 255;
                    });
            }
        }
    }
    
    // avx2-optimized probability rendering
    // note: this is kept for reference but the rayon-based approach is often more
    // scalable and easier to maintain. direct simd can be faster for very specific,
    // large, and contiguous memory operations if implemented carefully.
    #[cfg(target_arch = "x86_64")]
    fn render_probability_avx2(
        re: &[f64],
        im: &[f64],
        result: &mut [u8]
    ) {
        use std::arch::x86_64::*;

        // ensure slices have the same length
        assert_eq!(re.len(), im.len());
        assert_eq!(re.len() * 4, result.len());

        let chunks = re.len() / 4;
        let remainder_start = chunks * 4;

        unsafe {
            let re_ptr = re.as_ptr();
            let im_ptr = im.as_ptr();
            let res_ptr = result.as_mut_ptr();
            
            let scale = _mm256_set1_pd(255.0);

            for i in 0..chunks {
                let offset = i * 4;
                
                // load 4 f64 values for real and imaginary parts
                let real_vec = _mm256_loadu_pd(re_ptr.add(offset));
                let imag_vec = _mm256_loadu_pd(im_ptr.add(offset));

                // calculate probability: re*re + im*im
                let real_sq = _mm256_mul_pd(real_vec, real_vec);
                let imag_sq = _mm256_mul_pd(imag_vec, imag_vec);
                let prob_vec = _mm256_add_pd(real_sq, imag_sq);

                // scale to 0-255
                let scaled_vec = _mm256_mul_pd(prob_vec, scale);

                // convert f64 vector to i32 vector
                let int_vec = _mm256_cvttpd_epi32(scaled_vec);
                
                // pack i32 down to u8. this is a bit tricky.
                // pack 2x 128-bit lanes of 32-bit integers into 16-bit integers
                let packed_16 = _mm256_packus_epi32(int_vec, int_vec);
                // pack 128-bit lane of 16-bit integers into 8-bit integers
                let packed_8 = _mm_packus_epi16(
                    _mm256_extracti128_si256(packed_16, 0),
                    _mm256_extracti128_si256(packed_16, 0)
                );

                // we now have 4 intensity values in the lower 32 bits of packed_8
                let intensities = _mm_cvtsi128_si32(packed_8) as u32;

                // write the rgba pixels
                for j in 0..4 {
                    let intensity_byte = (intensities >> (j * 8)) as u8;
                    let pixel_offset = (offset + j) * 4;
                    *res_ptr.add(pixel_offset) = intensity_byte;
                    *res_ptr.add(pixel_offset + 1) = intensity_byte;
                    *res_ptr.add(pixel_offset + 2) = intensity_byte;
                    *res_ptr.add(pixel_offset + 3) = 255;
                }
            }
        }

        // handle the remainder using scalar operations
        for i in remainder_start..re.len() {
            let prob = re[i] * re[i] + im[i] * im[i];
            let intensity = (prob.min(1.0) * 255.0) as u8;
            let pixel_offset = i * 4;
            result[pixel_offset] = intensity;
            result[pixel_offset + 1] = intensity;
            result[pixel_offset + 2] = intensity;
            result[pixel_offset + 3] = 255;
        }
    }
    
    // initialize the mapping of qubits to nodes
    fn initialize_qubit_mapping(&mut self) {
        let num_nodes = self.config.nodes.len();
        
        let qubits_per_node = self.config.total_qubits / num_nodes;
        let remainder = self.config.total_qubits % num_nodes;
        
        let mut qubit_idx = 0;
        for node_id in 0..num_nodes {
            let node_qubits = qubits_per_node + if node_id < remainder { 1 } else { 0 };
            for _ in 0..node_qubits {
                if qubit_idx < self.config.total_qubits {
                    self.qubit_mapping.insert(qubit_idx, node_id);
                    qubit_idx += 1;
                }
            }
        }
        
        log::info!("initialized qubit mapping: {} qubits across {} nodes", 
                 self.config.total_qubits, num_nodes);
    }
    
    // initialize tcp communication backend
    fn initialize_tcp_backend(&mut self) -> Result<(), String> {
        let mut connections = HashMap::new();
        
        if self.config.node_id == 0 {
            let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
                .map_err(|e| format!("failed to create socket: {}", e))?;
            
            socket.set_nonblocking(true)
                .map_err(|e| format!("failed to set non-blocking: {}", e))?;
            
            socket.set_reuse_address(true)
                .map_err(|e| format!("failed to set so_reuseaddr: {}", e))?;
            
            #[cfg(target_os = "linux")]
            {
                use std::os::unix::io::AsRawFd;
                let fd = socket.as_raw_fd();
                
                let value: libc::c_int = 5; // queue length
                unsafe {
                    libc::setsockopt(
                        fd,
                        libc::IPPROTO_TCP,
                        libc::TCP_FASTOPEN,
                        &value as *const _ as *const libc::c_void,
                        std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                    );
                }
                
                let value: libc::c_int = 1;
                unsafe {
                    libc::setsockopt(
                        fd,
                        libc::IPPROTO_TCP,
                        libc::TCP_QUICKACK,
                        &value as *const _ as *const libc::c_void,
                        std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                    );
                }
            }
            
            socket.set_send_buffer_size(self.config.socket_send_buffer)
                .map_err(|e| format!("failed to set send buffer size: {}", e))?;
                
            socket.set_recv_buffer_size(self.config.socket_recv_buffer)
                .map_err(|e| format!("failed to set receive buffer size: {}", e))?;
            
            socket.set_tos(self.config.ip_tos as u32)
                .map_err(|e| format!("failed to set ip tos: {}", e))?;
            
            let addr = format!("0.0.0.0:{}", self.config.port).parse()
                .map_err(|e| format!("failed to parse address: {}", e))?;
            
            socket.bind(&addr.into())
                .map_err(|e| format!("failed to bind: {}", e))?;
            
            socket.listen(MAX_SOCKET_BACKLOG as i32)
                .map_err(|e| format!("failed to listen: {}", e))?;
            
            let listener: TcpListener = socket.into();
            
            log::info!("master node listening on port {} with optimized socket settings", 
                     self.config.port);
            
            let start_time = Instant::now();
            let timeout = Duration::from_secs(30);
            let mut connected_nodes = 0;
            
            while connected_nodes < self.config.nodes.len() - 1 {
                match listener.accept() {
                    Ok((stream, addr)) => {
                        let mut node_id_buf = [0u8; 8];
                        stream.try_clone()
                            .map_err(|e| format!("failed to clone stream: {}", e))?
                            .read_exact(&mut node_id_buf)
                            .map_err(|e| format!("failed to read node id: {}", e))?;
                            
                        let node_id = usize::from_le_bytes(node_id_buf);
                        log::info!("connected to node {} at {} with optimized connection", 
                                 node_id, addr);
                        
                        let mut streams = vec![stream];
                        for _ in 1..self.config.streams_per_node {
                            // accept additional streams for this node
                            if let Ok((add_stream, _)) = listener.accept() {
                                streams.push(add_stream);
                            } else {
                                return Err("failed to accept additional stream".to_string());
                            }
                        }
                        
                        let connection = OptimizedConnection::new(
                            streams,
                            node_id,
                            self.config.network_buffer_size,
                            self.config.use_zero_copy
                        ).map_err(|e| format!("failed to create optimized connection: {}", e))?;
                        
                        connections.insert(node_id, PLMutex::new(connection));
                        connected_nodes += 1;
                    },
                    Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                        if start_time.elapsed() > timeout {
                            return Err(format!("timed out waiting for connections. only connected to {}/{} nodes", 
                                               connected_nodes, self.config.nodes.len() - 1));
                        }
                        thread::sleep(Duration::from_millis(10));
                    },
                    Err(e) => {
                        return Err(format!("error accepting connection: {}", e));
                    }
                }
            }
        } else {
            let master_addr = &self.config.nodes[0];
            log::info!("worker node {} connecting to master at {}", self.config.node_id, master_addr);
            
            let mut retry_count = 0;
            let max_retries = 5;
            let mut last_error = String::new();
            
            while retry_count < max_retries {
                let mut streams = Vec::new();
                let mut connected = false;
                
                for _ in 0..self.config.streams_per_node {
                    let stream = TcpStream::connect(format!("{}:{}", master_addr, self.config.port));
                    match stream {
                        Ok(mut s) => {
                            s.set_nodelay(true)
                                .map_err(|e| format!("failed to set tcp_nodelay: {}", e))?;
                            
                            s.set_read_timeout(Some(Duration::from_secs(5)))
                                .map_err(|e| format!("failed to set read timeout: {}", e))?;
                            
                            // send node id
                            s.write_all(&self.config.node_id.to_le_bytes())
                                .map_err(|e| format!("failed to send node id: {}", e))?;
                            
                            streams.push(s);
                        },
                        Err(e) => {
                            last_error = format!("failed to connect to master: {}", e);
                            log::warn!("{} - retrying in 2 seconds ({}/{})", last_error, retry_count + 1, max_retries);
                            retry_count += 1;
                            thread::sleep(Duration::from_secs(2));
                            break;
                        }
                    }
                }
                
                if streams.len() == self.config.streams_per_node {
                    let connection = OptimizedConnection::new(
                        streams,
                        0,
                        self.config.network_buffer_size,
                        self.config.use_zero_copy
                    ).map_err(|e| format!("failed to create optimized connection: {}", e))?;
                    
                    connections.insert(0, PLMutex::new(connection));
                    connected = true;
                }
                
                if connected {
                    break;
                }
            }
            
            if retry_count >= max_retries {
                return Err(last_error);
            }
        }
        
        self.connections = Some(connections);
        Ok(())
    }
    
    // initialize mpi communication backend
    fn initialize_mpi_backend(&mut self) -> Result<(), String> {
        #[cfg(feature = "mpi")]
        {
            log::info!("initializing mpi backend");
            if let Some(world) = &self.mpi_world {
                let rank = world.rank() as usize;
                if rank != self.config.node_id {
                    return Err("MPI rank mismatch".to_string());
                }
                // Additional MPI setup, e.g., creating communicators
                Ok(())
            } else {
                Err("MPI world not initialized".to_string())
            }
        }
        
        #[cfg(not(feature = "mpi"))]
        {
            Err("mpi backend not compiled in. rebuild with --feature=mpi".to_string())
        }
    }
    
    // initialize rdma backend
    fn initialize_rdma_backend(&mut self) -> Result<(), String> {
        if !self.config.use_rdma {
            return Err("rdma not enabled in config".to_string());
        }
        
        // implementation for rdma using verbs or similar library
        // assume rdma-core or rust-rdma crate, but since not included, placeholder
        log::info!("initializing rdma backend");
        todo!("rdma backend implementation");
    }
    
    // initialize shared memory backend for local nodes
    fn initialize_shared_memory_backend(&mut self) -> Result<(), String> {
        // implementation for shared memory, using shmem or mmap
        log::info!("initializing shared memory backend");
        todo!("shared memory backend implementation");
    }
    
    // apply a single-qubit gate to the distributed state vector
    pub fn apply_single_qubit_gate(&mut self, qubit: usize, gate: Gate) -> Result<(), String> {
        let node_id = self.qubit_mapping.get(&qubit).ok_or_else(|| 
            format!("qubit {} not found in mapping", qubit))?;
            
        if *node_id == self.config.node_id {
            self.apply_local_gate(qubit, gate)?;
        } else {
            self.request_remote_gate(qubit, *node_id, gate)?;
        }
        
        self.synchronize_nodes()?;
        Ok(())
    }
    
    // placeholder for local gate application
    fn apply_local_gate(&mut self, qubit: usize, gate: Gate) -> Result<(), String> {
        log::info!("applying {:?} gate to local qubit {}", gate, qubit);
        // actual transformation code would go here
        Ok(())
    }
    
    // request a remote node to apply a gate
    fn request_remote_gate(&mut self, qubit: usize, node_id: usize, gate: Gate) -> Result<(), String> {
        log::info!("requesting node {} to apply {:?} gate to qubit {}", node_id, gate, qubit);
        
        // serialize gate request and send via io command
        let mut payload = BytesMut::new();
        payload.put_u32(qubit as u32);
        payload.put_u8(gate as u8); // assume gate can be cast to u8
        
        let message = NetworkMessage::new(
            MessageType::GateRequest,
            self.config.node_id as u16,
            node_id as u16,
            payload,
            self.config.compression_level
        );
        
        if let Some(tx) = &self.io_command_tx {
            tx.send(IoCommand::SendMessage { node_id, message })
                .map_err(|e| format!("failed to send gate request: {}", e))?;
        }
        
        Ok(())
    }
    
    // synchronize all nodes based on sync_mode
    fn synchronize_nodes(&mut self) -> Result<(), String> {
        match self.config.sync_mode {
            SyncMode::Barrier => self.synchronize_barrier(),
            SyncMode::LockFree => self.synchronize_lock_free(),
            SyncMode::Optimistic => self.synchronize_optimistic(),
            SyncMode::Eventually => self.synchronize_eventual(),
        }
    }
    
    // traditional barrier synchronization
    fn synchronize_barrier(&self) -> Result<(), String> {
        let barrier = self.barrier.read();
        let mut state = barrier.get();
        
        let old_gen = state.generation.load(Ordering::SeqCst);
        let new_count = state.count.fetch_add(1, Ordering::SeqCst) + 1;
        
        if new_count == state.total {
            state.count.store(0, Ordering::SeqCst);
            state.generation.fetch_add(1, Ordering::SeqCst);
            log::info!("all nodes synchronized at barrier generation {}", old_gen + 1);
        } else {
            while state.generation.load(Ordering::SeqCst) == old_gen {
                std::thread::yield_now();
            }
        }
        
        Ok(())
    }
    
    // lock-free synchronization
    fn synchronize_lock_free(&self) -> Result<(), String> {
        // implement lock-free sync, perhaps using atomic counters with spinloop
        // similar to barrier but optimized
        self.synchronize_barrier() // placeholder for now
    }
    
    // optimistic synchronization
    fn synchronize_optimistic(&self) -> Result<(), String> {
        // implement optimistic concurrency, with version checks
        // for quantum state, might involve state hashes or versions
        Ok(())
    }
    
    // eventual consistency synchronization
    fn synchronize_eventual(&self) -> Result<(), String> {
        // send async updates and assume eventual consistency
        // for non-critical ops
        Ok(())
    }

    // start the rendering process by distributing work to the workers.
    pub fn start_rendering(&self, params: RenderParams) -> Result<(), String> {
        let total_size = self.local_state.read().len();
        if total_size == 0 {
            return Ok(()); // nothing to render
        }
        
        let chunk_size = self.config.render_chunk_size;
        let work_tx = self.work_tx.as_ref().unwrap();

        for start_idx in (0..total_size).step_by(chunk_size) {
            let end_idx = (start_idx + chunk_size).min(total_size);
            let work_item = RenderWorkItem {
                start_idx,
                end_idx,
                parameters: params.clone(),
                priority: 0,
            };

            if let Err(e) = work_tx.send(work_item) {
                return Err(format!("failed to send render work: {}", e));
            }
        }

        Ok(())
    }

    // collect and assemble the final rendered image from the workers.
    pub fn collect_render_results(&self, timeout: Duration) -> Result<Vec<u8>, String> {
        let total_size = self.local_state.read().len();
        if total_size == 0 {
            return Ok(Vec::new());
        }

        let chunk_size = self.config.render_chunk_size;
        let expected_chunks = (total_size + chunk_size - 1) / chunk_size;
        
        let result_rx = self.result_rx.as_ref().unwrap();
        let rx_lock = result_rx.lock();

        let mut collected_chunks = HashMap::with_capacity(expected_chunks);
        let start_time = Instant::now();

        while collected_chunks.len() < expected_chunks {
            let remaining_timeout = timeout.saturating_sub(start_time.elapsed());
            if remaining_timeout.is_zero() {
                return Err(format!(
                    "timed out waiting for render results. collected {}/{} chunks",
                    collected_chunks.len(),
                    expected_chunks
                ));
            }

            match rx_lock.recv_timeout(remaining_timeout) {
                Ok(chunk) => {
                    collected_chunks.insert(chunk.start_idx, chunk);
                }
                Err(_) => {
                    return Err(format!(
                        "timed out while receiving a render chunk. collected {}/{} chunks",
                        collected_chunks.len(),
                        expected_chunks
                    ));
                }
            }
        }

        // assemble the final image from the chunks.
        let mut final_image = Vec::with_capacity(total_size * 4);
        for i in 0..expected_chunks {
            let start_idx = i * chunk_size;
            if let Some(chunk) = collected_chunks.get(&start_idx) {
                final_image.extend_from_slice(&chunk.data);
            } else {
                return Err(format!("missing chunk starting at index {}", start_idx));
            }
        }

        Ok(final_image)
    }
    
    // shutdown the distributed simulation
    pub fn shutdown(&mut self) -> Result<(), String> {
        log::info!("shutting down distributed simulation");
        
        if let Some(work_tx) = &self.work_tx {
            for _ in 0..self.worker_threads.len() {
                let termination_signal = RenderWorkItem {
                    start_idx: usize::MAX,
                    end_idx: usize::MAX,
                    parameters: RenderParams {
                        resolution: (0, 0),
                        depth: 0,
                        use_gpu: false,
                        visualization_type: VisualizationType::Probability,
                    },
                    priority: 0,
                };
                
                work_tx.send(termination_signal).ok();
            }
        }
        
        for handle in self.worker_threads.drain(..) {
            handle.join().ok();
        }
        
        if let Some(io_tx) = &self.io_command_tx {
            io_tx.send(IoCommand::Shutdown).ok();
        }
        
        for handle in self.io_threads.drain(..) {
            handle.join().ok();
        }
        
        if let Some(connections) = self.connections.take() {
            for (node_id, conn) in connections {
                log::debug!("closing connection to node {}", node_id);
            }
        }
        
        Ok(())
    }
}

// quantum gate types
#[derive(Debug, Clone, Copy)]
pub enum Gate {
    X = 0,
    Y = 1,
    Z = 2,
    H = 3,
    CNOT = 4,
    CZ = 5,
    SWAP = 6,
}

pub fn run_distributed_rendering() -> Result<(), String> {
    let config = DistributedConfig {
        total_qubits: 24,
        backend: CommunicationBackend::TCP,
        nodes: vec![
            "localhost".to_string(),
            // add other node addresses here if you want
            // "node1.example.com".to_string(),
            // "node2.example.com".to_string(),
            // "node3.example.com".to_string(),
        ],
        node_id: 0, // set differently for each node
        port: 9000,
        render_chunk_size: 4096,
        max_memory_per_node: 16 * 1024 * 1024 * 1024,
        compression_level: 3,
        streams_per_node: 8,
        ip_tos: 0x10, // minimize delay
        use_hybrid_computing: true,
        ..Default::default()
    };
    
    let mut sim = DistributedSimulation::new(config)?;
    
    sim.apply_single_qubit_gate(0, Gate::H)?;
    sim.apply_single_qubit_gate(1, Gate::H)?;
    // H gates for now

    let render_params = RenderParams {
        resolution: (1024, 1024),
        depth: 8,
        use_gpu: true, // Use GPU if available
        visualization_type: VisualizationType::Probability,
    };
    
    // start the rendering process
    sim.start_rendering(render_params)?;
    
    // collect the results with a timeout
    let timeout = Duration::from_secs(30);
    match sim.collect_render_results(timeout) {
        Ok(result) => {
            println!("rendering complete! generated {} bytes of visualization data", result.len());
            // here you would typically save the 'result' Vec<u8> to a file as an image.
        }
        Err(e) => {
            eprintln!("rendering failed: {}", e);
        }
    }
    
    sim.shutdown()?;
    
    Ok(())
}