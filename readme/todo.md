# QOA Development TODO

**THIS TODO LIST WILL DECREASE IN SIZE AS I FINISH IT**

- add multi-GPU support for quantum state simulation (via CUDA/OpenCL/Rust-CUDA, Vulkan, etc.) > ALMOST DONE
- add multi-threaded CPU+GPU coordination for hybrid workloads > STARTED NOT DONE

- Design and implement distributed simulation across multiple systems > STARTED NOT DONE

- Add support for frame-sliced distributed rendering in QOA visualizer
  - Assign time or frame ranges per system
  - Automate merge and encode process (e.g. ffmpeg concat, lossless)

## Lower Priority: Advanced Features, Robustness & Developer Experience

### Quantum State Management
- tensor network or sparse simulation for memory-limited systems
- support DPU acceleration for node rendering in quantum graph simulations

### Testing & Validation
- Write unit tests for all instructions and quantum operations
- Write integration tests for complex programs, including fusion simulation
- Add performance benchmarks for quantum operations
- Automate tests for edge cases, errors, and memory profiling
- Add regression tests for multi-GPU and multi-node execution

### Documentation & Examples
- Develop tutorials for fusion simulation and quantum programming
- Provide guides for troubleshooting, debugging, and performance optimization
- Document multi-node setup and GPU requirements
- Add examples of AVX/NEON/RVV vectorized instructions in `.qoa` programs

### rust stable, for now its only avaliable on nightly
