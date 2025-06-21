# QOA release v0.3.0 Development TODO

## THIS TODO LIST WILL DECREASE IN SIZE AS I FINISH IT

## High Priority:

# QUANTUM NOISE:

- Fix noise generation for outputs

- improve noise-gen and make it high quality

- confirm qubit indexing and ordering conventions used by the simulator.

- analyze the effect of the first Hadamard gate on qubit 0 starting from ∣0000⟩.

- analyze the effect of the second Hadamard gate on qubit 1 applied to the resulting superposition.

- fix the effect of measuring qubit 3 when it is already in state 0 across all superposed components.


- Implement THERMAL_AVG:  
  Sample Maxwell-Boltzmann energy distribution at given temperature (keV)

- Implement WKB_FACTOR:  
  Calculate Gamow tunneling factor exp(-2πη), where η depends on physical constants and energy

- Define key physical constants:  
  Proton mass, fine structure constant, Coulomb barrier, etc.

- Implement SQRT, EXP, LOG with domain and error checks

- Implement RAND:  
  Generate random floating-point number between 0 and 1

- Ensure proper NaN and error propagation for all math operations

## Quantum Gate Enhancements

- Implement RX, RY, RZ parametric rotation gates using angle values from registers

- Implement PHASE gate for arbitrary phase rotation on qubits

- Implement CNOT gate (controlled-NOT)

- Implement QRESET to reset qubit to |0⟩ state

- Update quantum state operations for parametric gates and normalize after application


## Control Flow & Program Structure

- Implement LOOP and ENDLOOP instructions with iteration counters on program counter stack

- Implement JMP (relative) and JMPABS (absolute) jump instructions

- Implement conditional jumps:  
  IFGT, IFLT, IFEQ, IFNE using epsilon-based floating-point comparisons

- Implement LABEL instruction for named jump targets

- Implement CALL and RET instructions with subroutine call stack management


## Medium Priority: I/O, Memory, Debugging & System

### Input/Output & Debugging

- Implement PRINTF with C-style formatted output of register values
- Implement PRINT and PRINTLN for string output and newlines
- Implement INPUT to read floating-point values from stdin into registers
- Implement DUMP_STATE to output quantum amplitudes and phases
- Implement DUMP_REGS to output all register values

### Memory & Stack

- Implement LOAD and STORE for memory-register data transfer
- Implement PUSH and POP stack operations on registers
- Implement ALLOC and FREE for dynamic memory management
- Add linear byte-addressable memory with bounds checking

### Comparison & Bitwise Logic

- Implement CMP to compare registers and set flags
- Implement AND, OR, XOR, NOT bitwise operations
- Implement SHL and SHR bit shift operations

### System & Debug Utilities

- Implement NOP (no operation) and BREAK (breakpoint) instructions
- Implement TIME instruction to get system timestamp in register
- Implement SEED instruction to seed RNG for reproducible results
- Implement VERSION instruction to output current QOA version
- Implement EXIT instruction to terminate program with exit code


## Lower Priority: Advanced Features, Robustness & Developer Experience

### Instruction Encoding & Parsing

- Extend opcode handling to cover 0x10 through 0xFF and variable-length instructions
- Improve parsing validation and error messaging
- Ensure backward compatibility with QOA v0.2.0 programs

### Quantum State Management

- Add serialization/deserialization of quantum state and registers for checkpoints
- Add validation to detect corrupted quantum states
- Optimize memory usage for large qubit simulations (sparse representations)
- Optionally add quantum decoherence simulation

### Testing & Validation

- Write unit tests for all instructions and quantum operations
- Write integration tests for complex programs, including fusion simulation
- Add performance benchmarks for quantum operations
- Automate tests for edge cases, errors, and memory profiling

### Documentation & Examples

- Write detailed instruction documentation with examples
- Create tutorials focused on fusion simulation and quantum programming
- Write troubleshooting and debugging guides
- Write performance optimization guides
- Document all physics constants and math formulas used