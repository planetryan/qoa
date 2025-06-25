# QOA release v0.3.0 Development TODO

## THIS TODO LIST WILL DECREASE IN SIZE AS I FINISH IT

## High Priority:

## Control Flow & Program Structure

- Fully implement Loops and such

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

- Implement BREAK (breakpoint) instruction
- Implement TIME instruction to get system timestamp in register
- Implement SEED instruction to seed RNG for reproducible results
- Implement EXIT instruction to terminate program with exit code


## Lower Priority: Advanced Features, Robustness & Developer Experience

### Instruction Encoding & Parsing

- Extend opcode handling to cover 0x10 through 0xFF and variable-length instructions
- Improve parsing validation and error messaging

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
