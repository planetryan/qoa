# QOA v0.2.5 Release Notes

**Release Date:** 26/06/2025  
**Status:** Pre Release

## Summary
QOA v0.2.5 significantly expands classical control capabilities, introducing a robust set of jump and subroutine instructions, comprehensive I/O features, and enhanced memory management.

## Changes

### Things I Added:
- Added `JMP` (relative) and `JMPABS` (absolute) jump instructions for flexible program flow.
- Implemented conditional jumps: `IFGT`, `IFLT`, `IFEQ`, `IFNE` for decision-making based on register values.
- Introduced `CALL_ADDR` and `RET_SUB` instructions for efficient subroutine calls and stack management.
- Added `PRINTF` for C-style formatted output, and `PRINT` / `PRINTLN` for simple string output.
- Implemented `INPUT` to read floating-point values from standard input into registers.
- Added `DUMP_STATE` and `DUMP_REGS` instructions for enhanced debugging and state inspection.
- Implemented `LOAD_REG_MEM` and `STORE_MEM_REG` for direct data transfer between registers and memory.
- Added `PUSH_REG` and `POP_REG` for stack operations on register values.
- Introduced `ALLOC` and `FREE` for dynamic memory management.
- Implemented `CMP` for comparing register values.
- Added bitwise operations: `AND_BITS`, `OR_BITS`, `XOR_BITS`, `NOT_BITS`.
- Implemented bit shift operations: `SHL` (shift left) and `SHR` (shift right).
- Added `BREAK_POINT` instruction for program debugging.
- Implemented `GET_TIME` to retrieve the system timestamp into a register.
- Added `SEED_RNG` to seed the random number generator for reproducible results.
- Introduced `EXIT_CODE` to terminate the program with a specified exit code.
- Added `Rand` for generating random floating-point numbers.
- Implemented mathematical functions: `Sqrt`, `Exp`, `Log`.
- Added register arithmetic operations: `RegAdd`, `RegSub`, `RegMul`, `RegDiv`, `RegCopy`.
- Introduced character I/O: `CharLoad` for loading character ASCII values into registers, and `CharOut` for printing characters from registers.
- Implemented `RegSet` for directly setting register values.
- Added `LoopStart` and `LoopEnd` instructions for defining program loops.

### Things I Improved:
- Enhanced classical control flow with new jump and subroutine instructions.
- Expanded I/O capabilities with new print and input instructions.
- Introduced dynamic memory management.
- Added comprehensive bitwise and arithmetic operations for classical registers.
- Improved debugging with state and register dumping, and breakpoints.
- Enhanced random number generation control.

## Migration of old QOA source files:
- No syntax changes or instruction deprecations were introduced for existing v0.2.4 instructions.
- Existing `.qoa` files remain compatible, but recompilation is recommended to leverage new features and avoid potential compiler warnings.
- You can also leverage the new instructions to implement more sophisticated quantum and optical protocols, including feedback loops, error correction, and advanced photonic state preparation.
