Quantum Optical Assembly (QOA) ISA Changelog — Version 0.2.0

Release Date: 15 June 2025

What's New
==========

Version 0.2.0 fixes some annoying inconsistencies in the binary format and cleans up the opcode assignments. The goal was to make QOA programs less likely to crash or hang in weird ways and easier to debug when they do.

If you're just getting started with QOA, this version should be much better to work with.

Main Changes
================

Opcode Changes
--------------

QInit moved from 0x01 to 0x04 - This will break your existing binaries. The old 0x01 assignment was causing parser conflicts, so I had to move it. You'll need to recompile anything you've already built. (Not like anyone uses qoa at the moment though)

Other opcodes that are updated:
  • QMEAS: 0x32 (was inconsistent before)
  • CHARLOAD: 0x31 (for printing ASCII chars)  
  • HALT: 0xFF (new opcode - finally gives you a clean way to stop execution)

Binary Format
-------------

.qexe files now need a proper header:
  • 4 bytes: magic number ("QEXE")
  • 1 byte: version number
  • 4 bytes: payload length (little-endian)

The old headerless format won't work anymore.

How It Works Now
================

Two-Pass Execution
------------------

The executor now runs through the program twice:

1. Pass 1: Scans for the highest qubit index so it knows how big to make the quantum state
2. Pass 2: runs instructions

This prevents the annoying crashes i'd get when trying to use qubit 5 but only having 3 qubits allocated.

Better Error Handling
---------------------

  • Bounds checking on instruction lengths
  • Clear error messages with payload positions when something goes wrong
  • Won't try to execute garbage data if binary is corrupted

Current Instruction Set
=======================

Instruction | Opcode | What it does           | Format
----------- | ------ | ---------------------- | --------------------------------------
QInit       | 0x04   | Initialize a qubit     | opcode + qubit_index
QGate       | 0x02   | Apply quantum gate     | opcode + qubit_index + 8-byte gate name
CHARLOAD    | 0x31   | Print ASCII character  | opcode + qubit_index + ascii_char
QMEAS       | 0x32   | Measure qubit          | opcode + qubit_index
HALT        | 0xFF   | Stop execution         | opcode only

Migration Notes
===============

old QOA binaries from 0.1.0 won't work. The QInit opcode change means:
1. Recompile from source, or
2. Hex edit binaries to change 0x01 to 0x04 where appropriate

What's Next
===========

The instruction set is getting more stable. Future versions might add multi qubit gates and parameterized operations, but the core format shouldn't change much from here.

Thanks for using QOA!

-- Rayan