# QOA v0.2.3 Release Notes

Release Date: 19/06/2025  
Status: Stable Pre Release

## Summary

QOA v0.2.3 introduces IonQ JSON compiler support, enabling conversion of ".qoa" source files into IonQ-compatible JSON circuit files. This integration facilitates running QOA circuits on IonQ’s simulators and real quantum hardware via their API.

## Changes

### Things I Added:

- Added a new "compile-json" command to the CLI to compile ".qoa" files into IonQ-compatible JSON format.
- Support for translating QOA assembly instructions (Hadamard, RZ, CZ, etc.) into IonQ JSON gate representations.
- Output JSON conforms to IonQ's expected schema, supporting key fields such as "circuit", "qubits", "shots", and "target".

### Things I Improved:

- Refined instruction parsing and encoding to ensure accurate gate translation in JSON output.
- Improved CLI argument parsing for enhanced user experience when compiling JSON output.

## Migration of old qoa source files:

- No breaking changes introduced.
- Users wishing to use IonQ hardware or simulators must recompile ".qoa" sources with the new "compile-json" command.
- Existing ".qexe" binaries and previous JSON outputs remain compatible.
