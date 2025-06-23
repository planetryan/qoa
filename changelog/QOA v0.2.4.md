# QOA v0.2.4 Release Notes

Release Date: 23/06/2025
Status: Pre Release

## Summary

QOA v0.2.4 significantly enhances quantum control with new feedback and conditional execution capabilities and of course, noise generation, alongside expanded optical photon manipulation. This version introduces direct controlled phase gates, quantum non-demolition measurements, and improved error handling, laying groundwork for more complex quantum algorithms and real-time system adjustments.

## Changes

### Things I Added:

- Added `DECOHERE_PROTECT n duration` instruction for decoherence mitigation.
- Added `FEEDBACK_CONTROL n measurement_reg` for adaptive quantum operations.
- Added `ENTANGLE_CLUSTER n1 n2 ... nN` for generating cluster states.
- Added `APPLY_CPHASE control target angle` for controlled phase gates.
- Added `APPLY_QND_MEASUREMENT n dest` for non-demolition measurements.
- Added `ERROR_SYNDROME n code dest` for error syndrome extraction.
- Added `PHOTON_BUNCHING_CONTROL n enable` for photon bunching management.
- Added `SINGLE_PHOTON_SOURCE_ON n` and `SINGLE_PHOTON_SOURCE_OFF n` for deterministic photon emission.
- Added `APPLY_LINEAR_OPTICAL_TRANSFORM matrix_addr src_modes dest_modes count` for complex optical transformations.
- Added `PHOTON_DETECT_COINCIDENCE n1 n2 ... nN dest` for multi-photon event detection.
- Added `CONTROLLED_SWAP control target1 target2` (Fredkin gate).
- Added `APPLY_DISPLACEMENT_FEEDBACK n feedback_reg` for feedback-driven displacement.
- Added `PHOTON_DETECT_WITH_THRESHOLD n threshold dest` for conditional photon detection.
- Added `APPLY_MULTI_QUBIT_ROTATION n1 n2 ... nN axis angles` for simultaneous rotations.
- Added `OPTICAL_SWITCH_CONTROL n state` for physical optical switch control.
- Added `APPLY_FEEDFORWARD_GATE n control_reg` for dynamically applied gates.
- Added `APPLY_NONLINEAR_SIGMA n param` for custom nonlinear optical operations.
- Added `MEASURE_WITH_DELAY n delay dest` for delayed measurements.
- Added `PHOTON_LOSS_CORRECTION n code` for optical error correction.
- Added `PHOTON_EMISSION_PATTERN n pattern duration` for patterned photon emission.
- Added `APPLY_SQUEEZING_FEEDBACK n feedback_reg` for feedback-driven squeezing.
- Added `APPLY_PHOTON_SUBTRACTION n` and `PHOTON_ADDITION n` for photon number state engineering.
- Added `APPLY_MEASUREMENT_BASIS_CHANGE n basis` for dynamic measurement basis changes.
- Added `CONTROLLED_PHASE_ROTATION control target angle` for conditional phase rotation.
- Added `.qx, .qox, .qex, .oex` and other executable formats, exiesting `.qexe` and other formats remains compatible.

### Things I Improved:

- Enhanced classical-quantum interaction with more explicit `LOAD_CLASSICAL` and `STORE_CLASSICAL` instructions.
- Expanded error detection with dedicated `APPLY_PHASE_FLIP` and `APPLY_BIT_FLIP` operations.
- Refined quantum state manipulation with `APPLY_T_GATE` and `APPLY_S_GATE`.
- Improved measurement flexibility with `MEASURE_IN_BASIS`.
- Extended optical system control with fine-grained photon operations.
- I improved Quantum Noise Generation significantly, but ideal enviorment still can be used

## Migration of old qoa source files:

- No syntax changes or instruction deprecations were introduced for existing v0.2.3 instructions.
- Existing `.qoa` files remain compatible, but I would recompile to avoid compiler errors
- Developers can now leverage the new instructions to implement more sophisticated quantum and optical protocols, including feedback loops, error correction, and advanced photonic state preparation.