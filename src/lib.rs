// ONLY in nightly

#![feature(portable_simd)]

// #![feature(stdarch_x86_mm_shuffle)] // if you are NOT on x86-64, comment this line out
// disabling this on x86-64 systems will not break QOA compiliation, but is reccomended for optimization and preformance.

// #![feature(rustc_attrs)] // this is a internal compiler feature. using internal features are discouraged.

pub mod distribute; // for distribution
pub mod instructions; // for instruction enum
pub mod runtime; // for runtime
pub mod vectorization; // for vectorization implementation
pub mod vulkan; // vulkan module
