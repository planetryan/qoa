// only in nightly

#![feature(portable_simd)]
#![feature(stdarch_x86_mm_shuffle)]
// #![feature(rustc_attrs)] // this is a internal compiler feature. using internal features are discouraged.

pub mod instructions; // for instruction enum
pub mod runtime; // for runtime
pub mod vectorization; // for vectorization implementation