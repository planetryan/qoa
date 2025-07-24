#![feature(portable_simd)] // enable access to the new portable simd api (for future use/consideration)

pub mod instructions; // for instruction enum
pub mod runtime; // for runtime
pub mod vectorization; // for vectorization implementation