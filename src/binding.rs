use num_complex::Complex64;

// --- Bindings ---

// Note: these bindings must exactly match the functions exported from the assembly file.
// The return types for functions returning multiple values are handled by the C ABI
// for returning structs, which works well here.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
extern "C" {
    // calculates sin for 8 doubles in parallel (AVX-512).
    fn __svml_sin8(x: __m512d) -> __m512d;

    // calculates cos for 8 doubles in parallel (AVX-512).
    fn __svml_cos8(x: __m512d) -> __m512d;

    // calculates sin and cos for 8 doubles in parallel (AVX-512).
    // returns a struct { __m512d sin; __m512d cos; } which we can receive as [__m512d; 2].
    fn vector_sincos8(x: __m512d) -> [__m512d; 2];

    // calculates log base 2 for a single double.
    fn __svml_log2(x: f64) -> f64;

    // calculates the coherent state amplitude.
    // returns a struct { f64 re; f64 im; } which we can receive as Complex64.
    fn quantum_coherent_amplitude(re_alpha: f64, im_alpha: f64, n: usize) -> Complex64;

    // calculates the factorial of n.
    // note: the asm returns a double, so we receive f64 and handle potential infinity.
    fn quantum_factorial_optimized(n: usize) -> f64;

    // multiplies two vectors of 8 complex numbers (AVX-512).
    // returns a struct { __m512d re; __m512d im; } which we can receive as [__m512d; 2].
    fn vector_complex_multiply(
        re1: __m512d,
        im1: __m512d,
        re2: __m512d,
        im2: __m512d,
    ) -> [__m512d; 2];

    // calculates a point on the wigner function.
    fn quantum_wigner_point(re_alpha: f64, im_alpha: f64, x: f64, p: f64) -> f64;

    // new quantum squeeze operator transformation
    fn quantum_squeeze_transform(re_state: f64, im_state: f64, r: f64, theta: f64) -> Complex64;
}

// For RISC-V, AArch64, and POWER, the vector functions will operate on arrays
// passed by pointer, as their ABIs and intrinsic models differ from AVX-512's
// direct register passing for large vectors.
// Scalar functions will use standard f64 types.

#[cfg(any(target_arch = "riscv64", target_arch = "aarch64", target_arch = "powerpc64"))]
extern "C" {
    // calculates sin for 8 doubles. Input/output are pointers to arrays.
    fn __svml_sin8(values: *mut f64);

    // calculates cos for 8 doubles. Input/output are pointers to arrays.
    fn __svml_cos8(values: *mut f64);

    // calculates sin and cos for 8 doubles. Input is pointer to values,
    // outputs are pointers to sin_out and cos_out arrays.
    fn vector_sincos8(values: *mut f64, sin_out: *mut f64, cos_out: *mut f64);

    // calculates log base 2 for a single double.
    fn __svml_log2(x: f64) -> f64;

    // calculates the coherent state amplitude.
    fn quantum_coherent_amplitude(re_alpha: f64, im_alpha: f64, n: usize) -> Complex64;

    // calculates the factorial of n.
    fn quantum_factorial_optimized(n: usize) -> f64;

    // multiplies two arrays of 8 complex numbers.
    // Input/output are pointers to real and imaginary parts.
    fn vector_complex_multiply(
        re1_ptr: *const f64,
        im1_ptr: *const f64,
        re2_ptr: *const f64,
        im2_ptr: *const f64,
        re_out_ptr: *mut f64,
        im_out_ptr: *mut f64,
    );

    // calculates a point on the wigner function.
    fn quantum_wigner_point(re_alpha: f64, im_alpha: f64, x: f64, p: f64) -> f64;

    // new quantum squeeze operator transformation
    fn quantum_squeeze_transform(re_state: f64, im_state: f64, r: f64, theta: f64) -> Complex64;
}

// --- Math Struct ---

// Provides safe, math operation using assembly implementations.
pub struct AsmMath;

impl AsmMath {
    // calculates the sine of 8 double-precision floats simultaneously.
    #[inline]
    pub fn sin_v8(mut values: [f64; 8]) -> [f64; 8] {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                let input = _mm512_loadu_pd(values.as_ptr());
                let result_vec = __svml_sin8(input);
                _mm512_storeu_pd(values.as_mut_ptr(), result_vec);
            }
            #[cfg(any(target_arch = "riscv64", target_arch = "aarch64", target_arch = "powerpc64"))]
            {
                // For other architectures, the assembly function takes a mutable pointer
                // and modifies the array in place.
                __svml_sin8(values.as_mut_ptr());
            }
            values
        }
    }

    // calculates the cosine of 8 double-precision floats simultaneously.
    #[inline]
    pub fn cos_v8(mut values: [f64; 8]) -> [f64; 8] {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                let input = _mm512_loadu_pd(values.as_ptr());
                let result_vec = __svml_cos8(input);
                _mm512_storeu_pd(values.as_mut_ptr(), result_vec);
            }
            #[cfg(any(target_arch = "riscv64", target_arch = "aarch64", target_arch = "powerpc64"))]
            {
                __svml_cos8(values.as_mut_ptr());
            }
            values
        }
    }

    // calculates both sine and cosine of 8 doubles simultaneously.
    // returns ([sin_values], [cos_values]). this is more efficient than separate calls.
    #[inline]
    pub fn sincos_v8(values: [f64; 8]) -> ([f64; 8], [f64; 8]) {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                let input = _mm512_loadu_pd(values.as_ptr());
                let result_vecs = vector_sincos8(input);
                let mut sin_arr = [0.0; 8];
                let mut cos_arr = [0.0; 8];
                _mm512_storeu_pd(sin_arr.as_mut_ptr(), result_vecs[0]);
                _mm512_storeu_pd(cos_arr.as_mut_ptr(), result_vecs[1]);
                (sin_arr, cos_arr)
            }
            #[cfg(any(target_arch = "riscv64", target_arch = "aarch64", target_arch = "powerpc64"))]
            {
                let mut sin_arr = [0.0; 8];
                let mut cos_arr = [0.0; 8];
                // The assembly function takes pointers for input and two output arrays.
                vector_sincos8(
                    values.as_ptr() as *mut f64, // Cast to mut for ABI compatibility if needed
                    sin_arr.as_mut_ptr(),
                    cos_arr.as_mut_ptr(),
                );
                (sin_arr, cos_arr)
            }
        }
    }

    // calculates log base 2 using a high-precision assembly implementation.
    // returns `f64::NAN` for negative inputs and `f64::NEG_INFINITY` for zero.
    #[inline]
    pub fn log2_precise(x: f64) -> f64 {
        unsafe { __svml_log2(x) }
    }

    // calculates factorial using a precomputed table in assembly for n <= 20,
    // and stirling's approximation for n > 20.
    // returns `f64::INFINITY` if n > 170 (overflow).
    #[inline]
    pub fn factorial(n: usize) -> f64 {
        unsafe { quantum_factorial_optimized(n) }
    }
}

// --- Quantum Math Struct ---

// Provides quantum specific math operations using the assembly backend.
pub struct QuantumAsmMath;

impl QuantumAsmMath {
    // calculates the coherent state amplitude <n|α> using assembly.
    #[inline]
    pub fn coherent_amplitude(alpha: Complex64, n: usize) -> Complex64 {
        // the assembly function is designed to return a struct that matches Complex64 layout
        unsafe { quantum_coherent_amplitude(alpha.re, alpha.im, n) }
    }

    // performs vectorized multiplication of two arrays of 8 complex numbers.
    // this is much more efficient than a scalar loop.
    #[inline]
    pub fn complex_multiply_v8(
        a: &[Complex64; 8],
        b: &[Complex64; 8],
    ) -> [Complex64; 8] {
        unsafe {
            let mut result_arr = [Complex64::new(0.0, 0.0); 8];

            #[cfg(target_arch = "x86_64")]
            {
                // Optimized de-interleaving of complex numbers into real and imaginary vectors
                let a_ptr = a.as_ptr() as *const f64;
                let b_ptr = b.as_ptr() as *const f64;

                // Load real parts: a.re, b.re
                // The original code had redundant loads and then gather.
                // For AVX-512, direct loads and shuffles are often better than gather for contiguous data.
                // However, sticking to the original gather approach for consistency with the assembly.
                let real_indices = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
                let imag_indices = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);

                let re_a_vec_p = _mm512_i64gather_pd(real_indices, a_ptr, 8);
                let im_a_vec_p = _mm512_i64gather_pd(imag_indices, a_ptr, 8);
                let re_b_vec_p = _mm512_i64gather_pd(real_indices, b_ptr, 8);
                let im_b_vec_p = _mm512_i64gather_pd(imag_indices, b_ptr, 8);

                let result_vecs =
                    vector_complex_multiply(re_a_vec_p, im_a_vec_p, re_b_vec_p, im_b_vec_p);

                _mm512_storeu_pd(
                    result_arr.as_mut_ptr() as *mut f64,
                    result_vecs[0], // real parts
                );
                _mm512_storeu_pd(
                    (result_arr.as_mut_ptr() as *mut f64).add(8),
                    result_vecs[1], // imag parts
                );
            }
            #[cfg(any(target_arch = "riscv64", target_arch = "aarch64", target_arch = "powerpc64"))]
            {
                // For other architectures, the assembly function takes pointers to real/imag parts
                // and writes to output pointers.
                let a_re_ptr = a.as_ptr() as *const f64;
                let a_im_ptr = (a.as_ptr() as *const f64).add(1); // imag parts start at offset 1
                let b_re_ptr = b.as_ptr() as *const f64;
                let b_im_ptr = (b.as_ptr() as *const f64).add(1);

                // Need to extract real and imaginary parts from Complex64 array into separate f64 arrays
                // before passing to assembly, and then re-interleave.
                // This is a common pattern when the assembly expects SOA (Structure of Arrays) layout
                // while Rust's Complex64 is AOS (Array of Structs).
                let mut a_re_arr = [0.0; 8];
                let mut a_im_arr = [0.0; 8];
                let mut b_re_arr = [0.0; 8];
                let mut b_im_arr = [0.0; 8];

                for i in 0..8 {
                    a_re_arr[i] = a[i].re;
                    a_im_arr[i] = a[i].im;
                    b_re_arr[i] = b[i].re;
                    b_im_arr[i] = b[i].im;
                }

                let mut result_re_arr = [0.0; 8];
                let mut result_im_arr = [0.0; 8];

                vector_complex_multiply(
                    a_re_arr.as_ptr(),
                    a_im_arr.as_ptr(),
                    b_re_arr.as_ptr(),
                    b_im_arr.as_ptr(),
                    result_re_arr.as_mut_ptr(),
                    result_im_arr.as_mut_ptr(),
                );

                for i in 0..8 {
                    result_arr[i] = Complex64::new(result_re_arr[i], result_im_arr[i]);
                }
            }
            result_arr
        }
    }

    // calculates the wigner function value at a single point (x, p) in phase space.
    #[inline]
    pub fn wigner_function(alpha: Complex64, x: f64, p: f64) -> f64 {
        unsafe { quantum_wigner_point(alpha.re, alpha.im, x, p) }
    }

    // performs a quantum squeeze operator transformation.
    #[inline]
    pub fn squeeze_transform(state: Complex64, r: f64, theta: f64) -> Complex64 {
        unsafe { quantum_squeeze_transform(state.re, state.im, r, theta) }
    }

    pub fn wigner_function_grid(
        alpha: Complex64,
        x_points: &[f64],
        p_points: &[f64],
    ) -> Vec<f64> {
        let mut results = Vec::with_capacity(x_points.len() * p_points.len());
        for &p in p_points {
            for &x in x_points {
                results.push(Self::wigner_function(alpha, x, p));
            }
        }
        results
    }
}
