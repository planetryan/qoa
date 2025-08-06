#[allow(unused_imports)] // for conditional imports
use clap::Parser;
use image::{ImageBuffer, ImageFormat, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use memmap2::MmapOptions;
use num_complex::Complex;
use parking_lot::{Mutex, RwLock};
use qoa::runtime::quantum_state::{NoiseConfig, QuantumState};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use realfft::RealFftPlanner;
use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use bumpalo::Bump;

// --- aligned buffer for cache-line alignment ---

// this struct can make sure that the vector data is aligned to a cache line boundary (64 bytes)
// which can improve simd performance by preventing unaligned memory accesses

#[repr(C, align(64))]
struct AlignedBuffer<T> {
    data: Vec<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Copy + Default> AlignedBuffer<T> {
    // creates a new aligned buffer with a given capacity, initializing elements with their default value
    fn new(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        // safety: set_len is unsafe but used here to pre-allocate and initialize memory
        // it's safe because we immediately fill it with default values
        unsafe { data.set_len(capacity); }
        Self {
            data,
            _phantom: std::marker::PhantomData,
        }
    }

    // gets an element at a given index without bounds checking for performance
    #[inline(always)]
    #[allow(dead_code)] // this method is not directly used in visualizer.rs but might be externally
    fn get(&self, idx: usize) -> T {
        // safety: unchecked access is safe assuming idx is within bounds,
        // which should be ensured by the caller in performance-critical loops
        unsafe { *self.data.get_unchecked(idx) }
    }

    // sets an element at a given index without bounds checking for performance
    #[inline(always)]
    #[allow(dead_code)] // this method is not directly used in visualizer.rs but might be externally
    fn set(&mut self, idx: usize, val: T) {
        // safety: unchecked access is safe assuming idx is within bounds
        unsafe { *self.data.get_unchecked_mut(idx) = val; }
    }

    // returns a mutable slice of the buffer's data
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    // returns an immutable slice of the buffer's data
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        &self.data
    }

    // resizes the buffer, filling new elements with default values
    #[allow(dead_code)] // this method is not directly used in visualizer.rs but might be externally
    fn resize(&mut self, new_len: usize) {
        let old_len = self.data.len();
        if new_len > old_len {
            self.data.resize(new_len, T::default());
        } else {
            self.data.truncate(new_len);
        }
    }
}

// --- simd audio module ---
mod audio_simd {
    use super::AlignedBuffer; 

    // on x86_64, we use runtime detection for avx512, avx2, and sse.
    #[cfg(target_arch = "x86_64")]
    mod platform_impl {
        use std::arch::x86_64::*;
        use super::AlignedBuffer;

        // public api for sum_squares_f32, dispatches to appropriate simd or scalar fallback
        pub fn sum_squares_f32(samples: &AlignedBuffer<f32>) -> f64 {
            if is_x86_feature_detected!("avx512f") {
                unsafe { sum_squares_avx512(samples) }
            } else if is_x86_feature_detected!("avx2") {
                unsafe { sum_squares_avx2(samples) }
            } else if is_x86_feature_detected!("sse") {
                unsafe { sum_squares_sse(samples) }
            } else {
                super::scalar_fallback::sum_squares_f32(samples.as_slice())
            }
        }

        // public api for mix_stereo_to_mono_f32, dispatches to appropriate simd or scalar fallback
        pub unsafe fn mix_stereo_to_mono_f32(samples: &[f32], output: &mut Vec<f32>) {
            if is_x86_feature_detected!("avx512f") {
                mix_stereo_to_mono_avx512(samples, output)
            } else if is_x86_feature_detected!("avx2") {
                mix_stereo_to_mono_avx2(samples, output)
            } else if is_x86_feature_detected!("sse3") {
                mix_stereo_to_mono_sse(samples, output)
            } else {
                super::scalar_fallback::mix_stereo_to_mono_f32(samples, output)
            }
        }

        // public api for calculate_spectral_centroid_f64, dispatches to appropriate simd or scalar fallback
        #[target_feature(enable = "avx2")] // Added target_feature
        pub unsafe fn calculate_spectral_centroid_f64( // Marked as unsafe fn
            spectrum_mags: &[f64],
            bin_width: f64,
        ) -> (f64, f64) {
            calculate_spectral_centroid_avx2(spectrum_mags, bin_width)
        }

        // calculates sum of squares for f32 samples using avx512 intrinsics
        #[target_feature(enable = "avx512f")]
        unsafe fn sum_squares_avx512(samples: &AlignedBuffer<f32>) -> f64 {
            let samples_slice = samples.as_slice();
            let chunks = samples_slice.chunks_exact(16);
            let remainder = chunks.remainder();
            let mut sum_vec = _mm512_setzero_ps();
            for chunk in chunks {
                // load data and perform fused multiply-add
                let data = _mm512_loadu_ps(chunk.as_ptr());
                sum_vec = _mm512_fmadd_ps(data, data, sum_vec);
            }
            // horizontal sum of the vector
            let mut total_sum = _mm512_reduce_add_ps(sum_vec) as f64;
            total_sum += remainder.iter().map(|&s| s as f64 * s as f64).sum::<f64>();
            total_sum
        }

        // calculates sum of squares for f32 samples using avx2 intrinsics
        #[target_feature(enable = "avx2", enable = "fma")]
        unsafe fn sum_squares_avx2(samples: &AlignedBuffer<f32>) -> f64 {
            let samples_slice = samples.as_slice();
            let chunks = samples_slice.chunks_exact(8);
            let remainder = chunks.remainder();
            let mut sum_vec = _mm256_setzero_ps();
            for chunk in chunks {
                // load data and perform fused multiply-add
                let data = _mm256_loadu_ps(chunk.as_ptr());
                sum_vec = _mm256_fmadd_ps(data, data, sum_vec);
            }
            // more efficient horizontal sum for avx2
            let mut total_sum = horizontal_sum_ps(sum_vec) as f64;
            total_sum += remainder.iter().map(|&s| s as f64 * s as f64).sum::<f64>();
            total_sum
        }

        // calculates sum of squares for f32 samples using sse intrinsics
        #[target_feature(enable = "sse")]
        unsafe fn sum_squares_sse(samples: &AlignedBuffer<f32>) -> f64 {
            let samples_slice = samples.as_slice();
            let chunks = samples_slice.chunks_exact(4);
            let remainder = chunks.remainder();
            let mut sum_vec = _mm_setzero_ps();
            for chunk in chunks {
                // load data and perform multiply-add
                let data = _mm_loadu_ps(chunk.as_ptr());
                sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(data, data));
            }
            // horizontal sum using sse instructions
            let mut total_sum = {
                let shuf = _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(2, 3, 0, 1));
                let sums = _mm_add_ps(sum_vec, shuf);
                let shuf = _mm_movehl_ps(sums, sums);
                let sums = _mm_add_ss(sums, shuf);
                _mm_cvtss_f32(sums)
            } as f64;
            total_sum += remainder.iter().map(|&s| s as f64 * s as f64).sum::<f64>();
            total_sum
        }

        // mixes stereo f32 samples to mono using avx512 intrinsics
        #[target_feature(enable = "avx512f")]
        unsafe fn mix_stereo_to_mono_avx512(samples: &[f32], output: &mut Vec<f32>) {
            let chunks = samples.chunks_exact(32); // 16 stereo pairs
            let remainder = chunks.remainder();
            output.reserve(samples.len() / 2);
            let mut buffer = [0.0f32; 16];
            let half_vec = _mm512_set1_ps(0.5);
            for chunk in chunks {
                // de-interleave using shuffles
                let data = _mm512_loadu_ps(chunk.as_ptr());
                let lefts = _mm512_shuffle_ps(data, data, 0b10_00_10_00);
                let rights = _mm512_shuffle_ps(data, data, 0b11_01_11_01);
                let sums = _mm512_add_ps(lefts, rights);
                let avgs = _mm512_mul_ps(sums, half_vec);
                _mm512_storeu_ps(buffer.as_mut_ptr(), avgs);
                output.extend_from_slice(&buffer);
            }
            for pair in remainder.chunks_exact(2) {
                output.push((pair[0] + pair[1]) * 0.5);
            }
        }

        // mixes stereo f32 samples to mono using avx2 intrinsics
        #[target_feature(enable = "avx2")]
        unsafe fn mix_stereo_to_mono_avx2(samples: &[f32], output: &mut Vec<f32>) {
            let chunks = samples.chunks_exact(16); // 8 stereo pairs
            let remainder = chunks.remainder();
            let half_vec = _mm256_set1_ps(0.5);
            output.reserve(samples.len() / 2);
            let mut buffer = [0.0f32; 8];
            for chunk in chunks {
                let left = _mm256_loadu_ps(chunk.as_ptr());
                let right = _mm256_loadu_ps(chunk.as_ptr().add(8));
                let sums = _mm256_hadd_ps(left, right);
                let avgs = _mm256_mul_ps(sums, half_vec);
                _mm256_storeu_ps(buffer.as_mut_ptr(), avgs);
                output.extend_from_slice(&buffer);
            }
            for pair in remainder.chunks_exact(2) {
                output.push((pair[0] + pair[1]) * 0.5);
            }
        }

        // mixes stereo f32 samples to mono using sse3 intrinsics
        #[target_feature(enable = "sse3")]
        unsafe fn mix_stereo_to_mono_sse(samples: &[f32], output: &mut Vec<f32>) {
            let chunks = samples.chunks_exact(8); // 4 stereo pairs
            let remainder = chunks.remainder();
            let half_vec = _mm_set1_ps(0.5);
            output.reserve(samples.len() / 2);
            let mut buffer = [0.0f32; 4];
            for chunk in chunks {
                let left = _mm_loadu_ps(chunk.as_ptr());
                let right = _mm_loadu_ps(chunk.as_ptr().add(4));
                let sums = _mm_hadd_ps(left, right);
                let avgs = _mm_mul_ps(sums, half_vec);
                _mm_storeu_ps(buffer.as_mut_ptr(), avgs);
                output.extend_from_slice(&buffer);
            }
            for pair in remainder.chunks_exact(2) {
                output.push((pair[0] + pair[1]) * 0.5);
            }
        }

        // calculates spectral centroid for f64 magnitudes using avx2 intrinsics
        #[target_feature(enable = "avx2")]
        unsafe fn calculate_spectral_centroid_avx2(
            spectrum_mags: &[f64],
            bin_width: f64,
        ) -> (f64, f64) {
            let chunks = spectrum_mags.chunks_exact(4);
            let chunks_len = chunks.len();
            let remainder = chunks.remainder();

            let mut sum_weighted_freq_vec = _mm256_setzero_pd();
            let mut sum_magnitudes_vec = _mm256_setzero_pd();

            let bin_width_vec = _mm256_set1_pd(bin_width);

            for (i, chunk) in chunks.enumerate() {
                let magnitudes = _mm256_loadu_pd(chunk.as_ptr());
                let freqs_base = _mm256_set1_pd((i * 4) as f64);
                let freqs_offset = _mm256_set_pd(3.0, 2.0, 1.0, 0.0); // reversed for correct order
                let freqs = _mm256_mul_pd(_mm256_add_pd(freqs_base, freqs_offset), bin_width_vec);

                let weighted = _mm256_mul_pd(freqs, magnitudes);

                sum_weighted_freq_vec = _mm256_add_pd(sum_weighted_freq_vec, weighted);
                sum_magnitudes_vec = _mm256_add_pd(sum_magnitudes_vec, magnitudes);
            }

            // more efficient horizontal sum for f64
            let mut total_sum_weighted_freq = horizontal_sum_pd(sum_weighted_freq_vec);
            let mut total_sum_magnitudes = horizontal_sum_pd(sum_magnitudes_vec);

            // process remainder
            // use the original `chunks` variable for `chunks_len` to avoid borrow issues
            for (i, &mag) in remainder.iter().enumerate() {
                let current_idx = chunks_len * 4 + i;
                total_sum_weighted_freq += current_idx as f64 * bin_width * mag;
                total_sum_magnitudes += mag;
            }

            (total_sum_weighted_freq, total_sum_magnitudes)
        }

        // calculates band energy for f64 magnitudes using avx2 intrinsics
        #[target_feature(enable = "avx2", enable = "fma")]
        pub unsafe fn calculate_band_energy_simd(
            spectrum_mags: &[f64],
            sample_rate: u32,
            low_hz: f64,
            high_hz: f64,
        ) -> f64 {
            if spectrum_mags.is_empty() || sample_rate == 0 || low_hz >= high_hz {
                return 0.0; // early return for invalid inputs
            }

            let nyquist_freq = sample_rate as f64 / 2.0;
            let bin_width = nyquist_freq / spectrum_mags.len() as f64;

            let start_bin = (low_hz / bin_width).floor().max(0.0) as usize;
            let end_bin = (high_hz / bin_width).ceil() as usize;

            if end_bin <= start_bin || start_bin >= spectrum_mags.len() {
                return 0.0;
            }

            let end_bin = end_bin.min(spectrum_mags.len() - 1);
            let slice = &spectrum_mags[start_bin..=end_bin];

            let chunks = slice.chunks_exact(4);
            let remainder = chunks.remainder();

            let mut sum_vec = _mm256_setzero_pd();

            for chunk in chunks {
                let mag_vec = _mm256_loadu_pd(chunk.as_ptr());
                sum_vec = _mm256_fmadd_pd(mag_vec, mag_vec, sum_vec);
            }

            // horizontal sum
            let result = horizontal_sum_pd(sum_vec);

            // add remainder
            let remainder_sum = remainder.iter().map(|&mag| mag * mag).sum::<f64>();

            result + remainder_sum
        }

        // performs a horizontal sum of a __m256d vector (4 f64 elements)
        #[target_feature(enable = "sse2", enable = "avx")] // ensure required features are enabled for this function
        unsafe fn horizontal_sum_pd(vec: __m256d) -> f64 {
            let sum = _mm_add_pd(
                _mm256_castpd256_pd128(vec),
                _mm256_extractf128_pd(vec, 1)
            );
            let sum = _mm_add_sd(sum, _mm_unpackhi_pd(sum, sum));
            _mm_cvtsd_f64(sum)
        }

        // performs a horizontal sum of a __m256ps vector (8 f32 elements)
        #[target_feature(enable = "sse", enable = "sse3", enable = "avx")] // ensure required features are enabled for this function
        unsafe fn horizontal_sum_ps(vec: __m256) -> f32 {
            let upper = _mm256_extractf128_ps(vec, 1);
            let lower = _mm256_castps256_ps128(vec);
            let sum_128 = _mm_add_ps(upper, lower);
            let sum_hadd = _mm_hadd_ps(sum_128, sum_128);
            let final_hadd = _mm_hadd_ps(sum_hadd, sum_hadd);
            _mm_cvtss_f32(final_hadd)
        }
    }

    // on aarch64, we assume neon is available.
    #[cfg(target_arch = "aarch64")]
    mod platform_impl {
        #[allow(unused_imports)] // this import is conditionally used
        use std::arch::aarch64::*;
        use super::AlignedBuffer;

        // public api for sum_squares_f32, dispatches to appropriate simd or scalar fallback
        #[target_feature(enable = "neon")]
        pub unsafe fn sum_squares_f32(samples: &AlignedBuffer<f32>) -> f64 {
            let samples_slice = samples.as_slice();
            let chunks = samples_slice.chunks_exact(4);
            let remainder = chunks.remainder();
            let mut sum_vec = vdupq_n_f32(0.0);
            for chunk in chunks {
                // load data and perform fused multiply-accumulate
                sum_vec = vfmaq_f32(sum_vec, unsafe { vld1q_f32(chunk.as_ptr()) }, unsafe { vld1q_f32(chunk.as_ptr()) });
            }
            // horizontal add
            let mut total_sum = vaddvq_f32(sum_vec) as f64;
            total_sum += remainder.iter().map(|&s| s as f64 * s as f64).sum::<f64>();
            total_sum
        }

        // public api for mix_stereo_to_mono_f32, dispatches to appropriate simd or scalar fallback
        #[target_feature(enable = "neon")]
        pub unsafe fn mix_stereo_to_mono_f32(samples: &[f32], output: &mut Vec<f32>) {
            let chunks = samples.chunks_exact(8); // 4 stereo pairs
            let remainder = chunks.remainder();
            output.reserve(samples.len() / 2);
            for chunk in chunks {
                let stereo_pairs = unsafe { vld2q_f32(chunk.as_ptr()) }; // de-interleaves into two vectors
                let sum_vec = vaddq_f32(stereo_pairs.0, stereo_pairs.1);
                let avg_vec = vmulq_n_f32(sum_vec, 0.5);
                let mut buffer = [0.0f32; 4];
                unsafe { vst1q_f32(buffer.as_mut_ptr(), avg_vec) };
                output.extend_from_slice(&buffer);
            }
            for pair in remainder.chunks_exact(2) {
                output.push((pair[0] + pair[1]) * 0.5);
            }
        }

        // public api for calculate_spectral_centroid_f64, dispatches to appropriate simd or scalar fallback
        #[target_feature(enable = "neon")] // Added target_feature
        pub unsafe fn calculate_spectral_centroid_f64( // Marked as unsafe fn
            spectrum_mags: &[f64],
            bin_width: f64,
        ) -> (f64, f64) {
            let chunks = spectrum_mags.chunks_exact(2); // neon for f64 typically uses 2 elements
            let remainder = chunks.remainder();

            let mut sum_weighted_freq_vec = vdupq_n_f64(0.0);
            let mut sum_magnitudes_vec = vdupq_n_f64(0.0);

            let bin_width_vec = vdupq_n_f64(bin_width);

            // Capture chunks length before it's consumed by `into_iter().enumerate()`
            let initial_chunks_len = chunks.len();

            for (i, chunk) in chunks.clone().enumerate() { // Clone chunks to allow subsequent use
                let magnitudes = unsafe { vld1q_f64(chunk.as_ptr()) };
                let freqs_base = vdupq_n_f64((i * 2) as f64);
                let freqs_offset = vsetq_lane_f64(1.0, vdupq_n_f64(0.0), 0);
                let freqs_offset = vsetq_lane_f64(0.0, freqs_offset, 1); // 0.0, 1.0

                let freqs = vmulq_f64(vaddq_f64(freqs_base, freqs_offset), bin_width_vec);

                let weighted = vmulq_f64(freqs, magnitudes);

                sum_weighted_freq_vec = vaddq_f64(sum_weighted_freq_vec, weighted);
                sum_magnitudes_vec = vaddq_f64(sum_magnitudes_vec, magnitudes);
            }

            let mut total_sum_weighted_freq = vaddvq_f64(sum_weighted_freq_vec);
            let mut total_sum_magnitudes = vaddvq_f64(sum_magnitudes_vec);

            // process remainder
            for (i, &mag) in remainder.iter().enumerate() {
                let current_idx = initial_chunks_len * 2 + i; // Use initial_chunks_len
                total_sum_weighted_freq += current_idx as f64 * bin_width * mag;
                total_sum_magnitudes += mag;
            }

            (total_sum_weighted_freq, total_sum_magnitudes)
        }

        // calculates band energy for f64 magnitudes using NEON intrinsics
        #[target_feature(enable = "neon")] // Added target_feature
        pub unsafe fn calculate_band_energy_simd(
            spectrum_mags: &[f64],
            sample_rate: u32,
            low_hz: f64,
            high_hz: f64,
        ) -> f64 {
            if spectrum_mags.is_empty() || sample_rate == 0 || low_hz >= high_hz {
                return 0.0; // early return for invalid inputs
            }

            let nyquist_freq = sample_rate as f64 / 2.0;
            let bin_width = nyquist_freq / spectrum_mags.len() as f64;

            let start_bin = (low_hz / bin_width).floor().max(0.0) as usize;
            let end_bin = (high_hz / bin_width).ceil() as usize;

            if end_bin <= start_bin || start_bin >= spectrum_mags.len() {
                return 0.0;
            }

            let end_bin = end_bin.min(spectrum_mags.len() - 1);
            let slice = &spectrum_mags[start_bin..=end_bin];

            let chunks = slice.chunks_exact(2);
            let remainder = chunks.remainder();

            let mut sum_vec = vdupq_n_f64(0.0);

            for chunk in chunks {
                let mag_vec = unsafe { vld1q_f64(chunk.as_ptr()) };
                sum_vec = vfmaq_f64(sum_vec, mag_vec, mag_vec);
            }

            // horizontal sum
            let result = vaddvq_f64(sum_vec);

            // add remainder
            let remainder_sum = remainder.iter().map(|&mag| mag * mag).sum::<f64>();

            result + remainder_sum
        }
    }

    // on riscv64, we use conditional compilation for the 'v' extension.
    #[cfg(target_arch = "riscv64")]
    mod platform_impl {
        use core::arch::riscv64::*;
        use super::AlignedBuffer;

        // public api for sum_squares_f32, dispatches to appropriate simd or scalar fallback
        #[target_feature(enable = "v")]
        pub unsafe fn sum_squares_f32(samples: &AlignedBuffer<f32>) -> f64 {
            let mut remaining_samples = samples.as_slice();
            let mut total_sum: f64 = 0.0;
            let acc_vec = unsafe { vfmv_s_f_f32m1(vfmv_v_f_f32m1(0.0, 1), 1) };

            while !remaining_samples.is_empty() {
                let vl = unsafe { vsetvl_e32m1(remaining_samples.len()) };
                let data_vec = unsafe { vle32_v_f32m1(remaining_samples.as_ptr(), vl) };
                let squared = unsafe { vfmul_vv_f32m1(data_vec, data_vec, vl) };
                let acc_vec = unsafe { vfredusum_vs_f32m1(squared, acc_vec, vl) };
                remaining_samples = &remaining_samples[vl..];
            }
            total_sum += unsafe { vfmv_f_s_f32m1_f32(acc_vec) } as f64;
            total_sum
        }

        // public api for mix_stereo_to_mono_f32, dispatches to appropriate simd or scalar fallback
        #[target_feature(enable = "v")]
        pub unsafe fn mix_stereo_to_mono_f32(samples: &[f32], output: &mut Vec<f32>) {
            output.resize(samples.len() / 2, 0.0);
            let mut remaining_samples = samples;
            let mut out_ptr = output.as_mut_ptr();

            while !remaining_samples.is_empty() {
                let pairs_to_process = remaining_samples.len() / 2;
                if pairs_to_process == 0 {
                    break;
                }
                let vl = unsafe { vsetvl_e32m1(pairs_to_process) };

                let left = unsafe { vlse32_v_f32m1(
                    remaining_samples.as_ptr(),
                    (std::mem::size_of::<f32>() * 2) as isize,
                    vl,
                ) };
                let right = unsafe { vlse32_v_f32m1(
                    remaining_samples.as_ptr().add(1),
                    (std::mem::size_of::<f32>() * 2) as isize,
                    vl,
                ) };

                let sum_vec = unsafe { vfadd_vv_f32m1(left, right, vl) };
                let avg_vec = unsafe { vfmul_vf_f32m1(sum_vec, 0.5, vl) };

                unsafe { vse32_v_f32m1(out_ptr, avg_vec, vl) };

                remaining_samples = &remaining_samples[vl * 2..];
                out_ptr = unsafe { out_ptr.add(vl) };
            }
        }

        // public api for calculate_spectral_centroid_f64, dispatches to appropriate simd or scalar fallback
        #[target_feature(enable = "v")] // Added target_feature
        pub unsafe fn calculate_spectral_centroid_f64( // Marked as unsafe fn
            spectrum_mags: &[f64],
            bin_width: f64,
        ) -> (f64, f64) {
            let mut remaining_mags = spectrum_mags;
            let mut total_sum_weighted_freq: f64 = 0.0;
            let mut total_sum_magnitudes: f64 = 0.0;
            let mut current_idx_base = 0;

            while !remaining_mags.is_empty() {
                let vl = unsafe { vsetvl_e64m1(remaining_mags.len()) };
                let mags_vec = unsafe { vle64_v_f64m1(remaining_mags.as_ptr(), vl) };

                // generate frequency indices
                let mut freqs_indices_vec = unsafe { vfmv_v_f_f64m1(0.0, vl) };
                for i in 0..vl {
                    freqs_indices_vec = unsafe { vset_v_f64m1(freqs_indices_vec, i, (current_idx_base + i) as f64) };
                }

                let freqs_vec = unsafe { vfmul_vf_f64m1(freqs_indices_vec, bin_width, vl) };
                let weighted_vec = unsafe { vfmul_vv_f64m1(freqs_vec, mags_vec, vl) };

                total_sum_weighted_freq += unsafe { vfmv_f_s_f64m1_f64(vfredusum_vs_f64m1(weighted_vec, vfmv_v_f_f64m1(0.0, vl), vl)) };
                total_sum_magnitudes += unsafe { vfmv_f_s_f64m1_f64(vfredusum_vs_f64m1(mags_vec, vfmv_v_f_f64m1(0.0, vl), vl)) };

                remaining_mags = &remaining_mags[vl..];
                current_idx_base += vl;
            }
            (total_sum_weighted_freq, total_sum_magnitudes)
        }

        // calculates band energy for f64 magnitudes using RISC-V V intrinsics
        #[target_feature(enable = "v")] // Added target_feature
        pub unsafe fn calculate_band_energy_simd(
            spectrum_mags: &[f64],
            sample_rate: u32,
            low_hz: f64,
            high_hz: f64,
        ) -> f64 {
            if spectrum_mags.is_empty() || sample_rate == 0 || low_hz >= high_hz {
                return 0.0; // early return for invalid inputs
            }

            let nyquist_freq = sample_rate as f64 / 2.0;
            let bin_width = nyquist_freq / spectrum_mags.len() as f64;

            let start_bin = (low_hz / bin_width).floor().max(0.0) as usize;
            let end_bin = (high_hz / bin_width).ceil() as usize;

            if end_bin <= start_bin || start_bin >= spectrum_mags.len() {
                return 0.0;
            }

            let end_bin = end_bin.min(spectrum_mags.len() - 1);
            let slice = &spectrum_mags[start_bin..=end_bin];

            let mut remaining_mags = slice;
            let mut total_sum: f64 = 0.0;
            let acc_vec = unsafe { vfmv_s_f_f64m1(vfmv_v_f_f64m1(0.0, 1), 1) };

            while !remaining_mags.is_empty() {
                let vl = unsafe { vsetvl_e64m1(remaining_mags.len()) };
                let data_vec = unsafe { vle64_v_f64m1(remaining_mags.as_ptr(), vl) };
                let squared = unsafe { vfmul_vv_f64m1(data_vec, data_vec, vl) };
                let acc_vec = unsafe { vfredusum_vs_f64m1(squared, acc_vec, vl) };
                remaining_mags = &remaining_mags[vl..];
            }
            total_sum += unsafe { vfmv_f_s_f64m1_f64(acc_vec) };
            total_sum
        }
    }

    // fallback for other architectures or when simd is disabled.
    mod scalar_fallback {

        // scalar fallback for sum of squares
        #[allow(dead_code)] // this function is used conditionally
        pub fn sum_squares_f32(samples: &[f32]) -> f64 {
            samples.iter().map(|&s| s as f64 * s as f64).sum()
        }

        // scalar fallback for mixing stereo to mono
        #[allow(dead_code)] // this function is used conditionally
        pub fn mix_stereo_to_mono_f32(samples: &[f32], output: &mut Vec<f32>) {
            output.reserve(samples.len() / 2);
            for pair in samples.chunks_exact(2) {
                output.push((pair[0] + pair[1]) * 0.5);
            }
        }

        // scalar fallback for calculating spectral centroid
        #[allow(dead_code)] // this function is used conditionally
        pub fn calculate_spectral_centroid_f64(
            spectrum_mags: &[f64],
            bin_width: f64,
        ) -> (f64, f64) {
            let (mut sum_weighted_freq, mut sum_magnitudes) = (0.0, 0.0);
            for (i, &mag) in spectrum_mags.iter().enumerate() {
                sum_weighted_freq += i as f64 * bin_width * mag;
                sum_magnitudes += mag;
            }
            (sum_weighted_freq, sum_magnitudes)
        }

        // scalar fallback for calculating band energy
        #[allow(dead_code)] // this function is used conditionally
        pub fn calculate_band_energy_f64(
            spectrum_mags: &[f64],
            sample_rate: u32,
            low_hz: f64,
            high_hz: f64,
        ) -> f64 {
            if spectrum_mags.is_empty() || sample_rate == 0 || low_hz >= high_hz {
                return 0.0; // early return for invalid inputs
            }

            let nyquist_freq = sample_rate as f64 / 2.0;
            let bin_width = nyquist_freq / spectrum_mags.len() as f64;
            let start_bin = (low_hz / bin_width).floor().max(0.0) as usize;
            let end_bin = (high_hz / bin_width).ceil() as usize;

            if end_bin <= start_bin || start_bin >= spectrum_mags.len() {
                return 0.0;
            }

            spectrum_mags[start_bin..=end_bin.min(spectrum_mags.len() - 1)]
                .iter()
                .map(|&mag| mag * mag)
                .sum()
        }
    }

    // public api for the audio_simd module
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64"))]
    pub use self::platform_impl::{
        calculate_band_energy_simd as calculate_band_energy_f64,
        calculate_spectral_centroid_f64, mix_stereo_to_mono_f32, sum_squares_f32,
    };

    // provide scalar fallback if no simd architecture is targeted
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64")))]
    pub use self::scalar_fallback::{
        calculate_band_energy_f64, calculate_spectral_centroid_f64, mix_stereo_to_mono_f32, sum_squares_f32,
    };
}

// --- image simd module ---
mod image_simd {
    #[allow(unused_imports)] // keep this import, it's used by Rgb<u8>
    use image::Rgb; 

    // define prefetch strategy for hardware-specific memory prefetching
    #[allow(dead_code)] // variants L2, L3, NonTemporal are not currently used
    pub enum PrefetchStrategy {
        L1,
        #[allow(dead_code)]
        L2,
        #[allow(dead_code)]
        L3,
        #[allow(dead_code)]
        NonTemporal,
    }

    // inline always to encourage inlining for performance-critical prefetching
    #[inline(always)]
    pub fn prefetch_data<T>(_ptr: *const T, _offset: usize, strategy: PrefetchStrategy) { // Added underscores to unused variables
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            match strategy {
                PrefetchStrategy::L1 => _mm_prefetch(_ptr.add(_offset) as *const i8, _MM_HINT_T0),
                PrefetchStrategy::L2 => _mm_prefetch(_ptr.add(_offset) as *const i8, _MM_HINT_T1),
                PrefetchStrategy::L3 => _mm_prefetch(_ptr.add(_offset) as *const i8, _MM_HINT_T2),
                PrefetchStrategy::NonTemporal => _mm_prefetch(_ptr.add(_offset) as *const i8, _MM_HINT_NTA),
            }
        }
        // aarch64 prefetch instructions (prfm) could be added here
        #[cfg(target_arch = "aarch64")]
        { // removed unnecessary unsafe block
            #[allow(unused_imports)] // this import is conditionally used
            use std::arch::aarch64::*;
            // prfm instructions are typically like prfm pldl1keep, [x0, #offset]
            // but direct Rust intrinsics for these are not as straightforward as x86.
            // this would require inline assembly or specific compiler support.
            // for now, we'll keep it as a placeholder.
            match strategy {
                PrefetchStrategy::L1 => { /* prfm pldl1keep, [ptr.add(offset)] */ },
                PrefetchStrategy::L2 => { /* prfm pldl2keep, [ptr.add(offset)] */ },
                PrefetchStrategy::L3 => { /* prfm pldl3keep, [ptr.add(offset)] */ },
                PrefetchStrategy::NonTemporal => { /* prfm pldl1strm, [ptr.add(offset)] */ },
            }
        }
        // riscv64 prefetch instructions (pref) could be added here
        #[cfg(target_arch = "riscv64")]
        unsafe {
            // riscv has 'pref' instruction, but no direct rust intrinsic.
            // would require inline assembly.
            match strategy {
                PrefetchStrategy::L1 => { /* pref 0, (ptr.add(offset)) */ },
                PrefetchStrategy::L2 => { /* pref 1, (ptr.add(offset)) */ },
                PrefetchStrategy::L3 => { /* pref 2, (ptr.add(offset)) */ },
                PrefetchStrategy::NonTemporal => { /* pref 3, (ptr.add(offset)) */ },
            }
        }
    }


    #[cfg(target_arch = "x86_64")]
    mod platform_impl {
        use std::arch::x86_64::*;
        use image::Rgb;

        // converts hsv to rgb for a batch of f64 values using avx2 intrinsics
        #[target_feature(enable = "avx2")]
        #[unsafe(link_section = ".text.hot")] // instruction cache optimization
        pub unsafe fn hsv_to_rgb_batch_simd(h_batch: &[f64], s_batch: &[f64], v_batch: &[f64]) -> Vec<(f64, f64, f64)> {
            let mut result = Vec::with_capacity(h_batch.len());
            let chunks = h_batch.chunks_exact(4);
            let remainder_start = chunks.len() * 4;

            for (i, h_chunk) in chunks.enumerate() {
                let s_chunk = &s_batch[i*4..(i+1)*4];
                let v_chunk = &v_batch[i*4..(i+1)*4];
                
                let zero_vec = _mm256_setzero_pd();
                let one_vec = _mm256_set1_pd(1.0);
                let two_vec = _mm256_set1_pd(2.0);
                let sixty_vec = _mm256_set1_pd(60.0);
                let three_sixty_vec = _mm256_set1_pd(360.0);
                let neg_zero_vec = _mm256_set1_pd(-0.0); // for abs value using andnot

                // load h, s, v values
                let h_vec = _mm256_loadu_pd(h_chunk.as_ptr());
                let s_vec = _mm256_loadu_pd(s_chunk.as_ptr());
                let v_vec = _mm256_loadu_pd(v_chunk.as_ptr());

                // normalize h to [0, 360)
                let h_norm = _mm256_sub_pd(
                    h_vec,
                    _mm256_mul_pd(
                        _mm256_floor_pd(_mm256_div_pd(h_vec, three_sixty_vec)),
                        three_sixty_vec
                    )
                );

                // calculate h_prime = h / 60.0
                let h_prime = _mm256_div_pd(h_norm, sixty_vec);

                // calculate c = v * s
                let c = _mm256_mul_pd(v_vec, s_vec);

                // calculate x = c * (1 - |h_prime % 2 - 1|)
                let h_prime_mod2 = _mm256_sub_pd(
                    h_prime,
                    _mm256_mul_pd(
                        _mm256_floor_pd(_mm256_div_pd(h_prime, two_vec)),
                        two_vec
                    )
                );
                let h_prime_mod2_sub1 = _mm256_sub_pd(h_prime_mod2, one_vec);
                let x = _mm256_mul_pd(
                    c,
                    _mm256_sub_pd(
                        one_vec,
                        _mm256_andnot_pd(neg_zero_vec, h_prime_mod2_sub1) // abs value using andnot
                    )
                );

                // calculate m = v - c
                let m = _mm256_sub_pd(v_vec, c);

                // determine rgb components based on h_prime ranges using blend/mask
                let mut r = zero_vec;
                let mut g = zero_vec;
                let mut b = zero_vec;

                // h_prime in [0, 1)
                let mask0 = _mm256_and_pd(_mm256_cmp_pd(h_prime, zero_vec, _CMP_GE_OQ), _mm256_cmp_pd(h_prime, one_vec, _CMP_LT_OQ));
                r = _mm256_blendv_pd(r, c, mask0);
                g = _mm256_blendv_pd(g, x, mask0);

                // h_prime in [1, 2)
                let mask1 = _mm256_and_pd(_mm256_cmp_pd(h_prime, one_vec, _CMP_GE_OQ), _mm256_cmp_pd(h_prime, two_vec, _CMP_LT_OQ));
                r = _mm256_blendv_pd(r, x, mask1);
                g = _mm256_blendv_pd(g, c, mask1);

                // h_prime in [2, 3)
                let mask2 = _mm256_and_pd(_mm256_cmp_pd(h_prime, two_vec, _CMP_GE_OQ), _mm256_cmp_pd(h_prime, _mm256_set1_pd(3.0), _CMP_LT_OQ));
                g = _mm256_blendv_pd(g, c, mask2);
                b = _mm256_blendv_pd(b, x, mask2);

                // h_prime in [3, 4)
                let mask3 = _mm256_and_pd(_mm256_cmp_pd(h_prime, _mm256_set1_pd(3.0), _CMP_GE_OQ), _mm256_cmp_pd(h_prime, _mm256_set1_pd(4.0), _CMP_LT_OQ));
                g = _mm256_blendv_pd(g, x, mask3);
                b = _mm256_blendv_pd(b, c, mask3);

                // h_prime in [4, 5)
                let mask4 = _mm256_and_pd(_mm256_cmp_pd(h_prime, _mm256_set1_pd(4.0), _CMP_GE_OQ), _mm256_cmp_pd(h_prime, _mm256_set1_pd(5.0), _CMP_LT_OQ));
                r = _mm256_blendv_pd(r, x, mask4);
                b = _mm256_blendv_pd(b, c, mask4);

                // h_prime in [5, 6)
                let mask5 = _mm256_and_pd(_mm256_cmp_pd(h_prime, _mm256_set1_pd(5.0), _CMP_GE_OQ), _mm256_cmp_pd(h_prime, _mm256_set1_pd(6.0), _CMP_LT_OQ));
                r = _mm256_blendv_pd(r, c, mask5);
                b = _mm256_blendv_pd(b, x, mask5);

                // add m
                let final_r = _mm256_add_pd(r, m);
                let final_g = _mm256_add_pd(g, m);
                let final_b = _mm256_add_pd(b, m);

                let mut r_arr = [0.0f64; 4];
                let mut g_arr = [0.0f64; 4];
                let mut b_arr = [0.0f64; 4];

                _mm256_storeu_pd(r_arr.as_mut_ptr(), final_r);
                _mm256_storeu_pd(g_arr.as_mut_ptr(), final_g);
                _mm256_storeu_pd(b_arr.as_mut_ptr(), final_b);

                for j in 0..4 {
                    result.push((r_arr[j], g_arr[j], b_arr[j]));
                }
            }

            // process remainder
            for i in remainder_start..h_batch.len() {
                result.push(super::scalar_fallback::hsv_to_rgb(h_batch[i], s_batch[i], v_batch[i]));
            }

            result
        }

        // blends two sets of pixels using avx512 intrinsics
        #[target_feature(enable = "avx512f", enable = "avx512dq", enable = "avx512vl", enable = "avx512bw")]
        #[unsafe(link_section = ".text.hot")] // instruction cache optimization
        pub unsafe fn blend_pixels_batch_avx512(pixels1: &[Rgb<u8>], pixels2: &[Rgb<u8>], alpha: f32) -> Vec<Rgb<u8>> {
            let chunks = pixels1.chunks_exact(16); // process 16 pixels at once for avx512
            let remainder = chunks.remainder();
            let mut result = Vec::with_capacity(pixels1.len());

            // capture chunks.len() before it's moved
            let chunks_len = chunks.len();

            for (i, chunk1) in chunks.enumerate() {
                let chunk2 = &pixels2[i*16..(i+1)*16];

                // load 16 pixels (48 bytes) into f32 vectors
                let mut r1_f = [0.0f32; 16];
                let mut g1_f = [0.0f32; 16];
                let mut b1_f = [0.0f32; 16];
                let mut r2_f = [0.0f32; 16];
                let mut g2_f = [0.0f32; 16];
                let mut b2_f = [0.0f32; 16];

                for j in 0..16 {
                    r1_f[j] = chunk1[j][0] as f32;
                    g1_f[j] = chunk1[j][0] as f32;
                    b1_f[j] = chunk1[j][0] as f32;
                    r2_f[j] = chunk2[j][0] as f32;
                    g2_f[j] = chunk2[j][0] as f32;
                    b2_f[j] = chunk2[j][0] as f32;
                }
                
                let alpha_vec = _mm512_set1_ps(alpha);
                let inv_alpha_vec = _mm512_set1_ps(1.0 - alpha);
                let zero_f32_vec = _mm512_set1_ps(0.0);
                let max_u8_f32_vec = _mm512_set1_ps(255.0);
                
                let r1_vec = _mm512_loadu_ps(r1_f.as_ptr());
                let g1_vec = _mm512_loadu_ps(g1_f.as_ptr());
                let b1_vec = _mm512_loadu_ps(b1_f.as_ptr());

                let r2_vec = _mm512_loadu_ps(r2_f.as_ptr());
                let g2_vec = _mm512_loadu_ps(g2_f.as_ptr());
                let b2_vec = _mm512_loadu_ps(b2_f.as_ptr());

                // blend using fma: r1*alpha + r2*(1-alpha)
                let r_blended = _mm512_fmadd_ps(r1_vec, alpha_vec, _mm512_mul_ps(r2_vec, inv_alpha_vec));
                let g_blended = _mm512_fmadd_ps(g1_vec, alpha_vec, _mm512_mul_ps(g2_vec, inv_alpha_vec));
                let b_blended = _mm512_fmadd_ps(b1_vec, alpha_vec, _mm512_mul_ps(b2_vec, inv_alpha_vec));

                // clamp to [0, 255] and convert to u8
                let r_clamped = _mm512_min_ps(_mm512_max_ps(r_blended, zero_f32_vec), max_u8_f32_vec);
                let g_clamped = _mm512_min_ps(_mm512_max_ps(g_blended, zero_f32_vec), max_u8_f32_vec);
                let b_clamped = _mm512_min_ps(_mm512_max_ps(b_blended, zero_f32_vec), max_u8_f32_vec);

                // convert f32 to i32
                let r_i32 = _mm512_cvtps_epi32(r_clamped); // __m512i (16 x i32)
                let g_i32 = _mm512_cvtps_epi32(g_clamped); // __m512i (16 x i32)
                let b_i32 = _mm512_cvtps_epi32(b_clamped); // __m512i (16 x i32)

                // Extract lower and upper 8 i32s as __m256i
                let r_i32_lo = _mm512_extracti64x4_epi64(r_i32, 0);
                let r_i32_hi = _mm512_extracti64x4_epi64(r_i32, 1);
                let g_i32_lo = _mm512_extracti64x4_epi64(g_i32, 0);
                let g_i32_hi = _mm512_extracti64x4_epi64(g_i32, 1);
                let b_i32_lo = _mm512_extracti64x4_epi64(b_i32, 0);
                let b_i32_hi = _mm512_extracti64x4_epi64(b_i32, 1);

                // Convert 8 i32s to 8 i16s (__m128i)
                let r_i16_lo = _mm256_cvtepi32_epi16(r_i32_lo);
                let r_i16_hi = _mm256_cvtepi32_epi16(r_i32_hi);
                let g_i16_lo = _mm256_cvtepi32_epi16(g_i32_lo);
                let g_i16_hi = _mm256_cvtepi32_epi16(g_i32_hi);
                let b_i16_lo = _mm256_cvtepi32_epi16(b_i32_lo);
                let b_i16_hi = _mm256_cvtepi32_epi16(b_i32_hi);

                // Convert 8 i16s to 8 i8s (__m128i)
                let r_u8_lo = _mm_cvtepi16_epi8(r_i16_lo);
                let r_u8_hi = _mm_cvtepi16_epi8(r_i16_hi);
                let g_u8_lo = _mm_cvtepi16_epi8(g_i16_lo);
                let g_u8_hi = _mm_cvtepi16_epi8(g_i16_hi);
                let b_u8_lo = _mm_cvtepi16_epi8(b_i16_lo);
                let b_u8_hi = _mm_cvtepi16_epi8(b_i16_hi);

                // Store the two __m128i results into the 16-element u8 array
                let mut r_arr = [0u8; 16];
                let mut g_arr = [0u8; 16];
                let mut b_arr = [0u8; 16];

                _mm_storeu_si128(r_arr.as_mut_ptr() as *mut __m128i, r_u8_lo);
                _mm_storeu_si128(r_arr.as_mut_ptr().add(8) as *mut __m128i, r_u8_hi);
                _mm_storeu_si128(g_arr.as_mut_ptr() as *mut __m128i, g_u8_lo);
                _mm_storeu_si128(b_arr.as_mut_ptr() as *mut __m128i, b_u8_lo);
                _mm_storeu_si128(b_arr.as_mut_ptr().add(8) as *mut __m128i, b_u8_hi);

                for j in 0..16 {
                    result.push(Rgb([r_arr[j], g_arr[j], b_arr[j]]));
                }
            }

            // process remainder
            for (p1, p2) in remainder.iter().zip(pixels2.iter().skip(chunks_len * 16)) {
                result.push(Rgb([
                    ((p1[0] as f32 * alpha + p2[0] as f32 * (1.0 - alpha)) as u8),
                    ((p1[1] as f32 * alpha + p2[1] as f32 * (1.0 - alpha)) as u8),
                    ((p1[2] as f32 * alpha + p2[2] as f32 * (1.0 - alpha)) as u8),
                ]));
            }

            result
        }


        // blends two sets of pixels using avx2 intrinsics
        #[target_feature(enable = "avx2", enable = "fma")]
        #[unsafe(link_section = ".text.hot")] // instruction cache optimization
        pub unsafe fn blend_pixels_batch_avx2(pixels1: &[Rgb<u8>], pixels2: &[Rgb<u8>], alpha: f32) -> Vec<Rgb<u8>> {
            let chunks = pixels1.chunks_exact(8); // process 8 pixels at once for avx2
            let remainder = chunks.remainder();
            let mut result = Vec::with_capacity(pixels1.len());

            // Capture chunks.len() before `chunks` is moved by `enumerate()`
            let chunks_len = chunks.len();

            for (i, chunk1) in chunks.enumerate() {
                let chunk2 = &pixels2[i*8..(i+1)*8];

                // manual extraction for 8 pixels (24 bytes) into f32 vectors
                let mut r1_f = [0.0f32; 8];
                let mut g1_f = [0.0f32; 8];
                let mut b1_f = [0.0f32; 8];
                let mut r2_f = [0.0f32; 8];
                let mut g2_f = [0.0f32; 8];
                let mut b2_f = [0.0f32; 8];

                for j in 0..8 {
                    r1_f[j] = chunk1[j][0] as f32;
                    g1_f[j] = chunk1[j][1] as f32;
                    b1_f[j] = chunk1[j][2] as f32;
                    r2_f[j] = chunk2[j][0] as f32;
                    g2_f[j] = chunk2[j][1] as f32;
                    b2_f[j] = chunk2[j][2] as f32;
                }

                let alpha_vec = _mm256_set1_ps(alpha);
                let inv_alpha_vec = _mm256_set1_ps(1.0 - alpha);
                let zero_f32_vec = _mm256_set1_ps(0.0);
                let max_u8_f32_vec = _mm256_set1_ps(255.0);
                
                let r1_vec = _mm256_loadu_ps(r1_f.as_ptr());
                let g1_vec = _mm256_loadu_ps(g1_f.as_ptr());
                let b1_vec = _mm256_loadu_ps(b1_f.as_ptr());

                let r2_vec = _mm256_loadu_ps(r2_f.as_ptr());
                let g2_vec = _mm256_loadu_ps(g2_f.as_ptr());
                let b2_vec = _mm256_loadu_ps(b2_f.as_ptr());

                // blend using fma: r1*alpha + r2*(1-alpha)
                let r_blended = _mm256_fmadd_ps(r1_vec, alpha_vec, _mm256_mul_ps(r2_vec, inv_alpha_vec));
                let g_blended = _mm256_fmadd_ps(g1_vec, alpha_vec, _mm256_mul_ps(g2_vec, inv_alpha_vec));
                let b_blended = _mm256_fmadd_ps(b1_vec, alpha_vec, _mm256_mul_ps(b2_vec, inv_alpha_vec));

                // clamp to [0, 255] and convert to u8
                let r_clamped = _mm256_min_ps(_mm256_max_ps(r_blended, zero_f32_vec), max_u8_f32_vec);
                let g_clamped = _mm256_min_ps(_mm256_max_ps(g_blended, zero_f32_vec), max_u8_f32_vec);
                let b_clamped = _mm256_min_ps(_mm256_max_ps(b_blended, zero_f32_vec), max_u8_f32_vec);

                // convert f32 to i32, then to u8
                let r_i32 = _mm256_cvtps_epi32(r_clamped);
                let g_i32 = _mm256_cvtps_epi32(g_clamped);
                let b_i32 = _mm256_cvtps_epi32(b_clamped);

                // extract and store
                let mut r_arr = [0i32; 8];
                let mut g_arr = [0i32; 8];
                let mut b_arr = [0i32; 8];

                _mm256_storeu_si256(r_arr.as_mut_ptr() as *mut __m256i, r_i32);
                _mm256_storeu_si256(g_arr.as_mut_ptr() as *mut __m256i, g_i32);
                _mm256_storeu_si256(b_arr.as_mut_ptr() as *mut __m256i, b_i32);

                for j in 0..8 {
                    result.push(Rgb([r_arr[j] as u8, g_arr[j] as u8, b_arr[j] as u8]));
                }
            }

            // process remainder
            for (p1, p2) in remainder.iter().zip(pixels2.iter().skip(chunks_len * 8)) {
                result.push(Rgb([
                    ((p1[0] as f32 * alpha + p2[0] as f32 * (1.0 - alpha)) as u8),
                    ((p1[1] as f32 * alpha + p2[1] as f32 * (1.0 - alpha)) as u8),
                    ((p1[2] as f32 * alpha + p2[2] as f32 * (1.0 - alpha)) as u8),
                ]));
            }

            result
        }
    }

    #[cfg(target_arch = "aarch64")]
    mod platform_impl {
        use std::arch::aarch64::*;
        use image::Rgb;

        // converts hsv to rgb for a batch of f64 values using NEON intrinsics
        #[target_feature(enable = "neon")]
        #[unsafe(link_section = ".text.hot")] // instruction cache optimization
        pub unsafe fn hsv_to_rgb_batch_simd(h_batch: &[f64], s_batch: &[f64], v_batch: &[f64]) -> Vec<(f64, f64, f64)> {
            let mut result = Vec::with_capacity(h_batch.len());
            let chunks = h_batch.chunks_exact(2); // NEON f64 vector is 2 elements
            let remainder_start = chunks.len() * 2;

            for (i, h_chunk) in chunks.enumerate() {
                let s_chunk = &s_batch[i*2..(i+1)*2];
                let v_chunk = &v_batch[i*2..(i+1)*2];

                let zero_vec = vdupq_n_f64(0.0);
                let one_vec = vdupq_n_f64(1.0);
                let two_vec = vdupq_n_f64(2.0);
                let three_vec = vdupq_n_f64(3.0);
                let four_vec = vdupq_n_f64(4.0);
                let five_vec = vdupq_n_f64(5.0);
                let six_vec = vdupq_n_f64(6.0);
                let sixty_vec = vdupq_n_f64(60.0);
                let three_sixty_vec = vdupq_n_f64(360.0);
                
                let h_vec = unsafe { vld1q_f64(h_chunk.as_ptr()) };
                let s_vec = unsafe { vld1q_f64(s_chunk.as_ptr()) };
                let v_vec = unsafe { vld1q_f64(v_chunk.as_ptr()) };

                let h_norm = vsubq_f64(
                    h_vec,
                    vmulq_f64(
                        vcvtq_f64_s64(vcvtaq_s64_f64(vdivq_f64(h_vec, three_sixty_vec))),
                        three_sixty_vec
                    )
                );

                let h_prime = vdivq_f64(h_norm, sixty_vec);
                let c = vmulq_f64(v_vec, s_vec);

                let h_prime_mod2 = vsubq_f64(
                    h_prime,
                    vmulq_f64(
                        vcvtq_f64_s64(vcvtaq_s64_f64(vdivq_f64(h_prime, two_vec))),
                        two_vec
                    )
                );
                let h_prime_mod2_sub1 = vsubq_f64(h_prime_mod2, one_vec);
                let x = vmulq_f64(
                    c,
                    vsubq_f64(
                        one_vec,
                        vabsq_f64(h_prime_mod2_sub1)
                    )
                );

                let m = vsubq_f64(v_vec, c);

                // h_prime comparison masks
                let mask0 = vcltq_f64(h_prime, one_vec); // h_prime < 1.0
                let mask1 = vandq_u64(vcgeq_f64(h_prime, one_vec), vcltq_f64(h_prime, two_vec)); // 1.0 <= h_prime < 2.0
                let mask2 = vandq_u64(vcgeq_f64(h_prime, two_vec), vcltq_f64(h_prime, three_vec)); // 2.0 <= h_prime < 3.0
                let mask3 = vandq_u64(vcgeq_f64(h_prime, three_vec), vcltq_f64(h_prime, four_vec)); // 3.0 <= h_prime < 4.0
                let mask4 = vandq_u64(vcgeq_f64(h_prime, four_vec), vcltq_f64(h_prime, five_vec)); // 4.0 <= h_prime < 5.0
                let mask5 = vandq_u64(vcgeq_f64(h_prime, five_vec), vcltq_f64(h_prime, six_vec)); // 5.0 <= h_prime < 6.0

                // initialize r, g, b components
                let mut r_comp = zero_vec;
                let mut g_comp = zero_vec;
                let mut b_comp = zero_vec;

                // select components based on masks
                r_comp = vbslq_f64(mask0, c, r_comp);
                g_comp = vbslq_f64(mask0, x, g_comp);

                r_comp = vbslq_f64(mask1, x, r_comp);
                g_comp = vbslq_f64(mask1, c, g_comp);

                g_comp = vbslq_f64(mask2, c, g_comp);
                b_comp = vbslq_f64(mask2, x, b_comp);

                g_comp = vbslq_f64(mask3, x, g_comp);
                b_comp = vbslq_f64(mask3, c, b_comp);

                r_comp = vbslq_f64(mask4, x, r_comp);
                b_comp = vbslq_f64(mask4, c, b_comp);

                r_comp = vbslq_f64(mask5, c, r_comp);
                b_comp = vbslq_f64(mask5, x, b_comp);

                // add m to all components
                let final_r = vaddq_f64(r_comp, m);
                let final_g = vaddq_f64(g_comp, m);
                let final_b = vaddq_f64(b_comp, m);

                let mut r_arr = [0.0f64; 2];
                let mut g_arr = [0.0f64; 2];
                let mut b_arr = [0.0f64; 2];

                unsafe { vst1q_f64(r_arr.as_mut_ptr(), final_r) };
                unsafe { vst1q_f64(g_arr.as_mut_ptr(), final_g) };
                unsafe { vst1q_f64(b_arr.as_mut_ptr(), final_b) };

                for j in 0..2 {
                    result.push((r_arr[j], g_arr[j], b_arr[j]));
                }
            }
            // process remainder
            for i in remainder_start..h_batch.len() {
                result.push(super::scalar_fallback::hsv_to_rgb(h_batch[i], s_batch[i], v_batch[i]));
            }
            result
        }

        // blends two sets of pixels using NEON intrinsics
        #[target_feature(enable = "neon")]
        #[unsafe(link_section = ".text.hot")] // instruction cache optimization
        pub unsafe fn blend_pixels_batch_simd(pixels1: &[Rgb<u8>], pixels2: &[Rgb<u8>], alpha: f32) -> Vec<Rgb<u8>> {
            let chunks = pixels1.chunks_exact(4); // 4 pixels = 12 bytes, can use 4 f32 lanes
            let remainder = chunks.remainder();
            let mut result = Vec::with_capacity(pixels1.len());

            // Capture chunks length before it's consumed by `into_iter().enumerate()`
            let initial_chunks_len = chunks.len();

            for (i, chunk1) in chunks.clone().enumerate() { // Clone chunks to allow subsequent use
                let chunk2 = &pixels2[i*4..(i+1)*4];

                // load and convert u8 to f32
                let mut r1_f = [0.0f32; 4];
                let mut g1_f = [0.0f32; 4];
                let mut b1_f = [0.0f32; 4];
                let mut r2_f = [0.0f32; 4];
                let mut g2_f = [0.0f32; 4];
                let mut b2_f = [0.0f32; 4];

                for j in 0..4 {
                    r1_f[j] = chunk1[j][0] as f32;
                    g1_f[j] = chunk1[j][1] as f32;
                    b1_f[j] = chunk1[j][2] as f32;
                    r2_f[j] = chunk2[j][0] as f32;
                    g2_f[j] = chunk2[j][1] as f32;
                    b2_f[j] = chunk2[j][2] as f32;
                }

                let alpha_vec = vdupq_n_f32(alpha);
                let inv_alpha_vec = vdupq_n_f32(1.0 - alpha);
                let max_val_vec = vdupq_n_f32(255.0);
                let zero_val_vec = vdupq_n_f32(0.0);
                
                let r1_vec = unsafe { vld1q_f32(r1_f.as_ptr()) };
                let g1_vec = unsafe { vld1q_f32(g1_f.as_ptr()) };
                let b1_vec = unsafe { vld1q_f32(b1_f.as_ptr()) };
                let r2_vec = unsafe { vld1q_f32(r2_f.as_ptr()) };
                let g2_vec = unsafe { vld1q_f32(g2_f.as_ptr()) };
                let b2_vec = unsafe { vld1q_f32(b2_f.as_ptr()) };

                // blend: p1 * alpha + p2 * (1-alpha)
                let r_result_f = vaddq_f32(vmulq_f32(r1_vec, alpha_vec), vmulq_f32(r2_vec, inv_alpha_vec));
                let g_result_f = vaddq_f32(vmulq_f32(g1_vec, alpha_vec), vmulq_f32(g2_vec, inv_alpha_vec));
                let b_result_f = vaddq_f32(vmulq_f32(b1_vec, alpha_vec), vmulq_f32(b2_vec, inv_alpha_vec));

                // clamp and convert back to u8
                let mut r_arr = [0.0f32; 4];
                let mut g_arr = [0.0f32; 4];
                let mut b_arr = [0.0f32; 4];

                unsafe { vst1q_f32(r_arr.as_mut_ptr(), vminq_f32(vmaxq_f32(r_result_f, zero_val_vec), max_val_vec)) };
                unsafe { vst1q_f32(g_arr.as_mut_ptr(), vminq_f32(vmaxq_f32(g_result_f, zero_val_vec), max_val_vec)) };
                unsafe { vst1q_f32(b_arr.as_mut_ptr(), vminq_f32(vmaxq_f32(b_result_f, zero_val_vec), max_val_vec)) };

                for j in 0..4 {
                    result.push(Rgb([r_arr[j] as u8, g_arr[j] as u8, b_arr[j] as u8]));
                }
            }

            // process remainder
            for (p1, p2) in remainder.iter().zip(pixels2.iter().skip(initial_chunks_len * 4)) { // Use initial_chunks_len
                result.push(Rgb([
                    ((p1[0] as f32 * alpha + p2[0] as f32 * (1.0 - alpha)) as u8),
                    ((p1[1] as f32 * alpha + p2[1] as f32 * (1.0 - alpha)) as u8),
                    ((p1[2] as f32 * alpha + p2[2] as f32 * (1.0 - alpha)) as u8),
                ]));
            }

            result
        }
    }

    // fallback for other architectures or when simd is disabled.
    mod scalar_fallback {
        use image::Rgb;

        // scalar fallback for hsv to rgb conversion
        pub fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
            let h = h.rem_euclid(360.0); // ensures h is always in [0, 360)
            let c = v * s;
            let h_prime = h / 60.0;
            let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
            let m = v - c;

            let (r, g, b) = if (0.0..1.0).contains(&h_prime) {
                (c, x, 0.0)
            } else if (1.0..2.0).contains(&h_prime) {
                (x, c, 0.0)
            } else if (2.0..3.0).contains(&h_prime) {
                (0.0, c, x)
            } else if (3.0..4.0).contains(&h_prime) {
                (0.0, x, c)
            } else if (4.0..5.0).contains(&h_prime) {
                (x, 0.0, c)
            } else if (5.0..6.0).contains(&h_prime) {
                (c, 0.0, x)
            } else {
                (0.0, 0.0, 0.0)
            };
            (r + m, g + m, b + m)
        }

        // scalar fallback for pixel blending
        #[allow(dead_code)] // this function is used conditionally
        pub fn blend_pixels(p1: &Rgb<u8>, p2: &Rgb<u8>, alpha: f32) -> Rgb<u8> {
            Rgb([
                ((p1[0] as f32 * alpha + p2[0] as f32 * (1.0 - alpha)) as u8),
                ((p1[1] as f32 * alpha + p2[1] as f32 * (1.0 - alpha)) as u8),
                ((p1[2] as f32 * alpha + p2[2] as f32 * (1.0 - alpha)) as u8),
            ])
        }

        // scalar fallback for hsv to rgb batch conversion
        #[allow(dead_code)] // this function is used conditionally
        pub fn hsv_to_rgb_batch(h_batch: &[f64], s_batch: &[f64], v_batch: &[f64]) -> Vec<(f64, f64, f64)> {
            h_batch.iter().zip(s_batch).zip(v_batch)
                .map(|((&h, &s), &v)| hsv_to_rgb(h, s, v))
                .collect()
        }

        // scalar fallback for pixel blending batch
        #[allow(dead_code)] // this function is used conditionally
        pub fn blend_pixels_batch(pixels1: &[Rgb<u8>], pixels2: &[Rgb<u8>], alpha: f32) -> Vec<Rgb<u8>> {
            pixels1.iter().zip(pixels2).map(|(p1, p2)| blend_pixels(p1, p2, alpha)).collect()
        }
    }

    // public api for the image_simd module, dispatching based on detected features
    #[cfg(target_arch = "x86_64")]
    pub mod dispatch_impl {
        use super::platform_impl;
        use image::Rgb;

        // hsv_to_rgb_batch dispatch
        pub fn hsv_to_rgb_batch(h_batch: &[f64], s_batch: &[f64], v_batch: &[f64]) -> Vec<(f64, f64, f64)> {
            if is_x86_feature_detected!("avx2") {
                unsafe { platform_impl::hsv_to_rgb_batch_simd(h_batch, s_batch, v_batch) }
            } else {
                super::scalar_fallback::hsv_to_rgb_batch(h_batch, s_batch, v_batch)
            }
        }

        // blend_pixels_batch dispatch
        pub fn blend_pixels_batch(pixels1: &[Rgb<u8>], pixels2: &[Rgb<u8>], alpha: f32) -> Vec<Rgb<u8>> {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
                unsafe { platform_impl::blend_pixels_batch_avx512(pixels1, pixels2, alpha) }
            } else if is_x86_feature_detected!("avx2") {
                unsafe { platform_impl::blend_pixels_batch_avx2(pixels1, pixels2, alpha) }
            } else {
                super::scalar_fallback::blend_pixels_batch(pixels1, pixels2, alpha)
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub mod dispatch_impl {
        use super::platform_impl;
        use image::Rgb;

        pub fn hsv_to_rgb_batch(h_batch: &[f64], s_batch: &[f64], v_batch: &[f64]) -> Vec<(f64, f64, f64)> {
            // assuming neon is generally available on aarch64, or use feature detection if specific neon extensions are needed
            unsafe { platform_impl::hsv_to_rgb_batch_simd(h_batch, s_batch, v_batch) }
        }

        pub fn blend_pixels_batch(pixels1: &[Rgb<u8>], pixels2: &[Rgb<u8>], alpha: f32) -> Vec<Rgb<u8>> {
            unsafe { platform_impl::blend_pixels_batch_simd(pixels1, pixels2, alpha) }
        }
    }


    // provide scalar fallback if no simd architecture is targeted or for other architectures
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub use self::scalar_fallback::{hsv_to_rgb_batch, blend_pixels_batch};

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    pub use self::dispatch_impl::{hsv_to_rgb_batch, blend_pixels_batch};

    // expose scalar hsv_to_rgb for single calls
    // Removed: pub use self::scalar_fallback::hsv_to_rgb;
    // Removed: pub use self::scalar_fallback::blend_pixels;
}

// --- fast approximate math functions ---

// fast approximate sine - much faster than sin() for visualization
#[inline(always)]
fn fast_sin(x: f64) -> f64 {
    // bhaskara i's approximation (within 2% error)
    let x = x % (2.0 * std::f64::consts::PI);
    let y = if x > std::f64::consts::PI {
        x - 2.0 * std::f64::consts::PI
    } else {
        x
    };

    (16.0 * y * (std::f64::consts::PI - y)) /
    (5.0 * std::f64::consts::PI * std::f64::consts::PI - 4.0 * y * (std::f64::consts::PI - y))
}


// --- quantum noise implementation ---

// this struct ensures that the permutation table is aligned to a cache line boundary (64 bytes)
#[derive(Clone)]
#[repr(align(64))]
struct AlignedPermutationTable {
    p: [u8; 512],
}

impl AlignedPermutationTable {
    // creates a new aligned permutation table with a given seed
    fn new(seed: u32) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
        let mut p_temp = [0u8; 256];

        // fill first 256 entries with 0..256
        for i in 0..256 {
            p_temp[i] = i as u8;
        }

        // shuffle the first 256 entries
        for i in (1..256).rev() {
            let j = rng.random_range(0..=i);
            p_temp.swap(i, j);
        }

        // duplicate for wrap-around
        let mut p = [0u8; 512];
        p[0..256].copy_from_slice(&p_temp);
        p[256..512].copy_from_slice(&p_temp);

        Self { p }
    }

    // gets a value from the permutation table without bounds checking
    #[inline(always)]
    fn get(&self, idx: usize) -> u8 {
        // safety: no bounds checking needed due to duplicated data and mask (idx & 255)
        unsafe { *self.p.get_unchecked(idx & 255) }
    }
}

#[derive(Clone)]
#[repr(align(64))] // add cache-line alignment for critical structs
pub struct PerlinNoise {
    #[allow(dead_code)]
    seed: u32,
    p: AlignedPermutationTable, // use the aligned permutation table
}

impl PerlinNoise {
    // creates a new perlin noise generator with a given seed
    fn new(seed: u32) -> Self {
        PerlinNoise {
            seed,
            p: AlignedPermutationTable::new(seed),
        }
    }

    // fade function for perlin noise
    #[inline(always)]
    fn fade(t: f64) -> f64 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    // linear interpolation function
    #[inline(always)]
    fn lerp(t: f64, a: f64, b: f64) -> f64 {
        a + t * (b - a)
    }

    // gradient function for perlin noise
    #[inline(always)]
    fn grad(hash: usize, x: f64, y: f64) -> f64 {
        match hash & 0xf {
            0x0 => x + y,
            0x1 => -x + y,
            0x2 => x - y,
            0x3 => -x - y,
            0x4 => x + x,
            0x5 => -x + x,
            0x6 => x - x,
            0x7 => -x - x,
            _ => x + y,
        }
    }

    // gets a single perlin noise value for given coordinates
    fn get(&self, x: f64, y: f64) -> f64 {
        let x_int = x.floor() as usize;
        let y_int = y.floor() as usize;

        let x_frac = x - x.floor();
        let y_frac = y - y.floor();

        let u = Self::fade(x_frac);
        let v = Self::fade(y_frac);

        // use aligned permutation table's get method
        let a = self.p.get(x_int) as usize + self.p.get(y_int) as usize;
        let b = self.p.get(x_int + 1) as usize + self.p.get(y_int) as usize;

        let n00 = Self::grad(self.p.get(a) as usize, x_frac, y_frac);
        let n10 = Self::grad(self.p.get(b) as usize, x_frac - 1.0, y_frac);
        let n01 = Self::grad(self.p.get(a + 1) as usize, x_frac, y_frac - 1.0);
        let n11 = Self::grad(self.p.get(b + 1) as usize, x_frac - 1.0, y_frac - 1.0);

        Self::lerp(v, Self::lerp(u, n00, n10), Self::lerp(u, n01, n11))
    }

    // gets a batch of perlin noise values, using simd if available
    fn get_batch(&self, coords: &[(f64, f64)]) -> Vec<f64> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { self.get_batch_simd(coords) }
            } else {
                coords
                    .par_iter()
                    .map(|(x, y)| self.get(*x, *y))
                    .collect()
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            coords
                .par_iter()
                .map(|(x, y)| self.get(*x, *y))
                .collect()
        }
    }

    // vectorized perlin noise batch generation using avx2
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn get_batch_simd(&self, coords: &[(f64, f64)]) -> Vec<f64> {
        use std::arch::x86_64::*;

        let chunks = coords.chunks_exact(4);
        let remainder = chunks.remainder();
        let mut result = Vec::with_capacity(coords.len());

        for chunk in chunks {
            let mut current_results = [0.0f64; 4];
            let six_vec = _mm256_set1_pd(6.0);
            let fifteen_vec = _mm256_set1_pd(15.0);
            let ten_vec = _mm256_set1_pd(10.0);
            let _one_vec = _mm256_set1_pd(1.0);

            // load x/y coordinates in correct order for _mm256_set_pd (3,2,1,0)
            let x_vec = _mm256_set_pd(chunk[3].0, chunk[2].0, chunk[1].0, chunk[0].0);
            let y_vec = _mm256_set_pd(chunk[3].1, chunk[2].1, chunk[1].1, chunk[0].1);

            // get integer and fractional parts
            let x_floor = _mm256_floor_pd(x_vec);
            let y_floor = _mm256_floor_pd(y_vec);
            let x_frac = _mm256_sub_pd(x_vec, x_floor);
            let y_frac = _mm256_sub_pd(y_vec, y_floor);

            // calculate fade(x) and fade(y) using polynomial: t^3 * (t * (t * 6 - 15) + 10)
            let x_frac_sq = _mm256_mul_pd(x_frac, x_frac);
            let x_frac_cube = _mm256_mul_pd(x_frac_sq, x_frac);
            let x_fade_term = _mm256_fmadd_pd(
                x_frac,
                _mm256_fmsub_pd(x_frac, six_vec, fifteen_vec),
                ten_vec
            );
            let u_vec = _mm256_mul_pd(x_frac_cube, x_fade_term);

            let y_frac_sq = _mm256_mul_pd(y_frac, y_frac);
            let y_frac_cube = _mm256_mul_pd(y_frac_sq, y_frac);
            let y_fade_term = _mm256_fmadd_pd(
                y_frac,
                _mm256_fmsub_pd(y_frac, six_vec, fifteen_vec),
                ten_vec
            );
            let v_vec = _mm256_mul_pd(y_frac_cube, y_fade_term);

            // for grad and lerp, we need to extract individual elements due to complex conditional logic
            // and table lookups. for a truly vectorized perlin, the gradient vectors and permutation
            // table lookups would need to be vectorized, which is highly complex for generic perlin.
            // here, we fall back to scalar grad and lerp for each of the 4 elements.
            let mut x_frac_arr = [0.0f64; 4];
            let mut y_frac_arr = [0.0f64; 4];
            let mut x_floor_arr = [0.0f64; 4];
            let mut y_floor_arr = [0.0f64; 4];

            _mm256_storeu_pd(x_frac_arr.as_mut_ptr(), x_frac);
            _mm256_storeu_pd(y_frac_arr.as_mut_ptr(), y_frac);
            _mm256_storeu_pd(x_floor_arr.as_mut_ptr(), x_floor);
            _mm256_storeu_pd(y_floor_arr.as_mut_ptr(), y_floor);

            let mut u_arr = [0.0f64; 4];
            let mut v_arr = [0.0f64; 4];
            _mm256_storeu_pd(u_arr.as_mut_ptr(), u_vec);
            _mm256_storeu_pd(v_arr.as_mut_ptr(), v_vec);

            for j in 0..4 {
                let x_int_j = x_floor_arr[j] as usize;
                let y_int_j = y_floor_arr[j] as usize;

                // use aligned permutation table's get method
                let a = self.p.get(x_int_j) as usize + self.p.get(y_int_j) as usize;
                let b = self.p.get(x_int_j + 1) as usize + self.p.get(y_int_j) as usize;

                let n00 = Self::grad(self.p.get(a) as usize, x_frac_arr[j], y_frac_arr[j]);
                let n10 = Self::grad(self.p.get(b) as usize, x_frac_arr[j] - 1.0, y_frac_arr[j]);
                let n01 = Self::grad(self.p.get(a + 1) as usize, x_frac_arr[j], y_frac_arr[j] - 1.0);
                let n11 = Self::grad(self.p.get(b + 1) as usize, x_frac_arr[j] - 1.0, y_frac_arr[j] - 1.0);

                current_results[j] = Self::lerp(v_arr[j], Self::lerp(u_arr[j], n00, n10), Self::lerp(u_arr[j], n01, n11));
            }
            result.extend_from_slice(&current_results);
        }

        // process remainder
        for &(x, y) in remainder {
            result.push(self.get(x, y));
        }

        result
    }
}

// --- public data structures ---

// arguments for the `visual` subcommand (video visualization).
#[derive(Debug, Parser)]
pub struct VisualArgs {
    #[arg(long, default_value = "1920x1080")]
    pub resolution: String,

    #[arg(long, default_value_t = 60)]
    pub fps: u32,

    #[arg(long)]
    pub ltr: bool,

    #[arg(long)]
    pub rtl: bool,

    #[arg(long)]
    pub bench: bool,

    // extra ffmpeg flags (e.g., -s, -r, -b:v, -pix_fmt, etc.)
    #[arg(long = "ffmpeg-flag", value_name = "FFMPEG_FLAG", num_args = 0.., action = clap::ArgAction::Append)]
    pub ffmpeg_flags: Vec<String>,

    #[arg(last = true, trailing_var_arg = true)]
    pub ffmpeg_args: Vec<String>,

    pub input: String,
    pub output: String,
}
// represents extracted audio features for a single frame.
#[derive(Debug, Default, Clone)]
#[repr(align(64))] // cache-line alignment
pub struct AudioFeatures {
    pub rms_loudness: f64,
    pub spectral_centroid: f64,
    pub low_freq_energy: f64,
    pub mid_freq_energy: f64,
    pub high_freq_energy: f64,
    pub spectrum: Vec<f64>,
    pub peak_frequency: f64,
    pub spectral_rolloff: f64,
    pub transient_strength: f64,
}

// represents the processed quantum state data and derived visual parameters.
#[derive(Debug, Default, Clone)]
#[repr(align(64))] // cache-line alignment
pub struct QuantumVisualData {
    pub base_brightness: f64,
    pub color_hue: f64,
    pub pattern_density: f64,
    pub distortion_magnitude: f64,
    pub flicker_intensity: f64,
    pub noise_seed: u64,
    pub chaos_factor: f64,
    pub interference_pattern: f64,
    pub quantum_entanglement: f64,
    #[allow(dead_code)] // this field is currently not read after being populated
    pub quantum_measurements: Vec<bool>,
    pub quantum_coherence: f64,
    pub flow_field_strength: f64,
    pub depth_modulation: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumDirection {
    None,
    Ltr,
    Rtl,
}

// --- quantum noise generation ---

pub struct QuantumNoiseGenerator {
    pub quantum_state: Arc<Mutex<QuantumState>>, // using parking_lot::mutex
    pub coherence_time: usize,
    pub current_time: AtomicU64, // use atomic for counter
}

impl QuantumNoiseGenerator {
    // creates a new quantum noise generator
    pub fn new(n_qubits: usize, _seed: u64) -> Self {
        Self {
            quantum_state: Arc::new(Mutex::new(QuantumState::new(
                n_qubits,
                Some(NoiseConfig::Random),
            ))),
            coherence_time: 30,
            current_time: AtomicU64::new(0), // initialize atomic counter
        }
    }

    // generates quantum noise for a given time
    pub fn quantum_noise(&mut self, time: f64) -> (f64, Vec<bool>) {
        let mut quantum_state = self.quantum_state.lock(); // parking_lot mutex lock
        let current_time_val = self.current_time.fetch_add(1, Ordering::SeqCst) as usize; // atomic increment

        if current_time_val % self.coherence_time == 0 {
            *quantum_state = QuantumState::new(quantum_state.n, Some(NoiseConfig::Random));
        }

        let mut measurements = Vec::with_capacity(quantum_state.n);
        let n_qubits = quantum_state.n;
        for q in 0..n_qubits {
            match quantum_state.measure(q) {
                Ok(val) => measurements.push(val == 1),
                Err(e) => {
                    // handle the error case, e.g., log it and push a default value
                    eprintln!("error measuring qubit {}: {}", q, e);
                    measurements.push(false); // default to false on error
                }
            }
        }
        let quantum_bits = measurements
            .iter()
            .enumerate()
            .fold(0u64, |acc, (i, val)| acc | (*val as u64) << i);
        (
            (quantum_bits.count_ones() as f64 / n_qubits as f64 * 0.8 + (time * 0.1).sin() * 0.2)
                * 2.0
                - 1.0,
            measurements,
        )
    }
}

// --- traits for abstraction ---

pub trait QoaAudioDecoder {
    fn decode_audio_file_to_samples(
        &self,
        audio_path: &Path,
    ) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>>;
}

pub trait QuantumProcessor {
    fn process_frame(
        &mut self,
        audio_features: &AudioFeatures,
        frame_index: usize,
        total_frames: usize,
    ) -> QuantumVisualData;
}

// --- implementation ---

pub struct AudioVisualizer {
    quantum_noise_gen: QuantumNoiseGenerator,
    prev_rms_loudness: Arc<RwLock<f64>>, // using rwlock for read-heavy data
    pub prev_hue: f64,
}

impl AudioVisualizer {
    // creates a new audio visualizer
    pub fn new() -> Self {
        // configure rayon thread pool once at application start
        configure_thread_pool();

        Self {
            quantum_noise_gen: QuantumNoiseGenerator::new(16, 42),
            prev_rms_loudness: Arc::new(RwLock::new(0.0)),
            prev_hue: 0.0,
        }
    }

    // calculates band energy from spectrum magnitudes
    #[inline(always)]
    fn calculate_band_energy_from_mags(
        spectrum_mags: &[f64],
        sample_rate: u32,
        low_hz: f64,
        high_hz: f64,
    ) -> f64 {
        unsafe {
            audio_simd::calculate_band_energy_f64(spectrum_mags, sample_rate, low_hz, high_hz)
        }
    }

    // extracts enhanced audio features
    fn extract_enhanced_features(
        spectrum: &[Complex<f32>],
        sample_rate: u32,
        window_samples: usize,
        audio_samples: &[f32],
        start_idx: usize,
        prev_rms: f64,
    ) -> AudioFeatures {
        let spectrum_mags: Vec<f64> = spectrum.iter().map(|c| c.norm() as f64).collect();
        let nyquist_freq = sample_rate as f64 / 2.0;
        let bin_width = nyquist_freq / spectrum.len() as f64;
        let rms_loudness = if window_samples > 0 {
            let end_idx = (start_idx + window_samples).min(audio_samples.len());
            let samples_slice = &audio_samples[start_idx..end_idx];
            
            // use alignedbuffer for sum_squares_f32
            let mut aligned_samples = AlignedBuffer::<f32>::new(samples_slice.len());
            aligned_samples.as_mut_slice().copy_from_slice(samples_slice);

            // Call to unsafe function `audio_simd::sum_squares_f32` is unsafe and requires unsafe block
            let sum_sq = unsafe { audio_simd::sum_squares_f32(&aligned_samples) };

            (sum_sq / samples_slice.len() as f64).sqrt()
        } else {
            0.0
        };

        let (sum_weighted_freq, sum_magnitudes) =
            unsafe { audio_simd::calculate_spectral_centroid_f64(&spectrum_mags, bin_width) };

        let spectral_centroid = if sum_magnitudes > 0.0 {
            sum_weighted_freq / sum_magnitudes
        } else {
            0.0
        };
        let low_freq_energy =
            Self::calculate_band_energy_from_mags(&spectrum_mags, sample_rate, 0.0, 200.0);
        let mid_freq_energy =
            Self::calculate_band_energy_from_mags(&spectrum_mags, sample_rate, 200.0, 2000.0);
        let high_freq_energy =
            Self::calculate_band_energy_from_mags(&spectrum_mags, sample_rate, 2000.0, 20000.0);
        let peak_bin = spectrum_mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let peak_frequency = peak_bin as f64 * bin_width;
        let total_energy: f64 = spectrum_mags.iter().sum();
        let mut cumulative_energy = 0.0;
        let mut rolloff_bin = 0;
        for (_i, &mag) in spectrum_mags.iter().enumerate() {
            cumulative_energy += mag;
            if cumulative_energy >= 0.85 * total_energy {
                rolloff_bin = _i;
                break;
            }
        }
        let spectral_rolloff = rolloff_bin as f64 * bin_width;

        // transient detection
        let rms_diff = (rms_loudness - prev_rms).abs();
        let transient_strength = (rms_diff * 50.0).min(1.0);

        AudioFeatures {
            rms_loudness,
            spectral_centroid,
            low_freq_energy,
            mid_freq_energy,
            high_freq_energy,
            spectrum: spectrum_mags,
            peak_frequency,
            spectral_rolloff,
            transient_strength,
        }
    }
}

// manually implement clone for audiovisualizer
impl Clone for AudioVisualizer {
    fn clone(&self) -> Self {
        Self {
            // create a new quantumnoisegenerator instance when cloning audiovisualizer
            quantum_noise_gen: QuantumNoiseGenerator::new(
                self.quantum_noise_gen.quantum_state.lock().n,
                0,
            ),
            // clone the content of prev_rms_loudness using a read lock
            prev_rms_loudness: Arc::new(RwLock::new(*self.prev_rms_loudness.read())),
            //hue speed
            prev_hue: 0.0,
        }
    }
}

impl QoaAudioDecoder for AudioVisualizer {
    // memory-map audio files for faster access
    fn decode_audio_file_to_samples(
        &self,
        audio_path: &Path,
    ) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
        info!("decoding audio from: {:?}", audio_path);
        match audio_path.extension().and_then(|s| s.to_str()) {
            Some("wav") => {
                let mut reader = hound::WavReader::open(audio_path)?;
                let spec = reader.spec();
                let num_channels = spec.channels as usize;

                info!(
                    "wav file has {} channels at {} hz",
                    num_channels, spec.sample_rate
                );

                if num_channels > 2 {
                    info!(
                        "audio has {} channels, only using first two channels. (averaging them)",
                        num_channels
                    );
                }

                // read all raw i16 samples first
                let all_raw_samples: Vec<i16> =
                    reader.samples().collect::<Result<_, hound::Error>>()?;
                let mut samples = Vec::with_capacity(all_raw_samples.len() / num_channels); // adjusted capacity

                // convert all raw i16 samples to f32 once
                let temp_f32_samples: Vec<f32> = all_raw_samples
                    .into_iter()
                    .map(|s| s as f32 / i16::MAX as f32)
                    .collect();

                if num_channels == 2 {
                    info!("processing stereo audio (simd)");
                    // use our new multi-architecture simd function.
                    unsafe { audio_simd::mix_stereo_to_mono_f32(&temp_f32_samples, &mut samples) };
                } else if num_channels == 1 {
                    info!("processing mono audio");
                    samples = temp_f32_samples; // directly use if mono
                } else {
                    info!("processing multi-channel audio (scalar, averaging channels)");
                    // fallback for > 2 channels
                    for chunk in temp_f32_samples.chunks(num_channels) {
                        samples.push(chunk.iter().sum::<f32>() / num_channels as f32);
                    }
                }
                Ok((samples, spec.sample_rate))
            }
            Some("qoa") => {
                info!("decoding qoa audio");
                let file_path = Path::new(audio_path);
                let file = fs::File::open(file_path)?;
                let _mmap = unsafe { MmapOptions::new().map(&file)? };
                Err(format!("qoa decoding is unavailable due0 to missing function `decode_to_vec_f32` in the `qoa` crate. please check your `qoa` dependency version and features.").into())
            }
            _ => Err(format!("unsupported audio format: {:?}", audio_path).into()),
        }
    }
}

impl QuantumProcessor for AudioVisualizer {
    fn process_frame(
        &mut self,
        audio_features: &AudioFeatures,
        _frame_index: usize,
        total_frames: usize,
    ) -> QuantumVisualData {
        let quantum_noise_value: f64;
        let quantum_measurements_vec: Vec<bool>;
        {
            // use fast_sin for performance
            let (noise_val, measurements) = self
                .quantum_noise_gen
                .quantum_noise(_frame_index as f64 / total_frames as f64);
            quantum_noise_value = noise_val;
            quantum_measurements_vec = measurements;
        }

        let current_rms = audio_features.rms_loudness;
        let mut prev_rms_write_guard = self.prev_rms_loudness.write();
        let prev_rms = *prev_rms_write_guard;
        *prev_rms_write_guard = current_rms;
        drop(prev_rms_write_guard);

        let rms_change = (current_rms - prev_rms).abs();

        let normalized_rms = current_rms.min(1.0).max(0.0);
        let normalized_centroid = audio_features.spectral_centroid.min(5000.0).max(0.0) / 5000.0;
        let _normalized_peak_freq = audio_features.peak_frequency.min(10000.0).max(0.0) / 10000.0;
        let normalized_rolloff = audio_features.spectral_rolloff.min(20000.0).max(0.0) / 20000.0;
        let normalized_transient = audio_features.transient_strength.min(1.0).max(0.0);

        // --- tones reactivity ---
        let total_energy = audio_features.low_freq_energy
            + audio_features.mid_freq_energy
            + audio_features.high_freq_energy
            + 1e-6;
        let normalized_bass = (audio_features.low_freq_energy / total_energy)
            .min(1.0)
            .max(0.0);
        let _normalized_mid = (audio_features.mid_freq_energy / total_energy)
            .min(1.0)
            .max(0.0);
        let _normalized_high = (audio_features.high_freq_energy / total_energy)
            .min(1.0)
            .max(0.0);

        let base_brightness = 1.0 + normalized_bass * 1.0 + normalized_rms * 0.3;

        // --- colour shift, and cycle ---
        let increment_per_frame = 0.01;
        let color_hue = (self.prev_hue + increment_per_frame) % 360.0;
        self.prev_hue = color_hue;

        let pattern_density = 4.0 + normalized_bass * 8.0 + normalized_rms * 1.5;
        let distortion_magnitude = rms_change * 0.0;
        let flicker_intensity = normalized_transient * (1.0 + normalized_bass * 1.0);

        let noise_seed = (quantum_noise_value * 1_000_000.0).abs() as u64;
        let chaos_factor =
            (audio_features.high_freq_energy / (audio_features.low_freq_energy + 1.0)).min(10.0)
                / 10.0;
        let interference_pattern = (normalized_rolloff + normalized_centroid) / 2.0;
        let quantum_entanglement = quantum_measurements_vec.iter().filter(|&b| *b).count() as f64
            / quantum_measurements_vec.len() as f64;
        let quantum_coherence = 1.0 - quantum_noise_value.abs();

        let flow_field_strength = normalized_rms * normalized_centroid;
        let depth_modulation = normalized_transient;

        QuantumVisualData {
            base_brightness,
            color_hue,
            pattern_density,
            distortion_magnitude,
            flicker_intensity,
            noise_seed,
            chaos_factor,
            interference_pattern,
            quantum_entanglement,
            quantum_measurements: quantum_measurements_vec,
            quantum_coherence,
            flow_field_strength,
            depth_modulation,
        }
    }
}

// thread-local bump allocator for zero-allocation rendering per frame
thread_local! {
    static FRAME_ALLOCATOR: RefCell<Bump> = RefCell::new(Bump::new());
}

// --- main visualization logic ---
// dispatch function for render_frame based on resolution
#[unsafe(link_section = ".text.hot")] // instruction cache optimization
pub fn render_frame(
    _frame_index: usize,
    width: u32,
    height: u32,
    quantum_data: &QuantumVisualData,
    perlin_noise: &PerlinNoise,
    spectrum_data: &[f64],
    spectrum_direction: SpectrumDirection,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    match (width, height) {
        (1920, 1080) => render_frame_1080p(
            _frame_index,
            width,
            height,
            quantum_data,
            perlin_noise,
            spectrum_data,
            spectrum_direction,
        ),
        (3840, 2160) => render_frame_4k(
            _frame_index,
            width,
            height,
            quantum_data,
            perlin_noise,
            spectrum_data,
            spectrum_direction,
        ),
        _ => render_frame_generic(
            _frame_index,
            width,
            height,
            quantum_data,
            perlin_noise,
            spectrum_data,
            spectrum_direction,
        ),
    }
}

// specialized implementation for 1080p with hard-coded constants and optimizations
#[inline(never)] // force it to be compiled separately
#[unsafe(link_section = ".text.hot")] // instruction cache optimization
fn render_frame_1080p(
    _frame_index: usize,
    width: u32,
    height: u32,
    quantum_data: &QuantumVisualData,
    perlin_noise: &PerlinNoise,
    spectrum_data: &[f64],
    spectrum_direction: SpectrumDirection,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    // constants for 1080p
    const WIDTH_F64: f64 = 1920.0;
    const HEIGHT_F64: f64 = 1080.0;
    const CENTER_X: f64 = 960.0;
    const CENTER_Y: f64 = 540.0;
    const BLOCK_SIZE: u32 = 64;

    let mut img = ImageBuffer::new(width, height);

    for y_block_start in (0..height).step_by(BLOCK_SIZE as usize) {
        for x_block_start in (0..width).step_by(BLOCK_SIZE as usize) {
            let y_block_end = (y_block_start + BLOCK_SIZE).min(height);
            let x_block_end = (x_block_start + BLOCK_SIZE).min(width);

            // process rows in parallel within the current block
            let block_pixel_data: Vec<Vec<(u32, u32, Rgb<u8>)>> = (y_block_start..y_block_end)
                .into_par_iter()
                .map(|y| {
                    // access thread-local allocator inside the parallel map closure
                    FRAME_ALLOCATOR.with(|allocator_cell| {
                        let allocator = &mut *allocator_cell.borrow_mut();
                        allocator.reset(); // reset allocator for each thread's work unit

                        let mut h_values: bumpalo::collections::Vec<'_, f64> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);
                        let mut s_values: bumpalo::collections::Vec<'_, f64> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);
                        let mut v_values: bumpalo::collections::Vec<'_, f64> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);

                        let mut row_coords_bump: bumpalo::collections::Vec<'_, (f64, f64)> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);
                        let mut noise_coords_bump: bumpalo::collections::Vec<'_, (f64, f64)> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);

                        for x in x_block_start..x_block_end {
                            let norm_x = (x as f64 - CENTER_X) / WIDTH_F64;
                            let norm_y = (y as f64 - CENTER_Y) / HEIGHT_F64;

                            // apply flow field, using fast_sin
                            let flow_x = fast_sin(norm_x + quantum_data.flow_field_strength * 0.1) * 0.05;
                            let flow_y = fast_sin(norm_y + quantum_data.flow_field_strength * 0.1 + std::f64::consts::PI / 2.0) * 0.05; // Use cos equivalent

                            row_coords_bump.push((norm_x + flow_x, norm_y + flow_y));

                            noise_coords_bump.push((
                                (norm_x + flow_x) * quantum_data.pattern_density * 10.0
                                    + quantum_data.noise_seed as f64 / 1000.0
                                    + quantum_data.chaos_factor * 5.0,
                                (norm_y + flow_y) * quantum_data.pattern_density * 10.0
                                    + quantum_data.noise_seed as f64 / 1000.0
                                    + quantum_data.chaos_factor * 5.0,
                            ));
                        }

                        let noise_values = perlin_noise.get_batch(&row_coords_bump);

                        let mut row_pixels_for_blending: bumpalo::collections::Vec<'_, (u32, u32, Rgb<u8>)> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);

                        // process each pixel in the row
                        for (x_idx, x) in (x_block_start..x_block_end).enumerate() {
                            let (norm_x, norm_y) = row_coords_bump[x_idx];
                            let noise_val = noise_values[x_idx];

                            // quantum interference pattern
                            let interference =
                                fast_sin(norm_x * 10.0) * fast_sin(norm_y * 10.0 + std::f64::consts::PI / 2.0) * quantum_data.interference_pattern; // Use cos equivalent
                            let final_noise = (noise_val + interference).max(-1.0).min(1.0);

                            let brightness_base = quantum_data.base_brightness;
                            let brightness_noise = (final_noise + 1.0) / 2.0; // scale noise from -1..1 to 0..1
                            let final_brightness =
                                (brightness_base * brightness_noise * (1.0 - quantum_data.flicker_intensity)
                                    + quantum_data.flicker_intensity * rand::random::<f64>())
                                .max(0.0)
                                .min(1.0);

                            // color mapping with hue based on audio features and quantum state
                            let hue = (quantum_data.color_hue
                                + normalized_centroid_for_color(&spectrum_data) * 180.0
                                + quantum_data.quantum_coherence * 90.0)
                                % 360.0;
                            let saturation = (0.7 + quantum_data.quantum_entanglement * 0.3).min(1.0);
                            let value = final_brightness;

                            h_values.push(hue);
                            s_values.push(saturation);
                            v_values.push(value);

                            // apply distortion - distorted_x and distorted_y are calculated here
                            // but the actual pixel setting is done later after batch hsv_to_rgb
                            let distorted_x = (x as f64
                                + fast_sin(norm_x * quantum_data.distortion_magnitude * 50.0)
                                + quantum_data.depth_modulation * 20.0 * fast_sin(noise_val * 2.0))
                                as u32;
                            let distorted_y = (y as f64
                                + fast_sin(norm_y * quantum_data.distortion_magnitude * 50.0 + std::f64::consts::PI / 2.0)
                                + quantum_data.depth_modulation * 20.0 * fast_sin(noise_val * 2.0 + std::f64::consts::PI / 2.0))
                                as u32;

                            // store pixel, will be blended later
                            if distorted_x < width && distorted_y < height {
                                // we push a dummy Rgb here, the actual color will be set after batch conversion
                                row_pixels_for_blending.push((distorted_x, distorted_y, Rgb([0, 0, 0])));
                            }
                        }

                        // batch hsv to rgb conversion
                        let rgb_colors = image_simd::hsv_to_rgb_batch(&h_values, &s_values, &v_values);

                        // update the stored pixels with actual colors
                        let mut final_row_pixels: Vec<(u32, u32, Rgb<u8>)> = Vec::with_capacity(row_pixels_for_blending.len());
                        let mut color_idx = 0;
                        for (_idx, (dx, dy, _)) in row_pixels_for_blending.into_iter().enumerate() {
                            if color_idx < rgb_colors.len() {
                                let (r, g, b) = rgb_colors[color_idx];
                                final_row_pixels.push((dx, dy, Rgb([
                                    (r * 255.0) as u8,
                                    (g * 255.0) as u8,
                                    (b * 255.0) as u8,
                                ])));
                                color_idx += 1;
                            }
                        }
                        final_row_pixels
                    })
                })
                .collect();

            // apply all pixel changes to the image sequentially within the block
            for row in block_pixel_data {
                // prefetch next row of pixels
                if y_block_start + BLOCK_SIZE < height {
                    image_simd::prefetch_data(img.as_ptr(), ((y_block_start + BLOCK_SIZE) * width) as usize, image_simd::PrefetchStrategy::L1);
                }

                // collect current pixels and new pixels for batch blending
                let mut current_pixels_to_blend: Vec<Rgb<u8>> = Vec::with_capacity(row.len());
                let mut new_pixels_to_blend: Vec<Rgb<u8>> = Vec::with_capacity(row.len());
                let mut coords_to_blend: Vec<(u32, u32)> = Vec::with_capacity(row.len());

                for (x, y, pixel) in row {
                    current_pixels_to_blend.push(*img.get_pixel(x, y));
                    new_pixels_to_blend.push(pixel);
                    coords_to_blend.push((x, y));
                }

                let blended_pixels = image_simd::blend_pixels_batch(
                    &current_pixels_to_blend,
                    &new_pixels_to_blend,
                    0.5,
                );

                for (_idx, (x, y)) in coords_to_blend.into_iter().enumerate() {
                    img.put_pixel(x, y, blended_pixels[_idx]);
                }
            }
        }
    }

    // spectrum visualization
    let spectrum_height = height / 4;
    let bar_width = width as f64 / spectrum_data.len() as f64;

    // pre-compute all bar coordinates and colors
    let mut bar_pixels: Vec<(u32, u32, Rgb<u8>)> = Vec::with_capacity(spectrum_data.len() * spectrum_height as usize);
    for (i, &magnitude) in spectrum_data.iter().enumerate() {
        let bar_height = (magnitude * 500.0).min(spectrum_height as f64) as u32;
        if bar_height == 0 { continue; }
        
        let (x_pos, start_y) = match spectrum_direction {
            SpectrumDirection::Ltr => ((i as f64 * bar_width) as u32, height - bar_height),
            SpectrumDirection::Rtl => (width - (i as f64 * bar_width) as u32 - bar_width as u32, 0),
            SpectrumDirection::None => continue,
        };
        
        // compute all pixels for this bar
        for y_offset in 0..bar_height {
            let plot_y = start_y + y_offset;
            for x_offset in 0..bar_width as u32 {
                let plot_x = x_pos + x_offset;
                if plot_x < width && plot_y < height {
                    bar_pixels.push((plot_x, plot_y, Rgb([255, 0, 0])));
                }
            }
        }
    }

    // batch process all bar pixels
    if !bar_pixels.is_empty() {
        let mut current_pixels: Vec<Rgb<u8>> = Vec::with_capacity(bar_pixels.len());
        let mut new_pixels: Vec<Rgb<u8>> = Vec::with_capacity(bar_pixels.len());
        let mut coords_to_blend: Vec<(u32, u32)> = Vec::with_capacity(bar_pixels.len()); 
        
        for (x, y, color) in &bar_pixels {
            current_pixels.push(*img.get_pixel(*x, *y));
            new_pixels.push(*color);
            coords_to_blend.push((*x, *y)); 
        }
        
        // batch blend operation
        let blended_pixels = image_simd::blend_pixels_batch(&current_pixels, &new_pixels, 0.2);
        
        // apply blended pixels
        for (_idx, (x, y)) in coords_to_blend.into_iter().enumerate() {
            img.put_pixel(x, y, blended_pixels[_idx]);
        }
    }
    img
}

// specialized implementation for 4k with hard-coded constants and optimizations
#[inline(never)] // force it to be compiled separately
#[unsafe(link_section = ".text.hot")] // instruction cache optimization
fn render_frame_4k(
    _frame_index: usize,
    width: u32,
    height: u32,
    quantum_data: &QuantumVisualData,
    perlin_noise: &PerlinNoise,
    spectrum_data: &[f64],
    spectrum_direction: SpectrumDirection,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    // constants for 4k
    const WIDTH_F64: f64 = 3840.0;
    const HEIGHT_F64: f64 = 2160.0;
    const CENTER_X: f64 = 1920.0;
    const CENTER_Y: f64 = 1080.0;
    const BLOCK_SIZE: u32 = 64;

    let mut img = ImageBuffer::new(width, height);

    for y_block_start in (0..height).step_by(BLOCK_SIZE as usize) {
        for x_block_start in (0..width).step_by(BLOCK_SIZE as usize) {
            let y_block_end = (y_block_start + BLOCK_SIZE).min(height);
            let x_block_end = (x_block_start + BLOCK_SIZE).min(width);

            // process rows in parallel within the current block
            let block_pixel_data: Vec<Vec<(u32, u32, Rgb<u8>)>> = (y_block_start..y_block_end)
                .into_par_iter()
                .map(|y| {
                    // access thread-local allocator inside the parallel map closure
                    FRAME_ALLOCATOR.with(|allocator_cell| {
                        let allocator = &mut *allocator_cell.borrow_mut();
                        allocator.reset(); // reset allocator for each thread's work unit

                        let mut h_values: bumpalo::collections::Vec<'_, f64> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);
                        let mut s_values: bumpalo::collections::Vec<'_, f64> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);
                        let mut v_values: bumpalo::collections::Vec<'_, f64> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);

                        let mut row_coords_bump: bumpalo::collections::Vec<'_, (f64, f64)> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);
                        let mut noise_coords_bump: bumpalo::collections::Vec<'_, (f64, f64)> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);

                        for x in x_block_start..x_block_end {
                            let norm_x = (x as f64 - CENTER_X) / WIDTH_F64;
                            let norm_y = (y as f64 - CENTER_Y) / HEIGHT_F64;

                            // apply flow field, using fast_sin
                            let flow_x = fast_sin(norm_x + quantum_data.flow_field_strength * 0.1) * 0.05;
                            let flow_y = fast_sin(norm_y + quantum_data.flow_field_strength * 0.1 + std::f64::consts::PI / 2.0) * 0.05; // Use cos equivalent

                            row_coords_bump.push((norm_x + flow_x, norm_y + flow_y));

                            noise_coords_bump.push((
                                (norm_x + flow_x) * quantum_data.pattern_density * 10.0
                                    + quantum_data.noise_seed as f64 / 1000.0
                                    + quantum_data.chaos_factor * 5.0,
                                (norm_y + flow_y) * quantum_data.pattern_density * 10.0
                                    + quantum_data.noise_seed as f64 / 1000.0
                                    + quantum_data.chaos_factor * 5.0,
                            ));
                        }

                        let noise_values = perlin_noise.get_batch(&row_coords_bump);

                        let mut row_pixels_for_blending: bumpalo::collections::Vec<'_, (u32, u32, Rgb<u8>)> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);

                        // process each pixel in the row
                        for (x_idx, x) in (x_block_start..x_block_end).enumerate() {
                            let (norm_x, norm_y) = row_coords_bump[x_idx];
                            let noise_val = noise_values[x_idx];

                            // quantum interference pattern
                            let interference =
                                fast_sin(norm_x * 10.0) * fast_sin(norm_y * 10.0 + std::f64::consts::PI / 2.0) * quantum_data.interference_pattern; // Use cos equivalent
                            let final_noise = (noise_val + interference).max(-1.0).min(1.0);

                            let brightness_base = quantum_data.base_brightness;
                            let brightness_noise = (final_noise + 1.0) / 2.0; // scale noise from -1..1 to 0..1
                            let final_brightness =
                                (brightness_base * brightness_noise * (1.0 - quantum_data.flicker_intensity)
                                    + quantum_data.flicker_intensity * rand::random::<f64>())
                                .max(0.0)
                                .min(1.0);

                            // color mapping with hue based on audio features and quantum state
                            let hue = (quantum_data.color_hue
                                + normalized_centroid_for_color(&spectrum_data) * 180.0
                                + quantum_data.quantum_coherence * 90.0)
                                % 360.0;
                            let saturation = (0.7 + quantum_data.quantum_entanglement * 0.3).min(1.0);
                            let value = final_brightness;

                            h_values.push(hue);
                            s_values.push(saturation);
                            v_values.push(value);

                            // apply distortion - distorted_x and distorted_y are calculated here
                            // but the actual pixel setting is done later after batch hsv_to_rgb
                            let distorted_x = (x as f64
                                + fast_sin(norm_x * quantum_data.distortion_magnitude * 50.0)
                                + quantum_data.depth_modulation * 20.0 * fast_sin(noise_val * 2.0))
                                as u32;
                            let distorted_y = (y as f64
                                + fast_sin(norm_y * quantum_data.distortion_magnitude * 50.0 + std::f64::consts::PI / 2.0)
                                + quantum_data.depth_modulation * 20.0 * fast_sin(noise_val * 2.0 + std::f64::consts::PI / 2.0))
                                as u32;

                            // store pixel, will be blended later
                            if distorted_x < width && distorted_y < height {
                                // we push a dummy Rgb here, the actual color will be set after batch conversion
                                row_pixels_for_blending.push((distorted_x, distorted_y, Rgb([0, 0, 0])));
                            }
                        }

                        // batch hsv to rgb conversion
                        let rgb_colors = image_simd::hsv_to_rgb_batch(&h_values, &s_values, &v_values);

                        // update the stored pixels with actual colors
                        let mut final_row_pixels: Vec<(u32, u32, Rgb<u8>)> = Vec::with_capacity(row_pixels_for_blending.len());
                        let mut color_idx = 0;
                        for (_idx, (dx, dy, _)) in row_pixels_for_blending.into_iter().enumerate() {
                            if color_idx < rgb_colors.len() {
                                let (r, g, b) = rgb_colors[color_idx];
                                final_row_pixels.push((dx, dy, Rgb([
                                    (r * 255.0) as u8,
                                    (g * 255.0) as u8,
                                    (b * 255.0) as u8,
                                ])));
                                color_idx += 1;
                            }
                        }
                        final_row_pixels
                    })
                })
                .collect();

            // apply all pixel changes to the image sequentially within the block
            for row in block_pixel_data {
                // prefetch next row of pixels
                if y_block_start + BLOCK_SIZE < height {
                    image_simd::prefetch_data(img.as_ptr(), ((y_block_start + BLOCK_SIZE) * width) as usize, image_simd::PrefetchStrategy::L1);
                }

                // collect current pixels and new pixels for batch blending
                let mut current_pixels_to_blend: Vec<Rgb<u8>> = Vec::with_capacity(row.len());
                let mut new_pixels_to_blend: Vec<Rgb<u8>> = Vec::with_capacity(row.len());
                let mut coords_to_blend: Vec<(u32, u32)> = Vec::with_capacity(row.len());

                for (x, y, pixel) in row {
                    current_pixels_to_blend.push(*img.get_pixel(x, y));
                    new_pixels_to_blend.push(pixel);
                    coords_to_blend.push((x, y));
                }

                let blended_pixels = image_simd::blend_pixels_batch(
                    &current_pixels_to_blend,
                    &new_pixels_to_blend,
                    0.5,
                );

                for (_idx, (x, y)) in coords_to_blend.into_iter().enumerate() {
                    img.put_pixel(x, y, blended_pixels[_idx]);
                }
            }
        }
    }

    // spectrum visualization
    let spectrum_height = height / 4;
    let bar_width = width as f64 / spectrum_data.len() as f64;

    // pre-compute all bar coordinates and colors
    let mut bar_pixels: Vec<(u32, u32, Rgb<u8>)> = Vec::with_capacity(spectrum_data.len() * spectrum_height as usize);
    for (i, &magnitude) in spectrum_data.iter().enumerate() {
        let bar_height = (magnitude * 500.0).min(spectrum_height as f64) as u32;
        if bar_height == 0 { continue; }
        
        let (x_pos, start_y) = match spectrum_direction {
            SpectrumDirection::Ltr => ((i as f64 * bar_width) as u32, height - bar_height),
            SpectrumDirection::Rtl => (width - (i as f64 * bar_width) as u32 - bar_width as u32, 0),
            SpectrumDirection::None => continue,
        };
        
        // compute all pixels for this bar
        for y_offset in 0..bar_height {
            let plot_y = start_y + y_offset;
            for x_offset in 0..bar_width as u32 {
                let plot_x = x_pos + x_offset;
                if plot_x < width && plot_y < height {
                    bar_pixels.push((plot_x, plot_y, Rgb([255, 0, 0])));
                }
            }
            }
    }

    // batch process all bar pixels
    if !bar_pixels.is_empty() {
        let mut current_pixels: Vec<Rgb<u8>> = Vec::with_capacity(bar_pixels.len());
        let mut new_pixels: Vec<Rgb<u8>> = Vec::with_capacity(bar_pixels.len());
        let mut coords_to_blend: Vec<(u32, u32)> = Vec::with_capacity(bar_pixels.len()); 
        
        for (x, y, color) in &bar_pixels {
            current_pixels.push(*img.get_pixel(*x, *y));
            new_pixels.push(*color);
            coords_to_blend.push((*x, *y)); 
        }
        
        // batch blend operation
        let blended_pixels = image_simd::blend_pixels_batch(&current_pixels, &new_pixels, 0.2);
        
        // apply blended pixels
        for (_idx, (x, y)) in coords_to_blend.into_iter().enumerate() {
            img.put_pixel(x, y, blended_pixels[_idx]);
        }
    }
    img
}

// generic render_frame implementation
#[inline(never)] // force it to be compiled separately
#[unsafe(link_section = ".text.hot")] // instruction cache optimization
fn render_frame_generic(
    _frame_index: usize,
    width: u32,
    height: u32,
    quantum_data: &QuantumVisualData,
    perlin_noise: &PerlinNoise,
    spectrum_data: &[f64],
    spectrum_direction: SpectrumDirection,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::new(width, height);
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;

    const BLOCK_SIZE: u32 = 64; // typical L1 cache line size, process in blocks

    // process image in blocks
    for y_block_start in (0..height).step_by(BLOCK_SIZE as usize) {
        for x_block_start in (0..width).step_by(BLOCK_SIZE as usize) {
            let y_block_end = (y_block_start + BLOCK_SIZE).min(height);
            let x_block_end = (x_block_start + BLOCK_SIZE).min(width);

            // process rows in parallel within the current block
            let block_pixel_data: Vec<Vec<(u32, u32, Rgb<u8>)>> = (y_block_start..y_block_end)
                .into_par_iter()
                .map(|y| {
                    // access thread-local allocator inside the parallel map closure
                    FRAME_ALLOCATOR.with(|allocator_cell| {
                        let allocator = &mut *allocator_cell.borrow_mut();
                        allocator.reset(); // reset allocator for each thread's work unit

                        // use bump allocator for temporary vectors
                        let mut h_values: bumpalo::collections::Vec<'_, f64> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);
                        let mut s_values: bumpalo::collections::Vec<'_, f64> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);
                        let mut v_values: bumpalo::collections::Vec<'_, f64> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);

                        // pre-compute coordinates and noise values for the entire row within the block
                        let mut row_coords_bump: bumpalo::collections::Vec<'_, (f64, f64)> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);
                        let mut noise_coords_bump: bumpalo::collections::Vec<'_, (f64, f64)> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);

                        for x in x_block_start..x_block_end {
                            let norm_x = (x as f64 - center_x) / width as f64;
                            let norm_y = (y as f64 - center_y) / height as f64;

                            // apply flow field, using fast_sin
                            let flow_x = fast_sin(norm_x + quantum_data.flow_field_strength * 0.1) * 0.05;
                            let flow_y = fast_sin(norm_y + quantum_data.flow_field_strength * 0.1 + std::f64::consts::PI / 2.0) * 0.05; // Use cos equivalent

                            row_coords_bump.push((norm_x + flow_x, norm_y + flow_y));

                            noise_coords_bump.push((
                                (norm_x + flow_x) * quantum_data.pattern_density * 10.0
                                    + quantum_data.noise_seed as f64 / 1000.0
                                    + quantum_data.chaos_factor * 5.0,
                                (norm_y + flow_y) * quantum_data.pattern_density * 10.0
                                    + quantum_data.noise_seed as f64 / 1000.0
                                    + quantum_data.chaos_factor * 5.0,
                            ));
                        }

                        let noise_values = perlin_noise.get_batch(&noise_coords_bump);

                        let mut row_pixels_for_blending: bumpalo::collections::Vec<'_, (u32, u32, Rgb<u8>)> = bumpalo::collections::Vec::with_capacity_in((x_block_end - x_block_start) as usize, allocator);

                        // process each pixel in the row
                        for (x_idx, x) in (x_block_start..x_block_end).enumerate() {
                            let (norm_x, norm_y) = row_coords_bump[x_idx];
                            let noise_val = noise_values[x_idx];

                            // quantum interference pattern
                            let interference =
                                fast_sin(norm_x * 10.0) * fast_sin(norm_y * 10.0 + std::f64::consts::PI / 2.0) * quantum_data.interference_pattern; // Use cos equivalent
                            let final_noise = (noise_val + interference).max(-1.0).min(1.0);

                            let brightness_base = quantum_data.base_brightness;
                            let brightness_noise = (final_noise + 1.0) / 2.0; // scale noise from -1..1 to 0..1
                            let final_brightness =
                                (brightness_base * brightness_noise * (1.0 - quantum_data.flicker_intensity)
                                    + quantum_data.flicker_intensity * rand::random::<f64>())
                                .max(0.0)
                                .min(1.0);

                            // color mapping with hue based on audio features and quantum state
                            let hue = (quantum_data.color_hue
                                + normalized_centroid_for_color(&spectrum_data) * 180.0
                                + quantum_data.quantum_coherence * 90.0)
                                % 360.0;
                            let saturation = (0.7 + quantum_data.quantum_entanglement * 0.3).min(1.0);
                            let value = final_brightness;

                            h_values.push(hue);
                            s_values.push(saturation);
                            v_values.push(value);

                            // apply distortion - distorted_x and distorted_y are calculated here
                            // but the actual pixel setting is done later after batch hsv_to_rgb
                            let distorted_x = (x as f64
                                + fast_sin(norm_x * quantum_data.distortion_magnitude * 50.0)
                                + quantum_data.depth_modulation * 20.0 * fast_sin(noise_val * 2.0))
                                as u32;
                            let distorted_y = (y as f64
                                + fast_sin(norm_y * quantum_data.distortion_magnitude * 50.0 + std::f64::consts::PI / 2.0)
                                + quantum_data.depth_modulation * 20.0 * fast_sin(noise_val * 2.0 + std::f64::consts::PI / 2.0))
                                as u32;

                            // store pixel, will be blended later
                            if distorted_x < width && distorted_y < height {
                                // we push a dummy Rgb here, the actual color will be set after batch conversion
                                row_pixels_for_blending.push((distorted_x, distorted_y, Rgb([0, 0, 0])));
                            }
                        }

                        // batch hsv to rgb conversion
                        let rgb_colors = image_simd::hsv_to_rgb_batch(&h_values, &s_values, &v_values);

                        // update the stored pixels with actual colors
                        let mut final_row_pixels: Vec<(u32, u32, Rgb<u8>)> = Vec::with_capacity(row_pixels_for_blending.len());
                        let mut color_idx = 0;
                        for (_idx, (dx, dy, _)) in row_pixels_for_blending.into_iter().enumerate() {
                            if color_idx < rgb_colors.len() {
                                let (r, g, b) = rgb_colors[color_idx];
                                final_row_pixels.push((dx, dy, Rgb([
                                    (r * 255.0) as u8,
                                    (g * 255.0) as u8,
                                    (b * 255.0) as u8,
                                ])));
                                color_idx += 1;
                            }
                        }
                        final_row_pixels
                    })
                })
                .collect();

            // apply all pixel changes to the image sequentially within the block
            for row in block_pixel_data {
                // prefetch next row of pixels
                if y_block_start + BLOCK_SIZE < height {
                    image_simd::prefetch_data(img.as_ptr(), ((y_block_start + BLOCK_SIZE) * width) as usize, image_simd::PrefetchStrategy::L1);
                }

                // collect current pixels and new pixels for batch blending
                let mut current_pixels_to_blend: Vec<Rgb<u8>> = Vec::with_capacity(row.len());
                let mut new_pixels_to_blend: Vec<Rgb<u8>> = Vec::with_capacity(row.len());
                let mut coords_to_blend: Vec<(u32, u32)> = Vec::with_capacity(row.len());

                for (x, y, pixel) in row {
                    current_pixels_to_blend.push(*img.get_pixel(x, y));
                    new_pixels_to_blend.push(pixel);
                    coords_to_blend.push((x, y));
                }

                let blended_pixels = image_simd::blend_pixels_batch(
                    &current_pixels_to_blend,
                    &new_pixels_to_blend,
                    0.5,
                );

                for (_idx, (x, y)) in coords_to_blend.into_iter().enumerate() {
                    img.put_pixel(x, y, blended_pixels[_idx]);
                }
            }
        }
    }

    // spectrum visualization
    let spectrum_height = height / 4;
    let bar_width = width as f64 / spectrum_data.len() as f64;

    // pre-compute all bar coordinates and colors
    let mut bar_pixels: Vec<(u32, u32, Rgb<u8>)> = Vec::with_capacity(spectrum_data.len() * spectrum_height as usize);
    for (i, &magnitude) in spectrum_data.iter().enumerate() {
        let bar_height = (magnitude * 500.0).min(spectrum_height as f64) as u32;
        if bar_height == 0 { continue; }
        
        let (x_pos, start_y) = match spectrum_direction {
            SpectrumDirection::Ltr => ((i as f64 * bar_width) as u32, height - bar_height),
            SpectrumDirection::Rtl => (width - (i as f64 * bar_width) as u32 - bar_width as u32, 0),
            SpectrumDirection::None => continue,
        };
        
        // compute all pixels for this bar
        for y_offset in 0..bar_height {
            let plot_y = start_y + y_offset;
            for x_offset in 0..bar_width as u32 {
                let plot_x = x_pos + x_offset;
                if plot_x < width && plot_y < height {
                    bar_pixels.push((plot_x, plot_y, Rgb([255, 0, 0])));
                }
            }
            }
    }

    // batch process all bar pixels
    if !bar_pixels.is_empty() {
        let mut current_pixels: Vec<Rgb<u8>> = Vec::with_capacity(bar_pixels.len());
        let mut new_pixels: Vec<Rgb<u8>> = Vec::with_capacity(bar_pixels.len());
        let mut coords_to_blend: Vec<(u32, u32)> = Vec::with_capacity(bar_pixels.len()); 
        
        for (x, y, color) in &bar_pixels {
            current_pixels.push(*img.get_pixel(*x, *y));
            new_pixels.push(*color);
            coords_to_blend.push((*x, *y)); 
        }
        
        // batch blend operation
        let blended_pixels = image_simd::blend_pixels_batch(&current_pixels, &new_pixels, 0.2);
        
        // apply blended pixels
        for (_idx, (x, y)) in coords_to_blend.into_iter().enumerate() {
            img.put_pixel(x, y, blended_pixels[_idx]);
        }
    }
    img
}

// configures rayon thread pool
pub fn configure_thread_pool() {
    // get the number of physical cores (not logical/hyperthreaded)
    let num_physical_cores = num_cpus::get_physical();
    // reserve one core for system tasks if we have many cores
    let compute_cores = if num_physical_cores > 4 { num_physical_cores - 1 } else { num_physical_cores };

    // create a thread pool with optimal configuration
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(compute_cores)
        // large stack for compute-intensive work
        .stack_size(8 * 1024 * 1024)
        // pin threads to cores (if on linux)
        .thread_name(|i| format!("compute-{}", i))
        .build()
        .unwrap();

    // set as global pool
    thread_pool.install(|| {
        // set cpu affinity if on linux
        #[cfg(target_os = "linux")]
        {
            let core_ids = core_affinity::get_core_ids().unwrap();

            std::thread::spawn(move || {
                for (i, id) in core_ids.into_iter().enumerate() {
                    if i < compute_cores {
                        let _ = core_affinity::set_for_current(id);
                    }
                }
            });
        }
    });
}


// converts hsv to rgb

fn normalized_centroid_for_color(spectrum_data: &[f64]) -> f64 {
    let (sum_weighted_freq, sum_magnitudes) =
        unsafe { audio_simd::calculate_spectral_centroid_f64(spectrum_data, 1.0 / spectrum_data.len() as f64) };

    if sum_magnitudes > 0.0 {
        sum_weighted_freq / sum_magnitudes
    } else {
        0.0
    }
}

pub fn parse_spectrum_direction(arg: Option<&str>) -> SpectrumDirection {
    match arg {
        Some("ltr") => SpectrumDirection::Ltr,
        Some("rtl") => SpectrumDirection::Rtl,
        _ => SpectrumDirection::None,
    }
}

// conversion & printing with visualizer & ffmpeg

pub fn run_qoa_to_video<D, P>(
    audio_decoder: &D,
    quantum_processor: P,
    input_audio_path: &std::path::Path,
    output_video_path: &std::path::Path,
    fps: u32,
    width: u32,
    height: u32,
    extra_ffmpeg_args: &[&str],
    spectrum_direction: SpectrumDirection,
    ffmpeg_flags: &[String],
) -> Result<(), Box<dyn std::error::Error>>
where
    D: QoaAudioDecoder + Send + Sync,
    P: QuantumProcessor + Send + Sync,
{
    use num_complex::Complex;
    use rand::Rng;
    use std::time::Instant;
    use std::{
        io::Read,
        process::{Command, Stdio},
        sync::{Arc, Mutex, RwLock},
        thread,
    };

    info!("starting qoa to video conversion...");

    let (audio_samples, sample_rate) =
        audio_decoder.decode_audio_file_to_samples(input_audio_path)?;

    let total_audio_samples = audio_samples.len();
    let samples_per_frame = (sample_rate as f64 / fps as f64) as usize;
    let total_frames = (total_audio_samples as f64 / samples_per_frame as f64).ceil() as usize;

    info!(
        "audio length: {} samples at {} hz",
        total_audio_samples, sample_rate
    );
    info!(
        "generating {} frames at {} fps ({} samples/frame)",
        total_frames, fps, samples_per_frame
    );

    let frames_dir = output_video_path.with_extension("frames");
    std::fs::create_dir_all(&frames_dir)?;

    let mut real_fft_planner = RealFftPlanner::<f32>::new();
    let rfft = real_fft_planner.plan_fft_forward(samples_per_frame);
    let perlin_noise_gen_seed = rand::rng().random::<u32>();
    let perlin_noise = PerlinNoise::new(perlin_noise_gen_seed);

    // --- frame progress bar ---
    let progress_bar = ProgressBar::new(total_frames as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} (est. time left: {eta_precise}) {msg}")
            .unwrap()
            .progress_chars("##->"),
    );
    progress_bar.set_message("generating frames...");

    let quantum_processor_arc = Arc::new(Mutex::new(quantum_processor));
    let prev_rms_arc = Arc::new(RwLock::new(0.0));
    let start_time = Instant::now();
    (0..total_frames)
        .into_par_iter()
        .map(|i| {
            let start_sample_idx = i * samples_per_frame;
            let end_sample_idx = (start_sample_idx + samples_per_frame).min(total_audio_samples);
            let current_frame_samples = &audio_samples[start_sample_idx..end_sample_idx];

            let mut input_buffer = rfft.make_input_vec();
            input_buffer[..current_frame_samples.len()].copy_from_slice(current_frame_samples);

            let mut output_buffer = rfft.make_output_vec();
            rfft.process(&mut input_buffer, &mut output_buffer).unwrap();

            let spectrum_slice: &[Complex<f32>] = output_buffer.as_slice();
            let prev_rms = *prev_rms_arc.read().unwrap();

            let audio_features = AudioVisualizer::extract_enhanced_features(
                spectrum_slice,
                sample_rate,
                samples_per_frame,
                &audio_samples,
                start_sample_idx,
                prev_rms,
            );

            let quantum_data = quantum_processor_arc.lock().unwrap().process_frame(
                &audio_features,
                i,
                total_frames,
            );

            *prev_rms_arc.write().unwrap() = audio_features.rms_loudness;

            let frame_img = render_frame(
                i,
                width,
                height,
                &quantum_data,
                &perlin_noise,
                &audio_features.spectrum,
                spectrum_direction,
            );

            let frame_path = frames_dir.join(format!("frame_{:05}.png", i));
            frame_img
                .save_with_format(&frame_path, ImageFormat::Png)
                .expect("failed to save frame");
            progress_bar.inc(1);
        })
        .count();

    progress_bar.finish_with_message("frames generated.");
    let elapsed = start_time.elapsed();
    println!("total time: {:.2} seconds", elapsed.as_secs_f64());

    // --- check all frames exist to avoid ffmpeg hang ---
    let missing_frames: Vec<String> = (0..total_frames)
        .map(|i| frames_dir.join(format!("frame_{:05}.png", i)))
        .filter(|p| !p.exists())
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    if !missing_frames.is_empty() {
        eprintln!("error: the following frames are missing:");
        for f in missing_frames.iter().take(10) {
            eprintln!("  {}", f);
        }
        if missing_frames.len() > 10 {
            eprintln!("  ...and {} more", missing_frames.len() - 10);
        }
        return Err("aborting: missing png frames for ffmpeg".into());
    }

    let mut frame_files: Vec<_> = std::fs::read_dir(&frames_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
        .collect();
    frame_files.sort();
    println!(
        "first 3 frames: {:?}",
        &frame_files[..3.min(frame_files.len())]
    );
    println!(
        "last 3 frames: {:?}",
        &frame_files[frame_files.len().saturating_sub(3)..]
    );

    // --- ffmpeg command to stitch frames and audio into a video ---
    info!("stitching video with ffmpeg...");

    let fps_str_owned = fps.to_string();

    let frames_path_string = frames_dir
        .join("frame_%05d.png")
        .to_string_lossy()
        .into_owned();
    let audio_path_string = input_audio_path.to_string_lossy().into_owned();
    let output_video_path_string = output_video_path.to_string_lossy().into_owned();
    let resolution_str_owned = format!("{}x{}", width, height);
    let _ = std::fs::remove_file(&output_video_path);
    let mut ffmpeg_command: Vec<String> = vec![
        "-y".to_string(),
        "-nostdin".to_string(),
        "-start_number".to_string(),
        "0".to_string(),
        "-framerate".to_string(),
        fps_str_owned.clone(),
        "-i".to_string(),
        frames_path_string.clone(),
        "-i".to_string(),
        audio_path_string.clone(),
        "-map".to_string(),
        "0:v:0".to_string(),
        "-map".to_string(),
        "1:a:0".to_string(),
        "-c:a".to_string(),
        "aac".to_string(),
        "-b:a".to_string(),
        "192k".to_string(),
        "-c:v".to_string(),
        "libx264".to_string(),
        "-preset".to_string(),
        "fast".to_string(),
        "-crf".to_string(),
        "18".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-s".to_string(),
        resolution_str_owned.clone(),
    ];

    // append user args last
    for flag in ffmpeg_flags {
        ffmpeg_command.push(flag.clone());
    }
    for arg in extra_ffmpeg_args {
        ffmpeg_command.push(arg.to_string());
    }
    ffmpeg_command.push(output_video_path_string.clone());
    // print the exact ffmpeg command for debugging
    println!("ffmpeg command: ffmpeg {}", ffmpeg_command[1..].join(" "));

    // --- ffmpeg output: print on both \r and \n ---

    let mut child = Command::new("ffmpeg")
        .args(&ffmpeg_command[1..])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to start ffmpeg");

    let mut stderr = child.stderr.take().expect("failed to open ffmpeg stderr");

    let h = thread::spawn(move || {
        let mut buf = [0u8; 4096];
        let mut line_buf = Vec::new();
        while let Ok(n) = stderr.read(&mut buf) {
            if n == 0 {
                break;
            }
            for &b in &buf[..n] {
                if b == b'\n' || b == b'\r' {
                    if !line_buf.is_empty() {
                        let s = String::from_utf8_lossy(&line_buf);
                        println!("ffmpeg stderr: {}", s);
                        line_buf.clear();
                    }
                } else {
                    line_buf.push(b);
                }
            }
        }
        if !line_buf.is_empty() {
            let s = String::from_utf8_lossy(&line_buf);
            println!("ffmpeg stderr: {}", s);
        }
    });

    let status = child.wait()?;
    h.join().unwrap();

    if !status.success() {
        return Err(format!("ffmpeg failed with status: {:?}", status).into());
    } else {
        println!("ffmpeg finished successfully.");
        Ok(())
    }
}
