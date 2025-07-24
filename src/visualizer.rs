#[allow(unused_imports)]

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
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// use std::time::Instant; // i havent removed this import because i might use it later.

#[allow(unused_imports)] // allow unused imports for simd types, as they are conditionally compiled
use std::simd::{f32x8, f32x16};
use rand::rngs::ThreadRng; // import ThreadRng directly


// --- quantum noise implementation ---

#[derive(Clone)]
#[repr(align(64))] // add cache-line alignment for critical structs
pub struct PerlinNoise {
    #[allow(dead_code)]
    seed: u32,
    p: Vec<usize>,
}

impl PerlinNoise {
    fn new(seed: u32) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
        let mut p: Vec<usize> = (0..256).collect();
        // shuffle p
        for i in (0..256).rev() {
            let j = rng.random_range(0..=i); // Changed: use random_range()
            p.swap(i, j);
        }
        let mut extended_p = p.clone();
        extended_p.extend_from_slice(&p); // duplicate for wrapping
        PerlinNoise {
            seed,
            p: extended_p,
        }
    }

    fn fade(t: f64) -> f64 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn lerp(t: f64, a: f64, b: f64) -> f64 {
        a + t * (b - a)
    }

    // simd-accelerated perlin noise calculations - conceptual, actual implementation is complex
    // this would require converting f64 to f32 or using f64x2/f64x4 if available and redesigning `grad`
    // to operate on packed vectors.
    // example for avx2 (f32x8) if `packed_simd` offered an easier way to abstract gradient
    #[cfg(target_feature = "avx2")]
    #[allow(dead_code)] // this function is intentionally unused as it's a conceptual example
    fn grad_simd(hash: f32x8, x: f32x8, y: f32x8) -> f32x8 {
        let mut result = f32x8::splat(0.0);
        for i in 0..8 {
            result[i] = match hash[i] as usize & 0xf {
                0x0 => x[i] + y[i],
                0x1 => -x[i] + y[i],
                0x2 => x[i] - y[i],
                0x3 => -x[i] - y[i],
                0x4 => x[i] + x[i],
                0x5 => -x[i] + x[i],
                0x6 => x[i] - x[i],
                0x7 => -x[i] - x[i],
                _ => x[i] + y[i],
            };
        }
        result
    }

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
            _ => x + y, // should not happen with proper gradient vector
        }
    }

    fn get(&self, x: f64, y: f64) -> f64 {
        let x_int = x.floor() as usize;
        let y_int = y.floor() as usize;

        let x_frac = x - x.floor();
        let y_frac = y - y.floor();

        let u = Self::fade(x_frac);
        let v = Self::fade(y_frac);

        let a = self.p[x_int & 255] + (y_int & 255);
        let b = self.p[(x_int + 1) & 255] + (y_int & 255);

        let n00 = Self::grad(self.p[a] & 255, x_frac, y_frac);
        let n10 = Self::grad(self.p[b] & 255, x_frac - 1.0, y_frac);
        let n01 = Self::grad(self.p[a + 1] & 255, x_frac, y_frac - 1.0);
        let n11 = Self::grad(self.p[b + 1] & 255, x_frac - 1.0, y_frac - 1.0);

        Self::lerp(v, Self::lerp(u, n00, n10), Self::lerp(u, n01, n11))
    }
}

// --- public data structures ---

/// arguments for the `visual` subcommand (video visualization).
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

    /// extra ffmpeg flags (e.g., -s, -r, -b:v, -pix_fmt, etc.)
    #[arg(long = "ffmpeg-flag", value_name = "FFMPEG_FLAG", num_args = 0.., action = clap::ArgAction::Append)]
    pub ffmpeg_flags: Vec<String>,

    #[arg(last = true, trailing_var_arg = true)]
    pub ffmpeg_args: Vec<String>,

    pub input: String,
    pub output: String,
}
/// represents extracted audio features for a single frame.
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

/// represents the processed quantum state data and derived visual parameters.
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

    pub fn quantum_noise(&mut self, time: f64) -> (f64, Vec<bool>) {
        let mut quantum_state = self.quantum_state.lock(); // parking_lot mutex lock
        let current_time_val = self.current_time.fetch_add(1, Ordering::SeqCst) as usize; // atomic increment

        if current_time_val % self.coherence_time == 0 {
            *quantum_state = QuantumState::new(quantum_state.n, Some(NoiseConfig::Random));
        }

        let mut measurements = Vec::with_capacity(quantum_state.n); // pre-allocate vector, now mutable
        let n_qubits = quantum_state.n;
        for q in 0..n_qubits {
            // fix: correctly handle the result of quantum_state.measure(q)
            // it returns a result<usize, string>, so we need to unwrap it or handle the error
            // assuming a successful measurement returns 0 or 1
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
            .fold(0u64, |acc, (i, &val)| acc | (val as u64) << i);
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
    pub fn new() -> Self {
        Self {
            quantum_noise_gen: QuantumNoiseGenerator::new(16, 42),
            prev_rms_loudness: Arc::new(RwLock::new(0.0)),
            prev_hue: 0.0, // <-- initialize
        }
    }

    fn calculate_band_energy_from_mags(
        spectrum_mags: &[f64],
        sample_rate: u32,
        low_hz: f64,
        high_hz: f64,
    ) -> f64 {
        let nyquist_freq = sample_rate as f64 / 2.0;
        let bin_width = nyquist_freq / spectrum_mags.len() as f64;
        let start_bin = (low_hz / bin_width).floor() as usize;
        let end_bin = (high_hz / bin_width).ceil() as usize;
        spectrum_mags[start_bin..=end_bin.min(spectrum_mags.len() - 1)]
            .iter()
            .map(|&mag| mag * mag)
            .sum()
    }

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

            // using f32x16 for avx512f if available, f32x8 for avx2, f32x4 for sse4.1, otherwise scalar
            #[cfg(target_feature = "avx512f")]
            let sum_sq = samples_slice
                .chunks_exact(16)
                .map(|chunk| {
                    let p = f32x16::from_slice(chunk);
                    p.as_array().iter().map(|&val| val as f64 * val as f64).sum::<f64>()
                })
                .sum::<f64>()
                + samples_slice
                    .chunks_exact(16)
                    .remainder()
                    .iter()
                    .map(|&s| s as f64 * s as f64)
                    .sum::<f64>();
            #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
            let sum_sq = samples_slice
                .chunks_exact(8)
                .map(|chunk| {
                    let p = f32x8::from_slice(chunk);
                    p.as_array().iter().map(|&val| val as f64 * val as f64).sum::<f64>()
                })
                .sum::<f64>()
                + samples_slice
                    .chunks_exact(8)
                    .remainder()
                    .iter()
                    .map(|&s| s as f64 * s as f64)
                    .sum::<f64>();
            #[cfg(all(
                not(target_feature = "avx512f"),
                not(target_feature = "avx2"),
                target_feature = "sse4.1"
            ))]
            let sum_sq = samples_slice
                .chunks_exact(4)
                .map(|chunk| {
                    let p = f32x4::from_slice(chunk);
                    p.as_array().iter().map(|&val| val as f64 * val as f64).sum::<f64>()
                })
                .sum::<f64>()
                + samples_slice
                    .chunks_exact(4)
                    .remainder()
                    .iter()
                    .map(|&s| s as f64 * s as f64)
                    .sum::<f64>();
            #[cfg(not(any(
                target_feature = "sse4.1",
                target_feature = "avx2",
                target_feature = "avx512f"
            )))]
            let sum_sq = samples_slice
                .iter()
                .map(|&s| s as f64 * s as f64)
                .sum::<f64>();

            (sum_sq / samples_slice.len() as f64).sqrt()
        } else {
            0.0
        };
        let (mut sum_weighted_freq, mut sum_magnitudes) = (0.0, 0.0);
        for (i, &mag) in spectrum_mags.iter().enumerate() {
            sum_weighted_freq += i as f64 * bin_width * mag;
            sum_magnitudes += mag;
        }
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
        for (i, &mag) in spectrum_mags.iter().enumerate() {
            cumulative_energy += mag;
            if cumulative_energy >= 0.85 * total_energy {
                rolloff_bin = i;
                break;
            }
        }
        let spectral_rolloff = rolloff_bin as f64 * bin_width;

        // transient detection
        let rms_diff = (rms_loudness - prev_rms).max(0.0); // only interested in positive changes
        let transient_strength = (rms_diff * 50.0).min(1.0); // amplify and clamp

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
                self.quantum_noise_gen.quantum_state.lock().n, // use current n_qubits
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

                // optimize channel mixing operations (conceptual)
                // if we were reading raw samples, we could interleave and then process with simd
                #[cfg(any(
                    target_feature = "sse4.1",
                    target_feature = "avx2",
                    target_feature = "avx512f"
                ))]
                {
                    // convert all raw i16 samples to f32 once
                    let temp_f32_samples: Vec<f32> = all_raw_samples.clone() // clone here
                        .into_iter()
                        .map(|s| s as f32 / i16::MAX as f32)
                        .collect();

                    if num_channels == 2 {
                        // process stereo pairs with simd
                        #[cfg(target_feature = "avx512f")]
                        {
                            for chunk in temp_f32_samples.chunks_exact(16) {
                                let packed_samples = f32x16::from_slice(chunk);
                                // sum pairs: [l0, r0, l1, r1, ...] -> [(l0+r0), (l1+l1), ...]
                                let mixed_pairs = f32x8::from_array([
                                    (packed_samples[0] + packed_samples[1]) * 0.5,
                                    (packed_samples[2] + packed_samples[3]) * 0.5,
                                    (packed_samples[4] + packed_samples[5]) * 0.5,
                                    (packed_samples[6] + packed_samples[7]) * 0.5,
                                    (packed_samples[8] + packed_samples[9]) * 0.5,
                                    (packed_samples[10] + packed_samples[11]) * 0.5,
                                    (packed_samples[12] + packed_samples[13]) * 0.5,
                                    (packed_samples[14] + packed_samples[15]) * 0.5,
                                ]);
                                samples.extend_from_slice(mixed_pairs.as_array());
                            }
                            // handle remainder
                            for pair in temp_f32_samples.chunks_exact(16).remainder().chunks(2) {
                                if let (Some(&l), Some(&r)) = (pair.get(0), pair.get(1)) {
                                    samples.push((l + r) * 0.5);
                                } else if let Some(&l) = pair.get(0) {
                                    samples.push(l);
                                }
                            }
                        }
                        #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
                        {
                            for chunk in temp_f32_samples.chunks_exact(8) {
                                let packed_samples = f32x8::from_slice(chunk);
                                // sum pairs: [l0, r0, l1, r1, ...] -> [(l0+r0), (l1+r1), ...]
                                let mixed_pairs = f32x4::from_array([
                                    (packed_samples[0] + packed_samples[1]) * 0.5,
                                    (packed_samples[2] + packed_samples[3]) * 0.5,
                                    (packed_samples[4] + packed_samples[5]) * 0.5,
                                    (packed_samples[6] + packed_samples[7]) * 0.5,
                                ]);
                                samples.extend_from_slice(mixed_pairs.as_array());
                            }
                            // handle remainder
                            for pair in temp_f32_samples.chunks_exact(8).remainder().chunks(2) {
                                if let (Some(&l), Some(&&r)) = (pair.get(0), pair.get(1)) {
                                    samples.push((l + r) * 0.5);
                                } else if let Some(&l) = pair.get(0) {
                                    samples.push(l);
                                }
                            }
                        }
                        #[cfg(all(
                            not(target_feature = "avx512f"),
                            not(target_feature = "avx2"),
                            target_feature = "sse4.1"
                        ))]
                        {
                            for chunk in temp_f32_samples.chunks_exact(4) {
                                let packed_samples = f32x4::from_slice(chunk);
                                // sum pairs: [l0, r0, l1, r1, ...] -> [(l0+r0), (l1+r1), ...]
                                let mixed_pairs = f32x2::from_array([
                                    (packed_samples[0] + packed_samples[1]) * 0.5,
                                    (packed_samples[2] + packed_samples[3]) * 0.5,
                                ]);
                                samples.extend_from_slice(mixed_pairs.as_array());
                            }
                            // handle remainder
                            for pair in temp_f32_samples.chunks_exact(4).remainder().chunks(2) {
                                if let (Some(&l), Some(&r)) = (pair.get(0), pair.get(1)) {
                                    samples.push((l + r) * 0.5);
                                } else if let Some(&l) = pair.get(0) {
                                    samples.push(l);
                                }
                            }
                        }
                    } else if num_channels == 1 {
                        info!("processing mono audio");
                        samples = temp_f32_samples; // directly use if mono
                    } else {
                        info!("processing stereo/multi-channel audio (averaging channels)");
                        let mut temp_all_samples: Vec<f32> =
                            Vec::with_capacity(all_raw_samples.len() * num_channels);
                        for &sample in &all_raw_samples {
                            temp_all_samples.push(sample as f32 / i16::MAX as f32);
                        }

                        // average all channels for each sample
                        for chunk in temp_all_samples.chunks(num_channels) {
                            samples.push(chunk.iter().sum::<f32>() / num_channels as f32);
                        }
                    }
                }
                #[cfg(not(any(
                    target_feature = "sse4.1",
                    target_feature = "avx2",
                    target_feature = "avx512f"
                )))]
                {
                    // scalar fallback for channel mixing
                    if num_channels == 2 {
                        info!("processing stereo audio (scalar)");
                        for sample_pair in all_raw_samples.chunks(2) {
                            let left =
                                sample_pair.get(0).copied().unwrap_or(0) as f32 / i16::MAX as f32;
                            let right =
                                sample_pair.get(1).copied().unwrap_or(0) as f32 / i16::MAX as f32;
                            samples.push((left + right) * 0.5);
                        }
                    } else if num_channels == 1 {
                        info!("processing mono audio (scalar)");
                        for &sample in &all_raw_samples {
                            samples.push(sample as f32 / i16::MAX as f32);
                        }
                    } else {
                        info!("processing multi-channel audio (scalar, averaging channels)");
                        for chunk in all_raw_samples.chunks(num_channels) {
                            let mut sum = 0.0f32;
                            let mut count = 0;
                            for i in 0..num_channels.min(2) {
                                // only average first two channels
                                if let Some(&sample) = chunk.get(i) {
                                    sum += sample as f32;
                                    count += 1;
                                }
                            }
                            if count > 0 {
                                samples.push((sum / count as f32) / i16::MAX as f32);
                            }
                        }
                    }
                }
                Ok((samples, spec.sample_rate))
            }
            Some("qoa") => {
                info!("decoding qoa audio");
                let file_path = Path::new(audio_path);
                let file = fs::File::open(file_path)?;
                let _mmap = unsafe { MmapOptions::new().map(&file)? };
                Err(format!("qoa decoding is unavailable due to missing function `decode_to_vec_f32` in the `qoa` crate. please check your `qoa` dependency version and features.").into())
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
            .min(1.0).max(0.0);
        let _normalized_high = (audio_features.high_freq_energy / total_energy)
            .min(1.0).max(0.0);

        let base_brightness = 1.0 + normalized_bass * 1.0 + normalized_rms * 0.3;

        // --- colour shift, and cycle ---
        let increment_per_frame = 0.01; // this is a good value, 1 is seizure inducing
        let color_hue = (self.prev_hue + increment_per_frame) % 360.0; // colour hue %
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

// --- main visualization logic ---
pub fn render_frame(
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

    let _total_pixels = width * height;
    let _quantum_bits_to_pixel_ratio =
        (quantum_data.quantum_measurements.len() as f64 / _total_pixels as f64).sqrt();

    for y in 0..height {
        for x in 0..width {
            // base coordinates
            let mut norm_x = (x as f64 - center_x) / width as f64;
            let mut norm_y = (y as f64 - center_y) / height as f64;

            // apply flow field strength
            let flow_x = (norm_x + quantum_data.flow_field_strength * 0.1).sin() * 0.05;
            let flow_y = (norm_y + quantum_data.flow_field_strength * 0.1).cos() * 0.05;
            norm_x += flow_x;
            norm_y += flow_y;

            let noise_val = perlin_noise.get(
                norm_x * quantum_data.pattern_density * 10.0
                    + quantum_data.noise_seed as f64 / 1000.0
                    + quantum_data.chaos_factor * 5.0, // incorporate chaos_factor
                norm_y * quantum_data.pattern_density * 10.0
                    + quantum_data.noise_seed as f64 / 1000.0
                    + quantum_data.chaos_factor * 5.0, // incorporate chaos_factor
            );

            // quantum interference pattern
            let interference = (norm_x * 10.0).sin() * (norm_y * 10.0).cos() * quantum_data.interference_pattern;
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
                + quantum_data.quantum_coherence * 90.0) // incorporate quantum_coherence
                % 360.0;
            let saturation = (0.7 + quantum_data.quantum_entanglement * 0.3).min(1.0);
            let value = final_brightness;

            let rgb_color = hsv_to_rgb(hue, saturation, value);

            // apply distortion
            let distorted_x = (x as f64 + (norm_x * quantum_data.distortion_magnitude * 50.0).sin()
                + quantum_data.depth_modulation * 20.0 * (noise_val * 2.0).sin()) as u32; // incorporate depth_modulation
            let distorted_y = (y as f64 + (norm_y * quantum_data.distortion_magnitude * 50.0).cos()
                + quantum_data.depth_modulation * 20.0 * (noise_val * 2.0).cos()) as u32; // incorporate depth_modulation

            let current_pixel: &mut Rgb<u8> = img.get_pixel_mut(
                // added type annotation
                distorted_x.min(width - 1),
                distorted_y.min(height - 1),
            );

            let new_pixel = Rgb([
                (rgb_color.0 * 255.0) as u8,
                (rgb_color.1 * 255.0) as u8,
                (rgb_color.2 * 255.0) as u8,
            ]);

            // simple blending (alpha compositing concept)
            let blended_pixel = Rgb([
                ((current_pixel[0] as f32 * 0.5 + new_pixel[0] as f32 * 0.5) as u8),
                ((current_pixel[1] as f32 * 0.5 + new_pixel[1] as f32 * 0.5) as u8),
                ((current_pixel[2] as f32 * 0.5 + new_pixel[2] as f32 * 0.5) as u8),
            ]);
            img.put_pixel(
                distorted_x.min(width - 1),
                distorted_y.min(height - 1),
                blended_pixel,
            );
        }
    }

    // spectrum visualization
    let spectrum_height = height / 4;
    let bar_width = width as f64 / spectrum_data.len() as f64;

    for (i, &magnitude) in spectrum_data.iter().enumerate() {
        let bar_height = (magnitude * 500.0).min(spectrum_height as f64);
        let start_y = match spectrum_direction {
            SpectrumDirection::Ltr => height - bar_height as u32,
            SpectrumDirection::Rtl => 0, // top of the image for rtl for now, can be adjusted
            SpectrumDirection::None => continue, // no spectrum rendering
        };

        let x_pos = match spectrum_direction {
            SpectrumDirection::Ltr => (i as f64 * bar_width) as u32,
            SpectrumDirection::Rtl => width - (i as f64 * bar_width) as u32 - bar_width as u32, // adjust for rtl
            SpectrumDirection::None => continue,
        };

        for y_offset in 0..(bar_height as u32) {
            let plot_y = start_y + y_offset;
            for x_offset in 0..(bar_width as u32) {
                let plot_x = x_pos + x_offset;
                if plot_x < width && plot_y < height {
                    let blended_pixel = Rgb([
                        (img.get_pixel(plot_x, plot_y)[0] as f32 * 0.8 + 255.0 * 0.2) as u8,
                        (img.get_pixel(plot_x, plot_y)[1] as f32 * 0.8 + 0.0 * 0.2) as u8,
                        (img.get_pixel(plot_x, plot_y)[2] as f32 * 0.8 + 0.0 * 0.2) as u8,
                    ]);
                    img.put_pixel(plot_x, plot_y, blended_pixel);
                }
            }
        }
    }
    img
}

// converts hsv to rgb.
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
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

fn normalized_centroid_for_color(spectrum_data: &[f64]) -> f64 {
    let mut sum_weighted_freq = 0.0;
    let mut sum_magnitudes = 0.0;
    for (i, &mag) in spectrum_data.iter().enumerate() {
        // use a simpler mapping for color, e.g., 0-1 range for frequency
        let normalized_freq = i as f64 / spectrum_data.len() as f64;
        sum_weighted_freq += normalized_freq * mag;
        sum_magnitudes += mag;
    }
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

// important function below related to conversion & printing with visualizer & ffmpeg,
// do not touch unless you know what you are doing!

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
    use rand::Rng; // keep this import for Rng trait
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

    let perlin_noise_gen_seed = ThreadRng::default().random::<u32>(); // Changed: use random()
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
        "-y".to_string(), // <-- must be first!!!
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
        .stdout(Stdio::null()) // we don't need ffmpeg stdout, only stderr for progress
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
