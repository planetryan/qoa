use duct::cmd;
use image::{ImageBuffer, ImageFormat, Rgb, Pixel};
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn, error};
use ndarray::Array1;
use num_complex::Complex;
use qoa::runtime::quantum_state::{NoiseConfig, QuantumState};
use rayon::prelude::*;
use realfft::RealFftPlanner;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// --- quantum noise implementation ---
#[derive(Clone)]
struct PerlinNoise {
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
            let j = rng.gen_range(0..=i);
            p.swap(i, j);
        }
        let mut extended_p = p.clone();
        extended_p.extend_from_slice(&p); // duplicate for wrapping
        PerlinNoise { seed, p: extended_p }
    }

    fn fade(t: f64) -> f64 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn lerp(t: f64, a: f64, b: f64) -> f64 {
        a + t * (b - a)
    }

    fn grad(hash: usize, x: f64, y: f64) -> f64 {
        match hash & 0xf {
            0x0 => x + y, 0x1 => -x + y, 0x2 => x - y, 0x3 => -x - y,
            0x4 => x + x, 0x5 => -x + x, 0x6 => x - x, 0x7 => -x - x,
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

    #[arg(last = true, trailing_var_arg = true)]
    pub ffmpeg_args: Vec<String>,

    pub input: String,
    pub output: String,
}

/// represents extracted audio features for a single frame.
#[derive(Debug, Default, Clone)]
pub struct AudioFeatures {
    pub rms_loudness: f64,
    pub spectral_centroid: f64,
    pub low_freq_energy: f64,
    pub mid_freq_energy: f64,
    pub high_freq_energy: f64,
    pub spectrum: Vec<f64>,
    pub peak_frequency: f64,
    pub spectral_rolloff: f64,
    pub transient_strength: f64, // new field for detecting sharp changes
}

/// represents the processed quantum state data and derived visual parameters.
#[derive(Debug, Default, Clone)]
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
#[derive(Clone)]
pub struct QuantumNoiseGenerator {
    pub quantum_state: Arc<Mutex<QuantumState>>,
    pub coherence_time: usize,
    pub current_time: usize,
}

impl QuantumNoiseGenerator {
    pub fn new(n_qubits: usize, _seed: u64) -> Self {
        Self {
            quantum_state: Arc::new(Mutex::new(QuantumState::new(n_qubits, Some(NoiseConfig::Random)))),
            coherence_time: 30,
            current_time: 0,
        }
    }

    pub fn quantum_noise(&mut self, time: f64) -> (f64, Vec<bool>) {
        let mut quantum_state = self.quantum_state.lock().unwrap();
        if self.current_time % self.coherence_time == 0 {
            *quantum_state = QuantumState::new(quantum_state.n, Some(NoiseConfig::Random));
        }
        self.current_time += 1;
        let mut measurements = Vec::new();
        let n_qubits = quantum_state.n;
        for q in 0..n_qubits {
            measurements.push(quantum_state.measure(q) == 1);
        }
        let quantum_bits = measurements.iter().enumerate()
            .fold(0u64, |acc, (i, &val)| acc | (val as u64) << i);
        ( (quantum_bits.count_ones() as f64 / n_qubits as f64 * 0.8 + (time * 0.1).sin() * 0.2) * 2.0 - 1.0, measurements )
    }
}

// --- traits for abstraction ---

pub trait QoaAudioDecoder {
    fn decode_audio_file_to_samples(&self, audio_path: &Path) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>>;
}

pub trait QuantumProcessor {
    fn process_frame(&mut self, audio_features: &AudioFeatures, frame_index: usize, total_frames: usize) -> QuantumVisualData;
}

// --- implementation ---

#[derive(Clone)]
pub struct AudioVisualizer {
    quantum_noise_gen: QuantumNoiseGenerator,
    #[allow(dead_code)] // this field is accessed via Arc<Mutex>, which the linter doesn't catch as a "read"
    prev_rms_loudness: Arc<Mutex<f64>>, // to detect transients
}

impl AudioVisualizer {
    pub fn new() -> Self {
        Self {
            quantum_noise_gen: QuantumNoiseGenerator::new(16, 42),
            prev_rms_loudness: Arc::new(Mutex::new(0.0)),
        }
    }

    fn calculate_band_energy_from_mags(spectrum_mags: &[f64], sample_rate: u32, low_hz: f64, high_hz: f64) -> f64 {
        let nyquist_freq = sample_rate as f64 / 2.0;
        let bin_width = nyquist_freq / spectrum_mags.len() as f64;
        let start_bin = (low_hz / bin_width).floor() as usize;
        let end_bin = (high_hz / bin_width).ceil() as usize;
        spectrum_mags[start_bin..=end_bin.min(spectrum_mags.len() - 1)].iter().map(|&mag| mag * mag).sum()
    }

    fn extract_enhanced_features(spectrum: &[Complex<f32>], sample_rate: u32, window_samples: usize, audio_samples: &[f32], start_idx: usize, prev_rms: f64) -> AudioFeatures {
        let spectrum_mags: Vec<f64> = spectrum.iter().map(|c| c.norm() as f64).collect();
        let nyquist_freq = sample_rate as f64 / 2.0;
        let bin_width = nyquist_freq / spectrum.len() as f64;
        let rms_loudness = if window_samples > 0 {
            let end_idx = (start_idx + window_samples).min(audio_samples.len());
            let samples_slice = &audio_samples[start_idx..end_idx];
            (samples_slice.iter().map(|&s| s as f64 * s as f64).sum::<f64>() / samples_slice.len() as f64).sqrt()
        } else { 0.0 };
        let (mut sum_weighted_freq, mut sum_magnitudes) = (0.0, 0.0);
        for (i, &mag) in spectrum_mags.iter().enumerate() {
            sum_weighted_freq += i as f64 * bin_width * mag;
            sum_magnitudes += mag;
        }
        let spectral_centroid = if sum_magnitudes > 0.0 { sum_weighted_freq / sum_magnitudes } else { 0.0 };
        let low_freq_energy = Self::calculate_band_energy_from_mags(&spectrum_mags, sample_rate, 0.0, 200.0);
        let mid_freq_energy = Self::calculate_band_energy_from_mags(&spectrum_mags, sample_rate, 200.0, 2000.0);
        let high_freq_energy = Self::calculate_band_energy_from_mags(&spectrum_mags, sample_rate, 2000.0, 20000.0);
        let peak_bin = spectrum_mags.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i).unwrap_or(0);
        let peak_frequency = peak_bin as f64 * bin_width;
        let total_energy: f64 = spectrum_mags.iter().sum();
        let mut cumulative_energy = 0.0;
        let mut rolloff_bin = 0;
        for (i, &mag) in spectrum_mags.iter().enumerate() {
            cumulative_energy += mag;
            if cumulative_energy >= 0.85 * total_energy { rolloff_bin = i; break; }
        }
        let spectral_rolloff = rolloff_bin as f64 * bin_width;

        // transient detection
        let rms_diff = (rms_loudness - prev_rms).max(0.0); // only interested in positive changes
        let transient_strength = (rms_diff * 50.0).min(1.0); // amplify and clamp

        AudioFeatures {
            rms_loudness, spectral_centroid, low_freq_energy, mid_freq_energy, high_freq_energy,
            spectrum: spectrum_mags, peak_frequency, spectral_rolloff, transient_strength
        }
    }
}

impl QoaAudioDecoder for AudioVisualizer {
    fn decode_audio_file_to_samples(&self, audio_path: &Path) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
        info!("decoding audio from: {:?}", audio_path);
        match audio_path.extension().and_then(|s| s.to_str()) {
            Some("wav") => {
                let mut reader = hound::WavReader::open(audio_path)?;
                let spec = reader.spec();
                if spec.channels > 1 { info!("audio has {} channels, averaging to mono. will only process the first channel for simplicity.", spec.channels); }
                let mut samples = Vec::new();
                for s in reader.samples::<i16>() {
                    // for multi-channel, we only take the first channel's data
                    // to keep the sample extraction simple and efficient.
                    // if averaging channels is desired, it should be done here.
                    // current implementation averages if channels_samples.len() == spec.channels, which effectively makes it mono.
                    // for now, keeping it as is, but noting that only first channel is used if it's strictly multi-channel
                    samples.push(s.unwrap() as f32 / i16::MAX as f32);
                }
                Ok((samples, spec.sample_rate))
            },
            Some("csv") => {
                let content = fs::read_to_string(audio_path)?;
                let mut lines = content.lines();
                let header_line = lines.next().ok_or("csv is empty")?;
                let headers: Vec<&str> = header_line.split(',').collect();
                let price_col_idx = headers.iter().position(|&h| h.trim().to_lowercase() == "price (eur)").ok_or("csv header 'price (eur)' not found")?;
                let mut samples = Vec::new();
                for (i, line) in lines.enumerate() {
                    if let Some(price_str) = line.split(',').nth(price_col_idx) {
                        samples.push(price_str.trim().parse::<f32>().unwrap_or_else(|_| {
                            warn!("could not parse price on line {}", i + 2);
                            0.0
                        }));
                    }
                }
                Ok((samples, 44100))
            },
            Some(ext) => Err(format!("unsupported format: {}", ext).into()),
            None => Err("file has no extension".into()),
        }
    }
}

impl QuantumProcessor for AudioVisualizer {
    fn process_frame(&mut self, audio_features: &AudioFeatures, frame_index: usize, _total_frames: usize) -> QuantumVisualData {
        let time = frame_index as f64 * 0.033;
        let (quantum_noise, quantum_measurements) = self.quantum_noise_gen.quantum_noise(time);
        let ones_count = quantum_measurements.iter().filter(|&&x| x).count();
        let quantum_coherence = (ones_count as f64 / quantum_measurements.len() as f64 - 0.5).abs() * 2.0;

        let base_brightness = ((audio_features.rms_loudness * 1.5 + audio_features.low_freq_energy * 1.5).powf(0.5) * (1.0 + quantum_noise.abs() * 0.2)).min(1.0);

        // more audio-responsive color hue
        let spectral_hue_factor = (audio_features.peak_frequency / 10000.0).min(1.0); // normalize peak freq to 0-1
        let energy_ratio_hue_factor = (audio_features.high_freq_energy / (audio_features.low_freq_energy + audio_features.mid_freq_energy + audio_features.high_freq_energy + 1e-6)).min(1.0);
        let color_hue = ((time * 0.05).fract() // base time evolution
            + audio_features.rms_loudness * 0.2 // general loudness adds to hue shift
            + quantum_noise * 0.1 // quantum influence
            + (audio_features.spectral_centroid / 10000.0).min(1.0) * 0.1 // spectral centroid
            + spectral_hue_factor * 0.3 // peak frequency influence
            + energy_ratio_hue_factor * 0.2 // high frequency ratio influence
        ).fract();

        let total_energy = audio_features.low_freq_energy + audio_features.mid_freq_energy + audio_features.high_freq_energy;
        let energy_ratios = if total_energy > 0.0 { (audio_features.low_freq_energy / total_energy, audio_features.mid_freq_energy / total_energy, audio_features.high_freq_energy / total_energy) } else { (0.33, 0.33, 0.34) };
        let spectral_complexity = if audio_features.spectrum.len() > 10 { (audio_features.spectrum.iter().map(|&x| (x - audio_features.spectrum.iter().sum::<f64>() / audio_features.spectrum.len() as f64).powi(2)).sum::<f64>() / audio_features.spectrum.len() as f64).sqrt() * 0.1 } else { 0.5 }.min(1.0);

        QuantumVisualData {
            base_brightness, color_hue,
            pattern_density: ((energy_ratios.1 * 0.5 + spectral_complexity * 0.3 + quantum_coherence * 0.2) * 1.5).min(1.0).max(0.0),
            distortion_magnitude: ((energy_ratios.2 * 0.6 + audio_features.rms_loudness * 0.2 + quantum_noise.abs() * 0.2) * 1.2).min(1.0).max(0.0),
            flicker_intensity: (audio_features.high_freq_energy * 5.0).min(1.0) + audio_features.transient_strength * 2.0, // flicker with transients
            noise_seed: quantum_measurements.iter().enumerate().fold(frame_index as u64, |acc, (i, &val)| acc ^ ((val as u64) << (i % 64))),
            chaos_factor: (quantum_coherence * 0.4 + quantum_noise.abs() * 0.3 + audio_features.transient_strength * 0.5).min(0.8), // chaos with transients
            interference_pattern: (quantum_coherence * 0.5 + spectral_complexity * 0.5).min(1.0),
            quantum_entanglement: (quantum_coherence * 0.6 + energy_ratios.1 * 0.2).min(1.0),
            quantum_measurements,
            quantum_coherence: quantum_coherence,
            flow_field_strength: (audio_features.rms_loudness * 2.0 + audio_features.spectral_centroid / 5000.0).min(1.0),
            depth_modulation: (audio_features.low_freq_energy * 3.0).min(1.0),
        }
    }
}

// --- hsv to rgb conversion ---
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [u8; 3] {
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let (r_prime, g_prime, b_prime) = if (0.0..1.0).contains(&h_prime) { (c, x, 0.0) }
    else if (1.0..2.0).contains(&h_prime) { (x, c, 0.0) } else if (2.0..3.0).contains(&h_prime) { (0.0, c, x) }
    else if (3.0..4.0).contains(&h_prime) { (0.0, x, c) } else if (4.0..5.0).contains(&h_prime) { (x, 0.0, c) }
    else if (5.0..6.0).contains(&h_prime) { (c, 0.0, x) } else { (0.0, 0.0, 0.0) };
    let m = v - c;
    [((r_prime + m) * 255.0).round() as u8, ((g_prime + m) * 255.0).round() as u8, ((b_prime + m) * 255.0).round() as u8]
}

// --- visualization functions ---

// this function will now primarily set the background to black
fn set_black_background(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, _width: u32, _height: u32) {
    // using `fill` is faster than iterating pixel by pixel
    img.pixels_mut().for_each(|p| *p = Rgb([0, 0, 0]));
}

fn generate_energetic_particles(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, quantum_data: &QuantumVisualData, audio_features: &AudioFeatures, width: u32, height: u32, perlin_noise: &PerlinNoise) {
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;

    // particle count based on overall audio energy and quantum pattern density
    let base_particle_count = 500;
    let audio_driven_particles = (audio_features.rms_loudness * 2000.0) as usize;
    let pattern_driven_particles = (quantum_data.pattern_density * 1000.0) as usize; // use pattern_density
    let additional_particles_from_transient = (audio_features.transient_strength * 1000.0) as usize;
    let particle_count = base_particle_count + audio_driven_particles + pattern_driven_particles + additional_particles_from_transient;

    let mut particle_rng_seed = quantum_data.noise_seed;
    for &measurement in &quantum_data.quantum_measurements {
        particle_rng_seed = particle_rng_seed.wrapping_add(measurement as u64);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(particle_rng_seed);

    // pre-calculate common factors outside the loop
    let width_f64 = width as f64;
    let height_f64 = height as f64;

    for _ in 0..particle_count {
        // particles originate from a small central area
        let initial_x = center_x + (rng.gen::<f64>() - 0.5) * width_f64 * 0.05;
        let initial_y = center_y + (rng.gen::<f64>() - 0.5) * height_f64 * 0.05;

        let mut x = initial_x;
        let mut y = initial_y;

        // determine initial direction radially outwards
        let dx = x - center_x;
        let dy = y - center_y;
        let mut angle = dy.atan2(dx);
        if dx == 0.0 && dy == 0.0 { // handle exact center case
            angle = rng.gen_range(0.0..std::f64::consts::PI * 2.0);
        }

        // speed influenced by overall loudness, high-frequency energy, and distortion_magnitude
        let base_speed = 3.0;
        let audio_speed_boost = (audio_features.rms_loudness * 10.0 + audio_features.high_freq_energy * 5.0).min(20.0);
        let distortion_speed_boost = quantum_data.distortion_magnitude * 15.0; // use distortion_magnitude
        let speed = base_speed + audio_speed_boost + distortion_speed_boost;

        let (mut vx, mut vy) = (angle.cos() * speed, angle.sin() * speed);

        // particle lifespan based on audio features and interference_pattern
        let base_lifespan = 20.0;
        let lifespan_boost = (audio_features.rms_loudness * 50.0).min(50.0) + (audio_features.spectral_rolloff / 1000.0).min(30.0);
        let interference_lifespan_mod = quantum_data.interference_pattern * 30.0; // use interference_pattern
        let lifespan = base_lifespan + lifespan_boost + interference_lifespan_mod;

        // particle color based on hue and spectral centroid, with blend towards white/orange
        // integrate base_brightness and quantum_entanglement into hue/saturation/value
        let h_base = quantum_data.color_hue * 360.0; // base hue from quantum data
        let h_offset_spectral = (audio_features.spectral_centroid / 5000.0).min(1.0) * 60.0; // shift hue based on spectral centroid (e.g., green to yellow)
        let entanglement_hue_shift = quantum_data.quantum_entanglement * 30.0; // use quantum_entanglement
        let particle_hue = (h_base + h_offset_spectral + entanglement_hue_shift).fract(); // keep hue within 0-1 range

        // saturation and value for warm colors (orange, white)
        let saturation_base = 0.8;
        let value_base = 0.8;

        // dynamic saturation and value based on audio and quantum_coherence
        let dynamic_saturation = (saturation_base + audio_features.rms_loudness * 0.2 + quantum_data.quantum_coherence * 0.1).min(1.0); // use quantum_coherence
        let dynamic_value = (value_base + audio_features.high_freq_energy * 0.2 + quantum_data.base_brightness * 0.1).min(1.0); // use base_brightness

        // pre-calculate values for speed
        let wobble_strength = quantum_data.chaos_factor * 0.5;
        let flow_influence = quantum_data.flow_field_strength * 0.1;
        let low_freq_energy_size_factor = audio_features.low_freq_energy * 50.0;
        let depth_modulation_size_factor = quantum_data.depth_modulation * 20.0;


        for i in 0..lifespan as usize {
            // check bounds at the start of loop for early exit
            if x < -10.0 || x >= width_f64 + 10.0 || y < -10.0 || y >= height_f64 + 10.0 { break; } // add buffer for larger particles

            let life_progress = i as f64 / lifespan;

            // trail intensity/alpha fades with life and influenced by audio and flicker_intensity
            let alpha = (1.0 - life_progress) * (audio_features.rms_loudness * 3.0).min(1.0) * (1.0 + quantum_data.flicker_intensity * (rng.gen::<f64>() - 0.5)).min(1.0).max(0.0); // use flicker_intensity

            // color morphing from orange to white (or similar)
            let current_hue = (particle_hue + (1.0 - life_progress) * 0.05).fract(); // slight hue shift over life
            let current_saturation = dynamic_saturation * (1.0 - life_progress * 0.7); // saturation decreases for whiter trails
            let current_value = dynamic_value * (0.5 + life_progress * 0.5); // value increases as it fades to white

            let new_rgb_color = hsv_to_rgb(current_hue * 360.0, current_saturation, current_value);

            // particle size/intensity linked to low frequency energy and depth_modulation
            let size = (low_freq_energy_size_factor + depth_modulation_size_factor).min(10.0);

            // apply slight perlin noise based "wobble" to particle path, influenced by chaos_factor and flow_field_strength
            let wobble_x = perlin_noise.get(x * 0.01 + i as f64 * 0.01, y * 0.01) * wobble_strength + perlin_noise.get(x * 0.005, y * 0.005 + i as f64 * 0.005) * flow_influence;
            let wobble_y = perlin_noise.get(y * 0.01 + i as f64 * 0.01, x * 0.01) * wobble_strength + perlin_noise.get(y * 0.005, x * 0.005 + i as f64 * 0.005) * flow_influence;

            vx += wobble_x;
            vy += wobble_y;

            // clamp velocity to maintain general outward direction
            let current_dist_from_center_sq = (x - center_x).powi(2) + (y - center_y).powi(2);
            if current_dist_from_center_sq > 0.01 { // avoid division by zero at center, use squared distance
                let current_dist_from_center = current_dist_from_center_sq.sqrt();
                let direction_x = (x - center_x) / current_dist_from_center;
                let direction_y = (y - center_y) / current_dist_from_center;
                let dot_product = vx * direction_x + vy * direction_y;
                if dot_product < 0.0 { // if moving inwards, gently push outwards
                    vx -= direction_x * dot_product * 0.1;
                    vy -= direction_y * dot_product * 0.1;
                }
            }

            if size > 0.5 { // draw as a line segment for trails or larger particles
                let prev_x = x - vx;
                let prev_y = y - vy;

                // draw steps proportional to distance moved, minimum 1
                let steps = (vx.abs().max(vy.abs()) * 2.0).ceil() as i32; // increase steps for smoother trails
                for s in 0..=steps {
                    let interp_x = prev_x + (vx / steps as f64) * s as f64;
                    let interp_y = prev_y + (vy / steps as f64) * s as f64;

                    let draw_alpha = alpha * (1.0 - (s as f64 / steps as f64) * 0.5); // fade trail

                    // iterate over a square area for thickness, optimized for bounds
                    let half_size_i = (size / 2.0).round() as i32;
                    let plot_x_start = (interp_x - half_size_i as f64).round() as i32;
                    let plot_y_start = (interp_y - half_size_i as f64).round() as i32;
                    let plot_x_end = (interp_x + half_size_i as f64).round() as i32;
                    let plot_y_end = (interp_y + half_size_i as f64).round() as i32;

                    for py_i32 in plot_y_start.max(0)..=(plot_y_end.min(height as i32 - 1)) {
                        for px_i32 in plot_x_start.max(0)..=(plot_x_end.min(width as i32 - 1)) {
                            let (px_u, py_u) = (px_i32 as u32, py_i32 as u32);
                            let current_pixel = img.get_pixel(px_u, py_u);
                            let blended_pixel = Rgb(current_pixel.channels().iter().zip(new_rgb_color.iter()).map(|(&old, &new)| {
                                ((old as f64 * (1.0 - draw_alpha)) + new as f64 * draw_alpha).round() as u8
                            }).collect::<Vec<u8>>().try_into().unwrap_or([0,0,0]));
                            img.put_pixel(px_u, py_u, blended_pixel);
                        }
                    }
                }
            } else { // draw as a single pixel for very small particles
                let plot_x = x.round() as u32;
                let plot_y = y.round() as u32;
                if plot_x < width && plot_y < height {
                    let current_pixel = img.get_pixel(plot_x, plot_y);
                    let blended_pixel = Rgb(current_pixel.channels().iter().zip(new_rgb_color.iter()).map(|(&old, &new)| {
                        ((old as f64 * (1.0 - alpha)) + new as f64 * alpha).round() as u8
                    }).collect::<Vec<u8>>().try_into().unwrap_or([0,0,0]));
                    img.put_pixel(plot_x, plot_y, blended_pixel);
                }
            }

            x += vx;
            y += vy;
        }
    }
}


// --- helper function to parse spectrum direction ---
pub fn parse_spectrum_direction(direction: Option<&str>) -> SpectrumDirection {
    match direction {
        Some("ltr") | Some("left-to-right") => SpectrumDirection::Ltr,
        Some("rtl") | Some("right-to-left") => SpectrumDirection::Rtl,
        _ => SpectrumDirection::None,
    }
}

// --- main visualization function ---
pub fn run_qoa_to_video(
    audio_decoder: &dyn QoaAudioDecoder, quantum_processor: impl QuantumProcessor + Send + Sync + 'static,
    input_audio_path: &Path, output_video_path: &Path, mut fps: u32, mut width: u32, mut height: u32,
    extra_ffmpeg_args: &[&str], spectrum_direction: SpectrumDirection,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("starting qoa video visualization process.");
    info!("input audio: {:?}, output video: {:?}", input_audio_path, output_video_path);

    let mut clean_ffmpeg_args: Vec<String> = Vec::new(); // changed to String to own the data
    let mut i = 0;
    while i < extra_ffmpeg_args.len() {
        match extra_ffmpeg_args[i] {
            "-r" => {
                if let Some(rate_str) = extra_ffmpeg_args.get(i + 1) {
                    if let Ok(rate) = rate_str.parse::<u32>() {
                        fps = rate;
                        info!("overriding framerate with ffmpeg arg: {}", fps);
                    } else {
                        warn!("invalid framerate value: {}", rate_str);
                    }
                    i += 1;
                } else {
                    warn!("-r option requires a value.");
                }
            },
            "-s" | "-video_size" => {
                if let Some(res_str) = extra_ffmpeg_args.get(i + 1) {
                    let parts: Vec<&str> = res_str.split('x').collect();
                    if parts.len() == 2 {
                        if let (Ok(w), Ok(h)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                            width = w;
                            height = h;
                            info!("overriding resolution with ffmpeg arg: {}x{}", width, height);
                        } else {
                            warn!("invalid resolution format: {}", res_str);
                        }
                    } else {
                        warn!("invalid resolution format: {}", res_str);
                    }
                    i += 1;
                } else {
                    warn!("-s or -video_size option requires a value.");
                }
            },
            "--ltr" | "--rtl" => {
                // these are handled by `parse_spectrum_direction` already
            },
            arg => {
                clean_ffmpeg_args.push(arg.to_string()); // convert to String
            }
        }
        i += 1;
    }

    info!("resolution: {}x{}, fps: {}", width, height, fps);
    info!("parsed spectrum direction: {:?}", spectrum_direction);

    let (audio_samples, sample_rate) = audio_decoder.decode_audio_file_to_samples(input_audio_path)?;
    let total_frames = ((audio_samples.len() as f64 / sample_rate as f64) * fps as f64).ceil() as usize;
    info!("decoded {} samples at {} hz, generating {} frames.", audio_samples.len(), sample_rate, total_frames);

    let frames_dir = output_video_path.with_extension("frames_tmp");
    fs::create_dir_all(&frames_dir)?;

    let fft_window_size = 2048;
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c_fft_shared = Arc::new(planner.plan_fft_forward(fft_window_size));
    let quantum_processor_arc = Arc::new(Mutex::new(quantum_processor));
    let prev_rms_loudness_arc = Arc::new(Mutex::new(0.0f64));

    let perlin_noise_arc = Arc::new(PerlinNoise::new(42));

    info!("generating video frames...");
    let bar = ProgressBar::new(total_frames as u64);
    bar.set_style(ProgressStyle::default_bar().template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?.progress_chars("#>-"));

    (0..total_frames).into_par_iter().for_each_init(
        || {
            let input_buffer = Array1::<f32>::zeros(fft_window_size);
            let output_buffer = Array1::<Complex<f32>>::zeros(fft_window_size / 2 + 1);
            let scratch_buffer = vec![Complex::new(0.0, 0.0); r2c_fft_shared.get_scratch_len()];
            (input_buffer, output_buffer, scratch_buffer)
        },
        |(input_buffer, output_buffer, scratch_buffer), i| {
            let r2c_fft = r2c_fft_shared.clone();

            let start_sample_index = (i as f64 * (sample_rate as f64 / fps as f64)).round() as usize;
            let window_samples = fft_window_size.min(audio_samples.len().saturating_sub(start_sample_index));
            if window_samples > 0 {
                let end_idx = start_sample_index + window_samples;
                input_buffer.as_slice_mut().unwrap()[..window_samples].copy_from_slice(&audio_samples[start_sample_index..end_idx]);
                for (idx, sample) in input_buffer.as_slice_mut().unwrap()[..window_samples].iter_mut().enumerate() {
                    *sample *= 0.5 * (1.0 - (2.0 * std::f64::consts::PI * idx as f64 / (window_samples - 1) as f64).cos()) as f32;
                }
            }
            if let Err(e) = r2c_fft.process_with_scratch(input_buffer.as_slice_mut().unwrap(), output_buffer.as_slice_mut().unwrap(), scratch_buffer) {
                eprintln!("fft processing error: {}", e);
                return;
            }

            let current_prev_rms = *prev_rms_loudness_arc.lock().unwrap();
            let audio_features = AudioVisualizer::extract_enhanced_features(output_buffer.as_slice().unwrap(), sample_rate, window_samples, &audio_samples, start_sample_index, current_prev_rms);
            *prev_rms_loudness_arc.lock().unwrap() = audio_features.rms_loudness;

            let quantum_data = quantum_processor_arc.lock().unwrap().process_frame(&audio_features, i, total_frames);

            let mut img = ImageBuffer::<Rgb<u8>, _>::from_pixel(width, height, Rgb([0, 0, 0]));

            set_black_background(&mut img, width, height);

            generate_energetic_particles(&mut img, &quantum_data, &audio_features, width, height, &perlin_noise_arc);

            if let Err(e) = img.save_with_format(frames_dir.join(format!("frame_{:05}.png", i)), ImageFormat::Png) {
                eprintln!("saving frame error: {}", e);
            }
            bar.inc(1);
        },
    );


    bar.finish_and_clear();
    info!("encoding video with ffmpeg...");

    // we need to create owned strings for these
    let fps_str_owned = fps.to_string();
    let frames_path_string = frames_dir.join("frame_%05d.png").to_string_lossy().into_owned();
    let audio_path_string = input_audio_path.to_string_lossy().into_owned();
    let output_video_path_string = output_video_path.to_string_lossy().into_owned();
    let resolution_str_owned = format!("{}x{}", width, height);

    let mut ffmpeg_command: Vec<&str> = vec![
        "-y",
        "-framerate", fps_str_owned.as_str(), // use .as_str() on the owned String
        "-i", frames_path_string.as_str(),
        "-i", audio_path_string.as_str(),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:a", "aac",
        "-b:a", "192k",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-threads", "0",
        "-s", resolution_str_owned.as_str(), // use .as_str() on the owned String
    ];

    // extend with references from the owned strings in clean_ffmpeg_args
    for arg in &clean_ffmpeg_args {
        ffmpeg_command.push(arg.as_str());
    }

    ffmpeg_command.push(output_video_path_string.as_str());

    let ffmpeg_output = cmd("ffmpeg", &ffmpeg_command).stderr_to_stdout().run()?;
    if !ffmpeg_output.status.success() {
        error!("ffmpeg failed: {}", String::from_utf8_lossy(&ffmpeg_output.stdout));
        return Err(format!("ffmpeg command failed: {}", String::from_utf8_lossy(&ffmpeg_output.stdout)).into());
    }

    info!("cleaning up temporary frames...");
    fs::remove_dir_all(&frames_dir)?;
    Ok(())
}
