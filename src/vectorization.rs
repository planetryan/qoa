#![allow(non_snake_case)]
#![allow(unused_imports)]

/*
    rust likes code to be safe, i agree with this, but for simd, sometimes direct, more unsafe methods are needed for preformance
    i could make it safer, but at a preformance penalty of ~15%, ive chosen preformance.

    if you feel like it is unsafe, feel free to modify it and make it safer.
*/

// --- common interface for vector operations ---
use num_complex::Complex64;
use rayon::prelude::*; // for parallel iterators in fallback
// use std::simd::{f32x8, f32x16};

pub fn apply_hadamard_vectorized(amps: &mut [Complex64], norm_factor: Complex64, mask_bit: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_hadamard_simd(amps, norm_factor, mask_bit);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_hadamard_simd(amps, norm_factor, mask_bit);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_hadamard_simd(amps, norm_factor, mask_bit);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_hadamard_simd(amps, norm_factor, mask_bit);
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let bit_val = 1 << mask_bit;
        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..amps.len() { 
            if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                let flipped_idx = i | bit_val; // get the index where mask_bit is 1
                if flipped_idx < amps.len() { // ensure flipped_idx is within bounds
                    let a_val = amps[i];
                    let b_val = amps[flipped_idx];

                    amps[i] = norm_factor * (a_val + b_val);
                    amps[flipped_idx] = norm_factor * (a_val - b_val);
                }
            }
        }
    }
}

pub fn apply_x_vectorized(amps: &mut [Complex64], mask_bit: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_x_simd(amps, mask_bit);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_x_simd(amps, mask_bit);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_x_simd(amps, mask_bit);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_x_simd(amps, mask_bit);
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let bit_val = 1 << mask_bit;
        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..amps.len() { 
            if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                let flipped_idx = i | bit_val; // get the index where mask_bit is 1
                if flipped_idx < amps.len() { // ensure flipped_idx is within bounds
                    amps.swap(i, flipped_idx);
                }
            }
        }
    }
}

pub fn apply_y_vectorized(amps: &mut [Complex64], mask_bit: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_y_simd(amps, mask_bit);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_y_simd(amps, mask_bit);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_y_simd(amps, mask_bit);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_y_simd(amps, mask_bit);
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let bit_val = 1 << mask_bit;
        let neg_i = Complex64::new(0.0, -1.0);
        let pos_i = Complex64::new(0.0, 1.0);

        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..amps.len() { 
            if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                let flipped_idx = i | bit_val; // get the index where mask_bit is 1
                if flipped_idx < amps.len() { // ensure flipped_idx is within bounds
                    let amp0 = amps[i];
                    let amp1 = amps[flipped_idx];
                    amps[i] = amp1 * neg_i;
                    amps[flipped_idx] = amp0 * pos_i;
                }
            }
        }
    }
}

pub fn apply_z_vectorized(amps: &mut [Complex64], mask_bit: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_z_simd(amps, mask_bit);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_z_simd(amps, mask_bit);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_z_simd(amps, mask_bit);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_z_simd(amps, mask_bit);
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let bit_val = 1 << mask_bit;
        // z-gate applies phase flip to |1> state of the target qubit
        amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            if (i & bit_val) != 0 { // if the mask_bit is 1 in current index i
                *amp = amp.scale(-1.0); // using scale for explicit negation
            }
        });
    }
}

pub fn apply_t_vectorized(amps: &mut [Complex64], mask_bit: usize) {
    let phase_factor = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_phaseshift_simd(amps, mask_bit, phase_factor);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_phaseshift_simd(amps, mask_bit, phase_factor);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_phaseshift_simd(amps, mask_bit, phase_factor);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_phaseshift_simd(amps, mask_bit, phase_factor);
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let bit_val = 1 << mask_bit;
        // t-gate applies phase shift to |1> state of the target qubit
        amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            if (i & bit_val) != 0 { // if the mask_bit is 1 in current index i
                *amp *= phase_factor;
            }
        });
    }
}

pub fn apply_s_vectorized(amps: &mut [Complex64], mask_bit: usize) {
    let phase_factor = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_2);
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_phaseshift_simd(amps, mask_bit, phase_factor);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_phaseshift_simd(amps, mask_bit, phase_factor);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_phaseshift_simd(amps, mask_bit, phase_factor);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_phaseshift_simd(amps, mask_bit, phase_factor);
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let bit_val = 1 << mask_bit;
        // s-gate applies phase shift to |1> state of the target qubit
        amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            if (i & bit_val) != 0 { // if the mask_bit is 1 in current index i
                *amp *= phase_factor;
            }
        });
    }
}

pub fn apply_phaseshift_vectorized(amps: &mut [Complex64], mask_bit: usize, angle: f64) {
    let phase_factor = Complex64::from_polar(1.0, angle);
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_phaseshift_simd(amps, mask_bit, phase_factor);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_phaseshift_simd(amps, mask_bit, phase_factor);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_phaseshift_simd(amps, mask_bit, phase_factor);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_phaseshift_simd(amps, mask_bit, phase_factor);
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let bit_val = 1 << mask_bit;
        // phase shift applies phase to |1> state of the target qubit
        amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            if (i & bit_val) != 0 { // if the mask_bit is 1 in current index i
                *amp *= phase_factor;
            }
        });
    }
}

pub fn apply_reset_vectorized(amps: &mut [Complex64], mask_bit: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_reset_simd(amps, mask_bit);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_reset_simd(amps, mask_bit);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_reset_simd(amps, mask_bit);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_reset_simd(amps, mask_bit);
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let bit_val = 1 << mask_bit;
        let _total_elements = amps.len(); // kept for clarity, but marked as unused
        let mut norm = 0.0;
        // first pass: calculate the norm of the |0> subspace for the target qubit
        for i in 0..amps.len() { 
            if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                norm += amps[i].norm_sqr();
            }
        }
        if norm > 1e-12 { // avoid division by zero
            let norm = norm.sqrt();
            // second pass: normalize amplitudes in the |0> subspace and zero out |1> subspace
            for i in 0..amps.len() { 
                if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                    amps[i] /= norm;
                } else { // if the mask_bit is 1 in current index i
                    amps[i] = Complex64::new(0.0, 0.0);
                }
            }
        } else {
            // if the |0> subspace has zero probability, set the state to |0>
            // this handles cases where the state was entirely in the |1> subspace
            if amps.len() > 0 { 
                amps[0] = Complex64::new(1.0, 0.0);
                for i in 1..amps.len() { 
                    amps[i] = Complex64::new(0.0, 0.0);
                }
            }
        }
    }
}

pub fn apply_swap_vectorized(amps: &mut [Complex64], q1: usize, q2: usize) {
    let q1_mask = 1 << q1;
    let q2_mask = 1 << q2;
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_swap_simd(amps, q1_mask, q2_mask);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_swap_simd(amps, q1_mask, q2_mask);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_swap_simd(amps, q1_mask, q2_mask);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_swap_simd(amps, q1_mask, q2_mask);
        }
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let total_elements = amps.len();
        let combined_mask = q1_mask | q2_mask;
        for i in 0..total_elements {
            if (i & combined_mask) == 0 {
                let idx_q1_0_q2_1 = i | q2_mask; // q1=0, q2=1
                let idx_q1_1_q2_0 = i | q1_mask; // q1=1, q2=0
                if idx_q1_0_q2_1 < total_elements && idx_q1_1_q2_0 < total_elements {
                    amps.swap(idx_q1_0_q2_1, idx_q1_1_q2_0);
                }
            }
        }
    }
}

pub fn apply_controlled_swap_vectorized(amps: &mut [Complex64], control: usize, target1: usize, target2: usize) {
    let control_mask = 1 << control;
    let target1_mask = 1 << target1;
    let target2_mask = 1 << target2;
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_controlled_swap_simd(amps, control_mask, target1_mask, target2_mask);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_controlled_swap_simd(amps, control_mask, target1_mask, target2_mask);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_controlled_swap_simd(amps, control_mask, target1_mask, target2_mask);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_controlled_swap_simd(amps, control_mask, target1_mask, target2_mask);
        }
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let total_elements = amps.len();
        let combined_mask = control_mask | target1_mask | target2_mask;
        for i in 0..total_elements {
            if (i & combined_mask) == 0 {
                let idx_c1_t1_0_t2_1 = i | control_mask | target2_mask;
                let idx_c1_t1_1_t2_0 = i | control_mask | target1_mask;
                if idx_c1_t1_0_t2_1 < total_elements && idx_c1_t1_1_t2_0 < total_elements {
                    amps.swap(idx_c1_t1_0_t2_1, idx_c1_t1_1_t2_0);
                }
            }
        }
    }
}

pub fn apply_rx_vectorized(amps: &mut [Complex64], mask_bit: usize, angle: f64) {
    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();
    // rx matrix: [[cos(a/2), -i*sin(a/2)], [-i*sin(a/2), cos(a/2)]]
    let m00 = Complex64::new(cos_half, 0.0);
    let m01 = Complex64::new(0.0, -sin_half);
    let m10 = Complex64::new(0.0, -sin_half);
    let m11 = Complex64::new(cos_half, 0.0);

    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_rx_simd(amps, mask_bit, m00, m01, m10, m11);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_rx_simd(amps, mask_bit, m00, m01, m10, m11);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_rx_simd(amps, mask_bit, m00, m01, m10, m11);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_rx_simd(amps, mask_bit, m00, m01, m10, m11);
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let bit_val = 1 << mask_bit;
        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..amps.len() { 
            if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                let flipped_idx = i | bit_val; // get the index where mask_bit is 1
                if flipped_idx < amps.len() { // ensure flipped_idx is within bounds
                    let amp0 = amps[i];
                    let amp1 = amps[flipped_idx];
                    amps[i] = m00 * amp0 + m01 * amp1;
                    amps[flipped_idx] = m10 * amp0 + m11 * amp1;
                }
            }
        }
    }
}

pub fn apply_ry_vectorized(amps: &mut [Complex64], mask_bit: usize, angle: f64) {
    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();
    // ry matrix: [[cos(a/2), -sin(a/2)], [sin(a/2), cos(a/2)]]
    let m00 = Complex64::new(cos_half, 0.0);
    let m01 = Complex64::new(-sin_half, 0.0);
    let m10 = Complex64::new(sin_half, 0.0);
    let m11 = Complex64::new(cos_half, 0.0);

    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_ry_simd(amps, mask_bit, m00, m01, m10, m11);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_ry_simd(amps, mask_bit, m00, m01, m10, m11);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_ry_simd(amps, mask_bit, m00, m01, m10, m11);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_ry_simd(amps, mask_bit, m00, m01, m10, m11);
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let bit_val = 1 << mask_bit;
        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..amps.len() { 
            if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                let flipped_idx = i | bit_val; // get the index where mask_bit is 1
                if flipped_idx < amps.len() { // ensure flipped_idx is within bounds
                    let amp0 = amps[i];
                    let amp1 = amps[flipped_idx];
                    amps[i] = m00 * amp0 + m01 * amp1;
                    amps[flipped_idx] = m10 * amp0 + m11 * amp1;
                }
            }
        }
    }
}

pub fn apply_rz_vectorized(amps: &mut [Complex64], mask_bit: usize, angle: f64) {
    // rz matrix: [[e^(-i*a/2), 0], [0, e^(i*a/2)]]
    let m00 = Complex64::from_polar(1.0, -angle / 2.0);
    let m11 = Complex64::from_polar(1.0, angle / 2.0);

    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_rz_simd(amps, mask_bit, m00, m11);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_rz_simd(amps, mask_bit, m00, m11);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_rz_simd(amps, mask_bit, m00, m11);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_rz_simd(amps, mask_bit, m00, m11);
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let bit_val = 1 << mask_bit;
        // rz-gate applies phase to |0> and |1> states of the target qubit
        amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                *amp *= m00;
            } else { // if the mask_bit is 1 in current index i
                *amp *= m11;
            }
        });
    }
}

pub fn apply_cnot_vectorized(amps: &mut [Complex64], control: usize, target: usize) {
    let control_mask = 1 << control;
    let target_mask = 1 << target;
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_cnot_simd(amps, control_mask, target_mask);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_cnot_simd(amps, control_mask, target_mask);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_cnot_simd(amps, control_mask, target_mask);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_cnot_simd(amps, control_mask, target_mask);
        }
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        let total_elements = amps.len();
        for i in 0..total_elements {
            if (i & control_mask) == control_mask && (i & target_mask) == 0 {
                let j = i | target_mask;
                if j < total_elements {
                    amps.swap(i, j);
                }
            }
        }
    }
}

pub fn apply_cz_vectorized(amps: &mut [Complex64], control_mask: usize, target_mask: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_cz_simd(amps, control_mask, target_mask);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_cz_simd(amps, control_mask, target_mask);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_cz_simd(amps, control_mask, target_mask);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_cz_simd(amps, control_mask, target_mask);
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        // cz-gate applies phase flip if both control and target qubits are 1
        amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            // changed condition to be more explicit for control qubit state
            if (i & (control_mask | target_mask)) == (control_mask | target_mask) {
                *amp = amp.scale(-1.0); // using scale for explicit negation
            }
        });
    }
}

pub fn apply_controlled_phase_rotation_vectorized(
    amps: &mut [Complex64],
    control_mask: usize,
    target_mask: usize,
    angle: f64,
) {
    let phase_factor = Complex64::from_polar(1.0, angle);
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_controlled_phase_rotation_simd(
                amps,
                control_mask,
                target_mask,
                phase_factor,
            );
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_controlled_phase_rotation_simd(
                amps,
                control_mask,
                target_mask,
                phase_factor,
            );
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_controlled_phase_rotation_simd(
                amps,
                control_mask,
                target_mask,
                phase_factor,
            );
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_controlled_phase_rotation_simd(
                amps,
                control_mask,
                target_mask,
                phase_factor,
            );
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        // controlled phase rotation applies phase if both control and target qubits are 1
        amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            // changed condition to be more explicit for control qubit state
            if (i & (control_mask | target_mask)) == (control_mask | target_mask) {
                *amp *= phase_factor;
            }
        });
    }
}

pub fn apply_reset_all_vectorized(amps: &mut [Complex64]) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            x86_64_simd::apply_reset_all_simd(amps);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            aarch64_neon::apply_reset_all_simd(amps);
        }
    }
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        unsafe {
            riscv64_rvv::apply_reset_all_simd(amps);
        }
    }
    #[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
    {
        unsafe {
            power_vsx::apply_reset_all_simd(amps);
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "riscv64", target_feature = "v"),
        all(target_arch = "powerpc64", target_feature = "vsx")
    )))]
    {
        // reset all sets the state to |0...0>
        amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
            if i == 0 {
                *amp = Complex64::new(1.0, 0.0);
            } else {
                *amp = Complex64::new(0.0, 0.0);
            }
        });
    }
}

// --- x86_64 simd implementations ---
// this module contains simd-accelerated implementations of quantum gates
// for x86_64 architectures.
#[cfg(target_arch = "x86_64")]
pub mod x86_64_simd {
    use super::*;
    use std::arch::x86_64::*;
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

    // complex64 elements per avx-512 register (not used directly, but good for context)
    const _AVX512_LANE_SIZE: usize = 4;
    // complex64 elements per avx register (not used directly, but good for context)
    const _AVX_LANE_SIZE: usize = 2;

    pub unsafe fn apply_hadamard_simd(
        amps: &mut [Complex64],
        norm_factor: Complex64,
        mask_bit: usize
    ) {
        let bit = 1 << mask_bit;
        let total_elements = amps.len();
        
        // process entries sequentially to ensure correctness
        // iterate over pairs (i, j) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            if (i & bit) == 0 { // if the mask_bit is 0 in current index i
                let j = i | bit; // get the index where mask_bit is 1
                if j < total_elements { // ensure j is within bounds
                    let a = amps[i];
                    let b = amps[j];
                    amps[i] = norm_factor * (a + b);
                    amps[j] = norm_factor * (a - b);
                }
            }
        }
    }

    // pauli-x: swap |0> ↔ |1>
    pub unsafe fn apply_x_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit = 1 << mask_bit;
        let total_elements = amps.len();
        
        for i in 0..total_elements {
            if (i & bit) == 0 {
                let j = i | bit;
                if j < total_elements {
                    unsafe {
                        let ptr_i = amps.as_ptr().add(i) as *const f64;
                        let ptr_j = amps.as_ptr().add(j) as *const f64;
                        let amp_i_vec = _mm_loadu_pd(ptr_i);
                        let amp_j_vec = _mm_loadu_pd(ptr_j);
                        _mm_storeu_pd(amps.as_mut_ptr().add(i) as *mut f64, amp_j_vec);
                        _mm_storeu_pd(amps.as_mut_ptr().add(j) as *mut f64, amp_i_vec);
                    }
                }
            }
        }
    }

    // pauli-y: |0>→i|1>, |1>→-i|0>
    pub unsafe fn apply_y_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit = 1 << mask_bit;
        let neg_i = Complex64::new(0.0, -1.0);
        let pos_i = Complex64::new(0.0, 1.0);
        let total_elements = amps.len();
        
        // iterate over pairs (i, j) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            if (i & bit) == 0 { // if the mask_bit is 0 in current index i
                let j = i | bit; // get the index where mask_bit is 1
                if j < total_elements { // ensure j is within bounds
                    let a0 = amps[i];
                    let a1 = amps[j];
                    amps[i] = a1 * neg_i;
                    amps[j] = a0 * pos_i;
                }
            }
        }
    }

    // pauli-z: flip phase of |1>
    pub unsafe fn apply_z_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit = 1 << mask_bit;
        let total_elements = amps.len();
        
        // iterate over all elements and apply phase flip if mask_bit is 1
        for i in 0..total_elements {
            if (i & bit) != 0 { // if the mask_bit is 1 in current index i
                amps[i] = amps[i].scale(-1.0); // using scale for explicit negation
            }
        }
    }

    // phase-shift p(φ): diag(1, e^{iφ})
    pub unsafe fn apply_phaseshift_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        phase_factor: Complex64
    ) {
        let bit = 1 << mask_bit;
        let total_elements = amps.len();
        
        // iterate over all elements and apply phase shift if mask_bit is 1
        for i in 0..total_elements {
            if (i & bit) != 0 { // if the mask_bit is 1 in current index i
                amps[i] *= phase_factor;
            }
        }
    }

    // reset (measure & collapse to |0>)
    pub unsafe fn apply_reset_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit = 1 << mask_bit;
        let total_elements = amps.len();
        let mut norm_sqr = 0.0;
        
        // first pass: calculate norm of the |0> subspace for the target qubit
        for i in 0..total_elements {
            if (i & bit) == 0 { // if the mask_bit is 0 in current index i
                norm_sqr += amps[i].norm_sqr();
            }
        }
        
        // second pass: normalize and collapse
        if norm_sqr > 1e-12 { // avoid division by zero
            let scale = 1.0 / norm_sqr.sqrt();
            for i in 0..total_elements {
                if (i & bit) == 0 { // if the mask_bit is 0 in current index i
                    amps[i] *= scale;
                } else { // if the mask_bit is 1 in current index i
                    amps[i] = Complex64::new(0.0, 0.0);
                }
            }
        } else {
            // if zero probability, set to |0> state
            // this handles cases where the state was entirely in the |1> subspace
            if total_elements > 0 {
                amps[0] = Complex64::new(1.0, 0.0);
                for i in 1..total_elements {
                    amps[i] = Complex64::new(0.0, 0.0);
                }
            }
        }
    }

    // general single-qubit rotation: matrix [[m00, m01], [m10, m11]]
    pub unsafe fn apply_rx_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        m00: Complex64, m01: Complex64,
        m10: Complex64, m11: Complex64
    ) {
        let bit = 1 << mask_bit;
        let total_elements = amps.len();
        
        // iterate over pairs (i, j) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            if (i & bit) == 0 { // if the mask_bit is 0 in current index i
                let j = i | bit; // get the index where mask_bit is 1
                if j < total_elements { // ensure j is within bounds
                    let amp0 = amps[i];
                    let amp1 = amps[j];
                    amps[i] = m00 * amp0 + m01 * amp1;
                    amps[j] = m10 * amp0 + m11 * amp1;
                }
            }
        }
    }

    #[inline(always)]
    pub unsafe fn apply_ry_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        m00: Complex64, m01: Complex64,
        m10: Complex64, m11: Complex64
    ) {
        // ry gate uses the same matrix application logic as rx
        unsafe {
            apply_rx_simd(amps, mask_bit, m00, m01, m10, m11);
        }
    }

    // rz rotation: diag(m00, m11)
    pub unsafe fn apply_rz_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        m00: Complex64, m11: Complex64
    ) {
        let bit = 1 << mask_bit;
        let total_elements = amps.len();
        
        // iterate over all elements and apply phase based on mask_bit
        for i in 0..total_elements {
            if (i & bit) == 0 { // if the mask_bit is 0 in current index i
                amps[i] *= m00;
            } else { // if the mask_bit is 1 in current index i
                amps[i] *= m11;
            }
        }
    }

    // s-gate = p(pi/2)
    pub unsafe fn apply_s_simd(amps: &mut [Complex64], mask_bit: usize) {
        let phase = Complex64::new(0.0, 1.0); // equivalent to e^(i*pi/2)
        unsafe {
            apply_phaseshift_simd(amps, mask_bit, phase);
        }
    }

    // t-gate = p(pi/4)
    pub unsafe fn apply_t_simd(amps: &mut [Complex64], mask_bit: usize) {
        let phase = Complex64::from_polar(1.0, FRAC_PI_4);
        unsafe {
            apply_phaseshift_simd(amps, mask_bit, phase);
        }
    }

    // cnot: if control=1, flip target
    pub unsafe fn apply_cnot_simd(
        amps: &mut [Complex64],
        control_mask: usize,
        target_mask: usize,
    ) {
        let total_elements = amps.len();
        for i in 0..total_elements {
            // process only when control qubit is 1 and target qubit is 0
            if (i & control_mask) == control_mask && (i & target_mask) == 0 {
                let j = i | target_mask; // state where target qubit is flipped to 1
                if j < total_elements { // bounds checking
                    unsafe {
                        let ptr_i = amps.as_ptr().add(i) as *const f64;
                        let ptr_j = amps.as_ptr().add(j) as *const f64;
                        let amp_i_vec = _mm_loadu_pd(ptr_i);
                        let amp_j_vec = _mm_loadu_pd(ptr_j);
                        _mm_storeu_pd(amps.as_mut_ptr().add(i) as *mut f64, amp_j_vec);
                        _mm_storeu_pd(amps.as_mut_ptr().add(j) as *mut f64, amp_i_vec);
                    }
                }
            }
        }
    }

    // cz: if control=1 and target=1, flip phase
    pub unsafe fn apply_cz_simd(
        amps: &mut [Complex64],
        control_mask: usize,
        target_mask: usize
    ) {
        let total_elements = amps.len();
        
        // iterate over all elements and apply phase flip if both control and target bits are 1
        for i in 0..total_elements {
            // changed condition to be more explicit for control qubit state
            if (i & (control_mask | target_mask)) == (control_mask | target_mask) {
                amps[i] = amps[i].scale(-1.0); // using scale for explicit negation
            }
        }
    }

    // swap: swap two qubits q1 and q2
    pub unsafe fn apply_swap_simd(
        amps: &mut [Complex64],
        q1_mask: usize,
        q2_mask: usize,
    ) {
        let total_elements = amps.len();
        let combined_mask = q1_mask | q2_mask;
        for i in 0..total_elements {
            // process base states where both qubits are 0
            if (i & combined_mask) == 0 {
                let idx_q1_0_q2_1 = i | q2_mask; // q1=0, q2=1
                let idx_q1_1_q2_0 = i | q1_mask; // q1=1, q2=0
                if idx_q1_0_q2_1 < total_elements && idx_q1_1_q2_0 < total_elements { // bounds checking
                    unsafe {
                        let ptr_a = amps.as_ptr().add(idx_q1_0_q2_1) as *const f64;
                        let ptr_b = amps.as_ptr().add(idx_q1_1_q2_0) as *const f64;
                        let amp_a_vec = _mm_loadu_pd(ptr_a);
                        let amp_b_vec = _mm_loadu_pd(ptr_b);
                        _mm_storeu_pd(amps.as_mut_ptr().add(idx_q1_0_q2_1) as *mut f64, amp_b_vec);
                        _mm_storeu_pd(amps.as_mut_ptr().add(idx_q1_1_q2_0) as *mut f64, amp_a_vec);
                    }
                }
            }
        }
    }

    // controlled-swap (fredkin): if control=1, swap targets
    pub unsafe fn apply_controlled_swap_simd(
        amps: &mut [Complex64],
        control_mask: usize,
        t1_mask: usize,
        t2_mask: usize,
    ) {
        let total_elements = amps.len();
        let combined_mask = control_mask | t1_mask | t2_mask;
        for i in 0..total_elements {
            // process base states where control, t1, and t2 are 0
            if (i & combined_mask) == 0 {
                let idx_c1_t1_0_t2_1 = i | control_mask | t2_mask; // control=1, t1=0, t2=1
                let idx_c1_t1_1_t2_0 = i | control_mask | t1_mask; // control=1, t1=1, t2=0
                if idx_c1_t1_0_t2_1 < total_elements && idx_c1_t1_1_t2_0 < total_elements { // bounds checking
                    unsafe {
                        let ptr_a = amps.as_ptr().add(idx_c1_t1_0_t2_1) as *const f64;
                        let ptr_b = amps.as_ptr().add(idx_c1_t1_1_t2_0) as *const f64;
                        let amp_a_vec = _mm_loadu_pd(ptr_a);
                        let amp_b_vec = _mm_loadu_pd(ptr_b);
                        _mm_storeu_pd(amps.as_mut_ptr().add(idx_c1_t1_0_t2_1) as *mut f64, amp_b_vec);
                        _mm_storeu_pd(amps.as_mut_ptr().add(idx_c1_t1_1_t2_0) as *mut f64, amp_a_vec);
                    }
                }
            }
        }
    }

    // controlled phase rotation: if control=1 and target=1, apply phase
    pub unsafe fn apply_controlled_phase_rotation_simd(
        amps: &mut [Complex64],
        control_mask: usize,
        target_mask: usize,
        phase_factor: Complex64,
    ) {
        let total_elements = amps.len();
        
        // iterate over all elements and apply phase if both control and target bits are 1
        for i in 0..total_elements {
            // changed condition to be more explicit for control qubit state
            if (i & (control_mask | target_mask)) == (control_mask | target_mask) {
                amps[i] *= phase_factor;
            }
        }
    }

    // reset entire state to |0...0>
    pub unsafe fn apply_reset_all_simd(amps: &mut [Complex64]) {
        if !amps.is_empty() {
            amps[0] = Complex64::new(1.0, 0.0);
            for i in 1..amps.len() {
                amps[i] = Complex64::new(0.0, 0.0);
            }
        }
    }
}

// --- aarch64 neon implementations ---
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod aarch64_neon {
    use super::*;
    use std::arch::aarch64::*;

    // helper for complex multiplication using neon intrinsics for float64x2_t
    // (a_re + i*a_im) * (b_re + i*b_im) = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
    #[target_feature(enable = "neon")] // removed #[inline(always)]
    unsafe fn mul_complex_f64x2(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        // a = [a_re, a_im]
        // b = [b_re, b_im]

        // extract components
        let a_re = vgetq_lane_f64(a, 0);
        let a_im = vgetq_lane_f64(a, 1);
        let b_re = vgetq_lane_f64(b, 0);
        let b_im = vgetq_lane_f64(b, 1);

        // compute real part: (a_re * b_re) - (a_im * b_im)
        let res_re = a_re * b_re - a_im * b_im;
        // compute imaginary part: (a_re * b_im) + (a_im * b_re)
        let res_im = a_re * b_im + a_im * b_re;

        // combine into a float64x2_t [res_re, res_im]
        vsetq_lane_f64(res_im, vsetq_lane_f64(res_re, vdupq_n_f64(0.0), 0), 1)
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_hadamard_simd(
        amps: &mut [Complex64],
        norm_factor: Complex64,
        mask_bit: usize,
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();
        let nf_vec = vsetq_lane_f64(norm_factor.im, vsetq_lane_f64(norm_factor.re, vdupq_n_f64(0.0), 0), 1);
        
        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let a_ptr = amps.as_ptr().add(i) as *const f64;
                    let b_ptr = amps.as_ptr().add(flipped_idx) as *const f64;

                    let a_vec = vld1q_f64(a_ptr);
                    let b_vec = vld1q_f64(b_ptr);

                    let sum_vec = vaddq_f64(a_vec, b_vec);
                    let diff_vec = vsubq_f64(a_vec, b_vec);

                    let res_a = mul_complex_f64x2(nf_vec, sum_vec);
                    let res_b = mul_complex_f64x2(nf_vec, diff_vec);

                    vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, res_a);
                    vst1q_f64(amps.as_mut_ptr().add(flipped_idx) as *mut f64, res_b);
                }
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_x_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let a_ptr = amps.as_ptr().add(i) as *const f64;
                    let b_ptr = amps.as_ptr().add(flipped_idx) as *const f64;

                    let a_vec = vld1q_f64(a_ptr);
                    let b_vec = vld1q_f64(b_ptr);

                    vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, b_vec);
                    vst1q_f64(amps.as_mut_ptr().add(flipped_idx) as *mut f64, a_vec);
                }
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_y_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();
        let neg_i = Complex64::new(0.0, -1.0);
        let pos_i = Complex64::new(0.0, 1.0);

        let neg_i_vec = vsetq_lane_f64(neg_i.im, vsetq_lane_f64(neg_i.re, vdupq_n_f64(0.0), 0), 1);
        let pos_i_vec = vsetq_lane_f64(pos_i.im, vsetq_lane_f64(pos_i.re, vdupq_n_f64(0.0), 0), 1);
        
        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let amp0_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp1_ptr = amps.as_ptr().add(flipped_idx) as *const f64;

                    let amp0_vec = vld1q_f64(amp0_ptr);
                    let amp1_vec = vld1q_f64(amp1_ptr);

                    let res0 = mul_complex_f64x2(neg_i_vec, amp1_vec);
                    let res1 = mul_complex_f64x2(pos_i_vec, amp0_vec);

                    vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, res0);
                    vst1q_f64(amps.as_mut_ptr().add(flipped_idx) as *mut f64, res1);
                }
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_z_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();
        // vector of all -1.0 to negate both real and imag parts
        let neg_one_vec = vdupq_n_f64(-1.0); 

        // iterate over all elements and apply phase flip if mask_bit is 1
        for i in 0..total_elements {
            if (i & bit_val) != 0 { // if the mask_bit is 1 in current index i
                unsafe {
                    let amp_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp_vec = vld1q_f64(amp_ptr);
                    let res_vec = vmulq_f64(amp_vec, neg_one_vec); // multiply by -1.0
                    vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, res_vec);
                }
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_phaseshift_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        phase_factor: Complex64,
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();
        let pf_vec = vsetq_lane_f64(phase_factor.im, vsetq_lane_f64(phase_factor.re, vdupq_n_f64(0.0), 0), 1);
        
        // iterate over all elements and apply phase shift if mask_bit is 1
        for i in 0..total_elements {
            if (i & bit_val) != 0 { // if the mask_bit is 1 in current index i
                unsafe {
                    let amp_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp_vec = vld1q_f64(amp_ptr);
                    let res_vec = mul_complex_f64x2(pf_vec, amp_vec);
                    vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, res_vec);
                }
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_reset_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();
        let zero_vec = vdupq_n_f64(0.0);
        let mut norm_sqr: f64 = 0.0;

        // first pass: calculate the norm of the |0> subspace for the target qubit
        for i in 0..total_elements {
            if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                unsafe {
                    let amp_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp_vec = vld1q_f64(amp_ptr);
                    let re = vgetq_lane_f64(amp_vec, 0);
                    let im = vgetq_lane_f64(amp_vec, 1);
                    norm_sqr += re * re + im * im;
                }
            }
        }
        
        // second pass: normalize and collapse
        if norm_sqr > 1e-12 { // avoid division by zero
            let scale = 1.0 / norm_sqr.sqrt();
            let scale_vec = vdupq_n_f64(scale); // broadcast scalar to vector

            for i in 0..total_elements {
                if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                    unsafe {
                        let amp_ptr = amps.as_ptr().add(i) as *const f64;
                        let amp_vec = vld1q_f64(amp_ptr);
                        let res_vec = vmulq_f64(amp_vec, scale_vec); // scale by 1/norm
                        vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, res_vec);
                    }
                } else { // if the mask_bit is 1 in current index i
                    unsafe {
                        vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, zero_vec);
                    }
                }
            }
        } else {
            // if zero probability, set to |0> state
            // this handles cases where the state was entirely in the |1> subspace
            if total_elements > 0 {
                unsafe {
                    let one_re_zero_im_vec = vsetq_lane_f64(0.0, vsetq_lane_f64(1.0, vdupq_n_f64(0.0), 0), 1);
                    vst1q_f64(amps.as_mut_ptr() as *mut f64, one_re_zero_im_vec);
                    for i in 1..total_elements {
                        vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, zero_vec);
                    }
                }
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_swap_simd(amps: &mut [Complex64], q1_mask: usize, q2_mask: usize) {
        let total_elements = amps.len();
        let combined_mask = q1_mask | q2_mask;
        for i in 0..total_elements {
            // process base states where both qubits are 0
            if (i & combined_mask) == 0 { // if both q1 and q2 bits are 0 in current index 'i'
                let idx_q1_0_q2_1 = i | q2_mask; // state with q1=0, q2=1
                let idx_q1_1_q2_0 = i | q1_mask; // state with q1=1, q2=0

                // these two indices form the pair that needs swapping.
                // no need for 'if i < j' because we explicitly construct the two indices from a base state.
                if idx_q1_0_q2_1 < total_elements && idx_q1_1_q2_0 < total_elements { // bounds checking
                    unsafe {
                        let a_ptr = amps.as_ptr().add(idx_q1_0_q2_1) as *const f64;
                        let b_ptr = amps.as_ptr().add(idx_q1_1_q2_0) as *const f64;

                        let a_vec = vld1q_f64(a_ptr);
                        let b_vec = vld1q_f64(b_ptr);

                        vst1q_f64(amps.as_mut_ptr().add(idx_q1_0_q2_1) as *mut f64, b_vec);
                        vst1q_f64(amps.as_mut_ptr().add(idx_q1_1_q2_0) as *mut f64, a_vec); // Corrected line
                    }
                }
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_controlled_swap_simd(
        amps: &mut [Complex64],
        control_mask: usize,
        t1_mask: usize,
        t2_mask: usize,
    ) {
        let total_elements = amps.len();
        let combined_mask = control_mask | t1_mask | t2_mask;
        for i in 0..total_elements {
            // process base states where control, t1, and t2 bits are all 0.
            // this ensures each relevant pair is visited exactly once.
            if (i & combined_mask) == 0 { // if control, t1, t2 bits are 0 in current index 'i'
                // construct the two states that need to be swapped when control is 1
                let idx_c1_t10_t21 = i | control_mask | t2_mask; // state with control=1, t1=0, t2=1
                let idx_c1_t11_t20 = i | control_mask | t1_mask; // state with control=1, t1=1, t2=0

                // these two indices form the pair that needs swapping.
                // no need for 'if i < j' because we explicitly construct the two indices from a base state.
                if idx_c1_t10_t21 < total_elements && idx_c1_t11_t20 < total_elements { // bounds checking
                    unsafe {
                        let ptr_a = amps.as_ptr().add(idx_c1_t10_t21) as *const f64;
                        let ptr_b = amps.as_ptr().add(idx_c1_t11_t20) as *const f64;

                        let amp_a_vec = vld1q_f64(ptr_a);
                        let amp_b_vec = vld1q_f64(ptr_b);

                        vst1q_f64(amps.as_mut_ptr().add(idx_c1_t10_t21) as *mut f64, amp_b_vec);
                        vst1q_f64(amps.as_mut_ptr().add(idx_c1_t11_t20) as *mut f64, amp_a_vec);
                    }
                }
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_rx_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        m00: Complex64, m01: Complex64,
        m10: Complex64, m11: Complex64
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let m00_vec = vsetq_lane_f64(m00.im, vsetq_lane_f64(m00.re, vdupq_n_f64(0.0), 0), 1);
        let m01_vec = vsetq_lane_f64(m01.im, vsetq_lane_f64(m01.re, vdupq_n_f64(0.0), 0), 1);
        let m10_vec = vsetq_lane_f64(m10.im, vsetq_lane_f64(m10.re, vdupq_n_f64(0.0), 0), 1);
        let m11_vec = vsetq_lane_f64(m11.im, vsetq_lane_f64(m11.re, vdupq_n_f64(0.0), 0), 1);
        
        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let amp0_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp1_ptr = amps.as_ptr().add(flipped_idx) as *const f64;

                    let amp0_vec = vld1q_f64(amp0_ptr);
                    let amp1_vec = vld1q_f64(amp1_ptr);

                    // res0 = m00 * amp0 + m01 * amp1
                    let term1_res0 = mul_complex_f64x2(m00_vec, amp0_vec);
                    let term2_res0 = mul_complex_f64x2(m01_vec, amp1_vec);
                    let res0 = vaddq_f64(term1_res0, term2_res0);

                    // res1 = m10 * amp0 + m11 * amp1
                    let term1_res1 = mul_complex_f64x2(m10_vec, amp0_vec);
                    let term2_res1 = mul_complex_f64x2(m11_vec, amp1_vec);
                    let res1 = vaddq_f64(term1_res1, term2_res1);

                    vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, res0);
                    vst1q_f64(amps.as_mut_ptr().add(flipped_idx) as *mut f64, res1);
                }
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_ry_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        m00: Complex64, m01: Complex64,
        m10: Complex64, m11: Complex64
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let m00_vec = vsetq_lane_f64(m00.im, vsetq_lane_f64(m00.re, vdupq_n_f64(0.0), 0), 1);
        let m01_vec = vsetq_lane_f64(m01.im, vsetq_lane_f64(m01.re, vdupq_n_f64(0.0), 0), 1);
        let m10_vec = vsetq_lane_f64(m10.im, vsetq_lane_f64(m10.re, vdupq_n_f64(0.0), 0), 1);
        let m11_vec = vsetq_lane_f64(m11.im, vsetq_lane_f64(m11.re, vdupq_n_f64(0.0), 0), 1);
        
        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let amp0_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp1_ptr = amps.as_ptr().add(flipped_idx) as *const f64;

                    let amp0_vec = vld1q_f64(amp0_ptr);
                    let amp1_vec = vld1q_f64(amp1_ptr);

                    // res0 = m00 * amp0 + m01 * amp1
                    let term1_res0 = mul_complex_f64x2(m00_vec, amp0_vec);
                    let term2_res0 = mul_complex_f64x2(m01_vec, amp1_vec);
                    let res0 = vaddq_f64(term1_res0, term2_res0);

                    // res1 = m10 * amp0 + m11 * amp1
                    let term1_res1 = mul_complex_f64x2(m10_vec, amp0_vec);
                    let term2_res1 = mul_complex_f64x2(m11_vec, amp1_vec);
                    let res1 = vaddq_f64(term1_res1, term2_res1);

                    vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, res0);
                    vst1q_f64(amps.as_mut_ptr().add(flipped_idx) as *mut f64, res1);
                }
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_rz_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        m00: Complex64, m11: Complex64
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let m00_vec = vsetq_lane_f64(m00.im, vsetq_lane_f64(m00.re, vdupq_n_f64(0.0), 0), 1);
        let m11_vec = vsetq_lane_f64(m11.im, vsetq_lane_f64(m11.re, vdupq_n_f64(0.0), 0), 1);

        // iterate over all elements and apply phase based on mask_bit
        for i in 0..total_elements {
            unsafe {
                let amp_ptr = amps.as_ptr().add(i) as *const f64;
                let amp_vec = vld1q_f64(amp_ptr);
                let res_vec;

                if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                    res_vec = mul_complex_f64x2(m00_vec, amp_vec);
                } else { // if the mask_bit is 1 in current index i
                    res_vec = mul_complex_f64x2(m11_vec, amp_vec);
                }
                vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, res_vec);
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_cnot_simd(amps: &mut [Complex64], control_mask: usize, target_mask: usize) {
        let total_elements = amps.len();

        // iterate over states where the target bit is 0.
        // this ensures each pair (i, i ^ target_mask) is visited exactly once,
        // specifically when 'i' is the smaller index of the pair.
        for i in 0..total_elements {
            // check if control qubit is 1 and target qubit is 0 in the current state 'i'
            if (i & control_mask) == control_mask && (i & target_mask) == 0 {
                let j = i | target_mask; // get the state where target qubit is 1 (control remains 1)
                if j < total_elements { // bounds checking
                    unsafe {
                        let a_ptr = amps.as_ptr().add(i) as *const f64;
                        let b_ptr = amps.as_ptr().add(j) as *const f64;

                        let amp_a_vec = vld1q_f64(a_ptr);
                        let amp_b_vec = vld1q_f64(b_ptr);

                        vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, amp_b_vec);
                        vst1q_f64(amps.as_mut_ptr().add(j) as *mut f64, amp_a_vec);
                    }
                }
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_cz_simd(amps: &mut [Complex64], control_mask: usize, target_mask: usize) {
        let total_elements = amps.len();
        let neg_one_vec = vdupq_n_f64(-1.0);
        
        // iterate over all elements and apply phase flip if both control and target bits are 1
        for i in 0..total_elements {
            // changed condition to be more explicit for control qubit state
            if (i & (control_mask | target_mask)) == (control_mask | target_mask) {
                unsafe {
                    let amp_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp_vec = vld1q_f64(amp_ptr);
                    let res_vec = vmulq_f64(amp_vec, neg_one_vec);
                    vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, res_vec);
                }
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_controlled_phase_rotation_simd(
        amps: &mut [Complex64],
        control_mask: usize,
        target_mask: usize,
        phase_factor: Complex64,
    ) {
        let total_elements = amps.len();
        
        let pf_vec = vsetq_lane_f64(phase_factor.im, vsetq_lane_f64(phase_factor.re, vdupq_n_f64(0.0), 0), 1);

        // iterate over all elements and apply phase if both control and target bits are 1
        for i in 0..total_elements {
            if (i & (control_mask | target_mask)) == (control_mask | target_mask) {
                unsafe {
                    let amp_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp_vec = vld1q_f64(amp_ptr);
                    let res_vec = mul_complex_f64x2(pf_vec, amp_vec);
                    vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, res_vec);
                }
            }
        }
    }

    #[target_feature(enable = "neon")] // added target feature
    pub unsafe fn apply_reset_all_simd(amps: &mut [Complex64]) {
        let total_elements = amps.len();

        let zero_vec = vdupq_n_f64(0.0);

        // set the state to |0...0>
        if total_elements > 0 {
            unsafe {
                // for the |0...0> state, set real part to 1.0 and imaginary to 0.0
                let one_re_zero_im_vec = vsetq_lane_f64(0.0, vsetq_lane_f64(1.0, vdupq_n_f64(0.0), 0), 1);
                vst1q_f64(amps.as_mut_ptr() as *mut f64, one_re_zero_im_vec);
                for i in 1..total_elements {
                    vst1q_f64(amps.as_mut_ptr().add(i) as *mut f64, zero_vec);
                }
            }
        }
    }
}

// --- riscv64 rvv implementations ---
#[cfg(all(target_arch = "riscv64", target_feature = "v"))]
pub mod riscv64_rvv {
    use super::*;
    use core::arch::riscv64::*;

    // helper for complex multiplication using rvv intrinsics
    // (a_re + i*a_im) * (b_re + i*b_im) = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
    #[inline(always)]
    unsafe fn mul_complex_rvv(
        a_re: float64m1_t,
        a_im: float64m1_t,
        b_re: float64m1_t,
        b_im: float64m1_t,
        vl: u64,
    ) -> (float64m1_t, float64m1_t) {
        unsafe {
            let res_re_term1 = vfmul_vv_f64m1(a_re, b_re, vl);
            let res_re_term2 = vfmul_vv_f64m1(a_im, b_im, vl);
            let res_re = vfsub_vv_f64m1(res_re_term1, res_re_term2, vl);

            let res_im_term1 = vfmul_vv_f64m1(a_re, b_im, vl);
            let res_im_term2 = vfmul_vv_f64m1(a_im, b_re, vl);
            let res_im = vfadd_vv_f64m1(res_im_term1, res_im_term2, vl);

            (res_re, res_im)
        }
    }

    pub unsafe fn apply_hadamard_simd(
        amps: &mut [Complex64],
        norm_factor: Complex64,
        mask_bit: usize,
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let nf_re_scalar = norm_factor.re;
        let nf_im_scalar = norm_factor.im;

        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) }; 
        if vl_complex_pair == 0 {
            return;
        }
        let nf_re_vec = unsafe { vfmv_v_f_f64m1(nf_re_scalar, vl_complex_pair) };
        let nf_im_vec = unsafe { vfmv_v_f_f64m1(nf_im_scalar, vl_complex_pair) };

        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    // load amps[i]
                    let (a_re, a_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(i) as *const f64, vl_complex_pair);
                    // load amps[flipped_idx]
                    let (b_re, b_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(flipped_idx) as *const f64, vl_complex_pair);

                    // sum = a + b
                    let sum_re = vfadd_vv_f64m1(a_re, b_re, vl_complex_pair);
                    let sum_im = vfadd_vv_f64m1(a_im, b_im, vl_complex_pair);

                    // diff = a - b
                    let diff_re = vfsub_vv_f64m1(a_re, b_re, vl_complex_pair);
                    let diff_im = vfsub_vv_f64m1(a_im, b_im, vl_complex_pair);

                    // result_a = norm_factor * sum
                    let (res_a_re, res_a_im) = mul_complex_rvv(nf_re_vec, nf_im_vec, sum_re, sum_im, vl_complex_pair);

                    // result_b = norm_factor * diff
                    let (res_b_re, res_b_im) = mul_complex_rvv(nf_re_vec, nf_im_vec, diff_re, diff_im, vl_complex_pair);

                    // store results
                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, res_a_re, res_a_im, vl_complex_pair);
                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(flipped_idx) as *mut f64, res_b_re, res_b_im, vl_complex_pair);
                }
            }
        }
    }

    pub unsafe fn apply_x_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }

        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let (a_re, a_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(i) as *const f64, vl_complex_pair);
                    let (b_re, b_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(flipped_idx) as *const f64, vl_complex_pair);

                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, b_re, b_im, vl_complex_pair);
                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(flipped_idx) as *mut f64, a_re, a_im, vl_complex_pair);
                }
            }
        }
    }

    pub unsafe fn apply_y_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();
        let neg_i = Complex64::new(0.0, -1.0);
        let pos_i = Complex64::new(0.0, 1.0);

        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }
        let neg_i_re_vec = unsafe { vfmv_v_f_f64m1(neg_i.re, vl_complex_pair) };
        let neg_i_im_vec = unsafe { vfmv_v_f_f64m1(neg_i.im, vl_complex_pair) };
        let pos_i_re_vec = unsafe { vfmv_v_f_f64m1(pos_i.re, vl_complex_pair) };
        let pos_i_im_vec = unsafe { vfmv_v_f_f64m1(pos_i.im, vl_complex_pair) };

        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let (amp0_re, amp0_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(i) as *const f64, vl_complex_pair);
                    let (amp1_re, amp1_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(flipped_idx) as *const f64, vl_complex_pair);

                    let (res0_re, res0_im) = mul_complex_rvv(neg_i_re_vec, neg_i_im_vec, amp1_re, amp1_im, vl_complex_pair);
                    let (res1_re, res1_im) = mul_complex_rvv(pos_i_re_vec, pos_i_im_vec, amp0_re, amp0_im, vl_complex_pair);

                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, res0_re, res0_im, vl_complex_pair);
                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(flipped_idx) as *mut f64, res1_re, res1_im, vl_complex_pair);
                }
            }
        }
    }

    pub unsafe fn apply_z_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }
        
        // iterate over all elements and apply phase flip if mask_bit is 1
        for i in 0..total_elements {
            if (i & bit_val) != 0 { // if the mask_bit is 1 in current index i
                unsafe {
                    let (mut amp_re, mut amp_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(i) as *const f64, vl_complex_pair);
                    amp_re = vfmul_vf_f64m1(amp_re, -1.0, vl_complex_pair);
                    amp_im = vfmul_vf_f64m1(amp_im, -1.0, vl_complex_pair);
                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, amp_re, amp_im, vl_complex_pair);
                }
            }
        }
    }

    pub unsafe fn apply_phaseshift_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        phase_factor: Complex64,
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }
        let pf_re_vec = unsafe { vfmv_v_f_f64m1(phase_factor.re, vl_complex_pair) };
        let pf_im_vec = unsafe { vfmv_v_f_f64m1(phase_factor.im, vl_complex_pair) };

        // iterate over all elements and apply phase shift if mask_bit is 1
        for i in 0..total_elements {
            if (i & bit_val) != 0 { // if the mask_bit is 1 in current index i
                unsafe {
                    let (amp_re, amp_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(i) as *const f64, vl_complex_pair);
                    let (res_re, res_im) = mul_complex_rvv(pf_re_vec, pf_im_vec, amp_re, amp_im, vl_complex_pair);
                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, res_re, res_im, vl_complex_pair);
                }
            }
        }
    }

    pub unsafe fn apply_reset_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }
        let zero_vec = unsafe { vfmv_v_f_f64m1(0.0, vl_complex_pair) };
        let mut norm_sqr = 0.0;

        // first pass: calculate the norm of the |0> subspace for the target qubit
        for i in 0..total_elements {
            if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                unsafe {
                    let (amp_re, amp_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(i) as *const f64, vl_complex_pair);
                    // extract scalar real and imaginary parts for norm calculation
                    let amp_re_scalar = vfmv_f_s_f64m1_f64(amp_re);
                    let amp_im_scalar = vfmv_f_s_f64m1_f64(amp_im);
                    norm_sqr += amp_re_scalar * amp_re_scalar + amp_im_scalar * amp_im_scalar;
                }
            }
        }

        // second pass: normalize and collapse
        if norm_sqr > 1e-12 { // avoid division by zero
            let scale_scalar = 1.0 / norm_sqr.sqrt();
            let scale_re_vec = unsafe { vfmv_v_f_f64m1(scale_scalar, vl_complex_pair) };
            let scale_im_vec = unsafe { vfmv_v_f_f64m1(0.0, vl_complex_pair) }; // imaginary part of scaling factor is 0

            for i in 0..total_elements {
                if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                    unsafe {
                        let (amp_re, amp_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(i) as *const f64, vl_complex_pair);
                        let (res_re, res_im) = mul_complex_rvv(scale_re_vec, scale_im_vec, amp_re, amp_im, vl_complex_pair);
                        vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, res_re, res_im, vl_complex_pair);
                    }
                } else { // if the mask_bit is 1 in current index i
                    unsafe {
                        vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, zero_vec, zero_vec, vl_complex_pair);
                    }
                }
            }
        } else {
            // if zero probability, set to |0> state
            // this handles cases where the state was entirely in the |1> subspace
            if total_elements > 0 {
                unsafe {
                    let one_re = vfmv_v_f_f64m1(1.0, vl_complex_pair);
                    let zero_im = vfmv_v_f_f64m1(0.0, vl_complex_pair);
                    vsseg2e64_v_f64m1(amps.as_mut_ptr() as *mut f64, one_re, zero_im, vl_complex_pair);
                    for i in 1..total_elements {
                        vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, zero_vec, zero_vec, vl_complex_pair);
                    }
                }
            }
        }
    }

    pub unsafe fn apply_swap_simd(amps: &mut [Complex64], q1_mask: usize, q2_mask: usize) {
        let total_elements = amps.len();
        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }

        // iterate over states where both q1_mask and q2_mask bits are 0.
        // this ensures each pair (i, i ^ target_mask) is visited exactly once,
        // specifically when 'i' is the smaller index of the pair.
        let combined_mask = q1_mask | q2_mask;
        for i in 0..total_elements {
            if (i & combined_mask) == 0 { // if both q1 and q2 bits are 0 in current index 'i'
                let idx_q1_0_q2_1 = i | q2_mask; // state with q1=0, q2=1
                let idx_q1_1_q2_0 = i | q1_mask; // state with q1=1, q2=0

                // these two indices form the pair that needs swapping.
                // no need for 'if i < j' because we explicitly construct the two indices from a base state.
                if idx_q1_0_q2_1 < total_elements && idx_q1_1_q2_0 < total_elements { // bounds checking
                    unsafe {
                        let (a_re, a_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(idx_q1_0_q2_1) as *const f64, vl_complex_pair);
                        let (b_re, b_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(idx_q1_1_q2_0) as *const f64, vl_complex_pair);

                        vsseg2e64_v_f64m1(amps.as_mut_ptr().add(idx_q1_0_q2_1) as *mut f64, b_re, b_im, vl_complex_pair);
                        vsseg2e64_v_f64m1(amps.as_mut_ptr().add(idx_q1_1_q2_0) as *mut f64, a_re, a_im, vl_complex_pair);
                    }
                }
            }
        }
    }

    pub unsafe fn apply_controlled_swap_simd(
        amps: &mut [Complex64],
        control_mask: usize,
        target1_mask: usize,
        target2_mask: usize,
    ) {
        let total_elements = amps.len();
        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }

        // iterate over states where control, t1, and t2 bits are all 0.
        // this ensures each relevant pair is visited exactly once.
        let combined_mask = control_mask | target1_mask | target2_mask;
        for i in 0..total_elements {
            if (i & combined_mask) == 0 { // if control, t1, t2 bits are 0 in current index 'i'
                // construct the two states that need to be swapped when control is 1
                let idx_c1_t10_t21 = i | control_mask | target2_mask; // state with control=1, t1=0, t2=1
                let idx_c1_t11_t20 = i | control_mask | target1_mask; // state with control=1, t1=1, t2=0

                // these two indices form the pair that needs swapping.
                // no need for 'if i < j' because we explicitly construct the two indices from a base state.
                if idx_c1_t10_t21 < total_elements && idx_c1_t11_t20 < total_elements { // bounds checking
                    unsafe {
                        let (a_re, a_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(idx_c1_t10_t21) as *const f64, vl_complex_pair);
                        let (b_re, b_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(idx_c1_t11_t20) as *const f64, vl_complex_pair);

                        vsseg2e64_v_f64m1(amps.as_mut_ptr().add(idx_c1_t10_t21) as *mut f64, b_re, b_im, vl_complex_pair);
                        vsseg2e64_v_f64m1(amps.as_mut_ptr().add(idx_c1_t11_t20) as *mut f64, a_re, a_im, vl_complex_pair);
                    }
                }
            }
        }
    }

    pub unsafe fn apply_rx_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        m00: Complex64,
        m01: Complex64,
        m10: Complex64,
        m11: Complex64,
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }
        let m00_re_vec = unsafe { vfmv_v_f_f64m1(m00.re, vl_complex_pair) };
        let m00_im_vec = unsafe { vfmv_v_f_f64m1(m00.im, vl_complex_pair) };
        let m01_re_vec = unsafe { vfmv_v_f_f64m1(m01.re, vl_complex_pair) };
        let m01_im_vec = unsafe { vfmv_v_f_f64m1(m01.im, vl_complex_pair) };
        let m10_re_vec = unsafe { vfmv_v_f_f64m1(m10.re, vl_complex_pair) };
        let m10_im_vec = unsafe { vfmv_v_f_f64m1(m10.im, vl_complex_pair) };
        let m11_re_vec = unsafe { vfmv_v_f_f64m1(m11.re, vl_complex_pair) };
        let m11_im_vec = unsafe { vfmv_v_f_f64m1(m11.im, vl_complex_pair) };

        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let (amp0_re, amp0_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(i) as *const f64, vl_complex_pair);
                    let (amp1_re, amp1_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(flipped_idx) as *const f64, vl_complex_pair);

                    // res0 = m00 * amp0 + m01 * amp1
                    let (term1_res0_re, term1_res0_im) = mul_complex_rvv(m00_re_vec, m00_im_vec, amp0_re, amp0_im, vl_complex_pair);
                    let (term2_res0_re, term2_res0_im) = mul_complex_rvv(m01_re_vec, m01_im_vec, amp1_re, amp1_im, vl_complex_pair);
                    let res0_re = vfadd_vv_f64m1(term1_res0_re, term2_res0_re, vl_complex_pair);
                    let res0_im = vfadd_vv_f64m1(term1_res0_im, term2_res0_im, vl_complex_pair);

                    // res1 = m10 * amp0 + m11 * amp1
                    let (term1_res1_re, term1_res1_im) = mul_complex_rvv(m10_re_vec, m10_im_vec, amp0_re, amp0_im, vl_complex_pair);
                    let (term2_res1_re, term2_res1_im) = mul_complex_rvv(m11_re_vec, m11_im_vec, amp1_re, amp1_im, vl_complex_pair);
                    let res1_re = vfadd_vv_f64m1(term1_res1_re, term2_res1_re, vl_complex_pair);
                    let res1_im = vfadd_vv_f64m1(term1_res1_im, term2_res1_im, vl_complex_pair);

                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, res0_re, res0_im, vl_complex_pair);
                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(flipped_idx) as *mut f64, res1_re, res1_im, vl_complex_pair);
                }
            }
        }
    }

    pub unsafe fn apply_ry_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        m00: Complex64,
        m01: Complex64,
        m10: Complex64,
        m11: Complex64,
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }
        let m00_re_vec = unsafe { vfmv_v_f_f64m1(m00.re, vl_complex_pair) };
        let m00_im_vec = unsafe { vfmv_v_f_f64m1(m00.im, vl_complex_pair) };
        let m01_re_vec = unsafe { vfmv_v_f_f64m1(m01.re, vl_complex_pair) };
        let m01_im_vec = unsafe { vfmv_v_f_f64m1(m01.im, vl_complex_pair) };
        let m10_re_vec = unsafe { vfmv_v_f_f64m1(m10.re, vl_complex_pair) };
        let m10_im_vec = unsafe { vfmv_v_f_f64m1(m10.im, vl_complex_pair) };
        let m11_re_vec = unsafe { vfmv_v_f_f64m1(m11.re, vl_complex_pair) };
        let m11_im_vec = unsafe { vfmv_v_f_f64m1(m11.im, vl_complex_pair) };
        
        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let (amp0_re, amp0_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(i) as *const f64, vl_complex_pair);
                    let (amp1_re, amp1_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(flipped_idx) as *const f64, vl_complex_pair);

                    // res0 = m00 * amp0 + m01 * amp1
                    let (term1_res0_re, term1_res0_im) = mul_complex_rvv(m00_re_vec, m00_im_vec, amp0_re, amp0_im, vl_complex_pair);
                    let (term2_res0_re, term2_res0_im) = mul_complex_rvv(m01_re_vec, m01_im_vec, amp1_re, amp1_im, vl_complex_pair);
                    let res0_re = vfadd_vv_f64m1(term1_res0_re, term2_res0_re, vl_complex_pair);
                    let res0_im = vfadd_vv_f64m1(term1_res0_im, term2_res0_im, vl_complex_pair);

                    // res1 = m10 * amp0 + m11 * amp1
                    let (term1_res1_re, term1_res1_im) = mul_complex_rvv(m10_re_vec, m10_im_vec, amp0_re, amp0_im, vl_complex_pair);
                    let (term2_res1_re, term2_res1_im) = mul_complex_rvv(m11_re_vec, m11_im_vec, amp1_re, amp1_im, vl_complex_pair);
                    let res1_re = vfadd_vv_f64m1(term1_res1_re, term2_res1_re, vl_complex_pair);
                    let res1_im = vfadd_vv_f64m1(term1_res1_im, term2_res1_im, vl_complex_pair);

                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, res0_re, res0_im, vl_complex_pair);
                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(flipped_idx) as *mut f64, res1_re, res1_im, vl_complex_pair);
                }
            }
        }
    }

    pub unsafe fn apply_rz_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        m00: Complex64,
        m11: Complex64,
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }
        let m00_re_vec = unsafe { vfmv_v_f_f64m1(m00.re, vl_complex_pair) };
        let m00_im_vec = unsafe { vfmv_v_f_f64m1(m00.im, vl_complex_pair) };
        let m11_re_vec = unsafe { vfmv_v_f_f64m1(m11.re, vl_complex_pair) };
        let m11_im_vec = unsafe { vfmv_v_f_f64m1(m11.im, vl_complex_pair) };

        // iterate over all elements and apply phase based on mask_bit
        for i in 0..total_elements {
            unsafe {
                let (amp_re, amp_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(i) as *const f64, vl_complex_pair);
                let (res_re, res_im);

                if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                    (res_re, res_im) = mul_complex_rvv(m00_re_vec, m00_im_vec, amp_re, amp_im, vl_complex_pair);
                } else { // if the mask_bit is 1 in current index i
                    (res_re, res_im) = mul_complex_rvv(m11_re_vec, m11_im_vec, amp_re, amp_im, vl_complex_pair);
                }
                vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, res_re, res_im, vl_complex_pair);
            }
        }
    }

    pub unsafe fn apply_cnot_simd(amps: &mut [Complex64], control_mask: usize, target_mask: usize) {
        let total_elements = amps.len();
        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }

        // iterate over states where the target bit is 0.
        // this ensures each pair (i, i ^ target_mask) is visited exactly once,
        // specifically when 'i' is the smaller index of the pair.
        for i in 0..total_elements {
            // check if control qubit is 1 and target qubit is 0 in the current state 'i'
            if (i & control_mask) == control_mask && (i & target_mask) == 0 {
                let j = i | target_mask; // get the state where target qubit is 1 (control remains 1)
                if j < total_elements { // bounds checking
                    unsafe {
                        let (a_re, a_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(i) as *const f64, vl_complex_pair);
                        let (b_re, b_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(j) as *const f64, vl_complex_pair);

                        vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, b_re, b_im, vl_complex_pair);
                        vsseg2e64_v_f64m1(amps.as_mut_ptr().add(j) as *mut f64, a_re, a_im, vl_complex_pair);
                    }
                }
            }
        }
    }

    pub unsafe fn apply_cz_simd(amps: &mut [Complex64], control_mask: usize, target_mask: usize) {
        let total_elements = amps.len();

        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }
        
        // iterate over all elements and apply phase flip if both control and target bits are 1
        for i in 0..total_elements {
            // changed condition to be more explicit for control qubit state
            if (i & (control_mask | target_mask)) == (control_mask | target_mask) {
                unsafe {
                    let (mut amp_re, mut amp_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(i) as *const f64, vl_complex_pair);
                    amp_re = vfmul_vf_f64m1(amp_re, -1.0, vl_complex_pair);
                    amp_im = vfmul_vf_f64m1(amp_im, -1.0, vl_complex_pair);
                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, amp_re, amp_im, vl_complex_pair);
                }
            }
        }
    }

    pub unsafe fn apply_controlled_phase_rotation_simd(
        amps: &mut [Complex64],
        control_mask: usize,
        target_mask: usize,
        phase_factor: Complex64,
    ) {
        let total_elements = amps.len();

        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }
        let pf_re_vec = unsafe { vfmv_v_f_f64m1(phase_factor.re, vl_complex_pair) };
        let pf_im_vec = unsafe { vfmv_v_f_f64m1(phase_factor.im, vl_complex_pair) };

        // iterate over all elements and apply phase if both control and target bits are 1
        for i in 0..total_elements {
            if (i & (control_mask | target_mask)) == (control_mask | target_mask) {
                unsafe {
                    let (amp_re, amp_im) = vlseg2e64_v_f64m1(amps.as_ptr().add(i) as *const f64, vl_complex_pair);
                    let (res_re, res_im) = mul_complex_rvv(pf_re_vec, pf_im_vec, amp_re, amp_im, vl_complex_pair);
                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, res_re, res_im, vl_complex_pair);
                }
            }
        }
    }

    pub unsafe fn apply_reset_all_simd(amps: &mut [Complex64]) {
        let total_elements = amps.len();

        let vl_complex_pair = unsafe { vsetvl(2, VLENB::V64, LMUL::M1) };
        if vl_complex_pair == 0 { return; }
        let zero_vec = unsafe { vfmv_v_f_f64m1(0.0, vl_complex_pair) };

        // set the state to |0...0>
        if total_elements > 0 {
            unsafe {
                // for the |0...0> state, set real part to 1.0 and imaginary to 0.0
                let one_re = vfmv_v_f_f64m1(1.0, vl_complex_pair);
                let zero_im = vfmv_v_f_f64m1(0.0, vl_complex_pair);
                vsseg2e64_v_f64m1(amps.as_mut_ptr() as *mut f64, one_re, zero_im, vl_complex_pair);
                for i in 1..total_elements {
                    vsseg2e64_v_f64m1(amps.as_mut_ptr().add(i) as *mut f64, zero_vec, zero_vec, vl_complex_pair);
                }
            }
        }
    }
}

// --- power64 vsx implementations ---
#[cfg(all(target_arch = "powerpc64", target_feature = "vsx"))]
pub mod power_vsx {
    use super::*;
    use core::arch::powerpc64::*;

    // helper for complex multiplication using vsx intrinsics for vector_f64
    // (a_re + i*a_im) * (b_re + i*b_im) = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
    #[inline(always)]
    unsafe fn mul_complex_vsx(a: vector_f64, b: vector_f64) -> vector_f64 {
        // a = [a_re, a_im]
        // b = [b_re, b_im]

        // this is a common pattern for complex multiplication with simd:
        // c.re = a.re * b.re - a.im * b.im
        // c.im = a.re * b.im + a.im * b.re

        // extract components using direct access to array elements
        let a_re = a.0[0];
        let a_im = a.0[1];
        let b_re = b.0[0];
        let b_im = b.0[1];

        // perform scalar multiplications
        let term1_re = a_re * b_re;
        let term2_re = a_im * b_im;
        let term1_im = a_re * b_im;
        let term2_im = a_im * b_re;

        // combine for final real and imaginary parts
        let res_re = term1_re - term2_re;
        let res_im = term1_im + term2_im;

        // combine into a vector_f64 [res_re, res_im]
        __vector_set_f64(res_re, res_im)
    }

    pub unsafe fn apply_hadamard_simd(
        amps: &mut [Complex64],
        norm_factor: Complex64,
        mask_bit: usize,
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let nf_vec = __vector_set_f64(norm_factor.re, norm_factor.im);
        
        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let a_ptr = amps.as_ptr().add(i) as *const f64;
                    let b_ptr = amps.as_ptr().add(flipped_idx) as *const f64;

                    let a_vec = lxvd2x(a_ptr, 0); // load two doubles into a vector_f64
                    let b_vec = lxvd2x(b_ptr, 0);

                    let sum_vec = __vaddfp(a_vec, b_vec); // vector add double-precision
                    let diff_vec = __vsubfp(a_vec, b_vec); // vector subtract double-precision

                    let res_a = mul_complex_vsx(nf_vec, sum_vec);
                    let res_b = mul_complex_vsx(nf_vec, diff_vec);

                    stxvd2x(res_a, amps.as_mut_ptr().add(i) as *mut f64, 0);
                    stxvd2x(res_b, amps.as_mut_ptr().add(flipped_idx) as *mut f64, 0);
                }
            }
        }
    }

    pub unsafe fn apply_x_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let a_ptr = amps.as_ptr().add(i) as *const f64;
                    let b_ptr = amps.as_ptr().add(flipped_idx) as *const f64;

                    let a_vec = lxvd2x(a_ptr, 0);
                    let b_vec = lxvd2x(b_ptr, 0);

                    stxvd2x(b_vec, amps.as_mut_ptr().add(i) as *mut f64, 0);
                    stxvd2x(a_vec, amps.as_mut_ptr().add(flipped_idx) as *mut f64, 0);
                }
            }
        }
    }

    pub unsafe fn apply_y_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();
        let neg_i = Complex64::new(0.0, -1.0);
        let pos_i = Complex64::new(0.0, 1.0);

        let neg_i_vec = __vector_set_f64(neg_i.re, neg_i.im);
        let pos_i_vec = __vector_set_f64(pos_i.re, pos_i.im);
        
        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let amp0_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp1_ptr = amps.as_ptr().add(flipped_idx) as *const f64;

                    let amp0_vec = lxvd2x(amp0_ptr, 0);
                    let amp1_vec = lxvd2x(amp1_ptr, 0);

                    let res0 = mul_complex_vsx(neg_i_vec, amp1_vec);
                    let res1 = mul_complex_vsx(pos_i_vec, amp0_vec);

                    stxvd2x(res0, amps.as_mut_ptr().add(i) as *mut f64, 0);
                    stxvd2x(res1, amps.as_mut_ptr().add(flipped_idx) as *mut f64, 0);
                }
            }
        }
    }

    pub unsafe fn apply_z_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();
        // vector of all -1.0 to negate both real and imag parts
        let neg_one_vec = __vector_splats_f64(-1.0); 

        // iterate over all elements and apply phase flip if mask_bit is 1
        for i in 0..total_elements {
            if (i & bit_val) != 0 { // if the mask_bit is 1 in current index i
                unsafe {
                    let amp_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp_vec = lxvd2x(amp_ptr, 0);
                    let res_vec = __vfmadd(amp_vec, neg_one_vec, __vector_splats_f64(0.0)); // multiply by -1.0
                    stxvd2x(res_vec, amps.as_mut_ptr().add(i) as *mut f64, 0);
                }
            }
        }
    }

    pub unsafe fn apply_phaseshift_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        phase_factor: Complex64,
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();
        let pf_vec = __vector_set_f64(phase_factor.re, phase_factor.im);
        
        // iterate over all elements and apply phase shift if mask_bit is 1
        for i in 0..total_elements {
            if (i & bit_val) != 0 { // if the mask_bit is 1 in current index i
                unsafe {
                    let amp_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp_vec = lxvd2x(amp_ptr, 0);
                    let res_vec = mul_complex_vsx(pf_vec, amp_vec);
                    stxvd2x(res_vec, amps.as_mut_ptr().add(i) as *mut f64, 0);
                }
            }
        }
    }

    pub unsafe fn apply_reset_simd(amps: &mut [Complex64], mask_bit: usize) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();
        let zero_vec = __vector_splats_f64(0.0);
        let mut norm_sqr: f64 = 0.0;

        // first pass: calculate the norm of the |0> subspace for the target qubit
        for i in 0..total_elements {
            if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                unsafe {
                    let amp_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp_vec = lxvd2x(amp_ptr, 0);
                    let re = amp_vec.0[0];
                    let im = amp_vec.0[1];
                    norm_sqr += re * re + im * im;
                }
            }
        }
        
        // second pass: normalize and collapse
        if norm_sqr > 1e-12 { // avoid division by zero
            let scale = 1.0 / norm_sqr.sqrt();
            let scale_vec = __vector_splats_f64(scale); // broadcast scalar to vector

            for i in 0..total_elements {
                if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                    unsafe {
                        let amp_ptr = amps.as_ptr().add(i) as *const f64;
                        let amp_vec = lxvd2x(amp_ptr, 0);
                        let res_vec = __vfmadd(amp_vec, scale_vec, __vector_splats_f64(0.0)); // scale by 1/norm
                        stxvd2x(res_vec, amps.as_mut_ptr().add(i) as *mut f64, 0);
                    }
                } else { // if the mask_bit is 1 in current index i
                    unsafe {
                        stxvd2x(zero_vec, amps.as_mut_ptr().add(i) as *mut f64, 0);
                    }
                }
            }
        } else {
            // if zero probability, set to |0> state
            // this handles cases where the state was entirely in the |1> subspace
            if total_elements > 0 {
                unsafe {
                    let one_re_zero_im_vec = __vector_set_f64(1.0, 0.0);
                    stxvd2x(one_re_zero_im_vec, amps.as_mut_ptr() as *mut f64, 0);
                    for i in 1..total_elements {
                        stxvd2x(zero_vec, amps.as_mut_ptr().add(i) as *mut f64, 0);
                    }
                }
            }
        }
    }

    pub unsafe fn apply_swap_simd(amps: &mut [Complex64], q1_mask: usize, q2_mask: usize) {
        let total_elements = amps.len();
        let combined_mask = q1_mask | q2_mask;
        for i in 0..total_elements {
            // process base states where both qubits are 0
            if (i & combined_mask) == 0 { // if both q1 and q2 bits are 0 in current index 'i'
                let idx_q1_0_q2_1 = i | q2_mask; // state with q1=0, q2=1
                let idx_q1_1_q2_0 = i | q1_mask; // q1=1, q2=0

                // these two indices form the pair that needs swapping.
                // no need for 'if i < j' because we explicitly construct the two indices from a base state.
                if idx_q1_0_q2_1 < total_elements && idx_q1_1_q2_0 < total_elements { // bounds checking
                    unsafe {
                        let a_ptr = amps.as_ptr().add(idx_q1_0_q2_1) as *const f64;
                        let b_ptr = amps.as_ptr().add(idx_q1_1_q2_0) as *const f64;

                        let amp_a_vec = lxvd2x(a_ptr, 0);
                        let amp_b_vec = lxvd2x(b_ptr, 0);

                        stxvd2x(amp_b_vec, amps.as_mut_ptr().add(idx_q1_0_q2_1) as *mut f64, 0);
                        stxvd2x(amp_a_vec, amps.as_mut_ptr().add(idx_q1_1_q2_0) as *mut f64, 0);
                    }
                }
            }
        }
    }

    pub unsafe fn apply_controlled_swap_simd(
        amps: &mut [Complex64],
        control_mask: usize,
        t1_mask: usize,
        t2_mask: usize,
    ) {
        let total_elements = amps.len();
        let combined_mask = control_mask | t1_mask | t2_mask;
        for i in 0..total_elements {
            // process base states where control, t1, and t2 bits are all 0.
            // this ensures each relevant pair is visited exactly once.
            if (i & combined_mask) == 0 { // if control, t1, t2 bits are 0 in current index 'i'
                // construct the two states that need to be swapped when control is 1
                let idx_c1_t10_t21 = i | control_mask | t2_mask; // state with control=1, t1=0, t2=1
                let idx_c1_t11_t20 = i | control_mask | t1_mask; // state with control=1, t1=1, t2=0

                // these two indices form the pair that needs swapping.
                // no need for 'if i < j' because we explicitly construct the two indices from a base state.
                if idx_c1_t10_t21 < total_elements && idx_c1_t11_t20 < total_elements { // bounds checking
                    unsafe {
                        let ptr_a = amps.as_ptr().add(idx_c1_t10_t21) as *const f64;
                        let ptr_b = amps.as_ptr().add(idx_c1_t11_t20) as *const f64;

                        let amp_a_vec = lxvd2x(ptr_a, 0);
                        let amp_b_vec = lxvd2x(ptr_b, 0);

                        stxvd2x(amp_b_vec, amps.as_mut_ptr().add(idx_c1_t10_t21) as *mut f64, 0);
                        stxvd2x(amp_a_vec, amps.as_mut_ptr().add(idx_c1_t11_t20) as *mut f64, 0);
                    }
                }
            }
        }
    }

    pub unsafe fn apply_rx_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        m00: Complex64,
        m01: Complex64,
        m10: Complex64,
        m11: Complex64,
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let m00_vec = __vector_set_f64(m00.re, m00.im);
        let m01_vec = __vector_set_f64(m01.re, m01.im);
        let m10_vec = __vector_set_f64(m10.re, m10.im);
        let m11_vec = __vector_set_f64(m11.re, m11.im);

        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let amp0_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp1_ptr = amps.as_ptr().add(flipped_idx) as *const f64;

                    let amp0_vec = lxvd2x(amp0_ptr, 0);
                    let amp1_vec = lxvd2x(amp1_ptr, 0);

                    // res0 = m00 * amp0 + m01 * amp1
                    let term1_res0 = mul_complex_vsx(m00_vec, amp0_vec);
                    let term2_res0 = mul_complex_vsx(m01_vec, amp1_vec);
                    let res0 = __vaddfp(term1_res0, term2_res0);

                    // res1 = m10 * amp0 + m11 * amp1
                    let term1_res1 = mul_complex_vsx(m10_vec, amp0_vec);
                    let term2_res1 = mul_complex_vsx(m11_vec, amp1_vec);
                    let res1 = __vaddfp(term1_res1, term2_res1);

                    stxvd2x(res0, amps.as_mut_ptr().add(i) as *mut f64, 0);
                    stxvd2x(res1, amps.as_mut_ptr().add(flipped_idx) as *mut f64, 0);
                }
            }
        }
    }

    pub unsafe fn apply_ry_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        m00: Complex64,
        m01: Complex64,
        m10: Complex64,
        m11: Complex64,
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let m00_vec = __vector_set_f64(m00.re, m00.im);
        let m01_vec = __vector_set_f64(m01.re, m01.im);
        let m10_vec = __vector_set_f64(m10.re, m10.im);
        let m11_vec = __vector_set_f64(m11.re, m11.im);
        
        // iterate over pairs (i, flipped_idx) where i has the mask_bit as 0
        // this ensures each pair is processed exactly once
        for i in 0..total_elements {
            let flipped_idx = i | bit_val;
            if (i & bit_val) == 0 && flipped_idx < total_elements { // if mask_bit is 0 in i and flipped_idx is valid
                unsafe {
                    let amp0_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp1_ptr = amps.as_ptr().add(flipped_idx) as *const f64;

                    let amp0_vec = lxvd2x(amp0_ptr, 0);
                    let amp1_vec = lxvd2x(amp1_ptr, 0);

                    // res0 = m00 * amp0 + m01 * amp1
                    let term1_res0 = mul_complex_vsx(m00_vec, amp0_vec);
                    let term2_res0 = mul_complex_vsx(m01_vec, amp1_vec);
                    let res0 = __vaddfp(term1_res0, term2_res0);

                    // res1 = m10 * amp0 + m11 * amp1
                    let term1_res1 = mul_complex_vsx(m10_vec, amp0_vec);
                    let term2_res1 = mul_complex_vsx(m11_vec, amp1_vec);
                    let res1 = __vaddfp(term1_res1, term2_res1);

                    stxvd2x(res0, amps.as_mut_ptr().add(i) as *mut f64, 0);
                    stxvd2x(res1, amps.as_mut_ptr().add(flipped_idx) as *mut f64, 0);
                }
            }
        }
    }

    pub unsafe fn apply_rz_simd(
        amps: &mut [Complex64],
        mask_bit: usize,
        m00: Complex64,
        m11: Complex64,
    ) {
        let bit_val = 1 << mask_bit;
        let total_elements = amps.len();

        let m00_vec = __vector_set_f64(m00.re, m00.im);
        let m11_vec = __vector_set_f64(m11.re, m11.im);

        // iterate over all elements and apply phase based on mask_bit
        for i in 0..total_elements {
            unsafe {
                let amp_ptr = amps.as_ptr().add(i) as *const f64;
                let amp_vec = lxvd2x(amp_ptr, 0);
                let res_vec;

                if (i & bit_val) == 0 { // if the mask_bit is 0 in current index i
                    res_vec = mul_complex_vsx(m00_vec, amp_vec);
                } else { // if the mask_bit is 1 in current index i
                    res_vec = mul_complex_vsx(m11_vec, amp_vec);
                }
                stxvd2x(res_vec, amps.as_mut_ptr().add(i) as *mut f64, 0);
            }
        }
    }

    pub unsafe fn apply_cnot_simd(amps: &mut [Complex64], control_mask: usize, target_mask: usize) {
        let total_elements = amps.len();

        // iterate over states where the target bit is 0.
        // this ensures each pair (i, i ^ target_mask) is visited exactly once,
        // specifically when 'i' is the smaller index of the pair.
        for i in 0..total_elements {
            // check if control qubit is 1 and target qubit is 0 in the current state 'i'
            if (i & control_mask) == control_mask && (i & target_mask) == 0 {
                let j = i | target_mask; // get the state where target qubit is 1 (control remains 1)
                if j < total_elements { // bounds checking
                    unsafe {
                        let a_ptr = amps.as_ptr().add(i) as *const f64;
                        let b_ptr = amps.as_ptr().add(j) as *const f64;

                        let amp_a_vec = lxvd2x(a_ptr, 0);
                        let amp_b_vec = lxvd2x(b_ptr, 0);

                        stxvd2x(amp_b_vec, amps.as_mut_ptr().add(i) as *mut f64, 0);
                        stxvd2x(amp_a_vec, amps.as_mut_ptr().add(j) as *mut f64, 0);
                    }
                }
            }
        }
    }

    pub unsafe fn apply_cz_simd(amps: &mut [Complex64], control_mask: usize, target_mask: usize) {
        let total_elements = amps.len();
        let neg_one_vec = __vector_splats_f64(-1.0);
        
        // iterate over all elements and apply phase flip if both control and target bits are 1
        for i in 0..total_elements {
            // changed condition to be more explicit for control qubit state
            if (i & (control_mask | target_mask)) == (control_mask | target_mask) {
                unsafe {
                    let amp_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp_vec = lxvd2x(amp_ptr, 0);
                    let res_vec = __vfmadd(amp_vec, neg_one_vec, __vector_splats_f64(0.0));
                    stxvd2x(res_vec, amps.as_mut_ptr().add(i) as *mut f64, 0);
                }
            }
        }
    }

    pub unsafe fn apply_controlled_phase_rotation_simd(
        amps: &mut [Complex64],
        control_mask: usize,
        target_mask: usize,
        phase_factor: Complex64,
    ) {
        let total_elements = amps.len();

        let pf_vec = __vector_set_f64(phase_factor.re, phase_factor.im);

        // iterate over all elements and apply phase if both control and target bits are 1
        for i in 0..total_elements {
            if (i & (control_mask | target_mask)) == (control_mask | target_mask) {
                unsafe {
                    let amp_ptr = amps.as_ptr().add(i) as *const f64;
                    let amp_vec = lxvd2x(amp_ptr, 0);
                    let res_vec = mul_complex_vsx(pf_vec, amp_vec);
                    stxvd2x(res_vec, amps.as_mut_ptr().add(i) as *mut f64, 0);
                }
            }
        }
    }

    pub unsafe fn apply_reset_all_simd(amps: &mut [Complex64]) {
        let total_elements = amps.len();

        let zero_vec = __vector_splats_f64(0.0);

        // set the state to |0...0>
        if total_elements > 0 {
            unsafe {
                // for the |0...0> state, set real part to 1.0 and imaginary to 0.0
                let one_re_zero_im_vec = __vector_set_f64(1.0, 0.0);
                stxvd2x(one_re_zero_im_vec, amps.as_mut_ptr() as *mut f64, 0);
                for i in 1..total_elements {
                    stxvd2x(zero_vec, amps.as_mut_ptr().add(i) as *mut f64, 0);
                }
            }
        }
    }
}
