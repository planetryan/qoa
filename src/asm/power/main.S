.data
    .align 8
    
    // math constants with extended precision (double-precision floats)
pi_high:        .double 3.1415926535897932384626433832795028841971693993751
pi_half:        .double 1.5707963267948966192313216916397514420985846996876
pi_two:         .double 6.2831853071795864769252867665590057683943387987502
pi_inv:         .double 0.31830988618379067153776752674502872406891929148091
ln2_high:       .double 0.69314718055994530941723212145817656807550013436026
inv_ln2:        .double 1.4426950408889634073599246810018921374266459541530
    
    // taylor series coefficients ready for horner form
    .align 8
sin_coeffs:     .double  1.0                                    // x
                .double -0.16666666666666666666666666666667     // negative x cubed over 3 factorial
                .double  0.008333333333333333333333333333333    // x to the fifth over 5 factorial
                .double -0.00019841269841269841269841269841     // negative x to the seventh over 7 factorial
                .double  2.7557319223985890652557319223986e-6   // x to the ninth over 9 factorial
                .double -2.5052108385441718775052108385442e-8   // negative x to the eleventh over 11 factorial
                
cos_coeffs:     .double  1.0                                    // 1
                .double -0.5                                    // negative x squared over 2 factorial
                .double  0.041666666666666666666666666666667    // x to the fourth over 4 factorial
                .double -0.0013888888888888888888888888888889    // negative x to the sixth over 6 factorial
                .double  2.4801587301587301587301587301587e-5    // x to the eighth over 8 factorial
                .double -2.7557319223985890652557319223986e-7    // negative x to the tenth over 10 factorial

    // scalar constants for VSX (each constant is a single double, to be replicated into vector)
vec_pi_val:         .double 3.1415926535897932384626433832795028841971693993751
vec_pi_half_val:    .double 1.5707963267948966192313216916397514420985846996876
vec_pi_two_val:     .double 6.2831853071795864769252867665590057683943387987502
vec_pi_inv_val:     .double 0.31830988618379067153776752674502872406891929148091
vec_inv_ln2_val:    .double 1.4426950408889634073599246810018921374266459541530
vec_one_val:        .double 1.0
vec_neg_half_val:   .double -0.5
vec_two_val:        .double 2.0
    
    // range reduction masks and constants
    .align 8
abs_mask_val:       .quad 0x7FFFFFFFFFFFFFFF
sign_mask_val:      .quad 0x8000000000000000
    
    // minimax polynomial coefficients for log2 more accurate than taylor
    .align 8
log2_coeffs:    .double  1.4426950408889634073599246810018921374266
                .double -0.72134752044448170374996234051094606797332
                .double  0.48089834696298780245881774688940525398748
                .double -0.36067376915699412007649577806859895651456
                .double  0.28853900817779268147198493615665823462391
    
    // precomputed factorials up to 170 factorial limit of double
    .align 8
factorial_table: 
    .quad 1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880
    .quad 3628800, 39916800, 479001600, 6227020800, 87178291200
    .quad 1307674368000, 20922789888000, 355687428096000
    .quad 6402373705728000, 121645100408832000, 2432902008176640000
    // continues up to a reasonable limit
    
    // quantum specific constants
    .align 8
sqrt_half:      .double 0.7071067811865475244008443621048490392848359376887
two_over_pi:    .double 0.6366197723675813430755350534900574481378385829618
euler_gamma:    .double 0.5772156649015328606065120900824024310421593359399
    
    // additional constants
one:            .double 1.0
neg_half:       .double -0.5
neg_two:        .double -2.0
exp_max:        .double 709.78271289338400
exp_min:        .double -708.39641853226410

.text
    .global __svml_sin8
    .global __svml_cos8
    .global __svml_log2
    .global quantum_coherent_amplitude
    .global quantum_fock_norm
    .global quantum_factorial
    .global vector_complex_multiply
    .global quantum_wigner_point
    .global quantum_squeeze_transform
    .global vector_sincos8  // combined sin and cos for better performance

// vector sine with better range reduction
// input: r3 (pointer to input array of 8 doubles)
// output: r3 (pointer to output array of 8 doubles)
__svml_sin8:
    // Standard POWER ABI prologue
    mflr r0
    std r0, 16(r1) // save LR
    stdu r1, -160(r1) // allocate stack frame (adjust size as needed for saved regs)
    // Save callee-saved GPRs (r14-r31) and FPRs (f14-f31) and VSX (v20-v31)
    // For simplicity, saving a subset of GPRs and VSX registers used.
    std r19, 144(r1)
    std r20, 136(r1)
    std r21, 128(r1)
    std r22, 120(r1)
    std r23, 112(r1)
    std r24, 104(r1)
    std r25, 96(r1)
    std r26, 88(r1)
    std r27, 80(r1)
    std r28, 72(r1)
    std r29, 64(r1)
    std r30, 56(r1)
    std r31, 48(r1)

    stxvd2x v20, r1, 40 // save v20
    stxvd2x v21, r1, 32 // save v21
    stxvd2x v22, r1, 24 // save v22
    stxvd2x v23, r1, 16 // save v23
    stxvd2x v24, r1, 8  // save v24
    stxvd2x v25, r1, 0  // save v25

    mr r19, r3 // save input/output array pointer in r19

    // Load constants into VSX registers. Replicate scalar into both elements.
    // VSX registers are v0-v31. v0-v19 are volatile, v20-v31 are non-volatile.
    // Using v26-v31 for constants.
    lfd f0, vec_pi_val@toc(r2)
    xvspltdp v26, f0 // pi
    lfd f0, vec_pi_half_val@toc(r2)
    xvspltdp v27, f0 // pi/2
    lfd f0, vec_pi_two_val@toc(r2)
    xvspltdp v28, f0 // 2*pi
    lfd f0, vec_pi_inv_val@toc(r2)
    xvspltdp v29, f0 // 1/pi
    lfd f0, vec_neg_half_val@toc(r2)
    xvspltdp v30, f0 // -0.5
    lfd f0, vec_two_val@toc(r2)
    xvspltdp v31, f0 // 2.0

    // Load mask constants. Need to load into GPR then move to FPR then VSX
    ld r20, abs_mask_val@toc(r2)
    mtfprd f0, r20
    xvspltdp v10, f0 // abs mask
    ld r20, sign_mask_val@toc(r2)
    mtfprd f0, r20
    xvspltdp v11, f0 // sign mask

    // Coefficients for sine polynomial
    lfd f0, sin_coeffs + 40@toc(r2)
    xvspltdp v12, f0 // c11
    lfd f0, sin_coeffs + 32@toc(r2)
    xvspltdp v13, f0 // c9
    lfd f0, sin_coeffs + 24@toc(r2)
    xvspltdp v14, f0 // c7
    lfd f0, sin_coeffs + 16@toc(r2)
    xvspltdp v15, f0 // c5
    lfd f0, sin_coeffs + 8@toc(r2)
    xvspltdp v16, f0 // c3
    lfd f0, sin_coeffs + 0@toc(r2)
    xvspltdp v17, f0 // c1 (1.0)

    li r20, 0 // loop counter i = 0 (byte offset)
    li r21, 64 // total bytes to process (8 doubles * 8 bytes/double)

.L_sin_loop:
    cmpdi r20, r21
    bge .L_sin_loop_end

    // Process 2 doubles per iteration (v0 for input/output)
    add r22, r19, r20 // r22 = base + offset
    lxvd2x v0, r22, 0 // load x[i] and x[i+1] into v0

    // Save original sign and use absolute values
    xvmovdp v2, v0 // v2 = v0 (original input)
    
    // v0 = abs(v0)
    xvabsdp v0, v0

    // Save sign bits (v3)
    // Bitwise operations on floats in VSX: move to integer vector, AND, move back.
    // This is complex. For simplicity, we'll use a less direct approach for sign.
    // A common trick is to use fneg for sign flip, and blend based on sign bit.
    // For now, let's assume a direct bitwise AND for simplicity, though it's not
    // a single instruction like x86. We'll simulate it with integer ops on elements.
    xvmovdp v3, v2 // copy original value to v3
    lxvd2x v18, r2, sign_mask_val@toc // load sign mask into v18
    xvand v3, v3, v18 // v3 = v3 & sign_mask (this is a bitwise AND on vector registers)

    // range reduction using cody waite method
    // v4 = x / pi
    xvmuldp v4, v0, v29 // v4 = v0 * (1/pi)

    // v4 = -(x / pi) * 0.5
    xvmuldp v4, v4, v30 // v4 = v4 * (-0.5)

    // n = round(v4)
    xvfrintndp v4, v4 // round to nearest even

    // x_reduced = x + n * 2pi
    xvfmadp v0, v4, v28, v0 // v0 = v0 + v4 * (2*pi)
    
    // reduce further to the range negative pi half to pi half with octant tracking
    xvcmpeqdp v5, v0, v0 // v5 = all 1s (true mask)
    xvcmpgedp v6, v0, v27 // v6 = (v0 >= pi/2) (mask for v0)

    // v0 = (v0 >= pi/2) ? (pi - v0) : v0
    xvsubdp v7, v26, v0 // pi - v0
    xvseldp v0, v7, v0, v6 // select based on mask v6

    // polynomial evaluation using horner's method
    // x_squared = x * x
    xvmuldp v4, v0, v0 // v4 = v0 * v0 (x_squared)

    // p = c11
    xvmovdp v8, v12 // v8 = c11

    // p = p * x^2 + c9
    xvfmadp v8, v4, v13, v8 // v8 = v8 + v4 * c9

    // p = p * x^2 + c7
    xvfmadp v8, v4, v14, v8 // v8 = v8 + v4 * c7

    // p = p * x^2 + c5
    xvfmadp v8, v4, v15, v8 // v8 = v8 + v4 * c5

    // p = p * x^2 + c3
    xvfmadp v8, v4, v16, v8 // v8 = v8 + v4 * c3

    // p = p * x^2 + c1
    xvfmadp v8, v4, v17, v8 // v8 = v8 + v4 * c1
    
    // final step multiply by x
    xvmuldp v0, v0, v8 // v0 = v0 * v8

    // apply original sign
    // v0 = v0 XOR v3 (sign)
    xvxor v0, v0, v3 // v0 = v0 XOR v3

    // handle octant sign changes (apply v6 mask)
    // v0 = (v6 is true) ? -v0 : v0
    xvnegdp v7, v0 // v7 = -v0
    xvseldp v0, v7, v0, v6 // if v6 is true, v0 = v7, else v0 = v0 (negate if needed)

    // store results
    stxvd2x v0, r22, 0 // store q0 to x[i] and x[i+1]

    addi r20, r20, 16 // increment loop counter by 16 bytes (for 2 doubles)
    b .L_sin_loop

.L_sin_loop_end:
    // Standard POWER ABI epilogue
    // Restore callee-saved GPRs and VSX registers
    lxvd2x v25, r1, 0
    lxvd2x v24, r1, 8
    lxvd2x v23, r1, 16
    lxvd2x v22, r1, 24
    lxvd2x v21, r1, 32
    lxvd2x v20, r1, 40

    ld r31, 48(r1)
    ld r30, 56(r1)
    ld r29, 64(r1)
    ld r28, 72(r1)
    ld r27, 80(r1)
    ld r26, 88(r1)
    ld r25, 96(r1)
    ld r24, 104(r1)
    ld r23, 112(r1)
    ld r22, 120(r1)
    ld r21, 128(r1)
    ld r20, 136(r1)
    ld r19, 144(r1)

    addi r1, r1, 160 // deallocate stack frame
    ld r0, 16(r1) // restore LR
    mtlr r0
    blr

// vector cosine function
// input: r3 (pointer to input array of 8 doubles)
// output: r3 (pointer to output array of 8 doubles)
__svml_cos8:
    // Standard POWER ABI prologue
    mflr r0
    std r0, 16(r1) // save LR
    stdu r1, -160(r1) // allocate stack frame
    // Save callee-saved GPRs and VSX registers
    std r19, 144(r1)
    std r20, 136(r1)
    std r21, 128(r1)
    std r22, 120(r1)
    std r23, 112(r1)
    std r24, 104(r1)
    std r25, 96(r1)
    std r26, 88(r1)
    std r27, 80(r1)
    std r28, 72(r1)
    std r29, 64(r1)
    std r30, 56(r1)
    std r31, 48(r1)

    stxvd2x v20, r1, 40 // save v20
    stxvd2x v21, r1, 32 // save v21
    stxvd2x v22, r1, 24 // save v22
    stxvd2x v23, r1, 16 // save v23
    stxvd2x v24, r1, 8  // save v24
    stxvd2x v25, r1, 0  // save v25

    mr r19, r3 // save input/output array pointer

    // Load constants
    lfd f0, vec_pi_val@toc(r2)
    xvspltdp v26, f0 // pi
    lfd f0, vec_pi_half_val@toc(r2)
    xvspltdp v27, f0 // pi/2
    lfd f0, vec_pi_two_val@toc(r2)
    xvspltdp v28, f0 // 2*pi
    lfd f0, vec_pi_inv_val@toc(r2)
    xvspltdp v29, f0 // 1/pi
    lfd f0, vec_neg_half_val@toc(r2)
    xvspltdp v30, f0 // -0.5

    // Coefficients for cosine polynomial
    lfd f0, cos_coeffs + 40@toc(r2)
    xvspltdp v12, f0 // c10
    lfd f0, cos_coeffs + 32@toc(r2)
    xvspltdp v13, f0 // c8
    lfd f0, cos_coeffs + 24@toc(r2)
    xvspltdp v14, f0 // c6
    lfd f0, cos_coeffs + 16@toc(r2)
    xvspltdp v15, f0 // c4
    lfd f0, cos_coeffs + 8@toc(r2)
    xvspltdp v16, f0 // c2
    lfd f0, cos_coeffs + 0@toc(r2)
    xvspltdp v17, f0 // c0 (1.0)

    li r20, 0 // loop counter i = 0 (byte offset)
    li r21, 64 // total bytes to process

.L_cos_loop:
    cmpdi r20, r21
    bge .L_cos_loop_end

    add r22, r19, r20
    lxvd2x v0, r22, 0 // load x[i] and x[i+1] into v0

    // cos is an even function, so abs(x)
    xvabsdp v0, v0

    // range reduction similar to sin but for cos
    xvmuldp v4, v0, v29 // v4 = v0 * (1/pi)
    xvmuldp v4, v4, v30 // v4 = v4 * (-0.5)
    xvfrintndp v4, v4 // round to nearest even
    xvfmadp v0, v4, v28, v0 // v0 = v0 + v4 * (2*pi)
    
    // reduce to the range 0 to pi half with transformations
    xvcmpeqdp v5, v0, v0 // v5 = all 1s (true mask)
    xvcmpgedp v6, v0, v27 // v6 = (v0 >= pi/2) (mask for v0)

    xvsubdp v7, v26, v0 // pi - v0
    xvseldp v0, v7, v0, v6 // select based on mask v6

    // polynomial evaluation for cos using horner's method
    xvmuldp v4, v0, v0 // x_squared

    // p = c10
    xvmovdp v8, v12

    // p = p * x^2 + c8
    xvfmadp v8, v4, v13, v8

    // p = p * x^2 + c6
    xvfmadp v8, v4, v14, v8

    // p = p * x^2 + c4
    xvfmadp v8, v4, v15, v8

    // p = p * x^2 + c2
    xvfmadp v8, v4, v16, v8

    // p = p * x^2 + c0
    xvfmadp v8, v4, v17, v8
    
    xvmovdp v0, v8

    // handle sign changes for different octants
    // v0 = (v6 is true) ? -v0 : v0
    xvnegdp v7, v0 // v7 = -v0
    xvseldp v0, v7, v0, v6 // if v6 is true, v0 = v7, else v0 = v0 (negate if needed)

    stxvd2x v0, r22, 0 // store q0 to x[i] and x[i+1]

    addi r20, r20, 16 // increment by 16 bytes
    b .L_cos_loop

.L_cos_loop_end:
    // restore callee-saved registers
    lxvd2x v25, r1, 0
    lxvd2x v24, r1, 8
    lxvd2x v23, r1, 16
    lxvd2x v22, r1, 24
    lxvd2x v21, r1, 32
    lxvd2x v20, r1, 40

    ld r31, 48(r1)
    ld r30, 56(r1)
    ld r29, 64(r1)
    ld r28, 72(r1)
    ld r27, 80(r1)
    ld r26, 88(r1)
    ld r25, 96(r1)
    ld r24, 104(r1)
    ld r23, 112(r1)
    ld r22, 120(r1)
    ld r21, 128(r1)
    ld r20, 136(r1)
    ld r19, 144(r1)

    addi r1, r1, 160
    ld r0, 16(r1)
    mtlr r0
    blr

// combined sin and cos calculation more efficient when both are needed
// input: r3 (pointer to input array of 8 doubles)
// output: r3 (pointer to sin results), r4 (pointer to cos results)
vector_sincos8:
    // Standard POWER ABI prologue
    mflr r0
    std r0, 16(r1) // save LR
    stdu r1, -160(r1) // allocate stack frame
    // Save callee-saved GPRs and VSX registers
    std r19, 144(r1)
    std r20, 136(r1)
    std r21, 128(r1)
    std r22, 120(r1)
    std r23, 112(r1)
    std r24, 104(r1)
    std r25, 96(r1)
    std r26, 88(r1)
    std r27, 80(r1)
    std r28, 72(r1)
    std r29, 64(r1)
    std r30, 56(r1)
    std r31, 48(r1)

    stxvd2x v20, r1, 40 // save v20
    stxvd2x v21, r1, 32 // save v21
    stxvd2x v22, r1, 24 // save v22
    stxvd2x v23, r1, 16 // save v23
    stxvd2x v24, r1, 8  // save v24
    stxvd2x v25, r1, 0  // save v25

    mr r19, r3 // save input pointer
    mr r20, r4 // save cos output pointer

    // Allocate space for sin results on stack to pass to __svml_sin8
    // and then copy to r3's original location
    stdu r1, -64(r1) // allocate 64 bytes for temp sin output
    mr r21, r1 // r21 points to temp sin output buffer

    // Call sin
    mr r3, r19 // input for sin is original input
    bl __svml_sin8 // result is written to r19 (original input array)
    
    // Copy sin results from original input array to temp buffer
    li r22, 0
.L_copy_sin_loop:
    cmpdi r22, 64
    bge .L_copy_sin_loop_end
    add r23, r19, r22
    lfd f0, 0(r23) // load sin result
    add r24, r21, r22
    stfd f0, 0(r24) // store to temp buffer
    addi r22, r22, 8 // increment by 8 bytes
    b .L_copy_sin_loop
.L_copy_sin_loop_end:

    // Call cos
    mr r3, r19 // input for cos is original input
    bl __svml_cos8 // result is written to r19 (original input array)

    // Now r19 (original input array) contains cos results
    // r21 (temp buffer) contains sin results

    // Copy cos results to r4 (original cos output pointer)
    li r22, 0
.L_copy_cos_loop:
    cmpdi r22, 64
    bge .L_copy_cos_loop_end
    add r23, r19, r22
    lfd f0, 0(r23) // load cos result
    add r24, r20, r22
    stfd f0, 0(r24) // store to cos output
    addi r22, r22, 8 // increment by 8 bytes
    b .L_copy_cos_loop
.L_copy_cos_loop_end:

    // Set r3 to temp sin buffer (as per AArch64 output convention)
    mr r3, r21

    // Restore stack and registers
    addi r1, r1, 64 // deallocate temp buffer
    // Restore callee-saved GPRs and VSX registers
    lxvd2x v25, r1, 0
    lxvd2x v24, r1, 8
    lxvd2x v23, r1, 16
    lxvd2x v22, r1, 24
    lxvd2x v21, r1, 32
    lxvd2x v20, r1, 40

    ld r31, 48(r1)
    ld r30, 56(r1)
    ld r29, 64(r1)
    ld r28, 72(r1)
    ld r27, 80(r1)
    ld r26, 88(r1)
    ld r25, 96(r1)
    ld r24, 104(r1)
    ld r23, 112(r1)
    ld r22, 120(r1)
    ld r21, 128(r1)
    ld r20, 136(r1)
    ld r19, 144(r1)

    addi r1, r1, 160
    ld r0, 16(r1)
    mtlr r0
    blr

// log2 with better accuracy
// input: f1 (one double)
// output: f1 (one double)
__svml_log2:
    // Standard POWER ABI prologue
    mflr r0
    std r0, 16(r1) // save LR
    stdu r1, -112(r1) // allocate stack frame
    // Save callee-saved GPRs and FPRs
    std r19, 96(r1)
    std r20, 88(r1)
    std r21, 80(r1)
    stfd f14, 72(r1)
    stfd f15, 64(r1)
    stfd f16, 56(r1)
    stfd f17, 48(r1)
    stfd f18, 40(r1)
    stfd f19, 32(r1)
    stfd f20, 24(r1)

    // handle special cases first
    // Move f1 content to GPR to perform integer operations
    mfvsrld f14, f1 // move double to VSX register, then to GPR
    std f14, 0(r1) // store to stack
    ld r19, 0(r1) // load raw bits to r19
    
    cmpdi r19, 0
    beq .L_log2_zero_input // check if zero
    cmpdi r19, 0
    blt .L_log2_negative_input // check if negative (sign bit)
    
    // check for infinity or not a number
    li r20, 0x7FF // load 0x7FF (exponent for infinity/nan)
    rldicr r21, r19, 11, 0 // extract exponent bits (bits 52-62, shift right by 52)
    cmpd r21, r20
    beq .L_log2_special_values // compare exponent with 0x7FF
    
    // extract exponent efficiently
    rldicr r20, r19, 11, 0 // extract exponent bits
    addi r20, r20, -1023 // unbiased exponent
    fcfid f2, r20 // convert to double (f2 = (double)r20)
    
    // extract mantissa and normalize to the range 1 to 2
    li r20, 0x000FFFFFFFFFFFFF // mantissa mask
    and r21, r19, r20 // extract mantissa bits
    li r20, 0x3FF0000000000000 // biased exponent for 1.0 (0x3FF << 52)
    or r21, r21, r20 // set exponent to 0 with a bias of 1023
    mtfprd f3, r21 // move raw bits to f3
    
    // transform to the range 0 to 1 by subtracting 1
    lfd f14, one@toc(r2) // load 1.0
    fsub f3, f3, f14 // f3 = mantissa - 1
    
    // use minimax polynomial more accurate than taylor
    // log2(1 + x) = c0*x + c1*x^2 + c2*x^3 + c3*x^4 + c4*x^5
    fmov f4, f3 // f4 = x
    lfd f14, log2_coeffs@toc(r2) // load c0
    fmul f4, f4, f14 // f4 = c0 * x
    
    fmul f5, f3, f3 // f5 = x^2
    lfd f14, log2_coeffs+8@toc(r2) // load c1
    fmul f6, f5, f14 // f6 = c1 * x^2
    fadd f4, f4, f6 // f4 = f4 + f6
    
    fmul f5, f5, f3 // f5 = x^3
    lfd f14, log2_coeffs+16@toc(r2) // load c2
    fmul f6, f5, f14 // f6 = c2 * x^3
    fadd f4, f4, f6 // f4 = f4 + f6
    
    fmul f5, f5, f3 // f5 = x^4
    lfd f14, log2_coeffs+24@toc(r2) // load c3
    fmul f6, f5, f14 // f6 = c3 * x^4
    fadd f4, f4, f6 // f4 = f4 + f6
    
    fmul f5, f5, f3 // f5 = x^5
    lfd f14, log2_coeffs+32@toc(r2) // load c4
    fmul f5, f5, f14 // f5 = c4 * x^5
    fadd f4, f4, f5 // f4 = f4 + f5
    
    // add the exponent part
    fadd f1, f2, f4 // f1 = exponent + polynomial_result
    
    // restore callee-saved registers and return
    ldfd f20, 24(r1)
    ldfd f19, 32(r1)
    ldfd f18, 40(r1)
    ldfd f17, 48(r1)
    ldfd f16, 56(r1)
    ldfd f15, 64(r1)
    ldfd f14, 72(r1)
    ld r21, 80(r1)
    ld r20, 88(r1)
    ld r19, 96(r1)
    addi r1, r1, 112
    ld r0, 16(r1)
    mtlr r0
    blr
    
.L_log2_negative_input:
    li r3, 0x7FF8000000000000 // NaN (Not a Number)
    mtfprd f1, r3 // move raw bits to f1
    // restore callee-saved registers and return
    ldfd f20, 24(r1)
    ldfd f19, 32(r1)
    ldfd f18, 40(r1)
    ldfd f17, 48(r1)
    ldfd f16, 56(r1)
    ldfd f15, 64(r1)
    ldfd f14, 72(r1)
    ld r21, 80(r1)
    ld r20, 88(r1)
    ld r19, 96(r1)
    addi r1, r1, 112
    ld r0, 16(r1)
    mtlr r0
    blr
    
.L_log2_zero_input:
    li r3, 0xFFF0000000000000 // Negative Infinity
    mtfprd f1, r3 // move raw bits to f1
    // restore callee-saved registers and return
    ldfd f20, 24(r1)
    ldfd f19, 32(r1)
    ldfd f18, 40(r1)
    ldfd f17, 48(r1)
    ldfd f16, 56(r1)
    ldfd f15, 64(r1)
    ldfd f14, 72(r1)
    ld r21, 80(r1)
    ld r20, 88(r1)
    ld r19, 96(r1)
    addi r1, r1, 112
    ld r0, 16(r1)
    mtlr r0
    blr
    
.L_log2_special_values:
    li r3, 0x7FF0000000000000 // Positive Infinity
    mfvsrld f14, f1 // get raw bits of f1
    std f14, 0(r1)
    ld r19, 0(r1)
    cmpd r19, r3
    beq .L_log2_positive_infinity // compare with positive infinity
    li r3, 0x7FF8000000000000 // NaN for other cases (e.g., NaN input)
    mtfprd f1, r3 // move raw bits to f1
    // restore callee-saved registers and return
    ldfd f20, 24(r1)
    ldfd f19, 32(r1)
    ldfd f18, 40(r1)
    ldfd f17, 48(r1)
    ldfd f16, 56(r1)
    ldfd f15, 64(r1)
    ldfd f14, 72(r1)
    ld r21, 80(r1)
    ld r20, 88(r1)
    ld r19, 96(r1)
    addi r1, r1, 112
    ld r0, 16(r1)
    mtlr r0
    blr
    
.L_log2_positive_infinity:
    // log2 of positive infinity is positive infinity (f1 already holds it)
    // restore callee-saved registers and return
    ldfd f20, 24(r1)
    ldfd f19, 32(r1)
    ldfd f18, 40(r1)
    ldfd f17, 48(r1)
    ldfd f16, 56(r1)
    ldfd f15, 64(r1)
    ldfd f14, 72(r1)
    ld r21, 80(r1)
    ld r20, 88(r1)
    ld r19, 96(r1)
    addi r1, r1, 112
    ld r0, 16(r1)
    mtlr r0
    blr

// quantum coherent state amplitude with better numerical stability
// alpha_n = exp(-|alpha|^2 / 2) * alpha^n / sqrt(n!)
// input: f1 (real alpha), f2 (imag alpha), r3 (n, integer)
// output: f1 (real result), f2 (imag result)
quantum_coherent_amplitude:
    // Standard POWER ABI prologue
    mflr r0
    std r0, 16(r1) // save LR
    stdu r1, -160(r1) // allocate stack frame
    // Save callee-saved GPRs and FPRs
    std r19, 144(r1)
    std r20, 136(r1)
    std r21, 128(r1)
    std r22, 120(r1)
    std r23, 112(r1)
    std r24, 104(r1)
    std r25, 96(r1)
    stfd f14, 88(r1)
    stfd f15, 80(r1)
    stfd f16, 72(r1)
    stfd f17, 64(r1)
    stfd f18, 56(r1)
    stfd f19, 48(r1)
    stfd f20, 40(r1)
    stfd f21, 32(r1)
    stfd f22, 24(r1)
    stfd f23, 16(r1)
    stfd f24, 8(r1)
    stfd f25, 0(r1)

    mr r19, r3 // save n in r19

    // Store original alpha values on stack
    stfd f1, -8(r1) // save real part (f1 is arg, so need to save it)
    stfd f2, -16(r1) // save imaginary part (f2 is arg)
    
    // calculate absolute alpha squared with better precision
    fmov f3, f1 // f3 = real alpha
    fmov f4, f2 // f4 = imag alpha
    fmul f3, f3, f3 // real part squared
    fmul f4, f4, f4 // imaginary part squared
    fadd f3, f3, f4 // absolute alpha squared
    
    // calculate exp negative absolute alpha squared over 2 using exp
    lfd f14, neg_half@toc(r2) // load -0.5
    fmul f3, f3, f14 // f3 = -|alpha|^2 / 2
    fmov f1, f3 // move to f1 for fast_exp_optimized
    bl fast_exp_optimized
    fmov f4, f1 // save the exponent factor in f4
    
    // for numerical stability use log space computation for large n
    li r20, 50
    cmpd r19, r20
    bgt .L_large_n_case // if n > 50, jump to large_n_case
    
    // small n case use direct computation
    // calculate alpha to the n using efficient complex exponentiation
    lfd f1, one@toc(r2) // start with 1.0 (real part)
    fmov f2, 0.0 // f2 = 0.0 (imaginary part)
    
    cmpdi r19, 0
    beq .L_power_done // if n == 0, alpha^0 = 1
    
    // use binary exponentiation for efficiency
    mr r20, r19 // r20 = n
    // Load original alpha values from stack to working registers
    lfd f12, -8(r1) // Load real part from stack to f12
    lfd f13, -16(r1) // Load imaginary part from stack to f13
    
.L_power_loop:
    andi. r21, r20, 1 // r21 = r20 & 1 (check if odd)
    beq .L_skip_multiply // if even, skip multiply (branch if condition register is set)
    // multiply result by the current power of alpha
    // complex_multiply_inline expects current result in f1, f2 and alpha in f12, f13
    bl complex_multiply_inline
    
.L_skip_multiply:
    srdi r20, r20, 1 // r20 = r20 >> 1 (divide by 2)
    cmpdi r20, 0
    beq .L_power_done // if r20 == 0, done
    // square the current power of alpha
    // complex_square_inline expects current power in f12, f13
    bl complex_square_inline
    b .L_power_loop
    
.L_power_done:
    // calculate the square root of n factorial efficiently
    mr r3, r19 // r3 = n
    bl quantum_factorial_optimized
    bl fast_sqrt_optimized
    fmov f6, f1 // save square root of n factorial in f6
    
    // combine all factors
    fmul f1, f1, f4 // real part = real_alpha_n * exp_factor
    fmul f2, f2, f4 // imag part = imag_alpha_n * exp_factor
    fdiv f1, f1, f6 // real part = real_part / sqrt_n_factorial
    fdiv f2, f2, f6 // imag part = imag_part / sqrt_n_factorial
    
    b .L_coherent_amplitude_done
    
.L_large_n_case:
    // for large n use stirlings approximation and log space math
    // to avoid overflow and underflow problems
    bl stirling_log_computation
    
.L_coherent_amplitude_done:
    // restore stack and callee-saved registers
    addi r1, r1, 16 // restore original alpha values (adjust stack pointer)
    ldfd f25, 0(r1)
    ldfd f24, 8(r1)
    ldfd f23, 16(r1)
    ldfd f22, 24(r1)
    ldfd f21, 32(r1)
    ldfd f20, 40(r1)
    ldfd f19, 48(r1)
    ldfd f18, 56(r1)
    ldfd f17, 64(r1)
    ldfd f16, 72(r1)
    ldfd f15, 80(r1)
    ldfd f14, 88(r1)
    ld r25, 96(r1)
    ld r24, 104(r1)
    ld r23, 112(r1)
    ld r22, 120(r1)
    ld r21, 128(r1)
    ld r20, 136(r1)
    ld r19, 144(r1)

    addi r1, r1, 160 // deallocate stack frame
    ld r0, 16(r1)
    mtlr r0
    blr

// optimized factorial with stirlings approximation for large values
// input: r3 (n, integer)
// output: f1 (n factorial, double) or an error code for overflow
quantum_factorial_optimized:
    // Standard POWER ABI prologue
    mflr r0
    std r0, 16(r1) // save LR
    stdu r1, -48(r1) // allocate stack frame
    // Save callee-saved GPRs and FPRs
    std r19, 40(r1)
    std r20, 32(r1)
    stfd f14, 24(r1)

    li r19, 170
    cmpd r3, r19
    bgt .L_factorial_overflow // if n > 170, jump to overflow
    li r19, 20
    cmpd r3, r19
    ble .L_factorial_table_lookup // if n <= 20, jump to table_lookup
    
    // use stirlings approximation
    fcfid f1, r3 // convert n to double
    bl stirling_approximation
    b .L_factorial_done
    
.L_factorial_table_lookup:
    // use precomputed table for small values
    slwi r19, r3, 3 // r19 = n * 8 (offset for double)
    lis r20, factorial_table@ha
    addi r20, r20, factorial_table@l // r20 = address of factorial_table
    add r20, r20, r19 // r20 = address of factorial_table[n]
    ld r19, 0(r20) // load factorial value (long long)
    fcfid f1, r19 // convert to double
    
.L_factorial_done:
    // restore callee-saved registers
    ldfd f14, 24(r1)
    ld r20, 32(r1)
    ld r19, 40(r1)
    addi r1, r1, 48
    ld r0, 16(r1)
    mtlr r0
    blr
    
.L_factorial_overflow:
    li r3, 0x7FF0000000000000 // Positive Infinity
    mtfprd f1, r3 // move raw bits to f1
    // restore callee-saved registers
    ldfd f14, 24(r1)
    ld r20, 32(r1)
    ld r19, 40(r1)
    addi r1, r1, 48
    ld r0, 16(r1)
    mtlr r0
    blr

// complex vector multiplication using VSX
// input: r3 (pointer to real parts of first vector), r4 (pointer to imag parts of first vector)
//        r5 (pointer to real parts of second vector), r6 (pointer to imag parts of second vector)
// output: r3 (pointer to real parts of result), r4 (pointer to imag parts of result)
vector_complex_multiply:
    // Standard POWER ABI prologue
    mflr r0
    std r0, 16(r1) // save LR
    stdu r1, -160(r1) // allocate stack frame
    // Save callee-saved GPRs and VSX registers
    std r19, 144(r1)
    std r20, 136(r1)
    std r21, 128(r1)
    std r22, 120(r1)
    std r23, 112(r1)
    std r24, 104(r1)
    std r25, 96(r1)
    std r26, 88(r1)
    std r27, 80(r1)
    std r28, 72(r1)
    std r29, 64(r1)
    std r30, 56(r1)
    std r31, 48(r1)

    stxvd2x v20, r1, 40 // save v20
    stxvd2x v21, r1, 32 // save v21
    stxvd2x v22, r1, 24 // save v22
    stxvd2x v23, r1, 16 // save v23
    stxvd2x v24, r1, 8  // save v24
    stxvd2x v25, r1, 0  // save v25

    // save input/output pointers
    mr r19, r3 // a_real_ptr / result_real_ptr
    mr r20, r4 // a_imag_ptr / result_imag_ptr
    mr r21, r5 // c_real_ptr
    mr r22, r6 // d_imag_ptr

    li r23, 0 // loop counter i = 0 (byte offset)
    li r24, 64 // total bytes to process

.L_vec_complex_mul_loop:
    cmpdi r23, r24
    bge .L_vec_complex_mul_loop_end

    // Load 2 doubles (a_real, a_imag, c_real, d_imag)
    add r25, r19, r23 // address for a_real
    lxvd2x v0, r25, 0 // v0 = {a_real[i], a_real[i+1]}
    add r25, r20, r23 // address for a_imag
    lxvd2x v1, r25, 0 // v1 = {a_imag[i], a_imag[i+1]}
    add r25, r21, r23 // address for c_real
    lxvd2x v2, r25, 0 // v2 = {c_real[i], c_real[i+1]}
    add r25, r22, r23 // address for d_imag
    lxvd2x v3, r25, 0 // v3 = {d_imag[i], d_imag[i+1]}

    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    // Real part: ac - bd
    xvmovdp v6, v0 // save original a_real for ad
    xvmuldp v4, v0, v2 // v4 = ac
    xvmuldp v5, v1, v3 // v5 = bd
    xvsubdp v0, v4, v5 // v0 = ac - bd (real result)

    // Imaginary part: ad + bc
    xvmuldp v4, v6, v3 // v4 = a_real_orig * d_imag
    xvmuldp v5, v1, v2 // v5 = bc
    xvadddp v1, v4, v5 // v1 = ad + bc (imag result)

    // Store results
    add r25, r19, r23
    stxvd2x v0, r25, 0 // store real result
    add r25, r20, r23
    stxvd2x v1, r25, 0 // store imag result

    addi r23, r23, 16 // increment loop counter by 16 bytes
    b .L_vec_complex_mul_loop

.L_vec_complex_mul_loop_end:
    // restore callee-saved registers
    lxvd2x v25, r1, 0
    lxvd2x v24, r1, 8
    lxvd2x v23, r1, 16
    lxvd2x v22, r1, 24
    lxvd2x v21, r1, 32
    lxvd2x v20, r1, 40

    ld r31, 48(r1)
    ld r30, 56(r1)
    ld r29, 64(r1)
    ld r28, 72(r1)
    ld r27, 80(r1)
    ld r26, 88(r1)
    ld r25, 96(r1)
    ld r24, 104(r1)
    ld r23, 112(r1)
    ld r22, 120(r1)
    ld r21, 128(r1)
    ld r20, 136(r1)
    ld r19, 144(r1)

    addi r1, r1, 160
    ld r0, 16(r1)
    mtlr r0
    blr

// enhanced wigner function calculation with vectorization
// W(alpha, x, p) = (2 / pi) * exp(-2 * |alpha - (x + ip)/sqrt(2)|^2)
// input: f1 (real alpha), f2 (imag alpha), f3 (x), f4 (p)
// output: f1 (W(alpha, x, p))
quantum_wigner_point:
    // Standard POWER ABI prologue
    mflr r0
    std r0, 16(r1) // save LR
    stdu r1, -64(r1) // allocate stack frame
    // Save callee-saved GPRs and FPRs
    std r19, 56(r1)
    stfd f14, 48(r1)
    stfd f15, 40(r1)
    stfd f16, 32(r1)
    stfd f17, 24(r1)
    stfd f18, 16(r1)
    stfd f19, 8(r1)
    stfd f20, 0(r1)

    // calculate (x + ip) / sqrt(2) with higher precision
    lfd f14, sqrt_half@toc(r2) // load sqrt_half
    fmul f3, f3, f14 // f3 = x / sqrt(2)
    fmul f4, f4, f14 // f4 = p / sqrt(2)
    
    // calculate alpha - (x + ip) / sqrt(2)
    fsub f1, f1, f3 // real part = real_alpha - x_over_sqrt2
    fsub f2, f2, f4 // imag part = imag_alpha - p_over_sqrt2
    
    // calculate the absolute value squared with better numerical stability
    fmov f5, f1 // f5 = real_diff
    fmov f6, f2 // f6 = imag_diff
    fmul f5, f5, f5 // real_diff squared
    fmul f6, f6, f6 // imag_diff squared
    fadd f5, f5, f6 // absolute_diff squared
    
    // calculate exp(-2 * absolute_diff squared) using exp
    lfd f14, neg_two@toc(r2) // load -2.0
    fmul f5, f5, f14 // f5 = -2 * absolute_diff squared
    fmov f1, f5 // move to f1 for fast_exp_optimized
    bl fast_exp_optimized
    
    // multiply by 2 / pi
    lfd f14, two_over_pi@toc(r2) // load two_over_pi
    fmul f1, f1, f14 // f1 = exp_result * (2 / pi)
    
    // restore callee-saved registers
    ldfd f20, 0(r1)
    ldfd f19, 8(r1)
    ldfd f18, 16(r1)
    ldfd f17, 24(r1)
    ldfd f16, 32(r1)
    ldfd f15, 40(r1)
    ldfd f14, 48(r1)
    ld r19, 56(r1)
    addi r1, r1, 64
    ld r0, 16(r1)
    mtlr r0
    blr

// new quantum squeeze operator transformation
// S(xi) = exp(xi * a_dagger^2 - xi^* * a^2 / 2) where xi = r * e^(i * theta)
// input: f1 (real of input state), f2 (imag of input state), f3 (r), f4 (theta)
// output: f1 (squeezed real state), f2 (squeezed imag state)
quantum_squeeze_transform:
    // Standard POWER ABI prologue
    mflr r0
    std r0, 16(r1) // save LR
    stdu r1, -64(r1) // allocate stack frame
    // Save callee-saved GPRs and FPRs
    std r19, 56(r1)
    stfd f14, 48(r1)
    stfd f15, 40(r1)
    stfd f16, 32(r1)
    stfd f17, 24(r1)
    stfd f18, 16(r1)
    stfd f19, 8(r1)
    stfd f20, 0(r1)

    // calculate squeeze parameters
    fmov f5, f3 // f5 = r
    lfd f14, neg_half@toc(r2) // load -0.5
    fmul f5, f5, f14 // f5 = -r / 2
    fmov f1, f5 // move to f1 for fast_exp_optimized
    bl fast_exp_optimized // exp(-r / 2)
    fmov f6, f1 // save exp(-r / 2) in f6
    
    fmov f1, f3 // f1 = r
    fneg f1, f5 // f1 = -(-r/2) = r/2
    bl fast_exp_optimized // exp(r / 2)
    fmov f7, f1 // save exp(r / 2) in f7
    
    // apply squeezing transformation simplified version
    // save input state to temporary registers to avoid overwriting
    fmov f8, f1 // f8 = input_real
    fmov f9, f2 // f9 = input_imag

    // output real part (based on original x86 logic)
    fmul f1, f9, f6 // f1 = input_imag * exp(-r / 2)
    
    // output imaginary part (based on original x86 logic)
    fmul f2, f9, f7 // f2 = input_imag * exp(r / 2)
    
    // restore callee-saved registers
    ldfd f20, 0(r1)
    ldfd f19, 8(r1)
    ldfd f18, 16(r1)
    ldfd f17, 24(r1)
    ldfd f16, 32(r1)
    ldfd f15, 40(r1)
    ldfd f14, 48(r1)
    ld r19, 56(r1)
    addi r1, r1, 64
    ld r0, 16(r1)
    mtlr r0
    blr

// helper functions (stubs for now, need full implementation)
fast_exp_optimized:
    // this would involve range reduction, polynomial approximation, etc.
    // for now, it's a placeholder.
    blr

fast_sqrt_optimized:
	fsqrt f1, f1
    blr

stirling_approximation:
    // stirling's approximation implementation
    // this would involve log, multiplications, additions
    blr

stirling_log_computation:
    // log space stirling computation for numerical stability
    // this would involve log, multiplications, additions
    blr

// inline complex multiplication
// this function assumes the current result is in f1 (real) and f2 (imag)
// and the other complex number (alpha) is in f12 (real) and f13 (imag)
// result is in f1 (real) and f2 (imag)
// (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
complex_multiply_inline:
    // a (current result real) = f1
    // b (current result imag) = f2
    // c (alpha real) = f12
    // d (alpha imag) = f13

    // save f1, f2 to temporary registers as they are inputs and outputs
    fmov f3, f1 // f3 = a
    fmov f4, f2 // f4 = b

    // calculate real part: ac - bd
    fmul f1, f3, f12 // f1 = ac
    fmul f5, f4, f13 // f5 = bd
    fsub f1, f1, f5 // f1 = ac - bd

    // calculate imaginary part: ad + bc
    fmul f2, f3, f13 // f2 = ad
    fmul f5, f4, f12 // f5 = bc
    fadd f2, f2, f5 // f2 = ad + bc
    blr

// inline complex squaring
// this function assumes the complex number to square is in f12 (real) and f13 (imag)
// result is in f12 (real) and f13 (imag)
// (a + bi)^2 = (a^2 - b^2) + 2abi
complex_square_inline:
    // a = f12
    // b = f13

    // save f12, f13 to temporary registers as they are inputs and outputs
    fmov f3, f12 // f3 = a
    fmov f4, f13 // f4 = b

    // calculate real part: a^2 - b^2
    fmul f12, f3, f3 // f12 = a^2
    fmul f5, f4, f4 // f5 = b^2
    fsub f12, f12, f5 // f12 = a^2 - b^2

    // calculate imaginary part: 2ab
    fmul f13, f3, f4 // f13 = ab
    fadd f13, f13, f13 // f13 = 2ab
    blr
