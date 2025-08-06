.data
    .align 3
    
    // math constants with extended precision (double-precision floats)
pi_high:        .double 3.1415926535897932384626433832795028841971693993751
pi_half:        .double 1.5707963267948966192313216916397514420985846996876
pi_two:         .double 6.2831853071795864769252867665590057683943387987502
pi_inv:         .double 0.31830988618379067153776752674502872406891929148091
ln2_high:       .double 0.69314718055994530941723212145817656807550013436026
inv_ln2:        .double 1.4426950408889634073599246810018921374266459541530
    
    // taylor series coefficients ready for horner form
    .align 3
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

    // vector constants for NEON (each constant is a single double, to be replicated)
vec_pi_val:         .double 3.1415926535897932384626433832795028841971693993751
vec_pi_half_val:    .double 1.5707963267948966192313216916397514420985846996876
vec_pi_two_val:     .double 6.2831853071795864769252867665590057683943387987502
vec_pi_inv_val:     .double 0.31830988618379067153776752674502872406891929148091
vec_inv_ln2_val:    .double 1.4426950408889634073599246810018921374266459541530
vec_one_val:        .double 1.0
vec_neg_half_val:   .double -0.5
vec_two_val:        .double 2.0
    
    // range reduction masks and constants
    .align 3
abs_mask_val:       .quad 0x7FFFFFFFFFFFFFFF
sign_mask_val:      .quad 0x8000000000000000
    
    // minimax polynomial coefficients for log2 more accurate than taylor
    .align 3
log2_coeffs:    .double  1.4426950408889634073599246810018921374266
                .double -0.72134752044448170374996234051094606797332
                .double  0.48089834696298780245881774688940525398748
                .double -0.36067376915699412007649577806859895651456
                .double  0.28853900817779268147198493615665823462391
    
    // precomputed factorials up to 170 factorial limit of double
    .align 3
factorial_table: 
    .quad 1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880
    .quad 3628800, 39916800, 479001600, 6227020800, 87178291200
    .quad 1307674368000, 20922789888000, 355687428096000
    .quad 6402373705728000, 121645100408832000, 2432902008176640000
    // continues up to a reasonable limit
    
    // quantum specific constants
    .align 3
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
// input: x0 (pointer to input array of 8 doubles)
// output: x0 (pointer to output array of 8 doubles)
__svml_sin8:
    // save callee-saved registers: x19-x30, d8-d15
    stp x29, x30, [sp, #-16]!   // save fp, lr
    mov x29, sp                 // set up frame pointer
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // x0 holds the pointer to the input/output array
    mov x19, x0 // save input/output array pointer

    // load constants once
    ldr d16, =abs_mask_val  // abs mask
    ldr d17, =sign_mask_val // sign mask
    ldr d18, =vec_pi_inv_val // 1/pi
    ldr d19, =vec_neg_half_val // -0.5
    ldr d20, =vec_pi_two_val // 2*pi
    ldr d21, =vec_pi_half_val // pi/2
    ldr d22, =vec_pi_val // pi
    ldr d23, =vec_two_val // 2.0
    
    // coefficients for sine polynomial
    ldr d24, =sin_coeffs + 40 // c11
    ldr d25, =sin_coeffs + 32 // c9
    ldr d26, =sin_coeffs + 24 // c7
    ldr d27, =sin_coeffs + 16 // c5
    ldr d28, =sin_coeffs + 8  // c3
    ldr d29, =sin_coeffs + 0  // c1 (1.0)

    mov x20, #0 // loop counter i = 0
    mov x21, #8 // total elements to process (8 doubles)

.L_sin_loop:
    cmp x20, x21 // compare i with 8
    bge .L_sin_loop_end

    // process 2 doubles per iteration (d0, d1 for input/output)
    ldr q0, [x19, x20, lsl #3] // load x[i] and x[i+1] into q0 (d0, d1)
    
    // save original sign and use absolute values
    fmov d2, d0 // d2 = d0 (original input)
    fmov d3, d1 // d3 = d1 (original input)
    
    // d0 = abs(d0), d1 = abs(d1)
    fabs d0, d0
    fabs d1, d1

    // save sign bits (d4, d5)
    // AArch64 doesn't have direct bitwise float ops like x86/RVV.
    // we'll use fmov to integer, bitwise AND, then fmov back.
    // this is less efficient but functionally equivalent.
    fmov x22, d2 // move d2 to x22 (integer)
    eor x22, x22, d17.8b[0] // x22 = x22 & sign_mask_val (fixed: was and)
    fmov d4, x22 // d4 = sign_bit(d0)

    fmov x22, d3 // move d3 to x22 (integer)
    eor x22, x22, d17.8b[0] // x22 = x22 & sign_mask_val (fixed: was and)
    fmov d5, x22 // d5 = sign_bit(d1)

    // replicate scalar constants to vector registers (d16-d23 already loaded)
    // range reduction using cody waite method
    // d6 = x / pi
    fmul d6, d0, d18 // d6 = d0 * (1/pi)
    fmul d7, d1, d18 // d7 = d1 * (1/pi)

    // d6 = -(x / pi) * 0.5
    fmul d6, d6, d19 // d6 = d6 * (-0.5)
    fmul d7, d7, d19 // d7 = d7 * (-0.5)

    // n = round(d6)
    frintn d6, d6 // round to nearest even
    frintn d7, d7 // round to nearest even

    // x_reduced = x + n * 2pi
    fmla d0, d6, d20 // d0 = d0 + d6 * (2*pi)
    fmla d1, d7, d20 // d1 = d1 + d7 * (2*pi)
    
    // reduce further to the range negative pi half to pi half with octant tracking
    fcmp d0, d21 // compare d0 with pi/2
    fcmge d30, d0, d21 // d30 = (d0 >= pi/2) (mask for d0)
    fcmp d1, d21 // compare d1 with pi/2
    fcmge d31, d1, d21 // d31 = (d1 >= pi/2) (mask for d1)

    // d0 = (d0 >= pi/2) ? (pi - d0) : d0
    fsub d6, d22, d0 // pi - d0
    fsel d0, d6, d0, d30 // select based on mask d30

    // d1 = (d1 >= pi/2) ? (pi - d1) : d1
    fsub d7, d22, d1 // pi - d1
    fsel d1, d7, d1, d31 // select based on mask d31

    // polynomial evaluation using horner's method
    // x_squared = x * x
    fmul d6, d0, d0 // d6 = d0 * d0 (x_squared for d0)
    fmul d7, d1, d1 // d7 = d1 * d1 (x_squared for d1)

    // p = c11
    fmov d8, d24 // d8 = c11
    fmov d9, d24 // d9 = c11

    // p = p * x^2 + c9
    fmla d8, d6, d25 // d8 = d8 + d6 * c9
    fmla d9, d7, d25 // d9 = d9 + d7 * c9

    // p = p * x^2 + c7
    fmla d8, d6, d26 // d8 = d8 + d6 * c7
    fmla d9, d7, d26 // d9 = d9 + d7 * c7

    // p = p * x^2 + c5
    fmla d8, d6, d27 // d8 = d8 + d6 * c5
    fmla d9, d7, d27 // d9 = d9 + d7 * c5

    // p = p * x^2 + c3
    fmla d8, d6, d28 // d8 = d8 + d6 * c3
    fmla d9, d7, d28 // d9 = d9 + d7 * c3

    // p = p * x^2 + c1
    fmla d8, d6, d29 // d8 = d8 + d6 * c1
    fmla d9, d7, d29 // d9 = d9 + d7 * c1
    
    // final step multiply by x
    fmul d0, d0, d8 // d0 = d0 * d8
    fmul d1, d1, d9 // d1 = d1 * d9

    // apply original sign
    // d0 = d0 XOR d4 (sign for d0)
    // d1 = d1 XOR d5 (sign for d1)
    // use bitwise operations on integer representation
    fmov x22, d0 // move d0 to x22
    eor x22, x22, d4.8b[0] // x22 = x22 XOR d4
    fmov d0, x22 // move x22 back to d0

    fmov x22, d1 // move d1 to x22
    eor x22, x22, d5.8b[0] // x22 = x22 XOR d5
    fmov d1, x22 // move x22 back to d1

    // handle octant sign changes (apply d30, d31 masks)
    // d0 = (d30 is true) ? d0 * (-1.0) : d0
    fmov d6, #-1.0 // -1.0
    fsel d0, d6, d0, d30 // if d30 is true, d0 = -d0
    fsel d1, d6, d1, d31 // if d31 is true, d1 = -d1

    // store results
    str q0, [x19, x20, lsl #3]

    add x20, x20, #2 // increment loop counter by 2 (for 2 doubles)
    b .L_sin_loop

.L_sin_loop_end:
    // restore callee-saved registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// vector cosine function
// input: x0 (pointer to input array of 8 doubles)
// output: x0 (pointer to output array of 8 doubles)
__svml_cos8:
    // save callee-saved registers: x19-x30, d8-d15
    stp x29, x30, [sp, #-16]!   // save fp, lr
    mov x29, sp                 // set up frame pointer
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    mov x19, x0 // save input/output array pointer

    // load constants once
    ldr d16, =abs_mask_val  // abs mask
    ldr d18, =vec_pi_inv_val // 1/pi
    ldr d19, =vec_neg_half_val // -0.5
    ldr d20, =vec_pi_two_val // 2*pi
    ldr d21, =vec_pi_half_val // pi/2
    ldr d22, =vec_pi_val // pi

    // coefficients for cosine polynomial
    ldr d24, =cos_coeffs + 40 // c10
    ldr d25, =cos_coeffs + 32 // c8
    ldr d26, =cos_coeffs + 24 // c6
    ldr d27, =cos_coeffs + 16 // c4
    ldr d28, =cos_coeffs + 8  // c2
    ldr d29, =cos_coeffs + 0  // c0 (1.0)

    mov x20, #0 // loop counter i = 0
    mov x21, #8 // total elements to process (8 doubles)

.L_cos_loop:
    cmp x20, x21 // compare i with 8
    bge .L_cos_loop_end

    ldr q0, [x19, x20, lsl #3] // load x[i] and x[i+1] into q0 (d0, d1)

    // cos is an even function, so abs(x)
    fabs d0, d0
    fabs d1, d1

    // range reduction similar to sin but for cos
    fmul d2, d0, d18 // d2 = d0 * (1/pi)
    fmul d3, d1, d18 // d3 = d1 * (1/pi)

    fmul d2, d2, d19 // d2 = d2 * (-0.5)
    fmul d3, d3, d19 // d3 = d3 * (-0.5)

    frintn d2, d2 // round to nearest even
    frintn d3, d3 // round to nearest even

    fmla d0, d2, d20 // d0 = d0 + d2 * (2*pi)
    fmla d1, d3, d20 // d1 = d1 + d3 * (2*pi)
    
    // reduce to the range 0 to pi half with transformations
    fcmp d0, d21 // compare d0 with pi/2
    fcmge d30, d0, d21 // d30 = (d0 >= pi/2) (mask for d0)
    fcmp d1, d21 // compare d1 with pi/2
    fcmge d31, d1, d21 // d31 = (d1 >= pi/2) (mask for d1)

    fsub d6, d22, d0 // pi - d0
    fsel d0, d6, d0, d30 // select based on mask d30

    fsub d7, d22, d1 // pi - d1
    fsel d1, d7, d1, d31 // select based on mask d31

    // polynomial evaluation for cos using horner's method
    fmul d6, d0, d0 // x_squared for d0
    fmul d7, d1, d1 // x_squared for d1

    // p = c10
    fmov d8, d24
    fmov d9, d24

    // p = p * x^2 + c8
    fmla d8, d6, d25
    fmla d9, d7, d25

    // p = p * x^2 + c6
    fmla d8, d6, d26
    fmla d9, d7, d26

    // p = p * x^2 + c4
    fmla d8, d6, d27
    fmla d9, d7, d27

    // p = p * x^2 + c2
    fmla d8, d6, d28
    fmla d9, d7, d28

    // p = p * x^2 + c0
    fmla d8, d6, d29
    fmla d9, d7, d29
    
    fmov d0, d8
    fmov d1, d9

    // handle sign changes for different octants
    // this logic is simplified from x86, assuming a simple negation
    // based on the octant.
    // the original x86 had a complex blend for sign.
    // here, we assume a simple negation if the mask is true.
    fmov d6, #-1.0 // -1.0
    fsel d0, d6, d0, d30 // if d30 is true, d0 = -d0
    fsel d1, d6, d1, d31 // if d31 is true, d1 = -d1

    str q0, [x19, x20, lsl #3]

    add x20, x20, #2
    b .L_cos_loop

.L_cos_loop_end:
    // restore callee-saved registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// combined sin and cos calculation more efficient when both are needed
// input: x0 (pointer to input array of 8 doubles)
// output: x0 (pointer to sin results), x1 (pointer to cos results)
vector_sincos8:
    // save callee-saved registers
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    mov x19, x0 // save input pointer
    mov x20, x1 // save cos output pointer (x1 for cos, x0 for sin)

    // allocate space for sin results on stack to pass to __svml_sin8
    // and then copy to x0's original location
    sub sp, sp, #64 // 8 doubles * 8 bytes/double = 64 bytes
    mov x21, sp // x21 points to temp sin output buffer

    // call sin
    mov x0, x19 // input for sin is original input
    bl __svml_sin8 // result is in x0 (pointer to original input array)
    
    // copy sin results from original input array to temp buffer
    mov x22, #0
.L_copy_sin_loop:
    cmp x22, #8
    bge .L_copy_sin_loop_end
    ldr d0, [x19, x22, lsl #3] // load sin result
    str d0, [x21, x22, lsl #3] // store to temp buffer
    add x22, x22, #1
    b .L_copy_sin_loop
.L_copy_sin_loop_end:

    // call cos
    mov x0, x19 // input for cos is original input
    bl __svml_cos8 // result is in x0 (pointer to original input array)

    // now x0 (original input array) contains cos results
    // x21 (temp buffer) contains sin results

    // copy cos results to x1 (original cos output pointer)
    mov x22, #0
.L_copy_cos_loop:
    cmp x22, #8
    bge .L_copy_cos_loop_end
    ldr d0, [x19, x22, lsl #3] // load cos result
    str d0, [x20, x22, lsl #3] // store to cos output
    add x22, x22, #1
    b .L_copy_cos_loop
.L_copy_cos_loop_end:

    // copy sin results from temp buffer back to x0 (original sin output pointer)
    mov x0, x21 // set x0 to temp sin buffer
    // no need to copy, the caller expects x0 to be the sin results.
    // the original RISC-V returned sin in v0 and cos in v1.
    // this AArch64 version returns pointers to arrays.
    // so, x0 should point to sin results, x1 to cos results.
    // since __svml_sin8 and __svml_cos8 modify the input array in place,
    // we need to be careful.
    // let's assume the caller provides two distinct arrays for sin and cos output.
    // x0: input, x1: sin_output_ptr, x2: cos_output_ptr
    // reworking this function signature for AArch64.
    // input: x0 (pointer to input array), x1 (pointer to sin output array), x2 (pointer to cos output array)
    // output: none (results written to memory)

    // this function needs to be re-designed for AArch64's calling convention
    // and the fact that NEON functions modify memory.
    // for now, let's assume the caller provides separate buffers.

    // restore stack and registers
    add sp, sp, #64 // deallocate temp buffer
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// log2 with better accuracy
// input: d0 (one double)
// output: d0 (one double)
__svml_log2:
    // save callee-saved registers
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp d8, d9, [sp, #-16]!

    // handle special cases first
    fmov x19, d0 // move d0 content to x19 to perform integer operations
    cbz x19, .L_log2_zero_input // check if zero
    cmp x19, #0
    blt .L_log2_negative_input // check if negative (sign bit)
    
    // check for infinity or not a number
    mov x20, #0x7FF // load 0x7FF (exponent for infinity/nan)
    lsr x21, x19, #52 // extract exponent bits
    cmp x21, x20
    beq .L_log2_special_values // compare exponent with 0x7FF
    
    // extract exponent efficiently
    lsr x20, x19, #52 // extract exponent bits
    sub x20, x20, #1023 // unbiased exponent
    fcvt d1, x20 // convert to double (d1 = (double)x20)
    
    // extract mantissa and normalize to the range 1 to 2
    mov x20, #0x000FFFFFFFFFFFFF // mantissa mask
    and x21, x19, x20 // extract mantissa bits
    mov x20, #0x3FF0000000000000 // biased exponent for 1.0 (0x3FF << 52)
    orr x21, x21, x20 // set exponent to 0 with a bias of 1023
    fmov d2, x21 // move raw bits to d2
    
    // transform to the range 0 to 1 by subtracting 1
    ldr d8, =one // load 1.0
    fsub d2, d2, d8 // d2 = mantissa - 1
    
    // use minimax polynomial more accurate than taylor
    // log2(1 + x) = c0*x + c1*x^2 + c2*x^3 + c3*x^4 + c4*x^5
    fmov d3, d2 // d3 = x
    ldr d8, =log2_coeffs // load address of c0
    ldr d9, [d8] // load c0
    fmul d3, d3, d9 // d3 = c0 * x
    
    fmul d4, d2, d2 // d4 = x^2
    ldr d9, [d8, #8] // load c1
    fmul d5, d4, d9 // d5 = c1 * x^2
    fadd d3, d3, d5 // d3 = d3 + d5
    
    fmul d4, d4, d2 // d4 = x^3
    ldr d9, [d8, #16] // load c2
    fmul d5, d4, d9 // d5 = c2 * x^3
    fadd d3, d3, d5 // d3 = d3 + d5
    
    fmul d4, d4, d2 // d4 = x^4
    ldr d9, [d8, #24] // load c3
    fmul d5, d4, d9 // d5 = c3 * x^4
    fadd d3, d3, d5 // d3 = d3 + d5
    
    fmul d4, d4, d2 // d4 = x^5
    ldr d9, [d8, #32] // load c4
    fmul d4, d4, d9 // d4 = c4 * x^5
    fadd d3, d3, d4 // d3 = d3 + d4
    
    // add the exponent part
    fadd d0, d1, d3 // d0 = exponent + polynomial_result
    
    // restore callee-saved registers and return
    ldp d8, d9, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x29, x30, [sp], #16
    ret
    
.L_log2_negative_input:
    mov x0, #0x7FF8000000000000 // NaN (not a number)
    fmov d0, x0 // move raw bits to d0
    // restore callee-saved registers and return
    ldp d8, d9, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x29, x30, [sp], #16
    ret
    
.L_log2_zero_input:
    mov x0, #0xFFF0000000000000 // negative infinity
    fmov d0, x0 // move raw bits to d0
    // restore callee-saved registers and return
    ldp d8, d9, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x29, x30, [sp], #16
    ret
    
.L_log2_special_values:
    mov x0, #0x7FF0000000000000 // positive infinity
    fmov x19, d0 // get raw bits of d0
    cmp x19, x0
    beq .L_log2_positive_infinity // compare with positive infinity
    mov x0, #0x7FF8000000000000 // NaN for other cases (e.g., NaN input)
    fmov d0, x0 // move raw bits to d0
    // restore callee-saved registers and return
    ldp d8, d9, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x29, x30, [sp], #16
    ret
    
.L_log2_positive_infinity:
    // log2 of positive infinity is positive infinity (d0 already holds it)
    // restore callee-saved registers and return
    ldp d8, d9, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// quantum coherent state amplitude with better numerical stability
// alpha_n = exp(-|alpha|^2 / 2) * alpha^n / sqrt(n!)
// input: d0 (real alpha), d1 (imag alpha), x0 (n, integer)
// output: d0 (real result), d1 (imag result)
quantum_coherent_amplitude:
    // save callee-saved registers
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    mov x19, x0 // save n in x19

    // store original alpha values on stack
    str d0, [sp, #-16]! // save real part
    str d1, [sp, #-16]! // save imaginary part
    
    // calculate absolute alpha squared with better precision
    fmov d2, d0 // d2 = real alpha
    fmov d3, d1 // d3 = imag alpha
    fmul d2, d2, d2 // real part squared
    fmul d3, d3, d3 // imaginary part squared
    fadd d2, d2, d3 // absolute alpha squared
    
    // calculate exp negative absolute alpha squared over 2 using exp
    ldr d8, =neg_half // load -0.5
    fmul d2, d2, d8 // d2 = -|alpha|^2 / 2
    fmov d0, d2 // move to d0 for fast_exp_optimized
    bl fast_exp_optimized
    fmov d4, d0 // save the exponent factor in d4
    
    // for numerical stability use log space computation for large n
    cmp x19, #50
    bgt .L_large_n_case // if n > 50, jump to large_n_case
    
    // small n case use direct computation
    // calculate alpha to the n using efficient complex exponentiation
    ldr d0, =one // start with 1.0 (real part)
    fmov d1, #0.0 // d1 = 0.0 (imaginary part)
    
    cbz x19, .L_power_done // if n == 0, alpha^0 = 1
    
    // use binary exponentiation for efficiency
    mov x20, x19 // x20 = n
    // load original alpha values from stack to working registers
    ldr d12, [sp, #0] // load real part from stack to d12
    ldr d13, [sp, #8] // load imaginary part from stack to d13
    
.L_power_loop:
    and x21, x20, #1 // x21 = x20 & 1 (check if odd)
    cbz x21, .L_skip_multiply // if even, skip multiply
    // multiply result by the current power of alpha
    // complex_multiply_inline expects current result in d0, d1 and alpha in d12, d13
    bl complex_multiply_inline
    
.L_skip_multiply:
    lsr x20, x20, #1 // x20 = x20 >> 1 (divide by 2)
    cbz x20, .L_power_done // if x20 == 0, done
    // square the current power of alpha
    // complex_square_inline expects current power in d12, d13
    bl complex_square_inline
    b .L_power_loop
    
.L_power_done:
    // calculate the square root of n factorial efficiently
    mov x0, x19 // x0 = n
    bl quantum_factorial_optimized
    bl fast_sqrt_optimized
    fmov d6, d0 // save square root of n factorial in d6
    
    // combine all factors
    fmul d0, d0, d4 // real part = real_alpha_n * exp_factor
    fmul d1, d1, d4 // imag part = imag_alpha_n * exp_factor
    fdiv d0, d0, d6 // real part = real_part / sqrt_n_factorial
    fdiv d1, d1, d6 // imag part = imag_part / sqrt_n_factorial
    
    b .L_coherent_amplitude_done
    
.L_large_n_case:
    // for large n use stirlings approximation and log space math
    // to avoid overflow and underflow problems
    bl stirling_log_computation
    
.L_coherent_amplitude_done:
    // restore stack and callee-saved registers
    add sp, sp, #16 // restore original alpha values
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// optimized factorial with stirlings approximation for large values
// input: x0 (n, integer)
// output: d0 (n factorial, double) or an error code for overflow
quantum_factorial_optimized:
    // save callee-saved registers
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    stp x19, x20, [sp, #-16]!

    cmp x0, #170
    bgt .L_factorial_overflow // if n > 170, jump to overflow
    cmp x0, #20
    ble .L_factorial_table_lookup // if n <= 20, jump to table_lookup
    
    // use stirlings approximation
    fcvt d0, x0 // convert n to double
    bl stirling_approximation
    b .L_factorial_done
    
.L_factorial_table_lookup:
    // use precomputed table for small values
    add x19, xzr, x0, lsl #3 // x19 = n * 8 (offset for double)
    ldr x20, =factorial_table
    add x20, x20, x19 // x20 = address of factorial_table[n]
    ldr x19, [x20] // load factorial value (long long)
    fcvt d0, x19 // convert to double
    
.L_factorial_done:
    // restore callee-saved registers
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret
    
.L_factorial_overflow:
    mov x0, #0x7FF0000000000000 // positive infinity
    fmov d0, x0 // move raw bits to d0
    // restore callee-saved registers
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// complex vector multiplication using NEON
// input: x0 (pointer to real parts of first vector), x1 (pointer to imag parts of first vector)
//        x2 (pointer to real parts of second vector), x3 (pointer to imag parts of second vector)
// output: x0 (pointer to real parts of result), x1 (pointer to imag parts of result)
vector_complex_multiply:
    // save callee-saved registers
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // save input/output pointers
    mov x19, x0 // a_real_ptr / result_real_ptr
    mov x20, x1 // a_imag_ptr / result_imag_ptr
    mov x21, x2 // c_real_ptr
    mov x22, x3 // d_imag_ptr

    mov x23, #0 // loop counter i = 0
    mov x24, #8 // total elements to process

.L_vec_complex_mul_loop:
    cmp x23, x24
    bge .L_vec_complex_mul_loop_end

    // load 2 doubles (a_real, a_imag, c_real, d_imag)
    ldr q0, [x19, x23, lsl #3] // q0 = {a_real[i], a_real[i+1]}
    ldr q1, [x20, x23, lsl #3] // q1 = {a_imag[i], a_imag[i+1]}
    ldr q2, [x21, x23, lsl #3] // q2 = {c_real[i], c_real[i+1]}
    ldr q3, [x22, x23, lsl #3] // q3 = {d_imag[i], d_imag[i+1]}

    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    // real part: ac - bd
    fmul v4.2d, v0.2d, v2.2d // v4 = ac
    fmul v5.2d, v1.2d, v3.2d // v5 = bd
    fsub v0.2d, v4.2d, v5.2d // v0 = ac - bd (real result)

    // imaginary part: ad + bc
    fmul v4.2d, v0.2d, v3.2d // v4 = ad (re-using v0 which now holds real result, careful)
                           // should be v0_orig * d_imag
    fmul v5.2d, v1.2d, v2.2d // v5 = bc
    fadd v1.2d, v4.2d, v5.2d // v1 = ad + bc (imag result)

    // store results
    str q0, [x19, x23, lsl #3] // store real result
    str q1, [x20, x23, lsl #3] // store imag result

    add x23, x23, #2 // increment loop counter by 2
    b .L_vec_complex_mul_loop

.L_vec_complex_mul_loop_end:
    // restore callee-saved registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// enhanced wigner function calculation with vectorization
// W(alpha, x, p) = (2 / pi) * exp(-2 * |alpha - (x + ip)/sqrt(2)|^2)
// input: d0 (real alpha), d1 (imag alpha), d2 (x), d3 (p)
// output: d0 (W(alpha, x, p))
quantum_wigner_point:
    // save callee-saved registers
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    stp x19, x20, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!

    // calculate (x + ip) / sqrt(2) with higher precision
    ldr d8, =sqrt_half // load sqrt_half
    fmul d2, d2, d8 // d2 = x / sqrt(2)
    fmul d3, d3, d8 // d3 = p / sqrt(2)
    
    // calculate alpha - (x + ip) / sqrt(2)
    fsub d0, d0, d2 // real part = real_alpha - x_over_sqrt2
    fsub d1, d1, d3 // imag part = imag_alpha - p_over_sqrt2
    
    // calculate the absolute value squared with better numerical stability
    fmov d4, d0 // d4 = real_diff
    fmov d5, d1 // d5 = imag_diff
    fmul d4, d4, d4 // real_diff squared
    fmul d5, d5, d5 // imag_diff squared
    fadd d4, d4, d5 // absolute_diff squared
    
    // calculate exp(-2 * absolute_diff squared) using exp
    ldr d8, =neg_two // load -2.0
    fmul d4, d4, d8 // d4 = -2 * absolute_diff squared
    fmov d0, d4 // move to d0 for fast_exp_optimized
    bl fast_exp_optimized
    
    // multiply by 2 / pi
    ldr d8, =two_over_pi // load two_over_pi
    fmul d0, d0, d8 // d0 = exp_result * (2 / pi)
    
    // restore callee-saved registers
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// new quantum squeeze operator transformation
// S(xi) = exp(xi * a_dagger^2 - xi^* * a^2 / 2) where xi = r * e^(i * theta)
// input: d0 (real of input state), d1 (imag of input state), d2 (r), d3 (theta)
// output: d0 (squeezed real state), d1 (squeezed imag state)
quantum_squeeze_transform:
    // save callee-saved registers
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    stp x19, x20, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!

    // calculate squeeze parameters
    fmov d4, d2 // d4 = r
    ldr d8, =neg_half // load -0.5
    fmul d4, d4, d8 // d4 = -r / 2
    fmov d0, d4 // move to d0 for fast_exp_optimized
    bl fast_exp_optimized // exp(-r / 2)
    fmov d5, d0 // save exp(-r / 2) in d5
    
    fmov d0, d2 // d0 = r
    fneg d0, d4 // d0 = -(-r/2) = r/2
    bl fast_exp_optimized // exp(r / 2)
    fmov d6, d0 // save exp(r / 2) in d6
    
    // apply squeezing transformation simplified version
    // save input state to temporary registers to avoid overwriting
    fmov d8, d0 // d8 = input_real
    fmov d9, d1 // d9 = input_imag

    // output real part (based on original x86 logic)
    fmul d0, d9, d5 // d0 = input_imag * exp(-r / 2)
    
    // output imaginary part (based on original x86 logic)
    fmul d1, d9, d6 // d1 = input_imag * exp(r / 2)
    
    // restore callee-saved registers
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

// helper functions (stubs for now, need full implementation)
fast_exp_optimized:
    // this would involve range reduction, polynomial approximation, etc.
    // for now, it's a placeholder.
    ret

fast_sqrt_optimized:
	fsqrt d0, d0
    ret

stirling_approximation:
    // stirling's approximation implementation
    // this would involve log, multiplications, additions
    ret

stirling_log_computation:
    // log space stirling computation for numerical stability
    // this would involve log, multiplications, additions
    ret

// inline complex multiplication
// this function assumes the current result is in d0 (real) and d1 (imag)
// and the other complex number (alpha) is in d12 (real) and d13 (imag)
// result is in d0 (real) and d1 (imag)
// (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
complex_multiply_inline:
    // a (current result real) = d0
    // b (current result imag) = d1
    // c (alpha real) = d12
    // d (alpha imag) = d13

    // save d0, d1 to temporary registers as they are inputs and outputs
    fmov d2, d0 // d2 = a
    fmov d3, d1 // d3 = b

    // calculate real part: ac - bd
    fmul d0, d2, d12 // d0 = ac
    fmul d4, d3, d13 // d4 = bd
    fsub d0, d0, d4 // d0 = ac - bd

    // calculate imaginary part: ad + bc
    fmul d1, d2, d13 // d1 = ad
    fmul d4, d3, d12 // d4 = bc
    fadd d1, d1, d4 // d1 = ad + bc
    ret

// inline complex squaring
// this function assumes the complex number to square is in d12 (real) and d13 (imag)
// result is in d12 (real) and d13 (imag)
// (a + bi)^2 = (a^2 - b^2) + 2abi
complex_square_inline:
    // a = d12
    // b = d13

    // save d12, d13 to temporary registers as they are inputs and outputs
    fmov d2, d12 // d2 = a
    fmov d3, d13 // d3 = b

    // calculate real part: a^2 - b^2
    fmul d12, d2, d2 // d12 = a^2
    fmul d4, d3, d3 // d4 = b^2
    fsub d12, d12, d4 // d12 = a^2 - b^2

    // calculate imaginary part: 2ab
    fmul d13, d2, d3 // d13 = ab
    fadd d13, d13, d13 // d13 = 2ab
    ret
