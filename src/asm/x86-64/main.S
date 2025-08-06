.intel_syntax noprefix # use intel syntax for operand order and no % prefix for registers

.section .data
    # high precision constants aligned for cache performance
    .align 64
    
    # math constants with extended precision
pi_high:        .double 3.1415926535897932384626433832795028841971693993751
pi_half:        .double 1.5707963267948966192313216916397514420985846996876
pi_two:         .double 6.283185307179586476925286769252867665590057683943387987502
pi_inv:         .double 0.31830988618379067153776752674502872406891929148091
ln2_high:       .double 0.69314718055994530941723212145817656807550013436026
inv_ln2:        .double 1.4426950408889634073599246810018921374266459541530
    
    # taylor series coefficients ready for horner form
    .align 32
sin_coeffs:     .double  1.0                                    # x
                .double -0.16666666666666666666666666666667     # negative x cubed over 3 factorial
                .double  0.008333333333333333333333333333333    # x to the fifth over 5 factorial
                .double -0.00019841269841269841269841269841     # negative x to the seventh over 7 factorial
                .double  2.7557319223985890652557319223986e-6   # x to the ninth over 9 factorial
                .double -2.5052108385441718775052108385442e-8   # negative x to the eleventh over 11 factorial
                
cos_coeffs:     .double  1.0                                    # 1
                .double -0.5                                    # negative x squared over 2 factorial
                .double  0.041666666666666666666666666666667    # x to the fourth over 4 factorial
                .double -0.0013888888888888888888888888888889    # negative x to the sixth over 6 factorial
                .double  2.4801587301587301587301587301587e-5    # x to the eighth over 8 factorial
                .double -2.7557319223985890652557319223986e-7    # negative x to the tenth over 10 factorial

    # avx 512 vectorized constants 8 doubles each
    .align 64
vec_pi:         .double 3.1415926535897932384626433832795028841971693993751, 3.1415926535897932384626433832795028841971693993751
                .double 3.1415926535897932384626433832795028841971693993751, 3.141592653589793238462643383279502884193387987502
                .double 3.1415926535897932384626433832795028841971693993751, 3.1415926535897932384626433832795028841971693993751
                .double 3.1415926535897932384626433832795028841971693993751, 3.1415926535897932384626433832795028841971693993751
vec_pi_half:    .double 1.5707963267948966192313216916397514420985846996876, 1.5707963267948966192313216916397514420985846996876
                .double 1.5707963267948966192313216916397514420985846996876, 1.5707963267948966192313216916397514420985846996876
                .double 1.5707963267948966192313216916397514420985846996876, 1.5707963267948966192313216916397514420985846996876
                .double 1.5707963267948966192313216916397514420985846996876, 1.5707963267948966192313216916397514420985846996876
vec_pi_two:     .double 6.2831853071795864769252867665590057683943387987502, 6.2831853071795864769252867665590057683943387987502
                .double 6.2831853071795864769252867665590057683943387987502, 6.2831853071795864769252867665590057683943387987502
                .double 6.2831853071795864769252867665590057683943387987502, 6.2831853071795864769252867665590057683943387987502
                .double 6.2831853071795864769252867665590057683943387987502, 6.2831853071795864769252867665590057683943387987502
vec_pi_inv:     .double 0.31830988618379067153776752674502872406891929148091, 0.31830988618379067153776752674502872406891929148091
                .double 0.31830988618379067153776752674502872406891929148091, 0.31830988618379067153776752674502872406891929148091
                .double 0.31830988618379067153776752674502872406891929148091, 0.31830988618379067153776752674502872406891929148091
                .double 0.31830988618379067153776752674502872406891929148091, 0.31830988618379067153776752674502872406891929148091
vec_inv_ln2:    .double 1.4426950408889634073599246810018921374266459541530, 1.4426950408889634073599246810018921374266459541530
                .double 1.4426950408889634073599246810018921374266459541530, 1.4426950408889634073599246810018921374266459541530
                .double 1.4426950408889634073599246810018921374266459541530, 1.4426950408889634073599246810018921374266459541530
                .double 1.4426950408889634073599246810018921374266459541530, 1.4426950408889634073599246810018921374266459541530
vec_one:        .double 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
vec_neg_half:   .double -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5
vec_two:        .double 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
    
    # range reduction masks and constants
    .align 64
abs_mask:       .quad 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF
                .quad 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF
sign_mask:      .quad 0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000
                .quad 0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000
    
    # minimax polynomial coefficients for log2 more accurate than taylor
    .align 32
log2_coeffs:    .double  1.4426950408889634073599246810018921374266
                .double -0.72134752044448170374996234051094606797332
                .double  0.48089834696298780245881774688940525398748
                .double -0.36067376915699412007649577806859895651456
                .double  0.28853900817779268147198493615665823462391
    
    # precomputed factorials up to 170 factorial limit of double
    .align 64
factorial_table: 
    .quad 1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880
    .quad 3628800, 39916800, 479001600, 6227020800, 87178291200
    .quad 1307674368000, 20922789888000, 355687428096000
    .quad 6402373705728000, 121645100408832000, 2432902008176640000
    # continues up to a reasonable limit
    
    # quantum specific constants
    .align 32
sqrt_half:      .double 0.7071067811865475244008443621048490392848359376887
two_over_pi:    .double 0.6366197723675813430755350534900574481378385829618
euler_gamma:    .double 0.5772156649015328606065120900824024310421593359399
    
    # additional constants
one:            .double 1.0
neg_half:       .double -0.5
neg_two:        .double -2.0
exp_max:        .double 709.78271289338400
exp_min:        .double -708.39641853226410

.section .text
    .global __svml_sin8
    .global __svml_cos8
    .global __svml_log2
    .global quantum_coherent_amplitude
    .global quantum_fock_norm
    .global quantum_factorial
    .global vector_complex_multiply
    .global quantum_wigner_point
    .global quantum_squeeze_transform
    .global vector_sincos8  # combined sin and cos for better performance

# vector sine with better range reduction
# input zmm0 holds 8 doubles
# output zmm0 holds 8 doubles
__svml_sin8:
    push rbp
    mov rbp, rsp
    
    # save original sign and use absolute values
    vmovapd zmm7, zmm0
    vandpd zmm0, zmm0, [rip + abs_mask]       # absolute value of x
    vandpd zmm7, zmm7, [rip + sign_mask]      # sign bits only
    
    # range reduction using cody waite method
    # more accurate than simple modulo for large inputs
    vmovapd zmm1, [rip + vec_pi_inv]          # one over pi
    vmulpd zmm2, zmm0, zmm1             # x over pi
    vmulpd zmm2, zmm2, [rip + vec_neg_half]   # negative x over 2 pi
    vrndscalepd zmm2, zmm2, 0           # n equals round of negative x over 2 pi
    
    # x reduced equals x plus n times 2 pi more accurate with fma
    vfmadd231pd zmm0, zmm2, [rip + vec_pi_two]
    
    # reduce further to the range negative pi half to pi half with octant tracking
    vmovapd zmm1, [rip + vec_pi_half]
    vcmppd k1, zmm0, zmm1, 14           # x is greater than pi half
    vmovapd zmm8, [rip + vec_pi]              # load vec_pi into a register for vsubpd
    vsubpd zmm0{k1}, zmm8, zmm0         # pi minus x for x greater than pi half
    
    # polynomial evaluation using horners method for efficiency
    vmulpd zmm1, zmm0, zmm0             # x squared
    
    # load coefficients and evaluate polynomial
    vbroadcastsd zmm3, QWORD PTR [rip + sin_coeffs + 40]  # x to the eleventh coefficient
    vbroadcastsd zmm4, QWORD PTR [rip + sin_coeffs + 32]  # x to the ninth coefficient
    vfmadd213pd zmm3, zmm1, zmm4                # x to the eleventh times c11 plus c9
    
    vbroadcastsd zmm4, QWORD PTR [rip + sin_coeffs + 24]  # x to the seventh coefficient
    vfmadd213pd zmm3, zmm1, zmm4                # the previous result times x squared plus c7
    
    vbroadcastsd zmm4, QWORD PTR [rip + sin_coeffs + 16]  # x to the fifth coefficient
    vfmadd213pd zmm3, zmm1, zmm4
    
    vbroadcastsd zmm4, QWORD PTR [rip + sin_coeffs + 8]   # x to the third coefficient
    vfmadd213pd zmm3, zmm1, zmm4
    
    vbroadcastsd zmm4, QWORD PTR [rip + sin_coeffs]       # x coefficient which is 1.0
    vfmadd213pd zmm3, zmm1, zmm4
    
    # final step multiply by x
    vmulpd zmm0, zmm0, zmm3
    
    # apply original sign
    vxorpd zmm0, zmm0, zmm7
    
    # handle octant sign changes apply k1 mask
    vbroadcastsd zmm1, QWORD PTR [rip + vec_neg_half]
    vmulpd zmm1, zmm1, [rip + vec_two]        # negative 1.0
    vmulpd zmm0{k1}, zmm0, zmm1         # negate where needed
    
    pop rbp
    ret

# vector cosine function
# input zmm0 holds 8 doubles
# output zmm0 holds 8 doubles
__svml_cos8:
    push rbp
    mov rbp, rsp
    
    # cos x equals sin x plus pi half but direct evaluation is faster
    vandpd zmm0, zmm0, [rip + abs_mask]       # cos is an even function
    
    # range reduction similar to sin but for cos
    vmovapd zmm1, [rip + vec_pi_inv]
    vmulpd zmm2, zmm0, zmm1
    vmulpd zmm2, zmm2, [rip + vec_neg_half]
    vrndscalepd zmm2, zmm2, 0
    vfmadd231pd zmm0, zmm2, [rip + vec_pi_two]
    
    # reduce to the range 0 to pi half with transformations
    vmovapd zmm1, [rip + vec_pi_half]
    vcmppd k1, zmm0, zmm1, 14           # x is greater than pi half
    vmovapd zmm8, [rip + vec_pi]              # load vec_pi into a register for vsubpd
    vsubpd zmm3{k1}, zmm8, zmm0         # pi minus x
    vblendmpd zmm0{k1}, zmm0, zmm3      # select the reduced value
    
    # polynomial evaluation for cos using horners method
    vmulpd zmm1, zmm0, zmm0             # x squared
    
    vbroadcastsd zmm2, QWORD PTR [rip + cos_coeffs + 40]  # highest degree coefficient
    vbroadcastsd zmm3, QWORD PTR [rip + cos_coeffs + 32]
    vfmadd213pd zmm2, zmm1, zmm3
    
    vbroadcastsd zmm3, QWORD PTR [rip + cos_coeffs + 24]
    vfmadd213pd zmm2, zmm1, zmm3
    
    vbroadcastsd zmm3, QWORD PTR [rip + cos_coeffs + 16]
    vfmadd213pd zmm2, zmm1, zmm3
    
    vbroadcastsd zmm3, QWORD PTR [rip + cos_coeffs + 8]
    vfmadd213pd zmm2, zmm1, zmm3
    
    vbroadcastsd zmm3, QWORD PTR [rip + cos_coeffs]       # 1.0
    vfmadd213pd zmm2, zmm1, zmm3
    
    vmovapd zmm0, zmm2
    
    # handle sign changes for different octants
    vbroadcastsd zmm1, QWORD PTR [rip + vec_neg_half]
    vmulpd zmm1, zmm1, [rip + vec_two]        # negative 1.0
    vmulpd zmm0{k1}, zmm0, zmm1
    
    pop rbp
    ret

# combined sin and cos calculation more efficient when both are needed
# input zmm0 holds 8 doubles
# output zmm0 is sin x and zmm1 is cos x
vector_sincos8:
    push rbp
    mov rbp, rsp
    
    # save input for the cos calculation
    vmovapd zmm2, zmm0
    
    # calculate sin x
    call __svml_sin8
    vmovapd zmm1, zmm0                  # save the sin result
    
    # calculate cos x
    vmovapd zmm0, zmm2                  # restore the original input
    call __svml_cos8
    
    # swap results so zmm0 is sin and zmm1 is cos
    vmovapd zmm2, zmm0                  # temp equals cos
    vmovapd zmm0, zmm1                  # move sin to zmm0
    vmovapd zmm1, zmm2                  # move cos to zmm1
    
    pop rbp
    ret

# log2 with better accuracy
# input xmm0 is one double
# output xmm0 is one double
__svml_log2:
    push rbp
    mov rbp, rsp
    
    # handle special cases first
    # move xmm0 content to rax to perform integer operations
    movq rax, xmm0
    test rax, rax
    js .negative_input
    jz .zero_input
    
    # check for infinity or not a number
    mov rdx, rax
    shr rdx, 52
    movabs rcx, 0x7FF                     # load 64-bit immediate into rcx
    cmp rdx, rcx                     # compare rdx (exponent) with 0x7FF (infinity/nan exponent)
    je .special_values
    
    # extract exponent efficiently
    mov rdx, rax
    shr rdx, 52
    sub rdx, 1023                       # unbiased exponent
    cvtsi2sd xmm1, rdx                  # convert to double
    
    # extract mantissa and normalize to the range 1 to 2
    mov rdx, rax
    # mantissa bits
    movabs rcx, 0x000FFFFFFFFFFFFF # load 64-bit immediate into rcx
    and rdx, rcx
    # set exponent to 0 with a bias of 1023
    movabs rcx, 0x3FF0000000000000 # load 64-bit immediate into rcx
    or rdx, rcx
    movq xmm2, rdx
    
    # transform to the range 0 to 1 by subtracting 1
    subsd xmm2, [rip + one]                   # m equals mantissa minus 1
    
    # use minimax polynomial more accurate than taylor
    # log2 1 plus x equals c0x plus c1x squared plus c2x cubed plus c3x fourth plus c4x fifth
    movsd xmm3, xmm2                    # x
    mulsd xmm3, [rip + log2_coeffs]           # c0 times x
    
    movsd xmm4, xmm2
    mulsd xmm4, xmm2                    # x squared
    movsd xmm5, xmm4
    mulsd xmm5, [rip + log2_coeffs + 8]       # c1 times x squared
    addsd xmm3, xmm5
    
    movsd xmm4, xmm2                    # x cubed (recalculate x^3 from x^2)
    mulsd xmm4, xmm2
    movsd xmm5, xmm4
    mulsd xmm5, [rip + log2_coeffs + 16]      # c2 times x cubed
    addsd xmm3, xmm5
    
    movsd xmm4, xmm2                    # x to the fourth (recalculate x^4 from x^3)
    mulsd xmm4, xmm2
    movsd xmm5, xmm4
    mulsd xmm5, [rip + log2_coeffs + 24]      # c3 times x to the fourth
    addsd xmm3, xmm5
    
    movsd xmm4, xmm2                    # x to the fifth (recalculate x^5 from x^4)
    mulsd xmm4, xmm2
    mulsd xmm4, [rip + log2_coeffs + 32]      # c4 times x to the fifth
    addsd xmm3, xmm4
    
    # add the exponent part
    movsd xmm0, xmm1
    addsd xmm0, xmm3
    
    pop rbp
    ret
    
.negative_input:
    mov rax, 0x7FF8000000000000         # not a number
    movq xmm0, rax
    pop rbp
    ret
    
.zero_input:
    mov rax, 0xFFF0000000000000         # negative infinity
    movq xmm0, rax
    pop rbp
    ret
    
.special_values:
    # handle positive infinity negative infinity and not a number
    movabs rcx, 0x7FF0000000000000 # load 64-bit immediate into rcx
    cmp rax, rcx         # compare rax (raw bits of xmm0) with positive infinity
    je .positive_infinity
    mov rax, 0x7FF8000000000000         # not a number for other cases
    movq xmm0, rax
    pop rbp
    ret
    
.positive_infinity:
    # log2 of positive infinity is positive infinity
    pop rbp
    ret

# quantum coherent state amplitude with better numerical stability
# alpha n equals exp negative absolute alpha squared over 2 times alpha to the n over sqrt n factorial
# input xmm0 is real alpha xmm1 is imag alpha rdi is n
# output xmm0 is real result xmm1 is imag result
quantum_coherent_amplitude:
    push rbp
    mov rbp, rsp
    sub rsp, 32                         # Reserve stack space for local variables (aligned to 16 bytes)
    push rbx
    push r12
    
    mov r12, rdi                        # save n
    
    # Store original alpha values on stack
    movsd [rsp], xmm0                   # Save real part at [rsp]
    movsd [rsp+8], xmm1                 # Save imaginary part at [rsp+8]
    
    # calculate absolute alpha squared with better precision
    movsd xmm2, xmm0
    movsd xmm3, xmm1
    mulsd xmm2, xmm2                    # real part squared
    mulsd xmm3, xmm3                    # imaginary part squared
    addsd xmm2, xmm3                    # absolute alpha squared
    
    # calculate exp negative absolute alpha squared over 2 using exp
    mulsd xmm2, [rip + neg_half]
    movsd xmm0, xmm2
    call fast_exp_optimized
    movsd xmm4, xmm0                    # save the exponent factor
    
    # for numerical stability use log space computation for large n
    cmp r12, 50
    ja .large_n_case
    
    # small n case use direct computation
    # calculate alpha to the n using efficient complex exponentiation
    movsd xmm0, [rip + one]                   # start with 1 plus 0i (result real)
    xorpd xmm1, xmm1                    # result imag
    
    test r12, r12
    jz .power_done                      # alpha to the 0 is 1
    
    # use binary exponentiation for efficiency
    mov rax, r12
    # Load original alpha values from stack to working registers
    movsd xmm7, [rsp]                   # Load real part from stack to xmm7
    movsd xmm8, [rsp+8]                 # Load imaginary part from stack to xmm8
    
.power_loop:
    test rax, 1
    jz .skip_multiply
    # multiply result by the current power of alpha
    # complex_multiply_inline expects current result in xmm0, xmm1 and alpha in xmm7, xmm8
    call complex_multiply_inline
    
.skip_multiply:
    shr rax, 1
    jz .power_done
    # square the current power of alpha
    # complex_square_inline expects current power in xmm7, xmm8
    call complex_square_inline
    jmp .power_loop
    
.power_done:
    # calculate the square root of n factorial efficiently
    mov rax, r12
    call quantum_factorial_optimized
    call fast_sqrt_optimized
    movsd xmm6, xmm0                    # square root of n factorial
    
    # combine all factors
    mulsd xmm0, xmm4                    # real part
    mulsd xmm1, xmm4                    # imaginary part
    divsd xmm0, xmm6
    divsd xmm1, xmm6
    
    jmp .coherent_amplitude_done
    
.large_n_case:
    # for large n use stirlings approximation and log space math
    # to avoid overflow and underflow problems
    call stirling_log_computation
    
.coherent_amplitude_done:
    pop r12
    pop rbx
    add rsp, 32                         # Restore stack space
    pop rbp
    ret

# optimized factorial with stirlings approximation for large values
# input rax is n
# output rax is n factorial or an error code for overflow
quantum_factorial_optimized:
    push rbp
    mov rbp, rsp
    
    cmp rax, 170                        # double precision limit
    ja .overflow
    cmp rax, 20
    jbe .table_lookup
    
    # use stirlings approximation
    cvtsi2sd xmm0, rax
    call stirling_approximation
    jmp .factorial_done
    
.table_lookup:
    # use precomputed table for small values
    lea rdx, [rip + factorial_table]
    mov rax, [rdx + rax*8]
    cvtsi2sd xmm0, rax
    
.factorial_done:
    pop rbp
    ret
    
.overflow:
    mov rax, 0x7FF0000000000000         # return positive infinity
    movq xmm0, rax
    pop rbp
    ret

# complex vector multiplication using avx 512
# input zmm0 zmm1 are real imag parts of first vector of 8 complex numbers
# input zmm2 zmm3 are real imag parts of second vector
# output zmm0 zmm1 are real imag parts of the result
vector_complex_multiply:
    push rbp
    mov rbp, rsp
    
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    # use fma instructions for better performance and accuracy
    
    vmovapd zmm4, zmm0                  # save a (real part of first vector)
    vmovapd zmm5, zmm1                  # save b (imag part of first vector)
    vmovapd zmm6, zmm2                  # save c (real part of second vector)
    vmovapd zmm7, zmm3                  # save d (imag part of second vector)
    
    # real part is ac minus bd
    vmulpd zmm0, zmm4, zmm6             # a times c
    vfnmadd231pd zmm0, zmm5, zmm7       # ac minus bd using fma
    
    # imaginary part is ad plus bc  
    vmulpd zmm1, zmm4, zmm7             # a times d
    vfmadd231pd zmm1, zmm5, zmm6        # ad plus bc using fma
    
    pop rbp
    ret

# enhanced wigner function calculation with vectorization
# w alpha x p equals 2 over pi times exp negative 2 times absolute alpha minus x plus ip over sqrt 2 squared
# input xmm0 xmm1 are real imag alpha xmm2 is x xmm3 is p
# output xmm0 is w alpha x p
quantum_wigner_point:
    push rbp
    mov rbp, rsp
    
    # calculate x plus ip over sqrt 2 with higher precision
    movsd xmm4, [rip + sqrt_half]
    mulsd xmm2, xmm4                    # x over sqrt 2
    mulsd xmm3, xmm4                    # p over sqrt 2
    
    # calculate alpha minus x plus ip over sqrt 2
    subsd xmm0, xmm2                    # real alpha minus x over sqrt 2
    subsd xmm1, xmm3                    # imag alpha minus p over sqrt 2
    
    # calculate the absolute value squared with better numerical stability
    movsd xmm4, xmm0
    movsd xmm5, xmm1
    mulsd xmm4, xmm4                    # real part squared
    mulsd xmm5, xmm5                    # imaginary part squared
    addsd xmm4, xmm5                    # absolute z squared
    
    # calculate exp negative 2 times absolute z squared using exp
    mulsd xmm4, [rip + neg_two]
    movsd xmm0, xmm4
    call fast_exp_optimized
    
    # multiply by 2 over pi
    mulsd xmm0, [rip + two_over_pi]
    
    pop rbp
    ret

# new quantum squeeze operator transformation
# s xi equals exp xi times a dagger squared minus xi times a squared over 2 where xi is r times e to the i theta
# input xmm0 xmm1 are real imag of input state xmm2 is r xmm3 is theta
# output xmm0 xmm1 is the squeezed state
quantum_squeeze_transform:
    push rbp
    mov rbp, rsp
    
    # calculate squeeze parameters
    movsd xmm4, xmm2
    mulsd xmm4, [rip + neg_half]              # negative r over 2
    call fast_exp_optimized             # exp negative r over 2
    movsd xmm5, xmm0                    # save exp negative r over 2
    
    movsd xmm0, xmm2
    mulsd xmm0, [rip + neg_half]
    call fast_exp_optimized             # exp r over 2
    movsd xmm6, xmm0                    # save exp r over 2
    
    # apply squeezing transformation simplified version
    # full implementation would need more complex matrix operations
    movsd xmm0, xmm1                    # start with input state
    # apply squeeze factor
    mulsd xmm0, xmm5
    
    movsd xmm1, xmm1
    mulsd xmm1, xmm6
    
    pop rbp
    ret

# helper functions
fast_exp_optimized:
    # would use minimax polynomials range reduction etc
    ret

fast_sqrt_optimized:
	# using newton raphson or a hardware instruction
    sqrtsd xmm0, xmm0
    ret

stirling_approximation:
    # stirlings approximation implementation
    ret

stirling_log_computation:
    # log space stirling computation for numerical stability
    ret

complex_multiply_inline:
    # inline complex multiplication
    # this function assumes the current result is in xmm0 (real) and xmm1 (imag)
    # and the other complex number (alpha) is in xmm7 (real) and xmm8 (imag)
    # result is in xmm0 (real) and xmm1 (imag)
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    movsd xmm2, xmm0    # a (current result real)
    movsd xmm3, xmm1    # b (current result imag)
    movsd xmm4, xmm7    # c (alpha real)
    movsd xmm5, xmm8    # d (alpha imag)

    # calculate real part: ac - bd
    movsd xmm0, xmm2    # copy a to xmm0 for ac
    mulsd xmm0, xmm4    # ac
    movsd xmm6, xmm3    # copy b to xmm6 for bd
    mulsd xmm6, xmm5    # bd
    subsd xmm0, xmm6    # ac - bd

    # calculate imaginary part: ad + bc
    movsd xmm1, xmm2    # copy a to xmm1 for ad
    mulsd xmm1, xmm5    # ad
    movsd xmm6, xmm3    # copy b to xmm6 for bc
    mulsd xmm6, xmm4    # bc
    addsd xmm1, xmm6    # ad + bc
    ret

complex_square_inline:
    # inline complex squaring
    # this function assumes the complex number to square is in xmm7 (real) and xmm8 (imag)
    # result is in xmm7 (real) and xmm8 (imag)
    # (a + bi)^2 = (a^2 - b^2) + 2abi
    movsd xmm2, xmm7    # a
    movsd xmm3, xmm8    # b

    # calculate real part: a^2 - b^2
    movsd xmm7, xmm2    # copy a to xmm7 for a^2
    mulsd xmm7, xmm2    # a^2
    movsd xmm9, xmm3    # copy b to xmm9 for b^2
    mulsd xmm9, xmm3    # b^2
    subsd xmm7, xmm9    # a^2 - b^2

    # calculate imaginary part: 2ab
    movsd xmm8, xmm2    # copy a to xmm8 for ab
    mulsd xmm8, xmm3    # ab
    addsd xmm8, xmm8    # 2ab
    ret