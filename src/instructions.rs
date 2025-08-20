// this file defines the instruction set architecture (ISA) for the quantum optical assembly (QOA)
// it includes the instruction enum, parsing logic, and byte serialization.

// defines the different types of instructions available in qoa.
// each variant represents a specific operation the qoa vm can perform.

#[allow(dead_code)]
// there isnt really any dead code in here, im just letting the compiler know the shortened
// aliases are intentional because they are not mapped like the full instructions, but they still do the same thing.
// rust sometimes wants strict conditions and unless you tell it explicilty it marks everything not used as warn or error.
#[derive(Debug, PartialEq, Clone)]
pub enum Instruction {
    // core quantum operations
    QINIT(u8), // initialize a qubit
    QI(u8),    // short name for QINIT

    QMEAS(u8), // measure a qubit
    QM(u8),    // short name for QMEAS

    H(u8),             // shorter name for applyhadamard
    HAD(u8),           // short name for applyhadamard
    APPLYHADAMARD(u8), // explicit applyhadamard instruction

    CONTROLLEDNOT(u8, u8), // apply controlled-not gate
    CNOT(u8, u8),          // short name for controllednot
    CN(u8, u8),            // shorter name for controllednot

    APPLYPHASEFLIP(u8), // apply z gate (phase flip)
    Z(u8),              // short name for applyphaseflip

    APPLYBITFLIP(u8), // apply x gate (bit flip)
    X(u8),            // short name for applybitflip

    APPLYTGATE(u8), // apply t gate
    T(u8),          // short name for applytgate

    APPLYSGATE(u8), // apply s gate
    S(u8),          // short name for applysgate

    PHASESHIFT(u8, f64), // apply a phase shift
    P(u8, f64),          // short name for phaseshift

    WAIT(u64), // wait for a specified number of cycles

    RESET(u8), // reset a qubit to |0>
    RST(u8),   // short name for reset
    RSTQ(u8),  // alt short name for reset

    SWAP(u8, u8), // swap two qubits

    CONTROLLEDSWAP(u8, u8, u8), // apply controlled-swap gate
    CS(u8, u8, u8),             // shorter name for CONTROLLEDSWAP
    CSWAP(u8, u8, u8),          // short name for controlledswap

    ENTANGLE(u8, u8),     // entangle two qubits
    ENTANGLEBELL(u8, u8), // create a bell state
    EBELL(u8, u8),        // short name for entanglebell
    EB(u8, u8),           // shorter name for enanglebell

    ENTANGLEMULTI(Vec<u8>), // entangle multiple qubits
    EMULTI(Vec<u8>),        // short name for entanglemulti
    EM(Vec<u8>),            // shorter name for enanglemulti

    ENTANGLECLUSTER(Vec<u8>), // create a cluster state
    ECLUSTER(Vec<u8>),        // short name for entanglecluster
    ECR(Vec<u8>),             // shorter name for entanglecluster

    ENTANGLESWAP(u8, u8, u8, u8), // perform an entanglement swap
    ESWAP(u8, u8, u8, u8),        // short name for entangleswap
    ESP(u8, u8, u8, u8),          // shorter name for entangleswap

    ENTANGLESWAPMEASURE(u8, u8, u8, u8, String), // entanglement swap with measurement
    ESWAPM(u8, u8, u8, u8, String),              // short name for entangleswapmeasure
    ESM(u8, u8, u8, u8, String),                 // shorter name for entangleswapmeasure

    ENTANGLEWITHCLASSICALFEEDBACK(u8, u8, String), // entangle with classical feedback
    EWCFB(u8, u8, String), // alt short name for entanglewithclassicalfeedback
    ECFB(u8, u8, String),  // short name for entanglewithclassicalfeedback

    ENTANGLEDISTRIBUTED(u8, String), // distributed entanglement
    EDIST(u8, String),               // short name for entangledistributed
    ED(u8, String),                  // shorter name for entangledistributed

    MEASUREINBASIS(u8, String), // measure in a specific basis
    MEASB(u8, String),          // short name for measureinbasis
    MIB(u8, String),            // shorter name for measureinbasis

    SYNC, // synchronize quantum operations

    RESETALL, // reset all qubits
    RSTALL,   // short name for resetall
    RSA,      // shorter name for resetall

    VERBOSELOG(u8, String), // log verbose messages
    VLOG(u8, String),       // short name for verboselog
    VL(u8, String),         // shorter name for verboselog

    SETPHASE(u8, f64), // set the phase of a qubit
    SETP(u8, f64),     // short name for setphase
    SPH(u8, f64),      // shorter name for setphase - new alias

    APPLYGATE(String, u8), // apply a named gate
    AGATE(String, u8),     // short name for applygate
    AG(String, u8),        // short name for applygate

    MEASURE(u8), // measure a qubit (alias for qmeas)
    MEAS(u8),    // short name for measure (alias for qmeas)
    M(u8),       // shorter name for measure (alias for qmeas)

    INITQUBIT(u8), // initialize a qubit (alias for qinit)
    IQ(u8),        // short name for initqubit (alias for qinit)
    QINITQ(u8),    // short name for initqubit (alias for qinit)

    LABEL(String), // represents a named label for jump targets (nop-instruction)

    // character printing operations
    CHARLOAD(u8, u8), // load a character into a register
    CLOAD(u8, u8),    // short name for charload
    CLD(u8, u8),      // shorter name for charload

    CHAROUT(u8), // output a character from a register
    COUT(u8),    // short name for charout
    CO(u8),      // short name for charout

    // ionq specific ISA gates
    RX(u8, f64), // apply x rotation gate

    RY(u8, f64), // apply y rotation gate

    RZ(u8, f64), // apply z rotation gate

    PHASE(u8, f64), // apply a phase gate (duplicate of phaseshift, kept for compatibility)
    PSE(u8, f64),   // short name for PHASE

    CZ(u8, u8), // apply controlled-z gate

    QRESET(u8), // reset a qubit (duplicate of reset, kept for compatibility)
    QR(u8),     // short name for QRESET

    THERMALAVG(u8, u8), // thermal average operation
    TAVG(u8, u8),       // short name for thermalavg
    TA(u8, u8),         // shorter name for thermalavg

    WKBFACTOR(u8, u8, u8), // wkb factor calculation
    WKBF(u8, u8, u8),      // short name for wkbfactor
    WF(u8, u8, u8),        // shorter name for wkbfactor

    // register set operation
    REGSET(u8, f64), // set a register to a floating-point value
    RSET(u8, f64),   // short name for regset
    RGST(u8, f64),   // alt short name for regset

    // loop control operations
    LOOPSTART(u8), // start a loop controlled by a register
    LSTART(u8),    // short name for loopstart
    LS(u8),        // shorter name for loopstart

    LOOPEND, // end a loop
    LEND,    // short name for loopend
    LE,      // short name for loopend

    // rotation operations
    APPLYROTATION(u8, char, f64), // apply a rotation around a specified axis
    ROT(u8, char, f64),           // short name for applyrotation
    AR(u8, char, f64),            // shorter name for applyrotation

    APPLYMULTIQUBITROTATION(Vec<u8>, char, Vec<f64>), // apply rotation to multiple qubits
    MROT(Vec<u8>, char, Vec<f64>),                    // short name for applymultiqubitrotation
    AMQR(Vec<u8>, char, Vec<f64>),                    // alt short name for applymultiqubitrotation

    CONTROLLEDPHASEROTATION(u8, u8, f64), // apply controlled phase rotation
    CPHASE(u8, u8, f64),                  // short name for controlledphaserotation
    CPR(u8, u8, f64),                     // shorter name for controlledphaserotation
    APPLYCPHASE(u8, u8, f64), // duplicate of controlledphaserotation, kept for compatibility

    APPLYKERRNONLINEARITY(u8, f64, u64), // apply kerr nonlinearity
    AKNL(u8, f64, u64),                  // short name for applykerrnonlinearity

    APPLYFEEDFORWARDGATE(u8, String), // apply feedforward gate
    AFFG(u8, String),                 // short name for applyfeedforwardgate

    DECOHERENCEPROTECT(u8, u64), // protect against decoherence
    DPROT(u8, u64),              // short name for decoherenceprotect
    DP(u8, u64),                 // shorter name for decoherenceprotect

    APPLYMEASUREMENTBASISCHANGE(u8, String), // change measurement basis
    AMBC(u8, String),                        // short name for applymeasurementbasischange

    // memory and classical operations
    LOAD(u8, String), // load value from classical variable into qubit register
    LD(u8, String),   // short name for load

    STORE(u8, String), // store qubit measurement result into classical variable
    STR(u8, String),   // short name for store

    LOADMEM(String, String), // load value from memory into classical register
    LMEM(String, String),    // short name for loadmem
    LM(String, String),      // shorter name for loadmem

    STOREMEM(String, String), // store value from classical register into memory
    SMEM(String, String),     // short name for storemem
    SM(String, String),       // shorter name for storemem

    LOADCLASSICAL(String, String), // load value from classical variable into classical register
    LCL(String, String),           // short name for loadclassical
    LC(String, String),            // shorter name for loadclassical

    STORECLASSICAL(String, String), // store value from classical register into classical variable
    SCL(String, String),            // short name for storeclassical
    SC(String, String),             // shorter name for storeclassical

    ADD(String, String, String), // add two classical registers

    SUB(String, String, String), // subtract two classical registers

    AND(String, String, String), // bitwise and for classical registers

    OR(String, String, String), // bitwise or for classical registers

    XOR(String, String, String), // bitwise xor for classical registers

    NOT(String), // bitwise not for a classical register

    PUSH(String), // push classical register value onto stack

    POP(String), // pop value from stack into classical register

    // classical control flow
    JUMP(String), // jump to a label
    JMP(i64),     // jump relative by offset (i64 for signed offset) - renamed from JUMP for clarity

    JUMPIFZERO(String, String), // conditional jump if classical register is zero
    JIZ(String, String),        // short name for jumpifzero

    JUMPIFONE(String, String), // conditional jump if classical register is one
    JIO(String, String),       // short name for jumpifone

    CALL(String),      // call a subroutine at a label
    CallLabel(String), // alias for CALL to avoid conflict with CHARLOAD's CL

    BARRIER, // synchronization barrier
    BR,      // short for barrier

    RETURN, // return from a subroutine
    RET,    // short for return

    TIMEDELAY(u8, u64), // introduce a time delay
    TDELAY(u8, u64),    // short name for timedelay
    TD(u8, u64),        // shorter name for timedelay

    RAND(u8), // generate a random number into a register
    RN(u8),   // short for rand

    SQRT(u8, u8), // square root operation
    SR(u8, u8),   // short for SQRT

    EXP(u8, u8), // exponential operation
    LOG(u8, u8), // logarithm operation

    // arithmetic operations on registers
    REGADD(u8, u8, u8), // add two registers (dest, op1, op2)
    RADD(u8, u8, u8),   // short name for regadd
    RGA(u8, u8, u8),    // shorter name for regadd

    REGSUB(u8, u8, u8), // subtract two registers (dest, op1, op2)
    RSUB(u8, u8, u8),   // short name for regsub
    RGS(u8, u8, u8),    // shorter name for regsub

    REGMUL(u8, u8, u8), // multiply two registers (dest, op1, op2)
    RMUL(u8, u8, u8),   // short name for regmul
    RM(u8, u8, u8),     // shorter name for regmul

    REGDIV(u8, u8, u8), // divide two registers (dest, op1, op2)
    RDIV(u8, u8, u8),   // short name for regdiv
    RD(u8, u8, u8),     // shorter name for regdiv

    REGCOPY(u8, u8), // copy value from one register to another (dest, src)
    RCOPY(u8, u8),   // short name for regcopy
    RC(u8, u8),      // shorter name for regcopy

    // optics and photonics operations
    PHOTONEMIT(u8), // emit a photon
    PEMIT(u8),      // short name for photonemit
    PE(u8),         // shorter name for photonemit

    PHOTONDETECT(u8), // detect a photon
    PDETECT(u8),      // short name for photondetect
    PD(u8),           // shorter name for photondetect

    PHOTONCOUNT(u8, String), // count photons
    PCOUNT(u8, String),      // short name for photoncount
    PC(u8, String),          // shorter name for photoncount

    PHOTONADDITION(u8), // perform photon addition
    PADD(u8),           // short name for photonaddition
    PA(u8),             // shorter name for photonaddition

    APPLYPHOTONSUBTRACTION(u8), // perform photon subtraction
    APSUB(u8),                  // short name for applyphotonsubtraction
    APS(u8),                    // shorter name for applyphotonsubtraction

    PHOTONEMISSIONPATTERN(u8, String, u64), // set photon emission pattern
    PEPAT(u8, String, u64),                 // short name for photonemissionpattern
    PEP(u8, String, u64),                   // shorter name for photonemissionpattern

    PHOTONDETECTWITHTHRESHOLD(u8, u64, String), // photon detection with threshold
    PDTHR(u8, u64, String),                     // short name for photondetectwiththreshold
    PDWT(u8, u64, String),                      // shorter name for photondetectwiththreshold

    PHOTONDETECTCOINCIDENCE(Vec<u8>, String), // detect photon coincidence
    PDCOIN(Vec<u8>, String),                  // short name for photondetectcoincidence
    PDC(Vec<u8>, String),                     // shorter name for photondetectcoincidence

    SINGLEPHOTONSOURCEON(u8), // turn on single photon source
    SPSON(u8),                // short name for singlephotonsourceon

    SINGLEPHOTONSOURCEOFF(u8), // turn off single photon source
    SPSOFF(u8),                // short name for singlephotonsourceoff

    PHOTONBUNCHINGCONTROL(u8, bool), // control photon bunching
    PBUNCH(u8, bool),                // short name for photonbunchingcontrol
    PBC(u8, bool),                   // shorter name for photonbunchingcontrol

    PHOTONROUTE(u8, String, String), // route a photon
    PROUTE(u8, String, String),      // short name for photonroute
    PR(u8, String, String),          // shorter name for photonroute

    OPTICALROUTING(u8, u8), // optical routing between qubits
    OROUTE(u8, u8),         // short name for opticalrouting
    OPTR(u8, u8),           // shorter name for opticalrouting

    SETOPTICALATTENUATION(u8, f64), // set optical attenuation
    SOATT(u8, f64),                 // short name for setopticalattenuation
    SOA(u8, f64),                   // shorter name for setopticalattenuation

    DYNAMICPHASECOMPENSATION(u8, f64), // dynamic phase compensation
    DPC(u8, f64),                      // short name for dynamicphasecompensation
    DPCMP(u8, f64),                    // shorter name for dynamicphasecompensation

    OPTICALDELAYLINECONTROL(u8, u64), // control optical delay line
    ODLC(u8, u64),                    // short name for opticaldelaylinecontrol
    ODL(u8, u64),                     // shorter name for opticaldelaylinecontrol

    CROSSPHASEMODULATION(u8, u8, f64), // cross-phase modulation
    CPM(u8, u8, f64),                  // short name for crossphasemodulation
    CPMOD(u8, u8, f64),                // shorter name for crossphasemodulation

    APPLYDISPLACEMENT(u8, f64), // apply displacement operator
    ADISP(u8, f64),             // short name for applydisplacement
    AD(u8, f64),                // shorter name for applydisplacement

    APPLYDISPLACEMENTFEEDBACK(u8, String), // apply displacement with feedback
    ADWF(u8, String),                      // alt short name for applydisplacementfeedback
    ADF(u8, String),                       // short name for applydisplacementfeedback
    ADFB(u8, String),                      // shorter name for applydisplacementfeedback

    APPLYDISPLACEMENTOPERATOR(u8, f64, u64), // apply displacement operator with duration
    ADO(u8, f64, u64),                       // short name for applydisplacementoperator
    ADOP(u8, f64, u64),                      // shorter name for applydisplacementoperator

    APPLYSQUEEZING(u8, f64), // apply squeezing operator
    ASQ(u8, f64),            // short name for applysqueezing
    AS(u8, f64),             // shorter name for applysqueezing

    APPLYSQUEEZINGFEEDBACK(u8, String), // apply squeezing with feedback
    ASWF(u8, String),                   // alt short name for applysqueezingfeedback
    ASF(u8, String),                    // short name for applysqueezingfeedback
    ASFB(u8, String),                   // shorter name for applysqueezingfeedback

    MEASUREPARITY(u8), // measure parity of a qubit
    MPAR(u8),          // short name for measureparity
    MP(u8),            // alt short name for measureparity

    MEASUREWITHDELAY(u8, u64, String), // measure with a delay
    MWD(u8, u64, String),              // short name for measurewithdelay
    MWDEL(u8, u64, String),            // shorter name for measurewithdelay

    OPTICALSWITCHCONTROL(u8, bool), // control optical switch
    OSC(u8, bool),                  // short name for opticalswitchcontrol
    OSW(u8, bool),                  // shorter name for opticalswitchcontrol

    PHOTONLOSSSIMULATE(u8, f64, u64), // simulate photon loss
    PLS(u8, f64, u64),                // short name for photonlosssimulate
    PLSIM(u8, f64, u64),              // shorter name for photonlosssimulate

    PHOTONLOSSCORRECTION(u8, String), // apply photon loss correction
    PLC(u8, String),                  // short name for photonlosscorrection
    PLCOR(u8, String),                // shorter name for photonlosscorrection

    // qubit measurement and error correction
    APPLYQNDMEASUREMENT(u8, String), // apply quantum non-demolition measurement
    AQND(u8, String),                // short name for applyqndmeasurement
    AQAD(u8, String),                // alt short name for applyqndmeasurement
    AQNM(u8, String),                // shorter name for applyqndmeasurement

    ERRORCORRECT(u8, String), // perform error correction
    ECORR(u8, String),        // short name for errorcorrect
    EC(u8, String),           // shorter name for errorcorrect - kept for this instruction

    ERRORSYNDROME(u8, String, String), // get error syndrome
    ESYN(u8, String, String),          // short name for errorsyndrome
    ES(u8, String, String),            // shorter name for errorsyndrome - kept for this instruction

    QUANTUMSTATETOMOGRAPHY(u8, String), // perform quantum state tomography
    QST(u8, String),                    // short name for quantumstatetomography
    QSTAT(u8, String),                  // shorter name for quantumstatetomography

    BELLSTATEVERIFICATION(u8, u8, String), // verify bell state
    BSV(u8, u8, String),                   // short name for bellstateverification
    BSTATE(u8, u8, String),                // shorter name for bellstateverification

    QUANTUMZENOEFFECT(u8, u64, u64), // apply quantum zeno effect
    QZE(u8, u64, u64),               // short name for quantumzenoeffect
    QZEN(u8, u64, u64),              // shorter name for quantumzenoeffect

    APPLYNONLINEARPHASESHIFT(u8, f64), // apply nonlinear phase shift
    ANLPS(u8, f64),                    // short name for applynonlinearphaseshift
    ANLP(u8, f64),                     // alt short name for applynonlinearphaseshift
    ANLPH(u8, f64),                    // shorter name for applynonlinearphaseshift

    APPLYNONLINEARSIGMA(u8, f64), // apply nonlinear sigma operation
    ANLS(u8, f64),                // short name for applynonlinearsigma
    ANLSI(u8, f64),               // shorter name for applynonlinearsigma

    APPLYLINEAROPTICALTRANSFORM(String, Vec<u8>, Vec<u8>, usize), // apply linear optical transform
    ALOT(String, Vec<u8>, Vec<u8>, usize), // short name for applylinearopticaltransform
    ALOPT(String, Vec<u8>, Vec<u8>, usize), // shorter name for applylinearopticaltransform

    PHOTONNUMBERRESOLVINGDETECTION(u8, String), // photon number resolving detection
    PNRD(u8, String),                           // short name for photonnumberresolvingdetection
    PNR(u8, String),                            // shorter name for photonnumberresolvingdetection

    FEEDBACKCONTROL(u8, String), // apply feedback control
    FBC(u8, String),             // short name for feedbackcontrol
    FBCONT(u8, String),          // shorter name for feedbackcontrol

    // miscellaneous operations
    SETPOS(u8, f64, f64), // set position of a qubit
    SPOS(u8, f64, f64),   // short name for setpos
    STP(u8, f64, f64),    // shorter name for setpos

    SETWL(u8, f64), // set wavelength of a qubit
    SWL(u8, f64),   // short name for setwl
    SW(u8, f64),    // shorter name for setwl

    WLSHIFT(u8, f64), // shift wavelength of a qubit
    WLS(u8, f64),     // short name for wlshift
    WLSH(u8, f64),    // shorter name for wlshift

    MOVE(u8, f64, f64), // move a qubit
    MOV(u8, f64, f64),  // short name for move
    MV(u8, f64, f64),   // shorter name for move

    COMMENT(String), // inline comment (nop-instruction)
    CMT(String),     // short name for comment
    CM(String),      // shorter name for comment

    MARKOBSERVED(u8), // mark a qubit as observed
    MOBS(u8),         // short name for markobserved
    MO(u8),           // shorter name for markobserved

    RELEASE(u8), // release a qubit
    REL(u8),     // short name for release
    RL(u8),      // shorter name for release

    HALT, // halt program execution
    HLT,  // short name for halt

    // new instructions for v0.3.0+

    // control flow & program structure
    JMPABS(u64), // jump absolute to instruction index
    JMPA(u64),   // alt short name for jmpabs
    JABS(u64),   // short name for jmpabs

    IFGT(u8, u8, i64), // if reg1 > reg2, jump relative by offset
    IGT(u8, u8, i64),  // short name for ifgt

    IFLT(u8, u8, i64), // if reg1 < reg2, jump relative by offset
    ILT(u8, u8, i64),  // short name for iflt

    IFEQ(u8, u8, i64), // if reg1 == reg2, jump relative by offset
    IEQ(u8, u8, i64),  // short name for ifeq

    IFNE(u8, u8, i64), // if reg1 != reg2, jump relative by offset
    INE(u8, u8, i64),  // short name for ifne

    CALLADDR(u64), // call subroutine at absolute address, push return address to stack
    CADDR(u64),    // short name for calladdr
    CA(u64),       // shorter name for calladdr

    RETSUB, // return from subroutine, pop return address from stack
    RS,     // short name for retsub

    // input/output & debugging
    PRINTF(String, Vec<u8>), // c-style formatted output (format string, register indices)
    PF(String, Vec<u8>),     // short name for printf

    PRINT(String),   // print a string literal
    PRINTLN(String), // print a string literal with newline
    PLN(String),     // short name for println

    INPUT(u8), // read floating-point value from stdin into register
    INP(u8),   // short name for input

    DUMPSTATE, // output quantum amplitudes and phases
    DSTATE,    // short name for dumpstate
    DS,        // shorter name for dumpstate

    DUMPREGS, // output all register values
    DREGS,    // short name for dumpregs
    DR,       // shorter name for dumpregs

    // memory & stack
    LOADREGMEM(u8, u64), // load value from memory address into register (reg, mem_addr)
    LRM(u8, u64),        // short name for loadregmem

    STOREMEMREG(u64, u8), // store value from register into memory address (mem_addr, reg)
    SMR(u64, u8),         // short name for storememreg

    PUSHREG(u8), // push register value onto stack
    PSHRG(u8),   // short name for pushreg
    PRG(u8),     // shorter name for pushreg

    POPREG(u8), // pop value from stack into register
    POPRG(u8),  // short name for popreg
    PPRG(u8),   // shorter name for popreg

    ALLOC(u8, u64), // allocate memory, store start address in register (reg_addr, size)
    ALC(u8, u64),   // short name for alloc

    FREE(u64), // free memory at address
    FRE(u64),  // short name for free

    // comparison & bitwise logic
    CMP(u8, u8), // compare reg1 and reg2, set internal flags (flags handled by runtime)

    ANDBITS(u8, u8, u8), // bitwise and (dest, op1, op2)
    ANDB(u8, u8, u8),    // short name for andbits
    AB(u8, u8, u8),      // shorter name for andbits

    ORBITS(u8, u8, u8), // bitwise or (dest, op1, op2)
    ORB(u8, u8, u8),    // short name for orbits
    OB(u8, u8, u8),     // shorter name for orbits

    XORBITS(u8, u8, u8), // bitwise xor (dest, op1, op2)
    XORB(u8, u8, u8),    // short name for xorbits
    XB(u8, u8, u8),      // shorter name for xorbits

    NOTBITS(u8, u8), // bitwise not (dest, op)
    NOTB(u8, u8),    // short name for notbits
    NB(u8, u8),      // shorter name for notbits

    SHL(u8, u8, u8), // shift left (dest, op, amount_reg)
    SHR(u8, u8, u8), // shift right (dest, op, amount_reg)

    // system & debug utilities
    BREAKPOINT, // breakpoint instruction
    BP,         // short name for breakpoint

    GETTIME(u8), // get system timestamp into register
    GTIME(u8),   // short name for gettime
    GT(u8),      // shorter name for gettime

    SEEDRNG(u64), // seed rng for reproducible results
    SRNG(u64),    // short name for seedrng

    EXITCODE(i32), // terminate program with exit code
    EXC(i32),      // short name for exitcode
    EX(i32),       // shorter name for exitcode
}

// helper functions for parsing
fn parse_u8(s: &str) -> Result<u8, String> {
    s.parse::<u8>().map_err(|_| format!("invalid u8: {}", s))
}

fn parse_f64(s: &str) -> Result<f64, String> {
    s.parse::<f64>().map_err(|_| format!("invalid f64: {}", s))
}

fn parse_u64(s: &str) -> Result<u64, String> {
    s.parse::<u64>().map_err(|_| format!("invalid u64: {}", s))
}

fn parse_i64(s: &str) -> Result<i64, String> {
    s.parse::<i64>().map_err(|_| format!("invalid i64: {}", s))
}

fn parse_string_literal(s: &str) -> Result<String, String> {
    if s.starts_with('"') && s.ends_with('"') {
        Ok(s[1..s.len() - 1].to_string())
    } else {
        Err(format!("invalid string literal: {}", s))
    }
}

fn parse_axis(s: &str) -> Result<char, String> {
    if s.len() == 1 && (s == "x" || s == "y" || s == "z") {
        Ok(s.chars().next().unwrap())
    } else {
        Err(format!("invalid axis: {}", s))
    }
}

fn parse_bool(s: &str) -> Result<bool, String> {
    match s.to_lowercase().as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("invalid boolean: {}", s)),
    }
}

fn parse_reg_list(tokens: &[&str]) -> Result<Vec<u8>, String> {
    tokens.iter().map(|s| parse_u8(s)).collect()
}

// helper function for parsing usize (not originally provided, but needed for APPLYLINEAROPTICALTRANSFORM)
fn parse_usize(s: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|_| format!("invalid usize: {}", s))
}

// helper function for parsing i32 (not originally provided, but needed for EXITCODE)
fn parse_i32(s: &str) -> Result<i32, String> {
    s.parse::<i32>().map_err(|_| format!("invalid i32: {}", s))
}

impl Instruction {
    // encodes the instruction into a byte vector for bytecode representation.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            // core quantum operations
            Instruction::QINIT(q)
            | Instruction::QINITQ(q)
            | Instruction::INITQUBIT(q)
            | Instruction::QI(q)
            | Instruction::IQ(q) => vec![0x04, *q],
            Instruction::QMEAS(q)
            | Instruction::MEASURE(q)
            | Instruction::MEAS(q)
            | Instruction::QM(q)
            | Instruction::M(q) => vec![0x32, *q],
            Instruction::H(q) | Instruction::HAD(q) | Instruction::APPLYHADAMARD(q) => {
                vec![0x05, *q]
            }
            Instruction::CONTROLLEDNOT(c, t) | Instruction::CNOT(c, t) | Instruction::CN(c, t) => {
                vec![0x17, *c, *t]
            }
            Instruction::APPLYPHASEFLIP(q) | Instruction::Z(q) => vec![0x06, *q],
            Instruction::APPLYBITFLIP(q) | Instruction::X(q) => vec![0x07, *q],
            Instruction::APPLYTGATE(q) | Instruction::T(q) => vec![0x0D, *q],
            Instruction::APPLYSGATE(q) | Instruction::S(q) => vec![0x0E, *q],
            Instruction::PHASESHIFT(q, angle) | Instruction::P(q, angle) => {
                let mut v = vec![0x08, *q];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::WAIT(cycles) => {
                let mut v = vec![0x09];
                v.extend(&cycles.to_le_bytes());
                v
            }
            Instruction::RESET(q)
            | Instruction::RST(q)
            | Instruction::QRESET(q)
            | Instruction::QR(q)
            | Instruction::RSTQ(q) => vec![0x0A, *q], // Changed RT to RSTQ
            Instruction::SWAP(q1, q2) => vec![0x0B, *q1, *q2],
            Instruction::CONTROLLEDSWAP(c, t1, t2)
            | Instruction::CSWAP(c, t1, t2)
            | Instruction::CS(c, t1, t2) => vec![0x0C, *c, *t1, *t2],
            Instruction::ENTANGLE(q1, q2) => vec![0x11, *q1, *q2],
            Instruction::ENTANGLEBELL(q1, q2)
            | Instruction::EBELL(q1, q2)
            | Instruction::EB(q1, q2) => vec![0x12, *q1, *q2],
            Instruction::ENTANGLEMULTI(qs) | Instruction::EMULTI(qs) | Instruction::EM(qs) => {
                let mut v = vec![0x13, qs.len() as u8];
                v.extend(qs.iter());
                v
            }
            Instruction::ENTANGLECLUSTER(qs) | Instruction::ECLUSTER(qs) | Instruction::ECR(qs) => {
                let mut v = vec![0x14, qs.len() as u8];
                v.extend(qs.iter());
                v
            }
            Instruction::ENTANGLESWAP(a, b, c, d)
            | Instruction::ESWAP(a, b, c, d)
            | Instruction::ESP(a, b, c, d) => vec![0x15, *a, *b, *c, *d],
            Instruction::ENTANGLESWAPMEASURE(a, b, c, d, label)
            | Instruction::ESWAPM(a, b, c, d, label)
            | Instruction::ESM(a, b, c, d, label) => {
                let mut v = vec![0x16, *a, *b, *c, *d];
                v.extend(label.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::ENTANGLEWITHCLASSICALFEEDBACK(q1, q2, signal)
            | Instruction::ECFB(q1, q2, signal)
            | Instruction::EWCFB(q1, q2, signal) => {
                let mut v = vec![0x19, *q1, *q2];
                v.extend(signal.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::ENTANGLEDISTRIBUTED(q, node)
            | Instruction::EDIST(q, node)
            | Instruction::ED(q, node) => {
                let mut v = vec![0x1A, *q];
                v.extend(node.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::MEASUREINBASIS(q, basis)
            | Instruction::MEASB(q, basis)
            | Instruction::MIB(q, basis) => {
                let mut v = vec![0x1B, *q];
                v.extend(basis.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::SYNC => vec![0x48],
            Instruction::RESETALL | Instruction::RSTALL | Instruction::RSA => vec![0x1C],
            Instruction::VERBOSELOG(q, msg)
            | Instruction::VLOG(q, msg)
            | Instruction::VL(q, msg) => {
                // Changed VL pattern
                let mut v = vec![0x87, *q];
                v.extend(msg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::SETPHASE(q, phase)
            | Instruction::SETP(q, phase)
            | Instruction::SPH(q, phase) => {
                // Changed STP to SPH
                let mut v = vec![0x1D, *q];
                v.extend(&phase.to_le_bytes());
                v
            }
            Instruction::APPLYGATE(name, q)
            | Instruction::AGATE(name, q)
            | Instruction::AG(name, q) => {
                let mut v = vec![0x02, *q];
                v.extend(name.as_bytes());
                // pad with zeros to a fixed length if necessary, or just null-terminate
                v.push(0); // null terminator
                v
            }
            Instruction::LABEL(_) => vec![], // LABEL is a nop-instruction, no bytecode

            // character printing operations
            Instruction::CHARLOAD(reg, val)
            | Instruction::CLOAD(reg, val)
            | Instruction::CLD(reg, val) => vec![0x31, *reg, *val],
            Instruction::CHAROUT(reg) | Instruction::COUT(reg) | Instruction::CO(reg) => {
                vec![0x18, *reg]
            }

            // ionq specific ISA gates
            Instruction::RX(q, angle) => {
                let mut v = vec![0x22, *q];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::RY(q, angle) => {
                let mut v = vec![0x23, *q];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::RZ(q, angle) => {
                let mut v = vec![0x0F, *q];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::PHASE(q, angle) | Instruction::PSE(q, angle) => {
                let mut v = vec![0x24, *q];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::CZ(c, t) => vec![0x1E, *c, *t],
            Instruction::THERMALAVG(q, param)
            | Instruction::TAVG(q, param)
            | Instruction::TA(q, param) => vec![0x1F, *q, *param],
            Instruction::WKBFACTOR(q1, q2, param)
            | Instruction::WKBF(q1, q2, param)
            | Instruction::WF(q1, q2, param) => vec![0x20, *q1, *q2, *param],

            // register set operation
            Instruction::REGSET(reg, val)
            | Instruction::RSET(reg, val)
            | Instruction::RGST(reg, val) => {
                let mut v = vec![0x21, *reg];
                v.extend(&val.to_le_bytes());
                v
            }

            // loop control operations
            Instruction::LOOPSTART(reg) | Instruction::LSTART(reg) | Instruction::LS(reg) => {
                vec![0x01, *reg]
            }
            Instruction::LOOPEND | Instruction::LEND | Instruction::LE => vec![0x10],

            // rotation operations
            Instruction::APPLYROTATION(q, axis, angle)
            | Instruction::ROT(q, axis, angle)
            | Instruction::AR(q, axis, angle) => {
                let mut v = vec![0x33, *q, *axis as u8];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::APPLYMULTIQUBITROTATION(qs, axis, angles)
            | Instruction::MROT(qs, axis, angles)
            | Instruction::AMQR(qs, axis, angles) => {
                let mut v = vec![0x34, *axis as u8, qs.len() as u8];
                v.extend(qs.iter());
                for a in angles {
                    v.extend(&a.to_le_bytes());
                }
                v
            }
            Instruction::CONTROLLEDPHASEROTATION(c, t, angle)
            | Instruction::CPHASE(c, t, angle)
            | Instruction::APPLYCPHASE(c, t, angle)
            | Instruction::CPR(c, t, angle) => {
                let mut v = vec![0x35, *c, *t];
                v.extend(&angle.to_le_bytes());
                v
            }
            Instruction::APPLYKERRNONLINEARITY(q_idx, strength, duration)
            | Instruction::AKNL(q_idx, strength, duration) => {
                let mut v = vec![0x37, *q_idx];
                v.extend(&strength.to_le_bytes());
                v.extend(&duration.to_le_bytes());
                v
            }
            Instruction::APPLYFEEDFORWARDGATE(q, reg) | Instruction::AFFG(q, reg) => {
                let mut v = vec![0x38, *q];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::DECOHERENCEPROTECT(q, duration)
            | Instruction::DPROT(q, duration)
            | Instruction::DP(q, duration) => {
                let mut v = vec![0x39, *q];
                v.extend(&duration.to_le_bytes());
                v
            }
            Instruction::APPLYMEASUREMENTBASISCHANGE(q, basis) | Instruction::AMBC(q, basis) => {
                let mut v = vec![0x3A, *q];
                v.extend(basis.as_bytes());
                v.push(0); // null terminator for string
                v
            }

            // memory and classical operations (originals)
            Instruction::LOAD(q, reg) | Instruction::LD(q, reg) => {
                let mut v = vec![0x3B, *q];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::STORE(q, reg) | Instruction::STR(q, reg) => {
                let mut v = vec![0x3C, *q];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::LOADMEM(reg, addr)
            | Instruction::LMEM(reg, addr)
            | Instruction::LM(reg, addr) => {
                let mut v = vec![0x3D];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v.extend(addr.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::STOREMEM(reg, addr)
            | Instruction::SMEM(reg, addr)
            | Instruction::SM(reg, addr) => {
                let mut v = vec![0x3E];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v.extend(addr.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::LOADCLASSICAL(reg, var)
            | Instruction::LCL(reg, var)
            | Instruction::LC(reg, var) => {
                let mut v = vec![0x3F];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v.extend(var.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::STORECLASSICAL(reg, var)
            | Instruction::SCL(reg, var)
            | Instruction::SC(reg, var) => {
                let mut v = vec![0x40];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v.extend(var.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::ADD(dst, src1, src2) => {
                let mut v = vec![0x41];
                v.extend(dst.as_bytes());
                v.push(0); // null terminator for string
                v.extend(src1.as_bytes());
                v.push(0); // null terminator for string
                v.extend(src2.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::SUB(dst, src1, src2) => {
                let mut v = vec![0x42];
                v.extend(dst.as_bytes());
                v.push(0); // null terminator for string
                v.extend(src1.as_bytes());
                v.push(0); // null terminator for string
                v.extend(src2.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::AND(dst, src1, src2) => {
                let mut v = vec![0x43];
                v.extend(dst.as_bytes());
                v.push(0); // null terminator for string
                v.extend(src1.as_bytes());
                v.push(0); // null terminator for string
                v.extend(src2.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::OR(dst, src1, src2) => {
                let mut v = vec![0x44];
                v.extend(dst.as_bytes());
                v.push(0); // null terminator for string
                v.extend(src1.as_bytes());
                v.push(0); // null terminator for string
                v.extend(src2.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::XOR(dst, src1, src2) => {
                let mut v = vec![0x45];
                v.extend(dst.as_bytes());
                v.push(0); // null terminator for string
                v.extend(src1.as_bytes());
                v.push(0); // null terminator for string
                v.extend(src2.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::NOT(reg) => {
                let mut v = vec![0x46];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::PUSH(reg) => {
                let mut v = vec![0x47];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::POP(reg) => {
                let mut v = vec![0x4F];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }

            // classical control flow (originals)
            Instruction::JUMP(label) => {
                // JUMP is now for labels only
                let mut v = vec![0x49];
                v.extend(label.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::JMP(offset) => {
                // JMP is for i64 offset
                let mut v = vec![0x90];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::JUMPIFZERO(cond, label) | Instruction::JIZ(cond, label) => {
                let mut v = vec![0x4A];
                v.extend(cond.as_bytes());
                v.push(0); // null terminator for string
                v.extend(label.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::JUMPIFONE(cond, label) | Instruction::JIO(cond, label) => {
                let mut v = vec![0x4B];
                v.extend(cond.as_bytes());
                v.push(0); // null terminator for string
                v.extend(label.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::CALL(label) | Instruction::CallLabel(label) => {
                // Use CallLabel for the alias
                let mut v = vec![0x4C];
                v.extend(label.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::RETURN | Instruction::RET => vec![0x4D], // Changed RT to RET
            Instruction::TIMEDELAY(q, cycles)
            | Instruction::TDELAY(q, cycles)
            | Instruction::TD(q, cycles) => {
                let mut v = vec![0x4E, *q];
                v.extend(&cycles.to_le_bytes());
                v
            }
            Instruction::RAND(reg) | Instruction::RN(reg) => vec![0x50, *reg],
            Instruction::SQRT(dst, src) | Instruction::SR(dst, src) => vec![0x51, *dst, *src],
            Instruction::EXP(dst, src) => vec![0x52, *dst, *src],
            Instruction::LOG(dst, src) => vec![0x53, *dst, *src],
            Instruction::REGADD(rd, ra, rb)
            | Instruction::RADD(rd, ra, rb)
            | Instruction::RGA(rd, ra, rb) => vec![0x54, *rd, *ra, *rb],
            Instruction::REGSUB(rd, ra, rb)
            | Instruction::RSUB(rd, ra, rb)
            | Instruction::RGS(rd, ra, rb) => vec![0x55, *rd, *ra, *rb],
            Instruction::REGMUL(rd, ra, rb)
            | Instruction::RMUL(rd, ra, rb)
            | Instruction::RM(rd, ra, rb) => vec![0x56, *rd, *ra, *rb],
            Instruction::REGDIV(rd, ra, rb)
            | Instruction::RDIV(rd, ra, rb)
            | Instruction::RD(rd, ra, rb) => vec![0x57, *rd, *ra, *rb],
            Instruction::REGCOPY(rd, ra) | Instruction::RCOPY(rd, ra) | Instruction::RC(rd, ra) => {
                vec![0x58, *rd, *ra]
            }

            // optics and photonics operations
            Instruction::PHOTONEMIT(q) | Instruction::PEMIT(q) | Instruction::PE(q) => {
                vec![0x59, *q]
            }
            Instruction::PHOTONDETECT(q) | Instruction::PDETECT(q) | Instruction::PD(q) => {
                vec![0x5A, *q]
            }
            Instruction::PHOTONCOUNT(q, reg)
            | Instruction::PCOUNT(q, reg)
            | Instruction::PC(q, reg) => {
                let mut v = vec![0x5B, *q];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::PHOTONADDITION(q) | Instruction::PADD(q) | Instruction::PA(q) => {
                vec![0x5C, *q]
            }
            Instruction::APPLYPHOTONSUBTRACTION(q)
            | Instruction::APSUB(q)
            | Instruction::APS(q) => vec![0x5D, *q],
            Instruction::PHOTONEMISSIONPATTERN(q, reg, cycles)
            | Instruction::PEPAT(q, reg, cycles)
            | Instruction::PEP(q, reg, cycles) => {
                let mut v = vec![0x5E, *q];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v.extend(&cycles.to_le_bytes());
                v
            }
            Instruction::PHOTONDETECTWITHTHRESHOLD(q, thresh, reg)
            | Instruction::PDTHR(q, thresh, reg)
            | Instruction::PDWT(q, thresh, reg) => {
                let mut v = vec![0x5F, *q];
                v.extend(&thresh.to_le_bytes());
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::PHOTONDETECTCOINCIDENCE(qs, reg)
            | Instruction::PDCOIN(qs, reg)
            | Instruction::PDC(qs, reg) => {
                let mut v = vec![0x60, qs.len() as u8];
                v.extend(qs.iter());
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::SINGLEPHOTONSOURCEON(q) | Instruction::SPSON(q) => vec![0x61, *q],
            Instruction::SINGLEPHOTONSOURCEOFF(q) | Instruction::SPSOFF(q) => vec![0x62, *q],
            Instruction::PHOTONBUNCHINGCONTROL(q, b)
            | Instruction::PBUNCH(q, b)
            | Instruction::PBC(q, b) => vec![0x63, *q, *b as u8],
            Instruction::PHOTONROUTE(q, from, to)
            | Instruction::PROUTE(q, from, to)
            | Instruction::PR(q, from, to) => {
                let mut v = vec![0x64, *q];
                v.extend(from.as_bytes());
                v.push(0); // null terminator for string
                v.extend(to.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::OPTICALROUTING(q1, q2)
            | Instruction::OROUTE(q1, q2)
            | Instruction::OPTR(q1, q2) => vec![0x65, *q1, *q2],
            Instruction::SETOPTICALATTENUATION(q, att)
            | Instruction::SOATT(q, att)
            | Instruction::SOA(q, att) => {
                let mut v = vec![0x66, *q];
                v.extend(&att.to_le_bytes());
                v
            }
            Instruction::DYNAMICPHASECOMPENSATION(q, phase)
            | Instruction::DPC(q, phase)
            | Instruction::DPCMP(q, phase) => {
                let mut v = vec![0x67, *q];
                v.extend(&phase.to_le_bytes());
                v
            }
            Instruction::OPTICALDELAYLINECONTROL(q, delay)
            | Instruction::ODLC(q, delay)
            | Instruction::ODL(q, delay) => {
                let mut v = vec![0x68, *q];
                v.extend(&delay.to_le_bytes());
                v
            }
            Instruction::CROSSPHASEMODULATION(c, t, stren)
            | Instruction::CPM(c, t, stren)
            | Instruction::CPMOD(c, t, stren) => {
                let mut v = vec![0x69, *c, *t];
                v.extend(&stren.to_le_bytes());
                v
            }
            Instruction::APPLYDISPLACEMENT(q, a)
            | Instruction::ADISP(q, a)
            | Instruction::AD(q, a) => {
                let mut v = vec![0x6A, *q];
                v.extend(&a.to_le_bytes());
                v
            }
            Instruction::APPLYDISPLACEMENTFEEDBACK(q, reg)
            | Instruction::ADF(q, reg)
            | Instruction::ADWF(q, reg)
            | Instruction::ADFB(q, reg) => {
                let mut v = vec![0x6B, *q];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::APPLYDISPLACEMENTOPERATOR(q, alpha, dur)
            | Instruction::ADO(q, alpha, dur)
            | Instruction::ADOP(q, alpha, dur) => {
                let mut v = vec![0x6C, *q];
                v.extend(&alpha.to_le_bytes());
                v.extend(&dur.to_le_bytes());
                v
            }
            Instruction::APPLYSQUEEZING(q, s) | Instruction::ASQ(q, s) | Instruction::AS(q, s) => {
                let mut v = vec![0x6D, *q];
                v.extend(&s.to_le_bytes());
                v
            }
            Instruction::APPLYSQUEEZINGFEEDBACK(q, reg)
            | Instruction::ASF(q, reg)
            | Instruction::ASWF(q, reg)
            | Instruction::ASFB(q, reg) => {
                let mut v = vec![0x6E, *q];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::MEASUREPARITY(q) | Instruction::MPAR(q) | Instruction::MP(q) => {
                vec![0x6F, *q]
            }
            Instruction::MEASUREWITHDELAY(q, u64_delay, reg)
            | Instruction::MWD(q, u64_delay, reg)
            | Instruction::MWDEL(q, u64_delay, reg) => {
                let mut v = vec![0x70, *q];
                v.extend(&u64_delay.to_le_bytes());
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::OPTICALSWITCHCONTROL(q, b)
            | Instruction::OSC(q, b)
            | Instruction::OSW(q, b) => vec![0x71, *q, *b as u8],
            Instruction::PHOTONLOSSSIMULATE(q, prob, seed)
            | Instruction::PLS(q, prob, seed)
            | Instruction::PLSIM(q, prob, seed) => {
                let mut v = vec![0x72, *q];
                v.extend(&prob.to_le_bytes());
                v.extend(&seed.to_le_bytes());
                v
            }
            Instruction::PHOTONLOSSCORRECTION(q, reg)
            | Instruction::PLC(q, reg)
            | Instruction::PLCOR(q, reg) => {
                let mut v = vec![0x73, *q];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }

            // qubit measurement and error correction
            Instruction::APPLYQNDMEASUREMENT(q, reg)
            | Instruction::AQND(q, reg)
            | Instruction::AQAD(q, reg)
            | Instruction::AQNM(q, reg) => {
                let mut v = vec![0x7C, *q];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::ERRORCORRECT(q, syndrome_type)
            | Instruction::ECORR(q, syndrome_type)
            | Instruction::EC(q, syndrome_type) => {
                let mut v = vec![0x7D, *q];
                v.extend(syndrome_type.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::ERRORSYNDROME(q, syndrome_type, result_reg)
            | Instruction::ESYN(q, syndrome_type, result_reg)
            | Instruction::ES(q, syndrome_type, result_reg) => {
                let mut v = vec![0x7E, *q];
                v.extend(syndrome_type.as_bytes());
                v.push(0); // null terminator for string
                v.extend(result_reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::QUANTUMSTATETOMOGRAPHY(q, basis)
            | Instruction::QST(q, basis)
            | Instruction::QSTAT(q, basis) => {
                let mut v = vec![0x7F, *q];
                v.extend(basis.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::BELLSTATEVERIFICATION(q1, q2, result_reg)
            | Instruction::BSV(q1, q2, result_reg)
            | Instruction::BSTATE(q1, q2, result_reg) => {
                let mut v = vec![0x80, *q1, *q2];
                v.extend(result_reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::QUANTUMZENOEFFECT(q, num_measurements, interval_cycles)
            | Instruction::QZE(q, num_measurements, interval_cycles)
            | Instruction::QZEN(q, num_measurements, interval_cycles) => {
                let mut v = vec![0x81, *q];
                v.extend(&num_measurements.to_le_bytes());
                v.extend(&interval_cycles.to_le_bytes());
                v
            }
            Instruction::APPLYNONLINEARPHASESHIFT(q, strength)
            | Instruction::ANLPS(q, strength)
            | Instruction::ANLP(q, strength)
            | Instruction::ANLPH(q, strength) => {
                let mut v = vec![0x82, *q];
                v.extend(&strength.to_le_bytes());
                v
            }
            Instruction::APPLYNONLINEARSIGMA(q, strength)
            | Instruction::ANLS(q, strength)
            | Instruction::ANLSI(q, strength) => {
                let mut v = vec![0x83, *q];
                v.extend(&strength.to_le_bytes());
                v
            }
            Instruction::APPLYLINEAROPTICALTRANSFORM(_name, input_qs, output_qs, _num_modes)
            | Instruction::ALOT(_name, input_qs, output_qs, _num_modes)
            | Instruction::ALOPT(_name, input_qs, output_qs, _num_modes) => {
                let mut v = vec![
                    0x84,
                    input_qs.len() as u8,
                    output_qs.len() as u8,
                    *_num_modes as u8,
                ];
                v.extend(input_qs.iter());
                v.extend(output_qs.iter());
                v
            }
            Instruction::PHOTONNUMBERRESOLVINGDETECTION(q, reg)
            | Instruction::PNRD(q, reg)
            | Instruction::PNR(q, reg) => {
                let mut v = vec![0x85, *q];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::FEEDBACKCONTROL(q, reg)
            | Instruction::FBC(q, reg)
            | Instruction::FBCONT(q, reg) => {
                let mut v = vec![0x86, *q];
                v.extend(reg.as_bytes());
                v.push(0); // null terminator for string
                v
            }

            // miscellaneous operations
            Instruction::SETPOS(q, x, y)
            | Instruction::SPOS(q, x, y)
            | Instruction::STP(q, x, y) => {
                let mut v = vec![0x74, *q];
                v.extend(&x.to_le_bytes());
                v.extend(&y.to_le_bytes());
                v
            }
            Instruction::SETWL(q, wl) | Instruction::SWL(q, wl) | Instruction::SW(q, wl) => {
                let mut v = vec![0x75, *q];
                v.extend(&wl.to_le_bytes());
                v
            }
            Instruction::WLSHIFT(q, wl_delta)
            | Instruction::WLS(q, wl_delta)
            | Instruction::WLSH(q, wl_delta) => {
                let mut v = vec![0x76, *q];
                v.extend(&wl_delta.to_le_bytes());
                v
            }
            Instruction::MOVE(q, dx, dy)
            | Instruction::MOV(q, dx, dy)
            | Instruction::MV(q, dx, dy) => {
                let mut v = vec![0x77, *q];
                v.extend(&dx.to_le_bytes());
                v.extend(&dy.to_le_bytes());
                v
            }
            Instruction::COMMENT(text) | Instruction::CMT(text) | Instruction::CM(text) => {
                let mut v = vec![0x88];
                v.extend(text.as_bytes());
                v.push(0); // null terminator for string
                v
            }
            Instruction::MARKOBSERVED(q) | Instruction::MOBS(q) | Instruction::MO(q) => {
                vec![0x79, *q]
            }
            Instruction::RELEASE(q) | Instruction::REL(q) | Instruction::RL(q) => vec![0x7A, *q],
            Instruction::HALT | Instruction::HLT => vec![0xFF],

            Instruction::BARRIER | Instruction::BR => vec![0x89],

            // new instructions for v0.3.0+

            // control flow & program structure
            // JMP (relative jump) is now distinct from JUMP (label jump)
            Instruction::JMPABS(addr) | Instruction::JABS(addr) | Instruction::JMPA(addr) => {
                let mut v = vec![0x91]; // explicit opcode for JMPABS
                v.extend(&addr.to_le_bytes());
                v
            }
            Instruction::IFGT(r1, r2, offset) | Instruction::IGT(r1, r2, offset) => {
                let mut v = vec![0x92, *r1, *r2];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::IFLT(r1, r2, offset) | Instruction::ILT(r1, r2, offset) => {
                let mut v = vec![0x93, *r1, *r2];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::IFEQ(r1, r2, offset) | Instruction::IEQ(r1, r2, offset) => {
                let mut v = vec![0x94, *r1, *r2];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::IFNE(r1, r2, offset) | Instruction::INE(r1, r2, offset) => {
                let mut v = vec![0x95, *r1, *r2];
                v.extend(&offset.to_le_bytes());
                v
            }
            Instruction::CALLADDR(addr) | Instruction::CADDR(addr) | Instruction::CA(addr) => {
                let mut v = vec![0x96];
                v.extend(&addr.to_le_bytes());
                v
            }
            Instruction::RETSUB | Instruction::RS => {
                // Consolidated RETSUB and RS
                vec![0x97]
            }
            Instruction::PRINTF(format_str, regs) | Instruction::PF(format_str, regs) => {
                let mut v = vec![0x98];
                v.extend(&(format_str.len() as u64).to_le_bytes()); // length of string
                v.extend(format_str.as_bytes()); // format string
                v.push(regs.len() as u8); // number of registers
                v.extend(regs); // register indices
                v
            }
            Instruction::PRINT(s) => {
                let mut v = vec![0x99];
                v.extend(&(s.len() as u64).to_le_bytes()); // length of string
                v.extend(s.as_bytes()); // the string itself
                v
            }
            Instruction::PRINTLN(s) | Instruction::PLN(s) => {
                let mut v = vec![0x9A];
                v.extend(&(s.len() as u64).to_le_bytes()); // length of string
                v.extend(s.as_bytes()); // the string itself
                v
            }
            Instruction::INPUT(reg) | Instruction::INP(reg) => {
                vec![0x9B, *reg]
            }
            Instruction::DUMPSTATE | Instruction::DSTATE | Instruction::DS => {
                vec![0x9C]
            }
            Instruction::DUMPREGS | Instruction::DREGS | Instruction::DR => {
                vec![0x9D]
            }
            Instruction::LOADREGMEM(reg, addr) | Instruction::LRM(reg, addr) => {
                let mut v = vec![0x9E, *reg];
                v.extend(&addr.to_le_bytes());
                v
            }
            Instruction::STOREMEMREG(addr, reg) | Instruction::SMR(addr, reg) => {
                let mut v = vec![0x9F];
                v.extend(&addr.to_le_bytes());
                v.push(*reg);
                v
            }
            Instruction::PUSHREG(reg) | Instruction::PSHRG(reg) | Instruction::PRG(reg) => {
                vec![0xA0, *reg]
            }
            Instruction::POPREG(reg) | Instruction::POPRG(reg) | Instruction::PPRG(reg) => {
                vec![0xA1, *reg]
            }
            Instruction::ALLOC(reg_addr, size) | Instruction::ALC(reg_addr, size) => {
                let mut v = vec![0xA2, *reg_addr];
                v.extend(&size.to_le_bytes());
                v
            }
            Instruction::FREE(addr) | Instruction::FRE(addr) => {
                let mut v = vec![0xA3];
                v.extend(&addr.to_le_bytes());
                v
            }
            Instruction::CMP(reg1, reg2) => {
                vec![0xA4, *reg1, *reg2]
            }
            Instruction::ANDBITS(d, o1, o2)
            | Instruction::ANDB(d, o1, o2)
            | Instruction::AB(d, o1, o2) => {
                vec![0xA5, *d, *o1, *o2]
            }
            Instruction::ORBITS(d, o1, o2)
            | Instruction::ORB(d, o1, o2)
            | Instruction::OB(d, o1, o2) => {
                vec![0xA6, *d, *o1, *o2]
            }
            Instruction::XORBITS(d, o1, o2)
            | Instruction::XORB(d, o1, o2)
            | Instruction::XB(d, o1, o2) => {
                vec![0xA7, *d, *o1, *o2]
            }
            Instruction::NOTBITS(d, o) | Instruction::NOTB(d, o) | Instruction::NB(d, o) => {
                vec![0xA8, *d, *o]
            }
            Instruction::SHL(d, o1, o2) => {
                vec![0xA9, *d, *o1, *o2]
            }
            Instruction::SHR(d, o1, o2) => {
                vec![0xAA, *d, *o1, *o2]
            }
            Instruction::BREAKPOINT | Instruction::BP => {
                vec![0xAB]
            }
            Instruction::GETTIME(reg) | Instruction::GTIME(reg) | Instruction::GT(reg) => {
                vec![0xAC, *reg]
            }
            Instruction::SEEDRNG(seed) | Instruction::SRNG(seed) => {
                let mut v = vec![0xAD];
                v.extend(&seed.to_le_bytes());
                v
            }
            Instruction::EXITCODE(code) | Instruction::EXC(code) | Instruction::EX(code) => {
                let mut v = vec![0xAE];
                v.extend(&code.to_le_bytes());
                v
            }
        }
    }

    // helper function to get the byte size of an instruction.
    // this method is currently not used in the provided code, but is kept for potential future use.
    #[allow(dead_code)] // added to suppress the unused method warning
    pub fn get_byte_size(&self) -> usize {
        match self {
            // nop-instructions, no bytecode
            Instruction::LABEL(_) => 0,
            Instruction::COMMENT(text) | Instruction::CMT(text) | Instruction::CM(text) => {
                1 + text.len() + 1
            } // opcode + string length + null terminator

            // 1-byte opcodes
            Instruction::SYNC => 1,
            Instruction::LOOPEND | Instruction::LEND | Instruction::LE => 1,
            Instruction::RESETALL | Instruction::RSTALL | Instruction::RSA => 1,
            Instruction::HALT | Instruction::HLT => 1,
            Instruction::RETSUB | Instruction::RS => 1,
            Instruction::BARRIER | Instruction::BR => 1,
            Instruction::DUMPSTATE | Instruction::DSTATE | Instruction::DS => 1,
            Instruction::DUMPREGS | Instruction::DREGS | Instruction::DR => 1,
            Instruction::BREAKPOINT | Instruction::BP => 1,
            Instruction::RETURN | Instruction::RET => 1, // Changed RT to RET

            // 2-byte opcodes (opcode + u8)
            Instruction::QINIT(_)
            | Instruction::QINITQ(_)
            | Instruction::INITQUBIT(_)
            | Instruction::QI(_)
            | Instruction::IQ(_) => 2,
            Instruction::QMEAS(_)
            | Instruction::MEASURE(_)
            | Instruction::MEAS(_)
            | Instruction::QM(_)
            | Instruction::M(_) => 2,
            Instruction::H(_) | Instruction::HAD(_) | Instruction::APPLYHADAMARD(_) => 2,
            Instruction::APPLYPHASEFLIP(_) | Instruction::Z(_) => 2,
            Instruction::APPLYBITFLIP(_) | Instruction::X(_) => 2,
            Instruction::APPLYTGATE(_) | Instruction::T(_) => 2,
            Instruction::APPLYSGATE(_) | Instruction::S(_) => 2,
            Instruction::RESET(_)
            | Instruction::RST(_)
            | Instruction::QRESET(_)
            | Instruction::QR(_)
            | Instruction::RSTQ(_) => 2, // Changed RT to RSTQ
            Instruction::CHAROUT(_) | Instruction::COUT(_) | Instruction::CO(_) => 2,
            Instruction::LOOPSTART(_) | Instruction::LSTART(_) | Instruction::LS(_) => 2,
            Instruction::RAND(_) | Instruction::RN(_) => 2,
            Instruction::PHOTONEMIT(_) | Instruction::PEMIT(_) | Instruction::PE(_) => 2,
            Instruction::PHOTONDETECT(_) | Instruction::PDETECT(_) | Instruction::PD(_) => 2,
            Instruction::PHOTONADDITION(_) | Instruction::PADD(_) | Instruction::PA(_) => 2,
            Instruction::APPLYPHOTONSUBTRACTION(_)
            | Instruction::APSUB(_)
            | Instruction::APS(_) => 2,
            Instruction::SINGLEPHOTONSOURCEON(_) | Instruction::SPSON(_) => 2,
            Instruction::SINGLEPHOTONSOURCEOFF(_) | Instruction::SPSOFF(_) => 2,
            Instruction::MEASUREPARITY(_) | Instruction::MPAR(_) | Instruction::MP(_) => 2,
            Instruction::OPTICALSWITCHCONTROL(_, _)
            | Instruction::OSC(_, _)
            | Instruction::OSW(_, _) => 3, // opcode + u8 + bool (1 byte)
            Instruction::INPUT(_) | Instruction::INP(_) => 2,
            Instruction::PUSHREG(_) | Instruction::PSHRG(_) | Instruction::PRG(_) => 2,
            Instruction::POPREG(_) | Instruction::POPRG(_) | Instruction::PPRG(_) => 2,
            Instruction::GETTIME(_) | Instruction::GTIME(_) | Instruction::GT(_) => 2,
            Instruction::MARKOBSERVED(_) | Instruction::MOBS(_) | Instruction::MO(_) => 2,
            Instruction::RELEASE(_) | Instruction::REL(_) | Instruction::RL(_) => 2,

            // 3-byte opcodes (two-qubit or reg/reg)
            Instruction::CONTROLLEDNOT(_, _) | Instruction::CNOT(_, _) | Instruction::CN(_, _) => 3,
            Instruction::SWAP(_, _) => 3,
            Instruction::ENTANGLE(_, _) => 3,
            Instruction::ENTANGLEBELL(_, _) | Instruction::EBELL(_, _) | Instruction::EB(_, _) => 3,
            Instruction::CZ(_, _) => 3,
            Instruction::THERMALAVG(_, _) | Instruction::TAVG(_, _) | Instruction::TA(_, _) => 3,
            Instruction::SQRT(_, _) | Instruction::SR(_, _) => 3,
            Instruction::EXP(_, _) => 3,
            Instruction::LOG(_, _) => 3,
            Instruction::OPTICALROUTING(_, _)
            | Instruction::OROUTE(_, _)
            | Instruction::OPTR(_, _) => 3,
            Instruction::CMP(_, _) => 3,
            Instruction::REGCOPY(_, _) | Instruction::RCOPY(_, _) | Instruction::RC(_, _) => 3,
            Instruction::NOTBITS(_, _) | Instruction::NOTB(_, _) | Instruction::NB(_, _) => 3,
            Instruction::PHOTONBUNCHINGCONTROL(_, _)
            | Instruction::PBUNCH(_, _)
            | Instruction::PBC(_, _) => 3,
            Instruction::CHARLOAD(_, _) | Instruction::CLOAD(_, _) | Instruction::CLD(_, _) => 3,

            // 4-byte opcodes (three regs)
            Instruction::CONTROLLEDSWAP(_, _, _)
            | Instruction::CSWAP(_, _, _)
            | Instruction::CS(_, _, _) => 4,
            Instruction::WKBFACTOR(_, _, _)
            | Instruction::WKBF(_, _, _)
            | Instruction::WF(_, _, _) => 4,
            Instruction::REGADD(_, _, _)
            | Instruction::RADD(_, _, _)
            | Instruction::RGA(_, _, _) => 4,
            Instruction::REGSUB(_, _, _)
            | Instruction::RSUB(_, _, _)
            | Instruction::RGS(_, _, _) => 4,
            Instruction::REGMUL(_, _, _)
            | Instruction::RMUL(_, _, _)
            | Instruction::RM(_, _, _) => 4,
            Instruction::REGDIV(_, _, _)
            | Instruction::RDIV(_, _, _)
            | Instruction::RD(_, _, _) => 4,
            Instruction::ANDBITS(_, _, _)
            | Instruction::ANDB(_, _, _)
            | Instruction::AB(_, _, _) => 4,
            Instruction::ORBITS(_, _, _) | Instruction::ORB(_, _, _) | Instruction::OB(_, _, _) => {
                4
            }
            Instruction::XORBITS(_, _, _)
            | Instruction::XORB(_, _, _)
            | Instruction::XB(_, _, _) => 4,
            Instruction::SHL(_, _, _) => 4,
            Instruction::SHR(_, _, _) => 4,

            // 9-byte opcodes (opcode + u64)
            Instruction::WAIT(_) => 1 + 8,
            Instruction::JMP(_) => 1 + 8,
            Instruction::JMPABS(_) | Instruction::JABS(_) | Instruction::JMPA(_) => 1 + 8,
            Instruction::CALLADDR(_) | Instruction::CADDR(_) | Instruction::CA(_) => 1 + 8,
            Instruction::FREE(_) | Instruction::FRE(_) => 1 + 8,
            Instruction::SEEDRNG(_) | Instruction::SRNG(_) => 1 + 8,

            // 10-byte opcodes (opcode + u8 + f64)
            Instruction::PHASESHIFT(_, _) | Instruction::P(_, _) => 1 + 1 + 8,
            Instruction::RX(_, _) => 1 + 1 + 8,
            Instruction::RY(_, _) => 1 + 1 + 8,
            Instruction::RZ(_, _) => 1 + 1 + 8,
            Instruction::PHASE(_, _) | Instruction::PSE(_, _) => 1 + 1 + 8,
            Instruction::REGSET(_, _) | Instruction::RSET(_, _) | Instruction::RGST(_, _) => {
                1 + 1 + 8
            }
            Instruction::TIMEDELAY(_, _) | Instruction::TDELAY(_, _) | Instruction::TD(_, _) => {
                1 + 1 + 8
            }
            Instruction::SETOPTICALATTENUATION(_, _)
            | Instruction::SOATT(_, _)
            | Instruction::SOA(_, _) => 1 + 1 + 8,
            Instruction::DYNAMICPHASECOMPENSATION(_, _)
            | Instruction::DPC(_, _)
            | Instruction::DPCMP(_, _) => 1 + 1 + 8,
            Instruction::OPTICALDELAYLINECONTROL(_, _)
            | Instruction::ODLC(_, _)
            | Instruction::ODL(_, _) => 1 + 1 + 8,
            Instruction::APPLYDISPLACEMENT(_, _)
            | Instruction::ADISP(_, _)
            | Instruction::AD(_, _) => 1 + 1 + 8,
            Instruction::APPLYSQUEEZING(_, _) | Instruction::ASQ(_, _) | Instruction::AS(_, _) => {
                1 + 1 + 8
            }
            Instruction::DECOHERENCEPROTECT(_, _)
            | Instruction::DPROT(_, _)
            | Instruction::DP(_, _) => 1 + 1 + 8,
            Instruction::LOADREGMEM(_, _) | Instruction::LRM(_, _) => 1 + 1 + 8,
            Instruction::ALLOC(_, _) | Instruction::ALC(_, _) => 1 + 1 + 8,
            Instruction::SETWL(_, _) | Instruction::SWL(_, _) | Instruction::SW(_, _) => 1 + 1 + 8,
            Instruction::WLSHIFT(_, _) | Instruction::WLS(_, _) | Instruction::WLSH(_, _) => {
                1 + 1 + 8
            }
            Instruction::SETPHASE(_, _) | Instruction::SETP(_, _) | Instruction::SPH(_, _) => {
                1 + 1 + 8
            } // Changed STP to SPH
            Instruction::APPLYNONLINEARPHASESHIFT(_, _)
            | Instruction::ANLPS(_, _)
            | Instruction::ANLP(_, _)
            | Instruction::ANLPH(_, _) => 1 + 1 + 8,
            Instruction::APPLYNONLINEARSIGMA(_, _)
            | Instruction::ANLS(_, _)
            | Instruction::ANLSI(_, _) => 1 + 1 + 8,

            // 11-byte opcodes (two u8 + f64)
            Instruction::CONTROLLEDPHASEROTATION(_, _, _)
            | Instruction::CPHASE(_, _, _)
            | Instruction::APPLYCPHASE(_, _, _)
            | Instruction::CPR(_, _, _) => 1 + 1 + 1 + 8,
            Instruction::CROSSPHASEMODULATION(_, _, _)
            | Instruction::CPM(_, _, _)
            | Instruction::CPMOD(_, _, _) => 1 + 1 + 1 + 8,
            Instruction::APPLYROTATION(_, _, _)
            | Instruction::ROT(_, _, _)
            | Instruction::AR(_, _, _) => 1 + 1 + 1 + 8,

            // 12-byte opcodes (two u8 + i64 offset)
            Instruction::IFGT(_, _, _) | Instruction::IGT(_, _, _) => 1 + 1 + 1 + 8,
            Instruction::IFLT(_, _, _) | Instruction::ILT(_, _, _) => 1 + 1 + 1 + 8,
            Instruction::IFEQ(_, _, _) | Instruction::IEQ(_, _, _) => 1 + 1 + 1 + 8,
            Instruction::IFNE(_, _, _) | Instruction::INE(_, _, _) => 1 + 1 + 1 + 8,

            // 18-byte opcodes (opcode + u8 + 2 f64 or u64 + f64)
            Instruction::APPLYKERRNONLINEARITY(_, _, _) | Instruction::AKNL(_, _, _) => {
                1 + 1 + 8 + 8
            }
            Instruction::APPLYDISPLACEMENTOPERATOR(_, _, _)
            | Instruction::ADO(_, _, _)
            | Instruction::ADOP(_, _, _) => 1 + 1 + 8 + 8,
            Instruction::PHOTONLOSSSIMULATE(_, _, _)
            | Instruction::PLS(_, _, _)
            | Instruction::PLSIM(_, _, _) => 1 + 1 + 8 + 8,
            Instruction::QUANTUMZENOEFFECT(_, _, _)
            | Instruction::QZE(_, _, _)
            | Instruction::QZEN(_, _, _) => 1 + 1 + 8 + 8,

            // variable length based on string/vec lengths
            Instruction::APPLYGATE(name, _)
            | Instruction::AGATE(name, _)
            | Instruction::AG(name, _) => 1 + 1 + name.len() + 1, // opcode + q + name length + null terminator
            Instruction::ENTANGLEMULTI(qs) | Instruction::EMULTI(qs) | Instruction::EM(qs) => {
                1 + 1 + qs.len()
            }
            Instruction::ENTANGLECLUSTER(qs) | Instruction::ECLUSTER(qs) | Instruction::ECR(qs) => {
                1 + 1 + qs.len()
            }
            Instruction::ENTANGLESWAP(_, _, _, _)
            | Instruction::ESWAP(_, _, _, _)
            | Instruction::ESP(_, _, _, _) => 5,
            Instruction::ENTANGLESWAPMEASURE(_, _, _, _, label)
            | Instruction::ESWAPM(_, _, _, _, label)
            | Instruction::ESM(_, _, _, _, label) => 1 + 4 + label.len() + 1,
            Instruction::ENTANGLEWITHCLASSICALFEEDBACK(_, _, signal)
            | Instruction::ECFB(_, _, signal)
            | Instruction::EWCFB(_, _, signal) => 1 + 2 + signal.len() + 1,
            Instruction::ENTANGLEDISTRIBUTED(_, node)
            | Instruction::EDIST(_, node)
            | Instruction::ED(_, node) => 1 + 1 + node.len() + 1,
            Instruction::MEASUREINBASIS(_, basis)
            | Instruction::MEASB(_, basis)
            | Instruction::MIB(_, basis) => 1 + 1 + basis.len() + 1,
            Instruction::VERBOSELOG(_, msg)
            | Instruction::VLOG(_, msg)
            | Instruction::VL(_, msg) => 1 + 1 + msg.len() + 1, // Changed VL pattern
            Instruction::APPLYFEEDFORWARDGATE(_, reg) | Instruction::AFFG(_, reg) => {
                1 + 1 + reg.len() + 1
            }
            Instruction::APPLYMEASUREMENTBASISCHANGE(_, basis) | Instruction::AMBC(_, basis) => {
                1 + 1 + basis.len() + 1
            }
            Instruction::LOAD(_, reg) | Instruction::LD(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::STORE(_, reg) | Instruction::STR(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::LOADMEM(reg, addr)
            | Instruction::LMEM(reg, addr)
            | Instruction::LM(reg, addr) => 1 + reg.len() + 1 + addr.len() + 1,
            Instruction::STOREMEM(reg, addr)
            | Instruction::SMEM(reg, addr)
            | Instruction::SM(reg, addr) => 1 + reg.len() + 1 + addr.len() + 1,
            Instruction::LOADCLASSICAL(reg, var)
            | Instruction::LCL(reg, var)
            | Instruction::LC(reg, var) => 1 + reg.len() + 1 + var.len() + 1,
            Instruction::STORECLASSICAL(reg, var)
            | Instruction::SCL(reg, var)
            | Instruction::SC(reg, var) => 1 + reg.len() + 1 + var.len() + 1,
            Instruction::ADD(dst, src1, src2) => {
                1 + dst.len() + 1 + src1.len() + 1 + src2.len() + 1
            }
            Instruction::SUB(dst, src1, src2) => {
                1 + dst.len() + 1 + src1.len() + 1 + src2.len() + 1
            }
            Instruction::AND(dst, src1, src2) => {
                1 + dst.len() + 1 + src1.len() + 1 + src2.len() + 1
            }
            Instruction::OR(dst, src1, src2) => 1 + dst.len() + 1 + src1.len() + 1 + src2.len() + 1,
            Instruction::XOR(dst, src1, src2) => {
                1 + dst.len() + 1 + src1.len() + 1 + src2.len() + 1
            }
            Instruction::NOT(reg) => 1 + reg.len() + 1,
            Instruction::PUSH(reg) => 1 + reg.len() + 1,
            Instruction::POP(reg) => 1 + reg.len() + 1,
            Instruction::JUMP(label) => 1 + label.len() + 1,
            Instruction::JUMPIFZERO(cond, label) | Instruction::JIZ(cond, label) => {
                1 + cond.len() + 1 + label.len() + 1
            }
            Instruction::JUMPIFONE(cond, label) | Instruction::JIO(cond, label) => {
                1 + cond.len() + 1 + label.len() + 1
            }
            Instruction::CALL(label) | Instruction::CallLabel(label) => 1 + label.len() + 1,
            Instruction::PHOTONCOUNT(_, reg)
            | Instruction::PCOUNT(_, reg)
            | Instruction::PC(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::PHOTONEMISSIONPATTERN(_, reg, _)
            | Instruction::PEPAT(_, reg, _)
            | Instruction::PEP(_, reg, _) => 1 + 1 + reg.len() + 1 + 8,
            Instruction::PHOTONDETECTWITHTHRESHOLD(_, _, reg)
            | Instruction::PDTHR(_, _, reg)
            | Instruction::PDWT(_, _, reg) => 1 + 1 + 8 + reg.len() + 1,
            Instruction::PHOTONDETECTCOINCIDENCE(qs, reg)
            | Instruction::PDCOIN(qs, reg)
            | Instruction::PDC(qs, reg) => 1 + 1 + qs.len() + reg.len() + 1,
            Instruction::PHOTONROUTE(_, from, to)
            | Instruction::PROUTE(_, from, to)
            | Instruction::PR(_, from, to) => 1 + 1 + from.len() + 1 + to.len() + 1,
            Instruction::APPLYDISPLACEMENTFEEDBACK(_, reg)
            | Instruction::ADF(_, reg)
            | Instruction::ADWF(_, reg)
            | Instruction::ADFB(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::APPLYSQUEEZINGFEEDBACK(_, reg)
            | Instruction::ASF(_, reg)
            | Instruction::ASWF(_, reg)
            | Instruction::ASFB(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::MEASUREWITHDELAY(_, _, reg)
            | Instruction::MWD(_, _, reg)
            | Instruction::MWDEL(_, _, reg) => 1 + 1 + 8 + reg.len() + 1,
            Instruction::PHOTONLOSSCORRECTION(_, reg)
            | Instruction::PLC(_, reg)
            | Instruction::PLCOR(_, reg) => 1 + 1 + reg.len() + 1,

            // qubit measurement and error correction
            Instruction::APPLYQNDMEASUREMENT(_, reg)
            | Instruction::AQND(_, reg)
            | Instruction::AQAD(_, reg)
            | Instruction::AQNM(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::ERRORCORRECT(_, syndrome_type)
            | Instruction::ECORR(_, syndrome_type)
            | Instruction::EC(_, syndrome_type) => 1 + 1 + syndrome_type.len() + 1,
            Instruction::ERRORSYNDROME(_, syndrome_type, result_reg)
            | Instruction::ESYN(_, syndrome_type, result_reg)
            | Instruction::ES(_, syndrome_type, result_reg) => {
                1 + 1 + syndrome_type.len() + 1 + result_reg.len() + 1
            }
            Instruction::QUANTUMSTATETOMOGRAPHY(_, basis)
            | Instruction::QST(_, basis)
            | Instruction::QSTAT(_, basis) => 1 + 1 + basis.len() + 1,
            Instruction::BELLSTATEVERIFICATION(_, _, result_reg)
            | Instruction::BSV(_, _, result_reg)
            | Instruction::BSTATE(_, _, result_reg) => 1 + 2 + result_reg.len() + 1,
            Instruction::APPLYLINEAROPTICALTRANSFORM(_name, input_qs, output_qs, _num_modes)
            | Instruction::ALOT(_name, input_qs, output_qs, _num_modes)
            | Instruction::ALOPT(_name, input_qs, output_qs, _num_modes) => {
                1 + 1 + 1 + 1 + input_qs.len() + output_qs.len()
            }
            Instruction::PHOTONNUMBERRESOLVINGDETECTION(_, reg)
            | Instruction::PNRD(_, reg)
            | Instruction::PNR(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::FEEDBACKCONTROL(_, reg)
            | Instruction::FBC(_, reg)
            | Instruction::FBCONT(_, reg) => 1 + 1 + reg.len() + 1,
            Instruction::STOREMEMREG(_, _) | Instruction::SMR(_, _) => 1 + 8 + 1,
            Instruction::EXITCODE(_) | Instruction::EXC(_) | Instruction::EX(_) => 1 + 4,

            Instruction::SETPOS(_, _, _)
            | Instruction::SPOS(_, _, _)
            | Instruction::STP(_, _, _) => 1 + 1 + 8 + 8,
            Instruction::MOVE(_, _, _) | Instruction::MOV(_, _, _) | Instruction::MV(_, _, _) => {
                1 + 1 + 8 + 8
            }

            Instruction::APPLYMULTIQUBITROTATION(qs, _axis, angles)
            | Instruction::MROT(qs, _axis, angles)
            | Instruction::AMQR(qs, _axis, angles) => 1 + 1 + 1 + qs.len() + (angles.len() * 8),
            Instruction::PRINTF(format_str, regs) | Instruction::PF(format_str, regs) => {
                1 + 8 + format_str.len() + 1 + regs.len()
            }
            Instruction::PRINT(s) => 1 + 8 + s.len(),
            Instruction::PRINTLN(s) | Instruction::PLN(s) => 1 + 8 + s.len(),
        }
    }

    // parses a single line of assembly code into an `Instruction` enum variant.
    // handles comments, labels, and various instruction formats.
    pub fn parse_instruction(line: &str) -> Result<Instruction, String> {
        use Instruction::*;
        let trimmed_line = line.trim();

        if trimmed_line.is_empty() {
            return Err("empty instruction line".into());
        }

        // handle full-line comments starting with ';'
        if trimmed_line.starts_with(';') {
            // return a COMMENT instruction directly
            return Ok(COMMENT(trimmed_line[1..].trim().to_string()));
        }

        // check for label definition (e.g., "MY_LABEL:")
        if trimmed_line.ends_with(':') {
            let label_name = trimmed_line.trim_end_matches(':').to_string();
            if label_name.is_empty() {
                return Err("empty label name".into());
            }
            // label is a nop-instruction, returned here for the first pass of compilation
            return Ok(LABEL(label_name));
        }

        // strip inline comments before tokenizing
        let instruction_part = trimmed_line.split(';').next().unwrap_or("").trim();

        if instruction_part.is_empty() {
            return Err("empty instruction after stripping comment".into());
        }

        // special handling for instructions with string literals
        let mut tokens: Vec<String> = Vec::new();
        let mut in_string = false;
        let mut current_token = String::new();

        for c in instruction_part.chars() {
            if c == '"' {
                if in_string {
                    current_token.push(c); // include the closing quote
                    tokens.push(current_token.clone());
                    current_token.clear();
                    in_string = false;
                } else {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                    current_token.push(c); // include the opening quote
                    in_string = true;
                }
            } else if c.is_whitespace() && !in_string {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
            } else {
                current_token.push(c);
            }
        }
        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        if tokens.is_empty() {
            return Err("empty instruction after tokenization".into());
        }

        // map short names to full names for parsing
        let op_str = tokens[0].to_uppercase();
        let op = match op_str.as_str() {
            "QI" => "QINIT",
            "QM" => "QMEAS",
            "HAD" | "H" => "APPLYHADAMARD",
            "CNOT" | "CN" => "CONTROLLEDNOT",
            "Z" => "APPLYPHASEFLIP",
            "X" => "APPLYBITFLIP",
            "T" => "APPLYTGATE",
            "S" => "APPLYSGATE",
            "P" => "PHASESHIFT",
            "RST" | "RSTQ" => "RESET", 
            "CSWAP" | "CS" => "CONTROLLEDSWAP",
            "EBELL" | "EB" => "ENTANGLEBELL",
            "EMULTI" | "EM" => "ENTANGLEMULTI",
            "ECLUSTER" | "ECR" => "ENTANGLECLUSTER",
            "ESWAP" | "ESP" => "ENTANGLESWAP",
            "ESWAPM" | "ESM" => "ENTANGLESWAPMEASURE",
            "ECFB" | "EWCFB" => "ENTANGLEWITHCLASSICALFEEDBACK",
            "EDIST" | "ED" => "ENTANGLEDISTRIBUTED",
            "MEASB" | "MIB" => "MEASUREINBASIS",
            "RSTALL" | "RSA" => "RESETALL",
            "VLOG" | "VL" => "VERBOSELOG",
            "SETP" | "SPH" => "SETPHASE", 
            "AGATE" | "AG" => "APPLYGATE",
            "MEAS" | "M" => "MEASURE",
            "QINITQ" | "IQ" => "INITQUBIT",
            "CLOAD" | "CLD" => "CHARLOAD",
            "COUT" | "CO" => "CHAROUT",
            "QR" => "QRESET",
            "TAVG" | "TA" => "THERMALAVG",
            "WKBF" | "WF" => "WKBFACTOR",
            "RSET" | "RGST" => "REGSET",
            "LSTART" | "LS" => "LOOPSTART",
            "LEND" | "LE" => "LOOPEND",
            "ROT" | "AR" => "APPLYROTATION",
            "MROT" | "AMQR" => "APPLYMULTIQUBITROTATION",
            "CPHASE" | "APPLYCPHASE" | "CPR" => "CONTROLLEDPHASEROTATION",
            "AKNL" => "APPLYKERRNONLINEARITY",
            "AFFG" => "APPLYFEEDFORWARDGATE",
            "DPROT" | "DP" => "DECOHERENCEPROTECT",
            "AMBC" => "APPLYMEASUREMENTBASISCHANGE",
            "LD" => "LOAD",
            "STR" => "STORE",
            "LMEM" | "LM" => "LOADMEM",
            "SMEM" | "SM" => "STOREMEM",
            "LCL" | "LC" => "LOADCLASSICAL",
            "SCL" | "SC" => "STORECLASSICAL",
            "JIZ" => "JUMPIFZERO",
            "JIO" => "JUMPIFONE",
            "CALL_LABEL" => "CallLabel",
            "BR" => "BARRIER",
            "RET" => "RETURN", 
            "TDELAY" | "TD" => "TIMEDELAY",
            "RN" => "RAND",
            "SR" => "SQRT",
            "RADD" | "RGA" => "REGADD",
            "RSUB" | "RGS" => "REGSUB",
            "RMUL" | "RM" => "REGMUL",
            "RDIV" | "RD" => "REGDIV",
            "RCOPY" | "RC" => "REGCOPY",
            "PEMIT" | "PE" => "PHOTONEMIT",
            "PDETECT" | "PD" => "PHOTONDETECT",
            "PCOUNT" | "PC" => "PHOTONCOUNT",
            "PADD" | "PA" => "PHOTONADDITION",
            "APSUB" | "APS" => "APPLYPHOTONSUBTRACTION",
            "PEPAT" | "PEP" => "PHOTONEMISSIONPATTERN",
            "PDTHR" | "PDWT" => "PHOTONDETECTWITHTHRESHOLD",
            "PDCOIN" | "PDC" => "PHOTONDETECTCOINCIDENCE",
            "PBUNCH" | "PBC" => "PHOTONBUNCHINGCONTROL",
            "PROUTE" | "PR" => "PHOTONROUTE",
            "OROUTE" | "OPTR" => "OPTICALROUTING",
            "SOATT" | "SOA" => "SETOPTICALATTENUATION",
            "DPC" | "DPCMP" => "DYNAMICPHASECOMPENSATION",
            "ODLC" | "ODL" => "OPTICALDELAYLINECONTROL",
            "CPM" | "CPMOD" => "CROSSPHASEMODULATION",
            "ADISP" | "AD" => "APPLYDISPLACEMENT",
            "ADF" | "ADWF" | "ADFB" => "APPLYDISPLACEMENTFEEDBACK",
            "ADO" | "ADOP" => "APPLYDISPLACEMENTOPERATOR",
            "ASQ" | "AS" => "APPLYSQUEEZING",
            "ASF" | "ASWF" | "ASFB" => "APPLYSQUEEZINGFEEDBACK",
            "MPAR" | "MP" => "MEASUREPARITY",
            "MWD" | "MWDEL" => "MEASUREWITHDELAY",
            "OSC" | "OSW" => "OPTICALSWITCHCONTROL",
            "PLS" | "PLSIM" => "PHOTONLOSSSIMULATE",
            "PLC" | "PLCOR" => "PHOTONLOSSCORRECTION",
            "AQND" | "AQAD" | "AQNM" => "APPLYQNDMEASUREMENT",
            "ECORR" | "EC" => "ERRORCORRECT",
            "ESYN" | "ES" => "ERRORSYNDROME",
            "QST" | "QSTAT" => "QUANTUMSTATETOMOGRAPHY",
            "BSV" | "BSTATE" => "BELLSTATEVERIFICATION",
            "QZE" | "QZEN" => "QUANTUMZENOEFFECT",
            "ANLPS" | "ANLP" | "ANLPH" => "APPLYNONLINEARPHASESHIFT",
            "ANLS" | "ANLSI" => "APPLYNONLINEARSIGMA",
            "ALOT" | "ALOPT" => "APPLYLINEAROPTICALTRANSFORM",
            "PNRD" | "PNR" => "PHOTONNUMBERRESOLVINGDETECTION",
            "FBC" | "FBCONT" => "FEEDBACKCONTROL",
            "SPOS" | "STP" => "SETPOS",
            "SWL" | "SW" => "SETWL",
            "WLS" | "WLSH" => "WLSHIFT",
            "MOV" | "MV" => "MOVE",
            "CMT" | "CM" => "COMMENT",
            "MOBS" | "MO" => "MARKOBSERVED",
            "REL" | "RL" => "RELEASE",
            "HLT" => "HALT",
            "JABS" | "JMPA" => "JMPABS",
            "IGT" => "IFGT",
            "ILT" => "IFLT",
            "IEQ" => "IFEQ",
            "INE" => "IFNE",
            "CADDR" | "CA" => "CALLADDR",
            "RS" => "RETSUB",
            "PF" => "PRINTF",
            "PLN" => "PRINTLN",
            "INP" => "INPUT",
            "DSTATE" | "DS" => "DUMPSTATE",
            "DREGS" | "DR" => "DUMPREGS",
            "LRM" => "LOADREGMEM",
            "SMR" => "STOREMEMREG",
            "PSHRG" | "PRG" => "PUSHREG",
            "POPRG" | "PPRG" => "POPREG",
            "ALC" => "ALLOC",
            "FRE" => "FREE",
            "ANDB" | "AB" => "ANDBITS",
            "ORB" | "OB" => "ORBITS",
            "XORB" | "XB" => "XORBITS",
            "NOTB" | "NB" => "NOTBITS",
            "BP" => "BREAKPOINT",
            "GTIME" | "GT" => "GETTIME",
            "SRNG" => "SEEDRNG",
            "EXC" | "EX" => "EXITCODE",
            "PSE" => "PHASE",
            _ => &op_str, // if no short name, use the original token
        };

        // Convert tokens to Vec<&str> for existing parsing logic
        let str_tokens: Vec<&str> = tokens.iter().map(AsRef::as_ref).collect();

        match op {
            // core quantum operations
            "QINIT" => {
                if str_tokens.len() == 2 {
                    Ok(QINIT(parse_u8(str_tokens[1])?))
                } else {
                    Err("qinit <qubit>".into())
                }
            }
            "QMEAS" => {
                if str_tokens.len() == 2 {
                    Ok(QMEAS(parse_u8(str_tokens[1])?))
                } else {
                    Err("qmeas <qubit>".into())
                }
            }
            "APPLYHADAMARD" => {
                if str_tokens.len() == 2 {
                    Ok(H(parse_u8(str_tokens[1])?))
                } else {
                    Err("usage: applyhadamard <qubit>".into())
                }
            }
            "CONTROLLEDNOT" => {
                if str_tokens.len() == 3 {
                    Ok(CONTROLLEDNOT(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                    ))
                } else {
                    Err("controllednot <c> <t>".into())
                }
            }
            "APPLYPHASEFLIP" => {
                if str_tokens.len() == 2 {
                    Ok(APPLYPHASEFLIP(parse_u8(str_tokens[1])?))
                } else {
                    Err("applyphaseflip <qubit>".into())
                }
            }
            "APPLYBITFLIP" => {
                if str_tokens.len() == 2 {
                    Ok(APPLYBITFLIP(parse_u8(str_tokens[1])?))
                } else {
                    Err("applybitflip <qubit>".into())
                }
            }
            "APPLYTGATE" => {
                if str_tokens.len() == 2 {
                    Ok(APPLYTGATE(parse_u8(str_tokens[1])?))
                } else {
                    Err("applytgate <qubit>".into())
                }
            }
            "APPLYSGATE" => {
                if str_tokens.len() == 2 {
                    Ok(APPLYSGATE(parse_u8(str_tokens[1])?))
                } else {
                    Err("applysgate <qubit>".into())
                }
            }
            "PHASESHIFT" => {
                if str_tokens.len() == 3 {
                    Ok(PHASESHIFT(
                        parse_u8(str_tokens[1])?,
                        parse_f64(str_tokens[2])?,
                    ))
                } else {
                    Err("phaseshift <qubit> <angle>".into())
                }
            }
            "WAIT" => {
                if str_tokens.len() == 2 {
                    Ok(WAIT(parse_u64(str_tokens[1])?))
                } else {
                    Err("wait <cycles>".into())
                }
            }
            "RESET" => {
                if str_tokens.len() == 2 {
                    Ok(RESET(parse_u8(str_tokens[1])?))
                } else {
                    Err("reset <qubit>".into())
                }
            }
            "SWAP" => {
                if str_tokens.len() == 3 {
                    Ok(SWAP(parse_u8(str_tokens[1])?, parse_u8(str_tokens[2])?))
                } else {
                    Err("swap <q1> <q2>".into())
                }
            }
            "CONTROLLEDSWAP" => {
                if str_tokens.len() == 4 {
                    Ok(CONTROLLEDSWAP(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_u8(str_tokens[3])?,
                    ))
                } else {
                    Err("controlledswap <c> <t1> <t2>".into())
                }
            }
            "ENTANGLE" => {
                if str_tokens.len() == 3 {
                    Ok(ENTANGLE(parse_u8(str_tokens[1])?, parse_u8(str_tokens[2])?))
                } else {
                    Err("entangle <q1> <q2>".into())
                }
            }
            "ENTANGLEBELL" => {
                if str_tokens.len() == 3 {
                    Ok(ENTANGLEBELL(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                    ))
                } else {
                    Err("entanglebell <q1> <q2>".into())
                }
            }
            "ENTANGLEMULTI" => {
                if str_tokens.len() >= 2 {
                    Ok(ENTANGLEMULTI(
                        str_tokens[1..]
                            .iter()
                            .map(|q| parse_u8(q))
                            .collect::<Result<_, _>>()?,
                    ))
                } else {
                    Err("entanglemulti <q1> <q2> ...".into())
                }
            }
            "ENTANGLECLUSTER" => {
                if str_tokens.len() >= 2 {
                    Ok(ENTANGLECLUSTER(
                        str_tokens[1..]
                            .iter()
                            .map(|q| parse_u8(q))
                            .collect::<Result<_, _>>()?,
                    ))
                } else {
                    Err("entanglecluster <q1> <q2> ...".into())
                }
            }
            "ENTANGLESWAP" => {
                if str_tokens.len() == 5 {
                    Ok(ENTANGLESWAP(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_u8(str_tokens[3])?,
                        parse_u8(str_tokens[4])?,
                    ))
                } else {
                    Err("entangleswap <a> <b> <c> <d>".into())
                }
            }
            "ENTANGLESWAPMEASURE" => {
                if str_tokens.len() == 6 {
                    Ok(ENTANGLESWAPMEASURE(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_u8(str_tokens[3])?,
                        parse_u8(str_tokens[4])?,
                        parse_string_literal(str_tokens[5])?,
                    ))
                } else {
                    Err("entangleswapmeasure <a> <b> <c> <d> <label>".into())
                }
            }
            "ENTANGLEWITHCLASSICALFEEDBACK" => {
                if str_tokens.len() == 4 {
                    Ok(ENTANGLEWITHCLASSICALFEEDBACK(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_string_literal(str_tokens[3])?,
                    ))
                } else {
                    Err("entanglewithclassicalfeedback <q1> <q2> <signal>".into())
                }
            }
            "ENTANGLEDISTRIBUTED" => {
                if str_tokens.len() == 3 {
                    Ok(ENTANGLEDISTRIBUTED(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("entangledistributed <qubit> <node>".into())
                }
            }
            "MEASUREINBASIS" => {
                if str_tokens.len() == 3 {
                    Ok(MEASUREINBASIS(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("measureinbasis <qubit> <basis>".into())
                }
            }
            "SYNC" => {
                if str_tokens.len() == 1 {
                    Ok(SYNC)
                } else {
                    Err("sync".into())
                }
            }
            "RESETALL" => {
                if str_tokens.len() == 1 {
                    Ok(RESETALL)
                } else {
                    Err("resetall".into())
                }
            }
            "VERBOSELOG" => {
                if str_tokens.len() == 3 {
                    // Updated to expect 3 tokens: VERBOSELOG, qubit, "message"
                    Ok(VERBOSELOG(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("verboselog <qubit> <message>".into())
                }
            }
            "SETPHASE" => {
                if str_tokens.len() == 3 {
                    Ok(SETPHASE(
                        parse_u8(str_tokens[1])?,
                        parse_f64(str_tokens[2])?,
                    ))
                } else {
                    Err("setphase <qubit> <phase>".into())
                }
            }
            "APPLYGATE" => {
                if str_tokens.len() == 3 {
                    Ok(APPLYGATE(
                        parse_string_literal(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                    ))
                } else {
                    Err("applygate <gate_name> <qubit>".into())
                }
            }
            "MEASURE" => {
                // handle MEASURE alias
                if str_tokens.len() == 2 {
                    Ok(MEASURE(parse_u8(str_tokens[1])?))
                } else {
                    Err("measure <qubit>".into())
                }
            }
            "INITQUBIT" => {
                // handle INITQUBIT alias
                if str_tokens.len() == 2 {
                    Ok(INITQUBIT(parse_u8(str_tokens[1])?))
                } else {
                    Err("initqubit <qubit>".into())
                }
            }
            "LABEL" => {
                if str_tokens.len() == 2 {
                    Ok(LABEL(str_tokens[1].to_string()))
                } else {
                    Err("label <name>".into())
                }
            }

            // char printing operations
            "CHARLOAD" => {
                if str_tokens.len() == 3 {
                    let reg = parse_u8(str_tokens[1])?;
                    let val_str = str_tokens[2];
                    if val_str.len() == 3 && val_str.starts_with('\'') && val_str.ends_with('\'') {
                        let ch = val_str.chars().nth(1).unwrap();
                        Ok(CHARLOAD(reg, ch as u8))
                    } else {
                        Err("usage: charload <reg> '<char>'".into())
                    }
                } else {
                    Err("usage: charload <reg> '<char>'".into())
                }
            }
            "CHAROUT" => {
                if str_tokens.len() == 2 {
                    Ok(CHAROUT(parse_u8(str_tokens[1])?))
                } else {
                    Err("usage: charout <reg>".into())
                }
            }
            // ionq specific ISA gates
            "RX" => {
                if str_tokens.len() == 3 {
                    Ok(RX(parse_u8(str_tokens[1])?, parse_f64(str_tokens[2])?))
                } else {
                    Err("rx <qubit> <angle>".into())
                }
            }
            "RY" => {
                if str_tokens.len() == 3 {
                    Ok(RY(parse_u8(str_tokens[1])?, parse_f64(str_tokens[2])?))
                } else {
                    Err("ry <qubit> <angle>".into())
                }
            }
            "RZ" => {
                if str_tokens.len() == 3 {
                    Ok(RZ(parse_u8(str_tokens[1])?, parse_f64(str_tokens[2])?))
                } else {
                    Err("rz <qubit> <angle>".into())
                }
            }
            "PHASE" => {
                if str_tokens.len() == 3 {
                    Ok(PHASE(parse_u8(str_tokens[1])?, parse_f64(str_tokens[2])?))
                } else {
                    Err("phase <qubit> <angle>".into())
                }
            }
            "CZ" => {
                if str_tokens.len() == 3 {
                    Ok(CZ(parse_u8(str_tokens[1])?, parse_u8(str_tokens[2])?))
                } else {
                    Err("cz <control_qubit> <target_qubit>".into())
                }
            }
            "QRESET" => {
                // handle QRESET alias
                if str_tokens.len() == 2 {
                    Ok(QRESET(parse_u8(str_tokens[1])?))
                } else {
                    Err("qreset <qubit>".into())
                }
            }
            "THERMALAVG" => {
                if str_tokens.len() == 3 {
                    Ok(THERMALAVG(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                    ))
                } else {
                    Err("thermalavg <qubit> <param>".into())
                }
            }
            "WKBFACTOR" => {
                if str_tokens.len() == 4 {
                    Ok(WKBFACTOR(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_u8(str_tokens[3])?,
                    ))
                } else {
                    Err("wkbfactor <q1> <q2> <param>".into())
                }
            }

            // register set operation
            "REGSET" => {
                if str_tokens.len() == 3 {
                    Ok(REGSET(parse_u8(str_tokens[1])?, parse_f64(str_tokens[2])?))
                } else {
                    Err("usage: regset <reg> <float_value>".into())
                }
            }

            // loop control operations
            "LOOPSTART" => {
                if str_tokens.len() == 2 {
                    Ok(LOOPSTART(parse_u8(str_tokens[1])?))
                } else {
                    Err("loopstart <reg>".into())
                }
            }
            "LOOPEND" => {
                if str_tokens.len() == 1 {
                    Ok(LOOPEND)
                } else {
                    Err("loopend".into())
                }
            }

            // rotation operations
            "APPLYROTATION" => {
                if str_tokens.len() == 4 {
                    Ok(APPLYROTATION(
                        parse_u8(str_tokens[1])?,
                        parse_axis(str_tokens[2])?,
                        parse_f64(str_tokens[3])?,
                    ))
                } else {
                    Err("applyrotation <q> <x|y|z> <angle>".into())
                }
            }
            "APPLYMULTIQUBITROTATION" => {
                if str_tokens.len() >= 4 {
                    let axis = parse_axis(str_tokens[2])?;
                    let qs = str_tokens[1]
                        .split(',')
                        .map(parse_u8)
                        .collect::<Result<Vec<_>, _>>()?;
                    let angles = str_tokens[3..]
                        .iter()
                        .map(|s| parse_f64(s))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(APPLYMULTIQUBITROTATION(qs, axis, angles))
                } else {
                    Err("applymultiqubitrotation <q1,q2,...> <x|y|z> <a1> <a2> ...".into())
                }
            }
            "CONTROLLEDPHASEROTATION" => {
                if str_tokens.len() == 4 {
                    Ok(CONTROLLEDPHASEROTATION(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_f64(str_tokens[3])?,
                    ))
                } else {
                    Err("controlledphaserotation <c> <t> <angle>".into())
                }
            }
            "APPLYKERRNONLINEARITY" => {
                if str_tokens.len() == 4 {
                    Ok(APPLYKERRNONLINEARITY(
                        parse_u8(str_tokens[1])?,
                        parse_f64(str_tokens[2])?,
                        parse_u64(str_tokens[3])?,
                    ))
                } else {
                    Err("applykerrnonlinearity <q> <strength> <duration>".into())
                }
            }
            "APPLYFEEDFORWARDGATE" => {
                if str_tokens.len() == 3 {
                    Ok(APPLYFEEDFORWARDGATE(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("applyfeedforwardgate <q> <reg>".into())
                }
            }
            "DECOHERENCEPROTECT" => {
                if str_tokens.len() == 3 {
                    Ok(DECOHERENCEPROTECT(
                        parse_u8(str_tokens[1])?,
                        parse_u64(str_tokens[2])?,
                    ))
                } else {
                    Err("decoherenceprotect <q> <duration>".into())
                }
            }
            "APPLYMEASUREMENTBASISCHANGE" => {
                if str_tokens.len() == 3 {
                    Ok(APPLYMEASUREMENTBASISCHANGE(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("applymeasurementbasischange <q> <basis>".into())
                }
            }

            // memory and classical operations
            "LOAD" => {
                if str_tokens.len() == 3 {
                    Ok(LOAD(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("load <qubit> <var>".into())
                }
            }
            "STORE" => {
                if str_tokens.len() == 3 {
                    Ok(STORE(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("store <qubit> <var>".into())
                }
            }
            "LOADMEM" => {
                if str_tokens.len() == 3 {
                    Ok(LOADMEM(
                        parse_string_literal(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("loadmem <reg> <mem>".into())
                }
            }
            "STOREMEM" => {
                if str_tokens.len() == 3 {
                    Ok(STOREMEM(
                        parse_string_literal(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("storemem <reg> <mem>".into())
                }
            }
            "LOADCLASSICAL" => {
                if str_tokens.len() == 3 {
                    Ok(LOADCLASSICAL(
                        parse_string_literal(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("loadclassical <reg> <var>".into())
                }
            }
            "STORECLASSICAL" => {
                if str_tokens.len() == 3 {
                    Ok(STORECLASSICAL(
                        parse_string_literal(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("storeclassical <reg> <var>".into())
                }
            }
            "ADD" => {
                if str_tokens.len() == 4 {
                    Ok(ADD(
                        parse_string_literal(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                        parse_string_literal(str_tokens[3])?,
                    ))
                } else {
                    Err("add <dst> <src1> <src2>".into())
                }
            }
            "SUB" => {
                if str_tokens.len() == 4 {
                    Ok(SUB(
                        parse_string_literal(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                        parse_string_literal(str_tokens[3])?,
                    ))
                } else {
                    Err("sub <dst> <src1> <src2>".into())
                }
            }
            "AND" => {
                if str_tokens.len() == 4 {
                    Ok(AND(
                        parse_string_literal(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                        parse_string_literal(str_tokens[3])?,
                    ))
                } else {
                    Err("and <dst> <src1> <src2>".into())
                }
            }
            "OR" => {
                if str_tokens.len() == 4 {
                    Ok(OR(
                        parse_string_literal(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                        parse_string_literal(str_tokens[3])?,
                    ))
                } else {
                    Err("or <dst> <src1> <src2>".into())
                }
            }
            "XOR" => {
                if str_tokens.len() == 4 {
                    Ok(XOR(
                        parse_string_literal(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                        parse_string_literal(str_tokens[3])?,
                    ))
                } else {
                    Err("xor <dst> <src1> <src2>".into())
                }
            }
            "NOT" => {
                if str_tokens.len() == 2 {
                    Ok(NOT(parse_string_literal(str_tokens[1])?))
                } else {
                    Err("not <reg>".into())
                }
            }
            "PUSH" => {
                if str_tokens.len() == 2 {
                    Ok(PUSH(parse_string_literal(str_tokens[1])?))
                } else {
                    Err("push <reg>".into())
                }
            }
            "POP" => {
                if str_tokens.len() == 2 {
                    Ok(POP(parse_string_literal(str_tokens[1])?))
                } else {
                    Err("pop <reg>".into())
                }
            }

            // classical control flow
            "JUMP" => {
                // This is for label jumps
                if str_tokens.len() == 2 {
                    Ok(JUMP(parse_string_literal(str_tokens[1])?))
                } else {
                    Err("jump <label>".into())
                }
            }
            "JMP" => {
                // This is for relative i64 jumps
                if str_tokens.len() == 2 {
                    Ok(JMP(parse_i64(str_tokens[1])?))
                } else {
                    Err("jmp <offset>".into())
                }
            }
            "JUMPIFZERO" => {
                if str_tokens.len() == 3 {
                    Ok(JUMPIFZERO(
                        parse_string_literal(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("jumpifzero <cond_reg> <label>".into())
                }
            }
            "JUMPIFONE" => {
                if str_tokens.len() == 3 {
                    Ok(JUMPIFONE(
                        parse_string_literal(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("jumpifone <cond_reg> <label>".into())
                }
            }
            "CALL" => {
                // This is for label calls
                if str_tokens.len() == 2 {
                    Ok(CALL(parse_string_literal(str_tokens[1])?))
                } else {
                    Err("call <label>".into())
                }
            }
            "RETURN" => {
                if str_tokens.len() == 1 {
                    Ok(RETURN)
                } else {
                    Err("return".into())
                }
            }
            "TIMEDELAY" => {
                if str_tokens.len() == 3 {
                    Ok(TIMEDELAY(
                        parse_u8(str_tokens[1])?,
                        parse_u64(str_tokens[2])?,
                    ))
                } else {
                    Err("timedelay <qubit> <cycles>".into())
                }
            }
            "RAND" => {
                if str_tokens.len() == 2 {
                    Ok(RAND(parse_u8(str_tokens[1])?))
                } else {
                    Err("rand <reg>".into())
                }
            }
            "SQRT" => {
                if str_tokens.len() == 3 {
                    Ok(SQRT(parse_u8(str_tokens[1])?, parse_u8(str_tokens[2])?))
                } else {
                    Err("sqrt <rd> <rs>".into())
                }
            }
            "EXP" => {
                if str_tokens.len() == 3 {
                    Ok(EXP(parse_u8(str_tokens[1])?, parse_u8(str_tokens[2])?))
                } else {
                    Err("exp <rd> <rs>".into())
                }
            }
            "LOG" => {
                if str_tokens.len() == 3 {
                    Ok(LOG(parse_u8(str_tokens[1])?, parse_u8(str_tokens[2])?))
                } else {
                    Err("log <rd> <rs>".into())
                }
            }
            // arithmetic operations on registers
            "REGADD" => {
                if str_tokens.len() == 4 {
                    Ok(REGADD(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_u8(str_tokens[3])?,
                    ))
                } else {
                    Err("regadd <rd> <ra> <rb>".into())
                }
            }
            "REGSUB" => {
                if str_tokens.len() == 4 {
                    Ok(REGSUB(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_u8(str_tokens[3])?,
                    ))
                } else {
                    Err("regsub <rd> <ra> <rb>".into())
                }
            }
            "REGMUL" => {
                if str_tokens.len() == 4 {
                    Ok(REGMUL(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_u8(str_tokens[3])?,
                    ))
                } else {
                    Err("regmul <rd> <ra> <rb>".into())
                }
            }
            "REGDIV" => {
                if str_tokens.len() == 4 {
                    Ok(REGDIV(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_u8(str_tokens[3])?,
                    ))
                } else {
                    Err("regdiv <rd> <ra> <rb>".into())
                }
            }
            "REGCOPY" => {
                if str_tokens.len() == 3 {
                    Ok(REGCOPY(parse_u8(str_tokens[1])?, parse_u8(str_tokens[2])?))
                } else {
                    Err("regcopy <rd> <ra>".into())
                }
            }

            // optics and photonics operations
            "PHOTONEMIT" => {
                if str_tokens.len() == 2 {
                    Ok(PHOTONEMIT(parse_u8(str_tokens[1])?))
                } else {
                    Err("photonemit <qubit>".into())
                }
            }
            "PHOTONDETECT" => {
                if str_tokens.len() == 2 {
                    Ok(PHOTONDETECT(parse_u8(str_tokens[1])?))
                } else {
                    Err("photondetect <qubit>".into())
                }
            }
            "PHOTONCOUNT" => {
                if str_tokens.len() == 3 {
                    Ok(PHOTONCOUNT(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("photoncount <qubit> <result_reg>".into())
                }
            }
            "PHOTONADDITION" => {
                if str_tokens.len() == 2 {
                    Ok(PHOTONADDITION(parse_u8(str_tokens[1])?))
                } else {
                    Err("photonaddition <qubit>".into())
                }
            }
            "APPLYPHOTONSUBTRACTION" => {
                if str_tokens.len() == 2 {
                    Ok(APPLYPHOTONSUBTRACTION(parse_u8(str_tokens[1])?))
                } else {
                    Err("applyphotonsubtraction <qubit>".into())
                }
            }
            "PHOTONEMISSIONPATTERN" => {
                if str_tokens.len() == 4 {
                    Ok(PHOTONEMISSIONPATTERN(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                        parse_u64(str_tokens[3])?,
                    ))
                } else {
                    Err("photonemissionpattern <qubit> <pattern_reg> <cycles>".into())
                }
            }
            "PHOTONDETECTWITHTHRESHOLD" => {
                if str_tokens.len() == 4 {
                    Ok(PHOTONDETECTWITHTHRESHOLD(
                        parse_u8(str_tokens[1])?,
                        parse_u64(str_tokens[2])?,
                        parse_string_literal(str_tokens[3])?,
                    ))
                } else {
                    Err("photondetectwiththreshold <qubit> <threshold> <result_reg>".into())
                }
            }
            "PHOTONDETECTCOINCIDENCE" => {
                if str_tokens.len() >= 3 {
                    let qs = str_tokens[1]
                        .split(',')
                        .map(parse_u8)
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(PHOTONDETECTCOINCIDENCE(
                        qs,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("photondetectcoincidence <q1,q2,...> <result_reg>".into())
                }
            }
            "SINGLEPHOTONSOURCEON" => {
                if str_tokens.len() == 2 {
                    Ok(SINGLEPHOTONSOURCEON(parse_u8(str_tokens[1])?))
                } else {
                    Err("singlephotonsourceon <qubit>".into())
                }
            }
            "SINGLEPHOTONSOURCEOFF" => {
                if str_tokens.len() == 2 {
                    Ok(SINGLEPHOTONSOURCEOFF(parse_u8(str_tokens[1])?))
                } else {
                    Err("singlephotonsourceoff <qubit>".into())
                }
            }
            "PHOTONBUNCHINGCONTROL" => {
                if str_tokens.len() == 3 {
                    Ok(PHOTONBUNCHINGCONTROL(
                        parse_u8(str_tokens[1])?,
                        parse_bool(str_tokens[2])?,
                    ))
                } else {
                    Err("photonbunchingcontrol <qubit> <true|false>".into())
                }
            }
            "PHOTONROUTE" => {
                if str_tokens.len() == 4 {
                    Ok(PHOTONROUTE(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                        parse_string_literal(str_tokens[3])?,
                    ))
                } else {
                    Err("photonroute <qubit> <from_port> <to_port>".into())
                }
            }
            "OPTICALROUTING" => {
                if str_tokens.len() == 3 {
                    Ok(OPTICALROUTING(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                    ))
                } else {
                    Err("opticalrouting <q1> <q2>".into())
                }
            }
            "SETOPTICALATTENUATION" => {
                if str_tokens.len() == 3 {
                    Ok(SETOPTICALATTENUATION(
                        parse_u8(str_tokens[1])?,
                        parse_f64(str_tokens[2])?,
                    ))
                } else {
                    Err("setopticalattenuation <qubit> <attenuation>".into())
                }
            }
            "DYNAMICPHASECOMPENSATION" => {
                if str_tokens.len() == 3 {
                    Ok(DYNAMICPHASECOMPENSATION(
                        parse_u8(str_tokens[1])?,
                        parse_f64(str_tokens[2])?,
                    ))
                } else {
                    Err("dynamicphasecompensation <qubit> <phase>".into())
                }
            }
            "OPTICALDELAYLINECONTROL" => {
                if str_tokens.len() == 3 {
                    Ok(OPTICALDELAYLINECONTROL(
                        parse_u8(str_tokens[1])?,
                        parse_u64(str_tokens[2])?,
                    ))
                } else {
                    Err("opticaldelaylinecontrol <qubit> <delay_cycles>".into())
                }
            }
            "CROSSPHASEMODULATION" => {
                if str_tokens.len() == 4 {
                    Ok(CROSSPHASEMODULATION(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_f64(str_tokens[3])?,
                    ))
                } else {
                    Err("crossphasemodulation <q1> <q2> <strength>".into())
                }
            }
            "APPLYDISPLACEMENT" => {
                if str_tokens.len() == 3 {
                    Ok(APPLYDISPLACEMENT(
                        parse_u8(str_tokens[1])?,
                        parse_f64(str_tokens[2])?,
                    ))
                } else {
                    Err("applydisplacement <qubit> <alpha>".into())
                }
            }
            "APPLYDISPLACEMENTFEEDBACK" => {
                if str_tokens.len() == 3 {
                    Ok(APPLYDISPLACEMENTFEEDBACK(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("applydisplacementfeedback <qubit> <feedback_reg>".into())
                }
            }
            "APPLYDISPLACEMENTOPERATOR" => {
                if str_tokens.len() == 4 {
                    Ok(APPLYDISPLACEMENTOPERATOR(
                        parse_u8(str_tokens[1])?,
                        parse_f64(str_tokens[2])?,
                        parse_u64(str_tokens[3])?,
                    ))
                } else {
                    Err("applydisplacementoperator <qubit> <alpha> <duration>".into())
                }
            }
            "APPLYSQUEEZING" => {
                if str_tokens.len() == 3 {
                    Ok(APPLYSQUEEZING(
                        parse_u8(str_tokens[1])?,
                        parse_f64(str_tokens[2])?,
                    ))
                } else {
                    Err("applysqueezing <qubit> <squeezing_factor>".into())
                }
            }
            "APPLYSQUEEZINGFEEDBACK" => {
                if str_tokens.len() == 3 {
                    Ok(APPLYSQUEEZINGFEEDBACK(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("applysqueezingfeedback <qubit> <feedback_reg>".into())
                }
            }
            "MEASUREPARITY" => {
                if str_tokens.len() == 2 {
                    Ok(MEASUREPARITY(parse_u8(str_tokens[1])?))
                } else {
                    Err("measureparity <qubit>".into())
                }
            }
            "MEASUREWITHDELAY" => {
                if str_tokens.len() == 4 {
                    Ok(MEASUREWITHDELAY(
                        parse_u8(str_tokens[1])?,
                        parse_u64(str_tokens[2])?,
                        parse_string_literal(str_tokens[3])?,
                    ))
                } else {
                    Err("measurewithdelay <qubit> <delay_cycles> <result_reg>".into())
                }
            }
            "OPTICALSWITCHCONTROL" => {
                if str_tokens.len() == 3 {
                    Ok(OPTICALSWITCHCONTROL(
                        parse_u8(str_tokens[1])?,
                        parse_bool(str_tokens[2])?,
                    ))
                } else {
                    Err("opticalswitchcontrol <qubit> <on|off>".into())
                }
            }
            "PHOTONLOSSSIMULATE" => {
                if str_tokens.len() == 4 {
                    Ok(PHOTONLOSSSIMULATE(
                        parse_u8(str_tokens[1])?,
                        parse_f64(str_tokens[2])?,
                        parse_u64(str_tokens[3])?,
                    ))
                } else {
                    Err("photonlosssimulate <qubit> <loss_probability> <seed>".into())
                }
            }
            "PHOTONLOSSCORRECTION" => {
                if str_tokens.len() == 3 {
                    Ok(PHOTONLOSSCORRECTION(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("photonlosscorrection <qubit> <correction_reg>".into())
                }
            }

            // qubit measurement and error correction
            "APPLYQNDMEASUREMENT" => {
                if str_tokens.len() == 3 {
                    Ok(APPLYQNDMEASUREMENT(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("applyqndmeasurement <qubit> <result_reg>".into())
                }
            }
            "ERRORCORRECT" => {
                if str_tokens.len() == 3 {
                    Ok(ERRORCORRECT(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("errorcorrect <qubit> <syndrome_type>".into())
                }
            }
            "ERRORSYNDROME" => {
                if str_tokens.len() == 4 {
                    Ok(ERRORSYNDROME(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                        parse_string_literal(str_tokens[3])?,
                    ))
                } else {
                    Err("errorsyndrome <qubit> <syndrome_type> <result_reg>".into())
                }
            }
            "QUANTUMSTATETOMOGRAPHY" => {
                if str_tokens.len() == 3 {
                    Ok(QUANTUMSTATETOMOGRAPHY(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("quantumstatetomography <qubit> <basis>".into())
                }
            }
            "BELLSTATEVERIFICATION" => {
                if str_tokens.len() == 4 {
                    Ok(BELLSTATEVERIFICATION(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_string_literal(str_tokens[3])?,
                    ))
                } else {
                    Err("bellstateverification <q1> <q2> <result_reg>".into())
                }
            }
            "QUANTUMZENOEFFECT" => {
                if str_tokens.len() == 4 {
                    Ok(QUANTUMZENOEFFECT(
                        parse_u8(str_tokens[1])?,
                        parse_u64(str_tokens[2])?,
                        parse_u64(str_tokens[3])?,
                    ))
                } else {
                    Err("quantumzenoeffect <qubit> <num_measurements> <interval_cycles>".into())
                }
            }
            "APPLYNONLINEARPHASESHIFT" => {
                if str_tokens.len() == 3 {
                    Ok(APPLYNONLINEARPHASESHIFT(
                        parse_u8(str_tokens[1])?,
                        parse_f64(str_tokens[2])?,
                    ))
                } else {
                    Err("applynonlinearphaseshift <qubit> <strength>".into())
                }
            }
            "APPLYNONLINEARSIGMA" => {
                if str_tokens.len() == 3 {
                    Ok(APPLYNONLINEARSIGMA(
                        parse_u8(str_tokens[1])?,
                        parse_f64(str_tokens[2])?,
                    ))
                } else {
                    Err("applynonlinearsigma <qubit> <strength>".into())
                }
            }
            "APPLYLINEAROPTICALTRANSFORM" => {
                if str_tokens.len() == 5 {
                    let transform_name = parse_string_literal(str_tokens[1])?;
                    let input_qubits_str = str_tokens[2];
                    let output_qubits_str = str_tokens[3];
                    let mode_count = parse_usize(str_tokens[4])?;

                    let input_qubits = input_qubits_str.split(',').map(parse_u8).collect::<Result<
                        Vec<_>,
                        _,
                    >>(
                    )?;
                    let output_qubits = output_qubits_str
                        .split(',')
                        .map(parse_u8)
                        .collect::<Result<Vec<_>, _>>()?;

                    Ok(APPLYLINEAROPTICALTRANSFORM(
                        transform_name,
                        input_qubits,
                        output_qubits,
                        mode_count,
                    ))
                } else {
                    Err("usage: applylinearopticaltransform <name> <in_q1,in_q2...> <out_q1,out_q2...> <num_modes>".into())
                }
            }
            "PHOTONNUMBERRESOLVINGDETECTION" => {
                if str_tokens.len() == 3 {
                    Ok(PHOTONNUMBERRESOLVINGDETECTION(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("photonnumberresolvingdetection <qubit> <result_reg>".into())
                }
            }
            "FEEDBACKCONTROL" => {
                if str_tokens.len() == 3 {
                    Ok(FEEDBACKCONTROL(
                        parse_u8(str_tokens[1])?,
                        parse_string_literal(str_tokens[2])?,
                    ))
                } else {
                    Err("feedbackcontrol <qubit> <classical_control_reg>".into())
                }
            }
            // miscellaneous operations
            "SETPOS" => {
                if str_tokens.len() == 4 {
                    Ok(SETPOS(
                        parse_u8(str_tokens[1])?,
                        parse_f64(str_tokens[2])?,
                        parse_f64(str_tokens[3])?,
                    ))
                } else {
                    Err("setpos <reg> <x> <y>".into())
                }
            }
            "SETWL" => {
                if str_tokens.len() == 3 {
                    Ok(SETWL(parse_u8(str_tokens[1])?, parse_f64(str_tokens[2])?))
                } else {
                    Err("setwl <reg> <wavelength>".into())
                }
            }
            "WLSHIFT" => {
                if str_tokens.len() == 3 {
                    Ok(WLSHIFT(parse_u8(str_tokens[1])?, parse_f64(str_tokens[2])?))
                } else {
                    Err("wlshift <reg> <delta_wavelength>".into())
                }
            }
            "MOVE" => {
                if str_tokens.len() == 4 {
                    Ok(MOVE(
                        parse_u8(str_tokens[1])?,
                        parse_f64(str_tokens[2])?,
                        parse_f64(str_tokens[3])?,
                    ))
                } else {
                    Err("move <reg> <dx> <dy>".into())
                }
            }
            "COMMENT" => {
                if str_tokens.len() >= 2 {
                    Ok(COMMENT(str_tokens[1..].join(" ")))
                } else {
                    Err("comment <text>".into())
                }
            }
            "MARKOBSERVED" => {
                if str_tokens.len() == 2 {
                    Ok(MARKOBSERVED(parse_u8(str_tokens[1])?))
                } else {
                    Err("markobserved <reg>".into())
                }
            }
            "RELEASE" => {
                if str_tokens.len() == 2 {
                    Ok(RELEASE(parse_u8(str_tokens[1])?))
                } else {
                    Err("release <reg>".into())
                }
            }
            "HALT" => {
                if str_tokens.len() == 1 {
                    Ok(HALT)
                } else {
                    Err("halt".into())
                }
            }

            // new instruction parsing for v0.3.0+
            "JMPABS" => {
                if str_tokens.len() == 2 {
                    Ok(JMPABS(parse_u64(str_tokens[1])?))
                } else {
                    Err("jmpabs <address>".into())
                }
            }
            "IFGT" => {
                if str_tokens.len() == 4 {
                    Ok(IFGT(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_i64(str_tokens[3])?,
                    ))
                } else {
                    Err("ifgt <reg1> <reg2> <offset>".into())
                }
            }
            "IFLT" => {
                if str_tokens.len() == 4 {
                    Ok(IFLT(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_i64(str_tokens[3])?,
                    ))
                } else {
                    Err("iflt <reg1> <reg2> <offset>".into())
                }
            }
            "IFEQ" => {
                if str_tokens.len() == 4 {
                    Ok(IFEQ(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_i64(str_tokens[3])?,
                    ))
                } else {
                    Err("ifeq <reg1> <reg2> <offset>".into())
                }
            }
            "IFNE" => {
                if str_tokens.len() == 4 {
                    Ok(IFNE(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_i64(str_tokens[3])?,
                    ))
                } else {
                    Err("ifne <reg1> <reg2> <offset>".into())
                }
            }
            "CALLADDR" => {
                if str_tokens.len() == 2 {
                    Ok(CALLADDR(parse_u64(str_tokens[1])?))
                } else {
                    Err("calladdr <address>".into())
                }
            }
            "RETSUB" => {
                if str_tokens.len() == 1 {
                    Ok(RETSUB)
                } else {
                    Err("retsub".into())
                }
            }
            "PRINTF" => {
                if str_tokens.len() >= 2 {
                    let format_str = parse_string_literal(str_tokens[1])?;
                    let regs = parse_reg_list(&str_tokens[2..])?;
                    Ok(PRINTF(format_str, regs))
                } else {
                    Err("printf <format_string> [reg1] [reg2] ...".into())
                }
            }
            "PRINT" => {
                if str_tokens.len() == 2 {
                    Ok(PRINT(parse_string_literal(str_tokens[1])?))
                } else {
                    Err("print <string>".into())
                }
            }
            "PRINTLN" => {
                if str_tokens.len() == 2 {
                    Ok(PRINTLN(parse_string_literal(str_tokens[1])?))
                } else {
                    Err("println <string>".into())
                }
            }
            "INPUT" => {
                if str_tokens.len() == 2 {
                    Ok(INPUT(parse_u8(str_tokens[1])?))
                } else {
                    Err("input <reg>".into())
                }
            }
            "DUMPSTATE" => {
                if str_tokens.len() == 1 {
                    Ok(DUMPSTATE)
                } else {
                    Err("dumpstate".into())
                }
            }
            "DUMPREGS" => {
                if str_tokens.len() == 1 {
                    Ok(DUMPREGS)
                } else {
                    Err("dumpregs".into())
                }
            }
            "LOADREGMEM" => {
                if str_tokens.len() == 3 {
                    Ok(LOADREGMEM(
                        parse_u8(str_tokens[1])?,
                        parse_u64(str_tokens[2])?,
                    ))
                } else {
                    Err("loadregmem <reg> <address>".into())
                }
            }
            "STOREMEMREG" => {
                if str_tokens.len() == 3 {
                    Ok(STOREMEMREG(
                        parse_u64(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                    ))
                } else {
                    Err("storememreg <address> <reg>".into())
                }
            }
            "PUSHREG" => {
                if str_tokens.len() == 2 {
                    Ok(PUSHREG(parse_u8(str_tokens[1])?))
                } else {
                    Err("pushreg <reg>".into())
                }
            }
            "POPREG" => {
                if str_tokens.len() == 2 {
                    Ok(POPREG(parse_u8(str_tokens[1])?))
                } else {
                    Err("popreg <reg>".into())
                }
            }
            "ALLOC" => {
                if str_tokens.len() == 3 {
                    Ok(ALLOC(parse_u8(str_tokens[1])?, parse_u64(str_tokens[2])?))
                } else {
                    Err("alloc <reg_addr> <size>".into())
                }
            }
            "FREE" => {
                if str_tokens.len() == 2 {
                    Ok(FREE(parse_u64(str_tokens[1])?))
                } else {
                    Err("free <address>".into())
                }
            }
            "CMP" => {
                if str_tokens.len() == 3 {
                    Ok(CMP(parse_u8(str_tokens[1])?, parse_u8(str_tokens[2])?))
                } else {
                    Err("cmp <reg1> <reg2>".into())
                }
            }
            "ANDBITS" => {
                if str_tokens.len() == 4 {
                    Ok(ANDBITS(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_u8(str_tokens[3])?,
                    ))
                } else {
                    Err("andbits <dest> <op1> <op2>".into())
                }
            }
            "ORBITS" => {
                if str_tokens.len() == 4 {
                    Ok(ORBITS(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_u8(str_tokens[3])?,
                    ))
                } else {
                    Err("orbits <dest> <op1> <op2>".into())
                }
            }
            "XORBITS" => {
                if str_tokens.len() == 4 {
                    Ok(XORBITS(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_u8(str_tokens[3])?,
                    ))
                } else {
                    Err("xorbits <dest> <op1> <op2>".into())
                }
            }
            "NOTBITS" => {
                if str_tokens.len() == 3 {
                    Ok(NOTBITS(parse_u8(str_tokens[1])?, parse_u8(str_tokens[2])?))
                } else {
                    Err("notbits <dest> <op>".into())
                }
            }
            "SHL" => {
                if str_tokens.len() == 4 {
                    Ok(SHL(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_u8(str_tokens[3])?,
                    ))
                } else {
                    Err("shl <dest> <op> <amount_reg>".into())
                }
            }
            "SHR" => {
                if str_tokens.len() == 4 {
                    Ok(SHR(
                        parse_u8(str_tokens[1])?,
                        parse_u8(str_tokens[2])?,
                        parse_u8(str_tokens[3])?,
                    ))
                } else {
                    Err("shr <dest> <op> <amount_reg>".into())
                }
            }
            "BREAKPOINT" => {
                if str_tokens.len() == 1 {
                    Ok(BREAKPOINT)
                } else {
                    Err("breakpoint".into())
                }
            }
            "GETTIME" => {
                if str_tokens.len() == 2 {
                    Ok(GETTIME(parse_u8(str_tokens[1])?))
                } else {
                    Err("gettime <reg>".into())
                }
            }
            "SEEDRNG" => {
                if str_tokens.len() == 2 {
                    Ok(SEEDRNG(parse_u64(str_tokens[1])?))
                } else {
                    Err("seedrng <seed>".into())
                }
            }
            "EXITCODE" => {
                if str_tokens.len() == 2 {
                    Ok(EXITCODE(parse_i32(str_tokens[1])?))
                } else {
                    Err("exitcode <code>".into())
                }
            }
            _ => Err(format!("unknown instruction '{}'", op)),
        }
    }
}
