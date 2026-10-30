// qoa liburutegia: mihiztatzailea, simulagailua eta inportazio/esportazioa.
// rust egonkorrean konpilatzen da, nightly ezaugarririk gabe.

// QOA 64 bitekoa da bakarrik: fisika 64 biteko koma higikorrean kalkulatzen du eta egoera handiek 64 biteko
// helbide-espazioa behar dute (2^32 anplitude = 64 GiB)
#[cfg(not(target_pointer_width = "64"))]
compile_error!(
    "QOA is 64-bit only: build it for a 64-bit target (x86-64, ARM64, RISC-V 64, POWER, IBM Z, LoongArch). / QOA 64 bitekoa da bakarrik: konpilatu 64 biteko helburu baterako (x86-64, ARM64, RISC-V 64, POWER, IBM Z, LoongArch)."
);

pub mod asm; // .qoa testua -> programa
pub mod dd; // batura zehatzak double-double aritmetikarekin (~106 bit): araua, probabilitateak
pub mod disasm; // desmihiztatzea eta analisirako esportazioak (qoa, listing, json, dot, glyph)
pub mod expr; // zenbakizko adierazpenak (pi/2, ...)
pub mod fock; // eragile optikoak fotoi-kopuruaren oinarrian
pub mod glyph; // glifo-sintaxia (.qg): QOA trinkoa, lerro bakoitzean agindu askorekin
pub mod gpu; // qubit-ateak GPUan: planifikazioa eta emulazioa
pub mod ionq; // ionq json esportazioa
pub mod isa; // agindu-multzoaren taula
pub mod kernel; // qubit-ateak: lasterketak, fusioa, cache-blokeak
pub mod lang; // irteeraren hizkuntza (euskara/ingelesa)
pub mod light; // audioa argi gisa zirkuitu optiko kuantiko batetik (`qoa visual`): mdct, kanalak, detekzioa
pub mod noise; // gailuen zarata-ereduak (kalibrazioak): ate-erroreak, T1/T2, irakurketa
pub mod program; // programa mihiztatua eta .qx formatua
pub mod qasm; // openqasm inportazioa eta esportazioa
pub mod report; // emaitzen txostena (testua eta json)
pub mod simd; // kernel beroak mihiztaduran: avx-512, avx2, neon, rvv
pub mod state; // egoera-bektorea: qubitak eta modu optikoak
pub mod vm; // makina birtuala eta shot-ak
#[cfg(feature = "vulkan")]
pub mod vulkan; // vulkan gailua (src/shaders/tile.comp)
