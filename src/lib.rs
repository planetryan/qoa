// qoa liburutegia: mihiztatzailea, simulagailua eta inportazio/esportazioa.
// rust egonkorrean konpilatzen da, nightly ezaugarririk gabe.

pub mod asm; // .qoa testua -> programa
pub mod expr; // zenbakizko adierazpenak (pi/2, ...)
pub mod fock; // eragile optikoak fotoi-kopuruaren oinarrian
pub mod ionq; // ionq json esportazioa
pub mod isa; // agindu-multzoaren taula
pub mod lang; // irteeraren hizkuntza (euskara/ingelesa)
pub mod program; // programa mihiztatua eta .qx formatua
pub mod qasm; // openqasm inportazioa eta esportazioa
pub mod report; // emaitzen txostena (testua eta json)
pub mod state; // egoera-bektorea: qubitak eta modu optikoak
pub mod vm; // makina birtuala eta shot-ak
