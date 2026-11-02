// openqasm 3-ren blokeak eta ate-modifikatzaileak: arrakasta arteko begizta, begizta finkoa, baldintzak eta aldatzaileak
// expect-only: 1011
OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
bit[4] c;
reset q[0];
ry(pi/3) q[0];
c[0] = measure q[0];
while (c[0] == false) {
  reset q[0];
  ry(pi/3) q[0];
  c[0] = measure q[0];
}
for uint i in [0:2] {
  x q[1];
}
if (c[0]) {
  ctrl @ x q[0], q[3];
} else {
  x q[2];
}
pow(2) @ s q[2];
inv @ ctrl(2) @ x q[0], q[1], q[2];
ctrl(2) @ x q[0], q[1], q[2];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
