// bell egoera openqasm 2.0-n: `qoa run examples/bell.qasm` zuzenean exekutatzen du
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
