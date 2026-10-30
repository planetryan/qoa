OPENQASM 2.0;
// examples/bell.qoa ibm torino gailurako transpilatua (tools/run_on_ibm.py --dry-run --save-model examples/bell-torino):
// gailuaren ate natiboak bakarrik (rz, sx, cz), eta q0, q1 gailuko 92 eta 99 qubitak dira. haren zarata-eredua
// bell-torino.json-en dago (fake_torino-ren kalibrazioa, qiskit-ibm-runtime 0.51):
//
//   qoa run examples/bell-torino.qasm --noise-model examples/bell-torino.json
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi) q[1];
cz q[0],q[1];
sx q[1];
rz(pi/2) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
