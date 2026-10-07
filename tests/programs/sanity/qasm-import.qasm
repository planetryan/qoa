// openqasm 2.0 osoa: gate definizioak, parametroak, erregistro-hedapena eta if
// expect-only: 111
OPENQASM 2.0;
include "qelib1.inc";
gate nire_x a { u3(pi,0,pi) a; }
gate bikoitza(theta) a, b { rx(theta) a; cx a, b; }
qreg q[3];
creg c[3];
nire_x q[0];
bikoitza(pi) q[1], q[2];
measure q -> c;
if (c == 7) barrier q;
