// fitxategi honek openqasm inportazioa eta esportazioa egiaztatzen ditu.
//
// proba nagusia: qoa-k berez mapatzen duen qelib1.inc-ko ate bakoitza openqasm 2.0-ren espezifikazioko
// definizio ofizialarekin konparatzen da (U eta CX-ekin eraikia, `gate` gisa inportatua eta zabaldua).
// bi bide independenteek egoera bera eman behar dute (fase global batera arte, openqasm 2.0-k ezin
// baitu fase globala ikusi).

use num_complex::Complex64 as C;
use qoa::lang::{self, Lang};
use qoa::vm::{self, RunConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

fn state_of_qoa(src: &str) -> Vec<C> {
    lang::set(Lang::En);
    let prog = qoa::asm::assemble(src, "t.qoa")
        .unwrap_or_else(|d| panic!("{}", qoa::asm::render(&d, src, "t.qoa")));
    let out = vm::run(
        &prog,
        &RunConfig {
            shots: 1,
            seed: 1,
            ..RunConfig::default()
        },
        &mut std::io::sink(),
    )
    .unwrap();
    out.final_state.amplitudes().to_vec()
}

fn state_of_qasm(src: &str) -> Vec<C> {
    lang::set(Lang::En);
    let qoa_src = qoa::qasm::to_qoa(src, "t.qasm")
        .unwrap_or_else(|e| panic!("import failed: {}\n{}", e, src));
    state_of_qoa(&qoa_src)
}

// fideltasuna |<a|b>|^2: 1 da bi egoerak fase global batera arte berdinak badira
fn fidelity(a: &[C], b: &[C]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| x.conj() * y)
        .sum::<C>()
        .norm_sqr()
}

// openqasm 2.0-ren qelib1.inc jatorrizkoko definizioak (U eta CX-ekin), `def_` aurrizkiarekin
const QELIB1_DEFINITIONS: &str = r#"
gate def_u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
gate def_u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
gate def_u1(lambda) q { U(0,0,lambda) q; }
gate def_cx c,t { CX c,t; }
gate def_x a { def_u3(pi,0,pi) a; }
gate def_y a { def_u3(pi,pi/2,pi/2) a; }
gate def_z a { def_u1(pi) a; }
gate def_h a { def_u2(0,pi) a; }
gate def_s a { def_u1(pi/2) a; }
gate def_sdg a { def_u1(-pi/2) a; }
gate def_t a { def_u1(pi/4) a; }
gate def_tdg a { def_u1(-pi/4) a; }
gate def_rx(theta) a { def_u3(theta,-pi/2,pi/2) a; }
gate def_ry(theta) a { def_u3(theta,0,0) a; }
gate def_rz(phi) a { def_u1(phi) a; }
gate def_cz a,b { def_h b; def_cx a,b; def_h b; }
gate def_cy a,b { def_sdg b; def_cx a,b; def_s b; }
gate def_swap a,b { def_cx a,b; def_cx b,a; def_cx a,b; }
gate def_ch a,b { def_h b; def_sdg b; def_cx a,b; def_h b; def_t b; def_cx a,b; def_t b; def_h b; def_s b; def_x b; def_s a; }
gate def_ccx a,b,c { def_h c; def_cx b,c; def_tdg c; def_cx a,c; def_t c; def_cx b,c; def_tdg c; def_cx a,c; def_t b; def_t c; def_h c; def_cx a,b; def_t a; def_tdg b; def_cx a,b; }
gate def_crz(lambda) a,b { def_u1(lambda/2) b; def_cx a,b; def_u1(-lambda/2) b; def_cx a,b; }
gate def_cu1(lambda) a,b { def_u1(lambda/2) a; def_cx a,b; def_u1(-lambda/2) b; def_cx a,b; def_u1(lambda/2) b; }
gate def_cu3(theta,phi,lambda) c,t { def_u1((lambda+phi)/2) c; def_u1((lambda-phi)/2) t; def_cx c,t; def_u3(-theta/2,0,-(phi+lambda)/2) t; def_cx c,t; def_u3(theta/2,phi,0) t; }
"#;

// (izena, parametroak, qubitak)
const GATES: &[(&str, usize, usize)] = &[
    ("u3", 3, 1),
    ("u2", 2, 1),
    ("u1", 1, 1),
    ("x", 0, 1),
    ("y", 0, 1),
    ("z", 0, 1),
    ("h", 0, 1),
    ("s", 0, 1),
    ("sdg", 0, 1),
    ("t", 0, 1),
    ("tdg", 0, 1),
    ("rx", 1, 1),
    ("ry", 1, 1),
    ("rz", 1, 1),
    ("cx", 0, 2),
    ("cz", 0, 2),
    ("cy", 0, 2),
    ("swap", 0, 2),
    ("ch", 0, 2),
    ("ccx", 0, 3),
    ("crz", 1, 2),
    ("cu1", 1, 2),
    ("cu3", 3, 2),
];

// egoera generiko bat sortzeko aurrizkia: u3 ausazkoak qubit guztietan, eta entrelazamendu pixka bat
fn scramble(rng: &mut StdRng, n: usize) -> String {
    let mut s = String::new();
    for q in 0..n {
        s.push_str(&format!(
            "U({},{},{}) q[{}];\n",
            rng.random_range(-PI..PI),
            rng.random_range(-PI..PI),
            rng.random_range(-PI..PI),
            q
        ));
    }
    for q in 0..n - 1 {
        s.push_str(&format!("CX q[{}],q[{}];\n", q, q + 1));
    }
    s
}

#[test]
fn every_native_qelib1_gate_matches_its_official_definition() {
    let mut rng = StdRng::seed_from_u64(11);
    let n = 4;
    for &(name, np, nq) in GATES {
        for _ in 0..3 {
            let prefix = scramble(&mut rng, n);
            let params: Vec<String> = (0..np)
                .map(|_| format!("{}", rng.random_range(-PI..PI)))
                .collect();
            let mut qs: Vec<usize> = (0..n).collect();
            for i in 0..nq {
                let j = rng.random_range(i..n);
                qs.swap(i, j);
            }
            let args: Vec<String> = qs[..nq].iter().map(|q| format!("q[{}]", q)).collect();
            let call = |gname: &str| {
                if np == 0 {
                    format!("{} {};", gname, args.join(","))
                } else {
                    format!("{}({}) {};", gname, params.join(","), args.join(","))
                }
            };
            let header = format!(
                "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n{}\nqreg q[{}];\n",
                QELIB1_DEFINITIONS, n
            );
            let native = format!("{}{}{}\n", header, prefix, call(name));
            let defined = format!("{}{}{}\n", header, prefix, call(&format!("def_{}", name)));
            let f = fidelity(&state_of_qasm(&native), &state_of_qasm(&defined));
            assert!(
                (f - 1.0).abs() < 1e-10,
                "{}: native and official definition differ (fidelity {})",
                name,
                f
            );
        }
    }
}

// qiskit-en qelib1.inc zabalduko ateak, matrize zehatzen aurka (fase barne)
#[test]
fn extended_gates_match_their_matrices() {
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let theta = 0.73;
    // |+0> egoeratik abiatuta (q0 = |+>, q1 = |0>)
    let start = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\n";
    let check = |gate: &str, expected: [C; 4]| {
        let got = state_of_qasm(&format!("{}{}\n", start, gate));
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).norm() < 1e-12,
                "{}: amplitude {} is {} but expected {}",
                gate,
                i,
                g,
                e
            );
        }
    };
    let z = C::new(0.0, 0.0);
    // rzz(θ) = exp(-iθ/2 Z⊗Z): |00> -> e^{-iθ/2}, |01> (q0=1) -> e^{iθ/2}
    check(
        &format!("rzz({}) q[0],q[1];", theta),
        [
            C::from_polar(r, -theta / 2.0),
            C::from_polar(r, theta / 2.0),
            z,
            z,
        ],
    );
    // rxx(θ) |q1 q0> = |0+>: X⊗X-k |0+> -> |1+>; exp(-iθ/2 XX) = cos - i sin XX
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    check(
        &format!("rxx({}) q[0],q[1];", theta),
        [
            C::new(r * c, 0.0),
            C::new(r * c, 0.0),
            C::new(0.0, -r * s),
            C::new(0.0, -r * s),
        ],
    );
    // csx: q0 = |+> kontrola, q1 helburua: |1>_q0 adarrean SX|0> = (1+i)/2 |0> + (1-i)/2 |1>
    check(
        "csx q[0],q[1];",
        [
            C::new(r, 0.0),
            C::new(0.5 * r, 0.5 * r),
            z,
            C::new(0.5 * r, -0.5 * r),
        ],
    );
    // cp(θ): fasea |11>-n bakarrik; hemen q1 = 0, beraz ezer ez
    check(
        &format!("cp({}) q[0],q[1];", theta),
        [C::new(r, 0.0), C::new(r, 0.0), z, z],
    );
    // cu(θ, φ, λ, γ): e^{iγ} U kontrolatua; U(θ,φ,λ)|0> = cos(θ/2)|0> + e^{iφ} sin(θ/2)|1>
    let (t, ph, la, ga) = (0.9, 0.4, -0.2, 0.3);
    let (ct, st) = ((t / 2.0f64).cos(), (t / 2.0f64).sin());
    let _ = la;
    check(
        &format!("cu({},{},{},{}) q[0],q[1];", t, ph, la, ga),
        [
            C::new(r, 0.0),
            C::from_polar(r * ct, ga),
            z,
            C::from_polar(r * st, ga + ph),
        ],
    );
    // sxdg · sx = identitatea
    check(
        "sx q[1];\nsxdg q[1];",
        [C::new(r, 0.0), C::new(r, 0.0), z, z],
    );
}

#[test]
fn multi_controlled_gates_flip_only_when_all_controls_are_one() {
    // c3x eta c4x oinarri-egoera guztietan
    for (gate, n) in [("c3x", 4usize), ("c4x", 5usize)] {
        for input in 0..(1usize << n) {
            let mut src = format!("OPENQASM 2.0;\nqreg q[{}];\n", n);
            for q in 0..n {
                if input >> q & 1 == 1 {
                    src.push_str(&format!("x q[{}];\n", q));
                }
            }
            let args: Vec<String> = (0..n).map(|q| format!("q[{}]", q)).collect();
            src.push_str(&format!("{} {};\n", gate, args.join(",")));
            let st = state_of_qasm(&src);
            let controls = (1usize << (n - 1)) - 1;
            let expected = if input & controls == controls {
                input ^ (1 << (n - 1))
            } else {
                input
            };
            assert!(
                (st[expected].norm() - 1.0).abs() < 1e-12,
                "{} on {:0w$b}",
                gate,
                input,
                w = n
            );
        }
    }
    // rccx: toffoli fase erlatibo batera arte (probabilitateak berdinak)
    for input in 0..8usize {
        let mut src = String::from("OPENQASM 2.0;\nqreg q[3];\n");
        for q in 0..3 {
            if input >> q & 1 == 1 {
                src.push_str(&format!("x q[{}];\n", q));
            }
        }
        src.push_str("rccx q[0],q[1],q[2];\n");
        let st = state_of_qasm(&src);
        let expected = if input & 3 == 3 { input ^ 4 } else { input };
        assert!(
            (st[expected].norm() - 1.0).abs() < 1e-12,
            "rccx on {:03b}",
            input
        );
    }
}

#[test]
fn import_handles_the_whole_language() {
    let src = r#"
        // iruzkinak, /* blokeak */ eta erregistro anitz
        OPENQASM 2.0;
        include "qelib1.inc";
        /* ate habiaratuak parametro-adierazpenekin */
        gate rot2(a, b) x, y { ry(a/2) x; ry(b*2) y; cx x, y; }
        gate outer(t) x, y { rot2(t, t/4) x, y; barrier x, y; }
        qreg a[2];
        qreg b[1];
        creg c[3];
        x b[0];
        outer(pi) a[0], a[1];
        h a;                         // erregistro osoa hedatuta
        h a;
        measure a -> c[0:1];
    "#;
    // c[0:1] ez da openqasm 2.0: errore argia eman behar du
    let err = qoa::qasm::to_qoa(src, "t.qasm").unwrap_err();
    assert!(
        err.message.contains("bad index") || err.message.contains("missing ]"),
        "{}",
        err
    );

    let ok = src
        .replace(
            "measure a -> c[0:1];",
            "measure a -> c[0];\nmeasure b[0] -> c[2];",
        )
        .replace(
            "measure a -> c[0];",
            "measure a[0] -> c[0];\nmeasure a[1] -> c[1];",
        );
    let qoa_src = qoa::qasm::to_qoa(&ok, "t.qasm").unwrap();
    assert!(qoa_src.contains("QINIT 3"), "{}", qoa_src);
    assert!(qoa_src.contains("BARRIER q0 q1"), "{}", qoa_src);
    assert!(qoa_src.contains("MEASURE q2 -> c2"), "{}", qoa_src);
    // outer(pi) = ry(pi/2) a0; ry(pi/2) a1; cx a0 a1 -> egoera ezaguna; b0 = 1
    let st = state_of_qasm(&ok.replace(
        "measure a[0] -> c[0];\nmeasure a[1] -> c[1];\nmeasure b[0] -> c[2];",
        "",
    ));
    let mut expected = state_of_qoa("QINIT 3\nX q2\nRY q0 pi/2\nRY q1 pi/2\nCNOT q0 q1\n");
    // h a; h a = identitatea
    let f = fidelity(&st, &expected);
    assert!((f - 1.0).abs() < 1e-12, "fidelity {}", f);
    expected.clear();
}

#[test]
fn if_statements_branch_on_classical_registers() {
    // c = 1 denean bakarrik x q[1]; q[0] beti 1, beraz q[1] ere bai
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\nif (c == 1) x q[1];\nif (c == 0) h q[1];\nmeasure q[1] -> c[0];\n";
    let qoa_src = qoa::qasm::to_qoa(src, "t.qasm").unwrap();
    let prog = qoa::asm::assemble(&qoa_src, "t").unwrap();
    let out = vm::run(
        &prog,
        &RunConfig {
            shots: 200,
            seed: 1,
            ..RunConfig::default()
        },
        &mut std::io::sink(),
    )
    .unwrap();
    assert_eq!(out.counts.len(), 1, "{:?}", out.counts);
    assert_eq!(out.counts[0].0.bits[0], Some(1));
}

#[test]
fn import_errors_say_what_is_wrong() {
    lang::set(Lang::En);
    let cases = [
        (
            "OPENQASM 2.0;\nqreg q[1];\nfoo q[0];\n",
            "unknown gate 'foo'",
        ),
        (
            "OPENQASM 2.0;\nqreg q[1];\nx r[0];\n",
            "unknown register 'r'",
        ),
        (
            "OPENQASM 2.0;\nqreg q[1];\nx q[3];\n",
            "q[3] is out of range",
        ),
        ("OPENQASM 2.0;\nqreg q[1];\nx q[0]\n", "missing its ;"),
        (
            "OPENQASM 2.0;\nopaque magic a;\nqreg q[1];\nmagic q[0];\n",
            "declared opaque",
        ),
        (
            "OPENQASM 2.0;\nqreg q[2];\ncx q[0], q[0];\n",
            "uses qubit 0 twice",
        ),
        (
            "OPENQASM 2.0;\nqreg q[1];\nrx q[0];\n",
            "rx takes 1 parameter(s), found 0",
        ),
        (
            "OPENQASM 2.0;\ngate g a { g a; }\nqreg q[1];\ng q[0];\n",
            "more than 64 levels deep",
        ),
        (
            "OPENQASM 3;\nqubit[1] q;\nctrl @ x q[0], q[0];\n",
            "cx uses qubit 0 twice",
        ),
        (
            "OPENQASM 3;\nqubit[2] q;\nnegctrl @ x q[0], q[1];\n",
            "gate modifier 'negctrl' is not supported",
        ),
        (
            "OPENQASM 3;\nqubit[1] q;\nctrl @ s q[0], q[1];\n",
            "no controlled form of 's'",
        ),
        (
            "OPENQASM 3;\nqubit[1] q;\npow(0.5) @ x q[0];\n",
            "only whole-number powers",
        ),
        (
            "OPENQASM 3;\nqubit[1] q;\nfor uint i in [0:3] { rx(i) q[0]; }\n",
            "loop variable 'i' is used inside the loop",
        ),
        (
            "OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nelse { x q[0]; }\n",
            "else without an if",
        ),
        (
            "OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nif (c[0] == maybe) { x q[0]; }\n",
            "found 'maybe'",
        ),
        (
            "OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nwhile (c[0]) {\n  x q[0];\n  y q[7];\n}\n",
            "line 6",
        ),
        (
            "OPENQASM 3;\nqubit[1] q;\nuint[4] n;\n",
            "'uint' is an OpenQASM 3 feature",
        ),
        (
            "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nif (c > 1) x q[0];\n",
            "only == conditions",
        ),
    ];
    for (src, want) in cases {
        let err = qoa::qasm::to_qoa(src, "t.qasm").unwrap_err();
        assert!(
            err.to_string().contains(want),
            "expected {:?}, got {}",
            want,
            err
        );
    }
}

#[test]
fn openqasm_3_basics_import() {
    let src = "OPENQASM 3;\ninclude \"stdgates.inc\";\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\nc[0] = measure q[0];\nc[1] = measure q[1];\n";
    let qoa_src = qoa::qasm::to_qoa(src, "t.qasm").unwrap();
    assert!(qoa_src.contains("MEASURE q1 -> c1"), "{}", qoa_src);
}

// ausazko qoa programa -> openqasm -> qoa: egoera bera
#[test]
fn export_then_import_round_trips() {
    let mut rng = StdRng::seed_from_u64(5);
    let ops = [
        "H q{a}",
        "X q{a}",
        "Y q{a}",
        "Z q{a}",
        "S q{a}",
        "SDG q{a}",
        "T q{a}",
        "TDG q{a}",
        "SX q{a}",
        "SXDG q{a}",
        "RX q{a} {p}",
        "RY q{a} {p}",
        "RZ q{a} {p}",
        "PHASE q{a} {p}",
        "U q{a} {p} {p} {p}",
        "CNOT q{a} q{b}",
        "CY q{a} q{b}",
        "CZ q{a} q{b}",
        "CH q{a} q{b}",
        "CSX q{a} q{b}",
        "CU q{a} q{b} {p} {p} {p}",
        "CPHASE q{a} q{b} {p}",
        "CRX q{a} q{b} {p}",
        "CRY q{a} q{b} {p}",
        "CRZ q{a} q{b} {p}",
        "SWAP q{a} q{b}",
        "CSWAP q{a} q{b} q{c}",
        "CCX q{a} q{b} q{c}",
        "MCZ q{a} q{b} q{c}",
        "BELL q{a} q{b}",
        "GHZ q{a} q{b} q{c}",
        "CLUSTER q{a} q{b} q{c}",
        "TOBASIS q{a} Y",
    ];
    for round in 0..20 {
        let n = 5;
        let mut src = format!("QINIT {}\n", n);
        for _ in 0..40 {
            let mut qs: Vec<usize> = (0..n).collect();
            for i in 0..3 {
                let j = rng.random_range(i..n);
                qs.swap(i, j);
            }
            let mut line = ops[rng.random_range(0..ops.len())]
                .replace("{a}", &qs[0].to_string())
                .replace("{b}", &qs[1].to_string())
                .replace("{c}", &qs[2].to_string());
            while line.contains("{p}") {
                line = line.replacen("{p}", &format!("{}", rng.random_range(-PI..PI)), 1);
            }
            src.push_str(&line);
            src.push('\n');
        }
        let prog = qoa::asm::assemble(&src, "t").unwrap();
        let qasm = qoa::qasm::from_program(&prog).unwrap_or_else(|e| panic!("{:?}\n{}", e, src));
        let f = fidelity(&state_of_qoa(&src), &state_of_qasm(&qasm));
        assert!(
            (f - 1.0).abs() < 1e-10,
            "round {}: fidelity {} for\n{}\n---\n{}",
            round,
            f,
            src,
            qasm
        );
    }
}

#[test]
fn export_refuses_what_openqasm_cannot_say() {
    lang::set(Lang::En);
    let cases = [
        ("QMODE 1\nDISPLACE o0 1\n", "OpenQASM has no optical modes"),
        // kopuru literaleko begiztak zabaltzen dira, baina erregistro bateko kopurua ez
        (
            "QINIT 1\nREGSET R1 2\nLOOPSTART R1\nH q0\nLOOPEND\n",
            "control flow",
        ),
        // atzerako jauzia (errepikatu arrakasta lortu arte) ez da OpenQASM 2.0
        (
            "QINIT 1\nberriz:\nH q0\nMEASURE q0\nJUMPIFZERO c0 berriz\n",
            "control flow",
        ),
        ("QINIT 1\nMEASURE q0 -> R0\n", "measurement into a register"),
        ("QINIT 6\nMCZ q0 q1 q2 q3 q4 q5\n", "phase on 6 qubits"),
    ];
    for (src, want) in cases {
        let prog = qoa::asm::assemble(src, "t").unwrap();
        let errs = qoa::qasm::from_program(&prog).unwrap_err();
        let text: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
        assert!(
            text.iter().any(|t| t.contains(want)),
            "expected {:?} in {:?}",
            want,
            text
        );
    }
}

#[test]
fn ionq_export_is_valid_json_in_the_v0_format() {
    let prog = qoa::asm::assemble(
        "QINIT 3\nH q0\nCNOT q0 q1\nCCX q0 q1 q2\nRZ q2 0.5\nSDG q1\nU q0 1 2 3\nMEASUREALL\n",
        "t",
    )
    .unwrap();
    let text = qoa::ionq::from_program(&prog, 100).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["input"]["format"], "ionq.circuit.v0");
    assert_eq!(v["input"]["qubits"], 3);
    assert_eq!(v["shots"], 100);
    let gates: Vec<&str> = v["input"]["circuit"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["gate"].as_str().unwrap())
        .collect();
    assert_eq!(gates, vec!["h", "x", "x", "rz", "si", "rz", "ry", "rz"]);
    assert_eq!(v["input"]["circuit"][1]["controls"], serde_json::json!([0]));
    assert_eq!(
        v["input"]["circuit"][2]["controls"],
        serde_json::json!([0, 1])
    );
    // modu optikoak eta neurketaren ondorengo ateak ez
    let photonic = qoa::asm::assemble("QMODE 1\nDISPLACE o0 1\n", "t").unwrap();
    assert!(qoa::ionq::from_program(&photonic, 1).is_err());
    let after = qoa::asm::assemble("QINIT 1\nMEASURE q0\nH q0\n", "t").unwrap();
    assert!(qoa::ionq::from_program(&after, 1).is_err());
}

#[test]
fn export_unrolls_loops_and_turns_skips_into_if() {
    // begizta literalak zabaltzen dira; bit baten araberako aurrerako jauzi bat `if` bihurtzen da, bit bakoitza
    // bere erregistroan
    let src = "QINIT 2\nLOOPSTART 3\nH q0\nLOOPEND\nMEASURE q0 -> c0\nJUMPIFZERO c0 gabe\nX q1\nCZ q0 q1\ngabe:\nMEASURE q1 -> c1\n";
    let prog = qoa::asm::assemble(src, "t").unwrap();
    let text = qoa::qasm::from_program(&prog).unwrap();
    assert_eq!(text.matches("h q[0];").count(), 3, "{}", text);
    assert!(
        text.contains("creg c0[1];") && text.contains("creg c1[1];"),
        "{}",
        text
    );
    assert!(
        text.contains("if(c0==1) x q[1];") && text.contains("if(c0==1) cz q[0],q[1];"),
        "{}",
        text
    );
    assert!(text.contains("measure q[1] -> c1[0];"), "{}", text);
}

#[test]
fn exported_control_flow_runs_the_same_after_import() {
    // QOA -> OpenQASM -> QOA: teleportazioak eta grover-ek kopuru berdin-berdinak ematen dituzte
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for name in [
        "examples/teleport.qoa",
        "examples/grover.qoa",
        "examples/grover-key.qoa",
    ] {
        let src = std::fs::read_to_string(root.join(name)).unwrap();
        let prog = qoa::asm::assemble(&src, name).unwrap();
        let text = qoa::qasm::from_program(&prog).unwrap();
        let back = qoa::asm::assemble(&qoa::qasm::to_qoa(&text, name).unwrap(), name).unwrap();
        let cfg = qoa::vm::RunConfig {
            shots: 4000,
            seed: 3,
            ..Default::default()
        };
        let a = qoa::vm::run(&prog, &cfg, &mut std::io::sink()).unwrap();
        let b = qoa::vm::run(&back, &cfg, &mut std::io::sink()).unwrap();
        // bit-erregistroak bereizita daude esportazioan; konparatu bitak
        let bits = |o: &qoa::vm::RunOutput| {
            let mut v: Vec<(String, u64)> = o
                .counts
                .iter()
                .map(|(r, n)| (qoa::report::bits_string(r).replace('-', "0"), *n))
                .collect();
            v.sort();
            v
        };
        let (ka, kb) = (bits(&a), bits(&b));
        // kopuru berdinak, bit-kopurua desberdina izan daitekeen arren (zeroak ezkerrean)
        let trim = |v: Vec<(String, u64)>| -> Vec<(String, u64)> {
            v.into_iter()
                .map(|(k, n)| (k.trim_start_matches('0').to_string(), n))
                .collect()
        };
        assert_eq!(trim(ka), trim(kb), "{}", name);
    }
}

#[test]
fn openqasm_3_export_keeps_loops_ifs_and_repeat_until_success() {
    lang::set(Lang::En);
    let v3 = |src: &str| {
        let prog = qoa::asm::assemble(src, "t")
            .unwrap_or_else(|d| panic!("{}", qoa::asm::render(&d, src, "t")));
        qoa::qasm::from_program_v3(&prog).unwrap_or_else(|e| panic!("{:?}\n{}", e, src))
    };
    // literal begiztak `for` gisa, zabaldu gabe
    let text = v3("QINIT 2\nLOOPSTART 3\nH q0\nLOOPSTART 2\nX q1\nLOOPEND\nLOOPEND\nMEASUREALL\n");
    assert!(
        text.starts_with("// exported")
            && text.contains("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nbit[2] c;\n"),
        "{}",
        text
    );
    assert!(
        text.contains(
            "for uint i0 in [0:2] {\n  h q[0];\n  for uint i1 in [0:1] {\n    x q[1];\n  }\n}\n"
        ),
        "{}",
        text
    );
    // arrakasta arte errepikatu: gorputza behin, gero `while`
    let text = v3(
        "QINIT 1\nberriz:\nRESET q0\nRY q0 1.0471975511965976\nMEASURE q0 -> c0\nJUMPIFZERO c0 berriz\n",
    );
    assert!(text.contains("reset q[0];\nry(1.0471975511965976) q[0];\nmeasure q[0] -> c[0];\nwhile (c[0] == false) {\n  reset q[0];\n  ry(1.0471975511965976) q[0];\n  measure q[0] -> c[0];\n}\n"), "{}", text);
    // aurrerako jauzia `if` bihurtzen da, eta JUMP batekin amaitutakoa if/else
    let text =
        v3("QINIT 2\nH q0\nMEASURE q0 -> c0\nJUMPIFZERO c0 gabe\nX q1\ngabe:\nMEASURE q1 -> c1\n");
    assert!(
        text.contains("if (c[0] == true) {\n  x q[1];\n}\n"),
        "{}",
        text
    );
    let text = v3(
        "QINIT 3\nH q0\nMEASURE q0 -> c0\nJUMPIFONE c0 bestela\nX q1\nJUMP eginda\nbestela:\nX q2\neginda:\nMEASUREALL\n",
    );
    assert!(
        text.contains("if (c[0] == false) {\n  x q[1];\n}\nelse {\n  x q[2];\n}\n"),
        "{}",
        text
    );
    // qelib1-eko izenak stdgates.inc-n ez daudenean, aldatzaileak edo baliokideak
    let text = v3(
        "QINIT 5\nSXDG q0\nCSX q0 q1\nCU1 q0 q1 0.5\nCU3 q0 q1 0.1 0.2 0.3\nMCX q0 q1 q2 q3\nMCX q0 q1 q2 q3 q4\n",
    );
    for want in [
        "inv @ sx q[0];",
        "ctrl @ sx q[0],q[1];",
        "cp(0.5) q[0],q[1];",
        "cu(0.1,0.2,0.3,0) q[0],q[1];",
        "ctrl(3) @ x q[0],q[1],q[2],q[3];",
        "ctrl(4) @ x q[0],q[1],q[2],q[3],q[4];",
    ] {
        assert!(text.contains(want), "{} not in\n{}", want, text);
    }
}

#[test]
fn openqasm_3_export_refuses_what_it_cannot_say() {
    lang::set(Lang::En);
    let cases = [
        ("QMODE 1\nDISPLACE o0 1\n", "OpenQASM has no optical modes"),
        // erregistro bateko kopurua: qiskit-ek eta IBM-k ez dute aldagai klasikorik onartzen
        (
            "QINIT 1\nREGSET R1 2\nLOOPSTART R1\nH q0\nLOOPEND\n",
            "OpenQASM 3",
        ),
        // atzerako jauzi bat begizta itxi baten barrutik kanpora
        (
            "QINIT 2\nhasi:\nH q0\nMEASURE q0 -> c0\nJUMPIFONE c0 kanpo\nJUMPIFZERO c0 hasi\nkanpo:\nH q1\n",
            "OpenQASM 3",
        ),
        ("QINIT 1\nMEASURE q0 -> R0\n", "measurement into a register"),
    ];
    for (src, want) in cases {
        let prog = qoa::asm::assemble(src, "t").unwrap();
        let errs = qoa::qasm::from_program_v3(&prog).unwrap_err();
        let text: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
        assert!(
            text.iter().any(|t| t.contains(want)),
            "expected {:?} in {:?}",
            want,
            text
        );
    }
}

// openqasm 3 -> qoa: blokeak eta ate-modifikatzaileak
#[test]
fn openqasm_3_blocks_and_modifiers_import() {
    lang::set(Lang::En);
    // modifikatzaileak: G eta gero inv @ G = ezer ez; pow(2) @ s = z; ctrl(2) @ x = ccx
    let mut rng = StdRng::seed_from_u64(17);
    let n = 3;
    for (a, b) in [
        ("u(0.3,0.4,0.5) q[0]; inv @ u(0.3,0.4,0.5) q[0];", ""),
        ("u2(0.3,0.4) q[1]; inv @ u2(0.3,0.4) q[1];", ""),
        (
            "cu(0.3,0.4,0.5,0.6) q[0],q[1]; inv @ cu(0.3,0.4,0.5,0.6) q[0],q[1];",
            "",
        ),
        ("sx q[0]; inv @ sx q[0];", ""),
        ("rxx(0.7) q[0],q[2]; inv @ rxx(0.7) q[0],q[2];", ""),
        ("inv @ s q[0];", "sdg q[0];"),
        ("inv @ t q[2];", "tdg q[2];"),
        ("pow(2) @ s q[1];", "z q[1];"),
        ("pow(-1) @ t q[1];", "tdg q[1];"),
        ("ctrl(2) @ x q[0],q[1],q[2];", "ccx q[0],q[1],q[2];"),
        ("ctrl @ ctrl @ x q[2],q[0],q[1];", "ccx q[2],q[0],q[1];"),
        ("ctrl @ rx(0.4) q[1],q[0];", "crx(0.4) q[1],q[0];"),
        ("inv @ ctrl @ rx(0.4) q[1],q[0];", "crx(-0.4) q[1],q[0];"),
        ("ctrl @ inv @ p(0.4) q[1],q[0];", "cp(-0.4) q[1],q[0];"),
        ("ctrl @ sx q[0],q[1];", "csx q[0],q[1];"),
    ] {
        let pre = format!("OPENQASM 3.0;\nqubit[{}] q;\n{}", n, scramble(&mut rng, n));
        let x = state_of_qasm(&format!("{}{}\n", pre, a));
        let y = state_of_qasm(&format!("{}{}\n", pre, b));
        assert!(fidelity(&x, &y) > 1.0 - 1e-12, "{} vs {}", a, b);
    }
    // blokeak, iruzkinekin eta lerro askotan
    let src = "OPENQASM 3.0;
include \"stdgates.inc\";
qubit[3] q;
bit[3] c;
// arrakasta arte
reset q[0];
h q[0];
c[0] = measure q[0];
while (!c[0]) {
  reset q[0];
  h q[0];
  c[0] = measure q[0];
}
for uint k in [0:2:5] {  // hiru aldiz
  x q[1];
}
if (c[0] == false) {
  x q[2];
} else if (c[1]) {
  h q[2];
  h q[2];
} else {
  x q[2];
  x q[2];
}
c[1] = measure q[1];
c[2] = measure q[2];
";
    let qoa_src = qoa::qasm::to_qoa(src, "t.qasm").unwrap_or_else(|e| panic!("{}", e));
    let prog = qoa::asm::assemble(&qoa_src, "t")
        .unwrap_or_else(|d| panic!("{}\n{}", qoa::asm::render(&d, &qoa_src, "t"), qoa_src));
    let out = vm::run(
        &prog,
        &RunConfig {
            shots: 200,
            seed: 4,
            ..RunConfig::default()
        },
        &mut std::io::sink(),
    )
    .unwrap();
    // c0 = 1 beti (begizta), c1 = 1 (hiru X), c2 = 0 (else if: h h)
    let keys: Vec<String> = out
        .counts
        .iter()
        .map(|(r, _)| qoa::report::bits_string(r))
        .collect();
    assert_eq!(keys, vec!["011".to_string()], "{}", qoa_src);
    // sakonera mugatua: 100 bloke habiaratuta errore bat da, ez pila-gainezkatze bat
    let deep = format!(
        "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\n{}x q[0];{}\n",
        "if (c[0]) {".repeat(100),
        "}".repeat(100)
    );
    let e = qoa::qasm::to_qoa(&deep, "t.qasm").unwrap_err();
    assert!(e.to_string().contains("nested more than 64 deep"), "{}", e);
}

// QOA -> OpenQASM 3 -> QOA: kopuru berdin-berdinak (hazi berarekin), begizta, if/else eta while barne
#[test]
fn openqasm_3_export_then_import_gives_the_same_counts() {
    lang::set(Lang::En);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources: Vec<(String, String)> = [
        "examples/until.qoa",
        "examples/teleport.qoa",
        "examples/grover.qoa",
        "examples/grover-key.qoa",
        "examples/qpe.qoa",
    ]
    .iter()
    .map(|n| {
        (
            n.to_string(),
            std::fs::read_to_string(root.join(n)).unwrap(),
        )
    })
    .collect();
    sources.push((
        "if/else".into(),
        "QINIT 3\nH q0\nMEASURE q0 -> c0\nJUMPIFONE c0 bestela\nX q1\nJUMP eginda\nbestela:\nH q2\nT q2\neginda:\nLOOPSTART 3\nRY q1 0.3\nLOOPEND\nSXDG q2\nCSX q2 q1\nMEASUREALL\n".into(),
    ));
    for (name, src) in sources {
        let prog = qoa::asm::assemble(&src, &name).unwrap();
        let text = qoa::qasm::from_program_v3(&prog).unwrap_or_else(|e| panic!("{:?}", e));
        let back_src =
            qoa::qasm::to_qoa(&text, &name).unwrap_or_else(|e| panic!("{}\n{}", e, text));
        let back = qoa::asm::assemble(&back_src, &name).unwrap();
        let cfg = RunConfig {
            shots: 3000,
            seed: 9,
            ..RunConfig::default()
        };
        let a = vm::run(&prog, &cfg, &mut std::io::sink()).unwrap();
        let b = vm::run(&back, &cfg, &mut std::io::sink()).unwrap();
        let bits = |o: &vm::RunOutput| {
            let mut v: Vec<(String, u64)> = o
                .counts
                .iter()
                .map(|(r, n)| (qoa::report::bits_string(r).replace('-', "0"), *n))
                .collect();
            v.sort();
            v
        };
        assert_eq!(bits(&a), bits(&b), "{}\n{}", name, text);
    }
}
