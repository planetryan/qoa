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
            "gate modifiers",
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
        ("QINIT 1\nLOOPSTART 2\nH q0\nLOOPEND\n", "control flow"),
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
