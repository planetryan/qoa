// fitxategi honek glifo-sintaxia (src/glyph.rs) egiaztatzen du: adierazpen bakoitzak sortzen duen QOA
// testu zehatza, glifo-programek QOA arrunteko bikoteen emaitza berak ematen dituztela, eta erroreek
// glifo-iturburuko leku zuzena seinalatzen dutela.

use qoa::lang::{self, Lang};
use qoa::vm::{self, RunConfig};

fn expand(src: &str) -> String {
    match qoa::glyph::expand(src) {
        Ok(e) => e.source,
        Err(d) => panic!("{}", qoa::asm::render(&d, src, "proba.qg")),
    }
}

fn errors(src: &str) -> Vec<(u32, u32, String)> {
    lang::set(Lang::En);
    match qoa::glyph::assemble(src, "proba.qg") {
        Ok(_) => panic!("ez luke mihiztatu behar: {}", src),
        Err(d) => d.into_iter().map(|d| (d.line, d.col, d.message)).collect(),
    }
}

#[test]
fn statements_expand_to_plain_qoa() {
    let cases: &[(&str, &str)] = &[
        // erregistroak ordenan esleitzen dira; erregistro osoa = elementu bakoitzeko
        ("A{2}; B{1}; H/A; X/B;", "QINIT 3\nH q0\nH q1\nX q2\n"),
        // adierazpena erabileran bertan
        ("H/R{3};", "QINIT 3\nH q0\nH q1\nH q2\n"),
        // indizeak, tarteak, hamaseitarra, parametroak
        (
            "Q{4}; RZ(pi/4)/Q[0x2]; CX/Q[0],Q[3]; X/Q[1..2];",
            "QINIT 4\nRZ q2 pi/4\nCNOT q0 q3\nX q1\nX q2\n",
        ),
        ("Q{1}; U(1.0, 0.5, -0.3)/Q;", "QINIT 1\nU q0 1.0 0.5 -0.3\n"),
        // bi erregistro elementuz elementu
        ("A{2}; B{2}; CX/A,B;", "QINIT 4\nCNOT q0 q2\nCNOT q1 q3\n"),
        // bideratzea: $> -> bihurtzen da
        (
            "Q{2}; #C{2}; M/Q$>#C;",
            "QINIT 2\nMEASURE q0 -> c0\nMEASURE q1 -> c1\n",
        ),
        ("~L{1}%5; COUNT/L$>#3;", "QMODE 1 5\nCOUNT o0 -> R3\n"),
        // modu-erregistroak eta qubitak batera
        (
            "E{2}; ~L{2}%4; EMIT/E,L; BS(pi/4)/L[0],L[1];",
            "QINIT 2\nQMODE 2 4\nEMIT q0 o0\nEMIT q1 o1\nBS o0 o1 pi/4\n",
        ),
        // inprimatzea: bit-erregistroak R31-era PACKBITS bidez
        (
            "#C{2}; PF\"x={} y={}\\n\"%#C%#4;",
            "PACKBITS R31 c0 c1\nPF \"x={} y={}\\n\" R31 R4\n",
        ),
        ("PF\"kaixo\";", "PF \"kaixo\"\n"),
        // erakustea, kontrola, begiztak, azpierrutinak, ihesa
        ("<<>>; <<#>>;", "DUMPSTATE\nDUMPREGS\n"),
        (
            "Q{1}; #C{1}; a:; ?#C>a; !#C[0]>b; >a; b:; END;",
            "QINIT 1\na:\nJUMPIFONE c0 a\nJUMPIFZERO c0 b\nJUMP a\nb:\nEND\n",
        ),
        (
            "Q{1}; *3{ H/Q; X/Q; }",
            "QINIT 1\nLOOPSTART 3\nH q0\nX q0\nLOOPEND\n",
        ),
        ("*#5{ PF\"a\"; };", "LOOPSTART R5\nPF \"a\"\nLOOPEND\n"),
        (
            "&f; END; f:; PF\"in\"; ^;",
            "CALL f\nEND\nf:\nPF \"in\"\nRETSUB\n",
        ),
        ("`QINIT 2`; `MEASUREALL`;", "QINIT 2\nMEASUREALL\n"),
        // QOA arrunteko eragigaiak bere horretan
        ("`QINIT 2`; CNOT/q0,q1;", "QINIT 2\nCNOT q0 q1\n"),
        // balio-eragigaietan zenbaki bat (edo parentesi arteko adierazpen bat) zenbaki bat da
        (
            "ADD/#1,#1,1; ADD/#2,#1,(pi/2); CHAROUT/(65); IFEQ/#1,10,a; a:; STORE/7,#3;",
            "ADD R1 R1 (1)\nADD R2 R1 (pi/2)\nCHAROUT (65)\nIFEQ R1 (10) a\na:\nSTORE (7) R3\n",
        ),
        // iruzkinak: `;`-z hasten den lerroa, eta // edonon
        (
            "; iruzkina\nQ{1}; // beste bat\nH/Q; // ;\n",
            "QINIT 1\nH q0\n",
        ),
    ];
    for (src, want) in cases {
        assert_eq!(&expand(src), want, "sarrera: {}", src);
    }
}

#[test]
fn glyph_programs_match_their_plain_twins() {
    // seed eta shot berekin, emaitza berdin-berdinak (hom.qg ez da bikia: fotoiak igorleekin sortzen ditu)
    let pairs = [
        ("examples/bell.qg", "examples/bell.qoa"),
        ("examples/teleport.qg", "examples/teleport.qoa"),
        ("examples/retry.qg", "examples/retry.qoa"),
    ];
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for (g, q) in pairs {
        let gs = std::fs::read_to_string(root.join(g)).unwrap();
        let qs = std::fs::read_to_string(root.join(q)).unwrap();
        let gp = qoa::glyph::assemble(&gs, g).unwrap();
        let qp = qoa::asm::assemble(&qs, q).unwrap();
        let cfg = RunConfig {
            shots: 3000,
            seed: 9,
            ..RunConfig::default()
        };
        let go = vm::run(&gp, &cfg, &mut std::io::sink()).unwrap();
        let qo = vm::run(&qp, &cfg, &mut std::io::sink()).unwrap();
        assert_eq!(go.counts, qo.counts, "{} eta {}", g, q);
    }
}

#[test]
fn errors_point_at_the_glyph_source() {
    // ; falta da: hurrengo karakterean
    let e = errors("Q{2};\nCX/Q[0] Q[1];\n");
    assert_eq!((e[0].0, e[0].1), (2, 9), "{:?}", e);
    assert!(e[0].2.contains("expected ;"));
    // mihiztatzailearen erroreak adierazpenera itzultzen dira (lerroa eta zutabea)
    let e = errors("Q{2};\nH/Q[0]; FROB/Q[1]; CX/Q[0],Q[0];\n");
    assert_eq!(e.len(), 2, "{:?}", e);
    assert_eq!((e[0].0, e[0].1), (2, 9));
    assert!(e[0].2.contains("unknown instruction 'FROB'"));
    assert_eq!((e[1].0, e[1].1), (2, 20));
    assert!(e[1].2.contains("uses qubit 0 twice"));
    // adierazi gabeko erregistroa, tartetik kanpoko indizea, tamaina desberdinak
    let e = errors("Q{2};\nH/Z[0];\n");
    assert!(e[0].2.contains("Z is not declared"), "{:?}", e);
    let e = errors("Q{2};\nH/Q[2];\n");
    assert!(e[0].2.contains("index 2 is past the end"), "{:?}", e);
    let e = errors("Q{2}; #C{3};\nM/Q$>#C;\n");
    assert_eq!((e[0].0, e[0].1), (2, 1));
    assert!(e[0].2.contains("same size"), "{:?}", e);
    // erregistro bat beste tamaina batekin berriz
    let e = errors("Q{2}; H/Q{3};");
    assert!(e[0].2.contains("already declared"), "{:?}", e);
    // zenbaki hutsa helbide baten lekuan (helmuga batean; balio-eragigaietan zenbaki bat zenbaki bat da)
    let e = errors("Q{1}; ADD/1,#1,#1;");
    assert!(e[0].2.contains("would be read as an address"), "{:?}", e);
    let e = errors("Q{1}; H/0;");
    assert!(e[0].2.contains("would be read as an address"), "{:?}", e);
    // begiztak
    let e = errors("Q{1}; *2{ H/Q;");
    assert!(e[0].2.contains("never closed"), "{:?}", e);
    let e = errors("Q{1}; H/Q; }");
    assert!(e[0].2.contains("closes no loop"), "{:?}", e);
    // ixten ez diren testua eta parentesia
    let e = errors("PF\"kaixo;\n");
    assert!(e[0].2.contains("never closed"), "{:?}", e);
    let e = errors("Q{1}; RZ(pi/4/Q;\n");
    assert!(e[0].2.contains("never closed"), "{:?}", e);
}

// [0x..] adierazpen baten ondoren: erregistroaren kokapena (memoria-helbidea); beste lekuetan elementu-indizea
#[test]
fn a_location_after_a_declaration_places_the_register() {
    assert_eq!(
        expand("Q{2}[0x3]; H/Q; X/Q[0x1];"),
        "QINIT 5\nH q3\nH q4\nX q4\n"
    );
    // kokatu gabeko hurrengoa kokatuen ondoren doa
    assert_eq!(
        expand("A{1}[0x2]; B{1}; #C{2}[0x4]; X/A; X/B; M/A$>#C[1];"),
        "QINIT 4\nX q2\nX q3\nMEASURE q2 -> c5\n"
    );
    let e = errors("A{2}; B{2}[0x1];");
    assert!(e[0].2.contains("B at 1..2 overlaps A (0..1)"), "{:?}", e);
    let e = errors("A{2}[0x0..0x1];");
    assert!(e[0].2.contains("location is one number"), "{:?}", e);
}

// CC: dei klasikoa helbide edo etiketa batera, argumentuak datu-pilan eta emaitza erregistro batera
#[test]
fn classical_calls_follow_the_stack_convention() {
    assert_eq!(
        expand("#C{2}; CC[0x10]%#C[0]%#2%(pi/2)$>#5; END; 0x10:; POP/#1; ^;"),
        "REGSET R31 pi/2\nPUSH R31\nPUSH R2\nPACKBITS R31 c0\nPUSH R31\nCALL addr_0x10\nPOP R5\nEND\naddr_0x10:\nPOP R1\nRETSUB\n"
    );
    // etiketa bat izenez, emaitzarik gabe; jauziak eta deiak helbideetara ere
    assert_eq!(
        expand("CC\"bat\"; >0x1f; &0x1F; bat:; ^; 0x1f:;"),
        "CALL bat\nJUMP addr_0x1f\nCALL addr_0x1f\nbat:\nRETSUB\naddr_0x1f:\n"
    );
    let e = errors("Q{1}; CC[0x10]%Q[0]; 0x10:; ^;");
    assert!(
        e[0].2.contains("no classical value until it is measured"),
        "{:?}",
        e
    );
    let e = errors("#C{1}; CC[0x10]$>#C[0]; 0x10:; ^;");
    assert!(e[0].2.contains("goes to a register"), "{:?}", e);
    let e = errors("CC; ^;");
    assert!(e[0].2.contains("CC needs what to call"), "{:?}", e);
    let e = errors("12; END;");
    assert!(e[0].2.contains("only as an address label"), "{:?}", e);
    // exekuzioan: argumentuak ordenan iristen dira eta emaitza itzultzen da
    let prog = qoa::glyph::assemble(
        "CC[0x8]%7%2$>#5; PF\"{}\"%#5; END; 0x8:; POP/#1; POP/#2; SUB/#3,#1,#2; PUSH/#3; ^;",
        "cc.qg",
    )
    .unwrap();
    let mut out = Vec::new();
    vm::run(&prog, &RunConfig::default(), &mut out).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "5");
}

#[test]
fn pr_prints_a_label_and_its_values() {
    assert_eq!(expand("PR\"x\"%#3;"), "PRINTF \"x = {}\\n\" R3\n");
    assert_eq!(
        expand("#C{2}; PR\"bits\"%#C%#1;"),
        "PACKBITS R31 c0 c1\nPRINTF \"bits = {} {}\\n\" R31 R1\n"
    );
    assert_eq!(expand("PR\"here\";"), "PRINTF \"here\\n\"\n");
    let e = errors("PR\"a{}\"%#1;");
    assert!(e[0].2.contains("cannot contain { or }"), "{:?}", e);
    let e = errors("PR%#1;");
    assert!(e[0].2.contains("one label"), "{:?}", e);
    // exekuzioan
    let prog = qoa::glyph::assemble("REGSET/#2,(1.5); PR\"R2\"%#2; END;", "pr.qg").unwrap();
    let mut out = Vec::new();
    vm::run(&prog, &RunConfig::default(), &mut out).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "R2 = 1.5\n");
}

#[test]
fn runtime_messages_name_the_glyph_line() {
    // DUMPSTATE-k eta exekuzio-erroreek glifo-iturburuko lerroa aipatzen dute, ez QOA hedatuarena
    lang::set(Lang::En);
    let src = "; iruzkina\n; beste bat\nQ{1};\n\nH/Q; <<>>;\n";
    let p = qoa::glyph::assemble(src, "proba.qg").unwrap();
    let mut out = Vec::new();
    vm::run(&p, &RunConfig::default(), &mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("state at line 5"), "{}", text);
}
