// fitxategi honek shot-ak prozesu-talde batean banatzea egiaztatzen du (`qoa run --mpi`-ren logika), MPI gabe:
// rank bakoitza hari bat da `LocalTeam`-ekin. emaitzak berdinak izan behar dira rank-kopurua edozein dela ere
// (shot bakoitzak bere hazia du), errore batek ez du inor zain uzten, eta sarrerak 0. rank-etik datoz.

use qoa::lang::{self, Lang};
use qoa::team::{LocalTeam, Team};
use qoa::vm::{self, RunConfig, RunOutput};
use std::sync::Arc;

fn single(src: &str, cfg: &RunConfig) -> RunOutput {
    lang::set(Lang::En);
    let prog = qoa::asm::assemble(src, "team.qoa").unwrap();
    vm::run(&prog, cfg, &mut std::io::sink()).unwrap()
}

// `ranks` hari, bakoitza rank bat; 0. rank-aren emaitza eta irteera
fn spread(src: &str, cfg: &RunConfig, ranks: usize) -> Vec<Result<(RunOutput, Vec<u8>), String>> {
    let prog = qoa::asm::assemble(src, "team.qoa").unwrap();
    let members = LocalTeam::new(ranks);
    std::thread::scope(|s| {
        let handles: Vec<_> = members
            .into_iter()
            .map(|m| {
                let mut cfg = cfg.clone();
                let prog = &prog;
                s.spawn(move || {
                    lang::set(Lang::En);
                    cfg.team = Some(Arc::new(m) as Arc<dyn Team>);
                    let mut out = Vec::new();
                    vm::run(prog, &cfg, &mut out)
                        .map(|o| (o, out))
                        .map_err(|e| e.message)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

const SAMPLED: &str = "QINIT 4\nH q0\nH q1\nRY q2 0.7\nCNOT q0 q3\nMEASUREALL\n";
const EVERY_SHOT: &str =
    "QINIT 2\nH q0\nMEASURE q0 -> c0\nJUMPIFZERO c0 skip\nX q1\nskip:\nH q1\nMEASURE q1 -> c1\n";

#[test]
fn every_rank_count_gives_the_same_counts() {
    for (src, noise) in [(SAMPLED, 0.0), (EVERY_SHOT, 0.0), (SAMPLED, 0.03)] {
        let cfg = RunConfig {
            shots: 2001,
            seed: 11,
            noise,
            ..RunConfig::default()
        };
        let want = single(src, &cfg);
        for ranks in [1, 2, 3, 5] {
            let got = spread(src, &cfg, ranks);
            let (out, _) = got[0].as_ref().unwrap();
            assert_eq!(out.counts, want.counts, "{} ranks", ranks);
            assert_eq!(out.method, want.method);
            assert_eq!(out.shots, 2001);
        }
    }
}

#[test]
fn only_rank_zero_prints_and_inputs_come_from_it() {
    let cfg = RunConfig {
        shots: 50,
        seed: 2,
        ..RunConfig::default()
    };
    let src = "QINIT 1\nPRINTF \"hello\\n\"\nH q0\nMEASURE q0 -> c0\n";
    let got = spread(src, &cfg, 3);
    assert_eq!(got[0].as_ref().unwrap().1, b"hello\n");
    for r in &got[1..] {
        assert!(r.as_ref().unwrap().1.is_empty());
    }
}

#[test]
fn an_error_on_one_rank_stops_every_rank() {
    // 1 neurtzen duten shot-ek begizta amaigabean geratzen dira (max-steps): lehen shot-ak 0 neurtzen du
    // hazi honekin, eta beste batzuek 1, beraz errorea rank batzuetan bakarrik gertatzen da
    let src = "QINIT 1\nH q0\nMEASURE q0 -> c0\nJUMPIFZERO c0 done\nloop:\nJUMP loop\ndone:\n";
    let cfg = RunConfig {
        shots: 40,
        seed: 1,
        max_steps: 500,
        ..RunConfig::default()
    };
    // lehen shot-a ondo doa (0 neurtzen du), errorea geroagoko shot batean dago
    let one = RunConfig {
        shots: 1,
        ..cfg.clone()
    };
    let prog = qoa::asm::assemble(src, "team.qoa").unwrap();
    assert!(vm::run(&prog, &one, &mut std::io::sink()).is_ok());
    let got = spread(src, &cfg, 3);
    let e = got[0].as_ref().unwrap_err();
    assert!(e.contains("500 steps"), "{}", e);
}

#[test]
fn memory_sizes_parse_and_limits_explain_themselves() {
    use qoa::state::{check_memory, parse_bytes};
    assert_eq!(parse_bytes("512G"), Some(512 << 30));
    assert_eq!(parse_bytes("64GiB"), Some(64 << 30));
    assert_eq!(parse_bytes("1.5T"), Some(3 << 39));
    assert_eq!(parse_bytes("4096m"), Some(4 << 30));
    assert_eq!(parse_bytes("123456"), Some(123456));
    assert_eq!(parse_bytes("lots"), None);
    assert_eq!(parse_bytes("-5G"), None);
    lang::set(Lang::En);
    let e = check_memory("everything", u128::MAX).unwrap_err();
    assert!(e.contains("QOA_MAX_MEMORY"), "{}", e);
    assert!(check_memory("a little", 1024).is_ok());
}

// ---------------------------------------------------------------- egoera banatua (`--distribute`)

// zirkuitu bat ate mota guztiekin, qubit altuetan ere (prozesuen arteko qubitak)
fn mixed_circuit(n: u32, depth: usize, measure: bool) -> String {
    let mut s = format!("QINIT {}\n", n);
    let mut x: u64 = 99;
    let mut next = |m: u64| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 33) % m
    };
    for _ in 0..depth {
        let q = next(n as u64);
        let r = (q + 1 + next(n as u64 - 1)) % n as u64;
        let a = next(1000) as f64 / 159.0;
        s.push_str(&match next(11) {
            0 => format!("H q{}\n", q),
            1 => format!("RX q{} {}\n", q, a),
            2 => format!("RZ q{} {}\n", q, a),
            3 => format!("CNOT q{} q{}\n", q, r),
            4 => format!("CZ q{} q{}\n", q, r),
            5 => format!("SWAP q{} q{}\n", q, r),
            6 => format!("CRX q{} q{} {}\n", q, r, a),
            7 => format!("T q{}\n", q),
            8 => format!("RY q{} {}\n", q, a),
            9 => format!("SX q{}\n", q),
            _ => {
                let mut t = (r + 1 + next(n as u64)) % n as u64;
                while t == q || t == r {
                    t = (t + 1) % n as u64;
                }
                format!("CSWAP q{} q{} q{}\n", q, r, t)
            }
        });
    }
    if measure {
        s.push_str("MEASUREALL\n");
    }
    s
}

#[test]
fn a_distributed_state_matches_one_process() {
    let src = mixed_circuit(9, 600, false);
    let cfg = RunConfig::default();
    let want = single(&src, &cfg);
    for ranks in [1, 2, 4, 8] {
        let cfg = RunConfig {
            distribute: true,
            ..cfg.clone()
        };
        let got = spread(&src, &cfg, ranks);
        let (out, _) = got[0].as_ref().unwrap();
        let worst = out
            .final_state
            .amplitudes()
            .iter()
            .zip(want.final_state.amplitudes())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(worst < 1e-12, "{} ranks: {}", ranks, worst);
        assert!(out.norm_error < 1e-12, "{}", out.norm_error);
    }
}

#[test]
fn distributed_shots_measure_and_sample_like_one_process() {
    for (src, shots) in [
        (mixed_circuit(8, 300, true), 3000),
        (
            EVERY_SHOT.replace("QINIT 2", "QINIT 4\nH q3\nCNOT q3 q2"),
            400,
        ),
    ] {
        let cfg = RunConfig {
            shots,
            seed: 5,
            ..RunConfig::default()
        };
        let want = single(&src, &cfg);
        for ranks in [2, 4] {
            let cfg = RunConfig {
                distribute: true,
                ..cfg.clone()
            };
            let got = spread(&src, &cfg, ranks);
            let (out, _) = got[0].as_ref().unwrap();
            assert_eq!(out.method, want.method);
            assert_eq!(out.counts, want.counts, "{} ranks", ranks);
        }
    }
}

#[test]
fn distributed_inspection_reads_the_whole_state() {
    // EXPECT eta BELLCHECK qubit lokal eta globaletan, eta DUMPSTATE
    let mut src = mixed_circuit(6, 120, false);
    src.push_str("EXPECT q0 X\nEXPECT q5 X\nEXPECT q5 Y\nEXPECT q4 Z\nBELLCHECK q0 q5\nBELLCHECK q4 q5\nBELLCHECK q1 q2\nDUMPSTATE\n");
    let cfg = RunConfig::default();
    lang::set(Lang::En);
    let prog = qoa::asm::assemble(&src, "team.qoa").unwrap();
    let mut want = Vec::new();
    vm::run(&prog, &cfg, &mut want).unwrap();
    for ranks in [2, 4] {
        let cfg = RunConfig {
            distribute: true,
            ..cfg.clone()
        };
        let got = spread(&src, &cfg, ranks);
        let (_, out) = got[0].as_ref().unwrap();
        assert_eq!(
            String::from_utf8_lossy(out),
            String::from_utf8_lossy(&want),
            "{} ranks",
            ranks
        );
    }
}

#[test]
fn distribute_explains_what_it_cannot_do() {
    let cfg = RunConfig {
        distribute: true,
        ..RunConfig::default()
    };
    let got = spread("QINIT 3\nH q0\n", &cfg, 3);
    assert!(got[0].as_ref().unwrap_err().contains("power-of-two"));
    let got = spread("QINIT 1\nH q0\n", &cfg, 4);
    assert!(got[0].as_ref().unwrap_err().contains("at least one qubit"));
    let got = spread("QMODE 1 4\nDISPLACE o0 0.5\n", &cfg, 2);
    assert!(got[0].as_ref().unwrap_err().contains("optical modes"));
    lang::set(Lang::En);
    let prog = qoa::asm::assemble("QINIT 2\nH q0\n", "team.qoa").unwrap();
    let e = vm::run(&prog, &cfg, &mut std::io::sink()).unwrap_err();
    assert!(e.message.contains("needs --mpi"), "{}", e.message);
}

#[test]
fn a_distributed_state_on_gpus_matches_one_process() {
    // prozesu bakoitzaren tokiko ateak GPUan (OpenCL, CI-n POCL); gailurik ez badago, saltatu
    let gpu = match qoa::gpu::Gpu::open_with(qoa::gpu::Api::OpenCl, 0) {
        Ok(g) => Arc::new(g),
        Err(why) => {
            eprintln!("OpenCL gabe, proba saltatua: {why}");
            return;
        }
    };
    let src = mixed_circuit(12, 400, false);
    let want = single(&src, &RunConfig::default());
    for ranks in [2, 4] {
        let cfg = RunConfig {
            distribute: true,
            gpu: Some(gpu.clone()),
            ..RunConfig::default()
        };
        let got = spread(&src, &cfg, ranks);
        let (out, _) = got[0].as_ref().unwrap();
        let worst = out
            .final_state
            .amplitudes()
            .iter()
            .zip(want.final_state.amplitudes())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(worst < 1e-12, "{} ranks: {}", ranks, worst);
    }
    assert!(gpu.stats().batches > 0);
}
