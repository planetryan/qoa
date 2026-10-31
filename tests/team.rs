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
