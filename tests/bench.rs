// fitxategi honek errendimendu-proba (src/bench.rs, `qoa --benchmark`) egiaztatzen du bertsio azkarrean: atal
// guztiak exekutatzen direla, puntuazioak finituak eta positiboak direla, guztizkoa haien batez besteko
// geometrikoa dela, txostenak atal bakoitza bere xehetasunekin erakusten duela, eta json-a irakurgarria dela. CLI-ak forma guztiak onartzen ditu: --benchmark, --bm, benchmark, bench eta bm.

use qoa::lang::{self, Lang};
use std::process::Command;

#[test]
fn the_quick_benchmark_measures_every_part() {
    lang::set(Lang::En);
    let mut seen = Vec::new();
    let r = qoa::bench::run(true, false, |name| seen.push(name.to_string()));
    let keys: Vec<&str> = r.items.iter().map(|i| i.key).collect();
    for want in [
        "memory",
        "assembler",
        "gates20",
        "sampling",
        "shots",
        "optics",
        "quad",
        "sums",
        "light",
        "wigner",
    ] {
        assert!(keys.contains(&want), "{} falta da: {:?}", want, keys);
    }
    assert_eq!(seen.len(), r.items.len());
    for i in &r.items {
        assert!(
            i.score.is_finite() && i.score > 0.0,
            "{}: {}",
            i.key,
            i.score
        );
        assert!(
            i.value > 0.0 && i.best_s > 0.0 && i.best_s <= i.median_s,
            "{:?}",
            i
        );
        assert!(!i.details.is_empty());
    }
    let geo = (r.items.iter().map(|i| i.score.ln()).sum::<f64>() / r.items.len() as f64).exp();
    assert!((r.score - geo).abs() < 1e-9 * geo);
    assert!(r.gpu.is_none() && r.gpu_note.is_some());

    let style = qoa::report::Style { color: false };
    let text = qoa::bench::render(&r, &style);
    for i in &r.items {
        assert!(
            text.contains(&i.name) && text.contains(i.unit),
            "{}",
            i.name
        );
    }
    assert!(text.contains("score") && text.contains("geometric mean") && text.contains("1000 ="));

    let json: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
    assert_eq!(json["items"].as_array().unwrap().len(), r.items.len());
    assert!(json["score"].as_f64().unwrap() > 0.0);
}

#[test]
fn every_way_of_asking_starts_it() {
    let qoa = env!("CARGO_BIN_EXE_qoa");
    let help = Command::new(qoa)
        .args(["--help"])
        .env("QOA_LANG", "en")
        .output()
        .unwrap();
    let help = String::from_utf8_lossy(&help.stdout);
    assert!(
        help.contains("--benchmark") && help.contains("bm"),
        "{}",
        help
    );
    for form in [
        &["bench", "--help"][..],
        &["bm", "--help"],
        &["benchmark", "--help"],
    ] {
        let o = Command::new(qoa)
            .args(form)
            .env("QOA_LANG", "en")
            .output()
            .unwrap();
        assert!(o.status.success(), "{:?}", form);
        assert!(
            String::from_utf8_lossy(&o.stdout).contains("--quick"),
            "{:?}",
            form
        );
    }
    // euskaraz ere (hizkuntza prozesu bakoitzeko behin finkatzen da, beraz CLI-tik)
    let eu = Command::new(qoa)
        .args(["bm", "--help"])
        .env("QOA_LANG", "eu")
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&eu.stdout).contains("puntuazio"),
        "{}",
        String::from_utf8_lossy(&eu.stdout)
    );
    // aukerarik gabe: laguntza eta errore-kodea
    let o = Command::new(qoa).env("QOA_LANG", "en").output().unwrap();
    assert!(!o.status.success());
}
