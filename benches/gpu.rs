// fitxategi honek GPUaren bidea neurtzen du (`qoa run --gpu`), CPUaren ondoan: egoera igo eta jaitsi, ate
// bakar bat, h geruza bat eta 355 ateko zirkuitu bat (5 geruza: h guztietan, cnot katea, rz guztietan).
//
//   cargo bench --bench gpu               # 20 eta 24 qubit
//   cargo bench --bench gpu -- 22 27      # qubit-kopuru jakin batzuk
//
// zutabeak: pasaldiak (etapak: egoeraren zeharkaldiak gailuan), kalkulua (gailuan, igo/jaitsi gabe), GB/s
// gailuan (2 * egoera * pasaldiak / kalkulua), mugimendua (igo + jaitsi, busetik) eta CPUaren denbora sorta
// berarekin (src/kernel.rs). software bidezko gailu batean (lavapipe) zenbakiek zuzentasuna erakusten dute
// bakarrik, ez GPU baten abiadura.

use qoa::gpu::Gpu;
use qoa::kernel::QGate;
use qoa::program::Gate;
use qoa::state::State;
use std::time::Instant;

fn h_layer(n: usize) -> Vec<QGate> {
    let h = Gate::H.matrix();
    (0..n)
        .map(|q| QGate::Mat {
            q,
            m: h,
            controls: 0,
        })
        .collect()
}

fn circuit(n: usize) -> Vec<QGate> {
    let mut gates = Vec::new();
    for layer in 0..5 {
        gates.extend(h_layer(n));
        for q in 0..n - 1 {
            gates.push(QGate::Mat {
                q: q + 1,
                m: Gate::X.matrix(),
                controls: 1 << q,
            });
        }
        for q in 0..n {
            gates.push(QGate::Mat {
                q,
                m: Gate::RZ(0.1 * (layer * n + q) as f64).matrix(),
                controls: 0,
            });
        }
    }
    gates
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let sizes = if sizes.is_empty() {
        vec![20, 24]
    } else {
        sizes
    };
    let gpu = match Gpu::open() {
        Ok(g) => g,
        Err(e) => {
            println!("GPUrik ez: {e}");
            return;
        }
    };
    let info = gpu.stats();
    println!(
        "gailua: {} ({}); cpu: {} kernelak, {} hari",
        info.device,
        info.device_type,
        qoa::simd::active().name(),
        rayon::current_num_threads()
    );
    println!(
        "{:>2}        {:<14} {:>4} {:>8}  {:>12} {:>13} {:>10} {:>8}",
        "n", "", "ate", "pasaldi", "kalkulua ms", "gailua GB/s", "mugitu ms", "cpu ms"
    );
    for n in sizes {
        let mut s = State::new(n, 0, 2).unwrap();
        // lehen sorta: gailuko bufferrak sortu (ez da neurtzen)
        s.apply_gates_on(&gpu, &h_layer(1)).unwrap();
        let cases = [
            ("h q0", h_layer(1)),
            (
                "h azkena",
                vec![QGate::Mat {
                    q: n - 1,
                    m: Gate::H.matrix(),
                    controls: 0,
                }],
            ),
            ("h geruza", h_layer(n)),
            ("zirkuitua", circuit(n)),
        ];
        for (name, gates) in cases {
            let before = gpu.stats();
            s.apply_gates_on(&gpu, &gates).unwrap();
            let after = gpu.stats();
            let passes = after.passes - before.passes;
            let compute = after.compute_seconds - before.compute_seconds;
            let moved = after.transfer_seconds - before.transfer_seconds;
            let t = Instant::now();
            s.apply_gates(&gates);
            let cpu = t.elapsed().as_secs_f64();
            let bytes = 2.0 * 16.0 * (1u64 << n) as f64 * passes as f64;
            println!(
                "{:>2} qubit  {:<14} {:>4} {:>8}  {:>12.2} {:>13.1} {:>10.2} {:>8.2}",
                n,
                name,
                gates.len(),
                passes,
                compute * 1e3,
                bytes / compute / 1e9,
                moved * 1e3,
                cpu * 1e3
            );
        }
    }
}
