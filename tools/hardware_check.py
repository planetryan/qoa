#!/usr/bin/env python3
# fitxategi honek erantzuten du: QOA programak benetako ordenagailu kuantiko batean exekuta litezke? kredentzialik
# gabe egin daitekeen guztia egiten du, eta txosten bat idazten du (markdown).
#
#   1. qubitak: `qoa export --to qasm` -> qiskit. QOA-ren emaitzak eta qiskit aer-enak konparatzen dira (shot
#      berdinekin, bariazio-distantzia osoa), beraz esportazioak programa bera ematen du.
#   2. hardwarea: zirkuitua IBM gailu errealen modeloetarako transpilatzen da (fake_torino: Heron, 133 qubit;
#      fake_sherbrooke: Eagle, 127 qubit; haien kalibrazio errealekin), eta gailu horien zarata-ereduarekin
#      simulatzen da: zenbat bizirik irauten duen emaitza zuzenak gaurko hardwarean.
#   3. fotoiak: QOA-ren optika linealeko programak (izpi-zatitzaileak, faseak, fotoi bakarrak) perceval-en
#      zirkuitu bihurtzen dira (Quandela-ren hardware fotonikoaren softwarea): QOA-k fotoi bakar baten
#      transferentzia-matrizea ematen du, eta bi simulagailuen fotoi anitzeko banaketak konparatzen dira.
#
#   python3 tools/hardware_check.py [--out txostena.md]
#
# behar ditu: qiskit, qiskit-aer, qiskit-ibm-runtime (gailu-modeloak), perceval-quandela. falta den zatia
# saltatu egiten da, eta txostenak hala esaten du. QOA lehenago konpilatu: cargo build --release.

import argparse
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QOA = os.path.join(ROOT, "target", "release", "qoa")

QUBIT_PROGRAMS = [
    "examples/bell.qoa",
    "examples/ghz.qoa",
    "examples/teleport.qoa",
    "examples/qpe.qoa",
    "examples/grover.qoa",
    "examples/grover-key.qoa",
    "examples/bell.qg",
]

SHOTS = 20000


def qoa(*args):
    r = subprocess.run([QOA, *args], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return r.stdout


# QOA-ren kopuruak: c[n-1]..c[0] ordenan, '-' idatzi gabeko bitentzat (0 gisa konparatzen da)
def qoa_counts(path, shots, seed=1):
    rep = json.loads(qoa("run", path, "--json", "--shots", str(shots), "--seed", str(seed)))
    out = {}
    for c in rep["counts"]:
        key = c["bits"].replace("-", "0")
        out[key] = out.get(key, 0) + c["count"]
    return out


# qiskit-en gakoak: erregistroak zuriunez bereizita, azkena ezkerrean; zuriuneak kenduta QOA-ren ordena bera da
def norm_keys(counts, width):
    out = {}
    for k, v in counts.items():
        key = k.replace(" ", "").zfill(width)[-width:]
        out[key] = out.get(key, 0) + v
    return out


def tvd(a, b):
    na, nb = sum(a.values()), sum(b.values())
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


def qubit_section(lines):
    try:
        from qiskit import qasm2, transpile
        from qiskit_aer import AerSimulator
    except ImportError as e:
        lines.append(f"Qiskit is not installed ({e}); qubit checks skipped.\n")
        return
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeTorino

        backends = [FakeTorino(), FakeSherbrooke()]
    except ImportError:
        backends = []

    lines.append("## Qubit programs: QOA, Qiskit Aer and IBM device models\n")
    lines.append(
        f"Each program is exported with `qoa export --to qasm`, loaded by Qiskit, and run on both simulators "
        f"({SHOTS} shots each). **TVD** is the total variation distance between the two outcome distributions "
        f"(0 = identical; with this many shots, sampling noise alone gives about 0.005). Then the circuit is "
        f"transpiled for real IBM devices (optimisation level 3) and run on a simulator with each device's "
        f"calibrated noise: **ideal mass** is the share of noisy shots that land on the outcomes the ideal program "
        f"gives at least 1% of the time (for Grover, that is the success rate), and **TVD noisy** is the distance "
        f"from the ideal distribution.\n"
    )
    header = "| program | qubits | gates | TVD QOA vs Aer |"
    sep = "|---|---|---|---|"
    for b in backends:
        header += f" {b.name}: 2-qubit gates / depth | ideal mass | TVD noisy |"
        sep += "---|---|---|"
    lines.append(header)
    lines.append(sep)
    ideal_sim = AerSimulator(method="statevector")
    for path in QUBIT_PROGRAMS:
        try:
            text = qoa("export", "--to", "qasm", path)
        except RuntimeError as e:
            lines.append(f"| `{path}` | export refused: {e.splitlines()[-1].strip()} |")
            continue
        qc = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        width = qc.num_clbits
        ours = qoa_counts(path, SHOTS)
        width = max(width, max(len(k) for k in ours))
        aer = norm_keys(ideal_sim.run(qc, shots=SHOTS, seed_simulator=1).result().get_counts(), width)
        row = f"| `{path}` | {qc.num_qubits} | {sum(qc.count_ops().values())} | {tvd(ours, aer):.4f} |"
        # emaitza esanguratsuak: probabilitate idealaren % 1 baino gehiago (grover-en ausazko hondarrak kanpo)
        total = sum(ours.values())
        support = {k for k, v in ours.items() if v / total > 0.01}
        for b in backends:
            try:
                tq = transpile(qc, backend=b, optimization_level=3, seed_transpiler=1)
                two = sum(v for k, v in tq.count_ops().items() if k in ("cz", "ecr", "cx"))
                noisy_sim = AerSimulator.from_backend(b)
                noisy = norm_keys(noisy_sim.run(tq, shots=4000, seed_simulator=1).result().get_counts(), width)
                mass = sum(v for k, v in noisy.items() if k in support) / sum(noisy.values())
                row += f" {two} / {tq.depth()} | {mass:.3f} | {tvd(ours, noisy):.3f} |"
            except Exception as e:  # noqa: BLE001
                row += f" failed: {type(e).__name__} | | |"
        lines.append(row)
    if not backends:
        lines.append("\nqiskit-ibm-runtime is not installed: device models skipped.")
    lines.append("")


# QOA-ren fotoi bakarreko transferentzia-matrizea: fotoi bat j moduan sartu eta irteerako anplitudeak irakurri
def transfer_matrix(network, modes):
    cols = []
    for j in range(modes):
        src = f"QMODE {modes} 2\nSINGLEPHOTON o{j}\n{network}"
        path = os.path.join(ROOT, "target", f"photonic_{j}.qoa")
        with open(path, "w") as f:
            f.write(src)
        rep = json.loads(qoa("run", path, "--json", "--top", "0"))
        col = [0j] * modes
        for a in rep["state"]:
            occ = [int(x) for x in a["basis"].strip("|>").split()]
            if sum(occ) == 1:
                col[occ.index(1)] = complex(a["re"], a["im"])
        cols.append(col)
    return [[cols[j][i] for j in range(modes)] for i in range(modes)]


def photonic_section(lines):
    try:
        import perceval as pcvl
    except ImportError as e:
        lines.append(f"Perceval is not installed ({e}); photonic checks skipped.\n")
        return
    lines.append("## Optical programs: QOA and Quandela's Perceval\n")
    lines.append(
        "For linear optics (beamsplitters, phases, single photons, photon counting), QOA gives the network's "
        "single-photon transfer matrix (one photon sent into each input), and Perceval, the software for "
        "Quandela's photonic processors, simulates the multi-photon input through that matrix (SLOS back end). "
        "**max |Δp|** is the largest difference between the two output distributions.\n"
    )
    lines.append("| network | modes | photons in | outcomes | max |Δp| |")
    lines.append("|---|---|---|---|---|")
    cases = [
        ("Hong-Ou-Mandel, 50:50 beamsplitter", 2, [1, 1], "BS o0 o1 pi/4\n"),
        ("Mach-Zehnder with phase pi/3", 2, [1, 0], "BS o0 o1 pi/4\nMODEPHASE o0 pi/3\nBS o0 o1 pi/4\n"),
        (
            "6-mode interferometer, 3 photons (boson sampling)",
            6,
            [1, 1, 1, 0, 0, 0],
            "".join(
                f"BS o{a} o{b} {0.3 + 0.17 * k} {0.11 * k}\nMODEPHASE o{a} {0.7 * k}\n"
                for k, (a, b) in enumerate([(0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (0, 1), (2, 3), (4, 5), (1, 2), (3, 4)])
            ),
        ),
    ]
    for name, modes, photons, network in cases:
        u = transfer_matrix(network, modes)
        proc = pcvl.Processor("SLOS", pcvl.Unitary(pcvl.Matrix(u)))
        proc.with_input(pcvl.BasicState(photons))
        probs = pcvl.algorithm.Sampler(proc).probs()["results"]
        theirs = {tuple(s): p for s, p in probs.items()}
        # QOA bera, fotoi anitzekin, fotoi-muga nahikoarekin
        n = sum(photons)
        src = f"QMODE {modes} {n + 1}\n" + "".join(
            f"FOCK o{i} {k}\n" for i, k in enumerate(photons) if k > 0
        ) + network
        path = os.path.join(ROOT, "target", "photonic_multi.qoa")
        with open(path, "w") as f:
            f.write(src)
        rep = json.loads(qoa("run", path, "--json", "--top", "0"))
        ours = {}
        for a in rep["state"]:
            occ = tuple(int(x) for x in a["basis"].strip("|>").split())
            ours[occ] = ours.get(occ, 0) + a["probability"]
        keys = set(ours) | set(theirs)
        worst = max(abs(ours.get(k, 0) - theirs.get(k, 0)) for k in keys)
        lines.append(f"| {name} | {modes} | {photons} | {len(keys)} | {worst:.1e} |")
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write the report here (default: print it)")
    args = ap.parse_args()
    if not os.path.exists(QOA):
        sys.exit("build QOA first: cargo build --release")
    lines = ["# Would QOA run on a real quantum computer? Measurements\n"]
    lines.append(f"Generated by `tools/hardware_check.py` with {qoa('--version').strip()}.\n")
    qubit_section(lines)
    photonic_section(lines)
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
