#!/usr/bin/env python3
# fitxategi honek QOA programa baten qubit-zatia IBM-ren ordenagailu kuantiko erreal batean exekutatzen du,
# doako "Open Plan"-arekin (IBM-ren dokumentazioaren arabera, 2026ko urrian: 10 minutu exekuzio 28 eguneko
# leiho mugikor bakoitzean), eta emaitza QOA-ren simulazio zehatzarekin konparatzen du.
#
#   python3 tools/run_on_ibm.py examples/bell.qoa              # gailu erreal bat (kontua eta token-a behar dira)
#   python3 tools/run_on_ibm.py examples/bell.qoa --dry-run    # IBM gailu baten modeloa, bere zaratarekin, kontu gabe
#   python3 tools/run_on_ibm.py --job <id>                     # lehenago bidalitako lan baten emaitzak
#
# token-a: --token, QISKIT_IBM_TOKEN ingurune-aldagaia, edo QiskitRuntimeService.save_account()-ekin gordetakoa.
# kontua: https://quantum.cloud.ibm.com (doakoa). minutuak mugatuak direnez, script-ak galdetu egiten du lan bat
# bidali aurretik (--yes galdera saltatzeko), eta lan bakoitzak zenbat erabili duen idazten du.
#
# behar ditu: qiskit, qiskit-ibm-runtime (eta --dry-run-erako qiskit-aer). QOA lehenago: cargo build --release.
# bidalitako lanak target/ibm-jobs.json-en gordetzen dira, emaitzak geroago --job-ekin jasotzeko.

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QOA = os.path.join(ROOT, "target", "release", "qoa.exe" if os.name == "nt" else "qoa")
JOBS = os.path.join(ROOT, "target", "ibm-jobs.json")

EU = os.environ.get("QOA_LANG", os.environ.get("LANG", "")).lower().startswith("eu")


def t(en, eu):
    return eu if EU else en


def die(msg, code=1):
    print(t("error: ", "errorea: ") + msg, file=sys.stderr)
    sys.exit(code)


def qoa(*args):
    r = subprocess.run([QOA, *args], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return r.stdout


# QOA-ren kopuru zehatzak (simulazioa): gakoak c[n-1]..c[0], '-' idatzi gabeko bitentzat (0 hardwarean)
def qoa_counts(path, shots):
    rep = json.loads(qoa("run", path, "--json", "--shots", str(shots), "--seed", "1"))
    out = {}
    for c in rep["counts"]:
        key = c["bits"].replace("-", "0")
        out[key] = out.get(key, 0) + c["count"]
    return out


# SamplerV2-ren emaitza QOA-ren gakoetara: erregistro bakoitzaren bitak bere bit klasiko globalean
def hardware_counts(pub, circuit):
    n = circuit.num_clbits
    per_reg = []
    for reg in circuit.cregs:
        strings = getattr(pub.data, reg.name).get_bitstrings()
        index = [circuit.find_bit(reg[j]).index for j in range(reg.size)]
        per_reg.append((strings, index))
    shots = len(per_reg[0][0]) if per_reg else 0
    out = {}
    for s in range(shots):
        bits = ["0"] * n
        for strings, index in per_reg:
            word = strings[s]
            for j, k in enumerate(index):
                bits[n - 1 - k] = word[len(word) - 1 - j]
        key = "".join(bits)
        out[key] = out.get(key, 0) + 1
    return out


def tvd(a, b):
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in set(a) | set(b))


def report(path, backend, shots, ideal, measured, isa, usage):
    total = sum(ideal.values())
    support = {k for k, v in ideal.items() if v / total > 0.01}
    mass = sum(v for k, v in measured.items() if k in support) / sum(measured.values())
    two = sum(v for k, v in isa.count_ops().items() if k in ("cz", "ecr", "cx", "rzz")) if isa is not None else None
    lines = [
        t("program     ", "programa    ") + path,
        t("device      ", "gailua      ") + backend,
        t("shots       ", "shot-ak     ") + str(shots),
    ]
    if isa is not None:
        lines.append(t("compiled    ", "konpilatua  ") + t(
            f"{two} two-qubit gates, depth {isa.depth()}", f"{two} bi qubiteko ate, sakonera {isa.depth()}"))
    if usage is not None:
        lines.append(t("QPU time    ", "QPU denbora ") + t(f"{usage} s of your free minutes", f"{usage} s zure doako minutuetatik"))
    lines.append(t("agreement   ", "adostasuna  ") + t(
        f"{mass:.1%} of shots on the outcomes QOA predicts; distance from QOA (TVD) {tvd(ideal, measured):.3f}",
        f"shot-en {mass:.1%} QOA-k aurreikusten dituen emaitzetan; QOA-tik distantzia (TVD) {tvd(ideal, measured):.3f}"))
    lines.append("")
    lines.append(t("outcome      QOA (exact)   device", "emaitza      QOA (zehatza)  gailua"))
    keys = sorted(set(ideal) | set(measured), key=lambda k: -(ideal.get(k, 0) / total + measured.get(k, 0) / shots))
    for k in keys[:16]:
        lines.append(f"{k:<12} {ideal.get(k, 0) / total:>10.1%}   {measured.get(k, 0) / shots:>7.1%}")
    if len(keys) > 16:
        lines.append(t(f"... {len(keys) - 16} more outcomes", f"... beste {len(keys) - 16} emaitza"))
    return "\n".join(lines) + "\n"


def load_jobs():
    try:
        with open(JOBS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_job(job_id, entry):
    jobs = load_jobs()
    jobs[job_id] = entry
    os.makedirs(os.path.dirname(JOBS), exist_ok=True)
    with open(JOBS, "w") as f:
        json.dump(jobs, f, indent=2)


def service(args):
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = args.token or os.environ.get("QISKIT_IBM_TOKEN")
    try:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=args.instance)
    except Exception as e:  # noqa: BLE001
        die(t(
            f"cannot log in to IBM Quantum ({e}). Create a free account at https://quantum.cloud.ibm.com, "
            "then pass --token or set QISKIT_IBM_TOKEN",
            f"ezin da IBM Quantum-en sartu ({e}). Sortu doako kontu bat https://quantum.cloud.ibm.com-en, "
            "eta gero eman --token edo ezarri QISKIT_IBM_TOKEN"))


def usage_seconds(job):
    try:
        return job.metrics().get("usage", {}).get("seconds")
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=t(
        "Run a QOA program's qubit part on a real IBM quantum computer (free Open Plan) and compare with QOA",
        "QOA programa baten qubit-zatia IBM-ren ordenagailu kuantiko erreal batean exekutatu (doako Open Plan) eta QOA-rekin konparatu"))
    ap.add_argument("program", nargs="?", help=t(".qoa, .qg or .qasm file", ".qoa, .qg edo .qasm fitxategia"))
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--backend", help=t("device name (default: the least busy one that fits)", "gailuaren izena (lehenetsia: sartzen den lanpetuena ez dena)"))
    ap.add_argument("--dry-run", action="store_true", help=t(
        "run on a local model of an IBM device with its measured noise (no account, no QPU time)",
        "IBM gailu baten modelo lokalean exekutatu, bere zarata neurtuarekin (konturik eta QPU denborarik gabe)"))
    ap.add_argument("--model", default="torino", help=t("device model for --dry-run: torino, sherbrooke, ...", "--dry-run-erako gailu-modeloa: torino, sherbrooke, ..."))
    ap.add_argument("--token")
    ap.add_argument("--instance", help=t("IBM Cloud instance (CRN), if you have several", "IBM Cloud instantzia (CRN), bat baino gehiago badituzu"))
    ap.add_argument("--yes", action="store_true", help=t("do not ask before using QPU time", "ez galdetu QPU denbora erabili aurretik"))
    ap.add_argument("--no-wait", action="store_true", help=t("submit and exit; fetch later with --job", "bidali eta irten; jaso geroago --job-ekin"))
    ap.add_argument("--job", help=t("fetch the results of a job submitted earlier", "lehenago bidalitako lan baten emaitzak jaso"))
    ap.add_argument("--out", help=t("also write the report to this file", "txostena fitxategi honetan ere idatzi"))
    args = ap.parse_args()

    if not os.path.exists(QOA):
        die(t("build QOA first: cargo build --release", "konpilatu QOA lehenik: cargo build --release"))
    try:
        from qiskit import qasm2
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2
    except ImportError as e:
        die(t(f"Qiskit is missing ({e}): pip install qiskit qiskit-ibm-runtime qiskit-aer",
              f"qiskit falta da ({e}): pip install qiskit qiskit-ibm-runtime qiskit-aer"))

    if args.job:
        entry = load_jobs().get(args.job)
        if entry is None:
            die(t(f"job {args.job} is not in {JOBS}", f"{args.job} lana ez dago {JOBS}-n"))
        job = service(args).job(args.job)
        print(t(f"job {args.job}: {job.status()}", f"{args.job} lana: {job.status()}"))
        circuit = qasm2.loads(entry["qasm"], custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        measured = hardware_counts(job.result()[0], circuit)
        text = report(entry["program"], entry["backend"], entry["shots"], entry["ideal"], measured, None, usage_seconds(job))
        print(text)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text)
        return

    if not args.program:
        ap.error(t("give a program, or --job", "eman programa bat, edo --job"))
    try:
        qasm = qoa("export", "--to", "qasm", args.program)
    except RuntimeError as e:
        die(t(f"this program cannot run on a qubit device as it is:\n{e}",
              f"programa hau ezin da qubit-gailu batean exekutatu dagoen bezala:\n{e}"), 2)
    circuit = qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    if circuit.num_clbits == 0:
        die(t("the program measures nothing, so a device would return nothing", "programak ez du ezer neurtzen, beraz gailu batek ez luke ezer itzuliko"), 2)
    ideal = qoa_counts(args.program, args.shots)

    if args.dry_run:
        import qiskit_ibm_runtime.fake_provider as fake

        name = "Fake" + args.model.capitalize()
        if not hasattr(fake, name):
            die(t(f"no device model named {args.model}", f"ez dago {args.model} izeneko gailu-modelorik"))
        backend = getattr(fake, name)()
        label = t(f"{backend.name} (local model with its measured noise, not the real device)",
                  f"{backend.name} (modelo lokala bere zarata neurtuarekin, ez gailu erreala)")
    else:
        svc = service(args)
        if args.backend:
            backend = svc.backend(args.backend)
        else:
            backend = svc.least_busy(operational=True, simulator=False, min_num_qubits=circuit.num_qubits)
        label = backend.name

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=1)
    isa = pm.run(circuit)
    two = sum(v for k, v in isa.count_ops().items() if k in ("cz", "ecr", "cx", "rzz"))
    print(t(f"{args.program} on {label}: {two} two-qubit gates, depth {isa.depth()}, {args.shots} shots",
            f"{args.program} {label}-n: {two} bi qubiteko ate, sakonera {isa.depth()}, {args.shots} shot"))
    if two > 200:
        print(t("warning: more than 200 two-qubit gates; today's noise will likely wash the answer out",
                "abisua: 200 bi qubiteko ate baino gehiago; gaurko zaratak erantzuna ezabatuko du ziurrenik"))

    if not args.dry_run and not args.yes:
        answer = input(t("this uses QPU time from your free minutes. continue? [y/N] ",
                         "honek zure doako minutuetako QPU denbora erabiltzen du. jarraitu? [b/E] "))
        if answer.strip().lower() not in ("y", "yes", "b", "bai"):
            print(t("not submitted", "ez da bidali"))
            return

    job = SamplerV2(mode=backend).run([isa], shots=args.shots)
    if not args.dry_run:
        save_job(job.job_id(), {
            "program": args.program,
            "backend": label,
            "shots": args.shots,
            "qasm": qasm,
            "ideal": ideal,
            "submitted": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(t(f"submitted job {job.job_id()} (saved in {JOBS})", f"{job.job_id()} lana bidalita ({JOBS}-n gordeta)"))
        if args.no_wait:
            print(t(f"fetch the results later: python3 tools/run_on_ibm.py --job {job.job_id()}",
                    f"jaso emaitzak geroago: python3 tools/run_on_ibm.py --job {job.job_id()}"))
            return
        print(t("waiting for the device (the free plan's queue can take a while; Ctrl-C is safe, use --job later)",
                "gailuaren zain (doako planaren ilarak denbora har dezake; Ctrl-C seguru da, erabili --job geroago)"))
    measured = hardware_counts(job.result()[0], circuit)
    text = report(args.program, label, args.shots, ideal, measured, isa, None if args.dry_run else usage_seconds(job))
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
