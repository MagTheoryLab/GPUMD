"""Matched GPUMD wall-off/on timing for a 108k-atom static spin system."""
import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path
import re


def write_system(path, cells=30):
    length = 3.6 * cells
    with path.open("w") as stream:
        stream.write(
            f'108000\nLattice="{length} 0 0 0 {length} 0 0 0 {length}" '
            'Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3:spin_vel:R:3\n')
        basis = ((0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0))
        spins = ((1.5, .5, .2), (.1, .1, .1), (.9, 0, 0))
        for i in range(cells):
            for j in range(cells):
                for k in range(cells):
                    for b, offset in enumerate(basis):
                        pos = tuple(3.6 * (q + r) for q, r in zip((i, j, k), offset))
                        spin = spins[(4 * (i * cells * cells + j * cells + k) + b) % 3]
                        fields = (*pos, 0, 0, 0, *spin, 0, 0, 0)
                        stream.write("Fe " + " ".join(f"{x:.8g}" for x in fields) + "\n")


def run_case(binary, root, label, model, xyz, steps):
    case = root / f"{label}-{time.monotonic_ns()}"
    case.mkdir()
    shutil.copyfile(model, case / "nep.txt")
    os.symlink(xyz, case / "model.xyz")
    (case / "run.in").write_text(
        "potential nep.txt\nvelocity 1 seed 731\nensemble nve\ntime_step 0\n"
        f"run {steps}\n")
    start = time.perf_counter()
    result = subprocess.run([str(binary)], cwd=case, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"{label} failed:\n{result.stdout}\n{result.stderr}")
    shutil.rmtree(case)
    return elapsed


def read_dump(path):
    lines = path.read_text().splitlines()
    fields = re.search(r"Properties=([^\s]+)", lines[1]).group(1).split(":")
    rows = []
    for line in lines[2:]:
        values = line.split()
        row, offset = {}, 0
        for i in range(0, len(fields), 3):
            name, kind, width = fields[i], fields[i + 1], int(fields[i + 2])
            data = values[offset:offset + width]
            row[name] = data if kind == "S" else list(map(float, data))
            offset += width
        rows.append(row)
    return rows


def validate_gpu_output(binary, root, models):
    xyz = root / "oracle.xyz"
    xyz.write_text(
        '4\nLattice="24 0 0 0 24 0 0 0 24" '
        'Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3:spin_vel:R:3\n'
        'Fe 1 1 1 0 0 0 1.5 .5 .2 0 0 0\n'
        'Fe 3 1 1 0 0 0 .1 .1 .1 0 0 0\n'
        'Fe 1 3 1 0 0 0 .9 0 0 0 0 0\n'
        'Ge 3 3 1 0 0 0 1.5 .5 .2 0 0 0\n')
    results = {}
    for label, model in models.items():
        case = root / f"oracle-{label}"
        case.mkdir()
        shutil.copyfile(model, case / "nep.txt")
        shutil.copyfile(xyz, case / "model.xyz")
        (case / "run.in").write_text(
            "potential nep.txt\nvelocity 1 seed 731\nensemble nve\ntime_step 0\n"
            "dump_xyz 1 result.xyz precision double force potential spin mforce virial\nrun 1\n")
        result = subprocess.run([str(binary)], cwd=case, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"GPU oracle {label} failed:\n{result.stdout}\n{result.stderr}")
        results[label] = read_dump(case / "result.xyz")
        shutil.rmtree(case)
    spins = ((1.5, .5, .2), (.1, .1, .1), (.9, 0, 0), (1.5, .5, .2))
    for i, spin in enumerate(spins):
        expected_energy, expected_mforce = 0.0, [0.0] * 3
        if i < 3:
            moment = math.sqrt(sum(x * x for x in spin))
            if moment > 1.3:
                excess = moment - 1.3
                expected_energy = .45 * excess**3 / 3
                expected_mforce = [-.45 * excess**2 * x / moment for x in spin]
            elif moment < .4:
                deficit = 1 - moment**2 / .4**2
                expected_energy = .8 * .4 * deficit**3 / 3
                expected_mforce = [1.6 * deficit**2 * x / .4 for x in spin]
        off, on = results["off"][i], results["on"][i]
        assert abs(on["energy_atom"][0] - off["energy_atom"][0] - expected_energy) < 2e-7
        assert max(abs(on["mforce"][d] - off["mforce"][d] - expected_mforce[d])
                   for d in range(3)) < 2e-7
        for key in ("forces", "virial"):
            assert max(abs(a - b) for a, b in zip(on[key], off[key])) < 2e-7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    source_model = args.model.read_text()
    xyz = args.workdir / "model.xyz"
    write_system(xyz)
    models = {"off": args.workdir / "off.nep", "on": args.workdir / "on.nep"}
    models["off"].write_text(source_model)
    header, blocks = source_model.split("@ parameters", 1)
    models["on"].write_text(header + "spin_wall_high 1.3 0.45 0 0\n"
                             "spin_wall_low 0.4 0.8 0 0\n@ parameters" + blocks)
    validate_gpu_output(args.binary, args.workdir, models)
    for label in models:
        run_case(args.binary, args.workdir, label, models[label], xyz, 2)
    rows = []
    pairs = []
    for repeat in range(args.repeats):
        order = ("off", "on") if repeat % 2 == 0 else ("on", "off")
        pair = {}
        for label in order:
            elapsed = run_case(args.binary, args.workdir, label, models[label], xyz, args.steps)
            row = {"wall": label, "repeat": repeat, "seconds": elapsed,
                   "atom_steps_per_second": 108000 * args.steps / elapsed}
            rows.append(row)
            pair[label] = elapsed
        pairs.append(100 * (pair["on"] / pair["off"] - 1))
    summary = {}
    for label in ("off", "on"):
        times = [row["seconds"] for row in rows if row["wall"] == label]
        summary[label] = {"median_seconds": statistics.median(times),
                          "median_atom_steps_per_second": statistics.median(
                              row["atom_steps_per_second"] for row in rows if row["wall"] == label),
                          "min_seconds": min(times), "max_seconds": max(times)}
    summary["paired_slowdown_percent"] = statistics.median(pairs)
    summary["paired_slowdown_min_percent"] = min(pairs)
    summary["paired_slowdown_max_percent"] = max(pairs)
    summary["atoms"] = 108000
    summary["steps"] = args.steps
    summary["repeats"] = args.repeats
    summary["trials"] = rows
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
