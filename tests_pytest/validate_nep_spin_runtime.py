#!/usr/bin/env python3
"""Standalone NEP_Spin validation for minimal CUDA hosts without pytest."""

import argparse
import concurrent.futures
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "nep_spin" / "spin_chiral_protocol"
GPUMD = Path(os.environ.get("GPUMD_COMMAND", ROOT / "src" / "gpumd"))
GPU_IDS = [
    item.strip()
    for item in os.environ.get("GPUMD_VALIDATOR_GPU_IDS", "").split(",")
    if item.strip()
]
GPU_POOL = queue.Queue()
for gpu_id in GPU_IDS:
    GPU_POOL.put(gpu_id)


def run_case(root, name, model_text, xyz_text, run_text=None):
    case = root / name
    case.mkdir()
    (case / "model.xyz").write_text(xyz_text)
    (case / "nep.txt").write_text(model_text)
    if run_text is None:
        run_text = (
            "potential nep.txt\n"
            "velocity 1\n"
            "ensemble nve\n"
            "time_step 0\n"
            "dump_xyz -1 0 1 result.xyz force potential spin mforce virial\n"
            "run 1\n")
    (case / "run.in").write_text(run_text)
    gpu_id = GPU_POOL.get() if GPU_IDS else None
    try:
        environment = os.environ.copy()
        if gpu_id is not None:
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        result = subprocess.run(
            [str(GPUMD)],
            cwd=case,
            env=environment,
            capture_output=True,
            text=True,
            check=False)
    finally:
        if gpu_id is not None:
            GPU_POOL.put(gpu_id)
    return case, result


def read_frame(path):
    lines = path.read_text().splitlines()
    atom_count = int(lines[0])
    rows = [line.split() for line in lines[-atom_count:]]
    values = [[float(value) for value in row[1:]] for row in rows]
    energy = float(
        re.search(r"\benergy=([-+0-9.eE]+)", lines[-atom_count - 1]).group(1))
    fields = {
        "energy": [energy],
        "position": [row[0:3] for row in values],
        "force": [row[3:6] for row in values],
        "spin": [row[6:9] for row in values],
        "mforce": [row[9:12] for row in values],
        "potential": [row[12] for row in values],
        "atom_virial": [row[13:22] for row in values],
    }
    return fields


def flatten(values):
    if not values or not isinstance(values[0], list):
        return values
    return [item for row in values for item in row]


def max_error(candidate, reference):
    return max(abs(a - b) for a, b in zip(flatten(candidate), flatten(reference)))


def replace_xyz_value(text, atom, field_offset, value):
    lines = text.splitlines()
    fields = lines[atom + 2].split()
    fields[field_offset] = f"{value:.17g}"
    lines[atom + 2] = " ".join(fields)
    return "\n".join(lines) + "\n"


def strain_xyz(text, position_axis, force_axis, epsilon):
    lines = text.splitlines()
    match = re.search(r'Lattice="([^"]+)"', lines[1])
    if match is None:
        raise ValueError("missing Lattice")
    lattice = [float(value) for value in match.group(1).split()]
    # GPUMD stores h by Cartesian row while extxyz lists the three lattice
    # vectors as rows. Thus h[force_axis, :] maps to this extxyz column.
    for lattice_vector in range(3):
        target = 3 * lattice_vector + force_axis
        source = 3 * lattice_vector + position_axis
        lattice[target] += epsilon * lattice[source]
    replacement = 'Lattice="' + " ".join(f"{value:.17g}" for value in lattice) + '"'
    lines[1] = lines[1][:match.start()] + replacement + lines[1][match.end():]
    for atom in range(int(lines[0])):
        fields = lines[atom + 2].split()
        fields[1 + force_axis] = f"{float(fields[1 + force_axis]) + epsilon * float(fields[1 + position_axis]):.17g}"
        lines[atom + 2] = " ".join(fields)
    return "\n".join(lines) + "\n"


def plateau_derivative(root, prefix, model, make_xyz, steps, executor):
    futures = {}
    for index, step in enumerate(steps):
        futures[(index, 1)] = executor.submit(
            run_case, root, f"{prefix}_p_{index}", model, make_xyz(step))
        futures[(index, -1)] = executor.submit(
            run_case, root, f"{prefix}_m_{index}", model, make_xyz(-step))

    energies = {}
    for key, future in futures.items():
        directory, result = future.result()
        energies[key] = read_frame(directory / "result.xyz")["energy"][0]
        if result.returncode:
            raise RuntimeError(
                f"{prefix} finite-difference run failed:\n"
                f"{result.stdout}{result.stderr}")

    derivatives = []
    for index, step in enumerate(steps):
        plus_energy = energies[(index, 1)]
        minus_energy = energies[(index, -1)]
        derivatives.append(-(plus_energy - minus_energy) / (2 * step))
    adjacent = [
        abs(derivatives[index] - derivatives[index + 1])
        for index in range(len(derivatives) - 1)]
    best = min(range(len(adjacent)), key=adjacent.__getitem__) + 1
    return derivatives[best], steps[best], adjacent[best - 1]


def check_self_fd(candidate, analytic, atol=4.0e-3, rtol=1.0e-2):
    error = abs(candidate - analytic)
    tolerance = atol + rtol * abs(analytic)
    return error, error / tolerance


def zero_model(compress, l_max, chiral, basis_size):
    structural_dim = 8
    spin_dim = 2 + 4 * compress + compress
    if l_max >= 1:
        spin_dim += 3 * compress
    if l_max >= 2:
        spin_dim += compress
    if l_max >= 3:
        spin_dim += compress
    if l_max >= 4:
        spin_dim += compress
    spin_dim += 2 * compress
    if l_max >= 1:
        spin_dim += compress
    if chiral:
        spin_dim += min(2, compress) + 2 * compress
    descriptor_dim = structural_dim + spin_dim
    hidden = 1
    ann_count = (descriptor_dim + 2) * hidden + 1
    radial_count = 5 * 7
    angular_count = 3 * 5
    spin_count = compress * (basis_size + 1)
    numeric_count = (
        ann_count + radial_count + angular_count + spin_count + descriptor_dim)
    header = (
        "nep4_spin1 1 Fe\n"
        "spin_mode 1 8\n"
        "spin_baseline 0\n"
        "spin_n_max 0 0\n"
        f"spin_basis_size {basis_size} {basis_size}\n"
        f"spin_l_max {l_max} 0 0\n"
        f"spin_compress {compress}\n"
        "spin_cutoff 6 6\n"
        f"spin_chiral {chiral}\n"
        "spin_scaler 1\n"
        "cutoff 6 5 88 60\n"
        "n_max 4 2\n"
        "basis_size 6 4\n"
        "l_max 1 0 0\n"
        f"ANN {hidden} 0\n")
    return header + "0\n" * numeric_count


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="full",
        help="quick checks one atom's position/spin derivatives; full checks all 3N components")
    parser.add_argument(
        "--workers",
        type=int,
        default=len(GPU_IDS) if GPU_IDS else 1,
        help="maximum concurrent GPUMD subprocesses across the assigned GPU pool")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.workers < 1:
        raise ValueError("--workers must be positive")
    if GPU_IDS and arguments.workers > len(GPU_IDS):
        raise ValueError(
            "--workers cannot exceed GPUMD_VALIDATOR_GPU_IDS count")
    model = (FIXTURE / "nep.txt").read_text()
    oracle = json.loads((FIXTURE / "fp64_oracle.json").read_text())
    report = {
        "oracle": {},
        "parser_positive": {},
        "parser_negative": {},
        "preflight_negative": {},
        "finite_difference": {},
        "mode": arguments.mode,
        "workers": arguments.workers,
        "gpu_count": len(set(GPU_IDS)) if GPU_IDS else 1,
    }
    with (
        tempfile.TemporaryDirectory(prefix="gpumd-nep-spin-") as temporary,
        concurrent.futures.ThreadPoolExecutor(
            max_workers=arguments.workers) as executor,
    ):
        root = Path(temporary)
        frames = {}
        for case, xyz_name in (
            ("large_box", "model_large_box.xyz"),
            ("small_pbc", "model_small_pbc.xyz"),
        ):
            directory, result = run_case(
                root, case, model, (FIXTURE / xyz_name).read_text())
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            frame = read_frame(directory / "result.xyz")
            frames[case] = frame
            report["oracle"][case] = {}
            for field in ("energy", "potential", "force", "mforce", "atom_virial"):
                error = max_error(frame[field], oracle[case][field])
                report["oracle"][case][field] = error
                if error > 2.0e-4:
                    raise AssertionError(f"{case} {field}: {error}")
            for field, atol in (
                ("energy", 1.0e-6),
                ("potential", 1.0e-6),
                ("force", 2.0e-6),
                ("mforce", 2.0e-7),
                ("atom_virial", 5.0e-6),
            ):
                candidate = flatten(frame[field])
                reference = flatten(oracle[case][field])
                ratios = [
                    abs(actual - expected) /
                    (atol + (0.0 if field in ("energy", "potential") else 2.0e-6 * abs(expected)))
                    for actual, expected in zip(candidate, reference)]
                report["oracle"][case][field + "_tight_ratio_max"] = max(ratios)
                if max(ratios) > 1.0:
                    raise AssertionError(
                        f"{case} {field} failed frozen FP64 derivative gate: {max(ratios)}")
            summed = [
                sum(row[component] for row in frame["atom_virial"])
                for component in range(9)]
            virial_error = max_error(summed, oracle[case]["virial"])
            report["oracle"][case]["virial_sum"] = virial_error
            if virial_error > 2.0e-4:
                raise AssertionError(f"{case} total raw9: {virial_error}")

        replicated_dir, replicated_result = run_case(
            root,
            "small_pbc_replicated_large",
            model,
            (FIXTURE / "model_small_pbc.xyz").read_text(),
            "replicate 5 5 5\n"
            "potential nep.txt\n"
            "velocity 1\n"
            "ensemble nve\n"
            "time_step 0\n"
            "dump_xyz -1 0 1 result.xyz force potential spin mforce virial\n"
            "run 1\n")
        if replicated_result.returncode:
            raise RuntimeError(replicated_result.stdout + replicated_result.stderr)
        replicated = read_frame(replicated_dir / "result.xyz")
        small = frames["small_pbc"]
        parity = {
            "energy_per_primitive": abs(
                replicated["energy"][0] / 125.0 - small["energy"][0])
        }
        mapped = {field: [] for field in ("potential", "force", "mforce", "atom_virial")}
        for atom, position in enumerate(replicated["position"]):
            reduced = [coordinate % 4.0 for coordinate in position]
            base = min(
                range(4),
                key=lambda candidate: sum(
                    min(
                        abs(reduced[axis] - small["position"][candidate][axis]),
                        4.0 - abs(reduced[axis] - small["position"][candidate][axis]))
                    ** 2
                    for axis in range(3)))
            for field in mapped:
                actual = replicated[field][atom]
                expected = small[field][base]
                mapped[field].append(
                    abs(actual - expected)
                    if field == "potential"
                    else max_error(actual, expected))
        for field, errors in mapped.items():
            parity[field] = max(errors)
        report["oracle"]["large_small_parity"] = parity
        if max(parity.values()) > 2.0e-4:
            raise AssertionError(f"large/small path parity failed: {parity}")

        mutations = {
            "legacy_keyword": model.replace("nep4_spin1", "nep4_spin", 1),
            "unknown_header": model.replace("spin_scaler 1", "spin_future 1", 1),
            "truncated": model.rsplit("\n", 2)[0] + "\n",
            "extra_numeric": model + "0\n",
            "unsupported_chiral": model.replace("spin_chiral 1", "spin_chiral 2", 1),
            "duplicate_header": model.replace("spin_dof_type Fe", "spin_baseline 0", 1),
            "wrong_count": model.replace("spin_mode 1 10", "spin_mode 1 9", 1),
            "two_layer_ann": model.replace("ANN 30 0", "ANN 30 1", 1),
            "nonfinite": re.sub(
                r"spin_baseline [^\n]+", "spin_baseline nan", model, count=1),
            "zero_spin_cutoff": re.sub(
                r"spin_cutoff [^\n]+", "spin_cutoff 0 5", model, count=1),
        }
        large_xyz = (FIXTURE / "model_large_box.xyz").read_text()
        for index, (name, mutated) in enumerate(mutations.items()):
            _, result = run_case(root, f"negative_{index}", mutated, large_xyz)
            report["parser_negative"][name] = result.returncode
            if result.returncode == 0:
                raise AssertionError(f"parser accepted {name}")

        variants = [
            (compress, l_max, chiral)
            for compress in range(1, 5)
            for l_max in range(5)
            for chiral in (0, 1)
        ]
        if arguments.mode == "quick":
            variants = [
                (1, 0, 0),
                (1, 4, 1),
                (2, 2, 0),
                (3, 3, 1),
                (4, 0, 1),
                (4, 4, 0),
            ]
        variant_futures = []
        for compress, l_max, chiral in variants:
            basis_size = (
                compress - 1 +
                ((compress + l_max + chiral) % (5 - compress)))
            variant_futures.append((
                compress,
                l_max,
                chiral,
                basis_size,
                executor.submit(
                    run_case,
                    root,
                    f"variant_{compress}_{l_max}_{chiral}",
                    zero_model(compress, l_max, chiral, basis_size),
                    large_xyz),
            ))
        for compress, l_max, chiral, basis_size, future in variant_futures:
            _, result = future.result()
            if result.returncode:
                raise AssertionError(
                    f"valid C={compress}, L={l_max}, chiral={chiral}, "
                    f"basis={basis_size} failed:\n{result.stdout}{result.stderr}")
        variant_count = len(variant_futures)
        report["parser_positive"]["shape_variants"] = variant_count

        baseline = frames["large_box"]
        missing_spin_xyz = large_xyz.replace(":spin:R:3", "")
        missing_spin_xyz = "\n".join(
            line if index < 2 else " ".join(line.split()[:4])
            for index, line in enumerate(missing_spin_xyz.splitlines())) + "\n"
        _, missing_result = run_case(root, "missing_spin", model, missing_spin_xyz)
        report["parser_negative"]["missing_spin"] = missing_result.returncode
        if missing_result.returncode == 0:
            raise AssertionError("NEP_Spin accepted model.xyz without spin")

        duplicate_spin_xyz = large_xyz.replace(
            ":spin:R:3", ":spin:R:3:spin:R:3")
        duplicate_spin_xyz = "\n".join(
            line if index < 2 else line + " " + " ".join(line.split()[4:7])
            for index, line in enumerate(duplicate_spin_xyz.splitlines())) + "\n"
        _, duplicate_result = run_case(
            root, "duplicate_spin", model, duplicate_spin_xyz)
        report["parser_negative"]["duplicate_spin"] = duplicate_result.returncode
        if duplicate_result.returncode == 0:
            raise AssertionError("duplicate spin property was accepted")

        nonperiodic_xyz = large_xyz.replace('pbc="T T T"', 'pbc="T T F"')
        _, nonperiodic_result = run_case(
            root, "nonperiodic", model, nonperiodic_xyz)
        report["parser_negative"]["nonperiodic"] = nonperiodic_result.returncode
        if nonperiodic_result.returncode == 0:
            raise AssertionError("NEP_Spin accepted a non-periodic box")

        cutoff_model = model.replace("cutoff 6 5 88 60", "cutoff 2 2 88 60", 1)
        cutoff_xyz_a = (
            '2\nLattice="16 0 0 0 16 0 0 0 16" '
            'Properties=species:S:1:pos:R:3:spin:R:3 pbc="T T T"\n'
            'Fe 0 0 0 1 0.2 0\nFe 4 0 0 0.4 -0.3 0.7\n')
        cutoff_xyz_b = cutoff_xyz_a.replace(
            "Fe 4 0 0 0.4 -0.3 0.7", "Fe 4 0 0 -0.4 0.3 0.7")
        cutoff_a_dir, cutoff_a_result = run_case(
            root, "cutoff_a", cutoff_model, cutoff_xyz_a)
        cutoff_b_dir, cutoff_b_result = run_case(
            root, "cutoff_b", cutoff_model, cutoff_xyz_b)
        if cutoff_a_result.returncode or cutoff_b_result.returncode:
            raise RuntimeError("cutoff separation case failed")
        cutoff_a = read_frame(cutoff_a_dir / "result.xyz")
        cutoff_b = read_frame(cutoff_b_dir / "result.xyz")
        cutoff_response = max_error(
            cutoff_a["mforce"][0], cutoff_b["mforce"][0])
        report["oracle"]["cutoff_separation_response"] = cutoff_response
        if cutoff_response <= 1.0e-6:
            raise AssertionError("spin neighbor outside structural cutoff had no response")

        overflow_model = model.replace("cutoff 6 5 88 60", "cutoff 6 5 1 1", 1)
        overflow_lines = large_xyz.splitlines()
        overflow_lines[0] = "5"
        overflow_lines.append("Fe 2.1 2.2 1.9 0.3 0.4 -0.2")
        overflow_xyz = "\n".join(overflow_lines) + "\n"
        _, overflow_result = run_case(
            root, "neighbor_overflow", overflow_model, overflow_xyz)
        overflow_output = overflow_result.stdout + overflow_result.stderr
        report["parser_negative"]["neighbor_overflow"] = overflow_result.returncode
        if (
            overflow_result.returncode == 0 or
            "required" not in overflow_output or
            "capacity" not in overflow_output
        ):
            raise AssertionError("neighbor overflow did not fail with required/capacity")

        unsupported = {
            "ensemble_langevin": "ensemble nvt_lan 300 300 100\n",
            "dump_observer": "dump_observer 1 1 observer.xyz\n",
            "dump_exyz": "dump_exyz 1\n",
            "deform": "deform 0 0 0\n",
            "mc": "mc canonical 1 1 300\n",
        }
        for index, (name, command) in enumerate(unsupported.items()):
            run_text = "potential nep.txt\n" + command
            _, result = run_case(
                root, f"unsupported_{index}", model, large_xyz, run_text)
            report["preflight_negative"][name] = result.returncode
            if result.returncode == 0:
                raise AssertionError(f"unsupported command accepted: {name}")

        steps = (3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4)
        force_errors = []
        force_ratios = []
        mforce_errors = []
        mforce_ratios = []
        selected_steps = []
        atoms_to_check = range(4) if arguments.mode == "full" else (0,)
        for atom in atoms_to_check:
            base_fields = large_xyz.splitlines()[atom + 2].split()
            for axis in range(3):
                position = float(base_fields[1 + axis])
                derivative, selected, _ = plateau_derivative(
                    root, f"position_{atom}_{axis}", model,
                    lambda h, a=atom, d=axis, value=position:
                        replace_xyz_value(large_xyz, a, 1 + d, value + h),
                    steps,
                    executor)
                error, ratio = check_self_fd(
                    derivative, baseline["force"][atom][axis])
                force_errors.append(error)
                force_ratios.append(ratio)
                selected_steps.append(selected)

                spin = float(base_fields[4 + axis])
                derivative, selected, _ = plateau_derivative(
                    root, f"spin_{atom}_{axis}", model,
                    lambda h, a=atom, d=axis, value=spin:
                        replace_xyz_value(large_xyz, a, 4 + d, value + h),
                    steps,
                    executor)
                error, ratio = check_self_fd(
                    derivative, baseline["mforce"][atom][axis])
                mforce_errors.append(error)
                mforce_ratios.append(ratio)
                selected_steps.append(selected)
        virial_errors = []
        virial_ratios = []
        total_raw9 = [
            sum(row[component] for row in baseline["atom_virial"])
            for component in range(9)]
        for position_axis in range(3):
            for force_axis in range(3):
                component = 3 * position_axis + force_axis
                derivative, selected, _ = plateau_derivative(
                    root, f"strain_{position_axis}_{force_axis}", model,
                    lambda h, p=position_axis, f=force_axis:
                        strain_xyz(large_xyz, p, f, h),
                    steps,
                    executor)
                error, ratio = check_self_fd(derivative, total_raw9[component])
                virial_errors.append(error)
                virial_ratios.append(ratio)
                selected_steps.append(selected)
        report["finite_difference"]["force_max"] = max(force_errors)
        report["finite_difference"]["force_tolerance_ratio_max"] = max(force_ratios)
        report["finite_difference"]["mforce_max"] = max(mforce_errors)
        report["finite_difference"]["mforce_tolerance_ratio_max"] = max(mforce_ratios)
        report["finite_difference"]["raw9_max"] = max(virial_errors)
        report["finite_difference"]["raw9_tolerance_ratio_max"] = max(virial_ratios)
        report["finite_difference"]["selected_steps"] = sorted(set(selected_steps))
        if max(force_ratios + mforce_ratios + virial_ratios) > 1.0:
            raise AssertionError(report["finite_difference"])

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
