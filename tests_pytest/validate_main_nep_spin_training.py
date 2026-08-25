#!/usr/bin/env python3
"""Standalone 4090 validation for the main_nep Spin NEP training path."""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NEP = Path(os.environ.get("NEP_COMMAND", ROOT / "src" / "nep"))
GPUMD = Path(os.environ.get("GPUMD_COMMAND", ROOT / "src" / "gpumd"))
CUDA_LIB = "/home/dwhe/opt/cuda-12.8/lib64"

NEP_IN = """\
type 2 Fe O
version 4
spin_mode 1
spin_dof_type Fe
spin_env_type Fe O
cutoff 3.0 3.0
n_max 0 0
basis_size 0 0
l_max 2 0 0
neuron 4
spin_chiral 0
spin_compress 1
spin_n_max 0 0
spin_basis_size 0 0
spin_l_max 1 0 0
spin_cutoff 3.5 2.5
lambda_m 1.0
lambda_tau 0.5
population 10
batch 3
generation 1
output_interval 1
save_potential 1 0 0
"""

TRAIN_XYZ = """\
4
Lattice="12 0 0 0 12 0 0 0 12" Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:mforce:R:3 energy=-4.0 virial="0 0 0 0 0 0 0 0 0"
Fe 1.0 1.0 1.0 0 0 0 1.0 0.1 0.0 0.2 -0.1 0.05
Fe 2.2 1.1 1.0 0 0 0 0.8 -0.2 0.1 -0.1 0.2 -0.05
O  1.4 2.3 1.1 0 0 0 0.0 0.0 0.0 9.0 9.0 9.0
O  2.8 2.1 1.4 0 0 0 0.1 0.0 0.0 9.0 9.0 9.0
4
Lattice="12 0 0 0 12 0 0 0 12" Properties=species:S:1:pos:R:3:force:R:3:moments:R:3:magnetic_forces:R:3 energy=-2.0 virial="0 0 0 0 0 0 0 0 0"
Fe 1.2 1.0 1.0 0 0 0 0.9 0.0 0.2 0.1 0.0 -0.2
O  2.1 1.4 1.0 0 0 0 0.0 0.0 0.0 8.0 8.0 8.0
O  1.5 2.4 1.2 0 0 0 0.0 0.1 0.0 8.0 8.0 8.0
O  2.8 2.5 1.5 0 0 0 0.0 0.0 0.1 8.0 8.0 8.0
4
Lattice="12 0 0 0 12 0 0 0 12" Properties=species:S:1:pos:R:3:force:R:3:spin:R:3 energy=-6.0 virial="0 0 0 0 0 0 0 0 0"
Fe 1.0 1.0 1.0 0 0 0 1.0 0.0 0.0
Fe 2.0 1.0 1.0 0 0 0 0.8 0.1 0.0
Fe 1.5 2.0 1.0 0 0 0 0.9 0.0 0.1
O  2.5 2.0 1.2 0 0 0 0.0 0.0 0.0
"""


def environment():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["LD_LIBRARY_PATH"] = CUDA_LIB + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def run(binary, cwd):
    return subprocess.run(
        [str(binary)],
        cwd=cwd,
        env=environment(),
        capture_output=True,
        text=True,
        check=False)


def write_training_case(directory, nep_in=NEP_IN, train_xyz=TRAIN_XYZ):
    directory.mkdir()
    (directory / "nep.in").write_text(nep_in)
    (directory / "train.xyz").write_text(train_xyz)


def predicted_columns(path, count, limit=None):
    rows = [
        [float(item) for item in line.split()[:count]]
        for line in path.read_text().splitlines()]
    return rows if limit is None else rows[:limit]


def maximum_error(left, right):
    if isinstance(left[0], list):
        return max(
            abs(a - b)
            for row_a, row_b in zip(left, right)
            for a, b in zip(row_a, row_b))
    return max(abs(a - b) for a, b in zip(left, right))


def validate_parser(root):
    root.mkdir()
    cases = {
        "unknown_dof": NEP_IN.replace("spin_dof_type Fe", "spin_dof_type X"),
        "dof_not_env": NEP_IN.replace("spin_env_type Fe O", "spin_env_type O"),
        "duplicate_lambda_alias": NEP_IN.replace(
            "lambda_tau 0.5", "lambda_mforce 1.0\nlambda_tau 0.5"),
        "legacy_spin_type": NEP_IN.replace(
            "spin_dof_type Fe", "spin_type Fe"),
        "unsupported_zbl": NEP_IN.replace(
            "spin_mode 1", "spin_mode 1\nzbl 1.0 2.0"),
    }
    results = {}
    for name, text in cases.items():
        case = root / name
        write_training_case(case, text.replace("generation 1", "generation 0"))
        result = run(NEP, case)
        results[name] = result.returncode
        if result.returncode == 0:
            raise AssertionError(f"{name} did not fail closed")

    lines = TRAIN_XYZ.splitlines()
    header = lines[1].replace(":spin:R:3", "")
    atoms = [
        " ".join(line.split()[:7] + line.split()[10:])
        for line in lines[2:6]]
    missing_spin = "\n".join([lines[0], header] + atoms) + "\n"
    case = root / "missing_spin"
    write_training_case(
        case, NEP_IN.replace("generation 1", "generation 0"), missing_spin)
    result = run(NEP, case)
    results["missing_spin"] = result.returncode
    if result.returncode == 0:
        raise AssertionError("missing spin did not fail closed")
    return results


def validate_training_and_runtime(root):
    root.mkdir()
    training = root / "training"
    write_training_case(training)
    result = run(NEP, training)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    checkpoint = (training / "nep.txt").read_text()
    if not checkpoint.startswith("nep4_spin1 2 Fe O \nspin_mode 1 10\n"):
        raise AssertionError("invalid counted Spin NEP checkpoint")
    baseline = [
        float(value)
        for value in re.search(
            r"^spin_baseline (.+)$", checkpoint, re.MULTILINE).group(1).split()]
    if maximum_error(baseline, [-2.0, 0.0]) > 1.0e-12:
        raise AssertionError(f"wrong total-energy baseline: {baseline}")

    prediction = root / "prediction"
    prediction.mkdir()
    shutil.copy(training / "nep.txt", prediction / "nep.txt")
    shutil.copy(training / "train.xyz", prediction / "train.xyz")
    prediction_in = re.sub(
        r"^(population|generation|output_interval|save_potential).*$",
        "",
        NEP_IN,
        flags=re.MULTILINE)
    prediction_in = prediction_in.replace(
        "version 4", "version 4\nprediction 1").replace(
            "lambda_m 1.0", "lambda_mforce 1.0")
    (prediction / "nep.in").write_text(prediction_in)
    result = run(NEP, prediction)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    if len((prediction / "mforce_train.out").read_text().splitlines()) != 8:
        raise AssertionError("unlabeled mforce frame was written to mforce_train.out")

    runtime = root / "runtime"
    runtime.mkdir()
    shutil.copy(training / "nep.txt", runtime / "nep.txt")
    (runtime / "model.xyz").write_text(
        "\n".join(TRAIN_XYZ.splitlines()[:6]) + "\n")
    (runtime / "run.in").write_text(
        "potential nep.txt\n"
        "velocity 1\n"
        "ensemble nve\n"
        "time_step 0\n"
        "dump_xyz -1 0 1 result.xyz force potential spin mforce virial\n"
        "run 1\n")
    result = run(GPUMD, runtime)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)

    lines = (runtime / "result.xyz").read_text().splitlines()
    atom_count = int(lines[-6])
    header = lines[-5]
    rows = [
        [float(item) for item in line.split()[1:]]
        for line in lines[-atom_count:]]
    runtime_energy = (
        float(re.search(r"\benergy=([-+0-9.eE]+)", header).group(1)) /
        atom_count)
    runtime_force = [row[3:6] for row in rows]
    runtime_mforce = [row[9:12] for row in rows]
    raw9 = [row[13:22] for row in rows]
    runtime_virial = [
        sum(row[index] for row in raw9) / atom_count
        for index in (0, 4, 8, 1, 5, 6)]

    main_energy = float(
        (prediction / "energy_train.out").read_text().split()[0])
    main_force = predicted_columns(
        prediction / "force_train.out", 3, atom_count)
    main_mforce = predicted_columns(
        prediction / "mforce_train.out", 3, atom_count)
    main_virial = predicted_columns(
        prediction / "virial_train.out", 6, 1)[0]
    report = {
        "baseline": baseline,
        "energy_per_atom": abs(main_energy - runtime_energy),
        "force": maximum_error(main_force, runtime_force),
        "mforce": maximum_error(main_mforce, runtime_mforce),
        "virial_per_atom": maximum_error(main_virial, runtime_virial),
        "inactive_mforce": max(
            abs(value) for row in runtime_mforce[2:] for value in row),
        "training_loss": (training / "loss.out").read_text().splitlines()[-1],
    }
    for name in ("energy_per_atom", "force", "mforce", "virial_per_atom"):
        if report[name] > 3.0e-5:
            raise AssertionError(f"{name} parity failed: {report[name]}")
    if report["inactive_mforce"] > 1.0e-12:
        raise AssertionError("inactive spin DOF received a public mforce")
    return report


def validate_chiral_training(root):
    root.mkdir()
    write_training_case(
        root / "training", NEP_IN.replace("spin_chiral 0", "spin_chiral 1"))
    result = run(NEP, root / "training")
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    checkpoint = (root / "training" / "nep.txt").read_text()
    if "\nspin_chiral 1\n" not in checkpoint:
        raise AssertionError("chiral training checkpoint lost spin_chiral")
    loss = (root / "training" / "loss.out").read_text().splitlines()[-1]
    if re.search(r"\b(?:nan|inf)\b", loss, re.IGNORECASE):
        raise AssertionError("chiral training produced a non-finite loss")
    return {"training_loss": loss}


def validate_full_batch_reporting(root):
    root.mkdir()
    lines = TRAIN_XYZ.splitlines()
    train_xyz = "\n".join(lines[:12]) + "\n"
    nep_in = NEP_IN.replace("batch 3", "batch 1 1")
    training = root / "training"
    write_training_case(training, nep_in, train_xyz)
    result = run(NEP, training)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)

    prediction = root / "prediction"
    prediction.mkdir()
    shutil.copy(training / "nep.txt", prediction / "nep.txt")
    shutil.copy(training / "train.xyz", prediction / "train.xyz")
    prediction_in = re.sub(
        r"^(population|generation|output_interval|save_potential).*$",
        "",
        nep_in,
        flags=re.MULTILINE)
    prediction_in = prediction_in.replace(
        "version 4", "version 4\nprediction 1")
    (prediction / "nep.in").write_text(prediction_in)
    result = run(NEP, prediction)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)

    mforce_rows = [
        [float(value) for value in line.split()]
        for line in (prediction / "mforce_train.out").read_text().splitlines()]
    frame_atom_lines = [lines[2:6], lines[8:12]]
    offset = 0
    mforce_rmse = []
    tau_rmse = []
    for atoms in frame_atom_lines:
        mforce_error_squared = 0.0
        tau_error_squared = 0.0
        active_count = 0
        for atom, row in zip(atoms, mforce_rows[offset:offset + len(atoms)]):
            tokens = atom.split()
            if tokens[0] != "Fe":
                continue
            spin = [float(value) for value in tokens[7:10]]
            residual = [
                row[component] - row[component + 3]
                for component in range(3)]
            tau = [
                spin[1] * residual[2] - spin[2] * residual[1],
                spin[2] * residual[0] - spin[0] * residual[2],
                spin[0] * residual[1] - spin[1] * residual[0],
            ]
            mforce_error_squared += sum(value * value for value in residual)
            tau_error_squared += sum(value * value for value in tau)
            active_count += 1
        offset += len(atoms)
        mforce_rmse.append(
            (mforce_error_squared / (3 * active_count)) ** 0.5)
        tau_rmse.append((tau_error_squared / (3 * active_count)) ** 0.5)

    expected_mforce = (
        sum(value * value for value in mforce_rmse) / len(mforce_rmse)) ** 0.5
    expected_tau = (
        sum(value * value for value in tau_rmse) / len(tau_rmse)) ** 0.5
    loss = [
        float(value)
        for value in (training / "loss.out").read_text().splitlines()[-1].split()]
    report = {
        "mforce": abs(loss[7] - expected_mforce),
        "tau": abs(loss[8] - expected_tau),
    }
    if report["mforce"] > 6.0e-6 or report["tau"] > 6.0e-6:
        raise AssertionError(f"full-batch report mismatch: {report}")
    return report


def main():
    global NEP, GPUMD
    parser = argparse.ArgumentParser()
    parser.add_argument("--nep", type=Path, default=NEP)
    parser.add_argument("--gpumd", type=Path, default=GPUMD)
    args = parser.parse_args()
    NEP = args.nep.resolve()
    GPUMD = args.gpumd.resolve()
    with tempfile.TemporaryDirectory(prefix="gpumd-main-nep-spin-") as tmp:
        root = Path(tmp)
        report = {
            "parser_negative_exit_codes": validate_parser(root / "parser"),
            "training_runtime": validate_training_and_runtime(root / "correctness"),
            "chiral_training": validate_chiral_training(root / "chiral"),
            "full_batch_reporting": validate_full_batch_reporting(
                root / "full_batch_reporting"),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
