#!/usr/bin/env python3
"""Validate spin3 training, checkpoint prediction, and GPUMD runtime parity."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEP = Path(os.environ.get("NEP_COMMAND", ROOT / "src" / "nep"))
GPUMD = Path(os.environ.get("GPUMD_COMMAND", ROOT / "src" / "gpumd"))

NEP_IN = """\
type 2 Fe Ge
version 4
spin_mode 3
spin_mforce_mode full
spin_dof_type Fe
spin_env_type Fe Ge
cutoff 6.0 5.0
n_max 0 0
basis_size 0 0
l_max 2 0 0
neuron 4
spin_compress 2
spin_basis_size 8 0
spin_l_max 2 0 0
spin_cutoff 6.0
spin_order 3
spin_soc 1
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
Lattice="24 0 0 0 24 0 0 0 24" Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:mforce:R:3 energy=-4.0 virial="0 0 0 0 0 0 0 0 0"
Fe 1.0 1.0 1.0 0 0 0 1.0 0.1 0.0 0.2 -0.1 0.05
Fe 2.2 1.1 1.0 0 0 0 0.8 -0.2 0.1 -0.1 0.2 -0.05
Ge 1.4 2.3 1.1 0 0 0 0.0 0.0 0.0 9.0 9.0 9.0
Ge 2.8 2.1 1.4 0 0 0 0.1 0.0 0.0 9.0 9.0 9.0
4
Lattice="24 0 0 0 24 0 0 0 24" Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:mforce:R:3 energy=-2.0 virial="0 0 0 0 0 0 0 0 0"
Fe 1.2 1.0 1.0 0 0 0 0.9 0.0 0.2 0.1 0.0 -0.2
Ge 2.1 1.4 1.0 0 0 0 0.0 0.0 0.0 8.0 8.0 8.0
Ge 1.5 2.4 1.2 0 0 0 0.0 0.1 0.0 8.0 8.0 8.0
Ge 2.8 2.5 1.5 0 0 0 0.0 0.0 0.1 8.0 8.0 8.0
4
Lattice="24 0 0 0 24 0 0 0 24" Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:mforce:R:3 energy=-6.0 virial="0 0 0 0 0 0 0 0 0"
Fe 1.0 1.0 1.0 0 0 0 1.0 0.0 0.0 0.0 0.1 0.0
Fe 2.0 1.0 1.0 0 0 0 0.8 0.1 0.0 0.1 0.0 0.0
Fe 1.5 2.0 1.0 0 0 0 0.9 0.0 0.1 0.0 0.0 0.1
Ge 2.5 2.0 1.2 0 0 0 0.0 0.0 0.0 7.0 7.0 7.0
"""


def run(binary, cwd):
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return subprocess.run(
        [str(binary)], cwd=cwd, env=env, capture_output=True, text=True,
        check=False)


def write_case(directory, nep_in=NEP_IN, train_xyz=TRAIN_XYZ):
    directory.mkdir()
    (directory / "nep.in").write_text(nep_in)
    (directory / "train.xyz").write_text(train_xyz)


def response_xyz():
    lines = TRAIN_XYZ.splitlines()
    output = []
    atom_count = int(lines[0])
    for coordinate in (-1.0, 0.0, 1.0):
        output.append(lines[0])
        output.append(
            f"{lines[1]} response_probe=rotation response_group=scan-a "
            f"response_coordinate={coordinate}")
        for atom in range(atom_count):
            fields = lines[2 + atom].split()
            if atom == 0:
                fields[7:10] = [
                    f"{math.cos(coordinate):.16e}",
                    f"{math.sin(coordinate):.16e}",
                    "0.0",
                ]
            output.append(" ".join(fields))
    return "\n".join(output) + "\n"


def columns(path, count, limit=None):
    rows = [
        [float(item) for item in line.split()[:count]]
        for line in path.read_text().splitlines()]
    return rows if limit is None else rows[:limit]


def flatten(values):
    if not isinstance(values, list):
        return [values]
    result = []
    for value in values:
        result.extend(flatten(value))
    return result


def maximum_error(left, right):
    left = flatten(left)
    right = flatten(right)
    if len(left) != len(right):
        raise AssertionError(f"shape mismatch: {len(left)} != {len(right)}")
    return max(abs(a - b) for a, b in zip(left, right))


def parity_tolerance(left, right):
    scale = max(abs(value) for value in flatten(left) + flatten(right))
    return 5.0e-5 + 5.0e-6 * scale


def validate_parser(root):
    root.mkdir()
    cases = {
        "invalid_basis": NEP_IN.replace("spin_basis_size 8 0", "spin_basis_size 7 0"),
        "invalid_second_basis": NEP_IN.replace("spin_basis_size 8 0", "spin_basis_size 8 1"),
        "invalid_order": NEP_IN.replace("spin_order 3", "spin_order 4"),
        "invalid_soc": NEP_IN.replace("spin_soc 1", "spin_soc 2"),
        "removed_spin_mode_1": NEP_IN.replace("spin_mode 3", "spin_mode 1"),
        "removed_spin_mode_2": NEP_IN.replace("spin_mode 3", "spin_mode 2"),
        "missing_mforce_mode": NEP_IN.replace("spin_mforce_mode full\n", ""),
        "invalid_mforce_mode": NEP_IN.replace(
            "spin_mforce_mode full", "spin_mforce_mode radial"),
        "invalid_spin_cutoff_arity": NEP_IN.replace(
            "spin_cutoff 6.0", "spin_cutoff 4.0 5.0 6.0"),
        "removed_spin_chiral": NEP_IN.replace("spin_soc 1", "spin_soc 1\nspin_chiral 1"),
        "curriculum_requires_o3": NEP_IN.replace("spin_order 3", "spin_order 2").replace(
            "lambda_tau 0.5", "lambda_tau 0.5\nspin_curriculum 1"),
        "response_requires_full_batch": NEP_IN.replace(
            "lambda_tau 0.5", "lambda_tau 0.5\nlambda_spin_response 0.3"),
    }
    results = {}
    for name, text in cases.items():
        case = root / name
        write_case(case, text.replace("generation 1", "generation 0"))
        result = run(NEP, case)
        results[name] = result.returncode
        if result.returncode == 0:
            raise AssertionError(f"{name} did not fail closed")
    tangent_case = root / "spin_tangent_is_not_a_label"
    tangent_input = NEP_IN.replace(
        "lambda_tau 0.5", "lambda_tau 0.5\nlambda_spin_response 0.3",
    ).replace("batch 3", "batch 3 1").replace("generation 1", "generation 0")
    tangent_xyz = response_xyz().replace(
        "mforce:R:3", "mforce:R:3:spin_tangent:R:3")
    write_case(tangent_case, tangent_input, tangent_xyz)
    result = run(NEP, tangent_case)
    results["spin_tangent_is_not_a_label"] = result.returncode
    if result.returncode == 0 or "not an input label" not in result.stdout + result.stderr:
        raise AssertionError("spin_tangent input was not rejected explicitly")
    return results


def validate_response_training(root):
    nep_in = NEP_IN.replace(
        "lambda_tau 0.5",
        "lambda_tau 0.5\nspin_curriculum 1\nlambda_spin_response 0.3",
    ).replace("batch 3", "batch 3 1").replace("generation 1", "generation 3")
    write_case(root, nep_in, response_xyz())
    result = run(NEP, root)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    if "O3 curriculum generation 1: perturbation_scale=0.000000" not in result.stdout:
        raise AssertionError("missing closed-O3 curriculum state")
    if "O3 curriculum generation 2: perturbation_scale=1.000000" not in result.stdout:
        raise AssertionError("missing full-O3 curriculum state")
    loss = (root / "loss.out").read_text().splitlines()[-1]
    if re.search(r"\b(?:nan|inf)\b", loss, re.IGNORECASE):
        raise AssertionError("response training produced a non-finite loss")
    return {"training_loss": loss, "curriculum_schedule": "0 -> 1"}


def validate_training_and_runtime(
        root, enable_zbl=False, spin_compress=2, typewise_spin_cutoff=False,
        mforce_mode="full", all_spin_types=False):
    root.mkdir()
    training = root / "training"
    nep_in = NEP_IN.replace("version 4", "version 4\nzbl 2.5") \
        if enable_zbl else NEP_IN
    nep_in = nep_in.replace(
        "spin_compress 2", f"spin_compress {spin_compress}")
    nep_in = nep_in.replace("spin_mforce_mode full", f"spin_mforce_mode {mforce_mode}")
    if all_spin_types:
        nep_in = nep_in.replace("spin_dof_type Fe\n", "spin_dof_type Fe Ge\n")
    if typewise_spin_cutoff:
        nep_in = nep_in.replace("spin_cutoff 6.0", "spin_cutoff 5.0 7.0")
    write_case(training, nep_in)
    result = run(NEP, training)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)

    checkpoint = (training / "nep.txt").read_text()
    expected_header = "nep4_spin3_zbl" if enable_zbl else "nep4_spin3"
    if not checkpoint.startswith(
            f"{expected_header} 2 Fe Ge \nspin_mode 3 11\n"):
        raise AssertionError("invalid counted spin3 checkpoint")
    if enable_zbl and "\nzbl 1.25 2.5\ncutoff " not in checkpoint:
        raise AssertionError("spin3 ZBL checkpoint is missing its zbl line")
    projection_size = 4 * spin_compress * spin_compress
    required = (
        "spin_basis_size 8\n",
        "spin_l_max 2\n",
        "spin_order 3\n",
        "spin_soc 1\n",
        "spin_scaler 1\n",
        "spin_dof_type Fe Ge\n" if all_spin_types else "spin_dof_type Fe\n",
        "spin_env_type Fe Ge\n",
    )
    for line in required:
        if line not in checkpoint:
            raise AssertionError(f"missing checkpoint line: {line.strip()}")
    for line in (
            f"spin_compress {spin_compress}\n",
            f"spin_projection_size {projection_size}\n"):
        if line not in checkpoint:
            raise AssertionError(f"missing checkpoint line: {line.strip()}")
    expected_spin_cutoff = (
        "spin_cutoff 5.0000000000000000e+00 7.0000000000000000e+00\n"
        if typewise_spin_cutoff else
        "spin_cutoff 6.0000000000000000e+00\n")
    if expected_spin_cutoff not in checkpoint:
        raise AssertionError("Spin3 checkpoint did not preserve spin_cutoff arity")

    loss = (training / "loss.out").read_text().splitlines()[-1]
    if re.search(r"\b(?:nan|inf)\b", loss, re.IGNORECASE):
        raise AssertionError("spin3 training produced a non-finite loss")
    loss_values = [float(value) for value in loss.split()]
    regularization_only = loss_values[2] + loss_values[3]
    if loss_values[1] <= regularization_only + 1.0e-4:
        raise AssertionError(
            "generation-one total loss does not contain the training fitness")

    spin_descriptor_dims = {1: 21, 2: 55, 3: 91}
    descriptor_dim = 3 + spin_descriptor_dims[spin_compress]
    q_scaler = [
        float(value) for value in checkpoint.splitlines()[-descriptor_dim:]]
    # The production magnetic scaler downscales but never amplifies channels.
    spin_scaler = q_scaler[3:]
    if any(not math.isfinite(value) or value <= 0 for value in q_scaler):
        raise AssertionError("spin3 q_scaler must be positive and finite")
    if max(spin_scaler) > 1.0:
        raise AssertionError(
            f"spin3 magnetic q_scaler must not amplify channels: max={max(spin_scaler)}")

    prediction = root / "prediction"
    prediction.mkdir()
    shutil.copy(training / "nep.txt", prediction / "nep.txt")
    shutil.copy(training / "train.xyz", prediction / "train.xyz")
    prediction_in = re.sub(
        r"^(population|generation|output_interval|save_potential).*$",
        "", nep_in, flags=re.MULTILINE)
    prediction_in = prediction_in.replace(
        "version 4", "version 4\nprediction 1")
    (prediction / "nep.in").write_text(prediction_in)
    result = run(NEP, prediction)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)

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
    atom_virial = [row[13:22] for row in rows]
    runtime_virial = [
        sum(row[index] for row in atom_virial) / atom_count
        for index in (0, 4, 8, 1, 5, 6)]

    main_energy = float(
        (prediction / "energy_train.out").read_text().split()[0])
    main_force = columns(prediction / "force_train.out", 3, atom_count)
    main_mforce = columns(prediction / "mforce_train.out", 3, atom_count)
    main_virial = columns(prediction / "virial_train.out", 6, 1)[0]
    report = {
        "energy_per_atom": abs(main_energy - runtime_energy),
        "force": maximum_error(main_force, runtime_force),
        "mforce": maximum_error(main_mforce, runtime_mforce),
        "virial_per_atom": maximum_error(main_virial, runtime_virial),
        "ge_mforce_max": max(
            abs(value) for row in runtime_mforce[2:] for value in row),
        "all_spin_types": all_spin_types,
        "training_loss": loss,
        "q_scaler_max": max(q_scaler),
        "zbl": enable_zbl,
        "spin_compress": spin_compress,
        "typewise_spin_cutoff": typewise_spin_cutoff,
        "mforce_mode": mforce_mode,
        "descriptor_dim": descriptor_dim,
    }
    for name in ("energy_per_atom", "force", "mforce", "virial_per_atom"):
        reference = {
            "energy_per_atom": [main_energy, runtime_energy],
            "force": [main_force, runtime_force],
            "mforce": [main_mforce, runtime_mforce],
            "virial_per_atom": [main_virial, runtime_virial],
        }[name]
        tolerance = parity_tolerance(*reference)
        report[f"{name}_tolerance"] = tolerance
        if report[name] > tolerance:
            raise AssertionError(
                f"{name} parity failed: {report[name]} > {tolerance}")
    if not all_spin_types and report["ge_mforce_max"] > 1.0e-12:
        raise AssertionError("inactive spin DOF received a public mforce")
    if all_spin_types and report["ge_mforce_max"] <= 1.0e-12:
        raise AssertionError("active Ge spin DOF was incorrectly masked")
    return report


def main():
    global NEP, GPUMD
    parser = argparse.ArgumentParser()
    parser.add_argument("--nep", type=Path, default=NEP)
    parser.add_argument("--gpumd", type=Path, default=GPUMD)
    args = parser.parse_args()
    NEP = args.nep.resolve()
    GPUMD = args.gpumd.resolve()
    with tempfile.TemporaryDirectory(prefix="gpumd-main-nep-spin3-") as tmp:
        root = Path(tmp)
        report = {
            "parser_negative_exit_codes": validate_parser(root / "parser"),
            "response_training": validate_response_training(root / "response"),
            "training_runtime": validate_training_and_runtime(root / "correctness"),
            "o3c3_training_runtime": validate_training_and_runtime(
                root / "o3c3", spin_compress=3),
            "all_types_training_runtime": validate_training_and_runtime(
                root / "all-types", all_spin_types=True),
            "all_types_o3c3_training_runtime": validate_training_and_runtime(
                root / "all-types-o3c3", spin_compress=3, all_spin_types=True),
            "rank_one_training_runtime": validate_training_and_runtime(
                root / "rank-one", spin_compress=1,
                typewise_spin_cutoff=True, mforce_mode="transverse"),
            "zbl_training_runtime": validate_training_and_runtime(
                root / "zbl", enable_zbl=True),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
