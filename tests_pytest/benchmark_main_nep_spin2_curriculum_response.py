#!/usr/bin/env python3
"""Small SNES A/B for the spin2 curriculum and grouped response loss."""

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def make_scan(radii):
    frames = []
    metadata = []
    angles = (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6)
    for group_index, radius in enumerate(radii):
        coupling = math.exp(-radius)
        dmi = 0.2 * coupling
        group = f"r{group_index}"
        for angle in angles:
            cosine = math.cos(angle)
            sine = math.sin(angle)
            energy = coupling * cosine + dmi * sine
            spin_1 = (1.0, 0.0, 0.0)
            spin_2 = (cosine, sine, 0.0)
            mforce_1 = (
                -(coupling * cosine + dmi * sine),
                -(coupling * sine - dmi * cosine),
                0.0,
            )
            mforce_2 = (-coupling, -dmi, 0.0)
            tangent_1 = (0.0, 0.0, 0.0)
            tangent_2 = (-sine, cosine, 0.0)
            spectator_spin_1 = (0.35 + 0.08 * sine, -0.62, 0.48 + 0.03 * cosine)
            spectator_spin_2 = (-0.41, 0.27 + 0.05 * cosine, 0.73 - 0.04 * sine)
            zero = (0.0, 0.0, 0.0)
            header = (
                'Lattice="20 0 0 0 20 0 0 0 20" '
                'Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:'
                'mforce:R:3:spin_tangent:R:3 '
                f'energy={energy:.16e} response_probe=rotation '
                f'response_group={group} response_coordinate={angle:.16e}'
            )
            atom_1 = "Fe 5 5 5 0 0 0 " + " ".join(
                f"{value:.16e}" for value in spin_1 + mforce_1 + tangent_1)
            atom_2 = f"Fe {5 + radius:.16e} 5 5 0 0 0 " + " ".join(
                f"{value:.16e}" for value in spin_2 + mforce_2 + tangent_2)
            atom_3 = (
                f"Fe 5.8 {6.6 + 0.07 * sine:.16e} 5.4 0 0 0 " + " ".join(
                    f"{value:.16e}" for value in spectator_spin_1 + zero + zero))
            atom_4 = (
                f"Fe 6.7 5.5 {6.3 + 0.06 * cosine:.16e} 0 0 0 " + " ".join(
                    f"{value:.16e}" for value in spectator_spin_2 + zero + zero))
            frames.extend(("4", header, atom_1, atom_2, atom_3, atom_4))
            metadata.append({
                "group": group,
                "coordinate": angle,
                "tangents": (tangent_1, tangent_2, zero, zero),
                "target_mforces": (mforce_1, mforce_2, zero, zero),
            })
    return "\n".join(frames) + "\n", metadata


def grouped_response_loss(predicted, metadata):
    groups = {}
    for frame, item in enumerate(metadata):
        generator_prediction = 0.0
        generator_target = 0.0
        for atom in range(4):
            tangent = item["tangents"][atom]
            generator_prediction += sum(
                predicted[4 * frame + atom][component] * tangent[component]
                for component in range(3))
            generator_target += sum(
                item["target_mforces"][atom][component] * tangent[component]
                for component in range(3))
        groups.setdefault(item["group"], []).append(
            (item["coordinate"], generator_prediction, generator_target))

    centered_prediction = []
    centered_target = []
    reliability = []
    mean_prediction = []
    mean_target = []
    for group in sorted(groups):
        members = groups[group]
        mean_x = statistics.fmean(point[0] for point in members)
        mean_p = statistics.fmean(point[1] for point in members)
        mean_t = statistics.fmean(point[2] for point in members)
        mean_prediction.append(mean_p)
        mean_target.append(mean_t)
        x = [point[0] - mean_x for point in members]
        target = [point[2] - mean_t for point in members]
        slope = sum(a * b for a, b in zip(x, target)) / max(
            sum(value * value for value in x), 2.220446049250313e-16)
        signal = [slope * value for value in x]
        noise = [a - b for a, b in zip(target, signal)]
        signal_power = statistics.fmean(value * value for value in signal)
        noise_power = statistics.fmean(value * value for value in noise)
        score = signal_power / (signal_power + noise_power + 1.1920929e-7)
        centered_prediction.extend(point[1] - mean_p for point in members)
        centered_target.extend(target)
        reliability.extend(max(0.05, score) for _ in members)

    def huber(value):
        absolute = abs(value)
        return 0.5 * value * value if absolute <= 1.0 else absolute - 0.5

    scale = max(
        math.sqrt(statistics.fmean(value * value for value in centered_target)),
        64.0 * 1.1920929e-7,
    )
    shape = sum(
        weight * huber((prediction - target) / scale)
        for prediction, target, weight in zip(
            centered_prediction, centered_target, reliability)
    ) / max(1.0, sum(reliability))
    mean_scale = max(
        math.sqrt(statistics.fmean(value * value for value in mean_target)),
        64.0 * 1.1920929e-7,
    )
    mean = statistics.fmean(huber((prediction - target) / mean_scale)
                            for prediction, target in zip(mean_prediction, mean_target))
    return shape + 0.25 * mean


def run_nep(binary, directory):
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    result = subprocess.run(
        [str(binary)], cwd=directory, env=environment,
        capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result


def predict_response(binary, model, xyz, metadata, directory):
    directory.mkdir()
    shutil.copy(model, directory / "nep.txt")
    (directory / "train.xyz").write_text(xyz)
    (directory / "nep.in").write_text(
        "type 1 Fe\nversion 4\nspin_mode 2\nspin_dof_type Fe\n"
        "cutoff 6 5\nn_max 0 0\nbasis_size 0 0\nl_max 2 0 0\nneuron 4\n"
        "spin_compress 1\nspin_basis_size 8 0\nspin_l_max 2 0 0\n"
        "spin_cutoff 6 6\nspin_order 3\nspin_soc 1\nprediction 1\n")
    run_nep(binary, directory)
    predicted = [
        tuple(float(value) for value in line.split()[:3])
        for line in (directory / "mforce_train.out").read_text().splitlines()
    ]
    return grouped_response_loss(predicted, metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nep", type=Path, default=ROOT / "src" / "nep")
    parser.add_argument("--generations", type=int, default=800)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    binary = args.nep.resolve()
    train_xyz, train_metadata = make_scan((2.0, 2.8))
    valid_xyz, valid_metadata = make_scan((2.3, 2.5))
    variants = {
        "baseline": (0, 0.0),
        "curriculum": (1, 0.0),
        "response": (0, 1.0),
        "curriculum_response": (1, 1.0),
    }
    raw = {name: [] for name in variants}
    with tempfile.TemporaryDirectory(prefix="spin2-snes-ab-") as temporary:
        root = Path(temporary)
        for repeat in range(args.repeats):
            for name, (curriculum, response_weight) in variants.items():
                case = root / f"{name}-{repeat}"
                case.mkdir()
                (case / "train.xyz").write_text(train_xyz)
                (case / "nep.in").write_text(
                    "type 1 Fe\nversion 4\nspin_mode 2\nspin_dof_type Fe\n"
                    "cutoff 6 5\nn_max 0 0\nbasis_size 0 0\nl_max 2 0 0\nneuron 4\n"
                    "spin_compress 1\nspin_basis_size 8 0\nspin_l_max 2 0 0\n"
                    "spin_cutoff 6 6\nspin_order 3\nspin_soc 1\n"
                    f"spin_curriculum {curriculum}\n"
                    f"lambda_spin_response {response_weight}\n"
                    "lambda_1 0.0001\nlambda_2 0.0001\n"
                    "lambda_e 1\nlambda_f 0\nlambda_v 0\nlambda_m 5\nlambda_tau 0\n"
                    "population 30\nbatch 14 1\n"
                    f"generation {args.generations}\noutput_interval {args.generations}\n"
                    f"save_potential {args.generations} 0 0\n")
                run_nep(binary, case)
                loss_columns = [float(value) for value in
                                (case / "loss.out").read_text().splitlines()[-1].split()]
                train_response = predict_response(
                    binary, case / "nep.txt", train_xyz, train_metadata,
                    case / "predict-train")
                valid_response = predict_response(
                    binary, case / "nep.txt", valid_xyz, valid_metadata,
                    case / "predict-valid")
                raw[name].append({
                    "total_loss": loss_columns[1],
                    "energy_rmse": loss_columns[4],
                    "mforce_rmse": loss_columns[7],
                    "tau_rmse": loss_columns[8],
                    "train_response_loss": train_response,
                    "valid_response_loss": valid_response,
                })
    summary = {}
    for name, runs in raw.items():
        summary[name] = {
            metric: statistics.median(run[metric] for run in runs)
            for metric in runs[0]
        }
    print(json.dumps({"raw": raw, "median": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
