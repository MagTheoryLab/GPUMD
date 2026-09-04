#!/usr/bin/env python3
"""Standalone SIB validation for minimal CUDA hosts without pytest/numpy."""

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "nep_spin3"
GPUMD = Path(os.environ.get("GPUMD_COMMAND", ROOT / "src" / "gpumd"))


def base_model():
    lines = (FIXTURE / "model_large_box.xyz").read_text().splitlines()
    lines[1] = lines[1].replace(
        "Properties=species:S:1:pos:R:3:spin:R:3",
        "Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3")
    for index in range(2, len(lines)):
        fields = lines[index].split()
        lines[index] = " ".join(fields[:4] + ["0", "0", "0"] + fields[4:])
    return "\n".join(lines) + "\n"


def replace_spins(model_text, spins):
    lines = model_text.splitlines()
    for row, spin in enumerate(spins, start=2):
        fields = lines[row].split()
        fields[-3:] = [f"{value:.17g}" for value in spin]
        lines[row] = " ".join(fields)
    return "\n".join(lines) + "\n"


def run_case(root, name, run_input, model_text=None):
    case = root / name
    case.mkdir()
    shutil.copy(FIXTURE / "nep4_spin3_o3c2_uniform.nep", case / "nep.txt")
    (case / "model.xyz").write_text(model_text or base_model())
    (case / "run.in").write_text(run_input)
    result = subprocess.run(
        [str(GPUMD)], cwd=case, capture_output=True, text=True, check=False)
    return case, result


def read_last_frame(path):
    lines = path.read_text().splitlines()
    number_of_atoms = int(lines[0])
    values = [
        [float(value) for value in line.split()[1:]]
        for line in lines[-number_of_atoms:]
    ]
    return {
        "spin": [row[4:7] for row in values],
        "mforce": [row[7:10] for row in values],
    }


def evaluate(root, name, model_text):
    case, result = run_case(
        root,
        name,
        "potential nep.txt\n"
        "ensemble nve\n"
        "time_step 0\n"
        "dump_xyz -1 0 1 state.xyz mass spin mforce\n"
        "run 1\n",
        model_text)
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    return read_last_frame(case / "state.xyz")


def cross(left, right):
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def cayley(direction, omega):
    half = [0.5 * value for value in omega]
    half_squared = sum(value * value for value in half)
    half_dot = sum(a * b for a, b in zip(half, direction))
    half_cross = cross(half, direction)
    return [
        ((1.0 - half_squared) * direction[k]
         + 2.0 * half_cross[k] + 2.0 * half[k] * half_dot)
        / (1.0 + half_squared)
        for k in range(3)
    ]


def assert_close(actual, expected, tolerance, label):
    error = max(
        abs(a - e)
        for actual_row, expected_row in zip(actual, expected)
        for a, e in zip(actual_row, expected_row)
    )
    if error > tolerance:
        raise AssertionError(f"{label}: max error {error:.6e} > {tolerance:.6e}")
    return error


def one_step_oracle(root):
    model = base_model()
    initial = evaluate(root, "initial", model)
    magnitudes = [norm(spin) for spin in initial["spin"]]
    directions = [
        [value / magnitude for value in spin]
        for spin, magnitude in zip(initial["spin"], magnitudes)
    ]
    alpha = 0.17
    gamma = 1200.0
    time_step_fs = 0.1
    drift = gamma * time_step_fs / 1000.0 / (1.0 + alpha * alpha)
    predictors = []
    for direction, field in zip(directions, initial["mforce"]):
        damping = cross(direction, field)
        omega = [drift * (field[k] + alpha * damping[k]) for k in range(3)]
        predictors.append(cayley(direction, omega))
    midpoint_directions = [
        [0.5 * (direction[k] + predictor[k]) for k in range(3)]
        for direction, predictor in zip(directions, predictors)
    ]
    midpoint_spins = [
        [magnitude * value for value in midpoint]
        for magnitude, midpoint in zip(magnitudes, midpoint_directions)
    ]
    midpoint = evaluate(root, "midpoint", replace_spins(model, midpoint_spins))
    expected = []
    for magnitude, direction, midpoint_direction, field in zip(
            magnitudes, directions, midpoint_directions, midpoint["mforce"]):
        damping = cross(midpoint_direction, field)
        omega = [drift * (field[k] + alpha * damping[k]) for k in range(3)]
        expected.append([magnitude * value for value in cayley(direction, omega)])

    case, result = run_case(
        root,
        "sib",
        "potential nep.txt\n"
        "ensemble nve_sib "
        f"alpha {alpha} gamma {gamma} stemp -1 seed 2468\n"
        f"time_step {time_step_fs}\n"
        "dump_xyz -1 0 1 state.xyz mass spin mforce\n"
        "run 1\n",
        model)
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    actual = read_last_frame(case / "state.xyz")
    spin_error = assert_close(actual["spin"], expected, 3.0e-7, "SIB Cayley oracle")
    endpoint = evaluate(root, "endpoint", replace_spins(model, actual["spin"]))
    field_error = assert_close(
        actual["mforce"], endpoint["mforce"], 3.0e-6, "endpoint mforce")
    norm_error = max(
        abs(norm(spin) - magnitude)
        for spin, magnitude in zip(actual["spin"], magnitudes)
    )
    if norm_error > 3.0e-7:
        raise AssertionError(f"spin norm error {norm_error:.6e}")
    return spin_error, field_error, norm_error


def noise_and_parser_checks(root):
    command = (
        "potential nep.txt\n"
        "ensemble nvt_sib 300 300 100 lattice off alpha 0.1 seed {seed}\n"
        "time_step 0.1\n"
        "dump_xyz -1 0 5 state.xyz mass spin mforce\n"
        "run 5\n")
    frames = []
    for name, seed in (("same_a", 314159), ("same_b", 314159), ("different", 271828)):
        case, result = run_case(root, name, command.format(seed=seed))
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)
        frames.append(read_last_frame(case / "state.xyz")["spin"])
    assert_close(frames[0], frames[1], 0.0, "same-seed reproducibility")
    seed_difference = max(
        abs(a - b)
        for left, right in zip(frames[0], frames[2])
        for a, b in zip(left, right)
    )
    if seed_difference <= 1.0e-8:
        raise AssertionError("different SIB seeds produced indistinguishable spins")

    invalid = [
        "nvt_sib 300 300 100 alpha",
        "nvt_sib 300 300 100 alpha -1",
        "nvt_sib 300 300 100 gamma 0",
        "nvt_sib 300 300 100 stemp -0.5",
        "nvt_sib 300 300 100 seed 0",
        "nvt_sib 300 300 100 lattice maybe",
        "nvt_sib 300 300 100 seed 1 seed 2",
        "nvt_sib 300 300 100 future 1",
        "nve_sib stemp 0",
        "nve_sib lattice off",
        "nve_sib alpha",
        "npt_sib temp 300 300 iso 0 0 lattice off",
    ]
    for index, ensemble in enumerate(invalid):
        _, result = run_case(
            root,
            f"invalid_{index}",
            "potential nep.txt\n" f"ensemble {ensemble}\n" "run 1\n")
        if result.returncode == 0:
            raise AssertionError(f"invalid SIB input was accepted: {ensemble}")
    return seed_difference, len(invalid)


def npt_smoke(root):
    case, result = run_case(
        root,
        "npt",
        "potential nep.txt\n"
        "ensemble npt_sib temp 300 300 iso 0 0 "
        "tperiod 100 pperiod 1000 alpha 0.1 stemp -1 seed 123\n"
        "time_step 0.01\n"
        "dump_xyz -1 0 1 state.xyz mass spin mforce\n"
        "run 1\n")
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    if "Integrate spins with the semi-implicit B (SIB) method" not in result.stdout:
        raise AssertionError("npt_sib did not report the SIB integrator")
    state = read_last_frame(case / "state.xyz")
    if not all(math.isfinite(value) for rows in state.values() for row in rows for value in row):
        raise AssertionError("npt_sib produced non-finite output")


def main():
    if not GPUMD.is_file():
        raise SystemExit(f"GPUMD executable not found: {GPUMD}")
    with tempfile.TemporaryDirectory(prefix="gpumd-spin-sib-") as temporary:
        root = Path(temporary)
        spin_error, field_error, norm_error = one_step_oracle(root)
        seed_difference, invalid_count = noise_and_parser_checks(root)
        npt_smoke(root)
    print(
        "SIB runtime validation passed: "
        f"nve_oracle=passed, spin_error={spin_error:.3e}, "
        f"endpoint_field_error={field_error:.3e}, "
        f"norm_error={norm_error:.3e}, seed_difference={seed_difference:.3e}, "
        f"invalid_inputs={invalid_count}, npt_smoke=passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"SIB runtime validation failed: {error}", file=sys.stderr)
        raise
