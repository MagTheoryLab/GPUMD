#!/usr/bin/env python3
"""Standalone TSPIN validation for minimal CUDA hosts without pytest/numpy."""

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "nep_spin" / "spin_chiral_protocol"
GPUMD = Path(os.environ.get("GPUMD_COMMAND", ROOT / "src" / "gpumd"))
K_B = 8.617343e-5
PI = 3.14159265358979
TIME_UNIT_CONVERSION = 1.018051e1
MASK64 = (1 << 64) - 1


def model_with_zero_lattice_velocity(xyz_name="model_large_box.xyz"):
    lines = (FIXTURE / xyz_name).read_text().splitlines()
    lines[1] = lines[1].replace(
        "Properties=species:S:1:pos:R:3:spin:R:3",
        "Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3")
    for index in range(2, len(lines)):
        fields = lines[index].split()
        lines[index] = " ".join(fields[:4] + ["0", "0", "0"] + fields[4:])
    return "\n".join(lines) + "\n"


def run_case(root, name, run_input, model_text=None, ordinary=False):
    case = root / name
    case.mkdir()
    if ordinary:
        shutil.copy(
            Path(__file__).parent / "fixtures" / "models" / "nep_C.txt",
            case / "nep.txt")
        shutil.copy(
            Path(__file__).parent / "fixtures" / "structures" / "C-nat16-rattled.xyz",
            case / "model.xyz")
    else:
        shutil.copy(FIXTURE / "nep.txt", case / "nep.txt")
        (case / "model.xyz").write_text(
            model_text if model_text is not None
            else model_with_zero_lattice_velocity())
    (case / "run.in").write_text(run_input)
    result = subprocess.run(
        [str(GPUMD)], cwd=case, capture_output=True, text=True, check=False)
    return case, result


def read_spin_frame(path):
    lines = path.read_text().splitlines()
    number_of_atoms = int(lines[-6])
    rows = [
        [float(value) for value in line.split()[1:]]
        for line in lines[-number_of_atoms:]
    ]
    return {
        "mass": [row[3] for row in rows],
        "spin": [row[4:7] for row in rows],
        "mforce": [row[7:10] for row in rows],
        "spin_velocity": [
            [value * TIME_UNIT_CONVERSION for value in row[10:13]]
            for row in rows
        ],
    }


def read_spin_temperatures(path, mass_factor):
    lines = path.read_text().splitlines()
    temperatures = []
    offset = 0
    while offset < len(lines):
        number_of_atoms = int(lines[offset])
        rows = [
            [float(value) for value in line.split()[1:]]
            for line in lines[offset + 2:offset + 2 + number_of_atoms]
        ]
        twice_kinetic_energy = 0.0
        for row in rows:
            mass = row[3]
            velocity = [
                value * TIME_UNIT_CONVERSION for value in row[7:10]
            ]
            twice_kinetic_energy += (
                mass * mass_factor *
                sum(value * value for value in velocity))
        temperatures.append(
            twice_kinetic_energy / (3.0 * number_of_atoms * K_B))
        offset += number_of_atoms + 2
    return temperatures


def splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def uniform_01(state):
    state = splitmix64(state)
    return state, (state >> 11) / 9007199254740992.0


def gaussian(seed, atom_index, component):
    state = seed
    state ^= (atom_index * 0xD1B54A32D192ED03) & MASK64
    state ^= (component * 0x9E3779B97F4A7C15) & MASK64
    state, u1 = uniform_01(state)
    while u1 <= 0.0:
        state, u1 = uniform_01(state)
    state, u2 = uniform_01(state)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * PI * u2)


def initial_spin_velocity(seed, mass, temperature, mass_factor):
    velocity = [
        [0.5 * gaussian(seed, atom + 1, component) for component in range(3)]
        for atom in range(len(mass))
    ]
    twice_kinetic_energy = sum(
        mass[atom] * mass_factor * sum(value * value for value in velocity[atom])
        for atom in range(len(mass))
    )
    target = 3.0 * len(mass) * K_B * temperature
    factor = math.sqrt(target / twice_kinetic_energy)
    return [[value * factor for value in row] for row in velocity]


def nhc(position, velocity, mass, twice_kinetic_energy,
        thermal_energy, degrees_of_freedom, half_time_step):
    weights = (
        0.784513610477560,
        0.235573213359357,
        -1.17767998417887,
        1.31518632068391,
        -1.17767998417887,
        0.235573213359357,
        0.784513610477560,
    )
    factor = 1.0
    for weight in weights:
        dt2 = half_time_step * weight / 4.0
        dt4 = dt2 * 0.5
        dt8 = dt4 * 0.5
        for _ in range(4):
            acceleration = velocity[-2] ** 2 / mass[-2] - thermal_energy
            velocity[-1] += dt4 * acceleration
            for index in range(len(velocity) - 2, -1, -1):
                scale = math.exp(-dt8 * velocity[index + 1] / mass[index + 1])
                if index == 0:
                    acceleration = (
                        twice_kinetic_energy -
                        degrees_of_freedom * thermal_energy)
                else:
                    acceleration = (
                        velocity[index - 1] ** 2 / mass[index - 1] -
                        thermal_energy)
                velocity[index] = scale * (
                    scale * velocity[index] + dt4 * acceleration)
            for index in range(len(position) - 1, -1, -1):
                position[index] += dt2 * velocity[index] / mass[index]
            local_factor = math.exp(-dt2 * velocity[0] / mass[0])
            twice_kinetic_energy *= local_factor * local_factor
            factor *= local_factor
            for index in range(len(velocity) - 1):
                scale = math.exp(-dt8 * velocity[index + 1] / mass[index + 1])
                if index == 0:
                    acceleration = (
                        twice_kinetic_energy -
                        degrees_of_freedom * thermal_energy)
                else:
                    acceleration = (
                        velocity[index - 1] ** 2 / mass[index - 1] -
                        thermal_energy)
                velocity[index] = scale * (
                    scale * velocity[index] + dt4 * acceleration)
            acceleration = velocity[-2] ** 2 / mass[-2] - thermal_energy
            velocity[-1] += dt4 * acceleration
    return factor


def max_error(candidate, expected):
    return max(
        abs(candidate[atom][component] - expected[atom][component])
        for atom in range(len(candidate))
        for component in range(3)
    )


def validate_one_step(root):
    static_model = model_with_zero_lattice_velocity()
    lines = static_model.splitlines()
    lines[1] = lines[1].replace(":spin:R:3", ":spin:R:3:spin_vel:R:3")
    for index in range(2, len(lines)):
        lines[index] += " 0 0 0"
    static_case, static_result = run_case(
        root,
        "static",
        "potential nep.txt\n"
        "ensemble nve\n"
        "time_step 0\n"
        "dump_xyz -1 0 1 state.xyz mass spin mforce spin_velocity\n"
        "run 1\n",
        "\n".join(lines) + "\n")
    if static_result.returncode != 0:
        raise RuntimeError(static_result.stdout + static_result.stderr)
    initial = read_spin_frame(static_case / "state.xyz")

    temperature = 300.0
    coupling = 100.0
    mass_factor = 1.5
    seed = 24681357
    time_step_fs = 0.1
    tspin_case, result = run_case(
        root,
        "tspin",
        "potential nep.txt\n"
        f"ensemble nvt_tspin {temperature} {temperature} {coupling} "
        f"mass_factor {mass_factor} seed {seed}\n"
        f"time_step {time_step_fs}\n"
        "dump_xyz -1 0 1 state.xyz mass spin mforce spin_velocity\n"
        "run 1\n")
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    actual = read_spin_frame(tspin_case / "state.xyz")

    dt = time_step_fs / TIME_UNIT_CONVERSION
    number_of_atoms = len(actual["mass"])
    velocity = initial_spin_velocity(
        seed, actual["mass"], temperature, mass_factor)
    position_nhc = [0.0] * 4
    velocity_nhc = [1.0, -1.0, 1.0, -1.0]
    thermal_energy = K_B * temperature
    mass_nhc = [thermal_energy * (dt * coupling) ** 2] * 4
    mass_nhc[0] *= 3.0 * number_of_atoms

    twice_kinetic_energy = sum(
        actual["mass"][atom] * mass_factor *
        sum(value * value for value in velocity[atom])
        for atom in range(number_of_atoms))
    factor = nhc(
        position_nhc, velocity_nhc, mass_nhc, twice_kinetic_energy,
        thermal_energy, 3.0 * number_of_atoms, 0.5 * dt)
    expected_spin = [[0.0] * 3 for _ in range(number_of_atoms)]
    for atom in range(number_of_atoms):
        for component in range(3):
            velocity[atom][component] *= factor
            velocity[atom][component] += (
                0.5 * dt * initial["mforce"][atom][component] /
                (actual["mass"][atom] * mass_factor))
            expected_spin[atom][component] = (
                initial["spin"][atom][component] +
                dt * velocity[atom][component])
            velocity[atom][component] += (
                0.5 * dt * actual["mforce"][atom][component] /
                (actual["mass"][atom] * mass_factor))

    twice_kinetic_energy = sum(
        actual["mass"][atom] * mass_factor *
        sum(value * value for value in velocity[atom])
        for atom in range(number_of_atoms))
    factor = nhc(
        position_nhc, velocity_nhc, mass_nhc, twice_kinetic_energy,
        thermal_energy, 3.0 * number_of_atoms, 0.5 * dt)
    expected_velocity = [
        [value * factor for value in row] for row in velocity
    ]
    spin_error = max_error(actual["spin"], expected_spin)
    velocity_error = max_error(actual["spin_velocity"], expected_velocity)
    if spin_error > 2.0e-7 or velocity_error > 2.0e-7:
        raise AssertionError(
            f"one-step oracle mismatch: spin={spin_error:.3e}, "
            f"spin_velocity={velocity_error:.3e}")
    return spin_error, velocity_error


def validate_fail_closed(root):
    bad_ensembles = (
        "nvt_tspin 300 300 100 mass_factor",
        "nvt_tspin 300 300 100 mass_factor 0",
        "nvt_tspin 300 300 100 mass_factor nan",
        "nvt_tspin 300 300 100 seed 0",
        "nvt_tspin 300 300 100 seed nope",
        "nvt_tspin 300 300 100 seed 1 seed 2",
        "nvt_tspin 300 300 100 mass_factor 1 mass_factor 2",
        "nvt_tspin 300 300 100 future 1",
    )
    for index, ensemble in enumerate(bad_ensembles):
        _, result = run_case(
            root,
            f"bad_{index}",
            "potential nep.txt\n"
            f"ensemble {ensemble}\n"
            "run 1\n")
        if result.returncode == 0:
            raise AssertionError(f"bad parser case was accepted: {ensemble}")

    _, ordinary = run_case(
        root,
        "ordinary",
        "potential nep.txt\n"
        "ensemble nvt_tspin 300 300 100\n"
        "run 1\n",
        ordinary=True)
    if ordinary.returncode == 0:
        raise AssertionError("nvt_tspin accepted a non-spin potential")
    return len(bad_ensembles) + 1


def validate_lifecycle(root):
    first_case, first = run_case(
        root,
        "lifecycle",
        "potential nep.txt\n"
        "ensemble nvt_tspin 300 300 100 seed 17\n"
        "time_step 0.1\n"
        "dump_restart 1\n"
        "run 1\n"
        "ensemble nvt_tspin 300 300 100 seed 999\n"
        "run 1\n")
    if first.returncode != 0:
        raise RuntimeError(first.stdout + first.stderr)
    if first.stdout.count("Initialized TSPIN velocities with seed") != 1:
        raise AssertionError("spin velocity was initialized more than once")
    reuse_message = "Reuse initialized spin velocities; nvt_tspin seed is not used."
    if reuse_message not in first.stdout:
        raise AssertionError("second run did not reuse spin velocity")

    _, restarted = run_case(
        root,
        "restart",
        "potential nep.txt\n"
        "ensemble nvt_tspin 300 300 100 seed 123\n"
        "time_step 0.1\n"
        "run 1\n",
        (first_case / "restart.xyz").read_text())
    if restarted.returncode != 0:
        raise RuntimeError(restarted.stdout + restarted.stderr)
    if reuse_message not in restarted.stdout:
        raise AssertionError("restart did not restore initialized spin velocity")


def validate_short_dynamics(root):
    mass_factor = 1.5
    case, result = run_case(
        root,
        "short_dynamics",
        "replicate 5 5 5\n"
        "potential nep.txt\n"
        f"ensemble nvt_tspin 300 300 100 mass_factor {mass_factor} seed 314159\n"
        "time_step 0.1\n"
        "dump_xyz -1 0 10 trajectory.xyz mass spin spin_velocity\n"
        "run 500\n",
        model_with_zero_lattice_velocity("model_small_pbc.xyz"))
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    temperatures = read_spin_temperatures(
        case / "trajectory.xyz", mass_factor)
    if not temperatures or not all(math.isfinite(value) for value in temperatures):
        raise AssertionError("short TSPIN dynamics produced a non-finite spin temperature")
    mean_temperature = sum(temperatures) / len(temperatures)
    if not 240.0 <= mean_temperature <= 360.0:
        raise AssertionError(
            f"short TSPIN mean temperature is {mean_temperature:.6g} K")
    return min(temperatures), max(temperatures), mean_temperature


def validate_fixed_spin_nhc(root):
    model_text = model_with_zero_lattice_velocity()
    initial_spin = [
        [float(value) for value in line.split()[7:10]]
        for line in model_text.splitlines()[2:]
    ]
    case, result = run_case(
        root,
        "fixed_spin_nhc",
        "potential nep.txt\n"
        "ensemble nvt_nhc 300 300 100\n"
        "time_step 0.1\n"
        "dump_xyz -1 0 10 state.xyz mass spin mforce\n"
        "run 10\n",
        model_text)
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    lines = (case / "state.xyz").read_text().splitlines()
    final_spin = [
        [float(value) for value in line.split()[5:8]]
        for line in lines[-len(initial_spin):]
    ]
    error = max_error(final_spin, initial_spin)
    if error != 0.0:
        raise AssertionError(f"nvt_nhc changed fixed spin by {error}")
    return error


def main():
    if not GPUMD.exists():
        raise FileNotFoundError(GPUMD)
    with tempfile.TemporaryDirectory(prefix="gpumd-tspin-validation-") as directory:
        root = Path(directory)
        spin_error, velocity_error = validate_one_step(root)
        negative_count = validate_fail_closed(root)
        validate_lifecycle(root)
        fixed_spin_error = validate_fixed_spin_nhc(root)
        minimum_temperature, maximum_temperature, mean_temperature = (
            validate_short_dynamics(root))
    print(
        "TSPIN validation passed: "
        f"one-step max spin error={spin_error:.3e}, "
        f"max spin_velocity error={velocity_error:.3e}, "
        f"fail-closed cases={negative_count}, lifecycle/restart=passed, "
        f"fixed-spin nvt_nhc error={fixed_spin_error:.1e}, "
        f"short spin T min/mean/max={minimum_temperature:.3f}/"
        f"{mean_temperature:.3f}/{maximum_temperature:.3f} K")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"TSPIN validation failed: {error}", file=sys.stderr)
        sys.exit(1)
