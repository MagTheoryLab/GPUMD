"""NEP_Spin parser, oracle, small-box and raw9 ownership regression tests."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "nep_spin" / "spin_chiral_protocol"
GPUMD = Path(os.environ.get("GPUMD_COMMAND", Path(__file__).parents[1] / "src" / "gpumd"))
ROW_MAJOR_FROM_GPUMD = np.array([0, 3, 4, 6, 1, 5, 7, 8, 2])


def _run(case_dir, model_text=None, xyz_name="model_large_box.xyz"):
    case_dir.mkdir()
    shutil.copy(FIXTURE / xyz_name, case_dir / "model.xyz")
    (case_dir / "nep.txt").write_text(
        model_text if model_text is not None else (FIXTURE / "nep.txt").read_text())
    (case_dir / "run.in").write_text(
        "potential nep.txt\n"
        "velocity 1\n"
        "ensemble nve\n"
        "time_step 0\n"
        "dump_xyz -1 0 1 result.xyz force potential spin mforce virial\n"
        "run 1\n")
    env = os.environ.copy()
    cuda_lib = "/home/dwhe/opt/cuda-12.8/lib64"
    env["LD_LIBRARY_PATH"] = cuda_lib + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return subprocess.run(
        [str(GPUMD)], cwd=case_dir, env=env, capture_output=True, text=True, check=False)


def _read_last_frame(path):
    lines = path.read_text().splitlines()
    atom_count = int(lines[-6])
    assert atom_count == 4
    header = lines[-5]
    energy = float(re.search(r"\benergy=([-+0-9.eE]+)", header).group(1))
    rows = [line.split() for line in lines[-4:]]
    values = np.array([[float(value) for value in row[1:]] for row in rows])
    return {
        "energy": np.array([energy]),
        "position": values[:, 0:3],
        "force": values[:, 3:6],
        "spin": values[:, 6:9],
        "mforce": values[:, 9:12],
        "potential": values[:, 12],
        "atom_virial": values[:, 13:22],
    }


def _reference(case):
    return {
        name: np.asarray(values, dtype=float)
        for name, values in json.loads((FIXTURE / "fp64_oracle.json").read_text())[case].items()
    }


def test_nep_spin_matches_frozen_fp64_oracle(tmp_path):
    tolerances = {
        "energy": 2.0e-4,
        "potential": 2.0e-4,
        "force": 2.0e-4,
        "mforce": 2.0e-4,
        "atom_virial": 2.0e-4,
    }
    for case, xyz_name in (
        ("large_box", "model_large_box.xyz"),
        ("small_pbc", "model_small_pbc.xyz"),
    ):
        result = _run(tmp_path / case, xyz_name=xyz_name)
        assert result.returncode == 0, result.stdout + result.stderr
        actual = _read_last_frame(tmp_path / case / "result.xyz")
        reference = _reference(case)
        for field, tolerance in tolerances.items():
            candidate = actual[field].reshape(-1)
            expected = reference[field].reshape(-1)
            assert np.max(np.abs(candidate - expected)) <= tolerance, field
        assert np.max(np.abs(actual["force"].sum(axis=0))) <= 2.0e-5
        assert np.max(np.abs(
            actual["atom_virial"].sum(axis=0) - reference["virial"])) <= 2.0e-4


@pytest.mark.parametrize(
    "mutate",
    [
        lambda text: text.replace("nep4_spin1", "nep4_spin", 1),
        lambda text: text.replace("spin_scaler 1", "spin_future 1", 1),
        lambda text: text.rsplit("\n", 2)[0] + "\n",
        lambda text: text + "0\n",
        lambda text: text.replace("spin_chiral 1", "spin_chiral 2", 1),
        lambda text: text.replace("spin_dof_type Fe", "spin_baseline 0", 1),
        lambda text: text.replace("spin_mode 1 10", "spin_mode 1 9", 1),
        lambda text: text.replace("ANN 30 0", "ANN 30 1", 1),
    ],
)
def test_nep_spin_parser_fails_closed(tmp_path, mutate):
    result = _run(tmp_path / "case", mutate((FIXTURE / "nep.txt").read_text()))
    assert result.returncode != 0


def test_nep_spin_requires_explicit_spin(tmp_path):
    xyz = (FIXTURE / "model_large_box.xyz").read_text()
    xyz = xyz.replace(":spin:R:3", "")
    xyz = "\n".join(
        " ".join(line.split()[:4]) if line.startswith("Fe ") else line
        for line in xyz.splitlines()) + "\n"
    case = tmp_path / "missing_spin"
    case.mkdir()
    (case / "model.xyz").write_text(xyz)
    shutil.copy(FIXTURE / "nep.txt", case / "nep.txt")
    (case / "run.in").write_text("potential nep.txt\nrun 1\n")
    result = subprocess.run(
        [str(GPUMD)], cwd=case, capture_output=True, text=True, check=False)
    assert result.returncode != 0
