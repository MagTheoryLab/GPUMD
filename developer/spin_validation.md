# Spin regression checks

Run from the repository root with Python, NumPy, pytest, a C++ compiler and,
for runtime checks, CUDA and an allocated NVIDIA GPU. Build `src/gpumd` and
`src/nep` for that GPU; `GPUMD_COMMAND` and `NEP_COMMAND` can select other builds.
Set `RESULTS` to an existing absolute directory outside the checkout.

Training-only commands, probe builds and the native/JIT × 1/2/4 GPU matrix are
listed in [Spin training validation](spin_training_validation.md).

## Routine checks

```sh
python -m pytest --noconftest -q -m 'not slow' tests_pytest/test_spin_*.py
python tests_pytest/validate_nep_spin3_runtime.py
python tests_pytest/validate_nep_spin3_runtime.py --fixtures o3c2_uniform --cases chiral_soc
python tests_pytest/validate_mfield_runtime.py
python tests_pytest/validate_main_nep_spin3_training.py
python tests_pytest/validate_spin_training_lifecycle.py
```

`--noconftest` avoids the ordinary NEP suite's ASE/Calorine fixtures. The CPU codec
checks the current model format and rejects development-only formats. Runtime checks cover independent energy,
force, magnetic force and virial oracles, integration formulas, noise scaling,
FIRE convergence and failure states, and Spin lattice restart continuity. TSPIN
and SIB each have one pytest implementation; the external-field SIB oracle is
part of that suite. Training checks cover actual errors, defaults, masks,
normalization, model export and restart. The main parity matrix has five cases:
C2 with selective spins, C3 with all spins and nontrivial restart layout, C1 with
typewise cutoff and torque supervision, ZBL, and a small periodic box with spin cutoff equal to radial cutoff. Lifecycle checks retain selected and all-type spin degrees of freedom, and mask excluded spins and magnetic labels without changing energy or atomic-force labels.
Detailed malformed model headers belong to the CPU codec suite; the CLI keeps a
representative rejection check. Spin rejects flexible ZBL in training and in
current block models; universal ZBL remains in the main parity matrix. The ordinary `tests_regression` suite
remains a separate regression gate.

The six files in `fixtures/nep_spin3` are two models, one structure and three
frozen numerical oracles. The typewise-cutoff case derives its header from the
C=2 model. SHA256 checks bind each oracle to its model. Never regenerate expected
values with the candidate implementation to make a failing test pass.
The nonmagnetic-Ge cases use independent TorchNEP FP64/autograd evaluations;
Ge spins are zero, while their positions still contribute geometric moments.
The original oracle was reproduced before updating these inputs; model parameters
and scalers were unchanged.

## Extended GPU checks

Run these when changing the corresponding numerical kernels, decomposition,
JIT or checkpoint contracts, and before merging those changes. The multi-GPU
scripts require four allocated GPUs visible through `CUDA_VISIBLE_DEVICES`.

```sh
python -m pytest --noconftest -q -m slow tests_pytest/test_spin_*.py
python tests_pytest/validate_main_nep_spin3_training.py --check-jit
python tests_pytest/validate_spin_training_multigpu.py --compile both --output "$RESULTS/training"
python tests_pytest/validate_spin_multigpu.py --binary src/gpumd \
  --model tests_pytest/fixtures/nep_spin3/nep_spin_o3c2_uniform.nep \
   --output "$RESULTS/runtime"
python tests_pytest/validate_spin_glsd_runtime.py --binary src/gpumd \
  --output "$RESULTS/glsd"
python -m pytest --noconftest -q tests_pytest/test_spin_lattice_restart.py
```

The slow pytest cases check TSPIN temperature statistics and restart trajectories
with the bundled NEP model; routine restart checks use a zero-force analytic
model. Run lattice restart pytest on one and four GPUs to cover both execution
paths. The GLSD runtime suite uses the bundled model by default and also checks
analytic trajectories, timestep convergence, diffusion and the restart stream.
No private Fe model or pre-refactor executable is required. Training multi-GPU
checks use multiple structures per batch and independently recomputed energy
errors. Optional historical binary comparisons are development evidence, not
prerequisites for these regressions.

## Focused compiled probes

Build against current sources and objects. These commands assume V100; change
`-arch=sm_70` for the allocated GPU. Rebuild the executable objects after source
changes. The small probes check contracts not exposed by the command-line output.

```sh
nvcc -std=c++14 -O3 -arch=sm_70 -Isrc tests_pytest/spin_glsd_analytic.cu \
  src/integrate/spin_glsd.cu -o "$RESULTS/spin_glsd_analytic"
python tests_pytest/validate_spin_glsd_analytic.py "$RESULTS/spin_glsd_analytic"
```

The magnetic-loss probe checks independent projections, including zero spins.
The restart probe distinguishes the SNES mean from the exported elite. The
response probes check rotation geometry, loss normalization and data isolation.
Initialization and variable-type assignment are checked by the restart probe;
torque and response-loss edge cases share the magnetic and response probes.
GLSD's harmonic probe supplies independent statistical expectations
that a bounded ANN potential cannot supply.

For integration temperature changes, link `integrate_temperature.cu` against the
GPUMD objects excluding `main_gpumd/main.o`; it checks the fixed initial cache and
per-step temperature ramps:

```sh
objects=$(find src/main_gpumd src/minimize src/phonon src/integrate src/mc \
  src/force src/measure src/model src/utilities -name '*.cu' \
  ! -path 'src/main_gpumd/main.cu' | sed 's/\.cu$/.o/')
nvcc -std=c++14 -O2 -arch=sm_70 -Isrc tests_pytest/integrate_temperature.cu \
  $objects -lcublas -lcusolver -lcufft -lpthread -o "$RESULTS/integrate_temperature"
"$RESULTS/integrate_temperature"
```

Keep timing, source/binary hashes and run logs outside the source tree. These
checks establish software contracts, not magnetic phase behavior or predictive
accuracy on unseen structures.
