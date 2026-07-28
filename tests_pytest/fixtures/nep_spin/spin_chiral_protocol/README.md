# Frozen NEP_Spin chiral protocol fixture

The model, original small-PBC total-output reference, and source structure were
copied without modification from NEPAdapters commit
`b4735bba1d02045ad31b7bae510bfdb393536f37`.

The independent oracle is the scalar, untabulated, non-OpenMP FP64 CPU path in
`NEPAdapters/tests/test_fp64_oracle.cpp` and
`NEPAdapters/engines/cpu/native/nep.cpp`. GPUMD output must never be used to
update the frozen references.

Original payload SHA256:

- `nep.txt`: `031ea97b5c88aa1daa9378292c60035fa2a746153520c7db0c94740a719a2111`
- `reference_small_pbc.txt`: `46cd75c83aa923a935a58afd7c9131885a719b955608f6f3628e6dd9d98891ce`
- `lammps_structure.json`: `89197f29d361af05a59712c2574d3d8bc17f7422b41da4200f930c45b276acf4`

`model_small_pbc.xyz` and `model_large_box.xyz` contain the same positions and
spins with 4 and 20 angstrom orthogonal cells.

`fp64_oracle.json` was generated from a temporary copy of that exact reference
commit, configured with `NEP_ADAPTERS_CPU_ENABLE_OPENMP=OFF` and
`NEP_ADAPTERS_CPU_USE_RADIAL_TABLE=OFF`. The temporary copy changes only the
batch-output spin-virial sink from atom 0 to `edge.j`; this activates the
documented neighbor (`n2`) ownership that the production LAMMPS path already
uses. It does not alter descriptor, energy, force, mforce, or total virial
math. Neither reference checkout was modified.

Frozen extension SHA256:

- `model_small_pbc.xyz`: `2d82ebb09cc1a14e9c677b578917f1130f987c96121d069c980667fcc6e83fd7`
- `model_large_box.xyz`: `8e130c4ce5a86e8c5fe61dd4c49cfb8d4c400c3ccfadc38c386a0b665caeb125`
- `fp64_oracle.json`: `4b419f376997aaf37e7d991fb0daba93ede2e2b7876b3a2e1f532cb789d93dde`
