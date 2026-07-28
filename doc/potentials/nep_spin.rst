NEP_Spin runtime
****************

``NEP_Spin`` evaluates canonical ``nep4_spin1`` models exported by a compatible
training implementation. It augments the ordinary NEP structural descriptor
with a spin descriptor and returns energy, per-atom potential, force, magnetic
force, and a complete per-atom raw9 virial.

Input contract
==============

The first line of the potential must start with ``nep4_spin1``. The following
``spin_mode 1 <count>`` line introduces a counted spin header. Required header
records, the numeric payload, and descriptor scalers are checked exactly.
Unknown or duplicate records, legacy keywords, truncated or extra payloads,
non-finite values, and unsupported descriptor shapes are rejected.

The model uses independent radial, angular, and spin cutoffs and neighbor
lists. The capacity in the ordinary ``cutoff`` record must cover neighbors out
to the largest of those cutoffs. Overflow is reported before descriptor or
force kernels consume the list.

``model.xyz`` must contain ``spin:R:3`` for every atom. There is no implicit
zero-spin fallback. Only full periodic boundary conditions are supported.

Outputs and units
=================

Energy and per-atom potential are in eV, force is in eV/Å, and magnetic force
is the negative derivative of energy with respect to the dimensionless spin.
The per-atom virial is written in row-major order
``xx xy xz yx yy yz zx zy zz``. Edge contributions use neighbor-atom
(``n2``) ownership.

Supported runtime boundary
==========================

This implementation supports one visible CUDA GPU, large boxes, expanded-cell
small boxes, and fixed-spin ``nve`` or ``nvt_nhc`` runs. HIP is source-level
compile-only until validated on AMD hardware.

NEP_Spin must be the only potential. Training inside GPUMD, ZBL combinations,
multiple potentials, multi-GPU, MDI, PIMD, minimization, force constants,
elastic calculations, Monte Carlo, deformation, HNEMD and spin heat-current
paths, ``dump_exyz``, and ``dump_observer`` are rejected. No spin integrator is
provided by this runtime.
