.. _kw_spin_mode:
.. index::
   single: spin_mode (keyword in nep.in)

:attr:`spin_mode`
==================

This keyword enables Spin NEP training. The syntax is::

  spin_mode <mode>

========  ================================================================
Value     Model
========  ================================================================
0         Ordinary non-spin NEP (default)
1         Spin NEP Lite; writes ``nep4_spin1``
2         Unified order/coupling Spin2; writes ``nep4_spin2``
========  ================================================================

Spin training requires ``version 4``, ``model_type 0``, and one hidden layer.
Every frame in ``train.xyz`` and ``test.xyz`` must contain a Cartesian
``spin:R:3`` property (accepted aliases are listed in
:ref:`train_test_xyz`). Add ``mforce:R:3`` to train magnetic forces.

Charge mode and type-dependent cutoffs are not supported. ZBL is supported
only with mode 2. ``fine_tune`` and ``import_q_scaler`` are not currently
supported for Spin NEP.

Mode 2 requires ``spin_basis_size 8 0`` and an explicit supported
``spin_l_max``. A complete, commonly used shape is::

  spin_mode        2
  spin_basis_size  8 0
  spin_l_max       2 0 0
  spin_compress    2
  spin_order       3
  spin_soc         1

The last four keywords determine the magnetic descriptor dimension. See
:ref:`nep_spin_dimensions` before changing this shape.
