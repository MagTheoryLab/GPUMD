.. _kw_spin_curriculum:
.. index::
   single: spin_curriculum (keyword in nep.in)

:attr:`spin_curriculum`
========================

This keyword enables a staged SNES search for third-order Spin2 channels::

  spin_curriculum <flag>

The flag can be 0 (default) or 1. It requires ``spin_mode 2`` and
``spin_order 3``.

When enabled, training is divided by the requested number of generations:

====================  ====================================================
Stage                 Order-3 ANN connection scale
====================  ====================================================
first third           0
second third          linear ramp from 0 to 1
final third           1
====================  ====================================================

The scale is applied both to SNES population perturbations and to SNES updates
of neural-network weights connected to order-3 spin coordinates. The lower
descriptor levels and all other parameters follow the ordinary SNES schedule
throughout. This changes the optimization path only; it does not change the
descriptor stored in the final model.

The curriculum can help the optimizer first fit the lower-order magnetic
signal before exposing the complete order-3 input. It is an optimization
option, not a guarantee of lower validation error, so its effect should be
checked against an otherwise identical run.
