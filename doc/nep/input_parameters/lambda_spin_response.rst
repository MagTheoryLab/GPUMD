.. _kw_lambda_spin_response:
.. index::
   single: lambda_spin_response (keyword in nep.in)

:attr:`lambda_spin_response`
=============================

This keyword sets the weight of grouped magnetic-response loss::

  lambda_spin_response <weight>

The weight must be non-negative and defaults to 0. A positive value requires
``spin_mode 3`` and full-batch evaluation::

  batch <size> 1

The second value ``1`` asks the trainer to evaluate all batches before the
grouped loss is formed; ``size`` still controls how many structures reside in
one batch.

For a response frame :math:`a`, the supplied tangent
:math:`\boldsymbol{t}_{ai}` defines a scalar generalized response

.. math::

   R_a=\sum_i \boldsymbol{M}_{ai}\cdot\boldsymbol{t}_{ai}.

The loss compares predicted and target :math:`R_a` values within each named
group. It robustly constrains both the centered response shape along the path
and, with a smaller weight, the group mean. This is useful when relative
magnetic response along a controlled rotation path matters in addition to
per-component magnetic-force RMSE.

Every participating ``train.xyz`` frame must provide ``mforce:R:3`` and
``spin_tangent:R:3`` together with::

  response_probe=rotation response_group=<name> response_coordinate=<value>

Each group needs at least three distinct ``response_coordinate`` values, and
all frames in a participating group must provide the tangent. The coordinate
does not have to be globally comparable between groups. It parameterizes one
path and is used to estimate a target-signal reliability weight; it does not
sort the frames. It is explicit because a collection of Cartesian spin
snapshots cannot in general reveal which atoms were rotated, the rotation
axis, or a nonuniform path parameter.

For a rotation angle :math:`\theta`, a natural tangent is
:math:`\boldsymbol{t}_{ai}=\partial\boldsymbol{s}_{ai}/\partial\theta` and a
natural ``response_coordinate`` is :math:`\theta` in radians. The trainer does
not construct this tangent from the group label. Set this tangent to zero for
atoms excluded by :ref:`spin_dof_type <kw_spin_dof_type>`, because the grouped
sum visits every atom while their predicted public magnetic force is masked to
zero. See :ref:`train_test_xyz` for the complete extended-XYZ contract.

With this tangent convention,
:math:`R_a=-\partial U_a/\partial\theta`. The centered shape term uses a Huber
loss normalized by the global centered-target RMS and weighted by a
coordinate-derived reliability score with a 0.05 floor. The final loss adds
0.25 times a separately normalized Huber loss of the group means.
