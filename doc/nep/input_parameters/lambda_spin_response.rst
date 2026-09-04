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

Every participating ``train.xyz`` frame must provide the converged DFT
``spin:R:3`` and ``mforce:R:3`` together with::

  response_probe=rotation response_group=<name> response_coordinate=<value>

Each group needs at least three distinct ``response_coordinate`` values, and
the trainer sorts each complete group by this coordinate and derives
``d spin / d response_coordinate`` with the same second-order, three-point
nonuniform finite difference used by TorchNEP. The coordinate does not have
to be globally comparable between groups.

For a rotation angle :math:`\theta`, ``response_coordinate`` should be
:math:`\theta` in radians. ``spin`` must contain the moments from the final
constrained-DFT state, not the initially requested constraint vectors. The
derived path tangent retains both transverse and longitudinal relaxation.
Atoms excluded by :ref:`spin_dof_type <kw_spin_dof_type>` are masked from the
derived response automatically. ``spin_tangent:R:3`` is not accepted as an
input label.

With this tangent convention,
:math:`R_a=-\partial U_a/\partial\theta`. Matching TorchNEP's
``neutral-group-balanced-mean-plus-centered-mforce-generator-v3`` contract,
each group contributes equally. A single group-balanced target-response RMS
normalizes both the centered-shape and group-mean Huber terms; the latter has
weight 0.25. Coordinates define the path derivative but do not create an
additional reliability weight.
