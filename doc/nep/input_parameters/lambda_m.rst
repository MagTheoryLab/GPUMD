.. _kw_lambda_m:
.. index::
   single: lambda_m (keyword in nep.in)

:attr:`lambda_m`
=================

This keyword sets the weight of Cartesian magnetic-force RMSE in the loss::

  lambda_m <weight>

The weight must be non-negative and defaults to 1. It multiplies the RMSE of
the Cartesian magnetic force

.. math::

   \boldsymbol{M}_i=-\frac{\partial U}{\partial\boldsymbol{s}_i},

whose unit is eV per unit spin. Only frames containing ``mforce:R:3`` and
atoms enabled by :ref:`spin_dof_type <kw_spin_dof_type>` enter this RMSE. A
frame without magnetic-force labels can still contribute energy, force, or
virial data.

Set ``lambda_m 0`` to remove the Cartesian magnetic-force RMSE from the
optimization objective. This does not change the analytic magnetic-force
definition or the model format. ``lambda_mforce`` is an accepted input alias,
but ``lambda_m`` is the canonical spelling used in the documentation and
training output.
