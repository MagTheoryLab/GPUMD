.. _kw_lambda_tau:
.. index::
   single: lambda_tau (keyword in nep.in)

:attr:`lambda_tau`
===================

This keyword sets the weight of spin-torque RMSE in the loss::

  lambda_tau <weight>

The weight must be non-negative and defaults to 0. For each labeled active
atom, the predicted and target quantities are converted to

.. math::

   \boldsymbol{\tau}_i
   = \boldsymbol{s}_i\times\boldsymbol{M}_i.

The resulting torque RMSE emphasizes magnetic-force components perpendicular
to the current spin. Only frames containing ``mforce:R:3`` and atom types
enabled by :ref:`spin_dof_type <kw_spin_dof_type>` contribute.

This term complements rather than replaces :ref:`lambda_m <kw_lambda_m>`:
the Cartesian loss constrains all components of :math:`\boldsymbol{M}_i`,
whereas the torque loss is insensitive to a component parallel to
:math:`\boldsymbol{s}_i`. A nonzero value is therefore most useful when
transverse spin dynamics is the quantity of interest.
