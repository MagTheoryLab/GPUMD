.. _kw_spin_dof_type:
.. index::
   single: spin_dof_type (keyword in nep.in)

:attr:`spin_dof_type`
======================

This keyword selects atom types whose spins are active degrees of freedom::

  spin_dof_type <type_1> [<type_2> ...]

At least one type must be listed. Names must be unique and must appear in
:ref:`type <kw_type>`. If the keyword is absent, all atom types are active.

Active types receive learned magnetic-force output and contribute to Cartesian
magnetic-force and torque RMSEs when the corresponding labels are present.
Inactive types receive zero public magnetic force and do not enter those two
RMSE counts. They can nevertheless contribute to a magnetic environment when
explicitly included by :ref:`spin_env_type <kw_spin_env_type>`; their spins
then act as frozen model inputs.

The grouped-response reduction visits every atom rather than applying the
RMSE mask. When :ref:`lambda_spin_response <kw_lambda_spin_response>` is used,
``spin_tangent`` must therefore be zero on inactive atoms.

For a model in which only Fe spins are dynamical, use::

  spin_dof_type Fe

Every active type must also appear in ``spin_env_type``.
