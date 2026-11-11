.. _kw_spin_type:
.. index::
   single: spin_type (keyword in nep.in)

:attr:`spin_type`
======================

Select the atom types that have independent spin degrees of freedom::

  spin_type Fe

At least one type must be listed. Names must be unique and must appear in
:ref:`type <kw_type>`. If omitted, all atom types are selected. The effective
selection is saved in the model and restored for prediction and restart;
an explicit conflicting selection is rejected.

For version 1, selected types are magnetic descriptor centers. Selected
types contribute to magnetic-force,
parallel/perpendicular and torque supervision when labels are present. All atom
types contribute structural and magnetic geometric environments within the
corresponding cutoffs. Only selected types contribute neighbor spin moments.

For atoms outside this selection, the reader sets the effective spin to zero
and ignores magnetic-force labels. Original data files need no modification;
energy and atomic-force labels are preserved. Their public magnetic force is
zero and they are excluded from magnetic loss counts. Their positions and
atomic forces remain active, including forces due to magnetic energy.

The grouped-response path derivative uses the selected DFT spin coordinates.
With :ref:`lambda_spin_response <kw_lambda_spin_response>`, derived tangents
are zero on excluded types.

With ``spin_version 2``, **every atom type is an energy and density-response
center**, including types excluded by ``spin_type``. For example, with
``type 3 Fe Bi O`` and ``spin_type Fe``, Bi and O centers can respond to
neighboring Fe spins and contribute magnetic energy and atomic forces.
This does not give Bi/O independent spin degrees of freedom. There are no
separate ``spin_dof_type`` or ``spin_env_type`` settings.
