.. _mforce_out_nep:
.. index::
   single: mforce_train.out (output file)
   single: mforce_test.out (output file)

``mforce_*.out``
=================

The ``mforce_train.out`` and ``mforce_test.out`` files contain the predicted
and target magnetic-force components of labeled configurations provided in
the :ref:`train.xyz and test.xyz input files <train_test_xyz>`. Configurations
without magnetic-force labels are omitted.

There are 6 columns. The first three columns are the :math:`x`, :math:`y`, and
:math:`z` magnetic-force components predicted by the :term:`NEP` model in eV
per unit spin. The last three columns are the corresponding target magnetic
forces. Each row corresponds to one atom in a labeled configuration; inactive
spin-DOF types have zero predicted magnetic force.
