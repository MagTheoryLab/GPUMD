.. _kw_spin_basis_size:
.. index::
   single: spin_basis_size (keyword in nep.in)

:attr:`spin_basis_size`
=======================

This keyword sets the largest radial and angular basis indices of the spin
descriptor::

  spin_basis_size <radial_size> <angular_size>

The default is ``8 0``. As for the ordinary NEP basis, a maximum index
:math:`K` represents :math:`K+1` functions because indexing starts from zero.

For ``spin_version 1``, the current radial construction uses exactly nine basis
functions and no separate angular basis. The only supported value is::

  spin_basis_size 8 0

The first value controls the radial basis count; the second must be zero and
is omitted from the model-file field ``spin_basis_size 8``.

For ``spin_version 2``, the spin density reuses the structural angular radial
coefficients. This keyword defaults to the angular entry of ``basis_size``
and must match it if given explicitly. Both ``K`` and ``K 0`` are accepted.
The same coefficients are evaluated with the ``spin_cutoff`` envelope,
which may differ from the structural angular cutoff.
