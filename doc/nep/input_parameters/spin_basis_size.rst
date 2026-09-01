.. _kw_spin_basis_size:
.. index::
   single: spin_basis_size (keyword in nep.in)

:attr:`spin_basis_size`
=======================

This keyword sets the largest radial and angular basis indices of the spin
descriptor::

  spin_basis_size <radial_size> <angular_size>

The default is ``3 3``. As for the ordinary NEP basis, a maximum index
:math:`K` represents :math:`K+1` functions because indexing starts from zero.

For ``spin_mode 2``, the current radial construction uses exactly nine basis
functions and no separate angular basis. The only supported value is::

  spin_basis_size 8 0

The first value controls the radial basis count; the second must be zero and
is not serialized in ``nep4_spin2``. This keyword must be set explicitly for
Spin2 because the default is not a valid Spin2 setting.
