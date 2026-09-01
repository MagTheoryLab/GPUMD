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

For ``spin_mode 1``, both values are non-negative, they must support the
corresponding :ref:`spin_n_max <kw_spin_n_max>` metadata, and the radial value
must support :ref:`spin_compress <kw_spin_compress>`. The first value controls
the active radial basis count. The second value is retained in the Lite model
header for compatibility but is not consumed by the current descriptor
kernel.

For ``spin_mode 2``, the current radial construction uses exactly nine basis
functions and no separate angular basis. The only supported value is::

  spin_basis_size 8 0

This keyword must therefore be set explicitly for Spin2; the default is not a
valid Spin2 setting.
