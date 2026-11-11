.. _kw_spin_basis_size:
.. index::
   single: spin_basis_size (keyword in nep.in)

:attr:`spin_basis_size`
=========================

Set the maximum radial basis index used by the spin descriptor::

  spin_basis_size K

This defaults to the angular entry of :ref:`basis_size <kw_basis_size>` and
must equal it when specified. Index :math:`K` means :math:`K+1` basis functions.

Spin descriptors reuse the structural angular radial coefficients and
evaluate them with the :ref:`spin_cutoff <kw_spin_cutoff>` envelope. The spin
cutoff may differ from the structural angular cutoff.
