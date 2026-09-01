.. _kw_spin_n_max:
.. index::
   single: spin_n_max (keyword in nep.in)

:attr:`spin_n_max`
===================

This Spin NEP Lite keyword records radial and angular maximum indices in the
compatibility header::

  spin_n_max <n_max_radial> <n_max_angular>

The default is ``3 0``. Both values are non-negative and cannot exceed the
corresponding :ref:`spin_basis_size <kw_spin_basis_size>`. They are parsed,
validated, and serialized for Spin NEP Lite checkpoint compatibility. The
current Lite descriptor kernel does not use either value to truncate its
channels; its active radial rank is controlled by ``spin_basis_size`` and
:ref:`spin_compress <kw_spin_compress>`.

This keyword is not part of the ``nep4_spin2`` protocol.
