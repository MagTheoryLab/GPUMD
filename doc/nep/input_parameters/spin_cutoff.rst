.. _kw_spin_cutoff:
.. index::
   single: spin_cutoff (keyword in nep.in)

:attr:`spin_cutoff`
====================

This keyword sets spin radial and angular cutoff distances in Å::

  spin_cutoff <radial_cutoff> <angular_cutoff>

Both values must be positive. If this keyword is absent, the ordinary radial
and angular :ref:`cutoff <kw_cutoff>` values are reused. This is often a good
starting point, but a shorter magnetic cutoff can reduce the size of the
independent spin neighbor list and the associated training/inference cost.

The current ``spin_mode 1`` and ``spin_mode 2`` descriptor kernels use the
radial value. The angular value is validated and retained in the Spin NEP Lite
header for compatibility, but it does not define a second neighbor list in the
current implementation. The shared input syntax still requires two positive
numbers. For example::

  spin_cutoff 6.0 4.0

stores a 6 Å spin cutoff in a Spin2 model. This setting does not change the
ordinary structural NEP cutoffs.
