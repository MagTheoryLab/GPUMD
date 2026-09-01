.. _kw_spin_chiral:
.. index::
   single: spin_chiral (keyword in nep.in)

:attr:`spin_chiral`
====================

This Spin NEP Lite keyword enables its optional chiral descriptor channel::

  spin_chiral <flag>

The flag can be 0 (default) or 1. It requires ``spin_mode 1``. A value of 1
adds the legacy optional chiral channel to the Spin NEP Lite descriptor.

This keyword is not part of the unified Spin2 protocol. Spin2 joint
spin--space information is controlled by :ref:`spin_soc <kw_spin_soc>` and
:ref:`spin_l_max <kw_spin_l_max>` instead.
