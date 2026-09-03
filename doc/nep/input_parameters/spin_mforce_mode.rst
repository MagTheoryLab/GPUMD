.. _kw_spin_mforce_mode:
.. index::
   single: spin_mforce_mode (keyword in nep.in)

:attr:`spin_mforce_mode`
=========================

Spin3 training requires this keyword explicitly::

  spin_mforce_mode full
  spin_mforce_mode transverse

``full`` compares all three Cartesian components of
:math:`-\partial U/\partial\boldsymbol{s}_i` and normalizes the RMSE by three
degrees of freedom per active spin. ``transverse`` first removes the component
parallel to :math:`\boldsymbol{s}_i`, excludes zero-length spins, and normalizes
by two degrees of freedom. The torque RMSE also excludes zero-length spins and
uses two degrees of freedom.

This is a training-data interpretation and is not serialized in ``nep.txt``.
Runtime ``mforce`` output always remains the full Cartesian energy derivative.
