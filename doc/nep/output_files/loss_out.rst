.. _loss_out:
.. index::
   single: loss.txt (output file)

``loss.out``
============

This files contains the terms that enter the :ref:`loss function <nep_loss_function>`, written every :ref:`output_interval <kw_output_interval>` generations (100 by default).

If a potential model is trained, each row contains the following fields::

  gen L_t L_1 L_2 L_e_train L_f_train L_v_train L_e_test L_f_test L_v_test

where

* :attr:`gen` is the current generation.
* :attr:`L_t` is the total loss function.
* :attr:`L_1` is the loss function related to the :math:`\mathcal{L}_1` regularization.
* :attr:`L_2` is the loss function related to the :math:`\mathcal{L}_2` regularization.
* :attr:`L_e_train` is the energy RMSE (in units of eV/atom) for the training set.
* :attr:`L_f_train` is the force RMSE (in units of eV/Å) for the training set.
* :attr:`L_v_train` is the virial RMSE (in units of eV/atom) for the training set.
* :attr:`L_e_test` is the energy RMSE (in units of eV/atom) for the test set.
* :attr:`L_f_test` is the force RMSE (in units of eV/Å) for the test set.
* :attr:`L_v_test` is the virial RMSE (in units of eV/atom) for the test set.

If a dipole model is trained, each row contains the following fields::

  gen L_t L_1 L_2 L_mu_train L_mu_test

where

* :attr:`L_mu_train` is the dipole RMSE (per atom) for the training set.
* :attr:`L_mu_test` is the dipole RMSE (per atom) for the test set.

If a polarizability model is trained, each row contains the following fields::

  gen L_t L_1 L_2 L_alpha_train L_alpha_test

where

* :attr:`L_alpha_train` is the polarizability RMSE (per atom) for the training set.
* :attr:`L_alpha_test` is the polarizability RMSE (per atom) for the test set.

If a Spin NEP potential is trained, each row contains the following fields::

  gen L_t L_1 L_2 L_e_train L_f_train L_v_train L_m_train L_tau_train L_e_test L_f_test L_v_test L_m_test L_tau_test

Here, :attr:`L_m` is the Cartesian magnetic-force RMSE in eV per unit spin and
:attr:`L_tau` is the RMSE of
:math:`\boldsymbol{s}_i\times\boldsymbol{M}_i`, also in eV per unit spin.
Only structures containing magnetic-force labels contribute to these two
terms. The grouped magnetic-response loss, when enabled, contributes to
:attr:`L_t` but currently has no separate output column.
