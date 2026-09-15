.. _kw_add_mfield:

:attr:`add_mfield`
=================

Apply a constant external magnetic induction to a group of atoms::

  add_mfield <group_method> <group_id> <Bx> <By> <Bz>

Grouping method and group ID select an existing model.xyz group, as in
add_efield. The three finite field components are in tesla (T). There is no
separate magnitude or direction normalization. For a 10 T field along z::

  add_mfield 0 0 0 0 10

To select all atoms, assign them to a common group in model.xyz. To select a
spatial region, assign its atoms to a group when preparing model.xyz. This is
an atom selection, not a moving spatial boundary: atoms retain their group
when they move. No separate region or gradient syntax is provided.

A Spin potential and spin:R:3 are required for a nonzero field. The field acts
on all selected magnetic moments, independently of the model spin_dof_type
response mask. It does not impose an integration constraint.

Each command replaces the previous field and group selection. The setting
persists across run commands. To turn it off::

  add_mfield 0 0 0 0 0

Spin is in muB, magnetic force is in eV/muB, and energy is in eV. With
c_B = 5.7883818060e-5, the added magnetic force is c_B times the field vector;
the added per-atom Zeeman energy is minus spin dotted with this force.
Both contributions are recomputed at initial, SIB midpoint, and endpoint
states. Output energies and magnetic forces include these contributions.
No additional g factor is applied. There is no direct mechanical force or
virial contribution. Time-dependent fields and linear gradients are not supported.
