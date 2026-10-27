/* This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include "model/atom.cuh"
#include "utilities/error.cuh"

namespace spin_dof {
inline void initialize(Atom& atom, const std::vector<int>& active_types)
{
  const int n = atom.number_of_atoms;
  if (!atom.has_spin || atom.spin_per_atom.size() != 3 * n ||
      atom.spin_velocity_per_atom.size() != 3 * n) {
    PRINT_INPUT_ERROR("NEP_Spin requires spin:R:3 in model.xyz.");
  }
  std::vector<double> spin(3 * n), velocity(3 * n);
  atom.spin_per_atom.copy_to_host(spin.data());
  atom.spin_velocity_per_atom.copy_to_host(velocity.data());
  atom.cpu_spin_dof.resize(n);
  for (int i = 0; i < n; ++i) {
    const int type = atom.cpu_type[i];
    if (type < 0 || type >= active_types.size()) {
      PRINT_INPUT_ERROR("Spin atom type is outside the model's element list.");
    }
    atom.cpu_spin_dof[i] = active_types[type];
    if (!atom.cpu_spin_dof[i]) {
      for (int d = 0; d < 3; ++d) {
        if (spin[d*n+i] != 0.0)
          PRINT_INPUT_ERROR("Atoms outside spin_type must have zero spin.");
        if (velocity[d*n+i] != 0.0)
          PRINT_INPUT_ERROR("Atoms outside spin_type must have zero spin velocity.");
      }
    }
  }
  atom.spin_dof.resize(n);
  atom.spin_dof.copy_from_host(atom.cpu_spin_dof.data());
}

static __global__ void mask_force(int n, const int* type, const int* active, double* force)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && !active[type[i]]) {
    force[i] = force[n+i] = force[2*n+i] = 0.0;
  }
}
} // namespace spin_dof
