/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.
*/

#pragma once
namespace
{
__global__ void add_spin_training_outputs(
  int atom_count,
  const double* spin_force,
  const double* spin_mforce,
  const double* spin_virial,
  float* force,
  float* mforce,
  float* virial)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  for (int component = 0; component < 3; ++component) {
    const int index = component * atom_count + atom;
    force[index] += static_cast<float>(spin_force[index]);
    mforce[index] = static_cast<float>(spin_mforce[index]);
  }
  constexpr int source_component[6] = {0, 1, 2, 3, 5, 7};
  for (int component = 0; component < 6; ++component) {
    virial[component * atom_count + atom] +=
      static_cast<float>(spin_virial[source_component[component] * atom_count + atom]);
  }
}

} // namespace
