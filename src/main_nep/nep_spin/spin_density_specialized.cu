/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.
*/

#ifdef NEP_SPECIALIZED_RUNTIME_BUILD
#include "model/box.cuh"
#include "force/nep_spin/spin_density_layout.cuh"
#include "nep_special_config.cuh"
#include "nep_spin.cuh"
#include "force/nep_spin/nep_spin_density.cuh"
#include "utilities/gpu_macro.cuh"

extern "C" void nep_spin_launch_descriptor(
  NEP_Spin_Model::Spin_Polynomial_Layout,
  int atom_count,
  int struct_dim,
  int num_types,
  int spin_basis_size,
  float spin_cutoff,
  const float* spin_cutoff_pair,
  const int* type,
  const int* spin_type_active,
  const double* position,
  const double* spin,
  const int* NN_spin,
  const int* NL_spin,
  const double* r12_spin,
  int r12_plane_size,
  const float* descriptor_coefficients,
  int spin_coefficient_offset,
  const float* projection,
  float* moments,
  float* descriptors)
{
  spin_density::CompiledLayout layout;
  if (layout.edge_cached && r12_plane_size > 0) {
    float* cache = const_cast<float*>(descriptor_coefficients) +
      NUM_TYPES_JIT*NUM_TYPES_JIT*layout.channels*(BASIS_SIZE_ANGULAR_JIT+1);
    layout.edge_radial = cache;
    spin_density::cache_radial_edges<spin_density::CompiledLayout, BASIS_SIZE_ANGULAR_JIT>
      <<<(r12_plane_size+127)/128, 128>>>(atom_count, r12_plane_size, NUM_TYPES_JIT,
        type, NN_spin, NL_spin, r12_spin, spin_cutoff_pair, descriptor_coefficients, cache);
    GPU_CHECK_KERNEL
  }
  spin_density::descriptors<<<(atom_count*spin_density::CompiledLayout::channels*spin_density::CompiledLayout::tile+63)/64, 64>>>(
    layout, atom_count, struct_dim, NUM_TYPES_JIT,
    N_MAX_ANGULAR_JIT, BASIS_SIZE_ANGULAR_JIT, Box{}, type, spin_type_active,
    position, spin, NN_spin, NL_spin, r12_spin, r12_plane_size, spin_cutoff_pair,
    descriptor_coefficients, projection, moments, descriptors);
  GPU_CHECK_KERNEL
}

extern "C" void nep_spin_launch_force(
  NEP_Spin_Model::Spin_Polynomial_Layout,
  int atom_count,
  int struct_dim,
  int num_types,
  int spin_basis_size,
  float spin_cutoff,
  const float* spin_cutoff_pair,
  const int* type,
  const int* spin_type_active,
  const double* position,
  const double* spin,
  const int* NN_spin,
  const int* NL_spin,
  const double* r12_spin,
  int r12_plane_size,
  const float* Fp,
  const float* descriptor_coefficients,
  int spin_coefficient_offset,
  const float* projection,
  const float* moments,
  double* force,
  double* mforce,
  double* virial)
{
  spin_density::CompiledLayout layout;
  if (layout.edge_cached)
    layout.edge_radial = descriptor_coefficients +
      NUM_TYPES_JIT*NUM_TYPES_JIT*layout.channels*(BASIS_SIZE_ANGULAR_JIT+1);
  spin_density::forces<<<(atom_count*spin_density::CompiledLayout::channels*spin_density::CompiledLayout::tile+63)/64, 64>>>(
    layout, atom_count, struct_dim, NUM_TYPES_JIT,
    N_MAX_ANGULAR_JIT, BASIS_SIZE_ANGULAR_JIT, Box{}, type, spin_type_active,
    position, spin, NN_spin, NL_spin, r12_spin, r12_plane_size, spin_cutoff_pair,
    descriptor_coefficients, projection, moments, Fp,
    force, mforce, virial, true);
  GPU_CHECK_KERNEL
}

extern "C" int nep_train_specialized_interface_version()
{
  return NEP_SPECIAL_CONFIG_INTERFACE_VERSION;
}
#endif
