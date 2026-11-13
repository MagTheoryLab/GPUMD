/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.
*/

#ifdef NEP_SPECIALIZED_RUNTIME_BUILD
#include "model/box.cuh"
#include "force/nep_spin/spin_layout.cuh"
#include "nep_special_config.cuh"
#include "nep_spin.cuh"
#include "force/nep_spin/spin_kernels.cuh"
#include "utilities/gpu_macro.cuh"

extern "C" void nep_spin_launch_descriptor_v2(
  int atom_count,
  int struct_dim,
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
  const float* projection,
  float* moments,
  float* descriptors,
  double* wide_spin_descriptors,
  const double* wide_radial)
{
  nep_spin::CompiledLayout layout;
  if (layout.edge_cached && r12_plane_size > 0) {
    float* cache = const_cast<float*>(descriptor_coefficients) +
      NUM_TYPES_JIT*NUM_TYPES_JIT*layout.channels*(BASIS_SIZE_ANGULAR_JIT+1);
    layout.edge_radial = cache;
    nep_spin::cache_radial_edges<nep_spin::CompiledLayout, BASIS_SIZE_ANGULAR_JIT>
      <<<(r12_plane_size+127)/128, 128>>>(atom_count, r12_plane_size, NUM_TYPES_JIT,
        type, NN_spin, NL_spin, r12_spin, spin_cutoff_pair, descriptor_coefficients, cache);
    GPU_CHECK_KERNEL
  }
#ifdef NEP_SPIN_WIDE_FORWARD
  auto* radial = wide_radial;
  auto* q = wide_spin_descriptors;
  struct_dim = 0;
#else
  auto* radial = descriptor_coefficients;
  auto* q = descriptors;
#endif
  nep_spin::descriptors<<<(atom_count*nep_spin::CompiledLayout::channels*nep_spin::CompiledLayout::tile+63)/64, 64>>>(
    layout, atom_count, struct_dim, NUM_TYPES_JIT,
    N_MAX_ANGULAR_JIT, BASIS_SIZE_ANGULAR_JIT, Box{}, type, spin_type_active,
    position, spin, NN_spin, NL_spin, r12_spin, r12_plane_size, spin_cutoff_pair,
    radial, projection, moments, q);
  GPU_CHECK_KERNEL
}

extern "C" void nep_spin_launch_force_v2(
  int atom_count,
  int struct_dim,
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
  const float* projection,
  const float* moments,
  double* force,
  double* mforce,
  double* virial,
  const double* wide_radial)
{
  nep_spin::CompiledLayout layout;
  if (layout.edge_cached)
    layout.edge_radial = descriptor_coefficients +
      NUM_TYPES_JIT*NUM_TYPES_JIT*layout.channels*(BASIS_SIZE_ANGULAR_JIT+1);
#ifdef NEP_SPIN_WIDE_FORWARD
  const auto* radial = wide_radial;
#else
  const auto* radial = descriptor_coefficients;
#endif
  nep_spin::forces<<<(atom_count*nep_spin::CompiledLayout::channels*nep_spin::CompiledLayout::tile+63)/64, 64>>>(
    layout, atom_count, struct_dim, NUM_TYPES_JIT,
    N_MAX_ANGULAR_JIT, BASIS_SIZE_ANGULAR_JIT, Box{}, type, spin_type_active,
    position, spin, NN_spin, NL_spin, r12_spin, r12_plane_size, spin_cutoff_pair,
    radial, projection, moments, Fp,
    force, mforce, virial, true);
  GPU_CHECK_KERNEL
}

extern "C" int nep_train_specialized_interface_version()
{
  return NEP_SPECIAL_CONFIG_INTERFACE_VERSION;
}
#endif
