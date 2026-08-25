/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under GPLv3 or later.

    The Spin NEP Lite descriptor and derivative math is shared with the
    production NEP_Spin runtime. Its model semantics follow TorchNEP
    MagTheoryLab commit e11369b6d1ec6fc6d1529e76754475a08b98744f.
*/

#include "nep_spin.cuh"
#include "dataset.cuh"
#include "parameters.cuh"
#include "model/box.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>

namespace {

using SimulationBox = Box;

struct TrainingSpinLayout {
  int channels = 0;
  int basis_count = 0;
  int l_max = 0;
  int chi_channels = 0;
  int rho0_offset = -1;
  int l1_rdot_offset = -1;
  int l1_cross_offset = -1;
  int l1_stf_offset = -1;
  int angular2_offset = -1;
  int angular3_offset = -1;
  int angular4_offset = -1;
  int geom_offset = -1;
  int rho0_dot_offset = -1;
  int raw1_dot_offset = -1;
  int chiral_offset = -1;
  int descriptor_dim = 0;
};

using SpinCoreLayout = TrainingSpinLayout;

enum class SpinVirialMode : int {
  disabled,
  center_owned,
  neighbor_owned,
  center_and_neighbor_float_sink,
};

constexpr int kSpinDeg2Count = 6;
constexpr int kSpinDeg3Count = 10;
constexpr int kSpinDeg4Count = 15;
constexpr double kPi = 3.14159265358979323846;
constexpr int kMaxSpinBasis = 8;
constexpr int kSpinChiralOReducedCount = 7;
constexpr int kSpinChiralHReducedCount = 9;
constexpr int kSpinChiralQohReducedCount = 50;

template <int C, int LMax>
struct SpinStaticLayout {
  static constexpr int Rho0Offset = 2 + 4 * C;
  static constexpr int L1RdotOffset = Rho0Offset + C;
  static constexpr int L1CrossOffset = L1RdotOffset + C;
  static constexpr int L1StfOffset = L1CrossOffset + C;
  static constexpr int Angular2Offset = Rho0Offset + C + (LMax >= 1 ? 3 * C : 0);
  static constexpr int Angular3Offset = Angular2Offset + (LMax >= 2 ? C : 0);
  static constexpr int Angular4Offset = Angular3Offset + (LMax >= 3 ? C : 0);
  static constexpr int GeomOffset = Angular4Offset + (LMax >= 4 ? C : 0);
  static constexpr int Rho0DotOffset = GeomOffset + C;
  static constexpr int Raw1DotOffset = Rho0DotOffset + C;
};

__device__ __constant__ unsigned short
kSpinChiralQohReducedPacked[kSpinChiralQohReducedCount] = {
  0, 32, 34, 49, 68, 70, 87, 100, 257, 259, 272, 274, 289, 291, 304, 306, 325,
  327, 328, 340, 342, 357, 517, 519, 532, 534, 549, 552, 564, 577, 579, 592, 594,
  611, 774, 791, 804, 824, 832, 849, 851, 866, 1041, 1056, 1073, 1075, 1094, 1109,
  1111, 1124};

__device__ __constant__ float
kSpinChiralQohReducedCoeff[kSpinChiralQohReducedCount] = {
  1.0f,  -2.0f, 1.0f,  2.0f,  1.0f,  -1.0f, 2.0f,  0.5f,  1.0f,  1.0f,
  1.0f,  1.0f,  -4.0f, -3.0f, -4.0f, -3.0f, -0.5f, -2.0f, -1.0f, -1.0f,
  -4.0f, -0.5f, 1.0f,  -2.0f, 1.0f,  2.0f,  -1.0f, -2.0f, 1.0f,  -2.0f,
  6.0f,  -4.0f, -8.0f, 2.0f,  1.0f,  1.0f,  -0.5f, 1.0f,  1.0f,  -2.0f,
  -4.0f, 1.0f,  1.0f,  2.0f,  -2.0f, 1.0f,  1.0f,  1.0f,  -2.0f, -0.5f};

__device__ __forceinline__ void minimum_image_delta(
  const SimulationBox& box,
  const double x,
  const double y,
  const double z,
  float& dx,
  float& dy,
  float& dz)
{
  dx = static_cast<float>(x);
  dy = static_cast<float>(y);
  dz = static_cast<float>(z);
  apply_mic(box, dx, dy, dz);
}

__device__ __forceinline__ void atomic_add_per_atom_virial_double(
  const int atom_stride,
  const int atom,
  const float x12,
  const float y12,
  const float z12,
  const float fx,
  const float fy,
  const float fz,
  double* virial)
{
  atomicAdd(&virial[atom], -static_cast<double>(x12 * fx));
  atomicAdd(&virial[atom_stride + atom], -static_cast<double>(y12 * fy));
  atomicAdd(&virial[2 * atom_stride + atom], -static_cast<double>(z12 * fz));
  atomicAdd(&virial[3 * atom_stride + atom], -static_cast<double>(x12 * fy));
  atomicAdd(&virial[4 * atom_stride + atom], -static_cast<double>(x12 * fz));
  atomicAdd(&virial[5 * atom_stride + atom], -static_cast<double>(y12 * fz));
  atomicAdd(&virial[6 * atom_stride + atom], -static_cast<double>(y12 * fx));
  atomicAdd(&virial[7 * atom_stride + atom], -static_cast<double>(z12 * fx));
  atomicAdd(&virial[8 * atom_stride + atom], -static_cast<double>(z12 * fy));
}

__device__ __forceinline__ void atomic_add_per_atom_virial_float(
  const int atom_stride,
  const int atom,
  const float x12,
  const float y12,
  const float z12,
  const float fx,
  const float fy,
  const float fz,
  float* virial)
{
  atomicAdd(&virial[atom], -x12 * fx);
  atomicAdd(&virial[atom_stride + atom], -y12 * fy);
  atomicAdd(&virial[2 * atom_stride + atom], -z12 * fz);
  atomicAdd(&virial[3 * atom_stride + atom], -x12 * fy);
  atomicAdd(&virial[4 * atom_stride + atom], -x12 * fz);
  atomicAdd(&virial[5 * atom_stride + atom], -y12 * fz);
  atomicAdd(&virial[6 * atom_stride + atom], -y12 * fx);
  atomicAdd(&virial[7 * atom_stride + atom], -z12 * fx);
  atomicAdd(&virial[8 * atom_stride + atom], -z12 * fy);
}

__device__ __forceinline__ void atomic_add_spin_transfer_float(
  int, int, float, float, float, const float*, float*)
{
}

#include "../force/nep_spin_descriptor.cuh"
#define find_mforce_onsite find_mforce_onsite_training
#define mask_inactive_spin_mforce mask_inactive_spin_mforce_training
#include "../force/nep_spin_force.cuh"
#undef mask_inactive_spin_mforce
#undef find_mforce_onsite

template <int C, int LMax>
void launch_spin_descriptor(
  int atom_count,
  int struct_dim,
  int num_types,
  int spin_basis_size,
  int spin_chiral,
  float spin_cutoff,
  TrainingSpinLayout layout,
  const int* type,
  const int* spin_dof_type_active,
  const int* spin_env_type_active,
  const double* position,
  const double* spin,
  const int* NN_spin,
  const int* NL_spin,
  const double* r12_spin,
  int r12_plane_size,
  const float* descriptor_coefficients,
  int spin_coefficient_offset,
  float* rho0,
  float* raw1,
  float* angular2,
  float* angular3,
  float* angular4,
  float* geom,
  float* rho0_dot,
  float* raw1_dot,
  float* polar,
  float* octupole,
  float* hexadecapole,
  float* chirals,
  float* descriptors)
{
  Box box{};
  if (spin_chiral) {
    find_spin_descriptor<C, LMax, true><<<atom_count, 128>>>(
      atom_count,
      atom_count,
      struct_dim,
      num_types,
      spin_basis_size,
      spin_cutoff,
      box,
      type,
      spin_dof_type_active,
      spin_env_type_active,
      position,
      spin,
      NN_spin,
      NL_spin,
      r12_spin,
      r12_plane_size,
      descriptor_coefficients,
      spin_coefficient_offset,
      rho0,
      raw1,
      angular2,
      angular3,
      angular4,
      geom,
      rho0_dot,
      raw1_dot,
      polar,
      octupole,
      hexadecapole,
      descriptors);
    const int work_items = atom_count * C;
    find_spin_descriptor_chiral<C><<<(work_items + 127) / 128, 128>>>(
      atom_count,
      atom_count,
      struct_dim,
      layout,
      spin,
      geom,
      raw1,
      polar,
      octupole,
      hexadecapole,
      chirals,
      descriptors);
  } else {
    find_spin_descriptor<C, LMax, false><<<atom_count, 128>>>(
      atom_count,
      atom_count,
      struct_dim,
      num_types,
      spin_basis_size,
      spin_cutoff,
      box,
      type,
      spin_dof_type_active,
      spin_env_type_active,
      position,
      spin,
      NN_spin,
      NL_spin,
      r12_spin,
      r12_plane_size,
      descriptor_coefficients,
      spin_coefficient_offset,
      rho0,
      raw1,
      angular2,
      angular3,
      angular4,
      geom,
      rho0_dot,
      raw1_dot,
      polar,
      octupole,
      hexadecapole,
      descriptors);
  }
  GPU_CHECK_KERNEL
}

template <int C>
void launch_spin_descriptor_lmax(
  int l_max,
  int atom_count,
  int struct_dim,
  int num_types,
  int spin_basis_size,
  int spin_chiral,
  float spin_cutoff,
  TrainingSpinLayout layout,
  const int* type,
  const int* spin_dof_type_active,
  const int* spin_env_type_active,
  const double* position,
  const double* spin,
  const int* NN_spin,
  const int* NL_spin,
  const double* r12_spin,
  int r12_plane_size,
  const float* descriptor_coefficients,
  int spin_coefficient_offset,
  float* rho0,
  float* raw1,
  float* angular2,
  float* angular3,
  float* angular4,
  float* geom,
  float* rho0_dot,
  float* raw1_dot,
  float* polar,
  float* octupole,
  float* hexadecapole,
  float* chirals,
  float* descriptors)
{
#define LAUNCH_DESCRIPTOR(L) \
  launch_spin_descriptor<C, L>( \
    atom_count, struct_dim, num_types, spin_basis_size, spin_chiral, spin_cutoff, \
    layout, type, spin_dof_type_active, spin_env_type_active, position, spin, \
    NN_spin, NL_spin, r12_spin, r12_plane_size, descriptor_coefficients, \
    spin_coefficient_offset, rho0, raw1, angular2, angular3, angular4, geom, \
    rho0_dot, raw1_dot, polar, octupole, hexadecapole, chirals, descriptors)
  switch (l_max) {
    case 0: LAUNCH_DESCRIPTOR(0); break;
    case 1: LAUNCH_DESCRIPTOR(1); break;
    case 2: LAUNCH_DESCRIPTOR(2); break;
    case 3: LAUNCH_DESCRIPTOR(3); break;
    case 4: LAUNCH_DESCRIPTOR(4); break;
  }
#undef LAUNCH_DESCRIPTOR
}

template <int C, int LMax>
void launch_spin_density_force(
  int atom_count,
  int struct_dim,
  int num_types,
  int spin_basis_size,
  float spin_cutoff,
  const int* type,
  const int* spin_dof_type_active,
  const int* spin_env_type_active,
  const double* position,
  const double* spin,
  const int* NN_spin,
  const int* NL_spin,
  const double* r12_spin,
  int r12_plane_size,
  const float* Fp,
  const float* descriptor_coefficients,
  int spin_coefficient_offset,
  const float* rho0,
  const float* angular2,
  const float* angular3,
  const float* angular4,
  const float* geom,
  const float* rho0_dot,
  const float* raw1,
  const float* raw1_dot,
  double* force,
  double* mforce,
  double* virial)
{
  constexpr int atoms_per_warp = 4;
  constexpr int edges_per_atom = 8;
  Box box{};
  find_force_spin_density<
    C,
    LMax,
    SpinVirialMode::neighbor_owned,
    false,
    atoms_per_warp,
    edges_per_atom><<<(atom_count + atoms_per_warp - 1) / atoms_per_warp, 32>>>(
    atom_count,
    atom_count,
    struct_dim,
    num_types,
    spin_basis_size,
    spin_cutoff,
    box,
    type,
    spin_dof_type_active,
    spin_env_type_active,
    position,
    spin,
    NN_spin,
    NL_spin,
    r12_spin,
    r12_plane_size,
    Fp,
    descriptor_coefficients,
    spin_coefficient_offset,
    rho0,
    angular2,
    angular3,
    angular4,
    geom,
    rho0_dot,
    raw1,
    raw1_dot,
    force,
    mforce,
    virial,
    nullptr,
    nullptr);
}

template <int C>
void launch_spin_chiral_force(
  int atom_count,
  int struct_dim,
  int num_types,
  int spin_basis_size,
  float spin_cutoff,
  TrainingSpinLayout layout,
  const int* type,
  const int* spin_dof_type_active,
  const int* spin_env_type_active,
  const double* position,
  const double* spin,
  const int* NN_spin,
  const int* NL_spin,
  const double* r12_spin,
  int r12_plane_size,
  const float* Fp,
  const float* descriptor_coefficients,
  int spin_coefficient_offset,
  const float* geom,
  const float* raw1,
  const float* polar,
  const float* octupole,
  const float* hexadecapole,
  const float* chirals,
  double* force,
  double* mforce,
  double* virial)
{
  constexpr int atoms_per_warp = 8;
  constexpr int edges_per_atom = 4;
  Box box{};
  find_force_spin_chiral<
    C,
    SpinVirialMode::neighbor_owned,
    false,
    atoms_per_warp,
    edges_per_atom><<<(atom_count + atoms_per_warp - 1) / atoms_per_warp, 32>>>(
    atom_count,
    atom_count,
    struct_dim,
    num_types,
    spin_basis_size,
    layout,
    spin_cutoff,
    box,
    type,
    spin_dof_type_active,
    spin_env_type_active,
    position,
    spin,
    NN_spin,
    NL_spin,
    r12_spin,
    r12_plane_size,
    Fp,
    descriptor_coefficients,
    spin_coefficient_offset,
    geom,
    raw1,
    polar,
    octupole,
    hexadecapole,
    chirals,
    force,
    mforce,
    virial,
    nullptr,
    nullptr);
}

__global__ void add_spin_training_outputs(
  int atom_count,
  const int* type,
  const float* baseline,
  const double* spin_force,
  const double* spin_mforce,
  const double* spin_virial,
  float* energy,
  float* force,
  float* mforce,
  float* virial)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  energy[atom] += baseline[type[atom]];
  for (int component = 0; component < 3; ++component) {
    const int index = component * atom_count + atom;
    force[index] += static_cast<float>(spin_force[index]);
    mforce[index] = static_cast<float>(spin_mforce[index]);
  }
  constexpr int source_component[6] = {0, 1, 2, 3, 5, 7};
  for (int component = 0; component < 6; ++component) {
    virial[component * atom_count + atom] += static_cast<float>(
      spin_virial[source_component[component] * atom_count + atom]);
  }
}

} // namespace

NEP_Spin_Trainer::NEP_Spin_Trainer(
  Parameters& para, int N, int version, int device_count)
  : NEP(para, N, version, device_count)
{
  spin_compress_ = para.spin_compress;
  spin_basis_size_ = para.spin_basis_size[0];
  spin_l_max_ = para.spin_l_max[0];
  spin_chiral_ = para.spin_chiral;
  struct_dim_ = para.dim_struct;
  spin_coefficient_offset_ =
    para.number_of_variables_descriptor - para.number_of_variables_descriptor_spin;
  num_types_ = para.num_types;
  spin_cutoff_ = para.spin_cutoff[0];

  layout_.channels = spin_compress_;
  layout_.basis_count = spin_basis_size_ + 1;
  layout_.l_max = spin_l_max_;
  layout_.chi_channels = std::min(2, spin_compress_);
  int offset = 2 + 4 * spin_compress_;
  layout_.rho0_offset = offset;
  offset += spin_compress_;
  if (spin_l_max_ >= 1) {
    layout_.l1_rdot_offset = offset;
    offset += spin_compress_;
    layout_.l1_cross_offset = offset;
    offset += spin_compress_;
    layout_.l1_stf_offset = offset;
    offset += spin_compress_;
  }
  if (spin_l_max_ >= 2) {
    layout_.angular2_offset = offset;
    offset += spin_compress_;
  }
  if (spin_l_max_ >= 3) {
    layout_.angular3_offset = offset;
    offset += spin_compress_;
  }
  if (spin_l_max_ >= 4) {
    layout_.angular4_offset = offset;
    offset += spin_compress_;
  }
  layout_.geom_offset = offset;
  offset += spin_compress_;
  layout_.rho0_dot_offset = offset;
  offset += spin_compress_;
  if (spin_l_max_ >= 1) {
    layout_.raw1_dot_offset = offset;
    offset += spin_compress_;
  }
  if (spin_chiral_) {
    layout_.chiral_offset = offset;
    offset += layout_.chi_channels + 2 * spin_compress_;
  }
  layout_.descriptor_dim = offset;

  for (int device_id = 0; device_id < device_count; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    auto& data = spin_data_[device_id];
    data.spin_dof_type_active.resize(num_types_);
    data.spin_env_type_active.resize(num_types_);
    data.spin_baseline.resize(num_types_);
    data.spin_dof_type_active.copy_from_host(para.spin_dof_type_active.data());
    data.spin_env_type_active.copy_from_host(para.spin_env_type_active.data());
    data.spin_baseline.copy_from_host(para.spin_baseline.data());
    data.rho0.resize(3 * spin_compress_ * N);
    data.raw1.resize(9 * spin_compress_ * N);
    data.angular2.resize(spin_l_max_ >= 2 ? 15 * spin_compress_ * N : 1);
    data.angular3.resize(spin_l_max_ >= 3 ? 21 * spin_compress_ * N : 1);
    data.angular4.resize(spin_l_max_ >= 4 ? 27 * spin_compress_ * N : 1);
    data.geom.resize(6 * spin_compress_ * N);
    data.rho0_dot.resize(3 * spin_compress_ * N);
    data.raw1_dot.resize(9 * spin_compress_ * N);
    data.polar.resize(spin_chiral_ ? 3 * spin_compress_ * N : 1);
    data.octupole.resize(
      spin_chiral_ ? kSpinChiralOReducedCount * spin_compress_ * N : 1);
    data.hexadecapole.resize(
      spin_chiral_ ? kSpinChiralHReducedCount * layout_.chi_channels * N : 1);
    data.chirals.resize(spin_chiral_ ? layout_.chi_channels * N : 1);
    data.force.resize(3 * N);
    data.mforce.resize(3 * N);
    data.virial.resize(9 * N);
  }
}

void NEP_Spin_Trainer::find_additional_descriptors(
  Parameters&, Dataset& dataset, int device_id)
{
  auto& data = spin_data_[device_id];
  TrainingSpinLayout layout;
  layout.channels = layout_.channels;
  layout.basis_count = layout_.basis_count;
  layout.l_max = layout_.l_max;
  layout.chi_channels = layout_.chi_channels;
  layout.rho0_offset = layout_.rho0_offset;
  layout.l1_rdot_offset = layout_.l1_rdot_offset;
  layout.l1_cross_offset = layout_.l1_cross_offset;
  layout.l1_stf_offset = layout_.l1_stf_offset;
  layout.angular2_offset = layout_.angular2_offset;
  layout.angular3_offset = layout_.angular3_offset;
  layout.angular4_offset = layout_.angular4_offset;
  layout.geom_offset = layout_.geom_offset;
  layout.rho0_dot_offset = layout_.rho0_dot_offset;
  layout.raw1_dot_offset = layout_.raw1_dot_offset;
  layout.chiral_offset = layout_.chiral_offset;
  layout.descriptor_dim = layout_.descriptor_dim;

#define LAUNCH_DESCRIPTOR(C) \
  launch_spin_descriptor_lmax<C>( \
    spin_l_max_, dataset.N, struct_dim_, num_types_, spin_basis_size_, \
    spin_chiral_, spin_cutoff_, layout, dataset.type.data(), \
    data.spin_dof_type_active.data(), data.spin_env_type_active.data(), \
    dataset.r_spin.data(), dataset.spin.data(), dataset.NN_spin.data(), \
    dataset.NL_spin.data(), dataset.r12_spin.data(), \
    dataset.N * dataset.max_NN_spin, annmb[device_id].c, \
    spin_coefficient_offset_, data.rho0.data(), data.raw1.data(), \
    data.angular2.data(), data.angular3.data(), data.angular4.data(), \
    data.geom.data(), data.rho0_dot.data(), data.raw1_dot.data(), \
    data.polar.data(), data.octupole.data(), data.hexadecapole.data(), \
    data.chirals.data(), nep_data[device_id].descriptors.data())
  switch (spin_compress_) {
    case 1: LAUNCH_DESCRIPTOR(1); break;
    case 2: LAUNCH_DESCRIPTOR(2); break;
    case 3: LAUNCH_DESCRIPTOR(3); break;
    case 4: LAUNCH_DESCRIPTOR(4); break;
  }
#undef LAUNCH_DESCRIPTOR
}

void NEP_Spin_Trainer::initialize_additional_outputs(
  Dataset& dataset, int device_id)
{
  auto& data = spin_data_[device_id];
  data.force.fill(0.0);
  data.mforce.fill(0.0);
  data.virial.fill(0.0);
  dataset.mforce.fill(0.0f);
}

void NEP_Spin_Trainer::find_additional_force(
  Parameters&, Dataset& dataset, int device_id)
{
  auto& data = spin_data_[device_id];
  const int block_size = 128;
  const int grid_size = (dataset.N + block_size - 1) / block_size;
  find_mforce_onsite_training<<<grid_size, block_size>>>(
    dataset.N,
    dataset.N,
    struct_dim_,
    dataset.type.data(),
    data.spin_dof_type_active.data(),
    dataset.spin.data(),
    nep_data[device_id].Fp.data(),
    data.mforce.data());

  TrainingSpinLayout layout;
  layout.channels = layout_.channels;
  layout.basis_count = layout_.basis_count;
  layout.l_max = layout_.l_max;
  layout.chi_channels = layout_.chi_channels;
  layout.rho0_offset = layout_.rho0_offset;
  layout.l1_rdot_offset = layout_.l1_rdot_offset;
  layout.l1_cross_offset = layout_.l1_cross_offset;
  layout.l1_stf_offset = layout_.l1_stf_offset;
  layout.angular2_offset = layout_.angular2_offset;
  layout.angular3_offset = layout_.angular3_offset;
  layout.angular4_offset = layout_.angular4_offset;
  layout.geom_offset = layout_.geom_offset;
  layout.rho0_dot_offset = layout_.rho0_dot_offset;
  layout.raw1_dot_offset = layout_.raw1_dot_offset;
  layout.chiral_offset = layout_.chiral_offset;
  layout.descriptor_dim = layout_.descriptor_dim;

#define LAUNCH_DENSITY(C, L) \
  launch_spin_density_force<C, L>( \
    dataset.N, struct_dim_, num_types_, spin_basis_size_, spin_cutoff_, \
    dataset.type.data(), data.spin_dof_type_active.data(), \
    data.spin_env_type_active.data(), dataset.r_spin.data(), dataset.spin.data(), \
    dataset.NN_spin.data(), dataset.NL_spin.data(), dataset.r12_spin.data(), \
    dataset.N * dataset.max_NN_spin, nep_data[device_id].Fp.data(), \
    annmb[device_id].c, spin_coefficient_offset_, data.rho0.data(), \
    data.angular2.data(), data.angular3.data(), data.angular4.data(), \
    data.geom.data(), data.rho0_dot.data(), data.raw1.data(), \
    data.raw1_dot.data(), data.force.data(), data.mforce.data(), data.virial.data())
#define LAUNCH_FOR_C(C) \
  switch (spin_l_max_) { \
    case 0: LAUNCH_DENSITY(C, 0); break; \
    case 1: LAUNCH_DENSITY(C, 1); break; \
    case 2: LAUNCH_DENSITY(C, 2); break; \
    case 3: LAUNCH_DENSITY(C, 3); break; \
    case 4: LAUNCH_DENSITY(C, 4); break; \
  } \
  if (spin_chiral_) { \
    launch_spin_chiral_force<C>( \
      dataset.N, struct_dim_, num_types_, spin_basis_size_, spin_cutoff_, layout, \
      dataset.type.data(), data.spin_dof_type_active.data(), \
      data.spin_env_type_active.data(), dataset.r_spin.data(), dataset.spin.data(), \
      dataset.NN_spin.data(), dataset.NL_spin.data(), dataset.r12_spin.data(), \
      dataset.N * dataset.max_NN_spin, nep_data[device_id].Fp.data(), \
      annmb[device_id].c, spin_coefficient_offset_, data.geom.data(), \
      data.raw1.data(), data.polar.data(), data.octupole.data(), \
      data.hexadecapole.data(), data.chirals.data(), data.force.data(), \
      data.mforce.data(), data.virial.data()); \
  }
  switch (spin_compress_) {
    case 1: LAUNCH_FOR_C(1); break;
    case 2: LAUNCH_FOR_C(2); break;
    case 3: LAUNCH_FOR_C(3); break;
    case 4: LAUNCH_FOR_C(4); break;
  }
#undef LAUNCH_FOR_C
#undef LAUNCH_DENSITY

  mask_inactive_spin_mforce_training<<<grid_size, block_size>>>(
    dataset.N,
    dataset.N,
    dataset.type.data(),
    data.spin_dof_type_active.data(),
    data.mforce.data());
  add_spin_training_outputs<<<grid_size, block_size>>>(
    dataset.N,
    dataset.type.data(),
    data.spin_baseline.data(),
    data.force.data(),
    data.mforce.data(),
    data.virial.data(),
    dataset.energy.data(),
    dataset.force.data(),
    dataset.mforce.data(),
    dataset.virial.data());
  GPU_CHECK_KERNEL
}
