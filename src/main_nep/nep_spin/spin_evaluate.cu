/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.
*/

#include "main_nep/dataset.cuh"
#include "model/box.cuh"
#include "nep_spin.cuh"
#include "spin_parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>

#include "spin_training_kernels.cuh"
#include "force/nep_spin/nep_spin_density.cuh"
#include <sstream>
#include "spin_density_compile.cuh"

void NEP_Spin_Model::initialize_spin(Spin_Parameters& para, int N, int device_count)
{
  spin_compress_ = para.spin_compress;
  spin_basis_size_ = para.spin_basis_size[0];
  spin_l_max_ = para.spin_l_max[0];
  spin_order_ = para.spin_order;
  spin_soc_ = para.spin_soc;
  struct_dim_ = para.dim_struct;
  spin_coefficient_offset_ =
    para.number_of_variables_descriptor - para.number_of_variables_descriptor_spin;
  spin_coefficient_offset_ -= para.number_of_variables_spin_projection;
  spin_projection_offset_ = spin_coefficient_offset_ + para.number_of_variables_descriptor_spin;
  num_types_ = para.num_types;
  spin_cutoff_ = para.spin_cutoff;

  polynomial_layout_ = make_spin_polynomial_layout<Spin_Polynomial_Layout>(
    spin_compress_, spin_l_max_, spin_order_, spin_soc_);

  for (int device_id = 0; device_id < device_count; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    auto& data = spin_data_[device_id];
    data.spin_type_active.resize(num_types_);
    data.spin_type_active.copy_from_host(para.spin_type_active.data());
    std::vector<float> spin_cutoff_pair(static_cast<std::size_t>(num_types_) * num_types_);
    for (int type_i = 0; type_i < num_types_; ++type_i) {
      for (int type_j = 0; type_j < num_types_; ++type_j) {
        spin_cutoff_pair[type_i * num_types_ + type_j] =
          0.5f * (para.spin_cutoff_by_type[type_i] + para.spin_cutoff_by_type[type_j]);
      }
    }
    data.spin_cutoff_pair.resize(spin_cutoff_pair.size());
    data.spin_cutoff_pair.copy_from_host(spin_cutoff_pair.data(), spin_cutoff_pair.size());
    if (para.spin_version == 2) {
      data.density.initialize(para.spin_compress, para.spin_l_max[0], para.spin_order, para.spin_soc);
      data.spin3_moments.resize(N * data.density.view.channels * data.density.view.components);
      if (spin_descriptor_jit_)
        data.density_radial.resize(num_types_ * num_types_ * spin_compress_ * (para.basis_size_angular+1));
    } else data.spin3_moments.resize(N * polynomial_layout_.moment_count);
    data.force.resize(3 * N);
    data.mforce.resize(3 * N);
    data.virial.resize(9 * N);
  }
}

void NEP_Spin_Model::find_additional_descriptors(Spin_Parameters& para, Dataset& dataset, int device_id)
{
  auto& data = spin_data_[device_id];
  auto& magnetic_data = magnetic(dataset);
  const int plane_size = dataset.N * magnetic_data.max_NN_spin;
  const float* projection = annmb[device_id].c + spin_projection_offset_;
  if (para.spin_version == 2 && spin_descriptor_jit_) {
    const int count = num_types_ * num_types_ * spin_compress_ * (para.basis_size_angular+1);
    const size_t edge_count = para.spin_l_max[0] == 2 && para.spin_order == 2 ?
      size_t(2) * spin_compress_ * plane_size : 0;
    if (data.density_radial.size() < static_cast<size_t>(count) + edge_count)
      data.density_radial.resize(static_cast<size_t>(count) + edge_count);
    spin_density::contract_radial<<<(count+127)/128, 128>>>(
      count, spin_compress_, para.n_max_angular, para.basis_size_angular,
      annmb[device_id].c + num_types_ * num_types_ *
        (para.n_max_radial+1) * (para.basis_size_radial+1),
      projection, data.density_radial.data());
    GPU_CHECK_KERNEL
  }
  if (para.spin_version == 2 && !spin_descriptor_jit_) {
    auto launch_density = [&](auto layout) {
      spin_density::descriptors<<<(dataset.N * spin_compress_ + 63) / 64, 64>>>(
        layout, dataset.N, struct_dim_, num_types_, para.n_max_angular,
        para.basis_size_angular, Box{}, dataset.type.data(), data.spin_type_active.data(),
        magnetic_data.r_spin.data(), magnetic_data.spin.data(), magnetic_data.NN_spin.data(),
        magnetic_data.NL_spin.data(), magnetic_data.r12_spin.data(), plane_size,
        data.spin_cutoff_pair.data(), annmb[device_id].c +
          num_types_ * num_types_ * (para.n_max_radial + 1) * (para.basis_size_radial + 1),
        projection, data.spin3_moments.data(), nep_data[device_id].descriptors.data());
    };
    if (data.density.view.components <= 100)
      launch_density(spin_density::DynamicLayout(data.density.view));
    else
      launch_density(spin_density::WideLayout(data.density.view));
    GPU_CHECK_KERNEL
    return;
  }
#define LAUNCH_SPIN3_DESCRIPTOR(C)                                                                 \
  launch_spin3_descriptor<C>(                                                                      \
    polynomial_layout_,                                                                            \
    dataset.N,                                                                                     \
    struct_dim_,                                                                                   \
    num_types_,                                                                                    \
    spin_basis_size_,                                                                              \
    spin_cutoff_,                                                                                  \
    data.spin_cutoff_pair.data(),                                                                  \
    dataset.type.data(),                                                                           \
    data.spin_type_active.data(),                                                                  \
    magnetic_data.r_spin.data(),                                                                   \
    magnetic_data.spin.data(),                                                                     \
    magnetic_data.NN_spin.data(),                                                                  \
    magnetic_data.NL_spin.data(),                                                                  \
    magnetic_data.r12_spin.data(),                                                                 \
    plane_size,                                                                                    \
    annmb[device_id].c,                                                                            \
    spin_coefficient_offset_,                                                                      \
    projection,                                                                                    \
    data.spin3_moments.data(),                                                                     \
    nep_data[device_id].descriptors.data())
  if (spin_descriptor_jit_) {
    spin_descriptor_jit_(
      polynomial_layout_,
      dataset.N,
      struct_dim_,
      num_types_,
      spin_basis_size_,
      spin_cutoff_,
      data.spin_cutoff_pair.data(),
      dataset.type.data(),
      data.spin_type_active.data(),
      magnetic_data.r_spin.data(),
      magnetic_data.spin.data(),
      magnetic_data.NN_spin.data(),
      magnetic_data.NL_spin.data(),
      magnetic_data.r12_spin.data(),
      plane_size,
      para.spin_version == 2 ? data.density_radial.data() : annmb[device_id].c,
      spin_coefficient_offset_,
      projection,
      data.spin3_moments.data(),
      nep_data[device_id].descriptors.data());
  } else {
    switch (spin_compress_) {
      case 1:
        LAUNCH_SPIN3_DESCRIPTOR(1);
        break;
      case 2:
        LAUNCH_SPIN3_DESCRIPTOR(2);
        break;
      case 3:
        LAUNCH_SPIN3_DESCRIPTOR(3);
        break;
      case 4:
        LAUNCH_SPIN3_DESCRIPTOR(4);
        break;
      case 5:
        LAUNCH_SPIN3_DESCRIPTOR(5);
        break;
      case 6:
        LAUNCH_SPIN3_DESCRIPTOR(6);
        break;
      case 7:
        LAUNCH_SPIN3_DESCRIPTOR(7);
        break;
      case 8:
        LAUNCH_SPIN3_DESCRIPTOR(8);
        break;
      case 9:
        LAUNCH_SPIN3_DESCRIPTOR(9);
        break;
      default:
        PRINT_INPUT_ERROR("spin_compress should be an integer from 1 to 9.");
        break;
    }
  }
#undef LAUNCH_SPIN3_DESCRIPTOR
}

void NEP_Spin_Model::initialize_additional_outputs(Dataset& dataset, int device_id)
{
  auto& data = spin_data_[device_id];
  auto& magnetic_data = magnetic(dataset);
  data.force.fill(0.0);
  data.mforce.fill(0.0);
  data.virial.fill(0.0);
  magnetic_data.mforce.fill(0.0f);
}

void NEP_Spin_Model::find_additional_force(Spin_Parameters& para, Dataset& dataset, int device_id)
{
  auto& data = spin_data_[device_id];
  auto& magnetic_data = magnetic(dataset);
  const int block_size = 128;
  const int grid_size = (dataset.N + block_size - 1) / block_size;
  const int plane_size = dataset.N * magnetic_data.max_NN_spin;
  const float* projection = annmb[device_id].c + spin_projection_offset_;
  if (para.spin_version == 2 && !spin_force_jit_) {
    auto launch_density = [&](auto layout) {
      spin_density::forces<<<(dataset.N * spin_compress_ + 63) / 64, 64>>>(
        layout, dataset.N, struct_dim_, num_types_, para.n_max_angular,
        para.basis_size_angular, Box{}, dataset.type.data(), data.spin_type_active.data(),
        magnetic_data.r_spin.data(), magnetic_data.spin.data(), magnetic_data.NN_spin.data(),
        magnetic_data.NL_spin.data(), magnetic_data.r12_spin.data(), plane_size,
        data.spin_cutoff_pair.data(), annmb[device_id].c +
          num_types_ * num_types_ * (para.n_max_radial + 1) * (para.basis_size_radial + 1),
        projection, data.spin3_moments.data(), nep_data[device_id].Fp.data(), data.force.data(),
        data.mforce.data(), data.virial.data(), true);
    };
    if (data.density.view.components <= 100)
      launch_density(spin_density::DynamicLayout(data.density.view));
    else
      launch_density(spin_density::WideLayout(data.density.view));
    add_spin_training_outputs<<<grid_size, block_size>>>(dataset.N, data.force.data(),
      data.mforce.data(), data.virial.data(), dataset.force.data(),
      magnetic_data.mforce.data(), dataset.virial.data());
    GPU_CHECK_KERNEL
    return;
  }
#define LAUNCH_SPIN3_FORCE(C)                                                                      \
  launch_spin3_force<C>(                                                                           \
    polynomial_layout_,                                                                            \
    dataset.N,                                                                                     \
    struct_dim_,                                                                                   \
    num_types_,                                                                                    \
    spin_basis_size_,                                                                              \
    spin_cutoff_,                                                                                  \
    data.spin_cutoff_pair.data(),                                                                  \
    dataset.type.data(),                                                                           \
    data.spin_type_active.data(),                                                                  \
    magnetic_data.r_spin.data(),                                                                   \
    magnetic_data.spin.data(),                                                                     \
    magnetic_data.NN_spin.data(),                                                                  \
    magnetic_data.NL_spin.data(),                                                                  \
    magnetic_data.r12_spin.data(),                                                                 \
    plane_size,                                                                                    \
    nep_data[device_id].Fp.data(),                                                                 \
    annmb[device_id].c,                                                                            \
    spin_coefficient_offset_,                                                                      \
    projection,                                                                                    \
    data.spin3_moments.data(),                                                                     \
    data.force.data(),                                                                             \
    data.mforce.data(),                                                                            \
    data.virial.data())
  if (spin_force_jit_) {
    spin_force_jit_(
      polynomial_layout_,
      dataset.N,
      struct_dim_,
      num_types_,
      spin_basis_size_,
      spin_cutoff_,
      data.spin_cutoff_pair.data(),
      dataset.type.data(),
      data.spin_type_active.data(),
      magnetic_data.r_spin.data(),
      magnetic_data.spin.data(),
      magnetic_data.NN_spin.data(),
      magnetic_data.NL_spin.data(),
      magnetic_data.r12_spin.data(),
      plane_size,
      nep_data[device_id].Fp.data(),
      para.spin_version == 2 ? data.density_radial.data() : annmb[device_id].c,
      spin_coefficient_offset_,
      projection,
      data.spin3_moments.data(),
      data.force.data(),
      data.mforce.data(),
      data.virial.data());
  } else {
    switch (spin_compress_) {
      case 1:
        LAUNCH_SPIN3_FORCE(1);
        break;
      case 2:
        LAUNCH_SPIN3_FORCE(2);
        break;
      case 3:
        LAUNCH_SPIN3_FORCE(3);
        break;
      case 4:
        LAUNCH_SPIN3_FORCE(4);
        break;
      case 5:
        LAUNCH_SPIN3_FORCE(5);
        break;
      case 6:
        LAUNCH_SPIN3_FORCE(6);
        break;
      case 7:
        LAUNCH_SPIN3_FORCE(7);
        break;
      case 8:
        LAUNCH_SPIN3_FORCE(8);
        break;
      case 9:
        LAUNCH_SPIN3_FORCE(9);
        break;
      default:
        PRINT_INPUT_ERROR("spin_compress should be an integer from 1 to 9.");
        break;
    }
  }
#undef LAUNCH_SPIN3_FORCE

  mask_inactive_spin_mforce_training<<<grid_size, block_size>>>(
    dataset.N, dataset.N, dataset.type.data(), data.spin_type_active.data(), data.mforce.data());
  add_spin_training_outputs<<<grid_size, block_size>>>(
    dataset.N,
    data.force.data(),
    data.mforce.data(),
    data.virial.data(),
    dataset.force.data(),
    magnetic_data.mforce.data(),
    dataset.virial.data());
  GPU_CHECK_KERNEL
}

void NEP_Spin_Model::initialize_spin_compiled(const Spin_Parameters& para)
{
  spin_descriptor_jit_ = nullptr;
  spin_force_jit_ = nullptr;
  compiled_spin_.reset();
  if (!para.nep_compile || (para.prediction != 0 && para.spin_version != 2))
    return;
  const auto layout = make_spin_polynomial_layout<Spin_Polynomial_Layout>(
    para.spin_compress, para.spin_l_max[0], para.spin_order, para.spin_soc);
  auto config = make_nep_compile_config(para);
  config.model_name = "Spin layout";
  config.source_file = "main_nep/nep_spin/spin_specialized.cu";
  std::ostringstream constants;
  constants << "#define NEP_SPIN_LAYOUT_JIT 1\nstruct SpinConstantLayout {\n";
  constants << "  enum : int { channels = " << layout.channels << " };\n";
  constants << "  enum : int { pair_count = " << layout.pair_count << " };\n";
  constants << "  enum : int { density_stride = " << layout.density_stride << " };\n";
  constants << "  enum : int { moment_count = " << layout.moment_count << " };\n";
  constants << "  enum : int { local_s2 = " << layout.local_s2 << " };\n";
  constants << "  enum : int { edge_l0_dot = " << layout.edge_l0_dot << " };\n";
  constants << "  enum : int { edge_l0_neighbor_s2 = " << layout.edge_l0_neighbor_s2 << " };\n";
  constants << "  enum : int { edge_l2_pair = " << layout.edge_l2_pair << " };\n";
  constants << "  enum : int { center_l2_environment = " << layout.center_l2_environment << " };\n";
  constants << "  enum : int { edge_l0_dot2 = " << layout.edge_l0_dot2 << " };\n";
  constants << "  enum : int { density_l0_self = " << layout.density_l0_self << " };\n";
  constants << "  enum : int { density_l1_longitudinal_self = "
            << layout.density_l1_longitudinal_self << " };\n";
  constants << "  enum : int { density_l1_axial_self = " << layout.density_l1_axial_self << " };\n";
  constants << "  enum : int { density_l1_stf_self = " << layout.density_l1_stf_self << " };\n";
  constants << "  enum : int { density_l1_product_self = " << layout.density_l1_product_self
            << " };\n";
  constants << "  enum : int { density_l2_product_self = " << layout.density_l2_product_self
            << " };\n";
  constants << "  enum : int { density_l0_dot_response = " << layout.density_l0_dot_response
            << " };\n";
  constants << "  enum : int { correlation_same_edge = " << layout.correlation_same_edge << " };\n";
  constants << "  enum : int { correlation_distinct_neighbor = "
            << layout.correlation_distinct_neighbor << " };\n";
  constants << "  enum : int { correlation_distinct_l1 = " << layout.correlation_distinct_l1
            << " };\n";
  constants << "  enum : int { correlation_distinct_l2 = " << layout.correlation_distinct_l2
            << " };\n";
  constants << "  enum : int { angular_l1_moment_offset = " << layout.angular_l1_moment_offset
            << " };\n";
  constants << "  enum : int { angular_l2_moment_offset = " << layout.angular_l2_moment_offset
            << " };\n";
  constants << "  enum : int { coupling_l11_axial = " << layout.coupling_l11_axial << " };\n";
  constants << "  enum : int { edge_l11_axial = " << layout.edge_l11_axial << " };\n";
  constants << "  enum : int { coupling_l22_axial = " << layout.coupling_l22_axial << " };\n";
  constants << "  enum : int { edge_l22_axial = " << layout.edge_l22_axial << " };\n";
  constants << "  enum : int { edge_l0_moment_gate = " << layout.edge_l0_moment_gate << " };\n";
  constants << "  enum : int { coupling_l11_dot_response = " << layout.coupling_l11_dot_response
            << " };\n";
  constants << "  enum : int { coupling_l111_p_m_x = " << layout.coupling_l111_p_m_x << " };\n";
  constants << "  enum : int { coupling_l22_dot_response = " << layout.coupling_l22_dot_response
            << " };\n";
  constants << "  enum : int { coupling_l111_p_qs_x = " << layout.coupling_l111_p_qs_x << " };\n";
  constants << "  enum : int { coupling_l112_edge_response = " << layout.coupling_l112_edge_response
            << " };\n";
  constants << "  enum : int { coupling_l111_bulk = " << layout.coupling_l111_bulk << " };\n";
  constants << "  enum : int { descriptor_dim = " << layout.descriptor_dim << " };\n";
  constants << "};\n";
  config.extra_config = constants.str();
  if (para.spin_version == 2) {
    const spin_density::Layout density(para.spin_compress, para.spin_l_max[0], para.spin_order, para.spin_soc);
    // Bound generated code size; larger layouts keep the generic CUDA implementation.
    if (density.terms.size() > 512) {
      printf("Spin version2: large coupling layout uses generic CUDA kernels.\n");
      return;
    }
    config.model_name = "Spin version2";
    config.source_file = "main_nep/nep_spin/spin_density_specialized.cu";
    config.extra_config = spin_density::compile_layout(density);
  }
  compiled_spin_.reset(new NEP_Compile(config));
  if (
    !compiled_spin_->is_valid() ||
    !compiled_spin_->load_symbol("nep_spin_launch_descriptor", spin_descriptor_jit_) ||
    !compiled_spin_->load_symbol("nep_spin_launch_force", spin_force_jit_)) {
    spin_descriptor_jit_ = nullptr;
    spin_force_jit_ = nullptr;
    compiled_spin_.reset();
  } else {
    printf(
      "Spin layout JIT ready: C=%d L=%d order=%d SOC=%d.\n",
      para.spin_compress,
      para.spin_l_max[0],
      para.spin_order,
      para.spin_soc);
  }
}
