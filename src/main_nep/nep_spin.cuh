/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under GPLv3 or later.
*/

#pragma once

#include "nep.cuh"

class NEP_Spin_Trainer : public NEP
{
public:
  NEP_Spin_Trainer(Parameters& para, int N, int version, int device_count);

protected:
  void find_additional_descriptors(
    Parameters& para, Dataset& dataset, int device_id) override;
  void initialize_additional_outputs(Dataset& dataset, int device_id) override;
  void find_additional_force(
    Parameters& para, Dataset& dataset, int device_id) override;

private:
  struct Spin_Layout {
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

  struct Spin_Data {
    GPU_Vector<int> spin_dof_type_active;
    GPU_Vector<int> spin_env_type_active;
    GPU_Vector<float> spin_baseline;
    GPU_Vector<float> rho0;
    GPU_Vector<float> raw1;
    GPU_Vector<float> angular2;
    GPU_Vector<float> angular3;
    GPU_Vector<float> angular4;
    GPU_Vector<float> geom;
    GPU_Vector<float> rho0_dot;
    GPU_Vector<float> raw1_dot;
    GPU_Vector<float> polar;
    GPU_Vector<float> octupole;
    GPU_Vector<float> hexadecapole;
    GPU_Vector<float> chirals;
    GPU_Vector<float> spin2_moments;
    GPU_Vector<float> spin2_pulls;
    GPU_Vector<double> force;
    GPU_Vector<double> mforce;
    GPU_Vector<double> virial;
  };

public:
  struct Spin_Polynomial_Layout {
    int channels = 0;
    int pair_count = 0;
    int density_stride = 38;
    int moment_count = 0;
    int local_s2 = -1;
    int edge_l0_dot = -1;
    int edge_l0_neighbor_s2 = -1;
    int edge_l2_pair = -1;
    int center_l2_environment = -1;
    int edge_l0_dot2 = -1;
    int density_l0_self = -1;
    int density_l1_longitudinal_self = -1;
    int density_l1_axial_self = -1;
    int density_l1_stf_self = -1;
    int density_l1_product_self = -1;
    int density_l2_product_self = -1;
    int density_l0_dot_response = -1;
    int correlation_same_edge = -1;
    int correlation_distinct_neighbor = -1;
    int coupling_l11_axial = -1;
    int edge_l11_axial = -1;
    int coupling_l22_axial = -1;
    int edge_l22_axial = -1;
    int edge_l0_moment_gate = -1;
    int coupling_l11_dot_response = -1;
    int coupling_l111_p_m_x = -1;
    int coupling_l22_dot_response = -1;
    int coupling_l111_p_qs_x = -1;
    int coupling_l112_edge_response = -1;
    int coupling_l111_bulk = -1;
    int descriptor_dim = 0;
  };

private:
  int spin_compress_ = 0;
  int spin_basis_size_ = 0;
  int spin_l_max_ = 0;
  int spin_chiral_ = 0;
  int spin_mode_ = 0;
  int spin_order_ = 0;
  int spin_soc_ = 0;
  int spin_projection_offset_ = 0;
  int struct_dim_ = 0;
  int spin_coefficient_offset_ = 0;
  int num_types_ = 0;
  float spin_cutoff_ = 0.0f;
  Spin_Layout layout_;
  Spin_Polynomial_Layout polynomial_layout_;
  Spin_Data spin_data_[16];
};
