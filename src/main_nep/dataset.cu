/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "dataset.cuh"
#include "mic.cuh"
#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

void Dataset::copy_structures(std::vector<Structure>& structures_input, int n1, int n2)
{
  Nc = n2 - n1;
  structures.resize(Nc);

  for (int n = 0; n < Nc; ++n) {
    int n_input = n + n1;
    structures[n].num_atom = structures_input[n_input].num_atom;
    structures[n].weight = structures_input[n_input].weight;
    structures[n].has_virial = structures_input[n_input].has_virial;
    structures[n].has_bec = structures_input[n_input].has_bec;
    structures[n].has_mforce = structures_input[n_input].has_mforce;
    structures[n].has_spin_response_metadata =
      structures_input[n_input].has_spin_response_metadata;
    structures[n].has_spin_response = structures_input[n_input].has_spin_response;
    structures[n].spin_response_probe = structures_input[n_input].spin_response_probe;
    structures[n].spin_response_group = structures_input[n_input].spin_response_group;
    structures[n].spin_response_coordinate =
      structures_input[n_input].spin_response_coordinate;
    structures[n].has_atomic_virial = structures_input[n_input].has_atomic_virial;
    structures[n].atomic_virial_diag_only = structures_input[n_input].atomic_virial_diag_only;
    structures[n].charge = structures_input[n_input].charge;
    structures[n].energy = structures_input[n_input].energy;
    structures[n].energy_weight = structures_input[n_input].energy_weight;
    structures[n].has_temperature = structures_input[n_input].has_temperature;
    structures[n].temperature = structures_input[n_input].temperature;
    structures[n].volume = structures_input[n_input].volume;
    for (int k = 0; k < 6; ++k) {
      structures[n].virial[k] = structures_input[n_input].virial[k];
    }
    for (int k = 0; k < 18; ++k) {
      structures[n].box[k] = structures_input[n_input].box[k];
    }
    for (int k = 0; k < 9; ++k) {
      structures[n].box_original[k] = structures_input[n_input].box_original[k];
    }
    for (int k = 0; k < 3; ++k) {
      structures[n].num_cell[k] = structures_input[n_input].num_cell[k];
    }

    structures[n].type.resize(structures[n].num_atom);
    structures[n].x.resize(structures[n].num_atom);
    structures[n].y.resize(structures[n].num_atom);
    structures[n].z.resize(structures[n].num_atom);
    structures[n].fx.resize(structures[n].num_atom);
    structures[n].fy.resize(structures[n].num_atom);
    structures[n].fz.resize(structures[n].num_atom);
    if (!structures_input[n_input].sx.empty()) {
      structures[n].sx.resize(structures[n].num_atom);
      structures[n].sy.resize(structures[n].num_atom);
      structures[n].sz.resize(structures[n].num_atom);
      structures[n].mfx.resize(structures[n].num_atom);
      structures[n].mfy.resize(structures[n].num_atom);
      structures[n].mfz.resize(structures[n].num_atom);
      if (structures[n].has_spin_response) {
        structures[n].spin_tangent_x.resize(structures[n].num_atom);
        structures[n].spin_tangent_y.resize(structures[n].num_atom);
        structures[n].spin_tangent_z.resize(structures[n].num_atom);
      }
    }
    structures[n].bec.resize(structures[n].num_atom * 9);

    for (int na = 0; na < structures[n].num_atom; ++na) {
      structures[n].type[na] = structures_input[n_input].type[na];
      structures[n].x[na] = structures_input[n_input].x[na];
      structures[n].y[na] = structures_input[n_input].y[na];
      structures[n].z[na] = structures_input[n_input].z[na];
      structures[n].fx[na] = structures_input[n_input].fx[na];
      structures[n].fy[na] = structures_input[n_input].fy[na];
      structures[n].fz[na] = structures_input[n_input].fz[na];
      if (!structures[n].sx.empty()) {
        structures[n].sx[na] = structures_input[n_input].sx[na];
        structures[n].sy[na] = structures_input[n_input].sy[na];
        structures[n].sz[na] = structures_input[n_input].sz[na];
        structures[n].mfx[na] = structures_input[n_input].mfx[na];
        structures[n].mfy[na] = structures_input[n_input].mfy[na];
        structures[n].mfz[na] = structures_input[n_input].mfz[na];
        if (structures[n].has_spin_response) {
          structures[n].spin_tangent_x[na] =
            structures_input[n_input].spin_tangent_x[na];
          structures[n].spin_tangent_y[na] =
            structures_input[n_input].spin_tangent_y[na];
          structures[n].spin_tangent_z[na] =
            structures_input[n_input].spin_tangent_z[na];
        }
      }
      for (int d = 0; d < 9; ++d) {
        structures[n].bec[na * 9 + d] = structures_input[n_input].bec[na * 9 + d];
      }
    }

    if (structures[n].has_atomic_virial != structures[0].has_atomic_virial) {
      throw std::runtime_error("All structures must have the same has_atomic_virial flag.");
    }
    if (structures[n].atomic_virial_diag_only != structures[0].atomic_virial_diag_only) {
      throw std::runtime_error("All structures must have the same atomic_virial_diag_only flag.");
    }
    if (structures[n].has_atomic_virial) {
      structures[n].avirialxx.resize(structures[n].num_atom);
      structures[n].avirialyy.resize(structures[n].num_atom);
      structures[n].avirialzz.resize(structures[n].num_atom);
      for (int na = 0; na < structures[n].num_atom; ++na) {
        structures[n].avirialxx[na] = structures_input[n_input].avirialxx[na];
        structures[n].avirialyy[na] = structures_input[n_input].avirialyy[na];
        structures[n].avirialzz[na] = structures_input[n_input].avirialzz[na];
      }
      if (!structures[n].atomic_virial_diag_only) {
        structures[n].avirialxy.resize(structures[n].num_atom);
        structures[n].avirialyz.resize(structures[n].num_atom);
        structures[n].avirialzx.resize(structures[n].num_atom);
        for (int na = 0; na < structures[n].num_atom; ++na) {
          structures[n].avirialxy[na] = structures_input[n_input].avirialxy[na];
          structures[n].avirialyz[na] = structures_input[n_input].avirialyz[na];
          structures[n].avirialzx[na] = structures_input[n_input].avirialzx[na];
        }
      }
    }
  }
}

void Dataset::find_has_type(Parameters& para)
{
  has_type.resize((para.num_types + 1) * Nc, false);
  for (int n = 0; n < Nc; ++n) {
    has_type[para.num_types * Nc + n] = true;
    for (int na = 0; na < structures[n].num_atom; ++na) {
      has_type[structures[n].type[na] * Nc + n] = true;
    }
  }
}

void Dataset::find_Na(Parameters& para)
{
  Na_cpu.resize(Nc);
  Na_sum_cpu.resize(Nc);

  N = 0;
  max_Na = 0;
  int num_virial_configurations = 0;
  for (int nc = 0; nc < Nc; ++nc) {
    Na_cpu[nc] = structures[nc].num_atom;
    Na_sum_cpu[nc] = 0;
  }

  for (int nc = 0; nc < Nc; ++nc) {
    N += structures[nc].num_atom;
    if (structures[nc].num_atom > max_Na) {
      max_Na = structures[nc].num_atom;
    }
    num_virial_configurations += structures[nc].has_virial;
  }

  for (int nc = 1; nc < Nc; ++nc) {
    Na_sum_cpu[nc] = Na_sum_cpu[nc - 1] + Na_cpu[nc - 1];
  }

  printf("Total number of atoms = %d.\n", N);
  printf("Number of atoms in the largest configuration = %d.\n", max_Na);
  if (para.train_mode == 0 || para.train_mode == 3) {
    printf("Number of configurations having virial = %d.\n", num_virial_configurations);
  }

  Na.resize(Nc);
  Na_sum.resize(Nc);
  Na.copy_from_host(Na_cpu.data());
  Na_sum.copy_from_host(Na_sum_cpu.data());
}

void Dataset::initialize_gpu_data(Parameters& para)
{
  std::vector<float> box_cpu(Nc * 18);
  std::vector<float> box_original_cpu(Nc * 9);
  std::vector<int> num_cell_cpu(Nc * 3);
  std::vector<float> r_cpu(N * 3);
  std::vector<double> r_spin_cpu;
  std::vector<double> spin_cpu;
  std::vector<int> type_cpu(N);
  if (para.spin_mode) {
    r_spin_cpu.resize(N * 3);
    spin_cpu.resize(N * 3);
  }

  if ((para.charge_mode || para.charge_vdw)) {
    charge.resize(N);
    charge_shifted.resize(N);
    charge_cpu.resize(N);
    charge_ref_cpu.resize(Nc);
    charge_ref_gpu.resize(Nc);
    bec.resize(N * 9);
    bec_cpu.resize(N * 9);
    bec_ref_cpu.resize(N * 9);
    bec_ref_gpu.resize(N * 9);
  }

  energy.resize(N);
  virial.resize(N * 6);
  force.resize(N * 3);
  if (para.spin_mode) {
    mforce.resize(N * 3);
    mforce_cpu.resize(N * 3);
    mforce_ref_cpu.resize(N * 3);
    mforce_ref_gpu.resize(N * 3);
    has_mforce_cpu.resize(Nc);
    has_mforce_gpu.resize(Nc);
  }
  energy_cpu.resize(N);
  virial_cpu.resize(N * 6);
  force_cpu.resize(N * 3);
  weight_cpu.resize(Nc);
  energy_ref_cpu.resize(Nc);
  energy_weight_cpu.resize(Nc);
  virial_ref_cpu.resize(Nc * 6);
  force_ref_cpu.resize(N * 3);
  if (structures[0].has_atomic_virial) {
    avirial_ref_cpu.resize(N * (structures[0].atomic_virial_diag_only ? 3 : 6));
  }
  temperature_ref_cpu.resize(N);

  for (int n = 0; n < Nc; ++n) {
    weight_cpu[n] = structures[n].weight;
    if ((para.charge_mode || para.charge_vdw)) {
      charge_ref_cpu[n] = structures[n].charge;
    }
    energy_ref_cpu[n] = structures[n].energy;
    energy_weight_cpu[n] = structures[n].energy_weight;
    if (para.spin_mode) {
      has_mforce_cpu[n] = structures[n].has_mforce;
    }
    for (int k = 0; k < 6; ++k) {
      virial_ref_cpu[k * Nc + n] = structures[n].virial[k];
    }
    for (int k = 0; k < 18; ++k) {
      box_cpu[k + n * 18] = structures[n].box[k];
    }
    for (int k = 0; k < 9; ++k) {
      box_original_cpu[k + n * 9] = structures[n].box_original[k];
    }
    for (int k = 0; k < 3; ++k) {
      num_cell_cpu[k + n * 3] = structures[n].num_cell[k];
    }
    for (int na = 0; na < structures[n].num_atom; ++na) {
      type_cpu[Na_sum_cpu[n] + na] = structures[n].type[na];
      r_cpu[Na_sum_cpu[n] + na] = structures[n].x[na];
      r_cpu[Na_sum_cpu[n] + na + N] = structures[n].y[na];
      r_cpu[Na_sum_cpu[n] + na + N * 2] = structures[n].z[na];
      if (para.spin_mode) {
        r_spin_cpu[Na_sum_cpu[n] + na] = structures[n].x[na];
        r_spin_cpu[Na_sum_cpu[n] + na + N] = structures[n].y[na];
        r_spin_cpu[Na_sum_cpu[n] + na + N * 2] = structures[n].z[na];
        spin_cpu[Na_sum_cpu[n] + na] = structures[n].sx[na];
        spin_cpu[Na_sum_cpu[n] + na + N] = structures[n].sy[na];
        spin_cpu[Na_sum_cpu[n] + na + N * 2] = structures[n].sz[na];
        mforce_ref_cpu[Na_sum_cpu[n] + na] = structures[n].mfx[na];
        mforce_ref_cpu[Na_sum_cpu[n] + na + N] = structures[n].mfy[na];
        mforce_ref_cpu[Na_sum_cpu[n] + na + N * 2] = structures[n].mfz[na];
      }
      force_ref_cpu[Na_sum_cpu[n] + na] = structures[n].fx[na];
      force_ref_cpu[Na_sum_cpu[n] + na + N] = structures[n].fy[na];
      force_ref_cpu[Na_sum_cpu[n] + na + N * 2] = structures[n].fz[na];
      temperature_ref_cpu[Na_sum_cpu[n] + na] = structures[n].temperature;
      if (structures[n].has_atomic_virial) {
        avirial_ref_cpu[Na_sum_cpu[n] + na] = structures[n].avirialxx[na];
        avirial_ref_cpu[Na_sum_cpu[n] + na + N] = structures[n].avirialyy[na];
        avirial_ref_cpu[Na_sum_cpu[n] + na + N * 2] = structures[n].avirialzz[na];
        if (!structures[n].atomic_virial_diag_only) {
          avirial_ref_cpu[Na_sum_cpu[n] + na + N * 3] = structures[n].avirialxy[na];
          avirial_ref_cpu[Na_sum_cpu[n] + na + N * 4] = structures[n].avirialyz[na];
          avirial_ref_cpu[Na_sum_cpu[n] + na + N * 5] = structures[n].avirialzx[na];
        }
      }
      if ((para.charge_mode || para.charge_vdw)) {
        for (int d = 0; d < 9; ++d) {
          bec_ref_cpu[Na_sum_cpu[n] + na + N * d] = structures[n].bec[na * 9 + d];
        }
      }
    }
  }

  type_weight_gpu.resize(NUM_ELEMENTS);
  if (para.spin_mode) {
    spin_dof_type_active_gpu.resize(para.num_types);
  }
  
  energy_ref_gpu.resize(Nc);
  energy_weight_gpu.resize(Nc);
  virial_ref_gpu.resize(Nc * 6);
  force_ref_gpu.resize(N * 3);
  if (structures[0].has_atomic_virial) {
    avirial_ref_gpu.resize(N * (structures[0].atomic_virial_diag_only ? 3 : 6));
  }
  temperature_ref_gpu.resize(N);
  type_weight_gpu.copy_from_host(para.type_weight_cpu.data());
  if ((para.charge_mode || para.charge_vdw)) {
    charge_ref_gpu.copy_from_host(charge_ref_cpu.data());
    bec_ref_gpu.copy_from_host(bec_ref_cpu.data());
  }
  energy_ref_gpu.copy_from_host(energy_ref_cpu.data());
  energy_weight_gpu.copy_from_host(energy_weight_cpu.data());
  virial_ref_gpu.copy_from_host(virial_ref_cpu.data());
  force_ref_gpu.copy_from_host(force_ref_cpu.data());
  if (para.spin_mode) {
    mforce_ref_gpu.copy_from_host(mforce_ref_cpu.data());
    has_mforce_gpu.copy_from_host(has_mforce_cpu.data());
    spin_dof_type_active_gpu.copy_from_host(para.spin_dof_type_active.data());
  }
  if (structures[0].has_atomic_virial) {
    avirial_ref_gpu.copy_from_host(avirial_ref_cpu.data());
  }
  temperature_ref_gpu.copy_from_host(temperature_ref_cpu.data());

  box.resize(Nc * 18);
  box_original.resize(Nc * 9);
  num_cell.resize(Nc * 3);
  r.resize(N * 3);
  if (para.spin_mode) {
    r_spin.resize(N * 3);
    spin.resize(N * 3);
  }
  type.resize(N);
  box.copy_from_host(box_cpu.data());
  box_original.copy_from_host(box_original_cpu.data());
  num_cell.copy_from_host(num_cell_cpu.data());
  r.copy_from_host(r_cpu.data());
  if (para.spin_mode) {
    r_spin.copy_from_host(r_spin_cpu.data());
    spin.copy_from_host(spin_cpu.data());
  }
  type.copy_from_host(type_cpu.data());
}

template <bool Spin>
static __global__ void gpu_find_neighbor_number(
  const int N,
  const int* Na,
  const int* Na_sum,
  const int* g_type,
  const int* g_atomic_numbers,
  const float* g_rc_radial,
  const float* g_rc_angular,
  const float rc_spin,
  const float* __restrict__ g_box,
  const float* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const float* x,
  const float* y,
  const float* z,
  int* NN_radial,
  int* NN_angular,
  int* NN_spin)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    const float* __restrict__ box = g_box + 18 * blockIdx.x;
    const float* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
    const int* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
    float x1 = x[n1];
    float y1 = y[n1];
    float z1 = z[n1];
    int t1 = g_type[n1];
    int count_radial = 0;
    int count_angular = 0;
    int count_spin = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      for (int ia = 0; ia < num_cell[0]; ++ia) {
        for (int ib = 0; ib < num_cell[1]; ++ib) {
          for (int ic = 0; ic < num_cell[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }
            float delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
            float delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
            float delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;
            float x12 = x[n2] + delta_x - x1;
            float y12 = y[n2] + delta_y - y1;
            float z12 = z[n2] + delta_z - z1;
            dev_apply_mic(box, x12, y12, z12);
            float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            int t2 = g_type[n2];
            float rc_radial = (g_rc_radial[t1] + g_rc_radial[t2]) * 0.5f;
            float rc_angular = (g_rc_angular[t1] + g_rc_angular[t2]) * 0.5f;
            if (distance_square < rc_radial * rc_radial) {
              count_radial++;
            }
            if (distance_square < rc_angular * rc_angular) {
              count_angular++;
            }
            if constexpr (Spin) {
              if (distance_square < rc_spin * rc_spin) {
                count_spin++;
              }
            }
          }
        }
      }
    }
    NN_radial[n1] = count_radial;
    NN_angular[n1] = count_angular;
    if constexpr (Spin) {
      NN_spin[n1] = count_spin;
    }
  }
}

template <bool Spin>
static __global__ void gpu_find_neighbor_list(
  const int N,
  const int* Na,
  const int* Na_sum,
  const int* g_type,
  const int* g_atomic_numbers,
  const float* g_rc_radial,
  const float* g_rc_angular,
  const float rc_spin,
  const float* __restrict__ g_box,
  const float* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const float* x,
  const float* y,
  const float* z,
  const int* NN_radial_sum,
  const int* NN_angular_sum,
  int* NN_radial,
  int* NL_radial,
  int* NN_angular,
  int* NL_angular,
  float* x12_radial,
  float* y12_radial,
  float* z12_radial,
  float* x12_angular,
  float* y12_angular,
  float* z12_angular,
  int spin_plane_size,
  int* NN_spin,
  int* NL_spin,
  double* r12_spin)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    const float* __restrict__ box = g_box + 18 * blockIdx.x;
    const float* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
    const int* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
    float x1 = x[n1];
    float y1 = y[n1];
    float z1 = z[n1];
    int t1 = g_type[n1];
    int count_radial = 0;
    int count_angular = 0;
    int count_spin = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      for (int ia = 0; ia < num_cell[0]; ++ia) {
        for (int ib = 0; ib < num_cell[1]; ++ib) {
          for (int ic = 0; ic < num_cell[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }
            float delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
            float delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
            float delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;
            float x12 = x[n2] + delta_x - x1;
            float y12 = y[n2] + delta_y - y1;
            float z12 = z[n2] + delta_z - z1;
            dev_apply_mic(box, x12, y12, z12);
            float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            int t2 = g_type[n2];
            float rc_radial = (g_rc_radial[t1] + g_rc_radial[t2]) * 0.5f;
            float rc_angular = (g_rc_angular[t1] + g_rc_angular[t2]) * 0.5f;
            if (distance_square < rc_radial * rc_radial) {
              int index = NN_radial_sum[n1] + count_radial;
              NL_radial[index] = n2;
              x12_radial[index] = x12;
              y12_radial[index] = y12;
              z12_radial[index] = z12;
              count_radial++;
            }
            if (distance_square < rc_angular * rc_angular) {
              int index = NN_angular_sum[n1] + count_angular;
              NL_angular[index] = n2;
              x12_angular[index] = x12;
              y12_angular[index] = y12;
              z12_angular[index] = z12;
              count_angular++;
            }
            if constexpr (Spin) {
              if (distance_square < rc_spin * rc_spin) {
                int index = n1 + N * count_spin;
                NL_spin[index] = n2;
                r12_spin[index] = x12;
                r12_spin[spin_plane_size + index] = y12;
                r12_spin[2 * spin_plane_size + index] = z12;
                count_spin++;
              }
            }
          }
        }
      }
    }
    NN_radial[n1] = count_radial;
    NN_angular[n1] = count_angular;
    if constexpr (Spin) {
      NN_spin[n1] = count_spin;
    }
  }
}

void Dataset::find_neighbor(Parameters& para)
{
  NN_radial.resize(N);
  NN_angular.resize(N);
  std::vector<int> NN_radial_cpu(N);
  std::vector<int> NN_angular_cpu(N);
  std::vector<int> NN_spin_cpu;
  if (para.spin_mode) {
    NN_spin.resize(N);
    NN_spin_cpu.resize(N);
  }

  std::vector<int> atomic_numbers_from_zero(para.atomic_numbers.size());
  for (int n = 0; n < para.atomic_numbers.size(); ++n) {
    atomic_numbers_from_zero[n] = para.atomic_numbers[n] - 1;
  }
  GPU_Vector<int> atomic_numbers(para.atomic_numbers.size());
  atomic_numbers.copy_from_host(atomic_numbers_from_zero.data());

  GPU_Vector<float> rc_radial(para.rc_radial.size());
  rc_radial.copy_from_host(para.rc_radial.data());
  GPU_Vector<float> rc_angular(para.rc_angular.size());
  rc_angular.copy_from_host(para.rc_angular.data());

  if (para.spin_mode) {
    gpu_find_neighbor_number<true><<<Nc, 256>>>(
      N,
      Na.data(),
      Na_sum.data(),
      type.data(),
      atomic_numbers.data(),
      rc_radial.data(),
      rc_angular.data(),
      para.spin_cutoff,
      box.data(),
      box_original.data(),
      num_cell.data(),
      r.data(),
      r.data() + N,
      r.data() + N * 2,
      NN_radial.data(),
      NN_angular.data(),
      NN_spin.data());
  } else {
    gpu_find_neighbor_number<false><<<Nc, 256>>>(
      N,
      Na.data(),
      Na_sum.data(),
      type.data(),
      atomic_numbers.data(),
      rc_radial.data(),
      rc_angular.data(),
      0.0f,
      box.data(),
      box_original.data(),
      num_cell.data(),
      r.data(),
      r.data() + N,
      r.data() + N * 2,
      NN_radial.data(),
      NN_angular.data(),
      nullptr);
  }
  GPU_CHECK_KERNEL

  NN_radial.copy_to_host(NN_radial_cpu.data());
  NN_angular.copy_to_host(NN_angular_cpu.data());
  if (para.spin_mode) {
    NN_spin.copy_to_host(NN_spin_cpu.data());
  }

  int sum_NN_radial = 0;
  int min_NN_radial = 10000;
  max_NN_radial = -1;
  for (int n = 0; n < N; ++n) {
    sum_NN_radial += NN_radial_cpu[n];
    if (NN_radial_cpu[n] < min_NN_radial) {
      min_NN_radial = NN_radial_cpu[n];
    }
    if (NN_radial_cpu[n] > max_NN_radial) {
      max_NN_radial = NN_radial_cpu[n];
    }
  }

  int sum_NN_angular = 0;
  int min_NN_angular = 10000;
  max_NN_angular = -1;
  for (int n = 0; n < N; ++n) {
    sum_NN_angular += NN_angular_cpu[n];
    if (NN_angular_cpu[n] < min_NN_angular) {
      min_NN_angular = NN_angular_cpu[n];
    }
    if (NN_angular_cpu[n] > max_NN_angular) {
      max_NN_angular = NN_angular_cpu[n];
    }
  }

  max_NN_spin = 0;
  int min_NN_spin = 0;
  long long sum_NN_spin = 0;
  if (para.spin_mode) {
    min_NN_spin = NN_spin_cpu.empty() ? 0 : NN_spin_cpu[0];
    for (int count : NN_spin_cpu) {
      min_NN_spin = std::min(min_NN_spin, count);
      max_NN_spin = std::max(max_NN_spin, count);
      sum_NN_spin += count;
    }
  }

  std::vector<int> NN_radial_sum_cpu(N, 0);
  std::vector<int> NN_angular_sum_cpu(N, 0);
  NN_radial_sum.resize(N);
  NN_angular_sum.resize(N);
  for (int n = 1; n < N; ++n) {
    NN_radial_sum_cpu[n] = NN_radial_sum_cpu[n - 1] + NN_radial_cpu[n - 1];
    NN_angular_sum_cpu[n] = NN_angular_sum_cpu[n - 1] + NN_angular_cpu[n - 1];
  }
  NN_radial_sum.copy_from_host(NN_radial_sum_cpu.data());
  NN_angular_sum.copy_from_host(NN_angular_sum_cpu.data());

  NL_radial.resize(sum_NN_radial);
  x12_radial.resize(sum_NN_radial);
  y12_radial.resize(sum_NN_radial);
  z12_radial.resize(sum_NN_radial);

  NL_angular.resize(sum_NN_angular);
  x12_angular.resize(sum_NN_angular);
  y12_angular.resize(sum_NN_angular);
  z12_angular.resize(sum_NN_angular);
  const long long spin_plane_size_wide =
    static_cast<long long>(N) * static_cast<long long>(max_NN_spin);
  if (spin_plane_size_wide > std::numeric_limits<int>::max()) {
    PRINT_INPUT_ERROR("Spin neighbor slot count exceeds the supported int range.\n");
  }
  const int spin_plane_size = static_cast<int>(spin_plane_size_wide);
  if (para.spin_mode) {
    NL_spin.resize(spin_plane_size);
    r12_spin.resize(3 * spin_plane_size);
  }

  if (para.spin_mode) {
    gpu_find_neighbor_list<true><<<Nc, 256>>>(
      N,
      Na.data(),
      Na_sum.data(),
      type.data(),
      atomic_numbers.data(),
      rc_radial.data(),
      rc_angular.data(),
      para.spin_cutoff,
      box.data(),
      box_original.data(),
      num_cell.data(),
      r.data(),
      r.data() + N,
      r.data() + N * 2,
      NN_radial_sum.data(),
      NN_angular_sum.data(),
      NN_radial.data(),
      NL_radial.data(),
      NN_angular.data(),
      NL_angular.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data(),
      spin_plane_size,
      NN_spin.data(),
      NL_spin.data(),
      r12_spin.data());
  } else {
    gpu_find_neighbor_list<false><<<Nc, 256>>>(
      N,
      Na.data(),
      Na_sum.data(),
      type.data(),
      atomic_numbers.data(),
      rc_radial.data(),
      rc_angular.data(),
      0.0f,
      box.data(),
      box_original.data(),
      num_cell.data(),
      r.data(),
      r.data() + N,
      r.data() + N * 2,
      NN_radial_sum.data(),
      NN_angular_sum.data(),
      NN_radial.data(),
      NL_radial.data(),
      NN_angular.data(),
      NL_angular.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data(),
      0,
      nullptr,
      nullptr,
      nullptr);
  }
  GPU_CHECK_KERNEL

  printf("Radial descriptor with a cutoff of %g A:\n", para.rc_radial_max);
  printf("    Minimum number of neighbors for one atom = %d.\n", min_NN_radial);
  printf("    Maximum number of neighbors for one atom = %d.\n", max_NN_radial);
  printf("    Average number of neighbors for one atom = %g.\n", sum_NN_radial / float(N));
  printf("Angular descriptor with a cutoff of %g A:\n", para.rc_angular_max);
  printf("    Minimum number of neighbors for one atom = %d.\n", min_NN_angular);
  printf("    Maximum number of neighbors for one atom = %d.\n", max_NN_angular);
  printf("    Average number of neighbors for one atom = %g.\n", sum_NN_angular / float(N));
  if (para.spin_mode) {
    printf("Spin descriptor with a maximum cutoff of %g A:\n", para.spin_cutoff);
    printf("    Minimum number of neighbors for one atom = %d.\n", min_NN_spin);
    printf("    Maximum number of neighbors for one atom = %d.\n", max_NN_spin);
    printf(
      "    Average number of neighbors for one atom = %g.\n",
      sum_NN_spin / float(N));
  }
#ifdef OUTPUT_NEIGHBOR_FOR_TRAIN
  FILE* fid = fopen("neighbor.txt", "a");
  for (int nc = 0; nc < Nc; ++nc) {
    int max_NN_radial = -1;
    int max_NN_angular = -1;
    for (int n = Na_sum_cpu[nc]; n < Na_sum_cpu[nc] + Na_cpu[nc]; ++n) {
      if (max_NN_radial < NN_radial_cpu[n]) {
        max_NN_radial = NN_radial_cpu[n];
      }
      if (max_NN_angular < NN_angular_cpu[n]) {
        max_NN_angular = NN_angular_cpu[n];
      }
    }
    fprintf(fid, "%d %d\n", max_NN_radial, max_NN_angular);
  }
  fclose(fid);
#endif
}

void Dataset::construct(
  Parameters& para, std::vector<Structure>& structures_input, int n1, int n2, int device_id)
{
  CHECK(gpuSetDevice(device_id));
  copy_structures(structures_input, n1, n2);
  find_has_type(para);
  error_cpu.resize(Nc);
  error_gpu.resize(Nc);

  find_Na(para);
  initialize_gpu_data(para);
  find_neighbor(para);
}

static __global__ void gpu_sum_force_error(
  bool use_weight,
  float force_delta,
  int* g_Na,
  int* g_Na_sum,
  int* g_type,
  float* g_type_weight,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_fx_ref,
  float* g_fy_ref,
  float* g_fz_ref,
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int N2 = N1 + g_Na[bid];
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    float fx_ref = g_fx_ref[n];
    float fy_ref = g_fy_ref[n];
    float fz_ref = g_fz_ref[n];
    float dx = g_fx[n] - fx_ref;
    float dy = g_fy[n] - fy_ref;
    float dz = g_fz[n] - fz_ref;
    float diff_square = dx * dx + dy * dy + dz * dz;
    if (use_weight) {
      float type_weight = g_type_weight[g_type[n]];
      diff_square *= type_weight * type_weight;
    }
    if (use_weight && force_delta > 0.0f) {
      float force_magnitude = sqrt(fx_ref * fx_ref + fy_ref * fy_ref + fz_ref * fz_ref);
      diff_square *= force_delta / (force_delta + force_magnitude);
    }
    s_error[tid] += diff_square;
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    error_gpu[bid] = s_error[0];
  }
}

std::vector<float> Dataset::get_rmse_force(Parameters& para, const bool use_weight, int device_id)
{
  CHECK(gpuSetDevice(device_id));
  const int block_size = 256;
  gpu_sum_force_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    use_weight,
    para.force_delta,
    Na.data(),
    Na_sum.data(),
    type.data(),
    type_weight_gpu.data(),
    force.data(),
    force.data() + N,
    force.data() + N * 2,
    force_ref_gpu.data(),
    force_ref_gpu.data() + N,
    force_ref_gpu.data() + N * 2,
    error_gpu.data());
  int mem = sizeof(float) * Nc;
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));

  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  std::vector<int> count_array(para.num_types + 1, 0);
  for (int n = 0; n < Nc; ++n) {
    float rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (has_type[t * Nc + n]) {
        rmse_array[t] += rmse_temp;
        count_array[t] += Na_cpu[n];
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / (count_array[t] * 3));
    }
  }
  return rmse_array;
}

template <bool Torque>
static __global__ void gpu_sum_mforce_error(
  const int N,
  const int* Na,
  const int* Na_sum,
  const int* type,
  const int* active_type,
  const int* has_mforce,
  const double* spin,
  const float* mforce,
  const float* mforce_ref,
  const int mforce_mode,
  float* error)
{
  const int configuration = blockIdx.x;
  const int begin = Na_sum[configuration];
  const int end = begin + Na[configuration];
  extern __shared__ float partial[];
  float sum = 0.0f;
  if (has_mforce[configuration]) {
    for (int atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
      if (!active_type[type[atom]]) {
        continue;
      }
      float predicted[3] = {
        mforce[atom],
        mforce[N + atom],
        mforce[2 * N + atom]};
      float reference[3] = {
        mforce_ref[atom],
        mforce_ref[N + atom],
        mforce_ref[2 * N + atom]};
      const double sx = spin[atom];
      const double sy = spin[N + atom];
      const double sz = spin[2 * N + atom];
      const double spin2 = sx * sx + sy * sy + sz * sz;
      if ((Torque || mforce_mode == 1) && spin2 <= 1.0e-20) {
        continue;
      }
      if constexpr (Torque) {
        const float predicted_tau[3] = {
          static_cast<float>(sy * predicted[2] - sz * predicted[1]),
          static_cast<float>(sz * predicted[0] - sx * predicted[2]),
          static_cast<float>(sx * predicted[1] - sy * predicted[0])};
        const float reference_tau[3] = {
          static_cast<float>(sy * reference[2] - sz * reference[1]),
          static_cast<float>(sz * reference[0] - sx * reference[2]),
          static_cast<float>(sx * reference[1] - sy * reference[0])};
        for (int component = 0; component < 3; ++component) {
          const float difference =
            predicted_tau[component] - reference_tau[component];
          sum += difference * difference;
        }
      } else {
        if (mforce_mode == 1) {
          const double inverse_spin2 = 1.0 / spin2;
          const double predicted_parallel =
            (sx * predicted[0] + sy * predicted[1] + sz * predicted[2]) * inverse_spin2;
          const double reference_parallel =
            (sx * reference[0] + sy * reference[1] + sz * reference[2]) * inverse_spin2;
          predicted[0] -= static_cast<float>(predicted_parallel * sx);
          predicted[1] -= static_cast<float>(predicted_parallel * sy);
          predicted[2] -= static_cast<float>(predicted_parallel * sz);
          reference[0] -= static_cast<float>(reference_parallel * sx);
          reference[1] -= static_cast<float>(reference_parallel * sy);
          reference[2] -= static_cast<float>(reference_parallel * sz);
        }
        for (int component = 0; component < 3; ++component) {
          const float difference = predicted[component] - reference[component];
          sum += difference * difference;
        }
      }
    }
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    error[configuration] = partial[0];
  }
}

template <bool Torque>
static std::vector<float> get_rmse_spin_impl(
  Dataset& dataset,
  Parameters& para,
  const bool use_weight,
  int device_id)
{
  CHECK(gpuSetDevice(device_id));
  const int block_size = 256;
  gpu_sum_mforce_error<Torque>
    <<<dataset.Nc, block_size, sizeof(float) * block_size>>>(
      dataset.N,
      dataset.Na.data(),
      dataset.Na_sum.data(),
      dataset.type.data(),
      dataset.spin_dof_type_active_gpu.data(),
      dataset.has_mforce_gpu.data(),
      dataset.spin.data(),
      dataset.mforce.data(),
      dataset.mforce_ref_gpu.data(),
      para.spin_mforce_mode,
      dataset.error_gpu.data());
  GPU_CHECK_KERNEL
  dataset.error_gpu.copy_to_host(dataset.error_cpu.data());

  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  std::vector<int> count_array(para.num_types + 1, 0);
  for (int configuration = 0; configuration < dataset.Nc; ++configuration) {
    if (!dataset.has_mforce_cpu[configuration]) {
      continue;
    }
    int active_count = 0;
    const auto& structure = dataset.structures[configuration];
    for (int atom = 0; atom < structure.num_atom; ++atom) {
      if (!para.spin_dof_type_active[structure.type[atom]]) {
        continue;
      }
      const double spin2 =
        structure.sx[atom] * structure.sx[atom] +
        structure.sy[atom] * structure.sy[atom] +
        structure.sz[atom] * structure.sz[atom];
      if ((!Torque && para.spin_mforce_mode == 0) || spin2 > 1.0e-20) {
        ++active_count;
      }
    }
    const float weighted_error =
      use_weight
        ? dataset.weight_cpu[configuration] * dataset.weight_cpu[configuration] *
            dataset.error_cpu[configuration]
        : dataset.error_cpu[configuration];
    for (int type = 0; type <= para.num_types; ++type) {
      if (dataset.has_type[type * dataset.Nc + configuration]) {
        rmse_array[type] += weighted_error;
        count_array[type] += active_count;
      }
    }
  }
  for (int type = 0; type <= para.num_types; ++type) {
    if (count_array[type] > 0) {
      const int degrees_of_freedom =
        Torque || para.spin_mforce_mode == 1 ? 2 : 3;
      rmse_array[type] = sqrt(
        rmse_array[type] / (degrees_of_freedom * count_array[type]));
    }
  }
  return rmse_array;
}

std::vector<float>
Dataset::get_rmse_mforce(Parameters& para, const bool use_weight, int device_id)
{
  return get_rmse_spin_impl<false>(*this, para, use_weight, device_id);
}

std::vector<float>
Dataset::get_rmse_tau(Parameters& para, const bool use_weight, int device_id)
{
  return get_rmse_spin_impl<true>(*this, para, use_weight, device_id);
}

static __global__ void gpu_sum_avirial_diag_only_error(
  const int N,
  int* g_Na,
  int* g_Na_sum,
  int* g_type,
  float* g_type_weight,
  float* g_virial,
  float* g_avxx_ref,
  float* g_avyy_ref,
  float* g_avzz_ref,
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int N2 = N1 + g_Na[bid];
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    float avxx_ref = g_avxx_ref[n];
    float avyy_ref = g_avyy_ref[n];
    float avzz_ref = g_avzz_ref[n];
    float dxx = g_virial[n] - avxx_ref;
    float dyy = g_virial[1 * N + n] - avyy_ref;
    float dzz = g_virial[2 * N + n] - avzz_ref;
    float diff_square = dxx * dxx + dyy * dyy + dzz * dzz;
    s_error[tid] += diff_square;
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    error_gpu[bid] = s_error[0];
  }
}

static __global__ void gpu_sum_avirial_error(
  const int N,
  int* g_Na,
  int* g_Na_sum,
  int* g_type,
  float* g_type_weight,
  float* g_virial,
  float* g_avxx_ref,
  float* g_avyy_ref,
  float* g_avzz_ref,
  float* g_avxy_ref,
  float* g_avyz_ref,
  float* g_avzx_ref,
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int N2 = N1 + g_Na[bid];
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    float avxx_ref = g_avxx_ref[n];
    float avyy_ref = g_avyy_ref[n];
    float avzz_ref = g_avzz_ref[n];
    float avxy_ref = g_avxy_ref[n];
    float avyz_ref = g_avyz_ref[n];
    float avzx_ref = g_avzx_ref[n];
    float dxx = g_virial[n] - avxx_ref;
    float dyy = g_virial[1 * N + n] - avyy_ref;
    float dzz = g_virial[2 * N + n] - avzz_ref;
    float dxy = g_virial[3 * N + n] - avxy_ref;
    float dyz = g_virial[4 * N + n] - avyz_ref;
    float dzx = g_virial[5 * N + n] - avzx_ref;
    float diff_square = dxx * dxx + dyy * dyy + dzz * dzz + dxy * dxy + dyz * dyz + dzx * dzx;
    s_error[tid] += diff_square;
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    error_gpu[bid] = s_error[0];
  }
}

std::vector<float> Dataset::get_rmse_avirial(Parameters& para, const bool use_weight, int device_id)
{
  CHECK(gpuSetDevice(device_id));
  const int block_size = 256;

  if (structures[0].atomic_virial_diag_only) {
    gpu_sum_avirial_diag_only_error<<<Nc, block_size, sizeof(float) * block_size>>>(
      N,
      Na.data(),
      Na_sum.data(),
      type.data(),
      type_weight_gpu.data(),
      virial.data(),
      avirial_ref_gpu.data(),
      avirial_ref_gpu.data() + N,
      avirial_ref_gpu.data() + N * 2,
      error_gpu.data());
  } else {
    gpu_sum_avirial_error<<<Nc, block_size, sizeof(float) * block_size>>>(
      N,
      Na.data(),
      Na_sum.data(),
      type.data(),
      type_weight_gpu.data(),
      virial.data(),
      avirial_ref_gpu.data(),
      avirial_ref_gpu.data() + N,
      avirial_ref_gpu.data() + N * 2,
      avirial_ref_gpu.data() + N * 3,
      avirial_ref_gpu.data() + N * 4,
      avirial_ref_gpu.data() + N * 5,
      error_gpu.data());
  }
  int mem = sizeof(float) * Nc;
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));

  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  std::vector<int> count_array(para.num_types + 1, 0);
  for (int n = 0; n < Nc; ++n) {
    float rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (has_type[t * Nc + n]) {
        rmse_array[t] += rmse_temp;
        count_array[t] += Na_cpu[n];
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / (count_array[t] * 6));
    }
  }
  return rmse_array;
}

static __global__ void
gpu_get_energy_shift(
  int* g_Na, 
  int* g_Na_sum, 
  float* g_pe, 
  float* g_pe_ref, 
  float* g_pe_weight, 
  float* g_energy_shift)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ float s_pe[];
  s_pe[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    s_pe[tid] += g_pe[n];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_pe[tid] += s_pe[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float diff = s_pe[0] / Na - g_pe_ref[bid];
    g_energy_shift[bid] = diff * g_pe_weight[bid];
  }
}

static __global__ void gpu_sum_pe_error(
  float energy_shift, 
  int* g_Na, 
  int* g_Na_sum, 
  float* g_pe, 
  float* g_pe_ref, 
  float* g_pe_weight, 
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ float s_pe[];
  s_pe[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    s_pe[tid] += g_pe[n];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_pe[tid] += s_pe[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float diff = s_pe[0] / Na - g_pe_ref[bid] - energy_shift;
    error_gpu[bid] = diff * diff * g_pe_weight[bid];
  }
}

std::vector<float> Dataset::get_rmse_energy(
  Parameters& para,
  float& energy_shift_per_structure,
  const bool use_weight,
  const bool do_shift,
  int device_id)
{
  CHECK(gpuSetDevice(device_id));
  energy_shift_per_structure = 0.0f;

  const int block_size = 256;
  int mem = sizeof(float) * Nc;

  if (do_shift) {
    gpu_get_energy_shift<<<Nc, block_size, sizeof(float) * block_size>>>(
      Na.data(), 
      Na_sum.data(), 
      energy.data(), 
      energy_ref_gpu.data(), 
      energy_weight_gpu.data(), 
      error_gpu.data());
    CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));
    float Nc_with_weight = 0.0f;
    for (int n = 0; n < Nc; ++n) {
      Nc_with_weight += energy_weight_cpu[n];
      energy_shift_per_structure += error_cpu[n];
    }
    if (Nc_with_weight > 0.0f) {
      energy_shift_per_structure /= Nc_with_weight;
    }
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    energy_shift_per_structure,
    Na.data(),
    Na_sum.data(),
    energy.data(),
    energy_ref_gpu.data(),
    energy_weight_gpu.data(), 
    error_gpu.data());
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));

  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  std::vector<int> count_array(para.num_types + 1, 0);
  for (int n = 0; n < Nc; ++n) {
    float rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (has_type[t * Nc + n]) {
        rmse_array[t] += rmse_temp;
        ++count_array[t];
      }
    }
  }
  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / count_array[t]);
    }
  }
  return rmse_array;
}

static __global__ void gpu_sum_virial_error(
  const int N,
  const float shear_weight,
  int* g_Na,
  int* g_Na_sum,
  float* g_virial,
  float* g_virial_ref,
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ float s_virial[];
  for (int d = 0; d < 6; ++d) {
    s_virial[d * blockDim.x + tid] = 0.0f;
  }

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    for (int d = 0; d < 6; ++d) {
      s_virial[d * blockDim.x + tid] += g_virial[d * N + n];
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (int d = 0; d < 6; ++d) {
        s_virial[d * blockDim.x + tid] += s_virial[d * blockDim.x + tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    float error_sum = 0.0f;
    for (int d = 0; d < 6; ++d) {
      float diff = s_virial[d * blockDim.x + 0] / Na - g_virial_ref[d * gridDim.x + bid];
      error_sum += (d >= 3) ? (shear_weight * diff * diff) : (diff * diff);
    }
    error_gpu[bid] = error_sum;
  }
}

std::vector<float> Dataset::get_rmse_virial(Parameters& para, const bool use_weight, int device_id)
{
  if (para.atomic_v) {
    return get_rmse_avirial(para, use_weight, device_id);
  }
  CHECK(gpuSetDevice(device_id));

  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  std::vector<int> count_array(para.num_types + 1, 0);

  int mem = sizeof(float) * Nc;
  const int block_size = 256;

  float shear_weight =
    (para.train_mode != 1) ? (use_weight ? para.lambda_shear * para.lambda_shear : 1.0f) : 0.0f;
  gpu_sum_virial_error<<<Nc, block_size, sizeof(float) * block_size * 6>>>(
    N,
    shear_weight,
    Na.data(),
    Na_sum.data(),
    virial.data(),
    virial_ref_gpu.data(),
    error_gpu.data());
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    if (structures[n].has_virial) {
      float rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
      for (int t = 0; t < para.num_types + 1; ++t) {
        if (has_type[t * Nc + n]) {
          rmse_array[t] += rmse_temp;
          count_array[t] += (para.train_mode != 1) ? 6 : 3;
        }
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / count_array[t]);
    }
  }
  return rmse_array;
}

static __global__ void gpu_sum_charge_error(
  int* g_Na, 
  int* g_Na_sum, 
  float* g_charge, 
  float* g_charge_ref,  
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ float s_charge[];
  s_charge[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    s_charge[tid] += g_charge[n];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_charge[tid] += s_charge[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float diff = (s_charge[0] - g_charge_ref[bid]) / Na;
    error_gpu[bid] = diff * diff;
  }
}

static __global__ void gpu_sum_bec_error(
  const int N,
  int* g_Na,
  int* g_Na_sum,
  float* g_bec,
  float* g_bec_ref,
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int N2 = N1 + g_Na[bid];
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    float diff_square = 0.0f;
    for (int d = 0; d < 9; ++d) {
      const float diff = g_bec[n + N * d] - g_bec_ref[n + N * d];
      diff_square += diff * diff;
    }
    s_error[tid] += diff_square;
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    error_gpu[bid] = s_error[0];
  }
}

std::vector<float> Dataset::get_rmse_charge(Parameters& para, int device_id)
{
  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  if (!(para.charge_mode || para.charge_vdw)) {
    return rmse_array;
  }

  CHECK(gpuSetDevice(device_id));

  std::vector<int> count_array(para.num_types + 1, 0);

  int mem = sizeof(float) * Nc;
  const int block_size = 256;

  gpu_sum_charge_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(),
    Na_sum.data(),
    charge.data(),
    charge_ref_gpu.data(),
    error_gpu.data());
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    float rmse_temp = error_cpu[n];
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (has_type[t * Nc + n]) {
        rmse_array[t] += rmse_temp;
        count_array[t] += 1;
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / count_array[t]);
    }
  }
  return rmse_array;
}

std::vector<float> Dataset::get_rmse_bec(Parameters& para, int device_id)
{
  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  if (!((para.charge_mode || para.charge_vdw) && para.has_bec)) {
    return rmse_array;
  }

  CHECK(gpuSetDevice(device_id));

  std::vector<int> count_array(para.num_types + 1, 0);

  int mem = sizeof(float) * Nc;
  const int block_size = 256;

  gpu_sum_bec_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    N,
    Na.data(),
    Na_sum.data(),
    bec.data(),
    bec_ref_gpu.data(),
    error_gpu.data());
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    if (structures[n].has_bec) {
      float rmse_temp = error_cpu[n];
      for (int t = 0; t < para.num_types + 1; ++t) {
        if (has_type[t * Nc + n]) {
          rmse_array[t] += rmse_temp / (Na_cpu[n]);
          count_array[t] += 9;
        }
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / count_array[t]);
    }
  }
  return rmse_array;
}
