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
#include "parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>

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
