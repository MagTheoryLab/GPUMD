/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    Copyright 2026 NEPAdapters contributors
    This file is part of GPUMD and is distributed under GPLv3 or later.

    The spin model protocol and mathematical layout are adapted from
    NEPAdapters (GPLv3+), commit b4735bba1d02045ad31b7bae510bfdb393536f37.
*/

#include "nep_spin.cuh"
#include "nep_kernels.cuh"
#include "nep_small_box.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

using SimulationBox = Box;
using SpinCoreLayout = NEP_Spin::Spin_Layout;

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
__device__ __constant__ float kSpinChiralQohReducedCoeff[kSpinChiralQohReducedCount] = {
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

#include "nep_spin_descriptor.cuh"
#include "nep_spin_force.cuh"

__device__ int required_small_box_capacity[1];

__device__ __forceinline__ void add_compensated(
  const float value, float& sum, float& compensation)
{
  const float corrected = value - compensation;
  const float updated = sum + corrected;
  compensation = (updated - sum) - corrected;
  sum = updated;
}

__global__ void find_local_neighbors_spin(
  const int N,
  const Box box,
  const float cutoff_radial_square,
  const float cutoff_angular_square,
  const float cutoff_spin_square,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int* __restrict__ NN_global,
  const int* __restrict__ NL_global,
  int* NN_radial,
  int* NL_radial,
  int* NN_angular,
  int* NL_angular,
  int* NN_spin,
  int* NL_spin)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }

  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  int count_radial = 0;
  int count_angular = 0;
  int count_spin = 0;

  for (int slot = 0; slot < NN_global[n1]; ++slot) {
    const int n2 = NL_global[static_cast<std::size_t>(N) * slot + n1];
    float x12 = x[n2] - x1;
    float y12 = y[n2] - y1;
    float z12 = z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    const float distance_square = x12 * x12 + y12 * y12 + z12 * z12;

    if (distance_square < cutoff_radial_square) {
      NL_radial[static_cast<std::size_t>(N) * count_radial++ + n1] = n2;
    }
    if (distance_square < cutoff_angular_square) {
      NL_angular[static_cast<std::size_t>(N) * count_angular++ + n1] = n2;
    }
    if (distance_square < cutoff_spin_square) {
      NL_spin[static_cast<std::size_t>(N) * count_spin++ + n1] = n2;
    }
  }

  NN_radial[n1] = count_radial;
  NN_angular[n1] = count_angular;
  NN_spin[n1] = count_spin;
}

__device__ __forceinline__ void apply_mic_spin_small_box(
  const Box& box, const NEP::ExpandedBox& ebox, double& x12, double& y12, double& z12)
{
  double sx12 =
    (box.cpu_h[9] * x12 + box.cpu_h[10] * y12 + box.cpu_h[11] * z12) /
    ebox.num_cells[0];
  double sy12 =
    (box.cpu_h[12] * x12 + box.cpu_h[13] * y12 + box.cpu_h[14] * z12) /
    ebox.num_cells[1];
  double sz12 =
    (box.cpu_h[15] * x12 + box.cpu_h[16] * y12 + box.cpu_h[17] * z12) /
    ebox.num_cells[2];
  if (box.pbc_x == 1)
    sx12 -= nearbyint(sx12);
  if (box.pbc_y == 1)
    sy12 -= nearbyint(sy12);
  if (box.pbc_z == 1)
    sz12 -= nearbyint(sz12);
  x12 = box.cpu_h[0] * ebox.num_cells[0] * sx12 +
        box.cpu_h[1] * ebox.num_cells[1] * sy12 +
        box.cpu_h[2] * ebox.num_cells[2] * sz12;
  y12 = box.cpu_h[3] * ebox.num_cells[0] * sx12 +
        box.cpu_h[4] * ebox.num_cells[1] * sy12 +
        box.cpu_h[5] * ebox.num_cells[2] * sz12;
  z12 = box.cpu_h[6] * ebox.num_cells[0] * sx12 +
        box.cpu_h[7] * ebox.num_cells[1] * sy12 +
        box.cpu_h[8] * ebox.num_cells[2] * sz12;
}

__global__ void find_neighbor_list_spin_small_box(
  const NEP::ParaMB paramb,
  const float spin_cutoff,
  const int N,
  const Box box,
  const NEP::ExpandedBox ebox,
  const int capacity,
  const int* type,
  const double* x,
  const double* y,
  const double* z,
  int* NN_radial,
  int* NL_radial,
  int* NN_angular,
  int* NL_angular,
  int* NN_spin,
  int* NL_spin,
  double* r12_radial,
  double* r12_angular,
  double* r12_spin,
  int* required_capacity)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }
  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  const int t1 = type[n1];
  const int plane_size = N * capacity;
  int count_radial = 0;
  int count_angular = 0;
  int count_spin = 0;
  for (int n2 = 0; n2 < N; ++n2) {
    for (int ia = 0; ia < ebox.num_cells[0]; ++ia) {
      for (int ib = 0; ib < ebox.num_cells[1]; ++ib) {
        for (int ic = 0; ic < ebox.num_cells[2]; ++ic) {
          if (n1 == n2 && ia == 0 && ib == 0 && ic == 0) {
            continue;
          }
          double x12 = x[n2] + box.cpu_h[0] * ia + box.cpu_h[1] * ib +
                       box.cpu_h[2] * ic - x1;
          double y12 = y[n2] + box.cpu_h[3] * ia + box.cpu_h[4] * ib +
                       box.cpu_h[5] * ic - y1;
          double z12 = z[n2] + box.cpu_h[6] * ia + box.cpu_h[7] * ib +
                       box.cpu_h[8] * ic - z1;
          apply_mic_spin_small_box(box, ebox, x12, y12, z12);
          const double d2 = x12 * x12 + y12 * y12 + z12 * z12;
          const int t2 = type[n2];
          const float rc_radial = 0.5f * (paramb.rc_radial[t1] + paramb.rc_radial[t2]);
          const float rc_angular = 0.5f * (paramb.rc_angular[t1] + paramb.rc_angular[t2]);
          if (d2 < rc_radial * rc_radial) {
            const int slot = count_radial++;
            if (slot < capacity) {
              const int index = n1 + N * slot;
              NL_radial[index] = n2;
              r12_radial[index] = x12;
              r12_radial[plane_size + index] = y12;
              r12_radial[2 * plane_size + index] = z12;
            }
          }
          if (d2 < rc_angular * rc_angular) {
            const int slot = count_angular++;
            if (slot < capacity) {
              const int index = n1 + N * slot;
              NL_angular[index] = n2;
              r12_angular[index] = x12;
              r12_angular[plane_size + index] = y12;
              r12_angular[2 * plane_size + index] = z12;
            }
          }
          if (d2 < spin_cutoff * spin_cutoff) {
            const int slot = count_spin++;
            if (slot < capacity) {
              const int index = n1 + N * slot;
              NL_spin[index] = n2;
              r12_spin[index] = x12;
              r12_spin[plane_size + index] = y12;
              r12_spin[2 * plane_size + index] = z12;
            }
          }
        }
      }
    }
  }
  NN_radial[n1] = min(count_radial, capacity);
  NN_angular[n1] = min(count_angular, capacity);
  NN_spin[n1] = min(count_spin, capacity);
  atomicMax(required_capacity, max(count_radial, max(count_angular, count_spin)));
}

__global__ void find_structural_descriptor(
  const NEP::ParaMB paramb,
  const NEP::ANN ann,
  const int N,
  const Box box,
  const int* NN_radial,
  const int* NL_radial,
  const int* NN_angular,
  const int* NL_angular,
  const double* r12_radial,
  const double* r12_angular,
  const int r12_plane_size,
  const int* type,
  const double* x,
  const double* y,
  const double* z,
  float* descriptor,
  float* sum_fxyz)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }
  const int t1 = type[n1];
  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  float q[MAX_DIM] = {0.0f};
  float q_compensation[MAX_DIM] = {0.0f};

  for (int i1 = 0; i1 < NN_radial[n1]; ++i1) {
    const int n2 = NL_radial[static_cast<size_t>(N) * i1 + n1];
    const int slot = N * i1 + n1;
    double x12 = r12_radial ? r12_radial[slot] : x[n2] - x1;
    double y12 = r12_radial ? r12_radial[r12_plane_size + slot] : y[n2] - y1;
    double z12 = r12_radial ? r12_radial[2 * r12_plane_size + slot] : z[n2] - z1;
    if (!r12_radial) {
      apply_mic(box, x12, y12, z12);
    }
    const float d12 = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
    const int t2 = type[n2];
    const float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    float fc12;
    find_fc(rc, rcinv, d12, fc12);
    float fn12[MAX_NUM_N];
    find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
    const int basis_count =
      (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gn12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_radial; ++k) {
        const int basis = n * (paramb.basis_size_radial + 1) + k;
        gn12 += fn12[k] * ann.c_type_pair[(t1 * paramb.num_types + t2) * basis_count + basis];
      }
      add_compensated(gn12, q[n], q_compensation[n]);
    }
  }

  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    float s[NUM_OF_ABC] = {0.0f};
    float s_compensation[NUM_OF_ABC] = {0.0f};
    const int abc_count = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
    for (int i1 = 0; i1 < NN_angular[n1]; ++i1) {
      const int n2 = NL_angular[n1 + N * i1];
      const int slot = N * i1 + n1;
      double x12 = r12_angular ? r12_angular[slot] : x[n2] - x1;
      double y12 = r12_angular ? r12_angular[r12_plane_size + slot] : y[n2] - y1;
      double z12 = r12_angular ? r12_angular[2 * r12_plane_size + slot] : z[n2] - z1;
      if (!r12_angular) {
        apply_mic(box, x12, y12, z12);
      }
      const float d12 = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
      const int t2 = type[n2];
      const float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      const float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
      float gn12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index = paramb.num_c_radial;
        c_index += (t1 * paramb.num_types + t2) *
                   ((paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
        c_index += n * (paramb.basis_size_angular + 1) + k;
        gn12 += fn12[k] * ann.c_type_pair[c_index];
      }
      float edge_s[NUM_OF_ABC] = {0.0f};
      accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, edge_s);
      for (int abc = 0; abc < abc_count; ++abc) {
        add_compensated(edge_s[abc], s[abc], s_compensation[abc]);
      }
    }
    find_q(
      paramb.L_max,
      paramb.has_q_222,
      paramb.has_q_1111,
      paramb.has_q_112,
      paramb.has_q_123,
      paramb.has_q_233,
      paramb.has_q_134,
      paramb.n_max_angular + 1,
      n,
      s,
      q + paramb.n_max_radial + 1);
    for (int abc = 0; abc < abc_count; ++abc) {
      sum_fxyz[static_cast<size_t>(N) * (n * abc_count + abc) + n1] = s[abc];
    }
  }
  const int struct_dim = paramb.n_max_radial + 1 + paramb.dim_angular;
  for (int d = 0; d < struct_dim; ++d) {
    descriptor[static_cast<size_t>(N) * d + n1] = q[d];
  }
}

__global__ void find_potential(
  const int N,
  const int descriptor_dim,
  const int hidden_neurons,
  const int num_types,
  const int* type,
  const float* descriptor,
  const float* parameters,
  const float* q_scaler,
  const float* spin_baseline,
  double* potential,
  float* Fp)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= N) {
    return;
  }
  const int atom_type = type[atom];
  const std::size_t type_block =
    static_cast<std::size_t>(hidden_neurons) * (descriptor_dim + 2);
  const float* w0 = parameters + static_cast<std::size_t>(atom_type) * type_block;
  const float* b0 = w0 + static_cast<std::size_t>(hidden_neurons) * descriptor_dim;
  const float* w1 = b0 + hidden_neurons;
  const float* b1 = parameters + static_cast<std::size_t>(num_types) * type_block;
  double q[MAX_DIM] = {0.0};
  for (int d = 0; d < descriptor_dim; ++d) {
    q[d] = static_cast<double>(descriptor[static_cast<std::size_t>(N) * d + atom]) *
           static_cast<double>(q_scaler[d]);
  }
  double energy = 0.0;
  double fp[MAX_DIM] = {0.0};
  for (int neuron = 0; neuron < hidden_neurons; ++neuron) {
    double projection = 0.0;
    for (int d = 0; d < descriptor_dim; ++d) {
      projection +=
        static_cast<double>(w0[neuron * descriptor_dim + d]) * q[d];
    }
    const double activation = tanh(projection - static_cast<double>(b0[neuron]));
    const double activation_derivative = 1.0 - activation * activation;
    energy += static_cast<double>(w1[neuron]) * activation;
    for (int d = 0; d < descriptor_dim; ++d) {
      fp[d] += static_cast<double>(w1[neuron]) * activation_derivative *
               static_cast<double>(w0[neuron * descriptor_dim + d]);
    }
  }
  energy -= static_cast<double>(b1[0]);
  potential[atom] += energy + static_cast<double>(spin_baseline[atom_type]);
  for (int d = 0; d < descriptor_dim; ++d) {
    Fp[static_cast<std::size_t>(N) * d + atom] =
      static_cast<float>(fp[d] * static_cast<double>(q_scaler[d]));
  }
}


std::vector<std::string> split(const std::string& line)
{
  std::istringstream input(line);
  std::vector<std::string> tokens;
  std::string token;
  while (input >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<std::string> next_tokens(std::ifstream& input)
{
  std::string line;
  while (std::getline(input, line)) {
    std::vector<std::string> tokens = split(line);
    if (!tokens.empty()) {
      return tokens;
    }
  }
  return {};
}

int parse_int(const std::string& token)
{
  std::size_t end = 0;
  const long value = std::stol(token, &end);
  if (end != token.size() || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error("invalid integer token: " + token);
  }
  return static_cast<int>(value);
}

double parse_double(const std::string& token)
{
  std::size_t end = 0;
  const double value = std::stod(token, &end);
  if (end != token.size() || !std::isfinite(value)) {
    throw std::runtime_error("invalid finite numeric token: " + token);
  }
  return value;
}

float parse_float(const std::string& token)
{
  const double value = parse_double(token);
  if (std::abs(value) > std::numeric_limits<float>::max()) {
    throw std::runtime_error("model parameter is outside the float range");
  }
  return static_cast<float>(value);
}

void require_line(
  const std::vector<std::string>& tokens, const char* name, const std::size_t arity)
{
  if (tokens.size() != arity || tokens.empty() || tokens[0] != name) {
    throw std::runtime_error(
      std::string("expected exact '") + name + "' line with " +
      std::to_string(arity - 1) + " value(s)");
  }
}

bool parse_flag(const std::string& token, const char* name, const bool allow_legacy_two = false)
{
  const int value = parse_int(token);
  if (value != 0 && value != 1 && !(allow_legacy_two && value == 2)) {
    throw std::runtime_error(std::string(name) + " must be 0 or 1");
  }
  return value != 0;
}

void check_unique_elements(const std::vector<std::string>& elements)
{
  const std::set<std::string> unique(elements.begin(), elements.end());
  if (unique.size() != elements.size()) {
    throw std::runtime_error("model element symbols must be unique");
  }
}

void check_active_types(
  const std::vector<std::string>& tokens,
  const std::vector<std::string>& elements,
  const char* name)
{
  if (tokens.size() != elements.size() + 1) {
    throw std::runtime_error(std::string(name) + " must list every model type exactly once");
  }
  std::set<std::string> listed(tokens.begin() + 1, tokens.end());
  std::set<std::string> expected(elements.begin(), elements.end());
  if (listed != expected || listed.size() != tokens.size() - 1) {
    throw std::runtime_error(std::string(name) + " must activate all model types exactly once");
  }
}

NEP_Spin::Spin_Layout make_spin_layout(const NEP_Spin::Model& model)
{
  NEP_Spin::Spin_Layout layout;
  layout.channels = model.spin_compress;
  layout.basis_count = model.spin_basis_size[0] + 1;
  layout.l_max = model.spin_l_max[0];
  layout.chi_channels = std::min(2, layout.channels);

  int offset = 2 + 4 * layout.channels;
  layout.rho0_offset = offset;
  offset += layout.channels;
  if (layout.l_max >= 1) {
    layout.l1_rdot_offset = offset;
    offset += layout.channels;
    layout.l1_cross_offset = offset;
    offset += layout.channels;
    layout.l1_stf_offset = offset;
    offset += layout.channels;
  }
  if (layout.l_max >= 2) {
    layout.angular2_offset = offset;
    offset += layout.channels;
  }
  if (layout.l_max >= 3) {
    layout.angular3_offset = offset;
    offset += layout.channels;
  }
  if (layout.l_max >= 4) {
    layout.angular4_offset = offset;
    offset += layout.channels;
  }
  layout.geom_offset = offset;
  offset += layout.channels;
  layout.rho0_dot_offset = offset;
  offset += layout.channels;
  if (layout.l_max >= 1) {
    layout.raw1_dot_offset = offset;
    offset += layout.channels;
  }
  if (model.spin_chiral != 0) {
    layout.chiral_offset = offset;
    offset += layout.chi_channels + 2 * layout.channels;
  }
  layout.descriptor_dim = offset;
  return layout;
}

template <int C, int LMax>
void launch_spin_descriptor(
  const NEP_Spin::Model& model,
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  NEP_Spin_Data& data,
  const double* slot_r12 = nullptr,
  int r12_plane_size = 0)
{
  if (model.spin_chiral) {
    find_spin_descriptor<C, LMax, true><<<type.size(), 128>>>(
      type.size(),
      type.size(),
      model.struct_descriptor_dim,
      model.num_types,
      model.spin_basis_size[0],
      static_cast<float>(model.spin_cutoff[0]),
      box,
      type.data(),
      position.data(),
      spin.data(),
      data.NN_spin.data(),
      data.NL_spin.data(),
      slot_r12,
      r12_plane_size,
      data.descriptor_parameters_type_pair.data(),
      model.radial_parameter_count + model.angular_parameter_count,
      data.rho0.data(),
      data.raw1.data(),
      data.angular2.data(),
      data.angular3.data(),
      data.angular4.data(),
      data.geom.data(),
      data.rho0_dot.data(),
      data.raw1_dot.data(),
      data.polar.data(),
      data.octupole.data(),
      data.hexadecapole.data(),
      data.descriptor.data());
    const int work_items = type.size() * C;
    find_spin_descriptor_chiral<C><<<(work_items + 127) / 128, 128>>>(
      type.size(),
      type.size(),
      model.struct_descriptor_dim,
      model.spin_layout,
      spin.data(),
      data.geom.data(),
      data.raw1.data(),
      data.polar.data(),
      data.octupole.data(),
      data.hexadecapole.data(),
      data.chirals.data(),
      data.descriptor.data());
  } else {
    find_spin_descriptor<C, LMax, false><<<type.size(), 128>>>(
      type.size(),
      type.size(),
      model.struct_descriptor_dim,
      model.num_types,
      model.spin_basis_size[0],
      static_cast<float>(model.spin_cutoff[0]),
      box,
      type.data(),
      position.data(),
      spin.data(),
      data.NN_spin.data(),
      data.NL_spin.data(),
      slot_r12,
      r12_plane_size,
      data.descriptor_parameters_type_pair.data(),
      model.radial_parameter_count + model.angular_parameter_count,
      data.rho0.data(),
      data.raw1.data(),
      data.angular2.data(),
      data.angular3.data(),
      data.angular4.data(),
      data.geom.data(),
      data.rho0_dot.data(),
      data.raw1_dot.data(),
      data.polar.data(),
      data.octupole.data(),
      data.hexadecapole.data(),
      data.descriptor.data());
  }
  GPU_CHECK_KERNEL
}

template <int C>
void launch_spin_descriptor_lmax(
  const NEP_Spin::Model& model,
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  NEP_Spin_Data& data,
  const double* slot_r12 = nullptr,
  int r12_plane_size = 0)
{
  switch (model.spin_l_max[0]) {
    case 0:
      launch_spin_descriptor<C, 0>(
        model, box, type, position, spin, data, slot_r12, r12_plane_size);
      break;
    case 1:
      launch_spin_descriptor<C, 1>(
        model, box, type, position, spin, data, slot_r12, r12_plane_size);
      break;
    case 2:
      launch_spin_descriptor<C, 2>(
        model, box, type, position, spin, data, slot_r12, r12_plane_size);
      break;
    case 3:
      launch_spin_descriptor<C, 3>(
        model, box, type, position, spin, data, slot_r12, r12_plane_size);
      break;
    case 4:
      launch_spin_descriptor<C, 4>(
        model, box, type, position, spin, data, slot_r12, r12_plane_size);
      break;
  }
}

void launch_spin_descriptors(
  const NEP_Spin::Model& model,
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  NEP_Spin_Data& data,
  const double* slot_r12 = nullptr,
  int r12_plane_size = 0)
{
  switch (model.spin_compress) {
    case 1:
      launch_spin_descriptor_lmax<1>(
        model, box, type, position, spin, data, slot_r12, r12_plane_size);
      break;
    case 2:
      launch_spin_descriptor_lmax<2>(
        model, box, type, position, spin, data, slot_r12, r12_plane_size);
      break;
    case 3:
      launch_spin_descriptor_lmax<3>(
        model, box, type, position, spin, data, slot_r12, r12_plane_size);
      break;
    case 4:
      launch_spin_descriptor_lmax<4>(
        model, box, type, position, spin, data, slot_r12, r12_plane_size);
      break;
  }
}

template <int C, int LMax>
void launch_spin_density_force(
  const NEP_Spin::Model& model,
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  NEP_Spin_Data& data,
  GPU_Vector<double>& force,
  GPU_Vector<double>& mforce,
  GPU_Vector<double>& virial,
  const double* slot_r12 = nullptr,
  int r12_plane_size = 0)
{
  constexpr int atoms_per_warp = 4;
  constexpr int edges_per_atom = 8;
  find_force_spin_density<
    C,
    LMax,
    SpinVirialMode::neighbor_owned,
    false,
    atoms_per_warp,
    edges_per_atom><<<(type.size() + atoms_per_warp - 1) / atoms_per_warp, 32>>>(
    type.size(),
    type.size(),
    model.struct_descriptor_dim,
    model.num_types,
    model.spin_basis_size[0],
    static_cast<float>(model.spin_cutoff[0]),
    box,
    type.data(),
    position.data(),
    spin.data(),
    data.NN_spin.data(),
    data.NL_spin.data(),
    slot_r12,
    r12_plane_size,
    data.Fp.data(),
    data.descriptor_parameters_type_pair.data(),
    model.radial_parameter_count + model.angular_parameter_count,
    data.rho0.data(),
    data.angular2.data(),
    data.angular3.data(),
    data.angular4.data(),
    data.geom.data(),
    data.rho0_dot.data(),
    data.raw1.data(),
    data.raw1_dot.data(),
    force.data(),
    mforce.data(),
    virial.data(),
    nullptr,
    nullptr);
}

template <int C>
void launch_spin_chiral_force(
  const NEP_Spin::Model& model,
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  NEP_Spin_Data& data,
  GPU_Vector<double>& force,
  GPU_Vector<double>& mforce,
  GPU_Vector<double>& virial,
  const double* slot_r12 = nullptr,
  int r12_plane_size = 0)
{
  constexpr int atoms_per_warp = 8;
  constexpr int edges_per_atom = 4;
  find_force_spin_chiral<
    C,
    SpinVirialMode::neighbor_owned,
    false,
    atoms_per_warp,
    edges_per_atom><<<(type.size() + atoms_per_warp - 1) / atoms_per_warp, 32>>>(
    type.size(),
    type.size(),
    model.struct_descriptor_dim,
    model.num_types,
    model.spin_basis_size[0],
    model.spin_layout,
    static_cast<float>(model.spin_cutoff[0]),
    box,
    type.data(),
    position.data(),
    spin.data(),
    data.NN_spin.data(),
    data.NL_spin.data(),
    slot_r12,
    r12_plane_size,
    data.Fp.data(),
    data.descriptor_parameters_type_pair.data(),
    model.radial_parameter_count + model.angular_parameter_count,
    data.geom.data(),
    data.raw1.data(),
    data.polar.data(),
    data.octupole.data(),
    data.hexadecapole.data(),
    data.chirals.data(),
    force.data(),
    mforce.data(),
    virial.data(),
    nullptr,
    nullptr);
}

template <int C>
void launch_spin_forces_lmax(
  const NEP_Spin::Model& model,
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  NEP_Spin_Data& data,
  GPU_Vector<double>& force,
  GPU_Vector<double>& mforce,
  GPU_Vector<double>& virial,
  const double* slot_r12 = nullptr,
  int r12_plane_size = 0)
{
  switch (model.spin_l_max[0]) {
    case 0:
      launch_spin_density_force<C, 0>(
        model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size);
      break;
    case 1:
      launch_spin_density_force<C, 1>(
        model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size);
      break;
    case 2:
      launch_spin_density_force<C, 2>(
        model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size);
      break;
    case 3:
      launch_spin_density_force<C, 3>(
        model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size);
      break;
    case 4:
      launch_spin_density_force<C, 4>(
        model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size);
      break;
  }
  if (model.spin_chiral) {
    launch_spin_chiral_force<C>(
      model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size);
  }
}

void launch_spin_forces(
  const NEP_Spin::Model& model,
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  NEP_Spin_Data& data,
  GPU_Vector<double>& force,
  GPU_Vector<double>& mforce,
  GPU_Vector<double>& virial,
  const double* slot_r12 = nullptr,
  int r12_plane_size = 0)
{
  switch (model.spin_compress) {
    case 1:
      launch_spin_forces_lmax<1>(
        model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size);
      break;
    case 2:
      launch_spin_forces_lmax<2>(
        model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size);
      break;
    case 3:
      launch_spin_forces_lmax<3>(
        model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size);
      break;
    case 4:
      launch_spin_forces_lmax<4>(
        model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size);
      break;
  }
  GPU_CHECK_KERNEL
}

} // namespace

int NEP_Spin::Body_Channels::count(void) const
{
  return l_max_3body + static_cast<int>(has_q_222) + static_cast<int>(has_q_1111) +
         static_cast<int>(has_q_112) + static_cast<int>(has_q_123) +
         static_cast<int>(has_q_233) + static_cast<int>(has_q_134);
}

NEP_Spin::NEP_Spin(const char* file_potential, const int num_atoms) : num_atoms_(num_atoms)
{
  try {
    read_model(file_potential);
  } catch (const std::exception& error) {
    PRINT_INPUT_ERROR(error.what());
  }

  rc = std::max({model_.cutoff_radial, model_.cutoff_angular, model_.spin_cutoff[0]});
  neighbor_.initialize(rc, num_atoms_, model_.neighbor_capacity);
  model_.neighbor_capacity = neighbor_.capacity();
  const std::size_t slots =
    static_cast<std::size_t>(num_atoms_) * static_cast<std::size_t>(model_.neighbor_capacity);
  data_.NN_radial.resize(num_atoms_);
  data_.NN_angular.resize(num_atoms_);
  data_.NN_spin.resize(num_atoms_);
  data_.NL_radial.resize(slots);
  data_.NL_angular.resize(slots);
  data_.NL_spin.resize(slots);
  data_.descriptor.resize(static_cast<std::size_t>(num_atoms_) * model_.descriptor_dim);
  data_.Fp.resize(static_cast<std::size_t>(num_atoms_) * model_.descriptor_dim);
  data_.f12x.resize(slots);
  data_.f12y.resize(slots);
  data_.f12z.resize(slots);
  const std::size_t C = static_cast<std::size_t>(model_.spin_compress);
  const std::size_t N = static_cast<std::size_t>(num_atoms_);
  data_.rho0.resize(3 * C * N);
  data_.raw1.resize(9 * C * N);
  data_.angular2.resize(model_.spin_l_max[0] >= 2 ? 15 * C * N : 1);
  data_.angular3.resize(model_.spin_l_max[0] >= 3 ? 21 * C * N : 1);
  data_.angular4.resize(model_.spin_l_max[0] >= 4 ? 27 * C * N : 1);
  data_.geom.resize(6 * C * N);
  data_.rho0_dot.resize(3 * C * N);
  data_.raw1_dot.resize(9 * C * N);
  const std::size_t chi_c = std::min<std::size_t>(2, C);
  data_.polar.resize(model_.spin_chiral ? 3 * C * N : 1);
  data_.octupole.resize(model_.spin_chiral ? 7 * C * N : 1);
  data_.hexadecapole.resize(model_.spin_chiral ? 9 * chi_c * N : 1);
  data_.chirals.resize(model_.spin_chiral ? chi_c * N : 1);
  const std::size_t sum_fxyz_size =
    N * static_cast<std::size_t>(model_.n_max_angular + 1) *
    static_cast<std::size_t>(
      (model_.body.l_max_3body + 1) * (model_.body.l_max_3body + 1) - 1);
  data_.sum_fxyz.resize(std::max<std::size_t>(sum_fxyz_size, 1));

  printf("Use NEP4 spin potential with %d atom type%s.\n", model_.num_types, model_.num_types == 1 ? "" : "s");
  printf(
    "    structural/spin descriptor dimensions = %d/%d.\n",
    model_.struct_descriptor_dim,
    model_.spin_descriptor_dim);
  printf(
    "    radial/angular/spin cutoffs = %g/%g/%g A.\n",
    model_.cutoff_radial,
    model_.cutoff_angular,
    model_.spin_cutoff[0]);
}

void NEP_Spin::read_model(const char* file_potential)
{
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    throw std::runtime_error(std::string("cannot open spin potential: ") + file_potential);
  }

  std::vector<std::string> tokens = next_tokens(input);
  if (tokens.size() < 3 || tokens[0] != "nep4_spin1") {
    throw std::runtime_error("NEP_Spin accepts only counted canonical nep4_spin1 models");
  }
  model_.num_types = parse_int(tokens[1]);
  if (
    model_.num_types < 1 || model_.num_types > NUM_ELEMENTS ||
    tokens.size() != static_cast<std::size_t>(model_.num_types + 2)) {
    throw std::runtime_error("invalid NEP_Spin model type count");
  }
  model_.elements.assign(tokens.begin() + 2, tokens.end());
  check_unique_elements(model_.elements);

  tokens = next_tokens(input);
  require_line(tokens, "spin_mode", 3);
  if (parse_int(tokens[1]) != 1) {
    throw std::runtime_error("only spin_mode 1 is supported");
  }
  const int spin_header_lines = parse_int(tokens[2]);
  if (spin_header_lines < 8 || spin_header_lines > 10) {
    throw std::runtime_error("counted spin header must contain 8 to 10 lines");
  }

  bool seen_baseline = false;
  bool seen_n_max = false;
  bool seen_basis_size = false;
  bool seen_l_max = false;
  bool seen_compress = false;
  bool seen_cutoff = false;
  bool seen_chiral = false;
  bool seen_scaler = false;
  bool seen_dof = false;
  bool seen_env = false;
  for (int line = 0; line < spin_header_lines; ++line) {
    tokens = next_tokens(input);
    if (tokens.empty()) {
      throw std::runtime_error("truncated counted spin header");
    }
    const std::string& name = tokens[0];
    if (name == "spin_baseline") {
      if (seen_baseline || tokens.size() != static_cast<std::size_t>(model_.num_types + 1)) {
        throw std::runtime_error("spin_baseline must occur once with one value per type");
      }
      seen_baseline = true;
      for (int type = 0; type < model_.num_types; ++type) {
        model_.spin_baseline.push_back(parse_double(tokens[type + 1]));
      }
    } else if (name == "spin_n_max") {
      require_line(tokens, "spin_n_max", 3);
      if (seen_n_max)
        throw std::runtime_error("duplicate spin_n_max");
      seen_n_max = true;
      model_.spin_n_max[0] = parse_int(tokens[1]);
      model_.spin_n_max[1] = parse_int(tokens[2]);
    } else if (name == "spin_basis_size") {
      require_line(tokens, "spin_basis_size", 3);
      if (seen_basis_size)
        throw std::runtime_error("duplicate spin_basis_size");
      seen_basis_size = true;
      model_.spin_basis_size[0] = parse_int(tokens[1]);
      model_.spin_basis_size[1] = parse_int(tokens[2]);
    } else if (name == "spin_l_max") {
      require_line(tokens, "spin_l_max", 4);
      if (seen_l_max)
        throw std::runtime_error("duplicate spin_l_max");
      seen_l_max = true;
      for (int i = 0; i < 3; ++i)
        model_.spin_l_max[i] = parse_int(tokens[i + 1]);
    } else if (name == "spin_compress") {
      require_line(tokens, "spin_compress", 2);
      if (seen_compress)
        throw std::runtime_error("duplicate spin_compress");
      seen_compress = true;
      model_.spin_compress = parse_int(tokens[1]);
    } else if (name == "spin_cutoff") {
      require_line(tokens, "spin_cutoff", 3);
      if (seen_cutoff)
        throw std::runtime_error("duplicate spin_cutoff");
      seen_cutoff = true;
      model_.spin_cutoff[0] = parse_double(tokens[1]);
      model_.spin_cutoff[1] = parse_double(tokens[2]);
    } else if (name == "spin_chiral") {
      require_line(tokens, "spin_chiral", 2);
      if (seen_chiral)
        throw std::runtime_error("duplicate spin_chiral");
      seen_chiral = true;
      model_.spin_chiral = parse_flag(tokens[1], "spin_chiral");
    } else if (name == "spin_scaler") {
      require_line(tokens, "spin_scaler", 2);
      if (seen_scaler || parse_int(tokens[1]) != 1)
        throw std::runtime_error("spin_scaler must occur once and equal 1");
      seen_scaler = true;
    } else if (name == "spin_dof_type") {
      if (seen_dof)
        throw std::runtime_error("duplicate spin_dof_type");
      seen_dof = true;
      check_active_types(tokens, model_.elements, "spin_dof_type");
    } else if (name == "spin_env_type") {
      if (seen_env)
        throw std::runtime_error("duplicate spin_env_type");
      seen_env = true;
      check_active_types(tokens, model_.elements, "spin_env_type");
    } else {
      throw std::runtime_error("unknown or unsupported spin header line: " + name);
    }
  }
  if (
    !seen_baseline || !seen_n_max || !seen_basis_size || !seen_l_max || !seen_compress ||
    !seen_cutoff || !seen_chiral || !seen_scaler) {
    throw std::runtime_error("counted spin header is missing a required line");
  }

  tokens = next_tokens(input);
  require_line(tokens, "cutoff", 5);
  model_.cutoff_radial = parse_double(tokens[1]);
  model_.cutoff_angular = parse_double(tokens[2]);
  model_.max_neighbors_global = parse_int(tokens[3]);
  model_.max_neighbors_angular = parse_int(tokens[4]);

  tokens = next_tokens(input);
  require_line(tokens, "n_max", 3);
  model_.n_max_radial = parse_int(tokens[1]);
  model_.n_max_angular = parse_int(tokens[2]);

  tokens = next_tokens(input);
  require_line(tokens, "basis_size", 3);
  model_.basis_size_radial = parse_int(tokens[1]);
  model_.basis_size_angular = parse_int(tokens[2]);

  tokens = next_tokens(input);
  if (tokens.size() < 2 || tokens.size() > 8 || tokens[0] != "l_max") {
    throw std::runtime_error("invalid l_max line");
  }
  model_.body.l_max_3body = parse_int(tokens[1]);
  if (tokens.size() >= 3)
    model_.body.has_q_222 = parse_flag(tokens[2], "q_222", true);
  if (tokens.size() >= 4)
    model_.body.has_q_1111 = parse_flag(tokens[3], "q_1111");
  if (tokens.size() >= 5)
    model_.body.has_q_112 = parse_flag(tokens[4], "q_112");
  if (tokens.size() >= 6)
    model_.body.has_q_123 = parse_flag(tokens[5], "q_123");
  if (tokens.size() >= 7)
    model_.body.has_q_233 = parse_flag(tokens[6], "q_233");
  if (tokens.size() >= 8)
    model_.body.has_q_134 = parse_flag(tokens[7], "q_134");

  tokens = next_tokens(input);
  require_line(tokens, "ANN", 3);
  model_.hidden_neurons = parse_int(tokens[1]);
  if (parse_int(tokens[2]) != 0) {
    throw std::runtime_error("NEP_Spin supports only a single hidden ANN layer");
  }

  if (
    model_.cutoff_radial <= 0.0 || model_.cutoff_angular <= 0.0 ||
    model_.spin_cutoff[0] <= 0.0 || model_.spin_cutoff[1] <= 0.0 ||
    model_.max_neighbors_global <= 0 || model_.max_neighbors_angular <= 0) {
    throw std::runtime_error("cutoffs and neighbor capacities must be finite and positive");
  }
  if (model_.n_max_radial < 0 || model_.n_max_radial > 12 ||
      model_.n_max_angular < 0 || model_.n_max_angular > 8 ||
      model_.basis_size_radial < 0 || model_.basis_size_radial > 16 ||
      model_.basis_size_angular < 0 || model_.basis_size_angular > 12 ||
      model_.body.l_max_3body < 0 || model_.body.l_max_3body > 8 ||
      model_.hidden_neurons < 1 || model_.hidden_neurons > 120) {
    throw std::runtime_error("ordinary NEP_Spin shape is outside the supported range");
  }
  if ((model_.body.has_q_222 || model_.body.has_q_112) && model_.body.l_max_3body < 2)
    throw std::runtime_error("q_222/q_112 require l_max_3body >= 2");
  if (model_.body.has_q_1111 && model_.body.l_max_3body < 1)
    throw std::runtime_error("q_1111 requires l_max_3body >= 1");
  if ((model_.body.has_q_123 || model_.body.has_q_233) && model_.body.l_max_3body < 3)
    throw std::runtime_error("q_123/q_233 require l_max_3body >= 3");
  if (model_.body.has_q_134 && model_.body.l_max_3body < 4)
    throw std::runtime_error("q_134 requires l_max_3body >= 4");

  if (
    model_.spin_compress < 1 || model_.spin_compress > 4 ||
    model_.spin_basis_size[0] < 0 || model_.spin_basis_size[0] + 1 > 8 ||
    model_.spin_compress > model_.spin_basis_size[0] + 1 ||
    model_.spin_basis_size[1] < 0 || model_.spin_n_max[0] < 0 ||
    model_.spin_n_max[1] < 0 || model_.spin_n_max[0] > model_.spin_basis_size[0] ||
    model_.spin_n_max[1] > model_.spin_basis_size[1] || model_.spin_l_max[0] < 0 ||
    model_.spin_l_max[0] > 4 || model_.spin_l_max[1] != 0 || model_.spin_l_max[2] != 0) {
    throw std::runtime_error("spin descriptor shape is outside the supported Lite range");
  }

  model_.struct_descriptor_dim =
    model_.n_max_radial + 1 + (model_.n_max_angular + 1) * model_.body.count();
  if ((model_.n_max_angular + 1) * model_.body.count() > 90) {
    throw std::runtime_error("structural angular descriptor dimension exceeds 90");
  }
  model_.spin_layout = make_spin_layout(model_);
  model_.spin_descriptor_dim = model_.spin_layout.descriptor_dim;
  if (model_.spin_descriptor_dim > 96) {
    throw std::runtime_error("spin descriptor dimension exceeds 96");
  }
  model_.descriptor_dim = model_.struct_descriptor_dim + model_.spin_descriptor_dim;

  const std::size_t types = static_cast<std::size_t>(model_.num_types);
  const std::size_t type_pairs = types * types;
  model_.ann_parameter_count =
    static_cast<std::size_t>(model_.descriptor_dim + 2) * model_.hidden_neurons * types + 1;
  model_.radial_parameter_count = static_cast<std::size_t>(model_.n_max_radial + 1) *
                                  (model_.basis_size_radial + 1) * type_pairs;
  model_.angular_parameter_count = static_cast<std::size_t>(model_.n_max_angular + 1) *
                                   (model_.basis_size_angular + 1) * type_pairs;
  model_.spin_parameter_count = static_cast<std::size_t>(model_.spin_compress) *
                                (model_.spin_basis_size[0] + 1) * type_pairs;
  model_.model_parameter_count = model_.ann_parameter_count + model_.radial_parameter_count +
                                 model_.angular_parameter_count + model_.spin_parameter_count;

  std::vector<float> parameters;
  parameters.reserve(model_.model_parameter_count + model_.descriptor_dim);
  for (std::size_t i = 0; i < model_.model_parameter_count + model_.descriptor_dim; ++i) {
    tokens = next_tokens(input);
    if (tokens.size() != 1) {
      throw std::runtime_error("truncated or malformed NEP_Spin numeric payload");
    }
    parameters.push_back(parse_float(tokens[0]));
  }
  if (!next_tokens(input).empty()) {
    throw std::runtime_error("extra header or numeric payload after NEP_Spin model");
  }
  data_.parameters.resize(parameters.size());
  data_.parameters.copy_from_host(parameters.data());

  const std::size_t descriptor_offset = model_.ann_parameter_count;
  const std::size_t ordinary_count =
    model_.radial_parameter_count + model_.angular_parameter_count;
  std::vector<float> descriptor_parameters(ordinary_count + model_.spin_parameter_count);
  const std::size_t radial_basis =
    static_cast<std::size_t>(model_.n_max_radial + 1) * (model_.basis_size_radial + 1);
  const std::size_t angular_basis =
    static_cast<std::size_t>(model_.n_max_angular + 1) * (model_.basis_size_angular + 1);
  for (std::size_t pair = 0; pair < type_pairs; ++pair) {
    for (std::size_t basis = 0; basis < radial_basis; ++basis) {
      descriptor_parameters[pair * radial_basis + basis] =
        parameters[descriptor_offset + basis * type_pairs + pair];
    }
    for (std::size_t basis = 0; basis < angular_basis; ++basis) {
      descriptor_parameters[
        model_.radial_parameter_count + pair * angular_basis + basis] =
        parameters[
          descriptor_offset + model_.radial_parameter_count + basis * type_pairs + pair];
    }
  }
  std::copy(
    parameters.begin() + descriptor_offset + ordinary_count,
    parameters.begin() + descriptor_offset + ordinary_count + model_.spin_parameter_count,
    descriptor_parameters.begin() + ordinary_count);
  data_.descriptor_parameters_type_pair.resize(descriptor_parameters.size());
  data_.descriptor_parameters_type_pair.copy_from_host(descriptor_parameters.data());

  std::vector<float> spin_baseline(model_.spin_baseline.begin(), model_.spin_baseline.end());
  data_.spin_baseline.resize(spin_baseline.size());
  data_.spin_baseline.copy_from_host(spin_baseline.data());

  paramb_.version = 4;
  paramb_.num_types = model_.num_types;
  paramb_.num_types_sq = model_.num_types * model_.num_types;
  paramb_.n_max_radial = model_.n_max_radial;
  paramb_.n_max_angular = model_.n_max_angular;
  paramb_.basis_size_radial = model_.basis_size_radial;
  paramb_.basis_size_angular = model_.basis_size_angular;
  paramb_.L_max = model_.body.l_max_3body;
  paramb_.num_L = model_.body.count();
  paramb_.has_q_222 = model_.body.has_q_222;
  paramb_.has_q_1111 = model_.body.has_q_1111;
  paramb_.has_q_112 = model_.body.has_q_112;
  paramb_.has_q_123 = model_.body.has_q_123;
  paramb_.has_q_233 = model_.body.has_q_233;
  paramb_.has_q_134 = model_.body.has_q_134;
  paramb_.dim_angular = (model_.n_max_angular + 1) * model_.body.count();
  paramb_.num_c_radial = static_cast<int>(model_.radial_parameter_count);
  paramb_.MN_radial = model_.neighbor_capacity;
  paramb_.MN_angular = model_.neighbor_capacity;
  paramb_.rc_radial_max = model_.cutoff_radial;
  paramb_.rc_radial_max_inv = 1.0f / model_.cutoff_radial;
  for (int type = 0; type < model_.num_types; ++type) {
    paramb_.rc_radial[type] = model_.cutoff_radial;
    paramb_.rc_angular[type] = model_.cutoff_angular;
  }

  ann_.dim = model_.descriptor_dim;
  ann_.num_neurons1 = model_.hidden_neurons;
  ann_.num_para_ann = static_cast<int>(model_.ann_parameter_count);
  ann_.num_para = static_cast<int>(model_.model_parameter_count);
  float* pointer = data_.parameters.data();
  for (int type = 0; type < model_.num_types; ++type) {
    ann_.w0[type] = pointer;
    pointer += model_.hidden_neurons * model_.descriptor_dim;
    ann_.b0[type] = pointer;
    pointer += model_.hidden_neurons;
    ann_.w1[type] = pointer;
    pointer += model_.hidden_neurons;
  }
  ann_.b1 = pointer;
  ann_.c_type_pair = data_.descriptor_parameters_type_pair.data();
  ann_.q_scaler = data_.parameters.data() + model_.model_parameter_count;

  const double enlarged = std::ceil(static_cast<double>(model_.max_neighbors_global) * 1.25);
  if (enlarged > std::numeric_limits<int>::max()) {
    throw std::runtime_error("neighbor capacity is too large");
  }
  model_.neighbor_capacity = static_cast<int>(enlarged);
}

void NEP_Spin::compute(
  Box&,
  const GPU_Vector<int>&,
  const GPU_Vector<double>&,
  GPU_Vector<double>&,
  GPU_Vector<double>&,
  GPU_Vector<double>&)
{
  PRINT_INPUT_ERROR("NEP_Spin requires spin inputs.");
}

void NEP_Spin::initialize_small_box(const Box& box)
{
  const double volume = box.get_volume();
  const double thickness[3] = {
    volume / box.get_area(0), volume / box.get_area(1), volume / box.get_area(2)};
  expanded_box_.num_cells[0] =
    box.pbc_x ? static_cast<int>(std::ceil(2.0 * rc / thickness[0])) : 1;
  expanded_box_.num_cells[1] =
    box.pbc_y ? static_cast<int>(std::ceil(2.0 * rc / thickness[1])) : 1;
  expanded_box_.num_cells[2] =
    box.pbc_z ? static_cast<int>(std::ceil(2.0 * rc / thickness[2])) : 1;

  const std::size_t cell_count =
    static_cast<std::size_t>(expanded_box_.num_cells[0]) *
    static_cast<std::size_t>(expanded_box_.num_cells[1]) *
    static_cast<std::size_t>(expanded_box_.num_cells[2]);
  if (
    cell_count == 0 ||
    cell_count > (std::numeric_limits<std::size_t>::max() - 1) /
                   static_cast<std::size_t>(num_atoms_)) {
    PRINT_INPUT_ERROR("NEP_Spin small-box image count overflows size_t.");
  }
  const std::size_t capacity =
    static_cast<std::size_t>(num_atoms_) * cell_count - 1;
  if (capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    PRINT_INPUT_ERROR("NEP_Spin small-box neighbor capacity exceeds int.");
  }
  const std::size_t plane_size = static_cast<std::size_t>(num_atoms_) * capacity;
  if (capacity != 0 && plane_size / capacity != static_cast<std::size_t>(num_atoms_)) {
    PRINT_INPUT_ERROR("NEP_Spin small-box slot count overflows size_t.");
  }
  if (plane_size > std::numeric_limits<std::size_t>::max() / (3 * sizeof(double))) {
    PRINT_INPUT_ERROR("NEP_Spin small-box displacement storage overflows size_t.");
  }
  if (plane_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    PRINT_INPUT_ERROR("NEP_Spin small-box slot plane exceeds int indexing.");
  }
  const int new_capacity = static_cast<int>(capacity);
  const bool resize_storage =
    !small_box_initialized_ || small_box_capacity_ != new_capacity;
  small_box_capacity_ = new_capacity;

  expanded_box_.h[0] = box.cpu_h[0] * expanded_box_.num_cells[0];
  expanded_box_.h[3] = box.cpu_h[3] * expanded_box_.num_cells[0];
  expanded_box_.h[6] = box.cpu_h[6] * expanded_box_.num_cells[0];
  expanded_box_.h[1] = box.cpu_h[1] * expanded_box_.num_cells[1];
  expanded_box_.h[4] = box.cpu_h[4] * expanded_box_.num_cells[1];
  expanded_box_.h[7] = box.cpu_h[7] * expanded_box_.num_cells[1];
  expanded_box_.h[2] = box.cpu_h[2] * expanded_box_.num_cells[2];
  expanded_box_.h[5] = box.cpu_h[5] * expanded_box_.num_cells[2];
  expanded_box_.h[8] = box.cpu_h[8] * expanded_box_.num_cells[2];
  expanded_box_.h[9] =
    expanded_box_.h[4] * expanded_box_.h[8] - expanded_box_.h[5] * expanded_box_.h[7];
  expanded_box_.h[10] =
    expanded_box_.h[2] * expanded_box_.h[7] - expanded_box_.h[1] * expanded_box_.h[8];
  expanded_box_.h[11] =
    expanded_box_.h[1] * expanded_box_.h[5] - expanded_box_.h[2] * expanded_box_.h[4];
  expanded_box_.h[12] =
    expanded_box_.h[5] * expanded_box_.h[6] - expanded_box_.h[3] * expanded_box_.h[8];
  expanded_box_.h[13] =
    expanded_box_.h[0] * expanded_box_.h[8] - expanded_box_.h[2] * expanded_box_.h[6];
  expanded_box_.h[14] =
    expanded_box_.h[2] * expanded_box_.h[3] - expanded_box_.h[0] * expanded_box_.h[5];
  expanded_box_.h[15] =
    expanded_box_.h[3] * expanded_box_.h[7] - expanded_box_.h[4] * expanded_box_.h[6];
  expanded_box_.h[16] =
    expanded_box_.h[1] * expanded_box_.h[6] - expanded_box_.h[0] * expanded_box_.h[7];
  expanded_box_.h[17] =
    expanded_box_.h[0] * expanded_box_.h[4] - expanded_box_.h[1] * expanded_box_.h[3];
  const double determinant =
    expanded_box_.h[0] *
      (expanded_box_.h[4] * expanded_box_.h[8] -
       expanded_box_.h[5] * expanded_box_.h[7]) +
    expanded_box_.h[1] *
      (expanded_box_.h[5] * expanded_box_.h[6] -
       expanded_box_.h[3] * expanded_box_.h[8]) +
    expanded_box_.h[2] *
      (expanded_box_.h[3] * expanded_box_.h[7] -
       expanded_box_.h[4] * expanded_box_.h[6]);
  for (int index = 9; index < 18; ++index) {
    expanded_box_.h[index] /= determinant;
  }

  if (resize_storage) {
    data_.NL_radial.resize(plane_size);
    data_.NL_angular.resize(plane_size);
    data_.NL_spin.resize(plane_size);
    data_.r12_radial.resize(3 * plane_size);
    data_.r12_angular.resize(3 * plane_size);
    data_.r12_spin.resize(3 * plane_size);
    data_.f12x.resize(plane_size);
    data_.f12y.resize(plane_size);
    data_.f12z.resize(plane_size);
  }
  small_box_initialized_ = true;
}

void NEP_Spin::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial,
  GPU_Vector<double>& mforce)
{
  const int N = type.size();
  if (
    N != num_atoms_ || position.size() != static_cast<std::size_t>(3 * N) ||
    spin.size() != static_cast<std::size_t>(3 * N)) {
    PRINT_INPUT_ERROR("NEP_Spin input sizes do not match the initialized atom count.");
  }
  if (!(box.pbc_x && box.pbc_y && box.pbc_z)) {
    PRINT_INPUT_ERROR("NEP_Spin requires periodic boundary conditions in all three directions.");
  }
  const double volume = box.get_volume();
  const double thickness_x = volume / box.get_area(0);
  const double thickness_y = volume / box.get_area(1);
  const double thickness_z = volume / box.get_area(2);
  const double minimum_safe_thickness = neighbor_.minimum_safe_box_thickness(rc);
  const bool is_small_box =
    (box.pbc_x && thickness_x < minimum_safe_thickness) ||
    (box.pbc_y && thickness_y < minimum_safe_thickness) ||
    (box.pbc_z && thickness_z < minimum_safe_thickness);

  constexpr int block_size = 64;
  const int grid_size = (N - 1) / block_size + 1;
  if (is_small_box) {
    initialize_small_box(box);
    int* gpu_required_capacity = nullptr;
    CHECK(gpuGetSymbolAddress(
      reinterpret_cast<void**>(&gpu_required_capacity), required_small_box_capacity));
    const int zero = 0;
    CHECK(gpuMemcpy(gpu_required_capacity, &zero, sizeof(int), gpuMemcpyHostToDevice));
    find_neighbor_list_spin_small_box<<<grid_size, block_size>>>(
      paramb_,
      static_cast<float>(model_.spin_cutoff[0]),
      N,
      box,
      expanded_box_,
      small_box_capacity_,
      type.data(),
      position.data(),
      position.data() + N,
      position.data() + 2 * N,
      data_.NN_radial.data(),
      data_.NL_radial.data(),
      data_.NN_angular.data(),
      data_.NL_angular.data(),
      data_.NN_spin.data(),
      data_.NL_spin.data(),
      data_.r12_radial.data(),
      data_.r12_angular.data(),
      data_.r12_spin.data(),
      gpu_required_capacity);
    GPU_CHECK_KERNEL
    int required_capacity = 0;
    CHECK(gpuMemcpy(
      &required_capacity, gpu_required_capacity, sizeof(int), gpuMemcpyDeviceToHost));
    if (required_capacity > small_box_capacity_) {
      PRINT_INPUT_ERROR(
        ("NEP_Spin small-box neighbor capacity overflow: required " +
         std::to_string(required_capacity) + " slots but capacity is " +
         std::to_string(small_box_capacity_) + ".")
          .c_str());
    }
    const int plane_size = N * small_box_capacity_;
    find_structural_descriptor<<<grid_size, block_size>>>(
      paramb_,
      ann_,
      N,
      box,
      data_.NN_radial.data(),
      data_.NL_radial.data(),
      data_.NN_angular.data(),
      data_.NL_angular.data(),
      data_.r12_radial.data(),
      data_.r12_angular.data(),
      plane_size,
      type.data(),
      position.data(),
      position.data() + N,
      position.data() + 2 * N,
      data_.descriptor.data(),
      data_.sum_fxyz.data());
    GPU_CHECK_KERNEL
    launch_spin_descriptors(
      model_, box, type, position, spin, data_, data_.r12_spin.data(), plane_size);
    find_potential<<<grid_size, block_size>>>(
      N,
      model_.descriptor_dim,
      model_.hidden_neurons,
      model_.num_types,
      type.data(),
      data_.descriptor.data(),
      data_.parameters.data(),
      data_.parameters.data() + model_.model_parameter_count,
      data_.spin_baseline.data(),
      potential.data(),
      data_.Fp.data());
    GPU_CHECK_KERNEL
    find_force_radial_small_box<true><<<grid_size, block_size>>>(
      paramb_,
      ann_,
      N,
      0,
      N,
      data_.NN_radial.data(),
      data_.NL_radial.data(),
      type.data(),
      data_.r12_radial.data(),
      data_.r12_radial.data() + plane_size,
      data_.r12_radial.data() + 2 * plane_size,
      data_.Fp.data(),
      false,
      force.data(),
      force.data() + N,
      force.data() + 2 * N,
      virial.data());
    GPU_CHECK_KERNEL
    find_force_angular_small_box<true><<<grid_size, block_size>>>(
      paramb_,
      ann_,
      N,
      0,
      N,
      data_.NN_angular.data(),
      data_.NL_angular.data(),
      type.data(),
      data_.r12_angular.data(),
      data_.r12_angular.data() + plane_size,
      data_.r12_angular.data() + 2 * plane_size,
      data_.Fp.data(),
      data_.sum_fxyz.data(),
      false,
      force.data(),
      force.data() + N,
      force.data() + 2 * N,
      virial.data());
    GPU_CHECK_KERNEL
    find_mforce_onsite<<<grid_size, block_size>>>(
      N, N, model_.struct_descriptor_dim, spin.data(), data_.Fp.data(), mforce.data());
    launch_spin_forces(
      model_,
      box,
      type,
      position,
      spin,
      data_,
      force,
      mforce,
      virial,
      data_.r12_spin.data(),
      plane_size);
    return;
  }

  neighbor_.find_neighbor_global(rc, box, type, position);
  constexpr int neighbor_block_size = 128;
  const int neighbor_grid_size = (N - 1) / neighbor_block_size + 1;
  find_local_neighbors_spin<<<neighbor_grid_size, neighbor_block_size>>>(
    N,
    box,
    static_cast<float>(model_.cutoff_radial * model_.cutoff_radial),
    static_cast<float>(model_.cutoff_angular * model_.cutoff_angular),
    static_cast<float>(model_.spin_cutoff[0] * model_.spin_cutoff[0]),
    position.data(),
    position.data() + N,
    position.data() + 2 * N,
    neighbor_.NN.data(),
    neighbor_.NL.data(),
    data_.NN_radial.data(),
    data_.NL_radial.data(),
    data_.NN_angular.data(),
    data_.NL_angular.data(),
    data_.NN_spin.data(),
    data_.NL_spin.data());
  GPU_CHECK_KERNEL

  find_structural_descriptor<<<grid_size, block_size>>>(
    paramb_,
    ann_,
    N,
    box,
    data_.NN_radial.data(),
    data_.NL_radial.data(),
    data_.NN_angular.data(),
    data_.NL_angular.data(),
    nullptr,
    nullptr,
    0,
    type.data(),
    position.data(),
    position.data() + N,
    position.data() + 2 * N,
    data_.descriptor.data(),
    data_.sum_fxyz.data());
  GPU_CHECK_KERNEL

  launch_spin_descriptors(model_, box, type, position, spin, data_);
  find_potential<<<grid_size, block_size>>>(
    N,
    model_.descriptor_dim,
    model_.hidden_neurons,
    model_.num_types,
    type.data(),
    data_.descriptor.data(),
    data_.parameters.data(),
    data_.parameters.data() + model_.model_parameter_count,
    data_.spin_baseline.data(),
    potential.data(),
    data_.Fp.data());
  GPU_CHECK_KERNEL

  find_force_radial<<<grid_size, block_size>>>(
    paramb_,
    ann_,
    N,
    0,
    N,
    box,
    data_.NN_radial.data(),
    data_.NL_radial.data(),
    type.data(),
    position.data(),
    position.data() + N,
    position.data() + 2 * N,
    data_.Fp.data(),
    false,
    force.data(),
    force.data() + N,
    force.data() + 2 * N,
    virial.data());
  GPU_CHECK_KERNEL

  find_partial_force_angular<<<grid_size, block_size>>>(
    paramb_,
    ann_,
    N,
    0,
    N,
    box,
    data_.NN_angular.data(),
    data_.NL_angular.data(),
    type.data(),
    position.data(),
    position.data() + N,
    position.data() + 2 * N,
    data_.Fp.data(),
    data_.sum_fxyz.data(),
    data_.f12x.data(),
    data_.f12y.data(),
    data_.f12z.data());
  GPU_CHECK_KERNEL
  find_properties_many_body(
    box,
    data_.NN_angular.data(),
    data_.NL_angular.data(),
    data_.f12x.data(),
    data_.f12y.data(),
    data_.f12z.data(),
    false,
    position,
    force,
    virial);

  find_mforce_onsite<<<grid_size, block_size>>>(
    N, N, model_.struct_descriptor_dim, spin.data(), data_.Fp.data(), mforce.data());
  launch_spin_forces(model_, box, type, position, spin, data_, force, mforce, virial);
}
