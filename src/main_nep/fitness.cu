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

/*----------------------------------------------------------------------------80
Get the fitness
------------------------------------------------------------------------------*/

#include "fitness.cuh"
#include "nep.cuh"
#include "nep_vdw.cuh"
#include "nep_charge.cuh"
#include "nep_spin.cuh"
#include "nep_charge_vdw.cuh"
#include "tnep.cuh"
#include "parameters.cuh"
#include "structure.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include "utilities/nep_parameters.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <vector>
#include <cstring>

namespace {

void load_spin_checkpoint_metadata(Parameters& para)
{
  std::ifstream input("nep.txt");
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Failed to open nep.txt for Spin NEP prediction.");
  }
  auto tokens = get_tokens(input);
  const std::string expected_tag = para.enable_zbl
    ? "nep4_spin3_zbl"
    : "nep4_spin3";
  if (tokens.size() != static_cast<std::size_t>(para.num_types + 2) ||
      tokens[0] != expected_tag ||
      get_int_from_token(tokens[1], __FILE__, __LINE__) != para.num_types) {
    PRINT_INPUT_ERROR("Spin NEP checkpoint header does not match nep.in.");
  }
  for (int type = 0; type < para.num_types; ++type) {
    if (tokens[type + 2] != para.elements[type]) {
      PRINT_INPUT_ERROR("Spin NEP checkpoint atom types do not match nep.in.");
    }
  }
  tokens = get_tokens(input);
  if (tokens.size() != 3 || tokens[0] != "spin_mode" ||
      get_int_from_token(tokens[1], __FILE__, __LINE__) != para.spin_mode) {
    PRINT_INPUT_ERROR("Spin NEP checkpoint has an invalid counted spin header.");
  }
  const int line_count = get_int_from_token(tokens[2], __FILE__, __LINE__);
  constexpr int minimum_lines = 9;
  if (line_count < minimum_lines || line_count > minimum_lines + 2) {
    PRINT_INPUT_ERROR("Spin NEP checkpoint has an invalid counted header length.");
  }

  bool seen_baseline = false;
  bool seen_basis_size = false;
  bool seen_l_max = false;
  bool seen_compress = false;
  bool seen_cutoff = false;
  bool seen_scaler = false;
  bool seen_order = false;
  bool seen_soc = false;
  bool seen_projection_size = false;
  bool seen_dof = false;
  bool seen_env = false;
  std::vector<int> dof(para.num_types, 1);
  std::vector<int> env;
  auto parse_active = [&](const std::vector<std::string>& line_tokens) {
    std::vector<int> active(para.num_types, 0);
    if (line_tokens.size() < 2) {
      PRINT_INPUT_ERROR("Spin type-selection line must enable at least one type.");
    }
    for (std::size_t index = 1; index < line_tokens.size(); ++index) {
      auto found =
        std::find(para.elements.begin(), para.elements.end(), line_tokens[index]);
      if (found == para.elements.end()) {
        PRINT_INPUT_ERROR("Spin checkpoint contains an unknown atom type.");
      }
      const int type = static_cast<int>(found - para.elements.begin());
      if (active[type]) {
        PRINT_INPUT_ERROR("Spin checkpoint contains a duplicate atom type.");
      }
      active[type] = 1;
    }
    return active;
  };
  for (int line = 0; line < line_count; ++line) {
    tokens = get_tokens(input);
    if (tokens.empty()) {
      PRINT_INPUT_ERROR("Spin NEP checkpoint has a truncated spin header.");
    }
    const std::string& keyword = tokens[0];
    if (keyword == "spin_baseline") {
      if (seen_baseline ||
          tokens.size() != static_cast<std::size_t>(para.num_types + 1)) {
        PRINT_INPUT_ERROR("Invalid spin_baseline in Spin NEP checkpoint.");
      }
      seen_baseline = true;
      para.spin_baseline.resize(para.num_types);
      for (int type = 0; type < para.num_types; ++type) {
        para.spin_baseline[type] =
          get_double_from_token(tokens[type + 1], __FILE__, __LINE__);
      }
    } else if (keyword == "spin_basis_size") {
      if (seen_basis_size || tokens.size() != 2 ||
          get_int_from_token(tokens[1], __FILE__, __LINE__) !=
            para.spin_basis_size[0]) {
        PRINT_INPUT_ERROR("spin_basis_size in nep.txt does not match nep.in.");
      }
      seen_basis_size = true;
    } else if (keyword == "spin_l_max") {
      if (seen_l_max || tokens.size() != 2) {
        PRINT_INPUT_ERROR("Invalid spin_l_max in Spin NEP checkpoint.");
      }
      if (get_int_from_token(tokens[1], __FILE__, __LINE__) !=
          para.spin_l_max[0]) {
        PRINT_INPUT_ERROR("spin_l_max in nep.txt does not match nep.in.");
      }
      seen_l_max = true;
    } else if (keyword == "spin_compress") {
      if (seen_compress || tokens.size() != 2 ||
          get_int_from_token(tokens[1], __FILE__, __LINE__) != para.spin_compress) {
        PRINT_INPUT_ERROR("spin_compress in nep.txt does not match nep.in.");
      }
      seen_compress = true;
    } else if (keyword == "spin_cutoff") {
      if (seen_cutoff ||
          (tokens.size() != 2 &&
           tokens.size() != static_cast<std::size_t>(para.num_types + 1))) {
        PRINT_INPUT_ERROR("Invalid spin_cutoff in Spin NEP checkpoint.");
      }
      for (int type = 0; type < para.num_types; ++type) {
        const std::size_t value_index = tokens.size() == 2 ? 1 : type + 1;
        const double value =
          get_double_from_token(tokens[value_index], __FILE__, __LINE__);
        if (std::abs(value - para.spin_cutoff_by_type[type]) >
            1.0e-6 * std::max(1.0, std::abs(value))) {
          PRINT_INPUT_ERROR("spin_cutoff in nep.txt does not match nep.in.");
        }
      }
      seen_cutoff = true;
    } else if (keyword == "spin_order") {
      if (seen_order || tokens.size() != 2 ||
          get_int_from_token(tokens[1], __FILE__, __LINE__) != para.spin_order) {
        PRINT_INPUT_ERROR("spin_order in nep.txt does not match nep.in.");
      }
      seen_order = true;
    } else if (keyword == "spin_soc") {
      if (seen_soc || tokens.size() != 2 ||
          get_int_from_token(tokens[1], __FILE__, __LINE__) != para.spin_soc) {
        PRINT_INPUT_ERROR("spin_soc in nep.txt does not match nep.in.");
      }
      seen_soc = true;
    } else if (keyword == "spin_projection_size") {
      if (seen_projection_size || tokens.size() != 2 ||
          get_int_from_token(tokens[1], __FILE__, __LINE__) !=
            para.number_of_variables_spin_projection) {
        PRINT_INPUT_ERROR("spin_projection_size in nep.txt does not match nep.in.");
      }
      seen_projection_size = true;
    } else if (keyword == "spin_scaler") {
      if (seen_scaler || tokens.size() != 2 ||
          get_int_from_token(tokens[1], __FILE__, __LINE__) != 1) {
        PRINT_INPUT_ERROR("spin_scaler in nep.txt must occur once and equal 1.");
      }
      seen_scaler = true;
    } else if (keyword == "spin_dof_type") {
      if (seen_dof) {
        PRINT_INPUT_ERROR("Duplicate spin_dof_type in Spin NEP checkpoint.");
      }
      dof = parse_active(tokens);
      seen_dof = true;
    } else if (keyword == "spin_env_type") {
      if (seen_env) {
        PRINT_INPUT_ERROR("Duplicate spin_env_type in Spin NEP checkpoint.");
      }
      env = parse_active(tokens);
      seen_env = true;
    } else {
      PRINT_INPUT_ERROR("Unknown line in counted Spin NEP checkpoint header.");
    }
  }
  if (!seen_baseline || !seen_basis_size || !seen_l_max ||
      !seen_compress || !seen_cutoff || !seen_scaler || !seen_order ||
      !seen_soc || !seen_projection_size) {
    PRINT_INPUT_ERROR("Spin NEP checkpoint is missing required metadata.");
  }
  if (!seen_env) {
    env = dof;
  }
  for (int type = 0; type < para.num_types; ++type) {
    if (dof[type] && !env[type]) {
      PRINT_INPUT_ERROR("spin_dof_type must be a subset of spin_env_type.");
    }
  }
  para.spin_dof_type_active = dof;
  para.spin_env_type_active = env;
  if (para.enable_zbl) {
    tokens = get_tokens(input);
    const std::size_t expected_size = para.use_typewise_cutoff_zbl ? 4 : 3;
    if (tokens.size() != expected_size || tokens[0] != "zbl") {
      PRINT_INPUT_ERROR("Spin3 ZBL checkpoint has an invalid zbl line.");
    }
    const double inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    const double outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (std::abs(inner - para.zbl_rc_inner) > 1.0e-6 ||
        std::abs(outer - para.zbl_rc_outer) > 1.0e-6 ||
        (para.use_typewise_cutoff_zbl &&
         std::abs(get_double_from_token(tokens[3], __FILE__, __LINE__) -
                  para.typewise_cutoff_zbl_factor) > 1.0e-6)) {
      PRINT_INPUT_ERROR("Spin3 ZBL checkpoint does not match nep.in.");
    }
  }
}

void fit_spin_energy_baseline(
  const std::vector<Structure>& structures, Parameters& para)
{
  const int T = para.num_types;
  std::vector<double> gram(T * T, 0.0);
  std::vector<double> rhs(T, 0.0);
  std::vector<double> counts(T);
  for (const auto& structure : structures) {
    std::fill(counts.begin(), counts.end(), 0.0);
    for (const int type : structure.type) {
      counts[type] += 1.0;
    }
    const double total_energy =
      static_cast<double>(structure.energy) * structure.num_atom;
    for (int i = 0; i < T; ++i) {
      rhs[i] += counts[i] * total_energy;
      for (int j = 0; j < T; ++j) {
        gram[i * T + j] += counts[i] * counts[j];
      }
    }
  }

  std::vector<double> eigenvectors(T * T, 0.0);
  for (int i = 0; i < T; ++i) {
    eigenvectors[i * T + i] = 1.0;
  }
  const int max_sweeps = std::max(16, 32 * T * T);
  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    int p = 0;
    int q = 0;
    double largest = 0.0;
    for (int i = 0; i < T; ++i) {
      for (int j = i + 1; j < T; ++j) {
        const double value = std::abs(gram[i * T + j]);
        if (value > largest) {
          largest = value;
          p = i;
          q = j;
        }
      }
    }
    double diagonal_scale = 0.0;
    for (int i = 0; i < T; ++i) {
      diagonal_scale = std::max(diagonal_scale, std::abs(gram[i * T + i]));
    }
    if (largest <=
        std::numeric_limits<double>::epsilon() * std::max(1.0, diagonal_scale)) {
      break;
    }
    const double app = gram[p * T + p];
    const double aqq = gram[q * T + q];
    const double apq = gram[p * T + q];
    const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    for (int k = 0; k < T; ++k) {
      if (k == p || k == q) {
        continue;
      }
      const double akp = gram[k * T + p];
      const double akq = gram[k * T + q];
      gram[k * T + p] = gram[p * T + k] = c * akp - s * akq;
      gram[k * T + q] = gram[q * T + k] = s * akp + c * akq;
    }
    gram[p * T + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    gram[q * T + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    gram[p * T + q] = gram[q * T + p] = 0.0;
    for (int k = 0; k < T; ++k) {
      const double vkp = eigenvectors[k * T + p];
      const double vkq = eigenvectors[k * T + q];
      eigenvectors[k * T + p] = c * vkp - s * vkq;
      eigenvectors[k * T + q] = s * vkp + c * vkq;
    }
  }

  double largest_eigenvalue = 0.0;
  for (int i = 0; i < T; ++i) {
    largest_eigenvalue =
      std::max(largest_eigenvalue, std::max(0.0, gram[i * T + i]));
  }
  const double eigenvalue_tolerance =
    largest_eigenvalue *
    std::max(structures.size(), static_cast<std::size_t>(T)) *
    std::numeric_limits<double>::epsilon();
  para.spin_baseline.assign(T, 0.0f);
  for (int column = 0; column < T; ++column) {
    const double eigenvalue = gram[column * T + column];
    if (eigenvalue <= eigenvalue_tolerance) {
      continue;
    }
    double projected_rhs = 0.0;
    for (int row = 0; row < T; ++row) {
      projected_rhs += eigenvectors[row * T + column] * rhs[row];
    }
    const double scale = projected_rhs / eigenvalue;
    for (int row = 0; row < T; ++row) {
      para.spin_baseline[row] +=
        static_cast<float>(eigenvectors[row * T + column] * scale);
    }
  }
}

struct SpinResponsePoint {
  std::string group;
  double coordinate;
  double prediction;
  double target;
};

void derive_spin_response_tangents(
  const Parameters& para,
  std::vector<Structure>& structures)
{
  std::map<std::string, std::vector<Structure*>> groups;
  for (auto& structure : structures) {
    structure.has_spin_response = 0;
    structure.spin_tangent_x.clear();
    structure.spin_tangent_y.clear();
    structure.spin_tangent_z.clear();
    if (structure.has_spin_response_metadata &&
        structure.spin_response_probe == "rotation") {
      groups[structure.spin_response_group].push_back(&structure);
    }
  }
  if (groups.empty()) {
    PRINT_INPUT_ERROR(
      "lambda_spin_response requires rotation response frames in train.xyz.\n");
  }

  const auto differs = [](const float left, const float right) {
    return std::abs(static_cast<double>(left) - static_cast<double>(right)) > 1.0e-6;
  };
  for (auto& item : groups) {
    auto& members = item.second;
    std::sort(members.begin(), members.end(), [](const Structure* left, const Structure* right) {
      return left->spin_response_coordinate < right->spin_response_coordinate;
    });
    if (members.size() < 3) {
      PRINT_INPUT_ERROR(
        "Each rotation response_group needs at least three distinct "
        "response_coordinate values.\n");
    }
    const Structure& reference = *members.front();
    for (std::size_t k = 0; k < members.size(); ++k) {
      Structure& structure = *members[k];
      if (!std::isfinite(structure.spin_response_coordinate)) {
        PRINT_INPUT_ERROR("response_coordinate must be finite.\n");
      }
      if (k > 0 &&
          structure.spin_response_coordinate == members[k - 1]->spin_response_coordinate) {
        PRINT_INPUT_ERROR(
          "Each rotation response_group needs distinct response_coordinate values.\n");
      }
      if (!structure.has_mforce) {
        PRINT_INPUT_ERROR(
          "Each rotation response frame requires mforce:R:3.\n");
      }
      if (structure.num_atom != reference.num_atom ||
          structure.type != reference.type) {
        PRINT_INPUT_ERROR(
          "A rotation response_group cannot change atom count, type, or order.\n");
      }
      for (int d = 0; d < 9; ++d) {
        if (differs(structure.box_original[d], reference.box_original[d])) {
          PRINT_INPUT_ERROR(
            "A rotation response_group cannot change its cell.\n");
        }
      }
      for (int atom = 0; atom < structure.num_atom; ++atom) {
        if (differs(structure.x[atom], reference.x[atom]) ||
            differs(structure.y[atom], reference.y[atom]) ||
            differs(structure.z[atom], reference.z[atom])) {
          PRINT_INPUT_ERROR(
            "A rotation response_group cannot change atomic positions.\n");
        }
        if (!std::isfinite(structure.sx[atom]) ||
            !std::isfinite(structure.sy[atom]) ||
            !std::isfinite(structure.sz[atom])) {
          PRINT_INPUT_ERROR(
            "Converged DFT spins in a rotation response_group must be finite.\n");
        }
      }
      structure.spin_tangent_x.resize(structure.num_atom);
      structure.spin_tangent_y.resize(structure.num_atom);
      structure.spin_tangent_z.resize(structure.num_atom);
    }

    double tangent_power = 0.0;
    for (std::size_t k = 0; k < members.size(); ++k) {
      const std::size_t first = k == 0 ? 0 : (k + 1 == members.size() ? k - 2 : k - 1);
      const std::size_t second = first + 1;
      const std::size_t third = first + 2;
      const double x0 = members[first]->spin_response_coordinate;
      const double x1 = members[second]->spin_response_coordinate;
      const double x2 = members[third]->spin_response_coordinate;
      const double x = members[k]->spin_response_coordinate;
      const double w0 = (2.0 * x - x1 - x2) / ((x0 - x1) * (x0 - x2));
      const double w1 = (2.0 * x - x0 - x2) / ((x1 - x0) * (x1 - x2));
      const double w2 = (2.0 * x - x0 - x1) / ((x2 - x0) * (x2 - x1));
      Structure& output = *members[k];
      for (int atom = 0; atom < output.num_atom; ++atom) {
        const double tx =
          w0 * members[first]->sx[atom] +
          w1 * members[second]->sx[atom] +
          w2 * members[third]->sx[atom];
        const double ty =
          w0 * members[first]->sy[atom] +
          w1 * members[second]->sy[atom] +
          w2 * members[third]->sy[atom];
        const double tz =
          w0 * members[first]->sz[atom] +
          w1 * members[second]->sz[atom] +
          w2 * members[third]->sz[atom];
        const bool active = para.spin_dof_type_active[output.type[atom]] != 0;
        output.spin_tangent_x[atom] = active ? static_cast<float>(tx) : 0.0f;
        output.spin_tangent_y[atom] = active ? static_cast<float>(ty) : 0.0f;
        output.spin_tangent_z[atom] = active ? static_cast<float>(tz) : 0.0f;
        if (active) {
          tangent_power += tx * tx + ty * ty + tz * tz;
        }
      }
      output.has_spin_response = 1;
    }
    if (tangent_power == 0.0) {
      PRINT_INPUT_ERROR(
        "A rotation response_group must contain a changing converged DFT spin path.\n");
    }
  }
}

void append_spin_response_points(
  const int device_id,
  Dataset& dataset,
  std::vector<SpinResponsePoint>& points)
{
  CHECK(gpuSetDevice(device_id));
  dataset.mforce.copy_to_host(dataset.mforce_cpu.data());
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    const Structure& structure = dataset.structures[nc];
    if (!structure.has_spin_response) {
      continue;
    }
    double prediction = 0.0;
    double target = 0.0;
    const int atom_offset = dataset.Na_sum_cpu[nc];
    for (int na = 0; na < structure.num_atom; ++na) {
      const int atom = atom_offset + na;
      prediction +=
        dataset.mforce_cpu[atom] * structure.spin_tangent_x[na] +
        dataset.mforce_cpu[dataset.N + atom] * structure.spin_tangent_y[na] +
        dataset.mforce_cpu[2 * dataset.N + atom] * structure.spin_tangent_z[na];
      target +=
        structure.mfx[na] * structure.spin_tangent_x[na] +
        structure.mfy[na] * structure.spin_tangent_y[na] +
        structure.mfz[na] * structure.spin_tangent_z[na];
    }
    points.push_back({
      structure.spin_response_group,
      structure.spin_response_coordinate,
      prediction,
      target});
  }
}

double huber_loss(const double residual)
{
  const double absolute = std::abs(residual);
  return absolute <= 1.0 ? 0.5 * residual * residual : absolute - 0.5;
}

float spin_response_loss(const std::vector<SpinResponsePoint>& points)
{
  std::map<std::string, std::vector<const SpinResponsePoint*>> groups;
  for (const auto& point : points) {
    groups[point.group].push_back(&point);
  }
  if (groups.empty()) {
    return 0.0f;
  }

  double group_power_sum = 0.0;
  for (const auto& item : groups) {
    const auto& members = item.second;
    double target_power = 0.0;
    for (const auto* point : members) {
      target_power += point->target * point->target;
    }
    group_power_sum += target_power / members.size();
  }
  const double scale = std::max(
    std::sqrt(group_power_sum / groups.size()), 1.0e-6);
  double shape_loss = 0.0;
  double mean_loss = 0.0;
  for (const auto& item : groups) {
    const auto& members = item.second;
    double mean_prediction = 0.0;
    double mean_target = 0.0;
    for (const auto* point : members) {
      mean_prediction += point->prediction;
      mean_target += point->target;
    }
    mean_prediction /= members.size();
    mean_target /= members.size();
    double group_shape_loss = 0.0;
    for (const auto* point : members) {
      const double residual =
        ((point->prediction - mean_prediction) -
         (point->target - mean_target)) / scale;
      group_shape_loss += huber_loss(residual);
    }
    shape_loss += group_shape_loss / members.size();
    mean_loss += huber_loss((mean_prediction - mean_target) / scale);
  }
  shape_loss /= groups.size();
  mean_loss /= groups.size();
  return static_cast<float>(shape_loss + 0.25 * mean_loss);
}

} // namespace

Fitness::Fitness(Parameters& para)
{
  int deviceCount;
  CHECK(gpuGetDeviceCount(&deviceCount));

  const bool spin_restart = para.spin_mode && !para.prediction &&
    std::ifstream("nep.restart").good();
  if (para.spin_mode && (para.prediction || spin_restart)) {
    load_spin_checkpoint_metadata(para);
  }
  std::vector<Structure> structures_train;
  read_structures(true, para, structures_train);
  if (para.lambda_spin_response > 0.0f) {
    derive_spin_response_tangents(para, structures_train);
  }
  if (para.spin_mode && !para.prediction && !spin_restart) {
    fit_spin_energy_baseline(structures_train, para);
    printf("Spin energy baseline:");
    for (const float value : para.spin_baseline) {
      printf(" %.10g", value);
    }
    printf("\n");
  }
  num_batches = (structures_train.size() - 1) / para.batch_size + 1;
  if (para.lambda_spin_response > 0.0f && num_batches != 1) {
    PRINT_INPUT_ERROR(
      "lambda_spin_response requires batch >= the number of training frames "
      "so every response group is complete in each fitness evaluation.\n");
  }
  printf("Number of devices = %d\n", deviceCount);
  printf("Number of batches = %d\n", num_batches);
  int batch_size_old = para.batch_size;
  para.batch_size = (structures_train.size() - 1) / num_batches + 1;
  if (batch_size_old != para.batch_size) {
    printf("Hello, I changed the batch_size from %d to %d.\n", batch_size_old, para.batch_size);
  }

  train_set.resize(num_batches);
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    train_set[batch_id].resize(deviceCount);
  }
  int count = 0;
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    const int batch_size_minimal = structures_train.size() / num_batches;
    const bool is_larger_batch =
      batch_id + batch_size_minimal * num_batches < structures_train.size();
    const int batch_size = is_larger_batch ? batch_size_minimal + 1 : batch_size_minimal;
    count += batch_size;
    printf("\nBatch %d:\n", batch_id);
    printf("Number of configurations = %d.\n", batch_size);
    for (int device_id = 0; device_id < deviceCount; ++device_id) {
      print_line_1();
      printf("Constructing train_set in device  %d.\n", device_id);
      CHECK(gpuSetDevice(device_id));
      train_set[batch_id][device_id].construct(
        para, structures_train, count - batch_size, count, device_id);
      print_line_2();
    }
  }

  std::vector<Structure> structures_test;
  has_test_set = read_structures(false, para, structures_test);
  if (has_test_set) {
    test_set.resize(deviceCount);
    for (int device_id = 0; device_id < deviceCount; ++device_id) {
      print_line_1();
      printf("Constructing test_set in device  %d.\n", device_id);
      CHECK(gpuSetDevice(device_id));
      test_set[device_id].construct(para, structures_test, 0, structures_test.size(), device_id);
      print_line_2();
    }
  }

  int N = -1;
  int Nc = -1;
  int N_times_max_NN_radial = -1;
  int N_times_max_NN_angular = -1;
  max_NN_radial = -1;
  max_NN_angular = -1;
  max_NN_spin = 0;
  if (has_test_set) {
    N = test_set[0].N;
    Nc = test_set[0].Nc;
    N_times_max_NN_radial = test_set[0].N * test_set[0].max_NN_radial;
    N_times_max_NN_angular = test_set[0].N * test_set[0].max_NN_angular;
    max_NN_radial = test_set[0].max_NN_radial;
    max_NN_angular = test_set[0].max_NN_angular;
    max_NN_spin = test_set[0].max_NN_spin;
  }
  for (int n = 0; n < num_batches; ++n) {
    if (train_set[n][0].N > N) {
      N = train_set[n][0].N;
    };
    if (train_set[n][0].Nc > Nc) {
      Nc = train_set[n][0].Nc;
    };
    if (train_set[n][0].N * train_set[n][0].max_NN_radial > N_times_max_NN_radial) {
      N_times_max_NN_radial = train_set[n][0].N * train_set[n][0].max_NN_radial;
    };
    if (train_set[n][0].N * train_set[n][0].max_NN_angular > N_times_max_NN_angular) {
      N_times_max_NN_angular = train_set[n][0].N * train_set[n][0].max_NN_angular;
    };

    if (train_set[n][0].max_NN_radial > max_NN_radial) {
      max_NN_radial = train_set[n][0].max_NN_radial;
    }
    if (train_set[n][0].max_NN_angular > max_NN_angular) {
      max_NN_angular = train_set[n][0].max_NN_angular;
    }
    if (train_set[n][0].max_NN_spin > max_NN_spin) {
      max_NN_spin = train_set[n][0].max_NN_spin;
    }
  }

  if (para.train_mode == 1 || para.train_mode == 2) {
    potential.reset(new TNEP(para, N, para.version, deviceCount));
  } else {
    if (para.charge_vdw) {
      potential.reset(new NEP_Charge_VDW(para, N, Nc, para.version, deviceCount));
    } else if (para.charge_mode) {
      potential.reset(new NEP_Charge(para, N, Nc, para.version, deviceCount));
    } else if (para.spin_mode) {
      potential.reset(new NEP_Spin_Trainer(para, N, para.version, deviceCount));
    } else if (para.vdw) {
      potential.reset(new NEP_VDW(para, N, Nc, para.version, deviceCount));
    } else {
      potential.reset(new NEP(para, N, para.version, deviceCount));
    }
  }

  if (para.prediction == 0) {
    fid_loss_out = my_fopen("loss.out", "a");
  }
}

Fitness::~Fitness()
{
  if (fid_loss_out != NULL) {
    fclose(fid_loss_out);
  }
}

void Fitness::initialize_q_scaler(
  Parameters& para, const float* parameter_center)
{
  if (para.prediction != 0 || para.fine_tune || para.import_q_scaler) {
    return;
  }

  int deviceCount;
  CHECK(gpuGetDeviceCount(&deviceCount));
  std::vector<float> scaler_solution(
    para.number_of_variables * deviceCount, para.initial_para);
  if (parameter_center != nullptr) {
    for (int device_id = 0; device_id < deviceCount; ++device_id) {
      std::copy(
        parameter_center,
        parameter_center + para.number_of_variables,
        scaler_solution.begin() + device_id * para.number_of_variables);
    }
  }

  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    potential->find_force(
      para,
      scaler_solution.data(),
      train_set[batch_id],
      true,
      deviceCount);
  }

  if (!para.spin_mode) {
    return;
  }

  std::vector<float> global_max(para.dim, -1.0e10f);
  std::vector<float> global_min(para.dim, 1.0e10f);
  std::vector<float> device_max(para.dim);
  std::vector<float> device_min(para.dim);
  std::vector<double> global_square_sum(para.dim, 0.0);
  std::vector<unsigned long long> global_count(para.dim, 0);
  std::vector<float> device_square_sum(para.dim);
  std::vector<unsigned long long> device_count(para.dim);
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    para.q_scaler_max[device_id].copy_to_host(device_max.data());
    para.q_scaler_min[device_id].copy_to_host(device_min.data());
    para.q_scaler_square_sum[device_id].copy_to_host(device_square_sum.data());
    para.q_scaler_count[device_id].copy_to_host(device_count.data());
    for (int d = 0; d < para.dim; ++d) {
      global_max[d] = std::max(global_max[d], device_max[d]);
      global_min[d] = std::min(global_min[d], device_min[d]);
      global_square_sum[d] += device_square_sum[d];
      global_count[d] += device_count[d];
    }
  }

  int unresolved_channels = 0;
  for (int d = 0; d < para.dim; ++d) {
    const float range = global_max[d] - global_min[d];
    const float magnitude = std::max(
      1.0f, std::max(std::abs(global_max[d]), std::abs(global_min[d])));
    const float resolution =
      64.0f * std::numeric_limits<float>::epsilon() * magnitude;
    if (!std::isfinite(range)) {
      PRINT_INPUT_ERROR(
        "Cannot initialize Spin NEP q_scaler from a non-finite "
        "descriptor channel.\n");
    }
    if (d >= para.dim_struct) {
      const float rms = global_count[d] == 0
        ? 0.0f
        : static_cast<float>(std::sqrt(
            global_square_sum[d] / static_cast<double>(global_count[d])));
      if (!std::isfinite(rms)) {
        PRINT_INPUT_ERROR("Cannot initialize Spin3 q_scaler from non-finite RMS.\n");
      }
      para.q_scaler_cpu[d] = 1.0f / std::max(1.0f, std::max(range, rms));
      if (range <= resolution && rms <= resolution) {
        ++unresolved_channels;
      }
    } else {
      if (range <= 1.0e-10f) {
        PRINT_INPUT_ERROR(
          "Cannot initialize Spin NEP q_scaler from a constant "
          "descriptor channel.\n");
      }
      para.q_scaler_cpu[d] = 1.0f / range;
    }
  }
  if (unresolved_channels > 0) {
    printf(
      "Spin3 q_scaler kept at 1 for %d numerically unresolved "
      "descriptor channel(s).\n",
      unresolved_channels);
  }
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    para.q_scaler_gpu[device_id].copy_from_host(para.q_scaler_cpu.data());
  }
  CHECK(gpuSetDevice(0));
}

void Fitness::compute(
  const int generation, 
  Parameters& para, 
  const float* population,
  float* fitness_energy,
  float* fitness_force,
  float* fitness_virial,
  float* fitness_charge,
  float* fitness_bec,
  float* fitness_mforce,
  float* fitness_tau,
  float* fitness_spin_response)
{
  int deviceCount;
  CHECK(gpuGetDeviceCount(&deviceCount));
  int population_iter = (para.population_size - 1) / deviceCount + 1;

  {
    std::vector<std::vector<SpinResponsePoint>> response_points;
    if (para.lambda_spin_response > 0.0f) {
      response_points.resize(para.population_size);
    }
    int batch_id = generation % num_batches;
    for (int n = 0; n < population_iter; ++n) {
      const float* individual = population + deviceCount * n * para.number_of_variables;
      potential->find_force(para, individual, train_set[batch_id], false, deviceCount);
      for (int m = 0; m < deviceCount; ++m) {
        const int population_index = deviceCount * n + m;
        if (population_index >= para.population_size) {
          continue;
        }
        if (para.lambda_spin_response > 0.0f) {
          append_spin_response_points(
            m,
            train_set[batch_id][m],
            response_points[population_index]);
        }
        float energy_shift_per_structure_not_used;
        auto rmse_energy_array = train_set[batch_id][m].get_rmse_energy(
          para, energy_shift_per_structure_not_used, true, true, m);
        auto rmse_force_array = train_set[batch_id][m].get_rmse_force(para, true, m);
        auto rmse_virial_array = train_set[batch_id][m].get_rmse_virial(para, true, m);
        auto rmse_charge_array = train_set[batch_id][m].get_rmse_charge(para, m);
        auto rmse_bec_array = train_set[batch_id][m].get_rmse_bec(para, m);
        auto rmse_mforce_array =
          para.spin_mode
            ? train_set[batch_id][m].get_rmse_mforce(para, true, m)
            : std::vector<float>(para.num_types + 1, 0.0f);
        auto rmse_tau_array =
          para.spin_mode
            ? train_set[batch_id][m].get_rmse_tau(para, true, m)
            : std::vector<float>(para.num_types + 1, 0.0f);

        for (int t = 0; t <= para.num_types; ++t) {
          fitness_energy[deviceCount * n + m + t * para.population_size] =
            para.lambda_e * rmse_energy_array[t];
          fitness_force[deviceCount * n + m + t * para.population_size] =
            para.lambda_f * rmse_force_array[t];
          fitness_virial[deviceCount * n + m + t * para.population_size] =
            para.lambda_v * rmse_virial_array[t];
          fitness_charge[deviceCount * n + m + t * para.population_size] =
            para.lambda_q * rmse_charge_array[t];
          fitness_bec[deviceCount * n + m + t * para.population_size] =
            para.lambda_z * rmse_bec_array[t];
          fitness_mforce[deviceCount * n + m + t * para.population_size] =
            para.lambda_m * rmse_mforce_array[t];
          fitness_tau[deviceCount * n + m + t * para.population_size] =
            para.lambda_tau * rmse_tau_array[t];
        }
      }
    }
    for (int p = 0; p < para.population_size; ++p) {
      const float value = para.lambda_spin_response > 0.0f
        ? para.lambda_spin_response * spin_response_loss(response_points[p])
        : 0.0f;
      for (int t = 0; t <= para.num_types; ++t) {
        fitness_spin_response[p + t * para.population_size] = value;
      }
    }
  }
}

void Fitness::output(
  bool is_stress,
  int num_components,
  FILE* fid,
  float* prediction,
  float* reference,
  Dataset& dataset)
{
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    for (int n = 0; n < num_components; ++n) {
      int offset = n * dataset.N + dataset.Na_sum_cpu[nc];
      float data_nc = 0.0f;
      for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
        data_nc += prediction[offset + m];
      }
      if (!is_stress) {
        fprintf(fid, "%g ", data_nc / dataset.Na_cpu[nc]);
      } else {
        fprintf(fid, "%g ", data_nc / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION);
      }
    }
    for (int n = 0; n < num_components; ++n) {
      float ref_value = reference[n * dataset.Nc + nc];
      if (is_stress) {
        if (ref_value > -1e5) {
          ref_value *= dataset.Na_cpu[nc] / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION;
        }
      }
      if (n == num_components - 1) {
        fprintf(fid, "%g\n", ref_value);
      } else {
        fprintf(fid, "%g ", ref_value);
      }
    }
  }
}

void Fitness::output_atomic(
  int num_components,
  FILE* fid,
  float* prediction,
  float* reference,
  Dataset& dataset)
{
for (int nc = 0; nc < dataset.Nc; ++nc) {
  int offset = dataset.Na_sum_cpu[nc];
  for (int m = 0; m < dataset.structures[nc].num_atom; ++m) {
    for (int n = 0; n < num_components; ++n) {
      int index = n * dataset.N + offset + m;
      fprintf(fid, "%g ", prediction[index]);
    }
    for (int n = 0; n < num_components; ++n) {
      float ref_value = reference[n * dataset.N + offset + m];
      if (n == num_components - 1) {
        fprintf(fid, "%g\n", ref_value);
      } else {
        fprintf(fid, "%g ", ref_value);
      }
    }
  }
}
}

void Fitness::write_nep_txt(FILE* fid_nep, Parameters& para, float* elite)
{
  if (para.train_mode == 0) { // potential model
    if (!(para.charge_mode || para.charge_vdw)) {
      if (para.version == 4) {
        if (para.spin_mode) {
          if (para.enable_zbl) {
            fprintf(fid_nep, "nep4_spin3_zbl %d ", para.num_types);
          } else {
            fprintf(fid_nep, "nep4_spin3 %d ", para.num_types);
          }
        } else if (para.enable_zbl) {
          if (para.vdw) {
            fprintf(fid_nep, "nep4_zbl_vdw %d ", para.num_types);
          } else {
            fprintf(fid_nep, "nep4_zbl %d ", para.num_types);
          }
        } else {
          if (para.vdw) {
            fprintf(fid_nep, "nep4_vdw %d ", para.num_types);
          } else {
            fprintf(fid_nep, "nep4 %d ", para.num_types);
          }
        }
      } 
    } else {
      if (para.charge_vdw) {
        if (para.enable_zbl) {
          fprintf(fid_nep, "nep4_zbl_charge_vdw %d ", para.num_types);
        } else {
          fprintf(fid_nep, "nep4_charge_vdw %d ", para.num_types);
        }
      } else {
        if (para.enable_zbl) {
          fprintf(fid_nep, "nep4_zbl_charge%d %d ", para.charge_mode, para.num_types);
        } else {
          fprintf(fid_nep, "nep4_charge%d %d ", para.charge_mode, para.num_types);
        }
      }
    }
  } else if (para.train_mode == 1) { // dipole model
    if (para.version == 4) {
      fprintf(fid_nep, "nep4_dipole %d ", para.num_types);
    }
  } else if (para.train_mode == 2) { // polarizability model
    if (para.version == 4) {
      fprintf(fid_nep, "nep4_polarizability %d ", para.num_types);
    }
  } else if (para.train_mode == 3) { // temperature model
    if (para.version == 4) {
      if (para.enable_zbl) {
        fprintf(fid_nep, "nep4_zbl_temperature %d ", para.num_types);
      } else {
        fprintf(fid_nep, "nep4_temperature %d ", para.num_types);
      }
    }
  }

  for (int n = 0; n < para.num_types; ++n) {
    fprintf(fid_nep, "%s ", para.elements[n].c_str());
  }
  fprintf(fid_nep, "\n");
  if (para.spin_mode) {
    const int spin_header_lines =
      9 +
      static_cast<int>(para.is_spin_dof_type_set) +
      static_cast<int>(para.is_spin_env_type_set);
    fprintf(fid_nep, "spin_mode %d %d\n", para.spin_mode, spin_header_lines);
    fprintf(fid_nep, "spin_baseline");
    for (const float value : para.spin_baseline) {
      fprintf(fid_nep, " %.16e", static_cast<double>(value));
    }
    fprintf(fid_nep, "\n");
    fprintf(fid_nep, "spin_basis_size %d\n", para.spin_basis_size[0]);
    fprintf(fid_nep, "spin_l_max %d\n", para.spin_l_max[0]);
    fprintf(fid_nep, "spin_compress %d\n", para.spin_compress);
    fprintf(fid_nep, "spin_cutoff");
    const bool uniform_spin_cutoff = std::all_of(
      para.spin_cutoff_by_type.begin() + 1,
      para.spin_cutoff_by_type.end(),
      [&](float value) {
        return std::abs(value - para.spin_cutoff_by_type.front()) <= 1.0e-7f;
      });
    const int cutoff_count = uniform_spin_cutoff ? 1 : para.num_types;
    for (int type = 0; type < cutoff_count; ++type) {
      fprintf(fid_nep, " %.16e", static_cast<double>(para.spin_cutoff_by_type[type]));
    }
    fprintf(fid_nep, "\n");
    fprintf(fid_nep, "spin_order %d\n", para.spin_order);
    fprintf(fid_nep, "spin_soc %d\n", para.spin_soc);
    fprintf(
      fid_nep,
      "spin_projection_size %d\n",
      para.number_of_variables_spin_projection);
    fprintf(fid_nep, "spin_scaler 1\n");
    if (para.is_spin_dof_type_set) {
      fprintf(fid_nep, "spin_dof_type");
      for (int type = 0; type < para.num_types; ++type) {
        if (para.spin_dof_type_active[type]) {
          fprintf(fid_nep, " %s", para.elements[type].c_str());
        }
      }
      fprintf(fid_nep, "\n");
    }
    if (para.is_spin_env_type_set) {
      fprintf(fid_nep, "spin_env_type");
      for (int type = 0; type < para.num_types; ++type) {
        if (para.spin_env_type_active[type]) {
          fprintf(fid_nep, " %s", para.elements[type].c_str());
        }
      }
      fprintf(fid_nep, "\n");
    }
  }
  if (para.enable_zbl) {
    if (para.flexible_zbl) {
      fprintf(fid_nep, "zbl 0 0\n");
    } else if (para.use_typewise_cutoff_zbl) {
      fprintf(fid_nep, "zbl %g %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer, para.typewise_cutoff_zbl_factor);
    } else {
      fprintf(fid_nep, "zbl %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer);
    }
  }

  fprintf(fid_nep, "cutoff %g %g ", para.rc_radial[0], para.rc_angular[0]);
  if (para.has_multiple_cutoffs) {
    for (int n = 1; n < para.num_types; ++n) {
      fprintf(fid_nep, "%g %g ", para.rc_radial[n], para.rc_angular[n]);
    }
  }
  fprintf(
    fid_nep,
    "%d %d\n",
    para.spin_mode ? std::max(max_NN_radial, max_NN_spin) : max_NN_radial,
    max_NN_angular);

  fprintf(fid_nep, "n_max %d %d\n", para.n_max_radial, para.n_max_angular);
  fprintf(fid_nep, "basis_size %d %d\n", para.basis_size_radial, para.basis_size_angular);
  fprintf(fid_nep, "l_max %d %d %d ", para.L_max, (para.has_q_222 ? 2 : 0), para.has_q_1111);
  if (para.has_q_112 || para.has_q_123 || para.has_q_233 || para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_112);
  }
  if (para.has_q_123 || para.has_q_233 || para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_123);
  }
  if (para.has_q_233 || para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_233);
  }
  if (para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_134);
  }
  fprintf(fid_nep, "\n");

  if (para.num_hidden_layers == 2) {
    fprintf(fid_nep, "ANN %d %d\n", para.num_neurons1, para.num_neurons2);
  } else {
    fprintf(fid_nep, "ANN %d %d\n", para.num_neurons1, 0);
  }

  std::vector<float> parameters_file(elite, elite + para.number_of_variables);
  const int descriptor_offset = para.number_of_variables_ann * (para.train_mode == 2 ? 2 : 1);
#ifdef USE_CJ
  const int num_channels = para.num_types;
#else
  const int num_channels = para.num_types * para.num_types;
#endif
  descriptor_parameters_to_basis_major(
    parameters_file.data(),
    descriptor_offset,
    num_channels,
    para.n_max_radial,
    para.n_max_angular,
    para.basis_size_radial,
    para.basis_size_angular);
  for (int m = 0; m < para.number_of_variables; ++m) {
    fprintf(fid_nep, "%15.7e\n", parameters_file[m]);
  }
  CHECK(gpuSetDevice(0));
  para.q_scaler_gpu[0].copy_to_host(para.q_scaler_cpu.data());
  for (int d = 0; d < para.q_scaler_cpu.size(); ++d) {
    fprintf(fid_nep, "%15.7e\n", para.q_scaler_cpu[d]);
  }
  if (para.flexible_zbl) {
    for (int d = 0; d < 10 * (para.num_types * (para.num_types + 1) / 2); ++d) {
      fprintf(fid_nep, "%15.7e\n", para.zbl_para[d]);
    }
  }
}

void Fitness::get_save_potential_label(Parameters& para, const int generation, std::string& label) {
    if (para.save_potential_format == 1) {
      time_t rawtime;
      time(&rawtime);
      struct tm* timeinfo = localtime(&rawtime);
      char buffer[200];
      strftime(buffer, sizeof(buffer), "nep_y%Y_m%m_d%d_h%H_m%M_s%S_generation", timeinfo);
      label = std::string(buffer) + std::to_string(generation + 1);
    } else {
      label = "nep_gen" + std::to_string(generation + 1);
    }
}

void Fitness::report_error(
  Parameters& para,
  const int generation,
  const float loss_total,
  const float loss_L1,
  const float loss_L2,
  float* elite)
{
  if (0 == (generation + 1) % para.output_interval) {
    int batch_id = generation % num_batches;
    potential->find_force(para, elite, train_set[batch_id], false, 1);
    float energy_shift_per_structure;
    auto rmse_energy_train_array =
      train_set[batch_id][0].get_rmse_energy(para, energy_shift_per_structure, false, true, 0);
    auto rmse_force_train_array = train_set[batch_id][0].get_rmse_force(para, false, 0);
    auto rmse_virial_train_array = train_set[batch_id][0].get_rmse_virial(para, false, 0);
    auto rmse_charge_train_array = train_set[batch_id][0].get_rmse_charge(para, 0);
    auto rmse_bec_train_array = train_set[batch_id][0].get_rmse_bec(para, 0);
    auto rmse_mforce_train_array =
      para.spin_mode
        ? train_set[batch_id][0].get_rmse_mforce(para, false, 0)
        : std::vector<float>(para.num_types + 1, 0.0f);
    auto rmse_tau_train_array =
      para.spin_mode
        ? train_set[batch_id][0].get_rmse_tau(para, false, 0)
        : std::vector<float>(para.num_types + 1, 0.0f);

    float rmse_energy_train = rmse_energy_train_array.back();
    float rmse_force_train = rmse_force_train_array.back();
    float rmse_virial_train = rmse_virial_train_array.back();
    float rmse_charge_train = rmse_charge_train_array.back();
    float rmse_bec_train = rmse_bec_train_array.back();
    float rmse_mforce_train = rmse_mforce_train_array.back();
    float rmse_tau_train = rmse_tau_train_array.back();

    // correct the last bias parameter in the NN
    if (para.train_mode == 0 || para.train_mode == 3) {
      elite[para.number_of_variables_ann - 1] += energy_shift_per_structure;
    }

    float rmse_energy_test = 0.0f;
    float rmse_force_test = 0.0f;
    float rmse_virial_test = 0.0f;
    float rmse_charge_test = 0.0f;
    float rmse_bec_test = 0.0f;
    float rmse_mforce_test = 0.0f;
    float rmse_tau_test = 0.0f;
    if (has_test_set) {
      potential->find_force(para, elite, test_set, false, 1);
      float energy_shift_per_structure_not_used;
      auto rmse_energy_test_array =
        test_set[0].get_rmse_energy(para, energy_shift_per_structure_not_used, false, false, 0);
      auto rmse_force_test_array = test_set[0].get_rmse_force(para, false, 0);
      auto rmse_virial_test_array = test_set[0].get_rmse_virial(para, false, 0);
      auto rmse_charge_test_array = test_set[0].get_rmse_charge(para, 0);
      auto rmse_bec_test_array = test_set[0].get_rmse_bec(para, 0);
      auto rmse_mforce_test_array =
        para.spin_mode
          ? test_set[0].get_rmse_mforce(para, false, 0)
          : std::vector<float>(para.num_types + 1, 0.0f);
      auto rmse_tau_test_array =
        para.spin_mode
          ? test_set[0].get_rmse_tau(para, false, 0)
          : std::vector<float>(para.num_types + 1, 0.0f);
      rmse_energy_test = rmse_energy_test_array.back();
      rmse_force_test = rmse_force_test_array.back();
      rmse_virial_test = rmse_virial_test_array.back();
      rmse_charge_test = rmse_charge_test_array.back();
      rmse_bec_test = rmse_bec_test_array.back();
      rmse_mforce_test = rmse_mforce_test_array.back();
      rmse_tau_test = rmse_tau_test_array.back();
    }

    FILE* fid_nep = my_fopen("nep.txt", "w");
    write_nep_txt(fid_nep, para, elite);
    fclose(fid_nep);

    if (0 == (generation + 1) % para.save_potential) {
      std::string filename;
      get_save_potential_label(para, generation, filename);
      filename += ".txt";

      FILE* fid_nep = my_fopen(filename.c_str(), "w");
      write_nep_txt(fid_nep, para, elite);
      fclose(fid_nep);
    }

    if (para.train_mode == 0 || para.train_mode == 3) {
      if (!(para.charge_mode || para.charge_vdw)) {
        if (para.spin_mode) {
          printf(
            "%-8d%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f\n",
            generation + 1,
            loss_total,
            loss_L1,
            loss_L2,
            rmse_energy_train,
            rmse_force_train,
            rmse_virial_train,
            rmse_mforce_train,
            rmse_tau_train,
            rmse_energy_test,
            rmse_force_test,
            rmse_virial_test,
            rmse_mforce_test,
            rmse_tau_test);
          fprintf(
            fid_loss_out,
            "%-8d%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f\n",
            generation + 1,
            loss_total,
            loss_L1,
            loss_L2,
            rmse_energy_train,
            rmse_force_train,
            rmse_virial_train,
            rmse_mforce_train,
            rmse_tau_train,
            rmse_energy_test,
            rmse_force_test,
            rmse_virial_test,
            rmse_mforce_test,
            rmse_tau_test);
        } else {
          // NEP models
          printf(
            "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n",
            generation + 1,
            loss_total,
            loss_L1,
            loss_L2,
            rmse_energy_train,
            rmse_force_train,
            rmse_virial_train,
            rmse_energy_test,
            rmse_force_test,
            rmse_virial_test);
          fprintf(
            fid_loss_out,
            "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n",
            generation + 1,
            loss_total,
            loss_L1,
            loss_L2,
            rmse_energy_train,
            rmse_force_train,
            rmse_virial_train,
            rmse_energy_test,
            rmse_force_test,
            rmse_virial_test);
        }
      } else {
        // qNEP models:
        printf(
          "%-8d %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f\n",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_charge_train,
          rmse_bec_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test,
          rmse_charge_test,
          rmse_bec_test);
        fprintf(
          fid_loss_out,
          "%-8d %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f\n",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_charge_train,
          rmse_bec_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test,
          rmse_charge_test,
          rmse_bec_test);
      }
    } else {
      // TNEP models:
      printf(
        "%-8d %-11.5f %-11.5f %-11.5f %-13.5f %-13.5f\n",
        generation + 1,
        loss_total,
        loss_L1,
        loss_L2,
        rmse_virial_train,
        rmse_virial_test);
      fprintf(
        fid_loss_out,
        "%-8d %-11.5f %-11.5f %-11.5f %-13.5f %-13.5f\n",
        generation + 1,
        loss_total,
        loss_L1,
        loss_L2,
        rmse_virial_train,
        rmse_virial_test);
    }
    fflush(stdout);
    fflush(fid_loss_out);

    if (has_test_set) {
      if (para.train_mode == 0 || para.train_mode == 3) {
        FILE* fid_force = my_fopen("force_test.out", "w");
        FILE* fid_energy = my_fopen("energy_test.out", "w");
        FILE* fid_virial = my_fopen("virial_test.out", "w");
        FILE* fid_stress = my_fopen("stress_test.out", "w");
        update_energy_force_virial(fid_energy, fid_force, fid_virial, fid_stress, test_set[0]);
        fclose(fid_energy);
        fclose(fid_force);
        fclose(fid_virial);
        fclose(fid_stress);
        if ((para.charge_mode || para.charge_vdw)) {
          FILE* fid_charge = my_fopen("charge_test.out", "w");
          update_charge(fid_charge, test_set[0]);
          fclose(fid_charge);
          if (para.has_bec) {
            FILE* fid_bec = my_fopen("bec_test.out", "w");
            update_bec(fid_bec, test_set[0]);
            fclose(fid_bec);
          }
        } else if (para.spin_mode) {
          FILE* fid_mforce = my_fopen("mforce_test.out", "w");
          update_mforce(fid_mforce, test_set[0]);
          fclose(fid_mforce);
        }
      } else if (para.train_mode == 1) {
        FILE* fid_dipole = my_fopen("dipole_test.out", "w");
        update_dipole(fid_dipole, test_set[0], para.atomic_v);
        fclose(fid_dipole);
      } else if (para.train_mode == 2) {
        FILE* fid_polarizability = my_fopen("polarizability_test.out", "w");
        update_polarizability(fid_polarizability, test_set[0], para.atomic_v);
        fclose(fid_polarizability);
      }
    }
  }

  if (0 == (generation + 1) % 1000) {
    predict(para, elite);
  }
}

void Fitness::update_energy_force_virial(
  FILE* fid_energy, FILE* fid_force, FILE* fid_virial, FILE* fid_stress, Dataset& dataset)
{
  dataset.energy.copy_to_host(dataset.energy_cpu.data());
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  dataset.force.copy_to_host(dataset.force_cpu.data());

  for (int nc = 0; nc < dataset.Nc; ++nc) {
    int offset = dataset.Na_sum_cpu[nc];
    for (int m = 0; m < dataset.structures[nc].num_atom; ++m) {
      int n = offset + m;
      fprintf(
        fid_force,
        "%g %g %g %g %g %g\n",
        dataset.force_cpu[n],
        dataset.force_cpu[n + dataset.N],
        dataset.force_cpu[n + dataset.N * 2],
        dataset.force_ref_cpu[n],
        dataset.force_ref_cpu[n + dataset.N],
        dataset.force_ref_cpu[n + dataset.N * 2]);
    }
  }

  output(false, 1, fid_energy, dataset.energy_cpu.data(), dataset.energy_ref_cpu.data(), dataset);

  output(false, 6, fid_virial, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
  output(true, 6, fid_stress, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
}

void Fitness::update_mforce(FILE* fid_mforce, Dataset& dataset)
{
  dataset.mforce.copy_to_host(dataset.mforce_cpu.data());
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    if (!dataset.structures[nc].has_mforce) {
      continue;
    }
    const int offset = dataset.Na_sum_cpu[nc];
    for (int atom = 0; atom < dataset.Na_cpu[nc]; ++atom) {
      const int index = offset + atom;
      fprintf(
        fid_mforce,
        "%g %g %g %g %g %g\n",
        dataset.mforce_cpu[index],
        dataset.mforce_cpu[dataset.N + index],
        dataset.mforce_cpu[2 * dataset.N + index],
        dataset.mforce_ref_cpu[index],
        dataset.mforce_ref_cpu[dataset.N + index],
        dataset.mforce_ref_cpu[2 * dataset.N + index]);
    }
  }
}

void Fitness::update_charge(FILE* fid_charge, Dataset& dataset)
{
  dataset.charge.copy_to_host(dataset.charge_cpu.data());
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
      fprintf(fid_charge, "%g\n", dataset.charge_cpu[dataset.Na_sum_cpu[nc] + m]);
    }
  }
}

void Fitness::update_bec(FILE* fid_bec, Dataset& dataset)
{
  dataset.bec.copy_to_host(dataset.bec_cpu.data());
  output_atomic(9, fid_bec, dataset.bec_cpu.data(), dataset.bec_ref_cpu.data(), dataset);
}

void Fitness::update_dipole(FILE* fid_dipole, Dataset& dataset, bool atomic)
{
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  if (!atomic) {
    output(false, 3, fid_dipole, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
  } else {
    output_atomic(3, fid_dipole, dataset.virial_cpu.data(), dataset.avirial_ref_cpu.data(), dataset);
  }
}

void Fitness::update_polarizability(FILE* fid_polarizability, Dataset& dataset, bool atomic)
{
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  if (!atomic) {
    output(false, 6, fid_polarizability, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
  } else {
    output_atomic(6, fid_polarizability, dataset.virial_cpu.data(), dataset.avirial_ref_cpu.data(), dataset);
  }
}

void Fitness::predict(Parameters& para, float* elite)
{
  if (para.train_mode == 0 || para.train_mode == 3) {
    FILE* fid_force = my_fopen("force_train.out", "w");
    FILE* fid_energy = my_fopen("energy_train.out", "w");
    FILE* fid_virial = my_fopen("virial_train.out", "w");
    FILE* fid_stress = my_fopen("stress_train.out", "w");
    FILE* fid_charge = nullptr;
    FILE* fid_bec = nullptr;
    FILE* fid_mforce = nullptr;
    if ((para.charge_mode || para.charge_vdw)) {
      fid_charge = my_fopen("charge_train.out", "w");
      if (para.has_bec) {
        fid_bec = my_fopen("bec_train.out", "w");
      }
    } else if (para.spin_mode) {
      fid_mforce = my_fopen("mforce_train.out", "w");
    }
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, 1);
      update_energy_force_virial(
        fid_energy, fid_force, fid_virial, fid_stress, train_set[batch_id][0]);
      if ((para.charge_mode || para.charge_vdw)) {
        update_charge(fid_charge, train_set[batch_id][0]);
        if (para.has_bec) {
          update_bec(fid_bec, train_set[batch_id][0]);
        }
      } else if (para.spin_mode) {
        update_mforce(fid_mforce, train_set[batch_id][0]);
      }
    }
    fclose(fid_energy);
    fclose(fid_force);
    fclose(fid_virial);
    fclose(fid_stress);
    if ((para.charge_mode || para.charge_vdw)) {
      fclose(fid_charge);
      if (para.has_bec) {
        fclose(fid_bec);
      }
    } else if (para.spin_mode) {
      fclose(fid_mforce);
    }
  } else if (para.train_mode == 1) {
    FILE* fid_dipole = my_fopen("dipole_train.out", "w");
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, 1);
      update_dipole(fid_dipole, train_set[batch_id][0], para.atomic_v);
    }
    fclose(fid_dipole);
  } else if (para.train_mode == 2) {
    FILE* fid_polarizability = my_fopen("polarizability_train.out", "w");
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, 1);
      update_polarizability(fid_polarizability, train_set[batch_id][0], para.atomic_v);
    }
    fclose(fid_polarizability);
  }
}
