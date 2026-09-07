// GPU-node interface checks for the host-side Spin3 training policies.
// Run in a directory containing a valid Spin3 nep.in (C1/C2/C3/C9).
#include "main_nep/parameters.cuh"
#include "main_nep/dataset.cuh"
#include "main_nep/spin_fitness.cuh"
#include "main_nep/spin_snes.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

int main()
{
  Parameters para;
  std::mt19937 rng(12345678), same_rng(12345678);
  std::vector<float> mu(para.number_of_variables), sigma(mu.size());
  auto same_mu = mu;
  auto same_sigma = sigma;
  spin_snes::initialize_search(para, rng, mu, sigma);
  spin_snes::initialize_search(para, same_rng, same_mu, same_sigma);
  assert(mu == same_mu && sigma == same_sigma && rng == same_rng);
  for (std::size_t i = 0; i < mu.size(); ++i) {
    assert(std::isfinite(mu[i]) && std::isfinite(sigma[i]) && sigma[i] > 0);
  }
  const int spin_offset = para.number_of_variables_ann +
    para.number_of_variables_descriptor - para.number_of_variables_descriptor_spin -
    para.number_of_variables_spin_projection;
  for (int i = 0; i < para.number_of_variables_descriptor_spin; ++i) {
    assert(sigma[spin_offset + i] == para.sigma0 * 0.1f);
  }
  std::vector<int> types(mu.size(), para.num_types);
  spin_snes::assign_variable_types(para, spin_offset, types);
  for (int i = 0; i < para.number_of_variables_descriptor_spin; ++i) {
    assert(types[spin_offset + i] == (i % (para.num_types * para.num_types)) / para.num_types);
  }
  assert(std::all_of(types.begin(), types.begin() + spin_offset,
    [&](int type) { return type == para.num_types; }));
  if (para.spin_order == 3) {
    std::vector<int> mask(mu.size(), 0);
    spin_snes::mark_curriculum(para, mask);
    const int expected = para.num_types * para.num_neurons1 *
      (para.dim - para.spin_order3_descriptor_start);
    assert(std::count(mask.begin(), mask.end(), 1) == expected);
    spin_snes::initialize_curriculum(para, mu);
    for (std::size_t i = 0; i < mu.size(); ++i) {
      assert(mu[i] == (mask[i] ? 0.0f : same_mu[i]));
    }
  }

  std::vector<Structure> structures(2);
  for (int t = 0; t < 2; ++t) {
    structures[t].num_atom = 1;
    structures[t].type = {t};
    structures[t].energy = -2.0f - t;
  }
  spin_fitness::fit_spin_energy_baseline(structures, para);
  assert(para.spin_baseline[0] == -2.0f && para.spin_baseline[1] == -3.0f);

  Dataset dataset;
  dataset.Nc = dataset.N = 3;
  dataset.Na_sum_cpu = {0, 1, 2};
  dataset.structures.resize(3);
  dataset.mforce_cpu.resize(9);
  dataset.mforce.resize(9);
  std::vector<float> mf(9, 0.0f);
  for (int i = 0; i < 3; ++i) {
    auto& frame = dataset.structures[i];
    frame.num_atom = 1;
    frame.has_spin_response = 1;
    frame.spin_response_group = "rotation";
    frame.spin_response_coordinate = i - 1;
    frame.spin_tangent_x = {1};
    frame.spin_tangent_y = frame.spin_tangent_z = {0};
    frame.mfx = {float(i - 1)};
    frame.mfy = frame.mfz = {0};
    mf[i] = i - 1;
  }
  dataset.mforce.copy_from_host(mf.data());
  spin_fitness::ResponseLoss exact;
  assert(exact.value() == 0);
  exact.append(0, dataset);
  assert(exact.value() == 0);
  for (int i = 0; i < 3; ++i) mf[i] += 1;
  dataset.mforce.copy_from_host(mf.data());
  spin_fitness::ResponseLoss shifted;
  shifted.append(0, dataset);
  const double expected = 0.25 * (1.0 / std::sqrt(2.0 / 3.0) - 0.5);
  assert(std::abs(shifted.value() - expected) < 1.0e-7);
  puts("spin training policy interface checks passed");
}
