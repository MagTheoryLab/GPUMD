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

#include "spin_tspin.cuh"
#include "model/atom.cuh"
#include "nhc.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{
std::uint64_t splitmix64(std::uint64_t value)
{
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

double uniform_01(std::uint64_t& state)
{
  state = splitmix64(state);
  return (state >> 11) * (1.0 / 9007199254740992.0);
}

double gaussian(std::uint64_t seed, std::uint64_t atom_index, int component)
{
  std::uint64_t state = seed;
  state ^= atom_index * 0xd1b54a32d192ed03ULL;
  state ^=
    static_cast<std::uint64_t>(component) * 0x9e3779b97f4a7c15ULL;
  double u1 = 0.0;
  do {
    u1 = uniform_01(state);
  } while (u1 <= 0.0);
  const double u2 = uniform_01(state);
  return std::sqrt(-2.0 * std::log(u1)) *
    std::cos(2.0 * PI * u2);
}
} // namespace

static __global__ void gpu_scale_spin_velocity(
  const int number_of_atoms,
  const double factor,
  double* spin_velocity)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom < number_of_atoms) {
    spin_velocity[atom] *= factor;
    spin_velocity[number_of_atoms + atom] *= factor;
    spin_velocity[2 * number_of_atoms + atom] *= factor;
  }
}

static __global__ void gpu_update_spin_velocity(
  const int number_of_atoms,
  const double half_time_step,
  const double mass_factor,
  const double* mass,
  const double* mforce,
  double* spin_velocity)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom < number_of_atoms) {
    const double coefficient =
      half_time_step / (mass[atom] * mass_factor);
    spin_velocity[atom] += coefficient * mforce[atom];
    spin_velocity[number_of_atoms + atom] +=
      coefficient * mforce[number_of_atoms + atom];
    spin_velocity[2 * number_of_atoms + atom] +=
      coefficient * mforce[2 * number_of_atoms + atom];
  }
}

static __global__ void gpu_update_spin(
  const int number_of_atoms,
  const double time_step,
  const double* spin_velocity,
  double* spin)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom < number_of_atoms) {
    spin[atom] += time_step * spin_velocity[atom];
    spin[number_of_atoms + atom] +=
      time_step * spin_velocity[number_of_atoms + atom];
    spin[2 * number_of_atoms + atom] +=
      time_step * spin_velocity[2 * number_of_atoms + atom];
  }
}

static __global__ void gpu_find_spin_kinetic_energy(
  const int number_of_atoms,
  const int number_of_rounds,
  const double mass_factor,
  const double* mass,
  const double* spin_velocity,
  double* twice_kinetic_energy)
{
  const int thread = threadIdx.x;
  __shared__ double kinetic_energy[1024];
  kinetic_energy[thread] = 0.0;
  for (int round = 0; round < number_of_rounds; ++round) {
    const int atom = round * blockDim.x + thread;
    if (atom < number_of_atoms) {
      const double vx = spin_velocity[atom];
      const double vy = spin_velocity[number_of_atoms + atom];
      const double vz = spin_velocity[2 * number_of_atoms + atom];
      kinetic_energy[thread] +=
        mass[atom] * mass_factor * (vx * vx + vy * vy + vz * vz);
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (thread < offset) {
      kinetic_energy[thread] += kinetic_energy[thread + offset];
    }
    __syncthreads();
  }
  if (thread == 0) {
    twice_kinetic_energy[0] = kinetic_energy[0];
  }
}

Spin_TSPIN::Spin_TSPIN(
  double initial_temperature,
  double temperature_coupling,
  double time_step,
  double mass_factor,
  int seed,
  Atom& atom)
  : mass_factor_(mass_factor), twice_kinetic_energy_(1)
{
  const int number_of_atoms = atom.number_of_atoms;
  if (!atom.has_spin ||
      atom.spin_per_atom.size() != 3 * number_of_atoms ||
      atom.mforce_per_atom.size() != 3 * number_of_atoms ||
      atom.spin_velocity_per_atom.size() != 3 * number_of_atoms ||
      atom.cpu_mass.size() != number_of_atoms ||
      atom.cpu_type.size() != number_of_atoms) {
    PRINT_INPUT_ERROR("nvt_tspin requires complete spin, mforce, mass, and type arrays.");
  }
  if (!(initial_temperature > 0.0) || !std::isfinite(initial_temperature) ||
      !(temperature_coupling >= 1.0) || !std::isfinite(temperature_coupling) ||
      !(time_step > 0.0) || !std::isfinite(time_step) ||
      !(mass_factor_ > 0.0) || !std::isfinite(mass_factor_)) {
    PRINT_INPUT_ERROR("nvt_tspin requires finite positive integration parameters.");
  }
  if (atom.spin_velocity_initialized &&
      atom.cpu_spin_velocity_per_atom.size() != 3 * number_of_atoms) {
    PRINT_INPUT_ERROR("Initialized nvt_tspin velocities have an invalid size.");
  }

  pos_nhc_[0] = pos_nhc_[1] = pos_nhc_[2] = pos_nhc_[3] = 0.0;
  vel_nhc_[0] = vel_nhc_[2] = 1.0;
  vel_nhc_[1] = vel_nhc_[3] = -1.0;

  const double thermal_energy = K_B * initial_temperature;
  const double tau = time_step * temperature_coupling;
  for (int index = 0; index < 4; ++index) {
    mas_nhc_[index] = thermal_energy * tau * tau;
  }
  mas_nhc_[0] *= 3.0 * atom.number_of_atoms;

  initialize_spin_velocity(initial_temperature, seed, atom);
}

void Spin_TSPIN::initialize_spin_velocity(
  double initial_temperature,
  int seed,
  Atom& atom)
{
  if (atom.spin_velocity_initialized) {
    const int number_of_atoms = atom.number_of_atoms;
    for (int component = 0; component < 3; ++component) {
      for (int atom_index = 0; atom_index < number_of_atoms; ++atom_index) {
        const double value = atom.cpu_spin_velocity_per_atom[
          component * number_of_atoms + atom_index];
        if (!std::isfinite(value)) {
          PRINT_INPUT_ERROR("Initialized nvt_tspin velocities must be finite.");
        }
      }
    }
    printf("Reuse initialized spin velocities; nvt_tspin seed is not used.\n");
    return;
  }

  const int number_of_atoms = atom.number_of_atoms;
  const int number_of_types = atom.cpu_type_size.size();
  atom.cpu_spin_velocity_per_atom.assign(3 * number_of_atoms, 0.0);
  for (int atom_index = 0; atom_index < number_of_atoms; ++atom_index) {
    for (int component = 0; component < 3; ++component) {
      atom.cpu_spin_velocity_per_atom[
        component * number_of_atoms + atom_index] =
        0.5 * gaussian(
          static_cast<std::uint64_t>(seed),
          static_cast<std::uint64_t>(atom_index + 1),
          component);
    }
  }

  std::vector<double> twice_kinetic_energy(number_of_types, 0.0);
  for (int atom_index = 0; atom_index < number_of_atoms; ++atom_index) {
    const int type = atom.cpu_type[atom_index];
    const double mass = atom.cpu_mass[atom_index] * mass_factor_;
    if (!(mass > 0.0) || !std::isfinite(mass)) {
      PRINT_INPUT_ERROR("nvt_tspin requires finite positive spin masses.");
    }
    double velocity_squared = 0.0;
    for (int component = 0; component < 3; ++component) {
      const double velocity = atom.cpu_spin_velocity_per_atom[
        component * number_of_atoms + atom_index];
      velocity_squared += velocity * velocity;
    }
    twice_kinetic_energy[type] += mass * velocity_squared;
  }

  for (int type = 0; type < number_of_types; ++type) {
    if (atom.cpu_type_size[type] == 0) {
      continue;
    }
    if (!(twice_kinetic_energy[type] > 0.0) ||
        !std::isfinite(twice_kinetic_energy[type])) {
      PRINT_INPUT_ERROR("Failed to initialize finite nvt_tspin velocities.");
    }
    const double target =
      3.0 * atom.cpu_type_size[type] * K_B * initial_temperature;
    const double factor =
      std::sqrt(target / twice_kinetic_energy[type]);
    for (int atom_index = 0; atom_index < number_of_atoms; ++atom_index) {
      if (atom.cpu_type[atom_index] != type) {
        continue;
      }
      for (int component = 0; component < 3; ++component) {
        atom.cpu_spin_velocity_per_atom[
          component * number_of_atoms + atom_index] *= factor;
      }
    }
  }

  atom.spin_velocity_per_atom.copy_from_host(
    atom.cpu_spin_velocity_per_atom.data());
  atom.spin_velocity_initialized = true;
  printf("Initialized TSPIN velocities with seed %d.\n", seed);
}

double Spin_TSPIN::find_twice_kinetic_energy(const Atom& atom)
{
  const int number_of_atoms = atom.number_of_atoms;
  gpu_find_spin_kinetic_energy<<<1, 1024>>>(
    number_of_atoms,
    (number_of_atoms - 1) / 1024 + 1,
    mass_factor_,
    atom.mass.data(),
    atom.spin_velocity_per_atom.data(),
    twice_kinetic_energy_.data());
  GPU_CHECK_KERNEL
  double twice_kinetic_energy = 0.0;
  twice_kinetic_energy_.copy_to_host(&twice_kinetic_energy);
  return twice_kinetic_energy;
}

double Spin_TSPIN::find_nhc_factor(
  const double time_step,
  const Atom& atom)
{
  const double twice_kinetic_energy = find_twice_kinetic_energy(atom);
  const double thermal_energy = K_B * temperature;
  const double degrees_of_freedom = 3.0 * atom.number_of_atoms;
  const double factor = nhc(
    4,
    pos_nhc_,
    vel_nhc_,
    mas_nhc_,
    twice_kinetic_energy,
    thermal_energy,
    degrees_of_freedom,
    time_step * 0.5);
  if (!std::isfinite(factor)) {
    PRINT_INPUT_ERROR("nvt_tspin thermostat produced a non-finite scale factor.");
  }
  return factor;
}

void Spin_TSPIN::compute1(const double time_step, Atom& atom)
{
  const int number_of_atoms = atom.number_of_atoms;
  const int number_of_blocks = (number_of_atoms - 1) / 128 + 1;
  const double factor = find_nhc_factor(time_step, atom);
  gpu_scale_spin_velocity<<<number_of_blocks, 128>>>(
    number_of_atoms, factor, atom.spin_velocity_per_atom.data());
  gpu_update_spin_velocity<<<number_of_blocks, 128>>>(
    number_of_atoms,
    time_step * 0.5,
    mass_factor_,
    atom.mass.data(),
    atom.mforce_per_atom.data(),
    atom.spin_velocity_per_atom.data());
  gpu_update_spin<<<number_of_blocks, 128>>>(
    number_of_atoms,
    time_step,
    atom.spin_velocity_per_atom.data(),
    atom.spin_per_atom.data());
  GPU_CHECK_KERNEL
}

void Spin_TSPIN::compute2(const double time_step, Atom& atom)
{
  const int number_of_atoms = atom.number_of_atoms;
  const int number_of_blocks = (number_of_atoms - 1) / 128 + 1;
  gpu_update_spin_velocity<<<number_of_blocks, 128>>>(
    number_of_atoms,
    time_step * 0.5,
    mass_factor_,
    atom.mass.data(),
    atom.mforce_per_atom.data(),
    atom.spin_velocity_per_atom.data());
  GPU_CHECK_KERNEL
  const double factor = find_nhc_factor(time_step, atom);
  gpu_scale_spin_velocity<<<number_of_blocks, 128>>>(
    number_of_atoms, factor, atom.spin_velocity_per_atom.data());
  GPU_CHECK_KERNEL
}
