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

#pragma once

#include <cstdio>
#include <string>
#include <vector>

class Parameters;
class Structure;
class Dataset;

// Spin-only training policy; callers retain batch/device orchestration.
namespace fitness_spin {

bool prepare_checkpoint(Parameters& para);
void prepare_training_data(Parameters& para, std::vector<Structure>& structures, bool spin_restart);
void validate_batches(const Parameters& para, int num_batches);
void write_mforce(FILE* fid_mforce, Dataset& dataset);
void print_loss_header();
void write_loss(FILE* fid_loss_out, int generation, float loss_total, float loss_L1, float loss_L2,
  float rmse_energy_train, float rmse_force_train, float rmse_virial_train,
  float rmse_mforce_train, float rmse_tau_train, float rmse_energy_test,
  float rmse_force_test, float rmse_virial_test, float rmse_mforce_test, float rmse_tau_test);

class ResponseLoss {
public:
  void append(int device_id, Dataset& dataset);
  float value() const;

private:
  struct Point {
    std::string group;
    double coordinate;
    double prediction;
    double target;
  };
  std::vector<Point> points_;
};

void load_spin_checkpoint_metadata(Parameters& para);
void fit_spin_energy_baseline(const std::vector<Structure>& structures, Parameters& para);
void derive_spin_response_tangents(const Parameters& para, std::vector<Structure>& structures);
void finalize_q_scaler(Parameters& para, int deviceCount);
void write_checkpoint_metadata(FILE* fid_nep, const Parameters& para);

} // namespace fitness_spin
