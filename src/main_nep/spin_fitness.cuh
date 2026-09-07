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
namespace spin_fitness {

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

} // namespace spin_fitness
