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

#include <random>
#include <vector>

class Parameters;

// Preserve RNG consumption and parameter ordering across fresh/restart paths.
namespace spin_snes {
void initialize_search(const Parameters& para, std::mt19937& rng,
  std::vector<float>& mu, std::vector<float>& sigma);
void initialize_curriculum(const Parameters& para, std::vector<float>& mu);
void mark_curriculum(const Parameters& para, std::vector<int>& curriculum_parameter);
void assign_variable_types(const Parameters& para, int offset, std::vector<int>& type_of_variable);
} // namespace spin_snes
