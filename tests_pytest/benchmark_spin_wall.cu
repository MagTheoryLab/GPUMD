/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/
#include "utilities/nep_io/spin_wall.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t error)
{
  if (error != cudaSuccess) {
    fprintf(stderr, "%s\n", cudaGetErrorString(error));
    std::exit(1);
  }
}

static float median(std::vector<float>& values)
{
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

template <typename T>
static T* device_copy(const std::vector<T>& values)
{
  T* result;
  check(cudaMalloc(&result, values.size() * sizeof(T)));
  check(cudaMemcpy(result, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  return result;
}

int main()
{
  cudaDeviceProp properties;
  check(cudaGetDeviceProperties(&properties, 0));
  printf("GPU,%s\n", properties.name);
  printf("atoms,median_us,p10_us,p90_us,Matom_per_s\n");
  for (const int n : {4096, 65536, 108000, 1048576}) {
    std::vector<int> types(n);
    std::vector<double> spin(3 * size_t(n)), output(4 * size_t(n), 0.0);
    for (int i = 0; i < n; ++i) {
      types[i] = i % 4;
      spin[i] = (i % 3 == 0 ? 2.5 : i % 3 == 1 ? 0.2 : 0.0);
      spin[n + i] = (i % 7) * 0.1;
      spin[2 * n + i] = (i % 11) * 0.03;
    }
    std::vector<Spin_Walls> host_walls(4);
    for (auto& wall : host_walls) {
      wall.side[Spin_Walls::high] = {1.0, 0.45};
      wall.side[Spin_Walls::low] = {0.5, 0.8};
    }
    int* d_type = device_copy(types);
    double* d_spin = device_copy(spin);
    double *d_energy, *d_mforce;
    check(cudaMalloc(&d_energy, n * sizeof(double)));
    check(cudaMalloc(&d_mforce, 3 * size_t(n) * sizeof(double)));
    Spin_Walls* d_walls = device_copy(host_walls);
    const int block = 128, grid = (n + block - 1) / block;
    for (int i = 0; i < 30; ++i)
      add_spin_wall<<<grid, block>>>(n, d_type, d_spin, d_walls, d_energy, d_mforce);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    std::vector<float> samples;
    for (int repeat = 0; repeat < 9; ++repeat) {
      check(cudaEventRecord(start));
      for (int i = 0; i < 250; ++i)
        add_spin_wall<<<grid, block>>>(n, d_type, d_spin, d_walls, d_energy, d_mforce);
      check(cudaGetLastError());
      check(cudaEventRecord(stop));
      check(cudaEventSynchronize(stop));
      float elapsed;
      check(cudaEventElapsedTime(&elapsed, start, stop));
      samples.push_back(elapsed * 1000.0f / 250.0f);
    }
    std::sort(samples.begin(), samples.end());
    const float med = median(samples);
    printf("%d,%.3f,%.3f,%.3f,%.3f\n", n, med, samples[1], samples[7], n / med);
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(d_type));
    check(cudaFree(d_spin));
    check(cudaFree(d_energy));
    check(cudaFree(d_mforce));
    check(cudaFree(d_walls));
  }
}
