/* This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
// Standalone sum/max/invalid-state oracle for both stages of the FIRE statistics.
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include "minimize/spin_fire_stats.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>

#define FIRE_CHECK(call) do { auto e = (call); if (e != gpuSuccess) { \
  std::fprintf(stderr, "%s\n", gpuGetErrorString(e)); std::exit(1); } } while (0)

void check(int n, int invalid_plane = -1, bool infinite = false) {
  using namespace spin_fire_stats;
  const int blocks = std::max(1, std::min((n+BLOCK-1)/BLOCK, 320));
  double *data, *partial, *out;
  FIRE_CHECK(gpuMallocManaged(&data, 31*std::max(1,n)*sizeof(double)));
  FIRE_CHECK(gpuMallocManaged(&partial, 15*blocks*sizeof(double)));
  FIRE_CHECK(gpuMallocManaged(&out, 44*sizeof(double)));
  double *spin=data, *force=data+3*n, *gradient=data+6*n, *velocity=data+9*n;
  double *previous_spin=data+12*n, *position=data+15*n, *previous=data+18*n;
  double *virial=data+21*n, *energy=data+30*n;
  for (int i=0; i<n; ++i) {
    energy[i]=-6.25+1e-5*(i%11-5);
    for (int c=0; c<3; ++c) {
      const int j=c*n+i;
      spin[j]=.7*std::sin(.13*i+c);
      gradient[j]=.3*std::cos(.17*i+c);
      force[j]=.6*std::sin(.19*i+c);
      velocity[j]=(i%2 ? -1 : 1)*.25*gradient[j];
      previous_spin[j]=spin[j]+1e-4*(c+1);
      position[j]=10+.01*i+c;
      previous[j]=position[j]-1e-5*(c+1);
    }
    for (int c=0; c<9; ++c) virial[c*n+i]=(c+1)*.1*std::sin(.23*i);
  }
  if (invalid_plane>=0) data[invalid_plane*n+n-1]=infinite ? std::numeric_limits<double>::infinity()
    : std::numeric_limits<double>::quiet_NaN();
  long double ref[22]={};
  for (int i=0; i<n; ++i) {
    ref[0]+=energy[i];
    if (!std::isfinite(energy[i])) ref[5]=1;
    long double gi2=0;
    for (int c=0; c<3; ++c) {
      const int j=c*n+i;
      const long double g=gradient[j], v=velocity[j], f=force[j];
      gi2+=g*g; ref[2]+=v*v; ref[3]+=g*v;
      if (!std::isfinite(spin[j]) || !std::isfinite(force[j]) ||
          !std::isfinite(g) || !std::isfinite(v)) ref[5]=1;
      ref[6]=std::fmax(ref[6],std::fabs((long double)spin[j]-previous_spin[j])/
        std::fmax(1.L,std::fmax(std::fabs((long double)spin[j]),std::fabs((long double)previous_spin[j]))));
      ref[7]+=f*f; ref[8]+=v*v; ref[9]+=f*v;
      ref[10]=std::fmax(ref[10],std::fabs(f));
      if (!std::isfinite(f) || !std::isfinite(v) || !std::isfinite(position[j])) ref[11]=1;
      ref[12]=std::fmax(ref[12],std::fabs((long double)position[j]-previous[j])/
        std::fmax(1.L,std::fmax(std::fabs((long double)position[j]),std::fabs((long double)previous[j]))));
    }
    ref[1]+=gi2; ref[4]=std::fmax(ref[4],gi2);
    for (int c=0; c<9; ++c) {
      ref[13+c]+=virial[c*n+i];
      if (!std::isfinite(virial[c*n+i])) ref[11]=1;
    }
  }
  collect_stats<<<1,BLOCK>>>(n,spin,force,energy,gradient,velocity,previous_spin,out);
  collect_lattice_stats<<<1,BLOCK>>>(n,position,previous,force,velocity,virial,out+7);
  collect_stats<<<blocks,BLOCK>>>(n,spin,force,energy,gradient,velocity,previous_spin,partial);
  reduce_partials<7,4,7><<<1,BLOCK>>>(blocks,partial,out+22);
  collect_lattice_stats<<<blocks,BLOCK>>>(n,position,previous,force,velocity,virial,partial);
  reduce_partials<15,3,6><<<1,BLOCK>>>(blocks,partial,out+29);
  FIRE_CHECK(gpuGetLastError()); FIRE_CHECK(gpuDeviceSynchronize());
  for (int version=0; version<2; ++version)
    for (int k=0; k<22; ++k) {
      const double got=out[version*22+k];
      bool pass=std::isnan(ref[k]) ? std::isnan(got) : std::isinf(ref[k]) ? got==ref[k]
        : std::fabs(got-ref[k])<=1e-10L+3e-12L*std::fabs(ref[k]);
      if (k==5 || k==11) pass=got==ref[k];
      if (!pass) {
        std::fprintf(stderr,"FAIL n=%d plane=%d version=%d field=%d ref=%.18Lg got=%.18g\n",
          n,invalid_plane,version,k,ref[k],got);
        std::exit(2);
      }
    }
  FIRE_CHECK(gpuFree(data)); FIRE_CHECK(gpuFree(partial)); FIRE_CHECK(gpuFree(out));
  std::printf("PASS n=%d blocks=%d invalid_plane=%d\n",n,blocks,invalid_plane);
}

int main() {
  for (int n : {0,1,255,256,257,4095,4096,4097,100013}) check(n);
  for (int plane : {0,3,6,9,15,21,30}) check(257,plane);
  check(257,6,true); check(257,30,true);
}
