#include <cstdlib>
#include <iostream>

#define BENCHMARK_STATIC_DEFINE
#include "benchmark/benchmark.h"
#include "treble/treble.hpp"

static __global__ void saxpy(float a, float* x, float* y, size_t n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    y[i] = a * x[i] + y[i];
  }
  __syncthreads();
}

static __host__ void kernel(float a, float* x, float* y, size_t n, int nbBlocks,
                            int threadsPerBlock) {
  // std::cout << "Blocks: " << nbBlocks << " Threads: " << threadsPerBlock
  //           << std::endl;
  saxpy<<<nbBlocks, threadsPerBlock>>>(a, x, y, n);
  cudaDeviceSynchronize();
}

static void manual_near_optimal_parameters(benchmark::State& state) {
  float *x, *y;
  int n = 100000000;

  cudaMalloc(&x, n * sizeof(float));
  cudaMalloc(&y, n * sizeof(float));
  for (auto _ : state) {
    kernel(1., x, y, n, 8000, 1024);
  }

  cudaFree(x);
  cudaFree(y);
}
BENCHMARK(manual_near_optimal_parameters)->MinTime(10);

static void manual_suboptimal_parameters(benchmark::State& state) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  float *x, *y;
  int n = 100000000;

  cudaMalloc(&x, n * sizeof(float));
  cudaMalloc(&y, n * sizeof(float));
  for (auto _ : state) {
    kernel(1., x, y, n, 10 * prop.multiProcessorCount, 256);
  }

  cudaFree(x);
  cudaFree(y);
}
BENCHMARK(manual_suboptimal_parameters)->MinTime(10);

static void self_tuning(benchmark::State& state) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  auto stfunc = treble::self_tuning(
      kernel, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3, std::placeholders::_4,
      treble::tunable_param{prop.multiProcessorCount, 1, prop.maxGridSize[0],
                            prop.multiProcessorCount},
      treble::tunable_param{prop.warpSize, 1, prop.maxThreadsDim[0],
                            prop.warpSize});

  float *x, *y;
  int n = 100000000;

  cudaMalloc(&x, n * sizeof(float));
  cudaMalloc(&y, n * sizeof(float));

  for (auto _ : state) {
    stfunc(1., x, y, n);
  }

  cudaFree(x);
  cudaFree(y);
}
BENCHMARK(self_tuning)->MinTime(10);

static void self_tuning_good_initial_guess(benchmark::State& state) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  auto stfunc = treble::st_back(
      kernel,
      treble::tunable_param{10 * prop.multiProcessorCount, 1,
                            prop.maxGridSize[0], prop.multiProcessorCount},
      treble::tunable_param{256, 1, prop.maxThreadsDim[0], prop.warpSize});

  float *x, *y;
  int n = 100000000;

  cudaMalloc(&x, n * sizeof(float));
  cudaMalloc(&y, n * sizeof(float));
  for (auto _ : state) {
    stfunc(1., x, y, n);
  }
  cudaFree(x);
  cudaFree(y);
}
BENCHMARK(self_tuning_good_initial_guess)->MinTime(10);

BENCHMARK_MAIN();
