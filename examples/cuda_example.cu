#include <cstdlib>
#include <iostream>

#include "treble/treble.hpp"

__global__ void saxpy(float a, float* x, float* y, size_t n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    y[i] = a * x[i] + y[i];
  }
  __syncthreads();
}

__host__ void kernel(float a, float* x, float* y, size_t n, int nbBlocks,
                     int threadsPerBlock) {
  std::cout << "Blocks: " << nbBlocks << " Threads: " << threadsPerBlock
            << std::endl;
  saxpy<<<nbBlocks, threadsPerBlock>>>(a, x, y, n);
  cudaDeviceSynchronize();
}

int main(int, char**) {
  // We start with one warp of threads per block, and a block number equal
  // to the number of streaming multiprocessors on  the default device.
  // We then optimize threads in units of warps and blocks in units of blocks/SM
  // with bounds determined by the queried device limits.
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  auto saxpy_with_self_tuning_launch_parameters = treble::st_back(
      kernel,
      treble::tunable_param{prop.multiProcessorCount, 1, prop.maxGridSize[0],
                            prop.multiProcessorCount},
      treble::tunable_param{prop.warpSize, 1, prop.maxThreadsDim[0],
                            prop.warpSize});

  float *x, *y;
  int n = 100000000;

  cudaMalloc(&x, n * sizeof(float));
  cudaMalloc(&y, n * sizeof(float));
  for (int i = 0; i < 10000; ++i) {
    saxpy_with_self_tuning_launch_parameters(static_cast<float>(i), x, y, n);
  }
  cudaFree(x);
  cudaFree(y);

  return EXIT_SUCCESS;
}