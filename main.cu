#include <cuComplex.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <iomanip>
#include <iostream>

using namespace std;

double dtime() {
  double tseconds = 0;
  struct timeval t;
  gettimeofday(&t, NULL);
  tseconds = (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
  return tseconds;
}

#define GPU_ERROR(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    cerr << "GPUassert: \"" << cudaGetErrorString(code) << "\"  in " << file
         << ": " << line << "\n";
    if (abort) exit(code);
  }
}

template <typename T>
__global__ void initKernel(T* data, size_t data_len) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int idx = tidx; idx < data_len; idx += gridDim.x * blockDim.x) {
    data[idx] = idx;
  }
}

template <typename T, int N, int M, int BLOCKSIZE>
__global__ void testfun(T* dA, T* dB, T* dC) {
  T* sA = dA + threadIdx.x + blockIdx.x * BLOCKSIZE * M;
  T* sB = dB + threadIdx.x + blockIdx.x * BLOCKSIZE * M;

  T sum = 0;

#pragma unroll 1
  for (int i = 0; i < M; i++) {
    T a = sA[i * BLOCKSIZE];
    T b = sB[i * BLOCKSIZE];
    T v = a - b;
    for (int i = 0; i < N; i++) {
      v = v * a - b;
    }
    sum += v;
  }
  if (threadIdx.x == 0) dC[blockIdx.x] = sum;
}

int main(int argc, char** argv) {
  typedef double dtype;
  const int M = 4000;
  const int N = PARN;
  const int BLOCKSIZE = 256;

  int deviceUsed;
  cudaGetDevice(&deviceUsed);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceUsed);
  int numBlocks;

  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, testfun<dtype, N, M, BLOCKSIZE>, BLOCKSIZE, 0));
  int blockCount = prop.multiProcessorCount * numBlocks;

  size_t data_len = (size_t)blockCount * BLOCKSIZE * M;
  dtype* dA = NULL;
  dtype* dB = NULL;
  dtype* dC = NULL;
  size_t iters = 10000;

  GPU_ERROR(cudaMalloc(&dA, data_len * sizeof(dtype)));
  GPU_ERROR(cudaMalloc(&dB, data_len * sizeof(dtype)));
  GPU_ERROR(cudaMalloc(&dC, data_len * sizeof(dtype)));
  initKernel<<<blockCount, 256>>>(dA, data_len);
  initKernel<<<blockCount, 256>>>(dB, data_len);
  initKernel<<<blockCount, 256>>>(dC, data_len);
  testfun<dtype, N, M, BLOCKSIZE><<<blockCount, BLOCKSIZE>>>(dA, dB, dC);
  cudaDeviceSynchronize();

  double start = dtime();
  for (size_t iter = 0; iter < iters; iter++) {
    testfun<dtype, N, M, BLOCKSIZE><<<blockCount, BLOCKSIZE>>>(dA, dB, dC);
  }
  cudaDeviceSynchronize();
  double end = dtime();
  GPU_ERROR(cudaGetLastError());

  cout << setw(3) << N << " " << setprecision(4) << setw(5)
       << iters * 2 * data_len * sizeof(dtype) / (end - start) * 1.0e-9 << "  "
       << setw(6) << iters * (2 + N * 2) * data_len / (end - start) * 1.0e-9
       << "\n";

  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
  GPU_ERROR(cudaFree(dC));
}
