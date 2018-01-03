/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_
#define TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <algorithm>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

// Usage of GetGpuLaunchConfig, GetGpu2DLaunchConfig, and
// GetGpu3DLaunchConfig:
//
// There are two versions of GetGpuLaunchConfig and GetGpu2DLaunchConfig, one
// version uses heuristics without any knowledge of the device kernel, the other
// version uses cudaOccupancyMaxPotentialBlockSize to determine the theoretical
// launch parameters that maximize occupancy. Currently, only the maximum
// occupancy version of GetGpu3DLaunchConfig is available.
//
// For large number of work elements, the convention is that each kernel would
// iterate through its assigned range. The return value of GetGpuLaunchConfig
// is struct GpuLaunchConfig, which contains all the information needed for the
// kernel launch, including: virtual number of threads, the number of threads
// per block and number of threads per block used inside a kernel launch.
// GetGpu2DLaunchConfig and GetGpu3DLaunchConfig does the same thing as
// GpuLaunchConfig. The only difference is the dimension. The macros
// GPU_1D_KERNEL_LOOP and GPU_AXIS_KERNEL_LOOP might be used to do inner loop.
//
/* Sample code:

__global__ void MyKernel1D(GpuLaunchConfig config, other_args...) {
  GPU_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    do_your_job_here;
  }
}

__global__ void MyKernel2D(Gpu2DLaunchConfig config, other_args...) {
  GPU_AXIS_KERNEL_LOOP(x, config.virtual_thread_count, x) {
    GPU_AXIS_KERNEL_LOOP(y, config.virtual_thread_count, y) {
      do_your_job_here;
    }
  }
}

__global__ void MyKernel3D(Gpu3DLaunchConfig config, other_args...) {
  GPU_AXIS_KERNEL_LOOP(x, config.virtual_thread_count, x) {
    GPU_AXIS_KERNEL_LOOP(y, config.virtual_thread_count, y) {
      GPU_AXIS_KERNEL_LOOP(z, config.virtual_thread_count, z) {
        do_your_job_here;
      }
    }
  }
}

void MyDriverFunc(const GPUDevice &d) {
  // use heuristics
  GpuLaunchConfig cfg1 = GetGpuLaunchConfig(10240, d);
  MyKernel1D <<<config.block_count,
                config.thread_per_block, 0, d.stream()>>> (cfg1, other_args...);
  Gpu2DLaunchConfig cfg2 = GetGpu2DLaunchConfig(10240, 10240, d);
  MyKernel2D <<<config.block_count,
                config.thread_per_block, 0, d.stream()>>> (cfg2, other_args...);
  Gpu3DLaunchConfig cfg3 = GetGpu3DLaunchConfig(4096, 4096, 100, d);
  MyKernel3D <<<config.block_count,
                config.thread_per_block, 0, d.stream()>>> (cfg3, other_args...);

  // maximize occupancy
  GpuLaunchConfig cfg4 = GetGpuLaunchConfig(10240, d, MyKernel1D, 0, 0 );
  MyKernel1D <<<config.block_count,
                config.thread_per_block, 0, d.stream()>>> (cfg4, other_args...);
  Gpu2DLaunchConfig cfg5 = GetGpu2DLaunchConfig(10240, 10240, d,
                                                  MyKernel1D, 0, 0);
  MyKernel2D <<<config.block_count,
                config.thread_per_block, 0, d.stream()>>> (cfg5, other_args...);
  Gpu3DLaunchConfig cfg6 = GetGpu3DLaunchConfig(4096, 4096, 100, d,
                                                  MyKernel1D, 0, 0);
  MyKernel3D <<<config.block_count,
                config.thread_per_block, 0, d.stream()>>> (cfg6, other_args...);
}

// See the test for this for more example:
//
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/gpu_kernel_helper_test.cu.cc

*/

#define GPU_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define GPU_AXIS_KERNEL_LOOP(i, n, axis)                                  \
  for (int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n.axis; \
       i += blockDim.axis * gridDim.axis)

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

struct GpuLaunchConfig {
  // Logical number of thread that works on the elements. If each logical
  // thread works on exactly a single element, this is the same as the working
  // element count.
  int virtual_thread_count = -1;
  // Number of threads per block.
  int thread_per_block = -1;
  // Number of blocks for Cuda kernel launch.
  int block_count = -1;
};

// Calculate the launch config we should use for a kernel launch.
// This is assuming the kernel is quite simple and will largely be
// memory-limited.
inline GpuLaunchConfig GetGpuLaunchConfig(int work_element_count,
                                            const GPUDevice& d) {
  GpuLaunchConfig config;

  // in case of invalid input, return the default value config, which has all -1
  if (work_element_count <= 0) {
    return config;
  }

  const int virtual_thread_count = work_element_count;

#if GOOGLE_CUDA
  const int physical_thread_count = std::min(
      d.getNumCudaMultiProcessors() * d.maxCudaThreadsPerMultiProcessor(),
      virtual_thread_count);
  const int thread_per_block = std::min(1024, d.maxCudaThreadsPerBlock());
  const int block_count =
      std::min(DIV_UP(physical_thread_count, thread_per_block),
               d.getNumCudaMultiProcessors());
#elif TENSORFLOW_USE_ROCM
  const int physical_thread_count = std::min(
      d.getNumHipMultiProcessors() * d.maxHipThreadsPerMultiProcessor(),
      virtual_thread_count);
  const int thread_per_block = std::min(1024, d.maxHipThreadsPerBlock());
  const int block_count =
      std::min(DIV_UP(physical_thread_count, thread_per_block),
               d.getNumHipMultiProcessors());
#endif

  config.virtual_thread_count = virtual_thread_count;
  config.thread_per_block = thread_per_block;
  config.block_count = block_count;
  return config;
}

// Calculate the launch config we should use for a kernel launch. This
// variant takes the resource limits of func into account to maximize occupancy.
template <typename DeviceFunc>
inline GpuLaunchConfig GetGpuLaunchConfig(int work_element_count,
                                            const GPUDevice& d, DeviceFunc func,
                                            size_t dynamic_shared_memory_size,
                                            int block_size_limit) {
  GpuLaunchConfig config;

  if (work_element_count <= 0) {
    return config;
  }

  int block_count = 0;
  int thread_per_block = 0;

#if GOOGLE_CUDA
  cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
      &block_count, &thread_per_block, func, dynamic_shared_memory_size,
      block_size_limit);
  CHECK_EQ(err, cudaSuccess);
#elif TENSORFLOW_USE_ROCM
  // XXX FIXME re-enable this after hipOccupancyMaxPotentialBlockSize is
  // implemented
  //hipError_t err = hipOccupancyMaxPotentialBlockSize(
  //    &block_count, &thread_per_block, func, dynamic_shared_memory_size,
  //    block_size_limit);
  //CHECK_EQ(err, hipSuccess);

  const int physical_thread_count = std::min(
      d.getNumHipMultiProcessors() * d.maxHipThreadsPerMultiProcessor(),
      work_element_count);
  thread_per_block = std::min(1024, d.maxHipThreadsPerBlock());
  block_count =
      std::min(DIV_UP(physical_thread_count, thread_per_block),
               d.getNumHipMultiProcessors());
#endif

  block_count =
      std::min(block_count, DIV_UP(work_element_count, thread_per_block));

  config.virtual_thread_count = work_element_count;
  config.thread_per_block = thread_per_block;
  config.block_count = block_count;
  return config;
}

struct Gpu2DLaunchConfig {
  dim3 virtual_thread_count = dim3(0, 0, 0);
  dim3 thread_per_block = dim3(0, 0, 0);
  dim3 block_count = dim3(0, 0, 0);
};

inline Gpu2DLaunchConfig GetGpu2DLaunchConfig(int xdim, int ydim,
                                                const GPUDevice& d) {
  Gpu2DLaunchConfig config;

  if (xdim <= 0 || ydim <= 0) {
    return config;
  }

  const int kThreadsPerBlock = 256;
  int block_cols = std::min(xdim, kThreadsPerBlock);
  // ok to round down here and just do more loops in the kernel
  int block_rows = std::max(kThreadsPerBlock / block_cols, 1);

#if GOOGLE_CUDA
  const int physical_thread_count =
      d.getNumCudaMultiProcessors() * d.maxCudaThreadsPerMultiProcessor();
#elif TENSORFLOW_USE_ROCM
  const int physical_thread_count =
      d.getNumHipMultiProcessors() * d.maxHipThreadsPerMultiProcessor();
#endif

  const int max_blocks = std::max(physical_thread_count / kThreadsPerBlock, 1);

  config.virtual_thread_count = dim3(xdim, ydim, 1);
  config.thread_per_block = dim3(block_cols, block_rows, 1);

  int grid_x = std::min(DIV_UP(xdim, block_cols), max_blocks);

  config.block_count = dim3(
      grid_x, std::min(max_blocks / grid_x, std::max(ydim / block_rows, 1)), 1);
  return config;
}

// Calculate the Cuda 2D and 3D launch config we should use for a kernel launch.
// This variant takes the resource limits of func into account to maximize
// occupancy.
using Gpu3DLaunchConfig = Gpu2DLaunchConfig;

template <typename DeviceFunc>
inline Gpu3DLaunchConfig GetGpu3DLaunchConfig(
    int xdim, int ydim, int zdim, const GPUDevice& d, DeviceFunc func,
    size_t dynamic_shared_memory_size, int block_size_limit) {
  Gpu3DLaunchConfig config;

  if (xdim <= 0 || ydim <= 0 || zdim <= 0) {
    return config;
  }

  int dev;

#if GOOGLE_CUDA
  cudaGetDevice(&dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
#elif TENSORFLOW_USE_ROCM
  hipGetDevice(&dev);
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, dev);
#endif

  int xthreadlimit = deviceProp.maxThreadsDim[0];
  int ythreadlimit = deviceProp.maxThreadsDim[1];
  int zthreadlimit = deviceProp.maxThreadsDim[2];
  int xgridlimit = deviceProp.maxGridSize[0];
  int ygridlimit = deviceProp.maxGridSize[1];
  int zgridlimit = deviceProp.maxGridSize[2];

  int block_count = 0;
  int thread_per_block = 0;

#if GOOGLE_CUDA
  cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
      &block_count, &thread_per_block, func, dynamic_shared_memory_size,
      block_size_limit);
  CHECK_EQ(err, cudaSuccess);
#elif TENSORFLOW_USE_ROCM
  // XXX FIXME re-enable this after hipOccupancyMaxPotentialBlockSize is
  // implemented
  //hipError_t err = hipOccupancyMaxPotentialBlockSize(
  //    &block_count, &thread_per_block, func, dynamic_shared_memory_size,
  //    block_size_limit);
  //CHECK_EQ(err, hipSuccess);

  const int physical_thread_count =
      d.getNumHipMultiProcessors() * d.maxHipThreadsPerMultiProcessor();
  thread_per_block = std::min(1024, d.maxHipThreadsPerBlock());
  block_count =
      std::min(DIV_UP(physical_thread_count, thread_per_block),
               d.getNumHipMultiProcessors());
#endif

#define MIN3(a, b, c) std::min((a), std::min((b), (c)))
  int threadsx = MIN3(xdim, thread_per_block, xthreadlimit);
  int threadsy =
      MIN3(ydim, std::max(thread_per_block / threadsx, 1), ythreadlimit);
  int threadsz =
      MIN3(zdim, std::max(thread_per_block / (threadsx * threadsy), 1),
           zthreadlimit);

  int blocksx = MIN3(block_count, DIV_UP(xdim, threadsx), xgridlimit);
  int blocksy =
      MIN3(DIV_UP(block_count, blocksx), DIV_UP(ydim, threadsy), ygridlimit);
  int blocksz = MIN3(DIV_UP(block_count, (blocksx * blocksy)),
                     DIV_UP(zdim, threadsz), zgridlimit);
#undef MIN3

  config.virtual_thread_count = dim3(xdim, ydim, zdim);
  config.thread_per_block = dim3(threadsx, threadsy, threadsz);
  config.block_count = dim3(blocksx, blocksy, blocksz);
  return config;
}

template <typename DeviceFunc>
inline Gpu2DLaunchConfig GetGpu2DLaunchConfig(
    int xdim, int ydim, const GPUDevice& d, DeviceFunc func,
    size_t dynamic_shared_memory_size, int block_size_limit) {
  return GetGpu3DLaunchConfig(xdim, ydim, 1, d, func,
                               dynamic_shared_memory_size, block_size_limit);
}

#if GOOGLE_CUDA
// Returns a raw reference to the current cuda stream.  Required by a
// number of kernel calls (for which StreamInterface* does not work), i.e.
// CUB and certain cublas primitives.
inline const cudaStream_t& GetGpuStream(OpKernelContext* context) {
  const cudaStream_t* ptr = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GPUStreamMemberHack()));
  return *ptr;
}
#endif

namespace gpu_helper {

template <typename IntType>
__device__ IntType upper_bound(IntType* first, IntType count, IntType val) {
  IntType* orig = first;
  IntType* it = nullptr;
  IntType step = 0;
  while (count > 0) {
    it = first;
    step = count / 2;
    it += step;
    if (!(val < *it)) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }

  return first - orig;
}

}  // namespace gpu_helper

template <typename T>
__device__ __host__ inline T ldg(const T* address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  return __ldg(address);
#else
  return *address;
#endif
}

template <>
__device__ __host__ inline std::complex<float> ldg(
    const std::complex<float>* address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  float2 mem = __ldg(reinterpret_cast<const float2*>(address));
  return std::complex<float>(mem.x, mem.y);
#else
  return *address;
#endif
}

template <>
__device__ __host__ inline std::complex<double> ldg(
    const std::complex<double>* address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  double2 mem = __ldg(reinterpret_cast<const double2*>(address));
  return std::complex<double>(mem.x, mem.y);
#else
  return *address;
#endif
}

// CUDA provides atomic ops, but not for all types.  We provide wrappers
// for some ops and provide implementation for all reasonable types.
#define GPU_ATOMIC_WRAPPER(op, T) \
  __device__ inline T GpuAtomic##op(T* address, T val)

#define USE_GPU_ATOMIC(op, T) \
  GPU_ATOMIC_WRAPPER(op, T) { return atomic##op(address, val); }

// For atomicAdd.
USE_GPU_ATOMIC(Add, int32);
USE_GPU_ATOMIC(Add, uint32);
USE_GPU_ATOMIC(Add, uint64);
USE_GPU_ATOMIC(Add, float);

// For atomicMax.
USE_GPU_ATOMIC(Max, int32);
USE_GPU_ATOMIC(Max, uint32);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
USE_GPU_ATOMIC(Max, uint64);
#else
// The uint64 overload of atomicMax() is only available for __CUDA_ARCH__ >=
// 350.  If not satisfied, we provide a custom implementation using atomicCAS().
GPU_ATOMIC_WRAPPER(Max, uint64) {
  uint64* address_as_ull = reinterpret_cast<uint64*>(address);
  uint64 old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, max(val, assumed));
  } while (assumed != old);

  return old;
}
#endif

// Custom implementation of atomicAdd for double.
// This implementation is copied from CUDA manual.
GPU_ATOMIC_WRAPPER(Add, double) {
  uint64* address_as_ull = reinterpret_cast<uint64*>(address);
  uint64 old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN
  } while (assumed != old);

  return __longlong_as_double(old);
}

// Helper functions for GpuAtomicAdd(half*, half), below.
//
// Note that if __CUDA_ARCH__ >= 530, we could probably use __hadd2()
// for a more efficient implementation, assuming that adding -0.0
// will never harm the neighboring value. In this version, we take special
// care to guarantee the bits of the untouched value are unchanged.
inline __device__ uint32 add_to_low_half(uint32 val, float x) {
  Eigen::half low_half;
  low_half.x = static_cast<uint16>(val & 0xffffu);
  low_half = static_cast<Eigen::half>(static_cast<float>(low_half) + x);
  return (val & 0xffff0000u) | low_half.x;
}

inline __device__ uint32 add_to_high_half(uint32 val, float x) {
  Eigen::half high_half;
  high_half.x = static_cast<uint16>(val >> 16);
  high_half = static_cast<Eigen::half>(static_cast<float>(high_half) + x);
  return (val & 0xffffu) | (high_half.x << 16);
}

// Custom implementation of atomicAdd for half. Note that we don't have
// atomicCAS() for anything less than 32 bits, so we need to include the
// other 16 bits in the operation.
//
// Unlike the other atomic adds, this version is going to be very slow
// under high concurrency, since most threads will be spinning on failing
// their compare-and-swap tests. (The fact that we get false sharing on the
// neighboring fp16 makes this even worse.) If you are doing a large reduction,
// you are much better off with doing the intermediate steps in fp32 and then
// switching to fp16 as late as you can in the calculations.
//
// Note: Assumes little endian.
GPU_ATOMIC_WRAPPER(Add, Eigen::half) {
  float val_as_float(val);
  intptr_t address_int = reinterpret_cast<intptr_t>(address);
  if ((address_int & 0x2) == 0) {
    // The half is in the first part of the uint32 (lower 16 bits).
    uint32* address_as_uint32 = reinterpret_cast<uint32*>(address);
    assert(((intptr_t)address_as_uint32 & 0x3) == 0);
    uint32 old = *address_as_uint32, assumed;

    do {
      assumed = old;
      old = atomicCAS(address_as_uint32, assumed,
                      add_to_low_half(assumed, val_as_float));

      // Note: uses integer comparison to avoid hang in case of NaN
    } while (assumed != old);

    Eigen::half ret;
    ret.x = old & 0xffffu;
    return ret;
  } else {
    // The half is in the second part of the uint32 (upper 16 bits).
    uint32* address_as_uint32 = reinterpret_cast<uint32*>(address_int - 2);
    assert(((intptr_t)address_as_uint32 & 0x3) == 0);
    uint32 old = *address_as_uint32, assumed;

    do {
      assumed = old;
      old = atomicCAS(address_as_uint32, assumed,
                      add_to_high_half(assumed, val_as_float));

      // Note: uses integer comparison to avoid hang in case of NaN
    } while (assumed != old);

    Eigen::half ret;
    ret.x = old >> 16;
    return ret;
  }
}

template <typename T>
__global__ void SetZero(const int nthreads, T* bottom_diff) {
  GPU_1D_KERNEL_LOOP(index, nthreads) { *(bottom_diff + index) = T(0); }
}

// For atomicSub.

// Custom implementation for sub by just negating the value.
#define WRAPPED_ATOMIC_SUB(T) \
  GPU_ATOMIC_WRAPPER(Sub, T) { return GpuAtomicAdd(address, -val); }

WRAPPED_ATOMIC_SUB(uint64);
WRAPPED_ATOMIC_SUB(int32);
WRAPPED_ATOMIC_SUB(uint32);
WRAPPED_ATOMIC_SUB(float);
WRAPPED_ATOMIC_SUB(double);

#undef WRAPPED_ATOMIC_SUB

// For atomicMul.
GPU_ATOMIC_WRAPPER(Mul, int32) {
  int32 old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, val * assumed);
  } while (assumed != old);
  return old;
}

GPU_ATOMIC_WRAPPER(Mul, uint32) {
  uint32 old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, val * assumed);
  } while (assumed != old);
  return old;
}

GPU_ATOMIC_WRAPPER(Mul, uint64) {
  uint64 old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, val * assumed);
  } while (assumed != old);
  return old;
}

GPU_ATOMIC_WRAPPER(Mul, float) {
  int32* address_as_int = reinterpret_cast<int32*>(address);
  int32 old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed,
                    __float_as_int(val * __int_as_float(assumed)));
  } while (assumed != old);
  return __int_as_float(old);
}

GPU_ATOMIC_WRAPPER(Mul, double) {
  uint64* address_as_ull = reinterpret_cast<uint64*>(address);
  uint64 old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val * __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

// For atomicDiv.
GPU_ATOMIC_WRAPPER(Div, int32) {
  int32 old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, assumed / val);
  } while (assumed != old);
  return old;
}

GPU_ATOMIC_WRAPPER(Div, uint32) {
  uint32 old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, assumed / val);
  } while (assumed != old);
  return old;
}

GPU_ATOMIC_WRAPPER(Div, uint64) {
  uint64 old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, assumed / val);
  } while (assumed != old);
  return old;
}

GPU_ATOMIC_WRAPPER(Div, float) {
  int32* address_as_int = reinterpret_cast<int32*>(address);
  int32 old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed,
                    __float_as_int(__int_as_float(assumed) / val));
  } while (assumed != old);
  return __int_as_float(old);
}

GPU_ATOMIC_WRAPPER(Div, double) {
  uint64* address_as_ull = reinterpret_cast<uint64*>(address);
  uint64 old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(__longlong_as_double(assumed) / val));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#undef USE_GPU_ATOMIC
#undef GPU_ATOMIC_WRAPPER

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T tf_min(const T& x, const T& y) {
  return x > y ? y : x;
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T tf_max(const T& x, const T& y) {
  return x < y ? y : x;
}

template <typename T>
__device__ EIGEN_ALWAYS_INLINE T GpuShuffle(T value, int srcLane,
                                             int width = warpSize) {
  return __shfl(value, srcLane, width);
}

template <typename T>
__device__ EIGEN_ALWAYS_INLINE T GpuShuffleUp(T value, int delta,
                                               int width = warpSize) {
  return __shfl_up(value, delta, width);
}

template <typename T>
__device__ EIGEN_ALWAYS_INLINE T GpuShuffleDown(T value, int delta,
                                                 int width = warpSize) {
  return __shfl_down(value, delta, width);
}

template <typename T>
__device__ EIGEN_ALWAYS_INLINE T GpuShuffleXor(T value, int laneMask,
                                                int width = warpSize) {
  return __shfl_xor(value, laneMask, width);
}


#if GOOGLE_CUDA
// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// A bug has been filed with NVIDIA and will be fixed in the next CUDA release.
// TODO(csigg): remove when the bug is fixed in the next CUDA release.
__device__ EIGEN_ALWAYS_INLINE double GpuShuffle(double value, int srcLane,
                                                  int width = warpSize) {
  unsigned lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(value));
  hi = __shfl(hi, srcLane, width);
  lo = __shfl(lo, srcLane, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(value) : "r"(lo), "r"(hi));
  return value;
}

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// A bug has been filed with NVIDIA and will be fixed in the next CUDA release.
// TODO(csigg): remove when the bug is fixed in the next CUDA release.
__device__ EIGEN_ALWAYS_INLINE double GpuShuffleUp(double value, int delta,
                                                    int width = warpSize) {
  unsigned lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(value));
  hi = __shfl_up(hi, delta, width);
  lo = __shfl_up(lo, delta, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(value) : "r"(lo), "r"(hi));
  return value;
}

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// A bug has been filed with NVIDIA and will be fixed in the next CUDA release.
// TODO(csigg): remove when the bug is fixed in the next CUDA release.
__device__ EIGEN_ALWAYS_INLINE double GpuShuffleDown(double value, int delta,
                                                      int width = warpSize) {
  unsigned lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(value));
  hi = __shfl_down(hi, delta, width);
  lo = __shfl_down(lo, delta, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(value) : "r"(lo), "r"(hi));
  return value;
}

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// A bug has been filed with NVIDIA and will be fixed in the next CUDA release.
// TODO(csigg): remove when the bug is fixed in the next CUDA release.
__device__ EIGEN_ALWAYS_INLINE double GpuShuffleXor(double value, int laneMask,
                                                     int width = warpSize) {
  unsigned lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(value));
  hi = __shfl_xor(hi, laneMask, width);
  lo = __shfl_xor(lo, laneMask, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(value) : "r"(lo), "r"(hi));
  return value;
}
#elif TENSORFLOW_USE_ROCM
// XXX FIXME these are rather inefficient implementation for double
// need to figure out the proper way to do it in GCN ISA

// XXX: should HIP / HCC introduce unsigned version of shfl operators?
__device__ EIGEN_ALWAYS_INLINE double GpuShuffle(double value, int srcLane,
                                                  int width = warpSize) {
  unsigned lo, hi;
  uint64_t tmp = static_cast<uint64_t>(value);
  lo = static_cast<unsigned>(tmp);
  hi = static_cast<unsigned>(tmp >> 32);
  hi = __shfl(static_cast<int>(hi), srcLane, width);
  lo = __shfl(static_cast<int>(lo), srcLane, width);
  return static_cast<double>(static_cast<uint64_t>(hi) << 32 | static_cast<uint64_t>(lo));
}

__device__ EIGEN_ALWAYS_INLINE double GpuShuffleUp(double value, int delta,
                                                    int width = warpSize) {
  unsigned lo, hi;
  uint64_t tmp = static_cast<uint64_t>(value);
  lo = static_cast<unsigned>(tmp);
  hi = static_cast<unsigned>(tmp >> 32);
  hi = __shfl_up(static_cast<int>(hi), delta, width);
  lo = __shfl_up(static_cast<int>(lo), delta, width);
  return static_cast<double>(static_cast<uint64_t>(hi) << 32 | static_cast<uint64_t>(lo));
}

__device__ EIGEN_ALWAYS_INLINE double GpuShuffleDown(double value, int delta,
                                                      int width = warpSize) {
  unsigned lo, hi;
  uint64_t tmp = static_cast<uint64_t>(value);
  lo = static_cast<unsigned>(tmp);
  hi = static_cast<unsigned>(tmp >> 32);
  hi = __shfl_down(static_cast<int>(hi), delta, width);
  lo = __shfl_down(static_cast<int>(lo), delta, width);
  return static_cast<double>(static_cast<uint64_t>(hi) << 32 | static_cast<uint64_t>(lo));
}

__device__ EIGEN_ALWAYS_INLINE double GpuShuffleXor(double value, int laneMask,
                                                     int width = warpSize) {
  unsigned lo, hi;
  uint64_t tmp = static_cast<uint64_t>(value);
  lo = static_cast<unsigned>(tmp);
  hi = static_cast<unsigned>(tmp >> 32);
  hi = __shfl_xor(static_cast<int>(hi), laneMask, width);
  lo = __shfl_xor(static_cast<int>(lo), laneMask, width);
  return static_cast<double>(static_cast<uint64_t>(hi) << 32 | static_cast<uint64_t>(lo));
}
#endif

}  // namespace tensorflow

#undef DIV_UP

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_
