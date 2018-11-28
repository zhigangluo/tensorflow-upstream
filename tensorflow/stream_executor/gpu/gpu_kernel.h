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

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_KERNEL_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_KERNEL_H_

#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/gpu/gpu_types.h"

namespace stream_executor {
namespace gpu {

// Wraps a GPUFunctionHandle to implement the platform-independent KernelInterface.
class GPUKernel : public internal::KernelInterface {
 public:
  GPUKernel()
      : gpu_function_(nullptr),
        arity_(0),
        preferred_cache_config_(KernelCacheConfig::kNoPreference) {}

  // Note that the function is unloaded when the module is unloaded, and the
  // module that the function is contained in is owned by the CUDAExecutor.
  ~GPUKernel() override {}

  // As arity cannot be reflected upon using the GPU API, the arity is
  // explicitly set during the GPUExecutor::GetKernel initialization process.
  void SetArity(unsigned arity) { arity_ = arity; }
  unsigned Arity() const override { return arity_; }

  // get (implementation specific) function handle for passing to the GPU API.
  GPUFunctionHandle GetFunctionHandle() const {
    DCHECK(gpu_function_ != nullptr);
    return gpu_function_;
  }

  // set (implementation specific) function handle for passing to the GPU API.
  void SetFunctionHandle(GPUFunctionHandle hnd) {
    DCHECK(hnd != nullptr);
    gpu_function_ = hnd;
  }

  // Sets the current kernel cache configuration preference.
  void SetPreferredCacheConfig(KernelCacheConfig config) override {
    preferred_cache_config_ = config;
  }

  // Returns the current kernel cache configuration preference.
  KernelCacheConfig GetPreferredCacheConfig() const override {
    return preferred_cache_config_;
  }

 private:
  GPUFunctionHandle gpu_function_;  // Wrapped GPU kernel handle.
  unsigned arity_;  // Number of formal parameters the kernel takes.

  // Preferred (but not required) cache configuration for this kernel.
  KernelCacheConfig preferred_cache_config_;
};

// Given a platform-independent kernel datatype, returns the (const) internal
// GPU platform implementation pointer.
inline const GPUKernel* AsGPUKernel(const KernelBase* kernel) {
  return static_cast<const GPUKernel*>(kernel->implementation());
}

// Given a platform-independent kernel datatype, returns the (non-const)
// internal GPU platform implementation pointer.
inline GPUKernel* AsGPUKernel(KernelBase* kernel) {
  return static_cast<GPUKernel*>(kernel->implementation());
}

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_KERNEL_H_
