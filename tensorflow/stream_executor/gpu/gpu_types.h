/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_TYPES_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_TYPES_H_

#if TENSORFLOW_USE_ROCM

#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hip/hip_complex.h"

#else // CUDA

#include "cuda/include/cuda.h"
#include "cuda/include/cuComplex.h"
#include "cuda/include/cuda.h"

#endif

namespace stream_executor {
namespace gpu {

#if TENSORFLOW_USE_ROCM

using GPUStreamHandle = hipStream_t;
using GPUEventHandle = hipEvent_t;
using GPUFunctionHandle = hipFunction_t;
using GPUFunctionAttribute = hipDeviceAttribute_t;  // not a typo!
using GPUDeviceHandle = hipDevice_t;
using GPUDevicePointer = hipDeviceptr_t;
using GPUDeviceAttribute = hipDeviceAttribute_t;
using GPUDeviceProperty = hipDeviceProp_t;
using GPUModuleHandle = hipModule_t;
using GPUStatus = hipError_t;
using GPUFuncCachePreference = hipFuncCache_t;
using GPUSharedMemConfig = hipSharedMemConfig;
using GPUComplexType = hipComplex;
using GPUDoubleComplexType = hipDoubleComplex;
 
#else // CUDA

using GPUStreamHandle = CUstream;
using GPUEventHandle = CUevent;
using GPUFunctionHandle = CUfunction;
using GPUFunctionAttribute = CUfunction_attribute;
using GPUDeviceHandle = CUdevice;
using GPUDevicePointer = CUdeviceptr;
using GPUDeviceAttribute = CUdevice_attribute;
using GPUDeviceProperty = CUdevprop;
using GPUModuleHandle = CUmodule;
using GPUStatus = CUresult;
using GPUFuncCachePreference = CUfunc_cache;
using GPUSharedMemConfig = CUsharedconfig;
using GPUComplexType = cuComplex;
using GPUDoubleComplexType = cuDoubleComplex;

#endif
 
}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_TYPES_H_
