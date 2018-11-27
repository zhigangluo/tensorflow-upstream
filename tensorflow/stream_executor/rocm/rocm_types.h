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

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_TYPES_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_TYPES_H_

#include "tensorflow/stream_executor/gpu/gpu_types.h"

namespace stream_executor {
namespace gpu {

hipStream_t AsHipStream(GPUStreamHandle hnd) {
  return reinterpret_cast<hipStream_t>(hnd);
}

hipStream_t* AsHipStreamPtr(GPUStreamHandle* hndPtr) {
  return reinterpret_cast<hipStream_t*>(hndPtr);
}

hipEvent_t AsHipEvent(GPUEventHandle hnd) {
  return reinterpret_cast<hipEvent_t>(hnd);
}

hipEvent_t* AsHipEventPtr(GPUEventHandle* hndPtr) {
  return reinterpret_cast<hipEvent_t*>(hndPtr);
}

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_TYPES_H_
