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

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_EVENT_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_EVENT_H_

#if GOOGLE_CUDA

#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"

#elif TENSORFLOW_USE_ROCM

#include "tensorflow/stream_executor/rocm/rocm_driver.h"
#include "tensorflow/stream_executor/rocm/rocm_stream.h"

#endif

#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/status.h"

namespace stream_executor {
namespace gpu {

// GPUEvent wraps a GPUevent in the platform-independent EventInterface
// interface.
class GPUEvent : public internal::EventInterface {
 public:
  
#if GOOGLE_CUDA
  
  explicit GPUEvent(cuda::CUDAExecutor* parent);
  
#elif TENSORFLOW_USE_ROCM
  
  explicit GPUEvent(rocm::ROCMExecutor* parent);
  
#endif
  
  ~GPUEvent() override;

  // Populates the GPU-platform-specific elements of this object.
  port::Status Init();

  // Deallocates any platform-specific elements of this object. This is broken
  // out (not part of the destructor) to allow for error reporting.
  port::Status Destroy();

#if GOOGLE_CUDA

  // Inserts the event at the current position into the specified stream.
  port::Status Record(cuda::CUDAStream* stream);

#elif TENSORFLOW_USE_ROCM
  
  // Inserts the event at the current position into the specified stream.
  port::Status Record(rocm::ROCMStream* stream);
  
#endif
  
  // Polls the GPU platform for the event's current status.
  Event::Status PollForStatus();

#if GOOGLE_CUDA

  // The underlying CUDA event element.
  const CUevent& gpu_event();
  
#elif TENSORFLOW_USE_ROCM
  
  // The underlying ROCM event element.
  const hipEvent_t& gpu_event();
  
#endif

  
 private:

#if GOOGLE_CUDA
  
  // The Executor used to which this object and CUevent are bound.
  cuda::CUDAExecutor* parent_;
  
  // The underlying CUDA event element.
  CUevent gpu_event_;
  
#elif TENSORFLOW_USE_ROCM
  
  // The Executor used to which this object and hipEvent_t are bound.
  rocm::ROCMExecutor* parent_;
  
  // The underlying ROCM event element.
  hipEvent_t gpu_event_;
  
#endif
  
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_EVENT_H_
