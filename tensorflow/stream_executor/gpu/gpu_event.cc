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

#include "tensorflow/stream_executor/gpu/gpu_event.h"

#if GOOGLE_CUDA

#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
  
#elif TENSORFLOW_USE_ROCM

#include "tensorflow/stream_executor/rocm/rocm_gpu_executor.h"
#include "tensorflow/stream_executor/rocm/rocm_stream.h"

#endif

#include "tensorflow/stream_executor/lib/statusor.h"

namespace stream_executor {
namespace gpu {

#if GOOGLE_CUDA
  
GPUEvent::GPUEvent(cuda::CUDAExecutor* parent)
    : parent_(parent), gpu_event_(nullptr) {}

#elif TENSORFLOW_USE_ROCM
  
GPUEvent::GPUEvent(rocm::ROCMExecutor* parent)
    : parent_(parent), gpu_event_(nullptr) {}

#endif
  
GPUEvent::~GPUEvent() {}

port::Status GPUEvent::Init() {
#if GOOGLE_CUDA
  
  return cuda::CUDADriver::CreateEvent(parent_->cuda_context(), &gpu_event_,
                                 cuda::CUDADriver::EventFlags::kDisableTiming);
  
#elif TENSORFLOW_USE_ROCM

  return rocm::ROCMDriver::CreateEvent(parent_->device_ordinal(), &gpu_event_,
                                 rocm::ROCMDriver::EventFlags::kDisableTiming);

#endif
}

port::Status GPUEvent::Destroy() {
#if GOOGLE_CUDA
  
  return cuda::CUDADriver::DestroyEvent(parent_->cuda_context(), &gpu_event_);
  
#elif TENSORFLOW_USE_ROCM

  return rocm::ROCMDriver::DestroyEvent(parent_->device_ordinal(), &gpu_event_);

#endif
}

#if GOOGLE_CUDA
  
port::Status GPUEvent::Record(cuda::CUDAStream* stream) {
  return cuda::CUDADriver::RecordEvent(parent_->cuda_context(), gpu_event_,
                                 stream->cuda_stream());
}
  
#elif TENSORFLOW_USE_ROCM

port::Status GPUEvent::Record(rocm::ROCMStream* stream) {
  return rocm::ROCMDriver::RecordEvent(parent_->device_ordinal(), gpu_event_,
                                 stream->rocm_stream());

}
#endif

Event::Status GPUEvent::PollForStatus() {
#if GOOGLE_CUDA
  
  port::StatusOr<CUresult> status =
      cuda::CUDADriver::QueryEvent(parent_->cuda_context(), gpu_event_);
  if (!status.ok()) {
    LOG(ERROR) << "Error polling for event status: "
               << status.status().error_message();
    return Event::Status::kError;
  }

  switch (status.ValueOrDie()) {
    case CUDA_SUCCESS:
      return Event::Status::kComplete;
    case CUDA_ERROR_NOT_READY:
      return Event::Status::kPending;
    default:
      LOG(INFO) << "Error condition returned for event status: "
                << status.ValueOrDie();
      return Event::Status::kError;
  }
  
#elif TENSORFLOW_USE_ROCM

  port::StatusOr<hipError_t> status =
      rocm::ROCMDriver::QueryEvent(parent_->device_ordinal(), gpu_event_);
  if (!status.ok()) {
    LOG(ERROR) << "Error polling for event status: "
               << status.status().error_message();
    return Event::Status::kError;
  }

  switch (status.ValueOrDie()) {
    case hipSuccess:
      return Event::Status::kComplete;
    case hipErrorNotReady:
      return Event::Status::kPending;
    default:
      LOG(INFO) << "Error condition returned for event status: "
                << status.ValueOrDie();
      return Event::Status::kError;
  }

#endif
}

#if GOOGLE_CUDA
  
const CUevent& GPUEvent::gpu_event() {
  return gpu_event_;
}
  
#elif TENSORFLOW_USE_ROCM

const hipEvent_t& GPUEvent::gpu_event() {
  return gpu_event_;
}

#endif

}  // namespace gpu
}  // namespace stream_executor
