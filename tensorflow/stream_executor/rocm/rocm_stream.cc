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

#include "tensorflow/stream_executor/rocm/rocm_stream.h"

#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/stream.h"

namespace stream_executor {
namespace gpu {

bool GPUStream::Init() {
  if (!GPUDriver::CreateStream(parent_->gpu_context(), &gpu_stream_)) {
    return false;
  }
  return GPUDriver::CreateEvent(parent_->gpu_context(), &completed_event_,
                                 GPUDriver::EventFlags::kDisableTiming)
      .ok();
}

void GPUStream::Destroy() {
  if (completed_event_ != nullptr) {
    port::Status status =
        GPUDriver::DestroyEvent(parent_->gpu_context(), &completed_event_);
    if (!status.ok()) {
      LOG(ERROR) << status.error_message();
    }
  }

  GPUDriver::DestroyStream(parent_->gpu_context(), &gpu_stream_);
}

bool GPUStream::IsIdle() const {
  return GPUDriver::IsStreamIdle(parent_->gpu_context(), gpu_stream_);
}

GPUStream *AsGPUStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return static_cast<GPUStream *>(stream->implementation());
}

GPUStreamHandle AsGPUStreamValue(Stream *stream) {
  DCHECK(stream != nullptr);
  return AsGPUStream(stream)->gpu_stream();
}

}  // namespace gpu
}  // namespace stream_executor
