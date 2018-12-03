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

// Defines the GPUStream type - the GPU-specific implementation of the generic
// StreamExecutor Stream interface.

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_STREAM_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_STREAM_H_

#include "tensorflow/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace gpu {

class GPUExecutor;

// Wraps a GPUStreamHandle in order to satisfy the platform-independent
// StreamInterface.
//
// Thread-safe post-initialization.
class GPUStream : public internal::StreamInterface {
 public:
  explicit GPUStream(GPUExecutor *parent)
      : parent_(parent), gpu_stream_(nullptr), completed_event_(nullptr) {}

  // Note: teardown is handled by a parent's call to DeallocateStream.
  ~GPUStream() override {}

  void *GpuStreamHack() override { return gpu_stream_; }
  void **GpuStreamMemberHack() override {
    return reinterpret_cast<void **>(&gpu_stream_);
  }

  // Explicitly initialize the GPU resources associated with this stream, used
  // by StreamExecutor::AllocateStream().
  bool Init();

  // Explicitly destroy the GPU resources associated with this stream, used by
  // StreamExecutor::DeallocateStream().
  void Destroy();

  // Returns true if no work is pending or executing on the stream.
  bool IsIdle() const;

  // Retrieves an event which indicates that all work enqueued into the stream
  // has completed. Ownership of the event is not transferred to the caller, the
  // event is owned by this stream.
  GPUEventHandle completed_event() { return completed_event_; }

  // Returns the GPUStreamHandle value for passing to the GPU API.
  //
  // Precond: this GPUStream has been allocated (otherwise passing a nullptr
  // into the NVIDIA library causes difficult-to-understand faults).
  GPUStreamHandle gpu_stream() const {
    DCHECK(gpu_stream_ != nullptr);
    return const_cast<GPUStreamHandle>(gpu_stream_);
  }

  GPUExecutor *parent() const { return parent_; }

 private:
  GPUExecutor *parent_;  // Executor that spawned this stream.
  GPUStreamHandle gpu_stream_;  // Wrapped GPU stream handle.

  // Event that indicates this stream has completed.
  GPUEventHandle completed_event_ = nullptr;
};

// Helper functions to simplify extremely common flows.
// Converts a Stream to the underlying GPUStream implementation.
GPUStream *AsGPUStream(Stream *stream);

// Extracts a GPUStreamHandle from a GPUStream-backed Stream object.
GPUStreamHandle AsGPUStreamValue(Stream *stream);

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_STREAM_H_