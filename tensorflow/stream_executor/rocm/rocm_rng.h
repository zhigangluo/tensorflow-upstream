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

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_RNG_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_RNG_H_

#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/rng.h"

namespace stream_executor {

class Stream;
template <typename ElemT>
class DeviceMemory;

namespace gpu {

// Opaque and unique identifier for the hipRAND plugin.
extern const PluginId kHipRandPlugin;

class GPUExecutor;

// ROCM-platform implementation of the random number generation support
// interface.
//
// Thread-safe post-initialization.
class ROCMRng : public rng::RngSupport {
 public:
  explicit ROCMRng(GPUExecutor *parent);

  // Retrieves a hiprng library generator handle. This is necessary for
  // enqueuing random number generation work onto the device.
  // TODO(leary) provide a way for users to select the RNG algorithm.
  bool Init();

  // Releases a hiprng library generator handle, if one was acquired.
  ~ROCMRng() override;

  // See rng::RngSupport for details on the following overrides.
  bool DoPopulateRandUniform(Stream *stream, DeviceMemory<float> *v) override;
  bool DoPopulateRandUniform(Stream *stream, DeviceMemory<double> *v) override;
  bool DoPopulateRandUniform(Stream *stream,
                             DeviceMemory<std::complex<float>> *v) override;
  bool DoPopulateRandUniform(Stream *stream,
                             DeviceMemory<std::complex<double>> *v) override;
  bool DoPopulateRandGaussian(Stream *stream, float mean, float stddev,
                              DeviceMemory<float> *v) override;
  bool DoPopulateRandGaussian(Stream *stream, double mean, double stddev,
                              DeviceMemory<double> *v) override;

  bool SetSeed(Stream *stream, const uint8 *seed, uint64 seed_bytes) override;

 private:
  // Actually performs the work of generating random numbers - the public
  // methods are thin wrappers to this interface.
  template <typename T>
  bool DoPopulateRandUniformInternal(Stream *stream, DeviceMemory<T> *v);
  template <typename ElemT, typename FuncT>
  bool DoPopulateRandGaussianInternal(Stream *stream, ElemT mean, ElemT stddev,
                                      DeviceMemory<ElemT> *v, FuncT func);

  // Sets the stream for the internal hiprng generator.
  //
  // This is a stateful operation, as the handle can only have one stream set at
  // a given time, so it is usually performed right before enqueuing work to do
  // with random number generation.
  bool SetStream(Stream *stream) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // mutex that guards the hipRAND handle for this device.
  mutex mu_;

  // GPUExecutor which instantiated this ROCMRng.
  // Immutable post-initialization.
  GPUExecutor *parent_;

  // hipRAND library handle on the device.
  hiprandGenerator_t rng_ GUARDED_BY(mu_);

  SE_DISALLOW_COPY_AND_ASSIGN(ROCMRng);
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_RNG_H_