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

#include "rocm/include/hiprng/hiprng.h"
#include "tensorflow/stream_executor/rocm/rocm_rng.h"

#include "tensorflow/stream_executor/rocm/rocm_activation.h"
#include "tensorflow/stream_executor/rocm/rocm_gpu_executor.h"
#include "tensorflow/stream_executor/rocm/rocm_helpers.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/stream_executor/rocm/rocm_stream.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/rng.h"

// Formats hiprngStatus_t to output prettified values into a log stream.
std::ostream &operator<<(std::ostream &in, const hiprngStatus_t &status) {
#define OSTREAM_HIPRNG_STATUS(__name) \
  case HIPRNG_STATUS_##__name:        \
    in << "HIPRNG_STATUS_" #__name;   \
    return in;

  switch (status) {
    OSTREAM_HIPRNG_STATUS(SUCCESS)
    OSTREAM_HIPRNG_STATUS(VERSION_MISMATCH)
    OSTREAM_HIPRNG_STATUS(NOT_INITIALIZED)
    OSTREAM_HIPRNG_STATUS(ALLOCATION_FAILED)
    OSTREAM_HIPRNG_STATUS(TYPE_ERROR)
    OSTREAM_HIPRNG_STATUS(OUT_OF_RANGE)
    OSTREAM_HIPRNG_STATUS(LENGTH_NOT_MULTIPLE)
    OSTREAM_HIPRNG_STATUS(LAUNCH_FAILURE)
    OSTREAM_HIPRNG_STATUS(PREEXISTING_FAILURE)
    OSTREAM_HIPRNG_STATUS(INITIALIZATION_FAILED)
    OSTREAM_HIPRNG_STATUS(ARCH_MISMATCH)
    OSTREAM_HIPRNG_STATUS(INTERNAL_ERROR)
    default:
      in << "hiprngStatus_t(" << static_cast<int>(status) << ")";
      return in;
  }
}

namespace perftools {
namespace gputools {
namespace rocm {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kHipRngPlugin);

namespace wrap {

#define PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(__name)                      \
  struct WrapperShim__##__name {                                    \
    template <typename... Args>                                     \
    hiprngStatus_t operator()(ROCMExecutor *parent, Args... args) { \
      rocm::ScopedActivateExecutorContext sac{parent};              \
      return ::__name(args...);                                     \
    }                                                               \
  } __name;

PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngCreateGenerator);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngDestroyGenerator);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngSetStream);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngGenerateUniform);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngGenerateUniformDouble);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngSetPseudoRandomGeneratorSeed);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngSetGeneratorOffset);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngGenerateNormal);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngGenerateNormalDouble);

}  // namespace wrap

template <typename T>
string TypeString();

template <>
string TypeString<float>() {
  return "float";
}

template <>
string TypeString<double>() {
  return "double";
}

template <>
string TypeString<std::complex<float>>() {
  return "std::complex<float>";
}

template <>
string TypeString<std::complex<double>>() {
  return "std::complex<double>";
}

ROCMRng::ROCMRng(ROCMExecutor *parent) : parent_(parent), rng_(nullptr) {}

ROCMRng::~ROCMRng() {
  if (rng_ != nullptr) {
    wrap::hiprngDestroyGenerator(parent_, rng_);
  }
}

bool ROCMRng::Init() {
  mutex_lock lock{mu_};
  CHECK(rng_ == nullptr);

  hiprngStatus_t ret =
      wrap::hiprngCreateGenerator(parent_, &rng_, HIPRNG_RNG_PSEUDO_DEFAULT);
  if (ret != HIPRNG_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create random number generator: " << ret;
    return false;
  }

  CHECK(rng_ != nullptr);
  return true;
}

bool ROCMRng::SetStream(Stream *stream) {
  hiprngStatus_t ret =
      wrap::hiprngSetStream(parent_, rng_, AsROCMStreamValue(stream));
  if (ret != HIPRNG_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for random generation: " << ret;
    return false;
  }

  return true;
}

// Returns true if std::complex stores its contents as two consecutive
// elements. Tests int, float and double, as the last two are independent
// specializations.
constexpr bool ComplexIsConsecutiveFloats() {
  return sizeof(std::complex<int>) == 8 && sizeof(std::complex<float>) == 8 &&
      sizeof(std::complex<double>) == 16;
}

template <typename T>
bool ROCMRng::DoPopulateRandUniformInternal(Stream *stream,
                                            DeviceMemory<T> *v) {
  mutex_lock lock{mu_};
  static_assert(ComplexIsConsecutiveFloats(),
                "std::complex values are not stored as consecutive values");

  if (!SetStream(stream)) {
    return false;
  }

  // std::complex<T> is currently implemented as two consecutive T variables.
  uint64 element_count = v->ElementCount();
  if (std::is_same<T, std::complex<float>>::value ||
      std::is_same<T, std::complex<double>>::value) {
    element_count *= 2;
  }

  hiprngStatus_t ret;
  if (std::is_same<T, float>::value ||
      std::is_same<T, std::complex<float>>::value) {
    ret = wrap::hiprngGenerateUniform(
        parent_, rng_, reinterpret_cast<float *>(ROCMMemoryMutable(v)),
        element_count);
  } else {
    ret = wrap::hiprngGenerateUniformDouble(
        parent_, rng_, reinterpret_cast<double *>(ROCMMemoryMutable(v)),
        element_count);
  }
  if (ret != HIPRNG_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do uniform generation of " << v->ElementCount()
               << " " << TypeString<T>() << "s at " << v->opaque() << ": "
               << ret;
    return false;
  }

  return true;
}

bool ROCMRng::DoPopulateRandUniform(Stream *stream, DeviceMemory<float> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool ROCMRng::DoPopulateRandUniform(Stream *stream, DeviceMemory<double> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool ROCMRng::DoPopulateRandUniform(Stream *stream,
                                    DeviceMemory<std::complex<float>> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool ROCMRng::DoPopulateRandUniform(Stream *stream,
                                    DeviceMemory<std::complex<double>> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

template <typename ElemT, typename FuncT>
bool ROCMRng::DoPopulateRandGaussianInternal(Stream *stream, ElemT mean,
                                             ElemT stddev,
                                             DeviceMemory<ElemT> *v,
                                             FuncT func) {
  mutex_lock lock{mu_};

  if (!SetStream(stream)) {
    return false;
  }

  uint64 element_count = v->ElementCount();
  hiprngStatus_t ret =
      func(parent_, rng_, ROCMMemoryMutable(v), element_count, mean, stddev);

  if (ret != HIPRNG_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do gaussian generation of " << v->ElementCount()
               << " floats at " << v->opaque() << ": " << ret;
    return false;
  }

  return true;
}

bool ROCMRng::DoPopulateRandGaussian(Stream *stream, float mean, float stddev,
                                     DeviceMemory<float> *v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        wrap::hiprngGenerateNormal);
}

bool ROCMRng::DoPopulateRandGaussian(Stream *stream, double mean, double stddev,
                                     DeviceMemory<double> *v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        wrap::hiprngGenerateNormalDouble);
}

bool ROCMRng::SetSeed(Stream *stream, const uint8 *seed, uint64 seed_bytes) {
  mutex_lock lock{mu_};
  CHECK(rng_ != nullptr);

  if (!CheckSeed(seed, seed_bytes)) {
    return false;
  }

  if (!SetStream(stream)) {
    return false;
  }

  // Requires 8 bytes of seed data; checked in RngSupport::CheckSeed (above)
  // (which itself requires 16 for API consistency with host RNG fallbacks).
  hiprngStatus_t ret = wrap::hiprngSetPseudoRandomGeneratorSeed(
      parent_, rng_, *(reinterpret_cast<const uint64 *>(seed)));
  if (ret != HIPRNG_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set rng seed: " << ret;
    return false;
  }

  ret = wrap::hiprngSetGeneratorOffset(parent_, rng_, 0);
  if (ret != HIPRNG_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to reset rng position: " << ret;
    return false;
  }
  return true;
}

}  // namespace rocm
}  // namespace gputools
}  // namespace perftools

namespace gpu = ::perftools::gputools;

REGISTER_MODULE_INITIALIZER(register_hiprng, {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::RngFactory>(
              gpu::rocm::kROCmPlatformId, gpu::rocm::kHipRngPlugin, "hipRNG",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::rng::RngSupport * {
                gpu::rocm::ROCMExecutor *rocm_executor =
                    dynamic_cast<gpu::rocm::ROCMExecutor *>(parent);
                if (rocm_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the hipRNG "
                      << "support library with a non-ROCM StreamExecutor";
                  return nullptr;
                }

                gpu::rocm::ROCMRng *rng = new gpu::rocm::ROCMRng(rocm_executor);
                if (!rng->Init()) {
                  // Note: Init() will log a more specific error.
                  delete rng;
                  return nullptr;
                }
                return rng;
              });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register hipRNG factory: "
               << status.error_message();
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::rocm::kROCmPlatformId,
                                                     gpu::PluginKind::kRng,
                                                     gpu::rocm::kHipRngPlugin);
});
