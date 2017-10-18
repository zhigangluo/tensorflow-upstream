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

// Include cuBLAS headers early, and then set EIGEN_HAS_ROCM_FP16
// if we have new enough ROCM (which we will only know after including
// rocm.h). This ensures that Eigen's Half.h does not attempt to make its own
// __half typedef if ROCM has already defined one (and conversely, that we do
// not include <rocm_fp16.h> after Half.h has made its typedef).
#include "rocm/include/rocm.h"
#include "rocm/include/cublas_v2.h"

#if ROCM_VERSION >= 7050
#define EIGEN_HAS_ROCM_FP16
#endif

#if ROCM_VERSION >= 8000
#define SE_ROCM_DATA_HALF ROCM_R_16F
#else
#define SE_ROCM_DATA_HALF CUBLAS_DATA_HALF
#endif

#include "tensorflow/stream_executor/rocm/rocm_blas.h"

#include <assert.h>
#include <complex>

#include "tensorflow/stream_executor/rocm/rocm_activation.h"
#include "tensorflow/stream_executor/rocm/rocm_gpu_executor.h"
#include "tensorflow/stream_executor/rocm/rocm_helpers.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/stream_executor/rocm/rocm_stream.h"
#include "tensorflow/stream_executor/rocm/rocm_timer.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace perftools {
namespace gputools {
namespace rocm {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuBlasPlugin);

namespace wrap {

#define PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(__name)                      \
  struct WrapperShim__##__name {                                    \
    static const char *kName;                                       \
    template <typename... Args>                                     \
    cublasStatus_t operator()(ROCMExecutor *parent, Args... args) { \
      rocm::ScopedActivateExecutorContext sac{parent};              \
      return ::__name(args...);                                     \
    }                                                               \
  } __name;                                                         \
  const char *WrapperShim__##__name::kName = #__name;

#define PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(__name) \
  PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(__name)

#define CUBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(cublasSnrm2)                    \
  __macro(cublasDnrm2)                    \
  __macro(cublasScnrm2)                   \
  __macro(cublasDznrm2)                   \
  __macro(cublasSdot)                     \
  __macro(cublasDdot)                     \
  __macro(cublasCdotu)                    \
  __macro(cublasCdotc)                    \
  __macro(cublasZdotu)                    \
  __macro(cublasZdotc)                    \
  __macro(cublasSscal)                    \
  __macro(cublasDscal)                    \
  __macro(cublasCscal)                    \
  __macro(cublasCsscal)                   \
  __macro(cublasZscal)                    \
  __macro(cublasZdscal)                   \
  __macro(cublasSaxpy)                    \
  __macro(cublasDaxpy)                    \
  __macro(cublasCaxpy)                    \
  __macro(cublasZaxpy)                    \
  __macro(cublasScopy)                    \
  __macro(cublasDcopy)                    \
  __macro(cublasCcopy)                    \
  __macro(cublasZcopy)                    \
  __macro(cublasSswap)                    \
  __macro(cublasDswap)                    \
  __macro(cublasCswap)                    \
  __macro(cublasZswap)                    \
  __macro(cublasIsamax)                   \
  __macro(cublasIdamax)                   \
  __macro(cublasIcamax)                   \
  __macro(cublasIzamax)                   \
  __macro(cublasIsamin)                   \
  __macro(cublasIdamin)                   \
  __macro(cublasIcamin)                   \
  __macro(cublasIzamin)                   \
  __macro(cublasSasum)                    \
  __macro(cublasDasum)                    \
  __macro(cublasScasum)                   \
  __macro(cublasDzasum)                   \
  __macro(cublasSrot)                     \
  __macro(cublasDrot)                     \
  __macro(cublasCrot)                     \
  __macro(cublasCsrot)                    \
  __macro(cublasZrot)                     \
  __macro(cublasZdrot)                    \
  __macro(cublasSrotg)                    \
  __macro(cublasDrotg)                    \
  __macro(cublasCrotg)                    \
  __macro(cublasZrotg)                    \
  __macro(cublasSrotm)                    \
  __macro(cublasDrotm)                    \
  __macro(cublasSrotmg)                   \
  __macro(cublasDrotmg)                   \
  __macro(cublasSgemv)                    \
  __macro(cublasDgemv)                    \
  __macro(cublasCgemv)                    \
  __macro(cublasZgemv)                    \
  __macro(cublasSgbmv)                    \
  __macro(cublasDgbmv)                    \
  __macro(cublasCgbmv)                    \
  __macro(cublasZgbmv)                    \
  __macro(cublasStrmv)                    \
  __macro(cublasDtrmv)                    \
  __macro(cublasCtrmv)                    \
  __macro(cublasZtrmv)                    \
  __macro(cublasStbmv)                    \
  __macro(cublasDtbmv)                    \
  __macro(cublasCtbmv)                    \
  __macro(cublasZtbmv)                    \
  __macro(cublasStpmv)                    \
  __macro(cublasDtpmv)                    \
  __macro(cublasCtpmv)                    \
  __macro(cublasZtpmv)                    \
  __macro(cublasStrsv)                    \
  __macro(cublasDtrsv)                    \
  __macro(cublasCtrsv)                    \
  __macro(cublasZtrsv)                    \
  __macro(cublasStpsv)                    \
  __macro(cublasDtpsv)                    \
  __macro(cublasCtpsv)                    \
  __macro(cublasZtpsv)                    \
  __macro(cublasStbsv)                    \
  __macro(cublasDtbsv)                    \
  __macro(cublasCtbsv)                    \
  __macro(cublasZtbsv)                    \
  __macro(cublasSsymv)                    \
  __macro(cublasDsymv)                    \
  __macro(cublasCsymv)                    \
  __macro(cublasZsymv)                    \
  __macro(cublasChemv)                    \
  __macro(cublasZhemv)                    \
  __macro(cublasSsbmv)                    \
  __macro(cublasDsbmv)                    \
  __macro(cublasChbmv)                    \
  __macro(cublasZhbmv)                    \
  __macro(cublasSspmv)                    \
  __macro(cublasDspmv)                    \
  __macro(cublasChpmv)                    \
  __macro(cublasZhpmv)                    \
  __macro(cublasSger)                     \
  __macro(cublasDger)                     \
  __macro(cublasCgeru)                    \
  __macro(cublasCgerc)                    \
  __macro(cublasZgeru)                    \
  __macro(cublasZgerc)                    \
  __macro(cublasSsyr)                     \
  __macro(cublasDsyr)                     \
  __macro(cublasCsyr)                     \
  __macro(cublasZsyr)                     \
  __macro(cublasCher)                     \
  __macro(cublasZher)                     \
  __macro(cublasSspr)                     \
  __macro(cublasDspr)                     \
  __macro(cublasChpr)                     \
  __macro(cublasZhpr)                     \
  __macro(cublasSsyr2)                    \
  __macro(cublasDsyr2)                    \
  __macro(cublasCsyr2)                    \
  __macro(cublasZsyr2)                    \
  __macro(cublasCher2)                    \
  __macro(cublasZher2)                    \
  __macro(cublasSspr2)                    \
  __macro(cublasDspr2)                    \
  __macro(cublasChpr2)                    \
  __macro(cublasZhpr2)                    \
  __macro(cublasSgemm)                    \
  __macro(cublasDgemm)                    \
  __macro(cublasCgemm)                    \
  __macro(cublasZgemm)                    \
  __macro(cublasSsyrk)                    \
  __macro(cublasDsyrk)                    \
  __macro(cublasCsyrk)                    \
  __macro(cublasZsyrk)                    \
  __macro(cublasCherk)                    \
  __macro(cublasZherk)                    \
  __macro(cublasSsyr2k)                   \
  __macro(cublasDsyr2k)                   \
  __macro(cublasCsyr2k)                   \
  __macro(cublasZsyr2k)                   \
  __macro(cublasCher2k)                   \
  __macro(cublasZher2k)                   \
  __macro(cublasSsyrkx)                   \
  __macro(cublasDsyrkx)                   \
  __macro(cublasCsyrkx)                   \
  __macro(cublasZsyrkx)                   \
  __macro(cublasCherkx)                   \
  __macro(cublasZherkx)                   \
  __macro(cublasSsymm)                    \
  __macro(cublasDsymm)                    \
  __macro(cublasCsymm)                    \
  __macro(cublasZsymm)                    \
  __macro(cublasChemm)                    \
  __macro(cublasZhemm)                    \
  __macro(cublasStrsm)                    \
  __macro(cublasDtrsm)                    \
  __macro(cublasCtrsm)                    \
  __macro(cublasZtrsm)                    \
  __macro(cublasStrmm)                    \
  __macro(cublasDtrmm)                    \
  __macro(cublasCtrmm)                    \
  __macro(cublasZtrmm)                    \
  __macro(cublasSgeam)                    \
  __macro(cublasDgeam)                    \
  __macro(cublasCgeam)                    \
  __macro(cublasZgeam)                    \
  __macro(cublasSdgmm)                    \
  __macro(cublasDdgmm)                    \
  __macro(cublasCdgmm)                    \
  __macro(cublasZdgmm)

PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasCreate)
PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasDestroy)
PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasSetStream)
PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasSetPointerMode)
PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasGetPointerMode)
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasSgemmBatched)
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasDgemmBatched)
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasCgemmBatched)
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasZgemmBatched)
CUBLAS_BLAS_ROUTINE_EACH(PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP)

#if ROCM_VERSION >= 7050
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasSgemmEx)
#endif

#if ROCM_VERSION >= 8000
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasGemmEx)
#endif

}  // namespace wrap

static string ToString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
#if ROCM_VERSION >= 8000
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    default:
      return port::StrCat("<invalid cublas status: ", status, ">");
  }
}

// cuBLAS has interfaces that permit pointers to be passed from either the host
// memory space or the device memory space; however, you must instruct it as to
// which address space those pointers are in with cublasSetPointerMode.
//
// This helper sets the cuBLAS pointer mode to a desired value for a cuBLAS call
// you are about to perform in a given scope.
//
// The prior cuBLAS pointer mode is retained and restored when this object goes
// out of scope.
class ScopedCublasPointerMode {
 public:
  // Note that, because the setting of the cublas pointer mode is fallible,
  // construction of this scoped datatype must be paired with a call to
  // Init().
  //
  // Parameters:
  //  handle: The cublas library handle to act upon in setting the pointer mode.
  explicit ScopedCublasPointerMode(ROCMExecutor *parent, cublasHandle_t handle)
      : parent_(parent), handle_(handle), ok_(false) {}

  // Attempts the switch to the requested scoped pointer mode, new_mode.
  //
  // Note that when false is returned, an appropriate error has already been
  // logged.
  bool Init(cublasPointerMode_t new_mode) {
    cublasStatus_t ret =
        wrap::cublasGetPointerMode(parent_, handle_, &old_mode_);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to get old cublas pointer mode: " << ToString(ret);
      return ok_ = false;
    }

    ret = wrap::cublasSetPointerMode(parent_, handle_, new_mode);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to set new cublas pointer mode: " << ToString(ret);
      return ok_ = false;
    }

    return ok_ = true;
  }

  // Switches back to the prior pointer mode, if the switch operation was
  // successful in the first place.
  ~ScopedCublasPointerMode() {
    if (ok_) {
      cublasStatus_t ret =
          wrap::cublasSetPointerMode(parent_, handle_, old_mode_);
      if (ret != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set former cublas pointer mode: "
                   << ToString(ret);
      }
    }
  }

 private:
  ROCMExecutor *parent_;   // Executor establishing this pointer mode for.
  cublasHandle_t handle_;  // Handle to the cuBLAS instance of interest.
  cublasPointerMode_t old_mode_;  // Prior cuBLAS pointer mode, to be restored.
  bool ok_;                       // Whether the change was successful.
};

bool ROCMBlas::Init() {
  cublasStatus_t ret = wrap::cublasCreate(parent_, &blas_);
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create cublas handle: " << ToString(ret);
    return false;
  }

  return true;
}

ROCMBlas::ROCMBlas(rocm::ROCMExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)), blas_(nullptr) {}

ROCMBlas::~ROCMBlas() {
  if (blas_ != nullptr) {
    wrap::cublasDestroy(parent_, blas_);
  }
}

bool ROCMBlas::SetStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsROCMStreamValue(stream) != nullptr);
  CHECK(blas_ != nullptr);
  cublasStatus_t ret =
      wrap::cublasSetStream(parent_, blas_, AsROCMStreamValue(stream));
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for cuBLAS calls: " << ToString(ret);
    return false;
  }

  return true;
}

namespace {

// Helper functions transforming blas arguments into cuBLAS arguments.

cublasOperation_t ROCMBlasTranspose(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return CUBLAS_OP_N;
    case blas::Transpose::kTranspose:
      return CUBLAS_OP_T;
    case blas::Transpose::kConjugateTranspose:
      return CUBLAS_OP_C;
    default:
      LOG(FATAL) << "Invalid value of blas::Transpose.";
  }
}

cublasFillMode_t ROCMBlasUpperLower(blas::UpperLower uplo) {
  switch (uplo) {
    case blas::UpperLower::kUpper:
      return CUBLAS_FILL_MODE_UPPER;
    case blas::UpperLower::kLower:
      return CUBLAS_FILL_MODE_LOWER;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

cublasDiagType_t ROCMBlasDiagonal(blas::Diagonal diag) {
  switch (diag) {
    case blas::Diagonal::kUnit:
      return CUBLAS_DIAG_UNIT;
    case blas::Diagonal::kNonUnit:
      return CUBLAS_DIAG_NON_UNIT;
    default:
      LOG(FATAL) << "Invalid value of blas::Diagonal.";
  }
}

cublasSideMode_t ROCMBlasSide(blas::Side side) {
  switch (side) {
    case blas::Side::kLeft:
      return CUBLAS_SIDE_LEFT;
    case blas::Side::kRight:
      return CUBLAS_SIDE_RIGHT;
    default:
      LOG(FATAL) << "Invalid value of blas::Side.";
  }
}

// ROCMDataType<T>::type translates from a C++ type (e.g. float) to a
// rocmDataType_t (e.g. ROCM_R_32F).  ROCMComputationType(ty) translates from a
// blas::ComputationType to a rocmDataType_t.
//
// These are used to build the argument type and computation type args to
// cublasGemmEx.  cublasGemmEx and rocmDataType_t are available only on
// ROCM >= 8.0.
#if ROCM_VERSION >= 8000
template <typename T>
struct ROCMDataType;

template <>
struct ROCMDataType<Eigen::half> {
  static constexpr rocmDataType_t type = SE_ROCM_DATA_HALF;
};

template <>
struct ROCMDataType<std::complex<Eigen::half>> {
  static constexpr rocmDataType_t type = ROCM_C_16F;
};

template <>
struct ROCMDataType<float> {
  static constexpr rocmDataType_t type = ROCM_R_32F;
};

template <>
struct ROCMDataType<std::complex<float>> {
  static constexpr rocmDataType_t type = ROCM_C_32F;
};

template <>
struct ROCMDataType<double> {
  static constexpr rocmDataType_t type = ROCM_R_64F;
};

template <>
struct ROCMDataType<std::complex<double>> {
  static constexpr rocmDataType_t type = ROCM_C_64F;
};

template <>
struct ROCMDataType<int> {
  static constexpr rocmDataType_t type = ROCM_R_32I;
};

template <>
struct ROCMDataType<int8> {
  static constexpr rocmDataType_t type = ROCM_R_8I;
};

template <>
struct ROCMDataType<std::complex<int8>> {
  static constexpr rocmDataType_t type = ROCM_C_8I;
};

template <>
struct ROCMDataType<uint8> {
  static constexpr rocmDataType_t type = ROCM_R_8U;
};

template <>
struct ROCMDataType<std::complex<uint8>> {
  static constexpr rocmDataType_t type = ROCM_C_8U;
};

rocmDataType_t ROCMComputationType(blas::ComputationType ty) {
  switch (ty) {
    case blas::ComputationType::kF16:
      return ROCM_R_16F;
    case blas::ComputationType::kF32:
      return ROCM_R_32F;
    case blas::ComputationType::kF64:
      return ROCM_R_64F;
    case blas::ComputationType::kI32:
      return ROCM_R_32I;
    case blas::ComputationType::kComplexF32:
      return ROCM_C_32F;
    case blas::ComputationType::kComplexF64:
      return ROCM_C_64F;
  }
}
#endif

}  // namespace

template <typename FuncT, typename... Args>
bool ROCMBlas::DoBlasInternalImpl(FuncT cublas_func, Stream *stream,
                                  bool pointer_mode_host, bool err_on_failure,
                                  Args... args) {
  mutex_lock lock{mu_};

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return false;
  }

  ScopedCublasPointerMode pointer_mode{parent_, blas_};
  if (!pointer_mode.Init(pointer_mode_host ? CUBLAS_POINTER_MODE_HOST
                                           : CUBLAS_POINTER_MODE_DEVICE)) {
    return false;
  }

  cublasStatus_t ret = cublas_func(parent_, blas_, args...);
  if (err_on_failure && ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to run cuBLAS routine " << cublas_func.kName << ": "
               << ToString(ret);
  }
  return ret == CUBLAS_STATUS_SUCCESS;
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(wrap::cublasSasum, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(wrap::cublasDasum, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(
      wrap::cublasScasum, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(
      wrap::cublasDzasum, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::cublasSaxpy, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        ROCMMemory(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::cublasDaxpy, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        ROCMMemory(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(wrap::cublasCaxpy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMComplex(&alpha), ROCMComplex(ROCMMemory(x)), incx,
                        ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(wrap::cublasZaxpy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMComplex(&alpha), ROCMComplex(ROCMMemory(x)), incx,
                        ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::cublasScopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::cublasDcopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(wrap::cublasCcopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMComplex(ROCMMemory(x)), incx,
                        ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(wrap::cublasZcopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMComplex(ROCMMemory(x)), incx,
                        ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *result) {
  return DoBlasInternal(
      wrap::cublasSdot, stream, false /* = pointer_mode_host */, elem_count,
      ROCMMemory(x), incx, ROCMMemory(y), incy, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *result) {
  return DoBlasInternal(
      wrap::cublasDdot, stream, false /* = pointer_mode_host */, elem_count,
      ROCMMemory(x), incx, ROCMMemory(y), incy, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  return DoBlasInternal(
      wrap::cublasCdotc, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
      ROCMComplex(ROCMMemoryMutable(result)));
}

bool ROCMBlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  return DoBlasInternal(
      wrap::cublasZdotc, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
      ROCMComplex(ROCMMemoryMutable(result)));
}

bool ROCMBlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  return DoBlasInternal(
      wrap::cublasCdotu, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
      ROCMComplex(ROCMMemoryMutable(result)));
}

bool ROCMBlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  return DoBlasInternal(
      wrap::cublasZdotu, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
      ROCMComplex(ROCMMemoryMutable(result)));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(wrap::cublasSnrm2, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(wrap::cublasDnrm2, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(
      wrap::cublasScnrm2, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(
      wrap::cublasDznrm2, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<float> *x, int incx,
                         DeviceMemory<float> *y, int incy, float c, float s) {
  return DoBlasInternal(
      wrap::cublasSrot, stream, true /* = pointer_mode_host */, elem_count,
      ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy, &c, &s);
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<double> *x, int incx,
                         DeviceMemory<double> *y, int incy, double c,
                         double s) {
  return DoBlasInternal(
      wrap::cublasDrot, stream, true /* = pointer_mode_host */, elem_count,
      ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy, &c, &s);
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<float>> *x, int incx,
                         DeviceMemory<std::complex<float>> *y, int incy,
                         float c, float s) {
  return DoBlasInternal(wrap::cublasCsrot, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMComplex(ROCMMemoryMutable(x)), incx,
                        ROCMComplex(ROCMMemoryMutable(y)), incy, &c, &s);
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<double>> *x, int incx,
                         DeviceMemory<std::complex<double>> *y, int incy,
                         double c, double s) {
  return DoBlasInternal(wrap::cublasZdrot, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMComplex(ROCMMemoryMutable(x)), incx,
                        ROCMComplex(ROCMMemoryMutable(y)), incy, &c, &s);
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<float> *a,
                          DeviceMemory<float> *b, DeviceMemory<float> *c,
                          DeviceMemory<float> *s) {
  return DoBlasInternal(wrap::cublasSrotg, stream,
                        false /* = pointer_mode_host */, ROCMMemoryMutable(a),
                        ROCMMemoryMutable(b), ROCMMemoryMutable(c),
                        ROCMMemoryMutable(s));
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<double> *a,
                          DeviceMemory<double> *b, DeviceMemory<double> *c,
                          DeviceMemory<double> *s) {
  return DoBlasInternal(wrap::cublasDrotg, stream,
                        false /* = pointer_mode_host */,
                        ROCMComplex(ROCMMemoryMutable(a)), ROCMMemoryMutable(b),
                        ROCMMemoryMutable(c), ROCMMemoryMutable(s));
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<float>> *a,
                          DeviceMemory<std::complex<float>> *b,
                          DeviceMemory<float> *c,
                          DeviceMemory<std::complex<float>> *s) {
  return DoBlasInternal(
      wrap::cublasCrotg, stream, false /* = pointer_mode_host */,
      ROCMComplex(ROCMMemoryMutable(a)), ROCMComplex(ROCMMemoryMutable(b)),
      ROCMComplex(ROCMMemoryMutable(c)), ROCMComplex(ROCMMemoryMutable(s)));
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<double>> *a,
                          DeviceMemory<std::complex<double>> *b,
                          DeviceMemory<double> *c,
                          DeviceMemory<std::complex<double>> *s) {
  return DoBlasInternal(
      wrap::cublasZrotg, stream, false /* = pointer_mode_host */,
      ROCMComplex(ROCMMemoryMutable(a)), ROCMComplex(ROCMMemoryMutable(b)),
      ROCMComplex(ROCMMemoryMutable(c)), ROCMComplex(ROCMMemoryMutable(s)));
}

bool ROCMBlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy,
                          const DeviceMemory<float> &param) {
  return DoBlasInternal(wrap::cublasSrotm, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy,
                        ROCMMemory(param));
}

bool ROCMBlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy,
                          const DeviceMemory<double> &param) {
  return DoBlasInternal(wrap::cublasDrotm, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy,
                        ROCMMemory(param));
}

bool ROCMBlas::DoBlasRotmg(Stream *stream, DeviceMemory<float> *d1,
                           DeviceMemory<float> *d2, DeviceMemory<float> *x1,
                           const DeviceMemory<float> &y1,
                           DeviceMemory<float> *param) {
  return DoBlasInternal(wrap::cublasSrotmg, stream,
                        false /* = pointer_mode_host */, ROCMMemoryMutable(d1),
                        ROCMMemoryMutable(d2), ROCMMemoryMutable(x1),
                        ROCMMemory(y1), ROCMMemoryMutable(param));
}

bool ROCMBlas::DoBlasRotmg(Stream *stream, DeviceMemory<double> *d1,
                           DeviceMemory<double> *d2, DeviceMemory<double> *x1,
                           const DeviceMemory<double> &y1,
                           DeviceMemory<double> *param) {
  return DoBlasInternal(wrap::cublasDrotmg, stream,
                        false /* = pointer_mode_host */, ROCMMemoryMutable(d1),
                        ROCMMemoryMutable(d2), ROCMMemoryMutable(x1),
                        ROCMMemory(y1), ROCMMemoryMutable(param));
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(wrap::cublasSscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::cublasDscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(
      wrap::cublasCsscal, stream, true /* = pointer_mode_host */, elem_count,
      ROCMComplex(&alpha), ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(
      wrap::cublasZdscal, stream, true /* = pointer_mode_host */, elem_count,
      ROCMComplex(&alpha), ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(
      wrap::cublasCscal, stream, true /* = pointer_mode_host */, elem_count,
      ROCMComplex(&alpha), ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(
      wrap::cublasZscal, stream, true /* = pointer_mode_host */, elem_count,
      ROCMComplex(&alpha), ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::cublasSswap, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::cublasDswap, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<float>> *x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(wrap::cublasCswap, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMComplex(ROCMMemoryMutable(x)), incx,
                        ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<double>> *x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(wrap::cublasZswap, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMComplex(ROCMMemoryMutable(x)), incx,
                        ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(wrap::cublasIsamax, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(wrap::cublasIdamax, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::cublasIcamax, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::cublasIzamax, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::cublasIsamin, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::cublasIdamin, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::cublasIcamin, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::cublasIzamin, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasSgbmv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, kl, ku, &alpha, ROCMMemory(a), lda,
      ROCMMemory(x), incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasDgbmv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, kl, ku, &alpha, ROCMMemory(a), lda,
      ROCMMemory(x), incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasCgbmv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, kl, ku, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasZgbmv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, kl, ku, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasSgemv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, &alpha, ROCMMemory(a), lda, ROCMMemory(x),
      incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasDgemv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, &alpha, ROCMMemory(a), lda, ROCMMemory(x),
      incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasCgemv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasZgemv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, float alpha,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasSger, stream, true /* = pointer_mode_host */, m, n, &alpha,
      ROCMMemory(x), incx, ROCMMemory(y), incy, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, double alpha,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasDger, stream, true /* = pointer_mode_host */, m, n, &alpha,
      ROCMMemory(x), incx, ROCMMemory(y), incy, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasCgerc, stream, true /* = pointer_mode_host */, m, n,
      ROCMComplex(&alpha), ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(ROCMMemory(y)), incy, ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasZgerc, stream, true /* = pointer_mode_host */, m, n,
      ROCMComplex(&alpha), ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(ROCMMemory(y)), incy, ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasCgeru, stream, true /* = pointer_mode_host */, m, n,
      ROCMComplex(&alpha), ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(ROCMMemory(y)), incy, ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasZgeru, stream, true /* = pointer_mode_host */, m, n,
      ROCMComplex(&alpha), ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(ROCMMemory(y)), incy, ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasChbmv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, k, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasZhbmv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, k, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasChemv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasZhemv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasCher, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, &alpha, ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasZher, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, &alpha, ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasCher2, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
      ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasZher2, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
      ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &ap,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasChpmv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(ap)), ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &ap,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasZhpmv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(ap)), ROCMComplex(ROCMMemory(x)), incx,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *ap) {
  return DoBlasInternal(
      wrap::cublasChpr, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemoryMutable(ap)));
}

bool ROCMBlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *ap) {
  return DoBlasInternal(
      wrap::cublasZhpr, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemoryMutable(ap)));
}

bool ROCMBlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *ap) {
  return DoBlasInternal(
      wrap::cublasChpr2, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
      ROCMComplex(ROCMMemoryMutable(ap)));
}

bool ROCMBlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *ap) {
  return DoBlasInternal(
      wrap::cublasZhpr2, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
      ROCMComplex(ROCMMemoryMutable(ap)));
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasSsbmv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, k, &alpha, ROCMMemory(a), lda, ROCMMemory(x),
      incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasDsbmv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), n, k, &alpha, ROCMMemory(a), lda, ROCMMemory(x),
      incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &ap,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::cublasSspmv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(ap),
                        ROCMMemory(x), incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &ap,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::cublasDspmv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(ap),
                        ROCMMemory(x), incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *ap) {
  return DoBlasInternal(wrap::cublasSspr, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
                        incx, ROCMMemoryMutable(ap));
}

bool ROCMBlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *ap) {
  return DoBlasInternal(wrap::cublasDspr, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
                        incx, ROCMMemoryMutable(ap));
}

bool ROCMBlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *ap) {
  return DoBlasInternal(wrap::cublasSspr2, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
                        incx, ROCMMemory(y), incy, ROCMMemoryMutable(ap));
}

bool ROCMBlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *ap) {
  return DoBlasInternal(wrap::cublasDspr2, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
                        incx, ROCMMemory(y), incy, ROCMMemoryMutable(ap));
}

bool ROCMBlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::cublasSsymv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(a), lda,
                        ROCMMemory(x), incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::cublasDsymv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(a), lda,
                        ROCMMemory(x), incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(wrap::cublasSsyr, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
                        incx, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *a, int lda) {
  return DoBlasInternal(wrap::cublasDsyr, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
                        incx, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(wrap::cublasSsyr2, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
                        incx, ROCMMemory(y), incy, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *a, int lda) {
  return DoBlasInternal(wrap::cublasDsyr2, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
                        incx, ROCMMemory(y), incy, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(wrap::cublasStbmv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, k, ROCMMemory(a), lda,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::cublasDtbmv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, k, ROCMMemory(a), lda,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  return DoBlasInternal(
      wrap::cublasCtbmv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, k, ROCMComplex(ROCMMemory(a)), lda,
      ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  return DoBlasInternal(
      wrap::cublasZtbmv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, k, ROCMComplex(ROCMMemory(a)), lda,
      ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(wrap::cublasStbsv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, k, ROCMMemory(a), lda,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::cublasDtbsv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, k, ROCMMemory(a), lda,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  return DoBlasInternal(
      wrap::cublasCtbsv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, k, ROCMComplex(ROCMMemory(a)), lda,
      ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  return DoBlasInternal(
      wrap::cublasZtbsv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, k, ROCMComplex(ROCMMemory(a)), lda,
      ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  return DoBlasInternal(
      wrap::cublasStpmv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, ROCMMemory(ap), ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(
      wrap::cublasDtpmv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, ROCMMemory(ap), ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::cublasCtpmv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(ap)),
                        ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::cublasZtpmv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(ap)),
                        ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  return DoBlasInternal(
      wrap::cublasStpsv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, ROCMMemory(ap), ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(
      wrap::cublasDtpsv, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, ROCMMemory(ap), ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::cublasCtpsv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(ap)),
                        ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::cublasZtpsv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(ap)),
                        ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(wrap::cublasStrmv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, ROCMMemory(a), lda,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::cublasDtrmv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, ROCMMemory(a), lda,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::cublasCtrmv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(a)),
                        lda, ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::cublasZtrmv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(a)),
                        lda, ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(wrap::cublasStrsv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, ROCMMemory(a), lda,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::cublasDtrsv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, ROCMMemory(a), lda,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::cublasCtrsv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(a)),
                        lda, ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::cublasZtrsv, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(a)),
                        lda, ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasGemm(
    Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64 m, uint64 n, uint64 k,
    float alpha, const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc) {
#if ROCM_VERSION >= 7050
  VLOG(1) << port::Printf(
      "doing cuBLAS SGEMM: at=%d bt=%d m=%llu n=%llu "
      "k=%llu alpha=%f a=%p lda=%d b=%p ldb=%d beta=%f "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);
  if (transa == blas::Transpose::kNoTranspose) {
    if (lda < static_cast<int64>(m)) {
      LOG(WARNING) << "GEMM lda was smaller than m (no transpose case); "
                      "precondition violation";
    }
  } else {
    if (lda < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM lda (" << lda << ") was smaller than k (" << k
                   << ") (transpose case); precondition violation";
    }
  }
  if (transb == blas::Transpose::kNoTranspose) {
    if (ldb < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM ldb (" << ldb << ") was smaller than k (" << k
                   << ") (no transpose case); precondition violation";
    }
  } else {
    if (ldb < static_cast<int64>(n)) {
      LOG(WARNING) << "GEMM ldb was smaller than n (transpose case); "
                      "precondition violation";
    }
  }
  // TODO(sesse): Consider supporting the Hgemm interface, which uses half
  // calculations internally (faster on newer devices, such as Pascal and TX1,
  // but less precise).
  return DoBlasInternal(
      wrap::cublasSgemmEx, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k, &alpha,
      ROCMMemory(a), SE_ROCM_DATA_HALF, lda, ROCMMemory(b), SE_ROCM_DATA_HALF,
      ldb, &beta, ROCMMemoryMutable(c), SE_ROCM_DATA_HALF, ldc);
#else
  LOG(ERROR) << "fp16 sgemm is not implemented in this cuBLAS version "
             << "(need at least ROCM 7.5)";
  return false;
#endif
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
  VLOG(1) << port::Printf(
      "doing cuBLAS SGEMM: at=%d bt=%d m=%llu n=%llu "
      "k=%llu alpha=%f a=%p lda=%d b=%p ldb=%d beta=%f "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);
  if (transa == blas::Transpose::kNoTranspose) {
    if (lda < static_cast<int64>(m)) {
      LOG(WARNING) << "GEMM lda was smaller than m (no transpose case); "
                      "precondition violation";
    }
  } else {
    if (lda < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM lda (" << lda << ") was smaller than k (" << k
                   << ") (transpose case); precondition violation";
    }
  }
  if (transb == blas::Transpose::kNoTranspose) {
    if (ldb < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM ldb (" << ldb << ") was smaller than k (" << k
                   << ") (no transpose case); precondition violation";
    }
  } else {
    if (ldb < static_cast<int64>(n)) {
      LOG(WARNING) << "GEMM ldb was smaller than n (transpose case); "
                      "precondition violation";
    }
  }
  return DoBlasInternal(
      wrap::cublasSgemm, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k, &alpha,
      ROCMMemory(a), lda, ROCMMemory(b), ldb, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasDgemm, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k, &alpha,
      ROCMMemory(a), lda, ROCMMemory(b), ldb, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasCgemm, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
      ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
      ROCMComplex(ROCMMemory(b)), ldb, ROCMComplex(&beta),
      ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasZgemm, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
      ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
      ROCMComplex(ROCMMemory(b)), ldb, ROCMComplex(&beta),
      ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

template <typename InT, typename OutT, typename CompT>
bool ROCMBlas::DoBlasGemmWithAlgorithmImpl(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const CompT &alpha, const DeviceMemory<InT> &a, int lda,
    const DeviceMemory<InT> &b, int ldb, const CompT &beta,
    DeviceMemory<OutT> *c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
// ROCM < version 8 and GPUs < sm_50 don't support cublasGemmEx.
#if ROCM_VERSION < 8000
  return false;
#else
  int cc_major, cc_minor;
  if (stream->parent()->GetDeviceDescription().rocm_compute_capability(
          &cc_major, &cc_minor) &&
      cc_major < 5) {
    return false;
  }

  struct TimerDeleter {
    void operator()(ROCMTimer *t) {
      t->Destroy();
      delete t;
    }
  };
  std::unique_ptr<ROCMTimer, TimerDeleter> timer;
  if (output_profile_result != nullptr) {
    timer.reset(new ROCMTimer(parent_));
    if (!timer->Init() || !timer->Start(AsROCMStream(stream))) {
      return false;
    }
  }

  rocmDataType_t rocm_in_type = ROCMDataType<InT>::type;
  // Since we are converting 'algorithm' to cublasGemmAlgo_t by static_cast,
  // we do the following compile-time check on the default value:
  static_assert(blas::kDefaultGemmAlgo == CUBLAS_GEMM_DFALT, "");
  bool result = DoBlasInternalFailureOK(
      wrap::cublasGemmEx, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k, &alpha,
      ROCMMemory(a), rocm_in_type, lda, ROCMMemory(b), rocm_in_type, ldb, &beta,
      ROCMMemoryMutable(c), ROCMDataType<OutT>::type, ldc,
      ROCMComputationType(computation_type),
      static_cast<cublasGemmAlgo_t>(algorithm));

  if (timer != nullptr && result) {
    // ROCMTimer will CHECK-fail if we Stop() it while the stream is in an error
    // state.
    if (!timer->Stop(AsROCMStream(stream))) {
      return false;
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algorithm);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }
  return result;
#endif
}

bool ROCMBlas::GetBlasGemmAlgorithms(
    std::vector<blas::AlgorithmType> *out_algorithms) {
// cublasGemmAlgo_t (and the function that accepts this type, cublasGemmEx)
// were first introduced in ROCM 8.
#if ROCM_VERSION >= 8000
  for (cublasGemmAlgo_t algo :
       {CUBLAS_GEMM_DFALT, CUBLAS_GEMM_ALGO0, CUBLAS_GEMM_ALGO1,
        CUBLAS_GEMM_ALGO2, CUBLAS_GEMM_ALGO3, CUBLAS_GEMM_ALGO4,
        CUBLAS_GEMM_ALGO5, CUBLAS_GEMM_ALGO6, CUBLAS_GEMM_ALGO7}) {
    out_algorithms->push_back(algo);
  }
#endif
  return true;
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, int alpha, const DeviceMemory<int8> &a, int lda,
    const DeviceMemory<int8> &b, int ldb, int beta, DeviceMemory<int> *c,
    int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const Eigen::half &alpha,
    const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb, const Eigen::half &beta,
    DeviceMemory<Eigen::half> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha, const DeviceMemory<float> &a, int lda,
    const DeviceMemory<float> &b, int ldb, float beta, DeviceMemory<float> *c,
    int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, double alpha, const DeviceMemory<double> &a, int lda,
    const DeviceMemory<double> &b, int ldb, double beta,
    DeviceMemory<double> *c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<float> alpha,
    const DeviceMemory<std::complex<float>> &a, int lda,
    const DeviceMemory<std::complex<float>> &b, int ldb,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<double> alpha,
    const DeviceMemory<std::complex<double>> &a, int lda,
    const DeviceMemory<std::complex<double>> &b, int ldb,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

template <typename T, typename FuncT>
port::Status ROCMBlas::DoBlasGemmBatchedInternal(
    FuncT cublas_func, Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64 m, uint64 n, uint64 k, T alpha,
    const port::ArraySlice<DeviceMemory<T> *> &a_ptrs_to_wrappers, int lda,
    const port::ArraySlice<DeviceMemory<T> *> &b_ptrs_to_wrappers, int ldb,
    T beta, const port::ArraySlice<DeviceMemory<T> *> &c_ptrs_to_wrappers,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  std::vector<T *> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  for (int i = 0; i < batch_count; ++i) {
    a_raw_ptrs.push_back(static_cast<T *>(a_ptrs_to_wrappers[i]->opaque()));
    b_raw_ptrs.push_back(static_cast<T *>(b_ptrs_to_wrappers[i]->opaque()));
    c_raw_ptrs.push_back(static_cast<T *>(c_ptrs_to_wrappers[i]->opaque()));
  }

  typedef typename ROCMComplexT<T>::type ROCM_T;

  const size_t size = batch_count * sizeof(ROCM_T *);

  // Device-side copy of pointers to matrices.
  DeviceMemory<ROCM_T *> a;
  DeviceMemory<ROCM_T *> b;
  DeviceMemory<ROCM_T *> c;

  // If temporary space is allocated for device-side copies of pointers to
  // matrices, that temporary space should not be freed until this function
  // returns. Although the values for these unique_ptrs are not set here, they
  // are declared at this scope so they will be destroyed when the function
  // returns.
  //
  // If a scratch allocator is provided, these pointers will not be used at all.
  std::unique_ptr<TemporaryDeviceMemory<ROCM_T *>> a_temporary;
  std::unique_ptr<TemporaryDeviceMemory<ROCM_T *>> b_temporary;
  std::unique_ptr<TemporaryDeviceMemory<ROCM_T *>> c_temporary;

  // Decide how to allocate device-side copy of pointers to matrices based on
  // whether a scratch allocator was passed.
  if (scratch_allocator != nullptr) {
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> a_bytes,
                        scratch_allocator->AllocateBytes(stream, size));
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> b_bytes,
                        scratch_allocator->AllocateBytes(stream, size));
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> c_bytes,
                        scratch_allocator->AllocateBytes(stream, size));
    a = DeviceMemory<ROCM_T *>(a_bytes);
    b = DeviceMemory<ROCM_T *>(b_bytes);
    c = DeviceMemory<ROCM_T *>(c_bytes);
  } else {
    SE_ASSIGN_OR_RETURN(a_temporary,
                        stream->AllocateTemporaryArray<ROCM_T *>(batch_count));
    SE_ASSIGN_OR_RETURN(b_temporary,
                        stream->AllocateTemporaryArray<ROCM_T *>(batch_count));
    SE_ASSIGN_OR_RETURN(c_temporary,
                        stream->AllocateTemporaryArray<ROCM_T *>(batch_count));
    a = DeviceMemory<ROCM_T *>(*a_temporary->mutable_device_memory());
    b = DeviceMemory<ROCM_T *>(*b_temporary->mutable_device_memory());
    c = DeviceMemory<ROCM_T *>(*c_temporary->mutable_device_memory());
  }

  if (!stream->ThenMemcpy(&a, a_raw_ptrs.data(), size).ok() ||
      !stream->ThenMemcpy(&b, b_raw_ptrs.data(), size).ok() ||
      !stream->ThenMemcpy(&c, c_raw_ptrs.data(), size).ok()) {
    return port::Status(port::error::INTERNAL,
                        "failed to copy memory from host to device in "
                        "ROCMBlas::DoBlasGemmBatched");
  }

  bool ok = DoBlasInternal(
      cublas_func, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
      ROCMComplex(&alpha), const_cast<const ROCM_T **>(ROCMMemory(a)), lda,
      const_cast<const ROCM_T **>(ROCMMemory(b)), ldb, ROCMComplex(&beta),
      const_cast<ROCM_T **>(ROCMMemory(c)), ldc, batch_count);

  if (ok) {
    return port::Status::OK();
  }
  return port::Status(port::error::INTERNAL,
                      "failed BLAS call, see log for details");
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha,
    const port::ArraySlice<DeviceMemory<float> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<float> *> &b_array, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<float> *> &c_array, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::cublasSgemmBatched, stream, transa, transb, m, n, k, alpha, a_array,
      lda, b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, double alpha,
    const port::ArraySlice<DeviceMemory<double> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<double> *> &b_array, int ldb,
    double beta, const port::ArraySlice<DeviceMemory<double> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::cublasDgemmBatched, stream, transa, transb, m, n, k, alpha, a_array,
      lda, b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<float> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b_array,
    int ldb, std::complex<float> beta,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::cublasCgemmBatched, stream, transa, transb, m, n, k, alpha, a_array,
      lda, b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<double> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b_array,
    int ldb, std::complex<double> beta,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::cublasZgemmBatched, stream, transa, transb, m, n, k, alpha, a_array,
      lda, b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasChemm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(b)), ldb,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasZhemm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(b)), ldb,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          float beta, DeviceMemory<std::complex<float>> *c,
                          int ldc) {
  return DoBlasInternal(wrap::cublasCherk, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
                        k, ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
                        &beta, ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          double beta, DeviceMemory<std::complex<double>> *c,
                          int ldc) {
  return DoBlasInternal(wrap::cublasZherk, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
                        k, ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
                        &beta, ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           float beta, DeviceMemory<std::complex<float>> *c,
                           int ldc) {
  return DoBlasInternal(wrap::cublasCher2k, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
                        k, ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
                        ROCMComplex(ROCMMemory(b)), ldb, &beta,
                        ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           double beta, DeviceMemory<std::complex<double>> *c,
                           int ldc) {
  return DoBlasInternal(wrap::cublasZher2k, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
                        k, ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
                        ROCMComplex(ROCMMemory(b)), ldb, &beta,
                        ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasSsymm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, &alpha, ROCMMemory(a),
      lda, ROCMMemory(b), ldb, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasDsymm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, &alpha, ROCMMemory(a),
      lda, ROCMMemory(b), ldb, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasCsymm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(b)), ldb,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasZsymm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(b)), ldb,
      ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          float beta, DeviceMemory<float> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasSsyrk, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k, &alpha,
      ROCMMemory(a), lda, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          double beta, DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasDsyrk, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k, &alpha,
      ROCMMemory(a), lda, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasCsyrk, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k,
      ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(&beta),
      ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasZsyrk, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k,
      ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(&beta),
      ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           float alpha, const DeviceMemory<float> &a, int lda,
                           const DeviceMemory<float> &b, int ldb, float beta,
                           DeviceMemory<float> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasSsyr2k, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k, &alpha,
      ROCMMemory(a), lda, ROCMMemory(b), ldb, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           double alpha, const DeviceMemory<double> &a, int lda,
                           const DeviceMemory<double> &b, int ldb, double beta,
                           DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasDsyr2k, stream, true /* = pointer_mode_host */,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k, &alpha,
      ROCMMemory(a), lda, ROCMMemory(b), ldb, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           std::complex<float> beta,
                           DeviceMemory<std::complex<float>> *c, int ldc) {
  return DoBlasInternal(wrap::cublasCsyr2k, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
                        k, ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
                        ROCMComplex(ROCMMemory(b)), ldb, ROCMComplex(&beta),
                        ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           std::complex<double> beta,
                           DeviceMemory<std::complex<double>> *c, int ldc) {
  return DoBlasInternal(wrap::cublasZsyr2k, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
                        k, ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
                        ROCMComplex(ROCMMemory(b)), ldb, ROCMComplex(&beta),
                        ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return DoBlasInternal(
      wrap::cublasStrmm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
      ROCMBlasDiagonal(diag), m, n, &alpha, ROCMMemory(a), lda,
      ROCMMemoryMutable(b), ldb, ROCMMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return DoBlasInternal(
      wrap::cublasDtrmm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
      ROCMBlasDiagonal(diag), m, n, &alpha, ROCMMemory(a), lda,
      ROCMMemoryMutable(b), ldb, ROCMMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  return DoBlasInternal(
      wrap::cublasCtrmm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
      ROCMBlasDiagonal(diag), m, n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemoryMutable(b)), ldb,
      ROCMComplex(ROCMMemoryMutable(b)), ldb);
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  return DoBlasInternal(
      wrap::cublasZtrmm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
      ROCMBlasDiagonal(diag), m, n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemoryMutable(b)), ldb,
      ROCMComplex(ROCMMemoryMutable(b)), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return DoBlasInternal(wrap::cublasStrsm, stream,
                        true /* = pointer_mode_host */, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, ROCMMemory(a),
                        lda, ROCMMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return DoBlasInternal(wrap::cublasDtrsm, stream,
                        true /* = pointer_mode_host */, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, ROCMMemory(a),
                        lda, ROCMMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  return DoBlasInternal(
      wrap::cublasCtrsm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
      ROCMBlasDiagonal(diag), m, n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemoryMutable(b)), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  return DoBlasInternal(
      wrap::cublasZtrsm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
      ROCMBlasDiagonal(diag), m, n, ROCMComplex(&alpha),
      ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemoryMutable(b)), ldb);
}

}  // namespace rocm

namespace gpu = ::perftools::gputools;

void initialize_cublas() {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::BlasFactory>(
              gpu::rocm::kROCmPlatformId, gpu::rocm::kCuBlasPlugin, "cuBLAS",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::blas::BlasSupport * {
                gpu::rocm::ROCMExecutor *rocm_executor =
                    dynamic_cast<gpu::rocm::ROCMExecutor *>(parent);
                if (rocm_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the cuBLAS "
                      << "support library with a non-ROCM StreamExecutor";
                  return nullptr;
                }

                gpu::rocm::ROCMBlas *blas =
                    new gpu::rocm::ROCMBlas(rocm_executor);
                if (!blas->Init()) {
                  // Note: Init() will log a more specific error.
                  delete blas;
                  return nullptr;
                }
                return blas;
              });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuBLAS factory: "
               << status.error_message();
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::rocm::kROCmPlatformId,
                                                     gpu::PluginKind::kBlas,
                                                     gpu::rocm::kCuBlasPlugin);
}

}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(register_cublas,
                            { perftools::gputools::initialize_cublas(); });
