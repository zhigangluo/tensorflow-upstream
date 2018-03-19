# TensorFlow ROCm port high-level design document

## Introduction

This document serves as the overall document to explain what was changed to allow TensorFlow 1.3.0 running on ROCm platform.

In this port efforts were made to try ensure logic for existing CUDA / NVPTX path stay as-is. Places where platform neutrality were broken are marked as **XXX**.

---
## Make system

- **configure**:
-- set default value of `TF_ENABLE_XLA` to *1*
-- added `TF_NEED_ROCM`, default value set to *1*
-- added `ROCM_TOOLKIT_PATH`, default value set to */opt/rocm*
- **third_party/gpus**:
-- added **rocm_configure.bzl** for ROCm platform
-- added **rocm/** directory for custom bazel functions for ROCm platform
-- added **crosstool/CROSSTOOL_hipcc.tpl** to add a new crosstool toolchain be used by **rocm_configure.bzl**
-- added **crosstool/clang/bin/crosstool_wrapper_driver_rocm.tpl** as the wrapper for compiler and linker
- **tensorflow/workspace.bzl**:
-- adopted `rocm_configrue()` to be ROCm-aware
-- changed how Eigen is fetched to cope with HIP on ROCm platform
-- removed some dead links
-- used LLVM 5.0.1 for XLA
- **tensorflow/tensorflow.bzl**:
-- renamed `tf_cuda_library()` to `tf_gpu_library`
-- renamed `cuda_py_tests()` to `gpu_py_tests()`
-- renamed `tf_cuda_test_tags()` to `tf_gpu_tests_tags()`
-- **XXX**: temporarily disabled dependencies introduced by `if_cuda()`
- **BUILD** files within TensorFlow directories
-- adopted naming changes introduced in **tensorflow/tensorflow.bzl**
-- added logic to load ROCm-specific functions such as `if_rocm()` or `if_rocm_is_configured()`

---
## Runtime

---
## GPU kernel implementation

[List of supported operators](rocm_docs/core_kernels.md)

---
## XLA

- **tensorflow/compiler/jit/xla_gpu_device.cc**
-- **XXX**: disable registering XLA devices for CUDA
- **tensorflow/compiler/xla/service/computation_placer.cc**
-- register for ROCm platofmr
- **tensorflow/compiler/xla/service/generic_transfer_manager.cc**
-- register for ROCm platform
- added the following files for AMDGPU backend
-- **tensorflow/compiler/xla/service/gpu/amdgpu_compiler.cc**
-- **tensorflow/compiler/xla/service/gpu/amdgpu_compiler.h**
-- **tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/amdgpu_backend_lib.cc**
-- **tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/amdgpu_backend_lib.h**
- renamed the following files for NVPTX backend
-- **tensorflow/compiler/xla/service/gpu/gpu_compiler.cc** to **tensorflow/compiler/xla/service/gpu/nvptx_compiler.cc**
-- **tensorflow/compiler/xla/service/gpu/gpu_compiler.h** to **tensorflow/compiler/xla/service/gpu/nvptx_compiler.h**
-- **tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h** to **tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/nvptx_backend_lib.h**
-- **tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc** to **tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/nvptx_backend_lib.hcc**
- **tensorflow/compiler/xla/service/gpu/BUILD**
-- modified rule `gpu_compiler()` to cope with file name changes and added depedency to *ROCm Device Libs*
- **tensorflow/compiler/xla/service/gpu/convolution_thunk.cc**
-- **XXX**: removed autotuning logic for *CuDNN*
- **tensorflow/compiler/xla/service/gpu/elemenmtal_ir_emitter.cc**
-- **XXX**: replaced logic dependent to *Libdevice* to *ROCm Device Libs*
- **tensorflow/compiler/xla/service/gpu/gpu_executable.cc**, **tensorflow/compiler/xla/service/gpu/gpu_executable.h**, **tensorflow/compiler/xla/service/gpu/kernel_thunk.cc**
-- renamed *ptx* to *text*
- **tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.cc**
-- **XXX**: changed logic to use `llvm::ConstantExpr::getAddrSpaceCast()` due to address space differences in AMDGPU
- **tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc**, **tensorflow/compiler/xla/service/gpu/ir_emission_utils.h**
-- **XXX**: replaced logic dependent to *Libdevice* to *ROCm Device Libs*
- **tensorflow/compiler/xla/service/gpu/ir_emitter.cc**
-- **XXX**: modified logic to cope with AMDGPU memory address spaces
- **tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.cc**
-- **XXX**: disabled NVPTX-specific logic
-- **XXX**: replaced logic dependent to *Libdevice* to *ROCm Device Libs*
- **tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.cc**
-- **XXX**: replaced logic dependent to *Libdevice* to *ROCm Device Libs*
- **tensorflow/compiler/xla/service/platform_util.cc**
-- added logic to check ROCm platform ISA version

