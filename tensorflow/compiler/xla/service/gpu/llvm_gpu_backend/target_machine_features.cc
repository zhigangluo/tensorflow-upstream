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

#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/target_machine_features.h"
#include "tensorflow/core/lib/strings/str_util.h"


namespace xla {
namespace gpu {

std::string AMDGPUMachineFeatures::simt_intrinsic(const std::string &name) {

    if (tensorflow::str_util::EndsWith(name, ".__thread_id_x")) {
        return "llvm.amdgcn.workitem.id.x";
    } else if (tensorflow::str_util::EndsWith(name, ".__thread_id_y")) {
        return "llvm.amdgcn.workitem.id.y";
    } else if (tensorflow::str_util::EndsWith(name, ".__thread_id_z")) {
        return "llvm.amdgcn.workitem.id.z";
    } else if (tensorflow::str_util::EndsWith(name, ".__thread_id_w")) {
        return "llvm.amdgcn.workitem.id.w";
    } else if (tensorflow::str_util::EndsWith(name, ".__block_id_x")) {
        return "llvm.amdgcn.workgroup.id.x";
    } else if (tensorflow::str_util::EndsWith(name, ".__block_id_y")) {
        return "llvm.amdgcn.workgroup.id.y";
    } else if (tensorflow::str_util::EndsWith(name, ".__block_id_z")) {
        return "llvm.amdgcn.workgroup.id.z";
    } else if (tensorflow::str_util::EndsWith(name, ".__block_id_w")) {
        return "llvm.amdgcn.workgroup.id.w";
    } else if (tensorflow::str_util::EndsWith(name, ".barrier")) {
        return "llvm.amdgcn.s.barrier";
    }
    return "";
}


}  // namespace cpu
}  // namespace xla
