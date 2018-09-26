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

#ifdef TENSORFLOW_USE_ROCM

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "tensorflow/core/kernels/rocm_fusion_ops.h"
#include "tensorflow/core/kernels/conv_2d.h" 
#include "tensorflow/core/kernels/conv_ops_gpu.h"

#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/activation_mode.h"

#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

  se::dnn::ActivationMode GetDnnActivationMode(ActivationMode activation_mode) {
    
    se::dnn::ActivationMode dnn_activation_mode;
    switch (activation_mode) {
    case ActivationMode::NONE:
      dnn_activation_mode = se::dnn::ActivationMode::kNone;
      break;
    case ActivationMode::RELU:
      dnn_activation_mode = se::dnn::ActivationMode::kRelu;
      break;
    default:
      LOG(FATAL) << "Activation mode " << activation_mode << " not supported";
    }

    return dnn_activation_mode;
  }


} // namespace tensorflow


#endif // TENSORFLOW_USE_ROCM
