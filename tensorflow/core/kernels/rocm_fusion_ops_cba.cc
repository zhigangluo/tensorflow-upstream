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

#include <string.h>
#include <map>
#include <vector>
#include <memory>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/activation_mode.h"
#include "tensorflow/core/kernels/conv_2d.h" 

namespace tensorflow {

  typedef Eigen::GpuDevice GPUDevice;

  template <typename Device, typename T>
  class ROCmFusionKernelCBA : public OpKernel {
  public :

    explicit ROCmFusionKernelCBA(OpKernelConstruction* ctx)
      : OpKernel(ctx) {

      OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
      
      OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_type_));

      string data_format_str;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
      OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_), errors::InvalidArgument("Invalid data format"));

      string filter_format_str("HWIO");
      OP_REQUIRES(ctx, FilterFormatFromString(filter_format_str, &filter_format_), errors::InvalidArgument("Invalid filter format"));

      OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
      
      string activation_mode_str;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("activation_mode", &activation_mode_str));
      OP_REQUIRES_OK(ctx, GetActivationModeFromString(activation_mode_str, &activation_mode_));
    }

    
    void Compute(OpKernelContext* ctx) override {
      
      VLOG(-1) << "_ROCmFusionKernelCBA invoked!";

      const Tensor& conv_input = ctx->input(0);
      const Tensor& filter = ctx->input(1);
      const Tensor& bias = ctx->input(2);

      const int32 batch_size = GetTensorDim(conv_input, data_format_, 'N');
      const int32 input_rows = GetTensorDim(conv_input, data_format_, 'H');
      const int32 input_cols = GetTensorDim(conv_input, data_format_, 'W');
      const int32 input_channels = GetTensorDim(conv_input, data_format_, 'C');
      
      const int32 filter_rows = GetFilterDim(filter, filter_format_, 'H');
      const int32 filter_cols = GetFilterDim(filter, filter_format_, 'W');
      const int32 output_channels = GetFilterDim(filter, filter_format_, 'O');
      
      int64 output_rows = 0, padding_rows = 0; 
      OP_REQUIRES_OK(ctx, GetWindowedOutputSize(input_rows, filter_rows, strides_[0], padding_type_, &output_rows, &padding_rows));

      int64 output_cols = 0, padding_cols = 0; 
      OP_REQUIRES_OK(ctx, GetWindowedOutputSize(input_cols, filter_cols, strides_[1], padding_type_, &output_cols, &padding_cols));

      TensorShape output_shape = ShapeFromFormat(data_format_, batch_size, output_rows, output_cols, output_channels);
      
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0 /*output index*/, output_shape, &output));

      if (output_shape.num_elements() != 0) {
	// we have got work to do
	// auto* stream = ctx->op_device_context()->stream();
	
	// if (padding_type_ == Padding::SAME) {
	// }

	Tensor* fusion_plan_output = output;
	Tensor temp_input, temp_output;
	if (data_format_ == FORMAT_NHWC) {
	  // allocate a temporary tensor to store the NCHW input
	  TensorShape nchw_shape_input = ShapeFromFormat(FORMAT_NCHW, batch_size, input_rows, input_cols, input_channels);
	  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape_input, &temp_input));

	  // convert the input tensor to NCHW format for the GPU
	  functor::NHWCToNCHW<GPUDevice, T, 4>
	    ()(ctx->eigen_device<GPUDevice>(),
	       conv_input.tensor<T,4>(),
	       temp_input.tensor<T,4>());

	  conv_input = &temp_input;

	  // allocate a temporary tensor to store the NCHW output
	  TensorShape nchw_shape_output = ShapeFromFormat(FORMAT_NCHW, batch_size, output_rows, output_cols, output_channels);
	  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape_output, &temp_output));
	  fusion_plan_output = &temp_output;
	}

	if (filter_format_ == FORMAT_HWIO) {
	  // allocate a temporary tensor to store the OIHW filter
	  TensorShape oihw_shape = ShapeFromFilterFormat(FORMAT_OIHW, filter.shape(), FORMAT_HWIO);
	  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape_input, &temp_input));

	  // convert the input tensor to NCHW format for the GPU
	  functor::NHWCToNCHW<GPUDevice, T, 4>
	    ()(ctx->eigen_device<GPUDevice>(),
	       conv_input.tensor<T,4>(),
	       temp_input.tensor<T,4>());
	}
	







	if (data_format_ == FORMAT_NHWC) {
	  // convert output back to NHWC format
	  functor::NCHWToNHWC<GPUDevice, T, 4>
	    ()(ctx->eigen_device<GPUDevice>(),
	       const_cast<const Tensor*>(fusion_plan_output)->tensor<T,4>(),
	       output->tensor<T,4>());
	}
      }
    }
    
  private:

    std::vector<int32> strides_;
    Padding padding_type_;
    TensorFormat data_format_;
    FilterTensorFormat filter_format_;
    std::vector<int32> dilations_;
    ActivationMode activation_mode_;
  };

  REGISTER_KERNEL_BUILDER(Name("_ROCmFusedConvBiasActv")
			  .Device(DEVICE_GPU)	
			  .TypeConstraint<float>("T"),
			  ROCmFusionKernelCBA<GPUDevice, float>);
  
} // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM
