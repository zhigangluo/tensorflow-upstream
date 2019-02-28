/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/amdgpu_compiler.h"

#include <stdlib.h>
#include <atomic>
#include <functional>
#include <mutex>  // NOLINT(build/c++11): only using std::call_once, not mutex.
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/convolution_group_converter.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/gpu/amdgpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_fused_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_padding_legalization.h"
#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_copy_insertion.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_support_checker.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_layout_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_sanitize_constant_names.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/amdgpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/miopen_conv_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_element_type_converter.h"
#include "tensorflow/compiler/xla/service/hlo_get_dimension_size_rewriter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_target_features.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/reduce_precision_insertion.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/rocm_rocdl_path.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/platform/tracing.h"

namespace xla {
namespace gpu {

/* static */ const char* AMDGPUCompiler::kTargetTriple = "amdgcn-amd-amdhsa";
/* static */ const char* AMDGPUCompiler::kDataLayout =
         "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32"
         "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
         "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5";

namespace {

namespace tracing = tensorflow::tracing;

// Returns the directory containing ROCm-Device-Libs files. This function is
// called in AMDGPUCompiler's constructor, so can't return an error. But
// AMDGPUCompiler::Compile will return an error when the wanted rocdl file
// doesn't exist in the folder this function returns.
string GetROCDLDir(const HloModuleConfig& config) {
  std::vector<string> potential_rocdl_dirs;
  const string datadir = config.debug_options().xla_gpu_cuda_data_dir();
  if (!datadir.empty()) {
    potential_rocdl_dirs.push_back(datadir);
  }
  potential_rocdl_dirs.push_back(tensorflow::ROCDLRoot());

  // Tries all potential ROCDL directories in the order they are inserted.
  // Returns the first directory that exists in the file system.
  for (const string& potential_rocdl_dir : potential_rocdl_dirs) {
    if (tensorflow::Env::Default()->IsDirectory(potential_rocdl_dir).ok()) {
      VLOG(2) << "Found ROCm-Device-Libs dir " << potential_rocdl_dir;
      return potential_rocdl_dir;
    }
    VLOG(2) << "Unable to find potential ROCm-Device-Libs dir "
            << potential_rocdl_dir;
  }

  // Last resort: maybe in the current folder.
  return ".";
}

// Runs optimization passes on the given HLO module.
//
// It takes a compiler pointer, as passes may compile and execute HLOs on the
// fly for cuDNN verification or other purposes.
Status OptimizeHloModule(HloModule* hlo_module, se::StreamExecutor* stream_exec,
                         DeviceMemoryAllocator* device_allocator,
                         Compiler* compiler) {
  {
    HloPassPipeline pipeline("optimization");
    pipeline.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                              /*allow_mixed_precision=*/false);
    pipeline.AddPass<DynamicIndexSplitter>();
    pipeline.AddPass<GpuHloSupportChecker>();
    ReducePrecisionInsertion::AddPasses(
        &pipeline, hlo_module->config().debug_options(),
        ReducePrecisionInsertion::PassTiming::BEFORE_OPTIMIZATION);

    // TODO(b/64094172): make Call work on GPU instead of inlining.
    pipeline.AddPass<CallInliner>();
    auto cost_model = [](HloInstruction* conv) {
      // We need a cost model for GPUs. Currently, do nothing.
      return false;
    };
    pipeline.AddPass<DotDecomposer>(false);
    pipeline.AddPass<ConvolutionGroupConverter>(
        cost_model,
        /*convert_batch_groups_only=*/true);
    // Convert BF16 operations to F32 operations so that the GPU backend can
    // support BF16 operations without directly implementing a BF16 lowering for
    // most ops.
    pipeline.AddPass<HloElementTypeConverter>(BF16, F32);

    {
      auto& pass =
          pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification");
      pass.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                            /*allow_mixed_precision=*/false);

      // If cudnn batchnorms are enabled, rewrite batchnorm HLOs to cudnn calls
      // where possible.  Not every batchnorm op can be implemented as a call to
      // cudnn, so decompose any remaining batchnorm ops into a soup of HLOs.
      if (hlo_module->config().debug_options().xla_gpu_use_cudnn_batchnorm()) {
        pass.AddPass<CudnnBatchNormRewriter>();
      }
      pass.AddPass<BatchNormExpander>(
          /*rewrite_training_op=*/true,
          /*rewrite_inference_op=*/true,
          /*rewrite_grad_op=*/true);

      pipeline.AddPass<HloGetDimensionSizeRewriter>();

      // BatchNormExpander can create zero-sized ops, so zero-sized HLO
      // elimination has to come after that pass.
      pipeline.AddPass<ZeroSizedHloElimination>();

      AlgebraicSimplifierOptions options;
      pass.AddPass<AlgebraicSimplifier>(options);
      pass.AddPass<SortSimplifier>();
      pass.AddPass<TupleSimplifier>();
      pass.AddPass<WhileLoopConstantSinking>();
      pass.AddPass<WhileLoopSimplifier>();
      pass.AddPass<HloDCE>();
      pass.AddPass<ReshapeMover>();
      pass.AddPass<HloConstantFolding>();
      pass.AddPass<ConditionalSimplifier>();

    }

    pipeline.AddPass<TransposeFolding>(
        [](const HloInstruction& dot,
           const TransposeFolding::OperandIndices& candidate_operands) {
          return ImplementedAsGemm(dot) ? candidate_operands
                                        : TransposeFolding::OperandIndices{};
        },
        TransposeFolding::NeverFoldTranspose);
    pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
    pipeline.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  {
    // Convert convolutions into CustomCalls to cudnn, then canonicalize them
    // (PadInsertion).
    HloPassPipeline pipeline("conv_canonicalization");
    pipeline.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                              /*allow_mixed_precision=*/false);
    pipeline.AddPass<CudnnConvRewriter>();
    pipeline.AddPass<CudnnFusedConvRewriter>();
    pipeline.AddPass<CudnnConvPaddingLegalization>();

    pipeline.AddPass<HloConstantFolding>();
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  {
    // Run layout assignment in a separate pipeline from
    // "post-layout-assignment" because we want everything after layout
    // assignment to have a layout-sensitive invariant-checker, but
    // HloPassPipeline also runs its invariant checker before any passes are
    // run, meaning, the pipeline that contains layout assignment cannot contain
    // a layout-sensitive verifier!
    HloPassPipeline pipeline("layout assignment");
    pipeline.AddPass<GpuLayoutAssignment>(
        hlo_module->mutable_entry_computation_layout(),
        LayoutAssignment::InstructionCanChangeLayout, stream_exec);
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  {
    HloPassPipeline pipeline("post-layout_assignment");
    pipeline.AddInvariantChecker<HloVerifier>(
        /*layout_sensitive=*/true,
        /*allow_mixed_precision=*/false,
        LayoutAssignment::InstructionCanChangeLayout);


     // The LayoutAssignment pass may leave behind kCopy instructions which are
    // duplicate or NOPs, so remove them with algebraic simplification and CSE.
    AlgebraicSimplifierOptions options;
    options.set_is_layout_sensitive(true);
    pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);


    // Choose the fastest algorithm for each conv.
    //
    // We pick the algorithm before fusion so we can generate better HLO. After
    // CudnnConvolutionRewriter, our convolutions are CustomCalls which return a
    // tuple (conv_result, scratch_memory), and the each conv uses 0 bytes of
    // scratch:
    //
    //   customcall = (f32[...], f32[0])
    //   return gte(customcall, 0)
    //
    // The algorithm picker then chooses the best algorithm, and potentially
    // increases the scratch space.  It replaces customcall with new_tuple,
    // giving us the following:
    //
    //   new_customcall = (f32[...], f32[N])
    //   new_tuple = tuple(gte(new_customcall, 0), constant f32[0])
    //   return gte(new_tuple, 0)
    //
    // The new tuple and gte instructions then be simplified away, because
    // nobody is expected to use the scratch value.
    //
    // However, if we were to run CudnnConvolutionAlgorithmPicker after fusion
    // the gte(customcall, 0) would probably already be into a fusion node.  We
    // can't simplify across HloComputation boundaries, so in this case we
    // wouldn't be able to simplify away the new_tuple bits.
    pipeline.AddPass<MiopenConvAlgorithmPicker>(
        stream_exec, device_allocator, compiler);
    // Clean up new_tuple described above.
    pipeline.AddPass<TupleSimplifier>();

    pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  {
    HloPassFix<HloPassPipeline> fusion("fusion");
    fusion.AddInvariantChecker<HloVerifier>(
        /*layout_sensitive=*/true,
        /*allow_mixed_precision=*/false,
        LayoutAssignment::InstructionCanChangeLayout);
    fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/false);
    fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/true);
    fusion.AddPass<FusionMerger>();
    fusion.AddPass<GpuMultiOutputFusion>();
    fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                           /*only_fusion_computations=*/true);
    fusion.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(fusion.Run(hlo_module).status());

    HloPassPipeline reduce_pipeline("reduce-precision");
    reduce_pipeline.AddInvariantChecker<HloVerifier>(
        /*is_layout_sensitive=*/true, /*allow_mixed_precision=*/false,
        LayoutAssignment::InstructionCanChangeLayout);
    ReducePrecisionInsertion::AddPasses(
        &reduce_pipeline, hlo_module->config().debug_options(),
        ReducePrecisionInsertion::PassTiming::AFTER_FUSION);
    StatusOr<bool> reduce_result = reduce_pipeline.Run(hlo_module);
    TF_RETURN_IF_ERROR(reduce_result.status());

    if (reduce_result.ValueOrDie()) {
      // Do another fusion pass, with the expectation that we may be able to
      // fuse the new ReducePrecision operations.
      TF_RETURN_IF_ERROR(fusion.Run(hlo_module).status());
    }
  }

  return Status::OK();
}

// Modifies the given HLO module so that it will be accepted by IrEmitter.
// Unlike optimization passes, the passes are necessary for correctness.
Status PrepareHloModuleForIrEmitting(HloModule* hlo_module) {
  // In some cases, we have to place the result of an instruction in a temporary
  // buffer. For instance, the buffer that holds an external parameter is
  // assumed immutable at this point, and should not be reused for output
  // (b/27180329). Therefore, in that case, we set the output to be a copy of
  // the parameter.
  HloPassPipeline pipeline("GPU-ir-emit-prepare");
  pipeline.AddInvariantChecker<HloVerifier>(
      /*layout_sensitive=*/true,
      /*allow_mixed_precision=*/false,
      LayoutAssignment::InstructionCanChangeLayout);

  // Copy insertion should be performed immediately before IR emission to avoid
  // inserting unnecessary copies (later pass adds an instruction which
  // materializes the value) or missing a necessary copy (later pass removes an
  // instruction which materializes a value). DCE must be run immediately before
  // (and sometime after) copy insertion, to avoid dead code from interfering
  // with the rewrites.
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<FlattenCallGraph>();
  pipeline.AddPass<GpuCopyInsertion>();
  pipeline.AddPass<GpuSanitizeConstantNames>();

  return pipeline.Run(hlo_module).status();
}

}  // namespace

AMDGPUCompiler::AMDGPUCompiler()
    : pointer_size_(llvm::DataLayout(kDataLayout)
                        .getPointerSize(0 /* default address space */)) {}

StatusOr<std::unique_ptr<HloModule>> AMDGPUCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    DeviceMemoryAllocator* device_allocator) {
  // We dump the post-optimization HLO in RunBackend so no need to dump it here.
  VLOG(3) << "*** HLO Before Optimization";
  XLA_VLOG_LINES(3, module->ToString());

  XLA_SCOPED_LOGGING_TIMER("AMDGPUCompiler::RunHloPasses");
  tracing::ScopedActivity activity("HLO Transforms", module->name(),
                                   /*is_expensive=*/true);
  TF_RETURN_IF_ERROR(
      OptimizeHloModule(module.get(), stream_exec, device_allocator, this));

  TF_RETURN_IF_ERROR(PrepareHloModuleForIrEmitting(module.get()));

  return std::move(module);
}

StatusOr<std::unique_ptr<Executable>> AMDGPUCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    DeviceMemoryAllocator* device_allocator) {
  XLA_SCOPED_LOGGING_TIMER("AMDGPUCompiler::RunBackend");

  TF_RET_CHECK(stream_exec != nullptr);

  llvm::LLVMContext llvm_context;
  std::string buffer;
  llvm::raw_string_ostream error(buffer);
  llvm::DiagnosticPrinterRawOStream printer(error);
  auto DiagnosticHandler = [](const llvm::DiagnosticInfo& diag_info,
                              void* Context) {
    auto printer = static_cast<llvm::DiagnosticPrinterRawOStream*>(Context);
    diag_info.print(*printer);
  };
  llvm_context.setDiagnosticHandlerCallBack(DiagnosticHandler, &printer);

  llvm::Module llvm_module(module->name().c_str(), llvm_context);
  // Set the target triple and the data layout.
  llvm_module.setTargetTriple(kTargetTriple);
  llvm_module.setDataLayout(kDataLayout);

  // Determine the HLO schedule, which is an ordering of HLO instructions.  This
  // is used by buffer assignment to enable buffer reuse, and the same ordering
  // must also be used to determine the thunk launch schedule.
  std::unique_ptr<StreamAssignment> stream_assignment = AssignStreams(*module);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<GpuHloSchedule> hlo_schedule,
      GpuHloSchedule::Build(*module, *stream_assignment, pointer_size_));

  // Run buffer analysis on the HLO graph. This analysis figures out which
  // temporary buffers are required to run the computation.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssigner::Run(
          module.get(), hlo_schedule->ConsumeHloOrdering(),
          BufferSizeBytesFunction(),
          /*color_alignment=*/
          [](LogicalBuffer::Color) { return kXlaAllocatedBufferAlignBytes; },
          /*allow_input_output_aliasing=*/false,
          /*allocate_buffers_for_constants=*/true));
  // BufferAssignment::Stats::ToString() and BufferAssignment::ToString()
  // include headers, so no need for us to print them ourselves.
  XLA_VLOG_LINES(1, buffer_assignment->GetStats().ToString());
  XLA_VLOG_LINES(2, buffer_assignment->ToString());
  VLOG(3) << "*** HLO After Optimization";
  XLA_VLOG_LINES(3, module->ToString());
  const string xla_dump_optimized_hlo_proto_to =
      module->config().debug_options().xla_dump_optimized_hlo_proto_to();
  if (!xla_dump_optimized_hlo_proto_to.empty()) {
    HloProto proto = MakeHloProto(*module, *buffer_assignment);
    TF_RETURN_IF_ERROR(protobuf_util::DumpProtoToDirectory(
        proto, xla_dump_optimized_hlo_proto_to, module->name()));
  }
  IrEmitterContext ir_emitter_context(module.get(), buffer_assignment.get(),
                                      &stream_exec->GetDeviceDescription(),
                                      &llvm_module);

  HloComputation* entry_computation = module->entry_computation();
  llvm_ir::AMDGPUMachineFeatures llvm_target_features = 
       llvm_ir::AMDGPUMachineFeatures::Singleton();

  IrEmitterUnnested ir_emitter(module->config(), entry_computation,
                               &ir_emitter_context, &llvm_target_features);

  TF_RETURN_IF_ERROR(ir_emitter.EmitConstantGlobals());

  {
    XLA_SCOPED_LOGGING_TIMER("AMDGPUCompiler::RunBackend - IR emission");
    TF_RETURN_IF_ERROR(entry_computation->Accept(&ir_emitter));
  }

  if (user_pre_optimization_hook_) {
    TF_CHECK_OK(user_pre_optimization_hook_(llvm_module));
  }
  string ir_module_string_before_opt;
  const bool embed_ir_in_executable =
      module->config().debug_options().xla_embed_ir_in_executable();
  if (VLOG_IS_ON(3) || embed_ir_in_executable) {
    ir_module_string_before_opt = llvm_ir::DumpModuleToString(llvm_module);
    VLOG(3) << "LLVM module before optimizations:";
    XLA_VLOG_LINES(3, ir_module_string_before_opt);
  }

  const string& ir_dump_directory =
      module->config().debug_options().xla_dump_ir_to();

  if (!ir_dump_directory.empty()) {
    TF_RETURN_IF_ERROR(llvm_ir::DumpIRToDirectory(
        /*directory_name=*/ir_dump_directory,
        /*hlo_module_name=*/module->name(), llvm_module,
        /*optimized=*/false));
  }

  {
    XLA_SCOPED_LOGGING_TIMER(
        "AMDGPUCompiler::RunBackend - Running LLVM verifier");

    std::string err;
    llvm::raw_string_ostream err_stream(err);

    // verifyModule() returns true if the module is broken.
    TF_RET_CHECK(!llvm::verifyModule(llvm_module, &err_stream))
        << "Invalid LLVM IR before optimizations:\n"
        << err_stream.str()
        << "\nThis probably indicates a bug in the HLO -> LLVM IR lowering. "
           "Rerun with --xla_dump_ir_to to get the IR. ";
  }

  int isa_version = 0;
  if (!stream_exec->GetDeviceDescription().
                    rocm_amdgpu_isa_version(&isa_version)) {
    LOG(WARNING)
        << "Couldn't get AMDGPU ISA version for device; assuming gfx803.";
    isa_version = 803;
  }
  if (rocdl_dir_.empty()) {
    // Compute rocdl_dir_ just once and cache it in this member.
    rocdl_dir_ = GetROCDLDir(module->config());
  }

  std::vector<uint8> hsaco;
  {
    XLA_SCOPED_LOGGING_TIMER("AMDGPUCompiler::Runbackend - CompileToHsaco");
    TF_ASSIGN_OR_RETURN(hsaco, CompileToHsaco(&llvm_module, isa_version,
                                              module->config(), rocdl_dir_));
  }

  if (!ir_dump_directory.empty()) {
    TF_RETURN_IF_ERROR(llvm_ir::DumpIRToDirectory(
        /*directory_name=*/ir_dump_directory,
        /*hlo_module_name=*/module->name(), llvm_module,
        /*optimized=*/true));
  }

  if (user_post_optimization_hook_) {
    TF_CHECK_OK(user_post_optimization_hook_(llvm_module));
  }
  VLOG(3) << "LLVM module after optimizations:";
  XLA_VLOG_LINES(3, llvm_ir::DumpModuleToString(llvm_module));

  auto thunk_schedule = absl::make_unique<ThunkSchedule>(
      ir_emitter.ConsumeThunkSequence(), std::move(stream_assignment),
      hlo_schedule->ThunkLaunchOrder());
  VLOG(3) << "Printing the thunk schedule...";
  XLA_VLOG_LINES(3, thunk_schedule->ToString());

  std::unique_ptr<HloProfileIndexMap> profile_index_map;
  std::unique_ptr<HloProfilePrinterData> profile_printer;

  if (module->config().hlo_profiling_enabled()) {
    HloCostAnalysis cost_analysis(ShapeSizeBytesFunction());
    cost_analysis.set_bytes_per_second(
        stream_exec->GetDeviceDescription().memory_bandwidth());
    TF_RETURN_IF_ERROR(module->entry_computation()->Accept(&cost_analysis));
    profile_index_map = absl::make_unique<HloProfileIndexMap>(*module);
    profile_printer = CreateHloProfilePrinterData(
        *profile_index_map, cost_analysis, entry_computation->name());
  }
 
  auto* amdgpu_executable = new AMDGPUExecutable(
        "", std::move(hsaco), isa_version, std::move(thunk_schedule),
        std::move(module), std::move(buffer_assignment),
        std::move(profile_printer), std::move(profile_index_map));
  if (embed_ir_in_executable) {
    DCHECK_NE("", ir_module_string_before_opt);
    amdgpu_executable->set_ir_module_string(ir_module_string_before_opt);
  }
  return std::unique_ptr<Executable>(amdgpu_executable);
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
AMDGPUCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                   const AotCompilationOptions& options) {
  return Unimplemented(
      "not yet implemented: AMDGPUCompiler::CompileAheadOfTime");
}

se::Platform::Id AMDGPUCompiler::PlatformId() const {
  return se::rocm::kROCmPlatformId;
}

}  // namespace gpu
}  // namespace xla

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      stream_executor::rocm::kROCmPlatformId,
      []() { return absl::make_unique<xla::gpu::AMDGPUCompiler>(); });
  return true;
}
static bool module_initialized = InitModule();
