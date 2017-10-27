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
#include <functional>
#include <utility>

#include "external/llvm/include/llvm/IR/DiagnosticInfo.h"
#include "external/llvm/include/llvm/IR/DiagnosticPrinter.h"
#include "external/llvm/include/llvm/IR/LLVMContext.h"
#include "external/llvm/include/llvm/IR/Module.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_folding.h"
#include "tensorflow/compiler/xla/service/gpu/copy_insertion.h"
#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/layout_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/amdgpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/pad_insertion.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/rocm_rocdl_path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/subprocess.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace gpu {

namespace {

// The triple that represents our target.
const char* kTargetTriple = "amdgcn--amdhsa-amdgiz";

// The data layout of the emitted module. Copied from computeDataLayout in
// AMDGPUTargetMachine.cpp.
const char* kDataLayout =
         "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32"
         "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
         "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5";

// Any address of a variable residing in global memory or returned by one of the
// memory allocation routines from the driver or runtime API is always aligned
// to at least 256 bytes.
constexpr int64 kMemoryAlignment = 256;

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
tensorflow::Status OptimizeHloModule(HloModule* hlo_module,
                                     const se::DeviceDescription& device_desc) {
  {
    HloPassPipeline pipeline("optimization");
    pipeline.AddInvariantChecker<HloVerifier>();
    {
      auto& pass =
          pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification");
      pass.AddPass<AlgebraicSimplifier>(
          /*is_layout_sensitive=*/false,
          [](const Shape&, const Shape&) { return false; });
      pass.AddPass<ReshapeMover>();
      pass.AddPass<HloConstantFolding>();
    }
    pipeline.AddPass<ConvolutionFolding>();
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
    HloPassFix<HloPassPipeline> fusion("fusion");
    fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/false);
    fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/true);
    fusion.AddPass<FusionMerger>();
    return fusion.Run(hlo_module).status();
  }
}

// Modifies the given HLO module so that it will be accepted by IrEmitter.
// Unlike optimization passes, the passes are necessary for correctness.
tensorflow::Status PrepareHloModuleForIrEmitting(HloModule* hlo_module) {
  // In some cases, we have to place the result of an instruction in a temporary
  // buffer. For instance, the buffer that holds an external parameter is
  // assumed immutable at this point, and should not be reused for output
  // (b/27180329). Therefore, in that case, we set the output to be a copy of
  // the parameter.
  HloPassPipeline pipeline("GPU-ir-emit-prepare");
  pipeline.AddInvariantChecker<HloVerifier>();
  pipeline.AddPass<PadInsertion>();
  pipeline.AddPass<GpuLayoutAssignment>(
      hlo_module->mutable_entry_computation_layout());
  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(
      /*is_layout_sensitive=*/true,
      [](const Shape&, const Shape&) { return true; });
  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
  // Copy insertion should be performed immediately before IR emission to avoid
  // inserting unnecessary copies (later pass adds an instruction which
  // materializes the value) or missing a necessary copy (later pass removes an
  // instruction which materializes a value). DCE must be run immediately before
  // (and sometime after) copy insertion, to avoid dead code from interfering
  // with the rewrites.
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<GpuCopyInsertion>();
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<FlattenCallGraph>();
  return pipeline.Run(hlo_module).status();
}

}  // namespace

AMDGPUCompiler::AMDGPUCompiler()
    : pointer_size_(llvm::DataLayout(kDataLayout).getPointerSize()) {}

StatusOr<std::unique_ptr<Executable>> AMDGPUCompiler::Compile(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec) {
  TF_RET_CHECK(stream_exec != nullptr);

  TF_RETURN_IF_ERROR(
      OptimizeHloModule(module.get(), stream_exec->GetDeviceDescription()));
  TF_RETURN_IF_ERROR(PrepareHloModuleForIrEmitting(module.get()));

  llvm::LLVMContext llvm_context;
  std::string buffer;
  llvm::raw_string_ostream error(buffer);
  llvm::DiagnosticPrinterRawOStream printer(error);
  auto DiagnosticHandler = [](const llvm::DiagnosticInfo& diag_info,
                              void* Context) {
    auto printer = static_cast<llvm::DiagnosticPrinterRawOStream*>(Context);
    diag_info.print(*printer);
  };
  llvm_context.setDiagnosticHandler(DiagnosticHandler, &printer);

  llvm::Module llvm_module(module->name().c_str(), llvm_context);
  // Set the target triple and the data layout.
  llvm_module.setTargetTriple(kTargetTriple);
  llvm_module.setDataLayout(kDataLayout);

  // Determine the HLO schedule, which is an ordering of HLO instructions.  This
  // is used by buffer assignment to enable buffer reuse, and the same ordering
  // must also be used to determine the thunk launch schedule.
  std::unique_ptr<StreamAssignment> stream_assignment = AssignStreams(*module);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloSchedule> hlo_schedule,
      HloSchedule::Build(*module, *stream_assignment, pointer_size_));

  // Run buffer analysis on the HLO graph. This analysis figures out which
  // temporary buffers are required to run the computation.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssigner::Run(module.get(), hlo_schedule->ConsumeHloOrdering(),
                          BufferSizeBytesFunction(), [](LogicalBuffer::Color) {
                            return kMemoryAlignment;
                          }));

  const string dump_debug_json_to =
      module->config().debug_options().xla_dump_debug_json_to();
  if (!dump_debug_json_to.empty()) {
    HloProto proto = MakeHloProto(*module, *buffer_assignment);
    TF_RETURN_IF_ERROR(protobuf_util::DumpJsonToDirectory(
        proto, dump_debug_json_to, module->name()));
  }

  IrEmitterContext ir_emitter_context(module.get(), buffer_assignment.get(),
                                      &stream_exec->GetDeviceDescription(),
                                      &llvm_module);

  HloComputation* entry_computation = module->entry_computation();
  IrEmitterUnnested ir_emitter(module->config(), entry_computation,
                               module->config().has_hybrid_result(),
                               &ir_emitter_context);
  TF_RETURN_IF_ERROR(
      entry_computation->root_instruction()->Accept(&ir_emitter));

  string ir_module_string_before_opt;
  const bool embed_ir_in_executable =
      module->config().debug_options().xla_embed_ir_in_executable();
  if (VLOG_IS_ON(2) || embed_ir_in_executable) {
    ir_module_string_before_opt = llvm_ir::DumpModuleToString(llvm_module);
    VLOG(2) << "LLVM module before optimizations:";
    XLA_VLOG_LINES(2, ir_module_string_before_opt);
  }

  // Reserve space for the HSACO to be generated for this module.
  std::vector<char>* hsaco;
  {
    tensorflow::mutex_lock lock(mutex_);
    generated_hsaco_.emplace_back(MakeUnique<std::vector<char>>());
    hsaco = generated_hsaco_.back().get();
  }
  
  // XXX FIXME fix stream_executor device_description
  //int cc_major, cc_minor;
  //if (!stream_exec->GetDeviceDescription().cuda_compute_capability(&cc_major,
  //                                                                 &cc_minor)) {
  //  LOG(WARNING)
  //      << "Couldn't get compute capability for device; assuming sm_20.";
  //  cc_major = 2;
  //  cc_minor = 0;
  //}
  if (rocdl_dir_.empty()) {
    // Compute rocdl_dir_ just once and cache it in this member.
    rocdl_dir_ = GetROCDLDir(module->config());
  }
  // XXX FIXME force use gfx803 for now
  TF_ASSIGN_OR_RETURN(*hsaco, CompileToHsaco(&llvm_module, "gfx803",
                                             module->config(), rocdl_dir_));

  VLOG(2) << "LLVM module after optimizations:";
  XLA_VLOG_LINES(2, llvm_ir::DumpModuleToString(llvm_module));

  auto thunk_schedule = MakeUnique<ThunkSchedule>(
      ir_emitter.ConsumeThunkSequence(), std::move(stream_assignment),
      hlo_schedule->ThunkLaunchOrder());
  VLOG(2) << "Printing the thunk schedule...";
  XLA_VLOG_LINES(2, thunk_schedule->ToString());

  auto* gpu_executable =
      new GpuExecutable(std::move(hsaco->data()), std::move(thunk_schedule), std::move(module),
                        std::move(buffer_assignment), ShapeSizeBytesFunction());
  if (embed_ir_in_executable) {
    DCHECK_NE("", ir_module_string_before_opt);
    gpu_executable->set_ir_module_string(ir_module_string_before_opt);
  }
  return std::unique_ptr<Executable>(gpu_executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> AMDGPUCompiler::Compile(
    std::vector<std::unique_ptr<HloModule>> modules,
    std::vector<se::StreamExecutor*> stream_execs) {
  return Unimplemented(
      "Compilation of multiple HLO modules is not yet supported on GPU.");
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
AMDGPUCompiler::CompileAheadOfTime(std::vector<std::unique_ptr<HloModule>> module,
                                const AotCompilationOptions& options) {
  return Unimplemented("not yet implemented: AMDGPUCompiler::CompileAheadOfTime");
}

se::Platform::Id AMDGPUCompiler::PlatformId() const {
  return se::rocm::kROCmPlatformId;
}

}  // namespace gpu
}  // namespace xla

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(se::rocm::kROCmPlatformId, []() {
    return xla::MakeUnique<xla::gpu::AMDGPUCompiler>();
  });
  return true;
}
static bool module_initialized = InitModule();
