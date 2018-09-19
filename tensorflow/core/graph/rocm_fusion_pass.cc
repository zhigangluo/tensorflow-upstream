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

#ifdef TENSORFLOW_USE_ROCM

#include <algorithm>
#include <functional>
#include <memory>
#include <queue>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/graph/rocm_fusion_pass.h"

namespace tensorflow {

  // Util routines - Start
  // These routines should be moved to a shared file if we end up with more than one rocm graph pass
  
  std::ostream& operator<<(std::ostream& out, const Node* n) {

    out << "id : " << n->id()
	<< ", cost_id : " << n->cost_id() 
	<< ", name : " << n->name() 
	<< ", type_string : " << n->type_string() 
	<< ", num_inputs : " << n->num_inputs()
	<< ", num_input_edges : " << n->in_edges().size()
	<< ", num_outputs : " << n->num_outputs() 
	<< ", num_output_edges : " << n->out_edges().size()
	<< ", requested_device : " << n->requested_device() 
	<< ", assigned_device : " << (n->has_assigned_device_name() ? n->assigned_device_name() : "None")
      ;
    
    return out;
  }


  bool isGpuPartition(StringPiece fullname) {
    const char* const kGPUDeviceStr = "GPU";
    DeviceNameUtils::ParsedName p;
    return (DeviceNameUtils::ParseFullName(fullname, &p) && (p.type == kGPUDeviceStr));
  }

  // is this node an instance of a convolution op for which we support fusion for?
  inline bool isOpConv(Node* n) {
    return (n->type_string() == "Conv2D");
  }

  // is this node an instance of a bias op for which we support fusion for?
  inline bool isOpBias(Node* n) {
    return (n->type_string() == "BiasAdd");
  }

  // is this node an instance of a activation op for which we support fusion for?
  inline bool isOpActv(Node* n) {
    return (n->type_string() == "Relu");
  }

  // is this node an instance of a batchnorm op for which we support fusion for?
  inline bool isOpNorm(Node* n) {
    return (n->type_string() == "FusedBatchNorm");
  }
  // Util routines - End

  
  class ROCmFusionPass : public GraphOptimizationPass {
  public:

    // optimization pass entry point,
    // application code will call this routine to run the pass
    Status Run(const GraphOptimizationPassOptions& options) override;

    // helper function that does all the work for this pass
    bool RunPass(std::unique_ptr<Graph>* g);

    void CheckFusionConvBiasActv(std::unique_ptr<Graph>* g, Node* n);

    void CheckFusionConvBiasNormActv(std::unique_ptr<Graph>* g, Node* n);

    void CheckFusionNormActv(std::unique_ptr<Graph>* g, Node* n);
    
  private:

    const int kVlogLevel_ = -1;
  };

  // Register the ROCmFusionPass with the registry.
  // At the point of this writing, only the *mkl* passes are in the
  // POST_PARTITIONING grouping. They work on nodes (placed on CPU) that are
  // exclusive to nodes this pass will work on (placed on GPU). So the
  // choice of phase number (1) is completely arbitrary.
  
  REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, // grouping
			1, // phase number
			ROCmFusionPass);
  
  bool RunROCmFusionPass(std::unique_ptr<Graph>* g) {
    return ROCmFusionPass().RunPass(g);
  }

  Status ROCmFusionPass::Run(const GraphOptimizationPassOptions& options) {

    // Check if the graph is present, should be either in
    // - options.graph (for all but POST_PARTITIONING grouping)
    // - options.partition_graphs (for POST_PARTITIONING_grouping)
    if (options.graph == nullptr && options.partition_graphs == nullptr) {
      return Status::OK();
    }

    for (auto& pg : *options.partition_graphs) {

      if (isGpuPartition(pg.first)) {

	VLOG(kVlogLevel_) << "Running ROCmFusionPass for partition : " << pg.first ;

	// Get the ownership of a graph
	std::unique_ptr<Graph>* ng = std::move(&(pg.second));
	
	// run the pass
	RunPass(ng);
	
	// Return the ownership of the graph
	pg.second.reset(ng->release());

      } else {
	
	VLOG(kVlogLevel_) << "Skipping ROCmFusionPass for partition : " << pg.first ;
      }
    }

    return Status::OK();
  }

  
  bool ROCmFusionPass::RunPass(std::unique_ptr<Graph>* g) {
    
    // DumpGraph("Before running ROCmFusionPass", &**g);

    std::vector<Node*> order;
    // GetReversePostOrder(**g, &order);  // This will give us topological sort.
    GetPostOrder(**g, &order);  // This will give us reverse topological sort.

    for (Node* n : order) {

      // VLOG(kVlogLevel_) << n;
      
      CheckFusionConvBiasNormActv(g, n);
      
      CheckFusionConvBiasActv(g, n);
      
      CheckFusionNormActv(g, n);
    }
    
    // DumpGraph("After running ROCmFusionPass", &**g);
    return true;
  }

  
  void ROCmFusionPass::CheckFusionConvBiasNormActv(std::unique_ptr<Graph>* g, Node* n4) {
    
    if (isOpNorm(n4)) {

      Node* n3 = nullptr;
      TF_CHECK_OK(n4->input_node(0, &n3));

      if (isOpActv(n3)) {
	Node* n2 = nullptr;
	TF_CHECK_OK(n3->input_node(0, &n2));

	if (isOpBias(n2)) {

	  Node* n1 = nullptr;
	  TF_CHECK_OK(n2->input_node(0, &n1));
	
	  if (isOpConv(n1)) {
	  
	    VLOG(kVlogLevel_) << "=============" << "Found Fusion Candidate CBNA : \n"
			      << "\t" << n1 << std::endl
			      << "\t" << n2 << std::endl
			      << "\t" << n3 << std::endl
			      << "\t" << n4 << std::endl
	      ;
	  }
	}
      }
    }
  }

  
  void ROCmFusionPass::CheckFusionConvBiasActv(std::unique_ptr<Graph>* g, Node* n3) {
    
    if (isOpActv(n3)) {

      Node* n2 = nullptr;
      TF_CHECK_OK(n3->input_node(0, &n2));

      if (isOpBias(n2)) {

  	Node* n1 = nullptr;
	TF_CHECK_OK(n2->input_node(0, &n1));
	
  	if (isOpConv(n1)) {
	  
	  VLOG(kVlogLevel_) << "=============" << "Found Fusion Candidate CBA : \n"
			    << "\t" << n1 << std::endl
			    << "\t" << n2 << std::endl
			    << "\t" << n3 << std::endl
	    ;
  	}
      }
    }
  }
  
  void ROCmFusionPass::CheckFusionNormActv(std::unique_ptr<Graph>* g, Node* n2) {
    
    if (isOpActv(n2)) {

      Node* n1 = nullptr;
      TF_CHECK_OK(n2->input_node(0, &n1));
	
      if (isOpNorm(n1)) {
	  
	VLOG(kVlogLevel_) << "=============" << "Found Fusion Candidate NA : \n"
			  << "\t" << n1 << std::endl
			  << "\t" << n2 << std::endl
			  << "\t" << n3 << std::endl
	  ;
      }
    }
  }
  
}  // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM
