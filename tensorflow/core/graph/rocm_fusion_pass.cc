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
#include <list>
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
namespace rocm_fusion_pass {

  const int kVlogLevel = -1;
  
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

  void DumpNodeList(int lvl, string message, std::list<Node*> nodes) {

    VLOG(lvl) << "===========";
    VLOG(lvl) << message;
    for (auto n : nodes) {
      VLOG(lvl) << "\t" << n;
    }
    VLOG(lvl) << "===========";
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


  // absract base class for an individual fusion operation
  class ROCmFusionBase {

  public :
    
    ROCmFusionBase(std::unique_ptr<Graph>* g, std::set<Node*>& fn)
      : graph_(g)
      , fused_nodes_(fn)
    {}

    virtual ~ROCmFusionBase() {}
    
    void DoFusion(Node* n) {
      if (IsFusionEligible(n)) {
	CreateFusionOp();
      }
    }

  protected:

    // routine to determine whether any sequence of nodes *ending* at the given
    // node is eligible for fusion. 
    virtual bool IsFusionEligible(Node* n) = 0;

    // routine to create a fused op corresponding to some sequence of nodes
    // *ending* at the given node (which was specified in the previous call).
    virtual void CreateFusionOp() = 0;
      
    std::unique_ptr<Graph>* graph_; 
    std::set<Node*>& fused_nodes_;
  };
  
  class ROCmFusionPass : public GraphOptimizationPass {
  public:

    // optimization pass entry point,
    // application code will call this routine to run the pass
    Status Run(const GraphOptimizationPassOptions& options) override;

    // helper function that does all the work for this pass
    bool RunPass(std::unique_ptr<Graph>* g);

  private:

    void InitializeFusions(std::vector<ROCmFusionBase*>& fusions,
			   std::unique_ptr<Graph>* g,
			   std::set<Node*>& fused_nodes);

    void RemoveFusedNodes(std::set<Node*>& fused_nodes);

  };

  // Register the ROCmFusionPass with the registry.
  // At the point of this writing, only the *mkl* passes are in the
  // POST_PARTITIONING grouping. They work on nodes (placed on CPU) that are
  // exclusive to nodes this pass will work on (placed on GPU). So the
  // choice of phase number (1) is completely arbitrary.
  
  REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, // grouping
			1, // phase number
			ROCmFusionPass);


  // Convolution-Bias-BatchNorm-Activation Fusion
  class ROCmFusionCBNA : public ROCmFusionBase {

  public:

    ROCmFusionCBNA(std::unique_ptr<Graph>* g, std::set<Node*>& fn)
      : ROCmFusionBase(g, fn)
      , conv_(nullptr)
      , bias_(nullptr)
      , norm_(nullptr)
      , actv_(nullptr)
    {}

  protected:

    bool IsFusionEligible(Node* n) override;

    void CreateFusionOp() override;

    Node* conv_;
    Node* bias_;
    Node* norm_;
    Node* actv_;
  };


  // Convolution-Bias-Activation Fusion
  class ROCmFusionCBA : public ROCmFusionBase {

  public:

    ROCmFusionCBA(std::unique_ptr<Graph>* g, std::set<Node*>& fn)
      : ROCmFusionBase(g, fn)
      , conv_(nullptr)
      , bias_(nullptr)
      , actv_(nullptr)
    {}

  protected:

    bool IsFusionEligible(Node* n) override;

    void CreateFusionOp() override;

    Node* conv_;
    Node* bias_;
    Node* actv_;
  };


  
  // BatchNorm-Activation Fusion
  class ROCmFusionNA : public ROCmFusionBase {

  public:

    ROCmFusionNA(std::unique_ptr<Graph>* g, std::set<Node*>& fn)
      : ROCmFusionBase(g, fn)
      , norm_(nullptr)
      , actv_(nullptr)
    {}

  protected:

    bool IsFusionEligible(Node* n) override;

    void CreateFusionOp() override;

    Node* norm_;
    Node* actv_;
  };

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

	VLOG(kVlogLevel) << "Running ROCmFusionPass for partition : " << pg.first ;

	// Get the ownership of a graph
	std::unique_ptr<Graph>* ng = std::move(&(pg.second));
	
	// run the pass
	RunPass(ng);
	
	// Return the ownership of the graph
	pg.second.reset(ng->release());

      } else {
	
	VLOG(kVlogLevel) << "Skipping ROCmFusionPass for partition : " << pg.first ;
      }
    }

    return Status::OK();
  }

  
  bool ROCmFusionPass::RunPass(std::unique_ptr<Graph>* g) {

    // DumpGraph("Before running ROCmFusionPass", &**g);

    std::vector<ROCmFusionBase*> fusions;
    std::set<Node*> fused_nodes;
    
    // Initialize a vector of all the fusion operations we currently support
    InitializeFusions(fusions, g, fused_nodes);
    
    std::vector<Node*> order;
    GetPostOrder(**g, &order);  // This will give us reverse topological sort.

    for (Node* n : order) {

      // VLOG(kVlogLevel) << n;

      if (fused_nodes.count(n)) { // we have fused this node...skip it
	continue;
      }
      
      for (auto fusion : fusions) {
	fusion->DoFusion(n);
      }
    }

    // Remove fused nodes from the graph
    RemoveFusedNodes(fused_nodes);
    
    // DumpGraph("After running ROCmFusionPass", &**g);
    return true;
  }

  void ROCmFusionPass::InitializeFusions(std::vector<ROCmFusionBase*>& fusions,
					 std::unique_ptr<Graph>* g,
					 std::set<Node*>& fused_nodes) {

    fusions.push_back(new ROCmFusionCBNA(g, fused_nodes));
    fusions.push_back(new ROCmFusionCBA(g, fused_nodes));
    fusions.push_back(new ROCmFusionNA(g, fused_nodes));
  }
  
  void ROCmFusionPass::RemoveFusedNodes(std::set<Node*>& fused_nodes) {
  }

  
  bool ROCmFusionCBNA::IsFusionEligible(Node* n4) {
    
    if (isOpActv(n4)) { // activation node
      Node* n3 = nullptr;
      TF_CHECK_OK(n4->input_node(0, &n3));
      if (isOpNorm(n3)) { // preceded by a batchnorm node
  	Node* n2 = nullptr;
  	TF_CHECK_OK(n3->input_node(0, &n2));
  	if (isOpBias(n2)) { // preceded by a bias node
  	  Node* n1 = nullptr;
  	  TF_CHECK_OK(n2->input_node(0, &n1));
	  if (isOpConv(n1)) { // preceded by a convolution node
	    conv_ = n1;
	    bias_ = n2;
	    norm_ = n3;
	    actv_ = n4;
	    DumpNodeList(kVlogLevel, "Found Fusion Candidate CBNA : ", {conv_, bias_, norm_, actv_});
	    return true;
	  }
	}
      }
    }
    
    return false;
  }
  
  
  void ROCmFusionCBNA::CreateFusionOp() {
    
  }

  bool ROCmFusionCBA::IsFusionEligible(Node* n3) {

    // First check whether we have the right sequence of ops
    bool is_eligible_sequence = false;
    if (isOpActv(n3)) { // activation node
      Node* n2 = nullptr;
      TF_CHECK_OK(n3->input_node(0, &n2));
      if (isOpBias(n2)) { // preceded by a bias node
  	Node* n1 = nullptr;
	TF_CHECK_OK(n2->input_node(0, &n1));
  	if (isOpConv(n1)) { // precedded by a convolution node
	  conv_ = n1;
	  bias_ = n2;
	  actv_ = n3;
	  DumpNodeList(kVlogLevel, "Found Fusion Candidate CBA : ", {conv_, bias_, actv_});
	  is_eligible_sequence = true;
  	}
      }
    }

    if (is_eligible_sequence) {
      return true;
    }

    return false;
  }
  
  void ROCmFusionCBA::CreateFusionOp() {
  }

  bool ROCmFusionNA::IsFusionEligible(Node* n2) {
    
    if (isOpActv(n2)) { // activation node
      Node* n1 = nullptr;
      TF_CHECK_OK(n2->input_node(0, &n1));
      if (isOpNorm(n1)) { // preceded by a batchnorm node
	norm_ = n1;
	actv_ = n2;
	DumpNodeList(kVlogLevel, "Found Fusion Candidate NA : ", {norm_, actv_});
	return true;
      }
    }
    return false;
  }
  
  void ROCmFusionNA::CreateFusionOp() {
  }

}  // namespace rocm_fusion_pass
}  // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM
