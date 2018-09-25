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

  // declaration of all the constants that are used in the rocm_fusion_pass namespace
  
  const int kVlogLevel = -1;
  
  // attribute name strings
  const string kAttr_T = "T";
  const string kAttr_strides = "strides";
  const string kAttr_padding = "padding";
  const string kAttr_data_format = "data_format";
  const string kAttr_dilations = "dilations";
  const string kAttr_activation_mode = "activation_mode";

  // Util routines - Start
  
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

  void DumpNodeList(int lvl, string message, std::list<const Node*> nodes) {

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
  inline bool isOpConv(const Node* n) {
    return (n->type_string() == "Conv2D");
  }

  // is this node an instance of a bias op for which we support fusion for?
  inline bool isOpBias(const Node* n) {
    return (n->type_string() == "BiasAdd");
  }

  // is this node an instance of a activation op for which we support fusion for?
  inline bool isOpActv(const Node* n) {
    return (n->type_string() == "Relu");
  }

  // is this node an instance of a batchnorm op for which we support fusion for?
  inline bool isOpNorm(const Node* n) {
    return (n->type_string() == "FusedBatchNorm");
  }
  // Util routines - End


  class ROCmFusionOpBase;

  class ROCmFusionPass : public GraphOptimizationPass {
  public:

    // optimization pass entry point,
    // application code will call this routine to run the pass
    Status Run(const GraphOptimizationPassOptions& options) override;

    // helper function that does all the work for this pass
    bool RunPass(std::unique_ptr<Graph>* g);

  private:

    void InitializeFusions(std::vector<ROCmFusionOpBase*>& fusions,
			   Graph* g,
			   std::set<const Node*>& fused_nodes);

    void RemoveFusedNodes(std::set<const Node*>& fused_nodes);

  };

  // Register the ROCmFusionPass with the registry.
  // At the point of this writing, only the *mkl* passes are in the
  // POST_PARTITIONING grouping. They work on nodes (placed on CPU) that are
  // exclusive to nodes this pass will work on (placed on GPU). So the
  // choice of phase number (1) is completely arbitrary.
  
  REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, // grouping
			1, // phase number
			ROCmFusionPass);


  // absract base class for an individual fusion operation
  class ROCmFusionOpBase {

  public :
    
    ROCmFusionOpBase(Graph* g, std::set<const Node*>& fn, string n)
      : graph_(g)
      , fused_nodes_(fn)
      , fusion_op_name_(n)
    {}

    virtual ~ROCmFusionOpBase() {}
    
    virtual bool DoFusion(const Node* n) = 0;

  protected:

    Graph* graph_; 
    std::set<const Node*>& fused_nodes_;
    string fusion_op_name_;
  };
  
  // Convolution-Bias-BatchNorm-Activation Fusion
  class ROCmFusionOpCBNA : public ROCmFusionOpBase {

  public:

    ROCmFusionOpCBNA(Graph* g, std::set<const Node*>& fn)
      : ROCmFusionOpBase(g, fn, "_ROCmFusedConvBiasNormActv")
    {}

    bool DoFusion(const Node* n) override;
    
  protected:

    struct FusionData {
      const Node* conv;
      const Node* bias;
      const Node* norm;
      const Node* actv;
    };

    bool IsFusionEligible(const Node* n, FusionData* d);
  };


  // Convolution-Bias-Activation Fusion
  class ROCmFusionOpCBA : public ROCmFusionOpBase {

  public:

    ROCmFusionOpCBA(Graph* g, std::set<const Node*>& fn)
      : ROCmFusionOpBase(g, fn, "_ROCmFusedConvBiasActv")
    {}

    bool DoFusion(const Node* n) override;
    
  protected:

    struct FusionData {
      const Node* conv;
      const Node* bias;
      const Node* actv;

      DataType data_type;
      std::vector<int32> strides;
      string padding;
      string data_format;
      std::vector<int32> dilations;
      string activation_mode;
    };

    bool IsFusionEligible(const Node* n, FusionData* d);

    void CreateFusionOp(const FusionData* d);
  };


  
  // BatchNorm-Activation Fusion
  class ROCmFusionOpNA : public ROCmFusionOpBase {

  public:

    ROCmFusionOpNA(Graph* g, std::set<const Node*>& fn)
      : ROCmFusionOpBase(g, fn, "_ROCmFusedNormActv")
    {}

    bool DoFusion(const Node* n) override;
    
  protected:

    struct FusionData {
      const Node* norm;
      const Node* actv;
    };
    
    bool IsFusionEligible(const Node* n, FusionData* d);
  };




  
  bool RunROCmFusionPass(std::unique_ptr<Graph>* g) {
    return ROCmFusionPass().RunPass(g);
  }
  
  Status ROCmFusionPass::Run(const GraphOptimizationPassOptions& options) {

    // skip the fusion pass if the env var TF_ROCM_DISABLE_FUSION is set
    const char* disable_fusion = getenv("TF_ROCM_DISABLE_FUSION");
    if (disable_fusion != nullptr) {
      return Status::OK();
    }
    
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

    std::vector<ROCmFusionOpBase*> fusions;
    std::set<const Node*> fused_nodes;
    
    // Initialize a vector of all the fusion operations we currently support
    InitializeFusions(fusions, g->get(), fused_nodes);
    
    std::vector<Node*> order;
    GetPostOrder(**g, &order);  // This will give us reverse topological sort.

    for (const Node* n : order) {

      // VLOG(kVlogLevel) << n;

      if (fused_nodes.count(n)) {
	// We have fused already this node...skip it
	// Note that we are traversing nodes in reverse topological order, and
	// matches are found by comparing node sequences from back to front, so
	// we will hit nodes that we have already fused into a fusion operation.
	continue;
      }
      
      for (auto fusion : fusions) {
	if (fusion->DoFusion(n)) {
	  // bail out after the first successful fusion ...
	  // cannot do more than one fusions on a op
	  break; 
	}
      }
    }

    // Remove fused nodes from the graph
    RemoveFusedNodes(fused_nodes);
    
    // DumpGraph("After running ROCmFusionPass", &**g);
    return true;
  }

  void ROCmFusionPass::InitializeFusions(std::vector<ROCmFusionOpBase*>& fusions,
					 Graph* g,
					 std::set<const Node*>& fused_nodes) {

    fusions.push_back(new ROCmFusionOpCBNA(g, fused_nodes));
    fusions.push_back(new ROCmFusionOpCBA(g, fused_nodes));
    fusions.push_back(new ROCmFusionOpNA(g, fused_nodes));
  }
  
  void ROCmFusionPass::RemoveFusedNodes(std::set<const Node*>& fused_nodes) {
  }


  // -------------------------------------------
  // ROCmFusionOpCBNA implementation
  // -------------------------------------------
  bool ROCmFusionOpCBNA::DoFusion(const Node* n4) {

    FusionData d;
    return IsFusionEligible(n4, &d);
  }

  bool ROCmFusionOpCBNA::IsFusionEligible(const Node* n4, FusionData* d) {
    
    // First check whether we have the right sequence of ops
    bool is_eligible = false;
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
	    d->conv = n1;
	    d->bias = n2;
	    d->norm = n3;
	    d->actv = n4;
	    DumpNodeList(kVlogLevel, "Found Fusion Candidate CBNA : ",
			 {d->conv, d->bias, d->norm, d->actv});
	    is_eligible = true;
	  }
	}
      }
    }
    return is_eligible;
  }
  
  
  // -------------------------------------------
  // ROCmFusionOpCBNA implementation
  // -------------------------------------------
  bool ROCmFusionOpCBA::DoFusion(const Node* n3) {

    bool did_fusion = false;
    FusionData d;
    if (IsFusionEligible(n3, &d)) {
      CreateFusionOp(&d);
      did_fusion = true;
    }
    return did_fusion;
  }

  bool ROCmFusionOpCBA::IsFusionEligible(const Node* n3, FusionData* d) {

    bool is_eligible = false;

    // First check whether we have the right sequence of ops
    if (isOpActv(n3)) { // activation node
      Node* n2 = nullptr;
      TF_CHECK_OK(n3->input_node(0, &n2));
      if (isOpBias(n2)) { // preceded by a bias node
  	Node* n1 = nullptr;
	TF_CHECK_OK(n2->input_node(0, &n1));
  	if (isOpConv(n1)) { // precedded by a convolution node
	  d->conv = n1;
	  d->bias = n2;
	  d->actv = n3;
	  DumpNodeList(kVlogLevel, "Found Fusion Candidate CBA : ",
		       {d->conv, d->bias, d->actv});
	  is_eligible = true;
  	}
      }
    }

    // Next check if the output of conv node feeds a node other then the bias node
    if (is_eligible) {
      for (const Edge* e : d->conv->out_edges()) {
	if ((e->src_output() == 0) && (e->dst() != d->bias)) {
	  VLOG(kVlogLevel) << "Skipping Fusion : "
			   << "Convolution output feeds a node other the the bias node : "
			   << e->dst()->name();
	  is_eligible = false;
	  break;
	}
      }
    }

    // Next check if the output of bias node feeds a node other then the activation node
    if (is_eligible) {
      for (const Edge* e : d->bias->out_edges()) {
	if ((e->src_output() == 0) && (e->dst() != d->actv)) {
	  VLOG(kVlogLevel) << "Skipping Fusion : "
			   << "Bias output feeds a node other the the activation node : "
			   << e->dst()->name();
	  is_eligible = false;
	  break;
	}
      }
    }

    // Next check if the datatype(s) are supported
    if (is_eligible) {
      is_eligible = false;

      DataType T_conv, T_bias, T_actv;
      TF_CHECK_OK(GetNodeAttr(d->conv->def(), kAttr_T, &T_conv));
      TF_CHECK_OK(GetNodeAttr(d->bias->def(), kAttr_T, &T_bias));
      TF_CHECK_OK(GetNodeAttr(d->actv->def(), kAttr_T, &T_actv));

      // only float type is supported for now
      if (T_conv == DT_FLOAT) {
	// all three types must match
	if ((T_conv == T_bias) && (T_conv == T_actv)) {
	  d->data_type = T_conv;
	  is_eligible = true;
	} else {
	  VLOG(kVlogLevel) << " DataTypes not matching for Fusion : "
			   << " " << DataType_Name(T_conv)
			   << " " << DataType_Name(T_bias)
			   << " " << DataType_Name(T_actv);
	}
      } else {
	VLOG(kVlogLevel) << " DataType not supported for Fusion : " << DataType_Name(T_conv);
      }
    }

    // Next check if the data format(s) are supported
    if (is_eligible) {
      is_eligible = false;

      string df_conv_str, df_bias_str;
      TF_CHECK_OK(GetNodeAttr(d->conv->def(), kAttr_data_format, &df_conv_str));
      TF_CHECK_OK(GetNodeAttr(d->bias->def(), kAttr_data_format, &df_bias_str));

      VLOG(kVlogLevel) << "ROCmFusionPass - data_format = " << df_conv_str;
      
      TensorFormat df_conv, df_bias;
      CHECK_EQ(FormatFromString(df_conv_str, &df_conv), true);
      CHECK_EQ(FormatFromString(df_bias_str, &df_bias), true);
      
      if ((df_conv == FORMAT_NHWC) || (df_conv == FORMAT_NCHW)) {
	if (df_conv == df_bias) {
	  d->data_format = ToString(df_conv);
	  is_eligible = true;
	} else {
	  VLOG(kVlogLevel) << " Data Formats not matching for Fusion : "
			   << " " << ToString(df_conv)
			   << " " << ToString(df_bias);
	}
      } else {
	VLOG(kVlogLevel) << " Data Format not supported for Fusion : " << ToString(df_conv);
      }
    }

    // Next check if the specified stride is supported
    if (is_eligible) {
      is_eligible = false;
      
      TF_CHECK_OK(GetNodeAttr(d->conv->def(), kAttr_strides, &d->strides));

      // TODO : add checks here
      is_eligible = true;
    }

    // Next check if the specified padding is supported
    if (is_eligible) {
      is_eligible = false;
      
      TF_CHECK_OK(GetNodeAttr(d->conv->def(), kAttr_padding, &d->padding));

      // TODO : add checks here
      is_eligible = true;
    }

    // Next check if the specified dilation is supported
    if (is_eligible) {
      is_eligible = false;
      
      TF_CHECK_OK(GetNodeAttr(d->conv->def(), kAttr_dilations, &d->dilations));

      // TODO : add checks here
      is_eligible = true;
    }

    // finally check if the specified activation is supported
    if (is_eligible) {
      is_eligible = false;

      // TODO : add checks here
      d->activation_mode = "Relu";
      is_eligible = true;
    }
    
    return is_eligible;
  }
  
  void ROCmFusionOpCBA::CreateFusionOp(const FusionData* d) {

    std::vector<const Edge*> conv_input_edges;
    TF_CHECK_OK(d->conv->input_edges(&conv_input_edges));
      
    std::vector<const Edge*> bias_input_edges;
    TF_CHECK_OK(d->bias->input_edges(&bias_input_edges));
    
    // create an instance of the fusion node
    NodeBuilder nb(d->conv->name(), fusion_op_name_);

    // populate input data edges
    nb.Input(conv_input_edges[0]->src(), conv_input_edges[0]->src_output());
    nb.Input(conv_input_edges[1]->src(), conv_input_edges[1]->src_output());
    nb.Input(bias_input_edges[1]->src(), bias_input_edges[1]->src_output());

    // populate attributes
    nb.Attr(kAttr_T, d->data_type);
    nb.Attr(kAttr_strides, d->strides);
    nb.Attr(kAttr_padding, d->padding);
    nb.Attr(kAttr_data_format, d->data_format);
    nb.Attr(kAttr_dilations, d->dilations);
    nb.Attr(kAttr_activation_mode, d->activation_mode);

    // populate the device
    nb.Device(d->conv->def().device());

    // create the new fusion node
    Node* fusion_node = nullptr;
    TF_CHECK_OK(nb.Finalize(graph_, &fusion_node));
    
    // populate the input control edges
    for (const Edge* e : d->conv->in_edges()) {
      if (e->IsControlEdge()) {
	CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
      }
    }
    for (const Edge* e : d->bias->in_edges()) {
      if (e->IsControlEdge()) {
	CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
      }
    }
    for (const Edge* e : d->actv->in_edges()) {
      if (e->IsControlEdge()) {
	CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
      }
    }
    
    // populate output data and control edges
    for (const Edge* e : d->conv->out_edges()) {
      if (e->IsControlEdge()) {
	CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
      }
    }
    for (const Edge* e : d->bias->out_edges()) {
      if (e->IsControlEdge()) {
	CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
      }
    }
    for (const Edge* e : d->actv->out_edges()) {
      if (e->IsControlEdge()) {
	CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
      } else {
	CHECK_NOTNULL(graph_->AddEdge(fusion_node, 0, e->dst(), e->dst_input()));
      }
    }

    // populate the device placement
    fusion_node->set_assigned_device_name(d->conv->assigned_device_name());

    VLOG(kVlogLevel) << "Created CBA Fusion Node: " << fusion_node;
    
    // add the now redundant nodes to the set of fused nodes
    fused_nodes_.insert(d->conv);
    fused_nodes_.insert(d->bias);
    fused_nodes_.insert(d->actv);

    // and remove them from the graph
    graph_->RemoveNode(const_cast<Node*>(d->conv));
    graph_->RemoveNode(const_cast<Node*>(d->bias));
    graph_->RemoveNode(const_cast<Node*>(d->actv));
  }


  // -------------------------------------------
  // ROCmFusionOpNA implementation
  // -------------------------------------------

  bool ROCmFusionOpNA::DoFusion(const Node* n2) {
    
    FusionData d;
    return IsFusionEligible(n2, &d);
  }
  
  bool ROCmFusionOpNA::IsFusionEligible(const Node* n2, FusionData* d) {
    
    // First check whether we have the right sequence of ops
    bool is_eligible = false;
    if (isOpActv(n2)) { // activation node
      Node* n1 = nullptr;
      TF_CHECK_OK(n2->input_node(0, &n1));
      if (isOpNorm(n1)) { // preceded by a batchnorm node
	d->norm = n1;
	d->actv = n2;
	DumpNodeList(kVlogLevel, "Found Fusion Candidate NA : ",
		     {d->norm, d->actv});
	is_eligible = true;
      }
    }
    return is_eligible;
  }
  
}  // namespace rocm_fusion_pass
}  // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM
